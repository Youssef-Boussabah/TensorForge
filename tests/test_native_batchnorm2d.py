"""Phase F, milestone F4 — NativeBatchNorm2d.

NCHW ``(N, C, H, W)`` batch normalization over the **same** shared private
implementation ``NativeBatchNorm1d`` already uses: this milestone supplies
a rank, a reduction-axis set, a broadcast layout, and the channels-last
permutation the rank-1 affine parameters need — nothing else.

The tests below are behavioral, and they lean hard on shapes where a wrong
axis or a trailing-broadcast mistake cannot accidentally look right: C, H,
and W are deliberately unequal almost everywhere.

Nothing here is a normalization *kernel*, C ABI symbol, ``NativeTensorCore``
method, or ``NativeTensor`` operation: F4 adds a module, and the guardrails
at the end pin exactly that.
"""

import gc
import os

import numpy as np
import pytest

import tensorforge as tf
from tensorforge.backends import cpp
from tensorforge.experimental import (
    NativeAdam, NativeBatchNorm1d, NativeBatchNorm2d, NativeConv2d,
    NativeFlatten, NativeLinear, NativeMaxPool2d, NativeModule,
    NativeParameter, NativeReLU, NativeSGD, NativeSequential, NativeTensor,
    load_native_checkpoint, save_native_checkpoint,
)
from tensorforge.experimental import _native_state
from tensorforge.experimental import native_batchnorm

needs_native = pytest.mark.skipif(
    not cpp.is_available(), reason="the experimental C++ backend is not built"
)

needs_fault_injection = pytest.mark.skipif(
    not (cpp.is_available() and cpp.fault_injection_available()),
    reason="the backend was built without the deterministic allocation "
           "fault-injection hook",
)


# ==========================================================================
# Fixtures, references, helpers
# ==========================================================================

@pytest.fixture
def live_storages(monkeypatch):
    """The ids of every NativeStorage currently open — the project's
    deterministic native-allocation instrumentation. The count is exact
    (it hooks close())."""
    open_ids = set()
    original_init = cpp.NativeStorage.__init__
    original_close = cpp.NativeStorage.close

    def tracked_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        open_ids.add(id(self))

    def tracked_close(self):
        original_close(self)
        open_ids.discard(id(self))

    monkeypatch.setattr(cpp.NativeStorage, "__init__", tracked_init)
    monkeypatch.setattr(cpp.NativeStorage, "close", tracked_close)
    return open_ids


def _collect():
    """Force the composed autograd graph's intermediate wrappers — which
    participate in reference cycles, a property of the Python-managed
    native autograd engine since Phase B, not of BatchNorm — to their
    deterministic collection point."""
    gc.collect()


def _channel_shape(values, channels):
    """``(C,)`` values reshaped for NCHW channel broadcasting."""
    return np.asarray(values, dtype=np.float64).reshape(1, channels, 1, 1)


def train_reference(x, gamma, beta, eps):
    """The population-variance NCHW BatchNorm, in NumPy — an external
    oracle, never run inside an armed tripwire. Reduces over N, H, and W
    and leaves the channel axis alone. Returns the output plus the ``(C,)``
    batch statistics that drove it."""
    x = np.asarray(x, dtype=np.float64)
    channels = x.shape[1]
    mean = x.mean(axis=(0, 2, 3), keepdims=True)
    var = ((x - mean) ** 2).mean(axis=(0, 2, 3), keepdims=True)
    normalized = (x - mean) / np.sqrt(var + eps)
    out = normalized * _channel_shape(gamma, channels) + _channel_shape(
        beta, channels
    )
    return out, mean.ravel(), var.ravel()


def eval_reference(x, running_mean, running_var, gamma, beta, eps):
    x = np.asarray(x, dtype=np.float64)
    channels = x.shape[1]
    normalized = (x - _channel_shape(running_mean, channels)) / np.sqrt(
        _channel_shape(running_var, channels) + eps
    )
    return normalized * _channel_shape(gamma, channels) + _channel_shape(
        beta, channels
    )


def stable_reference(x, gamma, beta, eps, momentum, running_mean, running_var):
    """The honest stable-framework oracle: move C last, flatten N/H/W into
    one sample dimension, run ``tensorforge.nn.BatchNorm1d``, and move the
    axes back. Same mathematics, a completely independent implementation."""
    x = np.asarray(x, dtype=np.float64)
    n, channels, height, width = x.shape
    flat = np.transpose(x, (0, 2, 3, 1)).reshape(-1, channels)
    module = tf.nn.BatchNorm1d(channels, eps=eps, momentum=momentum)
    module.gamma.data[:] = np.asarray(gamma, dtype=np.float64)
    module.beta.data[:] = np.asarray(beta, dtype=np.float64)
    module.running_mean = np.array(running_mean, dtype=np.float64)
    module.running_var = np.array(running_var, dtype=np.float64)
    out = module(tf.Tensor(flat)).data
    out = np.transpose(out.reshape(n, height, width, channels), (0, 3, 1, 2))
    return out, module.running_mean, module.running_var


def running_reference(running_mean, running_var, batch_mean, batch_var,
                      momentum):
    return (
        (1 - momentum) * running_mean + momentum * batch_mean,
        (1 - momentum) * running_var + momentum * batch_var,
    )


def _load(module, gamma=None, beta=None, running_mean=None, running_var=None):
    """Load nontrivial state through the public atomic loader (identity
    preserved), leaving unspecified entries alone."""
    supplied = {
        "gamma": gamma, "beta": beta,
        "running_mean": running_mean, "running_var": running_var,
    }
    values = {
        key: NativeTensor.from_array(np.asarray(value, dtype=np.float64))
        for key, value in supplied.items() if value is not None
    }
    module.load_state_dict(values, strict=False)
    for value in values.values():
        value.close()


def _module_state(module):
    return {
        "gamma_id": id(module.gamma),
        "beta_id": id(module.beta),
        "mean_id": id(module.running_mean),
        "var_id": id(module.running_var),
        "gamma": module.gamma.to_numpy().copy(),
        "beta": module.beta.to_numpy().copy(),
        "mean": module.running_mean.to_numpy().copy(),
        "var": module.running_var.to_numpy().copy(),
        "gamma_version": module.gamma.version,
        "beta_version": module.beta.version,
        "gamma_grad": None if module.gamma.grad is None
                      else module.gamma.grad.to_numpy().copy(),
        "beta_grad": None if module.beta.grad is None
                     else module.beta.grad.to_numpy().copy(),
    }


def _assert_state_unchanged(module, before):
    after = _module_state(module)
    for key in ("gamma_id", "beta_id", "mean_id", "var_id",
                "gamma_version", "beta_version"):
        assert after[key] == before[key], key
    for key in ("gamma", "beta", "mean", "var"):
        assert np.array_equal(after[key], before[key]), key
    for key in ("gamma_grad", "beta_grad"):
        if before[key] is None:
            assert after[key] is None, key
        else:
            assert np.array_equal(after[key], before[key]), key
    for tensor in (module.gamma, module.beta,
                   module.running_mean, module.running_var):
        assert tensor.closed is False


def _graph_objects(root):
    """Every object reachable from ``root`` through the autograd graph."""
    seen = {}

    def visit(node):
        if id(node) in seen:
            return
        seen[id(node)] = node
        for parent in node._parents:
            visit(parent)
        for resource in node._graph_resources:
            seen[id(resource)] = resource

    visit(root)
    return seen


def _graph_storage_ids(root):
    """Every native **storage** the graph can reach — stronger than the
    object walk, because a borrowing view of a buffer is a different
    object over the same bytes."""
    ids = set()
    for obj in _graph_objects(root).values():
        if isinstance(obj, NativeTensor) and not obj.closed:
            ids.add(id(obj._core.storage))
    return ids


def _central_difference(f, base, h=1e-6):
    base = np.asarray(base, dtype=np.float64)
    grad = np.zeros_like(base)
    flat = base.ravel()
    for i in range(flat.size):
        up = flat.copy()
        down = flat.copy()
        up[i] += h
        down[i] -= h
        grad.ravel()[i] = (
            f(up.reshape(base.shape)) - f(down.reshape(base.shape))
        ) / (2 * h)
    return grad


def _close_all(module):
    """§9's consequence of there being no ``NativeModule.close()``: the
    owner releases **both** parameters and buffers explicitly."""
    for tensor in module.parameters():
        tensor.close()
    for tensor in module.buffers():
        tensor.close()


# ==========================================================================
# Shared implementation proof
# ==========================================================================

# Everything both public shapes must inherit *by function identity*.
_SHARED_METHODS = (
    "forward", "_training_forward", "_eval_forward", "_mean_over",
    "_inverse_std", "_snapshot", "_blend", "_affine",
    "_commit_running_state", "_validate_forward", "_registered_running",
    "__init__", "__repr__",
)

# The only names a public subclass may declare: shape/layout configuration.
_SHAPE_CONFIG = {
    "_INPUT_NDIM", "_REDUCTION_AXES", "_TRAILING_DIMS", "_LAYOUT",
    "_CHANNELS_LAST",
}


@needs_native
def test_both_shapes_subclass_the_one_private_implementation():
    base = native_batchnorm._NativeBatchNorm
    assert issubclass(NativeBatchNorm1d, base)
    assert issubclass(NativeBatchNorm2d, base)
    assert NativeBatchNorm1d is not NativeBatchNorm2d
    # Neither is a subclass of the other: they are siblings.
    assert not issubclass(NativeBatchNorm2d, NativeBatchNorm1d)
    assert not issubclass(NativeBatchNorm1d, NativeBatchNorm2d)
    # The base is private and unexported.
    import tensorforge.experimental as experimental
    assert not hasattr(experimental, "_NativeBatchNorm")
    assert "_NativeBatchNorm" not in experimental.__all__


@needs_native
@pytest.mark.parametrize("method", _SHARED_METHODS)
def test_both_shapes_inherit_the_same_function_object(method):
    one = getattr(NativeBatchNorm1d, method)
    two = getattr(NativeBatchNorm2d, method)
    assert one is two, method
    # ...and it really is the base's, not a redefinition on either class.
    assert one is getattr(native_batchnorm._NativeBatchNorm, method), method
    assert method not in vars(NativeBatchNorm1d), method
    assert method not in vars(NativeBatchNorm2d), method


@needs_native
def test_native_batchnorm2d_declares_only_shape_configuration():
    declared = {
        name for name in vars(NativeBatchNorm2d) if not name.startswith("__")
    }
    assert declared <= _SHAPE_CONFIG, declared - _SHAPE_CONFIG
    assert declared == {
        "_INPUT_NDIM", "_REDUCTION_AXES", "_TRAILING_DIMS", "_LAYOUT",
        "_CHANNELS_LAST",
    }
    assert not any(callable(vars(NativeBatchNorm2d)[n]) for n in declared)
    assert NativeBatchNorm2d.__doc__            # a docstring, and no code


@needs_native
def test_the_configuration_is_exactly_the_nchw_contract():
    assert NativeBatchNorm2d._INPUT_NDIM == 4
    assert NativeBatchNorm2d._REDUCTION_AXES == (0, 2, 3)
    assert NativeBatchNorm2d._TRAILING_DIMS == 2
    assert NativeBatchNorm2d._LAYOUT == "(N, C, H, W)"
    assert NativeBatchNorm2d._CHANNELS_LAST == (0, 2, 3, 1)
    module = NativeBatchNorm2d(3)
    assert module._stat_shape == (1, 3, 1, 1)
    # The return leg is *derived*, so it can never disagree.
    assert module._channels_first == (0, 3, 1, 2)
    permutation = NativeBatchNorm2d._CHANNELS_LAST
    for axis, position in enumerate(module._channels_first):
        assert permutation[position] == axis
    _close_all(module)


@needs_native
def test_the_source_holds_one_implementation_of_each_shared_method():
    """A structural check against the obvious regression: someone
    "fixing" NCHW by copying a method onto the subclass."""
    source = open(native_batchnorm.__file__, encoding="utf-8").read()
    for method in _SHARED_METHODS:
        if method.startswith("__"):
            continue
        assert source.count(f"    def {method}(") == 1, method
    assert source.count("class _NativeBatchNorm(") == 1
    assert source.count("class NativeBatchNorm1d(") == 1
    assert source.count("class NativeBatchNorm2d(") == 1
    # Both public classes live in the same file.
    assert NativeBatchNorm1d.__module__ == NativeBatchNorm2d.__module__


# ==========================================================================
# Constructor, state, and registration (the inherited contract)
# ==========================================================================

@needs_native
def test_constructor_signature_is_exact():
    import inspect

    signature = inspect.signature(NativeBatchNorm2d.__init__)
    assert list(signature.parameters) == [
        "self", "num_features", "eps", "momentum"
    ]
    assert signature.parameters["eps"].default == 1e-5
    assert signature.parameters["momentum"].default == 0.1
    for kwargs in ({"affine": False}, {"track_running_stats": False},
                   {"dtype": "float64"}, {"device": "cpu"}, {"seed": 1},
                   {"requires_grad": False}, {"axis": 1},
                   {"layout": "NCHW"}, {"channels_last": True},
                   {"unbiased": False}):
        with pytest.raises(TypeError):
            NativeBatchNorm2d(3, **kwargs)


@needs_native
@pytest.mark.parametrize("bad,error", [
    (True, TypeError), (3.0, TypeError), ("3", TypeError), (None, TypeError),
    (np.int64(3), TypeError), (0, ValueError), (-2, ValueError),
])
def test_inherited_num_features_validation(bad, error, live_storages):
    baseline = len(live_storages)
    with pytest.raises(error):
        NativeBatchNorm2d(bad)
    assert len(live_storages) == baseline


@needs_native
@pytest.mark.parametrize("bad,error", [
    (True, TypeError), ("1e-5", TypeError), (None, TypeError),
    (0, ValueError), (-1.0, ValueError),
])
def test_inherited_eps_validation(bad, error, live_storages):
    baseline = len(live_storages)
    with pytest.raises(error):
        NativeBatchNorm2d(3, eps=bad)
    assert len(live_storages) == baseline


@needs_native
@pytest.mark.parametrize("bad,error", [
    (True, TypeError), ("0.1", TypeError), (None, TypeError),
    (-0.1, ValueError), (1.1, ValueError), (float("nan"), ValueError),
])
def test_inherited_momentum_validation(bad, error, live_storages):
    baseline = len(live_storages)
    with pytest.raises(error):
        NativeBatchNorm2d(3, momentum=bad)
    assert len(live_storages) == baseline


@needs_native
def test_accepted_configurations():
    for module in (NativeBatchNorm2d(1), NativeBatchNorm2d(6, eps=1e-2),
                   NativeBatchNorm2d(3, momentum=0.0),
                   NativeBatchNorm2d(3, momentum=1.0),
                   NativeBatchNorm2d(3, momentum=0.35)):
        assert isinstance(module.eps, float)
        assert isinstance(module.momentum, float)
        _close_all(module)


@needs_native
def test_state_initialization_and_order():
    module = NativeBatchNorm2d(4)
    assert np.array_equal(module.gamma.to_numpy(), np.ones(4))
    assert np.array_equal(module.beta.to_numpy(), np.zeros(4))
    assert np.array_equal(module.running_mean.to_numpy(), np.zeros(4))
    assert np.array_equal(module.running_var.to_numpy(), np.ones(4))
    for tensor in (module.gamma, module.beta,
                   module.running_mean, module.running_var):
        assert tensor.shape == (4,)               # buffers stay (C,)
        assert tensor.owns_core and tensor.contiguous
        assert tensor.dtype == "float64" and tensor.device == "cpu"
    assert isinstance(module.gamma, NativeParameter)
    assert type(module.running_mean) is NativeTensor
    assert module.gamma.requires_grad and module.beta.requires_grad
    assert not module.running_mean.requires_grad
    assert not module.running_var.requires_grad
    assert module.gamma.version == 0 and module.beta.version == 0
    assert [n for n, _ in module.named_parameters()] == ["gamma", "beta"]
    assert [n for n, _ in module.named_buffers()] == [
        "running_mean", "running_var"
    ]
    assert list(module.state_dict()) == [
        "gamma", "beta", "running_mean", "running_var"
    ]
    for snapshot in module.state_dict().values():
        snapshot.close()
    _close_all(module)


@needs_native
def test_optimizers_discover_only_the_affine_parameters():
    module = NativeBatchNorm2d(3)
    for optimizer in (NativeSGD(module.parameters(), lr=0.1),
                      NativeAdam(module.parameters(), lr=0.1)):
        discovered = [id(p) for p in optimizer.parameters()]
        assert discovered == [id(module.gamma), id(module.beta)]
        assert id(module.running_mean) not in discovered
        assert id(module.running_var) not in discovered
        close = getattr(optimizer, "close", None)
        if close is not None:
            close()
    _close_all(module)


@needs_native
def test_repr_is_deterministic_and_metadata_only():
    module = NativeBatchNorm2d(5, eps=1e-3, momentum=0.25)
    text = repr(module)
    assert text == "NativeBatchNorm2d(num_features=5, eps=0.001, momentum=0.25)"
    module.eval()
    assert repr(module) == text
    x = NativeTensor.from_array(np.zeros((2, 5, 2, 3)))
    module.train()
    module(x).close()
    assert repr(module) == text          # no running values, no address
    assert "0x" not in text
    x.close()
    _close_all(module)


@needs_fault_injection
def test_constructor_failure_releases_every_earlier_allocation(
    monkeypatch, live_storages
):
    """The inherited cleanup path, re-proved for the NCHW class: the
    constructor allocates gamma, beta, running_mean, running_var in that
    order, and forcing the 2nd/3rd/4th native allocation to fail must
    return to the pre-construction baseline with every earlier object
    closed."""
    for nth, already_created in ((2, 1), (3, 2), (4, 3)):
        created = []
        real_parameter = native_batchnorm.NativeParameter
        real_zeros = NativeTensor.zeros
        real_full = NativeTensor.full

        def spy_parameter(*args, **kwargs):
            parameter = real_parameter(*args, **kwargs)
            created.append(parameter)
            return parameter

        def spy_zeros(*args, **kwargs):
            tensor = real_zeros(*args, **kwargs)
            created.append(tensor)
            return tensor

        def spy_full(*args, **kwargs):
            tensor = real_full(*args, **kwargs)
            created.append(tensor)
            return tensor

        monkeypatch.setattr(native_batchnorm, "NativeParameter", spy_parameter)
        monkeypatch.setattr(
            native_batchnorm.NativeTensor, "zeros", staticmethod(spy_zeros)
        )
        monkeypatch.setattr(
            native_batchnorm.NativeTensor, "full", staticmethod(spy_full)
        )
        _collect()
        baseline = len(live_storages)
        cpp._arm_alloc_failure(nth)
        try:
            with pytest.raises(MemoryError):
                NativeBatchNorm2d(4)
        finally:
            cpp._arm_alloc_failure(0)
            cpp._require_library().tf_clear_error()
            monkeypatch.undo()
        assert len(created) == already_created, nth
        for tensor in created:
            assert tensor.closed is True, nth
        assert len(live_storages) == baseline, nth


# ==========================================================================
# NCHW shape validation
# ==========================================================================

@needs_native
@pytest.mark.parametrize("shape", [
    (1, 3, 1, 1), (1, 3, 2, 5), (4, 3, 2, 5), (2, 1, 3, 7),
    (3, 6, 1, 4), (2, 4, 5, 1), (7, 2, 3, 3),
])
def test_accepts_valid_nchw_shapes(shape):
    module = NativeBatchNorm2d(shape[1])
    x = NativeTensor.from_array(
        np.random.default_rng(sum(shape)).standard_normal(shape)
    )
    out = module(x)
    assert out.shape == shape
    out.close()
    x.close()
    _close_all(module)


@needs_native
@pytest.mark.parametrize("shape", [
    (), (3,), (4, 3), (2, 3, 5), (2, 3, 4, 5, 6),
])
def test_rejects_wrong_rank(shape, live_storages):
    module = NativeBatchNorm2d(3)
    x = NativeTensor.from_array(np.zeros(shape) if shape else np.zeros(()))
    baseline = len(live_storages)
    with pytest.raises(ValueError) as error:
        module(x)
    message = str(error.value)
    assert "NativeBatchNorm2d" in message
    assert "(N, C, H, W)" in message
    assert "C=3" in message
    assert str(tuple(x.shape)) in message
    assert len(live_storages) == baseline
    x.close()
    _close_all(module)


@needs_native
def test_rejects_nhwc_input(live_storages):
    """NHWC ``(2, 4, 5, 3)`` has the channels last. H=4 != C=3, so the
    channel check catches it — which is exactly why the tests elsewhere
    keep C, H, and W unequal."""
    module = NativeBatchNorm2d(3)
    x = NativeTensor.from_array(
        np.random.default_rng(1).standard_normal((2, 4, 5, 3))
    )
    baseline = len(live_storages)
    with pytest.raises(ValueError) as error:
        module(x)
    assert "(N, C, H, W)" in str(error.value)
    assert "(2, 4, 5, 3)" in str(error.value)
    assert len(live_storages) == baseline
    x.close()
    _close_all(module)


@needs_native
@pytest.mark.parametrize("shape", [(2, 5, 3, 4), (2, 1, 3, 4), (2, 8, 3, 4)])
def test_rejects_wrong_channel_count(shape, live_storages):
    module = NativeBatchNorm2d(3)
    x = NativeTensor.from_array(np.zeros(shape))
    baseline = len(live_storages)
    with pytest.raises(ValueError) as error:
        module(x)
    assert "C=3" in str(error.value) and str(shape) in str(error.value)
    assert len(live_storages) == baseline
    x.close()
    _close_all(module)


@needs_native
def test_rejects_right_element_count_with_wrong_layout(live_storages):
    """24 elements every time. Rank 4 with C at axis 1 is the contract, so
    ``(2, 3, 2, 2)`` and ``(4, 3, 2, 1)`` are both legitimate NCHW inputs
    while the rest are not — the layout is never silently reinterpreted to
    make the element count fit."""
    module = NativeBatchNorm2d(3)
    for valid in ((2, 3, 2, 2), (4, 3, 2, 1), (1, 3, 4, 2)):
        x = NativeTensor.from_array(np.arange(24.0).reshape(valid))
        out = module(x)
        assert out.shape == valid
        out.close()
        x.close()
    for wrong in ((2, 2, 2, 3), (2, 4, 3, 1), (24,), (2, 12), (2, 3, 4)):
        x = NativeTensor.from_array(np.arange(24.0).reshape(wrong))
        baseline = len(live_storages)
        with pytest.raises(ValueError):
            module(x)
        assert len(live_storages) == baseline
        x.close()
    _close_all(module)


@needs_native
@pytest.mark.parametrize("bad", [
    np.zeros((2, 3, 2, 2)), [[[[0.0]]]], (0.0,), 1.0, None, object(),
])
def test_rejects_non_native_input(bad, live_storages):
    module = NativeBatchNorm2d(3)
    baseline = len(live_storages)
    with pytest.raises(TypeError):
        module(bad)
    assert len(live_storages) == baseline
    _close_all(module)


@needs_native
def test_rejects_stable_framework_tensor(live_storages):
    module = NativeBatchNorm2d(3)
    baseline = len(live_storages)
    with pytest.raises(TypeError):
        module(tf.Tensor(np.zeros((2, 3, 2, 2))))
    assert len(live_storages) == baseline
    _close_all(module)


@needs_native
def test_rejects_closed_input(live_storages):
    module = NativeBatchNorm2d(3)
    x = NativeTensor.from_array(np.zeros((2, 3, 2, 2)))
    x.close()
    baseline = len(live_storages)
    with pytest.raises(RuntimeError):
        module(x)
    assert len(live_storages) == baseline
    _close_all(module)


@needs_native
@pytest.mark.parametrize("name", ["gamma", "beta", "running_mean", "running_var"])
def test_rejects_closed_state(name, live_storages):
    module = NativeBatchNorm2d(3)
    getattr(module, name).close()
    x = NativeTensor.from_array(np.zeros((2, 3, 2, 2)))
    baseline = len(live_storages)
    with pytest.raises(RuntimeError) as error:
        module(x)
    assert name in str(error.value)
    assert len(live_storages) == baseline      # nothing was built
    x.close()
    _close_all(module)


@needs_native
@pytest.mark.parametrize("name", ["running_mean", "running_var"])
def test_rejects_corrupted_running_buffer(name, live_storages):
    module = NativeBatchNorm2d(3)
    wrong = NativeTensor.zeros((5,))
    module._buffers[name] = module._buffers[name]._replace(tensor=wrong)
    x = NativeTensor.from_array(np.zeros((2, 3, 2, 2)))
    baseline = len(live_storages)
    with pytest.raises(ValueError) as error:
        module(x)
    assert name in str(error.value)
    assert len(live_storages) == baseline
    wrong.close()
    x.close()
    module.gamma.close()
    module.beta.close()


@needs_native
def test_validation_mutates_nothing(live_storages):
    module = NativeBatchNorm2d(3)
    before = _module_state(module)
    baseline = len(live_storages)
    for bad in (NativeTensor.from_array(np.zeros((2, 5, 2, 2))),
                NativeTensor.from_array(np.zeros((2, 3))),
                "not a tensor"):
        with pytest.raises((TypeError, ValueError)):
            module(bad)
        if isinstance(bad, NativeTensor):
            bad.close()
    _assert_state_unchanged(module, before)
    assert len(live_storages) == baseline
    _close_all(module)


# ==========================================================================
# Reduction-axis correctness
# ==========================================================================

@needs_native
def test_reduces_n_h_and_w_but_never_c():
    """One channel-varying, one N-varying, one H-varying, and one
    W-varying pattern in a single input. If any of N/H/W were skipped —
    or if C were reduced — the statistics would differ from the
    reference."""
    n, channels, height, width = 3, 4, 2, 5
    x = np.zeros((n, channels, height, width))
    grid_n, grid_h, grid_w = np.meshgrid(
        np.arange(n), np.arange(height), np.arange(width), indexing="ij"
    )
    x[:, 0] = 10.0 + grid_n                 # varies only across N
    x[:, 1] = 20.0 + grid_h                 # varies only across H
    x[:, 2] = 30.0 + grid_w                 # varies only across W
    x[:, 3] = 40.0                          # constant channel

    module = NativeBatchNorm2d(channels, momentum=1.0)
    xt = NativeTensor.from_array(x)
    out = module(xt)

    expected, batch_mean, batch_var = train_reference(
        x, np.ones(channels), np.zeros(channels), 1e-5
    )
    assert np.allclose(out.to_numpy(), expected, atol=1e-12)
    # One statistic per channel — the channel axis was not reduced.
    assert module.running_mean.shape == (channels,)
    assert np.allclose(module.running_mean.to_numpy(), batch_mean, atol=1e-12)
    assert np.allclose(module.running_var.to_numpy(), batch_var, atol=1e-12)
    # Hand-computed: each channel's mean is its own base plus the mean of
    # the axis it varies along, and the constant channel has zero variance.
    assert np.allclose(
        batch_mean,
        [10.0 + (n - 1) / 2, 20.0 + (height - 1) / 2,
         30.0 + (width - 1) / 2, 40.0],
        atol=1e-12,
    )
    assert batch_var[3] == 0.0
    assert np.all(batch_var[:3] > 0)
    # Channels stay separated: no channel's statistic leaked into another.
    assert len(set(np.round(batch_mean, 9))) == channels
    out.close()
    xt.close()
    _close_all(module)


@needs_native
@pytest.mark.parametrize("varying_axis", [0, 2, 3])
def test_each_reduced_axis_actually_participates(varying_axis):
    """Vary along exactly one of N, H, W. The batch variance must be
    non-zero — which it could not be if that axis were skipped."""
    shape = [2, 3, 4, 5]
    x = np.zeros(shape)
    index = [slice(None)] * 4
    for position in range(shape[varying_axis]):
        index[varying_axis] = position
        x[tuple(index)] = float(position)
    module = NativeBatchNorm2d(3, momentum=1.0)
    xt = NativeTensor.from_array(x)
    module(xt).close()
    _, batch_mean, batch_var = train_reference(
        x, np.ones(3), np.zeros(3), 1e-5
    )
    assert np.all(batch_var > 0)
    assert np.allclose(module.running_var.to_numpy(), batch_var, atol=1e-12)
    assert np.allclose(module.running_mean.to_numpy(), batch_mean, atol=1e-12)
    xt.close()
    _close_all(module)


@needs_native
def test_channel_variation_alone_does_not_enter_a_channel_statistic():
    """Every channel is internally constant, so every batch variance must
    be exactly zero even though the channels differ wildly from each
    other. A reduction that included axis 1 could not produce that."""
    x = np.empty((2, 3, 4, 5))
    for channel, value in enumerate((-100.0, 0.0, 250.0)):
        x[:, channel] = value
    module = NativeBatchNorm2d(3, momentum=1.0)
    xt = NativeTensor.from_array(x)
    out = module(xt)
    assert np.allclose(module.running_var.to_numpy(), 0.0, atol=1e-15)
    assert np.allclose(
        module.running_mean.to_numpy(), [-100.0, 0.0, 250.0], atol=1e-12
    )
    assert np.allclose(out.to_numpy(), 0.0, atol=1e-12)
    out.close()
    xt.close()
    _close_all(module)


@needs_native
def test_batch_statistics_carry_the_channel_broadcast_shape(monkeypatch):
    """The statistics really are ``(1, C, 1, 1)`` — observed through the
    shared reduction helper rather than asserted about the output."""
    seen = []
    real_mean_over = native_batchnorm._NativeBatchNorm._mean_over

    def spy(self, value, track):
        result = real_mean_over(self, value, track)
        seen.append(result.shape)
        return result

    monkeypatch.setattr(
        native_batchnorm._NativeBatchNorm, "_mean_over", spy
    )
    module = NativeBatchNorm2d(3)
    xt = NativeTensor.from_array(
        np.random.default_rng(5).standard_normal((2, 3, 4, 5))
    )
    module(xt).close()
    monkeypatch.undo()
    assert seen == [(1, 3, 1, 1), (1, 3, 1, 1)]   # batch mean, batch var
    xt.close()
    _close_all(module)


# ==========================================================================
# Channelwise affine correctness
# ==========================================================================

@needs_native
def test_gamma_and_beta_apply_per_channel_not_per_spatial_position():
    """C=3, H=2, W=5, all different, so a rank-1 ``(C,)`` parameter
    broadcast against the *trailing* axis could not even run — and a
    C==W coincidence could not hide a mistake."""
    n, channels, height, width = 2, 3, 2, 5
    rng = np.random.default_rng(11)
    x = rng.standard_normal((n, channels, height, width))
    gamma = np.array([2.0, -3.0, 0.5])
    beta = np.array([10.0, -20.0, 0.25])

    plain = NativeBatchNorm2d(channels)
    affine = NativeBatchNorm2d(channels)
    _load(affine, gamma=gamma, beta=beta)
    xt = NativeTensor.from_array(x)
    normalized = plain(xt).to_numpy()
    out = affine(xt).to_numpy()

    # Exactly gamma[c] * normalized + beta[c], at *every* (n, h, w).
    expected = normalized * _channel_shape(gamma, channels) + _channel_shape(
        beta, channels
    )
    assert np.allclose(out, expected, atol=1e-12)
    for channel in range(channels):
        assert np.allclose(
            out[:, channel],
            normalized[:, channel] * gamma[channel] + beta[channel],
            atol=1e-12,
        )
    # And explicitly *not* the wrong trailing-axis broadcast, which would
    # scale by spatial column. (It needs W == C to even be expressible;
    # here it is not, which is the point.)
    assert width != channels
    assert out.shape == (n, channels, height, width)
    xt.close()
    _close_all(plain)
    _close_all(affine)


@needs_native
def test_a_channel_only_gamma_touches_only_that_channel():
    channels = 3
    rng = np.random.default_rng(13)
    x = rng.standard_normal((2, channels, 2, 5))
    base = NativeBatchNorm2d(channels)
    tweaked = NativeBatchNorm2d(channels)
    _load(tweaked, gamma=[1.0, 7.0, 1.0], beta=[0.0, 0.0, 0.0])
    xt = NativeTensor.from_array(x)
    before = base(xt).to_numpy()
    after = tweaked(xt).to_numpy()
    assert np.allclose(after[:, 0], before[:, 0], atol=1e-12)
    assert np.allclose(after[:, 2], before[:, 2], atol=1e-12)
    assert np.allclose(after[:, 1], 7.0 * before[:, 1], atol=1e-12)
    xt.close()
    _close_all(base)
    _close_all(tweaked)


@needs_native
def test_affine_gradients_reduce_over_n_h_and_w():
    n, channels, height, width = 3, 4, 2, 5
    rng = np.random.default_rng(17)
    x = rng.standard_normal((n, channels, height, width))
    gamma = rng.standard_normal(channels) + 1.0
    beta = rng.standard_normal(channels)
    module = NativeBatchNorm2d(channels, eps=1e-3)
    _load(module, gamma=gamma, beta=beta)
    xt = NativeParameter(x)
    out = module(xt)
    upstream = rng.standard_normal((n, channels, height, width))
    loss = out.multiply(NativeTensor.from_array(upstream)).sum()
    loss.backward()

    assert module.gamma.grad.shape == (channels,)
    assert module.beta.grad.shape == (channels,)
    # beta's gradient is exactly the upstream summed over N, H, W.
    assert np.allclose(
        module.beta.grad.to_numpy(), upstream.sum(axis=(0, 2, 3)), atol=1e-10
    )
    # gamma's is the normalized activation dotted with the upstream, again
    # summed over N, H, W.
    normalized, _, _ = train_reference(x, np.ones(channels), np.zeros(channels), 1e-3)
    assert np.allclose(
        module.gamma.grad.to_numpy(),
        (normalized * upstream).sum(axis=(0, 2, 3)),
        atol=1e-10,
    )
    loss.close()
    out.close()
    xt.close()
    _close_all(module)


@needs_native
@pytest.mark.parametrize("training", [True, False])
def test_output_is_fresh_owning_contiguous_nchw(training):
    module = NativeBatchNorm2d(3)
    _load(module, gamma=[1.5, -0.5, 2.0], beta=[0.1, 0.2, 0.3],
          running_mean=[0.2, 0.3, 0.4], running_var=[1.5, 2.5, 3.5])
    module.train(training)
    xt = NativeParameter(np.random.default_rng(19).standard_normal((2, 3, 2, 5)))
    out = module(xt)
    assert type(out) is NativeTensor
    assert out.shape == (2, 3, 2, 5)
    assert out.owns_core is True
    assert out.contiguous is True             # never a borrowing transpose
    storages = {id(xt._core.storage), id(module.gamma._core.storage),
                id(module.beta._core.storage),
                id(module.running_mean._core.storage),
                id(module.running_var._core.storage)}
    assert id(out._core.storage) not in storages
    out.close()
    xt.close()
    _close_all(module)


# ==========================================================================
# Parameter mutation guard
# ==========================================================================

@needs_native
@pytest.mark.parametrize("training", [True, False])
def test_gamma_mutation_after_forward_stales_the_graph(training):
    """The load-bearing consequence of keeping ``gamma`` a **direct**
    rank-1 ``multiply`` operand instead of reshaping it to ``(1, C, 1, 1)``:
    the existing direct-parameter version guard still sees it. A reshaped
    parameter would be an unversioned view, and this backward would
    silently return a gradient for a value that was never in the forward."""
    module = NativeBatchNorm2d(3)
    _load(module, running_mean=[0.1, 0.2, 0.3], running_var=[1.0, 2.0, 3.0])
    module.train(training)
    xt = NativeParameter(np.random.default_rng(23).standard_normal((2, 3, 2, 5)))
    out = module(xt)
    loss = out.sum()

    module.gamma.copy_value_(NativeTensor.from_array([5.0, 5.0, 5.0]))
    with pytest.raises(RuntimeError, match="stale parameter value") as error:
        loss.backward()
    assert "NativeParameter" in str(error.value)
    assert xt.grad is None            # nothing was committed
    loss.close()
    out.close()
    xt.close()
    _close_all(module)


@needs_native
@pytest.mark.parametrize("training", [True, False])
def test_unmutated_gamma_backward_succeeds(training):
    module = NativeBatchNorm2d(3)
    module.train(training)
    xt = NativeParameter(np.random.default_rng(29).standard_normal((2, 3, 2, 5)))
    out = module(xt)
    loss = out.sum()
    loss.backward()
    assert xt.grad is not None and xt.grad.shape == (2, 3, 2, 5)
    assert module.gamma.grad is not None
    loss.close()
    out.close()
    xt.close()
    _close_all(module)


@needs_native
def test_gamma_is_a_direct_versioned_graph_operand():
    """Structural: the parameter object itself — not a view of it — is a
    graph parent, and the graph records its expected version."""
    module = NativeBatchNorm2d(3)
    xt = NativeParameter(np.random.default_rng(31).standard_normal((2, 3, 2, 5)))
    out = module(xt)
    reachable = _graph_objects(out)
    assert id(module.gamma) in reachable
    assert id(module.beta) in reachable
    recorded = [
        parameter
        for node in reachable.values()
        for _, parameter, _ in node._expected_versions
    ]
    assert any(parameter is module.gamma for parameter in recorded)
    out.close()
    xt.close()
    _close_all(module)


# ==========================================================================
# Training numerical parity
# ==========================================================================

@needs_native
@pytest.mark.parametrize("shape", [
    (1, 1, 1, 1), (2, 3, 2, 5), (4, 2, 3, 1), (3, 5, 1, 4), (5, 3, 4, 2),
])
def test_training_parity_with_the_numpy_reference(shape):
    rng = np.random.default_rng(sum(shape) * 7)
    x = rng.standard_normal(shape) * 3 + 1.5
    channels = shape[1]
    module = NativeBatchNorm2d(channels)
    xt = NativeTensor.from_array(x)
    out = module(xt)
    expected, batch_mean, batch_var = train_reference(
        x, np.ones(channels), np.zeros(channels), 1e-5
    )
    assert np.allclose(out.to_numpy(), expected, atol=1e-12)
    new_mean, new_var = running_reference(
        np.zeros(channels), np.ones(channels), batch_mean, batch_var, 0.1
    )
    assert np.allclose(module.running_mean.to_numpy(), new_mean, atol=1e-12)
    assert np.allclose(module.running_var.to_numpy(), new_var, atol=1e-12)
    out.close()
    xt.close()
    _close_all(module)


@needs_native
@pytest.mark.parametrize("shape", [(2, 3, 2, 5), (4, 2, 3, 3), (3, 4, 1, 6)])
def test_training_parity_with_the_stable_batchnorm1d_over_flattened_samples(shape):
    """The honest stable-framework comparison: NCHW → NHWC → ``(N*H*W, C)``
    is *the same problem* the stable 1-D BatchNorm solves, so the two must
    agree exactly."""
    rng = np.random.default_rng(sum(shape) * 13)
    channels = shape[1]
    x = rng.standard_normal(shape) * 2
    gamma = rng.standard_normal(channels) + 1.0
    beta = rng.standard_normal(channels)
    start_mean = rng.standard_normal(channels)
    start_var = np.abs(rng.standard_normal(channels)) + 0.5

    module = NativeBatchNorm2d(channels, eps=1e-3, momentum=0.3)
    _load(module, gamma=gamma, beta=beta,
          running_mean=start_mean, running_var=start_var)
    xt = NativeTensor.from_array(x)
    out = module(xt)

    expected, expected_mean, expected_var = stable_reference(
        x, gamma, beta, 1e-3, 0.3, start_mean, start_var
    )
    assert np.allclose(out.to_numpy(), expected, atol=1e-12)
    assert np.allclose(module.running_mean.to_numpy(), expected_mean, atol=1e-12)
    assert np.allclose(module.running_var.to_numpy(), expected_var, atol=1e-12)
    out.close()
    xt.close()
    _close_all(module)


@needs_native
def test_population_variance_and_eps_inside_the_root():
    """Hand-computed: one channel holding [1, 3] over N=2, H=W=1 has mean
    2 and **population** variance 1 (the sample variance would be 2). With
    eps=3 the normalizer is sqrt(1 + 3) = 2, never sqrt(1) + 3 = 4."""
    module = NativeBatchNorm2d(1, eps=3.0, momentum=1.0)
    xt = NativeTensor.from_array(np.array([1.0, 3.0]).reshape(2, 1, 1, 1))
    out = module(xt)
    assert np.allclose(out.to_numpy().ravel(), [-0.5, 0.5], atol=1e-15)
    assert np.allclose(module.running_var.to_numpy(), [1.0], atol=1e-15)
    assert np.allclose(module.running_mean.to_numpy(), [2.0], atol=1e-15)
    out.close()
    xt.close()
    _close_all(module)


@needs_native
def test_constant_and_near_constant_channels_stay_finite():
    x = np.random.default_rng(37).standard_normal((2, 3, 2, 5))
    x[:, 0] = 4.0                      # exactly constant
    x[:, 1] = 4.0
    x[0, 1, 0, 0] += 1e-9              # near-constant
    module = NativeBatchNorm2d(3, eps=1e-4)
    xt = NativeTensor.from_array(x)
    out = module(xt)
    expected, _, batch_var = train_reference(x, np.ones(3), np.zeros(3), 1e-4)
    assert np.all(np.isfinite(out.to_numpy()))
    assert np.allclose(out.to_numpy(), expected, atol=1e-10)
    assert batch_var[0] == 0.0
    assert np.allclose(out.to_numpy()[:, 0], 0.0, atol=1e-15)
    out.close()
    xt.close()
    _close_all(module)


@needs_native
@pytest.mark.parametrize("momentum", [0.0, 1.0, 0.1, 0.42])
def test_running_update_momentum_convention(momentum):
    rng = np.random.default_rng(int(momentum * 1000) + 41)
    x = rng.standard_normal((3, 3, 2, 5)) * 2 - 1
    start_mean = np.array([0.25, -1.5, 3.0])
    start_var = np.array([2.5, 0.75, 1.0])
    module = NativeBatchNorm2d(3, momentum=momentum)
    _load(module, running_mean=start_mean, running_var=start_var)
    xt = NativeTensor.from_array(x)
    module(xt).close()
    _, batch_mean, batch_var = train_reference(x, np.ones(3), np.zeros(3), 1e-5)
    expected_mean, expected_var = running_reference(
        start_mean, start_var, batch_mean, batch_var, momentum
    )
    assert np.allclose(module.running_mean.to_numpy(), expected_mean, atol=1e-14)
    assert np.allclose(module.running_var.to_numpy(), expected_var, atol=1e-14)
    if momentum == 0.0:
        assert np.array_equal(module.running_mean.to_numpy(), start_mean)
        assert np.array_equal(module.running_var.to_numpy(), start_var)
    if momentum == 1.0:
        assert np.allclose(module.running_mean.to_numpy(), batch_mean, atol=1e-15)
        assert np.allclose(module.running_var.to_numpy(), batch_var, atol=1e-15)
    xt.close()
    _close_all(module)


@needs_native
def test_consecutive_training_forwards_accumulate_exactly():
    rng = np.random.default_rng(43)
    batches = [rng.standard_normal((2, 3, 2, 5)) for _ in range(5)]
    module = NativeBatchNorm2d(3, momentum=0.2)
    running_mean = np.zeros(3)
    running_var = np.ones(3)
    for batch in batches:
        xt = NativeTensor.from_array(batch)
        module(xt).close()
        xt.close()
        _, batch_mean, batch_var = train_reference(
            batch, np.ones(3), np.zeros(3), 1e-5
        )
        running_mean, running_var = running_reference(
            running_mean, running_var, batch_mean, batch_var, 0.2
        )
        assert np.allclose(module.running_mean.to_numpy(), running_mean, atol=1e-12)
        assert np.allclose(module.running_var.to_numpy(), running_var, atol=1e-12)
    _close_all(module)


# ==========================================================================
# Evaluation numerical parity
# ==========================================================================

@needs_native
def test_eval_uses_the_stored_running_statistics():
    rng = np.random.default_rng(47)
    gamma, beta = rng.standard_normal(3), rng.standard_normal(3)
    running_mean = np.array([0.3, -0.2, 1.1])
    running_var = np.array([2.0, 0.5, 1.25])
    module = NativeBatchNorm2d(3, eps=1e-3)
    _load(module, gamma=gamma, beta=beta,
          running_mean=running_mean, running_var=running_var)
    module.eval()
    x = rng.standard_normal((2, 3, 2, 5)) * 4 + 2
    xt = NativeTensor.from_array(x)
    out = module(xt)
    expected = eval_reference(x, running_mean, running_var, gamma, beta, 1e-3)
    assert np.allclose(out.to_numpy(), expected, atol=1e-12)
    # And definitely not the batch's own statistics.
    batch_expected, _, _ = train_reference(x, gamma, beta, 1e-3)
    assert not np.allclose(out.to_numpy(), batch_expected, atol=1e-6)
    out.close()
    xt.close()
    _close_all(module)


@needs_native
def test_eval_never_mutates_the_running_buffers():
    module = NativeBatchNorm2d(3)
    _load(module, running_mean=[1.0, 2.0, 3.0], running_var=[4.0, 5.0, 6.0])
    module.eval()
    before = _module_state(module)
    xt = NativeTensor.from_array(
        np.random.default_rng(53).standard_normal((3, 3, 2, 5)) * 100
    )
    for _ in range(4):
        module(xt).close()
        _assert_state_unchanged(module, before)
    xt.close()
    _close_all(module)


@needs_native
def test_train_and_eval_differ_when_their_statistics_differ():
    module = NativeBatchNorm2d(3)
    _load(module, running_mean=[5.0, -5.0, 0.0], running_var=[9.0, 4.0, 1.0])
    x = np.random.default_rng(59).standard_normal((2, 3, 2, 5))
    xt = NativeTensor.from_array(x)
    module.eval()
    eval_out = module(xt).to_numpy()
    module.train()
    train_out = module(xt).to_numpy()
    assert not np.allclose(eval_out, train_out, atol=1e-6)
    xt.close()
    _close_all(module)


@needs_native
def test_eval_normalizes_a_single_sample_consistently():
    module = NativeBatchNorm2d(2)
    _load(module, running_mean=[1.0, -1.0], running_var=[4.0, 9.0])
    module.eval()
    x = np.array([[[[3.0, 5.0]], [[2.0, -4.0]]]])      # (1, 2, 1, 2)
    xt = NativeTensor.from_array(x)
    out = module(xt)
    expected = eval_reference(
        x, np.array([1.0, -1.0]), np.array([4.0, 9.0]),
        np.ones(2), np.zeros(2), 1e-5,
    )
    assert np.allclose(out.to_numpy(), expected, atol=1e-12)
    out.close()
    xt.close()
    _close_all(module)


@needs_native
def test_mode_toggles_change_no_state():
    module = NativeBatchNorm2d(3)
    before = _module_state(module)
    keys = list(module.state_dict())
    for _ in range(3):
        module.eval()
        module.train()
    _assert_state_unchanged(module, before)
    assert list(module.state_dict()) == keys
    for snapshot in module.state_dict().values():
        snapshot.close()
    _close_all(module)


# ==========================================================================
# Gradients
# ==========================================================================

def _train_objective(x, gamma, beta, eps, upstream):
    out, _, _ = train_reference(x, gamma, beta, eps)
    return float((out * upstream).sum())


def _eval_objective(x, running_mean, running_var, gamma, beta, eps, upstream):
    out = eval_reference(x, running_mean, running_var, gamma, beta, eps)
    return float((out * upstream).sum())


@needs_native
@pytest.mark.parametrize("shape,eps", [
    ((2, 3, 2, 5), 1e-5), ((3, 2, 1, 4), 1e-2), ((2, 4, 3, 1), 0.25),
])
def test_training_gradients_match_central_differences(shape, eps):
    rng = np.random.default_rng(sum(shape) * 19)
    channels = shape[1]
    x = rng.standard_normal(shape)
    gamma = rng.standard_normal(channels) + 1.0
    beta = rng.standard_normal(channels)
    upstream = rng.standard_normal(shape)

    module = NativeBatchNorm2d(channels, eps=eps)
    _load(module, gamma=gamma, beta=beta)
    xt = NativeParameter(x)
    out = module(xt)
    loss = out.multiply(NativeTensor.from_array(upstream)).sum()
    loss.backward()

    assert np.allclose(
        xt.grad.to_numpy(),
        _central_difference(
            lambda p: _train_objective(p, gamma, beta, eps, upstream), x
        ),
        atol=1e-6,
    )
    assert np.allclose(
        module.gamma.grad.to_numpy(),
        _central_difference(
            lambda p: _train_objective(x, p, beta, eps, upstream), gamma
        ),
        atol=1e-6,
    )
    assert np.allclose(
        module.beta.grad.to_numpy(),
        _central_difference(
            lambda p: _train_objective(x, gamma, p, eps, upstream), beta
        ),
        atol=1e-6,
    )
    assert xt.grad.shape == shape
    for tensor in (xt.grad, module.gamma.grad, module.beta.grad):
        assert tensor.dtype == "float64" and tensor.device == "cpu"
    loss.close()
    out.close()
    xt.close()
    _close_all(module)


@needs_native
def test_training_gradient_flows_through_the_nhw_batch_statistics():
    """A detached-statistics implementation would give
    ``upstream * gamma / std``. The real gradient differs, and with a
    constant upstream it sums to ~0 over each channel's N/H/W block —
    the signature of differentiating through the mean and the variance."""
    rng = np.random.default_rng(61)
    x = rng.standard_normal((3, 3, 2, 5))
    module = NativeBatchNorm2d(3)
    xt = NativeParameter(x)
    out = module(xt)
    upstream = np.ones((3, 3, 2, 5))
    loss = out.multiply(NativeTensor.from_array(upstream)).sum()
    loss.backward()
    grad = xt.grad.to_numpy()
    _, _, batch_var = train_reference(x, np.ones(3), np.zeros(3), 1e-5)
    detached = upstream / np.sqrt(_channel_shape(batch_var, 3) + 1e-5)
    assert not np.allclose(grad, detached, atol=1e-6)
    assert np.allclose(grad.sum(axis=(0, 2, 3)), 0.0, atol=1e-9)
    loss.close()
    out.close()
    xt.close()
    _close_all(module)


@needs_native
def test_training_gradient_of_a_near_constant_channel_is_finite():
    x = np.random.default_rng(67).standard_normal((2, 2, 2, 3))
    x[:, 0] = 1.0
    x[0, 0, 0, 0] += 1e-8
    module = NativeBatchNorm2d(2, eps=1e-2)
    xt = NativeParameter(x)
    out = module(xt)
    loss = out.multiply(NativeTensor.from_array(np.ones((2, 2, 2, 3)))).sum()
    loss.backward()
    assert np.all(np.isfinite(xt.grad.to_numpy()))
    assert np.all(np.isfinite(module.gamma.grad.to_numpy()))
    assert np.allclose(
        xt.grad.to_numpy(),
        _central_difference(
            lambda p: _train_objective(
                p, np.ones(2), np.zeros(2), 1e-2, np.ones((2, 2, 2, 3))
            ),
            x,
        ),
        atol=1e-5,
    )
    loss.close()
    out.close()
    xt.close()
    _close_all(module)


@needs_native
def test_finite_differences_through_the_native_module_do_not_leak_state():
    """Each probe builds a *fresh* module loaded with the same state, so
    the running statistics a training forward advances never accumulate
    across evaluations of the objective."""
    rng = np.random.default_rng(71)
    x = rng.standard_normal((2, 3, 2, 3))
    gamma = rng.standard_normal(3) + 1.0
    beta = rng.standard_normal(3)
    upstream = rng.standard_normal((2, 3, 2, 3))

    def native_objective(probe):
        module = NativeBatchNorm2d(3, eps=1e-3)
        _load(module, gamma=gamma, beta=beta)
        xt = NativeTensor.from_array(probe)
        out = module(xt)
        up = NativeTensor.from_array(upstream)
        loss = out.multiply(up).sum()
        value = float(loss.to_numpy())
        for tensor in (loss, out, up, xt):
            tensor.close()
        # Every probe starts from zeros/ones, never from a previous probe.
        assert not np.array_equal(module.running_mean.to_numpy(), np.zeros(3))
        _close_all(module)
        return value

    expected = _central_difference(native_objective, x)
    module = NativeBatchNorm2d(3, eps=1e-3)
    _load(module, gamma=gamma, beta=beta)
    xt = NativeParameter(x)
    out = module(xt)
    loss = out.multiply(NativeTensor.from_array(upstream)).sum()
    loss.backward()
    assert np.allclose(xt.grad.to_numpy(), expected, atol=1e-6)
    loss.close()
    out.close()
    xt.close()
    _close_all(module)


@needs_native
def test_eval_gradients_match_central_differences():
    rng = np.random.default_rng(73)
    shape = (2, 3, 2, 5)
    x = rng.standard_normal(shape)
    gamma = rng.standard_normal(3) + 1.0
    beta = rng.standard_normal(3)
    running_mean = rng.standard_normal(3)
    running_var = np.abs(rng.standard_normal(3)) + 0.5
    upstream = rng.standard_normal(shape)

    module = NativeBatchNorm2d(3, eps=1e-3)
    _load(module, gamma=gamma, beta=beta,
          running_mean=running_mean, running_var=running_var)
    module.eval()
    xt = NativeParameter(x)
    out = module(xt)
    loss = out.multiply(NativeTensor.from_array(upstream)).sum()
    loss.backward()

    def objective(probe_x=None, probe_gamma=None, probe_beta=None):
        return _eval_objective(
            x if probe_x is None else probe_x, running_mean, running_var,
            gamma if probe_gamma is None else probe_gamma,
            beta if probe_beta is None else probe_beta, 1e-3, upstream,
        )

    assert np.allclose(
        xt.grad.to_numpy(),
        _central_difference(lambda p: objective(probe_x=p), x), atol=1e-6
    )
    assert np.allclose(
        module.gamma.grad.to_numpy(),
        _central_difference(lambda p: objective(probe_gamma=p), gamma),
        atol=1e-6,
    )
    assert np.allclose(
        module.beta.grad.to_numpy(),
        _central_difference(lambda p: objective(probe_beta=p), beta),
        atol=1e-6,
    )
    assert module.running_mean.grad is None
    assert module.running_var.grad is None
    loss.close()
    out.close()
    xt.close()
    _close_all(module)


@needs_native
@pytest.mark.parametrize("training", [True, False])
def test_buffers_never_receive_gradients(training):
    module = NativeBatchNorm2d(3)
    module.train(training)
    xt = NativeParameter(np.random.default_rng(79).standard_normal((2, 3, 2, 5)))
    out = module(xt)
    loss = out.sum()
    loss.backward()
    assert module.running_mean.grad is None and module.running_var.grad is None
    assert not module.running_mean.requires_grad
    assert not module.running_var.requires_grad
    loss.close()
    out.close()
    xt.close()
    _close_all(module)


# ==========================================================================
# Graph-safe evaluation snapshots, at (1, C, 1, 1)
# ==========================================================================

def _eval_graph_with_control(module, x, upstream):
    """Build an eval graph plus the gradients its forward-time values
    imply, for an NCHW module."""
    channels = module.num_features
    running_mean = module.running_mean.to_numpy().copy()
    running_var = module.running_var.to_numpy().copy()
    gamma = module.gamma.to_numpy().copy()
    xt = NativeParameter(x)
    out = module(xt)
    loss = out.multiply(NativeTensor.from_array(upstream)).sum()
    inverse_std = 1.0 / np.sqrt(_channel_shape(running_var, channels) + module.eps)
    normalized = (x - _channel_shape(running_mean, channels)) * inverse_std
    control = {
        "x": upstream * _channel_shape(gamma, channels) * inverse_std,
        "gamma": (normalized * upstream).sum(axis=(0, 2, 3)),
        "beta": upstream.sum(axis=(0, 2, 3)),
    }
    return xt, out, loss, control


def _assert_control_gradients(xt, module, control):
    assert np.allclose(xt.grad.to_numpy(), control["x"], atol=1e-12)
    assert np.allclose(module.gamma.grad.to_numpy(), control["gamma"], atol=1e-12)
    assert np.allclose(module.beta.grad.to_numpy(), control["beta"], atol=1e-12)


@needs_native
def test_eval_graph_never_holds_a_registered_buffer():
    module = NativeBatchNorm2d(3)
    _load(module, running_mean=[0.5, 1.0, 1.5], running_var=[2.0, 3.0, 4.0])
    module.eval()
    xt = NativeParameter(np.random.default_rng(83).standard_normal((2, 3, 2, 5)))
    out = module(xt)
    reachable = _graph_objects(out)
    assert id(module.running_mean) not in reachable
    assert id(module.running_var) not in reachable
    assert id(module.gamma) in reachable and id(module.beta) in reachable
    buffer_storage = {
        id(module.running_mean._core.storage),
        id(module.running_var._core.storage),
    }
    assert not (buffer_storage & _graph_storage_ids(out))
    resources = out._graph_resources
    assert len(resources) == 2
    for resource in resources:
        assert resource.shape == (1, 3, 1, 1)       # the NCHW broadcast shape
        assert resource.owns_core is True
        assert resource.contiguous is True
        assert resource.requires_grad is False
        assert resource.is_leaf is True
        assert id(resource._core.storage) not in buffer_storage
    out.close()
    xt.close()
    _close_all(module)


@needs_native
def test_later_training_update_cannot_change_an_earlier_eval_backward():
    rng = np.random.default_rng(89)
    module = NativeBatchNorm2d(3, momentum=0.5)
    _load(module, gamma=[1.5, -0.5, 2.0], beta=[0.1, 0.2, -0.3],
          running_mean=[0.3, -0.2, 0.5], running_var=[2.0, 0.5, 1.25])
    module.eval()
    x = rng.standard_normal((2, 3, 2, 5))
    upstream = rng.standard_normal((2, 3, 2, 5))
    xt, out, loss, control = _eval_graph_with_control(module, x, upstream)

    module.train()
    other = NativeTensor.from_array(rng.standard_normal((3, 3, 2, 5)) * 9)
    module(other).close()
    other.close()
    assert not np.allclose(module.running_mean.to_numpy(), [0.3, -0.2, 0.5])

    loss.backward()
    _assert_control_gradients(xt, module, control)
    loss.close()
    out.close()
    xt.close()
    _close_all(module)


@needs_native
def test_later_buffer_only_state_load_cannot_change_an_earlier_eval_backward():
    rng = np.random.default_rng(97)
    module = NativeBatchNorm2d(3)
    _load(module, running_mean=[0.3, -0.2, 0.5], running_var=[2.0, 0.5, 1.25])
    module.eval()
    x = rng.standard_normal((2, 3, 2, 5))
    upstream = rng.standard_normal((2, 3, 2, 5))
    xt, out, loss, control = _eval_graph_with_control(module, x, upstream)
    versions = (module.gamma.version, module.beta.version)

    mean_core = module.running_mean._core
    var_core = module.running_var._core
    _load(module, running_mean=[9.0, 9.0, 9.0], running_var=[16.0, 16.0, 16.0])
    assert mean_core._closed is True and var_core._closed is True
    assert (module.gamma.version, module.beta.version) == versions

    loss.backward()
    _assert_control_gradients(xt, module, control)
    loss.close()
    out.close()
    xt.close()
    _close_all(module)


class _RunningStatHolder(NativeModule):
    """**Test-only** buffer-only module registering *existing* running
    buffers as persistent aliases, so ``load_native_checkpoint()`` can be
    driven over exactly those objects without touching ``gamma``/``beta``.
    Not a production helper, not exported, not a public API — two
    ``register_buffer`` calls on the objects it is handed."""

    def __init__(self, running_mean, running_var):
        super().__init__()
        self.register_buffer("running_mean", running_mean, persistent=True)
        self.register_buffer("running_var", running_var, persistent=True)


@needs_native
def test_buffer_only_checkpoint_load_cannot_change_an_earlier_eval_backward(
    tmp_path, live_storages
):
    """The F3 checkpoint-path proof, re-run for the NCHW shape."""
    _collect()
    baseline = len(live_storages)

    rng = np.random.default_rng(101)
    module = NativeBatchNorm2d(3)
    _load(module, gamma=[1.5, -0.5, 2.0], beta=[0.1, 0.2, -0.3],
          running_mean=[0.3, -0.2, 0.5], running_var=[2.0, 0.5, 1.25])
    module.eval()

    donor_mean = NativeTensor.from_array([7.0, 7.0, 7.0])
    donor_var = NativeTensor.from_array([25.0, 25.0, 25.0])
    donor = _RunningStatHolder(donor_mean, donor_var)
    assert donor.parameters() == []
    path = os.path.join(str(tmp_path), "running_stats.npz")
    save_native_checkpoint(path, donor, metadata={"kind": "running-stats"})
    donor_mean.close()
    donor_var.close()

    x = rng.standard_normal((2, 3, 2, 5))
    upstream = rng.standard_normal((2, 3, 2, 5))
    xt, out, loss, control = _eval_graph_with_control(module, x, upstream)
    resources = out._graph_resources
    snapshot_storages = {id(r._core.storage) for r in resources}
    graph_storages = _graph_storage_ids(out)

    mean_object, var_object = module.running_mean, module.running_var
    mean_core, var_core = mean_object._core, var_object._core
    old_storages = {id(mean_core.storage), id(var_core.storage)}
    assert not (old_storages & graph_storages)
    versions = (module.gamma.version, module.beta.version)

    holder = _RunningStatHolder(mean_object, var_object)
    assert holder.running_mean is module.running_mean
    metadata = load_native_checkpoint(path, holder)
    assert metadata == {"kind": "running-stats"}

    # The checkpoint path changed both values, kept both objects, replaced
    # and closed both cores, and moved no parameter version.
    assert np.allclose(module.running_mean.to_numpy(), [7.0, 7.0, 7.0])
    assert np.allclose(module.running_var.to_numpy(), [25.0, 25.0, 25.0])
    assert module.running_mean is mean_object and module.running_var is var_object
    assert mean_object._core is not mean_core and var_object._core is not var_core
    assert mean_core._closed is True and var_core._closed is True
    assert (module.gamma.version, module.beta.version) == versions

    loss.backward()
    _assert_control_gradients(xt, module, control)
    # ...and not the gradients the new statistics would give.
    new_inverse_std = 1.0 / np.sqrt(25.0 + module.eps)
    assert not np.allclose(
        xt.grad.to_numpy(),
        upstream * _channel_shape(module.gamma.to_numpy(), 3) * new_inverse_std,
        atol=1e-6,
    )
    assert not ({id(mean_object._core.storage), id(var_object._core.storage)}
                & graph_storages)
    assert all(resource.closed for resource in resources)
    assert out._graph_resources == ()
    assert not (snapshot_storages & live_storages)

    for tensor in (loss, out, xt, xt.grad,
                   module.gamma.grad, module.beta.grad):
        tensor.close()
    _close_all(module)
    del resources, mean_object, var_object, holder, donor
    _collect()
    assert len(live_storages) == baseline


@needs_native
def test_full_checkpoint_load_stales_the_graph_through_parameters_not_buffers(
    tmp_path
):
    """Unchanged from F3 and re-proved for NCHW: a full load also replaces
    ``gamma``/``beta``, so the existing parameter-version guard correctly
    rejects the earlier graph. That is a parameter contract, never a
    running-buffer effect."""
    rng = np.random.default_rng(103)
    donor = NativeBatchNorm2d(3)
    _load(donor, gamma=[3.0, 3.0, 3.0], beta=[1.0, 1.0, 1.0],
          running_mean=[7.0, 7.0, 7.0], running_var=[25.0, 25.0, 25.0])
    path = os.path.join(str(tmp_path), "bn2d.npz")
    save_native_checkpoint(path, donor)
    _close_all(donor)

    module = NativeBatchNorm2d(3)
    _load(module, running_mean=[0.3, -0.2, 0.5], running_var=[2.0, 0.5, 1.25])
    module.eval()
    x = rng.standard_normal((2, 3, 2, 5))
    upstream = rng.standard_normal((2, 3, 2, 5))
    xt, out, loss, _ = _eval_graph_with_control(module, x, upstream)
    resources = out._graph_resources
    mean_object, var_object = module.running_mean, module.running_var
    versions = (module.gamma.version, module.beta.version)

    load_native_checkpoint(path, module)
    assert module.running_mean is mean_object and module.running_var is var_object
    assert module.gamma.version == versions[0] + 1
    assert module.beta.version == versions[1] + 1

    with pytest.raises(RuntimeError, match="stale parameter value") as error:
        loss.backward()
    message = str(error.value)
    assert "NativeParameter" in message and "version" in message
    for buffer_word in ("running_mean", "running_var", "buffer"):
        assert buffer_word not in message, buffer_word
    assert all(not resource.closed for resource in resources)
    assert xt.grad is None
    reachable = _graph_objects(out)
    assert id(mean_object) not in reachable and id(var_object) not in reachable
    loss.close()
    out.close()
    xt.close()
    _close_all(module)


@needs_native
def test_eval_graph_survives_retain_graph_and_releases_once(live_storages):
    module = NativeBatchNorm2d(3)
    _load(module, running_mean=[0.3, -0.2, 0.5], running_var=[2.0, 0.5, 1.25])
    module.eval()
    rng = np.random.default_rng(107)
    x = rng.standard_normal((2, 3, 2, 5))
    upstream = rng.standard_normal((2, 3, 2, 5))
    xt, out, loss, control = _eval_graph_with_control(module, x, upstream)
    resources = out._graph_resources
    assert len(resources) == 2
    snapshot_storages = {id(r._core.storage) for r in resources}
    assert snapshot_storages <= live_storages

    loss.backward(retain_graph=True)
    assert all(not resource.closed for resource in resources)
    module.zero_grad()
    xt.zero_grad()
    loss.backward(retain_graph=True)
    _assert_control_gradients(xt, module, control)
    module.zero_grad()
    xt.zero_grad()

    loss.backward()                       # one-shot: releases the history
    _assert_control_gradients(xt, module, control)
    assert all(resource.closed for resource in resources)
    assert out._graph_resources == ()
    out._release_graph_resources()        # a second release is a no-op
    assert not (snapshot_storages & live_storages)

    loss.close()
    out.close()
    xt.close()
    _close_all(module)


@needs_native
def test_abandoned_eval_graph_releases_its_snapshots_on_close(live_storages):
    module = NativeBatchNorm2d(3)
    module.eval()
    xt = NativeParameter(np.random.default_rng(109).standard_normal((2, 3, 2, 5)))
    _collect()
    baseline = len(live_storages)
    out = module(xt)
    resources = out._graph_resources
    assert len(resources) == 2
    snapshot_storages = {id(r._core.storage) for r in resources}
    out.close()
    assert all(resource.closed for resource in resources)
    assert not (snapshot_storages & live_storages)
    del out, resources
    _collect()
    assert len(live_storages) == baseline
    xt.close()
    _close_all(module)


@needs_native
def test_training_graph_holds_no_buffer_and_no_snapshot_resource():
    module = NativeBatchNorm2d(3)
    xt = NativeParameter(np.random.default_rng(113).standard_normal((2, 3, 2, 5)))
    out = module(xt)
    reachable = _graph_objects(out)
    assert id(module.running_mean) not in reachable
    assert id(module.running_var) not in reachable
    buffer_storage = {id(module.running_mean._core.storage),
                      id(module.running_var._core.storage)}
    assert not (buffer_storage & _graph_storage_ids(out))
    assert out._graph_resources == ()
    out.close()
    xt.close()
    _close_all(module)


# ==========================================================================
# Atomic running-statistics updates
# ==========================================================================

@needs_native
def test_successful_update_advances_both_buffers_together():
    module = NativeBatchNorm2d(3, momentum=0.5)
    mean_core = module.running_mean._core
    var_core = module.running_var._core
    mean_id, var_id = id(module.running_mean), id(module.running_var)
    versions = (module.gamma.version, module.beta.version)
    x = np.random.default_rng(127).standard_normal((2, 3, 2, 5))
    xt = NativeTensor.from_array(x)
    module(xt).close()
    _, batch_mean, batch_var = train_reference(x, np.ones(3), np.zeros(3), 1e-5)
    expected_mean, expected_var = running_reference(
        np.zeros(3), np.ones(3), batch_mean, batch_var, 0.5
    )
    assert id(module.running_mean) == mean_id
    assert id(module.running_var) == var_id
    assert np.allclose(module.running_mean.to_numpy(), expected_mean, atol=1e-14)
    assert np.allclose(module.running_var.to_numpy(), expected_var, atol=1e-14)
    assert mean_core._closed is True and var_core._closed is True
    assert module.running_mean._core is not mean_core
    assert (module.gamma.version, module.beta.version) == versions
    xt.close()
    _close_all(module)


def _run_with_transaction_failure(monkeypatch, module, xt, *, stage_at=None,
                                  install_at=None):
    calls = {"stage": 0, "install": 0}
    real_stage = _native_state._stage_entry
    real_install = _native_state._install_core

    def stage(planned):
        calls["stage"] += 1
        if calls["stage"] == stage_at:
            raise MemoryError("injected staging failure")
        return real_stage(planned)

    def install(planned, core):
        calls["install"] += 1
        if calls["install"] == install_at:
            raise RuntimeError("injected install failure")
        return real_install(planned, core)

    if stage_at is not None:
        monkeypatch.setattr(_native_state, "_stage_entry", stage)
    if install_at is not None:
        monkeypatch.setattr(_native_state, "_install_core", install)
    with pytest.raises((MemoryError, RuntimeError)):
        module(xt)
    monkeypatch.undo()


@needs_native
@pytest.mark.parametrize("stage_at,install_at", [
    (1, None), (2, None), (None, 1), (None, 2),
])
def test_transaction_failure_leaves_both_buffers_untouched(
    monkeypatch, live_storages, stage_at, install_at
):
    module = NativeBatchNorm2d(3, momentum=0.5)
    _load(module, running_mean=[0.1, 0.2, 0.3], running_var=[1.5, 2.5, 3.5])
    xt = NativeParameter(np.random.default_rng(131).standard_normal((2, 3, 2, 5)))
    mean_core = module.running_mean._core
    var_core = module.running_var._core
    before = _module_state(module)
    _collect()
    baseline = len(live_storages)

    _run_with_transaction_failure(
        monkeypatch, module, xt, stage_at=stage_at, install_at=install_at
    )

    # Asserted immediately, with no gc.collect() in between.
    _assert_state_unchanged(module, before)
    assert module.running_mean._core is mean_core
    assert module.running_var._core is var_core
    assert mean_core._closed is False and var_core._closed is False
    assert xt.closed is False
    assert len(live_storages) == baseline

    module(xt).close()          # a later valid forward succeeds
    assert not np.array_equal(module.running_mean.to_numpy(), before["mean"])
    assert not np.array_equal(module.running_var.to_numpy(), before["var"])
    xt.close()
    _close_all(module)


@needs_native
def test_update_value_preparation_failure_leaves_running_state_unchanged(
    monkeypatch, live_storages
):
    module = NativeBatchNorm2d(3)
    _load(module, running_mean=[0.4, 0.5, 0.6], running_var=[2.0, 2.0, 2.0])
    xt = NativeParameter(np.random.default_rng(137).standard_normal((2, 3, 2, 5)))
    before = _module_state(module)
    _collect()
    baseline = len(live_storages)

    real_detach = NativeTensor.detach
    calls = {"n": 0}

    def detach(self):
        calls["n"] += 1
        if calls["n"] == 2:
            raise MemoryError("injected preparation failure")
        return real_detach(self)

    monkeypatch.setattr(NativeTensor, "detach", detach)
    with pytest.raises(MemoryError):
        module(xt)
    monkeypatch.undo()

    _assert_state_unchanged(module, before)
    assert len(live_storages) == baseline
    module(xt).close()
    xt.close()
    _close_all(module)


@needs_native
def test_running_update_moves_no_parameter_version():
    module = NativeBatchNorm2d(3)
    xt = NativeParameter(np.random.default_rng(139).standard_normal((2, 3, 2, 5)))
    out = module(xt)
    loss = out.sum()
    versions = (module.gamma.version, module.beta.version)
    other = NativeParameter(np.random.default_rng(140).standard_normal((3, 3, 2, 5)))
    module(other).close()
    assert (module.gamma.version, module.beta.version) == versions
    loss.backward()             # no stale-graph error
    assert xt.grad is not None
    loss.close()
    out.close()
    xt.close()
    other.close()
    _close_all(module)


# ==========================================================================
# Failed-forward cleanup
# ==========================================================================

class _Boom(RuntimeError):
    pass


def _fail_after(monkeypatch, method, successes):
    real = getattr(NativeTensor, method)
    state = {"n": 0}

    def wrapper(self, *args, **kwargs):
        state["n"] += 1
        if state["n"] > successes:
            raise _Boom(f"injected {method} failure #{state['n']}")
        return real(self, *args, **kwargs)

    monkeypatch.setattr(NativeTensor, method, wrapper)
    return state


# Training order for NCHW: three reductions for the mean (axes 0, 2, 3),
# the centering, the squaring, three more reductions for the variance, the
# eps constant, sqrt, reciprocal, the normalizing multiply, then the affine
# round trip (transpose, multiply, add, transpose back, contiguous_copy),
# then the detached statistics and the momentum blend.
_TRAIN_FAILURE_POINTS = [
    ("mean", 0), ("mean", 1), ("mean", 2),          # the batch mean's axes
    ("subtract", 0),                                 # centering
    ("multiply", 0),                                 # squaring
    ("mean", 3), ("mean", 4), ("mean", 5),           # the variance's axes
    ("add", 0), ("sqrt", 0), ("reciprocal", 0),      # eps, sqrt, reciprocal
    ("multiply", 1),                                 # normalize
    ("transpose", 0),                                # NCHW -> NHWC
    ("multiply", 2),                                 # affine scale
    ("add", 1),                                      # affine shift
    ("transpose", 1),                                # NHWC -> NCHW
    ("contiguous_copy", 0),                          # materialize the output
    ("detach", 0), ("detach", 1),                    # graph-free statistics
    ("reshape", 0), ("reshape", 1),                  # flatten to (C,)
    ("multiply", 3), ("multiply", 5),                # momentum blend
    ("add", 2), ("add", 3),
]


@needs_native
@pytest.mark.parametrize("method,successes", _TRAIN_FAILURE_POINTS)
def test_failed_training_forward_changes_nothing_and_leaks_nothing(
    monkeypatch, live_storages, method, successes
):
    module = NativeBatchNorm2d(3, momentum=0.5)
    _load(module, gamma=[1.5, 0.5, -2.0], beta=[0.1, 0.2, 0.3],
          running_mean=[0.7, 0.8, 0.9], running_var=[1.7, 1.8, 1.9])
    xt = NativeParameter(np.random.default_rng(149).standard_normal((2, 3, 2, 5)))
    before = _module_state(module)
    _collect()
    baseline = len(live_storages)

    _fail_after(monkeypatch, method, successes)
    with pytest.raises(_Boom):
        module(xt)
    monkeypatch.undo()

    # No gc.collect() before these assertions: cleanup is deterministic.
    assert len(live_storages) == baseline
    _assert_state_unchanged(module, before)
    assert xt.closed is False and xt.grad is None

    out = module(xt)
    loss = out.multiply(NativeTensor.from_array(np.ones((2, 3, 2, 5)))).sum()
    loss.backward()
    assert xt.grad is not None
    loss.close()
    out.close()
    xt.close()
    _close_all(module)


_EVAL_FAILURE_POINTS = [
    ("reshape", 0), ("contiguous_copy", 0),          # the mean snapshot
    ("reshape", 1), ("contiguous_copy", 1),          # the variance snapshot
    ("add", 0), ("sqrt", 0), ("reciprocal", 0),      # eps, sqrt, reciprocal
    ("subtract", 0), ("multiply", 0),                # centre and normalize
    ("transpose", 0), ("multiply", 1), ("add", 1),   # the affine round trip
    ("transpose", 1), ("contiguous_copy", 2),
]


@needs_native
@pytest.mark.parametrize("method,successes", _EVAL_FAILURE_POINTS)
def test_failed_eval_forward_changes_nothing_and_leaks_nothing(
    monkeypatch, live_storages, method, successes
):
    module = NativeBatchNorm2d(3)
    _load(module, running_mean=[0.4, 0.5, 0.6], running_var=[2.0, 3.0, 4.0])
    module.eval()
    xt = NativeParameter(np.random.default_rng(151).standard_normal((2, 3, 2, 5)))
    before = _module_state(module)
    _collect()
    baseline = len(live_storages)

    _fail_after(monkeypatch, method, successes)
    with pytest.raises(_Boom):
        module(xt)
    monkeypatch.undo()

    assert len(live_storages) == baseline
    _assert_state_unchanged(module, before)
    assert xt.closed is False
    module(xt).close()
    xt.close()
    _close_all(module)


# ==========================================================================
# Non-contiguous NCHW input
# ==========================================================================

@needs_native
@pytest.mark.parametrize("training", [True, False])
def test_non_contiguous_nchw_input(training):
    """A spatial transpose of an ``(N, C, H, H)`` base gives a genuinely
    non-contiguous NCHW view of the same rank and channel count."""
    rng = np.random.default_rng(157)
    base = rng.standard_normal((2, 3, 4, 4))
    swapped = np.transpose(base, (0, 1, 3, 2))

    module = NativeBatchNorm2d(3, eps=1e-3)
    reference_module = NativeBatchNorm2d(3, eps=1e-3)
    for target in (module, reference_module):
        _load(target, gamma=[1.5, -0.5, 2.0], beta=[0.25, 0.0, -0.5],
              running_mean=[0.2, 0.4, 0.6], running_var=[1.5, 2.5, 3.5])
        target.train(training)

    base_t = NativeParameter(base)
    view = base_t.transpose(0, 1, 3, 2)
    assert view.contiguous is False and view.owns_core is False
    assert view.shape == (2, 3, 4, 4)

    out = module(view)
    contiguous_input = NativeTensor.from_array(swapped)
    reference_out = reference_module(contiguous_input)

    # Parity with the contiguous equivalent, and with the oracle.
    assert np.allclose(out.to_numpy(), reference_out.to_numpy(), atol=1e-12)
    if training:
        expected, _, _ = train_reference(
            swapped, np.array([1.5, -0.5, 2.0]), np.array([0.25, 0.0, -0.5]), 1e-3
        )
        assert np.allclose(
            module.running_mean.to_numpy(),
            reference_module.running_mean.to_numpy(), atol=1e-14,
        )
        assert np.allclose(
            module.running_var.to_numpy(),
            reference_module.running_var.to_numpy(), atol=1e-14,
        )
    else:
        expected = eval_reference(
            swapped, np.array([0.2, 0.4, 0.6]), np.array([1.5, 2.5, 3.5]),
            np.array([1.5, -0.5, 2.0]), np.array([0.25, 0.0, -0.5]), 1e-3,
        )
    assert np.allclose(out.to_numpy(), expected, atol=1e-12)
    assert out.owns_core is True and out.contiguous is True
    assert out.shape == (2, 3, 4, 4)

    loss = out.multiply(NativeTensor.from_array(np.ones((2, 3, 4, 4)))).sum()
    loss.backward()
    assert base_t.grad.shape == (2, 3, 4, 4)
    assert np.all(np.isfinite(base_t.grad.to_numpy()))
    # The base and the view both stay usable afterwards.
    assert base_t.closed is False and view.closed is False
    assert np.allclose(view.to_numpy(), swapped, atol=1e-15)

    loss.close()
    out.close()
    reference_out.close()
    contiguous_input.close()
    view.close()
    base_t.close()
    _close_all(module)
    _close_all(reference_module)


# ==========================================================================
# NativeSequential CNN composition
# ==========================================================================

@needs_native
def test_composes_in_a_cnn_sequential():
    model = NativeSequential(
        NativeConv2d(2, 4, kernel_size=3, padding=1, seed=5),
        NativeBatchNorm2d(4),
        NativeReLU(),
        NativeMaxPool2d(kernel_size=2),
        NativeFlatten(),
        NativeLinear(4 * 3 * 2, 3, seed=6),
    )
    assert [name for name, _ in model.named_parameters()] == [
        "0.weight", "0.bias", "1.gamma", "1.beta", "5.weight", "5.bias",
    ]
    assert [name for name, _ in model.named_buffers()] == [
        "1.running_mean", "1.running_var",
    ]
    assert list(model.state_dict()) == [
        "0.weight", "0.bias", "1.gamma", "1.beta", "5.weight", "5.bias",
        "1.running_mean", "1.running_var",
    ]
    for snapshot in model.state_dict().values():
        snapshot.close()

    optimizer = NativeSGD(model.parameters(), lr=0.1)
    assert len(optimizer.parameters()) == 6
    buffer_ids = {id(b) for b in model.buffers()}
    assert not any(id(p) in buffer_ids for p in optimizer.parameters())

    xt = NativeParameter(np.random.default_rng(163).standard_normal((2, 2, 6, 5)))
    out = model(xt)
    assert out.shape == (2, 3)
    loss = out.multiply(NativeTensor.from_array(np.ones((2, 3)))).sum()
    loss.backward()
    # Backward reached the convolution, the BatchNorm, and the linear.
    batchnorm = model._modules["1"]
    conv = model._modules["0"]
    linear = model._modules["5"]
    for parameter in (conv.weight, conv.bias, batchnorm.gamma, batchnorm.beta,
                      linear.weight, linear.bias, xt):
        assert parameter.grad is not None
    assert not np.array_equal(batchnorm.running_mean.to_numpy(), np.zeros(4))

    model.eval()
    assert batchnorm.training is False
    model.train()
    assert batchnorm.training is True

    identities = [id(p) for _, p in model.named_parameters()]
    identities += [id(b) for _, b in model.named_buffers()]
    state = model.state_dict()
    model.load_state_dict(state)
    reloaded = [id(p) for _, p in model.named_parameters()]
    reloaded += [id(b) for _, b in model.named_buffers()]
    assert reloaded == identities
    for snapshot in state.values():
        snapshot.close()

    loss.close()
    out.close()
    xt.close()
    _close_all(model)


@needs_native
def test_cnn_sequential_train_and_eval_outputs_differ():
    model = NativeSequential(
        NativeConv2d(2, 3, kernel_size=3, padding=1, seed=8),
        NativeBatchNorm2d(3),
    )
    rng = np.random.default_rng(167)
    for _ in range(3):
        xt = NativeTensor.from_array(rng.standard_normal((3, 2, 4, 5)))
        model(xt).close()
        xt.close()
    probe = NativeTensor.from_array(rng.standard_normal((2, 2, 4, 5)))
    train_out = model(probe).to_numpy().copy()
    model.eval()
    eval_out = model(probe).to_numpy().copy()
    assert not np.allclose(train_out, eval_out, atol=1e-6)
    assert train_out.shape == (2, 3, 4, 5)
    probe.close()
    _close_all(model)


# ==========================================================================
# State and checkpoints
# ==========================================================================

@needs_native
def test_state_keys_order_and_independence():
    module = NativeBatchNorm2d(3)
    _load(module, running_mean=[1.0, 2.0, 3.0])
    state = module.state_dict()
    assert list(state) == ["gamma", "beta", "running_mean", "running_var"]
    snapshot = state["running_mean"].to_numpy().copy()
    xt = NativeTensor.from_array(
        np.random.default_rng(173).standard_normal((2, 3, 2, 5))
    )
    module(xt).close()
    assert np.array_equal(state["running_mean"].to_numpy(), snapshot)
    assert not np.array_equal(module.running_mean.to_numpy(), snapshot)
    for value in state.values():
        assert value.owns_core and value.requires_grad is False
        value.close()
    xt.close()
    _close_all(module)


@needs_native
def test_load_preserves_identities_and_moves_only_parameter_versions():
    module = NativeBatchNorm2d(3)
    identities = (id(module.gamma), id(module.beta),
                  id(module.running_mean), id(module.running_var))
    versions = (module.gamma.version, module.beta.version)
    _load(module, running_mean=[1.0, 2.0, 3.0], running_var=[4.0, 5.0, 6.0])
    assert (module.gamma.version, module.beta.version) == versions
    _load(module, gamma=[2.0, 2.0, 2.0], beta=[1.0, 1.0, 1.0])
    assert module.gamma.version == versions[0] + 1
    assert module.beta.version == versions[1] + 1
    assert (id(module.gamma), id(module.beta),
            id(module.running_mean), id(module.running_var)) == identities
    _close_all(module)


@needs_native
def test_invalid_mixed_load_is_atomic(live_storages):
    module = NativeBatchNorm2d(3)
    before = _module_state(module)
    _collect()
    baseline = len(live_storages)
    good = NativeTensor.from_array([5.0, 5.0, 5.0])
    bad = NativeTensor.from_array([1.0, 2.0])
    with pytest.raises(ValueError):
        module.load_state_dict(
            {"gamma": good, "running_mean": good, "running_var": bad},
            strict=False,
        )
    _assert_state_unchanged(module, before)
    good.close()
    bad.close()
    _collect()
    assert len(live_storages) == baseline
    _close_all(module)


@needs_native
def test_checkpoint_round_trip_reproduces_state_and_eval_output(tmp_path):
    from tensorforge.experimental import native_checkpoint

    assert native_checkpoint._FORMAT == "tensorforge.native_checkpoint"
    assert native_checkpoint._FORMAT_VERSION == 1

    source = NativeBatchNorm2d(3, eps=1e-3, momentum=0.4)
    _load(source, gamma=[1.1, 1.2, 1.3], beta=[-0.1, -0.2, -0.3])
    rng = np.random.default_rng(179)
    for _ in range(3):
        xt = NativeTensor.from_array(rng.standard_normal((2, 3, 2, 5)))
        source(xt).close()
        xt.close()
    expected = {name: tensor.to_numpy().copy()
                for name, tensor in source._state_named_tensors()}
    source.eval()
    probe = rng.standard_normal((2, 3, 2, 5))
    probe_t = NativeTensor.from_array(probe)
    expected_eval = source(probe_t).to_numpy().copy()

    path = os.path.join(str(tmp_path), "bn2d.npz")
    save_native_checkpoint(path, source, metadata={"milestone": "F4"})

    target = NativeBatchNorm2d(3, eps=1e-3, momentum=0.4)
    identities = (id(target.gamma), id(target.beta),
                  id(target.running_mean), id(target.running_var))
    metadata = load_native_checkpoint(path, target)
    assert metadata == {"milestone": "F4"}
    for name, value in expected.items():
        assert np.array_equal(getattr(target, name).to_numpy(), value), name
    assert (id(target.gamma), id(target.beta),
            id(target.running_mean), id(target.running_var)) == identities
    # Training mode is runtime state, never serialized.
    assert target.training is True
    target.eval()
    assert np.array_equal(target(probe_t).to_numpy(), expected_eval)
    load_native_checkpoint(path, target)
    assert target.training is False

    probe_t.close()
    _close_all(source)
    _close_all(target)


@needs_native
def test_hierarchical_checkpoint_keys_inside_a_sequential(tmp_path):
    model = NativeSequential(
        NativeConv2d(2, 3, kernel_size=3, padding=1, seed=11),
        NativeBatchNorm2d(3),
    )
    xt = NativeTensor.from_array(
        np.random.default_rng(181).standard_normal((2, 2, 4, 5))
    )
    model(xt).close()
    path = os.path.join(str(tmp_path), "cnn.npz")
    save_native_checkpoint(path, model)
    target = NativeSequential(
        NativeConv2d(2, 3, kernel_size=3, padding=1, seed=12),
        NativeBatchNorm2d(3),
    )
    load_native_checkpoint(path, target)
    for name, tensor in model._state_named_tensors():
        assert name in dict(target._state_named_tensors())
        assert np.array_equal(
            dict(target._state_named_tensors())[name].to_numpy(),
            tensor.to_numpy(),
        ), name
    assert "1.running_mean" in dict(target._state_named_tensors())
    xt.close()
    _close_all(model)
    _close_all(target)


# ==========================================================================
# NumPy tripwire
# ==========================================================================

_NUMERICAL_NUMPY = (
    "max", "amax", "argmax", "exp", "log", "sqrt", "reciprocal", "sum",
    "divide", "true_divide", "add", "subtract", "multiply", "matmul",
    "mean", "var", "std", "negative", "power", "square", "copyto",
    "transpose", "moveaxis", "swapaxes",
)
_DATA_NUMPY = ("empty", "frombuffer")


def _arm_numpy_tripwire(monkeypatch):
    def _tripwire(*args, **kwargs):
        raise AssertionError("tensor data was computed or materialized via NumPy")

    for name in _NUMERICAL_NUMPY + _DATA_NUMPY:
        monkeypatch.setattr(np, name, _tripwire)
    monkeypatch.setattr(cpp.NativeTensorCore, "to_numpy", _tripwire)
    monkeypatch.setattr(cpp.NativeTensorCore, "from_array", staticmethod(_tripwire))
    monkeypatch.setattr(cpp.NativeTensorView, "to_numpy", _tripwire)
    monkeypatch.setattr(cpp.NativeStorage, "from_array", staticmethod(_tripwire))
    monkeypatch.setattr(cpp.NativeStorage, "to_numpy", _tripwire)
    monkeypatch.setattr(cpp.NativeStorage, "copy_from", _tripwire)
    monkeypatch.setattr(NativeTensor, "to_numpy", _tripwire)


@needs_native
def test_nchw_training_and_evaluation_use_no_numpy(monkeypatch):
    rng = np.random.default_rng(191)
    shape = (2, 3, 2, 5)
    x = rng.standard_normal(shape)
    upstream = rng.standard_normal(shape)
    module = NativeBatchNorm2d(3, eps=1e-3, momentum=0.4)
    _load(module, gamma=rng.standard_normal(3), beta=rng.standard_normal(3),
          running_mean=[0.2, 0.4, 0.6], running_var=[1.5, 2.5, 3.5])
    gamma = module.gamma.to_numpy().copy()
    beta = module.beta.to_numpy().copy()
    xt = NativeParameter(x)
    up = NativeTensor.from_array(upstream)
    expected_train, batch_mean, batch_var = train_reference(x, gamma, beta, 1e-3)
    expected_mean, expected_var = running_reference(
        np.array([0.2, 0.4, 0.6]), np.array([1.5, 2.5, 3.5]),
        batch_mean, batch_var, 0.4,
    )
    expected_eval = eval_reference(
        x, expected_mean, expected_var, gamma, beta, 1e-3
    )

    _arm_numpy_tripwire(monkeypatch)
    train_out = module(xt)               # NCHW forward + running update
    loss = train_out.multiply(up).sum()  # scalar objective
    loss.backward()                      # native backward
    module.eval()
    eval_out = module(xt)                # NCHW eval forward
    eval_loss = eval_out.multiply(up).sum()
    eval_loss.backward()                 # eval backward
    monkeypatch.undo()

    assert np.allclose(train_out.to_numpy(), expected_train, atol=1e-12)
    assert np.allclose(module.running_mean.to_numpy(), expected_mean, atol=1e-12)
    assert np.allclose(module.running_var.to_numpy(), expected_var, atol=1e-12)
    assert np.allclose(eval_out.to_numpy(), expected_eval, atol=1e-12)
    assert xt.grad is not None
    for tensor in (eval_loss, eval_out, loss, train_out, up, xt):
        tensor.close()
    _close_all(module)


# ==========================================================================
# Ownership and live storage
# ==========================================================================

@needs_native
def test_construction_allocates_exactly_four_storages(live_storages):
    baseline = len(live_storages)
    module = NativeBatchNorm2d(3)
    assert len(live_storages) == baseline + 4
    _close_all(module)
    assert len(live_storages) == baseline


@needs_native
@pytest.mark.parametrize("training", [True, False])
def test_forward_backward_returns_to_baseline(live_storages, training):
    baseline = len(live_storages)
    module = NativeBatchNorm2d(3)
    module.train(training)
    xt = NativeParameter(np.random.default_rng(193).standard_normal((2, 3, 2, 5)))
    out = module(xt)
    loss = out.multiply(NativeTensor.from_array(np.ones((2, 3, 2, 5)))).sum()
    loss.backward()
    for tensor in (loss, out, xt, xt.grad,
                   module.gamma.grad, module.beta.grad):
        tensor.close()
    _close_all(module)
    _collect()
    assert len(live_storages) == baseline


@needs_native
@pytest.mark.parametrize("training", [True, False])
def test_repeated_cycles_do_not_grow_storage(live_storages, training):
    module = NativeBatchNorm2d(3)
    module.train(training)
    _collect()
    baseline = len(live_storages)
    for step in range(6):
        xt = NativeParameter(
            np.random.default_rng(step).standard_normal((2, 3, 2, 5))
        )
        out = module(xt)
        loss = out.multiply(NativeTensor.from_array(np.ones((2, 3, 2, 5)))).sum()
        loss.backward()
        for tensor in (loss, out, xt, xt.grad):
            tensor.close()
        module.zero_grad()
        _collect()
    assert len(live_storages) == baseline


@needs_native
def test_non_contiguous_cycles_do_not_grow_storage(live_storages):
    module = NativeBatchNorm2d(3)
    _collect()
    baseline = len(live_storages)
    for step in range(4):
        base_t = NativeParameter(
            np.random.default_rng(step).standard_normal((2, 3, 4, 4))
        )
        view = base_t.transpose(0, 1, 3, 2)
        out = module(view)
        loss = out.sum()
        loss.backward()
        for tensor in (loss, out, view, base_t, base_t.grad):
            tensor.close()
        module.zero_grad()
        _collect()
    assert len(live_storages) == baseline


@needs_native
def test_state_and_checkpoint_loading_return_to_baseline(live_storages, tmp_path):
    module = NativeBatchNorm2d(3)
    path = os.path.join(str(tmp_path), "own.npz")
    save_native_checkpoint(path, module)
    _collect()
    baseline = len(live_storages)
    for _ in range(3):
        _load(module, running_mean=[1.0, 2.0, 3.0])
        load_native_checkpoint(path, module)
        state = module.state_dict()
        module.load_state_dict(state)
        for value in state.values():
            value.close()
        _collect()
    assert len(live_storages) == baseline
    _close_all(module)


@needs_native
def test_cnn_sequential_returns_to_baseline(live_storages):
    _collect()
    baseline = len(live_storages)
    model = NativeSequential(
        NativeConv2d(2, 3, kernel_size=3, padding=1, seed=1),
        NativeBatchNorm2d(3),
        NativeReLU(),
    )
    for step in range(3):
        xt = NativeParameter(
            np.random.default_rng(step).standard_normal((2, 2, 4, 5))
        )
        out = model(xt)
        loss = out.sum()
        loss.backward()
        for tensor in (loss, out, xt, xt.grad):
            tensor.close()
        model.zero_grad()
        _collect()
    _close_all(model)
    _collect()
    assert len(live_storages) == baseline


@needs_native
def test_closing_state_twice_is_idempotent():
    module = NativeBatchNorm2d(3)
    _close_all(module)
    _close_all(module)
    for tensor in (module.gamma, module.beta,
                   module.running_mean, module.running_var):
        assert tensor.closed is True


# ==========================================================================
# Milestone guardrails
# ==========================================================================

@needs_native
def test_f4_completes_the_public_normalization_module_surface():
    import tensorforge.experimental as experimental

    for module in ("NativeLayerNorm", "NativeBatchNorm1d", "NativeBatchNorm2d"):
        assert module in cpp.NATIVE_MODULES, module
        assert module in experimental.__all__, module
        assert hasattr(experimental, module), module
    assert experimental.NativeBatchNorm2d is NativeBatchNorm2d
    # NativeBatchNorm2d is a *module*, in no other inventory.
    for inventory in (cpp.TENSOR_CORE_OPS, cpp.AUTOGRAD_OPS, cpp.RAW_KERNELS,
                      cpp.NATIVE_LOSSES, cpp.NATIVE_METRICS,
                      cpp.NATIVE_OPTIMIZERS, cpp.STATE_SUPPORT,
                      cpp.UNSUPPORTED):
        assert "NativeBatchNorm2d" not in inventory
    # Both normalization capability names have now left UNSUPPORTED...
    assert "batchnorm" not in cpp.UNSUPPORTED
    assert "layernorm" not in cpp.UNSUPPORTED
    # ...and the remaining boundary is exactly what it was.
    assert cpp.UNSUPPORTED == ("dropout", "float32", "cuda", "amp")
    # BatchNorm3d was never in scope.
    assert not hasattr(experimental, "NativeBatchNorm3d")
    assert "NativeBatchNorm3d" not in cpp.NATIVE_MODULES


@needs_native
def test_f4_adds_no_operation_core_method_kernel_or_abi_symbol():
    for name in ("batch_norm", "batchnorm", "batch_norm_forward",
                 "batch_norm_backward", "layer_norm", "normalize"):
        assert name not in cpp.TENSOR_CORE_OPS, name
        assert name not in cpp.AUTOGRAD_OPS, name
        assert name not in cpp.RAW_KERNELS, name
        assert not hasattr(cpp.NativeTensorCore, name), name
        assert not hasattr(NativeTensor, name), name
    for symbol in ("tf_core_batch_norm", "tf_core_batch_norm_forward",
                   "tf_core_batch_norm_backward", "tf_core_layer_norm"):
        assert symbol not in cpp._CHECKED_KERNELS, symbol
    assert cpp.SUPPORTED_DTYPES == ("float64",)
    assert cpp.SUPPORTED_DEVICES == ("cpu",)
    assert cpp.NATIVE_LOSSES == ("NativeMSELoss", "NativeCrossEntropyLoss")
    assert cpp.NATIVE_METRICS == ("native_accuracy",)
    assert cpp.NATIVE_OPTIMIZERS == ("NativeSGD", "NativeAdam")
    assert cpp.STATE_SUPPORT == (
        "persistent_buffers", "state_dict", "load_state_dict",
        "save_native_checkpoint", "load_native_checkpoint",
    )
    # No custom BatchNorm backward and no graph-node construction in the
    # module: backward is entirely the existing composed autograd.
    source = open(native_batchnorm.__file__, encoding="utf-8").read()
    assert "_from_op(" not in source
    assert "def _backward" not in source


@needs_native
def test_no_channels_last_public_mode_was_introduced():
    """The affine round trip is an internal step, not a layout option."""
    import tensorforge.experimental as experimental
    import inspect

    assert "channels_last" not in inspect.signature(
        NativeBatchNorm2d.__init__
    ).parameters
    module = NativeBatchNorm2d(3)
    for name in ("channels_last", "to_channels_last", "memory_format",
                 "layout", "data_format"):
        assert not hasattr(module, name), name
        assert not hasattr(experimental, name), name
    # The permutation is private configuration, and only that.
    assert NativeBatchNorm2d._CHANNELS_LAST == (0, 2, 3, 1)
    _close_all(module)
