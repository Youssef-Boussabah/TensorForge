"""Phase F, milestone F3 — NativeBatchNorm1d.

The first **stateful** native numerical module: ``(N, C)`` batch
normalization with differentiable training statistics, persistent native
``running_mean``/``running_var`` buffers advanced by a graph-free atomic
two-buffer transaction, and graph-safe immutable snapshots in evaluation
mode.

These tests are behavioral — values, gradients, shapes, identities,
versions, storage lifetime — rather than assertions about the
composition's internal shape. Nothing here is a normalization *kernel*, C
ABI symbol, ``NativeTensorCore`` method, or ``NativeTensor`` operation:
F3 adds a module, and the guardrails at the end pin exactly that.
"""

import gc
import os

import numpy as np
import pytest

import tensorforge as tf
from tensorforge.backends import cpp
from tensorforge.experimental import (
    NativeAdam, NativeBatchNorm1d, NativeLinear, NativeModule, NativeParameter,
    NativeReLU, NativeSGD, NativeSequential, NativeTensor,
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
    (it hooks close()), so a test may force ``gc.collect()`` to a defined
    point and still read a truthful count."""
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
    deterministic collection point. The live-storage count is exact
    regardless, so the baseline assertion still proves nothing leaked."""
    gc.collect()


def train_reference(x, gamma, beta, eps):
    """The population-variance training-mode BatchNorm, in NumPy — an
    external oracle, never run inside an armed tripwire. Returns the
    output plus the ``(C,)`` batch statistics that drove it."""
    x = np.asarray(x, dtype=np.float64)
    mean = x.mean(axis=0, keepdims=True)
    var = ((x - mean) ** 2).mean(axis=0, keepdims=True)
    out = (x - mean) / np.sqrt(var + eps) * gamma + beta
    return out, mean.ravel(), var.ravel()


def eval_reference(x, running_mean, running_var, gamma, beta, eps):
    x = np.asarray(x, dtype=np.float64)
    return (x - running_mean) / np.sqrt(running_var + eps) * gamma + beta


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
    """Everything a failed forward or a failed transaction must leave
    untouched, as plain comparable data plus object identities."""
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
    """Every object reachable from ``root`` through the autograd graph:
    the node itself, its parents transitively, and every native object a
    node's history owns (``_graph_resources``). This is the structural
    walk the §7 snapshot rule is proved against."""
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
    """The ids of every native **storage** the graph can reach.

    Stronger than the object-identity walk on its own: a borrowing view
    of a registered buffer is a *different* Python object but the *same*
    storage, and §7 forbids that just as firmly — the graph must depend
    on no byte the running buffers own."""
    ids = set()
    for obj in _graph_objects(root).values():
        if isinstance(obj, NativeTensor) and not obj.closed:
            ids.add(id(obj._core.storage))
    return ids


def _central_difference(f, base, h=1e-6):
    """Central-difference gradient of scalar ``f`` at ``base`` (an
    ndarray), returned with ``base``'s shape."""
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
    """The §9 consequence of there being no ``NativeModule.close()``: a
    stateful module's owner releases **both** its parameters and its
    buffers explicitly."""
    for tensor in module.parameters():
        tensor.close()
    for tensor in module.buffers():
        tensor.close()


# ==========================================================================
# Constructor validation
# ==========================================================================

@needs_native
@pytest.mark.parametrize("num_features", [1, 2, 7])
def test_accepts_positive_num_features(num_features):
    module = NativeBatchNorm1d(num_features)
    assert module.num_features == num_features
    assert module.eps == 1e-5
    assert module.momentum == 0.1
    _close_all(module)


@needs_native
@pytest.mark.parametrize("eps", [1e-3, 0.5, 2])
def test_accepts_positive_eps(eps):
    module = NativeBatchNorm1d(3, eps=eps)
    assert module.eps == float(eps)
    assert isinstance(module.eps, float)
    _close_all(module)


@needs_native
@pytest.mark.parametrize("momentum", [0.0, 1.0, 0.25, 0.9, 1])
def test_accepts_momentum_in_the_closed_unit_range(momentum):
    module = NativeBatchNorm1d(3, momentum=momentum)
    assert module.momentum == float(momentum)
    assert isinstance(module.momentum, float)
    _close_all(module)


@needs_native
@pytest.mark.parametrize("bad", [True, False, 3.0, "3", None, np.int64(3), 3 + 0j])
def test_rejects_non_int_num_features(bad, live_storages):
    baseline = len(live_storages)
    with pytest.raises(TypeError):
        NativeBatchNorm1d(bad)
    assert len(live_storages) == baseline


@needs_native
@pytest.mark.parametrize("bad", [0, -1, -7])
def test_rejects_non_positive_num_features(bad, live_storages):
    baseline = len(live_storages)
    with pytest.raises(ValueError):
        NativeBatchNorm1d(bad)
    assert len(live_storages) == baseline


@needs_native
@pytest.mark.parametrize("bad", [True, False, "1e-5", None, [1e-5], 1 + 2j])
def test_rejects_non_real_eps(bad, live_storages):
    baseline = len(live_storages)
    with pytest.raises(TypeError):
        NativeBatchNorm1d(3, eps=bad)
    assert len(live_storages) == baseline


@needs_native
@pytest.mark.parametrize("bad", [0, 0.0, -1e-5, -1])
def test_rejects_non_positive_eps(bad, live_storages):
    baseline = len(live_storages)
    with pytest.raises(ValueError):
        NativeBatchNorm1d(3, eps=bad)
    assert len(live_storages) == baseline


@needs_native
@pytest.mark.parametrize("bad", [True, False, "0.1", None, (0.1,), 0.1 + 0j])
def test_rejects_non_real_momentum(bad, live_storages):
    baseline = len(live_storages)
    with pytest.raises(TypeError):
        NativeBatchNorm1d(3, momentum=bad)
    assert len(live_storages) == baseline


@needs_native
@pytest.mark.parametrize("bad", [-0.1, 1.1, 2, -1, float("nan")])
def test_rejects_out_of_range_or_nan_momentum(bad, live_storages):
    baseline = len(live_storages)
    with pytest.raises(ValueError):
        NativeBatchNorm1d(3, momentum=bad)
    assert len(live_storages) == baseline


@needs_native
def test_rejected_construction_leaves_no_partial_module(live_storages):
    """Validation precedes allocation, so a rejected call creates no
    parameter, no buffer, and no registration at all."""
    baseline = len(live_storages)
    for call in (
        lambda: NativeBatchNorm1d(0),
        lambda: NativeBatchNorm1d(3, eps=-1.0),
        lambda: NativeBatchNorm1d(3, momentum=1.5),
        lambda: NativeBatchNorm1d(True),
    ):
        with pytest.raises((TypeError, ValueError)):
            call()
        assert len(live_storages) == baseline


@needs_native
def test_constructor_rejects_extra_arguments():
    """The public constructor is exactly (num_features, eps, momentum):
    no affine, track_running_stats, dtype, device, seed, or training."""
    for kwargs in ({"affine": False}, {"track_running_stats": False},
                   {"dtype": "float64"}, {"device": "cpu"},
                   {"requires_grad": False}, {"seed": 1},
                   {"unbiased": False}, {"axis": 0}, {"training": True}):
        with pytest.raises(TypeError):
            NativeBatchNorm1d(3, **kwargs)
    with pytest.raises(TypeError):
        NativeBatchNorm1d(3, 1e-5, 0.1, 7)


# ==========================================================================
# Initialization and registration
# ==========================================================================

@needs_native
def test_parameters_and_buffers_initialize_exactly():
    module = NativeBatchNorm1d(4)
    assert np.array_equal(module.gamma.to_numpy(), np.ones(4))
    assert np.array_equal(module.beta.to_numpy(), np.zeros(4))
    assert np.array_equal(module.running_mean.to_numpy(), np.zeros(4))
    assert np.array_equal(module.running_var.to_numpy(), np.ones(4))
    for tensor in (module.gamma, module.beta,
                   module.running_mean, module.running_var):
        assert tensor.shape == (4,)
        assert tensor.owns_core is True
        assert tensor.contiguous is True
        assert tensor.dtype == "float64" and tensor.device == "cpu"
    assert isinstance(module.gamma, NativeParameter)
    assert isinstance(module.beta, NativeParameter)
    assert type(module.running_mean) is NativeTensor
    assert type(module.running_var) is NativeTensor
    assert module.gamma.requires_grad is True
    assert module.beta.requires_grad is True
    assert module.running_mean.requires_grad is False
    assert module.running_var.requires_grad is False
    assert module.gamma.version == 0 and module.beta.version == 0
    assert module.gamma.is_leaf and module.beta.is_leaf
    assert module.running_mean.is_leaf and module.running_var.is_leaf
    _close_all(module)


@needs_native
def test_registration_and_state_order_is_exact():
    module = NativeBatchNorm1d(3)
    assert [name for name, _ in module.named_parameters()] == ["gamma", "beta"]
    assert [name for name, _ in module.named_buffers()] == [
        "running_mean", "running_var"
    ]
    assert list(module.state_dict()) == [
        "gamma", "beta", "running_mean", "running_var"
    ]
    for snapshot in module.state_dict().values():
        snapshot.close()
    parameters = module.parameters()
    buffers = module.buffers()
    assert [id(p) for p in parameters] == [id(module.gamma), id(module.beta)]
    assert [id(b) for b in buffers] == [
        id(module.running_mean), id(module.running_var)
    ]
    # Neither category leaks into the other.
    for buffer in buffers:
        assert all(id(buffer) != id(p) for p in parameters)
    _close_all(module)


@needs_native
def test_optimizers_discover_only_the_affine_parameters():
    module = NativeBatchNorm1d(3)
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
def test_buffer_identities_survive_updates_and_loads():
    module = NativeBatchNorm1d(3)
    mean_id = id(module.running_mean)
    var_id = id(module.running_var)
    x = NativeTensor.from_array(np.random.default_rng(0).standard_normal((4, 3)))
    module(x).close()
    assert id(module.running_mean) == mean_id
    assert id(module.running_var) == var_id
    _load(module, running_mean=[1.0, 2.0, 3.0], running_var=[2.0, 3.0, 4.0])
    assert id(module.running_mean) == mean_id
    assert id(module.running_var) == var_id
    x.close()
    _close_all(module)


@needs_native
def test_repr_is_deterministic_and_metadata_only():
    module = NativeBatchNorm1d(5, eps=1e-3, momentum=0.25)
    text = repr(module)
    assert text == "NativeBatchNorm1d(num_features=5, eps=0.001, momentum=0.25)"
    assert repr(module) == text
    module.eval()
    assert repr(module) == text          # training state is not in the repr
    x = NativeTensor.from_array(np.zeros((2, 5)))
    module.train()
    module(x).close()
    assert repr(module) == text          # running values are not either
    assert "0x" not in text
    x.close()
    _close_all(module)


# ==========================================================================
# Shape and input validation
# ==========================================================================

@needs_native
@pytest.mark.parametrize("batch", [1, 2, 5, 17])
def test_accepts_any_positive_batch_size(batch):
    module = NativeBatchNorm1d(3)
    x = NativeTensor.from_array(
        np.random.default_rng(batch).standard_normal((batch, 3))
    )
    out = module(x)
    assert out.shape == (batch, 3)
    out.close()
    x.close()
    _close_all(module)


@needs_native
@pytest.mark.parametrize("shape", [(), (6,), (2, 3, 1), (1, 2, 3, 1)])
def test_rejects_wrong_rank(shape, live_storages):
    module = NativeBatchNorm1d(3)
    x = NativeTensor.from_array(np.zeros(shape) if shape else np.zeros(()))
    baseline = len(live_storages)
    with pytest.raises(ValueError) as error:
        module(x)
    assert "3" in str(error.value) and str(tuple(x.shape)) in str(error.value)
    assert len(live_storages) == baseline
    x.close()
    _close_all(module)


@needs_native
def test_rejects_wrong_feature_count(live_storages):
    module = NativeBatchNorm1d(3)
    x = NativeTensor.from_array(np.zeros((4, 5)))
    baseline = len(live_storages)
    with pytest.raises(ValueError) as error:
        module(x)
    assert "3" in str(error.value) and "(4, 5)" in str(error.value)
    assert len(live_storages) == baseline
    x.close()
    _close_all(module)


@needs_native
def test_rejects_right_element_count_with_wrong_shape(live_storages):
    """12 elements, but (2, 2, 3) is not ``(N, 3)`` — the rank is part of
    the contract, never silently reinterpreted."""
    module = NativeBatchNorm1d(3)
    x = NativeTensor.from_array(np.zeros((2, 2, 3)))
    baseline = len(live_storages)
    with pytest.raises(ValueError):
        module(x)
    assert len(live_storages) == baseline
    x.close()
    _close_all(module)


@needs_native
@pytest.mark.parametrize("bad", [
    np.zeros((2, 3)), [[0.0, 0.0, 0.0]], (0.0, 0.0, 0.0), 1.0, None, object(),
])
def test_rejects_non_native_input(bad, live_storages):
    module = NativeBatchNorm1d(3)
    baseline = len(live_storages)
    with pytest.raises(TypeError):
        module(bad)
    assert len(live_storages) == baseline
    _close_all(module)


@needs_native
def test_rejects_stable_framework_tensor(live_storages):
    module = NativeBatchNorm1d(3)
    baseline = len(live_storages)
    with pytest.raises(TypeError):
        module(tf.Tensor(np.zeros((2, 3))))
    assert len(live_storages) == baseline
    _close_all(module)


@needs_native
def test_accepts_a_native_parameter_input():
    module = NativeBatchNorm1d(3)
    x = NativeParameter(np.random.default_rng(1).standard_normal((4, 3)))
    out = module(x)
    assert isinstance(out, NativeTensor) and not isinstance(out, NativeParameter)
    out.close()
    x.close()
    _close_all(module)


@needs_native
def test_rejects_closed_input(live_storages):
    module = NativeBatchNorm1d(3)
    x = NativeTensor.from_array(np.zeros((2, 3)))
    x.close()
    baseline = len(live_storages)
    with pytest.raises(RuntimeError):
        module(x)
    assert len(live_storages) == baseline
    _close_all(module)


@needs_native
@pytest.mark.parametrize("name", ["gamma", "beta", "running_mean", "running_var"])
def test_rejects_closed_state(name, live_storages):
    module = NativeBatchNorm1d(3)
    getattr(module, name).close()
    x = NativeTensor.from_array(np.zeros((2, 3)))
    baseline = len(live_storages)
    with pytest.raises(RuntimeError) as error:
        module(x)
    assert name in str(error.value)
    assert len(live_storages) == baseline  # no graph node was built
    x.close()
    _close_all(module)


@needs_native
@pytest.mark.parametrize("name", ["running_mean", "running_var"])
def test_rejects_corrupted_running_buffer_shape(name, live_storages):
    """A private test seam simulates state corruption: the registered
    buffer is swapped for a wrongly shaped one. Forward must refuse before
    building anything rather than broadcasting silently."""
    module = NativeBatchNorm1d(3)
    wrong = NativeTensor.zeros((5,))
    module._buffers[name] = module._buffers[name]._replace(tensor=wrong)
    x = NativeTensor.from_array(np.zeros((2, 3)))
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
def test_rejects_unregistered_running_buffer(live_storages):
    module = NativeBatchNorm1d(3)
    orphan = module.running_var
    module.register_buffer("running_var", None)
    x = NativeTensor.from_array(np.zeros((2, 3)))
    baseline = len(live_storages)
    with pytest.raises(RuntimeError) as error:
        module(x)
    assert "running_var" in str(error.value)
    assert len(live_storages) == baseline
    orphan.close()
    x.close()
    module.gamma.close()
    module.beta.close()
    module.running_mean.close()


@needs_native
def test_validation_mutates_nothing(live_storages):
    module = NativeBatchNorm1d(3)
    before = _module_state(module)
    baseline = len(live_storages)
    for bad in (NativeTensor.from_array(np.zeros((2, 4))), "not a tensor"):
        with pytest.raises((TypeError, ValueError)):
            module(bad)
        if isinstance(bad, NativeTensor):
            bad.close()
    _assert_state_unchanged(module, before)
    assert len(live_storages) == baseline
    _close_all(module)


# ==========================================================================
# Training-mode numerical parity
# ==========================================================================

@needs_native
@pytest.mark.parametrize("batch,features", [(1, 1), (2, 3), (5, 4), (9, 2)])
def test_training_parity_with_the_stable_batchnorm(batch, features):
    rng = np.random.default_rng(batch * 31 + features)
    x = rng.standard_normal((batch, features)) * 3.0 + 1.5
    module = NativeBatchNorm1d(features)
    reference = tf.nn.BatchNorm1d(features)
    xt = NativeTensor.from_array(x)
    out = module(xt)
    expected = reference(tf.Tensor(x))
    assert np.allclose(out.to_numpy(), expected.data, atol=1e-12)
    assert np.allclose(module.running_mean.to_numpy(),
                       reference.running_mean, atol=1e-12)
    assert np.allclose(module.running_var.to_numpy(),
                       reference.running_var, atol=1e-12)
    out.close()
    xt.close()
    _close_all(module)


@needs_native
def test_training_parity_with_nondefault_configuration():
    rng = np.random.default_rng(7)
    x = rng.standard_normal((6, 3))
    gamma = rng.standard_normal(3)
    beta = rng.standard_normal(3)
    module = NativeBatchNorm1d(3, eps=1e-2, momentum=0.3)
    _load(module, gamma=gamma, beta=beta,
          running_mean=[0.4, -0.7, 1.2], running_var=[2.0, 0.5, 3.0])
    xt = NativeTensor.from_array(x)
    out = module(xt)
    expected, batch_mean, batch_var = train_reference(x, gamma, beta, 1e-2)
    assert np.allclose(out.to_numpy(), expected, atol=1e-12)
    new_mean, new_var = running_reference(
        np.array([0.4, -0.7, 1.2]), np.array([2.0, 0.5, 3.0]),
        batch_mean, batch_var, 0.3,
    )
    assert np.allclose(module.running_mean.to_numpy(), new_mean, atol=1e-12)
    assert np.allclose(module.running_var.to_numpy(), new_var, atol=1e-12)
    out.close()
    xt.close()
    _close_all(module)


@needs_native
def test_training_uses_population_variance_and_eps_inside_the_root():
    """Hand-computed: column [1, 3] has mean 2 and **population**
    variance 1 (the sample variance would be 2). With eps=3 the
    normalizer is sqrt(1 + 3) = 2 — never sqrt(1) + 3 = 4."""
    module = NativeBatchNorm1d(1, eps=3.0, momentum=1.0)
    xt = NativeTensor.from_array([[1.0], [3.0]])
    out = module(xt)
    assert np.allclose(out.to_numpy(), [[-0.5], [0.5]], atol=1e-15)
    # The same population variance drives the running update.
    assert np.allclose(module.running_var.to_numpy(), [1.0], atol=1e-15)
    assert np.allclose(module.running_mean.to_numpy(), [2.0], atol=1e-15)
    out.close()
    xt.close()
    _close_all(module)


@needs_native
def test_constant_feature_column_is_finite_and_matches_the_reference():
    x = np.array([[2.0, 1.0], [2.0, 5.0], [2.0, -3.0]])
    module = NativeBatchNorm1d(2, eps=1e-4)
    xt = NativeTensor.from_array(x)
    out = module(xt)
    expected, _, batch_var = train_reference(x, np.ones(2), np.zeros(2), 1e-4)
    assert np.allclose(out.to_numpy(), expected, atol=1e-12)
    assert np.all(np.isfinite(out.to_numpy()))
    assert batch_var[0] == 0.0                    # the constant column
    assert np.allclose(out.to_numpy()[:, 0], 0.0, atol=1e-15)
    out.close()
    xt.close()
    _close_all(module)


@needs_native
@pytest.mark.parametrize("momentum", [0.0, 1.0, 0.1, 0.37, 0.5])
def test_running_update_follows_the_exact_momentum_convention(momentum):
    rng = np.random.default_rng(int(momentum * 1000) + 5)
    x = rng.standard_normal((5, 3)) * 2.0 - 1.0
    start_mean = np.array([0.25, -1.5, 3.0])
    start_var = np.array([2.5, 0.75, 1.0])
    module = NativeBatchNorm1d(3, momentum=momentum)
    _load(module, running_mean=start_mean, running_var=start_var)
    xt = NativeTensor.from_array(x)
    module(xt).close()
    _, batch_mean, batch_var = train_reference(x, np.ones(3), np.zeros(3), 1e-5)
    expected_mean, expected_var = running_reference(
        start_mean, start_var, batch_mean, batch_var, momentum
    )
    assert np.allclose(module.running_mean.to_numpy(), expected_mean, atol=1e-14)
    assert np.allclose(module.running_var.to_numpy(), expected_var, atol=1e-14)
    xt.close()
    _close_all(module)


@needs_native
def test_momentum_zero_leaves_the_running_values_numerically_equal():
    module = NativeBatchNorm1d(3, momentum=0.0)
    _load(module, running_mean=[0.5, -0.5, 2.0], running_var=[1.5, 3.0, 0.25])
    before_mean = module.running_mean.to_numpy().copy()
    before_var = module.running_var.to_numpy().copy()
    xt = NativeTensor.from_array(
        np.random.default_rng(2).standard_normal((6, 3)) * 10
    )
    for _ in range(3):
        module(xt).close()
        assert np.array_equal(module.running_mean.to_numpy(), before_mean)
        assert np.array_equal(module.running_var.to_numpy(), before_var)
    xt.close()
    _close_all(module)


@needs_native
def test_momentum_one_replaces_with_the_current_batch_statistics():
    module = NativeBatchNorm1d(3, momentum=1.0)
    _load(module, running_mean=[9.0, 9.0, 9.0], running_var=[9.0, 9.0, 9.0])
    x = np.random.default_rng(11).standard_normal((7, 3))
    xt = NativeTensor.from_array(x)
    module(xt).close()
    _, batch_mean, batch_var = train_reference(x, np.ones(3), np.zeros(3), 1e-5)
    assert np.allclose(module.running_mean.to_numpy(), batch_mean, atol=1e-15)
    assert np.allclose(module.running_var.to_numpy(), batch_var, atol=1e-15)
    xt.close()
    _close_all(module)


@needs_native
def test_consecutive_training_forwards_accumulate_exactly():
    rng = np.random.default_rng(13)
    batches = [rng.standard_normal((4, 3)) for _ in range(5)]
    module = NativeBatchNorm1d(3, momentum=0.2)
    reference = tf.nn.BatchNorm1d(3, momentum=0.2)
    running_mean = np.zeros(3)
    running_var = np.ones(3)
    for batch in batches:
        xt = NativeTensor.from_array(batch)
        module(xt).close()
        xt.close()
        reference(tf.Tensor(batch))
        _, batch_mean, batch_var = train_reference(
            batch, np.ones(3), np.zeros(3), 1e-5
        )
        running_mean, running_var = running_reference(
            running_mean, running_var, batch_mean, batch_var, 0.2
        )
        assert np.allclose(module.running_mean.to_numpy(), running_mean, atol=1e-12)
        assert np.allclose(module.running_var.to_numpy(), running_var, atol=1e-12)
        assert np.allclose(module.running_mean.to_numpy(),
                           reference.running_mean, atol=1e-12)
        assert np.allclose(module.running_var.to_numpy(),
                           reference.running_var, atol=1e-12)
    _close_all(module)


# ==========================================================================
# Evaluation-mode numerical parity
# ==========================================================================

@needs_native
def test_eval_uses_the_stored_running_statistics():
    rng = np.random.default_rng(17)
    gamma, beta = rng.standard_normal(3), rng.standard_normal(3)
    running_mean = np.array([0.3, -0.2, 1.1])
    running_var = np.array([2.0, 0.5, 1.25])
    module = NativeBatchNorm1d(3, eps=1e-3)
    _load(module, gamma=gamma, beta=beta,
          running_mean=running_mean, running_var=running_var)
    module.eval()
    x = rng.standard_normal((4, 3)) * 4 + 2
    xt = NativeTensor.from_array(x)
    out = module(xt)
    expected = eval_reference(x, running_mean, running_var, gamma, beta, 1e-3)
    assert np.allclose(out.to_numpy(), expected, atol=1e-12)
    # Not the current batch's own statistics.
    batch_expected, _, _ = train_reference(x, gamma, beta, 1e-3)
    assert not np.allclose(out.to_numpy(), batch_expected, atol=1e-6)
    out.close()
    xt.close()
    _close_all(module)


@needs_native
def test_eval_parity_with_the_stable_batchnorm():
    rng = np.random.default_rng(19)
    module = NativeBatchNorm1d(3)
    reference = tf.nn.BatchNorm1d(3)
    for _ in range(3):
        batch = rng.standard_normal((5, 3))
        xt = NativeTensor.from_array(batch)
        module(xt).close()
        xt.close()
        reference(tf.Tensor(batch))
    module.eval()
    reference.eval()
    probe = rng.standard_normal((2, 3))
    xt = NativeTensor.from_array(probe)
    out = module(xt)
    assert np.allclose(out.to_numpy(), reference(tf.Tensor(probe)).data, atol=1e-12)
    out.close()
    xt.close()
    _close_all(module)


@needs_native
def test_eval_never_mutates_the_running_buffers():
    module = NativeBatchNorm1d(3)
    _load(module, running_mean=[1.0, 2.0, 3.0], running_var=[4.0, 5.0, 6.0])
    module.eval()
    before = _module_state(module)
    xt = NativeTensor.from_array(
        np.random.default_rng(23).standard_normal((6, 3)) * 100
    )
    for _ in range(4):
        module(xt).close()
        _assert_state_unchanged(module, before)
    xt.close()
    _close_all(module)


@needs_native
def test_train_and_eval_differ_when_their_statistics_differ():
    module = NativeBatchNorm1d(3)
    _load(module, running_mean=[5.0, -5.0, 0.0], running_var=[9.0, 4.0, 1.0])
    x = np.random.default_rng(29).standard_normal((5, 3))
    xt = NativeTensor.from_array(x)
    module.eval()
    eval_out = module(xt).to_numpy()
    module.train()
    train_out = module(xt).to_numpy()
    assert not np.allclose(eval_out, train_out, atol=1e-6)
    xt.close()
    _close_all(module)


@needs_native
def test_mode_toggles_change_no_state():
    module = NativeBatchNorm1d(3)
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


@needs_native
def test_eval_single_sample_normalizes_consistently():
    """A batch of one has zero variance, so only eval mode gives a
    meaningful answer — the reason running statistics exist."""
    module = NativeBatchNorm1d(2)
    _load(module, running_mean=[1.0, -1.0], running_var=[4.0, 9.0])
    module.eval()
    xt = NativeTensor.from_array([[3.0, 2.0]])
    out = module(xt)
    expected = eval_reference(
        np.array([[3.0, 2.0]]), np.array([1.0, -1.0]), np.array([4.0, 9.0]),
        np.ones(2), np.zeros(2), 1e-5,
    )
    assert np.allclose(out.to_numpy(), expected, atol=1e-12)
    out.close()
    xt.close()
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
@pytest.mark.parametrize("batch,features,eps", [
    (4, 3, 1e-5), (2, 2, 1e-2), (6, 1, 1e-5), (3, 5, 0.25),
])
def test_training_gradients_match_central_differences(batch, features, eps):
    rng = np.random.default_rng(batch * 101 + features)
    x = rng.standard_normal((batch, features))
    gamma = rng.standard_normal(features) + 1.0
    beta = rng.standard_normal(features)
    upstream = rng.standard_normal((batch, features))

    module = NativeBatchNorm1d(features, eps=eps)
    _load(module, gamma=gamma, beta=beta)
    xt = NativeParameter(x)
    out = module(xt)
    loss = out.multiply(NativeTensor.from_array(upstream)).sum()
    loss.backward()

    expected_x = _central_difference(
        lambda probe: _train_objective(probe, gamma, beta, eps, upstream), x
    )
    expected_gamma = _central_difference(
        lambda probe: _train_objective(x, probe, beta, eps, upstream), gamma
    )
    expected_beta = _central_difference(
        lambda probe: _train_objective(x, gamma, probe, eps, upstream), beta
    )
    assert np.allclose(xt.grad.to_numpy(), expected_x, atol=1e-6)
    assert np.allclose(module.gamma.grad.to_numpy(), expected_gamma, atol=1e-6)
    assert np.allclose(module.beta.grad.to_numpy(), expected_beta, atol=1e-6)
    assert xt.grad.shape == (batch, features)
    assert module.gamma.grad.shape == (features,)   # reduced over the batch
    assert module.beta.grad.shape == (features,)
    for tensor in (xt.grad, module.gamma.grad, module.beta.grad):
        assert tensor.dtype == "float64" and tensor.device == "cpu"
    loss.close()
    out.close()
    xt.close()
    _close_all(module)


@needs_native
def test_training_gradient_flows_through_the_batch_statistics():
    """A detached-statistics implementation would give ``upstream *
    gamma / std`` for the input gradient. The real gradient differs, and
    (with a constant upstream) sums to ~0 down each column — the
    signature of differentiating through the mean and the variance."""
    rng = np.random.default_rng(41)
    x = rng.standard_normal((5, 3))
    module = NativeBatchNorm1d(3)
    xt = NativeParameter(x)
    out = module(xt)
    upstream = np.ones((5, 3))
    loss = out.multiply(NativeTensor.from_array(upstream)).sum()
    loss.backward()
    grad = xt.grad.to_numpy()
    _, _, batch_var = train_reference(x, np.ones(3), np.zeros(3), 1e-5)
    detached = upstream / np.sqrt(batch_var + 1e-5)
    assert not np.allclose(grad, detached, atol=1e-6)
    assert np.allclose(grad.sum(axis=0), 0.0, atol=1e-9)
    loss.close()
    out.close()
    xt.close()
    _close_all(module)


@needs_native
def test_training_gradient_of_a_near_constant_column_is_finite():
    x = np.array([[1.0, 0.5], [1.0, -0.5], [1.0 + 1e-8, 2.0]])
    module = NativeBatchNorm1d(2, eps=1e-2)
    xt = NativeParameter(x)
    out = module(xt)
    loss = out.multiply(NativeTensor.from_array(np.ones((3, 2)))).sum()
    loss.backward()
    assert np.all(np.isfinite(xt.grad.to_numpy()))
    assert np.all(np.isfinite(module.gamma.grad.to_numpy()))
    expected = _central_difference(
        lambda probe: _train_objective(probe, np.ones(2), np.zeros(2), 1e-2,
                                       np.ones((3, 2))),
        x, h=1e-6,
    )
    assert np.allclose(xt.grad.to_numpy(), expected, atol=1e-5)
    loss.close()
    out.close()
    xt.close()
    _close_all(module)


@needs_native
def test_training_finite_differences_through_the_native_module_itself():
    """The reference-based probes above are the primary check; this one
    differentiates the **native** forward directly. Each probe runs on a
    freshly built module loaded with the same state, so the running
    statistics a training forward advances never leak between
    evaluations."""
    rng = np.random.default_rng(43)
    x = rng.standard_normal((4, 3))
    gamma = rng.standard_normal(3) + 1.0
    beta = rng.standard_normal(3)
    upstream = rng.standard_normal((4, 3))

    def native_objective(probe):
        module = NativeBatchNorm1d(3, eps=1e-3)
        _load(module, gamma=gamma, beta=beta)
        xt = NativeTensor.from_array(probe)
        out = module(xt)
        up = NativeTensor.from_array(upstream)
        loss = out.multiply(up).sum()
        value = float(loss.to_numpy())
        for tensor in (loss, out, up, xt):
            tensor.close()
        _close_all(module)
        return value

    expected = _central_difference(native_objective, x)

    module = NativeBatchNorm1d(3, eps=1e-3)
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
@pytest.mark.parametrize("batch,features", [(3, 2), (5, 4)])
def test_eval_gradients_match_central_differences(batch, features):
    rng = np.random.default_rng(batch * 7 + features)
    x = rng.standard_normal((batch, features))
    gamma = rng.standard_normal(features) + 1.0
    beta = rng.standard_normal(features)
    running_mean = rng.standard_normal(features)
    running_var = np.abs(rng.standard_normal(features)) + 0.5
    upstream = rng.standard_normal((batch, features))

    module = NativeBatchNorm1d(features, eps=1e-3)
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

    assert np.allclose(xt.grad.to_numpy(),
                       _central_difference(lambda p: objective(probe_x=p), x),
                       atol=1e-6)
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
    # The running buffers never receive a gradient.
    assert module.running_mean.grad is None
    assert module.running_var.grad is None
    loss.close()
    out.close()
    xt.close()
    _close_all(module)


@needs_native
@pytest.mark.parametrize("training", [True, False])
def test_buffers_never_receive_gradients(training):
    module = NativeBatchNorm1d(3)
    module.train(training)
    xt = NativeParameter(np.random.default_rng(47).standard_normal((4, 3)))
    out = module(xt)
    loss = out.multiply(NativeTensor.from_array(np.ones((4, 3)))).sum()
    loss.backward()
    assert module.running_mean.grad is None
    assert module.running_var.grad is None
    assert module.running_mean.requires_grad is False
    assert module.running_var.requires_grad is False
    loss.close()
    out.close()
    xt.close()
    _close_all(module)


@needs_native
def test_no_custom_backward_exists():
    """Backward comes entirely from the existing composed autograd: the
    module defines no backward of its own and adds no operation."""
    for name in dir(NativeBatchNorm1d):
        assert "backward" not in name, name
    source = native_batchnorm.__file__
    text = open(source, encoding="utf-8").read()
    assert "_from_op(" not in text
    assert "def _backward" not in text


# ==========================================================================
# Running-statistics transaction atomicity
# ==========================================================================

def _training_state(module):
    return _module_state(module)


@needs_native
def test_successful_update_advances_both_buffers_together():
    module = NativeBatchNorm1d(3, momentum=0.5)
    mean_core = module.running_mean._core
    var_core = module.running_var._core
    mean_id, var_id = id(module.running_mean), id(module.running_var)
    gamma_version, beta_version = module.gamma.version, module.beta.version
    x = np.random.default_rng(53).standard_normal((4, 3))
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
    # The replaced cores were closed exactly once, and the new ones differ.
    assert mean_core._closed is True and var_core._closed is True
    assert module.running_mean._core is not mean_core
    assert module.running_var._core is not var_core
    assert module.gamma.version == gamma_version
    assert module.beta.version == beta_version
    xt.close()
    _close_all(module)


@needs_native
def test_successful_update_releases_each_replaced_core_exactly_once(monkeypatch):
    """The F1 commit boundary, observed through its release seam: two
    destinations, two replaced cores, two releases — no more, no fewer,
    and no live core among them."""
    module = NativeBatchNorm1d(3)
    mean_core = module.running_mean._core
    var_core = module.running_var._core
    released = []
    real_release = _native_state._release_core

    def release(core):
        released.append(core)
        return real_release(core)

    monkeypatch.setattr(_native_state, "_release_core", release)
    xt = NativeTensor.from_array(np.random.default_rng(51).standard_normal((4, 3)))
    module(xt).close()
    monkeypatch.undo()

    assert [id(core) for core in released] == [id(mean_core), id(var_core)]
    assert len(released) == len(set(id(core) for core in released))
    # The freshly installed cores were not among them.
    assert module.running_mean._core not in released
    assert module.running_var._core not in released
    assert not module.running_mean._core._closed
    assert not module.running_var._core._closed
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
    return calls


@needs_native
@pytest.mark.parametrize("stage_at,install_at", [
    (1, None),      # the first entry's staging fails
    (2, None),      # the second entry's staging fails
    (None, 1),      # the first core install fails
    (None, 2),      # interrupted *after* the first buffer was swapped
])
def test_transaction_failure_leaves_both_buffers_untouched(
    monkeypatch, live_storages, stage_at, install_at
):
    module = NativeBatchNorm1d(3, momentum=0.5)
    _load(module, running_mean=[0.1, 0.2, 0.3], running_var=[1.5, 2.5, 3.5])
    xt = NativeParameter(np.random.default_rng(59).standard_normal((4, 3)))
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
    assert len(live_storages) == baseline    # nothing staged leaked

    # A later valid forward succeeds and advances both together.
    module(xt).close()
    assert not np.array_equal(module.running_mean.to_numpy(), before["mean"])
    assert not np.array_equal(module.running_var.to_numpy(), before["var"])
    xt.close()
    _close_all(module)


@needs_native
def test_update_value_preparation_failure_leaves_running_state_unchanged(
    monkeypatch, live_storages
):
    """The failure happens while the graph-free replacement values are
    being prepared — before the transaction is even entered."""
    module = NativeBatchNorm1d(3)
    _load(module, running_mean=[0.4, 0.5, 0.6], running_var=[2.0, 2.0, 2.0])
    xt = NativeParameter(np.random.default_rng(61).standard_normal((4, 3)))
    before = _module_state(module)
    _collect()
    baseline = len(live_storages)

    real_detach = NativeTensor.detach
    calls = {"n": 0}

    def detach(self):
        calls["n"] += 1
        if calls["n"] == 2:          # the second statistic
            raise MemoryError("injected preparation failure")
        return real_detach(self)

    monkeypatch.setattr(NativeTensor, "detach", detach)
    with pytest.raises(MemoryError):
        module(xt)
    monkeypatch.undo()

    _assert_state_unchanged(module, before)
    assert len(live_storages) == baseline
    module(xt).close()               # a later forward still works
    xt.close()
    _close_all(module)


@needs_native
def test_running_update_moves_no_parameter_version_and_keeps_graphs_valid():
    """A training step advances the statistics; a graph built before it
    must still run — the buffers carry no version and gamma/beta were not
    replaced."""
    module = NativeBatchNorm1d(3)
    xt = NativeParameter(np.random.default_rng(67).standard_normal((4, 3)))
    out = module(xt)
    loss = out.multiply(NativeTensor.from_array(np.ones((4, 3)))).sum()
    versions = (module.gamma.version, module.beta.version)

    other = NativeParameter(np.random.default_rng(68).standard_normal((6, 3)))
    module(other).close()
    assert (module.gamma.version, module.beta.version) == versions

    loss.backward()                  # no stale-graph error
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
    """Let ``NativeTensor.<method>`` succeed ``successes`` times, then
    raise. A production-free seam: nothing in the library replaces these
    methods."""
    real = getattr(NativeTensor, method)
    state = {"n": 0}

    def wrapper(self, *args, **kwargs):
        state["n"] += 1
        if state["n"] > successes:
            raise _Boom(f"injected {method} failure #{state['n']}")
        return real(self, *args, **kwargs)

    monkeypatch.setattr(NativeTensor, method, wrapper)
    return state


# (method, successes-before-failure) covering, in training order: the
# batch mean, the centering, the squaring, the batch variance, the eps
# constant, sqrt, reciprocal, the normalization multiply, the affine
# multiply, the affine add, the detached statistics, and the
# momentum-update intermediates.
_TRAIN_FAILURE_POINTS = [
    ("mean", 0), ("mean", 1), ("subtract", 0), ("multiply", 0),
    ("multiply", 1), ("multiply", 2), ("add", 0), ("add", 1),
    ("sqrt", 0), ("reciprocal", 0), ("detach", 0), ("detach", 1),
    ("reshape", 0), ("reshape", 1), ("multiply", 3), ("multiply", 5),
    ("add", 2), ("add", 3),
]


@needs_native
@pytest.mark.parametrize("method,successes", _TRAIN_FAILURE_POINTS)
def test_failed_training_forward_changes_nothing_and_leaks_nothing(
    monkeypatch, live_storages, method, successes
):
    module = NativeBatchNorm1d(3, momentum=0.5)
    _load(module, gamma=[1.5, 0.5, -2.0], beta=[0.1, 0.2, 0.3],
          running_mean=[0.7, 0.8, 0.9], running_var=[1.7, 1.8, 1.9])
    xt = NativeParameter(np.random.default_rng(71).standard_normal((4, 3)))
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

    # A later normal forward and backward still work.
    out = module(xt)
    loss = out.multiply(NativeTensor.from_array(np.ones((4, 3)))).sum()
    loss.backward()
    assert xt.grad is not None
    loss.close()
    out.close()
    xt.close()
    _close_all(module)


_EVAL_FAILURE_POINTS = [
    ("reshape", 0), ("contiguous_copy", 0), ("reshape", 1),
    ("contiguous_copy", 1), ("add", 0), ("sqrt", 0), ("reciprocal", 0),
    ("subtract", 0), ("multiply", 0), ("multiply", 1), ("add", 1),
]


@needs_native
@pytest.mark.parametrize("method,successes", _EVAL_FAILURE_POINTS)
def test_failed_eval_forward_changes_nothing_and_leaks_nothing(
    monkeypatch, live_storages, method, successes
):
    module = NativeBatchNorm1d(3)
    _load(module, running_mean=[0.4, 0.5, 0.6], running_var=[2.0, 3.0, 4.0])
    module.eval()
    xt = NativeParameter(np.random.default_rng(73).standard_normal((4, 3)))
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


@needs_native
def test_failed_forward_after_the_transaction_seams_still_rolls_back(
    monkeypatch, live_storages
):
    """A staging failure *and* a commit-time failure both leave the
    pre-forward baseline, including the output graph that was already
    built."""
    module = NativeBatchNorm1d(3)
    xt = NativeParameter(np.random.default_rng(79).standard_normal((5, 3)))
    before = _module_state(module)
    _collect()
    baseline = len(live_storages)
    for stage_at, install_at in ((2, None), (None, 2)):
        _run_with_transaction_failure(
            monkeypatch, module, xt, stage_at=stage_at, install_at=install_at
        )
        assert len(live_storages) == baseline
        _assert_state_unchanged(module, before)
    xt.close()
    _close_all(module)


# ==========================================================================
# Graph-safe evaluation snapshots (§7)
# ==========================================================================

@needs_native
def test_eval_graph_never_holds_a_registered_buffer():
    module = NativeBatchNorm1d(3)
    _load(module, running_mean=[0.5, 1.0, 1.5], running_var=[2.0, 3.0, 4.0])
    module.eval()
    xt = NativeParameter(np.random.default_rng(83).standard_normal((4, 3)))
    out = module(xt)
    reachable = _graph_objects(out)
    assert id(module.running_mean) not in reachable
    assert id(module.running_var) not in reachable
    # The parameters *are* legitimately in the graph.
    assert id(module.gamma) in reachable and id(module.beta) in reachable
    buffer_storage = {
        id(module.running_mean._core.storage),
        id(module.running_var._core.storage),
    }
    # Stronger than identity: not one byte the running buffers own is
    # reachable from the graph, so a borrowing view cannot sneak in.
    assert not (buffer_storage & _graph_storage_ids(out))
    # And every graph-owned resource is independent owning storage.
    resources = out._graph_resources
    assert len(resources) == 2
    for resource in resources:
        assert resource.owns_core is True
        assert resource.contiguous is True
        assert resource.requires_grad is False
        assert resource.is_leaf is True
        assert resource.shape == (1, 3)
        assert id(resource._core.storage) not in buffer_storage
    out.close()
    xt.close()
    _close_all(module)


def _eval_graph_with_control(module, x, upstream):
    """Build an eval graph and the gradients the values *at forward time*
    imply. Returns the live loss plus the three control gradients."""
    running_mean = module.running_mean.to_numpy().copy()
    running_var = module.running_var.to_numpy().copy()
    gamma = module.gamma.to_numpy().copy()
    beta = module.beta.to_numpy().copy()
    xt = NativeParameter(x)
    out = module(xt)
    loss = out.multiply(NativeTensor.from_array(upstream)).sum()
    inverse_std = 1.0 / np.sqrt(running_var + module.eps)
    control = {
        "x": upstream * gamma * inverse_std,
        "gamma": (((x - running_mean) * inverse_std) * upstream).sum(axis=0),
        "beta": upstream.sum(axis=0),
        "beta_value": beta,
    }
    return xt, out, loss, control


def _assert_control_gradients(xt, module, control):
    assert np.allclose(xt.grad.to_numpy(), control["x"], atol=1e-12)
    assert np.allclose(module.gamma.grad.to_numpy(), control["gamma"], atol=1e-12)
    assert np.allclose(module.beta.grad.to_numpy(), control["beta"], atol=1e-12)


@needs_native
def test_later_training_update_cannot_change_an_earlier_eval_backward():
    rng = np.random.default_rng(89)
    module = NativeBatchNorm1d(3, momentum=0.5)
    _load(module, gamma=[1.5, -0.5, 2.0], beta=[0.1, 0.2, -0.3],
          running_mean=[0.3, -0.2, 0.5], running_var=[2.0, 0.5, 1.25])
    module.eval()
    x = rng.standard_normal((4, 3))
    upstream = rng.standard_normal((4, 3))
    xt, out, loss, control = _eval_graph_with_control(module, x, upstream)

    module.train()
    other = NativeTensor.from_array(rng.standard_normal((6, 3)) * 9)
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
def test_later_state_load_cannot_change_an_earlier_eval_backward():
    rng = np.random.default_rng(97)
    module = NativeBatchNorm1d(3)
    _load(module, running_mean=[0.3, -0.2, 0.5], running_var=[2.0, 0.5, 1.25])
    module.eval()
    x = rng.standard_normal((4, 3))
    upstream = rng.standard_normal((4, 3))
    xt, out, loss, control = _eval_graph_with_control(module, x, upstream)

    mean_core = module.running_mean._core
    var_core = module.running_var._core
    _load(module, running_mean=[9.0, 9.0, 9.0], running_var=[16.0, 16.0, 16.0])
    # The replaced cores really were closed; the graph does not read them.
    assert mean_core._closed is True and var_core._closed is True

    loss.backward()
    _assert_control_gradients(xt, module, control)
    loss.close()
    out.close()
    xt.close()
    _close_all(module)


@needs_native
def test_later_buffer_only_state_load_keeps_the_parameter_versions_still():
    """The `load_state_dict()` half of the buffer-only path: only the two
    running buffers move, so no parameter version moves and the earlier
    eval graph stays completely valid."""
    rng = np.random.default_rng(1013)
    module = NativeBatchNorm1d(3)
    _load(module, running_mean=[0.3, -0.2, 0.5], running_var=[2.0, 0.5, 1.25])
    module.eval()
    x = rng.standard_normal((4, 3))
    upstream = rng.standard_normal((4, 3))
    xt, out, loss, control = _eval_graph_with_control(module, x, upstream)
    versions = (module.gamma.version, module.beta.version)

    _load(module, running_mean=[7.0, 7.0, 7.0], running_var=[25.0, 25.0, 25.0])
    assert np.allclose(module.running_mean.to_numpy(), [7.0, 7.0, 7.0])
    assert (module.gamma.version, module.beta.version) == versions

    loss.backward()
    _assert_control_gradients(xt, module, control)
    loss.close()
    out.close()
    xt.close()
    _close_all(module)


class _RunningStatHolder(NativeModule):
    """**Test-only** buffer-only module: it registers *existing*
    ``running_mean`` / ``running_var`` objects as persistent buffer
    aliases and owns no parameters.

    It exists so a test can drive the real ``load_native_checkpoint()``
    path over exactly the ``NativeBatchNorm1d`` buffer objects **without**
    also replacing ``gamma``/``beta`` — the isolation a full-model
    checkpoint cannot give. It is deliberately not a production helper,
    not exported, and adds no public API: it is nothing but two
    ``register_buffer`` calls on the objects it is handed, which is the
    aliasing ``NativeModule`` has always supported."""

    def __init__(self, running_mean, running_var):
        super().__init__()
        self.register_buffer("running_mean", running_mean, persistent=True)
        self.register_buffer("running_var", running_var, persistent=True)


@needs_native
def test_buffer_only_checkpoint_load_cannot_change_an_earlier_eval_backward(
    tmp_path, live_storages
):
    """The checkpoint-path proof of the §7 snapshot rule.

    ``load_native_checkpoint()`` replaces the *same* registered
    ``running_mean``/``running_var`` objects the BatchNorm module holds —
    through the real archive path, not a state dictionary — while leaving
    ``gamma``/``beta`` untouched. An eval graph built before that load
    must still back-propagate the values it read at forward time."""
    _collect()
    baseline = len(live_storages)

    rng = np.random.default_rng(1011)
    module = NativeBatchNorm1d(3)
    _load(module, gamma=[1.5, -0.5, 2.0], beta=[0.1, 0.2, -0.3],
          running_mean=[0.3, -0.2, 0.5], running_var=[2.0, 0.5, 1.25])
    module.eval()

    # A compatible buffer-only checkpoint holding *different* statistics.
    donor_mean = NativeTensor.from_array([7.0, 7.0, 7.0])
    donor_var = NativeTensor.from_array([25.0, 25.0, 25.0])
    donor = _RunningStatHolder(donor_mean, donor_var)
    assert donor.parameters() == []          # buffer-only, by construction
    path = os.path.join(str(tmp_path), "running_stats.npz")
    save_native_checkpoint(path, donor, metadata={"kind": "running-stats"})
    donor_mean.close()
    donor_var.close()

    # The eval graph, and the gradients its forward-time values imply.
    x = rng.standard_normal((4, 3))
    upstream = rng.standard_normal((4, 3))
    xt, out, loss, control = _eval_graph_with_control(module, x, upstream)
    resources = out._graph_resources
    assert len(resources) == 2
    snapshot_storages = {id(resource._core.storage) for resource in resources}
    graph_storages_before = _graph_storage_ids(out)

    # Everything the load must not disturb, recorded before it runs.
    mean_object = module.running_mean
    var_object = module.running_var
    mean_core = mean_object._core
    var_core = var_object._core
    old_buffer_storages = {id(mean_core.storage), id(var_core.storage)}
    # The graph depended on none of the buffers' bytes even before the
    # load — object identity *and* storage.
    assert not (old_buffer_storages & graph_storages_before)
    old_mean = mean_object.to_numpy().copy()
    old_var = var_object.to_numpy().copy()
    gamma_version = module.gamma.version
    beta_version = module.beta.version

    # The holder aliases the module's *own* buffer objects, so the
    # checkpoint path writes straight into the live BatchNorm state.
    holder = _RunningStatHolder(mean_object, var_object)
    assert holder.running_mean is module.running_mean
    assert holder.running_var is module.running_var
    assert list(holder.state_dict().keys()) == ["running_mean", "running_var"]
    for snapshot in holder.state_dict().values():
        snapshot.close()

    metadata = load_native_checkpoint(path, holder)
    assert metadata == {"kind": "running-stats"}

    # (1) The checkpoint path really changed both running values.
    assert np.allclose(module.running_mean.to_numpy(), [7.0, 7.0, 7.0])
    assert np.allclose(module.running_var.to_numpy(), [25.0, 25.0, 25.0])
    assert not np.allclose(module.running_mean.to_numpy(), old_mean)
    assert not np.allclose(module.running_var.to_numpy(), old_var)
    # (2) Both Python buffer objects survived, in both registries.
    assert module.running_mean is mean_object
    assert module.running_var is var_object
    assert holder.running_mean is mean_object
    assert holder.running_var is var_object
    # (3) Both old cores were replaced and closed.
    assert mean_object._core is not mean_core
    assert var_object._core is not var_core
    assert mean_core._closed is True and var_core._closed is True
    # (4) No parameter version moved — this load touched no parameter.
    assert module.gamma.version == gamma_version
    assert module.beta.version == beta_version

    # (5) The earlier eval graph still runs...
    loss.backward()
    # (6) ...and reproduces the forward-time gradients exactly.
    _assert_control_gradients(xt, module, control)
    # (7) ...which are *not* the gradients the new statistics would give.
    new_inverse_std = 1.0 / np.sqrt(np.array([25.0, 25.0, 25.0]) + module.eps)
    new_input_gradient = upstream * module.gamma.to_numpy() * new_inverse_std
    assert not np.allclose(xt.grad.to_numpy(), new_input_gradient, atol=1e-6)
    # (8) Neither registered buffer was ever in the graph — by object
    # identity, and by storage in both its old and new incarnations.
    reachable = _graph_objects(out)
    assert id(mean_object) not in reachable
    assert id(var_object) not in reachable
    new_buffer_storages = {
        id(mean_object._core.storage), id(var_object._core.storage)
    }
    assert not (old_buffer_storages & graph_storages_before)
    assert not (new_buffer_storages & graph_storages_before)
    # (9) The one-shot backward released the snapshot storage exactly once.
    assert all(resource.closed for resource in resources)
    assert out._graph_resources == ()
    assert not (snapshot_storages & live_storages)
    out._release_graph_resources()          # a second release is a no-op
    assert all(resource.closed for resource in resources)

    # (10) Explicit cleanup returns native live storage to the baseline.
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
    """The complementary case, and the reason the buffer-only proof above
    needs its own test: a **full** BatchNorm checkpoint also replaces
    ``gamma``/``beta``, so the existing v3.7 parameter-version guard
    legitimately rejects an earlier graph as stale.

    This is a parameter contract, not a buffer one. BatchNorm neither
    bypasses nor weakens it, and the running-statistic snapshots stay
    safe throughout — proved here by reloading only the buffer half of
    the *same* archive into a second module, whose earlier eval graph
    then back-propagates the forward-time values unchanged."""
    rng = np.random.default_rng(101)
    donor = NativeBatchNorm1d(3)
    _load(donor, gamma=[3.0, 3.0, 3.0], beta=[1.0, 1.0, 1.0],
          running_mean=[7.0, 7.0, 7.0], running_var=[25.0, 25.0, 25.0])
    path = os.path.join(str(tmp_path), "bn.npz")
    save_native_checkpoint(path, donor)
    _close_all(donor)

    module = NativeBatchNorm1d(3)
    _load(module, running_mean=[0.3, -0.2, 0.5], running_var=[2.0, 0.5, 1.25])
    module.eval()
    x = rng.standard_normal((4, 3))
    upstream = rng.standard_normal((4, 3))
    xt, out, loss, control = _eval_graph_with_control(module, x, upstream)
    resources = out._graph_resources
    mean_object, var_object = module.running_mean, module.running_var
    versions = (module.gamma.version, module.beta.version)

    load_native_checkpoint(path, module)
    # Buffer identity survived here too, and the buffers are not the cause.
    assert module.running_mean is mean_object
    assert module.running_var is var_object
    # The cause is exactly this: both parameter versions moved.
    assert module.gamma.version == versions[0] + 1
    assert module.beta.version == versions[1] + 1

    with pytest.raises(RuntimeError, match="stale parameter value") as error:
        loss.backward()
    message = str(error.value)
    assert "NativeParameter" in message
    assert "version" in message
    for buffer_word in ("running_mean", "running_var", "buffer"):
        assert buffer_word not in message, buffer_word
    # The raise is deterministic and commits nothing: the snapshots are
    # still alive and the graph was not freed, so it repeats identically.
    assert all(not resource.closed for resource in resources)
    assert xt.grad is None
    with pytest.raises(RuntimeError, match="stale parameter value"):
        loss.backward()
    # Registered buffers were never in that graph either.
    reachable = _graph_objects(out)
    assert id(mean_object) not in reachable and id(var_object) not in reachable
    loss.close()
    out.close()
    xt.close()

    # The snapshot rule itself is unharmed: loading only the buffer half
    # of the *same* archive leaves an earlier eval graph fully valid.
    second = NativeBatchNorm1d(3)
    _load(second, running_mean=[0.3, -0.2, 0.5], running_var=[2.0, 0.5, 1.25])
    second.eval()
    xt2, out2, loss2, control2 = _eval_graph_with_control(second, x, upstream)
    versions2 = (second.gamma.version, second.beta.version)
    holder = _RunningStatHolder(second.running_mean, second.running_var)
    buffer_only = os.path.join(str(tmp_path), "buffers.npz")
    stats = NativeBatchNorm1d(3)
    load_native_checkpoint(path, stats)          # the archive's buffer half
    save_native_checkpoint(
        buffer_only, _RunningStatHolder(stats.running_mean, stats.running_var)
    )
    load_native_checkpoint(buffer_only, holder)
    assert np.allclose(second.running_mean.to_numpy(), [7.0, 7.0, 7.0])
    assert (second.gamma.version, second.beta.version) == versions2
    loss2.backward()
    _assert_control_gradients(xt2, second, control2)

    loss2.close()
    out2.close()
    xt2.close()
    _close_all(module)
    _close_all(second)
    _close_all(stats)


@needs_native
def test_eval_graph_survives_retain_graph_and_releases_once(live_storages):
    module = NativeBatchNorm1d(3)
    _load(module, running_mean=[0.3, -0.2, 0.5], running_var=[2.0, 0.5, 1.25])
    module.eval()
    rng = np.random.default_rng(103)
    x = rng.standard_normal((4, 3))
    upstream = rng.standard_normal((4, 3))
    xt, out, loss, control = _eval_graph_with_control(module, x, upstream)
    resources = out._graph_resources
    assert len(resources) == 2
    _collect()
    baseline = len(live_storages)

    loss.backward(retain_graph=True)
    assert all(not resource.closed for resource in resources)
    module.zero_grad()
    xt.zero_grad()
    loss.backward(retain_graph=True)      # readable again: still valid
    _assert_control_gradients(xt, module, control)
    module.zero_grad()
    xt.zero_grad()

    snapshot_storages = {id(resource._core.storage) for resource in resources}
    assert snapshot_storages <= live_storages

    loss.backward()                        # one-shot: releases the history
    _assert_control_gradients(xt, module, control)
    assert all(resource.closed for resource in resources)
    assert out._graph_resources == ()      # released exactly once
    out._release_graph_resources()         # a second call is a no-op
    assert all(resource.closed for resource in resources)
    # The snapshot storage really went away, deterministically and without
    # gc. The overall baseline legitimately moves (backward allocated
    # gradients), so the check is on the snapshot storages by identity.
    assert not (snapshot_storages & live_storages)
    assert baseline > 0

    loss.close()
    out.close()
    xt.close()
    _close_all(module)


@needs_native
def test_abandoned_eval_graph_releases_its_snapshots_on_close(live_storages):
    module = NativeBatchNorm1d(3)
    module.eval()
    xt = NativeParameter(np.random.default_rng(107).standard_normal((4, 3)))
    _collect()
    baseline = len(live_storages)
    out = module(xt)
    resources = out._graph_resources
    assert len(resources) == 2
    snapshot_storages = {id(resource._core.storage) for resource in resources}
    out.close()                             # never backwarded
    # Deterministic, without gc: closing the output released the graph's
    # snapshot state immediately and exactly once.
    assert all(resource.closed for resource in resources)
    assert not (snapshot_storages & live_storages)
    del out, resources
    _collect()                              # the ordinary graph wrappers
    assert len(live_storages) == baseline
    xt.close()
    _close_all(module)


@needs_native
def test_training_graph_holds_no_buffer_and_no_snapshot_resource():
    module = NativeBatchNorm1d(3)
    xt = NativeParameter(np.random.default_rng(109).standard_normal((4, 3)))
    out = module(xt)
    reachable = _graph_objects(out)
    assert id(module.running_mean) not in reachable
    assert id(module.running_var) not in reachable
    assert out._graph_resources == ()
    out.close()
    xt.close()
    _close_all(module)


# ==========================================================================
# State dictionary and checkpoints
# ==========================================================================

@needs_native
def test_state_dict_keys_order_and_independence():
    module = NativeBatchNorm1d(3)
    _load(module, running_mean=[1.0, 2.0, 3.0])
    state = module.state_dict()
    assert list(state) == ["gamma", "beta", "running_mean", "running_var"]
    snapshot = state["running_mean"].to_numpy().copy()
    xt = NativeTensor.from_array(np.random.default_rng(113).standard_normal((4, 3)))
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
    module = NativeBatchNorm1d(3)
    identities = (id(module.gamma), id(module.beta),
                  id(module.running_mean), id(module.running_var))
    versions = (module.gamma.version, module.beta.version)

    # Buffer-only load: no parameter version moves.
    _load(module, running_mean=[1.0, 2.0, 3.0], running_var=[4.0, 5.0, 6.0])
    assert (module.gamma.version, module.beta.version) == versions
    assert np.allclose(module.running_mean.to_numpy(), [1.0, 2.0, 3.0])

    # Parameter load: exactly one increment each.
    _load(module, gamma=[2.0, 2.0, 2.0], beta=[1.0, 1.0, 1.0])
    assert module.gamma.version == versions[0] + 1
    assert module.beta.version == versions[1] + 1

    assert (id(module.gamma), id(module.beta),
            id(module.running_mean), id(module.running_var)) == identities
    _close_all(module)


@needs_native
def test_invalid_mixed_load_is_atomic(live_storages):
    module = NativeBatchNorm1d(3)
    before = _module_state(module)
    _collect()
    baseline = len(live_storages)
    good = NativeTensor.from_array([5.0, 5.0, 5.0])
    bad = NativeTensor.from_array([1.0, 2.0])       # wrong shape
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
def test_checkpoint_round_trip_reproduces_all_four_tensors(tmp_path):
    from tensorforge.experimental import native_checkpoint

    assert native_checkpoint._FORMAT == "tensorforge.native_checkpoint"
    assert native_checkpoint._FORMAT_VERSION == 1

    source = NativeBatchNorm1d(3, eps=1e-3, momentum=0.4)
    _load(source, gamma=[1.1, 1.2, 1.3], beta=[-0.1, -0.2, -0.3])
    xt = NativeTensor.from_array(
        np.random.default_rng(127).standard_normal((6, 3))
    )
    source(xt).close()
    expected = {name: tensor.to_numpy().copy()
                for name, tensor in source._state_named_tensors()}
    path = os.path.join(str(tmp_path), "checkpoint.npz")
    save_native_checkpoint(path, source, metadata={"milestone": "F3"})

    target = NativeBatchNorm1d(3, eps=1e-3, momentum=0.4)
    identities = (id(target.gamma), id(target.beta),
                  id(target.running_mean), id(target.running_var))
    metadata = load_native_checkpoint(path, target)
    assert metadata == {"milestone": "F3"}
    for name, value in expected.items():
        assert np.array_equal(getattr(target, name).to_numpy(), value), name
    assert (id(target.gamma), id(target.beta),
            id(target.running_mean), id(target.running_var)) == identities

    # Training mode is runtime state, never serialized.
    target.eval()
    load_native_checkpoint(path, target)
    assert target.training is False

    xt.close()
    _close_all(source)
    _close_all(target)


@needs_native
def test_checkpoint_reproduces_evaluation_output(tmp_path):
    source = NativeBatchNorm1d(3)
    rng = np.random.default_rng(131)
    for _ in range(4):
        xt = NativeTensor.from_array(rng.standard_normal((5, 3)))
        source(xt).close()
        xt.close()
    source.eval()
    probe = rng.standard_normal((3, 3))
    probe_t = NativeTensor.from_array(probe)
    expected = source(probe_t).to_numpy().copy()

    path = os.path.join(str(tmp_path), "eval.npz")
    save_native_checkpoint(path, source)
    target = NativeBatchNorm1d(3)
    load_native_checkpoint(path, target)
    target.eval()
    assert np.array_equal(target(probe_t).to_numpy(), expected)

    probe_t.close()
    _close_all(source)
    _close_all(target)


# ==========================================================================
# NativeSequential composition
# ==========================================================================

@needs_native
def test_composes_inside_a_native_sequential():
    model = NativeSequential(
        NativeLinear(4, 3, seed=5),
        NativeBatchNorm1d(3),
        NativeReLU(),
        NativeLinear(3, 2, seed=6),
    )
    assert [name for name, _ in model.named_parameters()] == [
        "0.weight", "0.bias", "1.gamma", "1.beta", "3.weight", "3.bias",
    ]
    assert [name for name, _ in model.named_buffers()] == [
        "1.running_mean", "1.running_var",
    ]
    assert list(model.state_dict()) == [
        "0.weight", "0.bias", "1.gamma", "1.beta", "3.weight", "3.bias",
        "1.running_mean", "1.running_var",
    ]
    for snapshot in model.state_dict().values():
        snapshot.close()

    optimizer = NativeSGD(model.parameters(), lr=0.1)
    assert len(optimizer.parameters()) == 6
    buffer_ids = {id(b) for b in model.buffers()}
    assert not any(id(p) in buffer_ids for p in optimizer.parameters())

    xt = NativeParameter(np.random.default_rng(137).standard_normal((5, 4)))
    out = model(xt)
    assert out.shape == (5, 2)
    loss = out.multiply(NativeTensor.from_array(np.ones((5, 2)))).sum()
    loss.backward()
    assert xt.grad is not None
    batchnorm = model._modules["1"]
    assert not np.array_equal(batchnorm.running_mean.to_numpy(), np.zeros(3))

    # Mode propagation reaches the child.
    model.eval()
    assert batchnorm.training is False
    model.train()
    assert batchnorm.training is True

    # State loading preserves every identity.
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
def test_sequential_train_and_eval_outputs_differ():
    model = NativeSequential(NativeLinear(3, 3, seed=8), NativeBatchNorm1d(3))
    rng = np.random.default_rng(139)
    for _ in range(3):
        xt = NativeTensor.from_array(rng.standard_normal((6, 3)))
        model(xt).close()
        xt.close()
    probe = NativeTensor.from_array(rng.standard_normal((4, 3)))
    train_out = model(probe).to_numpy().copy()
    model.eval()
    eval_out = model(probe).to_numpy().copy()
    assert not np.allclose(train_out, eval_out, atol=1e-6)
    probe.close()
    _close_all(model)


# ==========================================================================
# Non-contiguous input
# ==========================================================================

@needs_native
@pytest.mark.parametrize("training", [True, False])
def test_non_contiguous_input(training):
    rng = np.random.default_rng(149)
    wide = rng.standard_normal((5, 6))
    module = NativeBatchNorm1d(3, eps=1e-3)
    _load(module, gamma=[1.5, -0.5, 2.0], beta=[0.25, 0.0, -0.5],
          running_mean=[0.2, 0.4, 0.6], running_var=[1.5, 2.5, 3.5])
    module.train(training)
    wide_t = NativeParameter(wide)
    view = wide_t.narrow(1, 2, 3)
    assert view.contiguous is False and view.owns_core is False

    out = module(view)
    slice_ = wide[:, 2:5]
    if training:
        expected, batch_mean, batch_var = train_reference(
            slice_, np.array([1.5, -0.5, 2.0]), np.array([0.25, 0.0, -0.5]), 1e-3
        )
        new_mean, new_var = running_reference(
            np.array([0.2, 0.4, 0.6]), np.array([1.5, 2.5, 3.5]),
            batch_mean, batch_var, 0.1,
        )
        assert np.allclose(module.running_mean.to_numpy(), new_mean, atol=1e-12)
        assert np.allclose(module.running_var.to_numpy(), new_var, atol=1e-12)
    else:
        expected = eval_reference(
            slice_, np.array([0.2, 0.4, 0.6]), np.array([1.5, 2.5, 3.5]),
            np.array([1.5, -0.5, 2.0]), np.array([0.25, 0.0, -0.5]), 1e-3,
        )
    assert np.allclose(out.to_numpy(), expected, atol=1e-12)
    assert out.owns_core is True and out.contiguous is True
    assert out.shape == (5, 3)

    loss = out.multiply(NativeTensor.from_array(np.ones((5, 3)))).sum()
    loss.backward()
    assert wide_t.grad.shape == (5, 6)
    assert np.allclose(wide_t.grad.to_numpy()[:, [0, 1, 5]], 0.0, atol=1e-15)

    loss.close()
    out.close()
    view.close()
    wide_t.close()
    _close_all(module)


@needs_native
def test_output_is_a_fresh_owning_contiguous_tensor():
    module = NativeBatchNorm1d(3)
    for training in (True, False):
        module.train(training)
        xt = NativeParameter(np.random.default_rng(151).standard_normal((4, 3)))
        out = module(xt)
        assert type(out) is NativeTensor
        assert out.owns_core is True and out.contiguous is True
        assert out.shape == (4, 3)
        storages = {id(xt._core.storage), id(module.gamma._core.storage),
                    id(module.beta._core.storage),
                    id(module.running_mean._core.storage),
                    id(module.running_var._core.storage)}
        assert id(out._core.storage) not in storages
        out.close()
        xt.close()
    # No per-forward tensor attribute is stored on the module.
    for name in dir(module):
        if name.startswith("__"):
            continue
        value = getattr(module, name, None)
        if isinstance(value, NativeTensor):
            assert name in ("gamma", "beta", "running_mean", "running_var"), name
    _close_all(module)


# ==========================================================================
# NumPy tripwire
# ==========================================================================

_NUMERICAL_NUMPY = (
    "max", "amax", "argmax", "exp", "log", "sqrt", "reciprocal", "sum",
    "divide", "true_divide", "add", "subtract", "multiply", "matmul",
    "mean", "var", "std", "negative", "power", "square", "copyto",
)
# The tensor-*data* entry/exit boundaries only. np.array / np.asarray /
# np.ascontiguousarray are deliberately absent: the native reductions build
# small int64 *shape* arrays with np.asarray (metadata, never tensor data),
# so arming them would flag a legitimate non-data use.
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
def test_training_and_evaluation_use_no_numpy(monkeypatch):
    # Construction and reference/upstream preparation are the allowed
    # host-entry boundary and happen before the tripwire is armed.
    rng = np.random.default_rng(157)
    x = rng.standard_normal((4, 3))
    upstream = rng.standard_normal((4, 3))
    module = NativeBatchNorm1d(3, eps=1e-3, momentum=0.4)
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
    train_out = module(xt)               # forward + running-stat update
    loss = train_out.multiply(up).sum()  # scalar objective
    loss.backward()                      # native backward
    module.eval()
    eval_out = module(xt)                # evaluation forward
    eval_loss = eval_out.multiply(up).sum()
    eval_loss.backward()                 # evaluation backward
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
    module = NativeBatchNorm1d(3)
    assert len(live_storages) == baseline + 4
    _close_all(module)
    assert len(live_storages) == baseline


@needs_native
@pytest.mark.parametrize("training", [True, False])
def test_forward_backward_returns_to_baseline(live_storages, training):
    baseline = len(live_storages)
    module = NativeBatchNorm1d(3)
    module.train(training)
    xt = NativeParameter(np.random.default_rng(163).standard_normal((4, 3)))
    out = module(xt)
    loss = out.multiply(NativeTensor.from_array(np.ones((4, 3)))).sum()
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
    module = NativeBatchNorm1d(3)
    module.train(training)
    _collect()
    baseline = len(live_storages)
    for step in range(6):
        xt = NativeParameter(
            np.random.default_rng(step).standard_normal((4, 3))
        )
        out = module(xt)
        loss = out.multiply(NativeTensor.from_array(np.ones((4, 3)))).sum()
        loss.backward()
        for tensor in (loss, out, xt, xt.grad):
            tensor.close()
        module.zero_grad()
        _collect()
    assert len(live_storages) == baseline   # module state only; no growth


@needs_native
def test_no_grad_forward_returns_to_baseline(live_storages):
    module = NativeBatchNorm1d(3)
    module.eval()
    _collect()
    baseline = len(live_storages)
    x = NativeTensor.from_array(np.random.default_rng(167).standard_normal((4, 3)))
    out = module(x)
    assert np.all(np.isfinite(out.to_numpy()))
    # gamma/beta always require grad, so a graph is built even for a
    # no-grad input: closing the output releases its snapshots at once,
    # and the ordinary graph wrappers go with the dropped reference.
    out.close()
    del out
    x.close()
    _collect()
    assert len(live_storages) == baseline


@needs_native
def test_state_and_checkpoint_loading_return_to_baseline(live_storages, tmp_path):
    module = NativeBatchNorm1d(3)
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
def test_sequential_composition_returns_to_baseline(live_storages):
    _collect()
    baseline = len(live_storages)
    model = NativeSequential(
        NativeLinear(4, 3, seed=1), NativeBatchNorm1d(3), NativeReLU()
    )
    for step in range(3):
        xt = NativeParameter(np.random.default_rng(step).standard_normal((5, 4)))
        out = model(xt)
        loss = out.multiply(NativeTensor.from_array(np.ones((5, 3)))).sum()
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
    module = NativeBatchNorm1d(3)
    _close_all(module)
    _close_all(module)
    for tensor in (module.gamma, module.beta,
                   module.running_mean, module.running_var):
        assert tensor.closed is True


@needs_fault_injection
def test_constructor_failure_releases_every_earlier_allocation(
    monkeypatch, live_storages
):
    """The constructor allocates, in order, gamma, beta, running_mean,
    and running_var. Forcing the 2nd, 3rd, and 4th native allocation to
    fail therefore injects a failure at the beta allocation, at the
    running_mean creation, and at the running_var creation — and each must
    return the native live-storage count to the pre-construction baseline,
    having closed every already-created object exactly once."""
    # (failing allocation, how many objects existed before it)
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
                NativeBatchNorm1d(4)
        finally:
            cpp._arm_alloc_failure(0)
            cpp._require_library().tf_clear_error()
            monkeypatch.undo()
        assert len(created) == already_created, nth
        for tensor in created:
            assert tensor.closed is True, nth
        # No gc.collect(): the cleanup is explicit and deterministic.
        assert len(live_storages) == baseline, nth


@needs_native
@pytest.mark.parametrize("failing_call,already_created", [(1, 3), (2, 4)])
def test_registration_failure_releases_every_earlier_allocation(
    monkeypatch, live_storages, failing_call, already_created
):
    """A failure in ``register_buffer`` itself (not in an allocation) is
    cleaned up on the same path: call 1 is the running_mean registration
    (gamma, beta, and running_mean exist), call 2 the running_var one (all
    four exist)."""
    from tensorforge.experimental.native_module import NativeModule

    real_register = NativeModule.register_buffer
    calls = {"n": 0}

    def register(self, name, tensor, persistent=True):
        calls["n"] += 1
        if calls["n"] == failing_call:
            raise _Boom("injected registration failure")
        return real_register(self, name, tensor, persistent=persistent)

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

    monkeypatch.setattr(NativeModule, "register_buffer", register)
    monkeypatch.setattr(native_batchnorm, "NativeParameter", spy_parameter)
    monkeypatch.setattr(
        native_batchnorm.NativeTensor, "zeros", staticmethod(spy_zeros)
    )
    monkeypatch.setattr(
        native_batchnorm.NativeTensor, "full", staticmethod(spy_full)
    )
    _collect()
    baseline = len(live_storages)
    with pytest.raises(_Boom):
        NativeBatchNorm1d(3)
    monkeypatch.undo()
    assert len(created) == already_created
    for tensor in created:
        assert tensor.closed is True
    assert len(live_storages) == baseline


# ==========================================================================
# Milestone guardrails
# ==========================================================================

@needs_native
def test_f3_ships_only_the_1d_module():
    import tensorforge.experimental as experimental

    assert "NativeBatchNorm1d" in experimental.__all__
    assert experimental.NativeBatchNorm1d is NativeBatchNorm1d
    assert "NativeBatchNorm1d" in cpp.NATIVE_MODULES
    # F4 has not started.
    assert "NativeBatchNorm2d" not in experimental.__all__
    assert not hasattr(experimental, "NativeBatchNorm2d")
    assert not hasattr(native_batchnorm, "NativeBatchNorm2d")
    assert "NativeBatchNorm2d" not in cpp.NATIVE_MODULES
    # The unqualified capability stays unsupported until both shapes ship.
    assert "batchnorm" in cpp.UNSUPPORTED


@needs_native
def test_f3_adds_no_operation_core_method_kernel_or_abi_symbol():
    for name in ("batch_norm", "batchnorm", "batch_norm_forward",
                 "batch_norm_backward", "normalize"):
        assert name not in cpp.TENSOR_CORE_OPS, name
        assert name not in cpp.AUTOGRAD_OPS, name
        assert name not in cpp.RAW_KERNELS, name
        assert not hasattr(cpp.NativeTensorCore, name), name
        assert not hasattr(NativeTensor, name), name
    for symbol in ("tf_core_batch_norm", "tf_core_batch_norm_forward",
                   "tf_core_batch_norm_backward"):
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


@needs_native
def test_the_shared_implementation_is_private_and_reusable():
    """One shared private implementation carries every behavior; the
    public class supplies only its shape configuration, so F4 can add the
    NCHW shape without a second BatchNorm implementation."""
    assert issubclass(NativeBatchNorm1d, native_batchnorm._NativeBatchNorm)
    assert NativeBatchNorm1d._INPUT_NDIM == 2
    assert NativeBatchNorm1d._REDUCTION_AXES == (0,)
    assert NativeBatchNorm1d._TRAILING_DIMS == 0
    # The subclass adds no behavior of its own: only the four shape
    # configuration slots, and not one callable.
    declared = {
        name for name in vars(NativeBatchNorm1d) if not name.startswith("__")
    }
    assert declared == {
        "_INPUT_NDIM", "_REDUCTION_AXES", "_TRAILING_DIMS", "_LAYOUT",
    }
    assert not any(callable(vars(NativeBatchNorm1d)[name]) for name in declared)
    import tensorforge.experimental as experimental
    assert not hasattr(experimental, "_NativeBatchNorm")


@needs_native
def test_no_public_buffer_mutation_api_was_added():
    """F3 uses the private F1 transaction; it adds no in-place mutation
    surface to ordinary NativeTensor and no BatchNorm-specific one.
    ``NativeParameter.copy_value_`` remains the only public controlled
    mutation primitive, and no ``NativeModule.close()`` was introduced."""
    for name in ("set_running_stats", "update_running_stats", "reset_running_stats",
                 "fill_", "copy_", "copy_value_"):
        assert not hasattr(NativeTensor, name) or name == "copy_value_", name
        assert not hasattr(NativeBatchNorm1d, name), name
    # copy_value_ stays exactly where it was: on NativeParameter only.
    assert hasattr(NativeParameter, "copy_value_")
    assert "copy_value_" not in vars(NativeTensor)
    import tensorforge.experimental as experimental
    from tensorforge.experimental.native_module import NativeModule
    assert not hasattr(NativeModule, "close")
    assert "replace_native_state" not in experimental.__all__
    assert not hasattr(experimental, "replace_native_state")
    module = NativeBatchNorm1d(3)
    assert not hasattr(module, "close")
    _close_all(module)
