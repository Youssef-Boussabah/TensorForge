"""Tests for NativeFlatten (Advanced C++ Phase D, milestone D1).

NativeFlatten is a parameter-free, buffer-free NativeModule that collapses
every non-batch dimension of a native tensor into one feature axis,
``(N, ...) -> (N, features)``. It is Python-composed from the existing
``reshape``/``contiguous_copy`` operations and their autograd — no new C++
kernel, no custom backward — and returns an **independent owning** result
so it composes safely in a NativeSequential (a bare reshape view would
dangle once the source transient is dropped; see
src/tensorforge/experimental/native_flatten.py and
docs/native_cnn_design.md §3.1).

Selector: python -m pytest -q -k native_flatten
"""

import numpy as np
import pytest

from tensorforge.backends import cpp
from tensorforge.experimental import (
    NativeFlatten,
    NativeLinear,
    NativeModule,
    NativeMSELoss,
    NativeParameter,
    NativeReLU,
    NativeSequential,
    NativeTensor,
    load_native_checkpoint,
    save_native_checkpoint,
)

needs_native = pytest.mark.skipif(
    not cpp.is_available(),
    reason="experimental C++ backend not built",
)


# ======================================================================
# Construction and module behavior
# ======================================================================


def test_native_flatten_is_a_parameter_free_module():
    flat = NativeFlatten()
    assert isinstance(flat, NativeModule)
    assert flat.parameters() == []
    assert list(flat.named_parameters()) == []
    assert flat.buffers() == []
    assert list(flat.named_buffers()) == []
    assert flat.state_dict() == {}
    assert flat.training is True
    assert repr(flat) == "NativeFlatten()"


def test_native_flatten_train_eval_flag_updates():
    flat = NativeFlatten()
    assert flat.eval() is flat
    assert flat.training is False
    assert flat.train() is flat
    assert flat.training is True


def test_native_flatten_exports_from_experimental_package():
    import tensorforge
    import tensorforge.experimental as experimental

    assert "NativeFlatten" in experimental.__all__
    assert experimental.NativeFlatten is NativeFlatten
    # Not leaked into the stable namespace, and not a stable Module.
    assert not hasattr(tensorforge, "NativeFlatten")
    assert not hasattr(tensorforge.nn, "NativeFlatten")
    assert not issubclass(NativeFlatten, tensorforge.nn.Module)


# ======================================================================
# Forward behavior
# ======================================================================


@needs_native
def test_native_flatten_collapses_nchw_to_features():
    x = NativeTensor.from_array(np.arange(24.0).reshape(2, 3, 4))  # N=2
    out = NativeFlatten()(x)
    assert type(out) is NativeTensor
    assert not isinstance(out, NativeParameter)
    assert out.shape == (2, 12)
    assert out.dtype == "float64" and out.device == "cpu"
    # Row-major order is preserved: (2, 3, 4) row-major == (2, 12) row-major.
    assert np.array_equal(out.to_numpy(), np.arange(24.0).reshape(2, 12))


@needs_native
def test_native_flatten_four_dimensional_nchw():
    values = np.arange(2 * 3 * 4 * 5, dtype=np.float64).reshape(2, 3, 4, 5)
    out = NativeFlatten()(NativeTensor.from_array(values))
    assert out.shape == (2, 3 * 4 * 5)
    assert np.array_equal(out.to_numpy(), values.reshape(2, -1))


@needs_native
def test_native_flatten_two_dimensional_input_preserves_shape():
    values = np.arange(6.0).reshape(3, 2)
    out = NativeFlatten()(NativeTensor.from_array(values))
    assert out.shape == (3, 2)
    assert np.array_equal(out.to_numpy(), values)


@needs_native
def test_native_flatten_single_element_feature_dimension():
    values = np.arange(3.0).reshape(3, 1)  # (N=3, 1)
    out = NativeFlatten()(NativeTensor.from_array(values))
    assert out.shape == (3, 1)
    assert np.array_equal(out.to_numpy(), values)


@needs_native
def test_native_flatten_higher_rank_flattens_every_non_batch_axis():
    values = np.arange(120.0).reshape(2, 2, 2, 3, 5)
    out = NativeFlatten()(NativeTensor.from_array(values))
    assert out.shape == (2, 2 * 2 * 3 * 5)
    assert np.array_equal(out.to_numpy(), values.reshape(2, -1))


@needs_native
def test_native_flatten_accepts_native_parameter_and_returns_plain_tensor():
    p = NativeParameter(np.arange(24.0).reshape(2, 3, 4))
    out = NativeFlatten()(p)
    assert type(out) is NativeTensor
    assert not isinstance(out, NativeParameter)
    assert out.shape == (2, 12)
    assert p.is_leaf is True  # the parameter stays a graph-free leaf


# ======================================================================
# Validation
# ======================================================================


@needs_native
def test_native_flatten_rejects_rank_below_two():
    flat = NativeFlatten()
    scalar = NativeTensor.from_array(3.0)  # rank 0
    vector = NativeTensor.from_array([1.0, 2.0, 3.0])  # rank 1
    for bad in (scalar, vector):
        with pytest.raises(ValueError, match="at least 2-D"):
            flat(bad)


@needs_native
def test_native_flatten_rejects_non_native_inputs():
    import tensorforge

    flat = NativeFlatten()
    for bad in (tensorforge.Tensor(np.zeros((2, 3))), np.zeros((2, 3)),
                [[1.0, 2.0]], 3.0, None):
        with pytest.raises(TypeError):
            flat(bad)


@needs_native
def test_native_flatten_rejects_closed_input():
    flat = NativeFlatten()
    x = NativeTensor.from_array(np.arange(24.0).reshape(2, 3, 4))
    x.close()
    with pytest.raises(RuntimeError, match="closed"):
        flat(x)


# ======================================================================
# View / copy ownership behavior
# ======================================================================


@needs_native
def test_native_flatten_result_owns_storage_and_survives_input_close():
    # The observable ownership contract: NativeFlatten returns an
    # independent owning tensor, so its lifetime never depends on the
    # input surviving (this is what makes it safe in a NativeSequential).
    x = NativeTensor.from_array(np.arange(24.0).reshape(2, 3, 4))  # contiguous
    out = NativeFlatten()(x)
    assert out.owns_core is True
    x.close()  # drop the input entirely
    assert np.array_equal(out.to_numpy(), np.arange(24.0).reshape(2, 12))


@needs_native
def test_native_flatten_non_contiguous_input_is_materialized_correctly():
    base = NativeTensor.from_array(np.arange(24.0).reshape(2, 3, 4))
    view = base.transpose(0, 2, 1)  # (2, 4, 3), non-contiguous
    assert view.contiguous is False
    out = NativeFlatten()(view)
    assert out.owns_core is True
    expected = np.arange(24.0).reshape(2, 3, 4).transpose(0, 2, 1).reshape(2, 12)
    assert np.array_equal(out.to_numpy(), expected)
    # The original non-contiguous source is not mutated by the flatten.
    assert np.array_equal(
        view.to_numpy(), np.arange(24.0).reshape(2, 3, 4).transpose(0, 2, 1)
    )
    assert np.array_equal(base.to_numpy(), np.arange(24.0).reshape(2, 3, 4))


@needs_native
def test_native_flatten_result_is_never_a_stable_tensor():
    import tensorforge

    out = NativeFlatten()(NativeTensor.from_array(np.arange(6.0).reshape(2, 3)))
    assert isinstance(out, NativeTensor)
    assert not isinstance(out, tensorforge.Tensor)


# ======================================================================
# NativeSequential integration
# ======================================================================


@needs_native
def test_native_flatten_in_sequential_forward_eval():
    # A view-returning module would dangle here (the ReLU output is a
    # transient dropped as the loop rebinds); the owning result survives.
    model = NativeSequential(NativeReLU(), NativeFlatten())
    x = NativeTensor.from_array(np.arange(-12.0, 12.0).reshape(2, 3, 4))
    out = model(x)
    assert out.shape == (2, 12)
    expected = np.maximum(np.arange(-12.0, 12.0).reshape(2, 3, 4), 0.0).reshape(2, 12)
    assert np.array_equal(out.to_numpy(), expected)


@needs_native
def test_native_flatten_bridges_into_native_linear():
    model = NativeSequential(NativeFlatten(), NativeLinear(12, 1, seed=0))
    x = NativeTensor.from_array(np.arange(24.0).reshape(2, 3, 4))
    out = model(x)
    assert out.shape == (2, 1)
    # Same as flattening first, then applying the linear directly.
    flat = NativeFlatten()(x)
    direct = model[1](flat)
    assert np.allclose(out.to_numpy(), direct.to_numpy())


# ======================================================================
# Autograd (inherited from reshape / contiguous_copy)
# ======================================================================


@needs_native
def test_native_flatten_has_no_module_backward_of_its_own():
    # No custom backward callback: the graph is only existing ops.
    assert "backward" not in NativeFlatten.__dict__
    x = NativeTensor.from_array(np.arange(24.0).reshape(2, 3, 4), requires_grad=True)
    out = NativeFlatten()(x)
    # The final op is the owning contiguous_copy; the graph underneath is
    # reshape -> contiguous_copy, all existing operations.
    assert out._op == "contiguous_copy"
    assert out.is_leaf is False


@needs_native
def test_native_flatten_contiguous_backward_restores_input_shape():
    x = NativeTensor.from_array(np.arange(24.0).reshape(2, 3, 4), requires_grad=True)
    NativeFlatten()(x).sum().backward()
    assert x.grad is not None
    assert x.grad.shape == (2, 3, 4)
    assert np.array_equal(x.grad.to_numpy(), np.ones((2, 3, 4)))


@needs_native
def test_native_flatten_contiguous_gradient_values_are_correct():
    # d/dx of sum(w . flatten(x)) is w reshaped back to x's shape.
    x = NativeTensor.from_array(np.arange(24.0).reshape(2, 3, 4), requires_grad=True)
    weights = np.arange(1.0, 25.0).reshape(2, 12)
    w = NativeTensor.from_array(weights)
    (NativeFlatten()(x).multiply(w)).sum().backward()
    assert np.array_equal(x.grad.to_numpy(), weights.reshape(2, 3, 4))


@needs_native
def test_native_flatten_non_contiguous_backward_reaches_source():
    base = NativeTensor.from_array(
        np.arange(24.0).reshape(2, 3, 4), requires_grad=True
    )
    view = base.transpose(0, 2, 1)  # (2, 4, 3) non-contiguous, requires grad
    NativeFlatten()(view).sum().backward()
    assert base.grad is not None
    assert base.grad.shape == (2, 3, 4)
    assert np.array_equal(base.grad.to_numpy(), np.ones((2, 3, 4)))


@needs_native
def test_native_flatten_shared_source_accumulates_gradients():
    # The same source feeds two flatten paths; gradients sum.
    x = NativeTensor.from_array(np.arange(24.0).reshape(2, 3, 4), requires_grad=True)
    flat = NativeFlatten()
    total = flat(x).sum().add(flat(x).sum())
    total.backward()
    assert np.array_equal(x.grad.to_numpy(), 2.0 * np.ones((2, 3, 4)))


@needs_native
def test_native_flatten_no_grad_input_builds_no_graph_state():
    x = NativeTensor.from_array(np.arange(24.0).reshape(2, 3, 4))  # requires_grad False
    out = NativeFlatten()(x)
    assert out.requires_grad is False
    assert out.is_leaf is True  # plain forward tensor, no graph node


@needs_native
def test_native_flatten_trains_in_sequential_composition():
    model = NativeSequential(NativeFlatten(), NativeLinear(12, 1, seed=0))
    x = NativeTensor.from_array(np.arange(24.0).reshape(2, 3, 4), requires_grad=True)
    target = NativeTensor.from_array(np.zeros((2, 1)))
    loss = NativeMSELoss()(model(x), target)
    loss.backward()
    # Gradients flow to the linear parameters and back through flatten to x.
    assert model[1].weight.grad is not None
    assert model[1].bias.grad is not None
    assert x.grad is not None and x.grad.shape == (2, 3, 4)


# ======================================================================
# State dictionary and checkpoint behavior
# ======================================================================


@needs_native
def test_native_flatten_state_dict_is_empty_and_loads_cleanly():
    flat = NativeFlatten()
    assert flat.state_dict() == {}
    result = flat.load_state_dict({})
    assert result.missing_keys == () and result.unexpected_keys == ()
    extra = {"x": NativeTensor.from_array([1.0])}
    with pytest.raises(ValueError, match="'x'"):
        flat.load_state_dict(extra)  # strict: unexpected key
    result = flat.load_state_dict(extra, strict=False)
    assert result.unexpected_keys == ("x",)


@needs_native
def test_native_flatten_adds_no_state_entries_in_sequential():
    model = NativeSequential(NativeFlatten(), NativeLinear(12, 2, seed=1))
    state = model.state_dict()
    # Only the linear (slot 1) contributes keys; flatten (slot 0) none.
    assert set(state) == {"1.weight", "1.bias"}


@needs_native
def test_native_flatten_checkpoint_round_trip(tmp_path):
    def build(seed):
        return NativeSequential(NativeFlatten(), NativeLinear(12, 2, seed=seed))

    source = build(seed=0)
    saved = {name: p.to_numpy() for name, p in source.named_parameters()}
    path = tmp_path / "cnn_flatten.npz"
    assert save_native_checkpoint(path, source) is None

    target = build(seed=9)  # different init, same architecture
    identities = [id(p) for p in target.parameters()]
    load_native_checkpoint(path, target)
    for name, parameter in target.named_parameters():
        assert np.array_equal(parameter.to_numpy(), saved[name])
    # Parameter identities preserved by the module loader (no schema change).
    assert [id(p) for p in target.parameters()] == identities

    # The restored model still runs the flatten bridge end to end.
    x = NativeTensor.from_array(np.arange(24.0).reshape(2, 3, 4))
    assert target(x).shape == (2, 2)


# ======================================================================
# Introspection and isolation
# ======================================================================


def test_native_flatten_in_native_module_inventory_not_raw_kernel():
    assert "NativeFlatten" in cpp.NATIVE_MODULES
    assert "NativeFlatten" not in cpp.RAW_KERNELS
    assert "NativeFlatten" not in cpp.list_kernels()
    # The Conv2d module and pooling remain unimplemented and unsupported
    # (the differentiable conv2d operation itself is implemented as of D6).
    assert "NativeConv2d" in cpp.UNSUPPORTED
    assert "maxpool2d" in cpp.UNSUPPORTED
    assert "flatten" not in cpp.UNSUPPORTED  # now implemented as the module
    info = cpp.backend_info()
    assert "NativeFlatten" in info["native_modules"]


def test_native_flatten_isolated_from_stable_framework():
    import tensorforge

    # The stable Flatten is untouched and still NumPy-backed.
    stable = tensorforge.nn.Flatten()
    t = tensorforge.Tensor(np.arange(24.0).reshape(2, 3, 4), requires_grad=True)
    out = stable(t)
    assert isinstance(out, tensorforge.Tensor)
    assert out.data.shape == (2, 12)
    out.sum().backward()
    assert isinstance(t.grad, np.ndarray)
