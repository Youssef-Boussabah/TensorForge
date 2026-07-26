"""Phase F, milestone F2 — NativeLayerNorm.

The first native normalization module: stateless (no buffers, identical
in train and eval), differentiable through the mean and the population
variance, and composed entirely from existing native operations. These
tests are behavioral — values, gradients, shapes, ownership, storage
lifetime — rather than assertions about the composition's internal shape.

Nothing here is a normalization *kernel*, C ABI symbol, NativeTensorCore
method, or NativeTensor operation: F2 adds a module, and the guardrails at
the end pin exactly that.
"""

import contextlib
import gc
import types

import numpy as np
import pytest

import tensorforge as tf
from tensorforge.backends import cpp
from tensorforge.experimental import (
    NativeLayerNorm, NativeLinear, NativeParameter, NativeReLU,
    NativeSequential, NativeTensor,
)

needs_native = pytest.mark.skipif(
    not cpp.is_available(), reason="the experimental C++ backend is not built"
)

needs_fault_injection = pytest.mark.skipif(
    not cpp.is_available(), reason="the experimental C++ backend is not built"
)


# --------------------------------------------------------------------------
# Fixtures, references, helpers
# --------------------------------------------------------------------------

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


def layernorm_reference(x, normalized_shape, eps, weight=None, bias=None):
    """The population-variance LayerNorm, in NumPy — an external oracle,
    never run inside an armed tripwire."""
    x = np.asarray(x, dtype=np.float64)
    k = len(normalized_shape)
    axes = tuple(range(x.ndim - k, x.ndim))
    mean = x.mean(axis=axes, keepdims=True)
    var = ((x - mean) ** 2).mean(axis=axes, keepdims=True)
    normalized = (x - mean) / np.sqrt(var + eps)
    if weight is not None:
        normalized = normalized * weight + bias
    return normalized


def _set_affine(module, weight, bias):
    """Load nontrivial affine values through the controlled mutation
    primitive (identity preserved), returning the arrays used."""
    module.weight.copy_value_(NativeTensor.from_array(weight))
    module.bias.copy_value_(NativeTensor.from_array(bias))


def _native_objective(module, x, upstream):
    """Run ``module`` on a fresh NativeParameter input, form the scalar
    ``sum(output * upstream)``, run backward, and return
    ``(loss, input_grad, weight_grad, bias_grad)`` as NumPy arrays. The
    input parameter and the module's parameters carry the gradients."""
    xt = NativeParameter(x)
    out = module(xt)
    up = NativeTensor.from_array(upstream)
    loss = out.multiply(up).sum()
    loss.backward()
    input_grad = xt.grad.to_numpy()
    weight_grad = module.weight.grad.to_numpy() if module.elementwise_affine else None
    bias_grad = module.bias.grad.to_numpy() if module.elementwise_affine else None
    value = float(loss.to_numpy())
    for t in (loss, out, up, xt):
        t.close()
    return value, input_grad, weight_grad, bias_grad


def _reference_objective(x, normalized_shape, eps, upstream, weight=None, bias=None):
    out = layernorm_reference(x, normalized_shape, eps, weight, bias)
    return float((out * np.asarray(upstream, dtype=np.float64)).sum())


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


# ==========================================================================
# Constructor validation
# ==========================================================================

@needs_native
def test_accepts_positive_int_normalized_shape():
    module = NativeLayerNorm(4)
    assert module.normalized_shape == (4,)
    assert module.eps == 1e-5
    assert module.elementwise_affine is True


@needs_native
def test_accepts_sequence_normalized_shapes_and_custom_eps():
    assert NativeLayerNorm((3,)).normalized_shape == (3,)
    assert NativeLayerNorm((3, 4)).normalized_shape == (3, 4)
    # A list is normalized to a tuple.
    module = NativeLayerNorm([2, 3, 4], eps=1e-3)
    assert module.normalized_shape == (2, 3, 4)
    assert isinstance(module.normalized_shape, tuple)
    assert module.eps == 1e-3
    assert isinstance(module.eps, float)


@needs_native
def test_accepts_affine_enabled_and_disabled():
    assert NativeLayerNorm(3, elementwise_affine=True).elementwise_affine is True
    off = NativeLayerNorm(3, elementwise_affine=False)
    assert off.elementwise_affine is False


@needs_native
@pytest.mark.parametrize("bad", [True, False])
def test_rejects_bool_normalized_shape(bad):
    with pytest.raises(TypeError):
        NativeLayerNorm(bad)


@needs_native
@pytest.mark.parametrize("bad", [0, -1, -7])
def test_rejects_nonpositive_int_normalized_shape(bad):
    with pytest.raises(ValueError):
        NativeLayerNorm(bad)


@needs_native
@pytest.mark.parametrize("bad", [(), []])
def test_rejects_empty_sequence(bad):
    with pytest.raises(ValueError):
        NativeLayerNorm(bad)


@needs_native
@pytest.mark.parametrize("bad", [3.0, "3", None, {3: 1}, (2, (3,))])
def test_rejects_wrong_type_normalized_shape(bad):
    with pytest.raises(TypeError):
        NativeLayerNorm(bad)


@needs_native
@pytest.mark.parametrize("bad", [(2, True), (2, 3.0), (2, "3"), (True,)])
def test_rejects_wrong_type_sequence_members(bad):
    with pytest.raises(TypeError):
        NativeLayerNorm(bad)


@needs_native
@pytest.mark.parametrize("bad", [(2, 0), (0, 3), (2, -1)])
def test_rejects_nonpositive_sequence_members(bad):
    with pytest.raises(ValueError):
        NativeLayerNorm(bad)


@needs_native
@pytest.mark.parametrize("bad", ["x", None, (1e-5,), [1e-5], 1j])
def test_rejects_wrong_type_eps(bad):
    with pytest.raises(TypeError):
        NativeLayerNorm(3, eps=bad)


@needs_native
def test_rejects_bool_eps():
    with pytest.raises(TypeError):
        NativeLayerNorm(3, eps=True)


@needs_native
@pytest.mark.parametrize("bad", [0, 0.0, -1e-5, -1])
def test_rejects_nonpositive_eps(bad):
    with pytest.raises(ValueError):
        NativeLayerNorm(3, eps=bad)


@needs_native
@pytest.mark.parametrize("bad", [1, 0, "yes", None])
def test_rejects_non_bool_elementwise_affine(bad):
    with pytest.raises(TypeError):
        NativeLayerNorm(3, elementwise_affine=bad)


@needs_native
def test_invalid_construction_allocates_no_native_storage(live_storages):
    baseline = len(live_storages)
    bad_calls = [
        lambda: NativeLayerNorm(0),
        lambda: NativeLayerNorm(-3),
        lambda: NativeLayerNorm(()),
        lambda: NativeLayerNorm((2, 0)),
        lambda: NativeLayerNorm((2, True)),
        lambda: NativeLayerNorm(3.0),
        lambda: NativeLayerNorm("3"),
        lambda: NativeLayerNorm(None),
        lambda: NativeLayerNorm(True),
        lambda: NativeLayerNorm(3, eps=0),
        lambda: NativeLayerNorm(3, eps=-1),
        lambda: NativeLayerNorm(3, eps=True),
        lambda: NativeLayerNorm(3, eps="x"),
        lambda: NativeLayerNorm(3, elementwise_affine=1),
    ]
    for call in bad_calls:
        with pytest.raises((TypeError, ValueError)):
            call()
        assert len(live_storages) == baseline  # nothing was ever allocated


# ==========================================================================
# Initialization and registration
# ==========================================================================

@needs_native
def test_affine_parameters_initialize_and_register():
    module = NativeLayerNorm((3, 4))
    assert isinstance(module.weight, NativeParameter)
    assert isinstance(module.bias, NativeParameter)
    assert np.array_equal(module.weight.to_numpy(), np.ones((3, 4)))
    assert np.array_equal(module.bias.to_numpy(), np.zeros((3, 4)))
    assert module.weight.shape == (3, 4)
    assert module.bias.shape == (3, 4)
    assert module.weight.requires_grad and module.bias.requires_grad
    assert module.weight.owns_core and module.bias.owns_core
    assert module.weight.contiguous and module.bias.contiguous
    assert module.weight.version == 0 and module.bias.version == 0
    # Deterministic registration order: weight, then bias.
    assert [name for name, _ in module.named_parameters()] == ["weight", "bias"]
    assert list(module.state_dict().keys()) == ["weight", "bias"]
    for snapshot in module.state_dict().values():
        snapshot.close()
    # No buffers.
    assert module.buffers() == []
    assert list(module.named_buffers()) == []


@needs_native
def test_affine_disabled_has_no_parameters_buffers_or_state():
    module = NativeLayerNorm((3, 4), elementwise_affine=False)
    assert module.weight is None
    assert module.bias is None
    assert list(module.parameters()) == []
    assert list(module.named_parameters()) == []
    assert module.buffers() == []
    assert module.state_dict() == {}
    # Forward still works.
    x = np.random.default_rng(0).standard_normal((2, 3, 4))
    out = module(NativeTensor.from_array(x))
    assert np.allclose(out.to_numpy(),
                       layernorm_reference(x, (3, 4), 1e-5), atol=1e-12)
    out.close()


@needs_native
def test_affine_disabled_allocates_no_affine_storage(live_storages):
    baseline = len(live_storages)
    module = NativeLayerNorm((5,), elementwise_affine=False)
    assert len(live_storages) == baseline  # no weight/bias storage
    del module


# ==========================================================================
# Shape validation
# ==========================================================================

@needs_native
def test_forward_accepts_matching_trailing_shapes():
    module = NativeLayerNorm((3, 4))
    # rank == k, and with leading dims.
    for shape in [(3, 4), (2, 3, 4), (5, 2, 3, 4)]:
        x = np.random.default_rng(1).standard_normal(shape)
        out = module(NativeTensor.from_array(x))
        assert out.shape == shape
        out.close()


@needs_native
def test_forward_one_trailing_dimension():
    module = NativeLayerNorm(4)
    x = np.random.default_rng(2).standard_normal((6, 4))
    out = module(NativeTensor.from_array(x))
    assert out.shape == (6, 4)
    out.close()


@needs_native
@pytest.mark.parametrize("shape", [(3,), (2, 3), (2, 5), (2, 3, 5), (5,)])
def test_forward_rejects_shape_mismatch(shape):
    # normalized_shape (3, 4): too-low rank, wrong final dim, wrong earlier
    # trailing dim, and right element count but wrong trailing shape.
    module = NativeLayerNorm((3, 4))
    x = np.zeros(shape)
    with pytest.raises(ValueError):
        module(NativeTensor.from_array(x))


@needs_native
def test_forward_rejects_right_numel_wrong_trailing_shape():
    module = NativeLayerNorm((3, 4))       # 12 trailing elements
    x = np.zeros((2, 12))                  # 12 trailing, but shape (12,) != (3,4)
    with pytest.raises(ValueError):
        module(NativeTensor.from_array(x))


@needs_native
def test_forward_error_names_the_shapes():
    module = NativeLayerNorm((3, 4))
    x = NativeTensor.from_array(np.zeros((2, 5)))
    with pytest.raises(ValueError, match=r"\(3, 4\)"):
        module(x)
    x.close()


@needs_native
def test_forward_rejects_wrong_input_types():
    module = NativeLayerNorm(3)
    for bad in (tf.Tensor(np.zeros((2, 3))), np.zeros((2, 3)), [1, 2, 3],
                (1, 2, 3), 3.0, object()):
        with pytest.raises(TypeError):
            module(bad)


@needs_native
def test_forward_rejects_closed_input():
    module = NativeLayerNorm(3)
    x = NativeTensor.from_array(np.zeros((2, 3)))
    x.close()
    with pytest.raises(RuntimeError):
        module(x)


@needs_native
def test_shape_validation_failure_allocates_nothing(live_storages):
    module = NativeLayerNorm((3, 4))
    baseline = len(live_storages)
    x = NativeTensor.from_array(np.zeros((2, 5)))
    after_input = len(live_storages)
    with pytest.raises(ValueError):
        module(x)
    assert len(live_storages) == after_input  # no graph node was built
    x.close()
    assert len(live_storages) == baseline


# ==========================================================================
# Numerical parity
# ==========================================================================

@needs_native
@pytest.mark.parametrize("normalized_shape,input_shape", [
    (4, (5, 4)),
    ((4,), (4,)),
    ((4,), (3, 4)),
    ((3, 4), (2, 3, 4)),
    ((3, 4), (3, 4)),
    ((2, 3, 4), (5, 2, 3, 4)),
])
def test_parity_with_numpy_reference(normalized_shape, input_shape):
    rng = np.random.default_rng(3)
    x = rng.standard_normal(input_shape)
    shape = (normalized_shape,) if isinstance(normalized_shape, int) else normalized_shape
    module = NativeLayerNorm(normalized_shape)
    out = module(NativeTensor.from_array(x))
    expected = layernorm_reference(x, shape, 1e-5, np.ones(shape), np.zeros(shape))
    assert np.allclose(out.to_numpy(), expected, atol=1e-12)
    out.close()


@needs_native
def test_parity_affine_disabled():
    rng = np.random.default_rng(4)
    x = rng.standard_normal((6, 3, 4))
    module = NativeLayerNorm((3, 4), elementwise_affine=False)
    out = module(NativeTensor.from_array(x))
    assert np.allclose(out.to_numpy(),
                       layernorm_reference(x, (3, 4), 1e-5), atol=1e-12)
    out.close()


@needs_native
def test_parity_nondefault_eps_and_affine():
    rng = np.random.default_rng(5)
    x = rng.standard_normal((6, 3))
    weight = rng.standard_normal(3)
    bias = rng.standard_normal(3)
    module = NativeLayerNorm(3, eps=1e-2)
    _set_affine(module, weight, bias)
    out = module(NativeTensor.from_array(x))
    expected = layernorm_reference(x, (3,), 1e-2, weight, bias)
    assert np.allclose(out.to_numpy(), expected, atol=1e-12)
    out.close()


@needs_native
def test_parity_with_stable_layernorm():
    rng = np.random.default_rng(6)
    x = rng.standard_normal((5, 3))
    weight = rng.standard_normal(3)
    bias = rng.standard_normal(3)

    native = NativeLayerNorm(3, eps=1e-3)
    _set_affine(native, weight, bias)
    native_out = native(NativeTensor.from_array(x)).to_numpy()

    stable = tf.LayerNorm(3, eps=1e-3)
    stable.weight.data[:] = weight
    stable.bias.data[:] = bias
    stable_out = stable(tf.Tensor(x)).data

    assert np.allclose(native_out, stable_out, atol=1e-12)


# ==========================================================================
# Population variance and epsilon ordering
# ==========================================================================

@needs_native
def test_population_variance_not_sample_variance():
    # x = [1, 2, 3]: mean 2, population var = 2/3, sample var = 1.
    # With eps ~ 0 the normalized value uses population variance.
    x = np.array([[1.0, 2.0, 3.0]])
    module = NativeLayerNorm(3, eps=1e-12, elementwise_affine=False)
    out = module(NativeTensor.from_array(x)).to_numpy()
    pop = (x - 2.0) / np.sqrt(2.0 / 3.0 + 1e-12)
    sample = (x - 2.0) / np.sqrt(1.0 + 1e-12)
    assert np.allclose(out, pop, atol=1e-6)
    assert not np.allclose(out, sample, atol=1e-3)


@needs_native
def test_epsilon_is_inside_the_square_root():
    # A deliberately large eps separates sqrt(var + eps) from sqrt(var)+eps.
    x = np.array([[1.0, 2.0, 3.0]])
    eps = 0.5
    module = NativeLayerNorm(3, eps=eps, elementwise_affine=False)
    out = module(NativeTensor.from_array(x)).to_numpy()
    var = np.array([[2.0 / 3.0]])
    inside = (x - 2.0) / np.sqrt(var + eps)          # correct
    outside = (x - 2.0) / (np.sqrt(var) + eps)       # the classic bug
    assert np.allclose(out, inside, atol=1e-12)
    assert not np.allclose(out, outside, atol=1e-3)


@needs_native
def test_constant_input_stays_finite():
    # var == 0 everywhere; eps keeps the result finite and zero.
    x = np.full((2, 4), 7.0)
    module = NativeLayerNorm(4, elementwise_affine=False)
    out = module(NativeTensor.from_array(x)).to_numpy()
    assert np.all(np.isfinite(out))
    assert np.allclose(out, 0.0, atol=1e-6)


# ==========================================================================
# Train / eval equivalence
# ==========================================================================

@needs_native
def test_train_and_eval_are_identical():
    rng = np.random.default_rng(7)
    x = rng.standard_normal((4, 3))
    module = NativeLayerNorm(3)
    _set_affine(module, rng.standard_normal(3), rng.standard_normal(3))
    xt = NativeTensor.from_array(x)

    module.train()
    train_out = module(xt).to_numpy()
    module.eval()
    eval_out = module(xt).to_numpy()
    assert np.array_equal(train_out, eval_out)

    # Repeated toggles change no state and keep producing the same output.
    keys_before = list(module.state_dict().keys())
    for _ in range(3):
        module.train()
        module.eval()
    again = module(xt).to_numpy()
    assert np.array_equal(again, eval_out)
    assert list(module.state_dict().keys()) == keys_before
    assert module.buffers() == []
    xt.close()


@needs_native
def test_forward_does_not_mutate_parameters_or_versions():
    rng = np.random.default_rng(8)
    x = rng.standard_normal((4, 3))
    module = NativeLayerNorm(3)
    weight_before = module.weight.to_numpy()
    bias_before = module.bias.to_numpy()
    vw, vb = module.weight.version, module.bias.version
    out = module(NativeTensor.from_array(x))
    assert np.array_equal(module.weight.to_numpy(), weight_before)
    assert np.array_equal(module.bias.to_numpy(), bias_before)
    assert module.weight.version == vw and module.bias.version == vb
    out.close()


# ==========================================================================
# Gradients (central finite differences)
# ==========================================================================

@needs_native
@pytest.mark.parametrize("normalized_shape,input_shape,eps,affine", [
    ((3,), (4, 3), 1e-5, True),
    ((3,), (4, 3), 1e-5, False),
    ((3, 4), (2, 3, 4), 1e-3, True),
    ((2, 3), (5, 2, 3), 1e-5, True),
    (4, (4,), 1e-5, True),
])
def test_finite_difference_gradients(normalized_shape, input_shape, eps, affine):
    rng = np.random.default_rng(9)
    shape = (normalized_shape,) if isinstance(normalized_shape, int) else normalized_shape
    x = rng.standard_normal(input_shape)
    upstream = rng.standard_normal(input_shape)
    weight = rng.standard_normal(shape) if affine else None
    bias = rng.standard_normal(shape) if affine else None

    module = NativeLayerNorm(normalized_shape, eps=eps, elementwise_affine=affine)
    if affine:
        _set_affine(module, weight, bias)
    _, gx, gw, gb = _native_objective(module, x, upstream)

    # input gradient
    gx_fd = _central_difference(
        lambda xv: _reference_objective(xv, shape, eps, upstream, weight, bias), x
    )
    assert gx.shape == x.shape
    assert np.allclose(gx, gx_fd, atol=1e-5)

    if affine:
        gw_fd = _central_difference(
            lambda wv: _reference_objective(x, shape, eps, upstream, wv, bias), weight
        )
        gb_fd = _central_difference(
            lambda bv: _reference_objective(x, shape, eps, upstream, weight, bv), bias
        )
        assert gw.shape == weight.shape and gb.shape == bias.shape
        assert np.allclose(gw, gw_fd, atol=1e-5)
        assert np.allclose(gb, gb_fd, atol=1e-5)


@needs_native
def test_weight_and_bias_gradients_reduce_over_leading_dims():
    # Affine parameters are (C,); the input is (N, M, C). The parameter
    # gradients must be reduced back to (C,) across the leading dims.
    rng = np.random.default_rng(10)
    x = rng.standard_normal((3, 2, 4))
    upstream = rng.standard_normal((3, 2, 4))
    module = NativeLayerNorm(4)
    _set_affine(module, rng.standard_normal(4), rng.standard_normal(4))
    _, gx, gw, gb = _native_objective(module, x, upstream)
    assert gx.shape == (3, 2, 4)
    assert gw.shape == (4,)
    assert gb.shape == (4,)


@needs_native
def test_gradient_dtype_and_device_match():
    rng = np.random.default_rng(11)
    x = rng.standard_normal((4, 3))
    module = NativeLayerNorm(3)
    xt = NativeParameter(x)
    out = module(xt)
    loss = out.multiply(NativeTensor.from_array(np.ones((4, 3)))).sum()
    loss.backward()
    for tensor, grad in ((xt, xt.grad), (module.weight, module.weight.grad),
                         (module.bias, module.bias.grad)):
        assert grad.dtype == tensor.dtype
        assert grad.device == tensor.device
        assert grad.shape == tensor.shape
    loss.close()
    out.close()


@needs_native
def test_one_shot_backward_frees_graph_and_zero_grad_works():
    rng = np.random.default_rng(12)
    x = rng.standard_normal((4, 3))
    module = NativeLayerNorm(3)
    xt = NativeParameter(x)
    out = module(xt)
    loss = out.multiply(NativeTensor.from_array(np.ones((4, 3)))).sum()
    loss.backward()
    with pytest.raises(RuntimeError, match="freed autograd graph"):
        loss.backward()
    # zero_grad clears every gradient.
    module.zero_grad()
    xt.zero_grad()
    assert module.weight.grad is None and module.bias.grad is None
    assert xt.grad is None
    # A fresh forward/backward works after zero_grad.
    out2 = module(xt)
    out2.sum().backward()
    assert xt.grad is not None
    loss.close()
    out.close()
    out2.close()
    xt.close()


@needs_native
def test_retain_graph_allows_a_second_backward():
    rng = np.random.default_rng(13)
    x = rng.standard_normal((4, 3))
    module = NativeLayerNorm(3, elementwise_affine=False)
    xt = NativeParameter(x)
    out = module(xt)
    loss = out.multiply(NativeTensor.from_array(np.ones((4, 3)))).sum()
    loss.backward(retain_graph=True)
    first = xt.grad.to_numpy().copy()
    loss.backward()  # a second pass accumulates
    assert np.allclose(xt.grad.to_numpy(), 2.0 * first, atol=1e-9)
    loss.close()
    out.close()
    xt.close()


@needs_native
def test_duplicate_use_of_centered_accumulates_both_contributions():
    # centered is used twice inside forward (centered*centered for the
    # variance, and centered*inv_std for the normalized value). If either
    # contribution were dropped, the input gradient would be wrong — the
    # finite-difference check below is exactly that guarantee for a case
    # where the mean/variance dependence is strong.
    rng = np.random.default_rng(14)
    x = rng.standard_normal((2, 5))
    upstream = rng.standard_normal((2, 5))
    module = NativeLayerNorm(5, elementwise_affine=False)
    _, gx, _, _ = _native_objective(module, x, upstream)
    gx_fd = _central_difference(
        lambda xv: _reference_objective(xv, (5,), 1e-5, upstream), x
    )
    assert np.allclose(gx, gx_fd, atol=1e-5)


# ==========================================================================
# Non-contiguous input
# ==========================================================================

@needs_native
def test_non_contiguous_input_forward_and_backward():
    rng = np.random.default_rng(15)
    base = rng.standard_normal((3, 5))
    param = NativeParameter(base)          # owning (3, 5)
    view = param.transpose()               # borrowing (5, 3), non-contiguous
    assert not view.contiguous

    module = NativeLayerNorm(3)
    weight = rng.standard_normal(3)
    bias = rng.standard_normal(3)
    _set_affine(module, weight, bias)

    out = module(view)
    expected = layernorm_reference(base.T, (3,), 1e-5, weight, bias)
    assert np.allclose(out.to_numpy(), expected, atol=1e-12)
    # Fresh, owning, contiguous output — never a borrowing view.
    assert out.owns_core and out.contiguous
    assert out is not view

    out.multiply(NativeTensor.from_array(np.ones((5, 3)))).sum().backward()
    # The gradient lands on the base parameter at its logical (3, 5) shape.
    assert param.grad.shape == (3, 5)

    out.close()
    view.close()      # closing the borrowing view leaves the base alive
    assert param.closed is False
    param.close()


@needs_native
def test_non_contiguous_matches_contiguous_equivalent():
    rng = np.random.default_rng(16)
    base = rng.standard_normal((4, 6))
    module = NativeLayerNorm(4, elementwise_affine=False)

    view = NativeParameter(base).transpose()      # (6, 4) non-contiguous
    contiguous = NativeTensor.from_array(base.T)  # (6, 4) contiguous copy
    from_view = module(view).to_numpy()
    from_contig = module(contiguous).to_numpy()
    assert np.allclose(from_view, from_contig, atol=1e-13)


# ==========================================================================
# Sequential composition
# ==========================================================================

@needs_native
def test_layernorm_inside_a_native_sequential():
    rng = np.random.default_rng(17)
    model = NativeSequential(
        NativeLinear(4, 3, seed=0),
        NativeReLU(),
        NativeLayerNorm(3),
    )
    x = rng.standard_normal((5, 4))
    out = model(NativeTensor.from_array(x))
    assert out.shape == (5, 3)

    # Hierarchical, ordered state keys reach the LayerNorm affine params.
    keys = list(model.state_dict().keys())
    assert keys == ["0.weight", "0.bias", "2.weight", "2.bias"]
    for snapshot in model.state_dict().values():
        snapshot.close()

    # Parameter traversal order includes both the linear and the norm.
    names = [name for name, _ in model.named_parameters()]
    assert names == ["0.weight", "0.bias", "2.weight", "2.bias"]

    # Backward reaches both the surrounding Linear and the LayerNorm params.
    out.close()
    xt = NativeParameter(x)
    loss = model(xt).multiply(NativeTensor.from_array(np.ones((5, 3)))).sum()
    loss.backward()
    layernorm = model[2]
    assert model[0].weight.grad is not None
    assert layernorm.weight.grad is not None
    assert layernorm.bias.grad is not None
    loss.close()
    xt.close()


@needs_native
def test_sequential_train_eval_propagates_to_layernorm():
    model = NativeSequential(NativeLinear(4, 3, seed=1), NativeLayerNorm(3))
    model.eval()
    assert model[1].training is False
    model.train()
    assert model[1].training is True


# ==========================================================================
# State behavior
# ==========================================================================

@needs_native
def test_state_dict_keys_and_independence():
    module = NativeLayerNorm(3)
    state = module.state_dict()
    assert list(state.keys()) == ["weight", "bias"]
    # Snapshots are independent owning copies.
    snap = state["weight"]
    assert snap.owns_core
    assert snap._core is not module.weight._core
    module.weight.copy_value_(NativeTensor.from_array(np.full(3, 9.0)))
    assert np.array_equal(snap.to_numpy(), np.ones(3))  # snapshot unaffected
    for value in state.values():
        value.close()


@needs_native
def test_non_affine_state_dict_is_empty():
    module = NativeLayerNorm(3, elementwise_affine=False)
    assert module.state_dict() == {}


@needs_native
def test_load_state_dict_preserves_identity_and_increments_versions():
    source = NativeLayerNorm(3)
    _set_affine(source, np.array([1.0, 2.0, 3.0]), np.array([0.5, 0.5, 0.5]))
    state = source.state_dict()

    target = NativeLayerNorm(3)
    weight_id = id(target.weight)
    bias_id = id(target.bias)
    vw, vb = target.weight.version, target.bias.version

    result = target.load_state_dict(state)
    assert not result.missing_keys and not result.unexpected_keys
    # Identity preserved (copied in place, not reassigned).
    assert id(target.weight) == weight_id and id(target.bias) == bias_id
    assert np.array_equal(target.weight.to_numpy(), np.array([1.0, 2.0, 3.0]))
    assert np.array_equal(target.bias.to_numpy(), np.array([0.5, 0.5, 0.5]))
    # Each matched parameter's version advanced by exactly one.
    assert target.weight.version == vw + 1
    assert target.bias.version == vb + 1
    for value in state.values():
        value.close()


@needs_native
def test_invalid_load_is_atomic(live_storages):
    target = NativeLayerNorm(3)
    before_w = target.weight.to_numpy()
    vw = target.weight.version
    baseline = len(live_storages)
    # A shape-mismatched bias makes the whole load fail; the weight (which
    # would have committed) must be rolled back and no version moves.
    bad = {
        "weight": NativeTensor.from_array(np.array([5.0, 6.0, 7.0])),
        "bias": NativeTensor.from_array(np.zeros(4)),
    }
    with pytest.raises(ValueError):
        target.load_state_dict(bad)
    assert np.array_equal(target.weight.to_numpy(), before_w)
    assert target.weight.version == vw
    for value in bad.values():
        value.close()
    assert len(live_storages) == baseline


@needs_native
def test_checkpoint_round_trip(tmp_path):
    from tensorforge.experimental import (
        save_native_checkpoint, load_native_checkpoint,
    )
    rng = np.random.default_rng(18)
    source = NativeLayerNorm((2, 3))
    _set_affine(source, rng.standard_normal((2, 3)), rng.standard_normal((2, 3)))
    path = tmp_path / "layernorm.npz"
    save_native_checkpoint(str(path), source)

    target = NativeLayerNorm((2, 3))
    # The checkpoint format/version are unchanged (still 1) — proven by the
    # native-checkpoint suite; here the affine values must round-trip exactly.
    load_native_checkpoint(str(path), target)
    assert np.array_equal(target.weight.to_numpy(), source.weight.to_numpy())
    assert np.array_equal(target.bias.to_numpy(), source.bias.to_numpy())


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
# so arming them would flag a legitimate non-data use — exactly the
# distinction the cross-entropy tripwire makes.
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
    monkeypatch.setattr(NativeTensor, "to_numpy", _tripwire)


@needs_native
def test_forward_and_backward_use_no_numpy(monkeypatch):
    # Construction and reference/upstream preparation are the allowed
    # host-entry boundary and happen before the tripwire is armed.
    rng = np.random.default_rng(19)
    x = rng.standard_normal((4, 3))
    upstream = np.ones((4, 3))
    module = NativeLayerNorm(3, eps=1e-3)
    _set_affine(module, rng.standard_normal(3), rng.standard_normal(3))
    xt = NativeParameter(x)
    up = NativeTensor.from_array(upstream)
    expected = layernorm_reference(
        x, (3,), 1e-3, module.weight.to_numpy(), module.bias.to_numpy()
    )

    _arm_numpy_tripwire(monkeypatch)
    out = module(xt)                 # native forward, no NumPy
    loss = out.multiply(up).sum()    # scalar objective, no NumPy
    loss.backward()                  # native backward, no NumPy
    monkeypatch.undo()

    assert np.allclose(out.to_numpy(), expected, atol=1e-12)
    assert xt.grad.shape == (4, 3)
    for t in (loss, out, up, xt):
        t.close()


@needs_native
def test_affine_disabled_forward_uses_no_numpy(monkeypatch):
    rng = np.random.default_rng(20)
    x = rng.standard_normal((2, 3, 4))
    module = NativeLayerNorm((3, 4), elementwise_affine=False)
    xt = NativeTensor.from_array(x)
    expected = layernorm_reference(x, (3, 4), 1e-5)

    _arm_numpy_tripwire(monkeypatch)
    out = module(xt)
    monkeypatch.undo()

    assert np.allclose(out.to_numpy(), expected, atol=1e-12)
    out.close()
    xt.close()


# ==========================================================================
# Ownership and leak behavior
# ==========================================================================

def _collect():
    """Force the composed autograd graph's intermediate wrappers — which
    participate in reference cycles, a property of the Python-managed
    native autograd engine since Phase B, not of LayerNorm — to their
    deterministic collection point. The live-storage count is exact
    regardless, so the baseline assertion still proves nothing leaked."""
    gc.collect()


@needs_native
def test_no_grad_forward_returns_to_baseline_without_gc(live_storages):
    # A forward that requires no grad builds no graph and therefore no
    # cycles: cleanup is immediate under reference counting.
    baseline = len(live_storages)
    module = NativeLayerNorm(3, elementwise_affine=False)
    x = NativeTensor.from_array(np.random.default_rng(21).standard_normal((4, 3)))
    out = module(x)
    assert np.all(np.isfinite(out.to_numpy()))   # usable for its lifetime
    out.close()
    x.close()
    assert len(live_storages) == baseline


@needs_native
def test_affine_forward_backward_returns_to_baseline(live_storages):
    baseline = len(live_storages)
    module = NativeLayerNorm(3)
    x = np.random.default_rng(22).standard_normal((4, 3))
    xt = NativeParameter(x)
    out = module(xt)
    loss = out.multiply(NativeTensor.from_array(np.ones((4, 3)))).sum()
    loss.backward()
    for t in (loss, out, xt, xt.grad,
              module.weight, module.weight.grad,
              module.bias, module.bias.grad):
        t.close()
    _collect()
    assert len(live_storages) == baseline


@needs_native
def test_affine_disabled_forward_backward_returns_to_baseline(live_storages):
    baseline = len(live_storages)
    module = NativeLayerNorm((3, 4), elementwise_affine=False)
    x = np.random.default_rng(23).standard_normal((2, 3, 4))
    xt = NativeParameter(x)
    out = module(xt)
    loss = out.multiply(NativeTensor.from_array(np.ones((2, 3, 4)))).sum()
    loss.backward()
    for t in (loss, out, xt, xt.grad):
        t.close()
    _collect()
    assert len(live_storages) == baseline


@needs_native
def test_repeated_cycles_do_not_grow_storage(live_storages):
    module = NativeLayerNorm(3)
    baseline = len(live_storages)
    for _ in range(6):
        xt = NativeParameter(np.random.default_rng(24).standard_normal((4, 3)))
        out = module(xt)
        loss = out.multiply(NativeTensor.from_array(np.ones((4, 3)))).sum()
        loss.backward()
        for t in (loss, out, xt, xt.grad):
            t.close()
        module.zero_grad()
        _collect()
    assert len(live_storages) == baseline  # module params only; no net growth


@needs_native
def test_retain_graph_then_release_returns_to_baseline(live_storages):
    baseline = len(live_storages)
    module = NativeLayerNorm(3, elementwise_affine=False)
    xt = NativeParameter(np.random.default_rng(25).standard_normal((4, 3)))
    out = module(xt)
    loss = out.multiply(NativeTensor.from_array(np.ones((4, 3)))).sum()
    loss.backward(retain_graph=True)
    loss.backward()  # final release
    for t in (loss, out, xt, xt.grad):
        t.close()
    _collect()
    assert len(live_storages) == baseline


@needs_native
def test_closed_weight_is_rejected_and_leaks_nothing(live_storages):
    module = NativeLayerNorm(3)
    module.weight.close()
    x = NativeTensor.from_array(np.zeros((2, 3)))
    baseline = len(live_storages)
    with pytest.raises(RuntimeError):
        module(x)
    assert len(live_storages) == baseline  # no graph node was built
    x.close()


@needs_native
def test_closed_bias_is_rejected():
    module = NativeLayerNorm(3)
    module.bias.close()
    x = NativeTensor.from_array(np.zeros((2, 3)))
    with pytest.raises(RuntimeError):
        module(x)
    x.close()


@needs_native
def test_explicit_parameter_closure_returns_to_baseline(live_storages):
    baseline = len(live_storages)
    module = NativeLayerNorm((3, 4))
    assert len(live_storages) == baseline + 2   # weight + bias
    module.weight.close()
    module.bias.close()
    assert len(live_storages) == baseline


@needs_native
def test_close_is_idempotent():
    module = NativeLayerNorm(3)
    weight = module.weight
    weight.close()
    weight.close()   # no double-free, no error
    assert weight.closed is True


@needs_fault_injection
def test_bias_allocation_failure_releases_weight(monkeypatch, live_storages):
    import tensorforge.experimental.native_layernorm as native_layernorm

    created = []
    real_parameter = native_layernorm.NativeParameter

    def spy(*args, **kwargs):
        param = real_parameter(*args, **kwargs)
        created.append(param)
        return param

    monkeypatch.setattr(native_layernorm, "NativeParameter", spy)
    baseline = len(live_storages)

    # weight is native allocation #1 (succeeds), bias is #2 (forced to fail).
    cpp._arm_alloc_failure(2)
    try:
        with pytest.raises(MemoryError):
            NativeLayerNorm(4)
    finally:
        cpp._arm_alloc_failure(0)
        cpp._require_library().tf_clear_error()

    assert len(created) == 1                 # only the weight became an object
    assert created[0].closed is True         # released deterministically
    assert len(live_storages) == baseline    # back to the exact baseline


# ==========================================================================
# Deterministic cleanup when forward raises after creating intermediates
# ==========================================================================
#
# A forward that raises part-way through must release every native
# intermediate it already created *immediately* — not eventually via
# garbage collection. This is stronger than the completed-graph leak tests
# above (which legitimately force `gc.collect()` for the returned graph's
# reference cycles): here no output is returned and no graph is handed to
# the caller, so the temporaries are unadopted and must be closed on the
# way out. It matters because the grad-building path puts each
# `sqrt`/`reciprocal` result node into a reference cycle (its backward
# closure captures the node itself), which reference counting alone cannot
# reclaim — so without explicit cleanup a failure after those nodes exist
# would leak until GC.


class _Boom(RuntimeError):
    """A distinctive injected forward failure."""


@contextlib.contextmanager
def _inject_boom(method_name, nth):
    """Make ``NativeTensor.<method_name>`` raise ``_Boom`` on its ``nth``
    entry, restoring it afterwards. A private test seam over an existing
    method — no production failure-control API is added."""
    original = getattr(NativeTensor, method_name)
    state = {"calls": 0}

    def patched(self, *args, **kwargs):
        state["calls"] += 1
        if state["calls"] == nth:
            raise _Boom(f"injected failure at {method_name} call #{nth}")
        return original(self, *args, **kwargs)

    setattr(NativeTensor, method_name, patched)
    try:
        yield
    finally:
        setattr(NativeTensor, method_name, original)


@pytest.fixture
def storage_tracker(monkeypatch):
    """The exact set of currently-open native storages — the project's
    deterministic instrumentation, hooking ``__init__``/``close`` so the
    set is truthful without any reliance on GC.

    Set **equality** against a baseline snapshot is the load-bearing
    assertion: an id present that was not at baseline is a leak, and a
    baseline id now absent means a caller-owned storage was wrongly closed.
    (A harmful double-*free* is impossible independently: ``close()`` and
    ``NativeStorage.close`` both guard on their handle, so the native
    memory is released at most once per object — verified in the codebase's
    existing lifetime tests — while benign idempotent ``close()`` calls
    happen throughout the engine and are not double-frees.)"""
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
    return types.SimpleNamespace(open_ids=open_ids)


@needs_native
@pytest.mark.parametrize("method,nth,affine,where", [
    # (1) During the variance reduction, after mean/centered/centered².
    ("mean", 2, True, "variance-reduction, affine"),
    ("mean", 2, False, "variance-reduction, non-affine"),
    # (2) During sqrt / reciprocal, after eps and variance+eps exist. The
    # reciprocal cases are the ones that leaked until GC before the fix.
    ("sqrt", 1, True, "sqrt, affine"),
    ("sqrt", 1, False, "sqrt, non-affine"),
    ("reciprocal", 1, True, "reciprocal, affine"),
    ("reciprocal", 1, False, "reciprocal, non-affine"),
    # (3) During the affine multiply / add, after the whole normalization
    # graph (including the self-cycling sqrt/reciprocal nodes) exists.
    ("multiply", 3, True, "affine multiply"),
    ("add", 2, True, "affine add"),
])
def test_mid_forward_failure_releases_temporaries_without_gc(
    method, nth, affine, where, storage_tracker
):
    x = NativeParameter(np.random.default_rng(30).standard_normal((4, 3)))
    module = NativeLayerNorm(3, elementwise_affine=affine)

    x_before = x.to_numpy()
    x_version = x.version
    if affine:
        weight_before = module.weight.to_numpy()
        bias_before = module.bias.to_numpy()
        weight_version = module.weight.version
        bias_version = module.bias.version

    baseline_ids = set(storage_tracker.open_ids)

    with _inject_boom(method, nth):
        with pytest.raises(_Boom):          # the injected exception is preserved
            module(x)

    # Immediate cleanup — deliberately NO gc.collect() here. Set equality is
    # exact in both directions: no id beyond baseline (nothing leaked), and
    # every baseline id still present (no caller-owned storage was closed).
    assert storage_tracker.open_ids == baseline_ids, (
        f"{where}: native storage did not return to baseline without GC"
    )

    # The input, weight, and bias remain open and unchanged (no
    # use-after-close, no mutation, no version move, no gradient).
    assert x.closed is False
    assert np.array_equal(x.to_numpy(), x_before)
    assert x.version == x_version
    assert x.grad is None
    if affine:
        assert module.weight.closed is False and module.bias.closed is False
        assert np.array_equal(module.weight.to_numpy(), weight_before)
        assert np.array_equal(module.bias.to_numpy(), bias_before)
        assert module.weight.version == weight_version
        assert module.bias.version == bias_version
        assert module.weight.grad is None and module.bias.grad is None

    # A later normal forward/backward still succeeds — the engine and the
    # caller-owned tensors were left in a usable state (had cleanup closed
    # or double-closed any of them, this would raise a closed-tensor error).
    out = module(x)
    out.multiply(NativeTensor.from_array(np.ones((4, 3)))).sum().backward()
    assert x.grad is not None
    if affine:
        assert module.weight.grad is not None and module.bias.grad is not None

    out.close()
    x.close()
    if affine:
        module.weight.close()
        module.bias.close()


@needs_native
def test_mid_forward_failure_on_multi_axis_releases_every_reduction(
    storage_tracker
):
    # A (H, W) normalized shape makes _mean_over take two single-axis means
    # per statistic; failing the very last reduction proves the tracked
    # intermediate means are released too, not just the final result.
    x = NativeParameter(np.random.default_rng(31).standard_normal((2, 3, 4)))
    module = NativeLayerNorm((3, 4))
    baseline_ids = set(storage_tracker.open_ids)

    # mean calls: 1,2 = mean chain; 3,4 = variance chain. Fail the 4th.
    with _inject_boom("mean", 4):
        with pytest.raises(_Boom):
            module(x)

    assert storage_tracker.open_ids == baseline_ids
    assert x.closed is False and module.weight.closed is False
    x.close()
    module.weight.close()
    module.bias.close()


# ==========================================================================
# Capability and guardrail tests
# ==========================================================================

def test_native_layernorm_is_exported():
    import tensorforge.experimental as experimental
    assert "NativeLayerNorm" in experimental.__all__
    assert experimental.NativeLayerNorm is NativeLayerNorm
    # It did not leak into the stable namespace.
    import tensorforge
    assert not hasattr(tensorforge, "NativeLayerNorm")


def test_native_layernorm_is_in_native_modules_only():
    assert "NativeLayerNorm" in cpp.NATIVE_MODULES
    # A module, not an operation, kernel, loss, or metric.
    for inventory in (cpp.TENSOR_CORE_OPS, cpp.AUTOGRAD_OPS, cpp.RAW_KERNELS,
                      cpp.NATIVE_LOSSES, cpp.NATIVE_METRICS, cpp.UNSUPPORTED):
        assert "NativeLayerNorm" not in inventory


def test_layernorm_left_unsupported_at_f2():
    # "layernorm" left UNSUPPORTED when F2 shipped NativeLayerNorm.
    # ("batchnorm" stayed until F4 shipped the second BatchNorm shape;
    # that removal is pinned in tests/test_native_batchnorm2d.py.)
    assert "layernorm" not in cpp.UNSUPPORTED
    assert "NativeLayerNorm" in cpp.NATIVE_MODULES


def test_f2_added_no_operation_kernel_core_or_abi():
    # No normalization operation at any layer, no raw kernel, no Core method.
    for name in ("layer_norm", "batch_norm", "layernorm", "batchnorm",
                 "layer_norm_forward", "layer_norm_backward"):
        assert name not in cpp.TENSOR_CORE_OPS, name
        assert name not in cpp.AUTOGRAD_OPS, name
        assert name not in cpp.RAW_KERNELS, name
        assert not hasattr(cpp.NativeTensorCore, name), name
    # No public NativeTensor normalization operation.
    for name in ("layer_norm", "batch_norm", "layernorm", "normalize"):
        assert not hasattr(NativeTensor, name), name
    # No functional layer_norm export.
    import tensorforge.experimental as experimental
    assert not hasattr(experimental, "layer_norm")
    assert "layer_norm" not in experimental.__all__


def test_layernorm_is_one_of_three_shipped_normalization_modules():
    # F2 shipped LayerNorm, F3 the (N, C) BatchNorm, F4 the NCHW one.
    # LayerNorm stays exactly what it was: a separate, stateless module.
    import tensorforge.experimental as experimental
    for module in ("NativeLayerNorm", "NativeBatchNorm1d",
                   "NativeBatchNorm2d"):
        assert module in cpp.NATIVE_MODULES, module
        assert module in experimental.__all__, module
    assert "layernorm" not in cpp.UNSUPPORTED
    assert "batchnorm" not in cpp.UNSUPPORTED
    # LayerNorm is not a BatchNorm: it shares no implementation with them
    # and still holds no buffers.
    from tensorforge.experimental import native_batchnorm
    assert not issubclass(NativeLayerNorm, native_batchnorm._NativeBatchNorm)


def test_other_capability_tuples_remain_exact():
    assert cpp.NATIVE_LOSSES == ("NativeMSELoss", "NativeCrossEntropyLoss")
    assert cpp.NATIVE_METRICS == ("native_accuracy",)
    assert cpp.NATIVE_OPTIMIZERS == ("NativeSGD", "NativeAdam")
    assert cpp.STATE_SUPPORT == (
        "persistent_buffers", "state_dict", "load_state_dict",
        "generator_state",   # Phase G, milestone G1 (in-memory only)
        "save_native_checkpoint", "load_native_checkpoint",
    )
    assert cpp.SUPPORTED_DTYPES == ("float64",)
    assert cpp.SUPPORTED_DEVICES == ("cpu",)
    assert cpp.UNSUPPORTED == ("dropout", "float32", "cuda", "amp")
