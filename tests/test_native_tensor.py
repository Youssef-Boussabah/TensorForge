"""Tests for the experimental NativeTensor wrapper (Stage 2, v1.8).

NativeTensor is a forward-only shell over NativeTensorCore:
constructors, metadata, to_numpy, and lifetime — no compute ops and no
view ops yet. Native-backend tests skip when the compiled library is
not built, following the same pattern as test_backends.py. See
docs/native_tensor_wrapper_design.md.
"""

import numpy as np
import pytest

from tensorforge.backends import cpp
from tensorforge.experimental import NativeTensor

needs_native = pytest.mark.skipif(
    not cpp.is_available(),
    reason="experimental C++ backend not built",
)


# -- import behavior --------------------------------------------------


def test_native_tensor_imports_cleanly():
    # Importing the experimental package and pulling NativeTensor out of
    # it must always work, built backend or not.
    from tensorforge.experimental import NativeTensor as _NT

    assert _NT is NativeTensor


def test_importing_tensorforge_does_not_import_experimental():
    """Importing tensorforge must not pull in the experimental package
    — it stays opt-in. A static check of the framework __init__ so it
    can't be fooled by another test importing experimental earlier."""
    import ast
    from pathlib import Path

    init = (
        Path(__file__).resolve().parent.parent
        / "src" / "tensorforge" / "__init__.py"
    )
    tree = ast.parse(init.read_text(encoding="utf-8"))
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)
        elif isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
    assert not any(
        module.startswith("tensorforge.experimental") for module in imported
    )


# -- constructors and metadata ----------------------------------------


@needs_native
def test_from_array_round_trips_through_to_numpy():
    x = np.array([[1.0, -2.0, 3.0], [4.0, 5.0, -6.0]])
    with NativeTensor.from_array(x) as t:
        result = t.to_numpy()
        assert isinstance(result, np.ndarray)
        assert result.dtype == np.float64
        assert np.array_equal(result, x)


@needs_native
def test_zeros_has_correct_shape_and_data():
    with NativeTensor.zeros((2, 3)) as t:
        assert t.shape == (2, 3)
        assert np.array_equal(t.to_numpy(), np.zeros((2, 3)))


@needs_native
def test_full_has_correct_shape_and_data():
    with NativeTensor.full((2, 2), 7.0) as t:
        assert t.shape == (2, 2)
        assert np.array_equal(t.to_numpy(), np.full((2, 2), 7.0))


@needs_native
def test_metadata_properties_are_correct():
    with NativeTensor.from_array(np.arange(6.0).reshape(2, 3)) as t:
        assert t.shape == (2, 3)
        assert t.strides == (3, 1)
        assert t.ndim == 2
        assert t.numel == 6
        assert t.contiguous is True


@needs_native
def test_to_numpy_returns_an_independent_copy():
    with NativeTensor.from_array([1.0, 2.0, 3.0]) as t:
        out = t.to_numpy()
        out[0] = 99.0
        # Mutating the materialized array must not touch native storage.
        assert np.array_equal(t.to_numpy(), [1.0, 2.0, 3.0])


# -- ownership --------------------------------------------------------


@needs_native
def test_constructors_own_their_core():
    for tensor in (
        NativeTensor.from_array([1.0, 2.0]),
        NativeTensor.zeros((2, 2)),
        NativeTensor.full((3,), 4.0),
    ):
        assert tensor.owns_core is True
        tensor.close()


# -- lifetime ---------------------------------------------------------


@needs_native
def test_close_is_idempotent_and_updates_closed():
    t = NativeTensor.zeros((2, 2))
    assert t.closed is False
    t.close()
    assert t.closed is True
    t.close()  # double close is safe
    assert t.closed is True


@needs_native
def test_to_numpy_after_close_fails_clearly():
    t = NativeTensor.zeros((2, 2))
    t.close()
    with pytest.raises(RuntimeError, match="closed"):
        t.to_numpy()


@needs_native
def test_metadata_after_close_fails_clearly():
    t = NativeTensor.from_array([[1.0, 2.0]])
    t.close()
    # Every layout property rejects access on a closed tensor.
    for attr in ("shape", "strides", "ndim", "numel", "contiguous"):
        with pytest.raises(RuntimeError, match="closed"):
            getattr(t, attr)
    # ...but lifetime state stays readable.
    assert t.closed is True
    assert t.owns_core is True


@needs_native
def test_context_manager_closes_on_exit():
    with NativeTensor.zeros((2, 2)) as t:
        assert t.closed is False
    assert t.closed is True


# -- repr -------------------------------------------------------------


@needs_native
def test_repr_shows_metadata_and_is_safe_when_closed():
    t = NativeTensor.zeros((2, 3))
    text = repr(t)
    # Metadata only — never the data itself.
    assert "NativeTensor" in text
    assert "shape=(2, 3)" in text
    t.close()
    # A closed tensor reprs safely, without touching released storage.
    assert repr(t) == "NativeTensor(closed)"


# -- forward compute --------------------------------------------------


@needs_native
def test_relu_matches_numpy_and_returns_owning_tensor():
    x = np.array([[1.0, -2.0], [-3.0, 4.0]])
    src = NativeTensor.from_array(x)
    out = src.relu()
    assert isinstance(out, NativeTensor)
    assert out.owns_core is True
    assert out.shape == (2, 2)
    assert np.array_equal(out.to_numpy(), np.maximum(x, 0.0))
    # The input is untouched and still usable.
    assert src.closed is False
    assert np.array_equal(src.to_numpy(), x)
    src.close()
    out.close()


@needs_native
def test_elementwise_binary_ops_match_numpy():
    x = np.array([[1.0, 2.0], [3.0, 4.0]])
    y = np.array([[10.0, 20.0], [30.0, 40.0]])
    a = NativeTensor.from_array(x)
    b = NativeTensor.from_array(y)
    for op, expected in (("add", x + y), ("subtract", x - y), ("multiply", x * y)):
        out = getattr(a, op)(b)
        assert isinstance(out, NativeTensor)
        assert out.owns_core is True
        assert out.shape == (2, 2)
        assert np.array_equal(out.to_numpy(), expected)
        out.close()
    a.close()
    b.close()


@needs_native
def test_elementwise_ops_broadcast_through_wrapper():
    # v1.17: the wrapper inherits broadcasting from NativeTensorCore with
    # no wrapper-specific code. (2, 3) with (3,) now broadcasts (matching
    # NumPy) rather than being rejected.
    x = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    row = np.array([10.0, 20.0, 30.0])
    a = NativeTensor.from_array(x)
    b = NativeTensor.from_array(row)
    for op, numpy_op in (("add", np.add), ("subtract", np.subtract), ("multiply", np.multiply)):
        out = getattr(a, op)(b)
        assert isinstance(out, NativeTensor)
        assert out.owns_core is True       # result is a fresh owning tensor
        assert out.shape == (2, 3)
        assert np.array_equal(out.to_numpy(), numpy_op(x, row))
        assert np.array_equal(a.to_numpy(), x)  # operands unchanged
        out.close()
    a.close()
    b.close()


@needs_native
def test_elementwise_ops_broadcast_scalar_through_wrapper():
    # scalar-operand broadcasting also rides through the wrapper.
    x = np.array([[1.0, -2.0], [3.0, 4.0]])
    a = NativeTensor.from_array(x)
    s = NativeTensor.from_array(5.0)  # shape ()
    assert np.array_equal(a.multiply(s).to_numpy(), x * 5.0)
    assert np.array_equal(s.add(a).to_numpy(), 5.0 + x)  # scalar on the left too
    a.close()
    s.close()


@needs_native
def test_elementwise_ops_reject_incompatible_shapes():
    # Genuinely incompatible shapes still raise a clear ValueError naming
    # both shapes — no silent NumPy fallback.
    a = NativeTensor.from_array(np.ones((2, 3)))
    bad = NativeTensor.from_array(np.ones((4, 3)))  # 2 vs 4 on axis 0
    for op in ("add", "subtract", "multiply"):
        with pytest.raises(ValueError, match="broadcast") as excinfo:
            getattr(a, op)(bad)
        assert "(2, 3)" in str(excinfo.value) and "(4, 3)" in str(excinfo.value)
    a.close()
    bad.close()


@needs_native
def test_binary_ops_reject_non_native_tensor_operand():
    a = NativeTensor.from_array([[1.0, 2.0]])
    for op in ("add", "subtract", "multiply", "matmul"):
        # A NumPy array is not a NativeTensor — a clear TypeError naming
        # NativeTensor, not a mysterious error about NativeTensorCore.
        with pytest.raises(TypeError, match="NativeTensor"):
            getattr(a, op)(np.array([[1.0, 2.0]]))
    a.close()


@needs_native
def test_binary_ops_reject_closed_operand():
    a = NativeTensor.from_array([[1.0, 2.0], [3.0, 4.0]])
    b = NativeTensor.from_array([[1.0, 2.0], [3.0, 4.0]])
    b.close()
    for op in ("add", "subtract", "multiply", "matmul"):
        with pytest.raises(RuntimeError, match="closed"):
            getattr(a, op)(b)
    a.close()


@needs_native
def test_matmul_matches_numpy():
    x = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    y = np.array([[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]])
    a = NativeTensor.from_array(x)
    b = NativeTensor.from_array(y)
    out = a.matmul(b)
    assert isinstance(out, NativeTensor)
    assert out.owns_core is True
    assert out.shape == (2, 2)
    assert np.allclose(out.to_numpy(), x @ y)
    a.close()
    b.close()
    out.close()


@needs_native
def test_matmul_rejects_incompatible_shapes():
    a = NativeTensor.from_array(np.ones((2, 3)))
    b = NativeTensor.from_array(np.ones((2, 3)))  # inner dims 3 vs 2
    with pytest.raises(ValueError, match="inner dimensions|shape"):
        a.matmul(b)
    a.close()
    b.close()


@needs_native
def test_compute_on_closed_self_fails_clearly():
    a = NativeTensor.zeros((2, 2))
    b = NativeTensor.zeros((2, 2))
    a.close()
    with pytest.raises(RuntimeError, match="closed"):
        a.relu()
    for op in ("add", "subtract", "multiply", "matmul"):
        with pytest.raises(RuntimeError, match="closed"):
            getattr(a, op)(b)
    b.close()


@needs_native
def test_compute_ops_chain():
    x = NativeTensor.from_array([[1.0, -1.0], [2.0, 3.0]])
    y = NativeTensor.from_array([[1.0, 1.0], [1.0, 1.0]])
    z = NativeTensor.from_array([[2.0, 0.0], [0.0, 2.0]])
    out = x.relu().add(y).matmul(z)
    expected = (np.maximum([[1.0, -1.0], [2.0, 3.0]], 0.0) + 1.0) @ [[2.0, 0.0], [0.0, 2.0]]
    assert np.allclose(out.to_numpy(), expected)
    for tensor in (x, y, z, out):
        tensor.close()


# -- view operations --------------------------------------------------


@needs_native
def test_reshape_borrows_and_matches_numpy():
    x = np.arange(6.0).reshape(2, 3)
    owner = NativeTensor.from_array(x)
    v = owner.reshape((3, 2))
    assert isinstance(v, NativeTensor)
    assert v.owns_core is False
    assert v.shape == (3, 2)
    assert np.array_equal(v.to_numpy(), x.reshape(3, 2))
    assert owner.closed is False
    v.close()
    owner.close()


@needs_native
def test_transpose_and_T_match_numpy():
    x = np.arange(6.0).reshape(2, 3)
    owner = NativeTensor.from_array(x)
    for v, expected in (
        (owner.transpose(), x.T),
        (owner.transpose(1, 0), np.transpose(x, (1, 0))),
        (owner.T, x.T),
    ):
        assert isinstance(v, NativeTensor)
        assert v.owns_core is False
        assert np.array_equal(v.to_numpy(), expected)
        v.close()
    assert owner.closed is False
    owner.close()


@needs_native
def test_narrow_matches_numpy_slicing():
    x = np.arange(12.0).reshape(3, 4)
    owner = NativeTensor.from_array(x)
    v = owner.narrow(1, 1, 2)  # keep columns 1..2
    assert v.owns_core is False
    assert v.shape == (3, 2)
    assert np.array_equal(v.to_numpy(), x[:, 1:3])
    assert owner.closed is False
    v.close()
    owner.close()


@needs_native
def test_contiguous_copy_owns_and_materializes_view():
    x = np.arange(6.0).reshape(2, 3)
    owner = NativeTensor.from_array(x)
    view = owner.transpose()  # non-contiguous
    assert view.contiguous is False
    copy = view.contiguous_copy()
    assert isinstance(copy, NativeTensor)
    assert copy.owns_core is True
    assert copy.contiguous is True
    assert np.array_equal(copy.to_numpy(), x.T)
    # Closing the independent copy leaves the original owner alive.
    copy.close()
    assert owner.closed is False
    assert np.array_equal(owner.to_numpy(), x)
    view.close()
    owner.close()


@needs_native
def test_closing_a_view_does_not_close_the_owner():
    x = np.arange(6.0).reshape(2, 3)
    owner = NativeTensor.from_array(x)
    view = owner.reshape((3, 2))
    view.close()
    assert view.closed is True
    assert owner.closed is False
    assert np.array_equal(owner.to_numpy(), x)  # owner still usable
    owner.close()


@needs_native
def test_closing_owner_invalidates_view_data_access():
    owner = NativeTensor.from_array(np.arange(6.0).reshape(2, 3))
    view = owner.transpose()
    owner.close()
    # The shared storage is gone, so the view's data access raises.
    with pytest.raises(RuntimeError, match="closed"):
        view.to_numpy()
    view.close()


@needs_native
def test_wrapper_inherits_contiguous_fast_path():
    """The v1.14 contiguous fast path lives below the wrapper in
    NativeTensorCore, so NativeTensor gets it with no code change. Both a
    plain contiguous tensor and a nonzero-offset contiguous view (a
    ``narrow`` along axis 0) compute correctly through the wrapper — the
    same values whether the fast or generic kernel runs beneath."""
    x = np.arange(15.0).reshape(5, 3) - 7.0
    owner = NativeTensor.from_array(x)
    other = NativeTensor.from_array(x)
    # Plain contiguous operands.
    assert np.array_equal(owner.relu().to_numpy(), np.maximum(x, 0.0))
    assert np.array_equal(owner.add(other).to_numpy(), x + x)
    # Nonzero-offset contiguous row slices still land on the fast path.
    rows = owner.narrow(0, 1, 3)      # x[1:4], contiguous, offset != 0
    rows2 = other.narrow(0, 1, 3)
    assert rows.contiguous is True
    assert np.array_equal(rows.relu().to_numpy(), np.maximum(x[1:4], 0.0))
    assert np.array_equal(rows.multiply(rows2).to_numpy(), x[1:4] * x[1:4])
    for tensor in (rows, rows2, owner, other):
        tensor.close()


@needs_native
def test_compute_works_on_views():
    x = np.array([[1.0, -2.0, 3.0], [4.0, 5.0, -6.0]])
    owner = NativeTensor.from_array(x)
    view = owner.transpose()  # (3, 2), strided
    # relu over a strided view
    assert np.array_equal(view.relu().to_numpy(), np.maximum(x.T, 0.0))
    # elementwise add of two matching strided views (hold both owners)
    owner2 = NativeTensor.from_array(x)
    other = owner2.transpose()
    assert np.array_equal(view.add(other).to_numpy(), x.T + x.T)
    # transposed view matmul: (3, 2) @ (2, 3)
    rhs = NativeTensor.from_array(np.ones((2, 3)))
    assert np.allclose(view.matmul(rhs).to_numpy(), x.T @ np.ones((2, 3)))
    for tensor in (owner, view, owner2, other, rhs):
        tensor.close()


@needs_native
def test_reductions_inherit_through_wrapper():
    # v1.19: NativeTensor.sum/mean delegate to NativeTensorCore, so the
    # wrapper gets reductions with no reduction-specific logic.
    x = np.arange(6.0).reshape(2, 3)
    a = NativeTensor.from_array(x)
    for method, numpy_op in (("sum", np.sum), ("mean", np.mean)):
        for axis, keep in ((None, False), (0, False), (1, True), (-1, False)):
            out = getattr(a, method)(axis=axis, keepdims=keep)
            assert isinstance(out, NativeTensor)
            assert out.owns_core is True
            expected = numpy_op(x, axis=axis, keepdims=keep)
            assert out.shape == np.shape(expected)
            assert np.allclose(out.to_numpy(), expected)
            out.close()
    assert np.array_equal(a.to_numpy(), x)  # input unchanged
    a.close()


@needs_native
def test_reductions_work_on_views_through_wrapper():
    x = np.arange(12.0).reshape(3, 4)
    owner = NativeTensor.from_array(x)
    view = owner.transpose()  # (4, 3), strided
    assert np.allclose(view.sum(axis=0).to_numpy(), x.T.sum(axis=0))
    assert np.allclose(view.mean().to_numpy(), x.T.mean())
    owner.close()
    view.close()


@needs_native
def test_reductions_on_closed_tensor_raise():
    a = NativeTensor.zeros((2, 2))
    a.close()
    with pytest.raises(RuntimeError, match="closed"):
        a.sum()
    with pytest.raises(RuntimeError, match="closed"):
        a.mean(axis=0)


@needs_native
def test_view_ops_reject_invalid_inputs():
    owner = NativeTensor.from_array(np.arange(6.0).reshape(2, 3))
    with pytest.raises(ValueError):
        owner.reshape((4, 2))  # 8 elements != 6
    with pytest.raises(ValueError):
        owner.transpose(0, 0)  # not a permutation
    with pytest.raises(ValueError):
        owner.narrow(1, 2, 5)  # out of bounds for size 3
    owner.close()


@needs_native
def test_dtype_device_defaults_and_delegate_to_core():
    for tensor in (
        NativeTensor.from_array([[1.0, 2.0], [3.0, 4.0]]),
        NativeTensor.zeros((2, 3)),
        NativeTensor.full((2,), 5.0),
    ):
        assert tensor.dtype == "float64"
        assert tensor.device == "cpu"
        tensor.close()


@needs_native
def test_dtype_device_explicit_args_and_from_array_default():
    with NativeTensor.zeros((2, 2), dtype="float64", device="cpu") as z:
        assert z.dtype == "float64" and z.device == "cpu"
    with NativeTensor.full((3,), 1.0, dtype="float64", device="cpu") as f:
        assert f.dtype == "float64" and f.device == "cpu"
    with NativeTensor.from_array([1.0], dtype=None) as a:  # None -> float64
        assert a.dtype == "float64" and a.device == "cpu"


@needs_native
def test_dtype_device_preserved_through_ops_and_views():
    x = np.arange(6.0).reshape(2, 3)
    a = NativeTensor.from_array(x)
    b = NativeTensor.from_array(x)
    for result in (a.relu(), a.add(b), a.sum(axis=0), a.mean(), a.transpose(), a.reshape((3, 2))):
        assert result.dtype == "float64"
        assert result.device == "cpu"
        if result.owns_core:
            result.close()
    a.close()
    b.close()


@needs_native
def test_constructors_reject_unsupported_dtype_device():
    for ctor in (
        lambda: NativeTensor.zeros((2, 2), dtype="float32"),
        lambda: NativeTensor.full((2,), 0.0, device="cuda"),
        lambda: NativeTensor.from_array([1.0], dtype="int64"),
    ):
        with pytest.raises(ValueError):
            ctor()


@needs_native
def test_dtype_device_rejected_after_close_like_other_metadata():
    t = NativeTensor.from_array([1.0, 2.0])
    t.close()
    # dtype/device follow the wrapper's other metadata: rejected on a
    # closed tensor (unlike the core, whose metadata stays readable).
    for attr in ("dtype", "device"):
        with pytest.raises(RuntimeError, match="closed"):
            getattr(t, attr)


@needs_native
def test_view_ops_on_closed_self_fail_clearly():
    owner = NativeTensor.from_array(np.arange(6.0).reshape(2, 3))
    owner.close()
    with pytest.raises(RuntimeError, match="closed"):
        owner.reshape((3, 2))
    with pytest.raises(RuntimeError, match="closed"):
        owner.transpose()
    with pytest.raises(RuntimeError, match="closed"):
        owner.narrow(0, 0, 1)
    with pytest.raises(RuntimeError, match="closed"):
        owner.contiguous_copy()
    with pytest.raises(RuntimeError, match="closed"):
        _ = owner.T


# -- guardrails -------------------------------------------------------


def test_no_operator_overloads_yet():
    """Compute is method-only for now — no Python operator sugar. These
    must not be custom NativeTensor methods yet."""
    for dunder in ("__add__", "__sub__", "__mul__", "__matmul__"):
        assert dunder not in vars(NativeTensor), (
            f"NativeTensor should not define {dunder} yet"
        )


@needs_native
def test_autograd_surface_present_with_forward_only_defaults():
    """As of v2.1 NativeTensor carries the native autograd surface, but a
    plain constructed tensor defaults to forward-only: requires_grad
    False, grad None, and it is a leaf. (Full autograd behavior is
    covered in test_native_autograd.py.)"""
    t = NativeTensor.zeros((2, 2))
    for attr in ("grad", "requires_grad", "is_leaf", "backward", "zero_grad", "detach"):
        assert hasattr(t, attr), f"NativeTensor should expose {attr}"
    assert t.requires_grad is False
    assert t.grad is None
    assert t.is_leaf is True
    t.close()


def test_native_tensor_is_not_tensorforge_tensor():
    from tensorforge import Tensor

    assert NativeTensor is not Tensor
    assert not issubclass(NativeTensor, Tensor)


# -- unavailable native backend ---------------------------------------


def test_constructors_fail_helpfully_when_backend_unavailable(monkeypatch):
    """When the library is unbuilt, a constructor must surface the
    build-instructions ImportError from the native runtime, never a
    mysterious AttributeError. Simulated so the test runs even where the
    backend is built."""
    def _unbuilt(*args, **kwargs):
        raise ImportError("The experimental C++ backend is not built ...")

    monkeypatch.setattr(
        cpp.NativeTensorCore, "zeros", staticmethod(_unbuilt)
    )
    with pytest.raises(ImportError, match="not built"):
        NativeTensor.zeros((2, 2))
