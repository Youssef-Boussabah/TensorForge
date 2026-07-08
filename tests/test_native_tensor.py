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
def test_elementwise_ops_reject_shape_mismatch_no_broadcasting():
    # (2, 3) with (3,) broadcasts in NumPy; the native path rejects it.
    a = NativeTensor.from_array(np.ones((2, 3)))
    row = NativeTensor.from_array([1.0, 2.0, 3.0])
    for op in ("add", "subtract", "multiply"):
        with pytest.raises(ValueError, match="broadcasting|shape"):
            getattr(a, op)(row)
    a.close()
    row.close()


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


# -- guardrails -------------------------------------------------------


@needs_native
def test_no_view_methods_yet():
    """v1.9 adds compute ops but not view ops — those are v1.10. The
    view methods exist on NativeTensorCore but must not be exposed on
    the wrapper yet."""
    t = NativeTensor.zeros((2, 2))
    for method in ("reshape", "transpose", "narrow", "T"):
        assert not hasattr(t, method), f"NativeTensor should not expose {method} yet"
    t.close()


def test_no_operator_overloads_yet():
    """Compute is method-only for now — no Python operator sugar. These
    must not be custom NativeTensor methods yet."""
    for dunder in ("__add__", "__sub__", "__mul__", "__matmul__"):
        assert dunder not in vars(NativeTensor), (
            f"NativeTensor should not define {dunder} yet"
        )


@needs_native
def test_no_autograd_surface():
    """A forward-only tensor carries none of Tensor's autograd machinery."""
    t = NativeTensor.zeros((2, 2))
    for attr in ("grad", "requires_grad", "backward", "_backward", "zero_grad"):
        assert not hasattr(t, attr), f"NativeTensor should not have {attr}"
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
