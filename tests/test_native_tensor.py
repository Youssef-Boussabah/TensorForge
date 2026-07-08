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


# -- guardrails -------------------------------------------------------


@needs_native
def test_no_compute_or_view_methods_yet():
    """v1.8 is a shell: even though NativeTensorCore has these, the
    wrapper must not expose them yet (compute is v1.9, views v1.10)."""
    t = NativeTensor.zeros((2, 2))
    for method in (
        "relu", "add", "subtract", "multiply", "matmul",  # compute (v1.9)
        "reshape", "transpose", "narrow", "T",            # views (v1.10)
    ):
        assert not hasattr(t, method), f"NativeTensor should not expose {method} yet"
    t.close()


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
