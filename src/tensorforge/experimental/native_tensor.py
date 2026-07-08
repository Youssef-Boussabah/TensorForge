"""NativeTensor — a minimal, forward-only wrapper over NativeTensorCore.

This is the Stage-2 wrapper described in
docs/native_tensor_wrapper_design.md, at its v1.8 shell stage:
constructors, metadata, conversion, and lifetime — no compute ops and
no view ops yet (those are v1.9 and v1.10).

NativeTensor is deliberately **not** tensorforge.Tensor: it carries no
autograd graph, no ``requires_grad``, no ``grad``, and no
``backward()``. It is a small convenience layer that owns (or borrows)
exactly one NativeTensorCore and exposes a Tensor-shaped surface for
experimenting with the native backend in isolation.

Conversion crosses the native boundary only by explicit call:
``from_array`` enters, ``to_numpy`` exits, both as copies. Lifetime is
explicit too — ``close()`` (or a ``with`` block) releases the native
storage an owning tensor holds; a closed tensor rejects metadata and
materialization clearly rather than reading freed layout.
"""

import numpy as np

from ..backends import cpp


class NativeTensor:
    """A forward-only native tensor: one NativeTensorCore plus an
    explicit ownership and lifetime story.

    Create one with ``from_array`` / ``zeros`` / ``full`` (each owns the
    core it wraps). Read ``shape`` / ``strides`` / ``ndim`` / ``numel``
    / ``contiguous`` for layout; ``to_numpy()`` to materialize a fresh
    float64 copy. Release native memory with ``close()`` or a ``with``
    block; ``owns_core`` and ``closed`` report lifetime state.

    Not tensorforge.Tensor, and no compute or view operations yet — see
    the module docstring and docs/native_tensor_wrapper_design.md.
    """

    __slots__ = ("_core", "_owns_core", "_closed")

    def __init__(self, core, owns_core=True):
        """Wrap an existing NativeTensorCore. Prefer the ``from_array``
        / ``zeros`` / ``full`` constructors; this is the low-level entry
        point (see also the internal ``_from_core``)."""
        if not isinstance(core, cpp.NativeTensorCore):
            raise TypeError(
                f"NativeTensor wraps a NativeTensorCore, got "
                f"{type(core).__name__}"
            )
        self._core = core
        self._owns_core = bool(owns_core)
        self._closed = False

    @classmethod
    def _from_core(cls, core, owns_core=True):
        """Internal: wrap a core produced elsewhere (e.g. a future
        compute or view op). Kept private so the public surface stays a
        small, forward-only shell and NativeTensorCore is not the normal
        way in."""
        return cls(core, owns_core=owns_core)

    # -- constructors -----------------------------------------------------

    @classmethod
    def from_array(cls, values):
        """A contiguous native tensor holding a copy of ``values``.

        This is the explicit *entry* boundary: array-like/NumPy data in,
        a new owning NativeTensor out, its data copied into fresh C++
        storage.
        """
        return cls._from_core(cpp.NativeTensorCore.from_array(values))

    @classmethod
    def zeros(cls, shape):
        """A row-major contiguous native tensor of ``shape``, all zeros."""
        return cls._from_core(cpp.NativeTensorCore.zeros(shape))

    @classmethod
    def full(cls, shape, fill_value):
        """A row-major contiguous native tensor of ``shape`` filled with
        ``fill_value``."""
        return cls._from_core(cpp.NativeTensorCore.full(shape, fill_value))

    # -- lifetime gate ----------------------------------------------------

    def _require_open(self):
        """The core behind an open tensor. Metadata and materialization
        both go through here, so a closed NativeTensor rejects them
        clearly instead of reading a released layout."""
        if self._closed:
            raise RuntimeError("this NativeTensor has been closed")
        return self._core

    # -- metadata (rejected after close) ----------------------------------

    @property
    def shape(self):
        return self._require_open().shape

    @property
    def strides(self):
        return self._require_open().strides

    @property
    def ndim(self):
        return self._require_open().ndim

    @property
    def numel(self):
        return self._require_open().numel

    @property
    def contiguous(self):
        return self._require_open().contiguous

    # -- lifetime state (always readable) ---------------------------------

    @property
    def closed(self):
        """True once ``close()`` has run. Readable even after close."""
        return self._closed

    @property
    def owns_core(self):
        """True if this tensor owns its NativeTensorCore (so ``close()``
        releases the native storage). Constructor-made tensors own their
        core; future borrowing views will not. Readable after close."""
        return self._owns_core

    # -- conversion -------------------------------------------------------

    def to_numpy(self):
        """Materialize into a fresh, independent float64 NumPy array.

        The explicit *exit* boundary: the returned array shares no
        mutable state with native storage. Raises RuntimeError if the
        tensor has been closed.
        """
        return self._require_open().to_numpy()

    # -- lifetime ---------------------------------------------------------

    def close(self):
        """Release this tensor's hold on its core. An owning tensor frees
        the native storage; a borrowing one would detach only itself.
        Idempotent — safe to call more than once."""
        if not self._closed:
            self._closed = True
            if self._owns_core:
                self._core.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
        return False

    def __del__(self):
        # Defensive cleanup only — correctness never depends on GC
        # timing; use close() or a with block.
        try:
            self.close()
        except Exception:
            pass

    def __repr__(self):
        if self._closed:
            return "NativeTensor(closed)"
        return (
            f"NativeTensor(shape={self._core.shape}, "
            f"contiguous={self._core.contiguous})"
        )
