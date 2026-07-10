"""NativeTensor — a native tensor wrapper over NativeTensorCore, with an
opt-in Python-managed autograd graph.

This is the Stage-2 wrapper described in
docs/native_tensor_wrapper_design.md. Its forward surface is complete —
constructors, metadata, ``to_numpy``, forward compute
(``relu``/``add``/``subtract``/``multiply``/``matmul``/``sum``/``mean``),
metadata-only views (``reshape``/``transpose``/``T``/``narrow``) and
``contiguous_copy`` — and an explicit ownership/lifetime story sits under
all of it.

As of v2.1 (Phase B), NativeTensor also carries the **native autograd
metadata skeleton and the reverse-topological backward driver**:
``requires_grad``, ``grad``, ``is_leaf``, ``zero_grad()``, ``detach()``,
and ``backward()``. The graph is **Python-managed at this layer** — the
raw forward runtime (NativeTensorCore) and the C++ kernels stay
autograd-unaware, exactly as the design requires. Gradients are
NativeTensor-backed native storage (never NumPy) and obey the v1.21
dtype/device metadata contract (``grad.dtype == tensor.dtype``,
``grad.device == tensor.device``, ``grad.shape == tensor.shape``).

As of v2.2, the core differentiable operations are **wired into that
engine**: ``add``/``subtract``/``multiply``/``relu``/``sum``/``mean``/
``matmul``/``reshape``/``transpose``/``T``/``contiguous_copy`` build
graph nodes when an operand requires grad (plain forward tensors
otherwise, exactly as before), with broadcasting handled on the way
back by an ``unbroadcast`` reduction. Backward math runs entirely on
native forward kernels at the ``NativeTensorCore`` level — building no
graph of its own and never touching NumPy. ``narrow`` stays outside
autograd (its backward needs a native scatter primitive; deferred to
v2.3). See docs/native_autograd_design.md.

A note on mutation: there are no in-place user arithmetic operations,
so the forward values the graph captures for backward (e.g. the
operands of ``multiply``) remain stable for the life of the graph —
unless a tensor is explicitly closed, in which case ``backward()``
raises clearly rather than reading freed storage.

NativeTensor is still **not** tensorforge.Tensor: the two autograd
engines never mix, no conversion is implicit, and the framework frontend
never imports this. Conversion crosses the native boundary only by
explicit call: ``from_array`` enters, ``to_numpy`` exits, both as copies.
Lifetime is explicit — ``close()`` (or a ``with`` block) releases the
native storage an owning tensor holds; a closed tensor rejects metadata,
materialization, and gradient operations clearly rather than reading
freed layout.
"""

import numpy as np

from ..backends import cpp


class NativeTensor:
    """A native tensor: one NativeTensorCore, an explicit ownership and
    lifetime story, and an opt-in Python-managed autograd graph.

    Create one with ``from_array`` / ``zeros`` / ``full`` (each owns the
    core it wraps; pass ``requires_grad=True`` to track gradients). Read
    ``shape`` / ``strides`` / ``ndim`` / ``numel`` / ``contiguous`` /
    ``dtype`` / ``device`` for metadata; ``to_numpy()`` to materialize a
    fresh float64 copy. For autograd read ``requires_grad`` / ``grad`` /
    ``is_leaf`` and call ``backward()`` / ``zero_grad()`` / ``detach()``.
    Release native memory with ``close()`` or a ``with`` block;
    ``owns_core`` and ``closed`` report lifetime state.

    Not tensorforge.Tensor — see the module docstring,
    docs/native_tensor_wrapper_design.md, and
    docs/native_autograd_design.md.
    """

    __slots__ = (
        "_core", "_owns_core", "_closed",
        # -- autograd metadata (opt-in; see docs/native_autograd_design.md)
        "_requires_grad", "_grad", "_parents", "_backward", "_op", "_is_leaf",
    )

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
        # A fresh wrapper is a forward-only leaf by default: no gradient
        # tracking and no graph edges. Constructors flip _requires_grad on
        # request; the internal _from_op records parents/backward/op for a
        # non-leaf differentiable result.
        self._requires_grad = False
        self._grad = None
        self._parents = ()
        self._backward = None
        self._op = ""
        self._is_leaf = True

    @classmethod
    def _from_core(cls, core, owns_core=True):
        """Internal: wrap a core produced elsewhere (a compute or view
        op). Kept private so NativeTensorCore is not the normal way in.
        The result is a forward-only leaf (``requires_grad=False``, no
        graph) — what every op returns when no operand requires grad."""
        return cls(core, owns_core=owns_core)

    @classmethod
    def _from_op(cls, core, parents, backward, op, owns_core=True):
        """Internal: build a **non-leaf** autograd graph node over
        ``core``.

        ``parents`` is a tuple of the NativeTensors this result was
        computed from; ``backward`` is a callable taking this node's
        upstream gradient (a NativeTensor) and accumulating a contribution
        into each parent (via ``parent._accumulate_grad(...)``); ``op`` is
        a short debug name. ``requires_grad`` is the OR of the parents' —
        if no parent needs grad, the result is a plain forward leaf with
        no recorded graph, matching the Python Tensor.

        This is the single internal entry for graph construction: the
        differentiable ops (v2.2) and the autograd tests use it, and it
        stays private so arbitrary graph building is never a public
        surface. The graph lives here, at the NativeTensor layer — the
        core and the C++ kernels remain autograd-unaware."""
        node = cls._from_core(core, owns_core=owns_core)
        if any(p._requires_grad for p in parents):
            node._requires_grad = True
            node._is_leaf = False
            node._parents = tuple(parents)
            node._backward = backward
            node._op = op
        return node

    # -- constructors -----------------------------------------------------

    @classmethod
    def from_array(cls, values, dtype=None, device="cpu", requires_grad=False):
        """A contiguous native tensor holding a copy of ``values``.

        This is the explicit *entry* boundary: array-like/NumPy data in,
        a new owning NativeTensor out, its data copied into fresh C++
        storage. ``dtype``/``device`` default to ``"float64"``/``"cpu"``
        and are rejected if unsupported. Pass ``requires_grad=True`` to
        make this a gradient-tracking leaf (see docs/native_autograd_design.md).
        """
        tensor = cls._from_core(
            cpp.NativeTensorCore.from_array(values, dtype=dtype, device=device)
        )
        tensor._init_requires_grad(requires_grad)
        return tensor

    @classmethod
    def zeros(cls, shape, dtype="float64", device="cpu", requires_grad=False):
        """A row-major contiguous native tensor of ``shape``, all zeros.
        ``dtype``/``device`` default to ``"float64"``/``"cpu"`` and are
        rejected if unsupported. Pass ``requires_grad=True`` for a
        gradient-tracking leaf."""
        tensor = cls._from_core(
            cpp.NativeTensorCore.zeros(shape, dtype=dtype, device=device)
        )
        tensor._init_requires_grad(requires_grad)
        return tensor

    @classmethod
    def full(cls, shape, fill_value, dtype="float64", device="cpu", requires_grad=False):
        """A row-major contiguous native tensor of ``shape`` filled with
        ``fill_value``. ``dtype``/``device`` default to
        ``"float64"``/``"cpu"`` and are rejected if unsupported. Pass
        ``requires_grad=True`` for a gradient-tracking leaf."""
        tensor = cls._from_core(
            cpp.NativeTensorCore.full(shape, fill_value, dtype=dtype, device=device)
        )
        tensor._init_requires_grad(requires_grad)
        return tensor

    def _init_requires_grad(self, requires_grad):
        """Validate and set a leaf's ``requires_grad`` flag. Only ``bool``
        is accepted (``int``/``None``/... raise a clear TypeError, so a
        stray ``requires_grad=1`` never silently passes). User-created
        tensors stay leaves whether or not they track gradients — the
        PyTorch convention."""
        if not isinstance(requires_grad, bool):
            raise TypeError(
                f"requires_grad must be a bool, got {type(requires_grad).__name__}"
            )
        self._requires_grad = requires_grad

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

    @property
    def dtype(self):
        """The element type tag (``"float64"``), delegated to the core.
        Rejected after close, like the other layout metadata."""
        return self._require_open().dtype

    @property
    def device(self):
        """The device tag (``"cpu"``), delegated to the core. Rejected
        after close, like the other layout metadata."""
        return self._require_open().device

    # -- autograd metadata (rejected after close) -------------------------

    @property
    def requires_grad(self):
        """Whether this tensor tracks gradients. Rejected after close,
        like the other metadata."""
        self._require_open()
        return self._requires_grad

    @property
    def grad(self):
        """The accumulated gradient (a NativeTensor) or ``None``. Starts
        ``None``, is filled lazily by ``backward()``, and is cleared to
        ``None`` by ``zero_grad()``. It obeys the dtype/device metadata
        contract: ``grad.dtype``/``grad.device``/``grad.shape`` match this
        tensor's. Rejected after close."""
        self._require_open()
        return self._grad

    @property
    def is_leaf(self):
        """True for a user-created tensor (a constructor result, or a
        ``detach()``) and for any result that does not require grad; False
        for a differentiable operation result. Rejected after close."""
        self._require_open()
        return self._is_leaf

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

    # -- forward compute (delegates to NativeTensorCore) ------------------
    #
    # As of v2.2 each compute op is differentiable: when an operand
    # requires grad the result is a graph node (via _from_op) whose
    # backward closure pushes native gradient contributions to the
    # parents. When nothing requires grad, the op returns a plain
    # forward tensor exactly as before — no graph metadata, no closure.
    # Backward math is computed at the NativeTensorCore level so it
    # never builds graph nodes of its own; parents skipped when they do
    # not require grad; contributions are always safe to retain (fresh
    # owning storage, or the upstream tensor itself, never a borrowing
    # view over a transient).

    def relu(self):
        """max(x, 0) elementwise, computed natively over this tensor's
        layout. Returns a new owning NativeTensor; the original stays
        open. Differentiable: the gradient passes through where the input
        was > 0 and is blocked elsewhere (0 included), via the fused
        native relu_backward kernel."""
        core = self._require_open()
        out_core = core.relu()
        if not self._requires_grad:
            return self._from_core(out_core)

        def _backward(upstream):
            contribution = self._require_open().relu_backward(
                upstream._require_open()
            )
            self._accumulate_grad(NativeTensor._from_core(contribution))

        return self._from_op(out_core, (self,), _backward, "relu")

    def add(self, other):
        """self + other elementwise, natively. Identical shapes or
        NumPy-style broadcasting. Returns a new owning NativeTensor.
        Differentiable: each operand receives the upstream gradient,
        summed back over any broadcast axes."""
        out_core = self._binary_forward("add", other)
        if not (self._requires_grad or other._requires_grad):
            return self._from_core(out_core)
        a, b = self, other

        def _backward(upstream):
            # d(a + b)/da = d(a + b)/db = 1.
            if a._requires_grad:
                a._accumulate_grad(_unbroadcast(upstream, a.shape))
            if b._requires_grad:
                b._accumulate_grad(_unbroadcast(upstream, b.shape))

        return self._from_op(out_core, (a, b), _backward, "add")

    def subtract(self, other):
        """self - other elementwise, natively. Identical shapes or
        NumPy-style broadcasting. Returns a new owning NativeTensor.
        Differentiable: the left operand receives the upstream gradient,
        the right its negation, each summed back over broadcast axes."""
        out_core = self._binary_forward("subtract", other)
        if not (self._requires_grad or other._requires_grad):
            return self._from_core(out_core)
        a, b = self, other

        def _backward(upstream):
            # d(a - b)/da = 1, d(a - b)/db = -1. Negation happens after
            # the unbroadcast reduction (sum commutes with it, and the
            # reduced tensor is smaller) and never mutates upstream —
            # the same object may flow to the other parent.
            if a._requires_grad:
                a._accumulate_grad(_unbroadcast(upstream, a.shape))
            if b._requires_grad:
                b._accumulate_grad(_negated(_unbroadcast(upstream, b.shape)))

        return self._from_op(out_core, (a, b), _backward, "subtract")

    def multiply(self, other):
        """self * other elementwise, natively. Identical shapes or
        NumPy-style broadcasting. Returns a new owning NativeTensor.
        Differentiable: da = upstream * b, db = upstream * a (native
        multiply), each summed back over broadcast axes. The graph keeps
        both operands alive, and with no in-place arithmetic their
        forward values stay valid until backward runs."""
        out_core = self._binary_forward("multiply", other)
        if not (self._requires_grad or other._requires_grad):
            return self._from_core(out_core)
        a, b = self, other

        def _backward(upstream):
            u_core = upstream._require_open()
            if a._requires_grad:
                contribution = NativeTensor._from_core(
                    u_core.multiply(b._require_open())
                )
                a._accumulate_grad(_unbroadcast(contribution, a.shape))
            if b._requires_grad:
                contribution = NativeTensor._from_core(
                    u_core.multiply(a._require_open())
                )
                b._accumulate_grad(_unbroadcast(contribution, b.shape))

        return self._from_op(out_core, (a, b), _backward, "multiply")

    def matmul(self, other):
        """(m, n) @ (n, p) matrix multiply, natively. 2-D only, no
        broadcasting. Returns a new owning NativeTensor. Differentiable:
        da = upstream @ b.T, db = a.T @ upstream, over the native matmul
        reading the transposed views directly (no materialization)."""
        out_core = self._binary_forward("matmul", other)
        if not (self._requires_grad or other._requires_grad):
            return self._from_core(out_core)
        a, b = self, other

        def _backward(upstream):
            u_core = upstream._require_open()
            if a._requires_grad:
                a._accumulate_grad(NativeTensor._from_core(
                    u_core.matmul(b._require_open().T)
                ))
            if b._requires_grad:
                b._accumulate_grad(NativeTensor._from_core(
                    a._require_open().T.matmul(u_core)
                ))

        return self._from_op(out_core, (a, b), _backward, "matmul")

    def sum(self, axis=None, keepdims=False):
        """Sum over ``axis`` (``None`` = all elements) natively, delegating
        to NativeTensorCore. Returns a new owning NativeTensor; the
        original stays open. Differentiable: every summed element receives
        the upstream gradient of its output cell, broadcast back natively
        to the input shape (reduced axes reinserted as size 1 first, so
        ``keepdims=False`` upstreams — the scalar shape () included —
        align correctly)."""
        core = self._require_open()
        out_core = core.sum(axis=axis, keepdims=keepdims)
        if not self._requires_grad:
            return self._from_core(out_core)
        x_shape = core.shape

        def _backward(upstream):
            self._accumulate_grad(_broadcast_back(upstream, x_shape, axis))

        return self._from_op(out_core, (self,), _backward, "sum")

    def mean(self, axis=None, keepdims=False):
        """Mean over ``axis`` (``None`` = all elements) natively, delegating
        to NativeTensorCore. Returns a new owning NativeTensor; the
        original stays open. Differentiable: sum's broadcast-back rule
        scaled by 1/count, where count is ``numel`` for ``axis=None`` and
        ``shape[axis]`` for a single (possibly negative) axis."""
        core = self._require_open()
        out_core = core.mean(axis=axis, keepdims=keepdims)
        if not self._requires_grad:
            return self._from_core(out_core)
        x_shape = core.shape
        if axis is None:
            count = core.numel
        else:
            count = x_shape[cpp._normalize_axis(axis, x_shape)]

        def _backward(upstream):
            # Scale before broadcasting back — same values, computed at
            # the smaller upstream shape. The scaling is a native
            # multiply against a broadcast scalar (no new kernel, no
            # NumPy, nothing mutated).
            u_core = upstream._require_open()
            scale = cpp.NativeTensorCore.full(
                (), 1.0 / count, dtype=u_core.dtype, device=u_core.device
            )
            scaled = NativeTensor._from_core(u_core.multiply(scale))
            scale.close()
            contribution = _broadcast_back(scaled, x_shape, axis)
            scaled.close()  # _broadcast_back returned independent storage
            self._accumulate_grad(contribution)

        return self._from_op(out_core, (self,), _backward, "mean")

    def _binary_forward(self, op_name, other):
        """Shared plumbing for the binary compute ops: require self and
        other open, require other to be a NativeTensor (a clear
        TypeError otherwise), then delegate to the core method — which
        enforces broadcast-compatible shapes / 2-D matmul and raises a
        clear ValueError on a mismatch. Returns the fresh owning result
        core; the caller decides whether it becomes a graph node."""
        core = self._require_open()
        if not isinstance(other, NativeTensor):
            raise TypeError(
                f"NativeTensor.{op_name} requires a NativeTensor operand, "
                f"got {type(other).__name__}"
            )
        other_core = other._require_open()
        return getattr(core, op_name)(other_core)

    # -- view operations (metadata only: no data is copied) ---------------

    def reshape(self, new_shape):
        """A view of the same storage with ``new_shape`` (row-major).

        Metadata only — the result borrows this tensor's storage
        (``owns_core`` is False), so closing it leaves this tensor
        alive. Requires a contiguous tensor and the same element count;
        an incompatible layout or count raises ValueError.
        Differentiable: a reshape is a pure relabeling, so its backward
        is the inverse reshape of the upstream gradient.
        """
        out_core = self._require_open().reshape(new_shape)  # validates
        if not self._requires_grad:
            return self._from_core(out_core, owns_core=False)
        original_shape = self.shape

        def _backward(upstream):
            u_core = upstream._require_open()
            if not u_core.contiguous:  # a user-supplied gradient view
                u_core = _native_copy(u_core)
            # The reshape is a borrowing view over the (transient)
            # upstream; materialize the contribution into independent
            # owning storage before it is retained as a grad.
            contribution = _native_copy(u_core.reshape(original_shape))
            self._accumulate_grad(NativeTensor._from_core(contribution))

        return self._from_op(out_core, (self,), _backward, "reshape",
                             owns_core=False)

    def transpose(self, *axes):
        """A view with permuted axes. Metadata only — the result borrows
        this tensor's storage (``owns_core`` is False).

        With no arguments, all axes are reversed (NumPy behavior).
        Explicit axes must be a complete permutation of ``range(ndim)``.
        Differentiable: the backward applies the inverse permutation to
        the upstream gradient.
        """
        core = self._require_open()
        out_core = core.transpose(*axes)  # validates the permutation
        if not self._requires_grad:
            return self._from_core(out_core, owns_core=False)
        # Recover the normalized permutation the core applied (same
        # rules: a single tuple/list argument unpacks, no arguments
        # reverses all axes) and build its pure-Python inverse.
        if len(axes) == 1 and isinstance(axes[0], (tuple, list)):
            axes = tuple(axes[0])
        if axes:
            permutation = tuple(int(axis) for axis in axes)
        else:
            permutation = tuple(reversed(range(core.ndim)))
        inverse = [0] * len(permutation)
        for position, axis in enumerate(permutation):
            inverse[axis] = position
        inverse = tuple(inverse)

        def _backward(upstream):
            # The adjoint of a permutation is its inverse permutation;
            # materialized so the retained grad owns its storage.
            u_core = upstream._require_open()
            contribution = _native_copy(u_core.transpose(inverse))
            self._accumulate_grad(NativeTensor._from_core(contribution))

        return self._from_op(out_core, (self,), _backward, "transpose",
                             owns_core=False)

    @property
    def T(self):
        """``transpose()`` with all axes reversed — NumPy's ``.T``
        semantics. A borrowing view."""
        return self.transpose()

    def narrow(self, dim, start, length):
        """A view keeping ``length`` positions of dimension ``dim`` from
        ``start``. Metadata only — the result borrows this tensor's
        storage (``owns_core`` is False). Out-of-bounds arguments raise
        ValueError; non-int arguments raise TypeError.

        **Not differentiable yet**: narrow's backward must scatter the
        upstream gradient into zeros of the original shape (un-narrowed
        positions get zero gradient), which needs a native scatter
        primitive the runtime does not have — deferred to the v2.3
        autograd-completion milestone rather than faked through NumPy.
        The result is always a plain forward tensor.
        """
        return self._from_core(
            self._require_open().narrow(dim, start, length), owns_core=False
        )

    def contiguous_copy(self):
        """A new **owning** NativeTensor with the same values in
        row-major contiguous native storage. Always copies (even when
        this tensor is already contiguous), so the result is independent
        of this one's lifetime. Differentiable: the forward copies each
        logical element unchanged, so the backward passes the upstream
        gradient through as-is — gradients live at the logical shape,
        making the parent's storage layout irrelevant."""
        out_core = self._require_open().contiguous_copy()
        if not self._requires_grad:
            return self._from_core(out_core)

        def _backward(upstream):
            self._accumulate_grad(upstream)

        return self._from_op(out_core, (self,), _backward, "contiguous_copy")

    # -- autograd (opt-in; Python-managed graph over native ops) ----------

    def _accumulate_grad(self, grad):
        """Add ``grad`` (a NativeTensor) into this tensor's ``.grad``.

        Gradients from multiple paths sum. The first contribution is
        adopted directly (no needless ``0 + g`` allocation); later ones
        are summed with the native ``add`` kernel — **no NumPy touches the
        gradient path**. A contribution to a tensor that does not require
        grad is dropped (the Python Tensor does the same). Only a
        NativeTensor gradient is accepted, and a closed tensor rejects
        accumulation (so a ``backward`` reaching a closed graph node
        raises rather than reading freed storage)."""
        self._require_open()
        if not self._requires_grad:
            return
        if not isinstance(grad, NativeTensor):
            raise TypeError(
                f"a native gradient must be a NativeTensor, got "
                f"{type(grad).__name__}"
            )
        if self._grad is None:
            self._grad = grad
        else:
            self._grad = self._grad.add(grad)

    def zero_grad(self):
        """Clear this tensor's gradient to ``None``.

        This is the v2.0 design choice — clear rather than allocate zeros
        (consistent with the framework's optimizers, and it sidesteps
        in-place mutation). Idempotent, and it leaves ``requires_grad``
        and all graph metadata untouched. Rejected on a closed tensor. The
        previous grad object is **dropped, not closed** — user code may
        hold a reference to it, and native storage is reclaimed by GC — so
        a live ``t.grad`` is never invalidated out from under the
        caller."""
        self._require_open()
        self._grad = None

    def detach(self):
        """A new leaf NativeTensor holding this tensor's current value,
        detached from the autograd graph: ``requires_grad=False``,
        ``grad=None``, no parents, no backward. Gradients never flow
        through it (a ``backward`` reaching a detach boundary stops there).

        It returns an **owning copy** (a native contiguous
        materialization), sharing no storage with this tensor — so its
        lifetime is fully independent (no double-close, no dangling core)
        and no NumPy round trip occurs. Rejected on a closed tensor."""
        core = self._require_open()
        return self._from_core(core.contiguous_copy())

    def backward(self, gradient=None):
        """Run reverse-mode autodiff from this tensor back to the leaves.

        Mirrors the Python Tensor engine (docs/autograd.md) natively:
        topologically sort the graph reachable through ``_parents``, seed
        this output's gradient, then walk that order **in reverse** so a
        node's gradient is complete before it propagates to its parents.
        Leaf gradients accumulate into ``.grad`` and are retained;
        non-leaf gradients are transient and cleared at the end (only
        leaves keep grad, matching PyTorch and the v2.0 design).

        ``gradient`` seeds the output:
        - a scalar output (``numel == 1``) defaults it to a native
          ``1.0``;
        - a non-scalar output **requires** an explicit ``gradient`` (else
          a clear ValueError) — no reduction is invented silently.
        An explicit ``gradient`` must be a NativeTensor matching this
        output's shape, dtype, and device (errors name expected vs.
        actual). No NumPy array ever enters the gradient path.

        Raises RuntimeError on a tensor that does not require grad, or that
        has been closed. ``retain_graph`` is intentionally not offered yet
        (see docs/native_autograd_design.md): the graph is rebuilt every
        call, so repeated ``backward()`` accumulates into leaf grads until
        ``zero_grad()`` clears them."""
        self._require_open()
        if not self._requires_grad:
            raise RuntimeError(
                "backward() called on a tensor that does not require grad"
            )

        seed = self._seed_gradient(gradient)

        # Post-order DFS over _parents, keyed by object identity. id() is
        # used explicitly (rather than a set of tensors) so the traversal
        # never depends on NativeTensor hashing/equality, and a node
        # reachable by several paths — or listed twice as a parent — is
        # still visited exactly once.
        topo = []
        visited = set()

        def build_topo(node):
            if id(node) in visited:
                return
            visited.add(id(node))
            for parent in node._parents:
                build_topo(parent)
            topo.append(node)

        build_topo(self)

        # Seed this output, then push gradients backward. Reverse
        # topological order guarantees each node's grad is fully
        # accumulated before its own backward rule runs.
        self._grad = seed
        for node in reversed(topo):
            if (
                node._requires_grad
                and node._backward is not None
                and node._grad is not None
            ):
                node._backward(node._grad)

        # Only leaves retain gradients; drop the transient non-leaf grads
        # (dropped, not closed — see zero_grad).
        for node in topo:
            if not node._is_leaf:
                node._grad = None

    def _seed_gradient(self, gradient):
        """Validate an explicit ``gradient`` or synthesize the default
        seed. Returns a NativeTensor matching this output's
        shape/dtype/device. (``self`` is already known open here.)"""
        core = self._core
        if gradient is None:
            if core.numel != 1:
                raise ValueError(
                    f"backward on a non-scalar output (shape {core.shape}) "
                    f"requires an explicit gradient"
                )
            # d(out)/d(out) = 1, as a native scalar-shaped tensor.
            return NativeTensor.full(
                core.shape, 1.0, dtype=core.dtype, device=core.device
            )
        if not isinstance(gradient, NativeTensor):
            raise TypeError(
                f"gradient must be a NativeTensor, got {type(gradient).__name__}"
            )
        g_core = gradient._require_open()
        if g_core.shape != core.shape:
            raise ValueError(
                f"gradient shape {g_core.shape} does not match output shape "
                f"{core.shape}"
            )
        if g_core.dtype != core.dtype or g_core.device != core.device:
            raise ValueError(
                f"gradient dtype/device {g_core.dtype}/{g_core.device} does "
                f"not match output {core.dtype}/{core.device}"
            )
        return gradient

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
        parts = [
            f"shape={self._core.shape}",
            f"contiguous={self._core.contiguous}",
        ]
        # Metadata only — never materialize data. The autograd marker
        # appears only when set, so forward-only reprs are unchanged.
        if self._requires_grad:
            parts.append("requires_grad=True")
        return f"NativeTensor({', '.join(parts)})"


# ---------------------------------------------------------------------------
# Autograd gradient-math helpers.
#
# These compute backward contributions at the NativeTensorCore level —
# native forward kernels only, no NumPy, and no graph nodes of their own
# (a gradient never requires grad). Each returns either the input tensor
# unchanged (only when it already is the exact contribution) or a fresh
# owning contiguous tensor, so a retained grad can never be a borrowing
# view over a transient whose owner is about to be dropped and closed.
# ---------------------------------------------------------------------------


def _native_copy(core):
    """A fresh owning row-major contiguous NativeTensorCore with the same
    values as ``core`` — computed natively (zeros + add reads any strided
    view through the existing kernels), so no NumPy round trip. Used to
    materialize borrowing views (reshape/transpose of an upstream
    gradient) into storage a retained grad can safely own."""
    zeros = cpp.NativeTensorCore.zeros(
        core.shape, dtype=core.dtype, device=core.device
    )
    result = zeros.add(core)
    zeros.close()
    return result


def _negated(tensor):
    """-tensor as a fresh owning NativeTensor, natively: a multiply
    against a broadcast native ``-1.0`` scalar. The runtime has no negate
    kernel; reusing the broadcasting multiply is the design's recommended
    composition (docs/native_autograd_design.md §7.5), and it never
    mutates its input — the same upstream object may also flow,
    un-negated, to the other parent."""
    core = tensor._require_open()
    neg_one = cpp.NativeTensorCore.full(
        (), -1.0, dtype=core.dtype, device=core.device
    )
    result = core.multiply(neg_one)
    neg_one.close()
    return NativeTensor._from_core(result)


def _unbroadcast(grad, target_shape):
    """Reduce ``grad`` (a NativeTensor at an op's broadcast output shape)
    to exactly ``target_shape`` — the adjoint of broadcasting: axes the
    forward expanded must have their gradient summed back down.

    The native reductions take one axis at a time, so the reductions run
    sequentially in a stable order that never shifts the remaining axis
    indices: (1) while the rank is too high, ``sum(axis=0)`` drops one
    leading rank-padded axis (always axis 0); (2) at equal rank, each
    stretched axis (target dim 1, grad dim > 1) is summed with
    ``keepdims=True`` (indices preserved). What can then remain is only a
    numel-preserving rank difference — the scalar-()-versus-(1,) family —
    fixed by a native reshape and materialized into owning storage.

    An exact-shape ``grad`` is returned unchanged (it is already the
    contribution); every other path returns a fresh owning contiguous
    NativeTensor of ``target_shape``. dtype/device are preserved and no
    NumPy touches the data."""
    target = tuple(target_shape)
    core = grad._require_open()
    if core.shape == target:
        return grad
    while core.ndim > len(target):
        core = core.sum(axis=0)
    if core.ndim == len(target):
        for axis, dim in enumerate(target):
            if dim == 1 and core.shape[axis] != 1:
                core = core.sum(axis=axis, keepdims=True)
    if core.shape != target:
        # Only rank padding of one-element shapes remains, e.g. () ->
        # (1,). reshape needs a contiguous source and returns a borrowing
        # view; _native_copy materializes it into independent storage.
        if not core.contiguous:
            core = _native_copy(core)
        core = _native_copy(core.reshape(target))
    return NativeTensor._from_core(core)


def _broadcast_back(upstream, x_shape, axis):
    """Broadcast a reduction's upstream gradient back to the reduced
    input's shape ``x_shape``, natively — sum/mean backward: every
    element that was folded into an output cell receives that cell's
    gradient. The upstream is first reshaped to the keepdims-compatible
    shape (reduced axes reinserted as size 1, so a ``keepdims=False``
    upstream — the scalar shape () included — lines up), then expanded by
    the existing broadcasting machinery via ``zeros(x_shape) + upstream``.
    Returns a fresh owning NativeTensor of exactly ``x_shape``;
    dtype/device preserved; no NumPy and no new kernel."""
    u_core = upstream._require_open()
    keep_shape = cpp.reduce_shape(x_shape, axis=axis, keepdims=True)
    transient = None
    if u_core.shape != keep_shape:
        if not u_core.contiguous:  # a user-supplied gradient view
            transient = u_core = _native_copy(u_core)
        u_core = u_core.reshape(keep_shape)  # borrowing view, used below
    zeros = cpp.NativeTensorCore.zeros(
        x_shape, dtype=u_core.dtype, device=u_core.device
    )
    result = zeros.add(u_core)
    zeros.close()
    if transient is not None:
        transient.close()
    return NativeTensor._from_core(result)
