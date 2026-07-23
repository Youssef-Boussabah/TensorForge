"""NativeTensor — a native tensor wrapper over NativeTensorCore, with an
opt-in Python-managed autograd graph.

This is the Stage-2 wrapper described in
docs/native_tensor_wrapper_design.md. Its forward surface is complete —
constructors, metadata, ``to_numpy``, forward compute
(``relu``/``add``/``subtract``/``multiply``/``matmul``/``sum``/``mean``,
plus the v3.11 optimizer math primitives ``sqrt``/``reciprocal``),
metadata-only views (``reshape``/``transpose``/``T``/``narrow``) and
``contiguous_copy`` — and an explicit ownership/lifetime story sits under
all of it. The v3.11 unary pair is differentiable through **saved
forward results**: their backwards read the recorded output, never the
parent's current value, so they record no expected parameter versions.

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
graph of its own and never touching NumPy. As of v2.3, ``narrow`` is
differentiable too: its backward **scatters** the upstream gradient into
a fresh zeros tensor of the parent's shape via the native
``narrow_backward`` kernel (the one new C++ kernel this milestone adds),
completing the view-backward set. As of v2.4, ``backward`` takes an
explicit ``retain_graph`` flag and the graph has a defined **lifetime**:
the default ``backward(retain_graph=False)`` is one-shot and releases the
traversed operation graph on success (a later backward through it raises
clearly), ``retain_graph=True`` keeps it for another pass, leaf gradients
accumulate across passes until ``zero_grad()``, and a failed pass rolls
back cleanly (no partial commit, no partial free). See
docs/native_autograd_design.md.

A note on mutation: there are no in-place user arithmetic operations,
so the forward values the graph captures for backward (e.g. the
operands of ``multiply``) remain stable for the life of the graph —
unless a tensor is explicitly closed, in which case ``backward()``
raises clearly rather than reading freed storage. The one sanctioned
mutation is the v3.7 NativeParameter value replacement
(``copy_value_`` / ``load_state_dict``), and it is version-guarded:
an operation whose backward reads a parameter's forward value
(``multiply``/``matmul``/``relu``) records the parameter's value
version at construction, and ``backward()`` validates every recorded
version *before* running any callback or touching any gradient — a
mutated-since-forward parameter raises a clear stale-graph
RuntimeError instead of silently computing gradients against the
wrong value. Value-independent backwards (``add``/``subtract``/
reductions/views) record nothing and stay valid across parameter
mutation, with mathematically correct gradients.

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
        # -- graph-lifetime state (v2.4): True once a one-shot backward has
        # released this non-leaf node's operation graph. Distinct from
        # _is_leaf so a freed non-leaf, a live non-leaf, and a genuine leaf
        # stay tellable apart.
        "_graph_freed",
        # -- stale-value guard (v3.7): on a non-leaf whose backward reads a
        # direct NativeParameter parent's forward value, a tuple of
        # (op_name, parameter, expected_version) entries recorded at
        # construction. backward() validates them before running anything;
        # graph cleanup releases them with _parents/_backward. Always ()
        # on leaves and value-independent results.
        "_expected_versions",
        # -- graph-owned native resources (Phase D, D9): a tuple of closeable
        # native objects this node's *history* owns — saved state its
        # backward needs that no parent keeps alive. Today the only user is
        # maxpool2d's private winner buffer. They are closed exactly once,
        # at the same deterministic point the graph is released (a one-shot
        # backward's cleanup, or close()), so retain_graph keeps them alive
        # for another pass and an abandoned graph still frees them. Always
        # () on leaves and on ops that save no native state.
        "_graph_resources",
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
        self._graph_freed = False
        self._expected_versions = ()
        self._graph_resources = ()

    @classmethod
    def _from_core(cls, core, owns_core=True):
        """Internal: wrap a core produced elsewhere (a compute or view
        op). Kept private so NativeTensorCore is not the normal way in.
        The result is a forward-only leaf (``requires_grad=False``, no
        graph) — what every op returns when no operand requires grad."""
        return cls(core, owns_core=owns_core)

    @classmethod
    def _from_op(cls, core, parents, backward, op, owns_core=True,
                 expected_versions=(), graph_resources=()):
        """Internal: build a **non-leaf** autograd graph node over
        ``core``.

        ``parents`` is a tuple of the NativeTensors this result was
        computed from; ``backward`` is a callable taking this node's
        upstream gradient (a NativeTensor) and accumulating a contribution
        into each parent (via ``parent._accumulate_grad(...)``); ``op`` is
        a short debug name. ``requires_grad`` is the OR of the parents' —
        if no parent needs grad, the result is a plain forward leaf with
        no recorded graph, matching the Python Tensor.
        ``expected_versions`` (v3.7) carries the stale-value guard: the
        ``(op_name, parameter, expected_version)`` entries for every
        direct NativeParameter parent whose forward value this op's
        backward reads — recorded only when a graph is actually built.
        ``graph_resources`` (D9) carries closeable **native objects this
        node's history owns** — saved forward state the backward needs that
        no parent keeps alive (today: maxpool2d's private winner buffer).
        They are adopted only when a graph is actually built and released
        exactly once when it is; when no parent requires grad **they are
        closed here immediately**, since nothing will ever consume them —
        so a no-grad forward can never leak saved state.

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
            node._expected_versions = tuple(expected_versions)
            node._graph_resources = tuple(graph_resources)
        else:
            # No graph, so no backward will ever read this saved state:
            # release it now rather than waiting for garbage collection.
            for resource in graph_resources:
                resource.close()
        return node

    def _release_graph_resources(self):
        """Close the native objects this node's graph history owns, exactly
        once (D9). Called at the deterministic graph-release points — a
        one-shot ``backward()``'s cleanup and ``close()`` — never on a
        retained graph, which still needs them. Clearing the tuple first
        makes a second call a no-op, so a freed graph can never
        double-close."""
        resources = self._graph_resources
        if not resources:
            return
        self._graph_resources = ()
        for resource in resources:
            resource.close()

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

        # relu's backward reads the input's forward value (the mask), so
        # a direct parameter input is stale-guarded (v3.7).
        return self._from_op(
            out_core, (self,), _backward, "relu",
            expected_versions=_versioned_value_reads("relu", (self,)),
        )

    def sqrt(self):
        """Elementwise square root, computed natively over this tensor's
        layout (v3.11 — an optimizer math primitive ahead of NativeAdam).
        Returns a new owning NativeTensor; the original stays open. IEEE
        float64 semantics: negative inputs give NaN, signed zeros are
        preserved, +inf gives +inf, NaN propagates.

        Differentiable: d(sqrt(x))/dx = 1 / (2*sqrt(x)), computed as
        ``0.5 * reciprocal(saved forward result)`` — the backward reads
        the **saved output**, never the parent's current value, so the
        graph records no expected parameter version (v3.7): mutating a
        direct parameter input after forward leaves this edge valid, and
        the gradient stays correct for the forward that was recorded. A
        closed saved output makes backward raise deterministically."""
        core = self._require_open()
        out_core = core.sqrt()
        if not self._requires_grad:
            return self._from_core(out_core)

        def _backward(upstream):
            # 1/(2*sqrt(x)) from the saved forward result: reciprocal
            # then a broadcast 0.5 scale — native cores only, and the
            # transient cores are closed as soon as they are consumed.
            result_core = result._require_open()
            inverse = result_core.reciprocal()
            half = cpp.NativeTensorCore.full(
                (), 0.5, dtype=result_core.dtype, device=result_core.device
            )
            local = inverse.multiply(half)
            half.close()
            inverse.close()
            contribution = upstream._require_open().multiply(local)
            local.close()
            self._accumulate_grad(NativeTensor._from_core(contribution))

        result = self._from_op(out_core, (self,), _backward, "sqrt")
        return result

    def reciprocal(self):
        """Elementwise 1/x, computed natively over this tensor's layout
        (v3.11 — an optimizer math primitive ahead of NativeAdam).
        Returns a new owning NativeTensor; the original stays open. IEEE
        float64 semantics: ±0.0 gives ±inf, ±inf gives ±0.0, NaN
        propagates — no exception and no warning.

        Differentiable: d(1/x)/dx = -1/x², computed as ``-(saved
        forward result)²`` — the backward reads the **saved output**,
        never the parent's current value, so the graph records no
        expected parameter version (v3.7): mutating a direct parameter
        input after forward leaves this edge valid, and the gradient
        stays correct for the forward that was recorded. A closed saved
        output makes backward raise deterministically."""
        core = self._require_open()
        out_core = core.reciprocal()
        if not self._requires_grad:
            return self._from_core(out_core)

        def _backward(upstream):
            # -(1/x)^2 from the saved forward result: square it, then a
            # broadcast -1.0 scale — native cores only.
            result_core = result._require_open()
            squared = result_core.multiply(result_core)
            neg_one = cpp.NativeTensorCore.full(
                (), -1.0, dtype=result_core.dtype, device=result_core.device
            )
            local = squared.multiply(neg_one)
            neg_one.close()
            squared.close()
            contribution = upstream._require_open().multiply(local)
            local.close()
            self._accumulate_grad(NativeTensor._from_core(contribution))

        result = self._from_op(out_core, (self,), _backward, "reciprocal")
        return result

    def exp(self):
        """Elementwise e**x, computed natively over this tensor's layout
        (Phase E, milestone E1 — the first stable-math primitive of the
        classification stack). Returns a new owning NativeTensor; the
        original stays open. IEEE float64 semantics: ``exp(0) == 1``,
        large positive arguments overflow to ``+inf``, large negative
        ones underflow toward ``+0``, ``-inf`` gives ``+0``, and NaN
        propagates — no clamping, no inserted bound.

        Differentiable: d(exp(x))/dx = exp(x), so the backward is simply
        ``upstream * saved forward output``. It reads the **saved
        output**, never the parent's current value, so the graph records
        no expected parameter version (v3.7): mutating a direct parameter
        input after forward leaves this edge valid, and the gradient
        stays correct for the forward that was recorded. A closed saved
        output makes backward raise deterministically, before any
        gradient is committed."""
        core = self._require_open()
        out_core = core.exp()
        if not self._requires_grad:
            return self._from_core(out_core)

        def _backward(upstream):
            # The derivative *is* the forward result, so no local
            # derivative has to be rebuilt: one native multiply of the
            # upstream by the saved output produces fresh owning storage.
            contribution = upstream._require_open().multiply(
                result._require_open()
            )
            self._accumulate_grad(NativeTensor._from_core(contribution))

        result = self._from_op(out_core, (self,), _backward, "exp")
        return result

    def log(self):
        """Elementwise natural logarithm, computed natively over this
        tensor's layout (Phase E, milestone E2 — the second stable-math
        primitive of the classification stack). Returns a new owning
        NativeTensor; the original stays open. IEEE float64 semantics with
        **no clamping and no inserted epsilon**: ``log(1) == 0``,
        ``log(±0) == -inf``, ``log(negative)`` is NaN, ``log(+inf) ==
        +inf``, and NaN propagates. Stability belongs in the fused losses
        (E4/E5), never in ``log`` itself.

        Differentiable: d(log(x))/dx = 1/x, computed as ``upstream *
        reciprocal(input)`` through the existing native ``reciprocal``
        primitive (no division operation exists or is added). Unlike
        ``exp``/``sqrt``/``reciprocal``, this backward **rereads the
        parent's live value** — the saved output ``log(x)`` cannot recover
        ``x`` cheaply or exactly — so a direct NativeParameter parent
        **is** version-guarded (v3.7): mutating it after forward makes
        ``backward()`` raise the deterministic stale-graph error before any
        gradient is committed anywhere in the graph, and the fix is a fresh
        forward pass."""
        core = self._require_open()
        out_core = core.log()
        if not self._requires_grad:
            return self._from_core(out_core)

        def _backward(upstream):
            # 1/x from the parent's *current* value (never from the saved
            # log output). The reciprocal is a transient owning core: it is
            # closed in a finally so a failing multiply cannot leak it, and
            # an exception from the cleanup never masks the original error.
            inverse = self._require_open().reciprocal()
            try:
                contribution = upstream._require_open().multiply(inverse)
            finally:
                inverse.close()
            self._accumulate_grad(NativeTensor._from_core(contribution))

        # The backward reads this parent's forward value, so a direct
        # parameter operand is stale-guarded — the same rule relu/multiply/
        # matmul already follow, through the same helper.
        return self._from_op(
            out_core, (self,), _backward, "log",
            expected_versions=_versioned_value_reads("log", (self,)),
        )

    def softmax(self, axis=-1):
        """Numerically stable softmax over one ``axis``, computed by the
        fused native kernel (Phase E, milestone E3 — the classification
        stack's first probability transform).

        ``axis`` is a plain int, negative allowed (NumPy-style); a bool,
        a float, a string, ``None``, or an out-of-range value raises, and
        rank 0 is rejected — softmax needs an axis to normalize over.
        Validation happens before any allocation. Returns a new owning
        row-major contiguous NativeTensor of the input's shape; the
        original stays open and unmutated. A non-contiguous input is
        handled by the Core layer's Policy-B copy-then-compute, so the
        result is contiguous whatever the input layout.

        Per slice the kernel fuses ``exp(x - max(x)) / sum(exp(x -
        max(x)))`` in float64 — the maximum shift means a large common
        offset cannot overflow. Exceptional values follow plain IEEE
        arithmetic: a NaN or ``+inf`` in a slice propagates, making that
        slice NaN.

        Differentiable, with the Jacobian-vector product written in
        closed form::

            dx = y * (upstream - sum(upstream * y, axis, keepdims=True))

        where ``y`` is the **saved forward output**. Like ``exp`` (and
        unlike ``log``), the backward never rereads the parent's value and
        never recomputes the softmax, so the graph records **no** expected
        parameter version: mutating a direct parameter input after forward
        leaves this edge valid and the gradient correct for the forward
        that ran. The whole backward is composed from existing
        graph-unaware Core operations — there is no dedicated softmax
        backward kernel."""
        core = self._require_open()
        # Normalize once, before anything is allocated, and hold the
        # normalized value in the closure so backward is independent of
        # the caller's variable and of any later rebinding. The Core
        # method validates identically (and again before *its* allocation).
        normalized_axis = cpp._normalize_axis(axis, core.shape)
        out_core = core.softmax(normalized_axis)
        if not self._requires_grad:
            return self._from_core(out_core)

        def _backward(upstream):
            # dx = y * (g - sum(g * y, axis, keepdims=True)), entirely at
            # the Core level. Each intermediate is fresh owning storage
            # closed in a finally as soon as its consumer has run, so a
            # failure at any stage leaks nothing and commits nothing.
            y_core = result._require_open()
            g_core = upstream._require_open()
            weighted = g_core.multiply(y_core)
            try:
                slice_dot = weighted.sum(axis=normalized_axis, keepdims=True)
            finally:
                weighted.close()
            try:
                # Broadcasting subtract: slice_dot keeps the reduced axis
                # at size 1, so it stretches back over the slice.
                centered = g_core.subtract(slice_dot)
            finally:
                slice_dot.close()
            try:
                contribution = y_core.multiply(centered)
            finally:
                centered.close()
            self._accumulate_grad(NativeTensor._from_core(contribution))

        result = self._from_op(out_core, (self,), _backward, "softmax")
        return result

    def log_softmax(self, axis=-1):
        """Numerically stable log-softmax over one ``axis``, computed by
        its own fused native kernel (Phase E, milestone E4).

        ``axis`` follows exactly ``softmax``'s rules: a plain int,
        negative allowed (NumPy-style); a bool, a float, a string,
        ``None``, or an out-of-range value raises, and rank 0 is rejected.
        Validation happens before any allocation. Returns a new owning
        row-major contiguous NativeTensor of the input's shape; the
        original stays open and unmutated. A non-contiguous input is
        handled by the Core layer's Policy-B copy-then-compute.

        Per slice the kernel fuses ``(x - max(x)) - log(sum(exp(x -
        max(x))))`` in float64. It is **never** ``softmax(x).log()``:
        that composition is exactly the precision loss this operation
        exists to avoid, since a probability below the float64 minimum
        rounds to 0 and its logarithm to ``-inf``, while the fused
        log-sum-exp form stays finite and accurate. Exceptional values
        follow plain IEEE arithmetic: a NaN or ``+inf`` in a slice makes
        that slice NaN; a ``-inf`` gets ``-inf`` and leaves its finite
        neighbours alone.

        Differentiable, with the Jacobian-vector product in closed form::

            dx = upstream - exp(y) * sum(upstream, axis, keepdims=True)

        where ``y`` is the **saved forward output** — ``exp(y)`` recovers
        the probabilities without ever rereading the input. Like ``exp``
        and ``softmax`` (and unlike ``log``), the graph therefore records
        **no** expected parameter version: mutating a direct parameter
        input after forward leaves this edge valid and the gradient
        correct for the forward that ran. The whole backward is composed
        from existing graph-unaware Core operations — there is no
        dedicated log-softmax backward kernel."""
        core = self._require_open()
        # Normalize once, before anything is allocated, and hold the
        # normalized value in the closure so backward is independent of
        # the caller's variable and of any later rebinding. The Core
        # method validates identically (and again before *its* allocation).
        normalized_axis = cpp._normalize_axis(axis, core.shape)
        out_core = core.log_softmax(normalized_axis)
        if not self._requires_grad:
            return self._from_core(out_core)

        def _backward(upstream):
            # dx = g - exp(y) * sum(g, axis, keepdims=True), entirely at
            # the Core level. Each intermediate is fresh owning storage
            # closed in a finally as soon as its consumer has run, so a
            # failure at any stage leaks nothing and commits nothing.
            y_core = result._require_open()
            g_core = upstream._require_open()
            probabilities = y_core.exp()
            try:
                slice_sum = g_core.sum(axis=normalized_axis, keepdims=True)
                try:
                    # Broadcasting multiply: slice_sum keeps the reduced
                    # axis at size 1, so it stretches back over the slice.
                    scaled = probabilities.multiply(slice_sum)
                finally:
                    slice_sum.close()
                try:
                    contribution = g_core.subtract(scaled)
                finally:
                    scaled.close()
            finally:
                probabilities.close()
            self._accumulate_grad(NativeTensor._from_core(contribution))

        result = self._from_op(out_core, (self,), _backward, "log_softmax")
        return result

    def cross_entropy(self, targets, reduction="mean"):
        """Fused multi-class cross-entropy over rank-2 logits, natively and
        differentiably (Phase E, milestone E6 — the autograd node over the
        E5 Core contract).

        ``self`` is the ``(batch_size, num_classes)`` logits block; the
        class axis is fixed at axis 1, so there is deliberately no ``axis``
        argument. ``targets`` is a one-dimensional sequence of integer
        class labels — a list/tuple of Python ints or a 1-D NumPy integer
        array, **never** a NativeTensor (the runtime has no integer dtype).
        ``reduction`` is exactly ``"mean"`` or ``"sum"``. Returns a
        **scalar** NativeTensor (shape ``()``), so ``loss.backward()``
        works with the engine's existing default seed.

        Everything numerical is the E5 Core contract, called once and
        unchanged: ``NativeTensorCore.cross_entropy_forward`` validates the
        rank, the dtype/device, the strict target rules (bools and
        floating-point labels rejected outright, nothing truncated), and
        the reduction *before any allocation*, copies the labels into an
        independently owned read-only ``int64`` array, applies Policy-B
        copy-then-compute to a strided view, and runs the single fused
        kernel that produces the scalar loss **and** the saved
        probabilities together. This method adds no second cross-entropy
        path and no arithmetic of its own.

        Backward consumes **only** the saved probabilities, that copied
        target array, the normalized reduction, and the native scalar
        upstream::

            grad[n, j] = upstream * (p[n, j] - [j == t_n]) / N

        (the ``/ N`` for ``"mean"`` only), through the E5 Core backward —
        whose signature does not even accept logits. So **backward never
        rereads the logits**, and the graph records **no expected
        parameter version**: mutating a direct NativeParameter logits
        parent after the forward pass leaves this edge valid and its
        gradient correct for the forward that ran (the ``maxpool2d``
        archetype, the deliberate contrast with ``log``).

        The saved probabilities are **private graph-owned state** (D9's
        ``graph_resources`` contract, reused unchanged): never a public
        tensor, never a parameter or buffer, never in a ``state_dict()``
        or a checkpoint. They are released exactly once, at the same
        deterministic points the graph history is — a one-shot
        ``backward()``'s cleanup or ``close()`` — so ``retain_graph=True``
        keeps them for another pass, a failed retryable backward leaves
        them alive, an abandoned graph still frees them, and a no-grad
        forward closes them immediately. The copied targets are ordinary
        immutable host metadata held by the backward closure and collected
        with it; mutating the caller's list or array after the forward
        cannot reach them.

        If graph construction itself raises, both the scalar loss and the
        saved probabilities are closed here before the exception
        propagates."""
        core = self._require_open()
        # One call into the E5 Core contract does all the validation and
        # all the math. Nothing is rechecked or recomputed at this layer.
        result = core.cross_entropy_forward(targets, reduction)
        # Unpack the record immediately. It is a plain __slots__ carrier
        # with no __del__, so once this frame holds the four fields the
        # record owns nothing that could later be double-closed — exactly
        # the position maxpool2d is in when its Core hands back
        # (out_core, winners).
        loss_core = result.loss
        probabilities = result.probabilities
        saved_targets = result.targets
        saved_reduction = result.reduction
        logits_t = self

        def _backward(upstream):
            # Saved probabilities + copied targets + reduction + the native
            # one-element upstream, whose storage handle and offset go
            # straight to the kernel (no NumPy extraction). The logits are
            # not an argument of the Core backward at all.
            grad_core = probabilities.cross_entropy_backward(
                saved_targets, upstream._require_open(), saved_reduction
            )
            contribution = NativeTensor._from_core(grad_core)
            try:
                logits_t._accumulate_grad(contribution)
            except BaseException:
                # _accumulate_grad adopts a contribution only on the
                # assignment that ends it — a closed parent, or a failing
                # native add for a second contribution, raises before that
                # — so an unadopted gradient is released here rather than
                # left to the __del__ safety net.
                contribution.close()
                raise

        try:
            # _from_op adopts the probabilities as this node's graph-owned
            # resource when a graph is built, and closes them immediately
            # when no parent requires grad. Same contract as maxpool2d's
            # winner buffer; no second lifetime system.
            return self._from_op(
                loss_core, (self,), _backward, "cross_entropy",
                graph_resources=(probabilities,),
            )
        except BaseException:
            # Nothing adopted either output: release both, in the same
            # saved-state-then-output order maxpool2d uses.
            probabilities.close()
            loss_core.close()
            raise

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

        # Each parent's gradient reads the *other* parent's forward
        # value, so both direct parameter operands are stale-guarded
        # (v3.7). A duplicate parent (a is b) simply records twice —
        # the check is idempotent.
        return self._from_op(
            out_core, (a, b), _backward, "multiply",
            expected_versions=_versioned_value_reads("multiply", (a, b)),
        )

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

        # da reads b's forward value and db reads a's, so both direct
        # parameter operands are stale-guarded (v3.7).
        return self._from_op(
            out_core, (a, b), _backward, "matmul",
            expected_versions=_versioned_value_reads("matmul", (a, b)),
        )

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

        Differentiable (v2.3): the backward **scatters** the upstream
        gradient into a fresh zeros tensor of this tensor's shape at the
        narrowed region (un-narrowed positions get zero gradient), via the
        native ``narrow_backward`` scatter kernel — no NumPy. The retained
        contribution is fresh owning contiguous storage; the parent's own
        layout is irrelevant because the gradient lives at the logical
        shape.
        """
        out_core = self._require_open().narrow(dim, start, length)  # validates
        if not self._requires_grad:
            return self._from_core(out_core, owns_core=False)
        # narrow's forward has already validated dim/start/length; capture
        # the normalized dim and start for the scatter (length is recovered
        # from the upstream's extent, and the original shape is read live so
        # a closed parent raises rather than reading freed layout).
        narrowed_dim = int(dim)
        narrowed_start = int(start)

        def _backward(upstream):
            original_shape = self._require_open().shape
            contribution = upstream._require_open().narrow_backward(
                narrowed_dim, narrowed_start, original_shape
            )
            self._accumulate_grad(NativeTensor._from_core(contribution))

        return self._from_op(out_core, (self,), _backward, "narrow",
                             owns_core=False)

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

    def conv2d(self, weight, bias=None, *, stride=1, padding=0):
        """2-D cross-correlation of this NCHW input against an OIHW
        ``weight`` (+ optional rank-1 ``bias``), natively and
        differentiably (Phase D, D6 — a **new fused primitive**, like
        ``matmul``, not a composition of existing ops).

        ``self`` is the ``(N, C, H, W)`` input; ``weight`` is an
        ``(O, C, kh, kw)`` NativeTensor; ``bias`` is ``None`` or a rank-1
        ``(O,)`` NativeTensor. ``stride``/``padding`` are an int or a
        length-2 ``(height, width)`` tuple (bools rejected). No dilation,
        groups, or channels-last; operands must be open CPU float64
        NativeTensors (stable ``Tensor`` and implicit conversion are
        rejected). Returns a fresh **owning** ``(N, O, out_h, out_w)``
        NativeTensor.

        The forward reuses ``NativeTensorCore.conv2d_forward`` (no forward
        kernel is duplicated in Python). Differentiable when any of
        input/weight/bias requires grad — otherwise a plain forward leaf.
        The Python-managed backward computes each gradient only for the
        parents that require it: the input gradient (rereading the weight
        value) and weight gradient (rereading the input value) run through
        the native ``conv2d_input_backward``/``conv2d_weight_backward`` Core
        ops, and the bias gradient reduces the upstream over batch and
        spatial axes via the existing native ``sum`` (no dedicated kernel).
        Conditional stale-value version tracking follows
        docs/native_cnn_design.md §8: a direct-parameter operand's version
        is recorded only when an *active* callback rereads its value."""
        core = self._require_open()
        if not isinstance(weight, NativeTensor):
            raise TypeError(
                f"conv2d requires a NativeTensor weight, got "
                f"{type(weight).__name__}"
            )
        weight_core = weight._require_open()
        has_bias = bias is not None
        if has_bias:
            if not isinstance(bias, NativeTensor):
                raise TypeError(
                    f"conv2d requires a NativeTensor bias or None, got "
                    f"{type(bias).__name__}"
                )
            bias_core = bias._require_open()
        else:
            bias_core = None
        # Forward via the Core wrapper — it validates ranks, channel
        # compatibility, dtype/device, output shape, and stride/padding
        # (bools rejected) before any allocation.
        out_core = core.conv2d_forward(
            weight_core, bias_core, stride=stride, padding=padding
        )
        parents = (self, weight) + ((bias,) if has_bias else ())
        if not any(p._requires_grad for p in parents):
            return self._from_core(out_core)

        input_t, weight_t, bias_t = self, weight, bias
        input_shape = core.shape
        weight_shape = weight_core.shape

        def _backward(upstream):
            u_core = upstream._require_open()
            # Input gradient — rereads the weight's forward value.
            if input_t._requires_grad:
                grad_in = u_core.conv2d_input_backward(
                    weight_t._require_open(), input_shape=input_shape,
                    stride=stride, padding=padding,
                )
                input_t._accumulate_grad(NativeTensor._from_core(grad_in))
            # Weight gradient — rereads the input's forward value.
            if weight_t._requires_grad:
                grad_w = u_core.conv2d_weight_backward(
                    input_t._require_open(), weight_shape=weight_shape,
                    stride=stride, padding=padding,
                )
                weight_t._accumulate_grad(NativeTensor._from_core(grad_w))
            # Bias gradient — reads only the upstream: sum over batch (axis
            # 0) then the two spatial axes, via existing native reductions
            # (no dedicated kernel). Both intermediates are closed; only the
            # final (O,) core survives to become the gradient.
            if has_bias and bias_t._requires_grad:
                reduced_batch = u_core.sum(axis=0)       # (O, out_h, out_w)
                try:
                    reduced_h = reduced_batch.sum(axis=1)  # (O, out_w)
                    try:
                        grad_b = reduced_h.sum(axis=1)     # (O,)
                    finally:
                        reduced_h.close()
                finally:
                    reduced_batch.close()
                bias_t._accumulate_grad(NativeTensor._from_core(grad_b))

        # Conditional version tracking (§8): record a value's version only
        # when an *active* callback rereads it. input-grad (runs iff input
        # needs grad) rereads the weight; weight-grad (runs iff weight needs
        # grad) rereads the input; bias-grad rereads neither.
        value_read_operands = []
        if input_t._requires_grad:
            value_read_operands.append(weight_t)
        if weight_t._requires_grad:
            value_read_operands.append(input_t)
        expected_versions = _versioned_value_reads("conv2d", value_read_operands)

        return self._from_op(
            out_core, parents, _backward, "conv2d",
            expected_versions=expected_versions,
        )

    def maxpool2d(self, *, kernel_size, stride=None, padding=0):
        """2-D max pooling over this NCHW input, natively and
        differentiably (Phase D, D9 — a fused primitive whose backward is
        driven entirely by the winners its own forward saved).

        ``self`` is the ``(N, C, H, W)`` input. ``kernel_size`` and
        ``stride`` are an int or a length-2 ``(height, width)`` tuple of
        ints ≥ 1 (bools rejected); ``stride=None`` means
        ``stride = kernel_size`` (non-overlapping windows, the stable
        convention). ``padding`` is an int or pair ≥ 0, applied
        symmetrically per axis. Returns a fresh **owning**
        ``(N, C, out_h, out_w)`` NativeTensor. Pooling has no parameters,
        so the only parent is the input.

        The forward reuses the D8 Core path, which also records — in
        **private** native storage the caller never sees — which input
        element won each window (docs/native_cnn_design.md §10/§12).
        Backward scatters the upstream gradient straight to those saved
        winners through the native ``maxpool2d_backward`` Core op:
        overlapping windows accumulate, a ``-1`` (padding-won) winner drops
        its gradient, and ties give the whole window's gradient to the
        first-occurrence winner recorded at forward time.

        Because backward reads **only** the saved winners and the upstream
        — never the input's current value, and never a recomputed maximum —
        this operation records **no expected parameter version**: mutating
        a directly versioned input after the forward pass cannot change the
        gradient routing and must not raise a stale-graph error. That is a
        deliberate contrast with ``conv2d``, whose gradients do reread
        operand values.

        The winner buffer is owned by this node's graph history and is
        released exactly when that history is, at one of the two
        deterministic points: a one-shot ``backward()``'s cleanup, or an
        explicit ``close()`` (a merely dropped graph reaches ``close()``
        through the ``__del__`` refcount/GC fallback).
        ``retain_graph=True`` keeps it for another pass; a no-grad forward
        (nothing requires grad) closes it immediately; and if graph
        construction itself raises, both the buffer and the pooled output
        are closed here before the exception propagates."""
        core = self._require_open()
        # Forward via the D8 Core path — it validates rank, dtype/device,
        # the kernel/stride/padding forms (bools and malformed pairs
        # rejected), the winner-exactness bound, and the output shape
        # before any allocation, and returns the pooled values plus the
        # private winner buffer.
        out_core, winners = core._maxpool2d_forward_with_winners(
            kernel_size=kernel_size, stride=stride, padding=padding
        )
        input_t = self
        input_shape = core.shape

        def _backward(upstream):
            # Saved winners + upstream only: no input value is reread and
            # no window maximum is recomputed.
            grad_in = upstream._require_open().maxpool2d_backward(
                winners, input_shape=input_shape
            )
            input_t._accumulate_grad(NativeTensor._from_core(grad_in))

        try:
            # _from_op adopts the winner buffer as graph-owned state when a
            # graph is built, and closes it immediately when one is not.
            return self._from_op(
                out_core, (self,), _backward, "maxpool2d",
                graph_resources=(winners,),
            )
        except BaseException:
            # Graph construction failed: release the saved state and the
            # output rather than leaking either.
            winners.close()
            out_core.close()
            raise

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

    def backward(self, gradient=None, retain_graph=False):
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

        **Graph lifetime (v2.4).** ``retain_graph`` must be a real ``bool``
        (validated *first*, before any traversal or gradient mutation; not
        coerced) — the default ``False`` makes ``backward()`` **one-shot**:
        after a successful pass the operation graph of every traversed
        non-leaf node is released (its ``_parents``/``_backward`` cleared so
        captured closures cannot keep parents alive, and the node marked
        freed), and a later ``backward()`` that reaches it raises a clear
        RuntimeError naming ``retain_graph=True`` as the remedy. Pass
        ``retain_graph=True`` to keep the graph for another pass; leaf
        gradients accumulate across passes until ``zero_grad()``. This is
        *not* full PyTorch parity — there is no per-node ``retain_grad`` and
        no double-backward. A genuine leaf has no graph to free (repeated
        ``backward()`` on a scalar leaf keeps accumulating) and is never
        marked freed. The tensor's stored value stays usable for forward
        computation after its graph is freed; only backward refuses to
        cross freed history.

        **Stale parameter values (v3.7).** Where an operation's backward
        must read a direct NativeParameter operand's forward value
        (``multiply``/``matmul``/``relu``), the graph records that
        parameter's value version at construction; this method validates
        every recorded version *before* any callback runs or any gradient
        changes. A parameter mutated after forward (``copy_value_`` or
        ``load_state_dict``) makes the old graph raise a distinct
        stale-value RuntimeError — deterministically, leaving gradients,
        graph structure, and versions untouched (the graph is *not*
        freed). ``retain_graph`` does not help; the remedy is a fresh
        forward pass. Value-independent backwards (add/subtract/
        reductions/views) record nothing and remain valid across
        parameter mutation.

        **Failure safety.** The whole pass is staged against a snapshot of
        every node's gradient (gradients are immutable — accumulation
        replaces the reference with a fresh native ``add``, never mutating
        in place), so if traversal or a callback raises, the references are
        restored: no leaf gradient is partially committed and no graph is
        partially freed. Cleanup runs only after the pass fully succeeds.

        Raises TypeError for a non-bool ``retain_graph``, and RuntimeError
        on a tensor that does not require grad, that has been closed, or
        whose graph has already been freed by a prior one-shot backward."""
        # Validate retain_graph before touching the graph or any gradient —
        # a bad value must not leave partial state, and it is never coerced.
        if not isinstance(retain_graph, bool):
            raise TypeError(
                f"retain_graph must be a bool, got {type(retain_graph).__name__}"
            )
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

        # Freed-graph detection: a one-shot backward releases the operation
        # graph, so any later traversal reaching a released non-leaf must
        # raise rather than silently treat it as a leaf and truncate
        # history (this catches both a repeated backward on the same output
        # and a new op built from a freed non-leaf value). Checked before
        # seeding, before any callback, and before any gradient is touched,
        # so a raise here changes nothing.
        for node in topo:
            if node._graph_freed:
                raise RuntimeError(
                    "backward() cannot traverse a freed autograd graph: this "
                    "graph was already released by a previous one-shot "
                    "backward() call. Pass retain_graph=True to that earlier "
                    "backward() if you need to run backward through the same "
                    "graph more than once."
                )

        # Stale-value detection (v3.7): every recorded expected version
        # must still match its parameter's current value version. Checked
        # for the whole traversal *before* the snapshot, the seed, any
        # callback, and any gradient commit — so a stale graph raises
        # deterministically with every gradient, every graph edge, and
        # every version untouched, and the failure repeats identically.
        # Versions are monotonic, so loading the old numerical value back
        # can never make a stale graph valid again: the fix is a new
        # forward pass. (This is deliberately *not* the freed-graph error,
        # and retain_graph does not help here.)
        for node in topo:
            for op_name, parameter, expected in node._expected_versions:
                current = parameter._version
                if current != expected:
                    raise RuntimeError(
                        f"backward() found a stale parameter value: a "
                        f"NativeParameter whose forward value {op_name!r} "
                        f"backward must read was modified after the forward "
                        f"pass (expected version {expected}, current version "
                        f"{current}). This graph cannot safely be reused — "
                        f"run the forward pass again after mutating or "
                        f"loading parameter values, then call backward() on "
                        f"the new output."
                    )

        # Stage the whole pass against a snapshot of every node's gradient,
        # so a mid-traversal failure rolls back cleanly (see the docstring's
        # failure-safety note) and never partially commits or partially
        # frees.
        snapshot = [(node, node._grad) for node in topo]
        try:
            # Seed this output. Going through _accumulate_grad means a
            # non-leaf root adopts the seed as its transient start, while a
            # leaf root accumulates it — so repeated backward on a scalar
            # leaf keeps summing. Reverse topological order then guarantees
            # each node's grad is complete before its own rule runs.
            self._accumulate_grad(seed)
            for node in reversed(topo):
                grad = node._grad
                if node._backward is not None and grad is not None:
                    node._backward(grad)
        except BaseException:
            for node, grad in snapshot:
                node._grad = grad
            raise

        # The pass succeeded. Drop the transient non-leaf gradients (only
        # leaves retain grad, both here and under retain_graph). With
        # retain_graph=False, also release the operation graph of every
        # traversed non-leaf node: clear its parents and backward closure
        # (so nothing keeps the parents alive) and mark it freed for
        # deterministic reuse errors. Never touched: tensor data, shape,
        # dtype/device, requires_grad, is_leaf, or any leaf gradient — a
        # freed non-leaf stays a non-leaf whose value is still usable.
        for node in topo:
            if node._is_leaf:
                continue
            node._grad = None
            if not retain_graph:
                node._parents = ()
                node._backward = None
                node._expected_versions = ()
                node._graph_freed = True
                # Release any native state this node's history owned (D9:
                # maxpool2d's private winner buffer) at the same
                # deterministic point the graph itself is released — not on
                # a retained graph, which may still run backward again, and
                # never twice.
                node._release_graph_resources()

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
        Any native state this node's graph history owns (D9: maxpool2d's
        private winner buffer) is released here too, so an explicit
        ``close()`` on an abandoned graph frees it deterministically. (An
        abandoned graph object that is merely *dropped* reaches this method
        through ``__del__``, which is the refcount/GC **fallback**, not a
        deterministic release point — the deterministic points are a
        one-shot ``backward()``'s history release and this method.)
        Idempotent — safe to call more than once."""
        if not self._closed:
            self._closed = True
            self._release_graph_resources()
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


def _versioned_value_reads(op_name, tensors):
    """The stale-value guard entries for one operation (v3.7):
    ``(op_name, tensor, current_version)`` for each tensor in
    ``tensors`` that carries a value version — i.e. each direct
    NativeParameter operand whose forward value the operation's
    backward will read. Detection is by the ``_version`` slot rather
    than an isinstance check so this module never imports the
    NativeParameter subclass (autograd stays independent of the module
    stack); NativeParameter is the only class that defines the slot.
    Plain tensors have no version and record nothing — they are
    immutable for the life of a graph (no in-place arithmetic exists),
    so only the controlled parameter mutation path needs guarding."""
    entries = []
    for tensor in tensors:
        version = getattr(tensor, "_version", None)
        if version is not None:
            entries.append((op_name, tensor, version))
    return tuple(entries)


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
