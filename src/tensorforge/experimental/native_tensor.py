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

As of Phase G milestone G3, ``dropout(p, *, generator)`` joins the
saved-state family: it takes an **explicit** ``NativeGenerator`` (there is
no default, global, or implicit native random stream), reserves exactly
one call, runs the stateless G2 Core with that reservation's seed and
index, and adopts the private multiplier mask as graph-owned state whose
backward is one ``multiply`` — so backward never rereads the input, never
redraws, and never touches the generator. The reservation is committed as
the last action before returning, and every failure before that abandons
it, so a failed forward consumes nothing. ``p == 0`` returns the input
object itself and consumes nothing either.

As of Phase K milestone K2, ``NativeTensor`` also carries the native line's
**index/result dtype**. ``from_int64_array(values, *, requires_grad=False)``
is the one public door through which an ``int64`` buffer can come into
existence anywhere in the repository: it takes an exact ``numpy.int64``
array and converts nothing — no dtype inference, no cast, no truncation, no
widening, no byte swap, and no "it was integral anyway" allowance — so every
value in ``[-(2**63), 2**63 - 1]`` survives, including the ones a float64
detour would round. The resulting tensor uses the **same** ownership, view,
copy, lifetime, and host-transfer machinery as a floating one:
``reshape``/``transpose``/``T``/``narrow`` are still metadata-only borrowing
views that cannot cast, ``contiguous_copy`` still allocates a fresh owning
output, and ``to_numpy``/``item``/``tolist`` still return fresh independent
host values. What it can never do is any of the roles Phase K fenced off at
K1: it cannot require gradients, build a graph, accumulate one, become a
``NativeParameter``, be registered as a buffer, be owned by an optimizer, be
declared in a checkpoint archive, or enter any floating operation. There is
no integer arithmetic, no integer reduction, no ``argmax``, no index
selection, and no casting in either direction.

NativeTensor is still **not** tensorforge.Tensor: the two autograd
engines never mix, no conversion is implicit, and the framework frontend
never imports this. Conversion crosses the native boundary only by
explicit call: ``from_array`` and ``from_int64_array`` enter, ``to_numpy``
/ ``item`` / ``tolist`` exit, all as copies.
Lifetime is explicit — ``close()`` (or a ``with`` block) releases the
native storage an owning tensor holds; a closed tensor rejects metadata,
materialization, and gradient operations clearly rather than reading
freed layout.
"""

import numpy as np

from ..backends import cpp
from .native_generator import NativeGenerator


class NativeTensor:
    """A native tensor: one NativeTensorCore, an explicit ownership and
    lifetime story, and an opt-in Python-managed autograd graph.

    Create one with ``from_array`` / ``zeros`` / ``full`` (each owns the
    core it wraps; pass ``requires_grad=True`` to track gradients), or —
    since Phase K milestone K2 — with ``from_int64_array``, the one public
    door to an exact, non-differentiable native ``int64`` tensor. Read
    ``shape`` / ``strides`` / ``ndim`` / ``numel`` / ``contiguous`` /
    ``dtype`` / ``device`` for metadata; ``to_numpy()`` / ``item()`` /
    ``tolist()`` to materialize a fresh independent host copy. For autograd
    read ``requires_grad`` / ``grad`` / ``is_leaf`` and call ``backward()``
    / ``zero_grad()`` / ``detach()``. Release native memory with
    ``close()`` or a ``with`` block; ``owns_core`` and ``closed`` report
    lifetime state.

    One class carries both roles deliberately (integer design §6): an
    ``int64`` tensor shares every line of the ownership, view, copy,
    lifetime, and host-transfer machinery, and is kept out of the roles it
    has no meaning in by dtype checks rather than by a second class. It can
    never require gradients, become a ``NativeParameter``, be registered as
    a buffer, be owned by an optimizer, be declared in a checkpoint, or
    enter a floating operation.

    Not tensorforge.Tensor — see the module docstring,
    docs/native_tensor_wrapper_design.md,
    docs/native_autograd_design.md, and
    docs/native_integer_tensors_design.md.
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
        point (see also the internal ``_from_core``).

        **Phase K: the core must carry a dtype a native tensor may have** —
        the wrapper-construction barrier of the integer design's §6.5
        table, one layer above ``NativeTensorCore.__init__``'s. The two are
        listed as separate barriers rather than one because they have
        different first authorities and different operands, and a single
        layer is a single point of failure.

        K1 set this gate to *floating only*, which is what kept a raw
        ``int64`` handle from becoming a tensor while the dtype existed
        solely as a C ABI representation. **K2 widened it to "floating or
        index"** and to nothing else, so a core produced by the private
        integer ingress can be wrapped — and an ``int64`` tensor is
        *still* refused by autograd, by ``NativeParameter``, by
        ``register_buffer`` at both persistence values, by both optimizers,
        by checkpoint entry validation, and by every floating operation
        entry, because every one of those barriers asks the **floating**
        predicate and none of them moved."""
        if not isinstance(core, cpp.NativeTensorCore):
            raise TypeError(
                f"NativeTensor wraps a NativeTensorCore, got "
                f"{type(core).__name__}"
            )
        cpp._require_tensor_dtype(core.dtype, "NativeTensor", role="core")
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
        core and the C++ kernels remain autograd-unaware.

        **Phase K, milestone K1: a non-differentiable result cannot become
        a graph node.** This is the structural backstop of the integer
        design's §6.5 table and the reason the unified object model is safe
        against operations written *after* this milestone: no future
        operation can accidentally produce a differentiable integer result,
        because the single graph-construction entry refuses to build one.
        It is a raise rather than an assertion, and it **closes the core it
        was handed** before raising — the core is this call's to publish or
        to release, so a rejected graph leaks nothing and live storage
        returns exactly to baseline. Differentiability is asked as its own
        question (§5.3), through the one floating predicate, rather than
        inferred from the parents."""
        if not cpp._is_floating_dtype(core.dtype):
            # This call was handed the core and the saved state; nothing
            # else can free either, so a rejection releases both before it
            # propagates — exactly what the no-graph branch below does with
            # ``graph_resources`` when no parent requires grad.
            for resource in graph_resources:
                resource.close()
            core.close()
            raise ValueError(
                f"a native autograd graph node must have a differentiable "
                f"dtype {cpp.SUPPORTED_DTYPES}, got {core.dtype!r} for "
                f"operation {op!r} (gradients of integers are ill-defined)"
            )
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

    @classmethod
    def from_int64_array(cls, values, *, requires_grad=False):
        """A contiguous native ``int64`` tensor holding an **exact** copy of
        an exact ``int64`` NumPy array (Phase K, milestone K2; see
        docs/native_integer_tensors_design.md §8).

        **This is the one public API in the repository through which an
        ``int64`` buffer can come into existence.** There is no ``dtype``
        argument, deliberately: the dtype is in the name, so it cannot be
        omitted, mistyped, or contradicted and no second authority exists.
        ``from_array``, ``zeros``, and ``full`` all validate through
        ``normalize_dtype`` and reject ``"int64"`` permanently, as does the
        public ``NativeStorage`` constructor, so no generic constructor
        changed what it accepts.

        **Nothing converts.** ``values`` must be exactly a
        ``numpy.ndarray`` — not a subclass, not a list, not a tuple, not a
        scalar — of exactly ``numpy.int64`` in native byte order. A
        ``float64`` array holding ``[1.0, 2.0]`` is rejected rather than
        accepted as "integral anyway"; so are ``int32``, ``uint64``,
        ``bool``, ``object``, and a byte-swapped ``>i8`` array. The
        asymmetry against the *floating* ingress, which has always
        converted, is the point (§8.3): a floating conversion is a rounding
        whose error is bounded and familiar, while an integer one is either
        a silent truncation or a silent reinterpretation. A **non-contiguous**
        exact-``int64`` array *is* accepted and is copied into fresh
        contiguous storage — rearranging where identical values live is
        layout normalization, not a cast (§8.4).

        Every value in ``[-(2**63), 2**63 - 1]`` survives exactly, including
        the ones beyond float64's exact integer range, which is what makes
        this a genuine integer boundary rather than a float64 detour.

        The shape is preserved exactly, at any rank including 0. A
        zero-element array is rejected, because the runtime cannot represent
        zero-element storage — an inherited limitation reported honestly and
        not worked around (§13.7).

        ``requires_grad`` is keyword-only and may only be ``False``. It
        exists rather than being omitted so that ``requires_grad=True``
        explains that ``int64`` tensors are non-differentiable instead of
        producing Python's generic unexpected-keyword ``TypeError``; a
        non-``bool`` raises ``TypeError`` and ``True`` raises ``ValueError``,
        both **before any native allocation**.

        The dtype this constructor names is still measured against the
        **public index/result registry** before anything else happens to
        the input: ``cpp._normalize_index_dtype`` is the canonical registry
        gate for this fixed-format door (§5.2), and it is asked at
        §26.1 step 2a — after both ``requires_grad`` checks and before the
        array is inspected, before the private Core ingress is entered, and
        before any allocation. A door whose dtype is in its *name* still
        answers to the registry rather than to itself.

        The result is an owning gradient-free leaf: it owns fresh storage,
        never aliases the caller's array (mutating it afterwards reaches
        nothing), and **the caller closes it**. Any failure closes
        everything it allocated, including under ``BaseException``, so live
        storage returns exactly to baseline."""
        # Argument validation first, before the array is even examined and
        # long before anything is allocated (§26.1 steps 1-2, §26.2).
        if not isinstance(requires_grad, bool):
            raise TypeError(
                f"requires_grad must be a bool, got "
                f"{type(requires_grad).__name__}"
            )
        if requires_grad:
            raise ValueError(
                "from_int64_array cannot create a gradient-tracking tensor: "
                "int64 tensors are non-differentiable, so requires_grad must "
                "be False (gradients of integers are ill-defined)"
            )
        # The index/result dtype authority, asked here and only here
        # (§26.1 step 2a). This constructor carries its dtype in its
        # *name* rather than in an argument, which is what keeps a second
        # authority from existing — but "the name says int64" is not the
        # same statement as "int64 is a dtype the runtime may build a
        # tensor at", and the second is the registry's to make. So the
        # canonical gate is asked before the array is examined and long
        # before anything is allocated: if ``cpp.INDEX_DTYPES`` ever
        # stopped listing ``"int64"``, this door would close here, at the
        # same pre-allocation step every other rejection uses, rather than
        # allocating storage the widened wrapper gate would then refuse.
        # There is deliberately no ``dtype=`` argument to pass the result
        # to: it is the gate that matters, and the private ingress below
        # names the one width it exists for.
        cpp._normalize_index_dtype("int64")
        core = cpp.NativeTensorCore._from_int64_array(values)
        try:
            # A published wrapper is already the gradient-free leaf this
            # constructor promises: ``__init__`` sets ``_requires_grad =
            # False`` on every new wrapper, and ``requires_grad=True`` was
            # rejected above, so a second flag-setting step after the
            # protected region would restate what construction guarantees
            # and would run *outside* the cleanup that owns this core.
            return cls._from_core(core)
        except BaseException:
            # The core is this call's to publish or to release; nothing else
            # holds it yet, so a failed wrapper construction frees it here
            # rather than leaving it to the ``__del__`` safety net.
            core.close()
            raise

    @classmethod
    def _typed_from_array(cls, values, dtype, device="cpu"):
        """Private: a contiguous tensor at an **internally representable**
        dtype holding a copy of ``values`` (Phase I, milestone I8).

        The tensor-level counterpart of
        ``cpp.NativeTensorCore._typed_from_array`` (I2), private for
        ``_typed_zeros``'s reason and completing the same small family: the
        typed ingress boundary, for state that arrives as host data already
        carrying its own width.

        Its one caller is the native checkpoint loader. A version-3 archive
        declares each entry's dtype explicitly and the loader has already
        proved the stored array matches that declaration exactly, in native
        byte order, so the tensor it stages must be built at the declared
        width — building it at float64 beside a float32 destination would
        be the mixed-dtype request the runtime refuses (design §9), and
        there is no cast that could reconcile them.

        Ingress is **not** a tensor cast (design §9.4): the host array's
        dtype is validated against ``dtype`` before anything is allocated,
        so this copies matching bits rather than converting between widths.
        It widens public construction by exactly nothing:
        ``NativeTensor.from_array(..., dtype="float32")`` raises unchanged.
        The result is always a gradient-free leaf — checkpointed state is
        installed through ``copy_value_`` and the loaders, never adopted as
        a graph node."""
        tensor = cls._from_core(
            cpp.NativeTensorCore._typed_from_array(values, dtype,
                                                   device=device)
        )
        tensor._init_requires_grad(False)
        return tensor

    @classmethod
    def _typed_zeros(cls, shape, dtype, device="cpu"):
        """Private: a row-major contiguous all-zero tensor of ``shape`` at
        an **internally representable** dtype (Phase I, milestone I7).

        The tensor-level counterpart of the Core's dtype-trusting zeroed
        constructor — the same shape validation, the same storage
        ownership, the same zeroed allocation, the same ``close()``
        semantics — differing from the
        public ``zeros`` in exactly one respect: the dtype is validated
        against the internal table rather than the public registry.

        It exists because a **module's** persistent numeric state has to be
        built at the module's own dtype: BatchNorm's ``running_mean`` is
        this call. Building it at float64 and then meeting a float32 input
        would be a mixed-dtype request, which the runtime refuses (design
        §9); building it through the public ``zeros`` is impossible while
        float32 is unsupported.

        Private on purpose, and it widens public construction by exactly
        nothing: ``NativeTensor.zeros(..., dtype="float32")`` raises
        unchanged. The result is always a gradient-free leaf — a running
        statistic is never trainable."""
        tensor = cls._from_core(
            cpp.NativeTensorCore.zeros(
                shape, dtype=dtype, device=device, _trusted_dtype=True
            )
        )
        tensor._init_requires_grad(False)
        return tensor

    @classmethod
    def _typed_full(cls, shape, fill_value, dtype, device="cpu"):
        """Private: a row-major contiguous tensor of ``shape`` filled with
        ``fill_value`` at an **internally representable** dtype (Phase I,
        milestone I7).

        The tensor-level counterpart of
        ``cpp.NativeTensorCore._typed_full`` (I4), private for
        ``_typed_zeros``'s reason and used for the same two things: a
        module's persistent numeric state at the module's dtype
        (BatchNorm's ``running_var``), and the **graph-dtype scalar
        constants** a composed normalization forward materializes — ``eps``,
        ``momentum``, and ``1 - momentum``. Those are the constants design
        §11.4 forbids introducing at literal float64 into a float32 graph.

        The scalar itself crosses the ABI as a ``double`` and is narrowed
        **once**, before the fill loop, inside ``tf_storage_fill`` (design
        §7.4). Converting a scalar argument is not casting a tensor.

        It widens public construction by exactly nothing:
        ``NativeTensor.full(..., dtype="float32")`` raises unchanged. The
        result is always a gradient-free leaf."""
        tensor = cls._from_core(
            cpp.NativeTensorCore._typed_full(
                shape, fill_value, dtype, device=device
            )
        )
        tensor._init_requires_grad(False)
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
        """The element type tag, delegated to the core: ``"float64"``,
        ``"float32"``, or — for a tensor built by ``from_int64_array`` and
        for the views and copies derived from it — ``"int64"``. A plain
        canonical string, never a dtype object, and there is deliberately no
        public ``is_integer`` / ``is_floating`` property beside it: this is
        the one authority. Rejected after close, like the other layout
        metadata."""
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
        """Materialize into a fresh, independent NumPy array of **exactly
        this tensor's dtype**, in its logical shape and row-major logical
        order.

        The explicit *exit* boundary: the returned array owns its host
        memory and shares no mutable state with native storage, so mutating
        it cannot reach the tensor and closing the tensor cannot invalidate
        it. Nothing is widened or reinterpreted on the way out — a float32
        tensor gives a float32 array and an ``int64`` tensor gives an exact
        ``numpy.int64`` array, including values beyond float64's exact
        integer range. A non-contiguous view materializes in logical C
        order. Raises RuntimeError if the tensor has been closed.
        """
        return self._require_open().to_numpy()

    def item(self):
        """The single element of a one-element tensor, as a **built-in
        Python scalar** (Phase K, milestone K2; integer design §8.8).

        ``int`` for an ``int64`` tensor — the complete signed 64-bit value,
        exactly, with no float intermediate and no NumPy scalar — and
        ``float`` for a ``float64`` or ``float32`` tensor, where the widening
        to a Python float is exact. Introduced dtype-general rather than
        integer-only because it has one meaning at every dtype, and two
        half-implementations would be worse than one.

        Requires ``numel == 1`` **at any rank**: a ``()``, a ``(1,)``, and a
        ``(1, 1, 1)`` tensor all qualify. Any other element count raises
        ``ValueError`` naming the actual count. A closed tensor rejects
        before any transfer.

        Built on ``to_numpy()`` — no new export, no second materialization
        path — so it pays one full materialization of a one-element tensor,
        which is exactly the cost of the transfer it needs. It builds no
        graph, touches no gradient, parameter, or version counter, allocates
        no native output, and retains nothing."""
        core = self._require_open()
        if core.numel != 1:
            raise ValueError(
                f"item() requires a tensor with exactly one element, got "
                f"{core.numel} (shape {core.shape})"
            )
        # ``ndarray.item`` is what turns a NumPy scalar into a built-in one:
        # an int64 element becomes a Python ``int`` and a float element a
        # Python ``float``, with no intermediate of the other kind.
        return core.to_numpy().item()

    def tolist(self):
        """This tensor's values as nested built-in Python containers, in its
        logical shape (Phase K, milestone K2; integer design §8.8).

        Exact Python ``int``s for an ``int64`` tensor and Python ``float``s
        for the floating dtypes — never a NumPy scalar, and never through a
        float intermediate. A rank-0 tensor returns the scalar itself,
        matching ``numpy.ndarray.tolist``. A non-contiguous view follows
        logical C order, because the materialization it is built on does.

        Dtype-general for ``item()``'s reason, built on ``to_numpy()`` for
        ``item()``'s reason, and it rejects a closed tensor before any
        transfer. It builds no graph, touches no gradient, parameter, or
        version counter, and retains nothing."""
        return self._require_open().to_numpy().tolist()

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
            # The constant is built at the **graph's** dtype (design §11.4),
            # through the private typed constructor — see ``_negated``.
            half = cpp.NativeTensorCore._typed_full(
                (), 0.5, result_core.dtype, device=result_core.device
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
            # The constant is built at the **graph's** dtype (design §11.4),
            # through the private typed constructor — see ``_negated``.
            neg_one = cpp.NativeTensorCore._typed_full(
                (), -1.0, result_core.dtype, device=result_core.device
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
        max(x)))`` **at the graph's dtype** (Phase I, milestone I6) — the
        maximum shift means a large common offset cannot overflow at either
        width. Exceptional values follow plain IEEE arithmetic: a NaN or
        ``+inf`` in a slice propagates, making that slice NaN.

        Differentiable, with the Jacobian-vector product written in
        closed form::

            dx = y * (upstream - sum(upstream * y, axis, keepdims=True))

        where ``y`` is the **saved forward output**. Every intermediate of
        that expression — the elementwise products, the axis reduction, and
        the broadcasting subtract — is an existing dtype-general Core
        operation, so the whole backward runs at the graph's dtype with no
        literal-float64 tensor anywhere in it. Like ``exp`` (and
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
        max(x))))`` **at the graph's dtype** (Phase I, milestone I6). It is
        **never** ``softmax(x).log()``: that composition is exactly the
        precision loss this operation exists to avoid, since a probability
        below the element type's smallest normal rounds to 0 and its
        logarithm to ``-inf``, while the fused log-sum-exp form stays
        finite and accurate — a margin that matters far sooner at float32,
        whose smallest normal is ~1.18e-38. Exceptional values
        follow plain IEEE arithmetic: a NaN or ``+inf`` in a slice makes
        that slice NaN; a ``-inf`` gets ``-inf`` and leaves its finite
        neighbours alone.

        Differentiable, with the Jacobian-vector product in closed form::

            dx = upstream - exp(y) * sum(upstream, axis, keepdims=True)

        where ``y`` is the **saved forward output** — ``exp(y)`` recovers
        the probabilities without ever rereading the input. As in
        ``softmax``, every intermediate is an existing dtype-general Core
        operation, so the backward runs at the graph's dtype throughout.
        Like ``exp``
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
        array, **never** a NativeTensor. Classification targets remain
        exact host-side label metadata under the Phase-E contract, and
        Phase K milestone K2 — which gave the runtime an ``int64``
        index/result dtype — did **not** widen cross-entropy to accept
        ``NativeTensor`` targets.
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

        Every numeric operand of that node carries **one** dtype (Phase I,
        milestone I6): the scalar loss, the saved probabilities, the
        upstream, and the logits gradient are all the logits' dtype, and
        the C ABI revalidates the agreement before it writes anything. The
        copied targets are the deliberate exception and are not a dtype at
        all — they stay host ``int64`` metadata at every width, are never
        inferred from the logits, and never become a tensor.

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
            #
            # ``1.0 / count`` is computed once in binary64 and narrowed once
            # into the constant, at the **graph's** dtype (design §7.4,
            # §11.4) — the same rule the forward ``mean`` follows through
            # ``tf_storage_scale``, so forward and backward scale by exactly
            # the same representable factor at either width.
            u_core = upstream._require_open()
            scale = cpp.NativeTensorCore._typed_full(
                (), 1.0 / count, u_core.dtype, device=u_core.device
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
        making the parent's storage layout irrelevant.

        The destination core is allocated before either publication branch
        is chosen, so it is this call's to publish or to release: every
        exception after it exists — including a ``BaseException``, and
        including a failure inside ``_from_op`` — closes it explicitly
        rather than leaving it to the ``__del__`` safety net, so live
        storage returns exactly to baseline. ``_from_op``'s own
        non-differentiable rejection already closes the core it was handed;
        core ``close()`` is idempotent, so the two cannot conflict."""
        out_core = self._require_open().contiguous_copy()
        try:
            if not self._requires_grad:
                return self._from_core(out_core)

            def _backward(upstream):
                self._accumulate_grad(upstream)

            return self._from_op(out_core, (self,), _backward,
                                 "contiguous_copy")
        except BaseException:
            out_core.close()
            raise

    def conv2d(self, weight, bias=None, *, stride=1, padding=0):
        """2-D cross-correlation of this NCHW input against an OIHW
        ``weight`` (+ optional rank-1 ``bias``), natively and
        differentiably (Phase D, D6 — a **new fused primitive**, like
        ``matmul``, not a composition of existing ops).

        ``self`` is the ``(N, C, H, W)`` input; ``weight`` is an
        ``(O, C, kh, kw)`` NativeTensor; ``bias`` is ``None`` or a rank-1
        ``(O,)`` NativeTensor. ``stride``/``padding`` are an int or a
        length-2 ``(height, width)`` tuple (bools rejected). No dilation,
        groups, or channels-last; operands must be open CPU NativeTensors
        of one dtype (stable ``Tensor`` and implicit conversion are
        rejected; mixed dtype raises before anything is allocated).
        Returns a fresh **owning** ``(N, O, out_h, out_w)`` NativeTensor
        at the operands' dtype.

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
        # Forward via the D8 Core path — it validates rank, the
        # kernel/stride/padding forms (bools and malformed pairs
        # rejected), the winner-exactness bound, and the output shape
        # before any allocation, and returns the pooled values (at the
        # input's dtype, I5) plus the private winner buffer (always
        # float64, design §13.3).
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

    def dropout(self, p, *, generator):
        """Inverted Dropout over this tensor, natively and differentiably
        (Phase G, milestone G3 — one autograd node over the G2 stateless
        Core, plus the generator call transaction of design §5).

        ``p`` is the drop probability in ``[0, 1)``, validated by the one
        shared normalizer the Core and (later) the module use, so the
        accepted/rejected matrix is identical at every layer: a ``bool`` is
        a ``TypeError``, a non-real is a ``TypeError``, and ``p == 1``,
        ``p > 1``, ``p < 0``, NaN, and ±inf are all ``ValueError``.
        ``generator`` is a **required keyword-only** ``NativeGenerator``:
        there is no default, no process-global stream, no module-global
        stream, no implicit per-call generator, and no NumPy or Python
        ``random`` fallback anywhere. Omitting it is a ``TypeError``.

        For ``0 < p < 1`` this returns a fresh **owning** contiguous tensor
        of this tensor's shape, where each logical element is kept and
        scaled by ``1 / (1 - p)`` or dropped to ``0.0``. The decision for
        logical (row-major) element ``i`` is a deterministic function of
        ``(seed, call_index, i, p)`` — never of the values, the address,
        the physical strides, or how the view was built, so a transposed,
        narrowed, or nonzero-offset input receives the same mask as a
        contiguous tensor of the same logical shape (Policy B, design §7.3).

        **``p == 0`` is identity** (design §6.2): this returns ``self`` —
        the caller's own object, un-copied — after validating the receiver,
        the generator, and the probability. It allocates nothing, calls no
        kernel, builds no graph node, and **consumes no generator call**.

        **The call transaction** (design §5). Everything that can be
        validated without the generator is validated first; then exactly
        one call is reserved, the Core runs **outside** the generator's
        lock with the reserved key, the graph node is built, and the
        reservation is committed as the **final** state-changing action
        before returning. So a successful stochastic forward consumes
        exactly one call — whether or not anything requires grad — and
        **every** ordinary failure before that commit abandons the
        reservation, leaving ``calls`` untouched and the same index free
        for the next forward. Backward consumes none, ever.

        The exact random key comes from the reservation, not from a later
        read: the index is the token's, and the seed is read while that
        reservation is live, which ``reseed``/``reset``/``load_state`` and
        every generator state transaction are refused during (design §3.6).
        So the key cannot describe a stream that no longer exists.

        **Graph behavior** (design §8.2). The result requires grad exactly
        when this tensor does; the only parent is this tensor; and the
        backward is ``grad_input = upstream * mask`` over the **private
        multiplier mask this forward saved**, through the existing native
        ``multiply`` — there is no Dropout backward kernel and no
        ``dropout_backward`` Core op (design §7.5). Backward therefore
        never rereads the input, never redraws, and never touches the
        generator, so this operation records **no expected parameter
        version**: mutating a directly versioned input afterwards leaves
        the gradient correct for the forward that ran and must not raise a
        stale-graph error (the ``maxpool2d``/``cross_entropy`` archetype,
        the deliberate contrast with ``log``). Reseeding, resetting, or
        replacing the generator's state afterwards cannot change an
        existing graph's gradient either. Higher-order autograd is not
        supported here, exactly as everywhere else in the native line: the
        backward computes at the graph-unaware Core level and produces
        graph-free gradients.

        The mask is **private graph-owned state** (D9's ``graph_resources``
        contract, reused unchanged — the third member of the family that
        already holds MaxPool2d's winners and cross-entropy's saved
        probabilities): never a public tensor, never a parameter or buffer,
        never in a ``state_dict()`` or a checkpoint. It is released exactly
        once, at the same deterministic points the graph history is — a
        one-shot ``backward()``'s cleanup or ``close()`` — so
        ``retain_graph=True`` keeps it for another pass, a failed retryable
        backward leaves it alive, an abandoned graph still frees it, and a
        **no-grad forward closes it immediately** (while still committing
        the call, because a draw happened). If graph construction itself
        raises, both the output and the mask are closed here.

        Zero-element inputs are contractually legal and would consume a
        call, but the native tensor representation rejects zero-size
        dimensions, so no empty tensor can be constructed to hand in
        today — the G2 reachability note, unchanged."""
        # -- validation first: nothing below reserves, allocates, or calls
        # a kernel, so a rejected call is inert on every axis (design §14).
        core = self._require_open()
        if not isinstance(generator, NativeGenerator):
            raise TypeError(
                f"dropout requires an explicit NativeGenerator, got "
                f"{type(generator).__name__}. The native line has no "
                f"default, global, or implicit random stream: construct a "
                f"NativeGenerator and pass generator=..."
            )
        # The one shared validator (design §6.1) — not a second rule.
        probability = cpp._normalize_dropout_probability(p, "dropout")
        if probability == 0.0:
            # Identity (design §6.2): the caller's own tensor, un-copied.
            # No reservation, no allocation, no kernel, no graph node, no
            # change to requires_grad, and no call consumed. This is the
            # one case where a Dropout result is not a fresh owning
            # tensor, matching the stable Dropout and the empty
            # NativeSequential forward; §13 records the aliasing.
            return self

        # -- reserve exactly one call. This is where a concurrent or
        # reentrant caller fails, and where exhaustion is refused — in both
        # cases before an index is minted. The generator's lock is released
        # before anything below runs, so it is never held across an
        # allocation or a kernel.
        token = generator._reserve_call()
        # Bound immediately, and the cleanup boundary entered as the very
        # next action: the only window this cannot cover is an asynchronous
        # exception between the two, which is a couple of bytecodes with no
        # Python code in it (design §3.6's residual window).
        result = None
        try:
            # The key belongs to the reservation: the token's index, and
            # the seed read while that reservation makes every state
            # replacement raise. `generator.calls` is deliberately NOT read
            # here — it is the committed *count*, and the reserved index is
            # the token's.
            seed = generator.seed
            call_index = token._index
            # One call into the G2 Core contract does all the numerical
            # work and its own validation, allocates output then mask, and
            # closes everything it allocated if any part fails.
            out_core, mask = core._dropout_forward_with_mask(
                probability, seed=seed, call_index=call_index
            )
            try:
                backward = _dropout_backward(self, mask)
                # _from_op adopts the mask as this node's graph-owned
                # resource when a graph is built, and closes it immediately
                # when no parent requires grad. Same contract as the
                # maxpool2d winner buffer; no second lifetime system.
                result = self._from_op(
                    out_core, (self,), backward, "dropout",
                    graph_resources=(mask,),
                )
            except BaseException:
                # Nothing adopted either object: release the saved state
                # and the output, in the same order maxpool2d uses.
                mask.close()
                out_core.close()
                raise
            # The last addressable failure point before the transaction
            # boundary (see _deliver_dropout_result).
            _deliver_dropout_result(result)
            # The transaction boundary. Everything after it is
            # irreversible: this index is spent whatever happens next.
            generator._commit_call(token)
            # Inside the `try` deliberately, so the commit-to-return
            # window is covered by the cleanup below rather than left
            # open. Nothing else may go here — no allocation, no
            # callback, no graph mutation, no formatting.
            return result
        except BaseException as error:
            # Nothing reached the caller either way, so the result is
            # released either way. What differs is the *reservation*, and
            # the difference is decided by the token's recorded outcome
            # rather than by a local flag — a commit can succeed and the
            # statement after it never run.
            _settle_failed_dropout(generator, token, result, error)
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
        raises rather than reading freed storage).

        **Phase K, milestone K1: a non-differentiable tensor never
        accumulates a gradient**, and says so rather than dropping the
        contribution silently. The distinction matters: a *frozen*
        parameter legitimately drops contributions (that is what
        ``requires_grad=False`` means), while a non-differentiable dtype is
        a request that can never be satisfied at all. Checked before the
        ``requires_grad`` drop and before any allocation, so a rejection
        changes no gradient anywhere."""
        self._require_open()
        if not cpp._is_floating_dtype(self.dtype):
            raise RuntimeError(
                f"a native tensor of dtype {self.dtype!r} cannot accumulate "
                f"a gradient: gradients exist only at the differentiable "
                f"dtypes {cpp.SUPPORTED_DTYPES}"
            )
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
        and no NumPy round trip occurs. Rejected on a closed tensor.

        The copy is allocated before the wrapper is published, so — exactly
        as in ``contiguous_copy`` — a failed publication closes it
        explicitly under every ``BaseException``, and only a successful
        transfer of ownership returns. The dtype is whatever this tensor
        carries, ``int64`` included: ``detach`` copies a value, so it is
        dtype-preserving rather than dtype-choosing."""
        out_core = self._require_open().contiguous_copy()
        try:
            return self._from_core(out_core)
        except BaseException:
            out_core.close()
            raise

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
        # Phase K, milestone K1: the differentiable-dtype gate, before the
        # graph is walked, before the seed is built, and before any
        # gradient is allocated or changed (integer design §9.2). A
        # non-differentiable output is a different failure from one that
        # merely does not track gradients, and reports as one.
        if not cpp._is_floating_dtype(self.dtype):
            raise RuntimeError(
                f"backward() called on a tensor of dtype {self.dtype!r}: "
                f"reverse-mode autodiff runs only at the differentiable "
                f"dtypes {cpp.SUPPORTED_DTYPES} (gradients of integers are "
                f"ill-defined)"
            )
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
            # d(out)/d(out) = 1, as a native scalar-shaped tensor — at the
            # **graph's** dtype (design §11.1/§11.4), through the private
            # typed constructor. A float64 seed for a float32 output would
            # be rejected by the very first gradient accumulation, and
            # rightly: the seed is a gradient of the output, so it has the
            # output's dtype by definition.
            return NativeTensor._from_core(
                cpp.NativeTensorCore._typed_full(
                    core.shape, 1.0, core.dtype, device=core.device
                )
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


def _dropout_backward(input_tensor, mask):
    """Build the Dropout backward closure over one saved multiplier mask
    (G3).

    A factory rather than an inline ``def`` for one reason: it makes
    "closure construction failed" an **addressable** failure position.
    Everything between reserving a generator call and committing it has to
    be provably failure-atomic, and a bare ``def`` inside the operation is
    the one step in that sequence a test cannot reach. This is the same
    reason ``native_generator._deliver_reservation`` exists — private, and
    doing nothing a caller could depend on beyond returning the closure.

    The closure consumes exactly two things: the upstream gradient and the
    graph-owned mask its own forward saved. It never rereads the input's
    values, never redraws, never touches a ``NativeGenerator``, and never
    mutates the mask — inverted Dropout's gradient is ``upstream * mask``,
    computed by the existing native ``multiply`` at the graph-unaware Core
    level, so there is no Dropout backward kernel (design §7.5) and the
    resulting gradient carries no graph of its own."""

    def backward(upstream):
        # Identical shapes (the output is the input's shape and the mask
        # is the output's), so no unbroadcast reduction is possible or
        # needed.
        grad_core = upstream._require_open().multiply(mask)
        contribution = NativeTensor._from_core(grad_core)
        try:
            input_tensor._accumulate_grad(contribution)
        except BaseException:
            # _accumulate_grad adopts a contribution only on the
            # assignment that ends it, so an unadopted gradient is
            # released here rather than left to the __del__ safety net.
            contribution.close()
            raise

    return backward


def _chain_cleanup_failure(error, cleanup_error):
    """Attach a failed cleanup step to the failure that caused it, keeping
    the **operation's** exception primary.

    The cleanup steps below are non-failing by contract — closing a
    result is idempotent, and abandoning a live matching reservation is
    two integer writes — so reaching here means something is already
    wrong. Substituting the cleanup error for the original would then
    report the *second* problem and hide the first, which is exactly
    backwards: the caller needs to know why the forward failed. So the
    original propagates and the cleanup failure is chained onto the end
    of its context chain, where the default traceback machinery still
    prints it. Nothing is swallowed, and an existing chain is appended to
    rather than overwritten.

    The resulting chain is **acyclic**, which takes one deliberate step
    (G6). A cleanup step raised while ``error`` was being handled gets an
    implicit ``__context__`` pointing straight back at ``error`` — so
    appending it without cutting that back-reference would close a loop,
    and a cyclic context chain makes every ordinary "follow
    ``__context__`` to the end" reader spin forever: this function on its
    next call, and any logging or error-reporting code the caller runs.
    The relationship is not lost; the link written below states it in the
    one direction that terminates."""
    if cleanup_error is error:
        return
    # Walk to the end of the existing chain, recording everything already
    # in it. The ``seen`` guard keeps this finite regardless of what any
    # other producer of ``__context__`` links left behind.
    tail = error
    seen = {id(error)}
    while tail.__context__ is not None and id(tail.__context__) not in seen:
        tail = tail.__context__
        seen.add(id(tail))
    if id(cleanup_error) in seen:
        # Already somewhere in the chain: appending it would be the loop.
        return
    # Cut the cleanup failure's own back-reference into that chain before
    # attaching it. Anything else it carries — a genuine inner cause of
    # its own — is left exactly as it is.
    node = cleanup_error
    inner = {id(cleanup_error)}
    while node.__context__ is not None and id(node.__context__) not in inner:
        if id(node.__context__) in seen:
            node.__context__ = None
            break
        node = node.__context__
        inner.add(id(node))
    tail.__context__ = cleanup_error


def _settle_failed_dropout(generator, token, result, error):
    """Clean up after a Dropout forward that raised, **outcome-aware**
    (G3; design §5 and §14).

    The two cases are genuinely different, and only the token knows which
    one this is:

    * **Before a successful commit** — the overwhelmingly common case, and
      every injectable one. No call has been consumed, so the reservation
      is abandoned: ``calls`` stays exactly where the forward found it,
      the slot is cleared, and the very next forward reuses the same
      unconsumed index and reproduces the mask this one would have.
    * **After a successful commit, before the caller receives the result**
      — an asynchronous exception in the commit-to-return window. That
      index is **irreversibly spent**: ``calls`` has already advanced
      exactly once and the reservation slot is already clear. Abandoning
      the committed token would raise "already committed" and mask the
      real failure, so it is not attempted; the generator simply carries
      on from its next index.

    In both cases the unreturned result is closed, which releases the
    graph-owned multiplier mask with it, so no caller can observe a
    partial result and native live storage returns to its baseline.

    The outcome is read from the token, never from a flag set after
    ``_commit_call``: a commit can succeed and the next statement never
    execute, and a cleanup that guessed wrong there would either strand a
    reservation or report the wrong error. Every step is attempted even
    if an earlier one fails — the discipline ``_native_state``'s
    post-commit release loop already uses — and a cleanup failure is
    chained onto ``error`` rather than substituted for it. This function
    never raises."""
    # Default to "not committed" if the query itself somehow fails:
    # attempting the cancellation is the safe guess, because
    # ``_abandon_call`` is exact-match and refuses a committed token
    # without changing anything, whereas skipping it could strand a live
    # reservation for the rest of the process.
    committed = False
    try:
        committed = generator._call_committed(token)
    except BaseException as cleanup_error:   # pragma: no cover - defensive
        _chain_cleanup_failure(error, cleanup_error)
    try:
        if result is not None:
            result.close()
    except BaseException as cleanup_error:   # pragma: no cover - defensive
        _chain_cleanup_failure(error, cleanup_error)
    if not committed:
        try:
            generator._abandon_call(token)
        except BaseException as cleanup_error:
            _chain_cleanup_failure(error, cleanup_error)


def _deliver_dropout_result(result):
    """The result → commit seam of the Dropout call transaction (G3).

    A deliberate no-op, and the exact analogue of
    ``native_generator._deliver_reservation``. It exists so the last
    window before the generator commit — the point at which the output
    exists, the graph owns the mask, and the call has *not* yet been
    consumed — is addressable by a test rather than only by argument. A
    failure here must cancel the reservation and release the whole result,
    so that the very next forward reuses the same unconsumed call index.

    Private and module-level: never exported, never referenced from a
    public API, and it does nothing a caller could depend on."""
    return result


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
    """A fresh owning row-major contiguous NativeTensorCore holding
    ``core``'s value — the native line's one **value-transfer** primitive.

    Every caller wants the same thing: an independent contiguous
    materialization of some tensor's current value, for any layout
    (contiguous, transposed, narrowed, offset, rank-0). It is the staging
    step behind ``NativeParameter.copy_value_``, both ``state_dict()``
    snapshot paths, both ``load_state_dict()`` staging paths, the
    BatchNorm running-statistics transaction, and the reshape/transpose/
    unbroadcast gradient materializations.

    **Phase H, milestone H5 — this is a copy, so it copies.** It delegates
    to ``NativeTensorCore.contiguous_copy()``: one uninitialized
    allocation (H1) and one identity gather. Before H5 it was spelled
    ``zeros(shape) + core`` — two allocations, a full zero-fill pass, and
    a full elementwise-addition pass — a composition that predates the
    E3.1 native gather and survived only because nothing had re-examined
    it. The gather is strictly less work at every size and strictly more
    faithful:

    - ``0.0 + (-0.0)`` is ``+0.0`` under IEEE-754, so the addition
      **normalized negative zero away**; a gather preserves it.
    - An addition **quiets a signaling NaN**; a gather preserves it.

    Both are now preserved, which is what makes this agree with the three
    value-copy paths that always used the gather — ``NativeParameter``
    construction, ``detach()``, and the ``to_numpy()``/``from_array``
    serialization boundary. Nothing else in the 17-pattern IEEE-754 sweep
    ever differed (see docs/native_cpu_performance_design.md §17.3).

    The result owns its storage, is contiguous at offset 0, aliases
    neither ``core`` nor anything else, and leaves ``core`` — value,
    layout, and ownership — untouched. A failed gather closes the fresh
    allocation before propagating, so no partially written core escapes
    and live storage returns to baseline. A closed ``core`` raises
    RuntimeError.

    This helper stays module-level and private: it is the seam the state
    and optimizer suites monkeypatch to inject a staging failure, and it
    is not a public mutation API."""
    return core.contiguous_copy()


def _negated(tensor):
    """-tensor as a fresh owning NativeTensor, natively: a multiply
    against a broadcast native ``-1.0`` scalar. The runtime has no negate
    kernel; reusing the broadcasting multiply is the design's recommended
    composition (docs/native_autograd_design.md §7.5), and it never
    mutates its input — the same upstream object may also flow,
    un-negated, to the other parent.

    The constant is built at **the operand's dtype**, through the private
    typed constructor (Phase I, milestone I4). A backward may not introduce
    a literal-float64 constant into a graph of another dtype (design §11.4):
    a float64 ``-1.0`` meeting a float32 gradient would be a mixed-dtype
    multiply, which the runtime refuses before allocating anything. The
    private route is what keeps that from also requiring public float32
    construction, which does not exist."""
    core = tensor._require_open()
    neg_one = cpp.NativeTensorCore._typed_full(
        (), -1.0, core.dtype, device=core.device
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
    dtype/device preserved; no NumPy and no new kernel.

    Phase I, milestone I4: the zero operand is built at **the upstream's
    dtype** through the private typed constructor, for the same reason
    ``_negated``'s ``-1.0`` is — a float64 zero meeting a float32 gradient
    would be a mixed-dtype add, rejected before any allocation. The
    algorithm, the allocation count, and the reduction axes are otherwise
    exactly what they were: this is still a genuine expansion (the adjoint
    of a reduction), not a copy."""
    u_core = upstream._require_open()
    keep_shape = cpp.reduce_shape(x_shape, axis=axis, keepdims=True)
    transient = None
    if u_core.shape != keep_shape:
        if not u_core.contiguous:  # a user-supplied gradient view
            transient = u_core = _native_copy(u_core)
        u_core = u_core.reshape(keep_shape)  # borrowing view, used below
    zeros = cpp.NativeTensorCore.zeros(
        x_shape, dtype=u_core.dtype, device=u_core.device, _trusted_dtype=True
    )
    result = zeros.add(u_core)
    zeros.close()
    if transient is not None:
        transient.close()
    return NativeTensor._from_core(result)
