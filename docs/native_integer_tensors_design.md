# Native integer tensors and indexing — Phase K architecture contract

**Phase K — Native Integer Tensors and Indexing.** This document is the
authoritative architecture contract for the phase. It is written **before**
any integer implementation exists, and milestone **K0** consists of exactly
this document, the status reconciliation it requires, and the contract
guardrails in `tests/test_native_phase_k.py` that keep it honest.

**Phase K is newly approved, and it was approved after Phase J closed.**
Phase J — deterministic native data pipeline and mini-batching — **remains
complete**: milestones J0 through J9 all landed, J9 closed the phase, and
nothing in Phase K reopens, supersedes, or re-describes any of it. The
repository deliberately closed Phase J at J9 **without committing to a
successor**, and Phase K was approved afterwards. That ordering is a fact
about how this project works and must not be rewritten: "the phase that
came next" and "the phase that was always planned next" are different
statements, and only the first is true here. Phase K was not part of the
Phase-I roadmap, was not planned during Phase J, and did not exist before
the branch that carries this document.

**Phase-K status: K0 through K8 complete. K0 through K8 are the
only completed Phase-K milestones. K9 is unstarted.**

**K0 adds no runtime behavior.** No integer dtype, no dtype code, no C++
enumerator, no storage change, no kernel, no C ABI symbol, no ctypes
declaration, no `NativeTensorCore` method, no `NativeTensor` operation, no
module, no optimizer change, no public export, no capability-registry
value, no checkpoint field, no checkpoint version, no optimizer-state
version, no loader-state version, no sampler-state version, no example, no
benchmark, no CTest, no build option, and no dependency. K0 is
architecture, contract, status reconciliation, and guardrails, and nothing
else. Runtime capability begins at **K1**.

**K1 adds the internal `int64` representation and every reachability
barrier, and nothing else.** What is now true internally: `TfDtype` /
`tf::Dtype` carry a third enumerator at code **2**, `create_storage` and
`destroy_storage_data` have an `Int64` arm, the four transfer boundaries
(`tf_storage_copy_from`, `tf_storage_copy_to`, `tf_storage_materialize`,
`tf_core_contiguous_copy`) move `int64` values **bit for bit**, and every
other handle-based export refuses an `int64` operand through the new
hidden-visibility `tf::require_floating` guard.

What was **not** true at K1, and was asserted false by test then: there was
no public `int64` Python capability of any kind. The Python dtype tables
(`_DTYPE_CODES`, `_DTYPE_ITEM_SIZES`, `_DTYPE_NUMPY`,
`_CHECKED_HOST_ARRAYS`) were **untouched at K1**, so no supported
TensorForge wrapper or public Python API could allocate or wrap `int64`
storage at K1; only the raw private C ABI could represent it, for isolation
and barrier testing. K1 added **no** C ABI symbol, **no** public Python
name, **no** registry value, and **no** version change; it added one native
CTest (24 → **25**) and moved no other inventory.

**K2 made the `int64` tensor publicly constructible, and it landed
atomically** — splitting it would have opened exactly the window §32.1
forbids. What is now true: the three Python dtype tables and the host
binding know `"int64"`; `INDEX_DTYPES == ("int64",)` exists beside an
**unmoved** `SUPPORTED_DTYPES` and is reported as
`backend_info()["index_dtypes"]`; `NativeStorage._from_int64_array` and
`NativeTensorCore._from_int64_array` are the private, exact,
non-converting ingress; the `NativeTensorCore.__init__` and
`NativeTensor.__init__` gates widened from "floating" to "floating **or**
index" and to nothing else; and **`NativeTensor.from_int64_array` is the
one public API in the repository through which an `int64` buffer can come
into existence**, beside the dtype-general `item()` and `tolist()`. Views,
copies, and host inspection work at `int64` through the machinery that
already existed. K2 added **no** C ABI symbol, **no** experimental export,
**no** CTest, **no** example, **no** benchmark, and **no** version change
of any kind, and every barrier it could meet had already landed at K1.

**K3 shipped the phase's first operation and its first C ABI symbol:
native `argmax`.** `NativeTensor.argmax(axis=None, keepdims=False)` and
`NativeTensorCore.argmax(axis=None, keepdims=False)` search a **floating**
tensor at either dtype, at any rank including 0, contiguous or not, and
return a **fresh owning contiguous `int64` tensor** — the first operation in
the runtime whose result dtype differs from its operand's, which is the
point of it. Behind them is one new export, `tf_core_argmax`, whose source
must be floating and whose destination must be exactly `int64`, and which
applies **neither** `tf::require_floating` **nor**
`tf::require_matching_dtype` to that destination because either would reject
every valid call (§22.8). The value rule of §17.5 is normative and is proved
row by row at both widths. `"argmax"` joined `TENSOR_CORE_OPS` and
**deliberately did not join `AUTOGRAD_OPS`**: an index has no derivative, so
the result is a plain leaf even when the input requires grad. The exports
went 54 → **55** and the native CTests 25 → **26**; nothing else moved, and
`max` was **not** shipped (§17.10).

**K4 shipped the phase's one index-*consuming* operation and its second and
final C ABI symbol: native `index_select`, forward only.**
`NativeTensor.index_select(axis, indices)` and
`NativeTensorCore.index_select(axis, indices)` take a **floating** source at
either dtype, any rank ≥ 1, contiguous or not, together with a rank-1
`int64` index tensor, and return a **fresh owning contiguous** tensor of the
**source's** dtype whose selected axis has `indices.numel` positions. It is
`argmax`'s mirror image: that one produces an index from values, this one
consumes one and produces values. Behind them is one new export,
`tf_core_index_select`, whose source and destination are floating and must
**agree** — the one place in the phase `tf::require_matching_dtype` is used,
and never across the floating/index boundary — and whose separate index
handle must be exactly `int64` (§22.9). Duplicates and order are preserved
exactly, negative and out-of-range indices are **rejected rather than
wrapped**, the complete bounds scan runs in Python *and* independently in
C++ before the first destination element is written, and values cross by
**object representation**, so both signed zeros, both infinities,
subnormals, and every NaN payload survive bit for bit. `"index_select"`
joined `TENSOR_CORE_OPS` and **deliberately did not join `AUTOGRAD_OPS`**: a
source with `requires_grad=True` is **rejected with a message naming
`detach()`** rather than silently detached, because the backward is a
scatter-add with its own contract and its own milestone and a graph-free
result from a gradient-tracking source would be a silent gradient hole
(§18.9). The exports went 55 → **56** — the phase maximum — and the native
CTests 26 → **27**; nothing else moved, and no general `gather`, `scatter`,
or embedding lookup was shipped (§18.1).

**K5 proved the whole of that, and moved nothing.** It is a
test-and-documentation milestone with **zero production code**: one new
module, `tests/test_native_integer_compatibility.py`, which drives the
checkpoint, the in-memory optimizer state, the Phase-J loader and sampler
states, the Phase-J delivery contract, `NativeCrossEntropyLoss`,
`native_accuracy`, and a real interrupted-and-resumed training run, and
shows that K1 through K4 left every one of them exactly where they found
it. No archive can declare an `int64` entry; the Phase-J delivery contract
still returns a floating `NativeTensor` feature batch and a **read-only
host `numpy.ndarray` target batch of dtype `int64`** — the shorthand
`(NativeTensor, numpy.int64)` used elsewhere in this document names that
host array's dtype, never a NumPy scalar, and the current-status surfaces
spell it out rather than rely on the shorthand; the caller-conversion route
works and needs no pipeline change; and a model trains, checkpoints, and
resumes **bit-identically** at both widths while `argmax` and
`index_select` are used beside it.

**K6 turned all of that into one end-user program, and moved exactly one
inventory.** It is the phase's end-to-end integration example and carries
**zero production code**: `examples/native_integer_indexing.py`, owned by
`tests/test_native_integer_indexing_example.py`. A deterministic native
classifier trains over the Phase-J pipeline, and at fixed evaluation points
on **both sides** of an interruption its logits become native `int64`
predictions through `argmax` which are then consumed by `index_select` over
a **detached** copy of the same logits — the two calls taking deliberately
different sources for the two reasons §17.9 and §18.9 give. The
interrupted-and-resumed run reproduces the uninterrupted one **exactly** at
float64 and float32 independently, every prediction index by exact integer
equality and every floating value by raw IEEE-754 bits. The example is
written entirely against the public experimental surface, closes every
native object it creates, leaves no file behind, claims no timing, and
returns live storage exactly to its baseline. **Examples went 16 → 17**, and
nothing else moved: no C ABI symbol (still **56**), no public Python name
(`__all__` still **25**), no CTest (still **27**), no benchmark (still
**9**), no registry value, and no version. The only file it touches under
`src/` is the package docstring's Phase-K status sentence, which carries no
capability.

**K7 is the adversarial hardening milestone, and it added zero production
code.** Its whole deliverable is
`tests/test_native_integer_hardening.py`, which attacks the shipped
integer stack rather than extending it: §27's four injection families at
**every actual allocating path**, resolved from the live call graph and
recorded as a traceable matrix in which a family that genuinely does not
apply to a path is an `N/A` with its technical reason rather than a
borrowed injection, and in which one export reached from two different
call sites gets **two rows** — `index_select`'s source and index Policy-B
materializations, the second driven by a call journal that delegates the
first to the real export so it can be reached at all; a complete
before/after fingerprint of the observable world after every rejection and
every injected failure, with a scan of the module's own AST making that
"every" literal; a `BaseException` through each cleanup-capable seam;
retained-reference proofs so no closure claim can rest on `__del__`
timing; the malformed-metadata **and** dtype-role matrices for **both**
exports, kept separate because their validation lists are, each asserting
every operand byte-identical after every rejection; the complete index
scan proved to precede the first destination byte; and a non-vacuity
control for every injector, every fingerprint component, and every parser.
Every allocation row fires the backend's **own** thread-local arm, armed
at the exact production seam it names. K7 found **no production defect**.
**No inventory moved**:
exports still **56**, CTests **27**, examples **17**, benchmarks **9**,
`__all__` **25**, every registry and every version exactly what K6 left;
the only file it touches under `src/` is the package docstring's Phase-K
status sentence, which carries no capability.

**K8 is the benchmark characterization milestone, and it added zero
production code too.** Its whole deliverable is
`benchmarks/benchmark_native_integer.py` and its owner
`tests/test_native_integer_benchmark.py`, and it **measures** the shipped
integer stack rather than changing it. It answers §31's four questions and
keeps them four — `integer_construction`, `host_materialization`,
`argmax`, `index_select` — with **no composed case at all**, because one
`argmax`-then-`index_select` number could not say which of the two
dominates and labelling it a composition would not fix that. Every case is
`native_only` and publishes **no ratio**: each of the four families
allocates native storage and transfers into or out of it while the
apparent host equivalent does not, which is exactly the fairness risk §31
names by name for `argmax` against `numpy.argmax`, and a conservative
absence is worth more than a ratio a reader would have to discount. The
correctness gates run to completion **before** the timing helper is
reached — proved both structurally, off the AST, and behaviourally with a
spy timer for **every** case — and they are exact: integer values and
`argmax` results by exact integer equality against an independent
transcription of §17.5's algorithm and that section's committed twelve-row
case table, floating payloads by raw IEEE-754 bits against a per-position
slice concatenation written without `numpy.take`. The three measured
dtypes are characterized separately and none is divided by another, and
`--dtype int64` is documented as selecting the **index/result** families
rather than as naming a compute width. No file is written in any mode, no
CLI option could ask for one, no duration is a pass/fail criterion, and
the AST shows the timed region of each family holding exactly one
operation call — with a non-contiguous operand's internal Policy-B
materialization deliberately **inside** it, because it is part of the
operation. K8 moves exactly one inventory: benchmarks **9 → 10**. Exports
stay **56**, CTests **27**, examples **17**, `__all__` **25**, and every
registry and version is exactly what K7 left; the only file it touches
under `src/` is the package docstring's Phase-K status sentence, which
carries no capability. **No native build was performed or required**, and
no measurement changed the runtime.

**`int64` is not a supported TensorForge native tensor dtype**, at K8 or
ever. It is an **index/result** dtype, in a separate registry, and the
distinction is the whole of the phase's taxonomy (§5.1). K3 and K4 are where
that distinction earns itself: one operation now *produces* `int64` and
another *consumes* it as a role operand, without `int64` ever becoming a
dtype a kernel computes at. The compute boundary is exactly what Phase I
established and Phase J left untouched, and no Phase-K milestone has moved
any of it:

| Row | Value at K0 | Value at K1 | Value at K2 | Value at K3 | Value at K4 | Value at K5 | Value at K6 | Value at K7 | Value at K8 |
|---|---|---|---|---|---|---|---|---|---|
| `SUPPORTED_DTYPES` | `("float64", "float32")` | unchanged | unchanged | unchanged | unchanged | unchanged | unchanged | unchanged | unchanged |
| `SUPPORTED_DEVICES` | `("cpu",)` | unchanged | unchanged | unchanged | unchanged | unchanged | unchanged | unchanged | unchanged |
| `UNSUPPORTED` | `("cuda", "amp")` | unchanged | unchanged | unchanged | unchanged | unchanged | unchanged | unchanged | unchanged |
| `RAW_KERNEL_DTYPES` | `("float64",)` | unchanged | unchanged | unchanged | unchanged | unchanged | unchanged | unchanged | unchanged |
| `normalize_dtype(None)` | `"float64"` | unchanged | unchanged | unchanged | unchanged | unchanged | unchanged | unchanged | unchanged |
| `normalize_dtype("int64")` | raises `ValueError` | unchanged | unchanged | unchanged | unchanged | unchanged | unchanged | unchanged | unchanged |
| `backend_info()["dtype"]` | `"float64"` | unchanged | unchanged | unchanged | unchanged | unchanged | unchanged | unchanged | unchanged |
| `backend_info()["stable_framework_integration"]` | `False` | unchanged | unchanged | unchanged | unchanged | unchanged | unchanged | unchanged | unchanged |
| `INDEX_DTYPES` | absent | absent | **`("int64",)`** | unchanged | unchanged | unchanged | unchanged | unchanged | unchanged |
| Python `_DTYPE_CODES` | `float64`, `float32` | unchanged | **+ `int64: 2`** | unchanged | unchanged | unchanged | unchanged | unchanged | unchanged |
| C++ `TfDtype` | `FLOAT64`, `FLOAT32` | **+ `INT64 = 2`** | unchanged | unchanged | unchanged | unchanged | unchanged | unchanged | unchanged |
| Public integer constructor | absent | absent | **`NativeTensor.from_int64_array`** | unchanged | unchanged | unchanged | unchanged | unchanged | unchanged |
| `TENSOR_CORE_OPS` | Phase-J set | unchanged | unchanged | **+ `"argmax"`** | **+ `"index_select"`** | unchanged | unchanged | unchanged | unchanged |
| `AUTOGRAD_OPS` | Phase-J set | unchanged | unchanged | unchanged | unchanged | unchanged | unchanged | unchanged | unchanged |
| Native checkpoint format · version · accepted | `tensorforge.native_checkpoint` · **3** · `(1, 2, 3)` | unchanged | unchanged | unchanged | unchanged | unchanged | unchanged | unchanged | unchanged |
| In-memory optimizer state version | **1** | unchanged | unchanged | unchanged | unchanged | unchanged | unchanged | unchanged | unchanged |
| Loader state format · version · accepted | `tensorforge.native_data_loader` · **1** · `(1,)` | unchanged | unchanged | unchanged | unchanged | unchanged | unchanged | unchanged | unchanged |
| Sampler state format · version · accepted | `tensorforge.native_sampler` · **1** · `(1,)` | unchanged | unchanged | unchanged | unchanged | unchanged | unchanged | unchanged | unchanged |
| Exported production `tf_*` symbols | **54** | **54** | **54** | **55** | **56** | **56** | **56** | **56** | **56** |
| Experimental Python exports | **25** | **25** | **25** | **25** | **25** | **25** | **25** | **25** | **25** |
| Native CTests · examples · benchmarks | **24** · **16** · **9** | **25** · **16** · **9** | **25** · **16** · **9** | **26** · **16** · **9** | **27** · **16** · **9** | **27** · **16** · **9** | **27** · **17** · **9** | **27** · **17** · **9** | **27** · **17** · **10** |

**`SUPPORTED_DTYPES` never gains `int64` — not in Phase K and not
afterwards** (§5). It is, and permanently remains, the **floating compute**
registry, and `normalize_dtype("int64")` keeps raising `ValueError`
forever. What Phase K added instead is one **separate, clearly named**
public registry, `INDEX_DTYPES == ("int64",)`, and it appeared exactly
once, at **K2**, in the same commit as the public integer constructor —
never earlier. That ordering — **prove first, then promise** — is the rule
Phase G used for `dropout` and Phase I used for `float32`, applied
unchanged.

The consequence is the property this contract is built around, and K2
preserved it exactly: **no existing generic constructor changed what it
accepts, at any Phase-K milestone.** `NativeStorage(...)`,
`NativeStorage.from_array`, `NativeTensorCore.from_array` / `zeros` /
`full`, and `NativeTensor.from_array` / `zeros` / `full` all validate
through `normalize_dtype`, which permanently rejects `"int64"` — so there
is no milestone at which one of them could have been narrowed and was not
(§5.4), and the one integer door is a **new** name rather than a widened
old one.

Four things no document, comment, test, or status surface may say, because
none of them is true and the guardrails in `tests/test_native_phase_k.py`
fail if one is written:

- `int64` is a *supported* dtype — it is an **index/result** dtype in its
  own registry, and `SUPPORTED_DTYPES` is the floating-compute registry;
- a `max`, a `max_with_indices`, or an `argmin` exists — none does, and
  `max` is declined **permanently** by §17.10 rather than deferred;
- a general `gather`, a `scatter`, a `scatter_add`, an embedding lookup, or
  an `index_select` **backward** exists — none does (§18.1, §18.9, §35);
- Phase K is **not** complete, and no milestone after K8 has landed.

Three things that were on this list and have been **moved rather than
deleted**, which is the discipline the list exists to demonstrate:

- "no native integer tensor exists" was true through K1, because K1 shipped
  a *representation* and the two are not the same thing. K2 shipped the
  tensor, so the sentence became false and the guardrail asserts its
  replacement — the tensor exists, is publicly constructible through exactly
  one door, and is still not a supported compute dtype.
- "no `argmax` is implemented" was true through K2 and is false from **K3**,
  which shipped it. A guardrail that kept banning the sentence "a native
  `argmax` exists" would force every status surface to under-report the
  project, which is the mirror of the failure the ban existed to catch — so
  the entry moved, and what replaced it is the narrower claim K3 did not
  earn.
- "index selection is available" was true as a prohibition through K3 and is
  false from **K4**, which shipped `index_select`. The entry moved for
  `argmax`'s reason, and what replaced it is the narrower claim K4 did not
  earn: the general `gather`, the `scatter`, the embedding, and the
  **backward**, none of which exists.

---

## 1. Status and historical positioning

### 1.1 Where the native line stands

Phases A through J are complete. The native line is a C++17 CPU runtime
behind a plain C ABI loaded with `ctypes`, with its own dtype-tagged
storage, strided runtime, Python-managed autograd, modules, losses, a
metric, optimizers, an explicit RNG, pickle-free checkpoints, and a
deterministic data pipeline. It computes at **float64 and float32** and at
no other width, and it still does. **Before Phase K** it had no integer
tensor of any kind — that is the baseline this contract was written
against, and it is a statement about the pre-K1 runtime rather than about
today's. K1 added the internal `int64` representation and K2 made the
tensor publicly constructible; `int64` is an **index/result** dtype in its
own registry and is still not a supported compute dtype (§5.1).

### 1.2 What Phase K is

Phase K is **newly approved**, after Phase J closed without a committed
successor. It adds a **carefully bounded native `int64` representation** so that
index-valued results can exist as native tensors rather than only as host
metadata. Concretely, by the end of the phase the native line will be able
to produce an index tensor from a floating reduction (`argmax`), consume an
index tensor to select along one axis (`index_select`), and move exact
integers in and out of native storage — with every one of those operations
non-differentiable, exact, CPU-only, and outside the parameter, optimizer,
buffer, and checkpoint surfaces.

### 1.3 What Phase K is not

Phase K is **not** permission to implement every integer feature. It is not
an integer arithmetic phase, not a general indexing phase, not an embedding
phase, not a dtype-expansion phase, and not a step toward CUDA. §35 states
the non-goals as boundaries, and §32's ladder is the whole of the approved
work.

### 1.4 The honest claim K0 makes

K0 claims exactly one thing: **the architecture below is decided.** Every
question a K1 implementer would otherwise have to invent an answer to is
answered here, with the alternatives recorded and rejected. Where the
evidence was insufficient to decide safely, the capability is **narrowed or
deferred** rather than left ambiguous — §35 lists what was deferred and
why.

---

## 2. Motivation

### 2.1 The concrete gap

**This is the gap as it stood when K0 was written — before K1 — and it is
recorded in the past tense because K2 closed the dtype half of it.** Three
places in the shipped runtime said the same thing in different words.

`src/tensorforge/experimental/native_metrics.py` stated it most directly:
`native_accuracy` materializes its logits through the public `to_numpy()`
boundary and takes a NumPy `argmax`, **because before K1 the runtime had
no native integer dtype for an index-producing reduction to return**.
The capability inventory in `src/tensorforge/backends/cpp.py` recorded the
same fact beside `NATIVE_METRICS`. `NativeTensorDataset.target_batch`
recorded the third form: class targets are host `int64` metadata at every
width, and no native integer tensor existed, was needed, or was implied.

None of those was a defect. Each was an accurate statement about a runtime
whose only element types were `float` and `double`. Phase K exists to
change the underlying fact rather than the wording.

**Where each of the three stands now.** K2 gave the runtime an exact
`int64` index/result dtype, so the first two *reasons* expired while both
*conclusions* stand, and §20.3 records the correction: `native_metrics.py`
and the `NATIVE_METRICS` comment now say that a native `argmax` is absent
because no milestone has shipped one — K3 owns it — rather than because
its result type is inexpressible. The third is unchanged and stays
unchanged: classification targets remain exact host-side label metadata
under the Phase-E contract, and no Phase-K milestone widens cross-entropy
or the data pipeline to accept a `NativeTensor` target (§19, §20.1–§20.2).

### 2.2 Why `int64` and why now

An index is not a small float. A float64 can represent every integer up to
2^53 exactly, which is why the private MaxPool2d winner buffer is a float64
plane offset with an explicitly checked 2^53 bound — a pin the project
accepted deliberately and documented as permanent. That trick does not
generalize: it needs a proof per call site, it silently loses exactness
beyond its bound, and it cannot express "this tensor holds indices" to
anything that reads it. A dtype that *is* an integer states the fact once,
in the one place the runtime already treats as authoritative — the storage
tag.

`int64` is the only candidate (§5.6) because it is the width every index
position in the C ABI already uses, the width NumPy's default integer is on
the platforms this project supports, and the width the existing host target
metadata already travels at. Adding it introduces no new integer width
anywhere; it lets an existing one be owned by native storage.

### 2.3 What becomes possible

- `argmax` as a native operation returning a native index tensor, instead
  of a host round trip (§17).
- Selection along one axis by an index tensor (§18), which is the first
  primitive from which prediction gathering, per-example lookup, and — in a
  **separately approved** future phase — embedding are built.
- A single, exact, inspectable representation for index-valued results,
  with `item()` and `tolist()` returning exact Python integers (§16).

### 2.4 What deliberately does not become possible

Integer arithmetic, integer reductions, integer comparison operations,
integer parameters, integer gradients, integer optimizer state, integer
checkpoint entries, casting between integer and floating tensors, and
native integer targets in the data pipeline. Each is refused in §35 with
its reason.

---

## 3. Current inherited boundary

This section records repository reality at K0, verified by reading the live
tree rather than by inheriting an earlier document's summary. Everything
here is a **constraint on Phase K**, not a description of Phase K.

### 3.1 The dtype model

Exactly **two** dtype authorities exist and they agree by construction
because the ABI codes are the same integers:

- `TfDtype` / `tf::Dtype` in `cpp/include/tf_internal.h`, today
  `TF_DTYPE_FLOAT64 = 0` and `TF_DTYPE_FLOAT32 = 1`, with
  `tf::dtype_from_code`, `tf::dtype_item_size`, `tf::dtype_name`,
  `tf::dtype_is_float32`, `tf::dtype_is_float64`, and
  `tf::dtype_checked_bytes` beside them;
- `_DTYPE_CODES`, `_DTYPE_ITEM_SIZES`, and `_DTYPE_NUMPY` in
  `src/tensorforge/backends/cpp.py`.

There is **no third table**, and there is **no exported dtype-query
symbol**: Python knows a storage's dtype because Python asked for it at
creation. `tf::storage_dtype` is a hidden-visibility accessor, not an
export.

The C++ header already records the intent this phase acts on: the codes are
frozen in the sense the `TfStatus` codes are, and the comment beside the
enumeration states that a hypothetical future dtype would take **2**.

### 3.2 Storage

`tf::Storage` is `{ void* data; int64_t size; Dtype dtype = Dtype::Float64; }`.
`data` is deliberately **untyped** — no `double*` member, no union of typed
pointers — and points at a genuine `float[size]` or `double[size]` array
object created by an array new-expression and type-erased afterwards, which
is what makes the kernels' `data[i]` and `data + i` well-defined under
C++17. `size` is a **logical element count**. The dtype tag is the single
authority for how the bytes are read, is assigned once before the handle is
published, and is immutable.

`create_storage` validates the positive size, checks `numel × itemsize`
through `tf::dtype_checked_bytes`, applies deterministic fault injection,
and then performs **one** dtype dispatch into a templated allocation body.
`destroy_storage_data` mirrors that dispatch to select the matching
`delete[]`. Neither switch has a `default:` label, so a new enumerator
without an arm is a compile-time diagnostic — and the project requires zero
compiler warnings.

**Zero-element storage does not exist**: `create_storage` rejects
`size <= 0` before anything is allocated. Phase K inherits that and does not
change it (§13.7).

### 3.3 Views and the tensor runtime

`NativeTensorView` binds a storage to shape, strides, and an offset, all in
**logical elements**. A view carries **no dtype of its own** — it reads the
storage's tag — so every view of one buffer necessarily agrees, and no view
operation can cast. Rank 0 is fully representable: a `()` view has
`ndim == 0`, `numel == 1`, `strides == ()`, is contiguous, and materializes
to a 0-d NumPy array. `NativeTensorCore.sum(axis=None)` already produces
one.

`NativeTensorCore` owns a `NativeStorage` and a `NativeTensorView`;
`NativeTensor` wraps a core and adds the Python-managed autograd graph.
`close()` exists exactly where something is owned. Every operation
allocates a **fresh owning contiguous** output that aliases neither
operand.

### 3.4 Dispatch

Each dtype-general handle-based export runs, in order:
`tf::require_matching_dtype` (every non-null operand carries the same tag)
and then **one** `tf::dispatch_dtype` read, feeding a two-arm `switch` whose
`Float32` arm calls the `float` instantiation and whose `Float64` arm
`break`s into the `double` instantiation below it. Exports that have not
been generalized call `tf::require_float64` instead, whose documented
default rule is: **if an operation has not been explicitly generalized, a
non-float64 handle is invalid for it.**

This is the single most important inherited fact for Phase K, and §22.4
turns it into a hard requirement: the *shape* of that dispatch —
`case Float32: … ; case Float64: break;` followed by a `double` fallthrough —
means a third enumerator would be **read as `double`** if the compile-time
diagnostic were ever ignored. Phase K does not rely on a warning to prevent
that.

### 3.5 Host transfer

`tf_storage_copy_from`, `tf_storage_copy_to`, and `tf_storage_materialize`
already take `void*` host positions and dispatch **once** on the storage
tag into templated same-type assignment loops. The host pointer carries no
dtype and cannot be made to; the contract is that the caller supplies a
contiguous host buffer of exactly the storage's element type, and the
Python wrapper enforces it per call through `_host_pointer`, which runs the
per-dtype `numpy.ctypeslib.ndpointer` check (element type, byte order,
C-contiguity) chosen from `_CHECKED_HOST_ARRAYS`.

Transfer performs **no arithmetic**, so it reproduces its source's object
representation exactly. That property is dtype-independent and carries to
`int64` unchanged.

### 3.6 Scalar and fill primitives

`tf_storage_fill(void*, double)` and `tf_storage_scale(void*, double)`
carry their scalar across the ABI as a `double`. A `double` represents
every integer in `[-(2^53), 2^53]` exactly and no integer outside it, so
**neither is an exact integer primitive** and §22.5 keeps both
floating-only permanently.

### 3.7 Autograd, parameters, optimizers

`NativeTensor.__slots__` carries the autograd metadata unconditionally —
`_requires_grad`, `_grad`, `_parents`, `_backward`, `_op`, `_is_leaf`,
`_graph_freed`, `_expected_versions`, `_graph_resources` — whether or not
gradients are tracked. `_from_op` is the **single** internal entry for
graph construction and sets `_requires_grad` to the OR of its parents'.
`_init_requires_grad` accepts `bool` only.

`NativeParameter.__init__` validates `requires_grad`, then canonicalizes
`dtype` through `normalize_module_dtype`, then rejects a stable-framework
object, then either copies a `NativeTensor` source of the **same** dtype or
converts host data through `NativeTensorCore._typed_from_array`. Both
optimizers own only `NativeParameter` objects; `NativeAdam` allocates moment
buffers at each parameter's own dtype.

### 3.8 The internal table is used as a proxy for "floating"

`cpp._normalize_internal_dtype` validates against `_DTYPE_CODES` — the
**representation** table — rather than against the public registry. Today
the two tables accept the same set, so the distinction is invisible. **The
moment `int64` enters `_DTYPE_CODES` it stops being invisible**, because
nine call sites currently use "the runtime can represent this" as a proxy
for "this is a floating dtype". Enumerated from the live source:

| Call site | What goes wrong if it is not narrowed |
|---|---|
| `_native_dtype.normalize_module_dtype` | The one validator the six state-owning constructors share, including `NativeParameter`. `NativeParameter(data, dtype="int64")` becomes legal the same day. |
| `native_checkpoint._validated_entry_dtype` | A hand-written version-3 archive could declare an `int64` model, buffer, or optimizer entry. |
| `cpp._narrowed_to_dtype` | Adam's bias-correction scalar would narrow through an integer NumPy type. |
| `NativeStorage._uninitialized` | An `int64` destination would join the H1 uninitialized-allocation audit without a row or a poison test. |
| `NativeStorage._typed_from_array` | It **casts** (`ascontiguousarray(values, dtype=…)`), so it would silently truncate floats to `int64`. |
| `NativeTensorCore._uninitialized` | As above, one layer up. |
| `NativeTensorCore._typed_from_array` | As above — the same silent truncation. |
| `NativeTensorCore._typed_full` | Fills through `tf_storage_fill(double)`, which is inexact above 2⁵³. |
| `NativeTensorCore.zeros(_trusted_dtype=True)` | Its callers are `sum` and `narrow_backward`, both floating-only accumulators. |

This is a genuine finding rather than a hypothetical, and it is the single
most important reason the ladder is ordered the way §32.1 orders it: adding
an enumerator has a silent, non-local effect on nine sites that never
mention `int64`.

**The resolution is a narrowing, not a new mechanism.** Because Phase K
adopts taxonomy **B** (§5.1), `SUPPORTED_DTYPES` *is* the floating registry
and `normalize_dtype` *is* the floating validator, so every one of the nine
sites simply calls `normalize_dtype`. All nine land at **K1**, one
milestone before `_DTYPE_CODES` gains the entry — so each is
behavior-preserving on the day it lands and preventive afterwards, and
there is no window at all rather than merely a short one.

### 3.9 What the inherited design authorities already decided

Phase K inherits more of this contract than it invents. Each row was read
from the live document, and each is a **constraint**, not a restatement.

| Authority | What it already binds |
|---|---|
| `native_dtype_device_metadata_design.md` §5 | `"int64"` is already in the **reserved future dtype vocabulary**, and dtype tags are deliberately validated lowercase strings, not NumPy objects. Phase K adds a vocabulary entry, not a vocabulary. |
| …§7 | Operation validation for a future integer dtype is **partly pre-decided**: `sum` "preserves the input dtype (a future int64 sum would be int64)", and an integer `mean` is *explicitly deferred and called out, not decided*. **Phase K narrows this**: it ships **no** integer reduction at all (§11.4), so the speculative int64 `sum` does not arrive here. Recorded as a deliberate narrowing rather than an oversight. |
| …§8 | No implicit promotion, no automatic device copying, no silent conversion, no silent NumPy fallback. Inherited verbatim (§12). |
| …§9 | `astype` / `to` / `cpu()` / `cuda()` are **reserved future items**, with the standing recommendation *not to add them at all* until the backend they target is real. Phase K adds none (§35). |
| …§10–§11 | *"A label that promises a capability the kernels lack is a lie the runtime must not tell"*, and the recommendation to **reject** rather than tag inertly. This is the decisive argument for taxonomy **B** (§5.1). |
| `native_tensor_wrapper_design.md` §4 | Owner/borrower semantics, idempotent `close()`, context managers, and the ban on hidden copies — *"Every copy is a named, explicit call."* §8.4's layout normalization is named and explicit; §15 keeps view and copy distinct. |
| …§5 | The explicit conversion contract: `from_array` enters, `to_numpy` exits, no implicit `__array__`, no auto-materialization, and **no `tensorforge.Tensor` in either direction**. §8 and §24 inherit it. |
| `native_autograd_design.md` §5.3 | A result's `requires_grad` is the OR of its differentiable inputs'. §9's rule 3 is the integer specialization. |
| …§10 | **Already states Phase K's central autograd rule**: *"`requires_grad` requires a float dtype … a future `int64`/`bool` native tensor should not accept `requires_grad=True` (gradients of integers are ill-defined)."* §9 implements an inherited contract. |
| `native_abi_error_contract.md` | `TF_GUARD_BEGIN`/`TF_GUARD_END`, the three status codes and their Python mapping, the thread-local slot cleared on entry, unguarded functions that never touch the slot, and *"a failed call never partially mutates caller-visible state"*. §22.7 and §27 inherit all of it. |
| `native_classification_design.md` §6 | The **existing precedent for strict integer validation**: 1-D only, exact length, real integer values, `bool` rejected, floats rejected *including integral ones*, nested rejected, negatives rejected, out-of-range rejected, errors naming the offending index and value. §14 mirrors this shape for index operands. It also records why Phase E did **not** add an integer dtype — *"a public integer `NativeTensor` is a much larger change: storage, ABI, dtype normalization, promotion rules"* — which is precisely Phase K's scope. |
| …§9.3 | *"C++ revalidates trust-boundary data — most importantly every target index it dereferences."* §22.8's C-side bounds re-scan is this rule applied, not a new one. |
| …§11 | The **layer-specific inventory contract** and its forbidden placements. §23.3 follows it: `argmax` and `index_select` join `TENSOR_CORE_OPS` only, never `AUTOGRAD_OPS`, never `RAW_KERNELS`, and `TENSOR_CORE_KERNELS` stays frozen at its historical five. |
| `native_normalization_design.md` §0 | The invariant list Phase K also preserves, including *"Failed operations must not partially mutate caller-visible state."* |
| …§1 | Phase F's exclusion list already names *"native integer tensors; indexing, `gather`, `max`, or `argmax`"* — the Phase-K subject, deferred there by name. Continuity, not coincidence. |
| `native_rng_dropout_design.md` §4.5 | The platform-independence discipline Phase K reuses in §29: explicit fixed-width types, never `unsigned long`/`size_t`, no fast-math, and a result that is a function of its inputs and nothing else. |
| …§6.4 | The precedent for reporting an **unreachable** contract honestly: zero-element tensors are specified, and the note records that the representation cannot construct one. §13.7 takes the same stance rather than inventing a workaround. |
| `native_cpu_performance_design.md` H1 audit | The output-initialization audit table, its per-kernel proof obligation, and the **rejected** `narrow_backward` row with its negative control. §27.3 declines the uninitialized path for both integer destinations and therefore adds no row. |
| `native_data_pipeline_design.md` | The five-phase handoff and its rollback order, which §19 refuses to disturb. |

### 3.10 Inventories at K0

54 exported `tf_*` symbols (verified from the `TF_EXPORT` inventory in
`cpp/src/*.cpp`), 24 native CTests (24 `.cpp` test files, 24 `add_test`
registrations), 16 examples, 9 benchmarks, 25 experimental exports.

---

## 4. Terminology

These words are used in exactly these senses throughout, and the guardrails
depend on the distinctions.

- **Storage dtype** — the immutable element-type tag on a
  `tf::Storage` / `NativeStorage`. The single authority for how the bytes
  are read.
- **Tensor dtype** — the dtype a `NativeTensorCore` / `NativeTensor`
  reports, which is always its storage's dtype. There is no second copy of
  it and no way to change it.
- **Floating compute dtype** — a dtype at which arithmetic kernels run:
  `float64` and `float32`, permanently, in Phase K.
- **Differentiable dtype** — a dtype a tensor may carry while
  `requires_grad` is true. Equal to the floating compute set today, but a
  **different question**, and reported and validated separately (§5.3).
- **Index/result dtype** — a dtype that may only be produced by, consumed
  by, or inspected through operations that treat its values as positions or
  as exact integers, never as arithmetic operands: `int64`.
- **Internally representable dtype** — a dtype `_DTYPE_CODES` and
  `tf::Dtype` know how to lay out. A superset of what is publicly promised
  during a rollout.
- **Checkpoint-persistable dtype** — a dtype a native checkpoint archive
  may declare for a model, buffer, or optimizer entry: `float64` and
  `float32`, permanently, in Phase K.
- **Raw-kernel dtype** — the element type the seven handle-free raw utility
  kernels accept: `float64`, permanently. Unchanged and unrelated.
- **Index operand** — an `int64` tensor argument whose values are read as
  positions. Never an arithmetic operand (§12.4).
- **Value transfer** — a same-type assignment between two objects, which
  performs no arithmetic and reproduces the source's object representation
  exactly.

---

## 5. Dtype taxonomy

### 5.1 The decision — taxonomy **B**, stated once and not left implicit

Two coherent taxonomies were available, and exactly one is adopted.

> **A.** `SUPPORTED_DTYPES` means every public native *tensor* dtype and
> gains `int64`, while every floating-only generic constructor stops using
> it and moves to a separate floating-compute validator **before** `int64`
> joins.
>
> **B.** `SUPPORTED_DTYPES` **remains the floating-compute registry**,
> unchanged in value and in meaning, and a separate, clearly named public
> **index/result-dtype registry** is introduced when support is proved.

**Phase K adopts B.** `SUPPORTED_DTYPES` stays `("float64", "float32")`
permanently; `normalize_dtype("int64")` raises `ValueError` permanently;
and a second public row appears at K2:

| Question | Row | Value after K2 |
|---|---|---|
| At what dtypes does the runtime **compute**? | `SUPPORTED_DTYPES` | `("float64", "float32")` — unchanged |
| What **index/result** dtypes exist as native tensors? | `INDEX_DTYPES` | `("int64",)` — new at K2 |
| What do I get if I **say nothing**? | `backend_info()["dtype"]` | `"float64"` — unchanged |
| What do the seven **raw utility kernels** take? | `RAW_KERNEL_DTYPES` | `("float64",)` — unchanged |

"What dtype can a native tensor have?" is answered by the **union** of the
first two rows, and that union is stated in prose and in `backend_info()`'s
own docstring rather than materialized as a fifth tuple — a derived value
is a fifth thing that can drift.

**Why B, on inherited evidence rather than taste.**

1. `docs/native_dtype_device_metadata_design.md` §10–§11 is explicit:
   *"a label that promises a capability the kernels lack is a lie the
   runtime must not tell"*, and it recommends **rejecting** an unsupported
   dtype at construction rather than tagging a tensor with a capability the
   kernels do not have. Under A, `SUPPORTED_DTYPES` would list `int64`
   while `add`, `subtract`, `multiply`, `matmul`, `sum`, `mean`, `relu`,
   and every other kernel rejects it — precisely that lie.
2. The registry's own docstring already frames it as a **compute**
   statement: *"The native kernels compute at both float32 and float64 on
   the CPU, so these are the legal values."* B preserves that meaning; A
   would silently redefine an existing row.
3. `normalize_dtype`'s error text is *"the native runtime supports
   {SUPPORTED_DTYPES}"*, and `normalize_dtype` is the gate for `zeros`,
   `full`, and `from_array` — all of which must keep rejecting `int64`
   (§8.6). Under A that message would name a dtype the same function then
   refuses at another door.
4. **B removes the coherence problem instead of satisfying it.** Under A,
   six generic constructors would have to be re-pointed at a new validator
   *before* the registry moved, and a single missed one would open exactly
   the window this contract exists to close. Under B **not one generic
   constructor changes**, at any milestone, so there is nothing to forget
   (§5.4).
5. It keeps a Phase-I guard **permanently** true rather than expiring it:
   `tests/test_native_phase_i.py` asserts `normalize_dtype("int64")` raises,
   and under B it always will.

The one Phase-I assertion B does expire is
`set(cpp._DTYPE_CODES) == set(cpp.SUPPORTED_DTYPES)`, whose stated purpose
is that *"a representation table that could hold a dtype the registry does
not is exactly the drift this guards"*. **K2 generalizes it rather than
deleting it**, to `set(_DTYPE_CODES) == set(SUPPORTED_DTYPES) |
set(INDEX_DTYPES)` — the same guarantee (nothing representable is
unpromised) over two registries instead of one. A milestone that added the
code without that generalization would be leaving the guard weaker than it
found it, and §33 forbids it.

**Order is contractual** inside each row: `SUPPORTED_DTYPES` stays
`float64` first (the default `None` selects), then `float32`.
`INDEX_DTYPES` has one member and gains none in Phase K.

### 5.2 The private partitions

Two private tables exist and stay private. Neither is a public registry and
neither is a generic dtype framework.

- `_DTYPE_CODES` — **internally representable**. Gained `"int64": 2` at
  **K2**, in the same commit as `INDEX_DTYPES`, so the two never disagree.
  `_DTYPE_ITEM_SIZES` and `_DTYPE_NUMPY` moved with it, and
  `_CHECKED_HOST_ARRAYS` gained an entry bound to the **already existing**
  `_CHECKED_I64_ARRAY` object.
- `_normalize_index_dtype(dtype)` — the private validator measured against
  the public `INDEX_DTYPES` tuple, with `normalize_dtype`'s exact
  canonicalization, `TypeError` for a non-string, and shape of
  `ValueError`. It has **no default** and does not accept `None`: every
  other dtype validator in the module treats `None` as `"float64"`, and an
  index dtype has no such fallback to offer.
  **It is the canonical registry gate for the phase's one fixed-format
  construction door, and it has exactly one production caller**:
  `NativeTensor.from_int64_array` asks it at §26.1 step 2a — after both
  `requires_grad` checks and before the input is inspected, before
  `NativeTensorCore._from_int64_array` is entered, and before anything is
  allocated. The door names its dtype rather than accepting one, and that
  name is still measured against the public registry, so `INDEX_DTYPES`
  and the public door cannot disagree: a registry that stopped listing
  `"int64"` would close the door at the same pre-allocation step every
  other rejection uses. **No floating constructor calls it, at any layer**,
  and it is not a second public registry, not a way around
  `normalize_dtype`, and not a generic dtype framework.
- `_is_index_dtype` / `_is_tensor_dtype` / `_require_tensor_dtype` — the
  K2 predicates behind the one widened gate. `_is_tensor_dtype` is computed
  from the two registries rather than stored as a third tuple, because a
  derived value materialized once is a third thing that can drift from the
  two it derives from. `_require_tensor_dtype` is asked at
  `NativeTensorCore.__init__` and `NativeTensor.__init__` and **nowhere
  else**; every other barrier still asks `_require_floating_dtype`.

**No new floating validator is needed, and none is added.** Under B,
`normalize_dtype` *is* the floating-compute validator, so every place that
must stay floating simply calls it. That is the whole of the narrowing
work, and it is a narrowing in the literal sense: each swap replaces
`_normalize_internal_dtype` (measured against the wider representation
table) with `normalize_dtype` (measured against the floating registry), and
because the two tables accept the same set **today**, every swap is
**behavior-preserving at the moment it lands** and preventive thereafter.

### 5.3 Differentiable is a separate question from floating

The differentiable set equals the floating compute set today, and Phase K
does not merge them. They answer different questions — "may a kernel do
arithmetic at this width?" versus "may a tensor at this width carry a
gradient?" — and a phase that introduced a floating dtype too narrow to
differentiate at, or an exact dtype that could carry a gradient, would
separate them again. The runtime asks the differentiability question
through one predicate (§9.2) that reads `SUPPORTED_DTYPES` today; the
predicate exists so the answer has one implementation, not so a second
table does.

This is **not a new rule**: `docs/native_autograd_design.md` §10 already
states it — *"`requires_grad` requires a float dtype … a future
`int64`/`bool` native tensor should **not** accept `requires_grad=True`
(gradients of integers are ill-defined)"*. Phase K implements an inherited
contract rather than inventing one.

### 5.4 Exact constructor reachability — every affected path, resolved

The complete list, read off the live source rather than inferred. "Rejects
`int64`" means the path validates through `normalize_dtype`, whose
accepted set never changes.

| Path | Validator today | Phase-K resolution |
|---|---|---|
| `NativeStorage.__init__(size, dtype=…)` — public | `normalize_dtype` | **unchanged; rejects `int64` permanently.** Public storage construction stays floating-only (§5.5) |
| `NativeStorage.__init__(…, _trusted_dtype=True)` | `_normalize_internal_dtype` | **accepts `int64` since K2** — the private allocation route |
| `NativeStorage.from_array` | `normalize_dtype`, then `np.ascontiguousarray(values, dtype=…)` | **unchanged; rejects `int64` permanently.** It *casts*, so it may never be an integer ingress |
| `NativeStorage._typed` | `_normalize_internal_dtype` | **accepts `int64` since K2** (allocation only) |
| `NativeStorage._uninitialized` | `_normalize_internal_dtype` | **narrowed to `normalize_dtype` at K1** — no `int64` destination uses the uninitialized path (§27.3), and the H1 poison audit is left untouched |
| `NativeStorage._typed_from_array` | `_normalize_internal_dtype` + cast | **narrowed to `normalize_dtype` at K1** — it casts |
| `NativeStorage._from_int64_array` | — | **new at K2**, private: exact-dtype validation, `np.ascontiguousarray(values)` with **no** `dtype=` argument, allocation through the zeroed `_typed` |
| `NativeTensorCore.from_array` | `normalize_dtype` + cast | **unchanged; rejects `int64`** |
| `NativeTensorCore.zeros` (public arm) | `normalize_dtype` | **unchanged; rejects `int64`** |
| `NativeTensorCore.zeros(_trusted_dtype=True)` | `_normalize_internal_dtype` | **narrowed to `normalize_dtype` at K1** — its callers are `sum` and `narrow_backward`, both floating |
| `NativeTensorCore.full` | `normalize_dtype` → `_typed_full` | **unchanged; rejects `int64`** |
| `NativeTensorCore._typed` | `_normalize_internal_dtype` | **accepts `int64` since K2** |
| `NativeTensorCore._uninitialized` | `_normalize_internal_dtype` | **narrowed to `normalize_dtype` at K1** |
| `NativeTensorCore._typed_from_array` | `_normalize_internal_dtype` + cast | **narrowed to `normalize_dtype` at K1** |
| `NativeTensorCore._typed_full` | `_normalize_internal_dtype`, fills through `tf_storage_fill(double)` | **narrowed to `normalize_dtype` at K1** — a `double` scalar is inexact above 2⁵³ |
| `NativeTensorCore._from_int64_array` | — | **new at K2**, private |
| `NativeTensor.from_array` / `zeros` / `full` | delegate to Core | **unchanged; reject `int64`** |
| `NativeTensor._typed_from_array` / `_typed_zeros` / `_typed_full` | delegate | **inherit the K1 narrowing** |
| `NativeTensor.from_int64_array` | — | **new at K2**, public, `_normalize_index_dtype` |
| `_native_dtype.normalize_module_dtype` | `_normalize_internal_dtype` | **narrowed to `normalize_dtype` at K1** — this is what keeps `NativeParameter(data, dtype="int64")` impossible |
| `NativeParameter.__init__` | `normalize_module_dtype` + source-dtype equality | inherits the narrowing, **and** gains an explicit non-floating-source rejection at K1 |
| `native_checkpoint._validated_entry_dtype` | `_normalize_internal_dtype` | **narrowed to `normalize_dtype` at K1** — no archive can declare an `int64` entry |
| `cpp._narrowed_to_dtype` | `_normalize_internal_dtype` | **narrowed to `normalize_dtype` at K1** — Adam's bias-correction scalar |

**Nine narrowings in total, and the split is exactly seven plus two:**

- **Seven constructor/backend narrowings** — `NativeStorage._uninitialized`,
  `NativeStorage._typed_from_array`,
  `NativeTensorCore.zeros(_trusted_dtype=True)`,
  `NativeTensorCore._uninitialized`, `NativeTensorCore._typed_from_array`,
  `NativeTensorCore._typed_full`, and `cpp._narrowed_to_dtype`.
- **Two state-validation narrowings** — `_native_dtype.normalize_module_dtype`
  (the validator the state-owning constructors share) and
  `native_checkpoint._validated_entry_dtype`.

`NativeTensorCore._typed_from_array` stays in this inventory and is
narrowed in the code: it takes `dtype=` through `np.ascontiguousarray`, so
it is a **converting** ingress and may never become an integer one.
`NativeTensor._typed_from_array` / `_typed_zeros` / `_typed_full` inherit
the narrowing by delegation and are not counted a second time.

Nine narrowings, one new private table entry, two new private
constructors, one new public constructor, and **zero changes to any
generic public constructor**. Every narrowing lands at **K1**, one
milestone *before* `int64` becomes representable in Python at all, and
each is behavior-preserving on the day it lands.

### 5.5 Public `NativeStorage(size, dtype="int64")` is **prohibited**

Decided explicitly, because the question is exactly where a "single public
door" claim usually stops being true.

`NativeStorage` is public in the backend module. Its `dtype` argument
validates through `normalize_dtype`, which under B never accepts
`"int64"`, so **public storage construction at `int64` raises `ValueError`
permanently**. `int64` storage is reachable only from the private
`_trusted_dtype` / `_typed` / `_from_int64_array` family inside
`tensorforge.backends.cpp`.

That is what makes the single-door claim literal rather than approximate:
`NativeTensor.from_int64_array` is the **only** public API in the
repository through which an `int64` buffer can come into existence, and
every other public constructor rejects the dtype by name before allocating
anything.

### 5.6 Checkpoint-persistable is a separate question again

`float64` and `float32`, permanently in Phase K, validated by
`_validated_entry_dtype` against `normalize_dtype` from K1 onward (§21).
An integer tensor is not persistable because it cannot be model state
(§10), not because the archive format could not describe one.

### 5.7 What does **not** change

`RAW_KERNEL_DTYPES == ("float64",)` — a permanent limitation of the seven
handle-free raw utility kernels, which take only `double*` and an element
count and so have no dtype to dispatch on. Phase K does not touch them and
does not read the public promise off that row.

`backend_info()["dtype"] == "float64"` and `normalize_dtype(None) ==
"float64"` — the default, at every constructor, factory, module, and
parameter, unchanged. **No public constructor's omitted `dtype` ever
selects `int64`**, and `int64` is never inferred from a host array (§8.3).

`SUPPORTED_DEVICES == ("cpu",)` and `UNSUPPORTED == ("cuda", "amp")` —
untouched by every Phase-K milestone.

### 5.8 The initial and only integer dtype

**`int64`, and nothing else.** Explicitly deferred and rejected for the
whole of Phase K: `int32`, `int16`, `int8`, `uint8`, every other unsigned
integer, `bool`, every complex dtype, `float16`, and `bfloat16`. A dtype is
not enabled because adding an enumerator is easy; each would need its own
construction contract, its own exactness argument, its own promotion
refusal at every operand position, and its own proof. A later, **separately
approved** phase may revisit any of them; Phase K may not.

`bool` deserves its own sentence because it is the tempting one: a boolean
mask would make advanced indexing look close. It is refused because a
boolean tensor's only use in this design would be masking, masking is
deferred (§35), and a dtype whose only consumer is deferred is a dtype with
no contract.

---

## 6. Public object-model decision

### 6.1 The three candidates

**A. One extended `NativeTensor`.** `int64` becomes another storage dtype;
`NativeStorage` → `NativeTensorCore` → `NativeTensor` carry it unchanged;
each operation family declares which dtypes it accepts.

**B. A separate public integer/index tensor** — `NativeIndexTensor` or
`NativeIntTensor`, its own class, its own hierarchy.

**C. An internal-only integer result representation** — `argmax` returns
something that is not a public tensor at all (a host array, or a private
opaque object consumed only by `index_select`).

### 6.2 The decision

**A — one extended `NativeTensor`.**

### 6.3 Why B was rejected

A separate public class sounds narrower and is not. It would duplicate
storage ownership, view ownership, `close()` semantics and idempotence,
`__del__` fallback, host transfer, shape/stride/offset metadata, contiguity
computation, contiguous copying, failure cleanup, live-storage accounting,
the C ABI handle protocol, and the `__enter__`/`__exit__` contract — every
one of which is load-bearing, individually proved by test, and identical
for an `int64` buffer.

Duplication is not merely wasteful here; it is the specific risk this
repository has spent ten phases avoiding. Two ownership implementations are
two places for the live-storage baseline to drift, two rollback orders, two
double-close stories. The narrowness B appears to buy is **not structural**
either: nothing about a second class prevents an integer parameter unless
the parameter constructor checks the dtype — which is exactly the check A
uses.

B would also make every future dtype-general operation choose between
accepting two unrelated types or being written twice, and it would make
`index_select(source, axis, indices)` a cross-class operation with no
shared base contract.

### 6.4 Why C was rejected

An internal-only result is honest for exactly one milestone and then
becomes a lie. `argmax`'s output must be inspectable (`item()`, `tolist()`,
`to_numpy()`), must be closeable, must be usable as `index_select`'s
operand, and must have shape and rank. Anything with those properties is a
tensor; calling it something else only removes it from the contracts that
already govern tensors. C also forecloses the phase's stated purpose — an
index-valued *result* the caller can hold, inspect, and pass — for no
safety gain.

### 6.5 How A prevents each way it could go wrong

The unified model is only defensible if the boundary is enforced
structurally rather than by convention. Each row below names the authority
and the milestone that ships it. **Every one of these is a rejection before
any allocation or mutation**, and every one has a Python authority *and* an
independent C-side authority where a handle can reach the ABI.

**Every barrier lands at K1 — one milestone before an `int64` tensor can
exist in Python at all.** The milestone column below is therefore uniform,
and that uniformity is the contract: §32.1 proves there is no intermediate
state in which a barrier is missing.

| Failure the unified model could allow | Structural prevention | Milestone |
|---|---|---|
| Integer autograd | `_from_op` raises when the result core's dtype is not differentiable; `backward()` rejects a non-differentiable dtype; `_accumulate_grad` rejects; and (from K2) `from_int64_array` accepts `requires_grad` only as `False`, rejecting before allocation | K1 |
| Integer parameters | `normalize_module_dtype` narrowed to `normalize_dtype` (§5.4), so `dtype="int64"` is impossible; `NativeParameter.__init__` additionally rejects a non-floating `NativeTensor` source by dtype | K1 |
| Optimizer ownership | Both optimizers accept only `NativeParameter` (already true, hence transitively closed) **and** gain a direct per-parameter floating-dtype check at registration, so the barrier is not only transitive | K1 |
| Floating arithmetic on integer storage | Python: each operation entry validates operand dtypes against `SUPPORTED_DTYPES` before touching the Core. C: every float-only export calls `tf::require_floating` (§22.4), a **second** authority, not a restatement | K1 |
| Mixed float/integer operands | `tf::require_matching_dtype` already rejects a tag mismatch; the Python entry rejects first, naming both dtypes | K1 |
| Unsupported integer methods | Each `NativeTensor` method with no integer meaning rejects by dtype with a message naming the operation, rather than failing somewhere below | K1 |
| Generic floating constructors | `normalize_dtype` rejects `"int64"` permanently, so `NativeStorage`, `NativeStorage.from_array`, `NativeTensorCore.from_array`/`zeros`/`full`, and `NativeTensor.from_array`/`zeros`/`full` need **no change at any milestone** (§5.4) | K1 (asserted, not changed) |
| Integer model state in a module | `NativeModule.register_buffer` rejects a non-floating tensor, for **both** `persistent=True` and `persistent=False` | K1 |
| Integer model state in a checkpoint | `_validated_entry_dtype` narrowed to `normalize_dtype`, so no archive can declare an `int64` model, buffer, or optimizer entry; and `state_dict()` can only contain parameters and buffers, which the two rows above already close — three independent layers | K1 |
| A raw core or wrapper built over integer storage | `NativeTensorCore.__init__` and `NativeTensor.__init__` reject a non-floating storage/core at K1, and at K2 that gate widens to "floating **or** index, and an index tensor carries every barrier above" | K1 → widened at K2 |

`tests/test_native_phase_k.py` parses this table and asserts that every
barrier's milestone precedes the first milestone at which an integer
tensor can be constructed; the K1 and K2 guardrails will prove each row by
executable rejection with a live-storage baseline.

### 6.6 What the object model does **not** gain

No `NativeIntTensor`, no `NativeIndexTensor`, no integer subclass, no
integer mixin, no dtype-parameterized base class, no `is_integer` public
property, no `is_floating` public property, and no public dtype object.
`tensor.dtype` remains a plain canonical string.

---

## 7. Storage and ownership model

### 7.1 Storage owns the dtype; views inherit it

Unchanged from §3.2 and §3.3, and this is the whole reason A works: an
`int64` view cannot disagree with its storage, a reshape cannot reinterpret
a buffer, and a chained view chain has exactly one dtype for its whole
length.

### 7.2 The C++ side at K1

`TfDtype` gains `TF_DTYPE_INT64 = 2` and `tf::Dtype` gains `Int64`, taking
the code the existing comment already reserves for a future dtype. The
following gain an arm, each of which is a `switch` with no `default:` so a
missing arm is a compile-time diagnostic:

- `tf::dtype_from_code` — code 2 → `Dtype::Int64`;
- `tf::dtype_item_size` — `sizeof(std::int64_t)`;
- `tf::dtype_name` — `"int64"`;
- `create_storage` — dispatches into
  `create_typed_storage<std::int64_t>`;
- `destroy_storage_data` — `delete[] static_cast<std::int64_t*>(...)`.

`create_typed_storage<T>` is used **unchanged**: its existing
`static_assert(std::is_trivially_destructible<T>::value)` holds for
`std::int64_t` — an `int64` is trivially destructible, so releasing the
array is the whole of destruction — its array new-expression creates a genuine
`std::int64_t[count]` object, and its RAII ordering is dtype-independent.
`tf::dtype_checked_bytes` is likewise unchanged and now checks
`numel × 8` for the new tag.

`tf::dtype_is_float32` and `tf::dtype_is_float64` are unchanged. A new
`tf::dtype_is_int64` is added only if a call site needs it; the *rule* every
kernel uses is `tf::require_floating` (§22.4), not a per-dtype predicate.

### 7.3 The Python side at K2

**Corrected heading.** This section was written at K0 under the title *"The
Python side at K1"*, which contradicted the header, §5.2, §32.1, and the
K1 ladder row — all four of which say the Python dtype tables are
**untouched at K1**, and all four of which are what K1 actually shipped.
The work below is K2's and always was; K2 fixed the heading rather than the
four statements that agreed with each other, and records the correction
here rather than silently renumbering.

`_DTYPE_CODES["int64"] = 2`, `_DTYPE_ITEM_SIZES["int64"] = 8`,
`_DTYPE_NUMPY["int64"] = np.int64`, and a **third** entry in
`_CHECKED_HOST_ARRAYS` bound to
`np.ctypeslib.ndpointer(dtype=np.int64, flags="C_CONTIGUOUS")` — which is
the **already existing** `_CHECKED_I64_ARRAY` object, reused rather than
rebuilt, so the class-label binding and the storage binding cannot diverge
in what they accept. (K0 said "a fourth entry", counting the three
`_CHECKED_*_ARRAY` objects rather than the two mapping entries; the table
had two members before K2 and has three now.)

`NativeStorage._typed(size, "int64")` is a legal private call from K2 on,
and is the **one** allocator that can produce an `int64` destination — for
`_from_int64_array` and for the integer arm of `contiguous_copy`, and for
nothing else. The public `NativeStorage(size, dtype=...)` constructor
validates through `normalize_dtype`, which never accepts `"int64"` — so
**public storage construction at `int64` is prohibited permanently**
(§5.5).

### 7.4 Ownership, exactly as it already is

A `NativeTensorCore` owns its `NativeStorage`; a `NativeTensorView` borrows
and never closes its parent's storage; a chained view keeps the whole chain
reachable. Every Phase-K operation allocates a **fresh owning contiguous**
output that aliases neither operand. `close()` exists exactly where
something is owned and is idempotent; `__del__` is only a fallback. None of
this is restated as a new rule for `int64` because none of it is dtype-
dependent — and §28 records the one consequence that is worth stating
explicitly.

### 7.5 No allocator, pool, or cache

Phase K introduces no memory pool, scratch workspace, arena, or persistent
cache of native storage, and may not. `int64` allocations go through the
same `create_storage` every other allocation goes through.

---

## 8. Host construction and transfer contract

### 8.1 One construction door

`int64` native tensors are created by **exactly one** public route:

```python
NativeTensor.from_int64_array(values, *, requires_grad=False)
```

There is no `dtype` argument — the dtype is in the name and cannot be
omitted, mistyped, or contradicted, so no second authority exists. There is
no `int64` route through `from_array`, `zeros`, or `full`, and §8.6 records
why.

`requires_grad` is accepted as a keyword-only parameter that may only be
`False`. It exists rather than being omitted so that
`requires_grad=True` produces a message explaining that `int64` tensors are
non-differentiable, instead of Python's generic unexpected-keyword
`TypeError`. A non-`bool` raises `TypeError`; `True` raises `ValueError`;
both **before any native allocation**.

**Everything beneath it is private.** The `NativeTensorCore` layer gains
`NativeTensorCore._from_int64_array(values)` and the storage layer gains
`NativeStorage._from_int64_array(values)` — both **leading-underscore
private**, neither exported, neither a supported way around the public
validator, exactly the standing rule for the existing `_typed*` family.
`NativeStorage` and `NativeTensorCore` gain **no public integer
constructor at all**, so `NativeTensor.from_int64_array` is the one public
**construction door** in the repository through which an `int64` buffer
can come into existence.

That claim is about *construction*, and it is deliberately not a claim
about the size of the public delta: K2 also adds the dtype-general
host-inspection methods `NativeTensor.item()` and `NativeTensor.tolist()`,
which construct nothing and work at every dtype a tensor may carry
(§23.1). Three public names, one public door.

### 8.2 The strict input contract

`values` must be, in this order (§26.1 fixes the order and its
consequences):

1. **exactly `numpy.ndarray`** — `type(values) is np.ndarray`, not a
   subclass. Subclasses (masked arrays, matrices, unit-carrying arrays)
   carry semantics a plain element copy silently discards, and this is the
   first strict boundary rather than a convenience layer. `TypeError`.
2. **exactly native `int64`, in native byte order** —
   `values.dtype == np.dtype(np.int64)`. That single comparison rejects
   every wrong width, both signedness errors, `bool`, `object`, and a
   byte-swapped `>i8` array, because a dtype whose byte order is not the
   host's is not equal to the native one. It is stated as one check on
   purpose: four separate checks would be four chances to disagree.
   `TypeError`.
3. **representable rank and shape** — any rank including 0; each dimension
   non-negative; the row-major stride and element-count arithmetic checked
   in Python arbitrary-precision integers before anything is allocated.
4. **non-empty** — `values.size >= 1`. A zero-element array is rejected
   because the runtime cannot represent zero-element storage (§3.2). This
   is an inherited limitation reported honestly, not a new integer rule,
   and it is **not** worked around.
5. **checked byte count** — `numel × 8` representable, proved in Python and
   re-proved at the C ABI boundary.
6. **allocation**, then **copy**.

### 8.3 What is explicitly refused

No dtype inference, no numeric cast, no truncation, no widening, no
reinterpretation, no `astype`, and no "it was integral anyway" allowance. A
`float64` array holding `[1.0, 2.0]` is rejected. A Python list is rejected.
A tuple is rejected. An `int32` array is rejected. A `uint64` array is
rejected. A `bool` array is rejected. A `>i8` array is rejected rather than
byte-swapped.

This is deliberately stricter than the surrounding runtime's *floating*
ingress, where `from_array` has always converted host data to the requested
element type (a float64 NumPy array becoming a float32 parameter is one
rounding at ingress). The asymmetry is intentional and is the whole point:
a floating ingress conversion is a rounding whose error is bounded and
familiar; an integer ingress conversion is either a silent truncation or a
silent reinterpretation, and neither has an honest error bound. **Integer
ingress converts nothing.**

### 8.4 Layout normalization is not conversion

A non-contiguous exact-`int64` array **is** accepted and is copied into
fresh contiguous storage. Rearranging where identical values live is layout
normalization; it changes no value and no type. It is implemented by
`np.ascontiguousarray(values)` with **no** `dtype` argument, applied
strictly after check 2, so it is structurally incapable of converting.

### 8.5 Ownership after construction

The result owns fresh storage. The caller's array is **never** aliased, and
mutating it afterwards reaches nothing. Two calls with the same array give
two independent tensors. A failed copy closes the storage it allocated
(including under `BaseException`) so live storage returns exactly to
baseline, and no caller ever observes a partly written buffer.

### 8.6 Why `zeros` and `full` stay floating-only

`zeros` at `int64` would be exact — value-initialized array new gives exact
integer zeros. `full` at `int64` would **not** be: the scalar crosses the
ABI as a `double` (§3.6), so any magnitude above 2^53 is silently rounded.
Shipping an exact `zeros` beside an inexact `full` is precisely the
asymmetric front door that decays into a bug report, and making `full`
exact would need either a new export or a widened scalar contract, neither
of which Phase K's budget spends.

**One construction door, provable, small.** A caller who wants integer
zeros writes `NativeTensor.from_int64_array(np.zeros(shape, np.int64))` and
pays one host allocation, which is the same cost the floating path pays for
`from_array`.

### 8.7 Host exit

`to_numpy()` on an `int64` tensor returns a **fresh, independent,
C-contiguous** NumPy array of dtype exactly `np.int64`, in the tensor's
logical shape, materialized in logical C order for a non-contiguous view.
Nothing is widened or reinterpreted; the returned array owns its host
memory and mutating it cannot reach native storage. The existing
`NativeTensorView.to_numpy` already allocates from the storage's own NumPy
dtype, so this needs the `_DTYPE_NUMPY` entry and nothing else.

A closed tensor rejects before any transfer.

### 8.8 `item()` and `tolist()`

Neither existed before K2. Both were introduced there, and both
**dtype-general** rather than integer-only, because each has one meaning
across every dtype and two half-implementations would be worse than one:

- `item()` requires `numel == 1` at any rank; returns a built-in `int` for
  `int64` and a built-in `float` for `float32`/`float64` (the float32
  widening to a Python float is exact). Any other element count raises
  `ValueError` naming the actual count.
- `tolist()` returns nested Python lists in the tensor's logical shape,
  with exact Python `int`s for `int64` and Python `float`s for the floating
  dtypes. A rank-0 tensor returns the scalar itself, matching NumPy's
  `ndarray.tolist`.

Both are implemented on top of `to_numpy()` — no new export, no new
traversal, no second materialization path. `item()` pays one full
materialization for a one-element tensor, which is exactly the cost of the
transfer it needs.

Both reject a closed tensor before transferring.

---

## 9. Autograd and optimizer boundary

### 9.1 The rules

1. An `int64` tensor **can never** require gradients.
2. `int64` construction with `requires_grad=True` rejects **before** native
   allocation.
3. Every `int64` operation result has `requires_grad is False`.
4. An `int64` result has no parents, no backward callback, no operation
   name, and no graph history; `is_leaf` is `True`.
5. An `int64` tensor never accumulates a gradient; `.grad` is always
   `None`.
6. `backward()` on an `int64` tensor rejects clearly.
7. An `int64` tensor can never become a `NativeParameter`.
8. An `int64` tensor can never be owned by `NativeSGD` or `NativeAdam`.
9. Optimizer state is floating only.
10. An `int64` tensor can never enter a mixed parameter collection.
11. An `int64` tensor **may** be retained as index metadata for a floating
    backward, and only under an explicit operation contract that names it.
12. A floating result selected by integer indices **may** be differentiable
    with respect to the floating source — in a separately approved future
    phase (§18.9); Phase K ships the forward only.
13. Integer index operands **never** receive gradients, under any
    operation, ever.

### 9.2 Where each rejection lives, and which authority is first

"Integers are non-differentiable" is not a contract. These are.

| Path | First authority | Error | Ordering guarantee |
|---|---|---|---|
| `from_int64_array(..., requires_grad=True)` | the constructor's own argument validation | `ValueError` | before dtype canonicalization, before any allocation |
| `from_int64_array(..., requires_grad=1)` | the same validation | `TypeError` | before allocation |
| `from_array(x, dtype="int64")` | `normalize_dtype` inside `from_array` | `ValueError` naming `from_int64_array` | before any allocation |
| `zeros(shape, dtype="int64")` / `full(...)` | `normalize_dtype` | `ValueError` | before any allocation |
| an op building a graph over an `int64` core | `_from_op`, which raises when the result core's dtype is not differentiable | `ValueError` | after the core exists, so it closes the core it was handed |
| `int64_tensor.backward()` | `backward()`'s own dtype guard | `RuntimeError` | before the graph is walked, before any gradient is allocated |
| `int64_tensor._accumulate_grad(g)` | the accumulator's dtype guard | `RuntimeError` | before allocation |
| `NativeParameter(int64_tensor)` | `normalize_module_dtype` (dtype argument) and the source-dtype comparison | `ValueError` | before any copy is allocated |
| `NativeSGD([...int64...])` / `NativeAdam` | the registration type check, then the per-parameter floating check | `TypeError` / `ValueError` | before any moment buffer is allocated |
| `module.register_buffer("b", int64_tensor)` | `register_buffer`'s dtype guard | `ValueError` | before registration |

The `_from_op` row is the structural backstop that makes the unified object
model safe against **future** operations: no operation written after K1 can
accidentally produce a differentiable integer result, because the single
graph-construction entry refuses to build one. It is deliberately a raise
rather than an assertion, and it closes the core it was given so a rejected
graph leaks nothing.

### 9.3 Index metadata retained for a floating backward

Rule 11 is the one place an integer tensor may be reachable from a graph,
and it is fenced:

- an integer tensor may be **saved** by a floating operation's forward as
  private graph-owned state, riding the existing `graph_resources` contract
  unchanged — released exactly once with the graph history, retained under
  `retain_graph=True`, kept alive across a failed retryable backward, freed
  by an abandoned graph's `close()`, closed immediately by a no-grad
  forward;
- it is **never** a parent, never in `_parents`, and never receives a
  gradient contribution;
- the operation that saves it must name it in its own contract.

**No Phase-K milestone exercises rule 11**, because Phase K's only
index-consuming operation ships forward-only (§18.9). The rule is stated
now so that a future differentiable selection inherits a decided contract
instead of inventing one.

### 9.4 Optimizer state

Neither optimizer has or gains a `dtype` or `device` argument: they own no
dtype to choose, only state that must match a parameter. `NativeAdam`'s
moment buffers are allocated at each parameter's own dtype, and since a
parameter is always floating, so is every moment. The in-memory optimizer
state format stays version **1** for the whole of Phase K.

---

## 10. Module and persistent-state boundary

### 10.1 The decisions

| Module state category | Phase-K position |
|---|---|
| Parameters | **Prohibited** for `int64`, permanently in Phase K |
| Optimizer state | **Prohibited** for `int64`, permanently in Phase K |
| Persistent buffers | **Prohibited** for `int64` in Phase K; deferred |
| Non-persistent buffers | **Prohibited** for `int64` in Phase K; deferred |
| Registered generators | Unrelated — a generator holds no tensor |
| Checkpoint-persisted integer state | **Prohibited**; deferred (§21) |

### 10.2 Why buffers are prohibited rather than merely undesigned

A persistent buffer is, by contract, serialized into `state_dict()` and
into a native checkpoint. Allowing an `int64` persistent buffer would
therefore require a checkpoint decision (§21) in the milestone that allowed
it, and Phase K's checkpoint position is "no change". A **non**-persistent
integer buffer would avoid the archive but would still make an integer
tensor part of a module's registered state, reachable by `buffers()`, and
subject to `load_state_dict`'s transactional replacement — three contracts
that would each need an integer proof for no capability the phase needs.

`register_buffer` therefore gains an explicit floating check at K1, for
**both** `persistent=True` and `persistent=False`.
Prohibition-with-a-check is stronger than prohibition-by-omission: it makes
the boundary a rejection a test can drive, rather than an accident waiting
for the first caller.

### 10.3 The unified model does not make integer checkpoint state reachable

Three independent layers, any one of which suffices (§6.5): a parameter
cannot be `int64`; a buffer cannot be `int64`; and a version-3 archive
entry cannot declare `int64`. They are listed as three because a single
layer is a single point of failure, and because each has a different first
authority and a different error.

---

## 11. Operation-family scope

The complete Phase-K integer capability matrix. **Supported** means a
milestone in §32 ships it; **absent** means no Phase-K milestone ships it
and §35 records it as a non-goal or a deferral.

### 11.1 Storage and metadata — supported

`dtype` identity, `shape`, `strides`, `ndim`, `numel`, `contiguous`,
`offset`, ownership, `closed`, `owns_core`, `close()`, `__enter__` /
`__exit__`, `__repr__` (metadata only, valid after close, never a value),
and host transfer. All inherited unchanged; none is dtype-dependent.

### 11.2 Views and copies — supported

| Operation | Result | dtype | Storage | Close | Differentiable |
|---|---|---|---|---|---|
| `reshape` | borrowing view | preserved | shared with the source | source owns; view does not close it | no |
| `transpose` | borrowing view | preserved | shared | as above | no |
| `T` | borrowing view | preserved | shared | as above | no |
| `narrow` | borrowing view | preserved | shared | as above | no |
| `contiguous_copy` | **owning** tensor | preserved | fresh | caller closes | no |

Every one preserves the source dtype exactly — a view cannot cast (§7.1) —
and every one is non-differentiable at `int64` because the source is. A
failure at any point closes whatever it allocated and leaves live storage
at baseline.

`narrow` **stays metadata-only and view-producing** and is never turned
into a copying operation (§18.1).

### 11.3 Inspection — supported

`to_numpy()`, `item()`, `tolist()` (§8.7, §8.8), and the exact integer
equality tests are built on. Tests compare integers with `==` and never
with a tolerance (§29.6).

### 11.4 Numeric integer operations — absent, by decision

Phase K adds **no** integer addition, subtraction, multiplication,
division, negation, absolute value, modulo, integer reduction (`sum`,
`mean`, `min`, `max`, `prod`), integer comparison as a general runtime
operation, integer matrix multiplication, integer convolution, integer
pooling, integer normalization, integer softmax, integer dropout, or
integer optimizer math.

Every existing arithmetic entry point rejects an `int64` operand by dtype,
in Python before the Core is touched and again in C before anything is
written (§6.5, §22.4).

**Overflow is therefore outside the implemented surface**, and Phase K does
not invent a wrapping rule or a checked-arithmetic rule that no operation
could reach. The two places integers are produced — `argmax`'s indices and
`index_select`'s validated positions — are bounded by the shapes they came
from and cannot overflow. Stating an overflow policy for arithmetic that
does not exist would be a promise about behavior no test could drive.

### 11.5 Index-producing operations

- `argmax` — **supported** (K3, §17).
- `argmin`, `nonzero`, `sort`/`argsort`, `top-k`, `unique`, `where`,
  `searchsorted`, `bincount`, `cumsum` — **absent**, deferred (§35).

`argmin` is deferred despite being one negation away, because it is a
public operation with its own tie, NaN, and validation contract, and
"it is nearly free to implement" is not a reason to add a public promise.

### 11.6 Index-consuming operations

- `index_select` — **supported, forward only** (K4, §18).
- Advanced indexing, `__getitem__` / `__setitem__` with tensors, boolean
  masks, arbitrary `take`, `gather` along multiple axes, `scatter` /
  `scatter_add` as public operations, and `embedding` as a module —
  **absent**, deferred (§35).

### 11.7 Naming

No document, comment, test, or message uses the phrase "basic integer
operations" or any equivalent hand-wave. Every operation is named, and
every absence is named.

---

## 12. Casting and promotion rejection

### 12.1 The inherited rule, unchanged

**No casting, no promotion, no mixed-dtype arithmetic.** A mismatch raises
before any allocation or mutation. There is no `astype`, no `to`, no
`.float()`, no `.double()`, no `.int()`, no `.long()`, no `map_location`,
and no global default dtype. Phase K adds none of these and may not.

### 12.2 The explicit rejection list

Each of the following raises, with the Python entry as first authority and
the C ABI as an independent second where a handle can reach it:

- floating tensor `+` integer tensor;
- integer tensor `+` floating tensor;
- integer tensor `*` floating tensor;
- floating `matmul` integer, and integer `matmul` floating;
- an `int32` NumPy array silently becoming `int64`;
- a `bool` array or a Python `bool` treated as 0 / 1;
- floating class targets truncated to integers;
- a Python `int`, a Python list, or a Python tuple silently creating a
  native tensor at any boundary that has not explicitly contracted for it;
- an integer result silently becoming a floating tensor, or the reverse;
- a checkpoint array reinterpreted across dtypes in either direction;
- any automatic stable ↔ native conversion.

### 12.3 What is *not* a cast

Two ingress behaviors are deliberately not casts and must not be described
as ones:

- **Layout normalization** — copying a non-contiguous exact-`int64` host
  array into contiguous storage (§8.4). No value and no type changes.
- **Floating host ingress** — `from_array` converting host data to the
  requested floating element type, which is the pre-existing documented
  behavior of the host→native boundary. Phase K does not extend it to
  `int64` (§8.3).

### 12.4 An index operand is not an arithmetic operand

`index_select` takes a floating source and an `int64` index tensor in one
call. That is **not** a mixed-dtype arithmetic operation, and the
distinction is enforced rather than asserted: the operation validates each
operand against the dtype **its role requires** — source floating,
destination floating and matching the source, index exactly `int64` — and
`tf::require_matching_dtype` is applied to the source/destination pair
only, never across the role boundary. A floating "index" tensor is rejected
by the index role check; an `int64` "source" is rejected by the source role
check. Neither is promoted, and neither is inferred from the other.

---

## 13. Exact integer semantics

### 13.1 Representation

Values are **signed 64-bit two's-complement integers**, exactly. The C++
side spells the type `std::int64_t`, which the standard requires to be
exact-width, two's-complement, and free of padding bits. A
`static_assert(sizeof(std::int64_t) == 8)` and a
`static_assert(static_cast<std::int64_t>(-1) == ~static_cast<std::int64_t>(0))`
are placed beside the existing floating `static_assert`s in
`cpp/include/tf_internal.h`, so a toolchain where the assumption fails is a
build error rather than a wrong result. The Python side asserts
`np.dtype(np.int64).itemsize == 8`.

The representable range is `[-(2**63), 2**63 - 1]`. `cpp._INT64_MIN` and
`cpp._INT64_MAX` already exist and are reused; no second bound is
introduced.

### 13.2 Element addressing

Shapes, strides, and offsets stay **logical element** counts at `int64`
exactly as at every other dtype. Bytes appear only at the allocation
boundary, through the one checked `numel × itemsize` conversion.

### 13.3 Negative values as stored values

A general `int64` tensor **may hold negative values**. `from_int64_array`
accepts them, `to_numpy`, `item`, and `tolist` return them exactly, views
and copies preserve them, and no operation clamps or normalizes them.
`int64` is a general exact integer dtype, not a "non-negative index" dtype.

### 13.4 Negative values *interpreted as indices*

Where an `int64` tensor is used as an **index operand**, a negative value
is **rejected**, not wrapped (§14.2). The two rules coexist without
tension: §13.3 is about what a tensor may contain, §13.4 about what one
operation accepts.

### 13.5 Duplicates and ordering

`index_select` preserves duplicate indices and input order **exactly**: the
output's *i*-th slice along the selected axis is the source slice at
`indices[i]`, for every *i*, with no deduplication, no sorting, and no
reordering.

### 13.6 Scalar and multidimensional index tensors

An index operand must be **rank exactly 1** (§14.3). A rank-0 index tensor
is rejected because the output rank would be ambiguous — does the selected
axis vanish or become size 1? — and a caller wanting a single position uses
`narrow(axis, i, 1)`, which already exists and is metadata-only. A rank ≥ 2
index tensor is rejected because it implies an output-shape rule Phase K
has not contracted.

### 13.7 Zero-sized dimensions and empty tensors

The runtime cannot represent zero-element storage (§3.2), so a zero-element
`int64` tensor cannot be constructed and an empty index tensor is a
permanent non-case rather than a special rule. `argmax` over an empty input
is likewise unreachable. Phase K reports this as an inherited limitation
and does **not** work around it, add an empty-tensor representation, or
special-case it.

### 13.8 Zero-dimensional inputs

Rank 0 is fully supported (§3.3). `argmax(axis=None)` over a rank-0 input
returns a rank-0 `int64` tensor holding `0`; `argmax` with an explicit axis
over a rank-0 input is out of range and rejected (§17.6).

### 13.9 Very large values

Any value in `[-(2**63), 2**63 - 1]` round-trips exactly through
construction, storage, views, `contiguous_copy`, `to_numpy`, `item`, and
`tolist`. A value outside the range cannot exist in an `int64` NumPy array
and therefore cannot reach construction. As an index it would be rejected
by the bounds scan (§14.4) long before it was used.

### 13.10 Deterministic traversal

Every Phase-K kernel traverses in a fixed logical order that is a function
of the shape metadata alone — never of a pointer value, an alignment, a
clock, an environment variable, or a CPU-feature probe. The results are
therefore identical on every platform, which §29.5 states as a requirement
rather than a hope.

### 13.11 Allocation before or after validation

**Every validation that can be performed without the destination is
performed before the destination is allocated.** This is stated once here
and applied at every boundary: input types, closed state, dtypes, ranks,
axis ranges, shape representability, and — the one that matters most —
**the complete index bounds scan** (§14.4). Allocation is the last step
before the kernel call.

---

## 14. Index validation

### 14.1 Where it happens

An index operand is validated in Python, in full, before the Core layer is
entered; and the C ABI export re-validates independently, because the C ABI
is a **second** authority and never a restatement of Python's. Neither
check may be removed because the other exists.

### 14.2 Negative indices reject

`index_select` rejects any negative index. It does **not** normalize
`-1` to "the last position".

The reasoning, recorded because the opposite convention is widespread: an
index tensor is *data*, and in this phase it is usually **computed** — the
whole point of `argmax` is to produce one. A negative value in computed
index data is a defect somewhere upstream. Wrapping it silently converts
that defect into a plausible, wrong answer at the far end of a training
run; rejecting it converts the same defect into an exception at the point
of use. TensorForge already takes this stance elsewhere: `narrow(dim,
start, length)` requires a non-negative start.

`argmax`'s output is always non-negative, so the two Phase-K operations
compose without a caller ever meeting this rule by accident.

### 14.3 Rank

Rank exactly 1 (§13.6).

### 14.4 The bounds scan

Every index is checked against `[0, axis_length)` **before the destination
is allocated**, and the scan is complete rather than incremental. A single
out-of-range value rejects the whole call, names the offending value and
the position it was found at, and leaves live storage exactly at baseline.

This is a deliberate strengthening over the obvious implementation, which
checks each index as it copies and therefore leaves a partly written
destination behind when it throws. Validating first costs one extra pass
over a rank-1 tensor and buys a rejection that writes nothing — which is
the rule the C ABI error contract already states for every self-validating
export.

### 14.5 Non-contiguous index views

Accepted. A non-contiguous or offset `int64` view is materialized
contiguously by the Core layer before the export is called (§18.5), so the
scan and the kernel both see one contiguous run. The *values* and their
order are identical either way, because `contiguous_copy` reproduces the
logical order exactly.

### 14.6 Closed operands

A closed source or a closed index tensor rejects before anything else about
the call is examined beyond argument types (§26.3).

---

## 15. View versus copy semantics

### 15.1 The rule

An operation is a **view** if and only if it produces no new storage. The
five view operations are listed in §11.2 and are metadata-only at every
dtype. Everything else — `contiguous_copy`, `argmax`, `index_select` —
allocates a fresh owning contiguous output.

### 15.2 A copying selection is never called a view

`index_select` allocates and copies. It is never described as a view, never
returns a borrowing wrapper, and never shares storage with its source. Its
result is the caller's, and the caller closes it.

### 15.3 `narrow` is not turned into a copy

`narrow` remains metadata-only and view-producing. Phase K does not extend
it to accept an index tensor, does not add a copying mode to it, and does
not deprecate it. `narrow` answers "one contiguous run along one axis";
`index_select` answers "these positions along one axis". They are different
questions with different costs, and conflating them would hide an
allocation behind an operation callers currently know is free.

### 15.4 Closing a source with a live view

Unchanged: a view borrows and never closes its parent's storage, a chained
view keeps the whole chain reachable, and operating on a view whose source
has been closed raises rather than reading released memory.

---

## 16. Scalar and host inspection

`to_numpy()`, `item()`, and `tolist()` are the three host-inspection
boundaries, contracted in §8.7 and §8.8: `item()` returns a built-in `int`
for a one-element `int64` tensor, `tolist()` returns exact Python integers
in the tensor's logical shape, and `to_numpy()` returns a fresh independent
`np.int64` array. Two additional points belong here.

### 16.1 `item()` on the phase's own results

`argmax(axis=None).item()` is the intended idiom for reading a single
predicted index, and it returns a built-in `int` — not a NumPy scalar, not
a 0-d array, not a float. Tests compare it with `==` against a Python
integer.

### 16.2 What inspection does not do

Inspection builds no graph, touches no gradient, no parameter, and no
version counter, allocates no native output, and retains nothing after
returning. A graph built before an inspection call is fully usable after
it. This is the same stance `native_accuracy` already takes, stated for the
new methods rather than assumed.

---

## 17. `argmax` contract

### 17.1 Signature and spelling

```python
NativeTensor.argmax(axis=None, keepdims=False)      # K3
NativeTensorCore.argmax(axis=None, keepdims=False)  # K3
```

**`axis` and `keepdims`**, matching this repository's established spelling
(`sum(axis=None, keepdims=False)`, `mean`, `softmax(axis=-1)`,
`log_softmax(axis=-1)`, `cpp.reduce_shape(shape, axis, keepdims)`). The
alternative spelling used by some other frameworks is not adopted; one
repository, one vocabulary.

### 17.2 Input

Floating only — `float64` or `float32` — open, and non-empty. An `int64`
input is rejected: `argmax` is a **floating** reduction that *produces* an
index, not an integer operation.

### 17.3 Output

A **fresh owning contiguous `int64` tensor**, always, at every input dtype.
The output dtype does not depend on the input dtype, which is the point.

Shapes, derived through the existing `cpp.reduce_shape` so there is one
shape authority:

| Call | Input shape | Output shape |
|---|---|---|
| `axis=None, keepdims=False` | any | `()` |
| `axis=None, keepdims=True` | rank *n* | `(1,) * n` |
| `axis=a, keepdims=False` | `(d0 … dn)` | `(d0 … dn)` with `da` removed |
| `axis=a, keepdims=True` | `(d0 … dn)` | `(d0 … dn)` with `da` replaced by 1 |

Negative `axis` is normalized by the existing `cpp._normalize_axis_checked`
before the shape is derived.

### 17.4 The index a result holds

- With `axis=None`, the **logical flat index** in row-major (C) order over
  the whole tensor — the same order `to_numpy()` produces.
- With an explicit `axis`, the position **along that axis**, in
  `[0, shape[axis])`.

### 17.5 Value semantics — the exact rule

No adjective carries any weight here. The rule is one algorithm, and every
exceptional case is an answer this algorithm produces rather than a
sentence about it.

**The run.** For `axis=None` the run is every element of the tensor in
**increasing row-major logical order** — the same order `to_numpy()`
produces — indexed `0 … numel-1`. For an explicit `axis`, each output
position has its own run: the elements along `axis` in **increasing
axis-index order**, indexed `0 … shape[axis]-1`. In both cases the returned
value is the index **within the run**.

**The algorithm**, applied to one run, left to right:

```
best_index = 0
best       = run[0]
for i in 1 .. len(run) - 1:
    v = run[i]
    if isnan(best):
        continue                      # nothing displaces an incumbent NaN
    if isnan(v) or v > best:          # a NaN displaces a non-NaN; else strict >
        best       = v
        best_index = i
return best_index
```

It is total, allocation-free, reads each element exactly once, and never
inspects a NaN's payload, its signalling bit, or its sign.

**The exact answer, case by case.** Let *i*₁ be the lowest index holding a
NaN, where one exists.

| Case | Returned index |
|---|---|
| No NaN, unique maximum | the index of that maximum |
| No NaN, several equal maxima | the **lowest** such index (strict `>` never displaces an equal) |
| No NaN, both signed zeros (`+0.0` and `-0.0`) among the maxima | the **lowest** index of either — IEEE comparison does not order the signed zeros, so they tie |
| No NaN, every element `-inf` | `0` — all tie |
| No NaN, `+inf` present | the lowest index holding `+inf` |
| **Exactly one NaN**, any other values | *i*₁ |
| **Several NaNs**, any other values | *i*₁ — the **first**; a later NaN cannot displace it, because `v > NaN` is false and the second clause requires the incumbent not to be NaN |
| NaN mixed with finite values | *i*₁, whatever the finite values are |
| NaN mixed with `+inf` | *i*₁ — `+inf` never displaces a NaN |
| NaN mixed with `-inf` | *i*₁ |
| NaN at index 0 | `0`, and the loop displaces nothing thereafter |
| Run of length 1 | `0`, NaN or not |

**Full reduction versus axis reduction.** The algorithm is identical; only
the run differs. For `axis=None` the answer is a flat row-major index in
`[0, numel)`; for an explicit `axis` it is a position in
`[0, shape[axis])`, computed independently for each output position, and
NaNs in one run never affect another.

**Contiguous versus non-contiguous input.** The answers are **identical**,
and this is a consequence rather than a coincidence: a non-contiguous input
is materialized contiguously first (§17.8), and `contiguous_copy`
reproduces logical order exactly, so the kernel sees the same sequence of
values in the same order either way. Traversal order is therefore a
function of shape metadata alone (§13.10) and never of strides, offsets,
addresses, or allocation history.

**Why this NaN rule, rather than skipping NaN.** A NaN in a logit array
means the forward pass is broken. A rule that *skips* NaN returns a
perfectly plausible index and hides the break until something much later
goes wrong; this rule makes the break visible at the first place a caller
looks.

**This is TensorForge's normative rule. K0 makes no compatibility claim
with NumPy or another framework. K3 tests every case directly.**

**Initialization is load-bearing.** `best` starts at `run[0]`, never at a
sentinel such as the type's lowest representable value. A sentinel start
makes an all-`-inf` run and an all-NaN run return `0` *by accident*; this
start makes them return `0` *by construction*, and the distinction is the
difference between a rule and a coincidence.

`argmax` performs no arithmetic, so there is nothing to reassociate and no
accumulation-order contract to state. Its results are bit-exact in the only
sense that applies to an integer: they are equal.

### 17.6 Validation order

1. `self` is an open `NativeTensor` (`RuntimeError` if closed);
2. input dtype is floating (`ValueError`);
3. input is non-empty — a permanent non-case (§13.7), asserted rather than
   reachable;
4. `axis` is `None` or an exact `int`; `bool` is rejected explicitly
   (`TypeError`);
5. `axis` is in `[-ndim, ndim)` after normalization (`ValueError`); a
   rank-0 input with **any** explicit axis is out of range;
6. `keepdims` is exactly `bool` (`TypeError`);
7. the output shape and its element count are representable
   (`ValueError`);
8. the destination is allocated;
9. the kernel runs.

A caller passing several invalid arguments receives the **first** error in
this order, deterministically.

### 17.7 Ownership and failure

The output is the caller's and **the caller closes it**. If allocation
fails, nothing is published and live storage returns to baseline. If the
kernel fails, the destination this call allocated is closed before the
exception propagates, including under `BaseException`. Operating on the
input after it is closed rejects; closing the input after the output exists
affects the output not at all, because the output owns fresh storage.

### 17.8 Non-contiguous input

Accepted, through **Policy-B copy-then-compute at the Core layer** — the
same pattern `softmax`, `log_softmax`, cross-entropy, Conv2d, MaxPool2d,
and Dropout already use. The export is contiguous-only and takes the
`(outer, axis_length, inner)` decomposition of the reduction axis; a
non-contiguous input is materialized contiguously first, and the temporary
is closed explicitly.

This is not a semantic compromise: `contiguous_copy` reproduces logical
order exactly, so the traversal order, the tie rule, and the NaN rule are
identical whether the input arrived contiguous or not — which is precisely
what makes the copy legitimate.

### 17.9 No graph, ever

The result has `requires_grad is False`, no parents, no backward, no `op`
name, and `is_leaf is True`, **even when the input requires grad**. This is
the one place `argmax` differs from every other operation on a
gradient-tracking tensor, and it is correct: the derivative of an index
with respect to a value does not exist. `"argmax"` joins `TENSOR_CORE_OPS`
and **does not** join `AUTOGRAD_OPS`, and the K3 guardrails assert both
halves.

### 17.10 `max` is not shipped

A kernel searching for the position of a maximum necessarily knows the
maximum. Phase K **does not** expose it. There is no `max`, no
`max_with_indices`, no tuple return, and no second output handle in the
export signature.

Reasons: a differentiable `max` is a genuinely different operation with a
graph node, a saved-index backward, and its own gradient contract; a
non-differentiable `max` would be a trap next to a `sum`/`mean` that are
differentiable; and "one implementation computes both" is an implementation
fact, not an API argument. A separately approved future phase may add
`max`; Phase K may not.

---

## 18. Gather / index-select contract

### 18.1 What this is not

Five distinct things are often conflated, and Phase K keeps them apart:

| Name | What it does | Phase-K position |
|---|---|---|
| `narrow` | metadata-only view of one contiguous run along one axis | exists, unchanged |
| `index_select` | **copies** the positions an index tensor names, along one axis | K4, forward only |
| general `gather` | element-wise, index tensor the shape of the output, along one axis | absent |
| embedding lookup | `index_select` on axis 0 of a weight, plus a scatter-add backward | absent, deferred |
| advanced indexing | arbitrary tensor/boolean/multi-axis subscripting | absent, deferred |

### 18.2 Signature

```python
NativeTensor.index_select(axis, indices)      # K4
NativeTensorCore.index_select(axis, indices)  # K4
```

`indices` is an `int64` **native tensor**, not a host array, not a list,
and not a Python `int`. A caller who has host indices constructs one
explicitly through `from_int64_array` — one visible conversion instead of a
hidden one, and the operation's own name never has to describe a second
input form.

### 18.3 Operands and roles

- **source** — floating (`float64` or `float32`), open, any rank ≥ 1, any
  layout;
- **axis** — exact `int`, `bool` rejected, negative normalized, in
  `[-ndim, ndim)`;
- **indices** — `int64`, open, **rank exactly 1**, every value in
  `[0, source.shape[axis])`, negatives rejected, duplicates allowed.

Each operand is validated against the dtype its **role** requires (§12.4).

### 18.4 Output

Shape = the source shape with `axis` replaced by `indices.numel` — one
axis, and exactly one dimension of the source changes size. dtype = the
**source's** dtype, preserved exactly. A **fresh owning contiguous**
floating tensor: it allocates and copies, so it is **never a view** and is
never called one, and it never shares storage with either operand. The
caller closes it.

Duplicates are preserved and order is preserved (§13.5).

### 18.5 Layout

The export is contiguous-only and takes the
`(outer, axis_length, index_count, inner)` decomposition; a non-contiguous
source or index view is materialized contiguously by the Core layer first,
through Policy-B copy-then-compute, and each temporary is closed
explicitly. Accepting non-contiguous operands matters: `transpose(...)`
results are ordinary, and rejecting them would make the operation unusable
after the most common view op.

### 18.6 Validation order

1. `axis` is an exact `int` (`bool` rejected) — `TypeError`;
2. `indices` is a `NativeTensor` — `TypeError`;
3. `self` is open — `RuntimeError`;
4. `indices` is open — `RuntimeError`;
5. source dtype is floating — `ValueError`;
6. index dtype is exactly `int64` — `ValueError`;
7. source does not require grad — `ValueError` naming `detach()` (§18.9);
8. `axis` is in range after normalization — `ValueError`;
9. `indices` rank is exactly 1 — `ValueError`;
10. the **complete** index bounds scan — `ValueError` naming the value and
    its position (§14.4);
11. output shape and element count representable — `ValueError`;
12. allocate the destination;
13. copy.

Steps 1–11 allocate nothing that survives. Step 12 is the first step that
can leave anything behind, and step 13 is the only step that writes.

### 18.7 Failure

Any failure at or after step 12 closes what it allocated — including under
`BaseException` — so live storage returns exactly to baseline and no caller
observes a partly written destination. A rejection at steps 1–11 leaves the
observable world byte-identical: both operands' storage, their metadata,
their graph state, and the live-storage count.

### 18.8 The result carries no graph

`requires_grad is False`, no parents, no backward, `is_leaf is True`.
`"index_select"` joins `TENSOR_CORE_OPS` and **does not** join
`AUTOGRAD_OPS`.

### 18.9 The gradient is deferred, and the source may not require grad

**Phase K ships `index_select` forward only.** A source with
`requires_grad=True` is **rejected**, with a message naming `detach()`.

Both halves of that are decisions:

*Why the backward is deferred.* It is a scatter-add into a zeroed
destination — for each *i*, add the *i*-th output slice into source
position `indices[i]`, accumulating where indices repeat. That is a real
numerical contract of its own: an accumulation order, a duplicate-index
rule, a zero-initialized destination that must stay zero-initialized
(exactly the `sum`/`narrow_backward` audit rows), and its own C ABI export.
No existing export expresses it — `tf_core_narrow_backward` scatters one
contiguous run, and composing it per index would allocate one full-size
temporary per index. Phase K's ABI budget deliberately does not spend a
third symbol (§22.3), and a gradient contract deserves its own milestone
and its own proof rather than a corner of one.

*Why rejecting beats silently detaching.* Returning a graph-free result
from a gradient-tracking source would be a silent gradient hole: the
forward would work, the loss would train, and one path's gradients would
simply be missing. This repository refuses that class of behavior
everywhere. A caller who genuinely wants a detached selection writes
`source.detach().index_select(...)` and says so.

When a separately approved phase adds the backward, the contract it must
meet is already fixed here: it accumulates **deterministically** into the
floating source in increasing index order, duplicate indices accumulate
rather than overwrite, the destination is zero-initialized and stays so,
the index tensor is retained as private graph-owned metadata under rule 11
(§9.3), and **the index tensor never receives a gradient**.

### 18.10 Why `index_select` rather than `gather`

`index_select` selects whole slices along one axis with a rank-1 index
tensor. `gather` selects element-wise with an index tensor shaped like the
output. The first has one obvious output shape, one bounds rule, one
traversal, and composes directly with `argmax`'s rank-1 output; the second
needs an index-shape-versus-source-shape compatibility contract Phase K has
no use for. One primitive, chosen for being the one the phase's other
operation produces input for.

---

## 19. Data-pipeline compatibility

### 19.1 The Phase-J default is unchanged

`NativeDataLoader` continues to deliver:

```
(NativeTensor floating features, numpy.int64 targets)
```

**No Phase-K milestone modifies Phase-J production code**, changes the
delivery contract, adds a loader or dataset option, or changes what
`__next__` returns. `native_dataset.py`, `native_sampler.py`,
`native_data_loader.py`, and `_native_permutation.py` are untouched for the
whole phase.

**K5 proved this against the live tree** rather than leaving it as a
promise: the delivered pair is a floating `NativeTensor` and a read-only
host `numpy.ndarray` of dtype `int64` at both feature widths, the ordering
and batching are still the sampler's plan, and no parameter on the dataset,
the sampler, the loader, or the loader's private iterator could request a
native label — `__next__` takes `self` alone, so there is nowhere for a
delivery option to sit. None of the four modules' executable code names
the integer door, `argmax`, or `index_select`.

### 19.2 Options assessed and not approved

| Option | Assessment |
|---|---|
| `NativeTensorDataset.native_target_batch(indices)` | Feasible once §8 exists. Adds a second closeable object per call and a second failure position to the dataset's contract. **Not approved in Phase K.** |
| Explicit caller conversion (`from_int64_array(loader_targets)`) | Needs nothing new after K2, costs one copy, and keeps every ownership contract where it is. **This is the supported route**, and it needs no milestone — **K5 proved it end to end** on a real delivered batch, including that the resulting tensor is still refused by every state-owning surface. |
| A separate adapter object | A third object in a three-object pipeline whose whole design is "three objects, one direction". Rejected. |
| A loader option defaulting to false | Rejected in Phase K — see §19.3. |

### 19.3 What any future native-target integration must define

Recorded now so a future phase inherits requirements rather than inventing
them: default-off behavior; who owns the delivered target tensor; who
closes it; the rollback set the five-phase handoff must extend to; how the
committed position's "advance if and only if delivered" invariant survives
a second allocation; checkpoint-metadata compatibility; the unchanged
host-target path beside it; and the absolute rule that **no existing
caller's output type changes silently**.

### 19.4 Why it is deferred

The batch handoff is the most carefully proved transaction in the
repository: five phases, one absolute invariant, a fixed rollback order
(restore position → clear the record → close the tensor), four distinct
injected failure positions, and a hardening matrix that fingerprints the
entire observable world after each. Adding a second owned object to that
transaction changes the rollback set and every one of those proofs.

Phase K's integer construction, lifecycle, and failure behavior should be
**stable and proved first**. Sequencing it the other way would mean
proving the hardest transaction in the project against a dtype whose own
failure behavior had not yet been demonstrated.

---

## 20. Classification and metric integration

### 20.1 Assessed separately

| Candidate | Phase-K position |
|---|---|
| `argmax` prediction indices | **Shipped at K3**, and shipped as a general reduction rather than as a classification feature |
| A native accuracy computation | **Absent.** Needs an integer equality reduction, which §11.4 refuses |
| Native `int64` targets for `NativeCrossEntropyLoss` | **Absent.** The host-target contract is unchanged for the whole phase |
| Gathering class logits by index | **Shipped at K4** as `index_select`, not as a classification feature |
| Confusion matrices | **Absent**, deferred |
| Embedding indices | **Absent**, deferred |

### 20.2 `NativeCrossEntropyLoss` is unchanged

It continues to accept its current strict **host** target contract, through
`cpp._prepare_class_targets`, for the whole of Phase K. A separately
assigned future milestone in a separately approved phase may change that;
Phase K may not, and no Phase-K milestone touches
`native_cross_entropy_loss.py` or the cross-entropy Core path.

**K5 proved that behaviorally** rather than by inspection alone: the
accepted host forms are unchanged and every one of them produces a
*bit-identical* loss, the rejected forms are unchanged, values and
gradients are correct at each width against a host oracle, repeat calls are
bit-identical, and a native `int64` target is refused by three separate
routes — the explicit construction door, a fresh `argmax` result, and a
view of one — each at the same shared host boundary rather than at a second
rule inside the module.

### 20.3 `native_accuracy` is unchanged, and stays honest

The existing host round trip — `to_numpy()` then `numpy.argmax` — remains
exactly as it is. It is honest, it is documented as a reporting helper, and
a native `argmax` does not oblige anyone to rewrite it in the same
milestone.

There is one **required documentation reconciliation**, and K0 assigned it
to the wrong milestone. `native_metrics.py`'s module docstring and the
`NATIVE_METRICS` comment in `backends/cpp.py` both stated that there is no
native `argmax` **because the runtime has no integer dtype**. K0 recorded
that this becomes false at **K3**, reasoning about the `argmax` half — but
the sentence is a conjunction, and its *second* conjunct expired one
milestone earlier: **K2 gave the runtime an exact `int64` index/result
dtype**, so from K2 the stated reason is simply untrue.

**K2 therefore corrected the reason and left the conclusion**, which was
accurate until K3: there was deliberately no native `argmax`, the dtype an
index-producing reduction would return existed, and the operation belonged
to a later milestone. Both surfaces said that.

**K3 owned the other half and has performed it.** `argmax` shipped, so both
surfaces now say what K0 specified — **a native `argmax` exists; this metric
still reports through the host boundary, deliberately** — rather than
deleting the honesty. The deliberateness is the substance, not a hedge:
rewriting `native_accuracy` in terms of the native `argmax` would still need
an integer *equality* reduction to compare predictions against targets, and
§11.4 refuses one. So the metric would end up materializing to the host
anyway, one operation later and one allocation heavier, with its one
explicit `to_numpy()` boundary harder to see. It stays exactly as it is.

Recording the misassignment here rather than quietly re-dating it is the
point: a contract that predicted the wrong milestone should say so, and both
halves of the sentence have now been repaired by the milestone that actually
owned each.

**K5 turned "it stays exactly as it is" into a test.** With
`NativeTensor.argmax`, `NativeTensor.index_select`, and both
`NativeTensorCore` counterparts patched to raise, `native_accuracy` still
succeeds — which is only possible if it calls none of them. It materializes
through `to_numpy()` exactly once, allocates no `int64` storage, builds no
graph, mutates neither operand, and reproduces NumPy's tie and
exceptional-value results across equal maxima, signed zeros, one NaN,
several NaNs, both infinities, and NaN beside an infinity. Those are
asserted against a **NumPy oracle**, never against `NativeTensor.argmax`: a
test that compared the two would be inventing exactly the equivalence this
section declines to claim.

### 20.4 Sequencing

Representation and every barrier (K1) → the integer tensor and its public
door (K2) → `argmax` (K3) → one index-consuming primitive (K4) →
integration proof (K5). No milestone combines two of these to save effort.

---

## 21. Checkpoint and state compatibility

### 21.1 The position

- **No checkpoint version change anywhere in Phase K.** The format stays
  `tensorforge.native_checkpoint`, version **3**, with `(1, 2, 3)`
  accepted.
- **No integer optimizer state**, no integer parameter state, no integer
  buffer state.
- **No new checkpoint field**, no reserved constant, no empty integer
  section, and **no version 4** — no version-4 constant is written into
  source, reserved, or accepted.
- Versions 1 and 2 remain historical float64-only formats permanently.
- Version 3 continues to represent the existing float32/float64 model,
  buffer, and optimizer families and is **not** silently reinterpreted.
- Loader state (`tensorforge.native_data_loader`) and sampler state
  (`tensorforge.native_sampler`) stay at version **1** with `(1,)`
  accepted.

**K5 proved every row of this against the live tree**, and from both sides:
a real archive's entries are floating at every role and at both widths, a
malformed copy declaring `int64` at a parameter, a persistent-buffer, an
optimizer-moment, or an optimizer-parameter entry is rejected before
anything is published and without allocating an `int64` storage, a
version-1 archive still loads under its legacy rules, and no version-4
constant is written, reserved, or accepted anywhere in the module — checked
by sweeping its version constants rather than by reading one of them.

### 21.2 Version 3 is semantically limited to floating entries — enforced

`_validated_entry_dtype` currently validates a declared entry dtype against
`cpp._normalize_internal_dtype`, i.e. against the **representation** table.
At K1 it is narrowed to validate against `normalize_dtype` instead — **one
milestone before `int64` is representable in Python at all** (§5.4) — so at
no point can an archive declare `int64` for a model, buffer, or optimizer
entry. The
existing version-1/version-2 float64-only rule is unchanged and still
applied after the new check.

That narrowing is not a restriction of an existing capability: no archive
that any shipped code could write has ever contained an `int64` entry.

### 21.3 Integer tensors are runtime objects, not state

An `int64` tensor is a value a caller holds between operations. It is not a
parameter (§9), not a buffer (§10), and therefore has no route into
`state_dict()` or an archive. Nothing about Phase K needs it to be
persisted: `argmax` results are recomputed from the model in one call, and
index tensors are either recomputed or are caller data.

### 21.4 Integer values as metadata

A caller who genuinely needs an index to survive a checkpoint puts a
**Python `int`** (or a list of them) in the existing validated JSON
metadata channel, exactly as Phase J's loader state travels. That channel
is unchanged, validates JSON-compatibility only, and interprets nothing.
This is a caller convention, and no production constant spells one.

### 21.5 What would justify a future version bump

Recorded so a future phase has a criterion rather than a temptation. A
checkpoint version bump would need **an actual persisted-state
requirement**: a module whose correctness after a resume depends on an
integer tensor it owns — a learned index table, a persistent integer
counter that is genuinely tensor-shaped, or an integer buffer some future
normalization needs. Convenience, symmetry, or "we have the dtype now" are
not requirements. Such a bump would be assigned to a milestone of a
separately approved phase, justified in that phase's contract, and would
carry its own exact-resume proof at the new dtype.

---

## 22. C ABI plan

### 22.1 What is preserved

Opaque handles; a strict C ABI with no C++ types crossing it; `ctypes`;
lazy library loading; the thread-local error contract with
`TF_GUARD_BEGIN`/`TF_GUARD_END`; source-inventory / built-library export
equality; the dtype travelling with the storage; one centralized dtype
dispatch per exported call; hidden default visibility with `TF_EXPORT` only
on functions Python declares; no pybind11; and no one-symbol-per-dtype
explosion.

### 22.2 What needs no new symbol

Verified against the live sources rather than assumed:

| Capability | Mechanism | New symbol |
|---|---|---|
| `int64` storage allocation | `tf_storage_create_typed(size, 2)` — the existing typed creator, plus one dtype code | **none** |
| `int64` uninitialized allocation | `tf_storage_create_uninitialized_typed` — exists; **no Phase-K path uses it** (§27.3) | none |
| host → `int64` storage | `tf_storage_copy_from` — already `void*`, already dispatches on the tag; gains an `Int64` arm | **none** |
| `int64` storage → host | `tf_storage_copy_to` — same | **none** |
| strided `int64` → contiguous host | `tf_storage_materialize` — same | **none** |
| `int64` views (`reshape`, `transpose`, `T`, `narrow`) | metadata-only in Python; no export is involved at all | **none** |
| `int64` `contiguous_copy` | `tf_core_contiguous_copy` — already dtype-general and dtype-strict; gains an `Int64` arm | **none** |
| `item()` / `tolist()` | built on `to_numpy()`, i.e. on `tf_storage_materialize` | **none** |
| `tf_storage_size`, `tf_storage_destroy` | dtype-independent already | **none** |

### 22.3 What needs a new symbol, and why

| Milestone | Symbol | Why no existing export can express it |
|---|---|---|
| **K3** | `tf_core_argmax` | No shipped export produces an index. `tf_core_sum` accumulates values with per-axis write-strides and has no notion of a position; a comparison-and-position search is a different traversal with a different destination dtype, and expressing it through existing symbols would need a value tensor plus a second pass that could not reproduce the first-occurrence rule. |
| **K4** | `tf_core_index_select` | No shipped export gathers by position. `tf_core_narrow_backward` scatters **one contiguous run**; `tf_core_contiguous_copy` is the identity map over one view's own layout. Composing selection from `narrow` per index would allocate one temporary per index and would still not express duplicates in one traversal. |

**Total Phase-K ABI delta: +2. Maximum: 56 exported production `tf_*`
symbols.** K1, K2, and K5 through K9 add **zero**. No milestone may
exceed this without a separately approved amendment to this contract.

**No exported storage-dtype query is added.** The Python-owned dtype
metadata model is still sound at three dtypes for exactly the reason it was
sound at two: Python knows a storage's dtype because Python asked for it at
creation, and every handle-based export reads the tag it already receives.
A `tf_storage_dtype` export would be a second authority for a value the
wrapper already owns.

### 22.4 The dispatch-safety requirement — the hard one

§3.4 records the shape of every dtype dispatch in the runtime:
`case Float32: … ; case Float64: break;` followed by a `double`
fallthrough. Adding a third enumerator makes each of those switches
non-exhaustive, which the compiler diagnoses — and the project requires
zero warnings, so it would be caught. **Phase K does not rely on that.**

At **K1**, every export that is not explicitly generalized to `int64` gains
an explicit guard:

```c++
// hidden visibility, in tf_internal.h, beside require_float64
bool require_floating(const char* operation,
                      std::initializer_list<const void*> handles) noexcept;
```

Same contract as the existing guards: total, `noexcept`, allocation-free, a
function of the storage tags alone; null handles pass so each export keeps
its own null validation and ordering; on rejection it records
`TF_ERROR_INVALID` naming the operation and the offending dtype, and the
caller returns **without touching any destination**.

`require_float64` is retained unchanged for the exports that are genuinely
float64-only; `require_floating` is the guard for the dtype-general
floating exports, and it is applied **before** `require_matching_dtype` so
that a mixed float/int call is rejected as "this operation is floating-only"
rather than as a tag mismatch.

The result: an `int64` handle reaching an arithmetic export is rejected by
a **check**, at the ABI, independently of Python, and independently of
whether anyone read a compiler warning.

### 22.5 Scalar-carrying exports stay floating-only

`tf_storage_fill(void*, double)` and `tf_storage_scale(void*, double)`
carry their scalar as a `double` and are therefore inexact above 2^53
(§3.6). Both gain `require_floating` at K1 and neither is ever an integer
path. `int64` storage is exact-zeroed by the creator, which needs no fill.

### 22.6 The two new signatures

```c
TF_EXPORT void tf_core_argmax(
    const void* src_handle, int64_t src_offset,   // floating, contiguous
    void*       dst_handle,                       // int64, caller-allocated, offset 0
    int64_t outer, int64_t axis_length, int64_t inner);

TF_EXPORT void tf_core_index_select(
    const void* src_handle, int64_t src_offset,   // floating, contiguous
    const void* idx_handle, int64_t idx_offset,   // int64,    contiguous
    void*       dst_handle,                       // floating, caller-allocated, offset 0
    int64_t outer, int64_t axis_length,
    int64_t index_count, int64_t inner);
```

Both follow the shipped conventions exactly: fixed-width `int64_t`
positions only (no `long`, no `long long`, no `size_t`); handles carry
per-operand offsets where a source may be offset and the caller-allocated
destination is always at offset 0; the `(outer, axis_length, inner)`
decomposition is the same one `tf_core_softmax_forward` and
`tf_core_log_softmax_forward` already take.

A **full** `argmax` reduction is expressed as
`outer = 1, axis_length = numel, inner = 1`, so one kernel covers both the
full and the per-axis case: no second symbol, no mode flag, no branch that
a caller controls.

### 22.7 Ownership, allocation, and error reporting

Outputs are **caller-allocated** in both, matching every fused export
shipped since Phase D. Neither allocates a tensor; neither frees one.
Both are guarded (`TF_GUARD_BEGIN` / `TF_GUARD_END_VOID()`), so no C++
exception escapes and a failure surfaces in Python as `MemoryError`,
`ValueError`, or `RuntimeError` through the existing `errcheck` hook.

### 22.8 Self-validation — `tf_core_argmax`

**The two exports do not share a validation list, and applying one to both
would be wrong.** `argmax` consumes a floating source and produces an
`int64` destination *by design*, so a `require_floating(destination)` or a
`require_matching_dtype(source, destination)` on it would reject **every
valid call**. It also has no index handle, so index validation has nothing
to validate.

**Required roles**

| Handle | Required dtype | Notes |
|---|---|---|
| source | `float32` **or** `float64` | `tf::require_floating` |
| destination | **exactly `int64`** | a dedicated int64-role check, never `require_floating` |
| — | — | **no** `require_matching_dtype` between source and destination |
| — | — | **no** index-input validation — no index handle exists |

Destination size is **exactly `outer * inner`** — one index per output
position — checked against the destination storage's own element count, and
source and destination must not alias.

**Validation order** (each step rejects before the next runs, and a
rejection writes nothing):

1. null handles;
2. **source is floating**;
3. **destination is `int64`**;
4. `outer`, `axis_length`, `inner`, and their checked products are valid
   (each positive, each product representable);
5. the source span implied by `src_offset` and the decomposition lies
   inside the source storage;
6. the destination element count is exactly `outer * inner` and its span
   lies inside the destination storage;
7. source and destination do not alias;
8. execute.

### 22.9 Self-validation — `tf_core_index_select`

**Required roles**

| Handle | Required dtype | Notes |
|---|---|---|
| source | `float32` **or** `float64` | `tf::require_floating` |
| destination | floating **and exactly the source's dtype** | `tf::require_floating`, then `tf::require_matching_dtype(source, destination)` |
| index | **exactly `int64`** | a dedicated int64-role check |

**`require_matching_dtype` is used here and only here, and only over the
floating source/destination pair.** It is never applied across a
floating/index role boundary — in either export — because the index operand
is not an arithmetic operand (§12.4) and a role mismatch is a *role*
error, not a promotion opportunity.

**Validation order:**

1. null handles;
2. **source is floating**;
3. **destination is floating**;
4. **source and destination dtypes match**;
5. **index is `int64`**;
6. `outer`, `axis_length`, `index_count`, `inner`, and their checked
   products are valid;
7. every span is valid — the source span, the index span
   (`index_count` elements from `idx_offset`), and the destination span
   (`outer * index_count * inner` elements);
8. aliasing is rejected — destination against source, and destination
   against index;
9. **every index is scanned and validated** against `[0, axis_length)`,
   completely, **before the first destination element is written**;
10. execute.

Step 9 is the one that most obviously must not be folded into the copy
loop: checking each index as it is used leaves a partly written
destination behind when it throws, and §14.4 forbids that. The C-side scan
is a **second** authority, not a restatement of Python's, and neither may
be removed because the other exists — the rule
`docs/native_classification_design.md` §9.3 already states for the
cross-entropy target indices.

### 22.10 What the two contracts share, and what they do not

Shared: `TF_GUARD_BEGIN`/`TF_GUARD_END_VOID()`, caller-allocated outputs,
fixed-width `int64_t` positions, null-handle rejection first, checked
product arithmetic, span containment, aliasing rejection, and
**no-write-on-rejection**.

Not shared, and the reason a single blanket paragraph is a defect rather
than a shorthand:

| | `tf_core_argmax` | `tf_core_index_select` |
|---|---|---|
| destination dtype | **`int64`** | **floating, matching the source** |
| `require_floating(destination)` | **never** — would reject every valid call | **always** |
| `require_matching_dtype(src, dst)` | **never** — would reject every valid call | **always** |
| index handle | **none** | required, exactly `int64` |
| index bounds scan | **not applicable** | required, complete, before any write |
| destination element count | exactly `outer * inner` | exactly `outer * index_count * inner` |

---

## 23. Python API plan

### 23.1 The complete public delta for Phase K

| Milestone | Surface | Delta |
|---|---|---|
| K1 | none | no public name |
| K2 | `NativeTensor` | `+ from_int64_array` (classmethod), `+ item()`, `+ tolist()` |
| K2 | `NativeTensorCore` | **no public name** — it gains only the private `_from_int64_array`, which never appears in the public delta |
| K2 | `cpp` | `INDEX_DTYPES = ("int64",)`; `backend_info()` gains `"index_dtypes"`. **`SUPPORTED_DTYPES` is not touched.** |
| K3 | `NativeTensor`, `NativeTensorCore` | `+ argmax(axis=None, keepdims=False)`; `TENSOR_CORE_OPS` gains `"argmax"` |
| K4 | `NativeTensor`, `NativeTensorCore` | `+ index_select(axis, indices)`; `TENSOR_CORE_OPS` gains `"index_select"` |
| K5–K9 | none | no public name |

**K2's row is three names, and exactly one of them is a door.**
`from_int64_array` is the only public construction or host-ingress door —
the one public API in the repository through which an `int64` buffer can
come into existence. `item()` and `tolist()` are **dtype-general
host-inspection** methods: they construct nothing, allocate no native
storage, and behave identically at `float64`, `float32`, and `int64`. Both
statements are true at once, and neither may be collapsed into the other:
the row is **three names**, exactly **one** of which is a construction
door, so the only singular claim that holds is the one about the *door*.
`item` / `tolist` are not removed to make a shorter sentence work, and
Storage and Core integer ingress stay private (§8.1, §23.4).

### 23.2 `tensorforge.experimental.__all__` stays at 25

**Phase K adds no new public experimental name at any milestone.** Every
new capability is a method on `NativeTensor` — which is the payoff of the
unified object model (§6) — and every new registry value is in
`tensorforge.backends.cpp`, which is a different surface with its own
guardrails.

That is a crisp, checkable, phase-level claim, and
`tests/test_native_phase_k.py` locks the K0 half of it now.

### 23.3 `AUTOGRAD_OPS` is unchanged for the whole phase

Neither `argmax` nor `index_select` is differentiable, so neither joins
`AUTOGRAD_OPS`, at any milestone. `TENSOR_CORE_KERNELS` — the deliberately
frozen historical registry — is untouched, as it has been since the
sum/mean/sqrt precedent. `RAW_KERNELS` is untouched.

### 23.4 Private names that stay private

`_normalize_index_dtype`, `_DTYPE_CODES`'s `int64` entry, the
`int64` `_CHECKED_HOST_ARRAYS` binding, `NativeStorage._typed`,
`NativeTensorCore._typed*`, and the internal decomposition helpers for the
two new kernels. None is exported, none appears in `__all__`, and none is a
supported way around a public validator — exactly the standing rule for the
existing `_typed*` family.

### 23.5 What no Python surface gains

No `dtype` argument on any class that owns no dtype-bearing state; no
`device` argument anywhere; no `astype` / `to` / `.int()` / `.long()`; no
`is_integer` / `is_floating` property; no dtype object; no global default
dtype; no `__getitem__` / `__setitem__` accepting tensors; no `__len__`; no
performance control of any kind.

---

## 24. Stable / native isolation

Unchanged and re-asserted for Phase K:

- `tensorforge.Tensor`, its autograd, `tensorforge.nn`,
  `tensorforge.optim`, `tensorforge.data`, and
  `tensorforge.serialization` are **unchanged**; no file under the stable
  framework needs a Phase-K code change, at any milestone;
- the stable public export list is unchanged;
- stable serialization stays entirely separate from the native checkpoint;
- `import tensorforge` never loads the native library, and a test proves
  it;
- no implicit stable ↔ native conversion, in either direction;
- no automatic backend selection or routing, and no environment variable
  that changes which line runs;
- **no stable API accepts a native integer tensor**, and no stable object
  may be passed to an integer constructor —
  `NativeParameter`'s existing `_reject_framework_object` guard is the
  precedent, and the K2 constructor applies the same rejection;
- `backend_info()["stable_framework_integration"] is False`, permanently.

---

## 25. Device and concurrency boundary

Phase K is **CPU only, synchronous, and externally locked if used
concurrently**.

Explicitly outside the phase: CUDA and every GPU backend; AMP and mixed
precision; `float16` and `bfloat16`; device movement; `.to()`, `.cpu()`,
`.cuda()`, `map_location`; streams; events; asynchronous execution;
workers; prefetch; any thread-safety claim; distributed execution; and
sparse tensors.

**No `device` argument exists anywhere and none may be added.**
`SUPPORTED_DEVICES` stays `("cpu",)` and `UNSUPPORTED` stays
`("cuda", "amp")` for every Phase-K milestone.

No Phase-K module contains a lock, thread, queue, future, or async
primitive; the new objects join no lock order; external locking is the
caller's job; **no Phase-K test starts a thread**, and none may. Concurrency
is a documented boundary, never a tested safety claim — the stance Phase J
locked, applied unchanged.

Any device architecture from an external implementation is explicitly not
imported into this design (§36).

---

## 26. Error precedence

A caller with several invalid inputs receives a **deterministic first
error**. The orders below are contractual; the guardrails protect the
*architecture* of the ordering by driving multi-fault calls, not by
freezing prose.

### 26.1 Integer construction

1. requested API route (`from_int64_array` versus a floating constructor
   with `dtype="int64"`);
2. `requires_grad` type, then value;
2a. the **index/result dtype authority** — `cpp._normalize_index_dtype`
   measured against the public `INDEX_DTYPES` registry (§5.2). It is the
   canonical registry gate for this fixed-format door, and it is asked
   here: after both `requires_grad` checks, before the input is inspected,
   before the private Core ingress is entered, and before anything is
   allocated. A sub-step rather than a renumbering, because it is a
   *registry* question asked of a constructor that carries no `dtype`
   argument — not a new check on a caller's value;
3. exact input type (`numpy.ndarray`, not a subclass);
4. exact NumPy dtype including native byte order;
5. rank and shape validity;
6. non-empty representability;
7. checked element-to-byte size;
8. device (defaulted, validated as it already is);
9. allocation;
10. copy.

Steps 1–8 allocate nothing. Step 9 is the first that can, and step 10 is
the only one that writes.

### 26.2 `requires_grad=True` on an integer constructor

Rejected at step 2 — **before allocation**, before the array is even
examined.

### 26.3 Integer operands reaching a floating operation

Dtype capability is rejected **before output allocation** and before the
Core layer is entered. Object type → closed state → dtype capability →
operand agreement → shape → allocation.

### 26.4 `argmax`

§17.6, in full.

### 26.5 `index_select`

§18.6, in full.

### 26.6 What the tests protect

That the ordering *is* the ordering — proved by calls that are invalid in
two ways at once, asserting which error arrives — and that a rejection at
any pre-allocation step leaves live storage at baseline. Exact message
wording is **not** frozen.

---

## 27. Allocation-failure behavior

### 27.1 The rule, per allocating path

For `from_int64_array`, `contiguous_copy`, `argmax`, and `index_select`:

- **validated before allocation** — everything in §26 up to the allocation
  step, including the complete index bounds scan;
- **owner of the fresh storage** — the new `NativeTensorCore`, from the
  moment the wrapper is successfully constructed;
- **ownership transfer** — to the caller, at return, and not before;
- **closed on failure** — everything the call allocated, in reverse order
  of allocation, including every Policy-B temporary;
- **`BaseException` follows the same path** — the cleanup is in a
  `finally`/`except BaseException` and is unconditional, so a
  `KeyboardInterrupt` between allocation and publication frees exactly what
  a `ValueError` would;
- **unchanged after rejection** — both operands' storage, metadata, graph
  state, `requires_grad`, `.grad`, every parameter version, both global
  RNGs, the filesystem, and every registry;
- **live storage returns exactly to baseline** — asserted by a tracker
  installed **outside** `monkeypatch`, so a mid-test `undo()` cannot
  silently disarm it.

### 27.2 Injected failure positions are distinct

Phase-K hardening (K7) must treat these as **four different injections**
and may not label one as another: host validation/normalization, native
allocation (through the existing thread-local arm only, disarmed in a
`finally` *and* an autouse fixture), host → native transfer, and kernel
execution. Every injection needs a non-vacuity control proving it can fire.

A **family** is not a position. Where one operation reaches the same
export from two different call sites, those are two positions and each
must be injected on its own: `index_select` materializes through
`tf_core_contiguous_copy` once for its floating source and once for its
`int64` index, and a stand-in that fails the export immediately reaches
only the first. Reaching the second requires delegating the first call to
the real export, and the two must be shown distinguishable by dtype, size,
or metadata rather than by ordering alone. Conversely, a cleanup
*invariant* checked at a position the matrix already names — reverse-order
release, or release **exactly once** — is not a new position and may not
be added as one: the matrix describes seams, not tests.

### 27.3 No `int64` path uses uninitialized allocation

`argmax`'s destination and `index_select`'s destination are
**zero-initialized**, through the ordinary `tf_storage_create_typed`
default. Phase K adds **no row** to the uninitialized-allocation audit
table in `docs/native_cpu_performance_design.md`, and therefore needs no
`int64` poison test.

This is a deliberate choice over a provable optimization. `index_select`
does overwrite every destination element, so an uninitialized destination
would be admissible — but admitting it would require an audit row, a poison
seam that can write a recognizable non-zero `int64` pattern, and a negative
control, all to save one pass over an output that is small by construction.
The permission is available to a later milestone that measures a reason;
Phase K does not take it.

---

## 28. Lifecycle and explicit close

Every rule is inherited and none is dtype-dependent; they are restated
because §6's unified object model is only as safe as its lifetime story.

- `close()` exists **exactly where something is owned**: on an `int64`
  `NativeTensor` that owns its core, and nowhere else.
- `close()` is **idempotent**; a second call is a no-op.
- `__del__` is a **fallback only**. No assertion anywhere may depend on
  collection timing; abandonment is proved by explicit `close()`.
- A **view** never closes its parent's storage; a chained view keeps the
  whole chain reachable; closing a source with a live view leaves the
  view's *storage* intact and its subsequent operations rejecting.
- **Operations on a closed tensor** raise `RuntimeError` before validating
  anything else about the call.
- **Every delivered output is the caller's** — the **caller closes** it.
  `argmax` and `index_select` results are closed by the caller; no
  operation retains a reference to one and no close path may reach one.
- **Context managers**: `with NativeTensor.from_int64_array(x) as t:` works
  exactly as it does for a floating tensor.
- Cleanup is explicit and never relies on garbage collection.

---

## 29. Cross-platform exactness

### 29.1 Width and representation

`std::int64_t` exists, is exactly 64 bits, has no padding, and is
two's-complement — required by the standard for the exact-width `intN_t`
types, and `static_assert`ed anyway (§13.1). NumPy `int64` is exactly 8
bytes, asserted on the Python side.

### 29.2 No integer-width ABI ambiguity

Every ABI position is `int64_t` (and `int32_t` for the dtype code). No
Phase-K declaration spells `long`, `long long`, `size_t`, `ssize_t`, or
`Py_ssize_t`. `ctypes.c_int64` is the only integer argument type the new
declarations use.

### 29.3 Checked allocation sizes

`tf::dtype_checked_bytes` proves `numel × 8` is representable before
anything is allocated, on both 64-bit and any narrower-`size_t` platform,
exactly as it already does for the floating widths.

### 29.4 Endianness

**Host-native only.** A byte-swapped host array is **rejected**, never
byte-swapped, at ingress (§8.2) and at egress (`to_numpy` allocates a
native-order array). No Phase-K code performs a byte swap, and no archive
gains a byte-order field.

### 29.5 Windows and Linux produce identical logical values

Every Phase-K kernel traverses in a fixed logical order determined by shape
metadata alone (§13.10), performs no floating-point arithmetic, and relies
on no platform-dependent overflow or conversion behavior. `argmax`'s
comparisons are IEEE comparisons of values the platform already agrees on
bit for bit (transfer is bit-preserving), and its NaN rule is expressed
with `isnan` rather than with a comparison whose result varies. Therefore
the integer values produced on Windows and on Linux are **equal**, and the
K9 validation asserts it directly rather than assuming it.

This is a different and **stronger** statement than the transcendental
one-ULP bound for `exp`/`log`, which exists precisely because libm differs
between MSVC and glibc. The two must never be conflated, and the one-ULP
bound is never tightened.

### 29.6 Exact comparison in tests

Integer comparisons in every Phase-K test use exact `==` on Python `int`s
or on NumPy `int64` arrays. **No floating-point tolerance is used for an
integer test, anywhere.** Where a Phase-K test compares floating values
(an `index_select` output against its source slices), it compares **raw
IEEE-754 bit patterns** through `uint32`/`uint64` views, never `allclose`,
`approx`, or any tolerance — the discipline every phase from C onward has
used, and each dtype is proved only against itself.

---

## 30. Testing ownership

### 30.1 Which module owns what

| Subject | Module |
|---|---|
| The phase contract, status, and presence/absence split | `tests/test_native_phase_k.py` (K0; expanded per milestone) |
| `int64` representation, C-side guards, and **every reachability barrier** | `tests/test_native_integer_barriers.py` (K1, shipped) + `cpp/tests/test_dtype_int64_storage.cpp` (K1, shipped) |
| Construction, ownership, views, copies, host exit | `tests/test_native_int64_tensor.py` (K2) |
| `argmax` | `tests/test_native_argmax.py` (K3) + a native CTest |
| `index_select` | `tests/test_native_index_select.py` (K4) + a native CTest |
| Checkpoint, state, pipeline, classification compatibility | `tests/test_native_integer_compatibility.py` (K5, shipped) |
| The end-to-end example | `tests/test_native_integer_indexing_example.py` (K6) |
| Adversarial hardening | `tests/test_native_integer_hardening.py` (K7) |
| The benchmark | `tests/test_native_integer_benchmark.py` (K8, shipped) + `benchmarks/benchmark_native_integer.py` (K8, shipped) |
| The closed boundary | `tests/test_native_phase_k_closure.py` (K9) |

Names are indicative; the *split* is contractual. A claim about closure
belongs in the closure module, not in the contract module — the boundary
Phase J established.

### 30.2 Discipline every Phase-K test module inherits

- **Exact equality only** for integers, and raw IEEE-754 bits for floating
  comparisons (§29.6).
- **Every rejection and every injected failure is followed by a complete
  before/after fingerprint of the observable world**, and **every injection
  and every parser has a non-vacuity control** — including the fingerprint
  itself, whose every component must be proved able to notice the change it
  exists for.
- Failure positions stay **distinct injections** (§27.2).
- Abandonment is proved by explicit `close()`; **no assertion may depend on
  collection timing**.
- The `live_storages` tracker installs itself **outside** `monkeypatch`.
- **Source scans read code, not prose**: docstrings and string literals are
  stripped through the **AST** first, and keyword-argument names are read
  too. A substring ban fails on the very sentence that documents the
  prohibition.
- Executable example code stays on **public APIs only**.
- **Every scanner needs a negative control.**
- No test starts a thread, touches the network, requires an ancestor
  commit, or depends on a total suite count.

---

## 31. Benchmark governance

K0 adds no benchmark. **K8** adds exactly one, and it inherits every
existing rule plus the ones this phase makes specific.

- **Correctness is gated before timing**, always. A failed gate publishes
  no timing and the CLI exits nonzero with clean stdout.
- **No speed is asserted anywhere**: no timing threshold, no performance
  budget, no committed duration, and **no CI job that fails on a number**.
- **No result file of any kind is written**, and there is no `--save`,
  `--output`, or `--baseline` option.
- **`int64` cases are separate cases** from `float32` and `float64` cases
  and are **never** presented as a ratio of one dtype to another.
- A case with no honest equivalent is labelled `native_only` and publishes
  **no ratio at all**. Never fabricate a comparison layer.
- **Never divide a native case that allocates by a host case that does
  not** — Phase J's rule, and it is the live risk here: a native `argmax`
  allocates an output tensor and `numpy.argmax` over an existing host array
  does not.
- Setup, cleanup, and temporaries stay outside the timer, and every native
  object is closed explicitly rather than left to GC.
- Medians with spread after warm-up; regressions, neutral results, and
  noise published as prominently as wins.
- **No cross-dtype speed ratio may be presented as a correctness or support
  claim**, and no hardware-independent speed promise is made anywhere.
- The benchmark answers **separate questions and never blurs them**:
  integer construction, host materialization, `argmax`, and `index_select`
  are four questions, and any composed case is labelled explicitly as a
  composition and is never a substitute for the parts.
- The benchmark is registered in **no** runtime inventory: a measurement
  tool is never a capability.

---

## 32. Final milestone ladder

Ten milestones, **one purpose each**. K0 through K8 are complete; K9 is
unstarted. No milestone combines two major operations, no milestone exists
to preserve a numbering, and K9 is the closure.

### 32.1 The window proof — there is no unsafe intermediate state

The ordering rule this ladder exists to satisfy: **no milestone may leave
an integer tensor constructible while any existing generic API can treat
it as floating or as model state.** The proof is a reachability argument
over exactly three states, and it is why the barriers are not spread
across the ladder.

**State after K0.** `int64` does not exist anywhere. Nothing to prove.

**State after K1.** An `int64` *buffer* could be allocated, but **no
supported TensorForge wrapper or public Python API could allocate or wrap
`int64` storage at K1; only the raw private C ABI could represent it, for
isolation and barrier testing** — a direct `tf_storage_create_typed(n, 2)`
through `ctypes`. It could not be reached from Python's own API surface,
because `_DTYPE_CODES` had no `"int64"` entry, so `NativeStorage._typed`,
`_uninitialized`, and the whole `_typed*` family rejected the name before
allocating. And even handed such a handle, nothing could be built over it:
`NativeTensorCore`'s and `NativeTensor`'s constructors rejected a
non-floating storage or core.
Meanwhile **every** barrier in §6.5 was already in place. So at K1 there
was no `int64` tensor, no `int64` core, and no route to a parameter, a
buffer, an optimizer, a graph, a checkpoint, or a floating kernel.

**State after K2 — today.** An `int64` tensor exists and is publicly
constructible through exactly one door — and every barrier it could meet
landed **one milestone earlier** and is re-proved against the real object
in `tests/test_native_int64_tensor.py`. `INDEX_DTYPES` appeared in the same
commit, so the promise and the capability were never out of step in either
direction. Two gates widened and no others: `NativeTensorCore.__init__` and
`NativeTensor.__init__`, from "floating" to "floating **or** index". Every
generic constructor still rejects `"int64"`, `SUPPORTED_DTYPES` did not
move, and the tensor is still refused by autograd, by `NativeParameter`, by
`register_buffer` at both persistence values, by both optimizers, by
checkpoint entry validation, and by every floating operation entry.

**The specific windows this ordering closes**, each of which a more
"natural" ladder would have opened:

| Hypothetical window | Why it cannot occur here |
|---|---|
| Integer tensor exists, `NativeParameter` not yet gated | `normalize_module_dtype` is narrowed at K1; construction is K2 |
| Integer tensor exists, `register_buffer` not yet gated | Both `persistent` values gated at K1 |
| Integer tensor exists, an optimizer accepts it | Both optimizers gated at K1 |
| Integer tensor exists, `_from_op` builds a graph over it | `_from_op` gated at K1 |
| Integer tensor exists, `backward()` walks it | `backward` / `_accumulate_grad` gated at K1 |
| Integer tensor exists, `add`/`matmul`/`sum` read it as `double` | Python entries gated at K1; `tf::require_floating` on every float-only export at K1 |
| Integer tensor exists, an archive declares an `int64` entry | `_validated_entry_dtype` narrowed at K1 |
| `int64` in a public registry before a barrier lands | `SUPPORTED_DTYPES` never gains it at all; `INDEX_DTYPES` appears at K2 |
| A generic constructor still routed through a registry that now admits `int64` | No generic constructor's validator ever admits `int64` (§5.4, §5.5) |

**The invariant, stated once:** for every barrier *b* and the first
milestone *c* at which an `int64` tensor can be constructed,
`milestone(b) < c`. §33 tabulates both sides, and
`tests/test_native_phase_k.py` checks the inequality structurally rather
than trusting the prose.

| # | Milestone | Purpose |
|---|---|---|
| **K0** | Architecture, taxonomy, API/ABI plan, and guardrails | **Complete.** This document, the status reconciliation, and `tests/test_native_phase_k.py`. Zero runtime. |
| **K1** | `int64` representation **and every reachability barrier**, with the dtype reachable only through the C ABI | **Complete.** `tf::Dtype::Int64`, the transfer arms, `tf::require_floating` on every float-only export, the nine Python narrowings, every §6.5 barrier, `tests/test_native_integer_barriers.py`, and `cpp/tests/test_dtype_int64_storage.cpp`. No public name, no export, no registry movement. |
| **K2** | The `int64` tensor: private storage ingress, construction, views, copies, host inspection, the one public door, and `INDEX_DTYPES` — **atomically** | **Complete.** The three dtype tables and the host binding, `INDEX_DTYPES` and `backend_info()["index_dtypes"]`, the generalized no-drift invariant, `_normalize_index_dtype`, the private `_from_int64_array` pair, the two widened wrapper gates, `NativeTensor.from_int64_array` / `item()` / `tolist()`, and `tests/test_native_int64_tensor.py`. No export, no `__all__` change, no CTest, no example, no benchmark, no version change. |
| **K3** | Native `argmax` — the phase's first operation and its first C ABI symbol | **Complete.** `cpp/include/tf_indexing_internal.h` (the templated traversal and the `tf::require_index` role guard), `cpp/src/indexing.cpp` (`tf_core_argmax`), one CTest (25 → **26**), the ctypes declaration and errcheck hook, `NativeTensorCore.argmax`, `NativeTensor.argmax`, `"argmax"` in `TENSOR_CORE_OPS`, the §20.3 reconciliation, and `tests/test_native_argmax.py`. Exports 54 → **55**. `AUTOGRAD_OPS`, `__all__`, every registry, every version, the examples, and the benchmarks are unmoved, and no `max` was shipped. |
| **K4** | `index_select`, forward only — the phase's one index-*consuming* operation and its second and final C ABI symbol | **Complete.** `tf::index_select_contiguous` beside K3's traversal in the same internal header, `tf_core_index_select` beside `tf_core_argmax` in the same translation unit, one CTest (26 → **27**), the ctypes declaration and errcheck hook, `NativeTensorCore.index_select`, `NativeTensor.index_select` with the `requires_grad` rejection naming `detach()`, `"index_select"` in `TENSOR_CORE_OPS`, and `tests/test_native_index_select.py`. Exports 55 → **56**, the phase maximum. `AUTOGRAD_OPS`, `__all__`, every registry, every version, the examples, and the benchmarks are unmoved, and no `gather`, `scatter`, embedding, or backward was shipped. |
| **K5** | Compatibility proof — checkpoint, state, data pipeline, classification | **Complete.** `tests/test_native_integer_compatibility.py`, and the status reconciliation. Zero production code: no export, no public name, no CTest, no example, no benchmark, no registry or version movement. |
| **K6** | End-to-end integration example and exact proof | **Complete.** `examples/native_integer_indexing.py` and `tests/test_native_integer_indexing_example.py`. Examples 16 → **17**; zero production code, no export, no `__all__` change, no CTest, no benchmark, no registry or version movement. |
| **K7** | Adversarial hardening — §27's four injection families at every actual allocating path (both of `index_select`'s Policy-B materialization call sites separately), the complete world fingerprint, `BaseException` cleanup, and the malformed-metadata and dtype-role matrices | **Complete.** `tests/test_native_integer_hardening.py`, and the status reconciliation. Zero production code: no export, no public name, no CTest, no example, no benchmark, no registry or version movement, and no defect found. |
| **K8** | Benchmark characterization — §31 in full, four separate questions and no composition | **Complete.** `benchmarks/benchmark_native_integer.py` and `tests/test_native_integer_benchmark.py`. Benchmarks 9 → **10**; zero production code, no export, no `__all__` change, no CTest, no example, no registry or version movement, and no optimization. |
| **K9** | Cross-platform validation and Phase-K closure | *Unstarted.* |

### K0 — Architecture, taxonomy, API/ABI plan, and guardrails · complete

This document; `tests/test_native_phase_k.py`; the narrow status
reconciliation the newly approved phase requires. **Zero runtime, zero
production source, zero export, zero registry movement.**

### K1 — `int64` representation and every reachability barrier · complete

**Layers:** `cpp/include/tf_internal.h`, `cpp/src/storage.cpp`, every
`cpp/src/*.cpp` that guards a float-only export,
`src/tensorforge/backends/cpp.py`,
`src/tensorforge/experimental/_native_dtype.py`, `native_tensor.py`,
`native_parameter.py`, `native_module.py`, `native_sgd.py`,
`native_adam.py`, `native_checkpoint.py`.

**C++ — the representation.** `TF_DTYPE_INT64 = 2` / `Dtype::Int64` and the
arms in `dtype_from_code`, `dtype_item_size`, `dtype_name`,
`create_storage`, and `destroy_storage_data`; the two `static_assert`s of
§13.1; the `Int64` arms in `tf_storage_copy_from`, `tf_storage_copy_to`,
`tf_storage_materialize`, and `tf_core_contiguous_copy`.

**C++ — the guard.** `tf::require_floating` (§22.4) added and applied to
**every** float-only export, including `tf_storage_fill` and
`tf_storage_scale`.

**Python — nothing becomes representable.** `_DTYPE_CODES`,
`_DTYPE_ITEM_SIZES`, `_DTYPE_NUMPY`, and `_CHECKED_HOST_ARRAYS` are
**untouched at K1**, so **no supported TensorForge wrapper or public
Python API can allocate or wrap `int64` storage at K1; only the raw
private C ABI can represent it, for isolation and barrier testing** — and
the Phase-I invariant `set(_DTYPE_CODES) == set(SUPPORTED_DTYPES)` stays
exactly true. K1 proves that no Python object can be built over a handle
obtained that way.

**Python — the nine narrowings** (§5.4), each behavior-preserving on the
day it lands because the two tables accept the same set today. **Seven
constructor/backend**: `NativeStorage._uninitialized`,
`NativeStorage._typed_from_array`,
`NativeTensorCore.zeros(_trusted_dtype=True)`,
`NativeTensorCore._uninitialized`, `NativeTensorCore._typed_from_array`,
`NativeTensorCore._typed_full`, and `cpp._narrowed_to_dtype` all move from
`_normalize_internal_dtype` to `normalize_dtype`. **Two
state-validation**: `_native_dtype.normalize_module_dtype` and
`native_checkpoint._validated_entry_dtype` do the same. Seven plus two is
the whole of it.

**Python — every barrier in §6.5**, without exception: `_from_op`,
`backward`, `_accumulate_grad`, `NativeParameter.__init__`,
`NativeModule.register_buffer` (both `persistent=True` and
`persistent=False`), `NativeSGD`, `NativeAdam`, every floating operation
entry, mixed float/integer requests, and the `NativeTensorCore.__init__` /
`NativeTensor.__init__` gates that reject a non-floating storage or core
outright. Each is keyed on a dtype predicate rather than on the existence
of `int64`, so each is testable at K1 by driving the C ABI directly.

**One status reconciliation was also assigned here, and deliberately not
to K0.** `src/tensorforge/experimental/__init__.py` read "Phase J … is the
latest phase", which K0 could not repair because K0 changes **no
production source at all** — and that module is production source, module
documentation or not. K1 is the first milestone that edits the package, so
K1 moved that sentence and removed the matching scoped exemption in
`tests/test_docs.py`, which the module's own text pinned. Until K1 the
sentence was knowingly one phase behind, recorded here rather than fixed
by widening K0's scope. **The exemption was removed rather than retained**:
one that outlives its reason is a guardrail that has stopped guarding, and
`tests/test_native_phase_k.py` now asserts its absence directly.

**No public name. No export. No registry movement. No Python-constructible
`int64` anything.** CTests +1 (`test_dtype_int64_storage.cpp`) → 25.

**As landed.** Two new test modules and one new C++ test file; no new
production module, no new example, no new benchmark, no build option, and
no dependency.

- **C++ representation** — `TF_DTYPE_INT64 = 2` / `tf::Dtype::Int64`; arms
  in `dtype_from_code`, `dtype_item_size` (`sizeof(std::int64_t)`),
  `dtype_name` (`"int64"`), `create_storage`
  (`create_typed_storage<std::int64_t>`, the existing
  trivially-destructible `static_assert` holding unchanged), and
  `destroy_storage_data`; the two §13.1 `static_assert`s. Every switch
  still has **no `default:`**, so a fourth enumerator without an arm stays
  a compile-time diagnostic. The dtype-code rejection message now names
  all three representable codes.
- **C++ transfer** — `Int64` arms on `tf_storage_copy_from`,
  `tf_storage_copy_to`, `tf_storage_materialize`, and
  `tf_core_contiguous_copy`, each a same-type assignment that performs no
  arithmetic and reproduces its source's object representation exactly.
  `tf_core_contiguous_copy` keeps `require_matching_dtype`, so an
  `int64`↔floating copy is still an invalid *request* rather than a
  conversion.
- **C++ barrier** — `tf::dtype_is_floating` and `tf::require_floating` in
  `tf_internal.h`, applied to **32** exports: 17 elementwise, 1 matmul, 2
  reduction, 4 classification, 3 conv2d, 2 pooling, 1 random, and
  `tf_storage_fill` / `tf_storage_scale`. It runs **before**
  `require_matching_dtype` everywhere, so a mixed float/integer call is
  reported as a role error rather than as a tag mismatch.
- **Error contract** — `tf_storage_fill` and `tf_storage_scale` gained
  `TF_GUARD_BEGIN` / `TF_GUARD_END_VOID()` **because** they can now record
  a rejection: a function that writes the slot must clear it on entry, or
  a code it recorded could go stale. They remain **unhooked**, so
  `_CHECKED_KERNELS` is unchanged at 36 and their per-call boundary cost is
  exactly what H7 left. The "unguarded functions never touch the slot" rule
  stays true because these two are no longer unguarded.
- **Python narrowings** — the nine of §5.4 (seven constructor/backend, two
  state-validation), each behavior-preserving on the day it landed.
- **Python barriers** — `cpp._is_floating_dtype` /
  `cpp._require_floating_dtype` as the one authority, plus
  `_native_dtype.require_floating_state_dtype` as the strict delegate the
  state-owning modules route through; gates on
  `NativeTensorCore.__init__`, `NativeTensor.__init__`,
  `NativeTensorCore._require_floating_operand` at every compute entry,
  `_require_matching_metadata` (both operands, floating-role first),
  `NativeStorage.fill`, `_from_op` (closing the core **and** the saved
  resources it was handed), `backward`, `_accumulate_grad`,
  `NativeParameter.__init__`, `NativeModule.register_buffer` at both
  `persistent` values, and both optimizers' registration loops.
- **Proof** — `cpp/tests/test_dtype_int64_storage.cpp` (the representation,
  the exact round trip at the signed extremes and beyond 2^53, and the
  32-row rejection table with its valid-call control) and
  `tests/test_native_integer_barriers.py` (the raw-ABI isolation proof,
  every Python barrier driven against a genuine `int64` handle, the
  observable-world fingerprint with its own non-vacuity control, and the
  structural scans proving no floating entry and no float-only export
  escapes the audit).

### K2 — The `int64` tensor, its public door, and `INDEX_DTYPES` · complete

**Layers:** `src/tensorforge/backends/cpp.py` (the three dtype tables, the
host binding, `INDEX_DTYPES`, `_normalize_index_dtype`, `backend_info`,
`NativeStorage._from_int64_array`, `NativeTensorCore._from_int64_array`),
`src/tensorforge/experimental/native_tensor.py`
(`from_int64_array`, `item`, `tolist`).

**Everything in this milestone lands atomically**, because splitting it
would create exactly the window §32.1 forbids:

- `_DTYPE_CODES["int64"] = 2`, `_DTYPE_ITEM_SIZES`, `_DTYPE_NUMPY`, and the
  `_CHECKED_HOST_ARRAYS` entry (reusing the existing `_CHECKED_I64_ARRAY`
  object, so the class-label binding and the storage binding cannot
  diverge);
- `INDEX_DTYPES = ("int64",)` and `backend_info()["index_dtypes"]`, with
  `SUPPORTED_DTYPES` **unmoved**;
- the Phase-I table invariant generalized to
  `set(_DTYPE_CODES) == set(SUPPORTED_DTYPES) | set(INDEX_DTYPES)` (§5.1);
- the private ingress `NativeStorage._from_int64_array` /
  `NativeTensorCore._from_int64_array` — exact-dtype validation,
  `np.ascontiguousarray(values)` with **no** `dtype=` argument, zeroed
  `_typed` allocation;
- the `NativeTensorCore.__init__` / `NativeTensor.__init__` gates widened
  from "floating only" to "floating **or** index";
- the **private** ingress helpers `NativeStorage._from_int64_array` and
  `NativeTensorCore._from_int64_array` — leading-underscore, unexported,
  and absent from the public delta;
- §8 in full: **the one public construction door
  `NativeTensor.from_int64_array`**, views and copies at `int64`,
  `to_numpy`, `item`, and `tolist`.

**The public API delta, stated exactly.** K2 adds **three** public
`NativeTensor` method names — `from_int64_array`, `item`, and `tolist` —
and the distinction between them is contractual, not cosmetic:

- **`NativeTensor.from_int64_array` is the only public construction or
  host-ingress door**: the one public API in the repository through which
  an `int64` buffer can come into existence. That is a claim about
  *construction*, and it is what "one door" means.
- **`item` and `tolist` are dtype-general host-inspection methods.** They
  construct nothing, allocate no native storage, and work identically at
  `float64`, `float32`, and `int64`; they are public names K2 adds, and
  they are not doors.
- **Storage and Core integer ingress remain private.**
  `NativeStorage._from_int64_array` and
  `NativeTensorCore._from_int64_array` are leading-underscore, unexported,
  and never appear in the public delta; neither class gains a **public**
  `from_int64_array`, and no public integer constructor exists at either
  lower layer.

Describing the delta as a single **name** is therefore **wrong**, and no
surface may do it: the only singular claim that holds is the one about the
**construction door**. `item` and `tolist` are part of the delta and are
not removed to make a shorter sentence true.

Every K1 barrier is **re-proved against a real `int64` tensor object**, not
merely against a dtype string.

**No export. No `__all__` change. `SUPPORTED_DTYPES` unchanged — and it
stays unchanged for the rest of the phase and afterwards.**

**As landed.** One new test module and no new production module, example,
benchmark, CTest, build option, or dependency. Everything above shipped in
one commit, and the two facts a reader must not conflate — *"an integer
tensor exists"* and *"`int64` is supported"* — stayed apart.

- **Registries** — `INDEX_DTYPES = ("int64",)` beside an **unmoved**
  `SUPPORTED_DTYPES`, reported as `backend_info()["index_dtypes"]`. The
  union of the two is stated in prose and in `backend_info`'s docstring and
  is deliberately **not** materialized as a fifth key. `_DTYPE_CODES`,
  `_DTYPE_ITEM_SIZES`, and `_DTYPE_NUMPY` gained `int64` at code **2**, 8
  bytes, `numpy.int64`, and `_CHECKED_HOST_ARRAYS` gained an entry bound to
  the **already existing** `_CHECKED_I64_ARRAY` object. The Phase-I
  no-drift guard was **generalized rather than deleted**, to
  `set(_DTYPE_CODES) == set(SUPPORTED_DTYPES) | set(INDEX_DTYPES)`, and is
  asserted as an exact equality rather than weakened to a subset check.
- **Validators and predicates** — `_normalize_index_dtype` (no default,
  `None` rejected, `TypeError` for a non-string), `_is_index_dtype`,
  `_is_tensor_dtype` (computed from the two registries, never stored), and
  `_require_tensor_dtype`. All private; none is exported.
- **Exact ingress** — `_exact_host_array` is the shared validator:
  `type(values) is np.ndarray`, then one `values.dtype != expected`
  comparison that rejects every wrong width, both signedness errors,
  `bool`, `object`, and a byte-swapped `>i8` array at once, then
  non-empty. Only then `np.ascontiguousarray(values)` with **no** `dtype=`
  argument — and **rank 0 is returned untouched**, because
  `ascontiguousarray` promotes a 0-d array to shape `(1,)`, which would be
  a silent rank change. `NativeStorage._from_int64_array` allocates through
  the **zeroed** `_typed` (§27.3: no integer path uses the uninitialized
  allocator, so the H1 audit table gains no row) and closes the storage on
  any failure including `BaseException`;
  `NativeTensorCore._from_int64_array` preserves the host shape exactly and
  closes what it allocated if the view or wrapper construction fails.
- **`copy_from` split by role** — the floating arm converts exactly as it
  always has; an index destination requires an already-exact host array
  through `_exact_host_array`. This is §8.3's *"integer ingress converts
  nothing"* applied at the one other place an integer buffer can be
  written, and it is why a `float64` array holding `[1.0, 2.0]` cannot
  reach `int64` storage through any route.
- **Two gates, and only two** — `NativeTensorCore.__init__` and
  `NativeTensor.__init__` moved from `_require_floating_dtype` to
  `_require_tensor_dtype`. Everything else still asks the floating
  predicate, which is what keeps the unified object model safe.
- **`contiguous_copy` destination** — the floating arm keeps its H1
  uninitialized allocation and its poison test untouched; an index
  destination takes the ordinary zeroed allocator.
- **Public surface** — `NativeTensor.from_int64_array(values, *,
  requires_grad=False)`, `NativeTensor.item()`, and
  `NativeTensor.tolist()`. `requires_grad` is validated **first**, before
  the array is examined and long before anything is allocated: a
  non-`bool` raises `TypeError` and `True` raises `ValueError`. `item()`
  and `tolist()` are dtype-general, built on `to_numpy()`, and return
  built-in scalars with no NumPy scalar and no float intermediate.
- **Proof** — `tests/test_native_int64_tensor.py`: the registry and table
  invariants, exact construction at the signed extremes and beyond 2^53,
  every rejection with its exception kind, host-mutation isolation, the
  view/copy inventory with owner/borrower and idempotent-close behavior,
  `to_numpy` / `item` / `tolist` exactness, the **complete K1 barrier
  matrix re-driven against a real `NativeTensor.from_int64_array` result**
  with a before/after fingerprint of the observable world after every
  rejection, injected-failure cleanup at each allocating step, and the
  absence half (no `argmax`, no `index_select`, 54 exports, 25 CTests, 25
  experimental names).
- **One K0 misassignment corrected** — §20.3's reconciliation of
  `native_metrics.py` and the `NATIVE_METRICS` comment was assigned to K3,
  but its *reason* clause ("the runtime has no integer dtype") expired at
  K2. K2 corrected the reason and left the conclusion; K3 still owns the
  `argmax` half. §7.3's heading was likewise corrected from K1 to K2, which
  is the milestone the work it describes always belonged to.

### K3 — Native `argmax` · complete

**Layers:** the new `cpp/src/indexing.cpp` and
`cpp/include/tf_indexing_internal.h`; `cpp/CMakeLists.txt` (the source list
is a glob, so only the CTest registration was edited); `backends/cpp.py`
(declaration, `_CHECKED_KERNELS`, `NativeTensorCore.argmax`,
`TENSOR_CORE_OPS`); `native_tensor.py` (`NativeTensor.argmax`); the §20.3
reconciliation of `native_metrics.py` and the `NATIVE_METRICS` comment.

**ABI +1 → 55.** CTests +1 → 26. `AUTOGRAD_OPS` unchanged.

**What landed, precisely.**

- **The traversal**, `tf::argmax_contiguous`, one template over the source
  element type in the new internal header, implementing §17.5's algorithm
  literally: `best` starts at `run[0]`, a strict `>` displaces it, a NaN
  displaces any non-NaN incumbent, and nothing displaces an incumbent NaN.
  It performs no arithmetic on values, allocates nothing, reads each element
  once per run, and never inspects a NaN's payload, sign, or signalling bit.
  The **output** type is not a template parameter and never becomes one: an
  index is `std::int64_t` at every source width.
- **The role guard**, `tf::require_index`, beside the traversal rather than
  in `tf_internal.h`, because it is the indexing family's question and K4
  reuses it. It completes the guard family: `require_float64` ("not
  generalized"), `require_floating` ("computing is floating-only"),
  `require_matching_dtype` ("operands must agree"), and `require_index`
  ("this operand is an index, and an index is exactly `int64`"). It is
  applied **instead of**, never beside, the two floating guards on the
  handle it governs.
- **The export**, `tf_core_argmax`, guarded with
  `TF_GUARD_BEGIN`/`TF_GUARD_END_VOID()` and validating in §22.8's order:
  null handles, source floating, destination `int64`, positive extents,
  checked products, offset sign, source span, an **exact** destination
  element count of `outer * inner`, aliasing, then one dispatch on the
  **source** dtype alone. The aliasing check is a deliberate backstop the
  role checks already make unreachable — one storage carries one dtype — and
  it is retained because the C ABI validates independently of what another
  check happens to imply.
- **The Core operation**, which validates the axis **before** `keepdims`
  (§17.6) and then asks the shared `reduce_shape` authority with arguments
  it can only accept, materializes a non-contiguous input through Policy-B,
  allocates a **zeroed** `int64` destination (§27.3), and closes every
  temporary and the destination under `BaseException`.
- **The tensor method**, which wraps the Core result as a plain leaf and
  never calls `_from_op`. `_from_op` would refuse to build an integer graph
  node anyway — that is K1's structural backstop — but the mechanism here is
  simply that no graph is built.

**What did not land, and why it is worth saying:** no `max`, no
`max_with_indices`, no tuple return, no second output handle, no `argmin`,
no backward export, no `AUTOGRAD_OPS` entry, no benchmark, no example, no
registry or version movement, and no change to `native_accuracy`'s runtime
behavior (§20.3).

### K4 — `index_select`, forward only · complete

**Layers:** the same two C++ files (`cpp/include/tf_indexing_internal.h`,
`cpp/src/indexing.cpp`), `cpp/CMakeLists.txt` (the source list is a glob, so
only the CTest registration was edited), `backends/cpp.py` (declaration,
`_CHECKED_KERNELS`, `TENSOR_CORE_OPS`, `_require_index_dtype`,
`_is_axis_int` / `_require_axis_int`, `NativeTensorCore.index_select`), and
`native_tensor.py` (`NativeTensor.index_select`).

**ABI +1 → 56 — the phase maximum.** CTests +1 → 27. `AUTOGRAD_OPS`
unchanged.

**What landed, precisely.**

- **The traversal**, `tf::index_select_contiguous`, one template over the
  *value* element type — the source and the destination together, which is
  sound because they are proved to agree — added **beside**
  `argmax_contiguous` rather than by generalizing it. The index pointer is
  always `std::int64_t` and is never a template parameter. It copies whole
  `inner`-element slices with `std::memcpy`, so every element crosses by
  **object representation**: it reads no value, performs no arithmetic,
  compares nothing, and therefore cannot canonicalize a signalling NaN or
  normalize a signed zero the way a floating assignment would be permitted
  to. It is `noexcept`, allocation-free, has no dtype branch inside the
  loop, and writes only inside `[0, outer * index_count * inner)`.
- **The export**, `tf_core_index_select`, guarded with
  `TF_GUARD_BEGIN`/`TF_GUARD_END_VOID()` and validating in §22.9's order:
  three null handles, source floating, destination floating,
  source/destination dtypes matching, index exactly `int64`, four positive
  extents, both checked products, both offset signs, the source span, the
  index span, an **exact** destination element count of
  `outer * index_count * inner`, the representability of one slice's byte
  count, both aliasing pairs, then the **complete** index bounds scan, then
  one dispatch on the **source** dtype alone. `tf::require_matching_dtype`
  is used here and only here, and only across the floating pair.
- **Its own validator**, `index_select_argument_error`, file-local and
  separate from K3's `argument_error`: three handles, two offsets, four
  extents, and two aliasing pairs are a different list, and one function
  covering both would have needed a mode flag. The two exports share only
  the operation-independent primitives — checked multiplication and
  addition, span containment, and the error-recording helper.
- **The Core operation**, which validates in §18.6's order — axis *type*,
  index operand *type*, source open, index open, source floating, index
  `int64`, axis range through the shared `_normalize_axis_checked`
  authority, index rank exactly 1, the **complete** bounds scan, then the
  output shape and count — with **everything before the allocation step
  allocating nothing**, proved by an allocator probe rather than by
  inspection. It materializes a non-contiguous source *or* index through
  Policy B as two separate temporaries, allocates a **zeroed** floating
  destination (§27.3), and closes every temporary in reverse allocation
  order and the destination under `BaseException`.
- **The tensor method**, which owns the one rule the Core cannot: a source
  with `requires_grad=True` is **rejected** with a message naming
  `detach()`, never silently detached. It wraps the Core result as a plain
  leaf and never calls `_from_op`.
- **Two small shared validators** on the Python side, each a sibling of an
  existing one rather than a new authority: `_require_index_dtype` (the
  Python half of `tf::require_index`) and `_require_axis_int`, which asks
  the **type** half of `_normalize_axis_checked`'s question on its own so
  that the type and the range can be reported at different points in one
  validation order. `_is_axis_int` is the single predicate both ask, so the
  accepted axis domain — Python `int`, NumPy integer scalar, `bool`
  rejected — cannot drift between them.

**What did not land, and why it is worth saying:** no general `gather`, no
`scatter`, no `scatter_add`, no embedding lookup, no `__getitem__` or
`__setitem__`, no advanced, boolean, or multi-axis indexing, no backward
export, no `AUTOGRAD_OPS` entry, no `argmin`, no `max`, no benchmark, no
example, and no registry or version movement. The gradient's contract is
already fixed by §18.9 for the separately approved phase that ships it.

### K5 — Compatibility proof · complete

**Zero production code**, and the whole milestone is one new test module,
`tests/test_native_integer_compatibility.py`, plus the status
reconciliation a landed milestone requires. It added no C ABI symbol
(still **56**), no public Python name (`__all__` still **25**), no CTest
(still **27**), no example (still **16**), no benchmark (still **9**), no
registry value, and no version of any kind. The only file it touched under
`src/` is the package docstring's Phase-K status sentence, which is
documentation and carries no capability.

**What it proves, against the live tree**, each claim driven end to end
rather than at the unit level the K1 and K2 modules already own:

- **No archive can declare an `int64` entry**, from both directions. Every
  archive a real `save_native_checkpoint` writes carries floating entries
  only — parameters, persistent buffers, optimizer parameter metadata, and
  both Adam moment families — at float64 and float32 independently, and
  the writer now **refuses to emit** a non-floating entry even from live
  state the public API cannot produce (see "the defect this proof found"
  below). A controlled malformed copy of a *real* archive that declares
  `int64` at a
  **parameter**, a **persistent-buffer**, an **optimizer-moment**, or an
  **optimizer-parameter** entry is rejected by the real parser, **before**
  any destination is published, leaving the destination's values,
  versions, gradients, moments, counters, generator states, and every
  object identity byte-identical, and without allocating a single `int64`
  storage — watched at the allocator rather than inferred.
- **The checkpoint format, current version, and accepted set are unmoved**
  — `tensorforge.native_checkpoint`, **3**, `(1, 2, 3)` — with **no
  version-4 constant written, reserved, or accepted**, proved by sweeping
  the module's own version constants rather than by reading one of them.
- **Versions 1 and 2 stay historical**: a version-1 archive still loads
  under its legacy rules (no generator section, bare-list moments), both
  keep their float64-only interpretation, neither may declare `int64`, and
  the two refusals carry *different* reasons because they are two rules.
  Versions 4, 5, 0, −1 and the malformed spellings (`"3"`, `3.0`, `True`,
  `None`) all reject through the established path.
- **Parameter, buffer, and optimizer state stay floating-only**, driven
  with a real tensor from K2's public door: `NativeParameter` from the
  tensor and by naming the dtype, `register_buffer` at **both** persistence
  values, and both optimizers'
  per-parameter dtype check beside the pre-existing type check — each
  rejection followed by a before/after fingerprint. The parameter role's
  authority is **located rather than assumed**: it is
  `NativeParameter.__init__` and only that. Neither
  `NativeModule.__setattr__` nor `register_parameter` re-checks a dtype —
  they delegate to `NativeParameterRegistry.register`, which validates the
  name and the type — so those paths are protected *by construction*, and
  the milestone measures that by driving both of them with a genuine
  `NativeParameter` forged through `__new__` over a real `int64` core
  rather than by asserting a barrier that is not there. **The
  optimizer-state
  version is unmoved at 1**, its live state is floating at every position,
  and an `int64` declared in its parameter metadata is refused as a
  mismatch against the live parameter — a second, separate authority from
  the archive's entry validator.
- **Loader and sampler state are unmoved** — `tensorforge.native_data_loader`
  and `tensorforge.native_sampler`, both version **1**, both accepting
  `(1,)` — with the exact key sets checked as *sets* (a substring ban would
  fire on the format tags themselves), the only dtype-valued field being
  the dataset's floating `feature_dtype`, nothing that grows with the
  sample count, a valid round trip on both objects reproducing the exact
  remaining order, and every malformed or unsupported version refused with
  the position left exactly as it was.
- **Phase J still delivers `(NativeTensor, numpy.ndarray of dtype int64)`**
  at both feature widths, with the target proved to be read-only host
  metadata and not a native tensor, the ordering and batching still the
  sampler's plan, and **no option anywhere** — on the dataset, the sampler,
  the loader, or its private iterator — through which a native label could
  be requested.
- **Explicit caller conversion works and needs no pipeline change.**
  `NativeTensor.from_int64_array(delivered_targets)` accepts the delivered
  read-only array, preserves every value exactly, is independent of the
  host array it came from, and is consumed by `index_select` — and the
  result is *still* refused as a parameter, a buffer, an optimizer's
  charge, a cross-entropy target, and a checkpoint entry. This is a
  caller's line of code **after** delivery and is never described as loader
  behavior.
- **`NativeCrossEntropyLoss` is behaviorally unchanged**: the same accepted
  host-target forms, every one of which gives *bit-identical* losses so the
  accepted set is one contract rather than several that agree numerically;
  the same rejected forms; correct values and gradients at each width
  against a host oracle; bit-identical repeat calls; and a native `int64`
  target refused by three separate routes — the explicit door, a fresh
  `argmax` result, and a view of one — at the same shared host boundary.
- **`native_accuracy` is behaviorally unchanged**, and the proof is the
  strong form: with `NativeTensor.argmax`, `NativeTensor.index_select`, and
  both `NativeTensorCore` counterparts patched to raise, the metric still
  succeeds, which is only possible if it calls none of them. It
  materializes through `to_numpy()` exactly once, allocates no `int64`
  storage, builds no graph, mutates neither logits nor targets, and still
  inherits NumPy's tie and exceptional-value conventions — asserted against
  a NumPy oracle rather than against `NativeTensor.argmax`, because §20.3
  declines to claim the two rules are equivalent.
- **Deterministic training, checkpointing, and resume stay bit-identical
  while `argmax` and `index_select` are used beside the training path.** A
  real classifier — trainable parameters, persistent BatchNorm buffers, a
  shared and an owned generator, Adam — trains a fixed schedule that
  crosses an epoch boundary, is interrupted **genuinely mid-epoch** with
  batches still owed, saved through the real checkpoint with the loader
  state as ordinary caller metadata, and resumed into an entirely fresh
  object graph that is **proved different before the load**. The
  uninterrupted and resumed runs agree bit for bit in every parameter,
  buffer, optimizer moment, counter, scalar, generator state, loader and
  sampler state, and in every per-step logits, loss, gradient, argmax index
  vector, and index-select selection — independently at float64 and
  float32, each dtype compared only against itself, through
  `uint32`/`uint64` views and never a tolerance. The omitted-loader-state
  leg is proved to **diverge**.
- **The evaluation path is observationally neutral.** An otherwise
  identical uninterrupted control that omits the indexing calls ends in a
  bit-identical trainable state, which is what proves the two operations
  consume no RNG draw, mutate no model, optimizer, or pipeline state,
  create no graph edge, and cannot change what a checkpoint would contain.
- **The evaluation path is exactly** `predictions = logits.argmax(axis=1)`
  followed by `logits.detach().index_select(1, predictions)`: the `argmax`
  is taken from the live gradient-tracking logits because §17.9 promises a
  plain leaf even then, and the `index_select` from a **detached** source
  because §18.9 rejects a `requires_grad=True` one rather than detaching it
  silently. `index_select` selects the same supplied index vector along one
  axis for **every** outer slice — it is not a per-row gather — so for a
  batch of `B` the result is `(B, B)` and the *diagonal* is each example's
  own predicted-class logit, verified on the host, with every column
  checked against the whole predicted-class column so duplicates and order
  are proved preserved. `BATCH > CLASSES` guarantees duplicate predicted
  classes by pigeonhole without the fixture distorting the model.

**The defect this proof found, and the chronology.** Driving the two
module registration routes for real — with a `NativeParameter` forged
through `__new__` over a real `int64` core, which is the only way to reach
them, because the public constructor refuses one — showed that
`save_native_checkpoint` trusted whatever dtype live state reported. A
module forced into that state produced an archive **declaring an `int64`
model entry**, which the loader then refused: the writer could emit a file
its own reader rejects. That was a **pre-existing gap in the writer**, not
something Phase K introduced and not reachable through any public API, and
the compatibility proof is what exposed it.

It was repaired in a **separate checkpoint-hardening change, committed
before K5**, so this milestone remains the test-and-documentation
compatibility milestone it is described as. The repair adds a save-side
persisted-dtype authority to `native_checkpoint.py` —
`_canonical_persisted_dtype`, the one question both sides now ask through
`cpp.normalize_dtype`, and `_validated_persisted_dtype`, the save-side
validator built on it — applied twice: in `_validate_model`'s **preflight**
over every `_state_named_tensors()` entry in traversal order, before a
single snapshot exists, and again at `_coherent_snapshot`'s
**serialization seam** over each model snapshot, each optimizer
parameter-metadata entry, and each Adam `m` and `v` snapshot, before the
manifest or the collision-safe temporary file exists. A rejection creates
no archive, leaves an existing destination byte-for-byte intact, leaves no
temporary behind, mutates no live state, and closes every snapshot already
taken. `_validated_entry_dtype`'s load behavior is unchanged and now
routes its rule 1 through the same shared question, so reader and writer
cannot drift apart. **No format, field, version, capability, registry,
export, CTest, example, or benchmark moved**, and the repair's own
regression lives in `tests/test_native_checkpoint.py` — the checkpoint
owner — not in the compatibility module, so the two changes stay
independently reviewable.

The forged parameter is **test-only and is never supported public usage**.
It exists because a schema guarantee has to survive state the public API
cannot produce: "no reachable caller can violate it" and "the serializer
enforces it" are different guarantees, and only the second makes the
archive schema true of the file rather than of the caller. The contract is
not narrowed to ordinary models to accommodate the finding.

Every scanner, fingerprint, and instrument in the module has a **negative
control**, including the live-storage tracker and the one narrow allowance
the module makes: a block that trained settles its count with a collection,
because an autograd graph's internal nodes are framework-owned cycles
rather than caller-closed objects, and a separate control proves a
collection cannot launder a tensor the test still holds and never closed.
Blocks that build no graph use the strict tracker.

This milestone replaces the notion of a separate serialization milestone.
There is no serialization work to do, and a milestone that assumed there
was would invite a format change nobody needs.

### K6 — End-to-end integration example and exact proof · complete

**Zero production code**, and the whole milestone is two new files —
`examples/native_integer_indexing.py` and its owner
`tests/test_native_integer_indexing_example.py` — plus the inventory and
status reconciliation a landed milestone requires. It added no C ABI symbol
(still **56**), no public Python name (`__all__` still **25**), no CTest
(still **27**), no benchmark (still **9**), no registry value, and no
version of any kind. It moved **exactly one** inventory: examples 16 →
**17**. The only file it touches under `src/` is the package docstring's
Phase-K status sentence, which is documentation and carries no capability.

**The program.** A deterministic native classifier —
`NativeLinear(5 → 8) → NativeReLU → NativeLinear(8 → 4)` with
`NativeCrossEntropyLoss` and `NativeAdam` — trains over the Phase-J pipeline
(`NativeTensorDataset` → `NativeBatchSampler` → `NativeDataLoader`) on
twenty-four fixed samples whose every feature value is a multiple of one
eighth, so the same literals seed both widths exactly. Ten shuffled steps
in batches of six cross two epoch boundaries; the interruption lands after
five, at epoch 1 cursor 1, with three batches still owed by the *active*
epoch. It is deliberately **smaller** than the Phase-J mini-batch
classifier: K6's subject is the indexing, and an extra stochastic or
normalizing layer would have added state to the resume proof without adding
anything to the indexing one.

**The evaluation path, at fixed steps on both sides of the interruption**
(1 and 4 before the checkpoint, 6 and 9 after it), taken from the step's own
logits so no second forward pass can be mistaken for the thing being
measured:

```python
predictions     = logits.argmax(axis=1)                       # K3, int64
detached_logits = logits.detach()
selected        = detached_logits.index_select(1, predictions)  # K4
```

The `argmax` reads the **live, gradient-tracking** logits because §17.9
promises a plain leaf even then; the `index_select` reads a **detached**
source because §18.9 rejects a `requires_grad=True` one rather than
detaching it silently. The two deliberately different sources are the
example's most instructive line, and the owner test drives the rejection
directly rather than only describing it.

**`index_select` is axis selection, not a per-row gather, and the example
says so in those words.** With logits of shape `(6, 4)` and predictions of
shape `(6,)` the result is `(6, 6)`: the *same ordered index vector* is
selected along the class axis for **every** row, so column *j* is the whole
source column `predictions[j]` and each example's own predicted-class logit
sits on the **diagonal**. The owner test rebuilds the whole result from the
recorded bit patterns and checks **every column** against its source column
and the diagonal against `logits[row, predictions[row]]` — recomputed rather
than read out of the example's own booleans. `BATCH_SIZE > NUM_CLASSES`
makes duplicate predicted classes a pigeonhole guarantee rather than luck,
and where an index repeats the two columns are proved bit-identical in their
original positions, which is the observable form of "duplicates and order
are preserved".

**What the proof compares.** For each dtype independently: an uninterrupted
run, an interrupted-and-resumed run through a real version-3 archive into an
entirely fresh object graph proved different *before* the load, and an
omitted-loader-state negative control proved to **diverge**. The two runs
agree on the completed step count, the delivered batch-index sequence, the
loader and sampler position, every parameter bit pattern, every Adam moment
and counter, every per-step logits and loss, every recorded prediction
index, every recorded selected value, and every evaluation shape and
metadata fact. **Prediction indices are compared as exact Python integers
and are never converted to a floating value**; floating values are compared
through `uint32`/`uint64` views. The **only** cross-dtype claims gated are
dtype-independent — the batch schedule, the permutations, the positions, the
evaluation steps, and the selection shapes. Whether the two widths happen to
predict the same classes is reported as an **observation** and deliberately
not required: it is legitimate for them to differ, and each run must still
reproduce itself exactly.

**Discipline.** The example is written against the **public** experimental
surface only, proved by an AST scan with a planted negative control; it
names no private runtime seam, assigns no private state, calls no
`numpy.argmax` (the indices come from the native operation, and a runtime
counter proves it), claims and measures no timing, reads no file, clock,
environment variable, or global RNG, closes every native object it creates
— the `argmax` result, the detached source, the `index_select` result, every
delivered batch, forward output, loss, and gradient — under `try`/`finally`,
leaves no file behind, and returns live native storage exactly to its
baseline. The storage tracker has its own non-vacuity control proving it
notices a deliberately retained tensor.

**No K7+ work landed here.** There is no allocation-failure injection, no
malformed-metadata matrix, no reentrancy or concurrency test, no benchmark,
and no closure claim; K7's, K8's, and K9's modules are asserted **absent**.

### K7 — Adversarial hardening · complete

`tests/test_native_integer_hardening.py`: §27's four injection positions at
every allocating path, the complete before/after world fingerprint after
every rejection, a `BaseException` through each cleanup, malformed-metadata
C-side negative controls, and a non-vacuity control for every injection and
every parser. **No production code** — and if it finds a defect, that
defect is fixed here and reported, rather than absorbed.

**What landed.** One new module, `tests/test_native_integer_hardening.py`,
plus status reconciliation, and **zero production code**. The four §27.2
families are resolved against the live call graph rather than assumed, and
recorded in the module as a traceable path-by-position matrix that a
guardrail in the same module checks: every row names either an owner test
that exists or an `N/A` with the technical reason the family cannot exist
on that path — `from_int64_array` runs no compute kernel, and `int64`
`contiguous_copy` has no host operand and no host→native transfer, so
neither is faked with a neighbour's injection.

Each of the four attacked paths is driven at **every** actual allocating
step, separately: `from_int64_array` at host validation, at the index
registry gate, at the `int64` storage allocation, at the host→storage
copy, at core/view construction, and at public publication; `int64`
`contiguous_copy` at the destination allocation, at
`tf_core_contiguous_copy`, and at publication; `argmax` at the Policy-B
temporary, at the `int64` destination, at the materialization kernel, at
`tf_core_argmax`, and at publication; `index_select` at the source
temporary, at the **index** temporary, at the destination, at **both** of
its Policy-B materialization call sites, at `tf_core_index_select`, and at
publication.

Those last two are **two rows, not one**, and the distinction is the
substantive one this milestone insisted on. `index_select` reaches
`tf_core_contiguous_copy` twice on a both-strided call — once through
`self._contiguous_temp` for the floating source, once through
`indices._contiguous_temp` for the `int64` index — and an injection that
fails the export immediately can only ever reach the **first**. The second
is therefore driven by a call **journal** that delegates call 1 to the real
export, so the failure happens with the source temporary genuinely
materialized, and the journal proves the two calls are distinguishable by
dtype, element count, and rank rather than by ordering alone. Both sites
run at float64 and float32 and under both an `Exception` and a
`BaseException`; the index-site failure additionally proves that **no
destination is allocated** after it and that both temporaries are closed
while the test still holds them. One representative failure is not allowed
to stand for both.

Every allocation row fires the backend's **own** thread-local arm, armed
immediately before the production seam it names so the position is exact
and no ordinal counting of unrelated allocations is involved; every kernel
and publication row runs under both an `Exception` and a `BaseException`
and at **both** floating widths, each width proved only against itself,
and retains the allocated core or storage in an external list so its
closure is proved by production cleanup **while a strong reference still
holds it**. The three-allocation `index_select` failure proves
reverse-order cleanup and a per-object release count proves **exactly
once** rather than merely *closed* — a cleanup **invariant** at a position
the matrix already names, deliberately traced as such rather than added as
a row, because re-describing one physical seam as two would inflate the
matrix without attacking anything new.

Around every rejection and every injected failure the module compares one
reusable fingerprint of the observable world — both operands by identity,
layout, graph state, gradient, and raw payload bytes; an unrelated
parameter with its version and gradient; a persistent **and** a
non-persistent buffer; a live optimizer; a registered generator; every
capability registry, dtype table, operation inventory, format, and
version; `experimental.__all__`; both global RNGs; the environment; a
watched directory; the live-storage count; and the native error slot —
and every component has a perturbation control proving it can notice the
change it exists for. That "every" is **literal, and checked**: a scan of
the module's own AST requires each non-`N/A` matrix row's owner to enter
the fingerprint, so a narrower check can never be reported as the complete
one. Where a rejection needs a deliberate instrument — an emptied
`INDEX_DTYPES`, a lowered `_INT64_MAX` — the instrument is applied first
and the snapshot taken after it, so what is proved unchanged is what the
*rejected call* touches, and the instrument is restored in a `finally`
with the ordinary path exercised afterwards.

The two exports keep **separate** malformed-metadata matrices *and*
separate dtype-role matrices, held to the same standard: every rejection
prefills every operand with distinctive values and asserts not one byte
moved in **any** of them — two handles for `argmax`, three for
`index_select` — allocates no native storage, and leaves the error slot
clean once the `errcheck` hook has run. The valid controls are preserved
and prove the permitted role combinations really execute: `argmax`'s
mixed-role call, which either forbidden guard would reject, and
`index_select`'s matching floating pair with an `int64` index. A late
invalid index following three valid ones is proved to leave the whole
destination byte-identical at both widths.

**K7 found no production defect**, and it moved no inventory: exports
**56**, CTests **27**, examples **17**, benchmarks **9**, `__all__`
**25**, every registry and every version exactly what K6 left. No native
build was performed or required. The only file it touches under `src/` is
the package docstring's Phase-K status sentence.

### K8 — Benchmark characterization · complete

`benchmarks/benchmark_native_integer.py` under §31 in full, owned by
`tests/test_native_integer_benchmark.py`. **Benchmarks 9 → 10. No
production code, no optimization.**

**What landed.** Two new files and nothing else outside them: the harness
and its owner. The harness answers §31's four questions as **four
workload families** — `integer_construction`, `host_materialization`,
`argmax`, `index_select` — over **sixteen** cases pinned in an exact
ordered inventory the owner test writes down independently. There is
**no composed case**: §31 permits one and this milestone declines it,
because a single `argmax`-then-`index_select` number cannot say which of
the two dominates, and the label "composition" would document that
weakness rather than remove it.

**Every case is `native_only`, and the registry has exactly one member.**
Each of the four families allocates native storage and transfers into or
out of it while the apparent host equivalent does not — construction
against `numpy.array`, materialization against a host-to-host copy,
`argmax` against `numpy.argmax` over an existing array, `index_select`
against `numpy.take` — so no honest denominator exists for any of them.
`argmax` is the one §31 names by name as the live fairness risk, and it is
resolved by publishing no ratio rather than by qualifying one. A second
reference label would have been an unused invitation, so the registry is
`("native_only",)` and the payload's `ratio_to_reference`,
`ratio_meaning`, and `reference` are literal `None` in the single place a
record is built — asserted off the AST, so no arithmetic could ever
produce one.

**The oracles are the harness's own.** `argmax` is gated against a direct
transcription of §17.5's algorithm and against that section's committed
**twelve-row** case table — unique maximum, equal maxima, both signed-zero
orders, all `-inf`, `+inf`, one NaN, several NaNs, a NaN against either
infinity, a NaN at index 0, and a length-1 run — run as known answers at
**both** floating widths inside every `argmax` gate. The rule is
transcribed rather than delegated: the owner test proves `argmax_run`
touches no array library, and proves the table *discriminates* by showing
that a skip-NaN rule and a last-maximum rule each fail it. That the
answers happen to coincide with `numpy.argmax` on these rows is a
coincidence §17.5 explicitly declines to promise, and nothing depends on
it. `index_select` is gated against a per-position slice concatenation
written without `numpy.take`, compared as whole arrays **and** position by
position, so a deduplicated, sorted, or reordered result fails.

**Correctness precedes timing, and it is proved twice.** Structurally, off
`run_case`'s AST: `check()` is called before `measure`. Behaviourally, with
a spy timer, for **every one of the sixteen cases**. Seven planted-defect
controls — a perturbed result at each family, a truncated shape, a
deduplicated selection, and a result claiming a gradient — each abort in
the gate with the timer never reached, clean stdout, and live storage
exactly at baseline; a sitecustomize-shim CLI run proves the same from
outside the process, with a nonzero exit and nothing on stdout.

**The timed region is exactly the operation**, read off the AST: each
family's `run` closure is a single `return` of a single call, and the four
expressions are pinned literally. A non-contiguous operand's internal
Policy-B materialization, the complete index bounds scan, and the
destination allocation stay **inside** that call, because they are part of
the operation; the harness calls no copy helper and reaches for no private
seam, which the owner test scans for. Setup and every `close()` are
outside the timer, proved behaviourally with a phase-aware clock. Every
measured sample is retained, no outlier is removed, and no timer overhead
is subtracted — the median is the headline and the IQR the spread, both
checked against known answers.

**The three measured dtypes stay three separate questions.** `--dtype`
names a *measured* dtype and its help text says so: `int64` selects the
index/result families and `float64`/`float32` select the floating source
width of `argmax` and `index_select`. Every record carries a `dtype_role`,
no case straddles the two registries, no payload key names two of the
three dtypes, and the owner test bans the wording that would imply
`SUPPORTED_DTYPES` had grown.

**K8 found no production defect**, performed no native build, and moved
one inventory: benchmarks **9 → 10**. Exports **56**, CTests **27**,
examples **17**, `__all__` **25**, every registry, and every version are
exactly what K7 left, and the only file it touches under `src/` is the
package docstring's Phase-K status sentence.

### K9 — Cross-platform validation and Phase-K closure

Fresh Windows Release and Debug builds with zero project warnings and the
full CTest suite green in each; the Clang ASan/UBSan matrix with
instrumentation **proved present**, a negative control proving the detector
can fail, the sanitized Python suites with zero diagnostics, and a
LeakSanitizer lifecycle returning live storage exactly to baseline; the
WSL/Linux CI-equivalent run; the §29.5 Windows/Linux equality proof; the
permanent closure guardrails in `tests/test_native_phase_k_closure.py`; and
the final inventory reconciliation. **No capability.**

---

## 33. Per-milestone API, ABI, inventory, and version deltas

`__all__` is 25 and `AUTOGRAD_OPS`, `RAW_KERNELS`, `TENSOR_CORE_KERNELS`,
`SUPPORTED_DEVICES`, `UNSUPPORTED`, `RAW_KERNEL_DTYPES`, the checkpoint
format and versions, the optimizer-state version, and the loader and
sampler state versions are unchanged in **every** row below. Only the
columns that move are shown.

| Milestone | Public Python | C ABI | CTests | Examples | Benchmarks | `SUPPORTED_DTYPES` | `INDEX_DTYPES` | Barriers | `TENSOR_CORE_OPS` |
|---|---|---|---|---|---|---|---|---|---|
| K0 | — | 54 | 24 | 16 | 9 | `(f64, f32)` | absent | none | — |
| K1 | — | 54 | **25** | 16 | 9 | `(f64, f32)` | absent | **all** | — |
| K2 | `from_int64_array`, `item`, `tolist` | 54 | 25 | 16 | 9 | `(f64, f32)` | **`("int64",)`** | all | — |
| K3 | `argmax` | **55** | **26** | 16 | 9 | `(f64, f32)` | `("int64",)` | all | **+`"argmax"`** |
| K4 | `index_select` | **56** | **27** | 16 | 9 | `(f64, f32)` | `("int64",)` | all | **+`"index_select"`** |
| K5 | — (zero production code) | 56 | 27 | 16 | 9 | `(f64, f32)` | `("int64",)` | all | — |
| K6 | — | 56 | 27 | **17** | 9 | `(f64, f32)` | `("int64",)` | all | — |
| K7 | — | 56 | 27 | 17 | 9 | `(f64, f32)` | `("int64",)` | all | — |
| K8 | — | 56 | 27 | 17 | **10** | `(f64, f32)` | `("int64",)` | all | — |
| K9 | — | 56 | 27 | 17 | 10 | `(f64, f32)` | `("int64",)` | all | — |

The **first milestone at which an `int64` tensor can be constructed is
K2**, and the Barriers column reads "all" from **K1** onward. That
inequality is §32.1's invariant, tabulated so a guardrail can check it
rather than read prose.

**No public promise moves before its proof, and `SUPPORTED_DTYPES` never
moves at all.** The only public registry movement in the phase is
`INDEX_DTYPES` appearing at **K2**, in the same commit as the public
constructor and one milestone *after* every barrier in §6.5 has landed. The
`int64` representation is reachable only through the raw private C ABI at
K1, then through the one public `NativeTensor.from_int64_array` constructor
backed by private Storage/Core helpers at K2 — the deliberate rollout
pattern this repository has used twice.

**56 is the Phase-K maximum export count.** A milestone that needs a third
symbol is a milestone this contract does not authorize.

---

## 34. Phase exit gate

Phase K may be declared complete only when **every** item holds.

**Capability**
1. `SUPPORTED_DTYPES == ("float64", "float32")` — **unmoved** — and
   `INDEX_DTYPES == ("int64",)`, both reported by `backend_info()`, with
   `normalize_dtype("int64")` still raising `ValueError`.
2. `SUPPORTED_DEVICES == ("cpu",)`, `UNSUPPORTED == ("cuda", "amp")`,
   `RAW_KERNEL_DTYPES == ("float64",)`, `backend_info()["dtype"] ==
   "float64"`, `normalize_dtype(None) == "float64"`.
2a. `set(_DTYPE_CODES) == set(SUPPORTED_DTYPES) | set(INDEX_DTYPES)` — the
   Phase-I no-drift guard, generalized rather than deleted (§5.1).
2b. Every generic constructor of §5.4 still rejects `"int64"`, and public
   `NativeStorage(size, dtype="int64")` still raises (§5.5).
3. Exactly **56** exported production `tf_*` symbols, source inventory and
   built library agreeing.
3a. `tf_core_argmax` rejects a non-floating source and a non-`int64`
   destination, and applies **neither** `require_floating` **nor**
   `require_matching_dtype` to its destination — proved by a valid call
   succeeding, which those checks would have rejected (§22.8).
3b. `tf_core_index_select` rejects a non-floating source, a non-floating
   destination, a source/destination dtype mismatch, and a non-`int64`
   index, and scans every index before writing anything (§22.9).
3c. `NativeTensor.from_int64_array` is the **only public construction or
   host-ingress door** for an `int64` buffer, and it asks
   `cpp._normalize_index_dtype` as its registry gate (§26.1 step 2a);
   `NativeStorage._from_int64_array` and
   `NativeTensorCore._from_int64_array` are private and unexported, and
   public `NativeStorage(size, dtype="int64")` still raises.
3d. The K2 public delta is **three** `NativeTensor` names —
   `from_int64_array` (the one door) plus the dtype-general
   host-inspection methods `item()` and `tolist()`, which construct
   nothing — and no surface describes it as one name.
4. `tensorforge.experimental.__all__` has exactly **25** names.
5. Checkpoint format version **3** with `(1, 2, 3)` accepted; optimizer
   state version **1**; loader and sampler state version **1** with `(1,)`
   accepted. **No version 4 constant exists anywhere.**

**Correctness**
6. Every §6.5 barrier proved by executable rejection, each leaving live
   storage exactly at baseline.
7. `argmax` proved against §17.5's table **row by row** — unique maximum,
   equal maxima, both signed zeros, all-`-inf`, `+inf`, one NaN, several
   NaNs, NaN with finite values, NaN with either infinity, NaN at index 0,
   and a length-1 run — at float32 and float64
   **separately**, contiguous and non-contiguous, full and per-axis, with
   and without `keepdims`, rank 0 through rank 4.
8. `index_select` proved against §18 in full — duplicates, order, bounds
   rejection before allocation, negative rejection, rank rejection,
   non-contiguous operands, `requires_grad` rejection — with floating
   values compared as raw IEEE-754 bits.
9. Exact integer round-trip through construction, views,
   `contiguous_copy`, `to_numpy`, `item`, and `tolist`, including
   `_INT64_MIN`, `_INT64_MAX`, and negative values.
10. `AUTOGRAD_OPS` unchanged; neither new operation appears in it.

**Integration**
11. A model trains, checkpoints, and resumes **bit-identically** with
    `argmax` and `index_select` in the evaluation path, at float32 and
    float64 independently (K6).
12. The Phase-J loader still delivers `(NativeTensor, numpy.int64)`, and no
    Phase-J production file changed anywhere in the phase.
13. Stable / native isolation intact; `stable_framework_integration` is
    `False`; `import tensorforge` loads no native library.

**Evidence**
14. Windows Release **and** Debug builds with zero project compiler,
    linker, and CMake warnings, and the full CTest suite green in each.
15. Clang ASan/UBSan with instrumentation **proved present**, a negative
    control proving the detector can fail, zero diagnostics from the
    sanitized native and Python suites, and a LeakSanitizer lifecycle
    returning live storage exactly to baseline with no suppression file.
16. The WSL/Linux CI-equivalent run green, and §29.5's Windows/Linux
    integer equality asserted directly.
17. The full `uv run pytest` suite green with zero skips.
18. The benchmark's correctness gates pass, it writes no result file, and
    it asserts no speed.
19. No status surface over-claims: no document says integer arithmetic,
    integer gradients, integer parameters, integer checkpoint state, a
    checkpoint version 4, `argmin`, `gather`, embedding, CUDA, or AMP
    exists.

---

## 35. Explicit non-goals

Permanently outside Phase K. Each would require a **separately approved**
phase with its own design contract; none may be described as begun,
planned, or scheduled.

**Dtypes** — `int32`, `int16`, `int8`, `uint8` and every unsigned integer,
`bool`, every complex dtype, `float16`, `bfloat16`, and any dtype-promotion
lattice.

**Conversion** — casting between any two tensor dtypes, promotion,
mixed-dtype arithmetic, `astype`, `to`, `.int()`, `.long()`, `.float()`,
`.double()`, `map_location`, dtype inference from a host array, and a
global default dtype.

**Integer numerics** — addition, subtraction, multiplication, division,
modulo, negation, absolute value, integer reductions, integer comparison
operations, integer matmul, integer convolution, integer pooling, integer
normalization, integer dropout, and integer optimizer math. Overflow is
consequently outside the implemented surface (§11.4).

**Index-producing** — `argmin`, `nonzero`, `sort`, `argsort`, `top-k`,
`unique`, `where`, `searchsorted`, `bincount`, `cumsum`.

**Index-consuming** — advanced indexing, `__getitem__` / `__setitem__` with
tensors, boolean masks, arbitrary `take`, multi-axis `gather`, `scatter`
and `scatter_add` as public operations, `index_select`'s **backward**, and
`embedding` as an operation or a module.

**Autograd and state** — integer gradients, integer parameters, integer
optimizer state, integer persistent or non-persistent buffers, integer
checkpoint entries, and checkpoint version 4.

**Pipeline and classification** — native `int64` targets in the data
pipeline, a loader or dataset option producing them, a native accuracy
computation, confusion matrices, and any change to
`NativeCrossEntropyLoss`'s host-target contract.

**Execution** — CUDA and every GPU backend, AMP and mixed precision, device
movement, a second device value, streams, events, asynchronous execution,
workers, prefetch, pinned memory, thread-safety claims, distributed
training, and sparse tensors.

**Runtime architecture** — a memory pool, scratch workspace or arena, a
persistent cache of native storage, SIMD intrinsics, threading, OpenMP,
BLAS, oneDNN, Eigen, im2col, general operator fusion, fast-math, cache
blocking, pybind11, a C++ ABI, any new required dependency, any new build
option, and any public performance control of any kind — no path selector,
threshold setter, dispatch tracer, profiling counter, benchmark hook,
"which path ran" query, or environment-variable dispatch.

**Also permanently absent, and asserted absent** — an exported
storage-dtype query symbol, a second dtype authority of any kind, an
integer poison or fault-injection hook beyond the one documented
allocation-failure arm, a public `is_integer` / `is_floating` property, a
public dtype object, and a `device` argument anywhere.

---

## 36. External-reference use policy

TensorForge is written from its own contracts. From time to time an
external implementation of a comparable runtime may be consulted as a
**private comparative engineering input** — a way to see how a problem has
been approached elsewhere before deciding how it should be approached here.

The policy is unconditional:

1. **No external project's name, owner, URL, file path, or identifier
   appears anywhere in this repository** — not in source, not in comments,
   not in documentation, not in tests, not in commit messages. The
   guardrails scan repository-owned text for exactly this and fail on it.
2. **No code is copied.** Any idea judged useful is re-derived against
   TensorForge's own contracts, in TensorForge's own vocabulary, and
   validated by TensorForge's own tests. An idea that cannot survive that
   re-derivation is not adopted.
3. **Every adopted idea must be independently justified in this document.**
   If the only argument for a decision is that someone else made it, it is
   not an argument, and the decision is not made.
4. **Weaker behavior is rejected explicitly, not silently.** Where this
   contract is stricter than a common approach — validating every index
   before allocating a destination rather than while copying; refusing
   silent host casts at an integer boundary; refusing negative index
   wraparound; refusing a second integer width; refusing to expose a
   maximum alongside its index — the reason is written down here so a later
   reader can weigh it rather than guess at it.
5. **No comparison with an external project may be recorded here**, either
   favourably or unfavourably. This document states what TensorForge does
   and why.

This section is generic by design and names nothing, because naming
something would itself violate rule 1.

---

## 37. Closure guardrail expectations

### 37.1 What `tests/test_native_phase_k.py` locks at K0

- **Status and history** — Phase K is newly approved *after* Phase J;
  Phase J is still complete; K0 is the **only** complete Phase-K milestone;
  K1 through K9 are unstarted; K0 is explicitly zero-runtime; no document
  claims an integer runtime exists.
- **Registries** — `SUPPORTED_DTYPES`, `SUPPORTED_DEVICES`, `UNSUPPORTED`,
  `RAW_KERNEL_DTYPES`, the default dtype, and
  `stable_framework_integration`, written independently of the module under
  test; `INDEX_DTYPES` absent; `normalize_dtype("int64")` raising.
- **Taxonomy and reachability** — the design selects taxonomy **B**
  explicitly; every generic constructor of §5.4 is named with its
  resolution; public `NativeStorage(size, dtype="int64")` is stated
  prohibited; and the ladder cannot move `int64` into a broad registry
  because `SUPPORTED_DTYPES` is identical in every row of §33.
- **Ordering** — every §6.5 barrier's milestone strictly precedes the
  first milestone at which an `int64` tensor is constructible, checked by
  parsing both tables rather than by reading the prose.
- **Inventories** — 54 exports, 25 experimental names, 24 CTests, 16
  examples, 9 benchmarks.
- **Versions** — checkpoint format/version/accepted set, optimizer state,
  loader state, sampler state; no future-version constant; no integer
  checkpoint field.
- **Runtime absence** — no public integer tensor name, no integer runtime
  module, no integer C++ production source, no integer or `argmax` or
  `index_select` C ABI symbol, no public integer constructor, no `int64`
  entry in `_DTYPE_CODES` / `_DTYPE_NUMPY` / `TfDtype` / any public
  registry, no casting or promotion operation, and no CUDA/AMP change —
  each pattern narrow enough not to reject the **legitimate existing** host
  `int64` target metadata, layout arrays, and class-label bindings.
- **Isolation** — the stable public API, stable import behavior, and stable
  serialization unchanged.
- **Design completeness** — every required topic is a real section with a
  real contract; the object-model alternatives are compared and one is
  selected; the taxonomy is unambiguous; the initial dtype is exactly
  `int64` with the others deferred; construction rejects silent casting;
  the autograd, parameter, and optimizer boundaries are explicit; the
  arithmetic scope is explicit; the `argmax` and `index_select` contracts
  are complete; the Phase-J default is preserved; the checkpoint position
  is explicit; the ABI maximum and per-milestone deltas are explicit; the
  exit gate and non-goals are explicit.
- **Over-claim scanners** that distinguish a negated or planned statement
  from a claim of current implementation, each with positive **and**
  negative controls.
- **An external-reference scanner** over repository-owned text only,
  shallow-clone safe, with a negative control.
- **A milestone-ladder parser** proving K0 complete, K1–K9 unstarted,
  identifiers unique and ordered, each row carrying scope, and a closure
  milestone present.
- **The `argmax` NaN rule** required to cover every case in §17.5's table —
  one NaN, several NaNs, NaN with finite values, NaN with either infinity,
  full and axis reductions, and contiguous and non-contiguous input — by
  required structure rather than by frozen prose.

### 37.2 What every later milestone must keep true

Each milestone moves names from the "absent" list to the "present" list in
**one** direction and never loosens a checker to do it. A milestone that
would require weakening an assertion instead of moving an entry is a
milestone whose scope is wrong.

### 37.3 What `tests/test_native_phase_k_closure.py` will own at K9

That every ladder row is complete; that every status surface says so; that
the final registries, inventories, and versions cannot drift; that the
export count is exactly 56; that `__all__` is exactly 25; and that closure
is **not** permission to name a successor — an invented milestone beyond
K9, or a phase after this one, would be a roadmap promise nobody
approved.

### 37.4 What no guardrail may do

No Phase-K test may assert an exact total pytest count, a benchmark number,
an error message's exact wording, a paragraph order, a line number, or a
character count other than the `CLAUDE.md` ceiling the project already
enforces. No Phase-K test may require network access, a Git ancestor, a
complete Git history, or an external package.
