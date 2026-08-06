# Roadmap

## Where the project is

**v3.0 is the completed Python framework release.** You
can define a model (including CNNs), train it, regularize it,
normalize it, evaluate it honestly, save it, and resume it bit-for-bit
— all from readable NumPy code, all tested, all documented. For the
version-by-version story, see [release_history.md](release_history.md);
for a two-minute overview, see [project_summary.md](project_summary.md).

## What's been built

**v0.x — foundations.** The Tensor and autograd engine (elementwise
ops, matmul, exp/log/tanh/sigmoid/relu/softmax with broadcasting-aware
gradients), the module system (Parameter, Module, Linear, activations,
Sequential), SGD, MSE and cross-entropy losses, and the first
examples: linear regression, XOR, and the multi-class spiral.

**v1.x — training basics and evaluation.** The accuracy metric, Adam,
mini-batching, gradient checking against finite differences,
save/load parameters, model summaries and parameter counting, frozen
parameters, train/validation splitting, evaluation helpers, binary
cross-entropy with a binary classification example, and checkpoints
that capture optimizer state so training can resume exactly.

**v2.x — deeper framework features.** Train/eval mode, Dropout (and an
example that uses it properly), eval-safe evaluators, BatchNorm1d with
module buffers, gradient clipping, the StepLR scheduler, scheduler
state in checkpoints — completing the training-resume story — and
image-shaped input: Conv2d, MaxPool2d, Flatten, and a tiny CNN
example; LayerNorm as the batch-independent normalization; optional
RNG state in checkpoints for bit-exact dropout resume; and a
release-readiness pass over docs and guardrail tests.

**Where things stand now.** The Python line is **complete at v3.0**, and
the experimental native line has completed **Phases A through J** — the
last of them, the deterministic native data pipeline and mini-batching,
closed at milestone J9, so **Phase J is the latest completed phase**.
**Phase K — Native Integer Tensors and Indexing — is the current phase,
and only K0 through K5 have landed.** Each phase's record is in its own
design document; the sections above are the narrative.

## The current phase — Phase K, K0 through K5 complete

**Phase K — Native Integer Tensors and Indexing — is the current phase,
and K0 through K5 are complete.** **K6 through K9 are unstarted.** Its
architecture contract is
[native_integer_tensors_design.md](native_integer_tensors_design.md).

**K0** added **design, documentation, and guardrails only** — no integer
dtype, no kernel, no C ABI symbol, no public export, no registry or
version movement. Runtime capability began at K1.

**K1** added the internal `int64` **representation** and **every**
reachability barrier, and nothing else. Internally, the C++ dtype model
gained a third enumerator at code 2, storage can allocate and destroy
genuine `std::int64_t[]` buffers, and the four transfer boundaries move
integer values bit for bit; every other handle-based export refuses an
`int64` operand at the ABI through the new `tf::require_floating` guard,
and the Python side gained the matching barriers in front of autograd,
parameters, buffers, both optimizers, checkpoint entries, wrapper
construction, and every floating operation. The Python dtype tables were
deliberately left untouched, so at K1 no supported wrapper or public
Python API could allocate or wrap `int64` storage at all.

**K2** made the `int64` tensor publicly constructible, and it landed
**atomically** — splitting it would have opened exactly the window the
ladder is ordered to close. The three Python dtype tables and the checked
host binding learned `"int64"`; `INDEX_DTYPES == ("int64",)` appeared
beside an **unmoved** `SUPPORTED_DTYPES` and is reported as
`backend_info()["index_dtypes"]`; the Phase-I no-drift guard was
**generalized** to `set(_DTYPE_CODES) == set(SUPPORTED_DTYPES) |
set(INDEX_DTYPES)` rather than deleted; the private
`NativeStorage._from_int64_array` / `NativeTensorCore._from_int64_array`
ingress arrived; and exactly two gates widened —
`NativeTensorCore.__init__` and `NativeTensor.__init__`, from "floating"
to "floating **or** index".

**`NativeTensor.from_int64_array(values, *, requires_grad=False)` is the
one public API in the repository through which an `int64` buffer can come
into existence**, beside the dtype-general `item()` and `tolist()`. It
converts nothing — exactly a `numpy.ndarray` of exactly native `int64`,
with a float array, an `int32` array, a `uint64` array, a `bool` array, an
`object` array, a byte-swapped `>i8` array, a list, and a scalar all
rejected — while a *non-contiguous* exact-`int64` array is accepted and
copied, because layout normalization is not conversion. Views, copies, and
exact host inspection work at `int64` through the machinery that already
existed. K2 added **no** C ABI symbol (still 54), **no** experimental
export (still 25), no CTest, no example, no benchmark, and no version
change.

**`int64` is still not a supported native tensor dtype**, and the
distinction is the point: it is an **index/result** dtype in its own
registry, `normalize_dtype("int64")` keeps raising, and **no generic
constructor changed what it accepts**. Every K1 barrier holds against a
real integer tensor — it cannot require gradients, build or enter a graph,
become a `NativeParameter`, be registered as a buffer at either
persistence value, be owned by either optimizer, be declared in a
checkpoint archive, or enter any floating operation.

**K3** added the phase's first operation and its first C ABI symbol:
native `argmax`. `NativeTensor.argmax(axis=None, keepdims=False)` and
`NativeTensorCore.argmax(axis=None, keepdims=False)` search a **floating**
tensor at either dtype, at any rank including 0, contiguous or not, and
return a fresh owning contiguous **`int64`** tensor — the first operation
whose result dtype differs from its operand's, which is exactly what an
index is. Shapes and axes come from the existing `reduce_shape` and
`_normalize_axis_checked` authorities, so `axis`/`keepdims` behave as they
do at `sum` and `mean`. The value rule is exact rather than adjectival:
equal maxima give the **lowest** index, `+0.0` and `-0.0` tie, an
all-`-inf` run gives 0, and the **first** NaN wins against every finite
value and either infinity. The result is a plain leaf **even when the input
requires gradients** — an index has no derivative — so `"argmax"` joined
`TENSOR_CORE_OPS` and deliberately not `AUTOGRAD_OPS`. Exports went
54 → **55** and native CTests 25 → **26**; nothing else moved.

**K4** added the phase's one index-*consuming* operation and its second and
final C ABI symbol: native `index_select`, forward only.
`NativeTensor.index_select(axis, indices)` and
`NativeTensorCore.index_select(axis, indices)` take a **floating** source
at either dtype, any rank ≥ 1, contiguous or not, together with a rank-1
**`int64`** index tensor — never a NumPy array, a list, a tuple, or a
Python `int` — and return a fresh owning contiguous tensor of the
**source's** dtype whose selected axis has `indices.numel` positions. It
is `argmax`'s mirror image, and the two compose directly. Duplicates and
order are preserved exactly, negative and out-of-range indices are
**rejected rather than wrapped**, the complete bounds scan runs in Python
and again independently in C++ before the first destination element is
written, and values cross by **object representation**, so signed zeros,
infinities, subnormals, and NaN payloads survive bit for bit. It is
**forward only**: a source with `requires_grad=True` is rejected with a
message naming `detach()` rather than silently detached, so
`"index_select"` joined `TENSOR_CORE_OPS` and deliberately not
`AUTOGRAD_OPS`. Exports went 55 → **56**, the phase maximum, and native
CTests 26 → **27**; nothing else moved.

**`int64` is still not a supported native tensor dtype** after all of that,
and K3 and K4 are where the distinction earns itself: one operation now
*produces* `int64` and another *consumes* it as a role operand, without
`int64` ever becoming a dtype a kernel computes at. There
is **no `max`** beside the `argmax` — a kernel that finds the position of a
maximum necessarily knows the maximum, and the phase deliberately does not
expose it — and **no general `gather`**, no `scatter`, no embedding lookup,
no `index_select` backward, no `argmin`, no integer arithmetic
or reduction, and no casting or promotion; those belong to later milestones
or to no milestone at all — prove first, then promise.

## The latest completed phase — Phase J, complete

**Phase J — Deterministic Native Data Pipeline and Mini-Batching — is
complete, and Phase J is the latest completed phase.** It was **newly
approved** when it
opened: the repository deliberately closed Phase I at I11 without
committing to a successor, and Phase J was approved afterwards, so it must
not be described as work that was already on the roadmap. Its architecture
contract is
[native_data_pipeline_design.md](native_data_pipeline_design.md).

**Milestones J0 through J9 have all landed, and J9 closed the phase.** J0
was an architecture, contract, and documentation
milestone and **added no runtime behavior at all** — no dataset, sampler,
or loader class, no helper module, no state serializer, no public export,
no C++, no C ABI symbol, no example, no benchmark, and no checkpoint or
optimizer-state change. Runtime capability began at **J1**.

**J1 shipped the finite host-backed dataset, `NativeTensorDataset`** —
the phase's first runtime, and **exactly one** new public experimental
name (`tensorforge.experimental.__all__` went from 22 names to 23). It
takes one owned host snapshot of the features and one of the class
targets — unconditional copies, so caller mutation, resize, or deletion
afterwards reaches nothing — at an **explicitly chosen** native feature
dtype that is never inferred from the input array, computes the locked
SHA-256 content fingerprint eagerly at construction, and turns any index
sequence into a fresh owning `NativeTensor` feature batch **the caller
closes** beside a fresh read-only host `int64` target batch. Order and
duplicate indices are preserved exactly, an empty request is refused, and
the dataset owns **no native storage between calls**. It added no C++, no
CMake entry, no C ABI symbol, no example, no benchmark, no checkpoint
field or version, no optimizer-state version, and no dependency.

**J2 shipped the deterministic batch planner, `NativeBatchSampler`** —
the phase's second runtime, and again **exactly one** new public
experimental name (`tensorforge.experimental.__all__` went from 23 names
to 24). It owns `batch_size`, `drop_last`, `shuffle`, the `seed`, the
`epoch`, and the `cursor`, and turns them into batch-index groups through
`epoch_permutation()`, `plan()`, and `next_batch_indices()`. Its
permutation **reuses the locked `tensorforge.splitmix64` derivation**
under one domain-separated epoch key schedule — no new RNG algorithm, no
new global or default generator, and no coupling to a live
`NativeGenerator` — with unbiased rejection-based bounded integers and a
downward Fisher–Yates sweep, in explicit 64-bit-masked Python integer
arithmetic that is bit-identical on every platform by construction. Every
permutation is a **pure function** of `(seed, epoch, length)`, so the
sampler holds no consumable stream: inspection and planning consume
nothing and may be repeated in any order. Its compact JSON-compatible
`state_dict()` carries the configuration, the position, and the dataset's
four identity fields — **no permutation and no payload** — and
`load_state_dict()` is transactional, validating dataset identity against
live reality and adopting the state's configuration before six
assignments that cannot fail. It allocates nothing native, materializes
no batch, and owns nothing releasable, so it has **no `close()`**. Its
derivation lives in the permanently private `_native_permutation` module,
which is exported by nothing. It added no C++, no CMake entry, no C ABI
symbol, no example, no benchmark, no checkpoint field or version, no
optimizer-state version, and no dependency.

**J3 shipped the native mini-batch loader, `NativeDataLoader`** — the
last of the phase's three public names, and again **exactly one** new
public experimental name (`tensorforge.experimental.__all__` went from 24
names to 25). `iter(loader)` returns a private one-epoch iterator that
captures the sampler's remaining batch count and supersedes any previous
one, and each `__next__` runs an explicit five-phase transaction — claim,
construct, publish, commit-and-deliver, rollback — whose single invariant
is that **the committed sampler position advances if and only if a batch
was successfully delivered to the caller**. A failure at any point closes
the undelivered feature tensor, restores the exact pre-delivery epoch and
cursor through the same non-failing write seam a state load uses, clears
the record on both owners, and leaves a retry returning the *same
indices and the same values*; that is asserted by injection at every
failure position, each with a non-vacuity control and a native
live-storage baseline. Delivered batches are the **caller's** — no close
path retains or can reach one — and the caller closes each feature
tensor. It added no C++, no CMake entry, no C ABI symbol, no example, no
benchmark, no checkpoint field or version, no optimizer-state version,
and no dependency; it is not thread-safe and contains no lock, thread,
queue, worker, prefetch, collate, or callback surface.

**J4 gave the loader its own in-memory state, and added no public name
at all** — the first Phase-J runtime milestone whose export delta is
**zero** (`tensorforge.experimental.__all__` stayed at 25).
`NativeDataLoader` gained exactly two methods. `state_dict()` returns a
compact tagged wrapper with **three** root keys — `format`
(`"tensorforge.native_data_loader"`), `format_version` (**1**), and
`sampler` — around the **unchanged** sampler state; the loader owns no
epoch, cursor, seed, shuffle, batch size, or drop-last field of its own,
so none is duplicated at the root. Every container is fresh at every
call, the whole structure is JSON-compatible and is accepted unchanged by
the checkpoint's existing metadata validator, and it carries no
permutation, no dataset content, and nothing whose size grows with the
number of samples. It is allowed between batches, after an iterator is
exhausted or superseded, with a closed dataset, and after the loader is
closed — and **refused** while a batch transaction is in flight, because
inside the commit-before-delivery window there is no honest answer.
`load_state_dict(state)` is transactional in the same sense the delivery
is: a closed guard, a transaction guard, and an active-iteration guard
run before the state is read at all, the wrapper is validated completely,
the **whole** nested sampler validation is delegated to the seam that
already owns it rather than restated, and only then does a commit run
that cannot fail — so a rejected load leaves the loader, the sampler, the
dataset, the position, the cache behavior, the iterator slot, and native
live storage byte-identical. Dataset identity is validated and never
adopted; the six configuration and position values **are** adopted, so a
deliberately differently configured loader takes the state's. The exit
gate is proved over two separate object graphs: a mid-epoch interruption
restored into a separately constructed dataset, sampler, and loader
reproduces the remaining batches exactly — identical indices, identical
raw IEEE-754 feature bits, identical targets — then the same canonical
next-epoch position and the same following epochs, at both dtypes, with
no tolerance anywhere and a negative control proving the sequences differ
without the restoration. It added no C++, no CMake entry, no C ABI
symbol, no example, no benchmark, no checkpoint field or version, no
optimizer-state version, and no dependency.

**J5 proved the caller-managed checkpoint-metadata workflow, and added no
production code at all.** It is the second consecutive Phase-J milestone
whose export delta is **zero**, and the only one so far whose diff
touches no file under `src/`:
`src/tensorforge/experimental/native_checkpoint.py` is unchanged. The
workflow is the one the contract has named since J0 — take
`loader.state_dict()`, pass it inside the `metadata` a caller already
controls, and after `load_native_checkpoint` returns, hand
`metadata[...]` back to `fresh_loader.load_state_dict(...)` — and J5 is
the evidence that it composes. Against **real** version-3 `.npz` archives
read with pickle disabled: the format stays
`tensorforge.native_checkpoint` version **3** with `(1, 2, 3)` accepted,
the manifest keeps the same six root keys, and the array inventory is
identical whether or not loader state is carried, so **the archive's own
capture set did not grow by one field**. Loader state lives only inside
caller metadata; there is no root field, no loader array, no serialized
permutation, and no dataset payload. `"training"`, `"data_loader"`, and
`"next_step"` are **caller conventions** that no production constant
spells: alternate nesting, alternate key names, and two loaders' states
side by side all round-trip unchanged. Restoration into an entirely fresh
model, optimizer, generator set, dataset, sampler, and loader — each
deliberately built wrong first — reproduces every parameter, persistent
buffer, Adam moment and step counter, hyperparameter, generator state and
**alias topology**, and all six loader values exactly, in raw IEEE-754
bit patterns with no tolerance anywhere, and then the exact next batch and
the exact remaining sequence. All three delivery boundaries are proved
through an archive: a **failed** delivery resumes the same candidate
batch, a **successful** one resumes the following batch, and an
epoch-boundary save resumes the canonical next epoch. The absence of
cross-object atomicity is proved rather than glossed — a checkpoint load
that succeeds followed by a loader load that fails leaves the first
restored and the second untouched, and the documented recovery is to
rebuild and repeat both calls. Non-coupling is asserted in both
directions by source inspection and by driving a real save and load with
the loader's state methods patched to record any call: neither fired.

**J6 shipped the deterministic mini-batch training example**,
`examples/native_minibatch_training.py`, and — like J5 — added **no
production code and no public name**: the third consecutive Phase-J
milestone with a zero export delta. The example inventory moved 15 →
**16**; the benchmarks stayed at **8**. It trains a
`Linear → BatchNorm1d → ReLU → Dropout → Linear → LayerNorm → Dropout →
Linear` classifier over shuffled mini-batches with `NativeAdam` and
`NativeCrossEntropyLoss`, two Dropout layers sharing one generator, and
proves an interrupted-and-resumed run **bit-for-bit identical** to an
uninterrupted one — the whole batch-index sequence, every feature batch's
raw bits, every target array and its flags, every loss, parameter,
buffer, Adam moment and counter, the generator state and alias topology,
the final loader `state_dict()`, and the evaluation output — at float32
and float64 independently, with no tolerance anywhere and no numeric
comparison between the two dtypes. The interruption is genuinely
mid-epoch, the resumed graph is entirely fresh and deliberately built
wrong first, and a negative control that omits the loader restoration
alone is proved to diverge. It uses **only public APIs**, asserted by an
AST scan with its own negative control.

**J7 shipped the cross-cutting adversarial hardening matrix**,
`tests/test_native_data_hardening.py`, and — like J5 and J6 — added **no
production code and no public name**: the fourth consecutive Phase-J
milestone with a zero export delta, and one that **found no production
defect**. Examples stayed at **16** and benchmarks at **8** through J7.
It asserts
every §12.7, §15, §16, and §17 row by injection rather than by argument:
each construction row, each iteration row — with the host gather, the
native allocation, the host→native transfer, and the target copy kept as
four distinct injections — the **commit step made to fail after the
candidate position was really applied**, a `BaseException` through the
same unconditional rollback, the reentrancy refusal matrix at all three
transaction phases, every abandonment position, the close ordering in
both directions, and a **checkpoint taken immediately after a failed
delivery proved to resume the same candidate batch** through a real
version-3 archive into an entirely fresh graph. Every rejection is
followed by a complete before/after fingerprint of the observable world —
including an unrelated parameter, buffer, optimizer, and registered
generator, the filesystem, both global RNGs, and every registry — and
every injection and every parser has its own non-vacuity control.
**Concurrency stays documented as unsupported rather than tested as
safe**: no lock was added, no Phase-J module contains one, no test starts
a thread, and external locking remains the caller's job.

**J8 shipped the data-pipeline benchmark**,
`benchmarks/benchmark_native_data_pipeline.py`, and — like J5, J6, and
J7 — added **no production code, no public name, and no optimization**:
the fifth consecutive Phase-J milestone with a zero export delta.
Examples stayed at **16** and benchmarks moved 8 → **9**. It answers four
separate questions rather than one blurred end-to-end number — what
immutable host dataset indexing costs, what deterministic batch planning
costs, what deterministic shuffled-permutation construction costs, and
what host→native batch materialization costs — with one clearly separate
composition case for a whole `next(iterator)` delivery. float32 and
float64 are measured **separately and never as a ratio of one to the
other**; every case is gated exactly, with no tolerance anywhere, before
the timing helper is reached; a case with no honest equivalent is
labelled `native_only` and publishes **no ratio at all**; cold and warm
permutation construction are separate cases and are never averaged;
medians come with an interquartile range after warm-up; setup,
per-repetition state reset, and every `close()` stay outside the timer;
and **no threshold, CI timing job, or result file** exists. The
measurements are one machine, one build, and one moment, and no runtime
change is derived from them.

**J9 closed the phase**, adding no production code, no public name, and
no export: it shipped `tests/test_native_phase_j_closure.py` — the
permanent closure guardrails — re-ran the complete validation matrix
(Windows Release and Debug, a Linux CI-equivalent, Clang ASan/UBSan with a
detector negative control, and a LeakSanitizer lifecycle over the whole
pipeline), and reconciled every inventory.

**What Phase J deliberately never had, and never will:** automatic loader
discovery. A loader's position can be serialized, carried through a
checkpoint archive, restored exactly, read in a worked example, relied on
to consume nothing when a delivery fails, and measured layer by layer —
but nothing discovers a loader for the caller, in either direction, and
none may be added.

What J0 resolved, so that later milestones inherit an unambiguous design
rather than re-deriving one: the three eventual public names —
`NativeTensorDataset` (J1), `NativeBatchSampler` (J2), and
`NativeDataLoader` (J3) — with the permutation helpers and the batch
iterator permanently private; a strict `numpy.ndarray`-only input contract
whose native feature dtype is **explicitly chosen and never inferred** from
the input array, still defaulting to float64; copied host snapshots taken
once at construction, so no caller mutation can reach a later batch; a
deterministic **SHA-256** dataset fingerprint over an explicit
little-endian canonical byte stream, so a restored position cannot be
applied to different data; a sampler owning `batch_size` and `drop_last`
and emitting batch-index groups, with `epoch` the active epoch, `cursor`
the batches already delivered, and the epoch boundary canonicalized
immediately so every position has exactly one representation; a
deterministic shuffle that **reuses the locked `tensorforge.splitmix64`
derivation** under a domain-separated key schedule — no new RNG algorithm,
no new global generator, and deliberately no coupling to a live
`NativeGenerator`, which exposes no bit derivation to couple to — with
unbiased rejection-based bounded integers, a downward Fisher–Yates sweep,
directly implementable pseudocode, and committed reference vectors; a
permutation that is a **pure function** of `(seed, epoch, length)`, so an
abandoned iterator, a rejected state load, and a failed batch consume
nothing by construction rather than by cleanup; one-epoch iterators with a
superseding `iter()` and a single atomic cursor commit after
materialization and immediately before handoff; caller-owned `NativeTensor`
feature batches beside fresh read-only host `int64` targets, which the
loader never retains; strict JSON-compatible sampler and loader state
schemas carrying no payload and **no serialized permutation**, because the
compact `(seed, epoch)` derivation reproduces it exactly; transactional
state loading whose commit is six assignments that cannot fail; an explicit
**caller-managed** checkpoint-metadata workflow over the unchanged
version-3 format, with cross-object atomicity explicitly **not** claimed;
and an exact interrupted-versus-uninterrupted resume contract compared in
raw IEEE-754 bit patterns, with no tolerance anywhere.

**Phase J moves no capability, at any milestone.** `SUPPORTED_DTYPES`,
`SUPPORTED_DEVICES`, `UNSUPPORTED`, and `RAW_KERNEL_DTYPES` are unchanged;
so are the **54** exported `tf_*` symbols, the 24 native CTests, checkpoint
version **3** with `(1, 2, 3)` accepted, and the in-memory optimizer state
version **1**. The phase plans no new C ABI export at all.

## Earlier completed phase — Phase I

**Phase I — Native Dtype Generalization and Float32 CPU Support — is
complete (I0–I11); Phase J closed after it, so the latest completed phase
is Phase J.** Milestone
I11 revalidated the whole dtype-general stack on Windows Release and Debug,
on a Linux CI-equivalent, and under Clang ASan/UBSan and LeakSanitizer,
reconciled every status surface, and closed the phase. Its architecture
contract is
[native_dtype_float32_design.md](native_dtype_float32_design.md).

**Since I9, `float32` and `float64` are both publicly supported native CPU
dtypes** — `SUPPORTED_DTYPES == ("float64", "float32")`, `UNSUPPORTED ==
("cuda", "amp")` — with float64 remaining the default everywhere, no
casting or promotion between them, and `RAW_KERNEL_DTYPES` still
`("float64",)` for the seven handle-free raw utility kernels.

**I0 was a design-and-reconciliation milestone: it shipped the contract,
its guardrail tests, and documentation, and no runtime behavior at all.**

**I1 built the foundation the rest of the phase stands on**: the C++ dtype
model (frozen ABI codes `0 = float64` and `1 = float32`, one item-size
authority, one canonical-name authority, and a total validated
conversion), dtype-tagged storage with an untyped owned buffer and checked
`numel × itemsize` allocation, and the two typed creation exports
`tf_storage_create_typed` and `tf_storage_create_uninitialized_typed`.
The library now exports **54** `tf_*` symbols, which is the count for the
whole phase — no later milestone adds one. The untyped creators remain,
unchanged, as thin float64 compatibility wrappers over the same shared
creation body.

**I2 made float32 storage movable.** The three exports that carry a
storage handle *and* a host buffer — `tf_storage_copy_from`,
`tf_storage_copy_to`, and `tf_storage_materialize` — became dtype-general
through a **source-level retype** of their host positions from `double*`
to `void*`: same symbols, same argument slots, same calling convention,
still 54 exports, and a previously compiled caller would link and run
identically. `tf_core_contiguous_copy`, the runtime's value-transfer
primitive, became dtype-preserving and dtype-strict, so a float32 view of
any layout can be materialized or copied storage-to-storage while a mixed
float32/float64 pair is rejected before anything is written. Internal
float32 values now round-trip through the host, through every view
transformation, and through the identity copy **bit for bit** — signed
zeros, infinities, subnormals, NaN payloads, and signalling NaNs included,
proved by raw IEEE-754 bit comparison at both widths. `RAW_KERNEL_DTYPES`
records the other half of the division: the seven handle-free raw utility
kernels take only `double*` and an element count, so they have no dtype to
dispatch on and stay float64.

**I3 made float32 storage computable, by the elementwise and unary Core
family and by nothing else.** `add`, `subtract`, `multiply`, `relu`,
`relu_backward`, `sqrt`, `reciprocal`, `exp`, and `log` — seventeen exports
across their strided and contiguous forms — validate that their operands
agree, dispatch **once** from the storage tag, and run one instantiation of
a templated kernel, with nothing below that point branching on dtype. All
three Phase-H traversal tiers (the contiguous row, H8's collapsed plan, and
the retained generic odometer) are instantiated for both element types from
the same source, so float64 runs the code Phase H measured and the widths
cannot drift apart. NumPy-style broadcasting works at float32 for every
layout it already worked at for float64, outputs preserve the operand
dtype, and mixed dtype is refused in the left, right, and destination
positions independently, before any allocation. float32 arithmetic is
genuinely binary32 — bit-identical to the binary32 oracle for the
IEEE-specified operations, and within a measured ULP bound for `exp` and
`log`, which are library functions with no correctly-rounded guarantee.
**I3 added no export.**

**I4 made float32 reduce, multiply matrices, propagate through views, and
differentiate.** `sum`, `mean`, `matmul`, and `narrow_backward` are
dtype-general, with H6's contiguous-block factorization and H2's `i`-`k`-`j`
row sweep — and the retained generic odometer and triple loop beside them —
instantiated for both element types from the same source, so every optimized
path keeps its oracle **per dtype** and float64 runs the code Phase H
measured. `tf_storage_scale` and `tf_storage_fill` became dtype-general with
their `(handle, double)` signatures unchanged: the scalar crosses as the
ABI's widest float and is narrowed **once, before the loop**, which is what
makes `mean`'s `1/count` deterministic and identical on every platform. On
top of those, private/internal float32 `NativeTensor` graphs now run forward
and backward through the whole set of Core operations landed so far —
elementwise, broadcasting, the views, the reductions, and matmul — with
every gradient, every backward temporary, and every materialized constant
carrying the graph's dtype.

**float32 accumulation is genuinely float32, and I4 is where that stopped
being a structural claim and became a measured one.** Every I3 operation
produced its result with a single correctly-rounded IEEE operation, for
which computing in binary64 and rounding once is *provably*
indistinguishable from computing in binary32 — so no runtime test could
separate the two, and I3 said so rather than inventing one. A sum is the
first place the difference is observable: on the vector `1.0` followed by
eight copies of `2**-24`, sequential binary32 stays at exactly `1.0` while
binary64-then-narrow lands four ULPs higher, and TensorForge is asserted
equal to the first and unequal to the second — on both reduction traversals
and both matmul paths. **I4 added no export.**

**I5 made float32 convolve and pool.** All three Conv2d directions and both
MaxPool2d directions are dtype-general, with H9's row-sweep and gather
traversals — and the retained Phase-D generic loops beside them —
instantiated for both element types from the same source, the geometry
predicates untouched, and Conv2d accumulating in the element type: the
binary32-versus-widened witness is proved in all three directions, on both
traversals of each. MaxPool2d's values follow the input dtype through the
identical comparison sequence at both widths, while the private **winner
buffer stays float64 at every value dtype** with its `2**53` exact-plane
bound unchanged — a float32 pool over a plane beyond float32's `2**24`
exact-integer range still records its offsets exactly, which is the
capability that decision preserves. Private float32 graphs differentiate
through convolution and pooling, with the winner riding the unchanged
graph-owned saved-state contract. **I5 added no export.**

**I6 made float32 classify.** Softmax, log-softmax, and the fused
cross-entropy forward and backward are dtype-general, with every
participating numeric handle checked for agreement and one dispatch per
export into templated kernels. The maximum scan, the shift, the
exponentials, the normalizing sum, the log-normalizer, the per-row loss,
the batch-loss accumulator, the mean divisor, and every backward
contribution all happen at the element type — and the batch-loss
accumulator is where that became a *measured* claim: on a batch whose first
row contributes exactly 200 and whose remaining 199 contribute ~6.1e-6
each, binary32 stays at exactly 200 while binary64-then-narrow lands
~1.2e-3 higher, and TensorForge is asserted equal to the first and unequal
to the second. Log-softmax is still its own fused log-sum-exp kernel and
never `softmax().log()`; the saved probabilities carry the graph dtype and
stay the only thing the backward reads; the class **targets stay host
`int64` metadata at every width**, so no integer tensor dtype appeared.
Private float32 graphs differentiate through all three operations with no
graph-structure change at all. **I6 added no export.**

I6 is also where the float32 stability statement gained its one honest
qualification. The maximum shift guarantees no *exponent* overflows; it
does not make the shifted value representable, so a slice whose **spread**
exceeds the element type's largest finite value overflows the shift to
`-inf`. `softmax` is unaffected and still exact; `log_softmax` reports
`-inf` and `cross_entropy` `+inf`, as values rather than errors. Those are
the correctly rounded IEEE results for quantities with no representation at
that width, and the same thing happens at binary64 past ~1.8e308. The
kernels were left alone — no widened intermediate, no clamp, no special
case — and the counterexample was written into the contract and asserted by
test in both directions.

**I7 made float32 a module dtype, and closed the last float64-only
kernel.** Six state-owning constructors — `NativeParameter`,
`NativeLinear`, `NativeConv2d`, `NativeLayerNorm`, `NativeBatchNorm1d`, and
`NativeBatchNorm2d` — take a keyword-only `dtype` accepting exactly the two
widths and defaulting to float64, through one shared validator. Affine
parameters, both BatchNorm running buffers, the evaluation snapshots, and
every constant a composed normalization forward materializes are at the
module's dtype; the atomic two-buffer running-statistics transaction gained
one dtype validation and nothing else. Initialization is unchanged at both
widths: the same local `default_rng(seed)` stream in the same order, so a
float32 layer with seed *S* holds exactly `float32(the float64 draw with
seed S)` — the seed contract does not become dtype-dependent.

Dropout was the last family out. The kernel is templated, the export keeps
its exact ABI shape with one operand-agreement guard and one dispatch, and
**the random derivation is untouched**: the uniform stays binary64 at every
width, so one `(seed, call_index, element count)` key drops exactly the
same elements at float32 and float64 and only the two multiplier values
differ. The kept multiplier is the binary64 reciprocal narrowed once, which
at float32 is an observable, separately witnessed property. Generator
state, its algorithm, its version, and the reserve → commit/abandon call
accounting are all unchanged, at both widths. **I7 added no export.**

**I8 delivered float32 optimizer state and native checkpoint version 3.**
Both `NativeSGD` and `NativeAdam` execute at float32 — Adam's `m` and `v`
carry their parameter's dtype, one optimizer may hold parameters of both
widths with independent dtype-consistent state per parameter, and neither
gained a `dtype` or `device` argument, because they own no dtype they
could choose. No C++ changed and **no export was added**: I3-I7 had already
generalized every operation the optimizers compose, so three constructors
moving to their private typed twins was the whole runtime change, and
H4's once-per-step scalar architecture is preserved whole (the caches key
on `(dtype, device)`, so a mixed collection builds one scalar set per
*active dtype* rather than one per parameter). Design §15.3's open
question was **resolved on measurement**: H4's Python bias-correction
reciprocal is an exact substitution at binary64 but not at binary32,
because the kernel divides by the **narrowed** denominator — the two
spellings differ by one ULP for a large fraction of inputs, the default
betas included — so the denominator is narrowed first and the reciprocal
taken of that, which is what the kernel does, at no allocation and no
kernel call, with float64 bit-identical to before. Native checkpoint
**version 3** declares every numeric entry's dtype explicitly, accepts
versions `(1, 2, 3)`, writes 3 on every new save whatever the model holds,
and carries Adam's moments as entry objects rather than bare archive names
so their metadata is stated rather than inferred positionally. float32
model values, persistent buffers, and Adam moments round-trip **bit for
bit**; a dtype disagreement is rejected in either direction with no cast,
no `map_location`, and no device movement; and versions 1 and 2 remain
float64-only formats permanently, never guessing a payload to be float32.
Every transactional, identity, aliasing, and rollback guarantee is
unchanged, and the **in-memory** optimizer state schema stayed at
version 1.

**None of I1 through I8 moved a public capability**, deliberately. The
checkpoint version moved at I8, which is a **schema** change design §16.1
always assigned to that milestone rather than a support claim. The gap
between internal capability and public promise was the phase's rollout
discipline, the same one Phase G used for `dropout`.

**I9 closed that gap, and it is the phase's one and only public registry
change.** It moved *after* the proof, not before: the integrated example
and its exact-resume proof were written and passing first, through the
already-approved private typed route and the six I7 module constructors,
with the registry still reading `("float64",)`; only then did
`SUPPORTED_DTYPES` become `("float64", "float32")` and `UNSUPPORTED` become
`("cuda", "amp")`; then the example's one ingress helper switched to the
public `NativeTensor.from_array(values, dtype=...)` and the whole proof was
rerun.

`examples/native_float32_training.py` is that proof: one deep model —
`Conv2d → BatchNorm2d → ReLU → MaxPool2d → Dropout → Flatten → Linear →
BatchNorm1d → ReLU → LayerNorm → Dropout → Linear`, raw logits into
`NativeCrossEntropyLoss`, trained by `NativeAdam`, with **two Dropout
layers sharing one registered generator** so the model carries a real alias
topology — run uninterrupted and interrupted at **each** dtype and compared
**only against itself**, in raw IEEE-754 bit patterns. Every loss, the
first resumed step's gradients, every parameter, every buffer, every Adam
moment and counter, the generator state, the alias topology, the next
Dropout mask, the final logits, the predictions, and the evaluation output
match exactly, and native live storage returns to baseline. A float32 run
is never required to reproduce a float64 one and nothing asserts that it
does.

I9 changed no C++, added no export (still 54) and no CTest (still 24),
moved no checkpoint field or version, and added no module, loss, optimizer,
operation, dependency, or benchmark change. Nothing about Phase H changed;
Phase H remains complete, and it closed at 52 exports.

What Phase I has delivered: float32 CPU tensors beside the existing float64
ones, dtype-tagged storage, dtype-aware handle-based operations, float32
autograd, modules, buffers, optimizers, and optimizer state, float32
deterministic Dropout over an unchanged generator, a dtype-aware checkpoint
version 3, exact deterministic float32 resume, public float32 support,
unchanged float64 behavior and performance, at **I10** the cross-cutting
adversarial hardening matrix and separate float32/float64 benchmark
characterization, and at **I11** the cross-platform revalidation, the
closure guardrails, and the final status reconciliation that closed the
phase. Nothing remains: the ladder is finished.

The decisions the contract locks, so later milestones inherit them rather
than re-deriving them:

- **Exactly two new C ABI symbols for the whole phase** —
  `tf_storage_create_typed` and `tf_storage_create_uninitialized_typed`,
  taking the library from 52 to **54**. *Delivered at I1.*
  Per-operation float32 exports are explicitly **rejected**: handle-based
  exports already receive their operands as opaque handles, so the dtype
  travels with the data and one narrow dispatch per call is enough.
- **Storage carries the dtype**, and it is the single authority, so every
  view of one buffer agrees and no view operation casts or reinterprets.
  Shapes, strides, and offsets stay measured in logical elements; bytes
  appear only at the allocation boundary, with checked
  `numel × itemsize` arithmetic. *Delivered at I1.*
- **No casting, no promotion, no mixed-dtype arithmetic.** A mismatch
  raises before any output is allocated or any state is mutated. *First
  reachable — and tested — at I2, on the identity copy; enforced across the
  whole elementwise family at I3, and across reductions, matmul,
  narrow-backward, and gradient accumulation at I4.*
- **The raw-buffer boundary is divided explicitly.** Every export that
  receives a storage handle can be dtype-general because the handle
  carries the dtype; the seven handle-free raw utility kernels cannot, and
  stay float64 permanently. *Recorded by `RAW_KERNEL_DTYPES` and reported
  by `backend_info()` at I2.*
- **float32 accumulates in float32**, with no hidden wider accumulator
  anywhere — that would be mixed precision, which is out of scope. *First
  enforced at I3; **witnessed by test at I4**, where accumulation makes the
  difference between binary32 and binary64-then-narrow observable, on both
  reduction traversals and both matmul paths.* *Extended at I5 to all three
  Conv2d directions on both traversals, and at I6 to the cross-entropy
  batch-loss accumulator — the one place in the classification family where
  the two policies can differ at all, since every other operation there is a
  single correctly-rounded operation per destination.* *At I7 the same split
  reached Dropout's kept multiplier, which is a **scalar** rather than an
  accumulator and is therefore governed by the narrow-once rule instead:
  computed in binary64 and narrowed once, witnessed at a probability where
  that provably differs from recomputing it in binary32.*
- **Checkpoint version 3** is designed but not activated at I0. Versions
  1 and 2 stay readable and are defined as float64-only formats that are
  never guessed to be float32.
- **Exact deterministic resume is proved separately for each dtype**, and
  a float32 run is never required to agree with a float64 one.
- **Every Phase-H float64 optimization is preserved**, and float32 and
  float64 are benchmarked separately, with no timing assertion, no
  committed number, and no result file — as in every phase before it.
  *Delivered at I10*, in a harness deliberately separate from Phase H's so
  that every number Phase H published keeps its meaning. I10 changed no
  production runtime code at all.
- **The public support registry moved at milestone I9** and at no other
  one: float32 was not declared supported until the whole training stack,
  optimizer state, checkpoint version 3, and the exact-resume proof all
  existed. *Delivered at I9.*

The ladder is I0 through I11 — **all twelve landed** (this line lagged a
milestone behind after I8, and that lag was repaired rather than rewritten
away): the contract (I0), the dtype model and
tagged storage (I1), typed transfer and materialization (I2), elementwise
execution (I3), reductions and matmul (I4), the convolution and pooling
kernels (I5), stable math and classification (I6), modules, buffers, and
Dropout (I7), optimizer state and checkpoint version 3 (I8), the public
integration and resume proof (I9), hardening and benchmarking (I10), and
cross-platform validation and closure (I11).

## Practical next steps

**Phase I is finished, and Phase J is the approved successor.** That
sentence read "no successor phase is defined" for as long as it was true —
Phase I closed at I11 without one, deliberately — and Phase J was approved
afterwards. It is recorded that way rather than rewritten, because "the
phase that came next" and "the phase that was always planned next" are
different facts.

Phase J's own design contract exists
([native_data_pipeline_design.md](native_data_pipeline_design.md), milestone
**J0**), and its runtime has begun at **J1** with `NativeTensorDataset`,
continued at **J2** with `NativeBatchSampler`, at **J3** with
`NativeDataLoader` and its transactional batch delivery, at **J4**
with the loader's own in-memory state and exact mid-epoch restoration,
at **J5** with the caller-managed checkpoint-metadata workflow proved
against real version-3 archives — a milestone that added no production
code and left the checkpoint module unchanged — and at **J6** with the
deterministic mini-batch training example and its exact
interrupted-versus-uninterrupted proof, which added no production code
either and took the example inventory from 15 to 16, at **J7** with
the cross-cutting adversarial hardening matrix, which added no production
code either and found no production defect, and at **J8** with the
data-pipeline characterization benchmark, which added no production code
and no optimization either and took the benchmark inventory from 8 to 9.
and at **J9** with the integration and closure milestone, which added no
production code either and closed the phase.

**Phase J is complete, and no milestone remains in it. Phase K is the
approved successor.** That paragraph read "no successor phase is defined"
for as long as it was true — Phase J closed at J9 without one,
deliberately — and Phase K was approved afterwards. It is recorded that way
rather than rewritten, because "the phase that came next" and "the phase
that was always planned next" are different facts, and only the first one
is true here.

**Phase K — Native Integer Tensors and Indexing** is that newly approved
successor. It has its own design contract
([native_integer_tensors_design.md](native_integer_tensors_design.md)),
and **K0, K1, and K2 have landed**. K0 is
architecture, contract, status reconciliation, and guardrails, and it
**added no runtime behavior at all**: no integer dtype or dtype code, no
C++ enumerator, no kernel, no C ABI symbol, no ctypes declaration, no
public export, no capability-registry movement, no checkpoint,
optimizer-state, loader-state, or sampler-state change, no example, no
benchmark, no CTest, and no dependency.

**K1 added the internal `int64` representation and every reachability
barrier, and no public capability at all.** The C++ dtype model gained a
third enumerator at code 2; `create_storage` and `destroy_storage_data`
gained an `Int64` arm; `tf_storage_copy_from`, `tf_storage_copy_to`,
`tf_storage_materialize`, and `tf_core_contiguous_copy` move integer
values bit for bit; and **32** float-only exports gained the
hidden-visibility `tf::require_floating` guard, which runs ahead of the
operand-agreement guard so a mixed float/integer call is refused as a role
error. On the Python side, nine trusted dtype paths were narrowed from the
representation table to the floating registry, and every §6.5 barrier
landed: wrapper construction, autograd (`_from_op`, `backward`,
`_accumulate_grad`), `NativeParameter`, `register_buffer` at **both**
`persistent` values, both optimizers, checkpoint entry validation, and
every floating operation entry. It added **no** C ABI symbol, **no**
public Python name, and **no** registry or version movement; the native
CTest inventory went from 24 to **25**.

**K2 made the `int64` tensor publicly constructible, atomically, and moved
no other capability.** `_DTYPE_CODES`, `_DTYPE_ITEM_SIZES`, and
`_DTYPE_NUMPY` gained `int64` at code **2**, 8 bytes, and `numpy.int64`,
and `_CHECKED_HOST_ARRAYS` gained an entry **reusing** the existing
`_CHECKED_I64_ARRAY` object; `INDEX_DTYPES == ("int64",)` appeared beside
an unmoved `SUPPORTED_DTYPES`, reported as
`backend_info()["index_dtypes"]`; the Phase-I no-drift guard was
generalized to `set(_DTYPE_CODES) == set(SUPPORTED_DTYPES) |
set(INDEX_DTYPES)`; `_normalize_index_dtype` and the
`_is_index_dtype` / `_is_tensor_dtype` / `_require_tensor_dtype`
predicates arrived; the private exact ingress
`NativeStorage._from_int64_array` and
`NativeTensorCore._from_int64_array` arrived, allocating through the
**zeroed** typed allocator so no integer destination joins the H1
uninitialized-allocation audit; and exactly two gates widened —
`NativeTensorCore.__init__` and `NativeTensor.__init__`. The whole public
delta is `NativeTensor.from_int64_array` plus the dtype-general `item()`
and `tolist()`. K2 added **no** C ABI symbol, **no** experimental export,
no CTest, no example, no benchmark, and no version change.

Every compute registry row is still exactly what
Phase J left: `SUPPORTED_DTYPES == ("float64", "float32")`,
`SUPPORTED_DEVICES == ("cpu",)`, `UNSUPPORTED == ("cuda", "amp")`,
`RAW_KERNEL_DTYPES == ("float64",)`, **25**
experimental names, **16** examples, and **9** benchmarks. The exported
`tf_*` symbol count was **54** through K2, **55** from K3, and is **56**
from K4 — the phase maximum, now reached.

**K4 shipped `index_select`, forward only**, over the phase's second and
final export `tf_core_index_select`: a floating source and a rank-1
`int64` index tensor in, a fresh owning contiguous tensor of the source's
dtype out, with duplicates and order preserved, negative and out-of-range
indices rejected rather than wrapped, a complete bounds scan in Python and
again in C++ before anything is written, and values copied by object
representation. A source with `requires_grad=True` is rejected with a
message naming `detach()`; the backward belongs to a later, separately
approved phase and its contract is already fixed in §18.9. Native CTests
went 26 → **27**; no example, benchmark, registry, or version moved.

**K5** is the **compatibility proof**, and it added **zero production
code**: one new test module,
`tests/test_native_integer_compatibility.py`, and the status
reconciliation a landed milestone requires. It proves against the live
tree that no checkpoint archive can declare an `int64` entry at any entry
role or any accepted version; that the checkpoint format and version
(`tensorforge.native_checkpoint`, **3**, accepting `(1, 2, 3)`), the
in-memory optimizer-state version (**1**), and the loader and sampler
state versions (**1**, accepting `(1,)`) are all exactly where Phase J
left them; that parameters, buffers of both persistence kinds, and both
optimizers still refuse a real `int64` tensor; that Phase J still delivers
a floating `NativeTensor` feature batch and a read-only host
`numpy.ndarray` target batch of dtype `int64`, with no option anywhere
requesting a native
label; that a caller may convert delivered targets explicitly through
`NativeTensor.from_int64_array` and consume them with `index_select`
without any pipeline change; that `NativeCrossEntropyLoss` and
`native_accuracy` are behaviorally unchanged, the metric proved not to
call the native `argmax` or `index_select` at all; and that a real
classifier trains, checkpoints, and resumes **bit-identically** at both
widths while `argmax` and a detached `index_select` run beside the
training path, with an observational control showing that indexing changes
no trainable state. Exports stayed **56**, CTests **27**, examples **16**,
benchmarks **9**, and `experimental.__all__` **25**.

**The proof found one real defect, and the chronology is part of the record.** Driving the two module registration routes with a deliberately forged `NativeParameter` — the only way to reach them, because the public constructor refuses an `int64` tensor — showed that `save_native_checkpoint` trusted whatever dtype live state reported, so the **writer** could emit an archive declaring an `int64` entry that its own loader then refused. That was a pre-existing gap in the writer, not something Phase K introduced and not reachable through any public API, and it was repaired in a **separate checkpoint-hardening change committed before K5**: a save-side persisted-dtype authority asking the same `cpp.normalize_dtype` question the loader asks, applied in `_validate_model`'s preflight and again at `_coherent_snapshot`'s serialization seam, with its own regression in `tests/test_native_checkpoint.py`. No format, field, version, capability, registry, export, CTest, example, or benchmark moved; the forged parameter is test-only and never supported public usage; and K5 itself remains the test-and-documentation compatibility milestone.

**`int64` is not a supported native tensor dtype** — it is an
index/result dtype in its own registry, and no generic constructor accepts
it. A native `argmax` exists from **K3** and a native `index_select` from
**K4**; **no `max`** exists beside them and
none is planned, no general `gather`, `scatter`, or embedding lookup
exists, no `index_select` backward exists, no `argmin` exists, no
integer arithmetic or reduction exists, no integer autograd, parameter,
buffer, optimizer state, or checkpoint entry exists, and
**K6 through K9 are unstarted**. What
K0 decides is the architecture: one extended `NativeTensor` rather than a
parallel integer class, `int64` as an exact non-differentiable
index/result dtype and the only integer dtype in the phase, one strict
`numpy.ndarray`-only construction door with no dtype inference and no
numeric cast, integer autograd/parameter/optimizer/buffer/checkpoint
barriers enforced in Python **and** independently at the C ABI, a complete
`argmax` contract (lowest-index ties, an exact case-by-case NaN rule in
which the **lowest-indexed** NaN is returned whatever the other values
are, signed zeros tying, no `max` exposed, no graph ever) and a complete
forward-only `index_select` contract (rank-1 `int64` indices, negatives
rejected, every index bounds-checked **before** the destination is
allocated, duplicates and order preserved), the Phase-J loader default
left exactly as it is, **no checkpoint version change**, and a C ABI
budget of **+2** symbols with a phase maximum of **56**.

**The dtype taxonomy is settled, and it is the narrow one.**
`SUPPORTED_DTYPES` **remains the floating-compute registry permanently**
and never gains `int64`; `normalize_dtype("int64")` keeps raising; and
therefore **not one generic constructor changes what it accepts at any
milestone** — `NativeStorage`, `NativeStorage.from_array`,
`NativeTensorCore.from_array`/`zeros`/`full`, and
`NativeTensor.from_array`/`zeros`/`full` all keep rejecting the dtype by
name, and public `NativeStorage(size, dtype="int64")` is prohibited. The
one public registry movement of the phase is a separate
`INDEX_DTYPES == ("int64",)` row, and it appeared at **K2**, in the same
commit as the public constructor: *prove first, then promise*, the rule
Phase G used for `dropout` and Phase I for `float32`.

**The ladder is ordered so that no unsafe window existed, and K1 is where
that ordering was paid for.** Every
reachability barrier — autograd, parameters, optimizers, module buffers,
checkpoint entries, every floating operation, and mixed float/integer
requests — landed at **K1**, while `int64` was reachable only through a
direct C ABI call and no Python object could be built over it. The first
milestone at which an integer tensor could be constructed is **K2**, one
milestone later, and every barrier is re-proved there against the real
object in `tests/test_native_int64_tensor.py`.

The ladder is K0 through K9: the contract (K0), the `int64`
representation and **every** reachability barrier (K1), the integer
tensor with its construction, ownership, views, host inspection, public
door, and `INDEX_DTYPES` — atomically (K2), `argmax` (K3),
`index_select` forward (K4), the compatibility proof (K5), the end-to-end
example and exact proof (K6), adversarial hardening (K7), benchmark
characterization (K8), and cross-platform validation and closure (K9).

Beyond Phase K, what the existing documents name as *possible* future work,
in no committed order and with nothing approved, is: further dtypes or
devices beyond the two Phase I delivers, and CUDA experiments. Neither has
started, neither is scheduled, and each would require a **separately
approved** phase with its own design contract before any of it may be
described as begun. Each would be a *capability* phase, and each is
deliberately outside everything shipped so far.

Two things Phase H recorded are worth carrying forward as *inputs* to
whichever comes next, rather than as work items in themselves:

- the small-operation cost floor is **allocation and Python object
  construction**, not the ctypes boundary (design §16.11.8), so any future
  attempt to move it is a binding- or execution-architecture decision;
- SIMD, threading/OpenMP, and BLAS are rejected **with measurements and
  with explicit reopening triggers** (design §11–§13 and §16.11.7), so
  reconsidering one means meeting its criterion, not relitigating it.

The rest of this section is the historical record of how the project got
here; what remains is expansion on its own terms:

- **Advanced branches** — the C++ backend experiment now has
  elementwise kernels, naive and cache-tiled 2-D matmuls, an
  introspection API, and a native runtime prototype: shape/stride
  metadata, a C++-owned NativeStorage buffer, NativeTensorView binding
  the two with native contiguous materialization, and NativeTensorCore
  composing it all into the first native tensor runtime object with
  metadata-only view operations (reshape, transpose, narrow), native
  compute over strided views (elementwise ops and matmul), a
  benchmark suite measuring NumPy, raw-buffer kernels, and the
  TensorCore runtime side by side, and a backend dispatch design plus
  Stage 1 of it — an explicit `get_backend("numpy"|"native")` API with
  no implicit routing and a polished conversion contract
  (`tensor_from_array` in, `to_numpy` out; see
  [backend_experiments.md](backend_experiments.md) and
  [dispatch_design.md](dispatch_design.md)). Stage 2 — a forward-only
  native tensor wrapper over `NativeTensorCore` — is now **designed**
  (purpose, non-goals, ownership/lifetime, conversion contract, minimal
  API, testing plan, and a staged build sequence; see
  [native_tensor_wrapper_design.md](native_tensor_wrapper_design.md)),
  and it is **now feature-complete as a forward-only wrapper**:
  `tensorforge.experimental.NativeTensor`, a forward-only wrapper over
  `NativeTensorCore` with constructors, metadata, `to_numpy`, an
  explicit ownership/lifetime story (v1.8), forward compute — `relu`,
  `add`, `subtract`, `multiply`, `matmul` with exact-shape/2-D behavior
  and no broadcasting (v1.9) — and metadata-only view ops: `reshape`,
  `transpose`, `T`, `narrow` returning borrowing wrappers, plus
  `contiguous_copy` returning an owning one (v1.10). No autograd, no
  operator overloads, not `tensorforge.Tensor`. It is now demonstrable
  too: a small deterministic example
  (`examples/native_tensor_demo.py`), a metadata-only `repr`, and a
  wrapper overview in the docs (v1.11); and honestly characterized —
  the benchmark suite times the wrapper's ops (strided views and
  `contiguous_copy` included) across NumPy, the raw-buffer kernels,
  `NativeTensorCore`, and `NativeTensor`, overheads included and with no
  performance assertions (v1.12). Acting on that finding — the
  elementwise cost is the generic shape/stride odometer traversal in the
  native runtime, not the wrapper — the contiguous elementwise fast path
  is now **designed** (v1.13): flat, index-free loops for contiguous
  `relu`/`add`/`subtract`/`multiply`, the odometer kept for strided
  views, placed in the `NativeTensorCore`/native-kernel layer so
  `NativeTensor` inherits it, bit-for-bit equivalent
  ([native_contiguous_fast_path_design.md](native_contiguous_fast_path_design.md))
  — and now **implemented** (v1.14): flat, index-free kernels beside the
  generic odometer ones, selected when every operand is row-major
  contiguous (nonzero offsets and scalars included) and falling back to
  the retained odometer path for strided views, proven bit-for-bit equal
  to it; `NativeTensor` inherited the change with no wrapper edits, and
  no broadcasting, reductions, autograd, `Tensor` integration, or CUDA
  came with it. Its impact is now **measured and reported** (v1.15): on a
  local run the contiguous elementwise rows moved to roughly raw-buffer-
  C++ speed (~1.5× NumPy at 1000×1000) while the strided-view rows stayed
  on the retained odometer (~2.5–3.5×), and `NativeTensor` tracked
  `NativeTensorCore` throughout — with matmul and `contiguous_copy`
  unchanged, numbers hardware-dependent, and no test asserting a speedup
  (see [backend_experiments.md](backend_experiments.md)). Building on
  that, native broadcasting is now **designed (v1.16) and implemented
  (v1.17)**, Phase A2 complete: it lifts the native elementwise ops from
  exact-shape-only to NumPy-style broadcasting (scalar↔tensor, same-rank
  size-1 stretching, left-padding with leading 1s) through a zero-stride
  read model that never materializes an expanded operand, with the v1.14
  fast path and generic odometer preserved for the same-shape case and
  **no new C++ kernel** (the existing odometer consumes the broadcast
  strides). `NativeTensor` and the explicit native backend inherited it
  with no wrapper edit, and results match NumPy exactly
  ([native_broadcasting_design.md](native_broadcasting_design.md)). The
  native reductions are now **designed (v1.18) and implemented (v1.19)** —
  Phase A3 complete for `sum`/`mean` (`axis`/`keepdims`, negative axes;
  `max`/`argmax`/`min`/`product` deferred) via a scatter-accumulate
  kernel that is the dual of broadcasting — where broadcasting reads
  through zero strides, a reduction writes through zero strides — reading
  any strided/offset input directly and writing a freshly allocated
  row-major contiguous output, with honest order-sensitive floating-point
  behavior (NumPy comparison to a tolerance, not bit-for-bit).
  `NativeTensor` and the explicit backends inherited `sum`/`mean` with no
  wrapper edit, and reductions stay forward-only — the broadcast-backward
  relationship (a broadcast backward is a reduction over the broadcast
  axes) is the recorded reason reductions precede native autograd
  ([native_reductions_design.md](native_reductions_design.md)). Building
  on that, the dtype/device metadata contract is now **designed (v1.20)
  and implemented (v1.21)**, Phase A4 complete and **Phase A closed in
  code**: `dtype`/`device` are explicit, inspectable, validated string
  tags (`"float64"`/`"cpu"`) owned by `NativeStorage` and surfaced
  read-only through `NativeTensorCore`/`NativeTensor`, with
  default-preserving constructor arguments (so every existing call is
  byte-for-byte unchanged), matching-dtype/device operation guards, a hard
  no-promotion/no-silent-conversion rule, and — per the design's
  reject-over-inert recommendation — rejection of any
  non-`float64`/non-`cpu` construction so no tensor advertises a dtype the
  kernels cannot compute. It is metadata only: no kernel, no compute
  change, `to_numpy` still float64, and pure `normalize_dtype`/
  `normalize_device` helpers validate the tags
  ([native_dtype_device_metadata_design.md](native_dtype_device_metadata_design.md)).
  **Phase B is under way.** Advanced C++ v2.0 — the native autograd design
  — is complete (a Python-managed reverse-mode graph at the `NativeTensor`
  layer, native gradients honoring the v1.21 `grad.dtype == tensor.dtype`
  / `grad.device == tensor.device` contract, broadcasting backward via A3
  reductions). **v2.1 implemented the autograd metadata skeleton** —
  opt-in `requires_grad`/`grad`/`is_leaf`, `zero_grad`/`detach`, and a
  reverse-topological `backward` driver with `NativeTensor`-backed
  gradients — and **v2.2 — Core Native Autograd Operations — is now
  implemented**: `add`/`subtract`/`multiply`/`relu`/`sum`/`mean`/
  `matmul`/`reshape`/`transpose`/`T`/`contiguous_copy` are differentiable
  (graph nodes when an operand requires grad, plain forward tensors
  otherwise), broadcasting backward runs through a native `unbroadcast`
  reduction, sum/mean broadcast their upstream back natively, and the one
  new C++ kernel is the fused `relu_backward` — with every rule verified
  against finite differences and a deterministic native demo
  (`examples/native_autograd_demo.py`). **v2.3 — Native Narrow Backward —
  is now implemented**, completing the view-backward set: `narrow(dim,
  start, length)` builds a graph node when its parent requires grad, and
  its backward **scatters** the upstream gradient into a fresh owning
  row-major contiguous zeros tensor of the parent's shape at the narrowed
  region (un-narrowed positions stay zero) through the one new C++ kernel
  `tf_core_narrow_backward`, the odometer dual of `tf_core_sum`. The
  gradient lives at the logical shape, so transposed, narrowed, and
  nonzero-offset parents all differentiate correctly, and there is no
  NumPy in the gradient path; `NativeTensorCore` and the C++ kernels still
  own no graph state. **v2.4 — Native Autograd Graph Lifetime Policy — is
  now implemented** (a Python-only `NativeTensor` change): `backward` takes
  a `retain_graph` flag (validated as a real bool first), the default
  `backward(retain_graph=False)` is one-shot and releases the traversed
  operation graph on success, a later backward through a freed graph raises
  a clear error (never silently truncating history), `retain_graph=True`
  keeps the graph for another pass, leaf gradients accumulate until
  `zero_grad()`, and a failed pass rolls back with no partial commit or
  partial free — explicitly not full PyTorch parity. **v2.5 — Native
  Autograd Benchmark Characterization — is now done** (a measurement-only
  milestone that changes no behavior): a reproducible harness
  (`benchmarks/benchmark_native_autograd.py`) times four modes —
  forward-native, forward+graph-construction, fresh forward+backward, and
  repeated retained backward — across five workloads, with a correctness
  gate, median/spread reporting, a JSON mode, and one honest
  hardware-specific snapshot carrying no speed assertions (see
  [native_autograd_benchmarks.md](native_autograd_benchmarks.md)). **v2.6 —
  Phase B Guardrails and Completion — is now done, closing Phase B in
  code**: cross-cutting guardrail tests
  (`tests/test_native_autograd_guardrails.py`) lock the completed engine's
  invariants (a NumPy-no-fallback runtime guard, `NativeTensor` /
  `tensorforge.Tensor` isolation, explicit-backend / no-implicit-dispatch
  behavior, gradient-ownership, graph-lifetime, detach, view+offset,
  closed-operand safety, the kernel-registry boundary, and the benchmark
  mode contract), and the final Phase B support matrix and the explicit
  divide-backward decision (deferred beyond Phase B) are documented, with no
  operation, kernel, or optimization added. **Phase C — a native training
  stack — is now under way: Advanced C++ v3.1 — NativeParameter and
  Parameter Registration Contract — is implemented**:
  `tensorforge.experimental.NativeParameter`, a `NativeTensor` subclass
  whose instances are always graph-free owning leaves (construction takes
  an independent owning contiguous copy of array-like data or an existing
  tensor's current value, inheriting no graph history; `requires_grad` is
  a validated real bool defaulting to `True`, `False` giving a frozen but
  registerable parameter; every operation, view, copy, and `detach()`
  returns a plain `NativeTensor` — parameter-ness never propagates; and
  identity is object identity, never value), plus
  `NativeParameterRegistry`, the minimal insertion-ordered registration
  contract the future `NativeModule` will embed (dot-free non-empty string
  names, `NativeParameter`-only slots with `None` unregistering,
  position-preserving replacement, alias-visible named traversal, and
  identity-deduplicated unique traversal). **v3.2 — NativeModule Core and
  Recursive Registration — is implemented on top of it**:
  `tensorforge.experimental.NativeModule`, the Python-side
  module-hierarchy core — attribute assignment registers parameters and
  child modules (ordinary values, plain `NativeTensor`s, and
  stable-framework objects stay ordinary attributes; one category per
  name, latest assignment wins, `None` unregisters), explicit
  `register_parameter`/`add_module` mirror assignment, recursive
  `parameters()`/`named_parameters()`/`modules()`/`named_modules()`
  traversal is deterministic depth-first with identity deduplication,
  first-discovered canonical dotted names, shared-parameter/-module
  handling, and cycle safety, plus recursive `zero_grad()` and
  bool-validated `train()`/`eval()` propagation — still with no layer,
  loss, optimizer, state_dict, or training loop, no storage ownership,
  and `tensorforge.Tensor`/`tensorforge.nn` untouched. **v3.3 — Native
  State Dictionary Contract — is implemented on top of that**: the
  in-memory, parameters-only state contract — `state_dict()` snapshots
  each unique parameter's value under its canonical dotted name into an
  independent owning graph-free `NativeTensor` (shared parameters once,
  first-discovered path wins, frozen included, no storage shared with
  the model in either direction), and
  `load_state_dict(state_dict, strict=True)` copies values back into the
  existing `NativeParameter` objects atomically (full preflight
  validation naming the failing key, strict/non-strict key handling with
  an immutable missing/unexpected result, exact shape/dtype/device
  matching with no casting/reshaping/broadcasting, stage-then-commit
  with rollback) while preserving parameter identity, registration,
  shared aliases, `requires_grad`/frozen state, gradients, and training
  flags — still with no layer, loss, optimizer, file serialization,
  checkpoint, or training loop. **v3.4 — NativeLinear — is implemented
  on top of all of it**: the first concrete native layer,
  `tensorforge.experimental.NativeLinear` — a `NativeModule` with a
  `(in_features, out_features)` `NativeParameter` weight (the
  `x @ weight` orientation) and optional `(out_features,)` bias,
  deterministic fan-in uniform initialization from a local seeded
  generator (global random state untouched), full argument validation before
  native allocation, strictly 2-D input semantics, forward as pure
  existing native operations (`matmul` + broadcast `add`) so the
  existing autograd supplies backward (verified analytically and by
  central finite differences), frozen-parameter support, deterministic
  `["weight", "bias"]` registration and state-dict keys, and full v3.3
  load compatibility. **v3.5 — NativeReLU and NativeSequential — is
  implemented, completing the first composable native model surface**:
  `NativeReLU`, a parameter-free shape-generic activation module over
  the existing native `relu()` and its existing backward, and
  `NativeSequential`, an ordered composition container whose children
  live in contiguous integer-string execution slots (`"0"`, `"1"`, ...)
  where execution order is the registered order — replacement preserves
  position, `append` takes the next slot, and gaps, non-slot child
  names, direct parameters, slot removal, and self-insertion are
  rejected — with position-based execution versus identity-deduplicated
  traversal/state for shared children, empty-sequence identity forward,
  nested composition, slot-derived state keys (`"0.weight"`,
  `"2.bias"`, nested `"0.0.weight"`), and a Linear→ReLU→Linear model
  verified end to end by exact references and central finite
  differences. **v3.6 — NativeMSELoss — is implemented, closing the
  forward side of the native training story**: a parameter-free loss
  module composing existing native operations (`subtract` →
  `multiply(diff, diff)` → `mean`/`sum`) into a scalar loss whose
  gradients come entirely from the existing autograd
  (duplicate-parent accumulation, subtract's sign split, and the
  existing native mean backward's `1/N` — no division, no manual
  backward), with exactly `"mean"`/`"sum"` reductions, strict
  exact-shape/no-broadcasting validation, empty state, and exact plus
  finite-difference verification through a full
  Linear→ReLU→Linear→MSE chain. **v3.7 — Native Parameter Mutation
  Safety and Versioning — is implemented**: every `NativeParameter`
  carries a read-only monotonic value version counting replacements of
  the owned value; `copy_value_(source)` is the one controlled no-grad
  mutation primitive (identity, gradients, `requires_grad`, and
  registrations preserved; native never-aliased copies; failure changes
  nothing); `load_state_dict` increments each matched canonical
  parameter once, after its atomic commit; and graphs record expected
  versions where backward reads a direct parameter operand's forward
  value (`multiply`/`matmul`/`relu`), so `backward()` raises a
  deterministic stale-graph error — before any callback or gradient
  commit — when such a parameter was mutated after forward, while
  value-independent graphs (add/subtract/reductions/views) stay valid
  with correct gradients. **v3.8 — NativeSGD — is implemented**: the
  first native optimizer — minimal SGD over identity-deduplicated
  `NativeParameter` objects with a strictly validated learning rate
  and a two-phase mutation-atomic `step()` that stages graph-free
  native updates and commits them through `copy_value_` (frozen and
  gradient-less parameters skipped, identities and gradients
  preserved, one version increment per updated parameter), plus
  preflighted `zero_grad()` — no momentum, weight decay, parameter
  groups, optimizer state, or training loop. **v3.9 — the native MLP
  training proof — is implemented**: `examples/native_mlp_training.py`
  trains a 2→8→ReLU→1 native MLP on fixed synthetic regression data
  for 25 deterministic SGD steps entirely through the native stack —
  a fresh graph every iteration, one version increment per parameter
  per step, stable identities, explicit lifetime handling, and a
  monotonic 99.5% loss reduction, all bit-reproducible across runs.
  **v3.10 — the integration checkpoint — is complete**: the branch's
  first major usable native training checkpoint, adding no numerical
  behavior — honest README/summary/architecture presentation of both
  framework lines, the canonical
  [native support matrix](native_support_matrix.md), documentation and
  export guardrails, and CI/repository-hygiene audits — leaving
  `advanced/cpp-backend` ready for its first pull request into `main`
  after validation. **v3.11 — native optimizer math primitives — is
  complete**: differentiable native `sqrt` and `reciprocal` through
  the whole stack (new odometer + contiguous fast-path C++ kernels,
  core methods, wrapper methods), with saved-forward-result backwards
  — each derivative reads the recorded output, never the parent's
  current value, so neither records a parameter version and mutation
  after forward leaves those edges valid — IEEE float64
  exceptional-value semantics locked by tests, and no general
  division (`reciprocal` + `multiply` compose what the stack needs):
  the reusable math NativeAdam requires. **v3.12 — NativeAdam — is
  complete**: the native adaptive optimizer — validated
  `lr`/`betas`/`eps`, identity-deduplicated parameters, eagerly
  allocated optimizer-owned native first/second-moment buffers and
  per-parameter step counters, bias-corrected graph-free updates
  staged at the core level (reciprocal + sqrt, no division) and
  committed through `copy_value_`, gradients retained until
  `zero_grad()`, mutation-atomic public failures, and an explicit
  state lifetime (`close()`) — with no weight decay, AMSGrad,
  parameter groups, or schedulers. **v3.13 — the native optimizer
  state contract — is complete**: in-memory
  `state_dict()`/`load_state_dict()` on both native optimizers — one
  versioned schema (format 1, exact optimizer type tag, ordered
  positional shape/dtype/device parameter metadata; no ids, names,
  values, or gradients), caller-owned independent NativeTensor m/v
  snapshots and per-parameter step counts for NativeAdam, exact
  validation with staged atomic loading that never touches parameter
  values, versions, gradients, or retained graphs, and a proven
  deterministic in-memory training continuation. **v3.14 — native
  checkpointing and deterministic file resume — is complete**:
  `save_native_checkpoint`/`load_native_checkpoint` persist the model
  plus optionally one native optimizer's state and JSON metadata to
  one explicit pickle-free NPZ archive (a versioned UTF-8/JSON
  manifest plus indexed float64 arrays; `allow_pickle=False` loading;
  no ids, gradients, versions, or graph data serialized), with strict
  full-archive validation before any live mutation, strict optimizer
  presence/type matching, atomic temporary-file replacement,
  deterministic bit-identical file resume
  (`examples/native_checkpoint_resume.py`), and no scheduler or
  random-state capture and no `map_location`. **v3.15 — Phase C
  guardrails and completion — is complete, closing Phase C in code**:
  a cross-cutting completion test suite (`tests/test_native_phase_c.py`)
  locks the integrated invariants that span several components — full
  NativeSGD and NativeAdam training lifecycles under a NumPy tripwire,
  the shared-parameter story end to end (registration → backward
  accumulation → optimizers → snapshots → checkpoints), mixed
  active/frozen/`grad=None`/zero-gradient collections, late parameter
  activation, repeated optimizer-state and checkpoint-resume cycles,
  failure recovery at every boundary, the four-way graph-staleness
  distinction, lifetime/close discipline, and the public surface — plus
  documentation completion, the finalized support matrix, and
  build/CI/hygiene verification, with **no new numerical behavior**.
  Phase C is **complete**; the intended sequence continues with
  the native CNN stack, the CUDA runtime, dtype/AMP work,
  Transformer/text experiments, distributed training, and the final
  portfolio release. CUDA/GPU experiments are still entirely future
  work. The Python framework stays the reference implementation.
- **The native roadmap** — the longer arc the explicit
  experimental native line (`tensorforge.backends`,
  `tensorforge.experimental`) is building toward, in phases, each landing
  only when the previous is tested and documented:
  - **Phase A — native CPU runtime.** A1: the contiguous elementwise
    fast path — **complete** (designed v1.13, implemented v1.14,
    benchmark impact reported v1.15). A2: broadcasting for elementwise
    ops — **complete** (designed v1.16, implemented v1.17). A3: reductions
    (sum/mean first; max/argmax/min/product later) — **complete**
    (designed v1.18, implemented v1.19). A4: explicit dtype and device
    metadata (float64/cpu) — **complete** (designed v1.20, implemented
    v1.21, metadata-only), which **closes Phase A in code**.
  - **Phase B — native autograd (complete).** The v2.0 design is complete
    (a Python-managed reverse-mode graph at the `NativeTensor` layer — see
    [native_autograd_design.md](native_autograd_design.md)); **v2.1
    implemented the metadata skeleton and reverse-topological backward
    driver** (opt-in `requires_grad`/`grad`/`is_leaf`,
    `zero_grad`/`detach`/`backward`, `NativeTensor`-backed gradients); and
    **v2.2 implemented the core backward operations** — add, subtract,
    multiply, relu (one new fused kernel), sum, mean, matmul,
    reshape/transpose/T, contiguous_copy, and broadcasting backward via a
    native `unbroadcast`, finite-difference-verified, with a
    deterministic native autograd demo; **v2.3 implemented native narrow
    backward** — the scatter that was v2.2's one deferral — through a
    second new fused kernel (`tf_core_narrow_backward`, the odometer dual
    of `sum`), completing the view-backward set with transposed / narrowed
    / nonzero-offset parents all handled; and **v2.4 implemented the graph
    lifetime policy** — a one-shot `backward(retain_graph=False)` that frees
    the traversed graph on success, opt-in `retain_graph=True` reuse,
    deterministic freed-graph errors, and snapshot-based failure safety (a
    Python-only change; no kernel touched); and **v2.5 characterized the
    stack** with a measurement-only benchmark harness (four modes across
    five workloads, correctness gate, median/spread reporting, JSON output,
    one hardware snapshot, no speed assertions); and **v2.6 completed Phase
    B** — cross-cutting guardrail tests
    (`tests/test_native_autograd_guardrails.py`) that lock the engine's
    invariants (NumPy-no-fallback runtime guard, `NativeTensor` /
    `tensorforge.Tensor` isolation, explicit-backend behavior,
    gradient-ownership, graph-lifetime, detach, view+offset, closed-operand
    safety, kernel-registry boundary, benchmark mode contract), the final
    Phase B support matrix, and the explicit divide-backward decision
    (deferred beyond Phase B), adding no operation, kernel, or optimization.
  - **Phase C — native training stack (complete).** **v3.1 —
    NativeParameter and Parameter Registration Contract — is complete**: a
    `NativeParameter` subclass of `NativeTensor` whose instances are
    always graph-free owning leaves with validated `requires_grad`
    (frozen parameters stay registerable), independent owning contiguous
    construction from array-like data or an existing tensor's current
    value, operation results that are always plain `NativeTensor`
    (parameter-ness never propagates), object-identity semantics for
    future optimizer state, and the minimal insertion-ordered
    `NativeParameterRegistry` (dot-free names, `None` unregisters,
    position-preserving replacement, alias and identity-deduplication
    rules). **v3.2 — NativeModule Core and Recursive Registration — is
    complete**: `NativeModule` with automatic assignment registration
    (one category per name, latest-assignment-wins collisions, `None`
    unregistering, ordinary attributes for everything that is not a
    `NativeParameter`/`NativeModule`), explicit
    `register_parameter`/`add_module` mirroring assignment, recursive
    `parameters()`/`named_parameters()`/`modules()`/`named_modules()`
    with deterministic depth-first order, identity deduplication,
    first-discovered canonical dotted names, shared-structure and cycle
    safety, recursive `zero_grad()`, and bool-validated
    `train()`/`eval()` propagation. **v3.3 — Native State Dictionary
    Contract — is complete**: in-memory, parameters-only
    `state_dict()`/`load_state_dict()` — canonical deterministic dotted
    keys, independent owning graph-free snapshot values (native copy
    path, no NumPy), strict/non-strict loading with an immutable
    missing/unexpected-keys result, exact shape/dtype/device validation,
    stage-then-commit atomicity with rollback, and full preservation of
    parameter identity, shared aliases, gradients, `requires_grad`, and
    training state — no file serialization, checkpoints, or optimizer
    state yet. **v3.4 — NativeLinear — is complete**: the first concrete
    native layer — `(in_features, out_features)` weight orientation,
    optional `(out_features,)` bias, deterministic seeded fan-in uniform
    initialization (local generator, global random state untouched), validated
    constructor and strictly 2-D input contract, forward as pure
    existing native `matmul` + broadcast `add` so the existing autograd
    is the backward implementation (exact analytical and
    finite-difference verified), frozen-parameter support, deterministic
    `["weight", "bias"]` keys, and full v3.3 state-dict compatibility —
    no losses, optimizers, containers, activations, or training loop
    yet. **v3.5 — NativeReLU and NativeSequential — is complete**: the
    parameter-free shape-generic `NativeReLU` over the existing native
    relu autograd (no in-place mode), and the `NativeSequential`
    ordered container — contiguous integer-string execution slots with
    the registration funnel enforcing that registered children and
    execution order never diverge (position-preserving replacement,
    contiguous `append`, rejection of gaps, non-slot names, direct
    parameters, slot removal, and self-insertion), a minimal
    `len`/`iter`/indexing/`append` surface, position-based execution
    with identity-deduplicated traversal and state for shared children,
    empty-sequence identity forward, nested composition, and exact plus
    finite-difference verified backward through a full
    Linear→ReLU→Linear model. **v3.6 — NativeMSELoss — is complete**:
    the first native loss — a parameter-free NativeModule composing
    native `subtract`/`multiply`/`mean`/`sum` into a scalar loss (mean
    default, sum the only alternative; exact string validation), with
    strict exact-shape/no-broadcasting and dtype/device validation
    before any graph construction, gradients supplied entirely by the
    existing autograd (duplicate-parent factor 2, subtract's target
    sign, the existing native mean backward's `1/N` — no division and
    no manual backward), empty state, train/eval independence, and
    exact plus finite-difference verification for both operands under
    both reductions and through a full Linear→ReLU→Linear→MSE model.
    **v3.7 — Native Parameter Mutation Safety and Versioning Contract —
    is complete**: a read-only monotonic value version on every
    NativeParameter counting replacements of the owned value, the
    controlled no-grad `copy_value_` mutation primitive (identity,
    gradients, `requires_grad`, and registrations preserved; native
    never-aliased owning copies; atomic failure behavior), state
    loading incrementing each matched canonical parameter exactly once
    after its atomic commit (rollback restores values and versions),
    forward-time expected-version capture on the value-sensitive
    operations (`multiply`/`matmul`/`relu` — the audited set whose
    backward reads direct-parent forward values), and a deterministic
    stale-graph backward error raised before any callback or gradient
    commit — while value-independent graphs (add/subtract/reductions/
    views) stay valid across mutation with correct gradients; shared
    parameters expose one version through every alias. **v3.8 —
    NativeSGD — is complete**: the first native optimizer — minimal
    SGD (`value ← value - lr * grad`) over identity-deduplicated open
    NativeParameter objects stored by strong reference in
    first-occurrence order (duplicate references and shared aliases:
    one entry, one update, one version increment per step), a strictly
    validated learning rate (real, non-bool, finite, strictly
    positive), and a two-phase mutation-atomic `step()` — preflight,
    frozen/`grad=None` skipping, exact gradient validation, graph-free
    native staging at the core level, and commits through the v3.7
    `copy_value_` path with gradients retained until a preflighted
    `zero_grad()` — verified through a one-step
    Sequential/Linear/ReLU/MSE integration; no momentum, weight decay,
    parameter groups, optimizer state, schedulers, or training loop.
    **v3.9 — the native MLP training proof — is complete**: the first
    complete multi-iteration native CPU training run, as an example
    plus integration tests with zero changes to the stack —
    `examples/native_mlp_training.py` trains
    `NativeSequential(NativeLinear(2, 8, seed=0), NativeReLU(),
    NativeLinear(8, 1, seed=1))` on 8 fixed synthetic regression
    samples for 25 steps of `NativeSGD(lr=0.1)`, with a completely
    fresh graph each iteration (no retained graphs — the v3.7 stale
    guard never fires in the loop, and deliberate retention across a
    step still raises), gradients confirmed present after backward,
    retained through `step()`, and cleared by `zero_grad()`, exactly
    one version increment per parameter per step, stable parameter
    identities/names/state keys, explicit per-iteration and
    end-of-run tensor release, a NumPy-compute tripwire over a full
    run, and a monotonic deterministic loss trajectory (2.107864 →
    0.009529, a 99.5% reduction) that repeats bit-identically.
    **v3.10 — the integration checkpoint — is complete**: no numerical
    changes — the canonical support matrix, corrected README/summary/
    architecture docs, documentation/export guardrails, and CI and
    hygiene audits, marking the first major usable native training
    checkpoint and PR readiness. **v3.11 — native optimizer math
    primitives — is complete**: native differentiable `sqrt` and
    `reciprocal` (kernels → bindings → core → wrapper → autograd),
    saved-forward-result backwards that record no parameter versions,
    IEEE float64 exceptional-value semantics, arbitrary strided/offset
    view support with fresh owning contiguous outputs, and finite-
    difference-verified gradients — the reusable math for the native
    adaptive optimizer, with general division still deliberately
    unshipped. **v3.12 — NativeAdam — is complete**: the native
    adaptive optimizer over the v3.7 mutation contract and the v3.11
    primitives — the NativeSGD parameter contract unchanged
    (identity-deduplicated open NativeParameters, position-named
    errors), strictly validated `lr`/`betas`/`eps`, eagerly allocated
    optimizer-owned graph-free moment buffers and per-parameter step
    counters (skipped parameters never age; late activation starts at
    t = 1; shared aliases advance once), bias-corrected updates staged
    entirely at the autograd-unaware core level and committed through
    `copy_value_` (one version increment per update, old moments
    closed only after their replacements are installed), preflighted
    `zero_grad()`, mutation-atomic public failure behavior with the
    documented asynchronous-interruption windows, and an explicit
    idempotent `close()` for the optimizer-owned state — with no
    weight decay, AMSGrad, parameter groups, or schedulers.
    **v3.13 — the native optimizer state contract — is complete**:
    `state_dict()`/`load_state_dict()` on NativeSGD and NativeAdam —
    a shared format-1 schema with an exact optimizer type tag,
    validated hyperparameters, and ordered positional
    shape/dtype/device parameter metadata (mapping across instances
    is positional over the deterministic identity-deduplicated
    parameter order; no object ids, names, parameter values,
    gradients, or graph data are serialized); NativeAdam adds
    per-parameter step counts plus caller-owned independent
    graph-free NativeTensor moment snapshots; loading is
    validate → stage → commit with exact validation (no casting,
    reshaping, broadcasting, or device movement), independent
    optimizer-owned copies of every input moment (caller state
    read-only, never adopted or consumed), replaced internal buffers
    closed only after installation, and mutation-atomic ordinary
    failures — never touching parameter values, versions, gradients,
    registrations, or retained graphs, with deterministic in-memory
    continuation proven against an uninterrupted run.
    **v3.14 — native checkpointing and deterministic file resume — is
    complete**: `save_native_checkpoint`/`load_native_checkpoint`
    over the existing state contracts — one pickle-free NPZ archive
    per checkpoint (format `"tensorforge.native_checkpoint"`,
    version 1) holding a UTF-8/JSON uint8 manifest (canonical model
    keys and positional optimizer metadata mapped explicitly to
    deterministic indexed float64 array names; user metadata
    included; nothing volatile serialized) plus the model parameter
    and optimizer moment arrays; validated save with every snapshot closed in a
    `finally` and an atomic collision-safe temporary-file
    `os.replace` (existing destinations survive failures, no
    temporary residue); validate → stage → commit loading under
    `allow_pickle=False` with strict optimizer presence/type
    matching, full pre-mutation validation of thirty-plus corruption
    cases, commits only through the existing module/optimizer
    loaders (model versions +1 each and retained sensitive graphs
    stale, per the existing contracts; optimizer loading moves no
    versions), deterministic bit-identical file resume for NativeAdam
    and next-step equivalence for NativeSGD, and a focused
    resume example. No scheduler state, random-state
    capture/restoration, `map_location`, partial loading, merging,
    sharding, compression, or encryption.
    **v3.15 — Phase C guardrails and completion — is complete,
    closing Phase C**: a cross-cutting completion test file
    (`tests/test_native_phase_c.py`) proving the components compose
    correctly under normal training, shared/frozen/`grad=None`/
    zero-gradient parameters, late activation, repeated snapshot/load
    and checkpoint-resume cycles, failure and corruption at every
    boundary, explicit native lifetime management, and the four-way
    stale-graph distinction; documentation completion and support-
    matrix finalization; and build/CI/hygiene verification — no new
    numerical behavior.
  - **Phase D — native CNN stack — COMPLETE (milestones D0–D12).** Every
    CNN layer — flatten, convolution, and max-pooling, operations and
    modules alike — has shipped; the deterministic native CNN training +
    checkpoint-resume proof runs end to end; and the phase closed with
    cross-cutting integration tests, honest CNN benchmarks, and ASan/UBSan
    validation. The **D0 architecture contract is written** —
    [native_cnn_design.md](native_cnn_design.md) locks the layouts
    (NCHW activations, OIHW convolution weights, cross-correlation), the
    argument and output-shape contracts, the non-contiguous-input policy
    (copy-then-compute at the wrapper), the fused-primitive/autograd
    ownership split, the max-pool winner-index representation, the C ABI
    families and C++/Python source organization, the full test and
    benchmark strategy, and the **D0–D12 milestone sequence**
    (`NativeFlatten`; native convolution forward and its input/weight/
    bias gradients; the convolution module; native max-pooling forward,
    backward, and module; and a deterministic native CNN
    training/checkpoint-resume proof). **D1 has shipped:** `NativeFlatten`,
    a parameter-free, buffer-free batch-preserving flatten
    Python-composed from the existing `reshape`/`contiguous_copy`
    operations and their autograd (no new kernel, no custom backward),
    returning an independent owning result so it composes safely in a
    `NativeSequential`. **D2 has shipped** the first native convolution
    code: an **internal** CPU float64 forward compute kernel
    (`tf::conv2d_forward_contiguous` — direct nested-loop
    cross-correlation, symmetric zero padding, optional bias), verified
    by a dependency-free C++ CTest binary against hand-computed cases and
    stable-framework parity. **D3 has shipped** the forward-only
    convolution *layer*: the exported, exception-guarded C ABI wrapper
    `tf_core_conv2d_forward` (self-validating, contiguous-only), its
    ctypes/`errcheck` registration, and `NativeTensorCore.conv2d_forward`
    — a Python-reachable, autograd-unaware Core method that validates
    shapes, computes the output shape in overflow-safe Python ints,
    copies non-contiguous operands (Policy B), and returns a fresh owning
    contiguous NCHW output matching the stable convolution to tolerance.
    **D4 has shipped** the **internal** CPU float64 convolution
    input-gradient compute kernel (`tf::conv2d_input_backward_contiguous`
    — a hidden C++ symbol: the deterministic scatter-add adjoint of the
    forward cross-correlation, zero-initializing its own output, verified
    by a dependency-free C++ CTest against hand-computed cases, stable
    parity, and central finite differences). Like D2 it is deliberately
    **not exposed to Python** — the exported backward C ABI wrapper, its
    Core method, and the autograd node are D6. **D5 has shipped** the
    **internal** CPU float64 convolution weight-gradient compute kernel
    (`tf::conv2d_weight_backward_contiguous` — a hidden C++ symbol,
    deterministic zero-initialized accumulation, verified against
    hand-computed cases, an explicit-zero padded-materialization oracle,
    stable parity, and central finite differences) and **locked and
    validated the bias-gradient path as a reuse of the existing native
    `sum` reduction** (`g.sum(0).sum(1).sum(1) → (O,)`, no dedicated
    kernel), proved in a focused Python contract test. **D6 completed the
    differentiable native convolution operation**: the exported guarded
    backward C ABI wrappers (`tf_core_conv2d_input_backward`,
    `tf_core_conv2d_weight_backward`), the Core backward methods, the bias
    gradient composed from the existing native `sum` reduction (no dedicated
    kernel), and the Python-managed **`NativeTensor.conv2d`** autograd
    primitive — forward reuse of the D3 wrapper, input/weight/bias
    gradients, deterministic `(input, weight[, bias])` parent ordering,
    conditional stale-value version tracking, and reuse of the existing
    backward snapshot/rollback engine — verified against stable parity,
    finite differences, and all `requires_grad` combinations. **D7 completed
    the trainable native convolution module**: an OIHW weight / optional
    `(O,)` bias native-parameter layer with deterministic uniform conv
    fan-in initialization (`bound = 1/sqrt(in_channels·kh·kw)`, a local
    generator with the global state untouched), 4-D NCHW input validation,
    and backward
    supplied entirely by the D6 `conv2d` autograd — no new kernel, C ABI
    symbol, or custom module backward. It registers in `NATIVE_MODULES`,
    exports from `tensorforge.experimental`, and rides the existing
    state_dict/checkpoint/optimizer paths unchanged. **D8 has shipped the
    forward-only native max-pooling layer**: the internal CPU float64
    compute kernel `tf::maxpool2d_forward_contiguous` (a hidden C++ symbol
    that produces the pooled values and the saved winner indices in one
    deterministic row-major pass — padding participates as a conceptual
    `-inf`, ties keep the first occurrence, and a completely padded window
    yields `-inf` with the `-1` sentinel), the exported guarded C ABI
    wrapper `tf_core_maxpool2d_forward` with its ctypes/`errcheck`
    registration, and `NativeTensorCore.maxpool2d_forward` — a
    Python-reachable, autograd-unaware Core method that validates the
    arguments and the `H*W ≤ 2^53` winner-exactness bound in Python ints
    before allocating anything, copies a non-contiguous input (Policy B),
    allocates the output and the **private** winner buffer in a
    failure-atomic order, and matches the stable pooling reference
    exactly. **D9 completed the differentiable native pooling
    operation**: the internal scatter-add kernel
    `tf::maxpool2d_backward_contiguous`, the exported guarded
    `tf_core_maxpool2d_backward` wrapper (which validates every saved
    winner — the sentinel or an exact in-range integer — before scattering,
    and never rounds), `NativeTensorCore.maxpool2d_backward`, and the
    Python-managed **`NativeTensor.maxpool2d`** autograd node. Its single
    input-gradient callback routes the upstream through the winners the
    forward saved — never rereading the input, never recomputing a maximum,
    and recording **no** parameter-version snapshot (a deliberate contrast
    with convolution) — with overlapping windows accumulating and padding
    winners dropped. The private winner buffer became graph-owned state
    released exactly when the graph history is (freed by a one-shot
    backward or `close()`, retained under `retain_graph=True`, and kept
    alive across a failed retryable backward). **D10 completed the native
    pooling layer**: a parameter-free, buffer-free module that normalizes
    its window arguments to `(height, width)` tuples (no stride means
    non-overlapping windows) and delegates its forward entirely to that
    operation — no new kernel, C ABI symbol, custom backward, parameters,
    buffers, or checkpoint schema, and no winner state held between calls.
    It exports from `tensorforge.experimental`, contributes no
    state-dictionary keys, and composes in a `NativeSequential` beside the
    convolution, activation, flatten, and linear layers, so the native
    optimizers ignore it naturally. **D11 proved the whole stack trains
    end to end**: `examples/native_cnn_training.py` learns a genuinely
    spatial target — the strongest bright-to-dark vertical edge of eight
    fixed 6×6 images — through convolution, activation, pooling, flatten,
    and a linear head with the native MSE loss and the native adaptive
    optimizer, dropping the loss from about 0.7713 to about 0.0111 in 40
    deterministic steps; and a run interrupted at step 15, checkpointed
    with its optimizer state and resumed into a completely fresh
    model/optimizer pair, reproduces the uninterrupted run **exactly**
    (loss history, final predictions, every parameter value, and every
    optimizer state entry), adding no kernel, operation, loss, optimizer,
    or checkpoint schema. **D12 closed the phase**: cross-cutting
    integration tests spanning several CNN components at once
    (`tests/test_native_phase_d.py`), honest CNN characterization
    benchmarks (`benchmarks/benchmark_native_cnn.py` — measurement only,
    no speed claims), **ASan/UBSan validation** of the whole native CNN
    stack under Clang on Linux with no TensorForge diagnostic, a
    LeakSanitizer pass over the instrumented native CTests, documentation
    reconciliation across every status surface, and the replacement of the
    milestone-era doc guardrails with durable semantic checks. See the
    [support matrix](native_support_matrix.md) for the finalized status.
  - **Phase E — Native Classification and Stable Math — complete
    (E0-E10).** The **E0 architecture contract
    is written** —
    [native_classification_design.md](native_classification_design.md)
    locks the scope, the public API surface (`exp`, `log`, `softmax`,
    `log_softmax`, a fused `cross_entropy` from raw logits,
    `NativeCrossEntropyLoss`, and a reporting-only `native_accuracy`), the
    numerical-stability strategy (maximum shift and log-sum-exp, never
    `softmax().log()`), the backward-read and versioning matrix (`log` is
    the one version-checked operation; everything else reads saved state),
    the `int64` target contract (**at Phase E** the native runtime had no
    integer dtype at all; classification targets remain exact host-side
    label metadata under that contract, and Phase K milestone K2 did not
    widen cross-entropy to accept `NativeTensor` targets),
    the graph-owned saved-probability lifetime, the contiguous-only C ABI
    families and the new `cpp/src/classification.cpp` unit, the capability
    inventory placements, the unchanged checkpoint format version 1, and
    the **E0–E10 milestone sequence**. **E0 added no numerical behavior**
    — it was a design-and-reconciliation milestone. **E1 has shipped the
    native exponential**: the C++ kernel on both the strided-odometer and
    contiguous execution paths, the two guarded C ABI exports
    (`tf_core_exp`/`tf_core_exp_contiguous`, which validate their own
    handles, layout metadata, spans, and overflow), their ctypes
    registration, `NativeTensorCore.exp()`, and the differentiable
    `NativeTensor.exp()` whose backward is `upstream ×` the **saved
    forward output** — never rereading the input, so it records **no**
    parameter-version snapshot and survives post-forward mutation, with
    plain IEEE semantics (no clamping, no inserted bound). **E2 has
    shipped the native logarithm** through the same four layers, reusing
    E1's self-validating export contract unchanged: plain IEEE
    `std::log` (`log(±0)` is `-inf`, `log(negative)` is NaN — values, not
    errors), with a backward that **rereads the live input** as
    `upstream × reciprocal(x)` (composed from the existing `reciprocal`;
    no division operation was added). That makes a direct
    `NativeParameter` parent **version-checked**: mutating it after
    forward raises the deterministic stale-graph error before any
    gradient is committed anywhere in the graph — the deliberate
    counterpart to `exp`'s saved-output edge, which stays valid across
    the same mutation. **E3 has shipped the stable native softmax**, the
    phase's first fused probability transform and the reason
    `cpp/src/classification.cpp` now exists: a maximum-shift kernel
    computing `exp(x - max(x)) / sum(exp(x - max(x)))` in one pass over
    any single axis (positive or negative, rank >= 1), behind a
    **contiguous-only** C ABI with the Core layer applying the Phase-D
    Policy-B copy-then-compute for strided views. Its backward is the
    closed-form `y * (upstream - sum(upstream * y, axis, keepdims))`
    **composed from existing Core operations** — no dedicated backward
    kernel — reading only the saved probabilities, so it records no
    parameter version. E3 added no `NativeSoftmax` module and no public
    `max`, `argmax`, or division. **E4 has shipped the stable native
    log-softmax**, the phase's second fused probability transform: its
    **own** log-sum-exp kernel computing
    `(x - max(x)) - log(sum(exp(x - max(x))))` in one pass over any
    single axis, **never** `softmax().log()` — no probability buffer is
    formed and no division happens, so a probability too small to
    represent (which the composed form would round to 0 and report as
    `-inf`) still gets an accurate finite log-probability. It reuses E3's
    contiguous-only C ABI shape, its trust-boundary validator (now shared
    by both exports), and the same Core-level Policy-B copy-then-compute.
    Its backward is the closed-form
    `upstream - exp(y) * sum(upstream, axis, keepdims)` **composed from
    existing Core operations** — no backward kernel; `exp(y)` recovers
    the probabilities from the saved log probabilities — so it too reads
    only the saved output and records no parameter version. E4 added no
    `NativeLogSoftmax` module and no `NLLLoss`. **E5 has shipped the
    fused cross-entropy Core contract** — and *only* the Core layer. Two
    new kernels in the same classification unit: a forward that, in one
    deterministic pass per row, computes the maximum, the log-sum-exp,
    the **saved probabilities**, and the per-example loss
    (`log(Σ exp(x − m)) − (x[target] − m)`, reduced by `"sum"` or once by
    the batch size for `"mean"`) — never `-log(p[target])`, never
    `softmax().log()`-then-index, never `log_softmax()`-then-gather — and
    a backward that turns those **saved probabilities**, the copied
    targets, the reduction, and a **native one-element upstream** into
    `upstream · (p − onehot) / N` **without ever rereading the logits**,
    which are not even an argument. Both guarded exports
    (`tf_core_cross_entropy_forward`/`tf_core_cross_entropy_backward`)
    are contiguous-only for tensor data, revalidate **every target index**
    themselves rather than trusting Python, and leave every destination
    byte-for-byte unchanged when they reject. Targets are not native
    tensors — classification targets remain exact host-side label
    metadata under the Phase-E contract, and Phase K milestone K2 did not
    widen cross-entropy to accept `NativeTensor` targets: they are strictly
    validated — `bool` and floating-point labels rejected outright,
    including integral ones like `1.0` — and copied into independently
    owned contiguous read-only `int64` metadata, so mutating the caller's
    list or array afterwards cannot reach the kernel. The forward's two
    outputs fail atomically: any failure closes everything it allocated
    and returns no partial result. **E5 added no
    `NativeTensor.cross_entropy`, no autograd node, and no graph-owned
    saved state** — the private probabilities were Core-level state the
    caller owned. **E6 has shipped that differentiable operation**:
    `NativeTensor.cross_entropy(targets, reduction="mean")`, a single
    autograd node over the E5 contract that adds **no kernel, no C ABI
    export, and no numerical change**. It calls the E5 forward once,
    returns a **scalar** `NativeTensor`, and adopts the private saved
    probabilities as **graph-owned state** through the same
    `_from_op(..., graph_resources=...)` contract the Phase-D pooling
    winner buffer established: retained under `retain_graph=True` and across a
    failed retryable backward, released exactly once when the graph
    history is (a one-shot `backward()` or `close()`), and closed
    immediately when nothing requires gradients. The copied `int64`
    targets ride in the backward closure as immutable metadata — no
    native integer tensor — so caller mutation after the forward cannot
    reach the gradient. Because backward consumes only that saved state
    and a native scalar upstream, **it never rereads the logits** and the
    node records **no expected parameter version**: mutating a direct
    `NativeParameter` logits parent with `copy_value_` afterwards neither
    raises a stale-graph error nor changes the gradient, even across a
    retained graph — the `maxpool2d` archetype, and the deliberate
    contrast with `log`. Failures are atomic throughout: an E5 forward
    failure returns no tensor and builds no node, a graph-construction
    failure closes both E5 outputs, and a backward failure commits no
    gradient, leaks no gradient core, keeps the probabilities for a
    retry, and leaves the graph honestly un-freed. **E7 has shipped the
    public classification surface** and, like E6, added no training
    mathematics. ``NativeCrossEntropyLoss(reduction="mean")`` is a
    parameter-free, buffer-free ``NativeModule`` whose entire forward is
    ``logits.cross_entropy(targets, reduction=self.reduction)`` — no Core
    call, no ABI call, no NumPy, no ``softmax``/``log_softmax``
    composition, and no second formula — so it inherits every E5/E6
    guarantee rather than restating any of them, validates its
    ``"mean"``/``"sum"`` reduction in the constructor with the
    operation's own validator, and contributes no ``state_dict()`` or
    checkpoint keys (the reduction is constructor configuration, not
    model state). ``native_accuracy(logits, targets) -> float`` is a
    deliberately **reporting-only** helper, and the honesty of that label
    is the point: there is no accuracy kernel, no C ABI export, no Core
    method, and no autograd node. (**At Phase E** the runtime had no
    integer dtype for an index-producing reduction to return. The integer
    result dtype arrived at Phase K, K2, and a **native ``argmax``
    arrived at K3** — and this helper still reports through the host
    boundary, deliberately: rewriting it over the native ``argmax`` would
    still need an integer *equality* reduction that no milestone ships, so
    it would materialize to the host anyway, one operation later, with its
    one explicit conversion harder to see.) It validates rank-2 logits and
    targets through the *same* private preparer the cross-entropy forward
    uses — so the strict accepted/rejected matrix is identical at both
    call sites by construction — then materializes the logits **once**
    through the explicit public ``to_numpy()`` boundary, takes
    ``numpy.argmax(axis=1)`` (ties to the first maximal index), and
    returns a plain ``float`` in ``[0.0, 1.0]``. It builds no graph,
    touches no gradient, parameter, or version, allocates no native
    storage at all, and retains nothing, so a graph built before the call
    is still usable after it. The two capabilities land in the two
    inventories that describe their layers — ``NATIVE_LOSSES`` and the
    new ``NATIVE_METRICS``, reported by ``backend_info()`` — and with
    that, no classification name remains listed as unsupported.
    **E8 has shipped the end-to-end proof**, and it too added no
    numerical operation or runtime capability — it is Python example,
    integration tests, and documentation only.
    ``examples/native_classification_training.py`` trains a
    ``NativeConv2d(1, 4, 3, seed=0)`` → ``NativeReLU`` →
    ``NativeMaxPool2d(2)`` → ``NativeFlatten`` → ``NativeLinear(16, 3,
    seed=1)`` classifier — a named ``NativeModule`` whose children are
    registered through the ordinary assignment path — on twelve fixed
    6×6 single-channel images in **three** classes (vertical bar,
    horizontal bar, diagonal line; four positions each, committed as
    source literals, labels host integers, nothing generated,
    downloaded, augmented, or shuffled). Its **raw logits** go straight
    to ``NativeCrossEntropyLoss`` — there is deliberately no softmax or
    log-softmax layer, because the fused E5/E6 kernel is what keeps the
    loss stable — and 40 full-batch ``NativeAdam(lr=0.05)`` steps take
    the loss from **1.159638 to 0.000101** (99.99%) and the reporting
    accuracy from **0.3333 to 1.0000**, with ``native_accuracy`` called
    only outside the training mathematics (it converts to the host on
    purpose). Interrupting at step **15**, checkpointing model **and**
    optimizer state through the existing pickle-free path (format
    **version 1**, no new keys, no graph data or target metadata
    serialized), and resuming into a **fresh** model/optimizer pair
    reproduces the uninterrupted run **exactly**: the whole remaining
    loss suffix, every parameter, both ``NativeAdam`` moment buffers,
    every step
    counter, the final logits, the predictions, and the accuracy. Two
    independent uninterrupted runs are exactly equal too, repeated steps
    retain no completed graph or saved probability and grow no native
    storage, and a tripwire proves one complete step reaches no NumPy
    numerical routine and converts no tensor data. It is an
    **integration proof on one fixed task** — not a benchmark, not a
    speed claim, and not a generalization claim.
    **E9 has shipped the honest characterization benchmark**, and it
    changed no numerical runtime file and tuned nothing.
    ``benchmarks/benchmark_native_classification.py`` measures the seven
    operations the phase built — ``exp``, ``log``, ``softmax``,
    ``log_softmax``, the fused cross-entropy forward, its backward alone
    (a fresh graph is built outside the timer every repetition; no graph
    is reused and ``retain_graph`` is never used to skip the rebuild),
    and one complete classification training step (``zero_grad`` →
    forward → loss → ``backward`` → ``NativeAdam.step()``, with model,
    optimizer, and dataset construction, checkpoint I/O,
    ``native_accuracy``, and cleanup all outside the timed region).
    **Correctness is gated before every measurement**: a case validates
    shape, finiteness, reference parity, and input non-mutation — plus
    gradients for the backward case, and a finite loss, a real parameter
    update, an advanced optimizer step counter, a released graph, closed
    transients, and stable-line parity for the training step — and a
    failed gate exits nonzero and publishes no timing. Each case is
    labelled with the reference it actually used: ``stable_tensorforge``
    where a stable operation exists, ``numpy`` for ``log_softmax``
    (the stable line has no direct one, and ``softmax().log()`` is
    deliberately not used as the reference), and ``native_only`` where no
    honest analogue would exist. Timing is ``time.perf_counter_ns`` with
    warm-up, repeated measurements, setup and cleanup outside the timer,
    and **median** reporting alongside min, max, and spread; ``--smoke``
    and ``--json`` modes exist and no result file is written. The
    observed ratio is a **local characterization**, never a speedup
    claim: no test asserts a speed, no timing number is committed as a
    promise, and there is no CI performance gate anywhere.
    **E10 closed the phase**, adding no numerical capability of any kind:
    cross-cutting integration tests (``tests/test_native_phase_e.py``)
    covering the classification stack as one system, **Release and Debug**
    native builds (10/10 CTests each, zero compiler warnings), Clang
    AddressSanitizer and UndefinedBehaviorSanitizer validation of the
    whole classification stack (zero diagnostics attributable to
    TensorForge), a practical LeakSanitizer pass finding **no** native
    leak with the live-storage counters returning to baseline, the
    complete Python regression suite, the conversion of milestone-era
    "not yet shipped" documentation guardrails into durable semantic
    ones, and reconciliation of every authoritative status surface.
    **Phase E is complete**, and it expanded nothing beyond float64/CPU
    and added no implicit stable/native dispatch. Deliberately outside
    Phase E: more native activations beyond it, a native RNG and dropout,
    a CPU optimization phase for the deliberately naive kernels, and
    build/packaging evolution. Native normalization then became its own
    phase, below.
  - **Phase F — Native Normalization and Stateful Buffers — complete
    (F0–F9 all shipped).** The **F0
    architecture contract is written** —
    [native_normalization_design.md](native_normalization_design.md)
    locks the phase's objective (a fully native, differentiable,
    state-safe normalization stack: `NativeLayerNorm`,
    `NativeBatchNorm1d`, and `NativeBatchNorm2d`), the public API and its
    naming (layer-norm `weight`/`bias`; batch-norm `gamma`/`beta` plus
    the `running_mean`/`running_var` buffers, matching the stable
    reference), and — most consequentially — the decision that
    normalization is **composed from existing native operations**
    (`mean`, `subtract`, `multiply`, `add`, `sqrt`, `reciprocal`,
    `reshape`, broadcasting, `contiguous_copy`) so the phase adds **no
    C++ kernel, no C ABI export, no ctypes declaration, and no
    `NativeTensorCore` method** and inherits an exact backward from the
    existing autograd — including differentiation through the batch mean
    and variance, which is never detached. It also locks the layer-norm
    contract (trailing-dimension normalization, population variance,
    `eps` inside the square root, no buffers, identical in train and eval
    mode), the two batch-norm shape contracts (`(N, C)` reducing over the
    batch; NCHW `(N, C, H, W)` reducing over N/H/W with `(1, C, 1, 1)`
    broadcasting), and three load-bearing safety rules: a **live mutable
    running buffer is never captured as a rereadable graph operand** (an
    eval forward takes independent, owning, graph-free snapshots, which
    is precisely why buffers need no value version — `multiply`'s
    backward rereads a live operand, and the existing stale-version check
    only covers direct `NativeParameter` parents); the two running
    buffers update as **one atomic transaction** (validate, stage
    graph-free values, commit preserving both buffers' Python identity,
    roll back completely on any failure or interruption, close replaced
    cores exactly once, move no parameter version); and **registration
    implies no exclusive ownership**, so stateful examples and tests
    close both `parameters()` and `buffers()` explicitly and no contract
    relies on garbage collection. Persistent running statistics ride the
    **existing** state-dictionary and pickle-free checkpoint
    infrastructure with the format **unchanged at version 1** — new
    persistent keys need no schema bump — and the eventual exact resume
    must reproduce the loss suffix, parameters, optimizer state, running
    means, running variances, final predictions, **and the
    evaluation-mode output**. The ladder is **F0–F9**: F0 (this
    contract and repository reconciliation), F1 (atomic native-buffer
    state transactions, extracted and generalized from the existing
    `load_state_dict` staging/commit/rollback, plus the `STATE_SUPPORT`
    persistent-buffer correction), F2 (`NativeLayerNorm`), F3
    (`NativeBatchNorm1d`), F4 (`NativeBatchNorm2d`), F5 (state,
    checkpoint, and graph-safety hardening), F6 (deterministic
    normalized training and exact resume), F7 (benchmark
    characterization, with no speed assertion), F8 (cross-cutting
    integration and semantic guardrails), and F9 (phase closure).
    **F0 added no numerical behavior** — it is a design-and-reconciliation
    milestone — and **F1 is complete**: the private atomic native-buffer
    state transaction (`_native_state.py`) that §8 of the contract calls
    for, now the single implementation behind
    `NativeModule.load_state_dict` (whose public behavior is unchanged),
    plus the `persistent_buffers` correction to `STATE_SUPPORT` — an
    under-reported capability that has existed since before Phase D. F1
    is state management and capability reporting only and added **no
    normalization mathematics**. **F2 is complete**: `NativeLayerNorm`,
    the first native normalization module — stateless (no buffers,
    identical in train and eval), differentiable through the mean and the
    population variance, and composed entirely from existing native
    operations (`sqrt(var + eps)`, no Bessel correction) with no kernel,
    ABI symbol, `NativeTensorCore` method, custom backward, or
    `NativeTensor` normalization operation; `"NativeLayerNorm"` is now in
    `NATIVE_MODULES` and the exports, and `"layernorm"` has left
    `UNSUPPORTED`. **F3 is complete**: `NativeBatchNorm1d`, the **first
    stateful native numerical module** — `(N, C)` batch normalization
    whose training statistics are differentiable (gradients flow through
    the batch mean *and* the population variance, never detached), whose
    `running_mean`/`running_var` are **persistent native buffers**
    advanced by `(1 − momentum)·running + momentum·batch` from the *same*
    batch statistics, computed graph-free and committed as one **atomic
    two-buffer transaction** through the F1 primitive (both identities
    preserved, replaced cores closed exactly once, no parameter version
    moved), and whose evaluation mode reads **independent owning
    graph-free snapshots** of those buffers, so a later training step, or a
    buffer-only `load_state_dict()`/`load_native_checkpoint()`, cannot
    change an earlier eval graph's gradient (a full checkpoint load that
    also replaces `gamma`/`beta` still stales that graph through the
    unchanged parameter-version rule — correct, and proved separately). It too is composed from existing native operations
    — no kernel, C ABI symbol, ctypes declaration, `NativeTensorCore`
    method, custom backward, or `NativeTensor.batch_norm` operation — and
    the native checkpoint format stays at version 1;
    `"NativeBatchNorm1d"` is now in `NATIVE_MODULES` and the exports.
    **F4 is complete**: `NativeBatchNorm2d`, NCHW `(N, C, H, W)` batch
    normalization reducing over N, H, and W — one population mean and
    variance per channel over `N * H * W` values — built on the **same**
    shared private implementation as the 1-D shape, which it extends with
    nothing but its rank, its reduction axes, its `(1, C, 1, 1)`
    broadcast layout, and the channels-last permutation its rank-1
    `gamma`/`beta` need (the activation is transposed for the affine
    application, never the parameters, so the existing direct-parameter
    stale-value guard is preserved exactly). Running buffers stay `(C,)`.
    `"NativeBatchNorm2d"` is now in `NATIVE_MODULES` and the exports,
    and with both shapes live **`batchnorm` has left `UNSUPPORTED`**,
    which now reads exactly `("dropout", "float32", "cuda", "amp")`.
    **The numerical normalization module surface is complete, and F5 has
    hardened it.** **F5 is complete**: the
    exhaustive state/checkpoint, ownership, and graph-safety hardening — a
    focused `tests/test_native_normalization_state.py` plus narrow
    additions to the generic buffer and checkpoint suites — proves §7–§10
    by executable test (canonical dotted buffer keys, independent state
    snapshots, strict/non-strict loads, exact never-casting metadata
    validation, mixed parameter/buffer transaction atomicity, buffer
    identity across state and checkpoint loads, exact eval-output
    reproduction, the buffer-only-versus-full stale-graph distinction, the
    save/corrupt-load failure boundaries, eval-graph snapshot safety under
    `retain_graph` and a failed retryable backward, and the live-storage
    baselines); it is **tests and documentation only** — no numerical
    behavior, no new capability, and the checkpoint format stays version
    1. **F6 is complete**: `examples/native_normalization_training.py`
    trains a `NativeLinear → NativeBatchNorm1d → NativeReLU →
    NativeLayerNorm → NativeLinear` regressor for 24 deterministic
    `NativeAdam` steps with `NativeMSELoss` (98.9% loss reduction), proves
    two uninterrupted runs bit-identical, and resumes an interrupted run
    into a fresh model/optimizer pair that reproduces the remaining losses,
    every parameter, the NativeAdam state, both running statistics
    (`running_mean`/`running_var`), the final training-step prediction, and
    the final evaluation-mode output exactly (format version 1 unchanged,
    training flags runtime-only) — one example and its integration test,
    adding no capability. **F7 is complete**:
    `benchmarks/benchmark_native_normalization.py` characterizes the stack
    with nine cases — the `NativeLayerNorm` forward and backward, the
    `NativeBatchNorm1d` training forward, evaluation forward, and
    backward, the `NativeBatchNorm2d` training forward, evaluation
    forward, and backward, and one complete F6-style normalized training
    step — each **correctness-gated before any timing**, six against
    `stable_tensorforge` equivalents on identical state and three (the
    `NativeBatchNorm2d` shapes) labelled `native_only` because the stable
    line has no public 2-D batch-normalization module to time against,
    though those keep a rigorous NumPy NCHW and transformed-oracle
    correctness gate. Medians are reported with min,
    max, and spread after warm-up; `--smoke` and `--json` modes exist;
    **no result file is written, no speed is asserted, no timing number is
    committed, and no CI timing threshold exists** — measurement only,
    adding no capability. **F8 is complete**:
    `tests/test_native_phase_f.py` proves the cross-cutting interactions —
    one integrated `NativeConv2d → NativeBatchNorm2d → NativeReLU →
    NativeMaxPool2d → NativeFlatten → NativeLinear → NativeBatchNorm1d →
    NativeReLU → NativeLayerNorm → NativeLinear` classifier over raw
    logits and the fused classification loss, trained by `NativeAdam` and
    resumed **exactly** from one version-1 checkpoint (all four
    running-statistic buffers, the final training logits, and the
    evaluation-mode logits, predictions, and accuracy included); the
    three saved-resource families coexisting in one eval graph and
    releasing exactly once; buffer mutation leaving an earlier graph
    valid while parameter mutation correctly stales it; the versioning
    archetypes; shared and frozen parameters; a non-contiguous NCHW
    input; strict stable/native separation; honest per-boundary failure
    atomicity (transactions are per module — one whole training step is
    *not* globally transactional); error-state recovery; the NumPy
    boundary; live-storage baselines; and reality-derived capability
    guardrails — tests and documentation only, adding no capability.
    **F9 is complete**: the phase closure — fresh Windows Release **and**
    Debug builds (Visual Studio 17 2022, MSVC 19.44.35228.0) each passing
    the full existing 10-test CTest suite with zero project warnings, and
    the active runtime proved to remain the Release DLL; a fresh Clang
    18.1.3 `address,undefined` build in WSL2 Ubuntu 24.04 whose
    instrumentation is *proved* (22 `__asan*` and 13 `__ubsan*` dynamic
    symbols; the library will not even load without the sanitizer
    runtime); 10/10 sanitized native CTests with leak detection enabled;
    1,968 sanitized normalization-focused Python tests with **zero ASan
    and zero UBSan diagnostics**; the F6 example reproducing its exact
    resume and the F7 benchmark passing all nine correctness gates under
    the sanitized library; and a practical LeakSanitizer lifecycle whose
    native live-storage counter returned **exactly** to baseline, with the
    remaining process-exit allocations identified honestly as
    CPython/NumPy shutdown retention containing no TensorForge frame and
    no suppression file. It is **validation and documentation only** — no
    numerical capability, no C++, no CTest, no ABI or ctypes surface, no
    example, no benchmark, and no production behavior changed — so
    **Phase F is complete (F0–F9)** and no normalization
    operation or kernel exists at all.
    Deliberately outside Phase F: dropout, a native RNG with its
    checkpoint state, further activations, more losses, schedulers, data
    loaders, native integer tensors, further dtypes or devices, CUDA,
    AMP, fused normalization kernels, and CPU optimization.
  - **Phase G — Native RNG and Dropout — is complete; G0 through G10
    have all landed.** The **G0 architecture contract is written** —
    [native_rng_dropout_design.md](native_rng_dropout_design.md) locks
    the phase's central split (random state is Python-managed; native
    random kernels are stateless and receive the complete key for one
    call), the `NativeGenerator` contract (an explicit unsigned 64-bit
    seed and call counter plus an algorithm identifier and version, no
    native resource and therefore no `close()`, identity equality, and no
    global or process-wide state anywhere), the deterministic
    counter-based algorithm and its known-answer requirements, the
    call-consumption transaction (**one successful stochastic forward
    consumes exactly one call**; a validation, allocation, kernel, or
    graph-construction failure consumes none, and neither does evaluation
    mode, `p == 0`, or backward) and the lock-protected, token-validated
    reservation protocol that carries it (one private lock covering
    reservation, commit, cancellation, and every state read and write,
    with native computation outside it; opaque single-use tokens so a
    stale, foreign, or duplicated commit changes nothing; at most one
    live reservation, so a concurrent or reentrant caller fails **before
    an index is minted** and no two callers can ever receive the same
    call index; and seed or counter replacement refused while a
    reservation is live — serialization for correctness, with parallel
    stochastic execution explicitly not claimed), the probability contract
    (`0 <= p < 1`, with `p == 1` rejected so inverted scaling never
    divides by zero), the stateless forward boundary (one new kernel
    producing the output **and** a private multiplier mask, no backward
    kernel, logical-order element indexing independent of physical
    strides), the differentiable operation whose backward reads **only**
    that graph-owned mask — never the input, never the generator — the
    module surface, generator registration as a fourth `NativeModule`
    state category, native checkpoint **version 2** whose generator
    section records the **alias topology** — every registered generator
    path and its canonical target, so shared-versus-independent generator
    identity is restored and not merely the states, with every topology
    mismatch failing in prevalidation before any live state changes — and
    an explicit version-1 compatibility rule that never fabricates a seed
    or counter, **whole-checkpoint transaction atomicity** (validate
    everything, stage everything that can allocate or raise, then commit
    under one rollback guard, so any ordinary synchronous failure — and
    any deliverable asynchronous one, including `KeyboardInterrupt` —
    restores parameters, persistent buffers, optimizer state, and
    generator state together with every object identity intact, leaving
    external process or interpreter death as the only documented
    exception), the ownership and failure matrices, and the **G0–G10**
    milestone sequence. **G0 added no numerical behavior**: it is design,
    documentation, and semantic guardrails only.
    **G1 is complete** — the generator state foundation:
    `NativeGenerator` (pure Python, no native storage, no `close()`; the
    four locked fields as read-only properties; atomic `state()` /
    `load_state()` / `reseed()` / `reset()`; exact-`int` seeds with one
    `secrets` entropy draw for `seed=None`; identity semantics with
    copying and pickling refused; and the lock-protected,
    token-validated reservation transaction — a two-phase claim /
    construct / publish / deliver sequence whose token is allocated with
    **no generator lock held**, so no callback-capable operation ever runs
    while a lock is held and a finalizer cannot invert the global
    multi-generator lock order, with the construction claim refusing every
    conflicting mutation in the meantime and an exact-match cleanup for a
    reservation that was published but never delivered, so a dropped token
    can never strand the generator) plus generators as a
    **fourth** `NativeModule` registration category with deterministic
    identity-deduplicated cycle-safe traversal and their own
    `generator_state_dict()` / `load_generator_state_dict()` surface,
    leaving `state_dict()` tensor-only and unchanged. **G1 generates no
    random values by itself.**
    **G2 is complete** — the deterministic stateless
    native Dropout-forward **Core**: the exact locked `tensorforge.splitmix64` derivation in
    unsigned 64-bit arithmetic (`mix64` finalizer, per-call stream key
    `mix64(seed + GOLDEN*(call_index + 1))`, per-element bits
    `mix64(stream + GOLDEN*(element + 1))`, uniform
    `(bits >> 11) * 2**-53`, dropped when `u < p`) as internal hidden
    `namespace tf` functions in the new `cpp/src/random.cpp` /
    `cpp/include/tf_random_internal.h`; the inverted-dropout float64 CPU
    kernel writing the output **and** the private multiplier mask in one
    pass; the self-validating guarded export `tf_core_dropout_forward`;
    its ctypes declaration carrying the whole key as two `c_uint64`
    arguments; `"dropout_forward"` in `TENSOR_CORE_OPS` and
    `"tf_core_dropout_forward"` in the checked-kernel inventory; the
    public `NativeTensorCore.dropout_forward(p, *, seed, call_index)` and
    the private `_dropout_forward_with_mask` that keeps the mask; a
    dependency-free CTest over both layers; and committed known-answer
    vectors asserted **identically** from C++ and Python. It is
    **stateless**: the complete random key arrives as two explicit
    integers, the Core reserves, commits, cancels, inspects, and mutates
    **no** `NativeGenerator`, and no C++ translation unit holds random
    state of any kind. Randomness is keyed by the **logical** row-major
    element index, so a transposed, narrowed, or nonzero-offset view
    receives the same mask as a contiguous tensor of the same logical
    shape. Both results are fresh owning contiguous cores that alias
    neither the input nor each other, and the two-result boundary is
    failure-atomic in C++ *and* in the Python wrapper.
    **G3 is complete** — the differentiable
    `NativeTensor.dropout(p, *, generator)` over that Core, and the whole
    milestone is one method plus one name, `"dropout"`, appended to
    `AUTOGRAD_OPS`: no C++, no C ABI symbol, no ctypes declaration, no
    `NativeTensorCore` method, no module, no export, and no
    checkpoint-format change. The `generator` is **required and
    keyword-only** — no default, process-global, or module-global stream,
    no implicit per-call generator, and no NumPy or Python `random`
    fallback — and `p` goes through the *same* shared validator the Core
    uses, so the accepted/rejected matrix is identical by construction.
    It owns the §5 call transaction: validate, then reserve one call,
    then run the G2 Core **outside** the generator's lock with the
    reservation's own seed and index (never a reread counter), then build
    the graph, then commit as the **last** state-changing action. So a
    successful stochastic forward consumes exactly one call — with or
    without gradients — while `p == 0` returns the caller's own tensor
    object having reserved, allocated, and consumed nothing, and every
    ordinary failure before the commit releases the result, abandons the
    reservation, and leaves the same unconsumed index for the next
    forward. Backward consumes none, ever. The private multiplier mask
    becomes **graph-owned** state through the unchanged `graph_resources`
    contract — the third member of the family beside the native pooling
    winner buffer and the fused loss's saved probabilities — released
    exactly once at the
    same deterministic points the graph history is, retained under
    `retain_graph=True`, kept alive across a failed retryable backward,
    freed by an abandoned graph's `close()`, and closed immediately by a
    no-grad forward. The backward is `upstream * mask` through the
    existing native `multiply`, so **no dropout backward kernel exists**;
    it never rereads the input, never redraws, and never touches a
    generator, and the node therefore records **no** expected parameter
    version, so mutating the input or reseeding, resetting, or reloading
    the generator afterwards cannot change an existing graph's gradient
    and must not raise.
    **G4 is complete** — `NativeDropout(p=0.5, seed=None,
    generator=None)`, the public module over that operation, plus its
    experimental export and one name (`"NativeDropout"`) appended to
    `NATIVE_MODULES`. Nothing else moved: no C++, no C ABI symbol, no
    ctypes declaration, no Core method, no autograd operation, and no
    checkpoint-format change. `p` goes through the *same* shared
    validator the Core and the operation use; `seed` and `generator` are
    **mutually exclusive**, so supplying both raises rather than quietly
    ignoring one; and the module either creates and owns a generator or
    registers the **exact** object supplied, never a copy — which is how
    two layers share one interleaved stream while the default gives every
    layer an independent one. The generator is first-class registered
    state (in `generators()`, `named_generators()`, and
    `generator_state_dict()`, and deliberately absent from
    `state_dict()`, which stays tensor-valued), a state load replaces it
    in place so identity and sharing survive, and the module owns **no**
    native storage. Training delegates to the differentiable operation,
    so a successful forward consumes exactly one call and a failed one
    none; evaluation returns the **input object itself**, consuming and
    allocating nothing, so an arbitrary number of eval forwards leaves no
    gap in the stream and the next training forward takes the next index;
    and `p == 0` is identity in both modes.
    **G5 is complete** — native checkpoint **format version 2** and exact
    generator restoration. The format *name* never moves; the version is
    now 2 and every new save writes 2, whether or not the model has
    generators. The manifest gained exactly one field, `"generators"`:
    `null` when the model registers none, so absence is stated rather
    than inferred, or `keys`/`entries`/`aliases` — the ordered canonical
    names, one `{algorithm, algorithm_version, seed, calls}` object each
    (seed and counter as **canonical decimal strings**, because a
    `uint64` above `2**53` is not representable in the IEEE double most
    JSON readers use), and the complete **registered path → canonical
    name** map. Generator state adds **no array** to the archive. A
    shared generator's state is written once while its *topology* is
    written in full, so two paths draw from one stream in the archive
    exactly when their aliases name the same canonical entry — sharing is
    **identity**, never state equality. Canonical names and both orders
    are functions of the model alone, so saving the same model twice is
    byte-identical. A load compares the archive against a real
    `named_generators()` traversal, strictly in both directions, and
    every mismatch — a missing or unexpected canonical key or registered
    path, an alias targeting an absent entry, a canonical name not
    self-mapped, a repeated JSON object key, saved-shared versus
    live-independent (or the reverse), a canonical name changed by a
    reordered registration, an algorithm or version mismatch, a malformed
    or out-of-range seed or counter — fails **in prevalidation, with
    nothing touched**. Generators are restored **in place**, so identity
    and every sharing relationship survive and the archive never
    constructs one. A save *or* a load is refused, leaving an existing
    destination byte-intact, while any target generator has a call
    reservation in flight. A **version-1** archive still loads into a
    model with no registered generators and is **rejected**, naming them,
    for one that has them — no seed and no counter is ever fabricated —
    while a v2 archive with generator state loaded into a generator-free
    model fails as an unexpected-generator error. And the load is **one
    transaction over the whole archive**: model, buffers, optimizer, and
    generators commit through their own loaders inside a single rollback
    guard, so any synchronous failure — a deliverable `KeyboardInterrupt`
    included — restores all four together, preserves every object
    identity, moves no parameter version, leaves graph-owned multiplier
    masks from earlier graphs untouched, and returns native live storage
    to baseline; only external process or interpreter death is outside
    that guarantee. It is **serializable** as well: every participating
    state replacement — the checkpoint load commit, `load_state_dict`,
    `load_generator_state_dict`, and both optimizers' state loads — plus
    the save snapshot runs under **one** private process-wide `RLock`, in
    the universal state-replacement lock order (that guard first, then
    every unique target generator lock in the global `id()` order, never
    the reverse), so two concurrent loads leave one archive's state
    followed by the other's rather than a mixture, and a save describes
    one coherent serial point. Generator reservations deliberately stay
    outside the guard, so a racing reservation precedes or follows a
    transaction and no state is replaced underneath a live token.
    Ordinary training mutation does not take the guard, so thread-safe
    concurrent training snapshots are not claimed. The whole registry
    footprint is one reporting-only
    name, `"checkpoint_generator_state"` in `STATE_SUPPORT`.
    **G6 is complete** — the hardening milestone. It executed §13 and §14
    of the design as adversarial tests in a new
    `tests/test_native_phase_g_hardening.py`: the reservation transition
    matrix, the exact `uint64` boundary, forced concurrent interleavings
    with bounded joins and no sleeps, the deterministic Core's structural
    key properties beside its committed vectors, every pre-commit and
    post-commit failure position of the call transaction over four
    exception classes, all four graph-owned saved-resource families in one
    graph, a 76-case checkpoint corruption matrix, whole-transaction
    rollback at every commit position, save-seam destination atomicity,
    and repeated success-and-failure lifecycle loops measured against a
    real native live-storage baseline. **It added no capability,
    operation, module, export, checkpoint field, or checkpoint version**
    and moved no registry value; it found and fixed exactly one runtime
    defect — a cleanup-failure `__context__` chain that could become
    cyclic — with a dedicated regression guard.
    Milestone **G7 is complete** — the end-to-end exact stochastic resume,
    and **no new capability**. `examples/native_dropout_training.py` trains
    `NativeLinear(4, 8)` -> `NativeBatchNorm1d(8)` -> `NativeReLU` ->
    `NativeDropout(p=0.5, seed=20240707)` -> `NativeLayerNorm(8)` ->
    `NativeLinear(8, 3)` over raw logits with `NativeCrossEntropyLoss` and
    `NativeAdam` on a fixed twelve-sample three-class task computed from an
    explicit formula, in three fixed batches on a schedule that is a **pure
    function of the training step**. It carries all four TensorForge-owned
    state families at once — parameters, the persistent native BatchNorm
    running buffers, a registered `NativeGenerator`, and NativeAdam moments with
    per-parameter step counters — so an incomplete restore diverges
    immediately. Two uninterrupted runs are bit-identical; an interrupted run
    checkpointed after 7 **completed** steps (deliberately mid-cycle in the
    batch schedule), whose model, optimizer, and generator are **released
    before the resume begins**, reloads into a completely fresh set built
    with a *different* native Dropout seed and reproduces the uninterrupted run by
    **exact equality**: the whole loss sequence, every parameter, both
    running statistics, every optimizer moment and step counter, the
    generator's algorithm/version/seed/calls, the final training logits, and
    the final evaluation output. Two negative controls make that meaningful —
    restoring all four families but restarting the batch schedule at 0
    **diverges**, and restoring everything but re-seeding the generator
    **diverges**. Evaluation is proved state-neutral (repeated eval passes
    leave `calls` bit-identical, produce identical outputs, restore the
    caller's mode, and leave a probed run's loss sequence equal to an
    unprobed one's), and a separate throwaway reload matches the restored
    module's next native Dropout output against `NativeTensorCore.dropout_forward`
    at the exact restored `(seed, call_index)`, advancing `calls` by exactly
    one. **External loop progress is carried explicitly**, as validated JSON
    metadata (`{"training_step": ..., "next_batch_index": ...}`), because
    checkpoint v2 captures TensorForge-owned state and **not** data-loader
    position, batch order, shuffle state, epoch counters, scheduler state,
    Python's `random`, or NumPy's global random state — a missing or inconsistent
    field raises rather than silently restarting from step 0.
    Reproducibility is exact **for the state actually captured**;
    full-program determinism is not claimed. The whole milestone is one
    example, one test module, and documentation: **no** C++, C ABI symbol,
    ctypes declaration, Core method, autograd operation, module, export,
    schema field, checkpoint version, benchmark, or registry value changed.
    Milestone **G8 is complete** — `benchmarks/benchmark_native_dropout.py`,
    the honest characterization, adding **no capability**. Thirty-five
    cases in eight families: the stateless Core against an **exact
    bit-for-bit** vectorized NumPy implementation of the same locked
    derivation, scalar-to-large size scaling, four physical layouts over
    one logical shape, a five-value probability sweep at three layers,
    the no-grad / differentiable / backward-only / forward-plus-backward
    operation layers, the module's training and identity paths, and one
    complete native Dropout training step — each gated for correctness **before**
    timing against the committed known-answer vectors, each recording its
    exact generator consumption, and all of them followed by an untimed
    lifecycle pass that returns native live storage to baseline. The
    operation and module cases are `native_only` and publish no ratio.
    **No speed assertion, no committed timing number, no CI timing
    threshold**, and no result file unless `--json-out` names one; the
    numbers are a machine-specific snapshot and nothing was optimized to
    improve one. Milestone **G9 is complete** — the cross-cutting Phase-G
    integration suite, `tests/test_native_phase_g.py`, adding **no
    capability** and changing no runtime file. One test-only model
    carries every registered state family at once (convolution, NCHW
    normalization buffers, pooling, two native Dropout layers over one
    shared generator, flatten, linear layers, 1-D normalization,
    native LayerNorm, and the fused loss over raw logits), and the suite
    proves
    the interactions: four saved-resource families in one graph released
    exactly once, exact version-2 resume into a fresh
    model/optimizer/generator set with a negative control that diverges,
    the generator-topology matrix with every mismatch rejected before any
    state changes, evaluation consuming no call anywhere, p == 0,
    non-contiguous NCHW and strided views, whole-state rollback at every
    commit position, four deterministic concurrency cases, a Phase A–F
    regression matrix, and live storage returning exactly to baseline
    across success and failure cycles. Milestone **G10 is complete** —
    the phase closure. The validation matrix ran with observed results:
    fresh Windows Release and Debug builds, each **11/11 CTests** with
    zero project warnings and the active runtime proved to stay the
    Release DLL; a fresh Clang 18.1.3 ASan+UBSan build in WSL2 with
    instrumentation proved rather than assumed, **11/11 sanitized
    CTests** with leak detection on, **3,166 sanitized Python tests**,
    the G7 example reproducing its exact resume, and the G8 benchmark
    smoke path passing every correctness gate — all with zero ASan and
    zero UBSan diagnostics; and a LeakSanitizer lifecycle returning
    native live storage exactly to baseline with no TensorForge frame in
    the remaining process-exit allocations and **no suppression file
    added**. Reproducibility stays exact only for
    the state actually captured (no Python `random`, no NumPy global
    random state, no data-loader position, and no scheduler state), and
    ordinary concurrent training is not claimed thread-safe.
    **`dropout` stayed listed unsupported for the whole of G0–G9** — G4
    implemented and exported `NativeDropout` and G5 persisted its stream,
    neither moving the boundary, because a capability
    whose value is exact reproducibility is not finished until
    reproducibility has been demonstrated under fresh Release and Debug
    builds and the sanitizers — and the name was removed at **G10**,
    after that matrix passed, leaving exactly `float32`, `cuda`, and
    `amp`. The claim it makes is narrow: native Dropout is supported in
    the **experimental native float64 CPU** backend, which says nothing
    about the stable framework, float32, CUDA, or AMP.
    Deliberately outside Phase G: a generic
    sampling or distribution API, global random state, NumPy
    global-random-state integration, parameter-initialization changes,
    data-loader shuffling,
    augmentation, 2-D and 3-D dropout variants, stochastic depth, attention
    dropout, integer tensors, embeddings, float32, CUDA, AMP,
    schedulers, new optimizers, CPU performance tuning, and any stable
    framework change.
  - **Phase H — Native CPU Performance and Runtime Efficiency — is
    complete.**
    Milestones H0 through H10 have all landed. (This entry read "is the latest *completed* phase" twice, which was accurate until Phase I closed at I11 and stale afterwards; it is repaired here rather than rewritten away. The latest completed phase is Phase J.) H10 re-measured the whole phase against a reconstructed and verified H0 baseline (52 cases, **zero checksum mismatches** — every figure compares implementations that produced bit-identical results), resolved the acceleration gate as three documented rejections with measurements (SIMD, threading/OpenMP, BLAS), assessed `tf_core_narrow_backward` and the small-operation boundary floor and implemented neither, ran the full Release/Debug/Linux/sanitizer/lifecycle matrix, and closed the phase. **Every shipped training workload is 1.50×–3.89× faster than at H0**, matmul 4.71×, the convolution kernels 2.59×–4.64×, reductions 3.78×–5.06×, with no allocation count or memory peak raised anywhere — and across the whole phase **no capability, dtype, device, registry value, public API, checkpoint field, or checkpoint version moved**, with exactly **one** C ABI symbol added (`tf_storage_create_uninitialized`, at H1): 51 → **52**. Its
    architecture contract is
    [native_cpu_performance_design.md](native_cpu_performance_design.md).
    **H0 is architecture, profiling, and baseline work: nothing was made
    faster.** It shipped that contract, the unified measurement harness
    `benchmarks/benchmark_native_cpu_performance.py`, that harness's
    behavioral contract tests, and documentation reconciliation — and no
    C++, C ABI symbol, ctypes declaration, `NativeTensorCore` method,
    autograd operation, module, loss, metric, optimizer, export,
    capability registry value, dtype, device, or checkpoint change. The
    native checkpoint format stays version 2 with versions 1 and 2
    supported, `UNSUPPORTED` still reads `("float32", "cuda", "amp")`,
    and **Phase G remains the latest completed phase**. The harness
    measures 26 cases (24 at H0, plus the two H3 added to
    decompose the per-call cost) across twelve workload families — dispatch
    overhead, elementwise, reductions, matmul, materialization, linear,
    convolution, normalization, stochastic, optimizer, training step, and
    in-memory state operations — separating up to nine implementation
    layers (NumPy, the stable line, the raw-buffer kernels,
    `NativeTensorCore`, `NativeTensor` with and without a graph,
    backward, an optimizer step, and a whole training step), with a
    correctness gate that runs **before** timing everywhere, honest
    reference labelling that publishes **no ratio** where no equivalent
    exists, `--smoke` / `--json` / `--case` / `--workload` and a focused
    `--profile CASE` mode, and no result file of any kind. Checkpoint
    file I/O is deliberately excluded from every training-step total, and
    the in-memory state surface is its own category. The evidence it
    produced is deliberately ranked and honest — the largest measured
    factors are an allocator behavior and a memory access pattern rather
    than raw arithmetic, the Python-side per-call metadata path costs
    several times the ctypes boundary it wraps, and the `NativeTensor`
    wrapper and its autograd graph node are measurably **not** a
    bottleneck. **Milestone H1 — the output-allocation contract — has
    since shipped**, the first Phase-H change to production code: **Milestone H1 — the output-allocation contract — has now shipped.** It removed the redundant zero-fill from output storage that a kernel provably overwrites in full, behind one new C ABI symbol (`tf_storage_create_uninitialized`) that matches the zero-initializing default in size validation, allocation-failure handling, error state, ownership, destruction, and live-storage accounting, and differs only in the buffer's initial contents. The zero-initializing path remains the default; there is **no** global allocator policy, environment variable, heuristic, memory pool, scratch arena, or public empty-tensor API, and every enabled call site opts in explicitly against a per-kernel audit table. `sum`/`mean` and `narrow_backward` are explicitly **rejected** and keep a zeroed destination: the first accumulates into its output, the second writes only the narrowed region and the untouched zeros *are* the gradient. Completeness is proved by deterministic **poison** tests that are injected **exclusively by test infrastructure, around the allocator**: the suite wraps the private uninitialized allocation helper, lets the real constructor allocate, fills the returned storage with a quiet NaN or a large finite pattern through the ordinary fill primitive, and hands that same storage to the real operation — so the pattern is in place after the real allocation and before the real kernel runs. **No poison-control mechanism exists in the production runtime**: no exported hook, no thread-local flag, no environment variable, no global mode. ASan and UBSan stay separate from the initialization proof — they do not detect uninitialized-value reads — and MemorySanitizer is not available here, so neither is claimed; negative controls prove the detector can actually fail. H1 is bit-identical: every enabled operation and a full training run are compared element-wise against the zero-initializing allocator. No capability, dtype, device, registry value, checkpoint field, or checkpoint version changed, and `tf_storage_create_uninitialized` is the **only** export it added, taking the library from the pre-H1 baseline of 51 exported `tf_*` symbols to **52**.
    The measured result is reported honestly rather than as a headline: isolated, the zero-fill is enormous and scales with the buffer (about 52x at 2 MB, 119x at 8 MB, 552x at 32 MB, and *negative* below roughly 16,000 elements, where it sits inside the noise). End to end it is much smaller and often inconclusive — clearly real for large memory-bound elementwise work (about 1.5-1.8x on an 8 MB output), small and variable for the `normalized_training_step` and `adam_step` cases, and with no measurable effect on `conv2d_forward`, `mlp_training_step`, or `matmul_square_contiguous`, whose arithmetic dwarfs its allocation. Those inconclusive and negative rows are published as such.
    **Milestone H2 — native matmul memory access — has since shipped**,
    and is the first Phase-H milestone to change how a numerical kernel
    executes. It swapped the production matmul's loop order from
    `i`-`j`-`k` to `i`-`k`-`j` over four destination rows at a time, so
    the innermost loop walks a *row* of the right operand and a row of
    the output sequentially instead of walking a column. **Cache blocking,
    which the milestone title anticipated, was measured against 22 blocked
    variants and rejected** — an unblocked full-width row sweep was faster
    at every non-trivial size — so H2 shipped the simpler superior design
    and recorded the negative blocking result. The pre-H2 triple loop is
    **retained verbatim as the shipped generic reference path** and still
    runs, chosen from stride metadata inside the kernel: a right operand
    whose column stride is 1, with a non-empty inner dimension and at
    least 8 result columns, takes the row sweep; a transposed right
    operand, a narrow result, or an empty inner dimension takes the
    generic path — which is the loop order that case already suits, so
    the fallback is a design choice rather than a gap. Dispatch is
    metadata-driven, deterministic, total, side-effect free, and
    independent of pointer values, alignment, timing, environment
    variables, and CPU-feature probes; a failed precondition is never an
    error. **H2 added no exported C ABI symbol** — the count is still 52 —
    and no kernel selector, block-size setter, dispatch tracer, or public
    dispatch control of any kind exists. The numerical agreement between
    the two paths is a **four-part contract**, not a blanket bit-identity
    claim: (1) accumulation order is preserved exactly; (2) **every
    non-NaN result is bit-identical**, asserted as raw IEEE-754 bit
    patterns rather than tolerances across shapes, layouts, signed zeros,
    infinities, denormals, largest finite magnitudes, gradients, Linear,
    both optimizers, deterministic training, and exact checkpoint resume;
    (3) NaNs occur in exactly the same positions on both paths and are
    always quiet; and (4) **NaN payload bits are deliberately outside
    TensorForge's numerical contract** and may differ. Ten source-level
    formulations were measured while trying to close (4), and the only
    structure that reproduces the reference's payloads is the `i`-`j`-`k`
    order H2 replaces, so parity is unavailable short of abandoning the
    optimization; MSVC Release differs on 162 of 208 results in a
    NaN-saturated matrix, MSVC Debug and Clang on none.
    H1's uninitialized-output contract still holds on both paths, for a
    different reason on each: the generic path never reads the
    destination, and the row sweep's `k == 0` pass assigns every element
    before anything accumulates into it — proved by poison tests over both
    paths with both patterns, plus a negative control. The measured result
    is reported honestly: roughly 4.1-4.7x at 384 cubed, 4.2-4.5x at 128
    cubed, about 4-6.8x on `NativeLinear` forward, 1.7-2.5x on its
    backward (only one of its two matmuls qualifies, by design), 2.0-2.4x
    on a 128x256 MLP training step, and **no measurable effect below
    roughly 32 cubed or on a small MLP step**, where a fixed ~10 microsecond
    per-call Python cost dominates and control cases whose code did not
    change at all vary by 0.50-1.44x. No capability, dtype, device,
    registry value, checkpoint field, or checkpoint version moved.
    **Milestone H3 — native metadata and dispatch efficiency — has since
    shipped**, and unlike H1 and H2 it is **Python-only**: no C++, no C
    ABI symbol, no ctypes declaration, and no kernel changed, so the
    library still exports exactly **52** `tf_*` symbols. H3 attacked the
    fixed per-operation cost B3 measured at 18.6-22.6 microseconds, of
    which only about 1.9 was the ctypes boundary and the rest was
    Python-side shape and stride work. The measured cause was redundant
    *re-validation*: one `shape_info` call ran `_as_int_tuple` **four**
    times over a tuple that was fully validated after the first pass,
    and computed the row-major strides **twice**, while
    `NativeTensorCore.zeros` validated the caller's shape a second
    complete time by calling `numel(shape)` and then constructing a view
    from the same raw shape. Instrumented call counts put that at
    **815** `_as_int_tuple` calls per MLP training step and 604 per
    `NativeAdam` step. H3 introduced **one normalization boundary** —
    the private `_normalized_layout`, performing exactly the checks
    `shape_info` always performed, in the same order and with the same
    messages, and normalizing the shape once — with the derived
    quantities computed by private `_checked` primitives that validate
    nothing *because there is nothing left to validate*. Each public
    helper (`row_major_strides`, `numel`, `reduce_shape`,
    `broadcast_shapes`) is now its own validation followed by the
    matching primitive, so the two can never disagree.
    `NativeTensorView` gained a private `_from_validated` constructor
    that skips **only** that normalization; both constructors funnel
    through one shared `_bind` that still performs the storage open
    check and the full reachable-offset bounds check, and the element
    count and contiguity flag are **derived inside** the private
    constructor rather than passed to it, so no caller can supply an
    inconsistent pair — which is why H3 has a separate private
    constructor rather than the misusable `validated=True` flag. Views
    also memoize their `int64` shape/stride arrays for the strided C
    ABI, **lazily** and **read-only**. That memoization cannot go stale:
    a view's layout is assigned exactly once, in `_bind`, and every
    layout-changing operation (`reshape`, `transpose`, `T`, `narrow`)
    returns a *new* view, so no invalidation is ever required and none
    exists. Nothing global was introduced — no shape cache, no stride
    interning, no weak-reference machinery, no thread-local state — and
    **no validation was removed**: every rejection still happens, with
    the same exception type, the same message, and the same shape-then-
    strides- then-offset ordering. Measured: `shape_info` 2.6-4.5x
    faster, view construction 3.2x, `_as_int_tuple` calls per MLP step
    **815 -> 149** and per CNN step **815 -> 150**; end to end, a one-
    element allocation 2.1x, a `reshape` 3.1x, a view chain 2.4x, a
    small `add` 1.56x, `NativeAdam` on a small MLP 1.42x, a **whole MLP
    training step 1.43x**, a **CNN training step 1.29x**, and a
    **normalized training step 1.51x**, which cut the `NativeAdam` step's gap
    against the stable line from 39.8x to 31.9x. Reported just as
    honestly: **large kernel-bound work shows no measurable change in
    either direction** — 384-cubed, 512-cubed and 128-cubed matmul, 256-
    squared elementwise, and 128-squared reduction all sit inside their
    own run-to-run spread, so H2's large-matmul result is intact. The
    layout- array cache is the weakest of the three changes and was kept
    on measured merit, not principle: isolated, it saves 0.6-1.5
    microseconds per *strided* small operation and nothing at all on
    large ones or on a contiguous training step, and even a deliberately
    cold-cache measurement is no slower than pre-H3. One methodology
    finding is published rather than buried: at the harness's default 11
    repetitions a case appeared to regress 35%, and at 201 repetitions
    the same case measured 1.19x *faster* — so no default-repetition
    figure is quoted as H3 evidence. Object footprint is unchanged for a
    cold view (byte-identical) and +328 bytes for one that actually
    takes a strided path; in a full MLP step only **5 of 134** views
    ever populate it, 1,560 bytes in total. All instrumentation was
    test-local or benchmark-local monkeypatching and subprocess A/B runs
    against a retained pre-H3 copy of the package — **no production
    counter, environment-variable profiler, or installed tracing mode
    exists**, and H3 added no public API of any kind: no cache control,
    statistic, reset, profiling counter, or dispatch selector. No
    capability, dtype, device, registry value, checkpoint field, or
    checkpoint version moved.
    **Milestone H4 — native optimizer step efficiency — has since
    shipped**, also **Python-only** (no C++, no C ABI symbol, no ctypes
    declaration, no kernel; still **52** exported `tf_*` symbols), and it
    is the first Phase-H milestone whose subject is a *training-stack*
    component rather than the tensor runtime. B4's counts were
    re-instrumented on the current code and confirmed exactly: **27
    native allocations per parameter per `NativeAdam.step()`, ten of them
    one-element** — eight broadcast scalar coefficients (`beta1`,
    `1 - beta1`, `beta2`, `1 - beta2`, both bias-correction terms, `eps`,
    and `lr`; the design said six, and `eps` and `lr` were the two it
    missed) plus two `reciprocal` outputs taken on one-element tensors —
    with `NativeSGD` at five per parameter, and eight of the thirteen
    binary operations in the native adaptive update taking the
    broadcasting path rather than the
    contiguous fast path. Three changes shipped. **The scalar
    coefficients are built once per step, not once per parameter**: a
    private per-step `_StepConstants` holder, keyed by `(dtype, device)`
    so it never assumes one dtype exists, with the two bias-correction
    terms cached per step *counter* so a parameter that skipped earlier
    steps still gets its own; it allocates nothing until the first entry
    asks for a coefficient (a step with no active parameter allocates
    nothing at all), is released before the commit begins, and is never
    stored on the optimizer, so no scalar survives a step, enters
    `state_dict()`, reaches a checkpoint, or has to be released by
    `close()`. `NativeSGD` does the same for its single `lr` scalar — the
    only change its evidence supported. **The bias-correction reciprocal
    is evaluated in Python**, removing one allocation and one kernel call
    per coefficient per parameter; this is an *exact substitution, not a
    reassociation*, because the kernel literally is
    `double op_reciprocal(double x) { return 1.0 / x; }`, a Python
    `float` and a C++ `double` are the same IEEE-754 binary64 value, and
    IEEE-754 division is correctly rounded — proved over **20,000+
    values** spanning the full exponent range, ±0, ±∞, the smallest
    subnormal, the largest finite magnitude, and every `1 - beta ** t`
    the optimizer actually forms, compared on **raw `uint64` bit
    patterns** with zero mismatches. **Temporaries are released at their
    last use** rather than all together at the end of the staged
    expression. Everything is **bit-identical** to the pre-H4
    composition, which is *retained in the test suite* and executed
    natively as the reference — 60 shape/step/hyperparameter
    combinations, a six-step run over four mixed shapes, and four SGD
    learning rates from `1e-9` to `1e12` — and a separate test pins the
    **exact operation sequence** a staged entry issues so a future
    reorder or fusion fails loudly. The two-phase contract is untouched:
    validation is still four complete passes in the same order with
    nothing moved behind a mutation, stage mutates no parameter, moment,
    counter, version, or gradient, the commit is still **one
    `copy_value_` and exactly one version increment per updated
    parameter**, gradients are read and never written by identity, value,
    and storage identity, and the per-entry commit boundary is *tested*
    by injecting a `copy_value_` failure rather than assumed infallible.
    Measured by alternating pre/post **subprocess** rounds so drift hits
    both arms equally, 366 samples per case: `NativeAdam.step()`
    **1.58x** at (128, 128), **1.54x** at (256, 256), **1.48x** on a
    four-parameter MLP whose largest weight is 256-squared, 1.21-1.22x on
    a small MLP, 1.15x on a first step, 1.09-1.12x on tiny parameters; a
    large MLP training step 1.23x, a small one 1.15x, a normalized step
    1.13x, a CNN step 1.09x; and in the shipped harness `adam_step`
    1.25x, cutting the gap against the stable line from **23.8x to
    19.7x**. Reported just as honestly: **a (512, 512)
    parameter is neutral** (1.02x — at that size the step is
    memory-bandwidth-bound and ten fewer one-element allocations are
    invisible), the **native Dropout training step is neutral**
    (0.99x), and
    **NativeSGD is neutral-to-slightly-positive** (1.03-1.07x), with one
    0.88x row identified as **noise** by a focused re-measurement whose
    post minima were lower in every pair. The noise floor is stated
    rather than assumed: the matmul control case, whose code H4 did not
    touch, varied **0.84x-1.26x** between arms, so no single reading
    inside that band is a result — and H2's large-matmul performance is
    intact. Memory moved with time, not against it: **peak live transient
    bytes during one `NativeAdam` step fell 2.6-3.0x** (1,966,160 to 655,424 for
    a (128, 128) parameter) and per-parameter allocations went 27 to
    **17**, so a four-parameter model allocates **76 instead of 108**.
    Six alternatives were measured and **rejected** with their reasons
    recorded: scalar materialization (faster below roughly 32K elements,
    *slower* above, and it would regress the harness's own profile
    configuration while adding a parameter-sized buffer per scalar
    operation); same-shape stride-0 views (identical kernel arguments by
    construction, but *four* NumPy layout arrays per call where the
    broadcast path builds three); adopting the staged core instead of
    `copy_value_`; giving `_native_copy` a `contiguous_copy`
    implementation (it would stop normalizing `-0.0` to `+0.0`, a real
    observable change in a helper shared far beyond the optimizer); a
    persistent per-optimizer scalar cache (the hidden scratch tensor the
    design forbids); and reassociating the update to fold scalars
    together (a floating-point order change that would break every
    exact-resume proof in the project). H4 added **no public API of any
    kind**, and no capability, dtype, device, registry value, checkpoint
    field, or checkpoint version moved.

    **Milestone H5 — native copy and mutation-transfer efficiency — is
    complete**, and is the first Phase-H milestone since H2 to change C++
    though **not the ABI** (still exactly **52** exported `tf_*` symbols).
    H5 replaced the native line's value-transfer primitive: `_native_copy`
    was `zeros(shape) + core` — two allocations, a zero-fill pass, and an
    elementwise-addition pass — and is now the E3.1 native identity gather,
    `contiguous_copy()`, at one uninitialized allocation and one pass. All
    **ten** call sites of that helper (`copy_value_` staging, both
    `state_dict()` snapshots, both `load_state_dict()` stagings, both
    normalization running-statistic commits, and the
    reshape/transpose/unbroadcast gradient materializations) are pure value
    transfers and were enabled; `_broadcast_back` was **rejected** because it
    is a genuine broadcast, not a copy. Over a fixed 18-pattern IEEE-754
    sweep, **exactly three** patterns moved: the addition normalized `-0.0`
    and quieted both signs of signaling NaN, and the gather preserves all
    three — and **no** NaN payload differed at all, so **H2's matmul
    payload carve-out does not generalize to copies**. The rule H5 states is
    the narrowest coherent one: **a value transfer reproduces its source's
    bits; an operation follows IEEE arithmetic.** One C++ change, inside the
    unchanged export: a metadata-driven second *traversal*
    (`tf::copy_prefers_contiguous`, hidden visibility, total, pure, no
    environment variable or CPU probe) that sweeps a row-major source with
    the flat loop and falls back to the retained odometer otherwise —
    bit-identical **by construction**, since the identity map performs no
    arithmetic, and proved by a new dependency-free CTest (13 to 14). Nothing
    became in-place; every call site still stages, so self-copy, own-storage
    views, own transposes, and sibling views all stay correct, and identity,
    storage, version, gradient, state-transaction, checkpoint, and
    exact-resume behavior are all unchanged. Measured by alternating
    pre/post subprocess rounds (control band 0.96x-1.05x) and by a separate
    pre-H5-library A/B: the traversal alone **2.5x-5.5x** on contiguous
    sources and **0.94x-1.02x** on transposed ones (the unchanged odometer —
    the design's own control); `copy_value_` **2.14x** at (512, 512),
    optimizer `state_dict()` 2.40x, `load_state_dict()` 1.69x, `NativeSGD`
    1.15-1.31x. Reported as honestly: **`NativeAdam.step()`, every training
    step, the normalization running-statistic update, and copies
    below ~16 K elements are
    all neutral**, the latter because two `int64` layout arrays cost ~1.1 us
    each at the ctypes boundary — measured, attributed, and left to a later
    dispatch milestone. Allocations fell everywhere and **no peak rose**:
    `copy_value_` 2 to 1, module state 4 to 2, optimizer state 16 to 8,
    `NativeAdam` 17 to **16** per parameter. The harness gained two cases
    (26 to 28), the
    ladder was **reordered** (reduction execution moved to H6), and no
    public API, capability, dtype, device, registry value, or checkpoint
    version moved.

    **Milestone H6 — native reduction execution efficiency — has since
    shipped**, the third Phase-H milestone to change C++ and, like H2 and H5,
    **not the ABI**: the library still exports exactly **52** `tf_*` symbols.
    Reductions were the last core family in the runtime that always paid the
    generic strided indexing cost.

    The pre-H6 kernel was re-read and re-measured rather than taken from H0's
    or H5's summaries, and the cost was **decomposed** instead of assumed. At
    `(256, 256)` `axis=0` a `core.sum` costs 99.7 us of which the **raw native
    call is 94.8 us — 95 %**; subtracting the three `ndpointer` conversions
    leaves the C++ traversal itself at ~91.6 us, **92 %** of the operation. The
    entire Python wrapper — axis normalization (0.4 us), output-shape
    construction (0.6 us), write-stride construction (0.5 us), the write-stride
    array (0.4 us), the H3-cached layout arrays (0.1 us), and the output
    allocation (3.2 us) — is about 5 us. So this was the **opposite** of B3:
    H3's subject was a fixed Python cost that dominated *small* operations,
    while a reduction of any real size is dominated by the compiled loop, and
    H6's only worthwhile target was the traversal.

    H6 therefore reused the dispatch shape H2 and H5 each proved — one hidden
    metadata predicate, inside the existing export, no new symbol, the
    pre-milestone traversal retained. New `cpp/include/tf_reduction_internal.h`
    declares three hidden-visibility `namespace tf` functions and
    `cpp/src/reduction.cpp` implements them: `tf::sum_generic_strided`, the
    **pre-H6 odometer retained as the shipped generic reference path** — the
    only path that can address a transposed, narrowed, non-unit-strided, or
    broadcast source at all, and the oracle every optimized result is compared
    against; `tf::reduce_prefers_contiguous_blocks`, the predicate; and
    `tf::sum_contiguous_blocks`, a flat walk over an `outer x mid x inner`
    factorization. The predicate is total, pure, allocation-free, and a
    function of layout metadata alone — never of a pointer value, an alignment,
    a clock, an environment variable, or a CPU-feature probe — and a false
    answer is a fallback, never an error. It accepts a reduction when (1) the
    source strides are exactly the row-major strides implied by the shape (the
    same definition `NativeTensorView` uses, so the two layers agree by
    construction), (2) the reduced axes — those with a zero *write* stride,
    which is how the kernel has always identified them — form **one contiguous
    run**, and (3) the kept axes carry exactly the row-major strides of the
    output formed by dropping that run. **Stride collapsing is implicit and
    bounded rather than a general layout compiler**: conditions 1 and 3 *are*
    the statement that adjacent axes of the same class have identical address
    progressions, so each group collapses by multiplication, nothing is cached
    or interned, and non-adjacent reduced axes (unreachable from Python, which
    still takes one `int` or `None`) simply fall back. `keepdims` needs no
    special case and the kernel cannot even observe it.

    **Per-output accumulation order is preserved exactly, and the source
    traversal order is not even reordered**: the `o`, `m`, `i` loop nest is the
    lexicographic order of the source's own row-major index, which is precisely
    what the odometer walks, and every destination cell is touched by exactly
    one `(o, i)` pair, so the cells are independent. Nothing is reassociated,
    no partial sums are combined, no accumulator width changes, and no FMA,
    Kahan, pairwise, tree, parallel, or horizontal-vector reduction exists.
    The `inner == 1` branch (a full reduction, or one whose reduced run is a
    suffix) uses a local accumulator **seeded from `dst[o]`**, which is what
    keeps the export's documented accumulate-into semantics identical on both
    paths; the `inner > 1` branch adds a contiguous source row elementwise into
    a contiguous destination row, where distinct `i` are distinct outputs, so
    any vectorization is across independent cells and never a horizontal
    reduction.

    **The signed-zero contract is proved, not assumed.** Both paths start from
    the destination's `+0.0`, and `+0.0 + -0.0` is `+0.0`, so the sum of any
    number of `-0.0` values is `+0.0` on both paths and matches NumPy; seeded
    with `-0.0` both keep `-0.0`. All-positive zeros, all-negative zeros,
    alternating zeros, `-0.0` first, `-0.0` last, `-0.0` mixed with finite
    values, a column of `-0.0`, and exactly cancelling finite values are each
    compared as **raw IEEE-754 bit patterns** at every axis, both `keepdims`
    values, and scalar and multi-output shapes. One case is recorded rather
    than idealized: the **rank-0** export branch is a genuine
    `dst[0] += src[offset]` against a zeroed destination, so a rank-0 `-0.0`
    sums to `+0.0` — exactly as before H6, and now pinned by a test.

    **The NaN rule is H6's own, measured rather than inherited from H2.**
    Contractual: NaN positions are identical on both paths; every NaN either
    path produces is quiet, and a signaling-NaN input is quieted by both with
    identical bits; and with **at most one NaN per accumulation** — every case
    that occurs in practice — the two paths are bit-identical, payload
    included. Not contractual: when **two or more** NaNs are accumulated into
    one destination cell the paths may select different payload bits, asserted
    in neither direction. Why parity is unavailable at any price was measured,
    not asserted: four spellings of the optimized accumulation were compared —
    `acc += x`, `acc = x + acc`, a named-temporary `acc = acc + x`, and
    `dst[o] += x` accumulating *through memory* exactly as the odometer does —
    and **all four selected the same NaN and all four differed from the
    odometer**, so the local accumulator is not the cause and removing it would
    recover nothing. The divergence comes from the odometer's destination index
    being a runtime-varying value, which changes which addend MSVC places in
    the `ADDSD` destination register; that is an instruction-selection decision
    C++ cannot express. The memory-accumulate spelling was also **1.2x-1.8x
    slower** on suffix reductions, so it bought nothing. Recorded as an
    observation rather than a promise: the block path keeps the **first** NaN
    in accumulation order, the odometer the **last**, and the block path's
    choice is the one **NumPy** makes — so where they differ, H6 moved the
    answer *toward* NumPy. **H5's copy rule does not apply here either**, for
    the same reason it made H5's claim strong: a value transfer performs no
    arithmetic and so has no operand roles to choose between, while a reduction
    is arithmetic. Three operations, three genuinely different rules.

    **H1's decision stands, and H6 confirms it rather than revisiting it.** The
    destination stays zero-initialized on both paths, because both *read* it —
    that is what accumulation means. Outcome B was rejected on two grounds,
    one measured and one semantic: the fill is 2,048 bytes against 524,288
    bytes of reads at `(256, 256)` `axis=0` and **8 bytes** at `axis=None`,
    under half a percent of the traffic against a traversal that was 92-95 % of
    the operation; and making the fast path *assign* its first contribution
    would give the two paths different behavior for a non-zero destination,
    breaking the export's accumulate-into contract and stopping the generic
    path from being the reference. H6 therefore adds **no poison test**,
    because it introduces no uninitialized destination; the H1 poison suite is
    untouched and still passes, `sum` reaching `zeros` and never
    `_uninitialized` is asserted structurally, and the accumulate-into behavior
    that makes the zero load-bearing has its own negative control at the ABI.

    Measured by building a **pre-H6 library** from the identical sources with
    only `reduction.cpp` restored, driving both through identical `ctypes`
    calls on identical data with every output proved **bit-identical before
    either side was timed**, over 15 alternating pre/post rounds; the machine's
    control band for this measurement is **0.90x-1.03x**. Kernel level: full
    reductions 1.19x at 1,024 elements rising to **3.96x** at `(512, 512)`;
    2-D axis reductions 3.24x at `(128, 128)` to **6.37x** at
    `(1024, 1024) axis=0`; and — the finding that was **not** predicted —
    3-D and 4-D reductions **8.60x-10.94x**, because the odometer's carry loop
    runs up to `ndim` iterations per element so its cost grows with rank while
    the block traversal's does not. The NCHW rows matter because that is the
    layout the convolution stack produces. Layer level, over 9 alternating
    subprocess rounds: `TensorCore.sum(axis=0)` 4.49x and `mean(axis=0)` 4.11x
    at `(256, 256)`, NCHW `sum(axis=1)` **8.56x** and `sum(axis=3)` **8.82x**,
    `NativeTensor.sum` 3.88x without a graph and 3.82x with one, `sum()`
    forward+backward 1.27x, `mean` forward+backward 1.23x, the **convolution
    bias gradient's three chained sums 1.46x**, `_unbroadcast` 1.15x, softmax
    backward 1.14x, log-softmax backward 1.10x, `NativeLayerNorm` forward
    1.16x, `NativeBatchNorm2d` backward 1.10x, cross-entropy forward and
    backward 1.05x. Against NumPy in the shipped harness the contiguous
    reduction gap closed from roughly 8-13x to **1.67x** (4-D middle axis),
    **2.43x** (axis 0), 2.90x (last axis), and 3.75x (full to scalar), while
    the transposed-view control stayed at 10.33x.

    Reported just as honestly. **Every training step is neutral** — MLP small
    0.99x, MLP large 1.03x, normalized 1.03x, CNN 1.01x, NativeDropout control 1.02x,
    all inside the control band — so **H6 does not make training faster**, and
    no reading should be quoted as if it did; a reduction is a small share of a
    step whose cost is the optimizer and the large matmuls. **Normalization is
    mostly neutral** too: NativeBatchNorm1d training forward 1.04x, eval 0.98x,
    backward 1.02x, NativeBatchNorm2d training forward 1.06x, eval 1.00x, NativeLayerNorm
    backward 1.01x, with only NativeLayerNorm forward and NativeBatchNorm2d backward
    clearly outside the band — which **narrows H7 rather than motivating it**,
    since what is left in those modules is the sheer count of broadcast
    elementwise operations rather than the reductions. **Tiny reductions are
    neutral** (1 element 1.00x, 16 elements 1.01x, `(8, 8)` axis 0 1.03x),
    because below roughly 1,000 elements the fixed ~7 us Python-plus-ctypes
    cost dominates — H3's and H5's documented boundary finding, left to a
    dispatch milestone. And one **real, repeatable ~10 % regression** is
    published rather than buried: a **2-D transposed source reduced over
    `axis=0`** measured 0.89x-0.93x across four independent 25-round runs,
    while the 3-D transposed `axis=0` fallback measured **1.04x-1.05x faster**
    and every other fallback 0.96x-1.01x. Both libraries run the *identical*
    odometer there, and the cause was **isolated**: in a standalone binary the
    extracted-function spelling versus the inline spelling measured 0.88x-1.67x
    with no stable direction, so the extracted call is not it — the remaining
    attribution is whole-translation-unit code layout, which is exactly the
    machine-specific tuning the design rejects chasing. It affects no shipped
    path and no end-to-end case regressed. A specialized register-blocked path
    for a small trailing extent (`inner=2` measured 1.75x, `inner=4` 1.77x —
    the weakest wins) was **rejected on complexity**. Methodology is published
    too: at 7 alternating rounds the fallback controls read 0.85x and at 21-25
    rounds the same cases read 0.90x-1.02x, the same lesson H3 and H5 each
    recorded, so no low-round figure is quoted as H6 evidence.

    **Memory moved not at all, and that is asserted rather than assumed**: a
    `sum` allocates **exactly one** native storage — its own output — on both
    paths, at every axis, under both `keepdims` values, and `mean` allocates
    the same one because its scale is in place. There is no scratch buffer,
    workspace, arena, or pool, and the odometer's counter is unchanged and only
    on the fallback path. A 10-step training run over a model carrying
    parameters, native BatchNorm running buffers, and NativeAdam moments produced a **bit-identical**
    allocation and live-count profile before and after H6, which also confirms
    that profile's oscillation is CPython's collector rather than a leak either
    version introduced.

    The harness gained three cases, 28 to **31**, following H5's
    separate-rather-than-average precedent: `reduction_last_axis` (the suffix
    form NativeLayerNorm's mean and both softmax backwards actually reduce over),
    `reduction_full_to_scalar` (every write stride 0 and a rank-0 output — the
    hottest reduction in the runtime, since every mean-reduced loss ends in
    it), and `reduction_middle_axis_4d` (kept axes on both sides, so all three
    block extents exceed 1, plus the rank-4 reading the 2-D cases cannot give),
    with `reduction_transposed_view` now explicitly the pair's control because
    the predicate rejects it. One dependency-free CTest was added,
    `cpp/tests/test_sum_reduction.cpp`, taking the native suite from 14 to
    **15**; it drives the predicate table, both traversals in isolation, the
    accumulate-into contract over a pre-filled destination, and the
    special-value matrix at the layer where those properties are actually
    decided. **No exported C ABI symbol, no new translation unit, no public
    control of any kind** — no path selector, threshold setter, block-size
    setter, dispatch tracer, profiling counter, environment variable, or
    "which path ran" query — and no SIMD, threading, OpenMP, BLAS, parallel
    reduction, memory pool, scratch workspace, or fast-math. Multi-axis
    reduction was **not** added: the kernel can factorize a contiguous reduced
    run, but the Python layer still accepts one `int` or `None`, with every
    signature, default, axis rule, `keepdims` behavior, error type, and error
    message exactly what they were. `tf_core_narrow_backward`, the odometer's
    scatter dual, was deliberately left alone — widening H6 to it would have
    made this a scatter milestone. No public API, capability, dtype, device,
    registry value, checkpoint field, or checkpoint version moved.

    **Milestone H7 — native Python/C ABI boundary efficiency — has since
    shipped**, and it is **Python-only**: no C++, no exported symbol, no kernel,
    no traversal, no arithmetic. The library still exports exactly **52** `tf_*`
    symbols.

    **The ladder was revised here, and the revision is recorded rather than
    retrofitted.** H0's H7 slot was *composed-module cost* — the normalization
    modules and the composed convolution bias gradient — explicitly conditional
    on a re-measurement after H1, H3, and H6. That condition was tested and
    **not met**: H6 made `mean` 3.9x-4.1x faster and moved the normalization
    modules almost not at all (`NativeLayerNorm` forward 1.16x,
    `NativeBatchNorm2d` backward 1.10x, everything else inside the
    0.90x-1.03x control band, the normalized training step 1.03x). So the milestone was **dropped on evidence**, its
    proposal and the evidence against it preserved in the design document
    rather than deleted, and the slot was refilled from the *same*
    measurements: H3, H5, and H6 had each ended by deferring the identical
    named cost — H5 "~1.1 us per layout array at the ctypes boundary, left to a
    later dispatch milestone", H6 "the fixed ~7 us Python-plus-ctypes cost,
    left to a dispatch milestone". Three milestones deferred one thing to a
    later dispatch milestone; H7 is that milestone. Composed-module allocation
    count remains conditional future scope, and is H8's subject.

    The cost was **decomposed rather than assumed**, and the claim that six
    kernels were involved was checked and found wrong. All 52 exports are
    configured in one file — no other module in the repository imports
    `ctypes` — and **57 of their argument positions are arrays**, every one
    formerly bound as `numpy.ctypeslib.ndpointer`. That binding re-verifies
    array-ness, exact dtype, and contiguity at **every call**, then constructs
    `obj.ctypes` and resolves it through `_as_parameter_`: two Python object
    constructions and three checks, per array, per call, measured at **~2.1 us
    per array position**. On real calls: `tf_core_add` on a 4x4 with three
    layout arrays cost **7.6 us**, of which **6.1 us** was the binding; the
    array-free `tf_core_add_contiguous` cost 0.9 us and is the control.

    Then the *frequency* was counted: an MLP training step makes 245 native
    calls carrying **101** array crossings, a normalized step 692 calls and
    **315**, a CNN step 242 and **104** — about **20-23 %** of each step's wall
    time. And the provenance was the finding that decided the architecture:
    **~85 % of those crossings are operation-local broadcast strides**, not the
    H3 per-view cache, so a design that only cached pointers per view would
    have captured a seventh of the available work.

    H7 ships **two bindings for two categories, and deliberately not one
    blanket policy**. *Data* positions keep the checked `ndpointer` binding —
    the seven public raw-buffer kernels (whose callers may pass anything), the
    `copy_from`/`copy_to`/`materialize` host conversion boundary, and the
    cross-entropy **class labels**, which are int64 like the layout metadata
    but stay checked because a label array's required length comes from the
    *logits*, a different object. *Layout metadata* positions — 32 of them
    across 13 exports — take `ctypes.POINTER(ctypes.c_int64)`, fed by exactly
    two private producers: `NativeTensorView._native_layout_pointers()`, which
    memoizes `data_as` over the **unchanged** H3 read-only NumPy arrays that
    remain the owning buffers, and `_layout_vector(values)`, which builds a
    fresh `(c_int64 * len(values))` for metadata belonging to one operation.

    **Nothing was weakened, and one thing was strengthened.** ctypes still
    type-checks every call: a trusted position rejects a NumPy array of any
    dtype, a differently typed pointer or vector, a `c_void_p`, a list, an
    int, bytes, and a string — a NumPy array being rejected is a deliberate
    consequence that makes the old binding unreachable by accident. Dtype,
    byte order, and contiguity are established *by construction* rather than
    re-checked. **The length/rank invariant — the one `ndpointer` never
    checked, because the ABI sees only a pointer and an `ndim`** — is now
    checkable for the first time: a vector carries its length in its type, and
    a cached pointer carries its owning array (NumPy's `data_as` attaches it),
    whose length is the view's rank. The suite asserts that per producer, per
    rank 0-4, and structurally over **every strided call in a real workload**.
    The one honest difference is `None`, which `ndpointer` rejected and a typed
    pointer converts to NULL; it is closed structurally — both producers are
    total and no public API takes a metadata pointer. (Only **three** of the
    thirteen exports reject a null metadata pointer in C++ as well, so the
    producers being total is the load-bearing part of the argument, not a
    belt-and-braces remark.)

    **Ownership is NumPy's guarantee, relied on and tested rather than
    assumed.** `data_as` stores the array on the pointer, so a cached pointer
    cannot outlive its buffer; `POINTER.from_address` was measured **faster**
    (0.9 us against 1.6 us) and **rejected outright** because it produces a
    pointer with no owner. Deriving the pointer from the array rather than
    building a second vector keeps exactly **one owning description** of a
    view's layout — a cached ctypes vector was measured fastest of all and
    rejected, because it would duplicate that description and lose H3's
    `writeable = False` protection for ~2 % of a training step. Proved with the
    cyclic collector **disabled**: no reference cycle (an explicit
    `gc.collect()` after dropping a view collects 0 objects), no native storage
    kept alive, no pointer surviving into a usable state after close, and
    operation-local vectors retained by nothing. There is no global pointer
    cache, no id-keyed table, and no `from_address`, `byref`, `addressof`,
    `id`, or weak-reference container **anywhere in the module's code** —
    enforced by parsing it, so the docstring recording *why* `from_address` is
    unused is not mistaken for using it. Binding configuration stays a
    load-time act: `argtypes`/`restype`/`errcheck` are assigned only inside the
    two loader functions, asserted by locating each assignment's enclosing
    function, so nothing reconfigures a shared function object per call and
    **no thread-safety claim is broadened**.

    Measured against a **retained pre-H7 `cpp.py` driving the same Release
    DLL**, over 11 alternating pre/post subprocess rounds, every case proved
    **bit-identical before either side was timed**; control bands 0.95x-1.10x
    (core) and 0.99x-1.05x (end to end). Core: a 1-element `sum` **1.94x**, a
    16-element `sum` 1.89x, `to_numpy` 16x16 **1.83x**, `sum(axis=0)` 16x16
    1.79x, a 4x4 `contiguous_copy` 1.73x, `narrow_backward` 1.73x, strided
    `relu_backward` 1.71x, scalar-broadcast `add` **1.67x**, 4-D NCHW
    `sum(axis=1)` 1.54x, row-broadcast `add` 1.41x, `sum(axis=0)` 256x256
    1.35x, strided `exp` 1.29x, transposed materialization 1.16x. End to end —
    and this is the result — the **native Dropout step 1.32x, the
    normalized step 1.31x, `NativeAdam` at (32, 32) 1.31x, the CNN step
    1.30x, the MLP step 1.28x**, `NativeLayerNorm` forward 1.23x,
    `NativeBatchNorm1d` eval 1.23x, `NativeAdam` at (128, 128) 1.14x,
    `NativeSGD` 1.13x, the large MLP step 1.08x. **H7 is the
    first Phase-H milestone to move every training step** — H4 moved them
    1.09x-1.23x and H5 and H6 were neutral on all of them — because the cost
    is paid per *call* and a step makes hundreds of them.

    Reported just as honestly: **large kernel-bound work is neutral**, exactly
    as the attribution predicts. 256-cubed matmul **0.99x** and 8-cubed matmul
    1.00x are controls that take no array at all, so **H2's result is
    structurally untouched**; contiguous 16x16 `add` 1.05x is the third
    array-free control; and 512x512 `copy` 1.02x, 256x256 `to_numpy` 1.04x,
    512x512 full `sum` 1.06x, 256x256 broadcast `multiply` 1.08x, and the large
    MLP step 1.08x are all at or inside the band. **H7 did not make matmul
    faster**, and no reading should say otherwise.

    A second, independent 11-round run reproduced every row: all cases again
    bit-identical, every control again holding (256-cubed matmul 0.98x, 8-cubed
    matmul 1.04x, contiguous 16x16 `add` 1.08x, 512x512 `copy` 1.01x), and every
    training step again improving, with individual ratios moving by roughly the
    control band's width in both directions (a 1-element `sum` 2.12x against
    1.94x, `NativeSGD` 1.26x against 1.13x, `NativeAdam` at (128, 128) 1.08x
    against 1.14x). The figures quoted are the first run's; the second is
    recorded in the design so no single number is read as more precise than the
    method supports.

    **Memory did not move, and that is asserted**: the same boundary workload
    allocates 5 native storages, peak 4 live, 584 peak bytes before and after —
    identical. A view's cold footprint is byte-identical; a view that actually
    takes a strided path costs **+296 bytes** for the pointer pair, and only
    **9 of 98** views in an MLP step ever populate it, which is H3's laziness
    argument unchanged. The harness gained three cases, 31 to **34**:
    `ctypes_boundary_strided` (the array-carrying twin of the existing
    array-free `ctypes_boundary`, so the two crossings are separated rather
    than averaged — measured **0.8 us** array-free against **1.3 us** with
    three layout arguments, where pre-H7 it would have been ~7 us), plus
    `elementwise_broadcast_scalar` and `elementwise_broadcast_row`, the two
    broadcast shapes the optimizer and the normalization modules actually use.
    Validation added a **sanitizer negative control**: under Clang ASan,
    test-only code handing `tf_core_sum` two-entry metadata with `ndim = 3`
    produces a `heap-buffer-overflow`, `READ of size 8`, `0 bytes after 16-byte
    region`, in `reduce_prefers_contiguous_blocks` — the exact H3 finding —
    which is what makes the **zero diagnostics across 2,834 sanitized tests** a
    real absence rather than a blind detector. No public API, capability,
    dtype, device, registry value, checkpoint field, or checkpoint version
    moved, and no C ABI symbol was added.

    **Milestone H8 — native elementwise traversal and composed allocation
    efficiency — has since shipped**, the fourth Phase-H milestone to change
    C++ and, like H2, H5, and H6, **not the ABI**: the library still exports
    exactly **52** `tf_*` symbols.

    H8 entered with **two** candidate tracks and an explicit instruction not to
    force both into production. The measurement kept one as a large result,
    kept a narrow piece of the other, and rejected the rest with reasons.
    **Track A — elementwise traversal — was confirmed and is the milestone.
    Track B — composed normalization allocation — was confirmed only as a
    memory result and is reported as timing-neutral.**

    The cost was **decomposed rather than assumed**, and the decomposition is
    what decided the architecture. Driving the generic strided walker and the
    flat contiguous walker through identical `ctypes` calls on identical
    *contiguous* data showed the odometer costing **1.60x-6.42x** the flat
    loop; a separate sweep showed that **all broadcasting is on the odometer**
    (there is no broadcast fast path at all — `_binary_core_op`'s path C builds
    per-operation broadcast strides and hands them to the same generic walker),
    at 2.0-3.6 ns per element. A standalone binary with an anti-hoisting guard
    then split the odometer's cost four ways at `(256, 256)` contiguous `add`:
    the shipped odometer-plus-function-pointer at **123.5 us**, templating
    alone **81.3 us (1.52x)**, collapsing alone **63.6 us (1.94x)**, and
    **both together 11.5 us (10.7x)**. Neither change is worth much alone and
    together they are worth an order of magnitude, because only their
    combination lets the compiler emit a vector loop — the odometer's carry
    chain blocks vectorization and so does an indirect call it cannot see
    through. The same run showed the **existing flat contiguous kernel was
    itself hobbled** by that indirect call: 21.0 us against 11.7 us for the
    identical loop with the operation as a compile-time constant.

    H8 therefore reused the dispatch shape H2, H5, and H6 each proved — one
    hidden metadata builder, inside the existing export, no new symbol, the
    pre-milestone traversal retained. New
    `cpp/include/tf_elementwise_internal.h` declares `tf::build_unary_plan` and
    `tf::build_binary_plan` plus the templated traversals that walk what they
    build; `cpp/src/elementwise.cpp` defines the builders and dispatches to
    them. A **plan is an operation-local normalized descriptor**: built on the
    stack, used by one call, dropped. Nothing is cached, interned, memoized, or
    shared between calls, and it applies exactly two transformations, both of
    which preserve the logical element sequence — **unit axes are dropped**,
    and **adjacent axes are merged** when `stride[outer] == stride[inner] *
    extent(inner)` for *every* operand at once, which is the statement that the
    two axes' address progressions form one arithmetic run. Axes are never
    reordered, split, or transposed. **This is not a layout compiler**; the
    bound is a fixed **4 axes**, which is every tensor the runtime can
    construct. The builders are total, pure, allocation-free, and a function of
    layout metadata alone — never a pointer value, an alignment, a clock, an
    environment variable, or a CPU-feature probe — and a rejection is a
    **fallback, never an error**: rank 0, an extent below 1, an element count
    not representable in `int64`, a collapsed rank still above 4, or an
    overflowing merge *test* all fall back to the retained odometer. (An
    overflowing merge *action* simply leaves the axes unmerged, which is always
    a valid description — collapsing is an optimization, never a correctness
    requirement.) `core_unary` and `core_binary`, the pre-H8 odometers, are
    retained **verbatim** as the shipped generic reference paths, still spelled
    with the odometer's counter.

    **Only operations IEEE-754 actually specifies take the templated path** —
    `add`, `subtract`, `multiply`, `relu_backward`, `relu`, `sqrt`,
    `reciprocal`, and the identity gather behind `tf_core_contiguous_copy`.
    **`exp` and `log` keep exactly the paths they had**, because they are
    library functions with no correctly-rounded guarantee and a toolchain that
    vectorized them through a vector-math library would be free to return
    different bits. Nothing is lost by excluding them: measured, the templated
    traversal is worth **1.05x** on both, inside this machine's noise. The
    pre-H8 flat *binary* kernel was removed rather than kept as a second copy
    that could drift, because the templated row with both strides 1 is the
    identical loop and is total, so no predicate can ever fall back to it.

    **The numerical contract is H8's own, measured over every ordered pair of
    14 IEEE-754 representatives x three operations x five layouts against a
    pre-H8 library built from identical sources.** (1) **Every result in which
    at most one operand is a NaN is bit-identical** — signed zeros, infinities,
    denormals, the smallest normal, the largest finite magnitudes, and a lone
    NaN of either sign with any payload included: **zero differing results**
    across all 15 combinations. (2) NaN **positions** are identical and every
    NaN the arithmetic produces is quiet (`relu_backward` and the identity
    gather **select** an operand rather than computing, so a signaling NaN
    legitimately survives them — identically on both traversals, exactly as H5
    established for the copy). (3) **Subtraction is bit-identical everywhere**,
    two-NaN pairs included, because it is not commutative and the compiler has
    no freedom over which operand reaches the destination register. (4) For
    **addition and multiplication with two NaN operands** the surviving payload
    is outside the contract and asserted in neither direction. **Part 4 is not
    something H8 introduced, and H8 narrows it**: measured on the pre-H8
    library, its *own* flat kernel and its *own* odometer already disagreed on
    **30 of 196** such pairs, while post-H8 the contiguous, same-shape strided,
    and row-broadcast paths agree exactly and only a transposed operand differs
    — on **5 of 196**. This is a *different* qualification from H2's and H6's,
    which concerned NaNs meeting inside an accumulation; here there is no
    accumulation at all, only operand order inside one commutative
    instruction. Nothing is reassociated, no FMA, no fast-math, no intrinsic,
    no `restrict`, and each functor spells the same expression its retained
    function-pointer twin spells, character for character.

    **H1's contract holds unchanged**: elementwise outputs stay uninitialized,
    and the plan walk writes the destination strictly left to right over
    exactly the logical element count — the same order and count the odometer
    produces — proved by poison injected by test infrastructure with two
    patterns over six layouts, with guard elements on both sides and a
    **negative control** showing the detector can fail.

    **Track B** shipped the one composed-allocation change the evidence
    supported: `_NativeBatchNorm` builds its `(1 - momentum, momentum)` pair
    **once per forward** instead of once per buffer — the per-step-constants
    shape H4 proved on the optimizer, never stored on the module, so no scalar
    survives a forward, enters `state_dict()`, or reaches a checkpoint — and
    each blend **releases its temporaries at last use** rather than holding
    them to the call's `finally`. Measured against a **retained pre-H8
    composition executed natively**, with running statistics proved
    bit-identical first: a `NativeBatchNorm1d` training forward goes **25 -> 23**
    allocations with **peak live storages 25 -> 17** and constant fills
    **5 -> 3**; `NativeBatchNorm2d` goes **30 -> 28**, **30 -> 22**, and **5 -> 3**.
    **Its timing effect is neutral** (1.007x-1.106x over 15 alternating
    rounds, only the smallest shape outside the band and only marginally):
    **Track B is a memory result, not a timing result**, and no reading should
    be quoted as if it were otherwise. Four alternatives were **rejected with
    reasons**: releasing the normalization graph temporaries early (proved
    impossible — every one is either read by a backward closure or must stay
    open to accumulate a gradient); adopting the blend's core into the running-
    state transaction instead of copying it (it would move numerical work
    inside the staging phase, changing a failure ordering F5 and F8 prove by
    test, to save two channel-sized copies); caching the eval-mode inverse
    standard deviation (the hidden mutable state this design forbids); and
    reshaping `gamma`/`beta` to `(1, C, 1, 1)` to skip `NativeBatchNorm2d`'s
    affine transposes (F4 rejected that for a semantic reason that has not changed —
    a reshaped parameter is unversioned, so the stale-parameter guard would
    silently stop firing).

    Measured against the pre-H8 library on identical `ctypes` calls with every
    case **bit-identical before either side was timed**, 11 alternating rounds,
    identical-code control band **0.97x-1.08x**: `multiply` row-broadcast
    `(256,256)+(256,)` **10.58x**, `add` strided same-shape `(256,256)`
    **9.67x**, `multiply` NCHW-stat `(32,16,16,16)` **7.15x**, `multiply`
    col-broadcast **6.70x**, `add` scalar-broadcast **6.31x**, rank-3 broadcast
    **6.18x**, NCHW same-shape 3.53x, NCHW-stat `(8,4,8,8)` 3.11x, transposed
    `add` 2.63x, strided `relu` 2.51x, transposed `copy` 2.31x, contiguous
    `sqrt` 2.03x, `reciprocal` 1.98x, `relu_backward` 1.86x, contiguous `relu`
    1.78x, contiguous `add`/`multiply` 1.76x, contiguous `copy` 1.68x. Layer
    and end to end, over 11 alternating **subprocess** rounds with all 31 cases
    bit-identical first: `TensorCore` row-broadcast `multiply` **6.81x**,
    `NativeTensor` broadcast multiply with a graph **6.33x**, scalar-broadcast
    `add` **5.12x**, **`NativeAdam.step()` at (128,128) 2.01x**, `relu` 1.59x,
    contiguous `add` 1.55x, **`NativeBatchNorm1d` eval forward 1.40x**, **`NativeBatchNorm2d`
    eval forward 1.36x**, **`NativeBatchNorm2d` training forward 1.33x**, **`NativeLayerNorm`
    forward 1.30x**, `NativeBatchNorm2d` fwd+bwd 1.25x, `NativeLayerNorm` non-affine 1.22x,
    `NativeBatchNorm1d` training forward and `NativeLayerNorm` fwd+bwd 1.21x, **the large MLP
    training step 1.19x**, `NativeBatchNorm1d` fwd+bwd 1.15x, `contiguous_copy`
    `(512,512)` 1.14x, **the normalized training step 1.08x**, the native Dropout step
    1.06x. **This is the milestone that finally moved the normalization
    modules** — which H6 measured as almost entirely neutral, and which is
    precisely why H0's composed-module H7 was dropped and this one entered.

    Reported just as honestly. **Small normalization shapes are neutral**:
    `NativeBatchNorm1d` training at `(32,16)` **0.98x**, `NativeBatchNorm2d`
    `(8,4,8,8)` 1.02x, `NativeLayerNorm` `(32,16)` 1.06x — below roughly 1,000 elements the fixed
    Python-plus-ctypes cost dominates, which is H3's, H5's, and H6's documented
    boundary finding, unchanged. **The CNN step is neutral (0.99x)** and the
    small MLP and SGD steps sit at the band edge (1.01-1.02x), because a
    convolution step's time is in `tf_core_conv2d_*`, which H8 did not touch.
    The `exp`/`log` controls read 0.97x-1.07x, exactly as the deliberate
    exclusion predicts, and `sum` and 128-cubed `matmul` are inside the band.
    One control is **published rather than buried**: **`matmul` 256 cubed reads
    0.93x-0.96x**, and a focused 25-round run shows the effect at that one size
    only — 64 cubed 1.014x, 128 cubed 1.035x, 256 cubed **0.921x**, 384 cubed
    0.994x — while the identical-code twin reads 0.969x on the same case.
    `matmul.cpp` is byte-identical source compiled with identical flags;
    `elementwise.cpp`'s object code grew 127 KB to 188 KB, moving every
    function's placement in the image. **This is the same whole-translation-
    unit code-layout effect H6 documented**, it is the machine-specific tuning
    the design rejects chasing, every matmul result is bit-identical, the H2
    CTest passes, and no end-to-end case regressed.

    **Memory: Track A moved none, and the odometer's heap-allocated counter is
    now removed on every plannable layout** — a strided elementwise call makes
    **one** allocation where it previously made two, which is a strict
    reduction and which re-anchored one existing fault-injection test (its
    assertion unchanged, its operand changed to a rank-5 reversed view the
    builder declines, with a **new** test asserting the other half). The
    harness gained **four** cases, 34 to **38**:
    `elementwise_broadcast_column` and `elementwise_broadcast_channel_4d`
    (following H5's and H6's separate-rather-than-average precedent — the row
    case stretches the leading axis, the column case the trailing one, and the
    NCHW case puts the stretched axis in the middle where neither side folds
    into it), plus `elementwise_unary_contiguous` and
    `elementwise_unary_transposed`, because every other elementwise case is
    binary and the one-source traversal was only ever visible averaged into a
    two-operand measurement. Native CTests 15 to **16**. **No exported C ABI
    symbol, no new translation unit, and no public control of any kind** — no
    path selector, plan inspector, collapse-mode flag, threshold setter,
    dispatch tracer, profiling counter, environment variable, or "which path
    ran" query — and no SIMD, threading, OpenMP, BLAS, memory pool, scratch
    workspace, general fusion, or fast-math. No public API, capability, dtype,
    device, registry value, checkpoint field, or checkpoint version moved.

    **Milestone H9 — native Conv2d execution efficiency — has since shipped**,
    the fifth Phase-H milestone to change C++ and, like H2, H5, H6, and H8,
    **not the ABI**: the library still exports exactly **52** `tf_*` symbols.

    **H9 was not in H0's ladder.** H0 pencilled that slot in for SIMD,
    threading, or optional BLAS, *conditional and presumed rejected*. It was
    not entered — none of the three qualified, and a larger, safer result was
    available in the same slot. H6 made reductions 3.2×–10.9× faster and left
    every training step neutral; H7 moved every training step but the CNN
    step's share came from its many small calls; H8 moved the normalization
    modules 1.21×–1.40× and the **CNN step stayed neutral at 0.99×** — because
    a convolution step's time is in `tf_core_conv2d_*`, which was still the
    unmodified Phase-D direct loop from D2–D5 while matmul, copy, reduction,
    and elementwise had each been revisited. Convolution was the last large
    compute family running its original correctness-first implementation, and
    it had become the majority of the one workload Phase H had never moved.
    The acceleration decision moved to H10's decision gate, where it is a
    decision rather than an implementation.

    The cost was **decomposed rather than assumed**, and the answer was H6's
    rather than H3's: timing the complete Core wrapper against the bare
    foreign call showed the Python wrapper is a fixed ≈ 8–12 µs, **66 %** of a
    toy `(4,1,6,6)` convolution but **0.2 %** at `(8,3,16,16)` and
    **≈ 0 %** at `(16,8,32,32)`. For any convolution with real work the
    compiled traversal is essentially **100 %** of the cost, so the C++ loop
    was the only target worth having. The composed **bias gradient was
    measured and found immaterial** — three chained `sum` reductions H6
    already made 3.9×–4.1× faster, producing an `(O,)` result beside two
    full-tensor gradients — so H9 changes nothing about it, and that is a
    recorded negative result rather than an oversight.

    All three pre-H9 kernels shared one shape: `n, o, i, j` outer, `c, p, q`
    inner, with the padded source coordinate recomputed and bounds-tested in
    the inner loops. That makes the innermost loop `kernel_width` — typically
    **3** — iterations long, recomputes a row bound that depends only on
    `(i, p)` once per input channel, and makes both gradients read-modify-write
    destinations that far-apart iterations revisit. H9 reuses the dispatch
    shape H2, H5, H6, and H8 each proved: **one hidden predicate, inside the
    existing export, no new symbol, the pre-milestone traversal retained**.
    `tf::conv2d_forward_generic`, `tf::conv2d_input_backward_generic`, and
    `tf::conv2d_weight_backward_generic` are the **Phase-D direct loops
    retained verbatim** as the shipped generic reference paths — reachable
    through ordinary production dispatch and the oracle every optimized result
    is compared against. Beside them: `tf::conv2d_forward_row_sweep`, whose
    nest becomes `n, o, i | c, p, q | j` accumulating into a bias-primed
    output row; `tf::conv2d_input_backward_gather`, which walks `grad_input`
    rows and gathers instead of scattering; and
    `tf::conv2d_weight_backward_gather`, which owns one destination at a time
    and sums it in a register, writing it **once** instead of
    `batch·out_h·out_w` times. Two file-local helpers compute the half-open
    run of kernel taps whose source lies inside the real input — that run is
    always contiguous, which is why solving for it and testing each candidate
    skip the identical taps.

    **The fast-path preconditions are one shared rule plus one
    direction-specific one**: `min(input_width, output_width) >= 4` for all
    three, and additionally **unit stride in both axes** for the input
    gradient. The minimum is measured, not tuned — at a swept extent of 1 the
    optimized forms ran **0.57×–0.93×**, at 2 they ran 1.04×–1.38×, and at 4
    they ran **1.91×–2.40×**. `min(input_width, output_width)` is used because
    it is the honest bound on all three inner loops, and keying the input
    gradient on `input_width` alone **was measured wrong** — a 5-wide input
    with a 1-wide output sweeps a single element and ran **0.73×**. The
    predicates are total, pure, allocation-free, and functions of the integer
    geometry alone — never a pointer value, an alignment, a clock, an
    environment variable, or a CPU-feature probe — and **a false answer is a
    fallback, never an error**. The input gradient alone needs unit stride
    because its gather walks the kernel offsets *downward* to reproduce the
    reference's ascending output order, and that inversion is one-for-one only
    at unit stride; the forward and weight gradient take their optimized paths
    at **every** stride. **The asymmetry is deliberate**, and the strided input
    gradient's 1.04× is the row that proves the fallback is really taken.

    **Per-destination accumulation order is preserved exactly in all three
    directions**, each with its own proof: the forward's `c, p, q` stay outer
    to `j`, so a destination still receives the same seed and the same taps in
    the same order; the input gradient's ascending-`o`, descending-`p`,
    descending-`q` walk *is* ascending `o`, `i`, `j` at unit stride; and the
    weight gradient's `n, i, j` nest is exactly the order its destination's
    contributions already arrived in. Nothing is reassociated, no partial sums
    are combined, no accumulator width changes, and there is no FMA,
    fast-math, tree or pairwise reduction, parallel accumulation, SIMD
    intrinsic, or threading anywhere.

    **The numerical contract is H9's own, measured against a pre-H9 library
    built from identical sources with only `conv2d.cpp` restored.**
    Contractual: (1) **every non-NaN result is bit-identical** — 256 ordered
    pairs of 16 IEEE-754 representatives × 3 directions, **zero non-NaN
    differences**, signed zeros, ±∞, denormals, the smallest normal, and the
    largest finite magnitudes included; (2) **NaN positions are identical** in
    all 768 comparisons; (3) **with at most one NaN reaching a destination the
    paths agree exactly, payload included** — 480 single-NaN configurations
    across five geometries, zero differences; (4) **signed zeros are
    bit-identical** across 80 sign-pattern configurations, with `−0.0`
    surviving only while every addend is `−0.0` and one `+0.0` making the sum
    `+0.0` — both halves asserted on both paths, because the sweep replaces a
    register accumulator with accumulate-into-memory and the weight gather
    does the reverse, exactly the rewrites that could change a zero's sign;
    (5) signalling NaNs are quieted identically and every NaN either path
    produces is quiet. **Not contractual**: when two or more NaNs reach one
    destination the surviving payload may differ (20/256, 20/256, and 29/256
    pairs), asserted in neither direction — the same qualification H2 and H6
    recorded, for the same instruction-selection reason, but **measured here
    rather than assumed from them**.

    **H1's contract holds on all three destinations**, for a different reason
    each: the forward primes every element of every output row with the bias
    before accumulating; the input gradient's gather zeroes each row it visits
    and visits every row; and the weight gradient's gather *assigns* every
    destination from a register, so it needs no zero-fill and never reads the
    destination at all. Proved by poison in two places — the C++ suite
    pre-fills each optimized destination with a quiet NaN and a large finite
    value across the whole geometry matrix, and the Python suite injects the
    same two patterns through the real private allocation seam for both the
    optimized and the fallback geometries — each with a **negative control**
    proving the detector can fail.

    **Layout handling is untouched**: the C ABI is contiguous-only by Policy B
    and the Core layer materializes any non-contiguous operand into a private
    copy, so **H9 is a geometry optimization, not a layout one** and broadened
    layout support by nothing. Autograd is untouched — same parent topology,
    same conditional version tracking, gradients created only for parents that
    require them, and the same `retain_graph`, repeated-backward,
    accumulation, and cleanup behaviour. Nothing became in-place.

    **Memory did not move, and that is asserted rather than assumed**: the
    same workload on both libraries reports **byte-identical** allocation
    counts, peak live storages, and peak bytes — a forward is 1 allocation /
    921,600 peak bytes, an input gradient 1 / 524,288, a weight gradient
    1 / 9,216, a `NativeConv2d` forward+backward 8 / 3,020,160, and a CNN
    training step 94 allocations / 34 peak live / 604,848 peak bytes. There is
    **no scratch buffer, workspace, arena, pool, padded copy, or im2col
    allocation anywhere**.

    Measured against the pre-H9 library on identical `ctypes` calls, every
    case **bit-identical before either side was timed**, 11 alternating rounds,
    with the two fallback geometries as the identical-code control at
    **1.00×–1.13×**: k1×1 `(8,16,32,32)` **6.23× / 8.37× / 5.40×** (forward /
    input / weight), `(8,16,32,32) → 32` k3×3 3.64× / 5.04× / 2.60×,
    `(16,8,32,32) → 16` 3.32× / 5.28× / 2.47×, padded `(8,8,32,32)` 2.91× /
    4.97× / 2.54×, `(8,3,16,16) → 8` 2.87× / 3.75× / 2.45×, prime extents
    2.61× / 4.57× / 2.77×, **stride 2** 2.41× / *1.04× (falls back)* / 2.41×,
    rectangular k3×5 2.30× / 3.76× / 2.22×, k5×5 1.97× / 3.84× / 2.09×. End to
    end over 9 alternating **subprocess** rounds, all 23 checksums identical
    first: `NativeConv2d` forward+backward **3.13×** padded and **3.09×**
    unpadded, forward **2.98×**, no-bias 2.46×, frozen 2.40×, stride-2 2.28×,
    and — the result that matters — **a CNN training step 1.86×** at
    `(8,3,32,32) → 16`, **1.38×** at `(8,3,16,16) → 8`, 1.27× with a native Dropout layer,
    **1.13×** at the shipped example's shape, and 1.11× with a native BatchNorm2d layer.
    **This is the first Phase-H milestone to move a CNN training step**, which
    H6 and H8 both measured as neutral.

    Reported just as honestly: a **small convolution is neutral** (1.06×
    forward, 1.20× forward+backward at `(4,1,8,8) → 4`), because below roughly
    a thousand output elements the fixed ≈ 10 µs Python-plus-ctypes cost
    dominates — H3's, H5's, H6's, and H8's documented boundary finding,
    unchanged; the **native BatchNorm2d and shipped-example CNN steps move least**
    (1.11×, 1.13×), because convolution is a smaller share of those steps; and
    **no control regressed** — at 21 alternating rounds matmul 256³ **0.98×**,
    the MLP training step **0.97×**, `contiguous_copy` 512² 1.01×, reduction
    1.07×, broadcast elementwise 1.07×, a control band of **0.97×–1.07×**. One
    methodology finding is published rather than buried: at 9 rounds the
    elementwise control read **0.93×** and looked like a regression, while at
    21 rounds the same case read **1.07×** — the lesson H3, H5, and H6 each
    recorded, so no low-round figure is quoted as H9 evidence.

    **Four candidates were rejected with reasons**: **im2col + matmul**
    (changes the accumulation order, which is the whole contract, and would
    allocate 8× the input at the profile shape); a **materialized padded
    input** (moves cost rather than removing it, when the tap-range helpers
    give the same branch-free inner loop with no allocation); **output-channel
    blocking** (the sweep already produces a long unit-stride inner loop, and
    blocking `o` would reintroduce a tuning constant for a second-order gain);
    and a **third "hoisted" path for small extents** (measured 1.5×–3.7× and
    never regressing, but it would have left the Phase-D reference unreachable
    for shapes that cost microseconds).

    Validation: Windows **Release and Debug**, both out-of-source with the
    Debug library written outside the repository so the active runtime stayed
    the Release DLL, **17/17 CTests each** with zero project compiler, linker,
    and CMake warnings; Clang 18.1.3 ASan/UBSan with **instrumentation
    proved** — 22 `__asan*` and 15 `__ubsan*` dynamic symbols beside the
    **52** exported `tf_*` symbols, independently confirming the export count
    on a second toolchain — **17/17 sanitized CTests**, **445 sanitized
    convolution/CNN/Phase-D/H1 tests**, the full sanitized native suite with
    **zero ASan and zero UBSan diagnostics**, and both shipped CNN examples
    reproducing their exact checkpoint resumes under it. A **sanitizer
    negative control** makes that absence real: handing the row sweep an input
    one row shorter than its declared geometry produces a
    `heap-buffer-overflow`, `READ of size 8`, inside
    `conv2d_forward_row_sweep`. A LeakSanitizer lifecycle returns native live
    storage **exactly to baseline (0)** at every checkpoint — core forward and
    both gradients over optimized *and* fallback geometries, module cycles,
    seven injected-failure cycles, abandoned graphs, and two complete CNN
    training runs — with the remaining process-exit allocations containing
    **no TensorForge frame** and no suppression file added.

    The harness gained **three** cases, 38 to **41** —
    `conv2d_forward_padded`, `conv2d_forward_strided`, and
    `conv2d_forward_fallback`, following the separate-rather-than-average
    precedent, because unlike H5/H6/H8 the chooser here is the *geometry*
    rather than the layout; all three are `native_only` and publish **no
    ratio**, and the fallback case is the family's control since its compiled
    path did not change. Native CTests 16 to **17**. **No exported C ABI
    symbol, no new translation unit, and no public control of any kind** — no
    path selector, block-size setter, traversal control, dispatch tracer,
    benchmark hook, profiling counter, environment variable, or "which path
    ran" query — and no SIMD, threading, OpenMP, BLAS, oneDNN, Eigen, memory
    pool, scratch workspace, im2col, or fast-math. No convolution option was
    added: no dilation, no groups, no channels-last, no new padding mode. No
    public API, capability, dtype, device, registry value, checkpoint field,
    or checkpoint version moved.

    The ladder ran **H0–H10 and ended there**: it was reordered at H5, revised at H7 (a milestone dropped on evidence), and extended at H9 (a slot reassigned), and H0's separate H11 closure slot was **not needed** because H10 carried closure itself. A memory pool, scratch allocation, SIMD, threading/OpenMP, and BLAS were **all finally rejected at H10, with measurements** — the disassembly showed elementwise, matmul, and reduction are already auto-vectorized; a CNN step's 198 native calls have a **1.20 µs median** with only two above 1 ms; and BLAS is **not bit-identical** (3.553e-15 at 64³), which would break every exact-resume proof. The criteria that would reopen each are recorded rather than an answer invented. Every number is a local characterization of one machine, reported with its spread, and asserted by no test.
    Deliberately outside Phase H: CUDA, float32/float16/bfloat16,
    casting, dtype promotion, AMP, Tensor Cores, pybind11, C++ autograd,
    implicit dispatch, Transformers, attention, embeddings, integer
    tensors, data loaders, distributed training, a memory pool, scratch
    allocation, SIMD, threading, OpenMP, BLAS, any required dependency,
    checkpoint format version 3, and **any CI timing threshold or
    committed performance number**.
  - **Then beyond (not started):** the CUDA
    runtime (where `device` gains a second value), an AMP / Tensor Core path
    (where `dtype` gains float16/bfloat16), Transformer / text examples,
    distributed / DDP, and a final benchmark / profiling / docs polish
    (the final portfolio release).
- **A larger synthetic image example** — more classes, bigger images,
  still dependency-free.
- **More docs** — deeper walkthroughs of individual layers, if the
  framework grows further.

## What this project is not

TensorForge is not production software and doesn't try to compete with
PyTorch or any real framework. It trades performance for readability
at every opportunity — that's the point. If it helps someone
understand what `loss.backward()` actually does, it has succeeded.
