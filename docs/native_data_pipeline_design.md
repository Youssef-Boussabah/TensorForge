# Deterministic native data pipeline and mini-batching — Phase J architecture contract

**Phase J — Deterministic Native Data Pipeline and Mini-Batching.** This
document is the authoritative architecture contract for the phase. It is
written **before** any data-pipeline implementation existed, and milestone
**J0** consisted of exactly this document, the status reconciliation it
required, and the contract guardrails that keep it honest.

**J0 added no runtime behavior.** No dataset class, no sampler class, no
loader class, no helper module, no state serializer, no shuffle helper, no
batching helper, no public export, no production import, no C++, no CMake
registration, no C ABI symbol, no example, no benchmark, no checkpoint
field, no checkpoint version, and no optimizer-state version. Runtime
capability began at **J1**, which added **exactly one** public name —
`NativeTensorDataset` (§3.3, §4, §5, §6) — continued at **J2**, which
added **exactly one** more — `NativeBatchSampler` (§3.4, §7, §8, §11.2,
§12.3, §12.4) — over the permanently private `_native_permutation`
derivation (§3.2, §8), and continued at **J3**, which added the last of
§3.1's three names, `NativeDataLoader` (§3.5, §3.6, §9, §10, §15, §17.3),
over the permanently private `_NativeBatchIterator` and `_deliver_batch`.

**Phase-J status: J0, J1, J2, J3, J4, J5, J6, J7, and J8 complete; J9 not
started.** What exists today is the dataset, the sampler, the loader, the
loader's own in-memory state, the caller-managed checkpoint-metadata
workflow, and a worked training program over all of it: a finite
host-backed dataset, a deterministic permutation and batch **planner** with
an explicit epoch and cursor, compact transactional sampler state, native
mini-batch iteration whose committed position advances **if and only if** a
batch was delivered, the three-key loader state schema (§11.3), the loader
`state_dict` and `load_state_dict` pair (§12.5), **exact in-memory
mid-epoch loader restoration** (**J4**), and — at **J5** — that same state
proved to survive a **real version-3 archive** as ordinary caller metadata
and to restore an exact continuation into entirely fresh objects. **J5
added no production code**: the checkpoint module is provably unchanged,
the format is still version 3 with `(1, 2, 3)` accepted, and the archive's
own capture set did not grow by one field.

New at **J6**: `examples/native_minibatch_training.py`, the **deterministic
native mini-batch training example** — the first end-user program to train
a native model through the pipeline rather than a hand-indexed array — and
its **exact interrupted-versus-uninterrupted training proof**, run
independently at float32 and float64 with every numeric comparison made in
raw IEEE-754 bits and never across dtypes. **J6 added no production code
and no public name either**: its whole diff is that example,
`tests/test_native_minibatch_training.py`, the narrow status edits landing
it requires, and documentation. The example inventory moved 15 → **16**;
`tensorforge.experimental.__all__` stayed at **25**.

New at **J7**: `tests/test_native_data_hardening.py`, the **adversarial
hardening matrix** — the cross-cutting evidence that every §12.7, §15,
§16, and §17 row holds under attack. It injects a failure at each §17.2
construction row and each §17.3 iteration row, separating the host
gather, the native allocation, the host→native transfer, and the target
copy into four distinct injections; it makes the **commit step fail after
the candidate position has really been applied**, not merely instead of
applying it; it drives a `BaseException` through the same path; and it
proves a **checkpoint taken immediately after a failed delivery** resumes
the same candidate batch through a real version-3 archive into an
entirely fresh object graph. Every rejection is followed by a complete
before/after fingerprint of the observable world — dataset, sampler,
loader, iterator, an unrelated parameter, buffer, optimizer, and
registered generator, the filesystem, the global RNGs, and every
registry — and every injection and every parser carries its own
non-vacuity control. **J7 found no production defect and changed no
production code**: its whole diff is that test module, the narrow
inventory edits landing it requires, and documentation. The example
inventory stays **16**, the benchmark inventory stayed at **8** through
J7, and `tensorforge.experimental.__all__` stays at **25**.

New at **J8**: `benchmarks/benchmark_native_data_pipeline.py`, the
**data-pipeline benchmark** (§23's J8 row, and the outcome recorded in
§23.2) — the local characterization of what each pipeline layer costs,
with float32 and float64 measured **separately and never as a ratio of
one to the other**, correctness gated **before** any timing, `native_only`
cases publishing **no ratio**, and **no result file** of any kind. **J8
changed no production code, added no public name, and shipped no
optimization**: its whole diff is that harness,
`tests/test_native_data_benchmark.py`, narrow inventory edits, and
documentation. Benchmarks moved 8 → **9**; examples stay **16** and
`tensorforge.experimental.__all__` stays at **25**.

**No automatic loader discovery exists yet**, and the phase closure has
not started: **J9 is the next implementation milestone**. §14's
statements about a *training* program describe shipped, executable
evidence, and §14.1's failed-delivery leg is now J5's archive proof
**and** J7's injection matrix rather than anything the public example
does.

Phase J is a **newly approved** direction. It was not part of the roadmap
while Phases A–I were being built: the repository deliberately closed
Phase I at I11 without committing to a successor, and Phase J was approved
afterwards. Nothing in this document may be read as describing work that
already existed.

**Phase I remains complete (I0–I11) and is the latest *completed* phase.**
Phase J is the latest phase. Nothing in Phase J revisits, reverses, widens,
or re-measures a Phase-I result.

**What Phase J will eventually deliver**, once J1–J9 have landed: a small,
finite, host-backed dataset; a deterministic batch sampler; and a native
mini-batch loader that yields `NativeTensor` feature batches beside copied
host `int64` class targets — with explicit epoch and cursor state, strict
JSON-compatible state serialization, transactional state restoration, exact
middle-of-epoch resume, exact future batch-index reproduction, explicit
caller-managed checkpoint-metadata integration, deterministic cleanup, and
unbroken stable/native isolation.

**What Phase J will not deliver, at any milestone.** It grants no dtype, no
device, no capability-registry value, no checkpoint version, and no
optimizer-state version. `SUPPORTED_DTYPES` stays `("float64", "float32")`,
`SUPPORTED_DEVICES` stays `("cpu",)`, `UNSUPPORTED` stays `("cuda",
"amp")`, and `RAW_KERNEL_DTYPES` stays `("float64",)`. The native
checkpoint stays `tensorforge.native_checkpoint` version **3** with `(1, 2,
3)` accepted, and the in-memory optimizer state stays version **1**. The
phase plans **no** new C ABI export: the production library exports **54**
`tf_*` symbols at J0 and is expected to export 54 at J9 (§22.3 states the
one narrow condition under which that could be revisited, and the evidence
that would be required first).

Related contracts this document builds on rather than replaces:
[native_dtype_float32_design.md](native_dtype_float32_design.md) (the dtype
model, the no-cast rule, and host-array ingress as a non-cast),
[native_rng_dropout_design.md](native_rng_dropout_design.md) (the
`tensorforge.splitmix64` derivation this phase reuses and the
explicit-state discipline it established),
[native_tensor_wrapper_design.md](native_tensor_wrapper_design.md) (native
tensor ownership and `close()`),
[native_classification_design.md](native_classification_design.md) (the
strict host `int64` class-target contract),
[native_cpu_performance_design.md](native_cpu_performance_design.md) (the
benchmark discipline J8 inherits), and
[native_abi_error_contract.md](native_abi_error_contract.md).

---

## 1. Objective and scope

### 1.1 What Phase J delivers

One deterministic, finite, host-backed data pipeline for the experimental
native training stack, in three objects:

- **`NativeTensorDataset`** — a finite dataset holding one owned,
  canonical, contiguous host snapshot of the features and one of the class
  targets, at an **explicitly chosen** native feature dtype, able to
  produce a `NativeTensor` feature batch and a copied host `int64` target
  batch for any index sequence.
- **`NativeBatchSampler`** — a deterministic order and batch planner. It
  owns `batch_size`, `drop_last`, `shuffle`, the `seed`, the `epoch`, and
  the `cursor`, and produces batch-index groups. It allocates nothing
  native and holds no consumable stream: every permutation is a **pure
  function** of `(seed, epoch, length)`.
- **`NativeDataLoader`** — the iteration surface over a sampler. It turns
  each planned index group into `(NativeTensor, numpy.ndarray)` and hands
  ownership of both to the caller.

Plus: strict state schemas for the sampler and the loader, a transactional
`load_state_dict` on each, an explicit caller-managed checkpoint-metadata
workflow, a deterministic mini-batch training example with an exact
interrupted-versus-uninterrupted proof, cross-cutting hardening, honest
per-dtype benchmark characterization, and cross-platform closure.

### 1.2 What Phase J does not deliver

Not a capability phase in any other direction. No dtype, device, casting,
promotion, mixed-dtype arithmetic, integer tensor, gather/scatter tensor
operation, embedding, sparse tensor, transform framework, user collation
hook, worker process, thread, prefetch, asynchronous iteration, pinned
memory, distributed sampler, network or filesystem dataset, streaming or
infinite dataset, memory map, checkpoint schema change, or stable-framework
change. §20 restates these as boundaries.

### 1.3 Why this phase, and why now

Every native training proof shipped so far — the MLP (v3.9), the CNN (D11),
the classifier (E8), the normalized regressor (F6), the stochastic model
(G7), and the dual-dtype float32 model (I9) — trains over a **fixed,
whole-batch, hand-indexed** array held by the example script. That was the
right scope for those phases, and it is now the last structural gap between
the native line and an ordinary training loop.

It is also the last place where "exact resume" is narrower than it reads.
Every phase from C onward proves exact checkpoint resume, and every one of
them has had to say the same honest limit out loud: *reproducibility is
exact for the state TensorForge captures, which is not a data loader, a
shuffle order, or an epoch counter.* Phase J does not change the checkpoint
to capture those; it makes them **explicit, serializable, and restorable by
the caller**, so a resumed run can reproduce its remaining batches exactly
without the archive growing a field or a version.

The prerequisites are all in place and none of them is new work: dtype-aware
native construction and transfer (I2, I9), a locked deterministic integer
derivation (G2), a strict host `int64` target contract (E5), a
JSON-validated checkpoint metadata channel validated by one authority on
both sides (I10), and an established transactional state-loading pattern
(v3.3, v3.13, G5).

---

## 2. Repository reality at J0 — the verified baseline

Every statement in this section was read out of the tree at J0, not
remembered. It is the reality the rest of the document is designed against,
and a later milestone that finds it false must say so rather than proceed.

### 2.1 The registries and versions Phase J must not move

Read from the live module at J0:

| Fact | Value at J0 |
|---|---|
| `cpp.SUPPORTED_DTYPES` | `("float64", "float32")` |
| `cpp.SUPPORTED_DEVICES` | `("cpu",)` |
| `cpp.UNSUPPORTED` | `("cuda", "amp")` |
| `cpp.RAW_KERNEL_DTYPES` | `("float64",)` |
| `cpp.normalize_dtype(None)` | `"float64"` |
| `cpp.backend_info()["dtype"]` | `"float64"` |
| `native_checkpoint._FORMAT` | `"tensorforge.native_checkpoint"` |
| `native_checkpoint._FORMAT_VERSION` | `3` |
| `native_checkpoint._SUPPORTED_FORMAT_VERSIONS` | `(1, 2, 3)` |
| `native_optimizer_state.FORMAT_VERSION` | `1` |
| Production `tf_*` exports (source) | **54** |
| Registered native CTests | **24** |
| Examples | **15** |
| Benchmark harnesses | **8** |

### 2.2 There is no existing dataset, sampler, loader, cursor, or epoch concept on the native line

Searched at J0 across `src/`, `cpp/`, `tests/`, `docs/`, `examples/`, and
`benchmarks/`. The findings:

- **`tensorforge.data.batches`** exists, and it is the **stable** line's
  mini-batch iterator over NumPy arrays. It is exported from the stable
  root package, is locked by `tests/test_public_api.py`, and is entirely
  outside the native line. Phase J neither uses, wraps, extends, imports,
  nor changes it (§18).
- **`tensorforge.train_test_split`** is likewise stable-only.
- No native module, class, function, or attribute anywhere spells
  *dataset*, *sampler*, *loader*, *cursor*, *epoch*, *shuffle*, or
  *collate*. The names are free, and Phase J is not colliding with or
  reinterpreting an existing concept.
- The words do appear in native **documentation and tests**, exclusively as
  statements of *absence*: the native checkpoint deliberately captures no
  data-loader position, shuffle order, or epoch counter, and
  `tests/test_docs.py` enforces that no status surface claims otherwise.
  §13.6 states precisely why Phase J leaves that guardrail true rather than
  retiring it.

### 2.3 Native construction, transfer, and the host boundary

- `NativeTensorCore.from_array(values, dtype=None, device="cpu")` performs
  `np.ascontiguousarray(values, dtype=_DTYPE_NUMPY[normalize_dtype(dtype)])`
  and copies the result into fresh storage. This is the explicit
  host-to-native **conversion** boundary, and it has always converted; it
  is **not** a tensor cast (dtype design §9.4).
- **The dtype is never inferred from the input array.** `dtype=None` means
  `"float64"`, so a `float32` NumPy array with no `dtype` argument produces
  **float64** storage. Phase J preserves this exactly (§19.2).
- `NativeStorage.__init__` rejects a non-positive size: *"size must be a
  positive int"*. `_as_shape` rejects any non-positive dimension: *"shape
  dimensions must be positive ints"*. **A native tensor with zero elements
  cannot exist.** This is load-bearing for §4.6 (empty datasets) and
  §7.5 (zero-batch epochs).
- `NativeStorage.to_numpy()` returns an array of **exactly** the storage's
  dtype and never widens.
- Every `NativeTensor` operation allocates a fresh owning contiguous
  output; `close()` is the contract and `__del__` is a fallback only.

### 2.4 The class-target contract already exists and is strict

`cpp._prepare_class_targets` is the Phase-E authority. It rejects
`str`/`bytes`, rejects a NumPy array that is `bool_` or non-integer,
rejects a non-1-D array, rejects scalars, rejects nested and ragged
sequences, requires the exact batch length, requires int64
representability, requires `0 <= value < num_classes`, and only then takes
an **independently owned, C-contiguous, read-only `np.int64`** copy. It is
deliberately stricter than `np.asarray(targets, dtype=np.int64)`, which
would truncate `1.9` and reinterpret `True`.

Two consequences Phase J inherits rather than re-decides: **targets are
host metadata, never native tensors** (the runtime has no integer dtype),
and **the number of classes is the model's fact, checked at every
`cross_entropy` call** — so the dataset must not become a second authority
on it (§4.4).

### 2.5 The deterministic derivation already exists, and it lives in C++

`cpp/include/tf_random_internal.h` and `cpp/src/random.cpp` hold the locked
`tensorforge.splitmix64` derivation, algorithm version 1:

```
kSplitMix64Golden = 0x9E3779B97F4A7C15

splitmix64_mix(x):  x ^= x >> 30;  x *= 0xBF58476D1CE4E5B9
                    x ^= x >> 27;  x *= 0x94D049BB133111EB
                    x ^= x >> 31
dropout_stream_key(seed, call)   = splitmix64_mix(seed + GOLDEN * (call + 1))
dropout_element_bits(key, index) = splitmix64_mix(key + GOLDEN * (index + 1))
dropout_uniform(bits)            = (bits >> 11) * 2**-53
```

All arithmetic is on `std::uint64_t` with wrapping (modulo 2**64)
semantics, spelled `std::uint64_t` explicitly so MSVC, GCC, and Clang
produce identical bits. A change to any constant, shift, multiplication
order, derivation, conversion, or comparison direction **must mint a new
`(algorithm, algorithm_version)` pair**.

There is **no Python-side bit derivation anywhere in the repository**, and
no exported symbol that returns raw random bits. `NativeGenerator` is a
pure-Python value holder — an algorithm identifier and version, a `uint64`
seed, and a counter of committed calls — with no `random`, `bits`,
`uniform`, `next`, or `mask` method, asserted absent by test. §8 is written
against exactly this reality.

### 2.6 The checkpoint metadata channel

`save_native_checkpoint(path, model, optimizer=None, metadata=None)`
validates `metadata` through `_validated_metadata`, which accepts, exactly
and recursively: `None`, `bool`, `int`, **finite** `float`, `str`,
`list`/`tuple` (tuples normalize to lists), and `str`-keyed `dict`. Exact
`type() is` checks, so a NumPy scalar is rejected even though `np.float64`
subclasses `float`. Cyclic containers are rejected.

Since milestone **I10** the **same authority runs on load**, so the set a
load accepts is exactly the set a save could have written.
`load_native_checkpoint` returns the metadata as an independent structure
that shares nothing with the archive.

Python `int` is arbitrary precision and `json` round-trips it exactly, so a
full unsigned 64-bit seed survives the channel without truncation. `bool`
is checked before `int`, so `True` stays `True` rather than becoming `1`.
**Every field §11 defines is JSON-native and passes this validator
unchanged** — which is why Phase J needs no checkpoint schema change, no
new root field, and no version 4.

### 2.7 The established state-loading pattern

Three shipped loaders set the pattern Phase J follows verbatim:

- `NativeModule.load_state_dict` (v3.3) — full preflight naming the failing
  key, stage-then-commit, rollback, identity preserved.
- `NativeSGD` / `NativeAdam` `load_state_dict` (v3.13) — one versioned
  schema, exact key set, exact `format_version`, exact type tag, positional
  parameter metadata validated against live reality, sequences accepted as
  tuple **or** list, no `strict=False`, no casting, no device movement.
- `NativeGenerator.load_state` (G1) — validate everything outside the lock,
  then two `__slots__` integer assignments that **cannot fail**, which is
  what makes rollback exact.

And one distinction Phase J adopts deliberately, taken from `NativeAdam`
(§12.4): **structural facts about live objects are validated against
reality and never adopted** (Adam validates each parameter's
shape/dtype/device); **configuration the state carries is adopted** (Adam's
Phase-3 commit replaces `lr`, `betas`, and `eps`).

### 2.8 Error-type conventions

`TypeError` for a wrong type — including `bool` where an `int` is required,
which the repository rejects everywhere (`_validate_uint64`,
`validate_step_counts`, `_validated_metadata`'s ordering). `ValueError` for
a well-typed but unacceptable value. `RuntimeError` for a lifecycle or
state conflict — a closed object, an outstanding reservation, a stale
graph. Messages name a stable field path (`state['sampler']['cursor']`),
never a memory address, and no exact message text is a public contract.

### 2.9 Lifecycle conventions

`close()` exists **exactly where something is owned**, is idempotent,
returns `None`, and is paired with `__enter__`/`__exit__` and a `closed`
property. `NativeGenerator` deliberately has **no** `close()`, because
inventing one "would advertise a lifetime that does not exist". §15 applies
that rule rather than a blanket one.

### 2.10 Concurrency reality

`_native_state_lock.state_transaction()` is a process-wide guard for
**state-replacement** participants, taken outermost, before any generator
lock, in a universal `id()`-sorted order. Ordinary training mutation does
**not** take it, and thread-safe concurrent training is explicitly not
offered. §16 explains why the Phase-J objects join neither the guard nor
the order.

---

## 3. Public API surface

### 3.1 The three eventual public names

| Class | Milestone | Module | Exported from `tensorforge.experimental` |
|---|---|---|---|
| `NativeTensorDataset` | J1 | `experimental/native_dataset.py` | at J1 |
| `NativeBatchSampler` | J2 | `experimental/native_sampler.py` | at J2 |
| `NativeDataLoader` | J3 | `experimental/native_data_loader.py` | at J3 |

**Three names, and no more.** Each is added to
`tensorforge.experimental.__all__` in the milestone that implements it, and
never before — the export inventory is a contract locked by test, and an
exported name whose class does not work is exactly the over-claim this
project's rollout discipline exists to prevent.

The names follow the repository's descriptive-and-exact convention
(`NativeBatchNorm1d`, `NativeCrossEntropyLoss`, `NativeMaxPool2d`):

- **`NativeTensorDataset`** says what it holds — in-memory tensors' worth
  of host data — and distinguishes it from any future dataset that would
  not.
- **`NativeBatchSampler`**, not `NativeSampler`, because it emits complete
  **batch** index groups and owns `batch_size` and `drop_last` (§7.1). A
  bare `NativeSampler` would name an object that emitted single indices,
  which this one does not.
- **`NativeDataLoader`** is the iteration surface.

### 3.2 Private names that stay private

| Name | Module | Why private |
|---|---|---|
| `_native_permutation.py` | `experimental/` | The derivation helpers (§8). A public bit-generation API would be a second RNG surface beside `NativeGenerator`, which §20 forbids. |
| `_NativeBatchIterator` | `native_data_loader.py` | Callers receive iterators from `iter(loader)`; they never construct one. Same stance as `NativeGenerator`'s reservation token. |
| `_deliver_batch` | `native_data_loader.py` | The §9.4 Phase-4 delivery seam. A module-level function that returns its record's `(features, targets)` and does nothing else, so the publish-to-delivery failure position is addressable and can be tested by monkeypatching the module attribute. **Not a hook**: it takes no user-supplied callable and no public callback exists. The direct analogue of `native_generator._deliver_reservation`. |
| `_claim_batch`, `_publish_pending`, `_rollback_pending`, `_complete_pending`, `_assign_position`, `_snapshot_position` | `NativeBatchSampler` | The §9.4 transaction primitives. Only the loader's iterator may open a transaction or move a cursor; a public advance would let a caller desynchronize an active iteration or strand a pending delivery. |
| `_begin_iteration`, `_end_iteration` | `NativeBatchSampler` | The active-iteration count behind §9.5's `load_state_dict` refusal. |

`_native_permutation.py` sits beside `_native_dtype.py`, `_native_state.py`,
and `_native_state_lock.py` — the established place for a private helper
that must not become an API. It imports `hashlib` and nothing from
`backends`, so the "normalization files reach no ctypes layer" property has
a direct precedent.

### 3.3 `NativeTensorDataset` — the eventual surface

```python
NativeTensorDataset(features, targets, *, dtype=None)
```

| Member | Kind | Contract |
|---|---|---|
| `samples` | property → `int` | Number of samples; `>= 1` always. |
| `feature_shape` | property → `tuple[int, ...]` | The **per-sample** shape. `()` for scalar samples. |
| `dtype` | property → `str` | `"float64"` or `"float32"` — the native dtype every feature batch will carry. |
| `device` | property → `str` | `"cpu"`. Present for uniformity; there is no `device` argument (§19.7). |
| `fingerprint` | property → `str` | 64 lowercase hex characters (§6). |
| `closed` | property → `bool` | |
| `__len__()` | → `int` | `samples`. |
| `identity()` | → `dict` | The four JSON-compatible identity fields (§6.4). Fresh dict each call. |
| `feature_batch(indices)` | → `NativeTensor` | An **owning** contiguous tensor of shape `(len(indices),) + feature_shape` at `self.dtype`. **The caller closes it.** |
| `target_batch(indices)` | → `numpy.ndarray` | A fresh, independently owned, C-contiguous, **read-only** `int64` array of shape `(len(indices),)`. |
| `close()` | → `None` | Releases both host snapshots. Idempotent. |
| `__enter__` / `__exit__` | | Standard; `__exit__` closes and returns `False`. |
| `__repr__` | | Metadata only: samples, per-sample shape, dtype, device, closed, and the first 12 fingerprint characters. Never data. |

There is **no public accessor for the host snapshots**. Returning them
would either alias caller-visible memory into the dataset (defeating §5) or
copy the whole dataset on a property read. `feature_batch` and
`target_batch` give exactly what is needed, and the second is read-only.

`identity()` is public because a caller may legitimately want to record
which dataset a run used without building a sampler.

### 3.4 `NativeBatchSampler` — the eventual surface

```python
NativeBatchSampler(dataset, *, batch_size, shuffle=False, seed=0,
                   drop_last=False)
```

| Member | Kind | Contract |
|---|---|---|
| `dataset` | property | The dataset object it was constructed with — identity, not a copy. |
| `batch_size`, `shuffle`, `seed`, `drop_last` | properties | Current configuration. Each may be **replaced by `load_state_dict`** (§12.4). |
| `epoch` | property → `int` | The epoch currently being consumed (§7.3). |
| `cursor` | property → `int` | Batches already delivered in the current epoch; always `0 <= cursor < batches_per_epoch` (§7.4). |
| `batches_per_epoch` | property → `int` | `>= 1` always (§7.5). Depends only on `(samples, batch_size, drop_last)`, so it is the same for every epoch. |
| `remaining` | property → `int` | `batches_per_epoch - cursor`. |
| `epoch_permutation(epoch=None)` | → `tuple[int, ...]` | The full order for an epoch (default: the current one). **Pure**: changes nothing. |
| `plan(epoch=None)` | → `tuple[tuple[int, ...], ...]` | The full batch plan for an epoch. **Pure**. |
| `next_batch_indices()` | → `tuple[int, ...]` | The indices the next delivered batch will use. **Pure**: does not consume. |
| `state_dict()` | → `dict` | §11.2. Fresh, JSON-compatible, sharing nothing. Describes the position after the last **successfully delivered** batch, so it is always the exact next batch. **`RuntimeError` while a §9.4 transaction is in flight** (§9.5). |
| `load_state_dict(state)` | → `None` | §11.4, §12.4. Transactional; identity preserved. **`RuntimeError`** while an iteration is active or a §9.4 transaction is in flight. |
| `__repr__` | | Configuration and position only. |

**No `close()`**, on `NativeGenerator`'s precedent: a sampler owns no native
storage, no host snapshot, and no file handle, so a `close()` would
advertise a lifetime it does not have (§15.3).

`epoch_permutation` and `plan` are public deliberately. They make the
determinism contract testable, and observable by a user, **through the
ordinary API** — which is why §10.3 needs no private debug index hook.

### 3.5 `NativeDataLoader` — the eventual surface

```python
NativeDataLoader(sampler)
```

| Member | Kind | Contract |
|---|---|---|
| `sampler` | property | The sampler object — identity. |
| `dataset` | property | `self.sampler.dataset`. Convenience only. |
| `closed` | property → `bool` | |
| `__iter__()` | → iterator | Returns a fresh `_NativeBatchIterator` for **one epoch** and supersedes any previous one; **`RuntimeError`** while a §9.4 transaction is in flight (§9.2). |
| `state_dict()` | → `dict` | §11.3. **`RuntimeError`** while a §9.4 transaction is in flight (§9.5); allowed at every other time, including after `close()`. |
| `load_state_dict(state)` | → `None` | §11.4, §12.5. **`RuntimeError`** when closed, when an iteration is active, or while a transaction is in flight. |
| `close()` | → `None` | **Rolls back any in-flight transaction** (§9.4, Phase 5), closes the resulting undelivered batch, then closes the active iterator. Never refused. Idempotent. **Never closes a delivered batch.** |
| `__enter__` / `__exit__` | | Standard. |
| `__repr__` | | Delegates the sampler's configuration and position; never data. |

**The loader takes a sampler, not a dataset plus six keyword arguments.**
The composition is then explicit, the dataset is named once, each
configuration value is spelled in exactly one constructor, and J2 is
independently useful and testable before J3 exists. The rejected
alternative — `NativeDataLoader(dataset, *, batch_size, shuffle, seed,
drop_last)` building its own sampler — is more convenient by one line and
duplicates every argument, every validation, and every error message across
two constructors.

**No `__len__` on the loader.** Mid-epoch it would have to mean either
"batches per epoch" or "batches remaining", and a caller reading the wrong
one would silently mis-schedule a resumed run. `loader.sampler
.batches_per_epoch` and `loader.sampler.remaining` each say which is meant.

### 3.6 The iterator

`_NativeBatchIterator` is private, is never exported, and is never
constructed by a caller. Its behavior is public:

- `__iter__()` returns `self`.
- `__next__()` runs the §9.4 transaction and returns
  `(NativeTensor, numpy.ndarray)`, or raises `StopIteration` at the end of
  the epoch. It raises `RuntimeError` if it is superseded, closed, or
  re-entered while its own transaction is in flight.
- `close()` **rolls back any in-flight transaction** — restoring the exact
  pre-delivery epoch and cursor and closing the undelivered batch —
  releases the active-iteration count, and detaches from the loader.
  Idempotent, never refused, and it **never** touches a delivered batch.
- `__enter__` / `__exit__`.

### 3.7 What must never be exported

`_native_permutation` and every name in it; `_NativeBatchIterator`;
`_deliver_batch` and every other transaction primitive of §3.2; any sampler
advance; any function returning raw random bits; any collate hook; any
worker, prefetch, or queue object; any global or default dataset, sampler,
or loader. Nothing is added to the **stable** root package (§18).

**`_deliver_batch` is a test seam, not an extension point**, and the
distinction is load-bearing: it accepts no user-supplied callable, is
reachable only by patching a private module attribute, and exposes no
public callback. §4.1 of `CLAUDE.md`'s "no production hook" rule is
respected because nothing here is compiled into or exported from the native
runtime at all — it is one private Python function.

---

## 4. Dataset input contract

### 4.1 `numpy.ndarray` exactly, for both arguments

`features` and `targets` must each be an object whose `type(...) is
numpy.ndarray`. Anything else is a `TypeError`.

This is **stricter than `NativeTensor.from_array`**, which accepts any
array-like, and the difference is deliberate. A single tensor is defined by
the values handed in; a *dataset* is defined by its sample count, its
per-sample shape, and its dtype, and all three must be answerable **before
anything is copied**. A nested list makes rank, raggedness, and element
type late discoveries inside a conversion. A caller who has lists writes
`numpy.asarray(...)` on their own side, one explicit line, where the
resulting shape is visible to them.

**Subclasses are rejected**, by `type(...) is` rather than `isinstance`.
A masked array is an `ndarray` subclass whose mask a snapshot would
silently discard, and a subclass that overrides indexing could make the
gather mean something other than a gather. This is the same exact-type
discipline `_validated_metadata` and `_validate_uint64` already use.

### 4.2 Feature dtype, rank, and shape

- **dtype**: `numpy.issubdtype(features.dtype, numpy.floating)` must hold —
  `float16`, `float32`, `float64`, and `longdouble`, in any byte order.
  Everything else is a `TypeError`: `bool_`, every integer type, complex,
  object, structured/void, string, bytes, datetime, and timedelta.
  - **Integer feature arrays are rejected**, and this is a decision rather
    than an oversight. Converting `int64` to `float32` silently loses
    exactness above `2**24`, and an integer feature column is as often
    categorical as numeric. A caller who means "these are real values"
    writes `.astype(numpy.float32)` and can see the conversion; a caller
    who meant categories is told rather than quietly given floats.
  - **`bool_` is rejected** for `_prepare_class_targets`' reason: `True` is
    not `1.0` here, and nothing is reinterpreted.
- **rank**: `features.ndim >= 1`. Axis 0 is the sample axis. A 0-d array
  has no sample axis and is a `ValueError`.
  - `ndim == 1` means **scalar feature samples**; `feature_shape` is `()`
    and a batch of `B` has shape `(B,)`.
- **shape**: `features.shape[0] >= 1` and **every** trailing dimension
  `>= 1`. A zero anywhere is a `ValueError`, because the native runtime
  cannot represent a zero-size dimension (§2.3) and a dataset that can
  never produce a batch is a construction error, not a runtime surprise.
- **values**: no constraint. `NaN`, `±inf`, subnormals, and signed zeros
  are ordinary float values; the dataset has no opinion and adds no check.

### 4.3 Target dtype, rank, and shape

- **dtype**: `numpy.issubdtype(targets.dtype, numpy.integer)` and **not**
  `numpy.bool_`. Any byte order. Everything else is a `TypeError`. This is
  §2.4's rule, at the same strictness and with the same reasoning.
- **rank**: `targets.ndim == 1` exactly; otherwise `ValueError` naming the
  shape.
- **length**: `targets.shape[0] == features.shape[0]` exactly; otherwise
  `ValueError` naming both counts.
- **values**: every value must be representable as `int64` (a `uint64`
  above `INT64_MAX` is a `ValueError` naming the index) and must be
  `>= 0` (a `ValueError` naming the index and the value).

### 4.4 The dataset is not a second authority on the class count

There is **no `num_classes` argument** and no upper-bound check on target
values.

The number of classes is a property of the *model* — of the logits' second
dimension — and `cross_entropy` already validates `0 <= value <
num_classes` on every call. A dataset that also held a class count could
disagree with the model, and the runtime would then have two authorities
for one fact. The `>= 0` check is kept because it is a **strict subset** of
what `cross_entropy` enforces at every possible `num_classes`, so it can
never disagree with it — it only moves an inevitable rejection earlier.

### 4.5 Layout, byte order, writeability, and aliasing of the inputs

All accepted, and none survives into the dataset:

- **Non-contiguous, strided, transposed, negatively strided, and sliced**
  inputs are accepted. The snapshot (§5.1) is a fresh C-contiguous array
  regardless of the input's layout.
- **Non-native byte order** is accepted. The conversion target is the
  native-order `numpy` dtype from the private `_DTYPE_NUMPY` table, so the
  snapshot construction byte-swaps once, at the boundary. This is a host
  normalization, not a reinterpretation, and it is why the snapshot's byte
  order is always native and provable.
- **Read-only** inputs are accepted; the snapshot is written, not the
  input.
- **The two inputs may be views of one buffer**, or the same object. Two
  independent snapshots are taken, so no aliasing survives.

### 4.6 Empty datasets are rejected

`features.shape[0] == 0` is a `ValueError` naming the count.

The native runtime rejects zero-element storage and zero-size dimensions
(§2.3), so an empty dataset could never produce a feature batch. Accepting
one would push a guaranteed failure from construction — where the caller
can see it — into the middle of a training loop, and would force every
downstream rule (`batches_per_epoch`, cursor range, the iterator's
countdown) to carry a degenerate branch that no correct program reaches.

### 4.7 The native feature dtype is chosen, never inferred

`dtype` accepts exactly `None`, `"float64"`, and `"float32"`, keyword-only,
validated by the **one** shared route `experimental._native_dtype
.normalize_module_dtype` — the same validator the six I7 state-owning
constructors use. `None` means `"float64"`. There are no aliases: not
`"f4"`, not `"single"`, not `"Float32"`, not `numpy.float32` the type, not
a `numpy.dtype` object. A non-string raises `TypeError`; any other string
raises `ValueError` naming it.

**The NumPy feature dtype never selects the native dtype.** A `float32`
NumPy array with `dtype` omitted produces a **float64** dataset and
**float64** feature batches. This is the Phase-I rule (§2.3) applied
without exception, and it is asserted in both directions by test.

There is **no `device` argument** (§19.7).

### 4.8 Validation order at construction

Ordered so that the cheapest and most fundamental rejections come first,
nothing is allocated before everything is checked, and a caller who got two
things wrong is told about the more basic one:

1. `dtype` — normalized first, so an unsupported request is refused before
   NumPy is asked to do any work at all (`from_array`'s own ordering).
2. `features` exact type → rank → dtype kind → shape (`samples >= 1`, every
   trailing dimension `>= 1`).
3. `targets` exact type → rank → dtype kind.
4. `targets` length against `samples`.
5. Target values: int64 representability, then non-negativity, each naming
   the first offending index.
6. Only then: the feature snapshot, the target snapshot, the fingerprint
   (§17.2 gives the failure behavior at each).

---

## 5. Dataset ownership and snapshot semantics

### 5.1 Copied snapshots, taken once, at construction

The dataset takes **exactly two** owned host arrays at construction and
never takes another:

```
_features = numpy.array(features, dtype=_DTYPE_NUMPY[dtype],
                        order="C", copy=True)
_targets  = numpy.array(targets,  dtype=numpy.int64,
                        order="C", copy=True)
```

Both are freshly allocated, C-contiguous, native byte order, owned by the
dataset, and reachable from nowhere else. The feature snapshot is at the
**chosen native dtype's NumPy counterpart**, which is the single most
important consequence of this design: the host→native conversion that
`from_array` performs per call happens here **once**, so every later batch
gather is a same-dtype row selection and every host→native transfer copies
matching bits.

**`copy=True` is load-bearing and is not a stylistic choice.** The obvious
spelling, `numpy.ascontiguousarray(values, dtype=...)`, returns the input
*unchanged* when it is already contiguous, native-order, and of the target
dtype — which would alias caller memory in exactly the common case, and
§5.2 forbids that. The rule is "snapshot", not "usually a snapshot", so the
copy is unconditional.

Borrowing was rejected. A borrowed feature array that the caller later
mutates would change the meaning of every future batch, would invalidate
the fingerprint silently, and would make an exact resume proof unprovable —
the batches would depend on host memory the state does not describe.
Deterministic reproduction is the phase's entire subject; hidden aliasing
is its exact negation.

### 5.2 Mutating the caller's arrays afterwards changes nothing

After construction the caller may mutate, resize, or delete `features` and
`targets` freely. The dataset's snapshots, its `feature_shape`, its
`fingerprint`, and every batch it will ever produce are unaffected. This is
asserted in both directions by test at J1.

### 5.3 Indexing returns copies, never views

`feature_batch(indices)` returns an **owning** `NativeTensor` whose native
storage aliases nothing — neither the snapshot (which is host memory) nor
any other tensor. `target_batch(indices)` returns a fresh host array that
shares no memory with the target snapshot, and is marked read-only so the
returned object cannot be edited in place either.

Mutating a returned batch therefore cannot reach the dataset, and closing
one cannot affect another. Two calls with the same indices produce two
independent objects with identical contents.

### 5.4 The dataset owns no native storage

Between calls the dataset holds two NumPy arrays and nothing else. It
allocates native storage only **inside** `feature_batch`, and every byte of
it belongs to the returned tensor, which the caller owns. So constructing,
holding, inspecting, and discarding a dataset leaves the native
live-storage count exactly where it was — the property `NativeGenerator`
already has, and one J7 asserts directly.

### 5.5 Close, and use after close

`close()` drops both snapshot references and sets `closed`. Idempotent;
returns `None`; never raises.

After close:

- `samples`, `feature_shape`, `dtype`, `device`, `fingerprint`, `closed`,
  `__len__`, `identity()`, and `__repr__` **still work**. They are
  metadata, they are what a state comparison needs, and the precedent is
  `NativeStorage.dtype`, which is explicitly "readable after close".
- `feature_batch` and `target_batch` raise `RuntimeError` naming the closed
  dataset, allocate nothing, and change nothing.

A dataset closed while a sampler or loader still references it is a
supported and deterministic situation: the sampler keeps working entirely
(it needs only `samples`, which survives), planning and state remain
correct, and the **loader** fails at materialization with that
`RuntimeError`, having advanced nothing (§9.6).

### 5.6 Dataset state is not serializable, and dataset contents are never in loader state

`NativeTensorDataset` has **no `state_dict()`** and no `load_state_dict()`.
A dataset is input, not training state: it is reconstructed by the caller
from the same source that produced it originally, exactly as a model's
*architecture* is reconstructed by the caller before its parameters are
loaded.

What crosses into sampler state is only the four **identity** fields of
§6.4 — a count, a shape, a dtype name, and a digest. No feature value, no
target value, no array, and no byte string ever enters a state dict or a
checkpoint through Phase J.

---

## 6. Dataset identity and compatibility

### 6.1 The question this must answer

A restored sampler has to decide whether the position it is about to adopt
means anything for the dataset actually in front of it. Cursor 7 of epoch 3
under a permutation of 800 samples describes specific rows; applied to a
different 800 rows it silently trains on the wrong data, and applied to 900
rows it may not even be in range.

### 6.2 A deterministic content fingerprint is selected

Compatibility is decided by **four** fields, checked in this order:
`samples`, `feature_shape`, `feature_dtype`, then `fingerprint`.

The first three are cheap, and each names a specific, understandable
mismatch. The fingerprint is the one that catches "the same shape, the same
dtype, different data" — a different train/validation split, a re-shuffled
source file, a re-generated synthetic set, a preprocessing change. Without
it, the compatibility guarantee would be *structural only*, which is
precisely the case a resume proof must exclude.

**Rejected alternatives**, and why:

- **A per-object random token.** It cannot survive a process boundary,
  which is the only boundary that matters for a checkpoint.
- **Python `id()`, or anything derived from it.** Process-local identity is
  explicitly forbidden, and ids are recycled.
- **A caller-supplied identifier.** It moves the entire guarantee to the
  caller's discipline and is silently wrong whenever they forget to change
  it. It is not added as an optional extra argument either: an optional
  weaker authority beside a strong one is a second authority.
- **Structural fields only.** Cheaper, and it accepts the exact case the
  phase exists to reject.

### 6.3 The exact fingerprint definition

Computed **once, eagerly, at construction**, immediately after both
snapshots exist, by **SHA-256** from `hashlib` (Python standard library —
no new dependency), reported as **64 lowercase hexadecimal characters**.

The hashed byte stream is exactly this concatenation, with no separators
beyond the ones written:

| # | Bytes |
|---|---|
| 1 | `b"tensorforge.native_dataset\x00"` — the domain tag |
| 2 | `b"fingerprint-v1\x00"` — the fingerprint schema version |
| 3 | the native feature dtype name in UTF-8 (`b"float64"` / `b"float32"`), then `b"\x00"` |
| 4 | the feature snapshot's rank, as one 8-byte little-endian unsigned integer |
| 5 | each feature-snapshot dimension in order, each as one 8-byte little-endian unsigned integer |
| 6 | the feature snapshot's elements, C order, **little-endian** |
| 7 | `b"targets\x00"` |
| 8 | the target count, as one 8-byte little-endian unsigned integer |
| 9 | the target elements, C order, **little-endian `int64`** |

**Endian normalization is explicit and mandatory.** The snapshots are in
*native* order, so on a big-endian machine their raw bytes differ from a
little-endian machine's for identical values. Steps 6 and 9 therefore
encode through `snapshot.astype(snapshot.dtype.newbyteorder("<"),
copy=False)`, which is a no-op returning the same object on a little-endian
host and a byte-swapped copy on a big-endian one. Steps 4, 5, and 8 use
`int.to_bytes(8, "little")`. **No Python `hash()`, no `str()` of an array,
no `repr`, no `pickle`, no `tobytes()` of a native-order buffer, and no
floating-point arithmetic** enters the digest.

Bytes are fed to the hasher in chunks over the contiguous buffer rather
than through one whole-array `tobytes()`, so a large dataset does not
double its peak host memory. The chunk size is an implementation detail and
provably cannot change the digest.

The rank and the dimensions are hashed **before** the elements so that two
datasets with the same elements in different shapes — `(6, 2)` and `(4, 3)`
— cannot collide. The dtype name is hashed even though a float32 and a
float64 snapshot of the same values already differ in bytes, because
stating it makes the digest self-describing rather than accidentally
distinct.

### 6.4 What travels in state

```json
{"samples": 800,
 "feature_shape": [1, 6, 6],
 "feature_dtype": "float32",
 "fingerprint": "3f1c...e90a"}
```

`samples` is an `int`, `feature_shape` a list of `int`s (the **per-sample**
shape — empty for scalar samples), `feature_dtype` one of the two dtype
names, `fingerprint` a 64-character lowercase hex `str`. All four are
JSON-native and pass `_validated_metadata` unchanged. No payload, no bytes,
no array, no length that would grow with the data.

### 6.5 Cost, and the collision assumption

The digest costs one SHA-256 pass over data that is being copied anyway, at
construction, once. It is paid whether or not the dataset is ever
checkpointed, which is a deliberate trade: an eager digest makes
`state_dict()` allocation-free and non-failing, and makes the cost visible
at the line where the dataset is built rather than at an arbitrary later
one. Lazy computation on first `state_dict()` was rejected for exactly
that reason.

The fingerprint detects **accidents** — the wrong dataset, mutated source
data, a different split, a changed preprocessing step. It is **not** an
adversarial integrity check, is not claimed to be one, and defends against
no deliberately constructed collision. That is stated rather than implied,
and no document may describe it as a security property.

If hashing raises (`MemoryError` on a very large dataset), construction
fails and both snapshots are released before the exception leaves the
constructor (§17.2).

---

## 7. Sampler architecture

### 7.1 The sampler owns batch size and drop-last, and emits batch groups

`batch_size` and `drop_last` live on the **sampler**, not the loader, and
the sampler emits complete **batch-index groups**, not individual indices.

The reason is the cursor. Exact mid-epoch resume requires a position that
is meaningful across a save and a restore, and the only position whose
meaning is stable is a **batch boundary**. If the loader owned the batch
size, the sampler's cursor would count something the sampler could not turn
into a batch, and two objects would jointly own one indivisible fact.
Emitting single indices has the same defect from the other side: a loader
that grouped them would decide where batches begin, so the sampler's state
would not determine the batches.

One owner, one cursor, one meaning. It is also what makes J2 a complete,
independently testable milestone with no `NativeTensor` allocation
anywhere in it.

### 7.2 Configuration validation

- `dataset` — must be a `NativeTensorDataset` (`TypeError` otherwise). A
  **closed** dataset is accepted: `samples` survives close (§5.5), the
  sampler needs nothing else, and refusing here would create a lifecycle
  rule with no purpose.
- `batch_size` — keyword-only, **required**, exact `int` (`bool` rejected
  as a `TypeError`), `>= 1` (`ValueError`). No upper bound is imposed by
  the sampler: `batch_size > samples` is legal with `drop_last=False` and
  simply gives one short batch per epoch. The platform limit is reached
  when the *native batch* is allocated, where `MemoryError` is the honest
  answer (§17.4), not at the plan.
- `shuffle` — keyword-only, **exact `bool`**; `0`, `1`, `""`, and `None`
  are each a `TypeError` (the `retain_graph` precedent).
- `seed` — keyword-only, exact `int`, `0 <= seed <= 2**64 - 1`, `bool`
  rejected. The identical domain and the identical validation
  `NativeGenerator` uses, deliberately: the phase does not invent a second
  seed contract.
- `drop_last` — keyword-only, exact `bool`.
- `epoch` and `cursor` are **not** constructor arguments. A new sampler
  starts at epoch 0, cursor 0; any other position arrives through
  `load_state_dict`, which is the one audited path into a position.

### 7.3 What `epoch` means

`epoch` is the **active** epoch — the one whose permutation is currently
being consumed — not the next one. A fresh sampler reads `epoch == 0`, and
its first batch comes from epoch 0's order.

### 7.4 What `cursor` means, and the canonical epoch boundary

`cursor` is **the number of batches already delivered in the current
epoch**. Batch `k` of an epoch covers permutation positions
`[k * batch_size, min((k + 1) * batch_size, samples))`.

The committed transition after a batch is successfully delivered is:

```
cursor += 1
if cursor == batches_per_epoch:
    epoch += 1
    cursor  = 0
```

**The epoch boundary is canonicalized immediately**, at the moment the last
batch of an epoch is delivered, rather than lazily on the next request.
Two consequences make this the right choice, and both are testable:

1. **One representation per position.** Two runs that have consumed the
   same number of batches always have byte-identical state dicts. A lazy
   rule would give "end of epoch 3" and "start of epoch 4" two spellings
   for one position, and every equality proof in J4, J5, J6, and J7 would
   have to normalize before comparing.
2. **The cursor range is a half-open interval**, `0 <= cursor <
   batches_per_epoch`, with no special case at the top. Validation is one
   comparison, and a cursor equal to the batch count is unambiguously
   invalid rather than ambiguously terminal.

`epoch` is bounded by the same unsigned 64-bit domain as `seed`. An advance
that would take it past `2**64 - 1` raises `RuntimeError` and moves
nothing, exactly as `NativeGenerator` refuses at an exhausted counter. This
is unreachable in practice and is specified so that it is not undefined.

### 7.5 `batches_per_epoch`, and why a zero-batch epoch cannot exist

```
drop_last = False -> batches_per_epoch = ceil(samples / batch_size)
drop_last = True  -> batches_per_epoch = samples // batch_size
```

With `drop_last=True` and `batch_size > samples` this would be **zero**,
and an epoch with no batches breaks the §7.4 transition: no batch is ever
delivered, so the epoch never advances, so repeated iteration spins on a
position that can never move.

**So the configuration is rejected at construction.** With
`drop_last=True`, `batch_size` must be `<= samples`; otherwise `ValueError`
naming both values and saying that dropping the last partial batch would
leave no batches at all. `batches_per_epoch >= 1` is then an invariant, the
§7.4 rule is total, and no code anywhere carries a zero-batch branch.

This also removes the question from `load_state_dict`: a state whose
`batch_size` and `drop_last` would produce zero batches is rejected by the
same rule, before any mutation (§12.4).

### 7.6 Behavior at every boundary

| Situation | Behavior |
|---|---|
| `samples == 1`, `batch_size == 1` | One batch per epoch; every epoch's permutation is `(0,)`; `drop_last` is irrelevant. |
| `batch_size > samples`, `drop_last=False` | One short batch per epoch of `samples` rows. |
| `batch_size > samples`, `drop_last=True` | Rejected at construction (§7.5). |
| `samples % batch_size == 0` | `drop_last` changes nothing; both give `samples // batch_size` full batches. |
| `samples % batch_size != 0`, `drop_last=False` | Final batch is short, with `samples % batch_size` rows. |
| `samples % batch_size != 0`, `drop_last=True` | The tail of the permutation is **not delivered this epoch**. A different epoch has a different permutation, so a given sample is not systematically excluded. |
| `cursor == 0` | The whole epoch is ahead; `remaining == batches_per_epoch`. |
| `cursor == batches_per_epoch - 1` | One batch left; delivering it canonicalizes to `(epoch + 1, 0)`. |
| At an epoch boundary | Exactly one state: `(epoch + 1, 0)` (§7.4). |
| Repeated iteration | Each `for` consumes one epoch's remaining batches, then the next `for` consumes the whole following epoch. |
| Abandoned iteration | Consumes nothing beyond the batches actually delivered (§7.7). |

### 7.7 The sampler has no consumable stream

This is the structural property the rest of the phase rests on, and it is
worth stating on its own: **a permutation is a pure function of `(seed,
epoch, length)`.** There is no reservation, no counter of draws, no
partially consumed sequence, and nothing to roll back.

Therefore, **by construction rather than by cleanup**:

- an abandoned iterator consumes nothing;
- a rejected `load_state_dict` consumes nothing;
- a failed batch materialization consumes nothing;
- `epoch_permutation()`, `plan()`, and `next_batch_indices()` consume
  nothing and may be called any number of times;
- two samplers with equal `(seed, epoch, cursor, batch_size, drop_last,
  shuffle)` over datasets of equal length produce identical remaining
  batch-index sequences forever.

The only state that ever moves is the `(seed, shuffle, batch_size,
drop_last, epoch, cursor)` tuple, and only through a **successfully
delivered** batch or a validated state load. A delivery that fails is
rolled back completely (§9.4, Phase 5), so it moves nothing either.

### 7.8 The permutation cache is not state

Recomputing a permutation for every batch would be `O(samples)` work per
batch. The sampler therefore keeps one private cached permutation keyed on
`(seed, epoch, samples)`, rebuilt whenever the key changes.

It is **not state and never appears anywhere**: not in `state_dict()`, not
in a checkpoint, not in `__repr__`, and not in any comparison. Dropping it
at any moment changes no observable behavior, because the value is a pure
function of the key. It is a Python tuple of `int`s and involves no native
storage, so it is not the persistent native cache, pool, arena, or
workspace that the project's absent-by-design list forbids.

---

## 8. The deterministic permutation

### 8.1 Requirements

Same permutation on every supported platform and Python build. No Python
global `random`, no NumPy global RNG, no NumPy `Generator`, no dictionary
or set ordering, no wall-clock or entropy seeding, no process-randomized
hashing, no ambient or hidden mutable global state, no platform-dependent
integer width, no platform-dependent overflow, no floating-point
randomness, no modulo bias, no new global or default generator, and **no
new RNG algorithm**.

### 8.2 The algorithm is `tensorforge.splitmix64`, reused

Phase J introduces **no new RNG algorithm**. It reuses the locked
`tensorforge.splitmix64` finalizer and golden constant of §2.5 — the same
constants, the same shifts, the same multiplication order, the same
wrapping 64-bit arithmetic — under a **different key schedule**, because
this consumer wants permutation indices rather than a keep/drop mask.

Reusing the derivation while minting a distinct key schedule is the same
relationship the existing derivation already has internally: one finalizer,
two applications, different arguments at each level.

### 8.3 The sampler is deliberately not coupled to a live `NativeGenerator`

It is not a `NativeGenerator`, does not hold one, does not accept one, and
does not consume calls from one. Three independent reasons, any one of
which is sufficient:

1. **It would entangle the data order with Dropout's stream.** A
   `NativeGenerator`'s state is `(seed, calls)`, and its call index is
   consumed by stochastic forwards. If shuffling drew from the same
   counter, changing the batch size — which changes how many forwards run
   per epoch — would change every Dropout mask in the model. Two knobs
   would silently move one stream.
2. **A permutation is not indexed by a call.** It is indexed by an
   **epoch**, which is not a monotonic count of anything the generator
   knows about, and a restored sampler must be able to reproduce epoch 9's
   order without having consumed epochs 0 through 8.
3. **It is not reachable.** `NativeGenerator` exposes no bit derivation at
   all (§2.5), and no export returns raw random bits. Coupling would
   therefore require inventing a Python bit API or adding a C ABI export —
   a new public random surface, which §20 forbids.

The sampler holds a plain `seed` integer in the **same** unsigned 64-bit
domain, validated by the **same** rules, so nothing about the seed contract
is a second convention.

### 8.4 Domain separation

The two consumers of one derivation are separated by one additive constant:

```
SAMPLER_DOMAIN = 0x54465F53414D504C          # the ASCII bytes "TF_SAMPL"
```

Without it, `epoch_key(seed, e)` would be bit-identical to
`dropout_stream_key(seed, e)`, so a user who passed the same seed to a
sampler and a generator would drive both from the same 64-bit sequence.
That is a benign but real accident, and one constant removes it.

**It is not a cryptographic separation and is not claimed to be one.** A
caller who deliberately chooses `seed' = seed + SAMPLER_DOMAIN` can still
align the two streams; the constant prevents the accident, not the
construction.

### 8.5 Pseudocode — precise enough to implement directly

```
MASK   = 2**64 - 1
GOLDEN = 0x9E3779B97F4A7C15
SAMPLER_DOMAIN = 0x54465F53414D504C

def splitmix64_mix(x):                  # tf::splitmix64_mix, in Python
    x &= MASK
    x ^= x >> 30
    x  = (x * 0xBF58476D1CE4E5B9) & MASK
    x ^= x >> 27
    x  = (x * 0x94D049BB133111EB) & MASK
    x ^= x >> 31
    return x

def epoch_key(seed, epoch):             # one full finalizer per epoch
    return splitmix64_mix((seed + SAMPLER_DOMAIN + GOLDEN * (epoch + 1)) & MASK)

def draw_bits(key, draw_index):         # one full finalizer per draw
    return splitmix64_mix((key + GOLDEN * (draw_index + 1)) & MASK)

def bounded(key, draw_index, bound):    # unbiased integer in [0, bound)
    # bound >= 1. limit is the largest multiple of bound that fits in 2**64,
    # so accepted values cover each residue exactly the same number of times.
    limit = (1 << 64) - ((1 << 64) % bound)
    while True:
        bits = draw_bits(key, draw_index)
        draw_index += 1
        if bits < limit:
            return bits % bound, draw_index

def permutation(seed, epoch, length):   # length >= 1
    order = list(range(length))
    if length < 2:
        return tuple(order)             # zero draws
    key   = epoch_key(seed, epoch)
    draws = 0
    for i in range(length - 1, 0, -1):  # Fisher-Yates, DOWNWARD
        j, draws = bounded(key, draws, i + 1)
        order[i], order[j] = order[j], order[i]
    return tuple(order)

def sample_order(seed, epoch, length, shuffle):
    return permutation(seed, epoch, length) if shuffle else tuple(range(length))

def batches_per_epoch(length, batch_size, drop_last):
    return length // batch_size if drop_last else -(-length // batch_size)

def batch_plan(seed, epoch, length, batch_size, drop_last, shuffle):
    order = sample_order(seed, epoch, length, shuffle)
    count = batches_per_epoch(length, batch_size, drop_last)
    return tuple(order[k * batch_size:(k + 1) * batch_size]
                 for k in range(count))
```

Every value is a Python `int`, which is arbitrary precision, so `& MASK` —
never a C type, never NumPy — is what makes the width exactly 64 bits on
every platform. There is no floating-point arithmetic anywhere in the
derivation: the `dropout_uniform` step is a Dropout-only conversion and has
no counterpart here.

### 8.6 The decisions inside that pseudocode

- **Bounded integers by rejection, not modulo.** `bits % bound` alone is
  biased whenever `bound` does not divide `2**64`. Rejecting `bits >= limit`
  removes the bias exactly rather than approximately. For any `bound <=
  2**32` the rejection probability is below `2**-32`, so the loop is
  essentially always one iteration — but correctness does not rest on that.
  A multiply-shift alternative was rejected: it would need 128-bit
  intermediate reasoning to state precisely and would be a second bounded-
  integer convention beside a simple one that is already exact.
- **Fisher–Yates, downward** (`i` from `length - 1` to `1`, `j` uniform in
  `[0, i]`, swap `order[i]` and `order[j]`). Exactly `length - 1` draws
  before rejections, and every one of the `length!` permutations is
  reachable. The upward variant is equally correct and is not used; the
  direction is part of the specification because the two produce different
  permutations from the same bits.
- **The permutation is built from the identity** `[0, 1, ..., length - 1]`
  in that order, then swapped in place.
- **`draw_index` counts every drawn value, including rejected ones.** A
  rejection therefore shifts all later draws by one, which is what keeps the
  result a pure function of `(seed, epoch, length)` regardless of where a
  rejection lands.
- **`length` is not mixed into the key.** It enters through the bounds, so
  two different lengths already produce different permutations from the
  same key; and the dataset identity check (§6) is what actually prevents a
  state from being applied to a differently sized dataset. Keeping the key
  schedule the same shape as the existing one is worth more than a
  redundant mixing step.
- **Sequential order is the identity permutation** at every seed and epoch,
  and draws nothing. `shuffle=False` is not "a shuffle with a fixed seed";
  it is a different, cheaper branch, and the reference vectors state it.

### 8.7 Draw accounting

| Situation | Draws consumed |
|---|---|
| `length == 1` | 0 |
| `length >= 2`, shuffled | `length - 1`, plus one per rejection |
| Sequential (`shuffle=False`) | 0, at any length |
| A permutation computed twice | The same, because it is a pure function |
| An abandoned iterator | 0 beyond those the delivered batches implied |
| A rejected `load_state_dict` | 0 |
| A failed materialization | 0 |

"Draws consumed" is bookkeeping inside one pure call, not a stream
position. Nothing in the sampler accumulates across calls (§7.7).

### 8.8 The Python implementation must be proved equal to the C++ one

One algorithm with two implementations can drift, so the equality is a
**gate**, not an assumption. J2 must prove it in two independent ways, and
both are implementable today with no new export:

1. **Committed known-answer vectors** for `splitmix64_mix`, `epoch_key`,
   and each reference permutation in §8.9, asserted as exact integers.
2. **A live cross-implementation check against the shipped C++ kernel.**
   The Python `epoch_key`/`draw_bits` pair computes, for arguments
   `(seed, epoch, index)`, exactly what
   `tf::dropout_stream_key`/`tf::dropout_element_bits` compute for
   `(seed, call_index, element_index)` — the schedules differ only by
   `SAMPLER_DOMAIN`, which the check omits. So for a Dropout forward at a
   known `(seed, call_index, p)` the kept/dropped pattern is
   `not (dropout_uniform(bits) < p)` elementwise, and the Python derivation
   must predict it exactly.

   **This was verified at J0 against the built library** before being
   written into the contract: over 48 combinations of
   `seed ∈ {0, 7, 0xFEDCBA9876543210, 2**64 - 1}`,
   `call_index ∈ {0, 1, 5}`, and `p ∈ {0.1, 0.25, 0.5, 0.9}`, at 4096
   elements each, the Python prediction matched the kernel's output on
   every element. J2 ships that check as a test rather than a claim.

   Each element is one bit of evidence about the mixing function, so a few
   thousand elements at several `p` values pin the implementation far more
   tightly than any vector list, and it re-runs on every platform the suite
   runs on — which is exactly where a two-implementation drift would appear.

### 8.9 Fixed reference vectors

All values below were computed at J0 and are the specification. A future
change to any of them is a change to the algorithm and requires a new
`(algorithm, algorithm_version)` pair, exactly as the Dropout vectors do.

**Empty and degenerate lengths.** `length == 0` has **no vector**: an
empty dataset is rejected at construction (§4.6), so a sampler over one
cannot exist. That rejection is itself the specified behavior for the
zero-length case.

**`splitmix64_mix` known answers** (also the C++ function's answers):

| input | output |
|---|---|
| `0x0000000000000000` | `0x0000000000000000` |
| `0x0000000000000001` | `0x5692161D100B05E5` |
| `0x9E3779B97F4A7C15` | `0xE220A8397B1DCDAF` |
| `0xFFFFFFFFFFFFFFFF` | `0xB4D055FCF2CBBD7B` |

**`epoch_key(seed, epoch)`**:

| seed | epoch | key |
|---|---|---|
| `0` | 0 | `0x66F32B8D4EDCDEF0` |
| `0` | 1 | `0xE205B4E09628466F` |
| `0` | 7 | `0x4487E9B41C8E68DF` |
| `7` | 0 | `0xE9D3E585001C46A4` |
| `7` | 1 | `0xED6B991DDB3B74AF` |
| `7` | 7 | `0xEDF383949681C2A9` |
| `0xFEDCBA9876543210` | 0 | `0x5EAC5CE0C7928FA5` |
| `0xFEDCBA9876543210` | 1 | `0x418ADC598C6E56E9` |
| `0xFEDCBA9876543210` | 7 | `0x4D3B0EE9DD189AB2` |
| `0xFFFFFFFFFFFFFFFF` | 0 | `0xA20C5EE669FCA87A` |
| `0xFFFFFFFFFFFFFFFF` | 1 | `0x097D6A1D7039DBCA` |
| `0xFFFFFFFFFFFFFFFF` | 7 | `0xBEF20F97FF2FBF91` |

**Complete shuffled permutations.** `0xFEDCBA9876543210` is the nontrivial
large seed; `0xFFFFFFFFFFFFFFFF` is the accepted upper bound of the seed
domain.

| length | seed | epoch | permutation |
|---|---|---|---|
| 1 | `0` | 0 | `[0]` |
| 1 | `0` | 7 | `[0]` |
| 1 | `7` | 0 | `[0]` |
| 1 | `7` | 7 | `[0]` |
| 1 | `0xFEDCBA9876543210` | 0 | `[0]` |
| 1 | `0xFEDCBA9876543210` | 7 | `[0]` |
| 1 | `0xFFFFFFFFFFFFFFFF` | 0 | `[0]` |
| 1 | `0xFFFFFFFFFFFFFFFF` | 7 | `[0]` |
| 2 | `0` | 0 | `[0, 1]` |
| 2 | `0` | 7 | `[1, 0]` |
| 2 | `7` | 0 | `[1, 0]` |
| 2 | `7` | 7 | `[1, 0]` |
| 2 | `0xFEDCBA9876543210` | 0 | `[0, 1]` |
| 2 | `0xFEDCBA9876543210` | 7 | `[1, 0]` |
| 2 | `0xFFFFFFFFFFFFFFFF` | 0 | `[0, 1]` |
| 2 | `0xFFFFFFFFFFFFFFFF` | 7 | `[0, 1]` |
| 5 | `0` | 0 | `[1, 0, 3, 4, 2]` |
| 5 | `0` | 7 | `[2, 1, 3, 0, 4]` |
| 5 | `7` | 0 | `[1, 2, 4, 3, 0]` |
| 5 | `7` | 7 | `[3, 0, 2, 1, 4]` |
| 5 | `0xFEDCBA9876543210` | 0 | `[2, 1, 0, 4, 3]` |
| 5 | `0xFEDCBA9876543210` | 7 | `[4, 3, 2, 1, 0]` |
| 5 | `0xFFFFFFFFFFFFFFFF` | 0 | `[4, 0, 3, 2, 1]` |
| 5 | `0xFFFFFFFFFFFFFFFF` | 7 | `[3, 2, 0, 4, 1]` |
| 8 | `0` | 0 | `[3, 6, 7, 0, 2, 5, 4, 1]` |
| 8 | `0` | 7 | `[4, 2, 0, 1, 7, 3, 5, 6]` |
| 8 | `7` | 0 | `[7, 5, 4, 0, 1, 3, 6, 2]` |
| 8 | `7` | 7 | `[1, 4, 7, 0, 3, 5, 6, 2]` |
| 8 | `0xFEDCBA9876543210` | 0 | `[1, 0, 5, 2, 6, 7, 4, 3]` |
| 8 | `0xFEDCBA9876543210` | 7 | `[2, 1, 6, 7, 4, 5, 3, 0]` |
| 8 | `0xFFFFFFFFFFFFFFFF` | 0 | `[0, 3, 1, 4, 6, 5, 2, 7]` |
| 8 | `0xFFFFFFFFFFFFFFFF` | 7 | `[6, 7, 1, 2, 5, 0, 4, 3]` |

Several of those rows are the identity permutation — every length-1
row trivially, and four of the length-2 rows. That is not a defect and
must not be "fixed": the identity is one of the `length!` outcomes and
excluding it would bias the sampler. The rows are kept precisely so that a
future implementation cannot quietly special-case it away.

**Sequential order** — `shuffle=False`, at **every** seed and **every**
epoch:

| length | order |
|---|---|
| 1 | `[0]` |
| 2 | `[0, 1]` |
| 5 | `[0, 1, 2, 3, 4]` |
| 8 | `[0, 1, 2, 3, 4, 5, 6, 7]` |

**Complete batch plans.** These fix the interaction of the permutation with
`batch_size` and `drop_last`.

`length = 8`, `seed = 7`, `epoch = 0`, shuffled — permutation
`[7, 5, 4, 0, 1, 3, 6, 2]`:

| `batch_size` | `drop_last` | plan |
|---|---|---|
| 3 | `False` | `[[7, 5, 4], [0, 1, 3], [6, 2]]` |
| 3 | `True` | `[[7, 5, 4], [0, 1, 3]]` |
| 4 | `False` | `[[7, 5, 4, 0], [1, 3, 6, 2]]` |
| 4 | `True` | `[[7, 5, 4, 0], [1, 3, 6, 2]]` |

`length = 5`, `seed = 0`, `epoch = 0`, shuffled — permutation
`[1, 0, 3, 4, 2]`:

| `batch_size` | `drop_last` | plan |
|---|---|---|
| 2 | `False` | `[[1, 0], [3, 4], [2]]` |
| 2 | `True` | `[[1, 0], [3, 4]]` |
| 5 | `False` | `[[1, 0, 3, 4, 2]]` |
| 5 | `True` | `[[1, 0, 3, 4, 2]]` |

`length = 5`, sequential, `batch_size = 2`:

| `drop_last` | plan |
|---|---|
| `False` | `[[0, 1], [2, 3], [4]]` |
| `True` | `[[0, 1], [2, 3]]` |

**Draw counts.** At `length ∈ {1, 2, 5, 8, 100, 1000}`, across all four
reference seeds and epochs 0–3, the observed draw count was exactly
`max(length - 1, 0)` in every case: **no rejection occurred anywhere in the
reference set**. The rejection branch is nonetheless part of the
specification and J2 must cover it by exercising `bounded` directly at a
`bound` chosen so that `limit` is small enough to force one.

---

## 9. Iterator and state-machine semantics

### 9.1 The loader is not itself an iterator

`iter(loader)` returns a **separate** `_NativeBatchIterator` object. The
loader is a long-lived configured surface that may be iterated many times;
an iterator is a short-lived one-epoch traversal. Merging them would make
`iter(loader)` return the loader, so two nested loops would silently share
one position.

### 9.2 One valid iterator at a time, and a new one supersedes the old

The loader holds a reference to at most one **current** iterator.
`iter(loader)` returns a fresh iterator and **supersedes** any previous
one, whose next `__next__` then raises `RuntimeError` naming the
supersession rather than yielding.

- The common pattern works: `for ... in loader: ... break`, then another
  `for`, starts cleanly from the committed position.
- Nested and concurrent iteration fails **loudly**, on the outer loop's
  next step, rather than interleaving two traversals over one cursor.
- Nothing depends on garbage collection: supersession is an explicit
  assignment made by `iter()`.

**With one exception, which the transaction of §9.4 requires:
`iter(loader)` is refused with `RuntimeError` while a claim or a
pending-delivery record is in flight**, and supersedes nothing. Superseding
mid-transaction would detach the iterator that owns the undelivered batch
and holds the only reference able to roll it back — stranding both a
position and a tensor, which is precisely the failure this section exists
to prevent. A transaction exists only inside `__next__`, so this is
reachable only from a reentrant caller, and refusing it there is the same
deterministic answer §9.5 gives every other reentrant operation.

**Rejected alternative:** refusing the second `iter()` in *all* cases until
the first is closed. It is marginally stricter about nesting and makes the
overwhelmingly common break-and-restart pattern require a manual `close()`
— and a user who forgets is blocked by an object they can no longer reach.
Supersession rejects the same misuse at the same place, one step later,
without stranding the loader, and the in-flight case above is carved out
narrowly rather than by widening the refusal to every call.

A superseded iterator's `close()` is still safe and still idempotent. It
cannot hold a pending batch — supersession is impossible while one exists —
so what it releases is its active-iteration count (§9.5).

### 9.3 One iterator is one epoch

An iterator captures, at construction, `to_yield = sampler.remaining`, and
counts it down. When it reaches zero it raises `StopIteration` and detaches
from the loader.

So a `for` loop consumes exactly the batches remaining in the current
epoch: a full epoch from a fresh position, and **the tail of an interrupted
epoch** when resuming mid-epoch — which is precisely the behavior an exact
mid-epoch resume needs. Because the epoch boundary is canonicalized
(§7.4), the next `for` then runs the whole of the following epoch.

The countdown is captured rather than re-read because the sampler's
`remaining` resets to a full epoch the moment the boundary is crossed; an
iterator re-reading it would never terminate. It is also why
`load_state_dict` is refused while an iteration is active (§9.5) — a load
would make a captured countdown describe a position that no longer exists.

### 9.4 The batch handoff is an explicit transaction

**No committed sampler position ever advances for a batch the caller did
not receive.** That is the contract, it holds at every failure position,
and it is what the rest of this section specifies.

A naive "materialize, advance the cursor, return" sequence cannot provide
it: an exception arriving between the advance and the return would leave a
position consumed and a batch unreachable, which would silently break both
the exact-resume proof (§14) and the checkpoint promise that a saved state
describes the *next* batch (§13.7). The project already refused that
trade once — `NativeGenerator`'s reservation is a four-phase claim →
construct → publish → deliver transaction precisely so that a failure
between publishing and delivering is *recoverable* rather than accepted.
Phase J adapts that discipline to data-loader semantics, where the
recoverable action is a full rollback rather than a cancellation.

Every `__next__` runs **five phases**.

#### Phase 1 — Claim

Under no lock, in this order:

1. Validate lifecycle: the iterator is open and not superseded, the loader
   is open, the dataset is open, and the countdown is nonzero (otherwise
   `StopIteration`).
2. **Reject another in-flight transaction.** If a claim or a
   pending-delivery record already exists on this sampler — whoever owns
   it — raise `RuntimeError` and change nothing. This is
   `_claim_reservation`'s rule, and it makes a reentrant `__next__` (from a
   finalizer, a callback, or a signal handler) a deterministic refusal
   instead of two interleaved traversals over one cursor.
3. Compute the candidate: the current committed position `before =
   (seed, shuffle, batch_size, drop_last, epoch, cursor)`, the batch
   `indices = sampler.next_batch_indices()`, and the post-delivery position
   `after`, by applying §7.4's rule to `before`.
4. Mint a **never-reused** serial and publish **only the internal claim**.

`before`, `after`, and `indices` are all pure functions of committed state,
so computing them mutates nothing. **The committed epoch and cursor do not
move in this phase.** A failure anywhere in it leaves no claim, because the
claim is written last.

#### Phase 2 — Construct

With the claim standing and **no committed state advanced**:

1. Gather the host feature rows and build the feature `NativeTensor`
   (§10.4, M1–M2).
2. Gather the targets into the fresh read-only host `int64` array
   (§10.4, M3).

**On any failure in this phase:** close every temporary `NativeTensor`
already allocated in this call, release the host target reference, clear
the claim, and re-raise. Epoch and cursor are unchanged, native live
storage returns to its pre-call value, the iterator and loader remain
usable, and a retry produces the **same indices and the same values**.
Nothing was consumed, because nothing was committed.

#### Phase 3 — Publish

Turn the claim into a **pending-delivery record**, which carries:

- the exact **pre-delivery** sampler state `before`;
- the **candidate post-delivery** sampler state `after`;
- the batch `indices`;
- **ownership of the undelivered feature and target batches**;
- the never-reused **serial**, plus the owning iterator's identity — the
  stale-token protection every cleanup below matches on.

The record is split across its two owners by what each owns, and the serial
is the join key:

- the **sampler** holds the integer half — the serial, `before`, and
  `after` — because that is what its own `state_dict()` and
  `load_state_dict()` guards must be able to see;
- the **iterator** holds the resource half — the feature tensor and the
  target array — because §15's rule puts owned resources on the object
  whose `close()` releases them, and because a sampler that transiently
  owned native storage would contradict §15.2.

Publication rechecks that the claim still matches this serial, exactly as
`_publish_reservation` does; a mismatch means internal state was already
broken and raises rather than proceeding. Once the record exists, **a
second `__next__`, a second `iter()`, `state_dict()`, `load_state_dict()`,
and any state replacement are all refused** (§9.5), so nothing can race
with it or overwrite it.

#### Phase 4 — Commit and deliver

This is one controlled transaction, and the candidate state is applied
**only** inside it:

1. Apply `after` to the sampler through the non-failing position
   assignment of §12.4 — six integer and boolean writes.
2. Pass the result through the **private delivery seam**
   `_deliver_batch(record)`.
3. Only if the seam returns: mark the transaction complete — clear the
   pending record on both owners, release the iterator's ownership of the
   batch **to the caller**, decrement the countdown, and return the tuple.

`_deliver_batch` is a private module-level function that returns its
argument's `(features, targets)` pair and does nothing else. It exists for
exactly one reason: **so that the publish-to-delivery failure position is
addressable and can be tested deliberately**, by monkeypatching the module
attribute to raise there. It is the direct analogue of
`native_generator._deliver_reservation`. It is never exported, never
referenced from any public API, takes no user-supplied callable, and is
**not** a hook: no public callback, no user extension point, and no
arbitrary code runs inside the transaction.

**Why commit-then-deliver, and not deliver-then-commit.** If the position
advanced *after* the seam returned, a failure in between would hand the
caller a batch the loader still considered unconsumed, and the next call
would deliver it a second time — two live tensors for one logical position,
and a resume that replays a batch. Committing first and rolling back on
failure makes the two error directions asymmetric in the safe direction:
the only recoverable state is "not yet delivered", and it is fully
recoverable.

#### Phase 5 — Failed delivery: complete rollback

If **anything** raises after the candidate state was applied and before the
seam returns, the transaction rolls back completely, in this order:

1. **Restore the exact pre-delivery position.** `before` is written back
   through the same non-failing assignment. This runs first precisely
   because it cannot fail, so the committed state is correct before any
   step that could conceivably raise.
2. **Clear the pending-delivery record** on both owners.
3. **Close the undelivered feature `NativeTensor`** and release the host
   target array reference.

Afterwards, all of the following hold and are asserted by test at J3 and
J7:

- epoch and cursor are **exactly** their pre-delivery values;
- **no logical batch position was consumed**;
- native live storage is back to its pre-call baseline;
- the iterator and the loader remain usable;
- **a retry is valid and returns the exact same batch indices and the exact
  same values** — a freshly allocated tensor with identical contents,
  because the permutation is a pure function of restored state (§7.7) and
  the dataset snapshot is immutable (§5.2).

The rollback runs from an unconditional `finally`, so no failure path can
skip it, and every step is **exact-match**: it acts only on a live pending
record whose serial *and* owning iterator match this transaction's, and
whose outcome is still unset. A newer reservation, a foreign iterator's
reservation, an already-completed one, and an already-rolled-back one are
each left strictly alone — `_release_undelivered`'s rule, for
`_release_undelivered`'s reason. That is also what makes the rollback
**idempotent**, so a `close()` that races the `finally` cannot double-roll:
whichever arrives first performs it, and the second matches nothing.

#### Phase 6 — Successful delivery

Once `_deliver_batch` returns:

- the committed sampler state **is** `after`, permanently;
- **ownership of the feature `NativeTensor` and the target array transfers
  completely to the caller**;
- the loader and the iterator **retain no reference** to the delivered
  batch — neither can close it, and neither tries;
- the pending-delivery record is cleared on both owners;
- the batch **cannot be rolled back and cannot be delivered twice**. The
  serial is never reused, and every cleanup path matches on it, so no later
  cleanup can reach a completed transaction.

**The seam's return is the ownership-transfer boundary**, and it is
unambiguous: before it the batch is the iterator's and a failure rolls
everything back; after it the batch is the caller's and the position is
consumed. There is no third state.

What the caller then does with a value the interpreter has handed them —
including dropping it because of an exception in *their* frame — is
caller-side, exactly as it is for any other owned `NativeTensor` they
discard (§10.5, §15.5). That is not the loader skipping a batch: the batch
was produced and delivered, and the committed position says so correctly.

### 9.5 State access while a transaction is in flight

A claim or a pending-delivery record exists only *inside* `__next__`.
Between batches there is none, so ordinary use — including the checkpoint
sequence of §13.5 — is unaffected. What follows governs reentrant arrivals:
a finalizer, a callback, or a signal handler reaching the loader while a
transaction is open.

| Operation, while a claim or pending record exists | Behavior |
|---|---|
| `sampler.state_dict()` / `loader.state_dict()` | **`RuntimeError`**, reading nothing. |
| `sampler.load_state_dict()` / `loader.load_state_dict()` | **`RuntimeError`**, mutating nothing. |
| A second `__next__` on the same iterator | **`RuntimeError`** at Phase 1 step 2, mutating nothing. |
| `iter(loader)` | **`RuntimeError`**; supersession is refused rather than performed (§9.2). |
| `iterator.close()` / `loader.close()` | **Performs the Phase-5 rollback**, then closes. Never refused (§15.3). |
| `dataset.close()` | Permitted; it touches no loader state. A pending batch is already built, so the in-flight transaction is unaffected; the *next* `__next__` fails in Phase 2. |
| Reading `epoch`, `cursor`, `remaining` | Permitted — these are plain integer reads and no promise is attached to them. |
| `next_batch_indices()`, `plan()`, `epoch_permutation()` | Permitted; all three are pure. |

**`state_dict()` is refused rather than answered, and that is the point.**
Inside Phase 4 the candidate position has been applied but the batch has
not been delivered, so there is no honest single answer: reporting `after`
would expose a committed cursor that skipped an undelivered batch, and
reporting `before` would contradict the object's own fields. **A state
snapshot must never be able to observe a skipped-but-undelivered
position**, so the ambiguous window is refused instead of captured.

This is exactly `snapshot_generator_states`' rule, applied to the same
problem: a generator whose next index has been decided but not committed
"has no single honest state to record, so it is refused rather than
captured ambiguously." A checkpoint's whole value is that the state it
writes was true.

**Outside a transaction, `state_dict()` is always allowed**, including
mid-iteration between batches and after `close()` (§15.3). It then
describes the position after the last **successfully delivered** batch,
which by §9.4 is always the exact next batch.

`load_state_dict()` is additionally refused whenever an iteration is
active at all — not only during a transaction — because an iterator's
captured countdown (§9.3) would otherwise describe a position that no
longer exists.

### 9.6 Failures, abandonment, and close during iteration

| Event | Result |
|---|---|
| Failure in Phase 1 (claim) | No claim published; nothing allocated; epoch and cursor unchanged; iterator usable; retry yields the same batch. |
| Failure in Phase 2 (construct) | Every temporary closed, claim cleared; epoch and cursor unchanged; live storage back to baseline; retry yields the same batch. |
| Failure in Phase 3 (publish) | Only reachable if the claim no longer matches, which already means broken internal state; raises, having advanced nothing, and the claim is cleared. |
| Failure in Phase 4 (commit or delivery seam) | **Full Phase-5 rollback**: pre-delivery epoch and cursor restored exactly, record cleared, undelivered batch closed, live storage back to baseline. No position consumed; retry yields the **same indices and the same values**. |
| Dataset closed mid-iteration | Phase 2 raises `RuntimeError`; claim cleared; nothing advanced. |
| Iterator abandoned **before** requesting a batch | Nothing was claimed and nothing is pending. The committed position is exactly the last delivered batch. Nothing consumed, nothing leaked. |
| Iterator abandoned **during** construction or delivery | Impossible to observe from outside: both are inside `__next__`, whose `finally` completes the rollback before the frame unwinds. If the abandonment is the *cause* (an asynchronous exception), the rollback still runs. |
| Iterator abandoned **after** a successful delivery | The delivered batch is the caller's and stays open. The committed position correctly records it as consumed. The sampler's active-iteration count is released by the iterator's `__del__` fallback. |
| `iterator.close()` | Rolls back any in-flight transaction, closes any undelivered batch, releases the active-iteration count, detaches. Idempotent. **Delivered batches are untouched.** |
| `loader.close()` during iteration | Closes the current iterator — and so performs its rollback — then marks the loader closed. Subsequent `iter()` raises `RuntimeError`. **Delivered batches are untouched.** |
| Caller raises after receiving a batch | The loader is unaffected and the position is correct. The batch is **the caller's**, and closing it is the caller's responsibility (§10.5). |
| `StopIteration` raised, then `__next__` again | `StopIteration` again. An exhausted iterator stays exhausted and never restarts an epoch. |

**Only a successfully delivered batch is ever consumed.** Every other row
above leaves the committed position exactly where it was.

### 9.7 Thread safety

None is claimed. See §16, which states precisely what the transaction's
claim does and does not protect against.

---

## 10. Batch materialization and ownership

### 10.1 The yielded structure

A plain **2-tuple**: `(feature_batch, target_batch)`.

- `feature_batch` — an owning `NativeTensor` of shape
  `(len(indices),) + dataset.feature_shape`, at `dataset.dtype`, on
  `"cpu"`, row-major contiguous, `requires_grad=False`, with no autograd
  history and no graph resources.
- `target_batch` — a fresh, independently owned, C-contiguous,
  **read-only** `numpy.ndarray` of dtype `int64` and shape
  `(len(indices),)`.

It unpacks directly (`for features, targets in loader:`) and adds **no new
public type**. A named tuple or dataclass was rejected: the export
inventory is a contract, and a new exported name that carries no behavior
buys nothing. Nothing prevents a later milestone from adding one; nothing
in Phase J needs it.

### 10.2 Why targets are copied per batch, and are read-only

They are copied because a view into the dataset's target snapshot would let
a caller mutate the dataset through a batch, and because
`cross_entropy` — which is where these go — already requires an
independently owned array (§2.4). They are read-only for the same reason
the Phase-E forward marks its own saved copy read-only: so the object a
caller holds cannot be edited in place and then re-used with a different
meaning.

A read-only array is still a perfectly good `cross_entropy` argument;
`_prepare_class_targets` reads and re-copies it.

### 10.3 Index arrays are not returned, and no private debug hook is added

The yielded tuple carries no index array. The indices are available —
before, during, and after — from the **public** sampler surface
(`next_batch_indices()`, `plan()`, `epoch_permutation()`), which is pure
and callable at any time.

That is a deliberate answer to "is a private index path needed for
determinism tests": **no**. Adding a private hook would create a second
route to a fact the public API already states, and a hook that exists only
for tests is exactly the kind of production surface the project refuses.

### 10.4 The materialization route

Per batch, in order. Steps **M1–M2** are what `dataset.feature_batch` does,
and together they are steps 2 and 3 of §9.4's transition; step **M3** is
`dataset.target_batch`, which is §9.4's step 4:

M1. `host = self._features[list(indices)]` — one NumPy fancy-index gather
   producing a fresh C-contiguous array of the snapshot's dtype, which is
   already the target native dtype's NumPy counterpart. Row order is
   exactly the index order.
M2. `features = NativeTensor.from_array(host, dtype=self.dtype)` — the
   public, explicit host→native boundary. Because `host.dtype` already
   equals the requested dtype and `host` is C-contiguous,
   `ascontiguousarray` returns it unchanged and the transfer copies
   **matching bits** with no conversion.
M3. `targets = numpy.ascontiguousarray(self._targets[list(indices)],
   dtype=numpy.int64)`, then `setflags(write=False)`.

So a batch costs exactly **one host gather + one host→native copy** for the
features and one host gather for the targets. There is no per-batch dtype
conversion, no intermediate native tensor, no reshape, and no view
chain.

**The temporary host gather buffer** is a local, is referenced by nothing
after the transfer, and is not cached, pooled, or reused between batches. A
reused scratch buffer was rejected: it is host memory rather than native,
so it is not literally the forbidden native workspace, but it would be
size-dependent hidden state on the loader with no measured justification,
and J8 is where such a thing would have to be justified if ever.

**Gather ordering** is the index order, exactly, with no sorting and no
deduplication. Sorting would change which row lands in which batch position
and would silently break the exact-batch proof.

**Duplicate indices** are permitted by `feature_batch`/`target_batch` — a
gather is a gather, and rejecting them would cost a set construction per
batch. The **sampler never produces one**: a permutation contains each
index exactly once and a batch is a contiguous slice of it. Both halves are
asserted at J1 and J2.

### 10.5 Ownership after yield

**The caller owns both objects, from the moment `_deliver_batch` returns.**
That seam's return is the ownership-transfer boundary (§9.4, Phase 6): the
pending-delivery record is cleared on both owners, so the loader and the
iterator retain no reference to a delivered batch, close none of them, and
cannot be made to — neither has any way to reach the tensor afterwards.

Before that boundary the batch is **iterator-owned**, and a failure closes
it as part of the Phase-5 rollback. There is no third state and no window
in which ownership is ambiguous.

So the caller must `close()` each feature batch — per iteration, in the
ordinary case:

```python
for features, targets in loader:
    logits = model(features)
    loss = criterion(logits, targets)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    loss.close()
    logits.close()
    features.close()
```

The target array is ordinary host memory and needs no close.

This is the same contract every other native object in the repository has,
and the J6 example demonstrates it explicitly rather than relying on `__del__`.

### 10.6 Cleanup at every materialization failure

| Failure point | Cleanup | State after |
|---|---|---|
| Feature gather (M1) raises | Nothing allocated natively; the partial host buffer is unreferenced; the claim is cleared. | Cursor and epoch unchanged; retry gives the same batch. |
| Native allocation (M2) raises `MemoryError` | `from_array` already closes its own storage on a failed copy; no tensor is published; the claim is cleared. | Unchanged; retry allowed. |
| Transfer (M2) raises | Same — the storage is closed inside `from_array` before the exception leaves it. | Unchanged; retry allowed. |
| Target gather (M3) raises **after** the feature tensor exists | The iterator closes the feature tensor before re-raising. **This is the one Phase-2 cleanup Phase J must write itself**, and J7 injects a failure exactly here. | Unchanged; live native storage returns to its pre-call value; retry allowed. |
| Publish (Phase 3) raises | The claim is cleared and the constructed batch is closed. | Unchanged. |
| Commit (Phase 4, step 1) raises | Only an `epoch` overflow can raise here (§7.4). **Full Phase-5 rollback.** | Unchanged. |
| **The delivery seam (Phase 4, step 2) raises** | **Full Phase-5 rollback**: the pre-delivery epoch and cursor are restored first, the pending record is cleared on both owners, then the undelivered `NativeTensor` is closed and the host target reference released. | **Unchanged — no logical batch position is consumed.** Retry yields the **same indices and the same values**; live storage returns to its pre-call baseline. |
| An asynchronous exception anywhere in Phase 4 | Identical to the row above: the rollback runs from an unconditional `finally` and cannot be skipped. | **Unchanged.** |
| Iterator closed mid-transaction | The same rollback, reached from `close()`; exact-match, so whichever of the two arrives first performs it and the second matches nothing. | Unchanged. |
| Loader closed mid-transaction | Delegates to its iterator's close, as above. | Unchanged. |
| Iterator or loader closed between batches | The active-iteration count is released; delivered batches untouched. | Unchanged. |

The invariant across the whole table, without exception: **the committed
cursor and epoch are unchanged whenever a batch is not successfully
delivered**, and native live storage returns to exactly its pre-call value.
A position is consumed if and only if the caller received the batch.

---

## 11. Loader and sampler state schemas

### 11.1 Constraints

Strictly JSON-compatible, so `_validated_metadata` accepts every field
unchanged and the whole structure survives a checkpoint round trip
(§2.6). Compact, so it costs nothing to carry. Complete, so `(seed,
epoch, cursor, batch_size, drop_last, shuffle)` plus dataset identity
determines every future batch exactly.

**The native checkpoint format is not modified.** No root field, no
version 4, no schema change of any kind.

### 11.2 Sampler state

```json
{
  "format": "tensorforge.native_sampler",
  "format_version": 1,
  "dataset": {
    "samples": 800,
    "feature_shape": [1, 6, 6],
    "feature_dtype": "float32",
    "fingerprint": "3f1c…e90a"
  },
  "seed": 20240612,
  "shuffle": true,
  "batch_size": 16,
  "drop_last": false,
  "epoch": 3,
  "cursor": 27
}
```

| Key | Type | Range / values |
|---|---|---|
| `format` | `str` | exactly `"tensorforge.native_sampler"` |
| `format_version` | `int` (not `bool`) | exactly `1` |
| `dataset` | `dict` | exactly the four keys below |
| `dataset.samples` | `int` (not `bool`) | `>= 1` |
| `dataset.feature_shape` | `list` of `int` (not `bool`) | each `>= 1`; may be empty (scalar samples) |
| `dataset.feature_dtype` | `str` | exactly `"float64"` or `"float32"` |
| `dataset.fingerprint` | `str` | exactly 64 lowercase hex characters |
| `seed` | `int` (not `bool`) | `0 <= seed <= 2**64 - 1` |
| `shuffle` | `bool` | exact `bool` |
| `batch_size` | `int` (not `bool`) | `>= 1` |
| `drop_last` | `bool` | exact `bool` |
| `epoch` | `int` (not `bool`) | `0 <= epoch <= 2**64 - 1` |
| `cursor` | `int` (not `bool`) | `0 <= cursor < batches_per_epoch` derived from the state's own `batch_size`/`drop_last` and `dataset.samples` |

`state_dict()` returns a **fresh** dict with fresh nested containers, at
every call, sharing nothing with the sampler or with a previous result.
`feature_shape` is emitted as a `list` rather than a tuple, matching what a
JSON round trip returns, so a saved-and-reloaded state compares equal to a
freshly produced one **without normalization**. `load_state_dict` accepts a
`tuple` there as well, following the v3.13 precedent that a caller may
legitimately have rebuilt the container.

Keys are emitted in the order shown. Nothing depends on it — validation is
by exact key **set** — and no test may assert dict ordering.

### 11.3 Loader state

```json
{
  "format": "tensorforge.native_data_loader",
  "format_version": 1,
  "sampler": { … exactly the §11.2 object … }
}
```

Exactly three keys. The loader owns no position of its own — it owns an
iterator slot and a pending batch, neither of which is state — so its state
is a tagged wrapper around its sampler's.

**Why a wrapper rather than exposing the sampler state directly:** the
loader is what a caller checkpoints, and the two objects' states must be
distinguishable in metadata. Without its own `format` tag, a loader state
and a sampler state would be the same JSON, so passing one where the other
was meant would be accepted silently. The wrapper is also where a future
loader-owned field would go without disturbing the sampler's schema.

### 11.4 What state must never contain

`NativeTensor` objects, `NativeParameter`s, `NativeGenerator`s, NumPy
arrays or scalars, `bytes`, `memoryview`s, tuples requiring non-JSON
interpretation, type objects, callables, any `id()` or address, any object
`repr`, **the full permutation**, any dataset content, and anything
executable.

**The permutation is derivable, so it is not serialized.** §8 proves it is
a pure function of `(seed, epoch, samples)`, all three of which are already
in the state. Serializing it would put an array whose length is the dataset
size inside a checkpoint manifest — megabytes of JSON for a modest dataset
— to carry information the eight bytes of `seed` already carry exactly.
This is the "unless J0 proves compact derivation impossible" case, and it
is proved possible.

### 11.5 No silent normalization

Loading never casts, coerces, truncates, clamps, wraps, rounds, defaults, or
ignores. In particular: `1` is not `True` and `True` is not `1`; `1.0` is
not `1`; `"1"` is not `1`; a NumPy `int64` is not an `int`; a missing key is
not a default; an unknown key is not ignored; and a cursor past the end is
not clamped. Every one of those is a rejection naming the field.

---

## 12. Validation and error ordering

### 12.1 Principles

Deterministic order, most fundamental first, so that a caller who got two
things wrong learns about the more basic one. Nothing is allocated or
mutated until everything is validated. Error types follow §2.8. Messages
name a stable field path; **no exact message text is a contract**, and no
test asserts one — the repository's policy, and Phase J does not change it.

### 12.2 `NativeTensorDataset.__init__`

The order is §4.8's. Every step precedes every allocation.

### 12.3 `NativeBatchSampler.__init__`

1. `dataset` type (`TypeError`).
2. `batch_size` type (`bool` first) → `>= 1`.
3. `shuffle` exact `bool`.
4. `seed` type (`bool` first) → uint64 range.
5. `drop_last` exact `bool`.
6. The §7.5 joint rule: `drop_last=True` and `batch_size > samples`
   (`ValueError` naming both).

Configuration is checked before the joint rule so a caller with a
`batch_size` of `0` is told that, not told about `drop_last`.

### 12.4 `NativeBatchSampler.load_state_dict`

**Phase 1 — validation, no mutation.** In order:

1. **Transaction guard** — refuse if a §9.4 claim or pending-delivery
   record is in flight (`RuntimeError`), reading nothing. **First of all**,
   because replacing a position underneath a live transaction would make
   the record's `before`/`after` pair describe a stream that no longer
   exists — `NativeGenerator._require_no_reservation`'s rule, and its
   reason.
2. **Iteration guard** — refuse if an iteration is active (`RuntimeError`,
   §9.5), because an iterator's captured countdown would otherwise describe
   a position that no longer exists.
3. **Container** — `dict` (`TypeError`).
4. **Exact key set** — `ValueError` naming missing and unexpected keys,
   sorted.
5. **`format`** — `str` (`TypeError`), then exactly
   `"tensorforge.native_sampler"` (`ValueError`).
6. **`format_version`** — `int`, not `bool` (`TypeError`), then exactly `1`
   (`ValueError`).
7. **`dataset`** — `dict` (`TypeError`), exact key set (`ValueError`), then
   each field's type, then each field's range.
8. **Dataset compatibility**, against the **live** dataset, in this order,
   each a `ValueError` naming both values:
   `samples` → `feature_shape` → `feature_dtype` → `fingerprint`.
   Structural first, digest last: a structural mismatch has an
   understandable message, while "the fingerprints differ" is only useful
   once the shapes agree.
9. **Configuration field types** — `seed`, `shuffle`, `batch_size`,
   `drop_last`, `epoch`, `cursor`, in schema order, `bool`-before-`int`
   at each integer field.
10. **Configuration field ranges** — `seed` in uint64, `batch_size >= 1`,
   `epoch` in uint64.
11. **The §7.5 joint rule**, evaluated against the **state's own**
    `batch_size` and `drop_last` and the live `samples`: a state that would
    produce zero batches is rejected.
12. **Cursor alignment** — `0 <= cursor < batches_per_epoch`, where
    `batches_per_epoch` is derived from the values validated in steps 10–11.
    Last, because it is the only rule that depends on several other fields
    being valid first.

**Phase 2 — commit.** Six assignments: `seed`, `shuffle`, `batch_size`,
`drop_last`, `epoch`, `cursor`. Every one is an `int` or a `bool` already
validated, so the commit **cannot fail**. That is what makes the
transaction exact without a rollback path — the `NativeGenerator
._assign_state` property, deliberately reproduced. The permutation cache
(§7.8) is invalidated as part of the commit; it is not state, and
invalidating it cannot fail either.

**This same non-failing primitive is what §9.4 uses**, in both directions:
Phase 4 applies the candidate position with it and Phase 5 restores the
pre-delivery position with it. A rollback that could itself raise would be
able to leave a loader half-restored, which is precisely the defect the
batch transaction exists to prevent — so the two share one write seam
rather than two spellings of one idea.

**Validated against reality, never adopted:** the four dataset identity
fields.
**Adopted from the state:** `seed`, `shuffle`, `batch_size`, `drop_last`,
`epoch`, `cursor`.

That split is `NativeAdam.load_state_dict`'s, applied unchanged (§2.7):
Adam validates each parameter's shape/dtype/device against the live object
and **replaces** `lr`, `betas`, and `eps` at commit. So a restored sampler
may legitimately report a different `batch_size` than its constructor was
given, exactly as a restored `NativeAdam` may report a different `lr`. It
is stated behavior, it is what makes the J5/J6 proof constructible without
the caller having to remember the original configuration, and it is
asserted by test.

**Object identity is preserved absolutely.** The sampler, the dataset, and
(for §12.5) the loader are the same objects afterwards. Nothing is
replaced, rebound, or recreated.

### 12.5 `NativeDataLoader.load_state_dict`

1. **Closed guard** — `RuntimeError` if the loader is closed. Restoring a
   position into a closed loader is meaningless, and refusing is more
   useful than silently succeeding.
2. **Transaction guard** — `RuntimeError` if a §9.4 claim or
   pending-delivery record is in flight, mutating nothing.
3. **Iteration guard** — `RuntimeError` if an iteration is active.
4. **Container** → **exact key set** → **`format`** (exactly
   `"tensorforge.native_data_loader"`) → **`format_version`** (exactly
   `1`) → **`sampler`** is a `dict`.
5. **Delegate the whole of §12.4's Phase 1** to the sampler, as a
   *validation-only* call that mutates nothing and returns the six values.
6. **Commit** through the sampler's non-failing assignment.

So a malformed inner sampler state is rejected **before** anything moves,
and the loader and its sampler can never end up half-restored. The
validate/assign split exists precisely so the loader can validate the inner
object without a two-phase protocol of its own.

`state_dict()` on the loader is allowed at any time, **including after
close** — reading a position is not a resource operation, and a caller who
closes a loader and then records where it stopped is doing something
reasonable. The **one** exception is a §9.4 transaction in flight, where it
raises `RuntimeError` rather than answer ambiguously (§9.5): a snapshot
must never be able to observe a committed cursor that skipped an
undelivered batch.

### 12.6 `feature_batch` / `target_batch`

1. Closed dataset (`RuntimeError`).
2. `indices` type: a `tuple`/`list` of exact `int`s, or a 1-D
   non-`bool` integer `numpy.ndarray` (`TypeError`).
3. Non-empty (`ValueError`) — a zero-row batch cannot become a native
   tensor (§2.3).
4. Every index in `[0, samples)` (`ValueError` naming the position and the
   value).
5. Only then: gather and allocate.

### 12.7 What a validation failure may never do

A rejected construction, state load, or batch request must not: consume a
shuffle draw (structurally impossible, §7.7); **consume a logical batch
position**; advance `cursor` or `epoch`; leave a claim or pending-delivery
record behind; leave an undelivered `NativeTensor` unclosed; allocate
persistent native storage; mutate a dataset, sampler, or loader; touch a
`NativeParameter`, buffer, version, gradient, or graph; touch a registered
`NativeGenerator`; write a file; or change any global or module state. J7
asserts each of these with a complete before/after fingerprint of the
observable world, following the I10 corruption-matrix precedent.

**A failed delivery is held to exactly this list**, which is what makes it
indistinguishable from a rejected request as far as observable state is
concerned: the only difference is that it happened later.

---

## 13. Checkpoint metadata integration

### 13.1 The format does not move

`tensorforge.native_checkpoint`, version **3**, accepted `(1, 2, 3)`, and
the in-memory optimizer state version **1**. No root field is added, no
manifest field changes, no version 4 exists, and no checkpoint code imports,
references, discovers, registers, or validates a Phase-J object.

### 13.2 The supported caller workflow

Loader state travels as **caller-supplied metadata**. TensorForge validates
it as recursively JSON-compatible and **preserves** it; it does not
interpret it.

**Saving:**

```python
loader_state = loader.state_dict()          # pure read; see §13.5
save_native_checkpoint(
    path, model, optimizer=optimizer,
    metadata={
        "training": {
            "next_step": step + 1,
            "data_loader": loader_state,
        }
    },
)
```

**Restoring, into entirely fresh objects:**

```python
metadata = load_native_checkpoint(path, model, optimizer)
training = metadata["training"]
loader.load_state_dict(training["data_loader"])
next_step = training["next_step"]
```

### 13.3 `"training"` and `"data_loader"` are the caller's names, not a schema

This document **recommends** those two keys and the J6 example uses them,
so that examples and tests speak one dialect. **No runtime code depends on
them.** `save_native_checkpoint` sees an ordinary nested dict;
`load_native_checkpoint` returns an ordinary nested dict; neither knows
that a loader exists. A caller may nest the state anywhere in their
metadata, under any key, or store several loaders' states side by side.

Keeping that distinction sharp is the point: the moment TensorForge knew
the key name, it would own the schema, and the next question would be
whether it should validate it — which is the implicit coupling §13.6
forbids.

### 13.4 Missing or malformed metadata

- **Absent** — `metadata.get("training", {}).get("data_loader")` is `None`
  and the caller decides what to do (start fresh, or fail). Passing that
  `None` to `load_state_dict` raises `TypeError`, having changed nothing.
- **Present but malformed** — rejected by §12.5's validation, atomically,
  leaving the loader exactly as it was. A corrupted archive therefore
  cannot half-restore a loader.
- **Present but for a different dataset** — rejected at §12.4 step 7 by the
  identity fields, naming which one differs.
- **Non-JSON-compatible at save time** — rejected by `_validated_metadata`
  before the archive is written. Every §11 field is JSON-native, so this
  can only fire on something else the caller put in the same dict.
- **`NaN`/`Infinity` smuggled into a hand-written archive** — rejected on
  load by the same validator, since I10.

### 13.5 Ordering, and what is and is not atomic

**Save order.** Build `loader.state_dict()` **first**, then call
`save_native_checkpoint`. Outside a §9.4 transaction the loader state is a
pure read of six values with no failure mode, and ordinary calling code is
never inside one — a transaction exists only between the phases of
`__next__`. (A reentrant caller *can* meet the §9.5 refusal, and that is the
deliberate answer there: a snapshot taken mid-handoff would be ambiguous.)

What matters for ordering is that **the loader must not be iterated between
the two calls**, or the archive would describe a position the run has
already left. `save_native_checkpoint`
takes its own coherent model/optimizer/generator snapshot under the
process-wide guard; the loader is not part of that snapshot and cannot be,
because it is not reachable from the model.

**Restore order.** `load_native_checkpoint` **first**, then
`loader.load_state_dict`. The checkpoint load is the transactional one; if
it fails, nothing was restored and the loader was never touched.

**What is atomic, precisely:**

- `load_native_checkpoint` is atomic **over the model, the optimizer, and
  every registered generator**. That guarantee is Phase G's and Phase I's
  and Phase J does not weaken it.
- `loader.load_state_dict` is atomic **over the loader and its sampler**
  (§12.5).
- `__next__` is atomic **over one batch handoff** (§9.4): the position
  advances if and only if the caller received the batch.
- **There is no cross-object atomicity between the first two, and none is
  claimed.** They are two calls on two unrelated objects. If the first
  succeeds and the second fails, the model is restored and the loader is
  not.

**Coordinating them is the caller's responsibility**, and the recommended
recovery is stated rather than left implicit: the archive is unchanged by a
failed loader load, so the caller discards the model, optimizer,
generators, and loader, rebuilds them, and repeats the whole sequence from
the file. Nothing partial survives that, because nothing was written.

This is honest rather than convenient. Providing genuine cross-object
atomicity would require the checkpoint runtime to know about loaders —
exactly the coupling §13.6 forbids — and it is not worth a schema change
and a permanent dependency to remove a two-line recovery from a rare path.

### 13.6 What is explicitly not added

No automatic loader discovery. No loader registration on a `NativeModule`.
No new checkpoint root field. No version 4. No import in either direction
between the checkpoint module and the data-pipeline modules. No automatic
dataset serialization. No `map_location`.

And the long-standing guardrail stays **true and stays enforced**: *the
native checkpoint captures no data-loader position, shuffle order, or epoch
counter.* Phase J does not make that false — it makes the position
something the **caller** can serialize and hand back, through a metadata
channel that already existed and is already validated. The checkpoint's own
capture set does not grow by one field.

### 13.7 `next_step`, and the epoch boundary

`next_step` is the index of the step that has **not** run yet. A save taken
after step `k` completes records `k + 1`, and the resumed loop is
`for step in range(next_step, total_steps)`.

The loader agrees **by construction**: its cursor advances **if and only
if** a batch was successfully delivered (§9.4), so its state always
describes "the next one". The two cannot drift by one, which is the error
every resume proof turns on.

Three consequences follow directly from the §9.4 transaction, and each is
asserted by test at J5:

- **`state_dict()` always describes the exact next batch.** There is no
  position it can report that a delivery skipped, because a skipped
  delivery is rolled back before any observer can see it — and while the
  transaction is open, `state_dict()` refuses rather than answers (§9.5).
- **A failed delivery followed by a checkpoint resumes from the same
  candidate batch.** The rollback restored the pre-delivery position, so
  the archive records it, and the restored loader re-plans the identical
  indices.
- **A successful delivery followed by a checkpoint resumes from the
  following batch**, because the position was committed exactly once, when
  ownership transferred.

So **checkpoint metadata cannot capture a skipped-but-undelivered
position.** There is no execution in which such a position exists as
committed state.

If the save lands exactly at an epoch boundary, the state reads
`(epoch + 1, 0)` — the canonical form of §7.4 — and the resumed loader
begins the next epoch's permutation from its first batch. One
representation, one behavior, and no boundary special case anywhere.

---

## 14. The exact interrupted-training contract

### 14.1 What J5 and J6 must prove

Two runs of the same deterministic training program:

- **Uninterrupted** — construct model, optimizer, generators, dataset,
  sampler, loader; train `N` steps; record everything.
- **Interrupted and resumed** — construct the same objects; train `S < N`
  steps; save a checkpoint with loader state in metadata; **discard every
  object**; construct **entirely fresh** ones; load; train the remaining
  `N - S` steps; record everything.

And one further leg, which the §9.4 transaction makes provable rather than
merely hoped for:

- **Interrupted by a failed delivery** — run to step `S`, inject a failure
  at the `_deliver_batch` seam, let it propagate, then checkpoint and
  resume. The resumed run must reproduce the uninterrupted run **exactly**,
  because the failed delivery consumed nothing: the checkpoint records the
  same candidate batch the failed call was about to deliver.

### 14.2 The resumed objects must be genuinely fresh

Fresh model, fresh optimizer, fresh registered generator objects, fresh
dataset, fresh sampler, fresh loader. No object identity from the first run
survives, and the proof asserts that (`a is not b` at every level,
including the generators reached through the model's alias topology).

**The fresh loader is deliberately constructed wrong**, and this is a
requirement rather than a flourish: it is built with a **different seed**
and a **different position** than the checkpoint holds — and, because
configuration is adopted (§12.4), it may legitimately be built with a
different `batch_size` too. If the restoration did not actually take, the
subsequent batches would differ, so the proof cannot pass vacuously.

The negative control makes that explicit: with the loader restoration
**omitted**, the batch indices must be proved **unequal**. And it must be
non-vacuous by construction — the interruption point `S` is chosen so that
it is **not** a multiple of `batches_per_epoch`, so the resume genuinely
lands mid-epoch. This is the discipline the I9 proof used for `SPLIT_STEP`.

### 14.3 What must match, and how exactly

| Subject | Comparison |
|---|---|
| The next batch's indices, immediately after restore | exact tuple equality |
| The entire remaining batch-index sequence to the end of the run | exact tuple equality, batch by batch |
| Every feature batch | **raw IEEE-754 bit patterns** — `to_numpy().view(uint32)` at float32, `view(uint64)` at float64 |
| Every target batch | exact `int64` array equality, plus dtype, shape, contiguity, and the read-only flag |
| Every model parameter | raw bit patterns |
| Every persistent buffer | raw bit patterns |
| Every Adam `m`, `v`, and step counter | raw bit patterns; counters by exact `int` equality |
| Optimizer hyperparameters | exact equality |
| Every registered generator's algorithm, version, seed, and calls | exact equality |
| The generator alias topology | exact structural equality |
| Loader `epoch`, `cursor`, `seed`, `batch_size`, `shuffle`, `drop_last` | exact equality |
| The full loader `state_dict()` at the end of both runs | exact equality |
| The loader `state_dict()` taken **after an injected delivery failure** and the one taken **before** it | exact equality — the rollback of §9.4 Phase 5 is what makes these the same object state |
| The loss suffix and the whole loss sequence | raw bit patterns |
| Final logits, predictions, and evaluation output | raw bit patterns |
| Native live storage at the end | exact equality with the pre-run baseline |

**No tolerance anywhere.** No `allclose`, no `atol`, no rounding, no
"approximately". This is the standard every exact-resume proof from Phase C
onward has met, and Phase J does not introduce the first exception.

### 14.4 Dtypes are proved separately

At float32 and at float64, each compared **only against itself**, exactly
as I9 requires. A float32 run is never required to reproduce a float64 one
and nothing asserts that it does — but the **batch indices** are asserted
**identical across dtypes**, because the permutation is a pure function of
`(seed, epoch, samples)` and carries no dtype at all. That is a real,
testable property of this design and it is proved in both directions.

### 14.5 Zero leaked native storage

Both runs return the native live-storage count exactly to its pre-run
baseline, with every feature batch closed explicitly by the loop and every
temporary closed at its last use. The J6 example demonstrates the
discipline; J7 asserts it under failure injection too.

**Including a failure injected at the delivery seam.** J7 patches
`_deliver_batch` to raise, and asserts that afterwards the epoch and cursor
are bit-for-bit their pre-call values, the loader `state_dict()` is equal to
the one taken before the call, live storage is back to its pre-call
baseline, and the very next `__next__` returns the **same indices and the
same values**. That is the executable form of "a failed delivery consumes
nothing", and it is the reason the seam exists at all (§9.4, Phase 4).

---

## 15. Lifecycle and close semantics

### 15.1 The rule

**`close()` exists exactly where something is owned**, and nowhere else.
Inventing a `close()` for an object that owns nothing "would advertise a
lifetime that does not exist" — `NativeGenerator`'s stated reason, applied
here rather than replaced by a blanket convention.

### 15.2 States and owners

| Object | Owns | `close()` | `closed` | Context manager |
|---|---|---|---|---|
| `NativeTensorDataset` | two host snapshots | **yes** | yes | yes |
| `NativeBatchSampler` | nothing | **no** | no | no |
| `NativeDataLoader` | the iterator slot | **yes** | yes | yes |
| `_NativeBatchIterator` | any undelivered pending batch, and one active-iteration count | **yes** | — | yes |

The sampler's integers and booleans need no release, and the permutation
cache is a Python tuple. **The sampler holds the integer half of a
pending-delivery record (§9.4, Phase 3) but never the batch itself**, which
is exactly why it still owns nothing releasable and still needs no
`close()`. A sampler is always usable; there is no closed sampler state to
reason about, no `RuntimeError` for using one, and no lifecycle question in
its state schema.

The loader owns the iterator *slot*, not the pending batch: it reaches an
undelivered batch only through its current iterator, which is the object
§15.1's rule makes responsible for releasing it.

### 15.3 Close behavior, in detail

- **Idempotent** everywhere; a second `close()` returns `None` and does
  nothing.
- **`NativeTensorDataset.close()`** drops both snapshots. Metadata stays
  readable (§5.5); `feature_batch`/`target_batch` raise `RuntimeError`.
- **`NativeDataLoader.close()`** closes the current iterator — which
  performs any in-flight rollback and releases the sampler's
  active-iteration count — then marks the loader closed. Afterwards
  `iter()` and `load_state_dict()` raise `RuntimeError`; `state_dict()`,
  `sampler`, `dataset`, and `repr` still work.
- **`_NativeBatchIterator.close()`** performs the §9.4 Phase-5 rollback if
  a transaction is in flight — restoring the exact pre-delivery epoch and
  cursor, clearing the pending record on both owners, then closing the
  undelivered `NativeTensor` and releasing the host target reference —
  releases the active-iteration count, and detaches. A closed iterator's
  `__next__` raises `RuntimeError`, not `StopIteration` — a closed
  traversal is a lifecycle error, not an exhausted one, and conflating them
  would let a `for` loop silently swallow a close.
- **`close()` is never refused, at any object, in any state.** It is the
  recovery path, so refusing it during a transaction would strand exactly
  the resources it exists to release. It is also **exact-match and
  therefore idempotent** against the transaction's own `finally`: whichever
  reaches the rollback first performs it, and the second finds no matching
  live record and does nothing. Neither can disturb a newer, foreign,
  already-completed, or already-rolled-back reservation.
- **No `close()` ever touches a delivered batch.** After the seam returns,
  the loader and the iterator hold no reference to it (§9.4, Phase 6).

### 15.4 Ordering, and cross-object close

- **Recommended order:** loader, then dataset. A `with` nest gives it for
  free.
- **Dataset closed first, loader still live:** supported and deterministic
  (§9.6). Planning and state keep working; materialization raises.
- **Loader closed while a yielded batch is alive:** the batch is untouched.
  It is the caller's, and closing the loader neither closes nor invalidates
  it.
- **Loader closed during iteration:** §9.6.
- **Dataset closed while a sampler exists:** the sampler is unaffected in
  every respect.

### 15.5 Garbage collection is a fallback only

`_NativeBatchIterator.__del__` releases the sampler's active-iteration
count and closes a pending batch, so an abandoned iterator cannot block a
loader forever. That is a **fallback**, exactly as `NativeTensor.__del__`
is: no test asserts a GC-timing behavior, and **no native-batch cleanup
relies on it** — every delivered batch is the caller's and every pending
batch is reachable through an explicit `close()`.

### 15.6 What the loader can and cannot clean up

**Can:** its own iterator, that iterator's undelivered pending batch, and
its own active-iteration count.

**Cannot, and must not try:** any batch it has already delivered. Once
Phase 6 of §9.4 runs, the loader holds no reference and the caller owns the
tensor. A loader that kept delivered batches alive so it could close them
would hold the entire epoch's native storage until the epoch ended, which
is a memory leak in the shape of a convenience.

The split is exactly the transaction's boundary: **undelivered pending
batches are iterator-owned and are closed by rollback or by `close()`;
delivered batches are caller-owned and are never closed by loader
shutdown.**

---

## 16. Reentrancy and concurrency

### 16.1 The contract

**The Phase-J objects are not thread-safe, and none of them contains a
lock.** One thread at a time per dataset, sampler, and loader.

Phase J adds **no worker process, no thread, no thread pool, no prefetch,
no queue, no future, no async iteration, and no lock.**

### 16.2 What is defined, and what is not

| Situation | Behavior |
|---|---|
| **Reentrant** `__next__`, `iter()`, `state_dict()`, or `load_state_dict()` arriving on the **same thread** during a transaction (a finalizer, a callback, a signal handler) | **Deterministic `RuntimeError`**, mutating nothing (§9.5). This is what the §9.4 claim is for, and it is a correctness mechanism rather than a concurrency one. |
| Reentrant `close()` on the same thread during a transaction | Performs the rollback; exact-match, so it cannot double-roll or disturb another reservation. |
| **Concurrent `__next__` from two threads** on one loader | **Undefined.** Not supported, not protected, not claimed. The claim check is not atomic without a lock, so two threads can both pass it. |
| Concurrent `__next__` on two loaders over one dataset | Safe in practice — the dataset is read-only after construction and each loader has its own state — but **not claimed**, because "safe in practice" is not a contract this phase is prepared to test at every path. |
| Concurrent `state_dict()` | **Undefined.** Single-threaded it is either a correct read or a deterministic refusal (§9.5); across threads the refusal check can itself be raced, so no atomicity is claimed. |
| Concurrent `load_state_dict()` | **Undefined**, and not protected. |
| `close()` during another thread's operation | **Undefined.** `close()` itself is idempotent and its rollback is exact-match. |
| Two active iterators | Deterministically resolved by supersession, or refused outright during a transaction (§9.2) — neither is a thread-safety mechanism and neither becomes one. |

**The claim guards reentrancy, not concurrency, and the difference is
stated rather than blurred.** A reentrant arrival is on the calling thread,
so it necessarily observes the claim that thread already published and is
refused deterministically. Two genuinely concurrent threads can both read
"no claim" before either writes one, and no lock prevents that because
Phase J adds none. What survives even a raced transaction is the
**exact-match** rule: no cleanup can ever roll back, close, or complete a
reservation that is not its own, so a race cannot corrupt an unrelated
batch or an unrelated position.

**External locking is required** for any concurrent use. That is the
documented answer, not a deficiency to be fixed later: it is exactly the
stance the repository already takes on ordinary training mutation, which
does not take the process-wide guard either.

### 16.3 The Phase-J objects join no lock order

They do **not** take `_native_state_lock.state_transaction()` and do not
enter the universal `id()`-sorted generator order.

The reason is structural rather than a shortcut: a loader is never
reachable from a `NativeModule`, so no checkpoint save, no checkpoint load,
no module `load_state_dict`, and no generator transaction can touch one —
there is nothing for it to serialize *with*. Joining the order would widen
a process-wide contract to cover an object no participant can reach, and
would put a lock acquisition on the batch path for no benefit.

A caller who saves a checkpoint from one thread while iterating in another
already has an unsynchronized program, and §13.5 says plainly that the
loader state must be read at a point where the loader is not advancing.

---

## 17. Allocation failure and rollback

### 17.1 The requirement

Every failure point has a stated cleanup, a stated post-state, and a stated
answer to "may the caller retry". J7 injects a failure at each and asserts
the row.

### 17.2 Construction

| Failure after… | Cleanup | Object state | Retry |
|---|---|---|---|
| Input normalization begins | Nothing allocated yet. | No object exists. | Yes |
| Feature snapshot allocated, target snapshot fails | The feature snapshot is released before the exception leaves the constructor. | No object exists; no reference survives. | Yes |
| Both snapshots allocated, fingerprint fails | Both snapshots released. | No object exists. | Yes |

No partially constructed dataset is ever observable, and construction
allocates **no native storage at all**, so a failed construction cannot
move the live-storage count.

### 17.3 Iteration

| Failure after… | Cleanup | Cursor / epoch | Iterator | Loader | Retry |
|---|---|---|---|---|---|
| Index plan built (Phase 1) | Nothing to clean; the plan is a tuple. The claim is cleared. | unchanged | usable | usable | yes, same batch |
| Host gather allocated (Phase 2, M1) | The buffer is unreferenced; the claim is cleared. | unchanged | usable | usable | yes, same batch |
| Feature `NativeTensor` allocation fails (Phase 2, M2) | `from_array` closes its own storage; the claim is cleared. | unchanged | usable | usable | yes, same batch |
| Feature transfer fails (Phase 2, M2) | Same. | unchanged | usable | usable | yes, same batch |
| Target batch copy fails **after** the feature tensor exists (Phase 2, M3) | **The iterator closes the feature tensor**; the claim is cleared. | unchanged | usable | usable | yes, same batch |
| Pending record publication fails (Phase 3) | The constructed batch is closed and the claim cleared. | unchanged | usable | usable | yes, same batch |
| **Candidate position applied, then the commit or the delivery seam raises (Phase 4)** | **Full Phase-5 rollback**: the pre-delivery position is restored first, the record is cleared on both owners, then the undelivered `NativeTensor` is closed and the host target reference released. | **unchanged** | usable | usable | **yes, same batch** |
| **An asynchronous exception anywhere in Phase 4** | Identical: the rollback runs from an unconditional `finally` and cannot be skipped. | **unchanged** | usable | usable | **yes, same batch** |
| `close()` arriving during Phase 4 | The same rollback, reached from `close()`; exact-match, so it and the `finally` cannot double-roll. | **unchanged** | closed | usable or closed | n/a |
| Iterator creation fails | The active-iteration count is released. | unchanged | none | usable | yes |

**Every row leaves the cursor and epoch unchanged.** That is the point of
the table, and it is why "retry, same batch" is meaningful throughout:
calling `__next__` again re-plans the identical indices from restored state
(§7.7) over an immutable dataset snapshot (§5.2), and produces a batch with
identical values. The retried batch is a **fresh allocation** — a new
`NativeTensor` object with the same contents — because the rolled-back one
was closed.

**No row advances the position, and no row consumes a logical batch.** A
position is consumed exactly once, at Phase 6, and only when the caller
received the batch.

### 17.4 A note on `MemoryError`

A `batch_size` large enough to exhaust memory raises `MemoryError` from the
native allocation, which is the honest answer and the one the runtime
already gives. The sampler imposes no upper bound on `batch_size` because
it cannot know the machine, and inventing a threshold would be a
performance control, which the project does not have.

### 17.5 State reconstruction

| Failure at… | Effect |
|---|---|
| Any Phase-1 validation step (§12.4, §12.5) | Nothing mutated. Every field, the position, the configuration, and the permutation cache are exactly as they were. |
| Phase-2 commit | **Cannot fail** — six assignments of already-validated `int`s and `bool`s. |

There is therefore no rollback path to test, because there is no failure to
roll back from. That is a design property, not an untested gap, and J4
asserts it by showing the commit consists only of assignments.

---

## 18. Stable / native isolation

Phase J belongs entirely to the experimental native stack. It preserves,
without exception:

- **No stable `Tensor` acceptance.** `NativeTensorDataset` accepts NumPy
  arrays; a `tensorforge.Tensor` is a `TypeError`, at the same place any
  other non-`ndarray` is.
- **No stable dataset coupling.** `tensorforge.data.batches` and
  `train_test_split` are not used, wrapped, imported, extended, or
  modified. The two mini-batch stories are separate, and the stable one is
  feature-frozen.
- **No implicit conversion** in either direction, and no bridge object.
- **No backend routing** based on input. There is one implementation and no
  dispatch.
- **No global loader**, default dataset, ambient sampler, or module-level
  singleton of any kind.
- **No change to stable training APIs, exports, or checkpoint formats.**
  `tests/test_public_api.py`'s locked stable surface does not gain a name.
- **`import tensorforge` still does not import the native backend**, and
  the test that proves it keeps passing. The Phase-J modules live under
  `tensorforge.experimental`, which the stable line never imports.
- **`backend_info()["stable_framework_integration"]` stays `False`.**

---

## 19. Dtype and target boundaries

1. **Feature batches may be float64 or float32**, and nothing else.
2. **An omitted `dtype` means float64**, at the dataset constructor as
   everywhere else. `None` means float64.
3. **The input NumPy dtype never chooses the native dtype.** A float32
   NumPy feature array with `dtype` omitted gives a **float64** dataset and
   float64 batches. Asserted in both directions.
4. **Targets are copied host `int64` arrays**, never native tensors.
5. **No native integer tensor** is introduced, needed, or implied.
6. **No mixed-dtype arithmetic, no promotion, no casting API**, no
   `astype`/`to`/`.float()`/`.double()`/`map_location`, and no global
   default dtype.
7. **No `device` argument** anywhere in the phase. `dataset.device` is a
   read-only `"cpu"` report, matching every other native object.
8. **No float16, no bfloat16, no additional dtype**, no AMP, no CUDA.

**Converting host feature values into the chosen native dtype is not a
casting feature.** It happens once, at construction, at the same explicit
host→native boundary `NativeTensorCore.from_array` has always used, on
**host** data that carries no native dtype. No native tensor changes dtype,
and none can. This is dtype design §9.4 applied unchanged, and moved
earlier so that per-batch transfer copies matching bits.

A dataset's dtype is fixed at construction. The only way to get the other
one is to construct another dataset — which is exactly the rule for every
native tensor in the repository.

---

## 20. Non-goals restated as boundaries

None of the following is delivered by any Phase-J milestone, and none may
be added under the heading of a Phase-J milestone:

native integer tensors · integer tensor arithmetic · embeddings · public
gather/scatter tensor operations · sparse tensors · CUDA · GPU execution ·
AMP · mixed precision · float16 · bfloat16 · dtype casting · dtype
promotion · device movement · `map_location` · multiprocessing workers ·
background workers · prefetch threads · asynchronous iteration · pinned
memory · distributed sampling · network datasets · filesystem downloads ·
a transforms framework · arbitrary user-defined collation · streaming
datasets · infinite datasets · memory mapping · checkpoint version 4 ·
optimizer-state version 2 · another RNG algorithm · another global or
default generator · a generic random-number API · implicit stable/native
conversion · new C ABI exports (§22.3) · timing assertions · performance
gates · generated benchmark-result files · external dependencies ·
external-project names or references.

---

## 21. Testing strategy

### 21.1 Per-milestone modules

| Module | Milestone | Subject |
|---|---|---|
| `tests/test_native_phase_j.py` | J0 | The contract guardrails (§25). |
| `tests/test_native_dataset.py` | J1 | Input validation, snapshot semantics, indexing, identity, close, failure cleanup. |
| `tests/test_native_sampler.py` | J2 | The derivation, the reference vectors, the C++ equivalence check, planning, boundaries, state. |
| `tests/test_native_data_loader.py` | J3 | Iteration, batch shape/dtype/ownership, the iterator state machine, cleanup. |
| `tests/test_native_loader_state.py` | J4 | State schemas, transactional loading, mismatch rejection, mid-epoch resume. |
| `tests/test_native_data_checkpoint.py` | J5 | The metadata workflow, fresh-object restoration, malformed metadata. |
| `tests/test_native_minibatch_training.py` | J6 | The example and its exact interrupted/uninterrupted proof. |
| `tests/test_native_data_hardening.py` | J7 | The cross-cutting adversarial matrix. |
| `tests/test_native_data_benchmark.py` | J8 | The benchmark's correctness gates and contract. |
| `tests/test_native_phase_j_closure.py` | J9 | Closure guardrails, in the shape of the Phase-H and Phase-I closure modules. |

### 21.2 Standing requirements

- **Every parser in a documentation or status test has a negative
  control**, per the I11 precedent: a check that finds nothing must be
  shown able to find something.
- **Bit-level claims use raw IEEE-754 bit patterns**, never tolerances.
- **Live-storage baselines** are asserted around every ownership and
  failure test.
- **No test asserts an exact error message**, a dict ordering, a timing, or
  a GC event.
- **The reference vectors of §8.9 are the specification**, committed on the
  test side as literal values written independently of the implementation
  — a known-answer set, not a regression convenience.
- **A NumPy-compute tripwire** guards the native paths, as in every
  previous phase: the feature batch's *values* must reach native storage
  through the transfer boundary, not through NumPy arithmetic.
- **The delivery seam is exercised by injection, not by inspection.** J3
  and J7 monkeypatch the private `_deliver_batch` module attribute to
  raise, and assert the §9.4 Phase-5 rollback in full: epoch and cursor
  restored exactly, the pending record cleared, the undelivered
  `NativeTensor` closed, live storage back to its pre-call baseline, no
  logical batch consumed, and the next `__next__` returning the **same
  indices and the same values**. Patching the module attribute is the
  established Phase-G technique — `native_generator._deliver_reservation`
  is tested the same way — and it requires no production hook, because the
  seam takes no user-supplied callable.
- **Every injection has a non-vacuity control**, proving the patched seam
  really did raise and the assertion is not passing because the failure
  never occurred.

---

## 22. Build, platform, and dependency requirements

### 22.1 No new dependency, and no build change

Phase J adds no dependency. `hashlib` and `json` are Python standard
library; NumPy is already the only numeric dependency. No build option, no
CMake change, no compiler flag, and no CI job is added.

### 22.2 No C++

No milestone in the ladder requires a kernel, a header, a CTest, or a
translation unit. The gather is a NumPy operation on host memory and the
transfer is the existing `tf_storage_copy_from` path. The CTest inventory
is expected to be **24** at J9, exactly as at J0.

### 22.3 The export count, and the one condition for revisiting it

The library exports **54** production `tf_*` symbols at J0, and Phase J
plans **none**. The design deliberately routes every batch through
`NativeTensor.from_array`, which needs no new symbol.

The only circumstance under which the question could be reopened is a
**measured** J8 finding that the host gather plus host→native copy is the
dominant cost of a realistic training step and that a native gather export
would remove it. Even then it would have to be **separately approved**, as
its own decision with its own evidence, its own review, and its own
milestone — not something J8 may do, since J8 is characterization only and
adds no capability. Absent that evidence, the answer is no.

### 22.4 Cross-platform requirements

J9 runs the standard matrix: Windows Release and Debug, a Linux
CI-equivalent, Clang ASan/UBSan with a negative control, and a
LeakSanitizer lifecycle. Because Phase J adds no C++, the sanitizer runs
exercise the **existing** kernels under the new Python paths — the
allocation and transfer traffic a loader generates is new even though the
code it runs is not, which is exactly what a leak lifecycle should see.

The derivation of §8 is pure Python integer arithmetic with explicit
`& MASK`, so it is bit-identical on every platform, word size, and Python
build by construction; §8.8's live cross-check re-proves it against the
compiled kernel on every platform the suite runs on.

---

## 23. Milestone ladder — J0 through J9

Each milestone lists its entry condition, scope, tests, documentation,
invariants, exclusions, and exit gate. The ladder is **evidence-driven**:
if repository reality contradicts a milestone's premise, that milestone is
narrowed, reordered, or dropped, and the revision is **recorded here rather
than rewritten away** — the precedent Phase H set three times and Phase I
followed.

### J0 — Architecture and API contract — **complete**

- **Entry:** Phase I complete and merged at I11; working tree clean; 54
  exports; 24 CTests; 15 examples; checkpoint version 3 with `(1, 2, 3)`;
  optimizer state version 1; 7,738 tests passing.
- **Scope:** inspect and record current reality (§2); create this contract;
  resolve the public API, the dataset input and ownership contracts,
  dataset identity, the sampler architecture, the deterministic derivation
  with pseudocode and fixed reference vectors, the iterator state machine,
  batch ownership, the state schemas, the validation ordering, the
  checkpoint-metadata workflow, the exact-resume contract, lifecycle,
  concurrency, rollback, isolation, dtype boundaries, non-goals, the
  ladder, and the exit gates.
- **Tests:** `tests/test_native_phase_j.py` — the durable contract
  guardrails of §25 — plus the targeted documentation-test updates the new
  phase requires.
- **Docs:** this file; Phase-J status on the roadmap, project summary,
  support matrix, architecture, backend experiments, README, and
  `CLAUDE.md`; an in-progress planning entry in the release history.
- **Invariants:** no production Python, C++, ABI, build, CI, example, or
  benchmark change.
- **Exclusions:** every runtime capability, including placeholder classes,
  empty modules, stubs, and `NotImplementedError` methods.
- **Exit gate:** §24.1.

### J1 — Host-backed dataset foundation — **complete**

- **Entry:** J0 merged.
- **Scope:** `NativeTensorDataset` — §4 validation, §5 snapshot semantics,
  §6 fingerprint, `feature_batch` / `target_batch`, `identity()`, close,
  and §17.2 construction-failure cleanup. Exported from
  `tensorforge.experimental`.
- **Tests:** `tests/test_native_dataset.py` — every accepted and rejected
  input class; the mutate-after-construction proof in both directions; view
  versus copy; duplicate and out-of-range indices; scalar samples; both
  dtypes; the no-inference rule; fingerprint determinism, sensitivity, and
  endian normalization; close, use-after-close, and repeated close;
  live-storage baselines.
- **Invariants:** no C++, no export, no checkpoint or optimizer change, no
  registry change. Adds exactly one public name.
- **Exclusions:** shuffling, batching, cursors, epochs, loader state.
- **Exit gate:** every §4/§5/§6/§17.2 rule asserted; `tensorforge
  .experimental.__all__` grows by exactly one; 54 exports; suite green.
- **Outcome:** met. `experimental/native_dataset.py` ships
  `NativeTensorDataset`, exported from `tensorforge.experimental`, whose
  `__all__` went from 22 names to 23 — `NativeTensorDataset` and no other.
  `tests/test_native_dataset.py` covers §4's accepted and rejected input
  classes and their **precedence**, §5's snapshot and alias rules, §6's
  digest against **independently computed known answers**, §12.6's index
  contract at both batch methods, §15's lifecycle, and §17.2's three
  construction-failure positions by injection with non-vacuity controls.
  The C ABI stayed at 54 exports, the CTests at 24, the examples at 15,
  the benchmarks at 8, the checkpoint at version 3 with `(1, 2, 3)`
  accepted, and the optimizer state at version 1; no C++ or CMake file was
  touched.

**Implementation clarifications recorded at J1**, none of which changes a
locked rule — the §23 discipline is to record rather than rewrite:

1. **"Exact `int`s" in §12.6 means `type(value) is int`.** A `tuple` or
   `list` index container therefore rejects a NumPy integer scalar as well
   as a `bool`, on the §4.1 exact-type discipline. This costs a caller
   nothing: a NumPy integer sequence is passed as the **array** the same
   clause already accepts, which is where NumPy integer widths are
   handled. Stated because "exact int" could otherwise be read as
   `_as_int_tuple`'s more permissive rule.
2. **A NumPy scalar feature argument fails the *type* rule, not the rank
   rule.** §4.2 makes a 0-d *array* a `ValueError` for having no sample
   axis; `numpy.float64(1.5)` is not an `ndarray` at all, so §4.1's
   `TypeError` fires first. Both are rejections, and the ordering is the
   one §4.8 already specifies.
3. **The two target-value checks cannot both fire for one array.** §4.8
   step 5 orders int64 representability before non-negativity; an unsigned
   dtype cannot hold a negative value and a signed one cannot exceed the
   int64 maximum, so the order is observable only as which message a given
   dtype produces. The ordering is implemented as specified regardless.
4. **The dataset has no `__del__`.** §15.5 permits a garbage-collection
   fallback where something is owned; the dataset owns two NumPy arrays
   and **no native resource**, so ordinary Python reclamation is already
   correct and a finalizer would advertise a lifetime it does not have —
   `NativeGenerator`'s stated reason for having no `close()` at all,
   applied rather than replaced. `close()` still exists, because the host
   snapshots genuinely are owned and releasing them early is meaningful.
5. **A wrong-*rank* index array raises `ValueError`, not `TypeError`.**
   §12.6 step 2 names one exception for the whole container check, but the
   step folds together two different faults, and §2.8 separates them:
   `TypeError` for a wrong type — a non-container, a non-integer or `bool`
   dtype, a non-`int` element — and `ValueError` for a well-typed but
   unacceptable value, which is what a 2-D or 0-d integer index array is.
   This is exactly the split §4.3 already specifies for the *targets*
   (dtype kind → `TypeError`, rank → `ValueError` naming the shape) and
   the one `_prepare_class_targets` has always used, so the alternative
   would have made the same fault raise two different exceptions in two
   places. The step order is unchanged: rank is checked before dtype, so
   a 2-D float array is reported as the more structural fault.

### J2 — Deterministic sampler — **complete**

- **Entry:** J1 merged.
- **Scope:** `_native_permutation.py` (§8) and `NativeBatchSampler` (§7) —
  order, planning, epoch and cursor semantics, the §7.5 construction rule,
  compact state (§11.2), and transactional loading (§12.4).
- **Tests:** `tests/test_native_sampler.py` — the §8.9 reference vectors as
  committed known answers; the §8.8 live cross-check against
  `tf_core_dropout_forward` with its own non-vacuity control; the rejection
  branch of `bounded` forced directly; permutation validity and
  epoch-to-epoch variation; every §7.6 boundary; the §7.7 no-consumption
  properties; state round-trip, mismatch rejection, and the non-failing
  commit.
- **Invariants:** no `NativeTensor` allocation anywhere in the milestone;
  no C++; no export; no generator coupling.
- **Exclusions:** materialization, iteration, checkpoints.
- **Exit gate:** every reference vector reproduced exactly; the C++
  equivalence check green; `__all__` grows by exactly one; 54 exports.
- **Outcome:** met. `experimental/_native_permutation.py` ships the §8
  derivation — importing **nothing at all**, so the whole bit path is
  built-in Python integer arithmetic with explicit `& MASK` — and
  `experimental/native_sampler.py` ships `NativeBatchSampler`, exported
  from `tensorforge.experimental`, whose `__all__` went from 23 names to
  24: `NativeBatchSampler` and no other. **Every §8.9 reference vector is
  reproduced exactly** — the four `splitmix64_mix` known answers, all
  twelve `epoch_key` vectors, all thirty-two permutations, the sequential
  orders, and all ten batch plans — written on the test side as literals
  rather than generated from the production helper. The §8.8 live
  cross-check runs at 48 `(seed, call_index, p)` combinations of 4,096
  elements each against the built `tf_core_dropout_forward`, with a
  non-vacuity control that mutates the multiplier, the golden constant,
  the first shift, and the domain separator in turn and proves each one
  breaks the prediction; a companion assertion keeps the sampler's own
  domain-separated key schedule provably **distinct** from Dropout's
  rather than letting the equivalence proof be misread. The rejection
  branch of `bounded` is forced directly at `bound = 2**63 + 1`, where
  `limit` makes roughly half of all draws fall out, against hard-coded
  residues and final draw indices. The C ABI stayed at 54 exports, the
  CTests at 24, the examples at 15, the benchmarks at 8, the checkpoint at
  version 3 with `(1, 2, 3)` accepted, and the optimizer state at version
  1; no C++ or CMake file was touched, and no `NativeTensor` is allocated
  anywhere in the milestone.

**Implementation clarifications recorded at J2**, none of which changes a
locked rule — the §23 discipline is to record rather than rewrite:

1. **The sampler shares `NativeGenerator`'s seed validator rather than
   restating it.** `native_sampler` imports exactly `_validate_uint64`
   and `UINT64_MAX` from `native_generator`. That is §8.3's "the same
   unsigned 64-bit domain, validated by the same rules" taken literally:
   a duplicated validator would be the second seed contract §8.3 exists
   to prevent, and it could drift. It is **not** the generator coupling
   §8.3 forbids, and the distinction is asserted rather than argued — the
   import set is pinned by test, no `NativeGenerator` is accepted, held,
   created, or consulted, no call is reserved or committed, and no
   generator instance is reachable from any sampler slot.
2. **`dataset` is checked with `isinstance`, not an exact type.** §4.1's
   exact-type discipline exists because an `ndarray` subclass could make
   a gather mean something other than a gather; the sampler reads only
   `samples` and `identity()`, which any subclass inherits intact, so the
   `_require_token` precedent (`isinstance`) is the right one here. §7.2
   says "must be a `NativeTensorDataset`", which this satisfies.
3. **The permutation cache holds only the *active* epoch.** §7.8 keys it
   on `(seed, epoch, samples)`; an explicit-epoch `epoch_permutation()`
   or `plan()` therefore recomputes without touching it. This is
   narrower than the key alone requires and is deliberate: a batch is only
   ever taken from the active epoch, so that is the only entry worth
   keeping, and it makes "an arbitrary-epoch inspection never touches the
   cache" a property a test can state. Because the value is a pure
   function of the key, the choice is unobservable — §7.8's "dropping it
   at any moment changes no observable behavior" is asserted directly.
4. **`shuffle=False` never reaches the bit path**, and that is proved
   rather than assumed: `sample_order` branches before `epoch_key`, and a
   test patches both `epoch_key` and `draw_bits` to record every call and
   asserts none happens at any seed, epoch, or length. §8.6 states
   sequential order is "a different, cheaper branch"; this is its
   executable form.
5. **§12.4's step 10 ordering is unobservable, and is implemented as
   written anyway.** The step orders every dataset-block field *type*
   before every field *range*, but the per-element rules for
   `feature_shape` are element types and element ranges at once. Element
   type is checked immediately before element range, per element. No
   state can distinguish the two orderings, because the only rule between
   them — the fingerprint's format — belongs to a different field.
6. **`_next_position` exists, is private, and mutates nothing.** §7.4's
   canonical transition is encoded now so J3 applies it through this
   object rather than redesigning the position semantics, and so the
   uint64 epoch-overflow refusal is testable at J2. It is called from no
   public path: **J2 performs no iteration and delivers no batch.**

### J3 — Native mini-batch loader — **complete**

- **Entry:** J2 merged.
- **Scope:** `NativeDataLoader`, `_NativeBatchIterator`, and the
  **§9.4 batch transaction** — the five phases, the private `_deliver_batch`
  seam, the pending-delivery record and its never-reused serial, the
  exact-match rollback, and the §9.5 refusals — plus §10's materialization
  and ownership, §15's lifecycle, and §17.3's cleanup.
- **Tests:** `tests/test_native_data_loader.py` — batch shapes, dtypes, and
  contiguity at both widths; scalar-sample batching; target contiguity,
  dtype, and read-only flag; the exact §9.4 commit point; **a failure
  injected at `_deliver_batch`, asserting the full rollback and that a
  retry yields the same indices and the same values**; supersession, and
  its refusal during a transaction; the §9.5 refusal matrix; exhaustion;
  close during a transaction; abandonment in each of its four distinct
  positions; per-failure cleanup with live-storage baselines; the NumPy
  tripwire.
- **Invariants:** no checkpoint integration; no C++; no export. **The
  committed position advances if and only if a batch was delivered**, and
  that is asserted at every failure position rather than argued.
- **Exclusions:** loader state serialization, checkpoints, examples.
- **Exit gate:** the §9.5, §9.6, §10.6, and §17.3 tables asserted row by
  row; the delivery-failure rollback proved with a non-vacuity control;
  `__all__` grows by exactly one; 54 exports.
- **Outcome:** met. `experimental/native_data_loader.py` ships
  `NativeDataLoader`, exported from `tensorforge.experimental`, whose
  `__all__` went from 24 names to 25: `NativeDataLoader` and no other.
  The module imports **exactly one name** — `NativeBatchSampler` — so it
  reaches no ctypes layer, no NumPy, no checkpoint, and no generator, and
  it constructs no lock, thread, queue, or worker. `_NativeBatchIterator`
  and `_deliver_batch` live in that module, are defined nowhere else, and
  are exported by nothing.

  **The §9.4 transaction is asserted at every failure position rather
  than argued**, each by injection with its own non-vacuity control and
  an exact before/after comparison of the position, the whole
  `state_dict()`, the captured countdown, and the native live-storage
  baseline: a claim failure before publication (which mints no serial), a
  feature-materialization failure, a **real native allocation failure**
  through the existing `tf_test_arm_alloc_failure` hook, a target-gather
  failure after the feature tensor exists, a publication failure, a
  commit failure injected into the structurally non-failing write seam,
  and the **delivery-seam failure at both dtypes** — after which the very
  next `__next__` returns the **same indices and the same values** in a
  freshly allocated tensor. Reentrant `iterator.close()` and
  `loader.close()` are driven from *inside* the seam, so a reentrant
  arrival is real rather than simulated, and the exact-match completion
  check refuses to hand back a batch a reentrant close already rolled
  back. Stale, foreign, and never-minted serials are proved to match
  nothing against a live newer transaction, and a rollback invoked four
  times is proved idempotent.

  The C ABI stayed at 54 exports, the CTests at 24, the examples at 15,
  the benchmarks at 8, the checkpoint at version 3 with `(1, 2, 3)`
  accepted, and the optimizer state at version 1; no C++ or CMake file
  was touched, and no loader state schema, format tag, or checkpoint
  integration was added.

**Implementation clarifications recorded at J3**, none of which changes a
locked rule — the §23 discipline is to record rather than rewrite:

1. **The dataset's open/closed check runs at §9.4 Phase 1 rather than
   inside Phase 2.** §9.4 step 1 already lists "the dataset is open"
   among the lifecycle validations, and §9.6's row places the resulting
   `RuntimeError` in Phase 2; both describe the same observable outcome —
   a `RuntimeError`, nothing allocated, nothing advanced — so the check
   is made where the lifecycle validations are, *before* a claim exists.
   The stated post-state is strictly stronger: no claim is ever published
   for a closed dataset, so there is none to clear. `feature_batch`'s own
   `RuntimeError` remains the second authority and is not removed.
2. **`sampler` is checked with `isinstance`, not an exact type**, for
   J2's recorded reason one level down: the loader reads only the
   sampler's documented surface, which any subclass inherits intact.
3. **The iterator releases its resource half unconditionally on every
   unsuccessful path**, while the *sampler* half stays strictly
   exact-match. An iterator owns at most one transaction's resources at a
   time and the only route that detaches them without releasing them is
   Phase 6, which hands them to the caller — so "release whatever this
   iterator still holds" can never reach a delivered batch, and it is
   what closes the tensor when a **reentrant** `close()` rolled the
   record back and the interrupted call then re-attached a freshly
   constructed one. Without it that one tensor would leak.
4. **`_commit_pending` marks the record committed *before* it writes.**
   The write is structurally non-failing, so the order is unobservable in
   production; it matters only under the J3 injection that makes it fail
   anyway, where marking first is what keeps the rollback's restore
   correct instead of concluding that nothing had been applied.
5. **The transaction serial advances at the claim, not at publication.**
   `NativeGenerator` advances its serial at publish, so a discarded claim
   reuses one; §9.4 says each claim receives a serial "never reused for
   the lifetime of the sampler", and advancing at the claim is the
   literal reading. A failed *delivery* therefore consumes a serial and a
   failed *claim* does not — serials are opaque, and skipping one costs
   nothing.
6. **`_commit_pending` is the private spelling of §3.2's committing
   half.** §3.2 lists `_assign_position`/`_snapshot_position`; J2 shipped
   the equivalent `_assign_state`/`_snapshot_state` and J3 reuses them
   rather than adding a second name for one seam, so the candidate
   commit and the rollback restore genuinely share one write path.

### J4 — Loader state and mid-epoch resume — **complete**

- **Entry:** J3 merged.
- **Scope:** §11.3's loader state, §12.5's transactional load, and the
  in-memory mid-epoch resume: a second loader restored from a first
  reproduces the exact remaining batch sequence.
- **Tests:** `tests/test_native_loader_state.py` — schema shape and exact
  key sets; every rejection in §12.4 and §12.5 with its precedence; the
  §11.5 no-normalization rules; mid-epoch, epoch-boundary, and
  final-batch restoration; dataset-mismatch rejection on each of the four
  identity fields independently; configuration adoption; identity
  preservation; refusal when closed or iterating.
- **Invariants:** no checkpoint coupling; no C++; no export; no new public
  name.
- **Exit gate:** exact remaining-sequence equality proved from a restored
  loader; every rejection leaves a byte-identical state.
- **Outcome:** met, and the milestone added **no public name at all** —
  `tensorforge.experimental.__all__` stayed at **25**, the first Phase-J
  runtime milestone whose export delta is zero. `NativeDataLoader` gained
  exactly two methods, `state_dict()` and `load_state_dict(state)`, over
  four private module constants (`_FORMAT`, `_FORMAT_VERSION`,
  `_SUPPORTED_FORMAT_VERSIONS`, `_STATE_FIELDS`) that are exported by
  nothing and are not a registry. The state is §11.3's schema exactly:
  **three** root keys around the **unchanged** §11.2 sampler object, with
  no configuration or position field duplicated at the wrapper's root.

  **The nested validation is delegated rather than restated.** The loader
  validates its own wrapper — container, exact key set, tag, version,
  nested container type — and then hands the whole inner object to the
  sampler's existing validation-only seam `_validate_state`, committing
  through the existing non-failing `_assign_state`. Both already existed
  at J2 and neither changed, so **no sampler production logic moved**:
  J4's only sampler edit is two docstring lines that said "once J3
  exists" of a wrapper that arrives at J4. That the delegation is real is
  asserted structurally — the loader's two state methods are proved to
  name **no** nested schema key as a string literal, against a negative
  control showing the same scan finds them in the sampler.

  **Every §12.5 guard is proved by precedence, not by claim.** The closed
  guard, the transaction guard, and the active-iteration guard are each
  driven with deliberately malformed arguments (`None`, `[]`,
  `{"bad": "state"}`), so "the guard ran before the state was inspected"
  is evidence. The transaction guards are driven from *inside* a live
  handoff through the private `_deliver_batch` seam at both the claim and
  the pending phase, where `state_dict()` is proved to refuse **while the
  sampler's raw fields already show the candidate position** — the
  executable form of "no snapshot may observe a skipped-but-undelivered
  position". A superseded iterator is proved to block a load until it
  releases.

  **Every rejection is compared against a complete before/after world
  fingerprint** — loader, sampler, and dataset identity, the closed
  states, all six configuration and position values, both `state_dict()`
  values, the next batch, the permutation, the plan, the iterator slot,
  the transaction record, the active-iteration set, and the native
  live-storage count — across nineteen fault classes covering the
  wrapper, the nested schema, the four dataset-identity fields, and the
  zero-batch joint rule.

  **The exit gate is proved over two genuinely separate object graphs**:
  a source dataset/sampler/loader interrupted mid-epoch, and a
  **separately constructed** target whose sampler is deliberately built
  with a different batch size, a different shuffle setting, a different
  seed, and a fresh position. After the load the target adopts all six
  values, keeps its own loader, sampler, and dataset objects, and
  reproduces the remaining tail exactly — identical index tuples,
  identical **raw IEEE-754 feature bits** through `uint32`/`uint64`
  views, identical `int64` targets with matching dtype, shape,
  contiguity, ownership, and read-only flag — then the same canonical
  next-epoch position and the same two following whole epochs, with
  native live storage returning exactly to baseline. **No tolerance is
  used anywhere.** It runs at float64 and float32, sequential and
  shuffled, drop-last false and true, and at every required position:
  fresh, genuine mid-epoch, final batch, epoch boundary, later epoch,
  short final batch, exact divisibility, one-batch epoch, one-sample
  dataset, and batch larger than the dataset. A negative control proves
  the sequences are **unequal** when the restoration is omitted, and the
  cross-dtype leg proves batch **indices** identical across equivalent
  float32 and float64 datasets while the two states remain
  non-interchangeable in both directions.

  No checkpoint production coupling exists in either direction, asserted
  by source inspection and by driving a real save and load with the
  loader's two methods patched to record any call: neither fired. The C
  ABI stayed at 54 exports, the CTests at 24, the examples at 15, the
  benchmarks at 8, the checkpoint at version 3 with `(1, 2, 3)` accepted,
  and the optimizer state at version 1; no C++ or CMake file was touched.

**Implementation clarifications recorded at J4**, none of which changes a
locked rule — the §23 discipline is to record rather than rewrite:

1. **The sampler needed no new seam.** §12.5 step 5 asks for "a
   *validation-only* call that mutates nothing and returns the six
   values", and §12.4's commit for "the sampler's non-failing
   assignment". J2 already shipped exactly those as `_validate_state` and
   `_assign_state`, and J3 already reuses the second in both directions.
   So J4 added no private sampler method, changed no sampler signature,
   and moved no sampler logic — which is the strongest available form of
   "one authority, not two".
2. **The loader shares the two schema-shaped rules rather than restating
   them.** `native_data_loader` imports `_require_exact_int` and
   `_require_exact_keys` beside `NativeBatchSampler`, on J2's recorded
   precedent for `_validate_uint64`: a duplicated exact-`int` or
   exact-key-set rule would be a second spelling of one convention, free
   to drift from the one the nested schema is held to. The module's
   pinned import set therefore grew from one name to three, all from the
   **same** module, so it still reaches no ctypes layer, no NumPy, no
   `json`, no checkpoint, and no generator.
3. **`state_dict()` takes the sampler's snapshot first, and that call is
   also the guard.** §9.5 requires a refusal mid-transaction, and
   `NativeBatchSampler.state_dict` already performs exactly that check.
   Calling it first means the loader has **one** transaction authority
   rather than a second that could disagree, and it means nothing at all
   is built when the refusal fires. The error therefore names the
   sampler; no message text is a contract, and §12.1 already says so.
4. **The `feature_shape` tuple latitude reaches the nested object only.**
   §11.2's single exception is a property of the sampler's dataset block,
   and delegation carries it through unchanged. The **wrapper** takes no
   such latitude in either position: its root must be an exact `dict` and
   so must `state["sampler"]`, and a tuple of items is refused at both.
5. **Two existing J3 assertions moved from absence to presence.** The
   loader's own module previously asserted `state_dict`,
   `load_state_dict`, and the format constants **absent**, and this
   phase's guardrail module did the same. Both now assert them present
   and unchanged in shape, with the checkpoint-absence half left exactly
   as it was — the milestone that ships a name is the milestone that
   moves it, and neither edit weakened anything else.

### J5 — Native checkpoint metadata integration — **complete**

- **Entry:** J4 merged.
- **Scope:** the §13 workflow, proved end to end against a real version-3
  archive, with **fresh** model, optimizer, generators, dataset, sampler,
  and loader.
- **Tests:** `tests/test_native_data_checkpoint.py` — the round trip;
  missing, malformed, wrong-dataset, and non-JSON metadata; the exact next
  batch after restore; the §13.5 ordering and the explicit absence of
  cross-object atomicity; the §13.6 non-coupling asserted by source
  inspection in both directions.
- **Invariants:** checkpoint version stays 3, accepted stays `(1, 2, 3)`,
  no root field, no import between checkpoint and pipeline modules, no
  automatic discovery.
- **Exit gate:** the fresh-object exact-next-batch proof green; the
  checkpoint module provably unchanged.
- **Outcome:** met, and the milestone's most important property is what it
  did **not** contain. J5 added **no production code whatsoever** — no
  module, no class, no method, no argument, no constant, no export, and no
  line of `src/`. `tensorforge.experimental.__all__` stayed at **25**, the
  second consecutive Phase-J milestone whose export delta is zero, and
  `src/tensorforge/experimental/native_checkpoint.py` is byte-identical to
  its J4 state. The whole milestone is
  `tests/test_native_data_checkpoint.py` plus documentation, which is the
  strongest available form of "the two halves already composed".

  **The workflow is proved against a real archive, never a stand-in.**
  Every primary proof writes an actual `.npz` through
  `save_native_checkpoint` and reads it back through
  `load_native_checkpoint` with `allow_pickle=False`. The manifest is
  inspected directly: `format` unchanged, `format_version` **3**, the same
  **six** root keys, and an array inventory containing only the manifest,
  `model::…`, and `optimizer::m::…`/`optimizer::v::…`. No array name
  contains a Phase-J word; no loader tag appears in the `model`,
  `optimizer`, or `generators` sections; and saving with and without
  loader state produces the **same** array inventory and manifests that
  differ in nothing but the caller's own `metadata` value. **The capture
  set did not grow by one field.**

  **The restoration is exact, into objects that share nothing with the
  saving graph.** The proof runs at float64 and float32, sequential and
  shuffled, drop-last false and true, over a model carrying trainable
  parameters, two persistent batch-norm buffers, and a **shared** generator
  alias topology — two Dropout layers on one `NativeGenerator` and a third
  on its own, a fact no per-generator state carries. The restored graph is
  deliberately built **wrong** in every family first: different parameter
  seeds, a different learning rate, different generator seeds, a
  separately constructed dataset, and a sampler with a different batch
  size, shuffle setting, seed, epoch, and cursor. After the two calls,
  every model parameter and persistent buffer, every Adam `m`, `v`, and
  step counter, every hyperparameter, every generator's algorithm,
  version, seed, and call count, the alias topology, and the six loader
  values compare **exactly** — raw IEEE-754 bit patterns through
  `uint32`/`uint64` views, exact `int64` targets with their dtype, shape,
  contiguity, ownership, and read-only flag. **No tolerance is used
  anywhere.** Object identity is preserved absolutely: the load constructs
  no generator, parameter, or buffer, and the loader keeps its own sampler
  and dataset. A negative control proves the continuation **differs** when
  the loader restoration is omitted and **agrees** the moment it is
  applied, so the gate cannot pass vacuously; the interruption point is
  chosen so the resume genuinely lands mid-epoch.

  **All three §13.7 delivery boundaries are proved through an archive.** A
  delivery failed at the `_deliver_batch` seam — with a non-vacuity record
  proving the seam ran *after* the candidate position had been applied and
  that `state_dict()` refused there — rolls back completely, and the
  checkpoint taken immediately afterwards resumes from the **same
  candidate batch**, delivering exactly those indices and those bits and
  advancing exactly once. A successful delivery resumes from the
  **following** batch with no replay. A save at an epoch boundary records
  the canonical `(epoch + 1, 0)` and resumes at the first batch of the next
  epoch, with no terminal or end-of-epoch representation anywhere.

  **The metadata boundary is proved to be the caller's.** `"training"`,
  `"data_loader"`, and `"next_step"` are conventions this repository
  speaks consistently and **no runtime code knows**: alternate nesting,
  alternate names, and two loaders' states side by side all round-trip
  unchanged, and the checkpoint module is asserted to contain none of
  those strings as a literal. Absent loader state yields `None` and no
  default; a malformed but JSON-compatible loader state is **preserved by
  the archive and rejected by the loader**, transactionally, across ten
  fault shapes; a wrong-dataset state is rejected on identity; non-JSON
  metadata is refused before the destination moves, leaving an existing
  archive byte-identical with no temporary file and every live object
  untouched. Returned metadata is a fresh, independent plain dict at every
  load.

  **The ordering and the honest atomicity boundary are both executable.**
  The loader snapshot precedes the save with no delivery in between,
  proved as an ordered event log and confirmed by the next delivery using
  exactly the indices the archive recorded. A failed checkpoint load is
  proved to leave a complete loader fingerprint untouched and to call
  neither loader state method. And the case the design refuses to
  paper over is asserted directly: a checkpoint load that **succeeds**
  followed by a loader load that **fails** on dataset identity leaves the
  model, optimizer, and generators restored and the loader unchanged —
  **nothing rolls back, because there is no cross-object transaction** —
  after which the documented recovery, discarding everything and repeating
  both calls from the same unchanged archive, succeeds.

  Non-coupling is asserted in both directions by AST inspection rather
  than by substring, and at runtime by patching the loader's two state
  methods to record any call during a real save and a real load: neither
  fired, and the caller's own two lines then fired exactly one each. The C
  ABI stayed at 54 exports, the CTests at 24, the examples at 15, the
  benchmarks at 8, the checkpoint at version 3 with `(1, 2, 3)` accepted,
  and the optimizer state at version 1; no C++, CMake, or ABI file was
  touched, and native live storage returns exactly to its baseline in
  every archive and continuation proof.

**Implementation clarifications recorded at J5**, none of which changes a
locked rule:

1. **The milestone found no production defect.** J5's instruction was to
   stop and report rather than silently repair, and there was nothing to
   report: every §13 statement held against the shipped runtime on the
   first attempt. The checkpoint's metadata validator already accepted
   every §11 field — verified at J0 (§23.1) and now exercised through real
   archives — and the loader's §12.5 ordering already rejected everything
   an archive could carry back.
2. **The fresh loader's differing start position is set through the public
   state-loading route**, never by assigning a private sampler field. §12.4
   adopts all six configuration and position values, so loading a
   deliberately edited state is itself an authoritative way to construct a
   different valid starting position — and it exercises the same seam the
   proof later depends on.
3. **The two absence assertions that moved are named.**
   `tests/test_native_loader_state.py` asserted
   `tests/test_native_data_checkpoint.py` absent, and this phase's
   guardrail module asserted the checkpoint workflow unproved. Both now
   assert the landed half present, with every other entry in the same
   lists left exactly as it was — the milestone that ships a thing is the
   milestone that moves it.

### J6 — Deterministic mini-batch training example — **complete**

- **Entry:** J5 merged.
- **Scope:** `examples/native_minibatch_training.py` — a meaningful float32
  model trained over shuffled mini-batches with `NativeAdam` and
  `NativeCrossEntropyLoss`, run uninterrupted and interrupted-and-resumed,
  compared by §14's exact contract, with explicit cleanup and no downloaded
  data. A float64 path is included where §14.4's cross-dtype index equality
  makes it informative.
- **Tests:** `tests/test_native_minibatch_training.py` — imports `train`
  and asserts the proof, the negative controls of §14.2, and the
  live-storage baseline.
- **Invariants:** uses only public APIs; no private `_typed`, `_from_core`,
  or `_native_permutation` call in executable code; no new capability.
- **Exit gate:** examples 15 → **16**; exact equality at every §14.3 row;
  the negative controls fail as required.
- **Outcome:** met, and — like J5 — the milestone added **no production
  code at all**: no module, class, method, argument, constant, or export,
  and no line of `src/` or `cpp/`.
  `tensorforge.experimental.__all__` stayed at **25**, the third
  consecutive Phase-J milestone whose export delta is zero, and the example
  inventory moved 15 → **16** while the benchmarks stayed at **8**.

  **The example is an ordinary training program, written entirely against
  the public surface.** `Linear(6→8) → BatchNorm1d(8) → ReLU →
  Dropout(0.25) → Linear(8→8) → LayerNorm(8) → Dropout(0.25) → Linear(8→3)`
  into `NativeCrossEntropyLoss` with `NativeAdam`, over a
  `NativeTensorDataset` → `NativeBatchSampler` → `NativeDataLoader` chain:
  24 samples of 6 features in 3 classes, `shuffle=True`, `batch_size=6`,
  `drop_last=False`, seed `20260803`, four batches per epoch, ten steps
  interrupted after five. **Two Dropout layers share one
  `NativeGenerator`**, so the model carries a real alias topology beside its
  parameters, its two BatchNorm running buffers, and Adam's moments and
  counters — every TensorForge-owned state family is load-bearing at once.
  That the example touches **no** private runtime seam is asserted by an
  **AST** scan rather than a substring ban, with a negative control proving
  the scanner catches a planted `_epoch` assignment, a `_deliver_batch`
  call, and a `_trusted_dtype=True` keyword — the last of which is only
  ever reachable *as a keyword*, so the scanner collects those too.

  **The interruption is genuinely mid-epoch, and the schedule is pinned as
  literal expected values.** After five completed steps the saved position
  is `(epoch 1, cursor 1)` — not zero, not the final step, not an epoch
  boundary, with **three** batches still owed by the active epoch and a
  five-step resumed suffix. The run crosses **two** epoch boundaries across
  three exercised epochs whose permutations are non-identity and mutually
  distinct. The three permutations, the twelve batch-index groups, the
  ten before/after positions, and the canonical final `(2, 2)` are written
  on the **test** side as committed literals, so a change to the seed, the
  batch size, the sweep direction, or the key schedule fails there rather
  than silently redefining what the proof proves.

  **The supported ordering is observed, not reconstructed.** `train()`
  appends a `("deliver", step)` entry to a journal as each batch actually
  arrives, and the proof appends its own `loader.state_dict()` and
  `save_native_checkpoint` entries at the real call sites; the journal's
  tail is then asserted to be exactly *deliver step 4 → snapshot → save*,
  which is the executable form of "no delivery happened between the
  snapshot and the archive". `next_step` is `SPLIT_STEP`, and one delivered
  batch is one completed step, so the two cannot drift by one.

  **The restore target is deliberately built wrong in every family, and
  proved so before the load**: different parameter seeds, a different
  generator seed, a different learning rate, a separately constructed
  dataset, and a loader with a different seed, a different batch size, a
  different shuffle setting, and a different position — reached by really
  delivering batches through the public iteration path, never by assigning
  a private field. After `load_native_checkpoint` and then
  `loader.load_state_dict`, every parameter, persistent buffer, Adam `m`,
  `v`, step counter, hyperparameter, generator state, alias topology, the
  six loader values, every feature batch's raw bits, every `int64` target
  array with its dtype, shape, contiguity, ownership and read-only flag,
  every loss, every logits tensor, and the final evaluation compare
  **exactly**. **No tolerance is used anywhere**, and object identity is
  preserved absolutely — the load constructs no generator, parameter, or
  buffer, and the loader keeps its own sampler and dataset.

  **The negative controls really fail.** Omitting `loader.load_state_dict`
  alone — everything else identical — is proved to give a different next
  batch, a different remaining index sequence, different losses, different
  parameters, and a different evaluation. The bit helper is proved to
  separate `+0.0` from `-0.0` and adjacent values at both widths, to refuse
  a wrong-width array, and to perform no conversion. The training-state
  claims are backed by state that moved: parameters, both running buffers,
  Adam moments and counters, the generator's call count, a non-constant
  loss sequence, and a changed evaluation output over identical inputs.

  **Cross-dtype, only the dtype-independent facts are compared** — the
  complete batch-index sequence, the permutations, the position
  progression, the next batch at the interruption, and the final loader
  position. Losses, logits, parameters, buffers, moments, and evaluation
  outputs are compared **only within their own dtype**, and the two dtypes
  are asserted to produce genuinely *different* bits, so a cross-dtype
  numeric equality would be false rather than merely unasserted.

  Native live storage returns **exactly** to baseline across the whole
  workflow, with every delivered feature batch, logits tensor, loss,
  gradient, evaluation tensor, parameter, buffer, optimizer, loader, and
  dataset closed explicitly; the checkpoint lives in a temporary directory
  that is removed automatically, and running the script leaves no file
  behind. The C ABI stayed at 54 exports, the CTests at 24, the benchmarks
  at 8, the checkpoint at version 3 with `(1, 2, 3)` accepted, the
  optimizer state at version 1, and the loader and sampler state formats at
  version 1; no C++, CMake, ABI, or dependency file was touched.

**Implementation clarifications recorded at J6**, none of which changes a
locked rule — the §23 discipline is to record rather than rewrite:

1. **The milestone found no production defect.** J6's instruction was to
   stop and report rather than silently repair, and there was nothing to
   report: the public surface J1–J5 shipped expressed the whole training
   workflow on the first attempt, and every §14 row was provable without a
   single private call.
2. **Evaluation goes through the dataset, not the loader.** §14.3 asks for
   final logits, predictions, and evaluation output; the example gathers
   the full dataset with the dataset's own public
   `feature_batch`/`target_batch` pair rather than iterating the loader.
   Evaluation has no position, no shuffle, and no epoch — it is one fixed
   gather of every row in index order — and routing it through the loader
   would advance a cursor that describes *training*. Nothing about §10's
   materialization contract changes; the same two methods are used.
3. **The example holds its retired objects only to keep `id()` honest.**
   §14.2 requires the resumed graph to share no object with the saving one.
   CPython recycles `id()` values, so measuring that against objects
   already collected would compare reusable addresses. The example
   therefore `close()`s every native resource of the interrupted run at the
   moment the archive is written — which is the ownership contract — and
   holds the emptied Python objects for exactly as long as the identity
   comparison runs, then clears them. None is ever passed into the restored
   graph, and none owns native storage by then.
4. **The one-epoch iterator rollover is the caller's, and it is two
   lines.** §9.3 makes one iterator one epoch, so a loop that outruns the
   captured countdown answers `StopIteration` by calling `iter(loader)`
   again and continuing at the canonical next-epoch position. Nothing
   resets or rebuilds the sampler, and no epoch or cursor is incremented by
   hand. The example's `train()` closes its iterator explicitly on the way
   out, so the active-iteration count is released there rather than by a
   finalizer — which is also what makes `load_state_dict` legal again
   immediately afterwards.
5. **`_LiveStorageMeter` is a measurement instrument, on the I9
   precedent.** The runtime exposes no live-allocation counter, and the
   §14.5 baseline claim has to be measured rather than asserted, so
   `main()` wraps the **public** `cpp.NativeStorage.__init__`/`close` to
   count open storages and restores both in a `finally`. It changes nothing
   about what either does, is not a private seam, and is not the release
   mechanism — explicit `close()` is, and the counter merely observes it.
6. **The failed-delivery boundary stays J5's and J7's.** §14.1's third leg
   requires injection at the private `_deliver_batch` seam. The public
   example must not do that and does not; J5 already proved through a real
   archive that a failed delivery consumes nothing and resumes from the
   same candidate batch, and J7 owns the complete matrix. J6 neither
   weakens nor restates that proof.

### J7 — Cross-cutting hardening — **complete**

- **Entry:** J6 merged.
- **Scope:** the adversarial matrix — malformed state at every field and
  every type; alias boundaries; allocation failure at every §17 row;
  **failure injected at the `_deliver_batch` seam, and at the commit step
  before it**, each proved to leave epoch, cursor, `state_dict()`, and live
  storage exactly as they were; **a checkpoint taken immediately after a
  failed delivery, proved to resume from the same candidate batch**;
  abandonment in each of its four positions; repeated iteration; close
  during a transaction; reentrancy through the §9.5 refusal matrix; the
  §16 concurrency contract asserted as *documented and unprotected* rather
  than as safety; checkpoint failure; rollback; live-storage baselines;
  stable/native isolation.
- **Tests:** `tests/test_native_data_hardening.py`, with a complete
  before/after world fingerprint after every rejection and its own
  non-vacuity control, following I10.
- **Invariants:** evidence only. Any production change J7 finds necessary
  is a **defect repair**, is narrow, and is recorded as such — the I10
  precedent, where exactly one loader-validation gap was repaired and
  everything else was evidence.
- **Exit gate:** every §12.7, §15, §16, and §17 row asserted.

#### J7 outcome

**No production defect was found, and no production code changed.** The
whole diff is `tests/test_native_data_hardening.py`, the narrow inventory
edits that move that file from absent to present in the four modules that
asserted its absence, and documentation. Every §3 capability row, both
state schemas, the checkpoint format and its accepted versions, the
optimizer-state version, the 54 exports, the 24 CTests, the 16 examples,
the 8 benchmarks, and the 25 experimental names are unchanged.

What the matrix established, beyond re-asserting J3–J6:

- **Every §17.2 construction row** by injection — validation before any
  snapshot, a failure between the two snapshots, and a failure at the
  digest — each proved to leave **no reference alive**, read from the
  raised exception's own traceback rather than argued.
- **Every §17.3 iteration row**, with the four Phase-2 failures kept
  genuinely distinct: the **host gather** (nothing native reached), the
  **native allocation** (the backend's own thread-local arm, disarmed in
  a `finally`), the **host→native transfer** (the storage existed and
  `from_array` closed it), and the **target copy** (the feature tensor
  existed, was open, and was closed before the exception escaped).
- **The commit step made to fail *after* the candidate was applied.** J3's
  injection raises instead of the assignment and so exercises a rollback
  from a position that never moved; J7's runs the assignment first, so the
  restore path is exercised with a position that really did move. The
  rollback's contracted **order** — restore, then clear, then close — is
  observed directly.
- **A `BaseException`** that is deliberately not an `Exception`, at both
  the commit and the seam, proving the `finally` unconditional.
- **The reentrancy matrix at three phases**, not two: claim, pending
  (record published, position not yet applied), and committed. Every
  refused operation raises `RuntimeError` while the original transaction
  survives untouched, and malformed load arguments are provably not
  inspected because the same arguments are a `TypeError` on an idle
  loader.
- **A checkpoint taken immediately after a failed delivery**, at both
  dtypes, through a real version-3 archive, restored into an entirely
  fresh and deliberately differently-configured graph, delivering the
  exact failed candidate once with bit-identical features and targets.
- **The §16 boundary asserted as a boundary**: no lock, thread, queue,
  future, or async primitive exists in any Phase-J module (read from the
  AST, so prose documenting the prohibition cannot satisfy it), the
  objects join no lock order, and the documentation says so in terms a
  parser checks — with a control proving the parser rejects a §16 that
  dropped or reversed the statement. **No test starts a thread and none
  claims a race is safe.**

Two contracted exceptions to "nothing moved" are asserted **explicitly**
rather than excluded quietly, because both are the never-reused rule
doing its job: a **failed delivery advances the serial counter**, and a
**failed iterator creation advances the participation-token counter**.
Neither is ever handed out again.

One structural fact was recorded rather than injected. There is no
"failed after the claim was published but before Phase 2" position to
test: `_claim_batch` writes `self._transaction` as its last statement
before `return`, and the only statements between that return and
`__next__`'s guarded block are a slot assignment and a local binding,
neither of which can raise. J7 asserts that shape from the AST, with a
negative control, instead of manufacturing a failure the runtime cannot
produce.

### J8 — Performance and transfer characterization — **complete**

- **Entry:** J7 merged.
- **Scope:** `benchmarks/benchmark_native_data_pipeline.py` — dataset
  indexing, batch planning, permutation construction, and host→native
  materialization, with **float32 and float64 measured separately and
  never as a ratio of one to the other**.
- **Tests:** `tests/test_native_data_benchmark.py` — the correctness gates,
  the case inventory, the CLI contract.
- **Invariants:** correctness gated **before** any timing; no speed
  asserted anywhere; no threshold, budget, or CI timing job; **no result
  file**; a case with no honest equivalent labelled `native_only` and
  publishing no ratio; medians with spread after warm-up; setup and
  cleanup outside the timer.
- **Exclusions:** optimization. A change motivated by a J8 measurement is a
  separate, separately reviewed decision — and §22.3 governs the only one
  that could touch the ABI.
- **Exit gate:** benchmarks 8 → **9**; no timing assertion anywhere in the
  repository. **Met** — see §23.2.

### J9 — Integration and closure — **not started**

- **Entry:** J8 merged.
- **Scope:** the full validation matrix (§22.4), final documentation and
  support-matrix reconciliation, artifact and ABI inventories, the exact
  deterministic resume proof re-run, and the closure guardrails.
- **Tests:** `tests/test_native_phase_j_closure.py`, in the shape of
  `tests/test_native_phase_h_closure.py` and
  `tests/test_native_phase_i_closure.py`, **every parser with a negative
  control**.
- **Exit gate:** §24.2.

### 23.1 Ladder adjustments made at J0

None. The ladder as approved survived repository inspection unchanged. Two
premises were **checked rather than assumed**, and both held, so no
milestone had to move:

- **J2's derivation is implementable with no new export and no new RNG
  algorithm.** Verified at J0 by reproducing the shipped C++ kernel's
  output from a Python implementation of the same finalizer (§8.8).
- **J5's metadata channel already carries every §11 field.** Verified at J0
  against `_validated_metadata`, which accepts `bool`, arbitrary-precision
  `int`, `str`, `list`, and `str`-keyed `dict`, and has run on both the
  save and the load side since I10 (§2.6).

One design decision was **made on inspected reality rather than
convenience**, and is recorded here because it constrains J1 and J2 rather
than J0: the native runtime cannot represent a zero-element tensor, so
empty datasets (§4.6) and zero-batch epochs (§7.5) are rejected at
construction instead of being carried as degenerate states.

### 23.2 J8 outcome — what was measured, and what was deliberately not

**J8 is characterization only. It shipped no optimization, no production
change, no public name, and no export.** Its whole diff is
`benchmarks/benchmark_native_data_pipeline.py`,
`tests/test_native_data_benchmark.py`, the narrow inventory edits landing
them requires, and documentation. Nothing below may be turned into a
threshold, a promise, or a justification for a runtime change: §22.3
governs the only measurement-driven decision that could ever touch the
ABI, and it is a **separately approved** milestone of its own, not
something J8 may take.

**Identity.** `BENCHMARK_NAME = "tensorforge.native_data_pipeline"`,
`BENCHMARK_VERSION = "1.0"`, `SCHEMA_VERSION = 1`. These name the
measurement payload; they are not package exports and no benchmark
registry exists inside `tensorforge`.

**The exact case inventory — 20 cases in five workload families, each
run separately at float64 and float32.** The first four families are the
four questions J8 undertook to answer, kept apart on purpose; the fifth
is a composition and never a substitute for them.

| Workload | Cases |
|---|---|
| `dataset_indexing` | `host_feature_gather_sequential`, `host_feature_gather_shuffled`, `host_feature_gather_duplicates`, `dataset_target_batch_sequential`, `dataset_target_batch_shuffled` |
| `batch_planning` | `plan_sequential_exact`, `plan_sequential_short_final`, `plan_shuffled_reference`, `plan_shuffled_large`, `next_batch_indices_fresh`, `next_batch_indices_mid_epoch` |
| `permutation_construction` | `permutation_cold_reference`, `permutation_cold_later_epoch`, `permutation_cold_large`, `permutation_cache_hit` |
| `host_to_native_materialization` | `feature_batch_small`, `feature_batch_large`, `feature_batch_shuffled`, `feature_batch_image` |
| `loader_delivery` | `loader_next_batch` |

**Reference decisions, and why.** Exactly two reference labels exist,
`numpy` and `native_only`, and every case declares one:

- The five `dataset_indexing` cases are **host-only** — they allocate no
  native storage at all — so an independently written NumPy expression
  over the identical snapshot, indices, dtype, and output shape *is* an
  honest same-operation reference, and each publishes a ratio together
  with an explicit `ratio_meaning`. For the feature-gather cases the
  reference is a second spelling of one NumPy gather, so the ratio is a
  NumPy-internal observation and the payload says so rather than letting
  it read as a TensorForge-versus-NumPy comparison. For the two
  `target_batch` cases the reference is the same gather, copy, and
  read-only publication written without the dataset's index validation,
  so the ratio is exactly that validation and dispatch.
- **Every other case is `native_only` and publishes no ratio at all.**
  Planning has no NumPy counterpart, and inventing one would time code
  the project does not ship. Permutation construction is a different
  algorithm under a different generator with a different contract, so a
  ratio would divide two operations that cannot produce the same answer.
  Materialization allocates and transfers where a NumPy gather does not,
  and §9's dividing rule refuses that comparison explicitly — which is
  precisely why the gather alone is measured honestly in
  `dataset_indexing` instead. Delivery is a transaction with no
  reference implementation.

**Correctness gates, all exact and all before timing.** Index tuples,
plans, and permutations by equality; feature values by raw IEEE-754 bits
**within one dtype**; targets by exact `int64` equality; dtype, shape,
device, ownership, contiguity, freshness, and the read-only flag by
identity. No `allclose`, no `pytest.approx`, and no tolerance appears
anywhere in the harness — every operation this pipeline performs is a
copy, a gather, or integer planning. The permutation and plan gates for
the length-8 configurations are **known-answer** checks against §8.9's
committed vectors; the larger configurations additionally prove the order
is a permutation, is a pure function of `(seed, epoch, length)`, and is
not the identity. A failed gate publishes no timing row, exits nonzero,
leaves stdout clean, and still releases everything the case allocated.

**Timing methodology.** `time.perf_counter_ns()`; one measured sample is
exactly one call; datasets, samplers, loaders, index sets, iterators,
restored positions, and cache warming are all built **outside** the
measured region, per repetition where the call advances state; every
native tensor is closed explicitly outside it. No sample is discarded, no
outlier is removed, and no timer overhead is subtracted. The headline is
the **median**; the spread is the **interquartile range**, published
beside p25, p75, the minimum, the maximum, and every raw sample.
Cold and warm permutation construction are **separate cases** and are
never averaged: a cold case builds a fresh sampler per repetition so its
cache is empty by construction, and the warm case's gate proves the timed
call is a genuine hit because the sampler returns the *same tuple
object*. No cache-control API exists and J8 added none.

**What the local runs showed, as observations only.** On the development
machine, planning and index inspection sit in the sub-microsecond to
low-microsecond range; permutation construction is linear in the sample
count and is the pipeline's one genuinely `O(samples)` per-epoch cost,
while a cache hit is a few hundred nanoseconds; a `feature_batch` is
dominated by the fixed per-call Python-and-ctypes cost at small batch
sizes, which is the architectural floor Phase H already recorded rather
than a defect; and one whole `next(iterator)` delivery is several times a
bare `feature_batch`, which is what the composition case exists to show.
**These are one machine, one build, and one moment.** They are not
cross-machine comparable, they are not a contract, no threshold is
derived from them, no CI job asserts one, and **no result file is
written**. The two dtypes are never divided by one another and neither is
ranked.

**What J8 did not find.** Nothing in these measurements meets §22.3's bar
for reopening the export count, and nothing in them is offered as
justification for a runtime change. The answer there remains no.

---

## 24. Exit gates

### 24.1 J0

- [x] This design document exists, is linked from the README, and is
      listed in the documentation map.
- [x] Every required design decision is resolved — §3 API surface, §4
      input contract, §5 ownership, §6 identity, §7 sampler, §8 shuffle,
      §9 iterator, §10 batch, §11 schemas, §12 validation, §13 checkpoint,
      §14 resume, §15 lifecycle, §16 concurrency, §17 rollback, §18
      isolation, §19 dtypes, §20 non-goals, §23 ladder — with no "TBD" and
      no unchosen alternative.
- [x] The deterministic derivation is specified as directly implementable
      pseudocode (§8.5) with every constant, mask, bound, rejection rule,
      and direction stated.
- [x] Fixed reference vectors are committed (§8.9), covering the
      documented empty-dataset rejection and lengths 1, 2, 5, and 8; epoch
      0 and a later epoch; seed 0 and a nontrivial large seed near the
      accepted upper bound; and sequential beside shuffled.
- [x] The state schemas are exact (§11), JSON-compatible, and carry no
      payload.
- [x] The validation order is exact (§12), with stated precedence.
- [x] Ownership and cleanup rules are exact (§10, §15, §17).
- [x] **The batch handoff is specified as an explicit transaction (§9.4)**,
      with claim, construct, publish, commit-and-deliver, rollback, and
      completion phases; a private patchable delivery seam; a never-reused
      reservation identity; and exact-match cleanup.
- [x] **A failed delivery is specified to leave epoch and cursor exactly
      unchanged**, to consume no logical batch position, to close the
      undelivered `NativeTensor`, and to leave a retry returning the same
      indices and the same values.
- [x] **No state snapshot can observe a skipped-but-undelivered
      position** (§9.5): `state_dict()` refuses while a transaction is in
      flight rather than answering ambiguously.
- [x] The checkpoint-metadata workflow is exact (§13), with cross-object
      atomicity explicitly **not** claimed.
- [x] The milestone ladder is complete (§23).
- [x] Status documentation is updated and says Phase J is newly approved,
      J0 is complete, and runtime work has not begun.
- [x] Contract and status tests pass.
- [x] No runtime capability added; no placeholder class, empty module,
      stub, or `NotImplementedError` method.
- [x] No public export added.
- [x] **54** production exports preserved.
- [x] **24** native CTests preserved.
- [x] **15** examples preserved.
- [x] Checkpoint version **3** and accepted `(1, 2, 3)` preserved.
- [x] Optimizer state version **1** preserved.
- [x] `SUPPORTED_DTYPES`, `SUPPORTED_DEVICES`, `UNSUPPORTED`, and
      `RAW_KERNEL_DTYPES` preserved.
- [x] Stable/native isolation preserved.

### 24.2 J9

- [ ] The complete Phase-J runtime: dataset, sampler, loader, state,
      checkpoint workflow, example, hardening, and benchmark.
- [ ] Exact mid-epoch resume proved.
- [ ] The §9.4 delivery transaction proved by injection at the seam: a
      failed delivery consumes nothing, and a checkpoint taken immediately
      after one resumes from the same candidate batch.
- [ ] Exact future permutation reproduction proved, including across
      dtypes (§14.4).
- [ ] Exact uninterrupted-versus-resumed training proved at each dtype,
      each against itself, with negative controls.
- [ ] Windows Release **and** Debug: zero project warnings, 24/24 CTests,
      54 exports, source and library sets equal.
- [ ] Linux CI-equivalent: zero warnings, 24/24 CTests, 54 exports.
- [ ] Clang ASan/UBSan with instrumentation proved present, the full suite
      green, zero diagnostics, and a negative control proving the detector
      works.
- [ ] LeakSanitizer with no suppression file, whose remaining reports
      carry no TensorForge frame.
- [ ] Native live storage returns exactly to baseline across the pipeline
      lifecycle.
- [ ] No ABI drift: 54 exports, unless separately approved under §22.3.
- [ ] Final documentation and support matrix reconciled.
- [ ] GitHub Actions green.
- [ ] No unsupported side capability: registries, checkpoint version,
      optimizer-state version, and the stable surface all unchanged.

---

## 25. What the Phase-J guardrails assert

`tests/test_native_phase_j.py` is the durable contract module. It asserts
**values and structure**, not wording, so ordinary prose improvements do
not require rewriting it, and it derives its premises from the live
registry, the live source, and real files wherever possible.

It pins that this document exists and is linked; that Phase J is presented
as newly approved **after** a completed Phase I rather than as pre-existing
roadmap work; that the ladder runs J0–J9 exactly once each in order, with
**exactly the landed milestones marked complete and every other one marked
not started** — driven from one list, so landing a milestone is a one-line
edit rather than a loosened checker, and carrying **both** an over-claim
and an under-claim negative control so neither direction can rot; that the
design resolves
each load-bearing decision in the section that owns it — the three public
class names and their eventual package, the copied-snapshot rule, the
content fingerprint with its algorithm and endian normalization, the
sampler's ownership of batch size and drop-last, the reuse of
`tensorforge.splitmix64` with no new RNG algorithm and no `NativeGenerator`
coupling, the unbiased bounded-integer rejection rule, the downward
Fisher–Yates direction, the canonical epoch boundary, the exact state
schemas and their format tags, the transactional non-failing commit, the
caller-managed checkpoint-metadata workflow with cross-object atomicity
explicitly disclaimed, and the exact-equality resume contract; that the
reference vectors are present for the required combinations; and that the
non-goals of §20 are all stated.

**It pins the batch-delivery transaction of §9.4 specifically**, because
that is the guarantee the rest of the phase's exact-resume claim rests on:
that the handoff is an explicit transaction with claim, construct, publish,
commit-and-deliver, and rollback phases; that a **failed delivery leaves
epoch and cursor unchanged**; that the **undelivered `NativeTensor` is
closed** by the rollback; that a **retry yields the same batch indices**;
that a **private delivery seam** exists and is named so the failure
position can be tested later; that **no logical batch is consumed before
successful delivery**; and that a **state snapshot cannot observe a skipped
undelivered batch**. Each of those has its own assertion, and the section
carries a negative-control mutation proving the guardrail **fails** if the
document is changed back to the earlier draft's concession — the one this
contract was corrected to remove.

That control is also why this paragraph describes the banned sentences
rather than quoting one: the scanner reads the whole document, so spelling a
concession out here would make the guardrail fail on the very section that
documents it.

**It also pins the unchanged runtime**, against the live module, the live
source, and the built library rather than against prose: `SUPPORTED_DTYPES
== ("float64", "float32")`, `SUPPORTED_DEVICES == ("cpu",)`, `UNSUPPORTED
== ("cuda", "amp")`, `RAW_KERNEL_DTYPES == ("float64",)`,
`normalize_dtype(None) == "float64"`, `backend_info()["dtype"] ==
"float64"`, checkpoint version **3** with `(1, 2, 3)` accepted, optimizer
state version **1**, **54** exports in source and in the built library,
**24** registered CTests, **15** examples, and Phase I still complete.

And it pins **presence and absence** as one split, so a name can only move
from the second set to the first in the milestone that ships it: exactly
the landed classes are exported from `tensorforge.experimental` and appear
in its `__all__`, each defined exactly once and in its own contracted
module, and every unlanded one is absent from the package, from `__all__`,
and from every module under `src/`. `_native_permutation` exists but is
**never exported**, and neither is any helper in it. Nothing Phase J adds
enters the stable public API. No status surface claims a Phase-J runtime
capability that has not landed, calls Phase J complete, or says a data
loader is supported.

Every parser in the module has a **negative control**, so a check that
finds nothing is shown able to find something.
