# Native RNG and Dropout — Phase G architecture contract

**Phase G — Native RNG and Dropout.** This document is the authoritative
architecture contract for the phase. It is written **before** any RNG or
Dropout implementation exists, and milestone **G0** consists of exactly
this document, the status reconciliation it required, and the semantic
guardrails that keep the contract honest. **G0 adds no numerical
behavior**: no generator, no kernel, no C ABI symbol, no ctypes
declaration, no `NativeTensorCore` method, no `NativeTensor` operation,
no module, no export, no registry change, and no checkpoint-format
change.

**Phase-G status: in progress. G0, G1, G2, G3, G4, G5, G6, and G7 are
complete; G8–G10 have not started.** Milestone **G1** shipped `NativeGenerator` and
generator registration on `NativeModule` — random **state** and its
ownership; it generates no random values by itself. Milestone **G2**
shipped the stateless Dropout-forward **Core**: §4's derivation, §7's
kernel, the guarded `tf_core_dropout_forward` export, and the
layer-qualified `NativeTensorCore.dropout_forward` /
`_dropout_forward_with_mask` pair, with committed known-answer vectors
asserted from both C++ and Python. Milestone **G3** shipped §8's
differentiable `NativeTensor.dropout(p, *, generator)` — one autograd
node over that Core, the graph-owned multiplier mask, the backward that
is one `multiply` against it, and §5's reserve → commit / abandon call
transaction — adding exactly one name, `"dropout"`, to `AUTOGRAD_OPS`.
Milestone **G4** shipped `NativeDropout`, the public module over that
operation: stochastic in training, the input object itself in evaluation,
identity at `p == 0`, over one registered `NativeGenerator` it either owns
(the default) or shares (an explicit one, stored as the exact object) —
adding exactly one name, `"NativeDropout"`, to `NATIVE_MODULES` and the
experimental exports. Milestone **G5** shipped §10 in full — native
checkpoint **format version 2**: the `"generators"` manifest section
(`keys`/`entries`/`aliases`) that persists every canonical generator's
state *and* the model's sharing topology, canonical decimal `uint64`
strings, the strict both-directions validation against a real
`named_generators()` traversal, the §10.6 version-1 compatibility rules,
and the §10.7 four-phase **whole-checkpoint** load transaction whose one
rollback guard spans the model, optimizer, and generator commits. It
added the reporting-only name `"checkpoint_generator_state"` to
`STATE_SUPPORT` and nothing else. Milestone **G6** hardened all of it:
§13 and §14 executed as adversarial tests — the reservation transition
matrix, the exact `uint64` boundary, forced concurrent interleavings, the
deterministic Core's structural key properties, every pre-commit and
post-commit failure position of §5's transaction, the four graph-owned
saved-resource families in one graph, a 76-case checkpoint corruption
matrix, whole-transaction rollback at every commit position, save-seam
destination atomicity, and repeated lifecycle loops measured against a
real native live-storage baseline. **G6 added no capability, operation,
module, export, checkpoint field, or checkpoint version** and moved no
registry value; it found and fixed exactly one runtime defect, recorded in
§5. Milestone **G7** delivered §11 end to end:
`examples/native_dropout_training.py` trains a classifier carrying all
four TensorForge-owned state families — parameters, persistent BatchNorm
buffers, a registered `NativeGenerator`, and `NativeAdam` moments — and
proves that an interrupted run resumed into a **completely fresh**
model/optimizer/generator set reproduces the uninterrupted run by exact
equality, with the external loop position carried as explicit, validated
metadata rather than claimed as automatic checkpoint state. **G7 added no
capability**: one example, one test module, and documentation.

That is a demonstrated exact stochastic resume and nothing above it. There
is no implicit, global, or default generator anywhere — the operation's
generator is **required and keyword-only**, and the module's is explicit
registered state; the Core still consumes **no** generator call and
touches no `NativeGenerator` at all; reproducibility is exact **for the
state actually captured**, and data-loader position, shuffle state,
scheduler state, Python's `random`, and NumPy's global RNG are captured by
nothing (§11.1); and `"dropout"` is still in `UNSUPPORTED`, deliberately,
until the G10 closure.

The capability boundary is therefore exactly what Phase F closed with,
except for the format version G5 was always going to move:

- `UNSUPPORTED == ("dropout", "float32", "cuda", "amp")`
- `SUPPORTED_DTYPES == ("float64",)`, `SUPPORTED_DEVICES == ("cpu",)`
- native checkpoint format `"tensorforge.native_checkpoint"` — the
  **name** never moves — at **format version 2** since G5, with version 1
  still loadable exactly where §10.6 says

`"dropout"` stays in `UNSUPPORTED` for the **whole** of G0–G9 and leaves
it **only at G10**, and only after the complete Phase-G closure matrix of
§18 has passed. G4 implements and publicly exports `NativeDropout`, but
it does **not** move the capability boundary: the registry reports a
*closed, validated* capability, not an *implemented* one (§19, G4). So

- through G9: `UNSUPPORTED == ("dropout", "float32", "cuda", "amp")`
- after a successful G10 closure: `UNSUPPORTED == ("float32", "cuda", "amp")`

The checkpoint format version became **2** at milestone **G5**, and not
before. Everything below describes what the remaining milestones *will*
do; the present tense is used for locked contracts, never for unshipped
behavior.

Related contracts this phase inherits and must not weaken:
[native_autograd_design.md](native_autograd_design.md) (graph lifetime,
`retain_graph`, failure rollback),
[native_cnn_design.md](native_cnn_design.md) (§5 Policy-B contiguity, §10
and §12 the private saved-state/`graph_resources` archetype),
[native_classification_design.md](native_classification_design.md)
(graph-owned saved probabilities, backward read contracts),
[native_normalization_design.md](native_normalization_design.md) (the
atomic registered-state transaction, mutable-state graph safety), and
[native_abi_error_contract.md](native_abi_error_contract.md) (no C++
exception crosses `extern "C"`).

---

## 1. Objective and scope

Phase G gives the experimental native line **explicit, deterministic,
serializable randomness** and the one consumer that motivates it:
inverted Dropout.

The phase's organizing idea is that **random state is Python-managed and
native kernels are stateless**. A native kernel is handed the complete
random key for one operation and computes a mask from it; it never reads,
holds, or advances any generator. That is the same split the whole native
line already uses for autograd — the graph lives in Python, the kernels
are graph-unaware — and it is what makes a checkpoint able to restore the
random stream exactly.

### 1.1 In scope

- deterministic explicit native RNG state (`NativeGenerator`)
- registered generator state as a first-class `NativeModule` category
- a stateless native CPU random-mask kernel behind the existing C ABI
- inverted Dropout (`0 <= p < 1`)
- the differentiable `NativeTensor.dropout(p, generator=...)` operation
- the `NativeDropout` module and its public export (the capability
  *boundary* moves only at closure — §19, G4 and G10)
- graph-owned saved multiplier masks
- native checkpoint **version 2** and exact generator restoration
- ownership, lifetime, and failure hardening
- cross-cutting integration tests, a deterministic example, an honest
  benchmark, sanitizer and leak closure, documentation and guardrails

### 1.2 Explicit non-goals

Phase G does **not** add: a generic `rand`/`randn`/Bernoulli/sampling or
distribution API; any global or process-wide random state; NumPy global
RNG integration; a parameter-initialization redesign; data-loader
shuffling or data augmentation; `Dropout2d`/`Dropout3d`; stochastic
depth; attention dropout; integer tensors or an indexing system;
embeddings; float32, CUDA, or AMP; schedulers; gradient clipping; new
optimizers; further activations or losses; CPU performance tuning; any
stable-framework change; implicit backend dispatch; production-readiness
claims; speed-superiority claims; or **parallel stochastic execution** —
the generator's lock (§3.6) serializes for correctness, and two threads
drawing from one generator is a deterministic error, not a feature.

`float32`, `cuda`, and `amp` remain in `UNSUPPORTED` when Phase G closes.
`"dropout"` leaves it at G10 and not before (§19).

---

## 2. Reference reading: Daedalus ML

The comparable design in
[Daedalus ML](https://github.com/JohnsonKayati/daedalus-ml) was read
before this contract was written (`python/daedalus/nn/dropout.py`,
`src/ops/cpu/unary_ops.cpp`, `python/daedalus/nn/module.py`,
`python/daedalus/checkpoint.py`). **No Daedalus code is copied**; what
follows is an architectural comparison. Where the two projects disagree,
TensorForge's existing ownership, failure-atomicity, checkpoint,
namespace, and stable/native separation contracts win.

### 2.1 Ideas taken

- **Stateless kernel, Python-side state.** Daedalus's CPU dropout kernel
  takes `(tensor, p, seed)` and holds nothing; the Python `Dropout`
  module owns the seed and a call counter. Phase G adopts exactly this
  split, and strengthens it: the kernel receives `(seed, call_index)` as
  two explicit `uint64` values and computes every element's draw from
  them, so no native object has random state at all.
- **A SplitMix64-family finalizer as the mixing function.** It is small,
  fully specifiable in integer arithmetic, dependency-free, and easy to
  pin with known-answer vectors. Phase G uses the same three-step
  finalizer constants.
- **Inverted dropout with the scale folded into the mask.** Daedalus
  writes `1/(1-p)` or `0` into a mask tensor and multiplies. Phase G does
  the same, which is what lets the backward be a single elementwise
  multiply against saved state.
- **`p` restricted to `[0, 1)`.** `p == 1` is rejected rather than
  special-cased, so `1/(1-p)` can never divide by zero.
- **OS entropy only to *create* a seed.** `seed=None` draws once from
  `secrets`; from then on the seed is an explicit integer. Phase G keeps
  this and adds the requirement that the drawn seed is immediately
  inspectable and serializable.

### 2.2 Ideas that require improvement

- **Key derivation by addition.** Daedalus computes each element's bits
  from `mix64(seed + i)` and each call's seed from
  `seed + counter * GOLDEN`. Two calls whose per-call seeds differ by
  less than the element count therefore share a suffix of the same
  underlying stream, so their masks correlate. Phase G derives a
  **per-call stream key through a full mix**, then derives each element
  from that key through a second full mix (§4), so call streams are
  decorrelated by construction.
- **The counter advances before the kernel runs.** Daedalus calls
  `_next_seed()` in the argument list, so a kernel that then raises has
  still consumed a call. Phase G makes the advance a **commit after the
  output is published** (§5): a failed forward consumes nothing.
- **Random state is invisible to `state_dict` and checkpoints.**
  Daedalus's counter is a plain Python attribute and its checkpoint
  format (version 1) captures no RNG state, so a resumed stochastic run
  cannot reproduce its masks. Phase G makes generators **registered
  module state** (§9) and adds a checkpoint section for them (§10), which
  is the whole reason the format version moves.
- **Implicit mask lifetime.** Daedalus builds a mask tensor and returns
  `mul(t, mask)`, leaving the mask alive through ordinary refcounting.
  TensorForge releases native storage deterministically, so Phase G makes
  the mask **graph-owned state** released exactly once at the existing
  graph-release points (§8).
- **An unsynchronized counter.** Daedalus's counter is a bare attribute
  incremented in place, so two overlapping calls — concurrent or
  reentrant — can read the same value and silently produce the same mask
  twice. Phase G puts reservation, commit, cancellation, and every state
  read and write behind one lock and hands out **opaque single-use
  tokens** (§3.6), so a duplicated call index is impossible and overlap
  is a deterministic error rather than a silent collision.
- **No algorithm identifier, no state validation, no exhaustion rule.**
  Phase G locks an algorithm identifier and version, validates every
  loaded state field, and defines counter exhaustion (§3, §4, §10).

### 2.3 Ideas that do not fit TensorForge

- **A CUDA branch inside the operation.** The native line is CPU-only and
  `device` has exactly one value; a device switch would be dead code and
  an untestable claim.
- **float32 masks.** The native runtime has one dtype. The mask is
  float64, and the bits-to-uniform conversion uses 53 bits accordingly,
  not Daedalus's 24-bit float32 conversion.
- **pybind11 bindings and C++-managed autograd.** TensorForge crosses the
  boundary with a plain C ABI and ctypes, and manages the graph in
  Python. The dropout node is a Python-managed node like every other one.
- **Returning the input tensor from the kernel layer on `p == 0`.**
  Daedalus's C++ `dropout` returns `t` itself for `p == 0`. In
  TensorForge that decision belongs to the Python layer, where ownership
  is tracked; the kernel is never asked a question whose answer is "no
  work" (§6.2).
- **A `secrets`-seeded module with no way to inspect or set the state.**
  TensorForge exposes the generator as an object with a readable,
  replaceable state, because the checkpoint contract requires it.

---

## 3. `NativeGenerator` — the public contract

`NativeGenerator` ships in milestone **G1**, in
`src/tensorforge/experimental/native_generator.py`.

### 3.1 What it is

A **pure-Python value holder**. It owns **no native storage**, allocates
nothing, and has **no `close()`** — inventing one would advertise a
lifetime that does not exist, and every ownership matrix in this project
would then have to lie. It is not a `NativeTensor`, not a
`NativeParameter`, not a buffer, and never enters the tensor state-dict
key space.

### 3.2 Constructor

```
NativeGenerator(seed=None)
```

- `seed=None` draws one 64-bit seed from operating-system entropy
  (`secrets.randbits(64)`). This is the **only** use of non-deterministic
  entropy anywhere in Phase G, it happens exactly once per constructed
  generator, and the drawn value is immediately readable through `.seed`
  and serializable. Nothing else in the phase consults the OS, the
  clock, the process id, an address, or any global RNG.
- An explicit `seed` must be an exact `int` (`bool` is rejected — `True`
  is not a seed) in `[0, 2**64 - 1]`. Python `int` is arbitrary
  precision, so there is no silent truncation: an out-of-range value
  raises `ValueError`, a non-int raises `TypeError`, and NumPy integer
  scalars are rejected (exact-type discipline, matching the checkpoint
  metadata validator).
- The call counter starts at `0`.
- Construction cannot fail after any state has been written: every
  argument is validated first.

### 3.3 State representation

Four fields, and exactly four:

| Field | Type | Meaning |
|---|---|---|
| `algorithm` | `str` | `"tensorforge.splitmix64"` (§4) |
| `algorithm_version` | `int` | `1` |
| `seed` | `int` | unsigned 64-bit, `0 <= seed < 2**64` |
| `calls` | `int` | unsigned 64-bit, number of **committed** stochastic calls |

All four are exposed as **read-only properties**. There is no public
attribute assignment: `generator.seed = 7` raises. State changes go
through the three explicit methods below, each of which validates
everything before mutating anything.

### 3.4 Methods

```
state()                      -> dict           # independent snapshot
load_state(state)            -> None           # atomic, identity-preserving
reseed(seed)                 -> None           # new seed, calls reset to 0
reset()                      -> None           # same seed, calls reset to 0
```

- `state()` returns a **fresh plain-Python dict** with exactly the four
  fields above. It shares nothing with the generator (ints are
  immutable), so mutating the returned dict affects nothing, and it is
  directly JSON-compatible except for the integer encoding rule in §9.4.
- `load_state(state)` requires a mapping with **exactly** those four
  keys, validates each (algorithm string equality, algorithm version
  equality, seed and counter type and range), and only then assigns. A
  rejected load changes **no** field — it is all-or-nothing by
  construction, because the assignment happens after all validation and
  cannot itself fail. The generator object's identity is preserved, so a
  module that registered it keeps the same object.
- `reseed(seed)` validates the seed exactly as the constructor does, then
  sets the seed and resets `calls` to `0`. `reset()` keeps the seed and
  sets `calls` to `0`. Both exist because a deterministic experiment
  frequently wants "the same stream again" without rebuilding the model.
- **`state()`, `load_state()`, `reseed()`, `reset()`, and every read of
  `calls` are performed under the generator's lock (§3.6), and the three
  mutating ones refuse while a reservation is active.** Replacing the
  seed or the counter underneath an in-flight draw would make the
  reserved index describe a stream that no longer exists, so it raises
  `RuntimeError` and changes nothing.
- There is **no** `advance(n)`, `jump()`, `split()`, `spawn()`, `clone()`,
  `__copy__`, or `__deepcopy__` in Phase G. A copied generator would
  silently produce identical masks in two places, which is the exact
  failure mode explicit state is meant to prevent. Sharing is done by
  sharing the object (§3.7).

### 3.5 Equality and identity

`NativeGenerator` defines **no** `__eq__` and **no** `__hash__` override:
identity is object identity, exactly like `NativeParameter`,
`NativeTensor`, and `NativeModule`. Two generators with equal state are
two generators. Every traversal, deduplication, and checkpoint rule in
this phase keys on `id()`, never on value equality. A test that wants to
compare states compares `state()` dicts explicitly.

### 3.6 The call reservation protocol, and its lock

The generator exposes three **private** methods used by exactly one
caller (`NativeTensor.dropout`, §8):

```
_reserve_call()      -> token   # opaque; carries the call index
_commit_call(token)  -> None    # publish: calls += 1, exactly once
_abandon_call(token) -> None    # failure: calls unchanged
```

#### The lock

`NativeGenerator` holds one private `threading.RLock` created in
`__init__` and never exposed or replaced.

The governing invariant is:

> **No user code, no callback, and no generator-owned allocation runs
> while a generator lock is held.**

Reservation-token construction — the one operation in the reservation
path that allocates, and therefore the one that can run interpreter
finalization — is deliberately performed **outside** the lock (see the
next subsection). That is what makes the invariant true, and it is what
makes the global multi-generator lock order (§9.6) unbreakable: a
finalizer cannot start a transaction while this thread owns one
generator's lock, so **finalizer or callback reentry cannot invert the
multi-generator lock order** — it can never own one lock and reach
backwards for another.

The lock protects exactly these operations, every one of which is now
straight-line integer work:

| Operation | Under the lock |
|---|---|
| reservation **claim** (phase 1 of `_reserve_call`) | yes — active check, **claim check**, exhaustion check, candidate index and serial read, claim write |
| reservation **token construction** (phase 2) | **no — no generator lock is held** |
| reservation **publication** (phase 3) | yes — claim match, active-slot write, single serial advance, claim clear |
| reservation cancellation (`_abandon_call`) | yes |
| reservation commit (`_commit_call`) | yes — token match and the single `calls += 1` |
| committed-outcome query (`_call_committed`) | yes — a read, and only a read (added at G3; see §5's outcome 3) |
| call-counter read (`.calls`) | yes |
| state inspection (`state()`, `.seed`, `.algorithm`, `.algorithm_version`) | yes |
| state replacement (`load_state()`) | yes |
| reseeding (`reseed()`) | yes |
| resetting (`reset()`) | yes |
| the multi-generator state transaction (§9.6) | yes — **every** target's lock, held together |

**Why it stays an `RLock`.** Two independent reasons, neither of which is
"to permit reentrant execution while holding the lock":

1. *Structural.* The multi-generator transaction (§9.6) holds every
   target's lock and then reaches its targets through the same private
   primitives every other caller uses — `_snapshot_state()` and
   `_assign_state()`, which take the lock themselves. Re-entering one
   lock the current thread already owns is how those primitives stay a
   single write seam instead of forking into locked and unlocked
   variants. On a plain `Lock` that is an immediate self-deadlock.
2. *Residual allocation.* CPython may begin a collection at any container
   allocation, and the remaining critical sections still allocate small
   objects — the dict `state()` returns, the tuple `_snapshot_state()`
   returns, and the message and traceback of every exception raised under
   the lock. An `RLock` turns a finalizer that re-enters through one of
   those into a deterministic refusal (for a mutation) or a correct read
   (for an inspection) instead of a permanent hang.

An `RLock` weakens nothing: across threads it behaves exactly like a
`Lock`, and re-acquiring a lock the current thread already owns never
blocks, so it can never reorder acquisitions.

#### Reservation construction: claim, construct, publish, deliver

`_reserve_call` is a **two-phase transaction** — a claim taken and
released under the lock, a construction performed holding nothing, a
publication under the lock, and finally delivery of the token to its
caller:

**Phase 1 — claim.** Under the lock: reject an active reservation →
reject a claim already in progress → reject an exhausted counter →
capture the candidate call index and reservation serial → publish **only**
the internal construction claim. No active reservation is published,
`calls` does not advance, and the next reservation serial does not
advance. Release the lock.

**Phase 2 — construct, outside the lock.** Build the token while holding
no generator lock. If construction raises — `MemoryError`,
`KeyboardInterrupt`, or any explicit failure — a `finally` reacquires the
lock, verifies the matching claim still stands, clears it, publishes no
active reservation, leaves `calls` and the next serial unchanged, and
re-raises the original exception. The generator is immediately reusable.

**Phase 3 — publish.** Reacquire the lock, verify the claim still matches
the candidate serial and index, publish the active reservation, advance
the never-reused serial exactly once, clear the claim, release.

**Phase 4 — deliver.** Hand the token back to its caller.

The **construction claim** is what makes the window between phase 1 and
phase 3 safe. While it stands, each of the following fails with
`RuntimeError` and mutates nothing, whether it arrives from another
thread or reentrantly:

- another reservation,
- `load_state()`, `reseed()`, `reset()`,
- `replace_generator_states()` (§9.6) and any checkpoint-style
  replacement built on it.

State *inspection* is unaffected and keeps working, exactly as the lock
table says. No other state-changing operation can occur between the claim
and the publication, because the claim blocks all of them. Neither the
claim nor the token is public.

#### The four failure positions, and why they need different cleanup

Failures at the four positions are **not** interchangeable, and clearing
the claim covers only the first two:

| Where it fails | Claim | Active reservation | Cleanup | `calls` | Serial |
|---|---|---|---|---|---|
| **construction** (phase 2) | still standing | never published | clear the matching claim | unchanged | unchanged — none is skipped |
| **publication** (phase 3) | still standing | never published | clear the matching claim | unchanged | unchanged |
| **after publication, before delivery** | **already cleared** | **published, and its only token is being dropped** | cancel the exactly-matching reservation | unchanged | **consumed** |
| **delivered** (phase 4 returns) | already cleared | published, caller holds the token | none — both cleanups match nothing | unchanged until commit | consumed |

The third row is the one worth stating explicitly, because it is easy to
get wrong: **once a reservation is published the claim is gone, so
clearing the claim does nothing for the publication-to-return window.**
An asynchronous exception arriving there — a `KeyboardInterrupt`, a
`MemoryError` — would otherwise leave an active reservation that *no
caller has a token for*: it could never be committed or abandoned, and
every later reservation on that generator would be refused for the rest
of the process. The generator would be permanently stranded.

So a failed delivery runs its own cleanup, which under the generator's
own lock:

- verifies the live reservation matches **this token's generator, serial,
  *and* index**, and that the token is still unfinished;
- clears that reservation **without advancing `calls`**;
- leaves the consumed reservation serial consumed — serials are opaque
  and never reused, so burning one costs nothing, while restoring one
  could hand a later reservation a serial an existing token already
  carries;
- marks the token finished, so a caller that somehow still holds it
  cannot resurrect the reservation;
- and, if the match fails for any reason, **changes nothing at all**.

That last point is the safety property: the cleanup can never cancel a
**newer** reservation, a **foreign** generator's reservation, or one that
was already **committed** or **abandoned**. A failed delivery may consume
an internal reservation serial; it never consumes a call index and never
advances `calls`.

The cleanup takes **only its own generator's lock** and does no
callback-capable work while holding it, so it never participates in the
global multi-generator acquisition order (§9.6) and cannot violate it.

One residual window is **not** covered, and is stated rather than papered
over: an asynchronous exception delivered after `_reserve_call`'s cleanup
has run and while the frame is returning, before the caller binds the
token. No Python code can run there. It is bounded to a couple of
bytecodes, and the intended caller (§8) binds the token and enters its
`try`/`finally` as its next action, so the operation-level cleanup covers
what the generator-level cleanup structurally cannot.

**Native numerical computation happens outside the lock.** The lock is
released before the Core forward allocates anything or the kernel runs,
and reacquired only to commit or abandon. No native call, no C ABI entry,
and no `NativeTensor` construction ever occurs while the lock is held, so
the lock can never be held across a native failure or an arbitrarily long
kernel.

#### The token

`_reserve_call()` returns an **opaque token**, not a bare integer. The
token carries enough identity to make a stale, foreign, duplicated, or
mismatched commit impossible to mistake for a valid one: the owning
generator's identity, the reserved call index, and a
monotonically-increasing per-generator reservation serial that is never
reused within the generator's lifetime. The token is inert — it exposes
no mutating behavior, holds no lock, and is meaningful only to the
generator that minted it.

The generator keeps **at most one** active reservation. `_commit_call`
and `_abandon_call` both compare the supplied token against that active
slot under the lock, by generator identity *and* serial:

| Token presented | Result |
|---|---|
| the current active reservation | commit advances `calls` by exactly one and clears the slot; abandon clears the slot and advances nothing |
| a token from a **different** generator | `RuntimeError`; **no** state changes on either generator |
| a token already committed | `RuntimeError`; `calls` does **not** advance a second time |
| a token already abandoned | `RuntimeError`; nothing changes |
| a token whose serial is stale (a newer reservation is active) | `RuntimeError`; the active reservation is **untouched** |
| any token when no reservation is active | `RuntimeError`; nothing changes |
| not a token at all | `TypeError`; nothing changes |

So: **commit advances exactly once, and only for the currently active
matching reservation. Cancel clears only the currently active matching
reservation and never advances.** Every rejected token leaves `seed`,
`calls`, and the active slot exactly as they were.

One further private method, `_call_committed(token)`, answers a
**read-only** question about that outcome: did *this* generator commit
*this exact* token? It creates, clears, and matches no reservation, moves
no counter, consumes no serial, and touches no claim. It exists because
§5's outcome 3 — an exception after a successful commit and before the
result is returned — needs cleanup that behaves differently from every
pre-commit failure, and a boolean set after `_commit_call` cannot tell
the two apart when the commit succeeds and that assignment never runs. A
foreign generator's token answers `False` rather than raising, since the
question is "did *I* commit it", and a non-token raises `TypeError` like
commit and abandon do. A token's outcome is **never** public.

#### What the lock buys, and what it does not

- **A second reservation while one is active fails deterministically and
  receives no call index at all.** Both overlap cases are covered:
  **concurrent** use (another thread) and **reentrant** use (the same
  thread re-entering through a callback, a signal handler, or a `__del__`
  running mid-call). Either way it raises `RuntimeError` *before* an
  index is minted — so it cannot consume, duplicate, or skip a call.
- **No two threads can ever receive the same successful call index.**
  Minting the index, recording the active reservation, and advancing the
  counter all happen inside the same lock, so the index a thread receives
  is unique for the generator's lifetime and only one thread can be
  holding a live reservation at a time.
- **Counter exhaustion (§4.6) is checked while holding the lock**, so the
  boundary cannot be raced past by two callers reading `calls` at once.
- **`load_state`, `reseed`, `reset`, and checkpoint generator commits
  refuse while a reservation is active**, with `RuntimeError` and no
  change. There is no "safe alternative" that permits it: the reserved
  index is only meaningful relative to the seed it was reserved under.
  A checkpoint load therefore fails in **prevalidation** (§10.5) if any
  target generator has a live reservation, which is before anything is
  staged or committed — never mid-commit.

**This is serialization for correctness, not a performance feature, and
Phase G does not claim parallel stochastic execution.** The lock exists
so that misuse becomes a deterministic exception and a committed call
index is provably unique — not so that two threads can draw
simultaneously. They cannot: one of them raises. The rest of the native
line remains a single-threaded contract, and nothing here changes that.

### 3.7 Sharing and aliasing

- A generator may be registered on several modules, or several times on
  one module. Traversal deduplicates by identity, so it appears **once**
  in `named_generators()`, once in the generator state dict, and once in
  a checkpoint, under its first-discovered canonical name (§9.4). This
  is exactly the shared-parameter rule.
- Sharing a generator is how two `NativeDropout` layers draw from one
  stream: the second layer's forward uses the next call index.
- Two independent generators (the default: each `NativeDropout` builds
  its own) produce independent streams and are restored independently.
- Nothing aliases a generator implicitly: registering stores the exact
  object, never a copy, matching every other registration in the native
  line.

---

## 4. The random algorithm

### 4.1 Identifier

`algorithm = "tensorforge.splitmix64"`, `algorithm_version = 1`.

The pair `(algorithm, algorithm_version)` is the compatibility key.
Nothing else identifies the stream.

### 4.2 The finalizer

All arithmetic is on unsigned 64-bit integers with **wrapping** (modulo
`2**64`) semantics.

```
GOLDEN = 0x9E3779B97F4A7C15

mix64(x):
    x ^= x >> 30;   x = x * 0xBF58476D1CE4E5B9
    x ^= x >> 27;   x = x * 0x94D049BB133111EB
    x ^= x >> 31
    return x
```

### 4.3 Derivation

```
stream(seed, call_index)          = mix64(seed + GOLDEN * (call_index + 1))
bits(seed, call_index, element)   = mix64(stream + GOLDEN * (element + 1))
```

Two full finalizer applications separate the per-call stream from the
per-element draw, so two different call indices cannot produce
overlapping element sequences by a simple offset — the defect §2.2
identifies in the reference design. `call_index + 1` and `element + 1`
keep the zero case from degenerating to `mix64(seed)` twice.

**What this guarantees, exactly** (characterized at G6, pinned by test).
`GOLDEN` is odd, so multiplication by it is invertible modulo `2**64`, and
`mix64` is a bijection. Therefore:

- **Within one seed, `stream` is injective in `call_index`.** No two call
  indices a generator ever issues can share a stream — which is the
  property a generator actually promises, and the one exact resume rests
  on. The same argument makes `bits` injective in `element` within one
  call.
- **Across different seeds, collisions exist and are unavoidable.**
  `(seed, call_index)` is 128 bits folded into a 64-bit stream key, so by
  counting some pairs must collide, and one is exactly computable:
  `GOLDEN * 2**63 == 2**63` mod `2**64`, hence
  `2**63 + GOLDEN * (2**63 + 1) == GOLDEN == 0 + GOLDEN * (0 + 1)` — so
  `(seed=2**63, call=2**63)` produces the *same* mask as `(seed=0,
  call=0)`, for every tensor and every `p`.

The second point weakens nothing in the contract and is recorded so it is
never mistaken for a defect and never "fixed" by changing a locked
derivation. Sharing a stream is **identity** (§3.7) — two generators are
two generators, and two generators with the same seed and counter stay two
entries everywhere — so the contract never claims that distinct
`(seed, call_index)` pairs index distinct streams. A test pins both halves
against the real kernel.

### 4.4 Bits to the Dropout decision

```
u(bits) = (bits >> 11) * 2**-53          # float64 in [0.0, 1.0)
drop    = u < p
multiplier = 0.0            if drop
             1.0 / (1.0 - p) otherwise
```

`bits >> 11` yields the top 53 bits, at most `2**53 - 1`, which is exactly
representable in float64; multiplying by the power of two `2**-53` is
exact. So `u` is a uniform value on `[0, 1)` with `2**-53` granularity and
no rounding surprise, and `P(u < p) = p` up to that granularity. The
comparison is `<`, so `p == 0` would drop nothing (and is short-circuited
before any draw anyway, §6.2).

`1.0 / (1.0 - p)` is computed **once per call** in float64 and reused for
every element, so every kept element carries the identical multiplier and
the mask holds exactly two distinct values.

### 4.5 Platform independence

- Only unsigned integer shifts, multiplies, and xor are used; C++
  guarantees wrapping for unsigned arithmetic, so there is no
  implementation-defined or undefined behavior anywhere in the bit path.
- The implementation uses `std::uint64_t` explicitly, never `unsigned
  long`, `size_t`, or a signed type, so word size and signedness cannot
  vary between MSVC and Clang/GCC.
- The only floating-point operations are the exact `>> 11` conversion,
  one division for the scale, and the comparison — no transcendental
  functions, no `long double`, no fused-multiply-add opportunity that
  could reassociate, and no dependence on fast-math flags. The project
  does not enable fast-math in any build configuration, and Phase G must
  not introduce one.
- No `std::random_device`, no `std::mt19937`, no `<random>` distribution,
  no Python `random`, and no NumPy global state is used for per-element
  generation — or anywhere else in the phase except the single
  `secrets.randbits(64)` seed draw of §3.2.
- The result is a deterministic function of `(seed, call_index,
  element_index)` and nothing else: not the memory address, not the
  thread, not the traversal order, not the physical strides, not the
  allocation history, not the number of preceding operations.

### 4.6 Counter exhaustion

The counter never wraps. Write `UINT64_MAX = 2**64 - 1`.

`calls` is a **count of committed calls**, not an index space, so
`UINT64_MAX` is a **reachable, valid** value — it is what the counter
holds after the last representable successful call. It is not a sentinel
and is not reserved.

A reservation uses the **current** `calls` value as its call index.
`_reserve_call()` refuses when `calls >= UINT64_MAX` and raises
`RuntimeError`, consuming nothing and minting no token. The check happens
**under the lock** (§3.6), so two callers cannot both observe the last
usable value. Exactly:

| State | Reserve | Reserved index | After commit |
|---|---|---|---|
| `calls < UINT64_MAX - 1` | succeeds | `calls` | `calls + 1` |
| `calls == UINT64_MAX - 1` | **succeeds** | `UINT64_MAX - 1` | **`UINT64_MAX`** |
| `calls == UINT64_MAX` | **refused** (`RuntimeError`) | — | `UINT64_MAX`, unchanged |

So the largest usable call *index* is `UINT64_MAX - 1`, the largest
reachable *count* is `UINT64_MAX`, and `calls` stays inside
`[0, UINT64_MAX]` inclusive at every point.

Abandoning does not consume the index: cancelling a reservation made at
`calls == UINT64_MAX - 1` leaves `calls == UINT64_MAX - 1`, and the next
reservation legitimately takes that same unconsumed index again. No
failed, stale, malformed, foreign, duplicate, or concurrent operation
moves the boundary state — every one of them leaves `calls` exactly where
it was (§14).

Loading a state whose `calls` is at the boundary is legal (§10.5) — the
refusal happens at the next attempted forward, not at load time, because
a checkpoint must round-trip exactly what was saved. Recovery is
`reset()` or `reseed()`, which return the counter to `0`.

### 4.7 Known-answer requirements

Milestone G2 commits fixed vectors, computed once and thereafter treated
as the specification:

- `mix64` on a fixed list of inputs including `0`, `1`, `2**63`, and
  `2**64 - 1`
- `stream(seed, call_index)` for a fixed seed across several call indices
- `bits`, `u`, and the full multiplier mask for a small fixed tensor at a
  fixed `(seed, call_index, p)`
- the same mask reproduced from a transposed (non-contiguous) view of the
  same logical tensor
- **the equality-threshold vector** (below), which pins the comparison
  itself

#### The equality-threshold vector

The vectors above pin the *bit path*. They do **not**, on their own, pin
the *comparison direction*, and that is a real gap rather than a
theoretical one: the committed mask vectors use `p = 0.25` and
`p = 0.75`, no committed word converts to either value, so replacing the
locked `u < p` with `u <= p` reproduces every one of those patterns
unchanged and escapes the suite. The bits-to-uniform vectors do not close
it either — they prove what `u` *is*, not how it is compared.

One further vector closes it, and it is chosen so that nothing new enters
the stream: the seed, call index, logical element index, and raw word are
**already committed** as `mixed_seed_call0`'s third element. Only the
probability is new, and it is chosen to land exactly on that element's
uniform value.

| Field | Value |
|---|---|
| `seed` | `0x0123456789ABCDEF` |
| `call_index` | `0` |
| logical element index | `2` |
| raw word `bits` | `0xA2A1796FEB7EF314` |
| `u` | `0x1.4542f2dfd6fdep-1` (`0.635276403259464`) |

`u` is strictly inside `(0, 1)` and is exactly the locked 53-bit
conversion of that word, so both `p == u` and the next representable
double above it are legal probabilities.

Run over the first **four** elements of that stream, the vector pins:

| `p` | keep pattern | what it proves |
|---|---|---|
| `u` | `0010` | **equality means keep** — the strict `<` rule keeps the element whose uniform value equals `p` |
| `u` (under the rejected `<=`) | `0000` | the two rules genuinely disagree here: this is the negative control |
| `nextafter(u, 1.0)` | `0000` | **the adjacent larger probability means drop** |

So the vector pins **all three** properties together: the comparison is
**strict**, equality is a **keep**, and one ULP more is a **drop**. The
kept element additionally carries exactly `1 / (1 - p)` and its output is
exactly `input * mask`, so the multiplier and the output rule are pinned
at the boundary too.

`std::nextafter(u, 1.0)` (C++) and `math.nextafter(u, 1.0)` (Python) are
used **only** to form the adjacent probability. The authoritative seed,
call index, logical index, raw word, expected `u`, and expected keep
patterns are hardcoded constants on both sides.

Both suites drive the **production** path — the internal kernel, the
exported C ABI wrapper, and (in Python) the public and private Core
methods — never a duplicated test-only comparison helper. Each also
carries a **negative control** proving the vector discriminates: it
computes what a `<=` kernel would have produced from the same derivation,
with only the operator changed, and asserts it disagrees with what the
production kernel produced. Without that control the equality assertion
could be satisfied vacuously; with it, a production change from `<` to
`<=` fails both the native and the Python boundary tests.

These vectors are asserted identically on Windows (MSVC) and Linux
(Clang/GCC). A test-only Python reference implementation of §4.2–§4.4
lives in the test suite and is compared against the kernel; it is
**never** production code, because a second production implementation
would create a second source of truth and a silent NumPy fallback path.

### 4.8 If the algorithm ever changes

A change to the finalizer, the derivation, the bits-to-uniform
conversion, or the comparison **must** introduce a new
`(algorithm, algorithm_version)` pair. Checkpoints written under the old
pair must then be rejected with a clear error naming both pairs. Silently
reinterpreting a saved seed under a different algorithm is forbidden: the
saved `calls` would no longer describe the stream it was taken from.

---

## 5. Call-counter transaction semantics

**One successful stochastic Dropout forward consumes exactly one
generator call.** The transaction boundary is the point at which the
operation's result has been fully constructed and is about to be returned
to the caller.

The reservation is owned by exactly one layer — `NativeTensor.dropout`
(§8), shipped at G3 and still the only caller of the reservation
protocol. Neither `NativeTensorCore` nor any C++ code ever touches a
generator, so no lower layer can commit a call that a higher layer then
fails to complete.

Ordered contract for `NativeTensor.dropout(p, generator=g)`:

1. Validate `p`, the generator, and the input. *No reservation yet.*
2. If `p == 0.0`, return the input unchanged. **No reservation, no call.**
3. `token = generator._reserve_call()` — under the generator's lock,
   which is released before step 4. This is where a concurrent or
   reentrant caller fails, and where exhaustion is refused.
4. Call the Core forward with `(seed, token.index)`, **outside the
   lock**. It allocates the output and the mask and runs the kernel.
5. Build the autograd node, adopting the mask as graph-owned state (or
   closing it immediately when nothing requires grad).
6. `generator._commit_call(token)` — **the transaction boundary**, which
   reacquires the lock, matches the token against the active reservation,
   and advances `calls` exactly once.
7. Return the result.

The seed used in step 4 is read under the lock in step 3 and carried
forward, so a `reseed()` racing this forward cannot change the stream the
reserved index describes — it is refused outright while the reservation
is live (§3.6).

#### The three outcomes, and why cleanup must know which one it is

Step 6 is the boundary, and the cleanup on either side of it is **not**
the same. There are exactly three outcomes:

| # | Outcome | Calls consumed | Reservation | Result | Exception |
|---|---|---|---|---|---|
| 1 | a failure at steps 3–5, or at step 6 **before** the commit takes effect | **none** — `calls` is exactly where the forward found it | abandoned; the slot is cleared and the **same index stays retryable** | closed, releasing the graph-owned mask with it | the original propagates |
| 2 | commit succeeds and the result is returned | **exactly one** | committed; the slot is clear | delivered to the caller | none |
| 3 | commit **succeeds** and an exception arrives before the caller receives the result | **exactly one — irreversibly** | already committed; the slot is already clear | closed, releasing the graph-owned mask with it | the original propagates |

Outcome 3 is the asynchronous commit-to-return window. The window itself
may be unavoidable, but its *cleanup* is not, and it must be correct:

- **The call is spent.** Once `_commit_call` returns, `calls` has
  advanced and that index describes a draw that really happened. Nothing
  may claim otherwise, and the generator carries on from its **next**
  index — the spent one is never handed out again.
- **The committed token is not abandoned.** `_abandon_call` on it would
  raise "already committed" (§3.6's token table, unchanged), and that
  cleanup error would then stand in for the failure the caller actually
  needs to see. So the cancellation is simply not attempted.
- **The unreturned result is still released.** No caller received it, so
  it is closed exactly as in outcome 1, which releases the graph-owned
  mask with it and returns native live storage to its baseline. Nothing
  partial is observable and no graph escapes.
- **The original exception propagates unchanged.** It is never replaced
  by a stale-token or already-committed cleanup error.

**The outcome is read from the token, not from a flag.** A local boolean
set after `_commit_call` cannot distinguish outcomes 1 and 3: the commit
can succeed and the assignment never execute — which is precisely the
injected case a test must be able to produce. The operation therefore
asks the generator, through the private read-only `_call_committed(token)`
query (§3.6), whether *that exact token* committed on *that* generator.
The query creates, clears, and matches no reservation, moves no counter,
consumes no serial, and touches no claim; a token's outcome is never
public.

Both sides are proved by deterministic injection rather than by real
signal timing: a `_commit_call` wrapper that raises *instead of*
committing exercises outcome 1, and one that calls the **real** commit
and then raises exercises outcome 3, over `KeyboardInterrupt`,
`MemoryError`, and a custom `BaseException` that is deliberately not an
`Exception`. A tripwire asserts `_abandon_call` is never invoked after a
confirmed commit.

Cleanup itself never raises. Every step is attempted even if an earlier
one fails — the discipline `_native_state`'s post-commit release loop
already uses — and because the cleanup steps are non-failing by contract,
a failure among them means something is already wrong: the **operation's**
exception stays primary and the cleanup failure is chained onto its
context chain rather than substituted for it, so nothing is swallowed and
nothing is hidden.

**The chain must also be acyclic** — the one runtime defect milestone G6
found and fixed, recorded here rather than only in a commit message. A
cleanup step that raises while the operation's failure is being handled
gets an *implicit* `__context__` pointing straight back at that failure,
so appending it to the end of the failure's own chain without cutting that
back-reference closes a two-element **cycle**. CPython's own traceback
formatter tolerates one, but every straightforward "follow `__context__`
to the end" reader does not — including the chaining helper itself on a
second cleanup failure, and any logging or error-reporting code the caller
runs. The chaining therefore cuts the cleanup failure's back-reference
into the chain it is joining (its own genuine inner cause, if it has one,
is left alone) and does nothing at all when the cleanup failure is already
in the chain. The relationship is not lost; it is stated in the one
direction that terminates. Reachable through the ordinary
`_abandon_call`-fails path, so it is a real defect rather than a
theoretical one, and it has a dedicated regression guard.

| Event | Consumes a call? |
|---|---|
| successful stochastic training forward (`0 < p < 1`) | **yes, exactly one** |
| successful stochastic forward with `requires_grad=False` everywhere | **yes** (a draw happened) |
| successful stochastic forward on an **empty** tensor | **yes** (§6.4) |
| `p == 0` | no |
| evaluation mode (`NativeDropout` in eval) | no |
| `p` / generator / input validation failure | no |
| native output allocation failure | no |
| native mask allocation failure | no |
| native kernel failure | no |
| graph-node construction failure | no |
| backward (one-shot, retained, or repeated) | no |
| abandoning or closing a graph | no |
| a stale-parameter or freed-graph backward error | no |
| checkpoint save | no |
| checkpoint load failure at any stage | no |
| successful checkpoint load | no — it **sets** `calls`, it does not advance it |
| counter exhausted | no (the reservation is refused) |
| a second reservation while one is active | no (refused before an index is minted) |
| commit or abandon with a stale, foreign, or already-finished token | no |
| commit itself failing **before** it takes effect | no — outcome 1: the reservation is abandoned and the index stays retryable |
| an exception **after** a successful commit, before the result is returned | **yes, exactly one — irreversibly.** Outcome 3: the index is spent, the committed token is not abandoned, the unreturned result is released, and the original exception propagates |

Two consequences worth stating explicitly, because they are what makes
resume exact:

- After a successful forward, **nothing** the caller does to the graph
  changes the counter. Retaining it, backwarding through it repeatedly,
  abandoning it, or closing it all leave `calls` where the forward left
  it.
- The counter is a property of the generator, not of any tensor. Two
  models sharing one generator interleave deterministically in call
  order.

---

## 6. Dropout probability semantics

### 6.1 Validation

`p` is validated identically everywhere it is accepted (the operation,
the Core method, and the module constructor all call one shared
validator, so the accepted/rejected matrix is identical by construction —
the pattern Phase E used for cross-entropy targets).

| Input | Result |
|---|---|
| `bool` (`True`/`False`) | `TypeError` — a bool is not a probability |
| non-real (`str`, `None`, list, complex, tensor) | `TypeError` |
| `int` `0` or `1` | `0` accepted (normalizes to `0.0`); `1` rejected |
| real in `[0.0, 1.0)` | accepted, normalized to `float(p)` |
| `p == 1.0` | `ValueError` — **rejected** (§6.3) |
| `p > 1.0` | `ValueError` |
| `p < 0.0` | `ValueError` |
| `NaN` | `ValueError`, with a message naming NaN explicitly |
| `+inf` / `-inf` | `ValueError` |

`numbers.Real` is the accepted abstract type (excluding `bool`), so a
NumPy float scalar is accepted at the Python surface and normalized with
`float(p)` — the same latitude the stable `Dropout` gives. The canonical
stored form is always a plain Python `float`.

### 6.2 `p == 0`

Identity. The operation returns the input tensor **unchanged and
un-copied**, builds no graph node, allocates nothing, calls no kernel, and
consumes no generator call. This matches the stable `tensorforge.nn.Dropout`
(`if not self.training or self.p == 0.0: return x`) and the
`NativeSequential` empty-sequence forward, both of which already return
the input object itself. §12 records the aliasing consequence.

### 6.3 `p == 1` is rejected

Locked: `0 <= p < 1`.

At `p == 1` inverted Dropout's multiplier `1/(1-p)` is a division by zero,
and every alternative is worse than an error: producing `inf`
multipliers poisons the gradient, special-casing to an all-zero output
silently changes the layer's expected-value contract, and defining
`0 * inf = 0` invents a rule that the mask arithmetic does not have. The
stable framework already rejects `p == 1`, so accepting it natively would
also break the parity the two lines otherwise keep. A caller who wants a
zeroing layer can multiply by zero explicitly.

### 6.4 Empty tensors

A tensor with zero elements is a legal input. The forward allocates
zero-element output and mask cores, the kernel draws nothing, and the
operation **still consumes one call** — the call index belongs to the
operation, not to the number of draws. This keeps the counter a function
of the sequence of forwards rather than of the data, which is what a
resume needs when a final batch is ragged.

**Reachability note (recorded at G2).** The kernel and the C ABI
implement this rule exactly — a `count` of `0` draws nothing, writes
nothing, and succeeds — but the native tensor representation, which
predates Phase G, rejects zero-size dimensions outright, so no empty
`NativeTensorCore` can currently be constructed to exercise it from
Python. G2 proves the case at the two layers where it is reachable and
pins the representation's limit with an explicit test. The contract above
is unchanged and becomes reachable, with no kernel change, whenever
zero-size shapes become expressible; that is a tensor-representation
change with its own stride conventions and is not part of Phase G.

---

## 7. The native forward contract

### 7.1 Layers

Milestone **G2** adds, bottom to top:

1. `tf::dropout_forward_contiguous(...)` — an **internal** C++ compute
   kernel (a hidden symbol in the new `cpp/src/random.cpp`), directly
   testable by a CTest binary.
2. `tf_core_dropout_forward(...)` — the exported, exception-guarded C ABI
   wrapper, self-validating at the trust boundary, added to
   `_CHECKED_KERNELS`.
3. `NativeTensorCore._dropout_forward_with_mask(p, *, seed, call_index)`
   — the private Core method returning `(output, mask)`.
4. `NativeTensorCore.dropout_forward(p, *, seed, call_index)` — the
   public Core method returning the output only, closing the mask
   deterministically before returning.

This is exactly the D8/D9 pooling shape: a private helper that keeps the
saved state, and a public Core method that discards it.

### 7.2 Signature and layout

```
TF_EXPORT void tf_core_dropout_forward(
    void*    input_handle,  int64_t input_offset,
    void*    output_handle,                 /* caller-allocated, offset 0 */
    void*    mask_handle,                   /* caller-allocated, offset 0 */
    int64_t  count,                         /* element count */
    uint64_t seed,
    uint64_t call_index,
    double   p);
```

- **Contiguous storage only.** Non-contiguous inputs are handled at the
  Core layer by Policy B (`docs/native_cnn_design.md` §5): a private
  owning contiguous copy is materialized, used, and closed as soon as the
  native call returns. No stride metadata crosses the ABI.
- `seed` and `call_index` are the complete random key. The kernel holds
  no state between calls and has no way to obtain any.
- The wrapper validates its own arguments — non-null handles, storage
  spans large enough for `count` from each offset, `count >= 0`,
  `0.0 <= p < 1.0` and finite — and writes **nothing** to either
  destination when it rejects, per the Phase-E self-validating export
  precedent.

### 7.3 Tensor contract

| Aspect | Contract |
|---|---|
| dtype / device | `float64` / `cpu` only; anything else is a `ValueError` before allocation |
| rank | any rank ≥ 0; a 0-d scalar has one element and gets one draw |
| empty | allowed; zero draws, zero-element results, one call consumed |
| contiguity | any; non-contiguous rides Policy B |
| element order | **logical row-major index over the logical shape** |
| input mutation | never; the input is read-only to the kernel |
| aliasing | output and mask share storage with neither the input nor each other |
| output layout | fresh **owning** row-major contiguous core, input's shape |
| mask layout | fresh **owning** row-major contiguous core, input's shape |
| mask dtype | `float64`, holding exactly `0.0` or `1/(1-p)` |

**Logical-layout independence** is a locked property, not an accident:
because Policy B materializes a non-contiguous input into row-major
contiguous storage before the kernel runs, the kernel's flat traversal
index *is* the logical row-major index. A transposed view and its
`contiguous_copy()` therefore receive the **same mask** for the same
`(seed, call_index)`, and the mask never depends on physical strides,
offsets, or how the tensor was built. G2 tests this directly (§4.7).

### 7.4 Two allocations, one atomic result

The Core method allocates in a deterministic order — **output first, then
mask** — and if either allocation or the kernel call fails it closes
whichever objects it created, closes any Policy-B temporary, leaves the
caller's input untouched, and returns nothing. No partial result is ever
observable. This is the same failure-atomic shape
`_maxpool2d_forward_with_winners` already uses.

The alternative designs were considered and rejected:

- *A single structured native result object* would add a new
  cross-boundary type for one caller; two cores match the existing
  precedent and need no new ABI concept.
- *A mask-only kernel plus the existing differentiable `multiply`* (the
  Daedalus shape) is attractive because it needs no new autograd node —
  but the mask would then be a graph **parent**, freed only by garbage
  collection rather than at the deterministic graph-release points, which
  violates the project's native-lifetime discipline. It is rejected for
  that reason alone; the numerics would have been identical.

### 7.5 No backward kernel

There is **no** `tf_core_dropout_backward` and no
`NativeTensorCore.dropout_backward`. The gradient of inverted Dropout is
`upstream * mask`, and `NativeTensorCore.multiply` already computes
exactly that over two owning contiguous float64 cores. Phase G adds one
forward kernel and no more — the same "compose the backward from existing
Core operations" decision Phase E made for `softmax`/`log_softmax`.

### 7.6 No generator inside the runtime

`NativeTensorCore` and every C++ translation unit remain **generator-free**:
no generator object, no counter, no seed storage, no static or
thread-local random state, no lazy initialization. The Core method's
`seed` and `call_index` are plain integers supplied by the caller. A
guardrail test asserts that no C++ source defines a random-state symbol
and that no Core method mutates a generator.

---

## 8. Autograd contract

### 8.1 Surface

```
NativeTensor.dropout(p, *, generator)
```

A **method**, keyword-only generator, **no default and no fallback**.
There is no functional `tensorforge.experimental.dropout(...)` helper and
no module-level convenience wrapper in Phase G: one surface, one
contract. Omitting the generator is a `TypeError`, not an implicit global
stream.

Milestone **G3**, shipped. Returns a fresh **owning** tensor of the
input's shape, except for the `p == 0` identity passthrough of §6.2.

### 8.2 Graph behavior

| Question | Answer |
|---|---|
| result `requires_grad` | exactly when the input does |
| parents | `(input,)` |
| saved state | the mask core, via `graph_resources=(mask,)` |
| backward formula | `grad_input = upstream * mask` (Core `multiply`) |
| does backward reread the input? | **no** |
| does backward use the generator? | **no** |
| expected parameter versions | `()` — none recorded |
| higher-order autograd | not supported (as everywhere in the native line: backward callbacks compute at the graph-unaware Core level and produce graph-free gradients) |
| `retain_graph=True` | mask retained; a second backward reproduces the identical gradient |
| one-shot `backward()` | mask released exactly once during the graph cleanup |
| abandoned graph | mask released exactly once by `close()` (or the `__del__` fallback) |
| no-grad forward | the mask is closed immediately by `_from_op`; the call is still committed |

**Backward never rereads the input.** It consumes exactly two things —
the graph-owned multiplier mask its own forward saved, and the upstream
gradient — and it never touches the generator. Because of that:

- mutating the input after the forward (including `copy_value_` on a
  directly versioned `NativeParameter` input) cannot change the gradient,
  and must **not** raise a stale-graph error — this is deliberately the
  `maxpool2d`/`cross_entropy` archetype, not the `log`/`multiply` one;
- reseeding, resetting, advancing, or replacing the generator's state
  after the forward cannot change an existing graph's gradient;
- a later `load_state_dict` or `load_native_checkpoint` cannot change an
  existing graph's saved mask. (A *full* checkpoint load also replaces
  parameters, so the unchanged v3.7 parameter-version rule may still
  stale such a graph through some *other* node — that is a parameter
  contract, never a Dropout effect, exactly as §7 of the normalization
  design says for BatchNorm snapshots.)

### 8.3 Ownership transfer at construction

The mask is created by the Core forward and is owned by the operation
until `_from_op` either adopts it into the graph history or closes it.
If graph construction itself raises, the operation closes **both** the
mask and the output, abandons the reservation, and re-raises — the same
`try/except BaseException` shape `maxpool2d` and `cross_entropy` already
use. The mask is never reachable from user code, never a public
`NativeTensor`, never a parameter or buffer, never in a `state_dict()`,
and never in a checkpoint.

---

## 9. `NativeModule` generator registration

Milestone **G1**. Generators become a **fourth registration category**
beside parameters, buffers, and child modules, built with the existing
conventions rather than a new object model.

### 9.1 Storage and reserved names

- A new insertion-ordered registry `_generators` (`name ->
  NativeGenerator`) created in `NativeModule.__init__` with
  `object.__setattr__`.
- `_RESERVED_NAMES` grows to
  `{"_parameters", "_modules", "_buffers", "_generators", "training"}`.
  Adding a name to that frozenset is the only change to existing
  registration behavior, and it can only reject a name that was already
  pathological.

### 9.2 APIs

```
register_generator(name, generator)     # explicit form; None unregisters
generators(recurse=True)                # unique generators, list
named_generators(prefix="", recurse=True)  # (dotted_name, generator)
generator_state_dict()                  # {canonical_name: state dict}
load_generator_state_dict(state, strict=True) -> LoadStateDictResult
```

- **Assignment registers.** `module.g = NativeGenerator(...)` registers,
  mirroring `NativeParameter` and `NativeModule`. A `NativeGenerator` is
  an unambiguous native type, so it does not need the explicit-call
  discipline buffers have (a plain `NativeTensor` is ambiguous; a
  generator is not).
- `register_generator` is the strict form: a non-generator value raises
  `TypeError` (assignment would store an ordinary attribute), and `None`
  unregisters, raising `KeyError` when nothing is registered under the
  name.
- `module.g = None` and `del module.g` unregister, exactly as for
  parameters and children.
- **One category per name.** Registering a generator evicts the name from
  the parameter, buffer, and child registries and from `__dict__`;
  registering a parameter, buffer, or child evicts it from `_generators`.
  Replacement within `_generators` preserves the slot's position; a name
  that moves between registries is appended to the target.
- Name validation reuses `_validate_registration_name` (non-empty str,
  no dots) and the reserved-name rejection.

### 9.3 `__getattr__` / `__delattr__`

Extended to consult `_generators` in the existing order, after parameters
and buffers and before child modules, so a registered generator reads
back as the exact object that was registered.

### 9.4 Traversal

`named_generators()` rides the **same** `named_modules()` walk as
parameters and buffers: deterministic pre-order depth-first, a module's
own generators before its descendants', deduplicated by object identity
with the first-discovered dotted name winning, cycle-safe. `generators()`
is the same walk without names. `recurse=False` restricts to direct
generators.

### 9.5 Why generators are **not** in `state_dict()`

`state_dict()` is contractually `{canonical_name: NativeTensor}` — every
consumer (its own validator, `load_state_dict`, the checkpoint model
section, every existing test) depends on the values being tensors. A
generator is not a tensor and has no shape, dtype, or device, so putting
one in that mapping would either break those consumers or force a
tensor-shaped lie (encoding two `uint64`s as float64 elements, which
cannot represent them exactly).

Generators therefore get their **own state section**:
`generator_state_dict()` returns an insertion-ordered
`{canonical_name: {"algorithm", "algorithm_version", "seed", "calls"}}`
of independent plain dicts. `state_dict()` is byte-for-byte unchanged for
every existing model, and a model with no generators produces an empty
generator state dict.

### 9.6 Atomic replacement

`load_generator_state_dict(state, strict=True)` follows the established
validate → stage → commit shape and returns the same
`LoadStateDictResult(missing_keys, unexpected_keys)` type
`load_state_dict` returns:

1. validate `strict` and the mapping type;
2. compute canonical keys, missing, and unexpected; under `strict=True`
   any of either raises `ValueError` reporting both lists;
3. hand every matching `(canonical_name, generator, state)` to the
   shared generator transaction below.

#### The multi-generator transaction

The replacement itself lives beside the lock it has to reason about, as
`replace_generator_states(entries)` — the generator analogue of
`_native_state.replace_native_state`. Its ordering is **validate → lock
→ recheck → snapshot → commit**:

1. **Validate.** Every entry's state is checked against its generator
   (exact four-key mapping, algorithm and algorithm-version equality
   against the **live** generator, seed and counter type and range), with
   the canonical name naming it in errors. Targets are deduplicated by
   **identity**; one generator supplied twice with *different* states is
   a conflict and is rejected outright, so an aliased key can never
   half-apply.
2. **Lock.** Every unique target's lock is acquired — all of them, held
   together — in one **global order that is independent of the caller**:
   sorted by object identity. A user-visible order (canonical names,
   mapping order, registration order) differs between two modules that
   share generators, which is precisely the case that deadlocks; identity
   does not. Locks release in reverse order.
3. **Recheck.** *While every lock is held*, each target is rechecked for
   a reservation — published **or holding a construction claim** (§3.6).
   This is the check that matters, and it cannot be raced: no target can
   begin a reservation without the lock this transaction is holding.
4. **Snapshot.** Each target's previous `(seed, calls)` is captured.
5. **Commit.** The writes run. They are integer assignments and **cannot
   fail**, so the only way out of the loop early is an asynchronous
   exception, and the rollback restores from the snapshots using the same
   non-failing primitive — **before any lock is released**. No other
   thread can ever observe a partially committed transaction.

**No reservation may begin on any target between the recheck and the end
of the commit.** A concurrent reservation therefore has exactly two
outcomes: it wins the lock first, and the load then rejects without
mutating anything; or it waits, and observes the finished state. It can
never overlap the commit, and it can never have its seed replaced
underneath a live token's index.

The global order also holds for **every** entry into this transaction,
including one reached from a finalizer. That follows from §3.6: because
reservation tokens are constructed with no generator lock held, no thread
can be inside a finalizer while owning a generator lock. A transaction
started from a finalizer therefore begins owning nothing and takes the
global order like any other caller — it cannot start from the middle of
the order and reach backwards, so finalizer or callback reentry cannot
invert the multi-generator lock order.

Any failure — validation, conflict, or reservation — leaves every
generator's state, identity, and reservation exactly as they were, with
no new reservation and no partially loaded target. Because generators own
no native storage, no failure can move the native live-storage count
either.

This is deliberately **not** routed through
`_native_state.replace_native_state` — that primitive replaces
`NativeTensorCore`s and owns native storage lifetimes, and a generator
owns none. Reusing it would require inventing a fake core. The two
transactions stay separate and each stays small.

Loading a generator state moves **no** parameter version, touches no
tensor, and makes no graph stale.

---

## 10. Checkpoint format version 2

Milestone **G5** — **shipped**. The format **name** stays
`"tensorforge.native_checkpoint"` forever; only the version moves, and it
moved once, at G5. Everything in this section is implemented exactly as
written, with two recorded strengthenings noted in §10.5 and §10.7.

### 10.1 Manifest change

`_MANIFEST_KEYS` becomes
`{"format", "format_version", "model", "optimizer", "generators", "metadata"}`.

The new top-level field is `"generators"`:

```json
"generators": null
```

when the model has no registered generators, or

```json
"generators": {
  "keys": ["features.3.generator", "classifier.1.generator"],
  "entries": {
    "features.3.generator": {
      "algorithm": "tensorforge.splitmix64",
      "algorithm_version": 1,
      "seed": "12297829382473034410",
      "calls": "37"
    },
    "classifier.1.generator": { "...": "..." }
  },
  "aliases": {
    "features.3.generator": "features.3.generator",
    "features.7.generator": "features.3.generator",
    "classifier.1.generator": "classifier.1.generator"
  }
}
```

Three fields, and exactly three:

- **`keys`** — the ordered canonical name list.
- **`entries`** — one state object per canonical name, mapping exactly
  `keys`, in the same order. This is the `keys`/`entries` shape the model
  section already uses, so the loader's existing structural validator
  style applies unchanged.
- **`aliases`** — the complete **registered-path → canonical-name**
  map (§10.3). This is the new part, and it is what makes the archive
  describe the model's generator *topology* rather than only its state.

### 10.2 No arrays

Generator state adds **no** array to the NPZ payload. It is four scalar
fields per generator and lives entirely inside the manifest JSON. The
array-name space (`model::NNNNNN`, `optimizer::m::NNNNNN`,
`optimizer::v::NNNNNN`) is untouched, and the existing
duplicate-reference / missing-array / unreferenced-extra checks need no
new cases.

### 10.3 Shared generators and the alias topology

A shared generator's **state** is written exactly **once**, under its
canonical name, exactly as a shared parameter's tensor is. But unlike a
shared parameter, a generator's *sharing topology is itself semantic
state*: two Dropout layers sharing one generator consume one interleaved
stream, while two independent generators consume two. Restoring the
states without restoring the topology would silently accept a model whose
stochastic behavior after the resume is different from the one that was
saved — which is exactly the class of "looks exact, is not" failure §10.6
refuses elsewhere. So version 2 records the topology explicitly.

**What is preserved.** The `"generators"` section preserves, together and
verifiably:

1. each canonical generator's complete state (algorithm, version, seed,
   counter),
2. **every** registered generator path in the model, not only the
   canonical ones,
3. the alias → canonical relationship for each of those paths,
4. and therefore the shared-versus-independent identity topology: two
   paths are shared in the archive **iff** their alias entries name the
   same canonical entry.

**Canonical-name selection is deterministic**, and it is the rule the
native line already uses everywhere: the **first name discovered by the
`named_generators()` pre-order depth-first walk** (§9.4) wins — a
module's own generators before its descendants', in registration order,
deduplicated by object identity. That walk is fully determined by the
module tree and the registration order, so the same model always produces
the same canonical names.

**Serialization order is deterministic.** `keys` and `entries` are in
`named_generators()` canonical order. `aliases` is in **full traversal
order** — every registered path in the order the same walk visits it,
*without* identity deduplication. Both orders are functions of the model
alone, so saving the same model twice produces byte-identical manifests.

**Every canonical name also appears in `aliases`**, mapped to itself.
There is no implicit "canonical names are omitted" rule: `aliases` is the
complete path set, so `set(aliases) ⊇ set(keys)`, `aliases[k] == k` for
every `k` in `keys`, and the alias map alone answers "what generator
objects does this model have, and which paths share them". This costs one
short string pair per canonical generator and removes an entire class of
"is a missing alias an error or a shortcut?" ambiguity.

**Alias cycles cannot be expressed.** An alias value must be a member of
`keys`, and a `keys` member always maps to itself, so the relation is a
one-step map into a fixed set, not a chain. There is nothing to follow
and no cycle to detect — the representation is chosen so the failure mode
does not exist. A load nevertheless asserts the one-step property
directly (`aliases[aliases[x]] == aliases[x]`) rather than relying on the
argument.

**Comparison is strict model traversal.** On load, the archive's alias
map is compared against the live model's own `named_generators()`
traversal — every registered path and its canonical target — and any
difference in either direction fails. Concretely:

- an archive that shares two paths while the live model has two
  independent generators **fails**;
- an archive with two independent generators while the live model shares
  one **fails**;
- an archive whose canonical name for a shared generator differs from the
  live model's (because registration order changed) **fails**, naming
  both;
- a path present in one and absent from the other **fails**, naming it.

**Loading never replaces a generator object.** A matched load calls each
live generator's `load_state`, so identity is preserved and every module
that registered it keeps the same object — the same identity-preserving
rule `load_state_dict` follows for parameters and buffers. The archive
never constructs a `NativeGenerator`.

### 10.4 Integer encoding

`seed` and `calls` are serialized as **canonical decimal strings**:
`^(0|[1-9][0-9]*)$`, no sign, no leading zeros, no separators, at most 20
digits, parsing to a value in `[0, 2**64 - 1]`.

JSON has no integer width, and while Python's `json` round-trips
arbitrary-precision ints exactly, a `uint64` above `2**53` is not
representable in the IEEE double that most JSON readers use. Writing the
manifest so that *any* conforming reader can inspect it without silent
precision loss is worth two string fields. `algorithm_version` stays a
plain JSON int (it is small by construction). The existing
`step_counts` ints are unchanged — they are bounded by the number of
optimizer steps, not by `2**64`.

### 10.5 Validation

**Every check below happens in prevalidation — before any staging and
therefore before any live model, buffer, optimizer, or generator state
changes.** A topology mismatch is detected while the model is still
completely untouched.

| Condition | Result |
|---|---|
| `"generators"` field absent from a v2 manifest | `ValueError` (field-set check) |
| `"generators"` is neither `null` nor an object with exactly `keys`/`entries`/`aliases` | `ValueError` |
| `entries` does not map exactly `keys`, in order | `ValueError` |
| a canonical entry the aliases reference is **missing** from `entries` | `ValueError` naming it |
| an **unexpected** canonical entry no alias references | `ValueError` naming it |
| archive canonical key set ≠ the live model's canonical generator names | `ValueError` naming missing and unexpected keys |
| an entry is not an object with exactly the four fields | `ValueError` |
| `algorithm` ≠ the live generator's algorithm | `ValueError` naming both |
| `algorithm_version` ≠ the live generator's | `ValueError` naming both |
| `seed`/`calls` not a canonical decimal string | `ValueError` |
| `seed`/`calls` outside `[0, 2**64 - 1]` | `ValueError` |
| a duplicate name in `keys` | `ValueError` |
| a duplicate path in `aliases` (a repeated JSON object key) | `ValueError` |
| an alias **missing** for a path the live model registers | `ValueError` naming the path |
| an **unexpected** alias for a path the live model does not register | `ValueError` naming the path |
| an alias whose value is not a member of `keys` (a target with no entry) | `ValueError` naming both |
| a canonical name absent from `aliases`, or `aliases[k] != k` for a canonical `k` | `ValueError` |
| `aliases[aliases[x]] != aliases[x]` (a multi-step or cyclic relation) | `ValueError` |
| archive shares two paths, live model has independent generators | `ValueError` naming the paths |
| archive has independent generators, live model shares one | `ValueError` naming the paths |
| the canonical name of a shared generator differs from the live model's | `ValueError` naming both |
| a malformed path (empty, non-string, or not a valid dotted name) in `keys` or `aliases` | `ValueError` |
| any target generator has an **active reservation** (§3.6) | `RuntimeError`, nothing staged |

Matching is **strict in both directions**, like the model section and the
optimizer presence rule: an archive may not omit a generator or a
registered path the model has, and may not carry one the model does not.
The comparison is against a real `named_generators()` traversal of the
live model, not against a name list the caller supplies.

**Two strengthenings recorded at G5**, neither weakening anything above:

1. **The manifest rejects a repeated JSON object key anywhere**, not only
   in `aliases`. Python's `json` silently keeps the *last* occurrence,
   which for the alias map would turn "this archive names one path twice,
   with two different canonical targets" into "this archive is fine" — a
   topology corruption that reads as valid. The loader parses with an
   `object_pairs_hook` that raises instead, and applies it to the whole
   manifest because no section benefits from a silently dropped key.
2. **A save is refused while any registered generator has a reservation
   in flight** — published *or* holding a construction claim — with the
   same `RuntimeError` a load raises, before the temporary file exists
   and therefore before the destination can be touched. §10.5 already
   refused such a *load*; a save has the same problem for the same
   reason, because a generator whose next index has been decided but not
   committed has no single honest state to record. Every registered
   generator's state is read in **one** locked snapshot
   (`snapshot_generator_states`, the read half of §9.6's transaction,
   taking the same global `id()` lock order), so the states an archive
   carries were true *together* rather than a microsecond apart.

### 10.6 Version-1 compatibility

Locked:

- **New saves always write version 2**, whether or not the model has
  generators. A model without generators writes `"generators": null`, so
  absence is explicit rather than inferred from a missing field.
- **A version-1 archive remains loadable** into a model that has **no**
  registered generators. Its manifest is validated against the v1 field
  set (no `"generators"` key), and the load proceeds exactly as it does
  today.
- **A version-1 archive loaded into a model that has registered
  generators fails** with a clear error naming the generators that the
  archive cannot supply. **No seed and no counter is ever fabricated** —
  not zero, not a fresh entropy draw, not the generator's current value.
  A silently invented stream would produce a resume that looks exact and
  is not, which is the single worst outcome available here.
- **A version-2 archive with a non-null `"generators"` section loaded
  into a model with no generators fails** as an unexpected-generator
  error.
- Any other `format_version` fails as it does today.

The loader accepts `{1, 2}` and dispatches on the value; there is no
"latest wins", no upgrade-in-place, and no silent rewrite of an old file.

### 10.7 Whole-checkpoint transaction contract

`load_native_checkpoint` is **one transaction over the whole archive**,
not three independent per-component transactions. It has exactly four
phases, and the guarantee attached to each is stated separately because
they are genuinely different guarantees.

#### Phase 1 — prevalidation (nothing has been touched)

The **entire** archive is opened, decoded, and validated first: the
container, the UTF-8/JSON manifest, the format name and version, the
complete field set, the model section, the optimizer section, the
generator section including the full §10.5 alias topology, the metadata,
and every array reference. Every cross-check against the **live** model,
optimizer, and generator traversals happens here.

If anything fails, **nothing whatsoever has changed** — no parameter, no
buffer, no optimizer slot, no generator, no object identity, no version
counter. Not one component is partially inspected into a live object.

#### Phase 2 — staging (everything that can fail, fails here)

Every value the commit will need is fully materialized before any of it
is installed:

- every model tensor and persistent buffer value, as owning native cores;
- every optimizer moment array and step count;
- every metadata value;
- every generator's validated `(algorithm, algorithm_version, seed,
  calls)` state, its canonical entry, and its alias relationship;
- and, for rollback, a **snapshot or replacement object for every live
  target that the commit will overwrite** — the previous cores for the
  parameter and buffer transaction, the previous optimizer values, and
  each generator's previous four-field state.

**Every operation that can allocate or raise happens in this phase**:
array decoding, native allocation, dtype and shape checking, contiguity
materialization, string parsing, integer range checking. A staging
failure closes every staged core in `finally` and leaves the live model,
optimizer, and generators untouched — the transaction aborts with the
same "nothing changed" guarantee as Phase 1, and native live storage
returns to its baseline.

#### Phase 3 — commit (atomic under any ordinary synchronous exception)

Commit order is **model → optimizer → generators**, and the whole commit
is wrapped in one rollback guard.

Every individual commit step is either **non-failing by construction** —
a pointer swap into an already-validated slot, or four assignments of
immutable integers for a generator — or **covered by complete rollback**.
The F1 state primitive already provides this shape for parameters and
buffers, with an explicit COMMIT BOUNDARY and exactly-once close; Phase G
extends the guard to span the optimizer and generator commits as well
rather than leaving three guards that each only protect their own
component.

**If any ordinary synchronous exception is raised anywhere in the commit,
the rollback restores all of:**

- every model parameter,
- every persistent buffer,
- the complete optimizer state (moments and per-parameter step counts),
- every generator's algorithm, algorithm version, seed, and call counter.

And afterwards:

- **no partially loaded component is observable** — a caller that catches
  the exception sees exactly the pre-load model, optimizer, and
  generators;
- **every generator object identity is unchanged** (states are replaced
  in place; the loader never constructs or substitutes a generator);
- every parameter and buffer object identity is unchanged, per the
  existing F1 contract;
- **graph-owned saved masks from graphs built before the load are
  unchanged** — they are private native state owned by graph history,
  reachable from no registry, and no phase of a load touches them
  (§8.2);
- staged cores are closed exactly once and native live storage returns to
  baseline.

Rollback itself is chosen to be unfailable: it restores previously-held
objects and immutable integers, allocating nothing and calling nothing
that can raise.

#### Phase 4 — the one honest exception

The **only** documented exception to whole-checkpoint atomicity is
**external asynchronous termination of the process or death of the
interpreter** — `SIGKILL`, a power loss, a hard interpreter crash — which
no in-process rollback can survive by definition, and which this project
does not pretend to handle.

An **asynchronous exception that is nevertheless deliverable to Python**
(`KeyboardInterrupt`, a signal handler raising, `GeneratorExit`) is
**not** an exception to the guarantee: the commit guard catches
`BaseException`, rolls everything back, and re-raises, the same
`try/except BaseException` discipline the rest of the native line already
uses. This is a strengthening of the earlier per-component wording, which
left a `KeyboardInterrupt` between two commits able to leave the model
restored and the optimizer stale. It no longer can.

The four cases are therefore distinct and must be tested distinctly:

| Class | When | Guarantee |
|---|---|---|
| prevalidation failure | Phase 1 | nothing inspected into any live object; no staging occurred |
| staging failure | Phase 2 | nothing committed; staged natives closed; live storage back to baseline |
| synchronous commit failure | Phase 3 | full rollback of model, buffers, optimizer, and generators; identities preserved; nothing partial observable |
| asynchronous process/interpreter death | outside Python | **the only** uncovered case; explicitly not claimed |

#### How G5 implements it

The commit still goes through each component's **own** loader —
`NativeModule.load_state_dict`, `optimizer.load_state_dict`,
`replace_generator_states` — so no component grew a second loading path.
What G5 added is the guard around all three, in the private
`_native_checkpoint_transaction` module, plus the **rollback snapshots**
Phase 2 already required: an independent owning copy of every parameter
and persistent buffer (with its current version), the optimizer's
complete `state_dict()`, and every generator's `(seed, calls)`. Because
every allocation the rollback could need happens in staging, the rollback
itself is plain attribute assignment and cannot raise:

- generators are written back through `_assign_state`, the same
  non-failing integer seam the multi-generator transaction uses;
- the optimizer's scalars and step counts are reassigned and each live
  moment tensor **swaps cores** with its snapshot;
- every parameter and buffer swaps cores with its snapshot and has its
  version written straight back, so a rolled-back load moves no version
  and stales no graph.

Swapping (rather than handing over) keeps ownership trivially correct on
every path: the live object always owns exactly one core, the loader's
snapshot wrapper always owns exactly one core, and the loader's existing
`finally` closes every staged tensor and every snapshot exactly once
whether the transaction committed or rolled back — which is why native
live storage returns to baseline either way.

One consequence is stated rather than glossed: `NativeAdam.load_state_dict`
**releases** the moment buffers it replaces, so a rolled-back load
restores the optimizer's moments **by value into its current buffer
objects** rather than restoring the original buffer objects. Those
buffers are private optimizer internals with no public identity contract
(a *successful* load replaces them outright), while every publicly
identified object — each `NativeParameter`, each persistent buffer, each
`NativeGenerator`, and the optimizer itself — is the same object
afterwards on both paths.

**Lock order.** See §10.8: the commit runs under one shared
state-transaction guard, with generator locks taken **under** it in the
existing global `id()`-sorted order.

---

## 10.8 Serializability — the shared state-transaction guard

Milestone **G5** — shipped alongside §10.7, and the other half of the same
guarantee.

§10.7 makes a checkpoint load atomic with respect to **failure**. That is
not the same as atomic with respect to **other threads**. Every native
state-replacement path was already individually all-or-nothing —
`replace_native_state` for parameters and persistent buffers,
`replace_generator_states` for generators, each optimizer's
`load_state_dict`, and the whole-checkpoint transaction over all of them
— and two of them running concurrently could still *each* succeed and
leave the model from one archive beside the optimizer or the generators
from the other. Deadlock freedom does not prevent that: a **hybrid final
state assembled from two checkpoints** is a corruption no per-component
guarantee can see, and it is exactly the "looks exact, is not" failure
this phase refuses everywhere else.

So G5 adds one requirement to every participating replacement: the
execution must have a **valid serial order**. After two concurrent
operations finish, the complete live state equals one of them followed by
the other — never a mixture.

### 10.8.1 The guard

One private, process-wide `threading.RLock` in `_native_state_lock.py`,
reached only through `state_transaction()`. It is not exported, not
reachable from `tensorforge.experimental.__all__`, and never handed to a
caller.

An `RLock`, and a single global one, for three reasons:

- **Reentrancy is required, not incidental.** The checkpoint transaction
  holds the guard and then calls the components' *own* public loaders,
  each of which takes it again. A plain `Lock` self-deadlocks on the
  first nested call, and the alternative — a second, lock-free internal
  entry point per component — is a duplicate commit path, precisely the
  kind of divergence this design avoids elsewhere.
- **One universal outer order.** A per-model or per-object lock needs a
  registry, a lifetime, and an ordering rule between unrelated models —
  and two transactions whose target sets *partially* overlap are the case
  that deadlocks. One process-wide lock has one order by construction.
- **Correctness over unrelated-model parallelism.** The critical sections
  are state replacement and checkpoint snapshotting, not training.

### 10.8.2 The universal state-replacement lock order

1. the shared state-transaction guard, **always first**;
2. then, when generators are involved, every unique target's lock in the
   existing global `id()`-sorted order (§9.6);
3. nothing acquires them in the opposite order, ever.

Both are taken together by `native_generator.locked_generators`, which is
also where the reservation recheck happens — so every path that touches
generator state gets item 1 before item 2 *by construction* rather than by
each caller remembering to. Even deciding the order (`_ordered_targets`)
happens inside the guard, which makes the property checkable at that seam
rather than only arguable from the source.

**Generator reservations deliberately do not participate.**
`_reserve_call` takes only its own generator's lock and never the guard.
That asymmetry is what keeps the two systems from inverting, and it gives
a racing reservation exactly two outcomes: it wins its generator's lock
first and completes before a transaction can take it, or it waits and
begins after the transaction has released it. No state replacement ever
happens underneath a live token.

### 10.8.3 Who participates

The commit portion of all of these, and they nest freely through the
`RLock`:

| Path | What it takes |
|---|---|
| `load_native_checkpoint` (the §10.7 commit) | guard, then every target generator lock |
| `replace_native_state` / `NativeModule.load_state_dict` | guard only |
| `replace_generator_states` / `load_generator_state_dict` | guard, then target generator locks |
| `NativeSGD.load_state_dict` | guard only |
| `NativeAdam.load_state_dict` | guard only |
| `save_native_checkpoint`'s snapshot | guard, then target generator locks |

There is exactly **one** guard object, shared by every module above; a
second lock anywhere would reintroduce the ordering problem it exists to
remove.

### 10.8.4 The load transaction under the guard

Ordinary work stays outside it. Archive parsing, the complete §10.5
validation, array decoding, and the staged `NativeTensor` values are all
produced before the guard is acquired — none of it touches live state.
Then, in order: acquire the guard; acquire every unique target generator
lock in `id()` order; recheck reservations and construction claims while
they are held; **capture the rollback snapshots**; commit model →
optimizer → generators; roll back — still holding both locks — if anything
raises; release the generator locks in reverse; release the guard.

The rollback snapshots moved *inside* the guard at this milestone, and
that is load-bearing rather than tidy: a snapshot taken before the guard
could describe a model another transaction has since replaced, and rolling
back to it would undo work this load never touched. They must reflect the
state at the real commit boundary, so they are captured at it.

Because the whole commit happens under the guard, a failed load's
**partial state is never observable** through any participating
operation: a second load waiting on the guard begins only after the first
has rolled back and released it.

### 10.8.5 The save snapshot

A save has the same problem in mirror image: a concurrent replacement
landing between the model snapshot and the optimizer or generator
snapshot would produce an archive describing a model that never existed.
So the whole snapshot — model state, persistent buffers, optimizer state,
and the generator topology and states — is captured under the same guard,
with generator locks taken under it as usual. The guard is released once
the complete immutable payload and its manifest exist; NPZ encoding and
the disk write happen outside it, because they touch no live state.

### 10.8.6 What this does not claim

The guard serializes **state replacement and checkpoint snapshotting**,
and nothing else. Ordinary training mutation — an optimizer `step()`, a
`copy_value_`, a backward accumulating gradients — deliberately does
**not** take it. So "thread-safe concurrent training snapshots" is *not*
claimed and must not appear on any surface: a save that overlaps a
concurrent `step()` can still capture a torn training state, because the
step never participates. What is claimed, exactly, is that participating
operations serialize with respect to each other.

One nuance is worth stating rather than leaving to be discovered: a
`NativeBatchNorm1d`/`2d` training forward commits its running statistics
through `replace_native_state`, so that particular training-time mutation
*does* participate and therefore cannot tear a concurrent save. That is a
consequence of routing every registered-state replacement through one
primitive, not a widening of the claim — `step()`, `copy_value_`, and
gradient accumulation still do not participate, so a concurrent training
loop can still produce a torn snapshot through them.

---

## 11. Exact stochastic resume

Milestone **G7** defines "exact resume" as the following, all under
`NativeAdam` on a fixed dataset with a `NativeDropout` in the model:

An interrupted run, checkpointed at step *k* and resumed into a
**completely fresh** model/optimizer/generator set, reproduces the
uninterrupted run **by exact equality** in:

- the remaining loss sequence,
- every model parameter,
- the full optimizer state (moments and per-parameter step counters),
- every persistent BatchNorm `running_mean` / `running_var`,
- every generator's `algorithm`, `algorithm_version`, `seed`, and `calls`,
- the **generator sharing topology** — which layers draw from one stream
  and which draw from their own (§10.3), since a resume that restored the
  states but not the topology would diverge on the very next step,
- every subsequent Dropout multiplier mask,
- the final training-mode prediction,
- the final evaluation-mode output.

Two uninterrupted runs from the same explicit seeds are likewise
bit-identical.

### 11.1 Deliberately outside the contract

The checkpoint captures model state, optimizer state, generator state,
and JSON metadata. It does **not** capture, and Phase G does not claim to
reproduce:

- data-loader position, batch order, or shuffle state (the native line has
  no data loader),
- any data-augmentation state,
- learning-rate scheduler state (the native line has no scheduler),
- Python's `random` module state,
- NumPy's global RNG state,
- the stable framework's RNG capture (`save_checkpoint(rng_state=True)`
  is a separate system on the other line and is untouched),
- thread counts, environment, or wall-clock behavior.

Reproducibility is therefore **exact for the state actually captured**,
and the documentation must say so in those words. Full-program
determinism is not claimed.

---

## 12. Stable / native separation

Unchanged and restated because Phase G touches a capability the stable
line already has:

- `tensorforge.nn.Dropout` keeps its current behavior, its NumPy RNG, and
  its checkpoint interaction. Phase G changes **no stable file**.
- There is no automatic backend dispatch and no implicit conversion
  between `tensorforge.Tensor` and `NativeTensor` in either direction.
- `NativeDropout` rejects stable tensors; `tensorforge.nn` modules reject
  native ones. Neither optimizer accepts the other line's objects.
- The native capability is reached only by explicit import from
  `tensorforge.experimental` (or `tensorforge.backends` for the Core and
  registry layers). `import tensorforge` never imports any of it.
- Phase G does not redefine, extend, or deprecate any stable public API,
  and adds no name to the stable top-level namespace.
- The two RNGs are unrelated: seeding NumPy does not affect a
  `NativeGenerator`, and a `NativeGenerator` never touches NumPy's global
  state.

---

## 13. Ownership and lifecycle matrix

"Live storage" means the native storage counter the test suite baselines.

| Object | Created by | Owner | Aliases | Released at | On failure | Affects live storage? |
|---|---|---|---|---|---|---|
| `NativeGenerator` | user or `NativeDropout.__init__` | the Python reference holder | any module that registered it | ordinary Python GC — **no `close()`, nothing to release** | nothing to clean | **no** |
| the generator's `threading.Lock` | `NativeGenerator.__init__` | the generator, privately | none — never exposed, never replaced | with the generator | never held across a native call, an allocation, or a failure | no |
| an active reservation | `_reserve_call` | the generator's single active slot, under the lock | the opaque token held by the one in-flight forward | `_commit_call` or `_abandon_call`, exactly once | abandoned on every failure path; the slot is cleared and `calls` is unchanged | no |
| a reservation token | `_reserve_call` | the calling forward | none — inert, generator-scoped serial, never reused | dropped when the forward ends | a stale, foreign, or already-finished token is refused and changes nothing | no |
| module-owned generator | `NativeDropout.__init__` | the module's `_generators` registry (reference only) | readable as `module.generator` | when the module and every other reference is dropped | unregistering never invalidates it | no |
| shared generator | the first owner | whoever holds the reference; registries only reference | every registration path that reaches it | as above; one object, one lifetime | as above | no |
| native Dropout output | Core forward | the returned `NativeTensor` (owning) | none | caller's `close()`, or GC fallback | closed by the operation if graph construction fails | **yes** |
| native multiplier mask | Core forward | the operation, then the **graph history** via `graph_resources` | none — never a public tensor | one-shot `backward()` cleanup, `close()`, or immediately when no grad is required; **exactly once** | closed by the operation on any pre-publication failure | **yes** |
| Policy-B contiguous temporary | Core forward | the Core method | none | `finally`, as soon as the native call returns | closed on every path | yes, transiently |
| saved autograd state (the mask) | Core forward | the graph node's `_graph_resources` | none | `_release_graph_resources`, exactly once | see mask row | yes |
| retained graph (`retain_graph=True`) | `backward` | the caller | — | the caller's `close()` or a later one-shot backward | mask stays alive across a **failed** retryable backward | yes, until released |
| abandoned graph | forward | the graph node | — | `close()`, or the `__del__` refcount/GC fallback | — | yes, until released |
| failed forward, before any allocation | — | — | — | nothing was allocated | reservation abandoned; `calls` unchanged | no |
| failed forward, after output allocated, before mask | Core forward | Core method | — | output closed immediately | reservation abandoned | returns to baseline |
| failed forward, after both allocated, kernel raised | Core forward | Core method | — | both closed immediately | reservation abandoned | returns to baseline |
| failed forward, after Core success, graph construction raised | Core forward | the operation | — | output **and** mask closed by the operation | reservation abandoned | returns to baseline |
| `p == 0` / eval passthrough | — | the **caller's** input tensor | the result **is** the input object | the caller's own `close()` | — | no |
| failed checkpoint prevalidation (incl. alias topology) | — | — | — | nothing staged or committed | model, optimizer, generators all untouched | no |
| failed checkpoint staging | staging | the loader | — | every staged core closed in `finally` | nothing committed; live state untouched | returns to baseline |
| failed checkpoint **commit**, any component | staging | the loader | — | every staged core closed in `finally` | **whole-transaction rollback** (§10.7 Phase 3): parameters, buffers, optimizer, and generator states all restored; every object identity preserved | returns to baseline |
| rollback snapshots (previous cores, previous optimizer values, previous generator states) | staging | the loader's rollback guard | — | released when the transaction ends, success or failure | this is what makes the commit unfailable | transiently |
| graph-owned masks during **any** checkpoint load | an earlier forward | that graph's history | none | with their own graph, exactly once | untouched by every load phase, success or failure | unchanged |
| module destruction | — | — | — | registries are references only; nothing is closed | — | no |
| interpreter shutdown | — | — | — | the project's existing contract: explicit `close()` is the guarantee, GC is a fallback, and process-exit retention by CPython/NumPy is not a TensorForge leak | — | measured against the live-storage baseline, not against process exit |

The single most important row: the **mask is graph-owned and released
exactly once**. It is the third member of the family that already
contains MaxPool2d winners and cross-entropy saved probabilities, and it
uses the same `graph_resources` mechanism with no second lifetime system.

Because the `p == 0` / eval passthrough returns the caller's own tensor,
the result must not be closed independently of the input. This is the one
case in which a Dropout result is not a fresh owning tensor, it matches
the stable `Dropout` and the empty `NativeSequential`, and G3/G4 tests
assert `result is input` so the aliasing is a checked contract rather
than an accident.

---

## 14. Failure matrix

"Advances" always means the generator's committed `calls` counter.

| Failure | Exception | Counter advances? | Live model state changes? | Native allocations remain? | Partial output observable? |
|---|---|---|---|---|---|
| invalid `p` (type) | `TypeError` | no | no | no | no |
| invalid `p` (range, NaN, inf) | `ValueError` | no | no | no | no |
| invalid seed type | `TypeError` | no | no | no | no |
| seed out of `[0, 2**64)` | `ValueError` | no | no | no | no |
| invalid counter type / range in a loaded state | `TypeError` / `ValueError` | no | no | no | no |
| unsupported dtype (not float64) | `ValueError` | no | no | no | no |
| unsupported device (not cpu) | `ValueError` | no | no | no | no |
| closed input tensor | `RuntimeError` | no | no | no | no |
| non-`NativeTensor` input (incl. a stable `Tensor`) | `TypeError` | no | no | no | no |
| `generator` omitted | `TypeError` | no | no | no | no |
| `generator` is not a `NativeGenerator` | `TypeError` | no | no | no | no |
| reservation already outstanding (reentrant or concurrent use) | `RuntimeError` | no — **no index is minted** | no | no | no |
| reservation already **claimed** (another thread, or reentrant from a finalizer) | `RuntimeError` | no — no index is minted | no | no | no |
| reservation-token construction itself fails (incl. `MemoryError`, `KeyboardInterrupt`) | the original exception | no | no — the `finally` clears the matching claim, publishes nothing, and leaves the serial unchanged | no | no |
| publication fails (the claim no longer matches) | `RuntimeError` | no | no — no reservation is published; the claim is cleared | no | no |
| **published, then delivery fails** (async exception before the caller receives the token) | the original exception | no | no — the exactly-matching reservation is cancelled, `calls` untouched; the serial is consumed | no | no |
| failed-delivery cleanup meets a newer, foreign, committed, or abandoned reservation | — (silent no-op) | no | no — that reservation is left strictly alone | no | no |
| `load_state` / `reseed` / `reset` / the multi-generator load, while a claim stands | `RuntimeError` | no | no | no | no |
| a multi-generator load reached **from inside** token construction, naming the claimed generator | `RuntimeError` | no | no — not even its co-targets are written | no | no |
| multi-generator load: a target reserves before the locks are taken | `RuntimeError` | no | no — no target is written | no | no |
| multi-generator load: conflicting states for one object through aliases | `ValueError` | no | no | no | no |
| commit with a token from another generator | `RuntimeError` | no — on **either** generator | no | no | no |
| commit with an already-committed token | `RuntimeError` | no — never twice | no | no | no |
| commit with an already-abandoned or stale-serial token | `RuntimeError` | no; the active reservation is untouched | no | no | no |
| commit or abandon when no reservation is active | `RuntimeError` | no | no | no | no |
| abandon with a foreign or already-finished token | `RuntimeError` | no; the active reservation is untouched | no | no | no |
| a non-token value passed to commit/abandon | `TypeError` | no | no | no | no |
| `load_state` / `reseed` / `reset` while a reservation is active | `RuntimeError` | no | no — seed and counter unchanged | no | no |
| counter exhausted (`calls >= 2**64 - 1`, checked under the lock) | `RuntimeError` | no | no | no | no |
| native output allocation failure | `MemoryError` | no | no | no — nothing was allocated yet | no |
| native mask allocation failure | `MemoryError` | no | no | no — the output is closed | no |
| kernel rejection at the ABI boundary | `ValueError` | no | no | no — both results closed | no — destinations are untouched by a rejecting kernel |
| other native failure | `RuntimeError` | no | no | no | no |
| Python wrapper (`NativeTensor`) creation failure | original exception | no | no | no — the Core results are closed | no |
| graph-node construction failure | original exception | no | no | no — mask **and** output closed | no |
| saved-state attachment failure | original exception | no | no | no — mask closed | no |
| the commit itself failing **before** it takes effect | original exception | no — the reservation is abandoned and the index stays retryable | no | no — result and mask closed | no |
| an exception **after** a successful commit, before the result is returned (§5, outcome 3) | original exception, **not** a cleanup error | **yes, exactly once — irreversibly**; the committed token is *not* abandoned | no | no — the unreturned result and its graph-owned mask are closed | no — no result escaped |
| a cleanup step itself failing during either of the above | the **operation's** original exception, with the cleanup failure chained onto it | as above for that outcome | no | every remaining step is still attempted | no |
| backward on a freed graph | `RuntimeError` | no | no | unchanged | no gradient committed |
| stale-parameter backward elsewhere in the graph | `RuntimeError` | no | no | unchanged; the mask stays alive for a retry | no gradient committed |
| checkpoint save: invalid path/model/optimizer/metadata | `TypeError` / `ValueError` | no | no | no | no file created |
| checkpoint save: write failure | `OSError` | no | no | no | destination byte-intact, no temporary left |
| checkpoint load: archive/JSON/UTF-8 corruption | `ValueError` | no | no | no | no |
| checkpoint load: unsupported `format_version` | `ValueError` | no | no | no | no |
| checkpoint load: v1 archive, model has generators | `ValueError` | no | no | no | no state invented |
| checkpoint load: missing generator entry | `ValueError` | no | no | no | no |
| checkpoint load: unexpected generator entry | `ValueError` | no | no | no | no |
| checkpoint load: generator algorithm / version mismatch | `ValueError` | no | no | no | no |
| checkpoint load: malformed seed/counter string or range | `ValueError` | no | no | no | no |
| checkpoint load: duplicate generator key or duplicate alias path | `ValueError` | no | no | no | no |
| checkpoint load: missing or unexpected alias path | `ValueError` naming the path | no | no | no | no |
| checkpoint load: alias targeting an absent canonical entry | `ValueError` | no | no | no | no |
| checkpoint load: unexpected canonical entry no alias references | `ValueError` | no | no | no | no |
| checkpoint load: canonical name missing from `aliases`, or not self-mapped | `ValueError` | no | no | no | no |
| checkpoint load: multi-step / cyclic alias relation | `ValueError` | no | no | no | no |
| checkpoint load: saved **shared** generator, live **independent** generators | `ValueError` naming the paths | no | no | no | no |
| checkpoint load: saved **independent** generators, live **shared** generator | `ValueError` naming the paths | no | no | no | no |
| checkpoint load: canonical name changed (registration order differs) | `ValueError` naming both | no | no | no | no |
| checkpoint load: malformed generator path | `ValueError` | no | no | no | no |
| checkpoint load: a target generator has an active reservation | `RuntimeError` | no | no | no | no |
| checkpoint load: **staging** failure (allocation, decode, parse) | propagates | no | no — nothing committed | staged cores closed in `finally` | no |
| checkpoint load: **synchronous commit** failure, model component | propagates | no | **fully rolled back** — parameters, buffers, optimizer, and generators all restored; identities preserved | staged cores closed in `finally` | **no** |
| checkpoint load: **synchronous commit** failure, optimizer component | propagates | no | as above — the model commit is rolled back too | staged cores closed in `finally` | **no** |
| checkpoint load: **synchronous commit** failure, generator component | propagates | no | as above — model and optimizer commits rolled back too | staged cores closed in `finally` | **no** |
| deliverable asynchronous exception mid-commit (`KeyboardInterrupt`, signal) | `KeyboardInterrupt` | no | **fully rolled back** — this is *not* an exception to atomicity | staged cores closed in `finally` | **no** |
| external process kill / interpreter death mid-commit | none reaches Python | undefined — the process is gone | undefined | undefined | **the one documented uncovered case** (§10.7 Phase 4) |
| graph-owned masks during any of the above | — | no | **unchanged in every case** | unchanged | no |

Every failure above leaves the generator's `calls` exactly where it was.
That is the single invariant the whole phase's resume story depends on,
and G6 exists to prove it boundary by boundary.

---

## 15. Testing strategy

Per milestone; none of these prescribes a performance gate.

**G1 — generator and registration.** Construction and validation matrices
(seed types, ranges, bools, NumPy scalars, `None`); `seed=None` produces
an in-range explicit seed and two generators differ; read-only
properties; `state()` independence; `load_state` validation and
all-or-nothing behavior; `reset`/`reseed`; identity (not value) equality;
absence of `close`, `copy`, `__deepcopy__`; exhaustion refusal.
**Reservation and locking (§3.6):** a reentrant `_reserve_call` while one
is active raising and minting no index; a **concurrent** attempt from a
second thread raising and minting no index; a stress loop over many
threads proving **no call index is ever handed out twice** and that
`calls` equals exactly the number of committed reservations; duplicate
commit refused with no second advance; duplicate cancel refused; a token
from a *different* generator refused on both; a stale-serial token
refused with the active reservation intact; commit/abandon with no active
reservation; a non-token argument raising `TypeError`; `load_state`,
`reseed`, and `reset` each refusing while a reservation is active and
changing nothing; and the lock proved **released** across the gap between
reserve and commit, so it is never held while a caller runs arbitrary
work between the two.
**Construction outside the lock (§3.6):** the token constructor
monkeypatched to observe, from inside itself, that **no generator lock is
held** — both by ownership and by an independent thread acquiring the
lock outright; that phase 1 published *only* the claim, with no active
reservation, no counter movement, and no serial movement; and then to
re-enter `_reserve_call`, `reseed`, `reset`, `load_state`, the module
load, and `replace_generator_states` directly — every one naming the
claimed generator raising rather than hanging and mutating nothing (not
even its co-targets), a transaction over *unrelated* generators
completing normally, and reversing the caller's mapping order changing
neither outcome — while state *inspection* keeps working.
**Construction failure:** `MemoryError`, `KeyboardInterrupt`, and a plain
`RuntimeError` each releasing the claim, publishing no reservation,
leaving `calls` unchanged, skipping no serial, leaving state replacement
and a later reserve/commit immediately usable, and moving no native
storage.
**Delivery failure (the publication-to-return window):** injected through
the private delivery seam rather than by real signal timing —
`KeyboardInterrupt`, `MemoryError`, and a custom `BaseException` each
propagating unchanged and leaving **no** claim, **no** active
reservation, `calls` unchanged, no lock held, state replacement working,
and a later reserve/commit succeeding; the call *index* unconsumed while
the opaque serial is consumed; repeated failures never stranding the
generator and moving no native storage; the same at
`calls == UINT64_MAX - 1`, where the last representable index stays
retryable and a later commit still reaches `UINT64_MAX`. **Exact-match
safety**, driven through the real seam wherever possible: the cleanup
cannot cancel a **newer** reservation taken inside the window, cannot
undo a **commit** made inside it, is inert on an **already abandoned**
token, ignores a **foreign** generator's token, ignores a **stale
serial**, ignores a **mismatched index**, and does nothing when no
reservation is live; and a discarded token is refused by both
`_commit_call` and `_abandon_call` afterwards.
**Cross-generator deadlock regression:** one thread reserving on the
generator that sorts **second** globally while its constructor starts a
replacement naming both in reverse caller order, and a second thread
gated so that it provably owns the first generator's lock before that
constructor runs — the exact inversion that hangs if the token is built
under the lock. Both threads must finish within bounded joins, neither
generator may be partially written, and the reservation must still commit
exactly one index. (Deterministic, event-gated, no sleeps, daemon threads
so a regression fails rather than hangs.)
**The multi-generator transaction:** a reservation taken between
validation and lock acquisition rejecting the whole load with nothing
written and the reservation still committable; a reservation racing the
commit never observing partial state; two modules holding the same
generators in **opposite** order loaded concurrently — with the threads
forced to interleave, so a caller-derived acquisition order provably
deadlocks and the global one provably does not; mapping order not
affecting the lock order; a shared generator locked and written exactly
once; conflicting alias states rejected; and a failed multi-target load
leaving every generator unchanged.
Registration: assignment and explicit registration,
`None`/`del` unregistration, one-category-per-name against all three
existing categories, reserved-name rejection, replacement position,
deterministic traversal order, identity deduplication, shared generators,
cycles, `recurse=False`, `generator_state_dict()` / strict and non-strict
`load_generator_state_dict` including atomic rollback, and proof that
`state_dict()` is unchanged for existing models and that no parameter
version moves.

**G2 — Core forward.** Known-answer vectors (§4.7) for `mix64`, the
stream, the bits, `u`, and full masks, **plus the equality-threshold
vector that pins the comparison direction** — `p == u` keeps, and
`nextafter(u, 1.0)` drops — with a negative control proving a `<=` kernel
would fail both the native and the Python boundary tests;
Windows/Linux equality asserted
against committed vectors; same `(seed, call_index)` reproduces the mask
bit-for-bit; a different `call_index` produces a different mask (and a
statistically sane keep rate); scalars; ranks 0–5;
contiguous, transposed, narrowed, and nonzero-offset inputs producing the
**same** logical mask; input non-mutation; no aliasing between input,
output, and mask; output and mask both owning and contiguous; the mask
holding exactly two values; the ABI's own validation rejecting bad
handles, spans, counts, and `p` without writing to destinations;
allocation-failure injection at each of the two allocations; a
Python-side wrapper failure between the two results proving the first is
still closed; live-storage
baseline restored after success and after each injected failure; a
dependency-free CTest binary over the internal kernel and the guarded
export. The **empty** case is asserted at the kernel and the ABI
(`count == 0` draws and writes nothing), plus an explicit test pinning
the tensor representation's rejection of zero-size dimensions, because no
empty core can be constructed today (see the G2 milestone block). Two
tests assert the separation directly: repeated Core calls leave a live
`NativeGenerator`'s `seed`, `calls`, and reservation slot bit-identical,
and passing a generator where an integer belongs is a plain `TypeError`.
Two further notes the vectors made necessary: a *thresholded* mask is one
bit per element, so two genuinely different streams can agree over a
short tensor by chance (at `p = 0.5`, seed 11 with call indices 4 and 5
agree over eight elements) — the committed **bit** vectors are the strong
statement, and the mask-difference checks use a wide enough sample that
an accidental agreement is not a realistic outcome.

**G3 — differentiable operation.** Forward equals `input * mask` for a
known mask; `p == 0` returns the input object; missing/invalid generator
rejected; exactly one call consumed per success and none per failure at
each boundary; no-grad forward consumes a call and closes the mask
immediately; gradient equals `upstream * mask` (checked against an
explicit reference and by finite differences on the *fixed* mask);
`retain_graph` reproducing the gradient; repeated backward; abandoned
graph release; mask lifetime and exactly-once release; no input reread
(mutating a parameter input afterwards changes nothing and raises
nothing); later generator mutation and later checkpoint load leaving an
existing graph's gradient equal to a clean control; graph-construction
failure closing both results and abandoning the reservation; a **failed
forward leaving no reservation outstanding**, so the very next forward
succeeds; a concurrent second `dropout` on the same generator raising
without consuming a call; and no two successful forwards on one generator
ever sharing a call index.

**G4 — module.** `p` validation; train vs eval; eval consumes no call and
returns the input object; `p == 0` likewise; a created generator versus a
supplied one — distinguished by **identity and registration** (a supplied
generator is the exact object given; a created one is a normally
registered fresh object), with an explicit assertion that **no public
`owns_generator` attribute exists** and that sharing a created generator
with a second module leaves no stale ownership claim behind;
`seed` and `generator` mutually exclusive; two modules sharing
one generator interleaving by call order; one module used twice; empty
tensor `state_dict()`; the generator registered under a stable canonical
name; repr; composition inside `NativeSequential`; capability registry
and export reconciliation (`"NativeDropout"` in `NATIVE_MODULES` and the
exports, nothing added to any operation inventory, and — asserted
explicitly, because it is the deliberate decision of §19/G4 —
**`"dropout"` still in `UNSUPPORTED`**, which still reads
`("dropout", "float32", "cuda", "amp")`).

**G5 — checkpoint v2.** Round-trip with and without generators; format
version is 2 and the format name is unchanged; `"generators": null` when
there are none; the three-field `keys`/`entries`/`aliases` shape; shared
generator state written **once** while **every** registered path appears
in `aliases`; canonical names self-mapped; deterministic canonical
selection and deterministic serialization order (saving the same model
twice is byte-identical); strict missing/unexpected canonical keys;
missing, unexpected, duplicate, and malformed alias paths; an alias
targeting an absent entry; an unexpected canonical entry; a multi-step
alias relation rejected; **saved-shared versus live-independent** and
**saved-independent versus live-shared** both rejected; a changed
canonical name rejected naming both; strict comparison against a real
`named_generators()` traversal; algorithm and version mismatch; malformed
and out-of-range seed/counter strings; duplicate keys; v1 archives still
loadable into generator-free models; v1 into a model **with** generators
failing and inventing nothing; v2 non-null into a generator-free model
failing; generator identity preserved across every load; no parameter
version moved by generator loading; **every topology mismatch proved to
fail in prevalidation, with the live model, buffers, optimizer, and
generators bit-identical to before the attempt**; the four §10.7 phases
tested distinctly — prevalidation failure, staging failure, and an
**injected synchronous commit failure in each of the model, optimizer,
and generator components**, each proving full rollback of *all four*
state families with every object identity preserved and nothing partial
observable; a deliverable `KeyboardInterrupt` injected mid-commit rolling
back identically; masks in a pre-existing graph unchanged after every
failed load; live-storage baselines restored after each; no benchmark or
result artifact written. **Serializability (§10.8)** is tested by forced
interleaving with barriers and events and bounded joins, never by sleeps:
a commit trace recorded at the real mutation seams *inside* the guard must
hold one contiguous run per thread, and the final state must equal exactly
one operation's result — two concurrent checkpoint loads of two different
archives (both caller orders, a seam forced at each commit position, and a
generator-free model so the property is not quietly resting on generator
locks); a checkpoint load against `load_state_dict`, against each
optimizer's `load_state_dict`, and against `load_generator_state_dict`; a
save snapshot against each of those replacements, with the resulting
archive proved to be one coherent serial point; the guard proved to be one
private reentrant `RLock` shared by every participant and held at every
commit seam; every ordered generator-lock acquisition proved to happen
with the guard already held and in sorted `id()` order; a reservation
racing a transaction proved to precede or follow it with no seed moving
under a live token; a failed load proved to roll back completely *before*
the next load observes anything; and the honest boundary — an optimizer
`step()` and a `copy_value_` proved **not** to take the guard.

**G6 — hardening.** The complete §14 matrix executed, each row asserting
counter, model state, native allocation, and observability; the §3.6
token matrix — concurrent and reentrant reservations, stale tokens,
duplicate commit, duplicate cancel, foreign tokens, and state replacement
attempted during a live reservation — under threads as well as
sequentially, including the no-duplicate-index stress proof; live-storage
baselines across success and failure cycles; a NumPy tripwire proving the
mask path reaches no NumPy numerical routine; `_CHECKED_KERNELS`,
`AUTOGRAD_OPS`, and the C++ sources proving no random state exists in the
runtime; and an explicit guard that `"dropout"` is **still** in
`UNSUPPORTED` at this milestone.

**G7 — resume.** Two uninterrupted runs bit-identical; an interrupted run
resumed into a fresh model/optimizer/generator set reproducing everything
listed in §11 by exact equality; the example importable with a
`train(...)` returning stats and a guarded `main()`.

**G8 — benchmark.** Every correctness gate runs before any timing; a
failed gate publishes nothing and exits nonzero; `--case`, `--family`,
`--smoke` (`--quick`), `--json`, and `--json-out`; no result file unless
`--json-out` names one; no asserted duration.

**G9 — integration.** §17.

**G10 — closure.** §18, including the capability-boundary move itself.

---

## 16. Benchmark contract

Milestone **G8**, shipped: `benchmarks/benchmark_native_dropout.py`,
`BENCHMARK_NAME = "tensorforge.native_dropout"`, version `"1.0"`, result
payload `schema_version` `"1.0"` — a number local to the harness, and
deliberately **not** the native checkpoint format version, which stays 2.

This section originally locked **six** cases. The shipped harness
measures those six under exactly those names and adds the
characterization the six alone cannot give — size scaling, layout,
probability, and the separate no-grad / differentiable / backward /
identity layers — for **35** cases in eight families, in this order:

| Family | Cases |
|---|---|
| `baseline` | `python_call_floor` |
| `core_reference` | `core_dropout_forward`, `core_dropout_forward_with_mask` |
| `size_scaling` | `scaling_core_scalar`, `scaling_core_tiny_vector`, `scaling_core_small`, `scaling_core_medium`, `scaling_core_large` |
| `layout` | `layout_contiguous`, `layout_transposed`, `layout_narrowed_noncontiguous`, `layout_offset_contiguous` |
| `probability` | `probability_core_p000`, `probability_core_p010`, `probability_core_p050`, `probability_core_p090`, `probability_core_pmax`, `probability_tensor_p010`, `probability_tensor_p050`, `probability_tensor_p090`, `probability_tensor_pmax`, `probability_module_p010`, `probability_module_p050`, `probability_module_p090`, `probability_module_pmax` |
| `tensor_operation` | `tensor_dropout_nograd_forward`, `tensor_dropout_forward`, `tensor_dropout_backward`, `tensor_dropout_forward_backward`, `tensor_dropout_p0_identity` |
| `module` | `module_training_forward`, `module_eval_forward`, `module_training_p0_identity`, `module_eval_p0_identity` |
| `training_step` | `dropout_training_step` |

The probability sweep is `0`, `0.1`, `0.5`, `0.9`, and
`nextafter(1.0, 0.0)` — never `p == 1`, which §6.3 rejects. At the
operation and module layers `p == 0` is a *different code path*
(identity), so those two layers' zero rows are the identity cases in the
`tensor_operation` and `module` families rather than repeated rows in the
sweep; the **Core** has no `p == 0` short-circuit and its zero row still
allocates, draws, and writes.

Methodology, following the Phase-E and Phase-F harnesses:

- **Correctness is gated before timing, always.** A global prologue pins
  the harness's reference to the committed G2 vectors and then pins the
  **native kernel** to the same vectors; each case's own gate then runs
  before the timing helper is ever reached. A failed gate publishes no
  timing and exits nonzero with clean stdout.
- Gates include: the **known-answer masks** for all seven committed
  `(seed, call_index, p)` vectors, verified element by element, plus the
  equality-threshold vector that pins the strict `u < p` comparison;
  output equal to `input * mask`; gradient equal to `upstream * mask`
  against the graph-owned mask; the generator's `calls` advanced by
  exactly the number of successful stochastic forwards and by **zero**
  for every evaluation and `p == 0` case; identity cases returning the
  caller's own object; the four layouts receiving one identical logical
  mask; the graph released and the mask closed; and the native
  live-storage counter returning exactly to its baseline.
- `time.perf_counter_ns`, warm-up, **one call per sample**, every sample
  retained, setup and cleanup outside the timer, a fresh model and
  optimizer per training-step repetition, median reported with min, max,
  spread, p25/p75, and the median absolute deviation. The one exception
  to one-call-per-sample is the **identity-dispatch** rows (evaluation
  mode and `p == 0`), which allocate nothing and sit far below the useful
  resolution of a single reading: they run a short **calibrated** inner
  loop whose iteration count is reported, and `python_call_floor`
  measures the same loop around a trivial Python call so that floor is
  visible rather than assumed. Those rows measure **dispatch**, not mask
  generation, and the report says so.
- Every stochastic case owns **one** generator for its whole run, so call
  indices advance monotonically; the consumed range is recorded and
  verified **exactly** against the number of cycles the harness
  performed. No generator is reset inside a timed region, no reservation
  is reused, and no case approaches counter exhaustion.
- An **untimed lifecycle verification** runs after the cases: repeated
  create/use/release cycles over every family, returning native live
  storage exactly to baseline with no reservation outstanding and every
  graph, mask, and gradient released. Its instrumentation is
  benchmark-local — G8 adds no runtime API.
- `--case`, `--family`, `--warmup`, `--repetitions`, `--smoke`
  (`--quick` is an alias), `--json`, and `--json-out PATH`; a fully
  JSON-native payload with no NaN or Infinity. **No result file of any
  kind is written unless `--json-out` explicitly names a destination**:
  there is no default path, no results directory, and no committed
  artifact.
- **No speed assertion, no committed timing number, and no CI timing
  threshold anywhere.** Published ratios name their numerator and
  denominator; the word "speedup" appears nowhere.

**Reference labels.** A stable-framework comparison is deliberately
omitted: `tensorforge.nn.Dropout` draws from a different RNG algorithm,
so no mask-for-mask comparison exists, and timing two different mask
distributions against each other would be a comparison of RNG
implementations dressed up as a Dropout benchmark. What the harness times
instead is:

- `numpy` — an **exact vectorized NumPy implementation of the same
  locked derivation**, doing the same work: the stream key, one 64-bit
  word per logical element, the top-53-bit uniform, the strict `u < p`
  test, the `1/(1 - p)` multiplier mask, and the output multiply, with
  both arrays allocated (and, for a non-contiguous input, the same
  Policy-B materialization). Agreement is asserted **bit for bit**, never
  to a tolerance. Only the stateless Core carries this label, because
  only the Core has a semantically equivalent NumPy expression. The
  reference is **benchmark-local**, never production code: a second
  production implementation would be a second source of truth and a
  silent NumPy fallback path.
- `native_only` — every `NativeTensor` and `NativeDropout` case. Those
  layers own a generator call transaction, native ownership, and (where
  applicable) an autograd graph that no NumPy expression has, so **no
  timing ratio is published** for them. Their gates are still exact,
  against the same reference at the reserved call index.
- `harness_baseline` — the `python_call_floor` row only.

Layer costs are reported as **approximate layered differences** between
adjacent measured cases (operation minus Core, differentiable minus
no-grad, module minus operation, forward-plus-backward minus forward),
explicitly labelled as descriptive gaps rather than a causal
decomposition — each side is an independent measurement with its own
noise, and the two sides do slightly different ownership work. A negative
value is reported as measured rather than hidden.

No benchmark-driven numerical shortcut is permitted: the harness composes
shipped public APIs only and may not add a fast path, a cached mask, or a
reused graph that the ordinary code path does not have. G8 is
**measurement only**: no runtime file changed, nothing was optimized to
improve a number, and the results are a machine-specific snapshot rather
than a performance contract.

---

## 17. Phase-G integration target

Milestone **G9**: `tests/test_native_phase_g.py`, one test-only model:

```
NativeConv2d(1, 4, 3) -> NativeBatchNorm2d(4) -> NativeReLU
  -> NativeMaxPool2d(2) -> NativeDropout(p, seed=...)
  -> NativeFlatten
  -> NativeLinear(16, 8) -> NativeBatchNorm1d(8) -> NativeReLU
  -> NativeLayerNorm(8) -> NativeDropout(p, seed=...)
  -> NativeLinear(8, 3)            # raw logits
  -> NativeCrossEntropyLoss
```

trained with `NativeAdam` on the existing fixed twelve-image three-class
dataset, interrupted, checkpointed at format **version 2**, and resumed
into a fresh model/optimizer pair.

It must prove, in one place, that the phase's new state coexists with
everything already there:

- **four** saved-resource families alive in one graph — Dropout
  multiplier masks, BatchNorm eval snapshots, MaxPool2d winner indices,
  and cross-entropy saved probabilities — each released **exactly once**
  with the graph history, with no registered buffer object *or storage*
  reachable from the graph;
- exact resume of the loss suffix, every parameter, the NativeAdam state,
  all four running-statistic buffers, **both generators' seeds and call
  counters**, the subsequent masks, the final training logits, and the
  final evaluation-mode logits, predictions, and accuracy;
- buffer-only and generator-only mutation leaving an earlier eval graph's
  gradients equal to a clean control, while a full checkpoint load or a
  `copy_value_` on an affine parameter correctly stales it through the
  unchanged parameter rule;
- the Phase-E versioning archetypes (saved-output `exp`, live-reread
  `log`, saved-probability cross-entropy) meeting Dropout masks;
- shared versus independent generators across the two Dropout layers,
  including the shared case deduplicating to one state **entry** while
  both registered paths appear in the archive's alias map, one restore
  reaching both layers, and a resume into a model whose sharing topology
  *differs* being **rejected in prevalidation** with nothing changed;
- eval mode consuming no generator call anywhere in the model;
- a **non-contiguous** NCHW input through the whole stack in both modes,
  with the mask proved layout-independent;
- shared and frozen parameters unaffected;
- strict stable/native separation;
- honest per-boundary failure atomicity, including the counter staying
  put after each failure, and a **whole-checkpoint synchronous commit
  failure over the fully integrated model** rolling back parameters,
  all four normalization buffers, the optimizer state, and both
  generators together, with external process death named as the only
  uncovered case (§10.7 Phase 4);
- a NumPy/conversion tripwire over one complete integrated step;
- native live-storage baselines across success **and** failure cycles.

---

## 18. Phase-closure requirements

Milestone **G10** closes the phase only when all of the following are
recorded with observed results. **This list is the gate on the capability
boundary**: `"dropout"` leaves `UNSUPPORTED` as the *last* act of G10,
after every item below has passed. If any item fails, the boundary does
not move and the phase is not closed.

- the full Python suite, with exact passed/skipped totals
- the focused Phase-G suites
- a **fresh Windows Release** build and its full CTest run
- a **fresh Windows Debug** build and its full CTest run, with the active
  runtime proved to remain the Release library
- a fresh **Clang ASan** build in WSL2 with instrumentation *proved*
  (dynamic sanitizer symbols present; the library refuses to load without
  the runtime)
- **UBSan** under the same build
- **LeakSanitizer** with `detect_leaks=1`
- the sanitized focused Python tests, with counts and zero diagnostics
  attributable to TensorForge
- the exact stochastic-resume example reproducing its resume under the
  sanitized library
- the benchmark smoke path passing every correctness gate under the
  sanitized library, writing no result file
- a practical lifecycle test returning native **live storage exactly to
  baseline**, with any remaining process-exit allocations attributed
  honestly and **no suppression file added**
- documentation reconciliation across every status surface
- **the capability-boundary move itself**, performed only after
  everything above has passed: `"dropout"` is removed from `UNSUPPORTED`,
  which then reads exactly `("float32", "cuda", "amp")`, while
  `"NativeDropout"` is already in `NATIVE_MODULES` and the exports from
  G4; `float32`, `cuda`, and `amp` stay unsupported; dtype and device are
  unchanged
- durable semantic guardrails replacing the milestone-era absence checks,
  including one that pins the post-closure tuple

---

## 19. Milestone ladder

| Milestone | Scope | Status |
|---|---|---|
| G0 | Architecture contract and design lock | **Complete** |
| G1 | `NativeGenerator` and module generator-state ownership | **Complete** |
| G2 | Deterministic native Dropout-forward Core | **Complete** |
| G3 | Differentiable `NativeTensor` Dropout | **Complete** |
| G4 | `NativeDropout` module and public export | **Complete** |
| G5 | Checkpoint version 2 and exact RNG restoration | **Complete** |
| G6 | RNG, graph, ownership, and checkpoint hardening | **Complete** (tests, one narrow fix, and documentation — no capability) |
| G7 | Deterministic stochastic training and exact resume | **Complete** (one example and its tests — no capability) |
| G8 | Honest native Dropout benchmark | **Complete** (one harness and its tests — measurement only, no capability) |
| G9 | Cross-cutting Phase-G integration | Not started |
| G10 | Phase-G closure, and `"dropout"` leaving `UNSUPPORTED` | Not started |

### G0 — Architecture contract and design lock

- **Objective.** Lock every architectural decision in this document
  before any RNG or Dropout code exists.
- **Scope.** This document, the minimum status reconciliation naming
  Phase G, and semantic guardrails asserting the contract.
- **Files.** `docs/native_rng_dropout_design.md` (new);
  `docs/roadmap.md`, `docs/project_summary.md`,
  `docs/backend_experiments.md`, `docs/native_support_matrix.md`,
  `README.md`, `CLAUDE.md` (status only); `tests/test_docs.py`
  (guardrails).
- **Public contract.** None. No API is added.
- **Ownership rules.** None introduced.
- **Failure rules.** None introduced.
- **Tests.** Documentation guardrails only (§20).
- **Docs.** This file; Phase G named as in progress everywhere it
  belongs.
- **Forbidden.** Any runtime, C++, ABI, ctypes, registry, export,
  checkpoint-format, example, or benchmark change. No placeholder module,
  empty class, stub kernel, or unreachable future-milestone code.
- **Done when.** Every decision in §3–§18 is resolved, the guardrails
  pass, `UNSUPPORTED` and the checkpoint version are unchanged, and the
  full Python suite passes.

### G1 — `NativeGenerator` and module generator-state ownership

**Complete.**

- **Objective.** Explicit, inspectable, serializable random state, and a
  first-class place for it in the module hierarchy.
- **Scope.** `NativeGenerator` (§3); the `_generators` registry and its
  APIs (§9); `generator_state_dict` / `load_generator_state_dict`. **No
  randomness is generated in this milestone** — a generator holds state
  and hands out call indices; nothing draws from it yet.
- **Files.** New `src/tensorforge/experimental/native_generator.py`;
  `native_module.py`; `src/tensorforge/experimental/__init__.py` (export
  `NativeGenerator`); new `tests/test_native_generator.py`;
  `src/tensorforge/backends/cpp.py` (`STATE_SUPPORT` gains
  `"generator_state"` — **reporting only**, placed between
  `load_state_dict` and `save_native_checkpoint` so it sits with the
  other in-memory state surfaces).

  `"generator_state"` is a *capability* name, like `"persistent_buffers"`
  beside it, covering `register_generator`, `generators()` /
  `named_generators()`, and the `generator_state_dict()` /
  `load_generator_state_dict()` pair. It reports **in-memory state
  only** and claims nothing more — and it still does: the file half is
  G5's separate `"checkpoint_generator_state"`. At G1 no generator state
  was written to or read from a checkpoint and no random value was
  generated anywhere. No other registry moved — `UNSUPPORTED` still reads
  `("dropout", "float32", "cuda", "amp")`, dtype and device are
  unchanged, and G1 left the checkpoint format at version 1.
- **Public contract.** §3.2–§3.5, §9.2–§9.6.
- **Ownership.** The generator owns nothing native; it owns one private
  `threading.RLock` and at most one reservation — a construction claim
  (§3.6) or a published one. Registries hold references only; identity is
  preserved across every state load.
- **Failure.** Constructor and `load_state` validate before mutating.
  Reservation creation is the §3.6 two-phase claim / construct / publish /
  deliver transaction: the token is allocated with **no generator lock
  held**, so no callback-capable operation ever runs while a lock is held;
  a failed construction releases the claim in `finally`, publishes
  nothing, and skips no serial; and a failure *after* publication but
  before the caller receives its token cancels the exactly-matching
  reservation instead, since the claim is already gone by then. Neither
  advances `calls`, and a caller never loses the only token for a live
  reservation. `load_generator_state_dict` runs the §9.6
  multi-generator transaction — validate → lock every unique target in
  the global identity order → recheck for reservations *and construction
  claims* under those locks → snapshot → non-failing commit, with the
  rollback completing before any lock is released. Concurrent and
  reentrant reservations, stale/foreign/duplicate tokens, state
  replacement during a live *or claimed* reservation, and exhaustion all
  raise without changing `seed`, `calls`, the serial, or the active slot.
  Nothing deadlocks: conflicting operations are rejected by the claim,
  overlapping loads share one acquisition order, and — because
  construction holds no lock — a finalizer cannot invert that order.
- **Tests.** The G1 block of §15.
- **Docs.** This file's status; the support matrix's state section.
- **Forbidden.** No kernel, no ABI symbol, no ctypes declaration, no Core
  method, no `NativeTensor` operation, no module, no Dropout, no
  checkpoint-format change. `"dropout"` stays in `UNSUPPORTED`.
- **Done when.** Generators register, traverse, deduplicate, snapshot,
  and restore atomically, and `state_dict()` is provably unchanged.

### G2 — Deterministic native Dropout-forward Core

**Complete.**

- **Objective.** The stateless native mask/output kernel and its Core
  boundary.
- **Scope.** §4 and §7: the internal kernel, the guarded export, the
  ctypes declaration, and both Core methods.
- **Files.** New `cpp/src/random.cpp` and `cpp/include/tf_random_internal.h`;
  new `cpp/tests/test_dropout_forward.cpp` and its `CMakeLists.txt`
  target; `src/tensorforge/backends/cpp.py` (declaration,
  `_CHECKED_KERNELS`, `TENSOR_CORE_OPS` gains `dropout_forward`); new
  `tests/test_native_dropout_core.py`.
- **Public contract.** §7.2–§7.4, shipped as
  `NativeTensorCore.dropout_forward(p, *, seed, call_index)` and the
  private `_dropout_forward_with_mask` that keeps the mask.
- **Ownership.** Two fresh owning contiguous cores; the private helper's
  caller owns the mask; Policy-B temporaries closed in `finally`.
- **Failure.** §14 rows for allocation, kernel, and validation; nothing
  partially observable; live storage returns to baseline.
- **Tests.** The G2 block of §15, plus the CTest binary.
- **Docs.** Support matrix Core-operation table; this file's status.
- **Forbidden.** No generator is touched by any Core or C++ code; no
  autograd node; no module; no backward kernel; `"dropout"` stays in
  `UNSUPPORTED`.
- **Done when.** The committed known-answer vectors reproduce on Windows
  and Linux and every layout produces the same logical mask.

**What shipped, exactly.** The `"tensorforge.splitmix64"` derivation of
§4.2–§4.4 as four internal `namespace tf` functions
(`splitmix64_mix`, `dropout_stream_key`, `dropout_element_bits`,
`dropout_uniform`), the `dropout_forward_contiguous` kernel, one guarded
export `tf_core_dropout_forward`, one ctypes declaration whose `seed` and
`call_index` cross as `c_uint64`, one entry in `TENSOR_CORE_OPS`
(`"dropout_forward"`) and one in `_CHECKED_KERNELS`, and the two Core
methods. The keep/drop decision is a deterministic function of
`(seed, call_index, logical_element_index, p)` and nothing else — not the
input values, not the address, not the strides, not the traversal order.
Committed known-answer vectors for `mix64`, the stream key, the element
bits, the bits-to-uniform conversion, seven full keep/drop patterns, and
the **equality-threshold vector** that pins the strict `<` comparison
(§4.7 — `p == u` keeps, `nextafter(u, 1.0)` drops, with a negative
control proving the vector discriminates `<` from `<=`)
are asserted **identically** in `cpp/tests/test_dropout_forward.cpp` and
`tests/test_native_dropout_core.py`, so neither side can redefine the
stream or the comparison alone; a test-only Python reference of §4.2–§4.4 exists in the
suite (never in production) and is pinned to those vectors before it is
allowed to generate any expectation.

**The empty-tensor row is not reachable from the Core, and that is
recorded rather than papered over.** §6.4 and §7.3 say a zero-element
tensor is a legal input, and the kernel and the C ABI implement exactly
that (`count == 0` draws nothing, writes nothing, and returns `TF_OK`).
But the native tensor representation predates this phase and **rejects
zero-size dimensions outright** — `shape` dimensions must be positive
ints, and `NativeStorage` requires a positive size — so no empty
`NativeTensorCore` can be constructed to hand in. G2 therefore proves the
empty case at the kernel and ABI layers, where it *is* reachable, and
pins the representation's limit with an explicit test. Nothing about the
contract changed: when zero-size shapes become expressible, the kernel
path is already correct. Making them expressible is a tensor-representation
change with its own stride conventions, and is deliberately not smuggled
into an RNG milestone.

**The `p == 0` split is preserved.** §6.2's identity bypass — returning
the caller's own tensor, allocating nothing, calling no kernel, consuming
no call — belongs to the operation layer (G3). At the Core, `p == 0` is a
legal probability that the kernel actually computes: the strict `<`
comparison drops nothing, so the mask is all `1.0` and the output equals
the input. `p == 1` is rejected at every layer, in Python and again at the
ABI.

### G3 — Differentiable `NativeTensor` Dropout

**Complete.**

- **Objective.** One autograd node over the G2 contract, with the
  call-consumption transaction.
- **Scope.** §5 and §8: `NativeTensor.dropout(p, *, generator)`, the
  graph-owned mask, the backward, and the reserve/commit/abandon
  transaction.
- **Files.** `native_tensor.py`; `cpp.py` (`AUTOGRAD_OPS` gains
  `dropout`).
- **Public contract.** §8.1–§8.2.
- **Ownership.** §13's mask, output, reservation, and token rows.
- **Failure.** No path advances the counter except a published success;
  every failure abandons the reservation, so no forward can be blocked by
  a leaked one.
- **Tests.** The G3 block of §15.
- **Docs.** Support matrix operation table; the autograd design's
  saved-state note.
- **Forbidden.** No module; no functional helper; no default generator;
  no global state; `"dropout"` stays in `UNSUPPORTED`.
- **Done when.** Gradients, lifetime, and the counter transaction are all
  proved, including at every failure boundary.

**What shipped, exactly.** `NativeTensor.dropout(p, *, generator)` in
`native_tensor.py`, and one name — `"dropout"` — appended to
`AUTOGRAD_OPS`. Nothing else moved: no C++, no C ABI symbol, no ctypes
declaration, no `NativeTensorCore` method, no module, no export, no
checkpoint-format change, and no other registry value. In particular the
backward is the existing `multiply` over the saved mask, so no
`dropout_backward` kernel or Core op was written (§7.5).

The operation is ordered exactly as §5 requires. Validation that needs no
generator runs first — the receiver is open, `generator` is a
`NativeGenerator`, and `p` goes through the **same**
`_normalize_dropout_probability` the Core and the future module use, so
the accepted/rejected matrix is identical by construction rather than by
duplication. `p == 0` then returns `self` — the caller's own object — with
no reservation, no allocation, no kernel call, and no graph node (§6.2).
Otherwise one call is reserved; the token is bound and the cleanup
boundary entered as the very next action; the key is the **reservation's**
(`token._index` for the index, and the seed read while that live
reservation makes every state replacement raise, so `generator.calls` is
never mistaken for the reserved index); the G2 Core runs outside the
generator's lock; `_from_op` adopts the mask through the unchanged
`graph_resources` contract; and `_commit_call` is the last state-changing
action before the result is returned.

Two private module-level seams make the transaction's failure positions
addressable rather than merely argued, in the same spirit as G1's
`_deliver_reservation`: `_dropout_backward(input_tensor, mask)` builds the
backward closure, and `_deliver_dropout_result(result)` is the deliberate
no-op between a fully constructed result and the commit. Neither is
exported or referenced from a public API. Every failure before the commit
takes effect — an invalid probability or generator, a closed receiver, an
exhausted
counter, a reservation conflict, a Core validation or allocation failure,
a Python wrapper failure, a backward-closure or graph-node or
graph-resource-attachment failure, a no-grad mask-cleanup failure, a
delivery failure, or a commit that raises *instead of* committing —
releases the result and the saved mask, abandons the
reservation, and re-raises, so `calls` is untouched and the very next
forward reuses the same unconsumed index and reproduces the committed
vector it would have produced.

A failure *after* a successful commit is **outcome 3** of §5 and is
handled differently on purpose: that index is irreversibly spent and is
never handed out again, the committed token is **not** abandoned (which
would raise "already committed" and mask the real failure), the
unreturned result and its graph-owned mask are still released, and the
original exception propagates unchanged. `_settle_failed_dropout`
implements both sides and decides between them from the token's recorded
outcome — through the private read-only `NativeGenerator._call_committed`
query — rather than from a local flag, because a commit can succeed and
the statement after it never run. It attempts every cleanup step even if
an earlier one fails, and chains any cleanup failure onto the operation's
exception instead of substituting for it.

Backward reads exactly two things, the upstream gradient and the
graph-owned mask, so it never rereads the input, never redraws, and never
reserves, commits, cancels, inspects, or mutates a generator. The node
therefore records **no** expected parameter version: mutating a directly
versioned input afterwards leaves the gradient correct for the forward
that ran and raises nothing, while a *full* checkpoint load still stales
such a graph through some **other** node's parameter rule — the
`maxpool2d`/`cross_entropy` archetype, unchanged. Reseeding, resetting,
loading a state, or running the multi-generator transaction after the
forward likewise cannot reach an existing graph.

The mask rides the existing lifetime mechanism with no second system: it
is released exactly once at the same deterministic points the graph
history is, `retain_graph=True` keeps it for another pass, a failed
retryable backward leaves it alive, an abandoned graph frees it through
`close()` (with `__del__` as the fallback), and a **no-grad** forward
closes it immediately while still committing the call, because a draw
happened.

One thing G3 did **not** get is a no-grad *context*: the native line has
none to add support for. Its graph is opt-in — a result is differentiable
only when a parent requires grad — so `detach()` is the equivalent, and it
takes the ordinary no-grad path and still consumes exactly one call.
Higher-order autograd is not supported, matching every other native
operation: the backward computes at the graph-unaware Core level and
produces a graph-free gradient.

The residual asynchronous window §3.6 documents is narrowed but not
closed, and is recorded rather than papered over. The operation binds the
token and enters its `try` as its next action, does no avoidable
callback-capable work before that boundary, commits last, and places no
allocation, callback, graph mutation, logging, or formatting after the
commit. What no pure-Python code can cover is an asynchronous exception
delivered *between* `_reserve_call` returning and the caller's `try`
being entered. It is a couple of bytecodes with no Python statement in
them, it is not a test-injectable or ordinary synchronous window, and
every window that *is* one has a test.

The **commit-to-return** window is different, and G3's revision draws the
distinction properly: the window still exists, but the `return` sits
*inside* the `try`, so an exception arriving there is caught and cleaned
up rather than escaping with an unreturned result and a spent index
unaccounted for. That cleanup is **outcome-aware** — §5's three outcomes
— and the outcome is read from the token through the private
`_call_committed` query, never from a flag set after `_commit_call`,
because the commit can succeed and that assignment never run. So an
interruption there consumes exactly one call (honestly reported, never
rolled back), releases the unreturned result and its graph-owned mask,
does **not** abandon the committed token, and propagates the original
exception rather than an "already committed" cleanup error.

### G4 — `NativeDropout` module and public export

**Complete.**

- **Objective.** The public layer — implemented and exported, but **not**
  yet advertised as a supported capability.
- **Scope.** §6 and the module contract; export reconciliation.
- **Files.** New `src/tensorforge/experimental/native_dropout.py`;
  `__init__.py` (export `NativeDropout`); `cpp.py` (`NATIVE_MODULES`
  gains `"NativeDropout"`).
- **Public contract.** `NativeDropout(p=0.5, seed=None, generator=None)`;
  training stochastic, evaluation identity, `p == 0` identity, neither
  consuming a call; the generator registered as module state.
- **Ownership.** The module owns its generator when it created one and
  merely references a supplied one; it owns no native storage. Ownership
  is expressed by **identity and registration**, never by a stored flag:
  the public surface is `p`, `generator`, `training`, and the ordinary
  `NativeModule` methods, and **no `owns_generator` attribute exists**.
- **Failure.** Constructor validation before any state; forward
  delegating every runtime failure to G3.
- **Tests.** The G4 block of §15, **including an explicit assertion that
  `UNSUPPORTED` still reads `("dropout", "float32", "cuda", "amp")`**.
- **Docs.** README, support matrix, project summary, backend
  experiments, this file — each describing Dropout as implemented at G4
  and **still listed unsupported** until G10.
- **Forbidden.** **Any change to `UNSUPPORTED`.** No checkpoint-format
  change; no `Dropout2d`; no functional helper; no stable-framework
  change; no public ownership flag or equivalent property.
- **Done when.** The module works in both modes, `"NativeDropout"` is in
  `NATIVE_MODULES` and the exports, and `"dropout"` is **still** in
  `UNSUPPORTED`.

**What shipped, exactly.** `src/tensorforge/experimental/native_dropout.py`,
the export in `__init__.py`, and one name — `"NativeDropout"` — appended
to `NATIVE_MODULES`. Nothing else moved: no C++, no C ABI symbol, no
ctypes declaration, no `NativeTensorCore` method, no `NativeTensor`
operation, no `AUTOGRAD_OPS` entry, no `STATE_SUPPORT` entry, no
checkpoint-format change, and no other registry value.

`NativeDropout(p=0.5, seed=None, generator=None)`. `p` goes through the
**same** `_normalize_dropout_probability` the Core and the operation use —
a third rule would be a third place for the matrix to drift — and is
stored as a plain `float`. `seed` and `generator` are **mutually
exclusive**: supplying both raises `TypeError` rather than silently
preferring one, because a quietly ignored seed is exactly the
"looks reproducible, is not" failure explicit state exists to prevent.
Without an explicit generator the module builds and owns
`NativeGenerator(seed)` (fresh OS entropy at `seed=None`); with one it
registers **that exact object**, never a copy, which is how two layers
share one interleaved stream (§3.7). Every argument is validated before a
generator is created or registered, so a rejected construction draws no
entropy, registers nothing, allocates nothing, and leaves a supplied
generator bit-identical.

**Which construction path ran is deliberately not recorded.** The public
surface is exactly `p`, `generator`, `training`, and the ordinary
`NativeModule` methods — there is **no ownership flag**, public or
private. "This module created its generator" is true of one moment in the
constructor, not durably: the same generator can be handed to a second
module a line later, at which point a stored Boolean would assert an
exclusivity the object graph contradicts, and a public mutable one would
additionally let a caller change the claim without changing any
registration, lifetime, or behavior. Ownership and sharing are read where
they are actually recorded — generator **identity** and the **registered
topology** (`a.generator is b.generator`, and `named_generators()` over
the model, which is also exactly what the §10.3 checkpoint alias topology
persists). The distinction still matters behaviorally, and it is the
observable one: the default gives every layer an independent stream, an
explicit generator gives several layers one interleaved stream.

The generator is registered under the canonical name `"generator"` and is
readable as `module.generator`. It is the **fourth** registration
category, so it appears in `generators()`, `named_generators()`, and
`generator_state_dict()` and is deliberately absent from `state_dict()`,
which stays contractually `{name: NativeTensor}`. A module with a Dropout
in it therefore has an unchanged tensor state dict. `load_generator_state_dict()`
replaces state in place, so identity — and any sharing — survives a load.
The module owns **no native storage**: constructing, registering,
running, and discarding one moves the live-storage count only by the
outputs its forwards return, and dropping the module never closes,
resets, or mutates the generator.

Forward is three cases and the layering is deliberate. Input validation
runs **first** (a `TypeError` for a non-`NativeTensor`, a `RuntimeError`
for a closed one), so evaluation mode is not a way to hand back an
invalid tensor. Training delegates to `NativeTensor.dropout`, which owns
the whole call transaction — so a successful training forward consumes
exactly one call and a failed one consumes none, and the module can add
no failure hole to a transaction it does not implement. Evaluation
returns the **input object itself**, consuming no call and allocating
nothing, so an arbitrary number of eval forwards leaves **no gap in the
stream**: a training forward at index *n* followed by any number of eval
forwards is followed by a training forward at index *n + 1*. `p == 0` is
identity too and is deliberately **not** short-circuited in the module:
§6.2 assigns that rule to the operation, which already returns the
caller's own tensor before reserving, allocating, or drawing anything, and
a second copy of the rule could only ever disagree with the first.

**The version-1 checkpoint limitation was recorded rather than glossed,
and G5 closed it.** At G4 the format was still version 1 with no
generator section, so saving a model containing a `NativeDropout`
preserved its parameters and buffers and **silently omitted the random
stream**; loading one left the live generator exactly as it found it and
— the important half — **fabricated nothing**. That omission was the
honest behavior of the format as it then stood, and it was pinned by
test. **G5** moved the format to version 2, added the `"generators"`
section, and added the §10.6 rejection rule that makes a v1 archive
loaded into a generator-bearing model an error instead of a quiet
omission. What is still missing after G5 is the *end-to-end* proof — an
interrupted stochastic training run resumed into a fresh
model/optimizer/generator set (§11) — which is G7. That, together with
the unrun closure matrix, is precisely why `"dropout"` stays in
`UNSUPPORTED`.

*Why `"dropout"` does **not** move here:* Phase F moved its capability
names at the module milestones (`"layernorm"` at F2, `"batchnorm"` at
F4), and Phase G deliberately does not follow that precedent. The
difference is what the capability carries. LayerNorm and BatchNorm were
composed from already-validated native operations and added no new kernel,
ABI symbol, or persisted state. Dropout adds a **new random algorithm**
whose cross-platform reproducibility is the whole point, a **new C ABI
kernel**, a **new registered state category**, and a **new checkpoint
format version** — and a capability whose value is exact reproducibility
is not a capability until reproducibility has actually been demonstrated.
Advertising it at G4 would mean the registry says "supported" while the
committed known-answer vectors have not been reproduced under a fresh
Release build, a fresh Debug build, and ASan/UBSan, and while checkpoint
v2 does not yet exist to persist the stream at all. So for Phase G the
registry reports a **closed, validated** capability: `"dropout"` stays in
`UNSUPPORTED` for the whole of G0–G9 and is removed at G10, as the last
act of the closure matrix in §18. The export and `NATIVE_MODULES` entry
land at G4 and honestly describe what exists; `UNSUPPORTED` describes what
is finished.

### G5 — Checkpoint version 2 and exact RNG restoration

**Complete.**

- **Objective.** Persist and restore generator state exactly.
- **Scope.** §10 in full.
- **Files.** `native_checkpoint.py`; `cpp.py` (`STATE_SUPPORT` reporting);
  the private `_native_state_lock.py` guard and the participating
  loaders (`_native_state.py`, `native_generator.py`, `native_sgd.py`,
  `native_adam.py`).
- **Public contract.** Format name unchanged; `_FORMAT_VERSION` becomes
  **2**; the `"generators"` manifest section with its
  `keys`/`entries`/`aliases` topology (§10.1, §10.3); v1 compatibility
  per §10.6.
- **Ownership.** No new native allocation for generator state; generator
  commits are pure Python; rollback snapshots per §10.7 Phase 2.
- **Concurrency.** §10.8: one private shared state-transaction `RLock`,
  outermost, with generator locks under it in the global `id()` order;
  every participating replacement (checkpoint load, model state load,
  both optimizer state loads, generator state load) and the save
  snapshot run under it, so concurrent operations **serialize** rather
  than merely avoiding deadlock.
- **Failure.** §14's checkpoint rows; **all four §10.7 phases
  implemented and distinguished** — every validation and topology check
  in prevalidation, everything that can allocate or raise in staging, and
  one rollback guard spanning the model, optimizer, and generator commits
  so that any ordinary synchronous exception (and any deliverable
  asynchronous one) restores all four state families.
- **Tests.** The G5 block of §15.
- **Docs.** Checkpoint sections of the support matrix and backend
  experiments; this file.
- **Forbidden.** No scheduler or dataloader state; no `map_location`; no
  partial or remapped loading; no fabricated seed or counter; no array
  added to the NPZ payload. **`"dropout"` stays in `UNSUPPORTED`.**
- **Done when.** v2 round-trips exactly including the alias topology, v1
  stays loadable exactly where §10.6 says, every mismatch fails before
  anything changes, an injected commit failure in each component leaves
  the model, buffers, optimizer, and generators bit-identical to their
  pre-load state, and two concurrent loads leave the complete state equal
  to one archive or the other — never a mixture.

**What shipped, exactly.** `native_checkpoint.py` (format version 2, the
`"generators"` section, its validators, and the four-phase load), the new
private `_native_checkpoint_transaction.py` (the one rollback guard
spanning the model, optimizer, and generator commits), the private
`_native_state_lock.py` shared state-transaction guard together with the
participating loaders that now take it (`_native_state.py`,
`native_generator.py`, `native_sgd.py`, `native_adam.py`), the private
`locked_generators` / `snapshot_generator_states` helpers in
`native_generator.py`,
one private traversal helper `NativeModule._named_generator_paths` (the
undeduplicated path walk the alias map is built from and compared
against), and one reporting-only registry name —
`"checkpoint_generator_state"` appended to `STATE_SUPPORT`. Nothing else
moved: **no C++, no C ABI symbol, no ctypes declaration, no
`NativeTensorCore` method, no autograd operation, no module, no export,
and no public entry point** — persistence rides the existing
`save_native_checkpoint` / `load_native_checkpoint` pair.

`"checkpoint_generator_state"` is a *separate* name from G1's
`"generator_state"` precisely because that one was explicitly scoped to
memory: through G4 a save preserved parameters and buffers and silently
omitted the random stream. The two names now read as what they are — the
in-memory surface and the file surface. Neither is a Dropout capability
claim: `UNSUPPORTED` still reads `("dropout", "float32", "cuda", "amp")`,
`SUPPORTED_DTYPES` is `("float64",)`, and `SUPPORTED_DEVICES` is
`("cpu",)`.

**What G5 deliberately did not do.** It proves *exact generator
restoration* — the state, the identity, the topology, and the next
Dropout mask against the G2 Core at the restored index — but **not** the
end-to-end §11 resume: an interrupted stochastic training run reproduced
into a fresh model/optimizer/generator set is milestone **G7**, with its
example and integration test. No benchmark, no training example, no
hardening suite, and no result artifact of any kind exists.

### G6 — RNG, graph, ownership, and checkpoint hardening

**Status: complete.**

- **Objective.** Prove §13 and §14 by executable test.
- **Scope.** Tests and documentation, plus the one narrow fix a hardening
  test exposed.
- **Files.** New `tests/test_native_phase_g_hardening.py` for the
  cross-cutting G1–G5 invariants; the focused suites keep their own narrow
  matrices.
- **Public contract.** Unchanged.
- **Forbidden.** Any registry, export, or schema movement. **`"dropout"`
  stays in `UNSUPPORTED`.** If a defect is found, it is fixed with the
  narrowest possible change and recorded explicitly.
- **Done when.** Every §14 row and every §13 row has a test, including
  the §3.6 concurrency and token rows and the §10.7 four-phase rows.

**What it proved.** The reservation transition matrix — every invalid
token transition asserting *five* invariants at once (no counter movement,
no active-reservation change, no construction-claim change, no serial
reuse, no native-storage movement) — and the four §3.6 failure positions
distinguished by whether a serial was consumed. The exact `uint64`
boundary as §4.6's table, row by row, with the final index proved
retryable until committed and repeated exhaustion failures freezing every
field. Forced concurrent interleavings under barriers and events with
bounded joins and no sleeps: no duplicate call index, unrelated generators
independent, no torn state read, a reservation racing a state replacement
provably preceding or following it in both orders, and a transaction
started *from inside* token construction refused rather than deadlocked.
The deterministic Core's **structural** key properties (§4.3's
characterization) beside its committed vectors, plus the probability
extremes and layout independence through real transposed and narrowed
views. Every pre-commit position of §5's transaction × `RuntimeError`,
`MemoryError`, `KeyboardInterrupt`, and a non-`Exception` `BaseException`,
each proving the retry reproduces the mask the failure would have
produced; and every post-commit position proving the index spent exactly
once. All **four** graph-owned saved-resource families — a Dropout mask,
MaxPool2d winners, BatchNorm eval snapshots, and cross-entropy
probabilities — coexisting in one graph and releasing exactly once, with
branched, chained, shared-generator, retained, failed-retryable, and
abandoned graphs each measured. A 76-case checkpoint corruption matrix,
every case failing before any live change with the model, buffers,
optimizer, and generators bit-identical. Whole-transaction rollback
injected at every commit position × the same four exception classes, with
identities, versions, active reservations, and pre-load graph masks all
proved untouched. Save-seam destination atomicity at all seven positions.
And repeated lifecycle loops — success and failure — returning native live
storage exactly to a measured baseline with no monotonic growth.

**The one defect found and fixed.** `_chain_cleanup_failure` closed a
**cycle** in the `__context__` chain when a cleanup step failed (§5). The
fix cuts the cleanup failure's implicit back-reference before appending it
and is inert when it is already in the chain; the original exception is
still primary and the cleanup failure still reachable. Nothing else in the
G1–G5 runtime changed — no C++, no C ABI symbol, no ctypes declaration, no
Core method, no operation, no module, no export, no schema field, and no
registry value.

### G7 — Deterministic stochastic training and exact resume

**Status: complete.**

- **Objective.** The end-to-end proof of §11.
- **Scope.** One example plus its integration test.
- **Files.** New `examples/native_dropout_training.py` and
  `tests/test_native_dropout_training.py`.
- **Forbidden.** No capability, operation, kernel, schema field,
  benchmark, or export. No timing. No stable-framework import.
  **`"dropout"` stays in `UNSUPPORTED`.**
- **Done when.** Two uninterrupted runs are bit-identical and the
  interrupted resume reproduces every item in §11 exactly.

**The model.** `NativeDropoutClassifier` —
`NativeLinear(4, 8, seed=0)` → `NativeBatchNorm1d(8)` → `NativeReLU` →
`NativeDropout(p=0.5, seed=20240707)` → `NativeLayerNorm(8)` →
`NativeLinear(8, 3, seed=1)` — over raw logits with
`NativeCrossEntropyLoss` and `NativeAdam(lr=0.05)`. It is the smallest
model that carries **all four** TensorForge-owned state families at once:
trainable parameters, persistent BatchNorm running buffers, a registered
`NativeGenerator`, and optimizer moments with per-parameter step counters.
Miss any one on restore and the trajectory diverges immediately — which
two deliberate negative controls prove rather than assume.

**The data and the schedule.** Twelve four-feature samples over three
classes, computed from an explicit arithmetic formula over the sample
index; every value is a quarter or an eighth and therefore exact in
float64. Three fixed batches of four, and step *s* always trains on batch
`s % 3` — the schedule is a **pure function of the step**, which is the
entire reason the external loop position collapses to one integer.
Nothing is shuffled, generated randomly, augmented, loaded, or
downloaded, and neither NumPy's global RNG nor Python's `random` is
touched.

**External loop state, carried honestly.** Checkpoint v2 captures
TensorForge-owned state and nothing else, so the loop position travels as
ordinary JSON metadata — `{"training_step": k, "next_batch_index": k % 3,
"lr": ...}` — and is **validated, never defaulted**, by the example's
`validated_progress`: a missing field, a `bool` where an `int` belongs, an
out-of-range step, or a `next_batch_index` disagreeing with the schedule
all raise. Silently restarting from step 0 is exactly the failure that
check exists to prevent, because such a resume would still converge and
still be a different run.

**What was proved.** Two uninterrupted runs are bit-identical across the
loss sequence, every parameter, the running statistics, the whole
optimizer state, the generator state, the final training logits, and the
final evaluation output. An interrupted run — checkpointed after
`split_step` **completed** steps (7, deliberately not a multiple of 3, so
the resume lands mid-cycle), with the interrupted model, optimizer, and
generator **released before the resume begins** — reloads into a
completely fresh set built with a *different* Dropout seed and reproduces
every §11 item by exact equality. The two negative controls are what make
that meaningful: a resume that restores all four families but restarts the
batch schedule at 0 **diverges**, and one that restores everything but
re-seeds the generator **diverges**. Evaluation is proved state-neutral —
repeated eval passes leave `calls` bit-identical, produce identical
outputs, restore the caller's mode, and leave the loss sequence of a probed
run equal to an unprobed one. `run_next_mask_proof()` closes the loop back
to the G2 Core: a **throwaway** reload (so the resumed run is untouched)
pushes a fixed probe through the restored `NativeDropout` and matches
`NativeTensorCore.dropout_forward` at the exact restored
`(seed, call_index)`, advancing `calls` by exactly one. The module's
private mask is never exposed; the Core supplies a reference *output*.

**Scope.** One example, its test module, and documentation. **No
capability, operation, kernel, C ABI symbol, ctypes declaration, Core
method, module, export, schema field, checkpoint version, benchmark, or
registry value changed**, and no runtime file was touched. The example
defines no public training API — none of its helpers is exported.

### G8 — Honest native Dropout benchmark

**Complete.** Measurement and characterization only.

- **Objective.** Measurement only, per §16.
- **Files.** New `benchmarks/benchmark_native_dropout.py` and
  `tests/test_native_dropout_benchmark.py`, plus status documentation and
  its guardrails. **No runtime file changed.**
- **What shipped.** The 35-case, eight-family harness of §16: the
  stateless Core against an exact bit-for-bit vectorized NumPy reference,
  scalar-through-131,072-element size scaling, four physical layouts over
  one logical shape, the five-value probability sweep at three layers,
  the no-grad / differentiable / backward-only / forward-plus-backward
  operation layers, the module's training and identity paths, and one
  complete Dropout training step — each correctness-gated first, each
  labelled with the reference it used, each reporting median with min,
  max, spread, p25/p75, and MAD, and each recording its exact generator
  consumption. Plus the untimed lifecycle verification that returns
  native live storage exactly to baseline.
- **What it found, and did not fix.** The characterization is reported as
  measured, including the results that are unflattering and the one that
  is surprising (the Core's cost varies strongly with `p` on the machine
  measured, which is a property of the data-dependent keep/drop branch,
  not a defect). **Nothing was optimized to improve a number**, and no
  runtime fast path, cache, relaxed validation, or altered ownership was
  added.
- **Forbidden, and observed.** No speed assertion, committed timing
  number, CI timing threshold, automatic result file, or capability
  change: **`"dropout"` stays in `UNSUPPORTED`**, which still reads
  `("dropout", "float32", "cuda", "amp")`.
- **Done when.** Every gate runs before timing and a failed gate
  publishes nothing. Observed: a corrupted forward and a non-finite
  forward both abort with `measure` never reached, stdout clean, and the
  exit status nonzero.

### G9 — Cross-cutting Phase-G integration

- **Objective.** §17.
- **Files.** New `tests/test_native_phase_g.py`.
- **Forbidden.** Any capability, operation, kernel, ABI, schema, example,
  benchmark, or export change. **`"dropout"` stays in `UNSUPPORTED`** —
  G9 is the last milestone at which that is still true.
- **Done when.** Every claim in §17 is asserted from the live registry,
  the real tree, or an executed workload.

### G10 — Phase-G closure and the capability boundary

- **Objective.** §18, and the single registry change the whole phase has
  been earning.
- **Files.** Documentation and documentation-guardrail tests, **plus one
  registry line**: `src/tensorforge/backends/cpp.py`, where `"dropout"`
  is removed from `UNSUPPORTED`, which then reads exactly
  `("float32", "cuda", "amp")`.
- **Ordering.** The removal is the **last** step. Every item in §18 —
  both builds, both CTest runs, ASan, UBSan, LeakSanitizer, the sanitized
  Python suites, the resume example, and the benchmark gates — is
  recorded with observed results **first**. A failure anywhere in that
  matrix means the boundary does not move and the phase does not close.
- **Forbidden.** Any numerical capability, C++, CTest, ABI, ctypes,
  example, benchmark, or production numerical file change. The
  `UNSUPPORTED` edit is the **only** permitted non-documentation change,
  and it adds no code path — `float32`, `cuda`, and `amp` stay listed,
  and `SUPPORTED_DTYPES` and `SUPPORTED_DEVICES` are untouched.
- **Done when.** Every item in §18 is recorded with observed results, the
  durable guardrails are in place, and `UNSUPPORTED` reads
  `("float32", "cuda", "amp")`.

---

## 20. What G0's guardrails assert

The semantic tests added with this document check the contract, not its
prose. They derive their premises from the live registry, the live
exports, the real file tree, and the C++ sources, and they assert:

1. Phase G exists and is named **Native RNG and Dropout**.
2. `G0` through `G10` each appear exactly once in the milestone ladder,
   in order.
3. No surface claims Dropout, `NativeDropout`, or a native RNG is
   already supported, implemented, shipped, or available.
4. `UNSUPPORTED` is exactly `("dropout", "float32", "cuda", "amp")`.
5. `SUPPORTED_DTYPES == ("float64",)` and `SUPPORTED_DEVICES == ("cpu",)`.
6. The checkpoint format name is unchanged and the format version is
   still **1**, while the design states version **2** as future work.
7. The design forbids global RNG state and requires stateless native
   kernels.
8. One successful stochastic forward consumes exactly one generator call,
   and failures, evaluation, `p == 0`, and backward consume none.
9. Dropout's backward uses a saved private multiplier mask.
10. Version-1 checkpoint compatibility is explicitly defined.
11. Stable/native separation is explicit, and float32, CUDA, and AMP are
    explicit non-goals.
12. No `NativeDropout`/`NativeGenerator` export, no Dropout entry in any
    operation or module inventory, and no RNG/Dropout C ABI symbol,
    ctypes declaration, Core method, or C++ source unit exists yet.
13. No registry value or runtime capability changed in G0.
14. **`"dropout"` remains in `UNSUPPORTED` through G9** — G4 exports the
    module without moving the boundary, every milestone G4–G9 forbids the
    move, and **only G10 removes it**, after the §18 closure matrix, to
    leave exactly `("float32", "cuda", "amp")`.
15. **Reservation operations are lock-protected and token-validated**:
    one lock covering reservation creation, cancellation, commit, counter
    reads, state inspection, state replacement, reseed, and reset;
    native computation outside it; opaque tokens with generator identity
    and a non-reused serial; commit advancing exactly once for the
    active matching reservation only; cancel never advancing; stale,
    foreign, duplicated, and finished tokens rejected without state
    change.
16. **Concurrent or reentrant reservations cannot receive duplicate call
    indices**: a second reservation while one is active fails before an
    index is minted, no two threads ever receive the same successful
    index, exhaustion is checked under the lock, and state replacement is
    refused while a reservation is live — with parallel stochastic
    execution explicitly *not* claimed.
17. **Checkpoint v2 records alias topology, not only canonical state
    entries**: every registered generator path, its canonical target,
    self-mapped canonical names, deterministic canonical selection and
    serialization order, and the shared-versus-independent distinction.
18. **Alias and topology mismatches fail during prevalidation**, before
    any live model, buffer, optimizer, or generator state changes.
19. **A synchronous checkpoint commit failure is fully rolled back**
    across model parameters, persistent buffers, optimizer state, and
    generator state together, with object identities preserved, no
    partially loaded component observable, and pre-existing graph-owned
    masks unchanged.
20. **External asynchronous process termination or interpreter death is
    the only documented exception** to whole-checkpoint atomicity — a
    deliverable `KeyboardInterrupt` is explicitly *not* an exception —
    and the four failure classes (prevalidation, staging, synchronous
    commit, asynchronous death) are distinguished.
