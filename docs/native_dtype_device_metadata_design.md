# Native dtype / device metadata — design

This is a **design document, not an implementation.** It specifies
explicit **dtype** and **device** metadata for the native runtime
(`NativeStorage` / `NativeTensorCore` / `NativeTensor`). No kernels
change and no compute behavior changes in this milestone (v1.20); a
metadata-only implementation is proposed for v1.21.

For where this sits, see [backend_experiments.md](backend_experiments.md)
(the native runtime), [dispatch_design.md](dispatch_design.md) (the
dtype/device risks a future Tensor bridge must answer), and the Phase A
designs it completes:
[native_contiguous_fast_path_design.md](native_contiguous_fast_path_design.md),
[native_broadcasting_design.md](native_broadcasting_design.md),
[native_reductions_design.md](native_reductions_design.md).

## 1. Why dtype/device metadata is Phase A4

Phase A built a native CPU runtime that can lay out, view, broadcast,
combine, and reduce tensors — but every one of those pieces silently
assumes **one** element type (`float64`) on **one** device (CPU). That
assumption is baked in, not stated. A4 makes it *explicit*: it gives the
runtime a place to record "what type are these elements" and "where does
this buffer live", without yet changing what the arithmetic does.

It is deliberately last in Phase A, and deliberately before Phase B
(native autograd), because it is a **contract**, not a feature — and the
things that come next cannot be designed honestly without it:

- **Native autograd (Phase B) needs dtype/device-aware tensors.** A
  gradient must match its parameter's dtype and live on the same device;
  a backward that mixes float64 and float32, or CPU and CUDA, is a bug.
  Autograd has to *assume* a metadata contract, so that contract must
  exist first.
- **A CUDA runtime needs device-aware storage.** "Where does this buffer
  live" is the first question a device abstraction asks; `device` is that
  field. Adding it now (CPU-only, but real) means the CUDA branch extends
  a contract instead of inventing one.
- **An AMP / Tensor Core path needs dtype-aware kernels.** Mixed
  precision is entirely about choosing float16/bfloat16 vs float32 per
  op; there is nothing to choose until tensors carry a dtype.
- **Backend dispatch needs clear metadata contracts.** The dispatch
  design ([dispatch_design.md](dispatch_design.md)) already lists dtype
  and device mismatch as two of the hardest questions any Tensor↔native
  bridge must answer. This milestone answers the native-side half.

So A4 is the seam every later phase attaches to. Getting it explicit,
inspectable, and honest now is worth more than any single new op.

## 2. Current limitation

Today the native runtime is **float64-CPU-only, implicitly**:

- `NativeStorage` allocates a C++ `double[]` buffer — float64, no other
  option, and it never records that fact.
- `NativeStorage.from_array` / `NativeTensorCore.from_array` call
  `np.ascontiguousarray(values, dtype=np.float64)`, coercing any input to
  float64 with no dtype argument and no record of intent.
- Every kernel in `cpp/kernels.cpp` takes `double*`; there is no
  float32/int64/bool code path.
- Nothing has a `device`; the buffer is heap CPU memory, unstated.
- `NativeTensorCore` / `NativeTensor` expose `shape`, `strides`,
  `offset`, `ndim`, `numel`, `contiguous` — but **no `dtype`, no
  `device`.**

The limitation is not that float64/CPU is wrong (it is the right and only
supported case today); it is that the assumption is **invisible**. You
cannot ask a native tensor what type it is, and you cannot be told "no"
when you would want a type the runtime cannot handle.

## 3. Goal

Add **explicit, inspectable dtype and device metadata** to the native
runtime **without changing any compute behavior**:

- Native tensors gain `dtype` and `device` you can read.
- The defaults are exactly today's behavior: `dtype="float64"`,
  `device="cpu"` — so existing code and tests are unchanged.
- The metadata is **explicit**, not silently inferred from NumPy after
  the fact: the runtime records what it was asked for (or its default),
  and validates it, rather than reading a dtype back off an array and
  hoping.
- Unsupported combinations are **rejected clearly** (§7, §11), so no
  tensor ever claims a dtype/device the kernels cannot actually compute.

Non-goal: no new numeric kernels, no float32/int64/bool arithmetic, no
CUDA allocation, no promotion, no casting. Those are later milestones the
*contract* enables — this milestone is the contract.

## 4. Where the metadata lives

**Recommendation: dtype and device are owned by `NativeStorage`**, and
surfaced (read-only) upward through `NativeTensorCore` and `NativeTensor`.

The reasoning follows the existing ownership model:

- **`NativeStorage` owns the buffer.** dtype (the element type of the
  bytes) and device (where those bytes physically live) are properties of
  the *allocation*, not of a logical view of it. When CUDA arrives,
  "device" is precisely "which allocator/memory this storage used" — a
  storage fact. So both tags belong with the storage that owns the
  memory.
- **Views share storage, so they share dtype/device.** A
  `NativeTensorCore` view (`reshape`/`transpose`/`narrow`) borrows the
  same `NativeStorage`; it must report the same dtype and device as its
  owner, automatically, because it *is* the same bytes. Deriving
  `NativeTensorCore.dtype` / `.device` from `self._storage` gives that for
  free and makes it impossible for a view to disagree with its storage.
- **`NativeTensorCore` exposes them** as `dtype` and `device` properties
  (delegating to storage), beside `shape`/`strides`/`contiguous`.
- **`NativeTensor` exposes them** as `dtype` and `device` properties
  (delegating to its core), beside its existing metadata — the wrapper
  stays thin and owns nothing (see
  [native_tensor_wrapper_design.md](native_tensor_wrapper_design.md)).

An alternative — putting dtype/device on `NativeTensorCore` instead of
`NativeStorage` — was considered and is weaker: a core is a *view*, and
two views over one buffer could then disagree about the buffer's type,
which is nonsensical. Keeping the tags on the storage makes them a single
source of truth. (If a future op ever needs a *view* to reinterpret bytes
as a different dtype — a genuine reinterpret-cast — that is an explicit,
separately designed operation, not the default.)

## 5. Metadata model and representation

- **dtype is a canonical lowercase string tag**: `"float64"` now, with
  `"float32"`, `"int64"`, `"bool"`, and later `"float16"` / `"bfloat16"`
  as the reserved future vocabulary (§ future targets). A string (not a
  raw `numpy.dtype`) is chosen deliberately:
  - it is **JSON-friendly**, matching the framework's checkpoint style
    (`.npz` + JSON metadata) so a future native checkpoint can round-trip
    dtype/device trivially;
  - it is a **closed, validated vocabulary** the runtime controls, not an
    open set inherited from NumPy (which has dozens of dtypes the kernels
    will never support);
  - the NumPy correspondence stays **explicit** in one mapping table
    (`"float64" → np.float64`), used only at the conversion boundaries
    (`from_array`, `to_numpy`), never as the source of truth.
- **device is a canonical lowercase string tag**: `"cpu"` now, `"cuda"`
  (and perhaps `"cuda:0"`-style indices) reserved for the future CUDA
  branch.
- **Defaults**: `dtype="float64"`, `device="cpu"`. Chosen so that every
  existing construction path produces exactly today's tensor.
- **Future dtype targets** (vocabulary reserved, not implemented):
  `float32` and `int64` and `bool` first (the common cases), then
  `float16` / `bfloat16` much later, specifically for the AMP / Tensor
  Core path where reduced precision is the point.
- **Future device targets**: `cpu` now; `cuda` later, only when a CUDA
  backend exists.

The tags could later be promoted to small frozen `Dtype` / `Device`
value objects if that proves cleaner, but plain validated strings are the
simplest thing that is honest and serializable, and are the
recommendation for v1.21.

## 6. Proposed constructor / API behavior

Future signatures (names/shape may shift on implementation):

```python
NativeTensorCore.from_array(values, dtype=None, device="cpu")
NativeTensorCore.zeros(shape, dtype="float64", device="cpu")
NativeTensorCore.full(shape, fill_value, dtype="float64", device="cpu")

NativeTensor.from_array(values, dtype=None, device="cpu")
NativeTensor.zeros(shape, dtype="float64", device="cpu")
NativeTensor.full(shape, fill_value, dtype="float64", device="cpu")
```

- **`dtype=None` on `from_array`** means "float64" for v1.21 (the only
  supported dtype). It is `None`-defaulted rather than `"float64"`-
  defaulted so a *later* milestone can define `None` as "infer from the
  array's dtype" without changing the signature — but v1.21 does **not**
  infer; `None` resolves to `"float64"` and the array is still coerced to
  float64, exactly as today. This keeps `from_array` byte-for-byte
  backward compatible.
- **`zeros`/`full` default `dtype="float64"`** explicitly (there is no
  array to infer from).
- **`device="cpu"`** everywhere; any other value is rejected in v1.21.
- **Existing call sites are unchanged**: every current
  `from_array(x)` / `zeros(shape)` / `full(shape, v)` keeps working and
  produces a `float64`/`cpu` tensor, because the new parameters default
  to today's behavior.
- **Validation at construction** (§11): a `dtype` outside the supported
  set, or a `device` outside it, raises immediately with a clear message
  — the tensor is never created in an unsupported state.
- **The explicit backend** (`NativeBackend` / `NumpyBackend`) may thread
  `dtype`/`device` through `tensor_from_array` / `zeros` / `full` if and
  when it fits the common surface; not required for the metadata contract
  itself and can wait.

## 7. Operation validation rules

With only float64/CPU supported, these rules are mostly *guards that
cannot yet fire* — but they are specified now so the behavior is defined
the moment a second dtype exists:

- **Elementwise binary ops (`add`/`subtract`/`multiply`)** require
  **matching dtype and matching device** on both operands. A mismatch
  raises (no implicit promotion, no auto-copy). The result carries the
  operands' shared dtype/device.
- **`matmul`** likewise requires matching dtype and device; the result
  carries them. (matmul itself is unchanged; only the guard is added.)
- **Reductions**: `sum` **preserves the input dtype** (a float64 sum is
  float64; a future int64 sum would be int64). `mean` is float-valued —
  with the current float64-only runtime it **stays float64**; a future
  **integer `mean`** needs its own explicit rule (NumPy promotes integer
  `mean` to float64), which is deferred and called out here, not decided.
- **`relu`** requires a **numeric** dtype (it is `max(x, 0)`); on a future
  `bool` dtype it would be rejected or specified separately. Float64
  today, so no restriction bites yet.
- **`to_numpy`** returns an array whose dtype matches the **stored**
  dtype once non-float64 exists (`"float32"` → a float32 array). Today it
  returns float64, unchanged.
- **Broadcasting and views** carry dtype/device through unchanged: a
  broadcast/transpose/narrow of a float64/cpu tensor is float64/cpu.
- **Error messages name both values** on both operands, e.g.
  `"add requires matching dtype and device, got float64/cpu and
  float32/cpu"`, so a mismatch is never mysterious.

Because v1.21 supports only `float64`/`cpu`, none of these mismatches can
actually be constructed yet (there is no way to make a non-float64
tensor). The rules are the *specification* the guards enforce; their
tests become meaningful when a second dtype lands (§9).

## 8. No-promotion rule

Explicit and load-bearing, mirroring the dispatch design's "no silent
switching":

- **No implicit dtype promotion.** float64 + float32 does not quietly
  become float64; it raises. Promotion, if ever added, is a separate
  designed milestone with its own casting rules.
- **No automatic CPU/GPU copying.** A CPU tensor and a (future) CUDA
  tensor do not silently move to meet; a device mismatch raises. Movement
  is always an explicit call (`to`/`cpu`/`cuda`, §9-future).
- **No silent conversion between float64 and float32** (or any pair).
- **No silent NumPy fallback.** An unsupported dtype/device does not fall
  back to a NumPy computation; it raises. (Same governing rule as every
  prior native milestone.)

## 9. Explicit conversion / casting — future, not this milestone

Reserved as **future design items**, listed so the contract anticipates
them; **none is implemented in v1.20, and none should be implemented in
v1.21's metadata-only pass**:

- **`astype(dtype)`** — an explicit dtype cast, producing a new tensor.
  Needs per-pair casting kernels (float64→float32, etc.), so it waits for
  those kernels.
- **`to(device)`** — an explicit device move, producing a new tensor on
  the target device. Waits for a real second device.
- **`cpu()` / `cuda()`** — device-specific conveniences over `to`. **A
  `cuda()` method should not exist until a CUDA backend exists**; if a
  placeholder is ever wanted, it must raise a clear "CUDA backend not
  available" error *by design* rather than silently no-op or fake a
  device. The recommendation is to **not add these methods at all** until
  the backend they target is real, to avoid a dead/misleading surface.

Casting and movement are where "no silent conversion" (§8) is enforced:
they are the *only* ways dtype or device changes, and they are always
explicit, named, copy-producing calls.

## 10. Relation to the current C++ kernels

- The kernels in `cpp/kernels.cpp` are **float64 CPU kernels**, full
  stop: `double*` buffers, heap CPU memory. There is no non-float64 and
  no non-CPU code path.
- v1.21 must **not pretend** non-float64/non-CPU compute exists. Metadata
  is a label; a label that promises a capability the kernels lack is a
  lie the runtime must not tell.
- Therefore a metadata-only implementation must **not expose unsupported
  dtype/device combinations as working compute**. The safe way to
  guarantee that is §11: reject them at construction, so an unsupported
  tensor never comes into being and no kernel is ever handed one.

## 11. Reject vs. inert metadata — recommendation

Two ways to ship metadata before the kernels catch up:

- **(A) Reject** any `dtype != "float64"` or `device != "cpu"` at
  construction, with a clear error. Only float64/cpu tensors can exist;
  the metadata is real but currently single-valued.
- **(B) Inert/experimental**: allow constructing a tensor *tagged*
  `float32` (say), but keep the bytes float64 and the compute float64 —
  the tag is decorative.

**Recommendation: (A), reject.** Option B manufactures exactly the lie
§10 warns against — a tensor that says `float32` but computes float64,
which will silently corrupt results the moment anyone trusts the tag. (A)
keeps the runtime honest: the `dtype`/`device` fields are real and
enforced, they simply have one legal value each for now, and every future
dtype/device becomes *available* precisely when its kernels/allocators
land (flipping it from "rejected" to "supported" in one place). Honesty
over premature generality — the same principle as "no silent fallback".

## 12. Interaction with existing features

Metadata is additive and touches no compute path:

- **The v1.14 contiguous fast path is unchanged** — same-shape float64
  contiguous ops still take the flat kernel; the dtype guard (all
  float64) always passes.
- **The v1.17 broadcasting implementation is unchanged** — broadcast
  results carry the operands' shared dtype/device.
- **The v1.19 reductions are unchanged** — `sum` preserves dtype, `mean`
  stays float64 (§7).
- **The `NativeTensor` wrapper stays thin** — it gains `dtype`/`device`
  properties that delegate downward; it owns and validates nothing.
- **The explicit backend API** exposes dtype/device consistently *if and
  when* it is threaded through; `backend_info` can advertise the
  supported sets (§13).
- **No `tensorforge.Tensor` integration, no autograd, no CUDA** — all
  remain future phases; this milestone only records metadata for the
  native runtime.

## 13. backend_info / introspection

`backend_info()` (and the native backend's `backend_info`) may report the
supported vocabulary so callers can discover it without trial and error,
e.g. adding `"supported_dtypes": ("float64",)` and
`"supported_devices": ("cpu",)` alongside the existing `"dtype":
"float64"` field. This stays honest — it lists exactly what the kernels
support, and grows only when they do. It is optional for the contract but
recommended, since "what can this backend actually hold?" is a natural
introspection question and the answer is now explicit.

## 14. Proposed tests (for the v1.21 implementation)

Metadata-only tests; no new compute to verify:

- **defaults**: a default `NativeTensor` / `NativeTensorCore` has
  `dtype == "float64"` and `device == "cpu"`.
- **exposure**: `NativeTensorCore` and `NativeTensor` both expose
  `dtype` and `device` (and a view derives them from its storage/core, so
  a transpose/narrow reports the same dtype/device as its owner).
- **back-compat**: `from_array(x)` still produces a float64/cpu tensor
  whose `to_numpy()` is byte-for-byte what it is today; `zeros`/`full`
  expose float64/cpu.
- **explicit args**: `zeros(shape, dtype="float64", device="cpu")` and
  `full(...)` accept the explicit defaults and expose them.
- **operations preserve metadata**: `add`/`multiply`/`matmul`/`sum`/
  `mean`/`relu`/views of a float64/cpu tensor are float64/cpu.
- **rejection**: an unsupported `dtype` (e.g. `"float32"`) and an
  unsupported `device` (e.g. `"cuda"`) each raise a clear error naming
  the offending value and the supported set — the safer-option guard
  (§11).
- **mismatch guards** become real tests once a second dtype exists; until
  then they are specified (§7) but cannot be constructed, so the suite
  documents them as pending rather than asserting an impossible case.
- **`to_numpy` correctness** is unchanged (float64 array out).
- **closed-object behavior**: reading `dtype`/`device` on a closed
  `NativeTensor` follows the existing rule for its metadata (matching how
  `shape`/`strides` behave on a closed tensor today — decide and test
  consistently: either both stay readable or both raise, per the current
  contract).
- **introspection**: `backend_info()` reports the supported dtype/device
  sets if that field is added (§13).

No performance tests; no timing.

## 15. Benchmark impact

**None.** Metadata is a construction-time label and a set of guards that,
for the only supported dtype/device, always pass. There are no new
compute rows to add and no performance claim to make. If a future dtype
introduces a genuinely different kernel (float32 SIMD, say), *that* op
gets benchmarked when it lands — not the metadata. Benchmarks stay
focused on compute changes, not on the metadata contract.

## 16. How this closes Phase A — and what is next

With this design, **Phase A — native CPU runtime — is complete on
paper**. The runtime now has:

- storage and views (v0.7–v1.1),
- the `NativeTensor` wrapper (v1.8–v1.11),
- elementwise ops and matmul (v1.2–v1.3),
- the contiguous fast path (v1.14),
- broadcasting (v1.17),
- reductions (v1.19),
- and a dtype/device metadata contract (this design, v1.20).

Every one of these is designed against the same principles: explicit over
implicit, honest over general, NumPy as the reference, no silent
fallback. The metadata contract is the last piece because it is the seam
Phase B and beyond attach to.

**Recommendation: do one small implementation milestone, v1.21 — native
dtype/device metadata (float64/cpu only), *before* Phase B.** The
reasoning:

- It is **small and low-risk**: read-only `dtype`/`device` properties,
  default-`float64`/`cpu` constructor args, a validated allow-list that
  currently has one entry each, and the reject-on-unsupported guard. No
  kernels, no compute change, fully backward compatible.
- It gives **native autograd (Phase B) a real contract to build on**
  rather than a paper one — a backward can read `grad.dtype ==
  param.dtype` and `grad.device == param.device` against fields that
  actually exist and are enforced.
- It **closes Phase A in code**, not just in docs, matching the project's
  design-then-implement cadence (every prior Phase A design shipped an
  implementation milestone).

So the recommended sequence is **v1.21 (metadata implementation, float64/
cpu only) → v2.0 (Phase B: native autograd design).** Moving straight to
v2.0 is defensible if metadata is judged premature, but shipping the
inert-but-honest contract first is the smaller, safer step and keeps the
"designed → implemented" rhythm intact.

Then Phase B and beyond follow the long arc:
**native autograd** → **native training stack** → **CUDA runtime**
(where `device` grows a second value) → **AMP / Tensor Core** (where
`dtype` grows float16/bfloat16) → **Transformer / text** examples →
**distributed / DDP** → a final **benchmark / profiling / docs** polish
(the final portfolio release). Each lands only when the previous is
tested and documented; the Python framework remains the reference
implementation throughout, and the dtype/device contract designed here is
what makes the device and precision phases expressible at all.
