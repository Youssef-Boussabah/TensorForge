# Native dtype generalization and float32 CPU support — Phase I architecture contract

**Phase I — Native Dtype Generalization and Float32 CPU Support.** This
document is the authoritative architecture contract for the phase. It is
written **before** any dtype-generalization implementation exists, and
milestone **I0** consists of exactly this document, the status
reconciliation it required, and the semantic guardrails that keep the
contract honest. **I0 adds no runtime behavior**: no dtype, no storage
change, no kernel, no C ABI symbol, no ctypes declaration, no
`NativeTensorCore` method, no `NativeTensor` operation, no module, no
optimizer, no export, no registry change, and no checkpoint-format
change.

**Phase-I status: I0, I1, and I2 complete. I3 through I11 are not
started.** The native runtime is **publicly** float64 CPU only today and
stays that way until milestone I9. `SUPPORTED_DTYPES` still reads
`("float64",)`, `UNSUPPORTED` still reads `("float32", "cuda", "amp")`,
and the native checkpoint format is still
`tensorforge.native_checkpoint` version **2** with versions **(1, 2)**
accepted.

What I1 changed, and only this: the library now exports **54** production
`tf_*` symbols — the 52 Phase H closed with, plus the two typed storage
creators of §6.2, which are the only two symbols the whole phase adds.
Storage is dtype-tagged (§4.1) and the C++ dtype model of §3 exists, so
float32 storage is **allocatable through the C ABI**.

What I2 changed, and only this: float32 storage is now **movable**. The
three transfer exports of §7.3 became dtype-general through a source-level
retype of their host positions (no new symbol, no ABI change, still 54),
`tf_core_contiguous_copy` — the runtime's value-transfer primitive — became
dtype-preserving and dtype-strict, and `RAW_KERNEL_DTYPES` records that the
seven handle-free raw kernels stay float64. Internal float32 values can be
copied in, copied out, viewed, materialized through any layout, and copied
storage-to-storage, **bit for bit**. Nothing computes on them: every
arithmetic, reduction, matmul, conv2d, pooling, classification, dropout,
normalization, optimizer, module, and checkpoint path still rejects a
float32 handle with `TF_ERROR_INVALID` before touching memory, and so do
`tf_storage_fill` and `tf_storage_scale`.

Internal allocation and transfer capability is not public support — that
distinction is §27.1's, and the registry moves at I9.

**Phase H remains complete (H0–H10) and is the latest *completed*
phase.** Nothing in Phase I revisits, reverses, or re-measures a Phase-H
result. Phase H made the float64 runtime faster; Phase I makes the
runtime *dtype-general* without giving any of that back.

Related contracts this document builds on rather than replaces:
[native_dtype_device_metadata_design.md](native_dtype_device_metadata_design.md)
(the v1.20/v1.21 metadata model and its no-promotion rule),
[native_abi_error_contract.md](native_abi_error_contract.md),
[native_cpu_performance_design.md](native_cpu_performance_design.md)
(Phase H, including the H1 output-allocation audit and the §4.2 dispatch
shape), [native_autograd_design.md](native_autograd_design.md),
[native_cnn_design.md](native_cnn_design.md),
[native_classification_design.md](native_classification_design.md),
[native_normalization_design.md](native_normalization_design.md), and
[native_rng_dropout_design.md](native_rng_dropout_design.md).

---

## 1. Objective and scope

### 1.1 What Phase I delivers

A native CPU runtime whose tensors are **either float32 or float64**,
with each dtype fully and independently supported through the whole
stack:

- native float32 CPU tensors, beside the existing float64 ones;
- dtype-tagged native storage (no physically `double`-only buffer);
- dtype-aware, **handle-based** operations, with the dtype travelling
  with the data rather than with the call site;
- float32 autograd, including every saved-resource family;
- float32 modules, initialization, and persistent buffers;
- float32 optimizers and float32 optimizer state;
- float32 deterministic RNG and Dropout integration, over the
  **unchanged** generator algorithm;
- dtype-aware native checkpoint **format version 3**;
- exact deterministic float32 checkpoint resume;
- exact preservation of every float64 behavior and of Phase-H float64
  performance;
- cross-platform, sanitizer, and lifecycle validation for both dtypes;
- a final integrated float32 training example.

### 1.2 What Phase I does not deliver

None of the following is in scope, and none may be introduced under the
cover of a Phase-I milestone:

CUDA or any GPU backend · AMP or mixed precision · float16 · bfloat16 ·
integer tensors · boolean tensors · complex tensors · automatic casting ·
explicit casting (`astype`, `.float()`, `.double()`) · dtype promotion ·
mixed-dtype arithmetic · device transfers or `to()`/`cpu()`/`cuda()` ·
`map_location` · stable-backend/native-backend implicit dispatch ·
environment-variable backend or dtype selection · data loaders ·
distributed execution · pybind11 · cffi · BLAS · Eigen · oneDNN · OpenMP
or threading · SIMD intrinsics · a memory pool · a scratch arena ·
C++-managed autograd · a new public stable-framework integration layer.

The §4.3 "deliberately absent" family of `CLAUDE.md` is unchanged by this
phase. Phase I is a **dtype** phase, not an acceleration phase and not a
device phase.

### 1.3 Why this phase, and why now

Every earlier native phase built a capability *at one width*. The
runtime's physical assumption — one `double*` per buffer — is the last
structural thing standing between it and a second dtype, and it is
cheaper to remove now than after another capability is layered on top of
it. Phase H is the reason the order is this way round: an efficiency
phase over a dtype-general runtime would have had to measure and defend
two widths at once, while a dtype phase over an already-optimized runtime
inherits the optimized traversals and only has to prove it did not
disturb them.

float32 in particular is the honest next dtype: it is the width real CPU
training uses when memory bandwidth matters, it is what every subsequent
device or mixed-precision experiment would need first, and it is the one
dtype whose semantics are fully specified by the same IEEE-754 standard
the runtime already obeys — so nothing about correctness has to be
invented, only generalized.

---

## 2. Repository reality at I0 — the verified baseline

Everything in this section was read out of the tree at
`1b6cc17305c7ffc6502e27c32b45661480e05f9d`, not inferred from names.
It is the *starting* state; every decision from §3 onward is a *future*
Phase-I decision unless it says otherwise.

### 2.1 How native storage is physically allocated today

`cpp/src/storage.cpp` holds one shared creation body,
`create_storage(int64_t size, bool zero_initialize)`, used by both
exported constructors. It rejects `size <= 0`, consults the test-only
allocation fault-injection hook, and then allocates

```cpp
std::unique_ptr<double[]> data(
    zero_initialize ? new (std::nothrow) double[count]()
                    : new (std::nothrow) double[count]);
```

followed by a separately allocated `Storage` node that adopts the buffer
(`data.release()`), so a failed metadata allocation frees the buffer
rather than leaking it. Destruction is `delete[] storage->data; delete
storage;`.

### 2.2 Whether storage uses `double*`, `void*`, bytes, or element counts

`cpp/include/tf_internal.h`:

```cpp
struct Storage {
    double* data;
    int64_t size;
};
```

The buffer is **physically `double`**, the size is a **logical element
count**, and there is no byte arithmetic anywhere in the storage layer —
`new double[count]` derives the byte size implicitly. There is no dtype
tag, no capacity field, and no alignment field. `as_storage(handle)` is a
plain `static_cast` from `void*`; every compute translation unit reads
and writes `.data` as `double*`.

### 2.3 Where float64 is hardcoded at the Python/native boundary

`src/tensorforge/backends/cpp.py` is the only module in the repository
that imports `ctypes`, and it hardcodes float64 in five distinguishable
places:

1. `_CHECKED_F64_ARRAY = np.ctypeslib.ndpointer(dtype=np.float64,
   flags="C_CONTIGUOUS")`, the single checked binding used for every
   caller-facing data buffer position;
2. `SUPPORTED_DTYPES = ("float64",)` and `normalize_dtype`, which returns
   `"float64"` for `None` and rejects everything outside the tuple;
3. `NativeStorage.copy_from` / `from_array`, which call
   `np.ascontiguousarray(values, dtype=np.float64).ravel()`;
4. `NativeStorage.to_numpy` and `NativeTensorView.to_numpy`, which
   allocate `np.empty(..., dtype=np.float64)` destinations;
5. five explicit gates — `if self.dtype != "float64" or self.device !=
   "cpu": raise ValueError(...)` — on `cross_entropy_forward`,
   `cross_entropy_backward`, `maxpool2d_forward`, `maxpool2d_backward`,
   and `dropout_forward`.

`backend_info()` additionally reports `"dtype": "float64"` and
`"device": "cpu"` as flat strings beside the two registries.

### 2.4 How NumPy arrays enter and leave native storage

Ingress is `NativeStorage.copy_from` → `tf_storage_copy_from(handle,
const double* src)`, an element loop over `storage->size`. The Python
side converts *whatever it was given* to contiguous float64 and flattens
it, so a Python list or an int64 array is silently converted at this
boundary — this is the **explicit host-to-native conversion boundary**,
not a tensor cast, and it has always behaved this way.

Egress is two paths: `tf_storage_copy_to(handle, double* dst)` for the
flat storage, and `tf_storage_materialize(handle, double* dst, shape,
strides, offset, ndim)` for a strided view, which walks an odometer and
writes a row-major destination. Both write into a NumPy buffer the Python
side allocated.

Native-to-native transfer never touches a host buffer: `contiguous_copy`
uses `tf_core_contiguous_copy`, storage to storage.

### 2.5 How dtype is represented today

Only as a **Python string tag**, `"float64"`, validated by
`normalize_dtype` and stored on `NativeStorage._dtype`.
`NativeTensorCore.dtype` and `NativeTensor.dtype` both *delegate* to that
storage tag; `NativeTensorView` has no dtype of its own. There is **no
dtype anywhere in C++** and none crosses the C ABI. The tag is
descriptive metadata that today can only ever hold one value, and
`docs/native_dtype_device_metadata_design.md` (v1.20/v1.21) is the
contract that put it there, deliberately, so that a second dtype would
have a place to be recorded.

### 2.6 How shapes, strides, offsets, and spans are measured

In **logical elements**, everywhere, without exception:

- `NativeTensorView` holds `shape`, `strides`, `offset`, `numel`, and a
  contiguity flag, all element-valued; `_bind` bounds-checks the whole
  reachable offset range `[low, high]` against `storage.size - 1`,
  negative strides included;
- the strided C ABI takes `const int64_t* shape`, `const int64_t*
  strides`, `int64_t offset`, `int64_t ndim`, all in elements;
- `tf_storage_size` returns an element count;
- the H3 per-view `int64` layout arrays and the H7 typed pointers into
  them describe elements;
- the conv2d, pooling, classification, and dropout exports take element
  offsets and element counts.

Nothing in the runtime measures a byte today.

### 2.7 How each state family represents numeric data

| State | Representation today |
|---|---|
| Tensors / parameters / gradients | `NativeTensor` → `NativeTensorCore` → `NativeStorage` (a float64 buffer) |
| Persistent buffers (BatchNorm running mean/var) | registered `NativeTensor`s, same representation |
| BatchNorm eval snapshots | independent owning `NativeTensor` copies, graph-owned |
| Optimizer moments (`m`, `v`) | `NativeTensor`s, one per parameter |
| Optimizer step counters | **Python ints**, not tensors |
| Optimizer hyperparameters (`lr`, `betas`, `eps`) | **Python floats**, materialized per step as rank-0 `NativeTensor`s at the parameter's dtype |
| Dropout multiplier mask | a graph-owned `NativeTensorCore` |
| MaxPool2d winners | a graph-owned `NativeTensorCore` holding flat plane offsets **encoded as float64 values** |
| Cross-entropy saved probabilities | a graph-owned `NativeTensorCore` |
| Cross-entropy targets | a host `int64` NumPy array copied independently — **metadata, never a tensor** |
| Generator state | `(algorithm, algorithm_version, seed, calls)`, `uint64`s as Python ints |
| Checkpoint payloads | one float64 NumPy array per model/buffer/moment entry in an `npz`, plus a JSON manifest |

### 2.8 Exported C ABI functions that receive raw typed buffers

Ten of the 52 take a host `double*` (or `const double*`):

- **Handle-free float64 utilities (7):** `tf_elementwise_add`,
  `tf_elementwise_subtract`, `tf_elementwise_multiply`,
  `tf_elementwise_divide`, `tf_relu`, `tf_matmul`, `tf_matmul_tiled`.
  These take only raw buffers and an element count — no storage handle at
  all — and are the reference/benchmark set `RAW_KERNELS` advertises.
- **Handle-plus-host-buffer transfer (3):** `tf_storage_copy_from`,
  `tf_storage_copy_to`, `tf_storage_materialize`.

Two more take a raw `const int64_t* targets` host array beside their
handles: `tf_core_cross_entropy_forward` and
`tf_core_cross_entropy_backward`. Every strided export additionally takes
`const int64_t*` **layout metadata** (shape/strides/write-strides), which
is metadata rather than data.

Three take a scalar `double` **by value**: `tf_storage_fill`,
`tf_storage_scale`, and `tf_core_dropout_forward`'s probability.

### 2.9 Exported C ABI functions that operate only through opaque handles

The remaining 42, of which 33 are `tf_core_*` compute exports (unary,
binary, contiguous fast paths, matmul, sum, narrow-backward, the three
conv2d directions, both pooling directions, softmax, log-softmax, both
cross-entropy directions, dropout-forward), 6 are storage lifecycle
(`tf_storage_create`, `tf_storage_create_uninitialized`,
`tf_storage_destroy`, `tf_storage_size`, `tf_storage_fill`,
`tf_storage_scale`), and 5 are error/introspection (`tf_last_error_code`,
`tf_last_error_message`, `tf_clear_error`, `tf_test_arm_alloc_failure`,
`tf_fault_injection_available`).

**This is the load-bearing fact of the whole phase**: the overwhelming
majority of the compute surface already addresses its operands through
opaque handles, so a dtype tag placed on the storage behind those handles
reaches every one of them without a signature change.

### 2.10 Internal C++ kernels that assume `double`

All of them, in every translation unit:

- `elementwise.cpp` / `tf_elementwise_internal.h` — the H8 collapsed-plan
  traversals are already `template <class Op>` with `static inline double
  apply(...)` operation structs (`AddOp`, `SubtractOp`, `MultiplyOp`,
  `ReluOp`, `ReluBackwardOp`, `SqrtOp`, `ReciprocalOp`, `IdentityOp`),
  plus the retained function-pointer odometers taking `double (*)(double,
  double)`. `exp` and `log` are deliberately excluded from the templated
  traversals and keep the retained paths.
- `matmul.cpp` / `tf_matmul_internal.h` — `matmul_generic_strided` and
  the H2 `matmul_row_sweep`, plus the hidden
  `tf::matmul_prefers_row_sweep` predicate.
- `reduction.cpp` / `tf_reduction_internal.h` — `sum_generic_strided`,
  the H6 `sum_contiguous_blocks`, and
  `tf::reduce_prefers_contiguous_blocks`.
- `conv2d.cpp` / `tf_conv2d_internal.h` — three directions, each with a
  retained generic loop, an H9 optimized traversal, and a hidden
  predicate.
- `pooling.cpp` — forward with its winner buffer, and the backward
  scatter that validates every winner value.
- `classification.cpp` — softmax, log-softmax, and the fused
  cross-entropy pair, all with the maximum shift and log-sum-exp.
- `random.cpp` / `tf_random_internal.h` — `dropout_uniform` produces a
  `double` on `[0, 1)` as `(bits >> 11) * 2**-53`, and
  `dropout_forward_contiguous` writes `double` output and mask.
- `storage.cpp` — as §2.1.

There is **one** `Storage` struct and **one** item-size assumption
(implicit in `new double[]`), so there is no scattered size table to
consolidate — which is exactly what makes a single central dtype
authority achievable rather than aspirational.

### 2.11 Python constructors and factories that assume `np.float64`

`NativeStorage.from_array` / `copy_from` / `to_numpy`;
`NativeTensorView.from_array` / `to_numpy`; `NativeTensorCore.from_array`
(via storage) and `to_numpy`; `NativeTensor.from_array` / `to_numpy`;
`NativeParameter` construction (through `NativeTensor`);
`NativeLinear.__init__` and `NativeConv2d.__init__`, which draw their
initial values with `numpy.random.Generator.uniform(...)` into a float64
host array; the checkpoint save path (`snapshot.to_numpy()` → float64
archive arrays) and load path (`NativeTensor.from_array(arrays[...])`);
and `native_accuracy`, which materializes logits through `to_numpy()` and
takes a NumPy argmax.

None of these has a dtype argument that reaches NumPy: `dtype` is
accepted and validated as metadata, and the NumPy conversion is
unconditionally `np.float64`.

### 2.12 Tests and registries that lock float64-only support

- `cpp.SUPPORTED_DTYPES == ("float64",)` and `cpp.UNSUPPORTED ==
  ("float32", "cuda", "amp")` — asserted in
  `tests/test_native_phase_h_closure.py`
  (`test_the_support_boundary_is_exactly_what_phase_h_inherited`),
  `tests/test_docs.py`
  (`test_no_future_phase_is_claimed_by_phase_e_closure`,
  `test_dropout_float32_cuda_and_amp_stay_unsupported_through_phase_f`),
  and the backend-info guardrails.
- Behavioral, not just registry:
  `test_the_unsupported_capabilities_really_are_unreachable` constructs
  `NativeTensorCore.from_array(..., dtype="float32")` and requires it to
  raise.
- `docs/native_support_matrix.md` lists float32 under **Unsupported or
  future**, and `tests/test_docs.py`
  (`test_native_support_matrix_is_canonical_and_honest`) asserts that
  `"float32"` appears **after** the `## Unsupported or future` heading and
  **not before** it.

### 2.13 Tests that lock the production export count at 52

Three, independently:

- `tests/test_native_phase_h_closure.py` — `FINAL_EXPORT_COUNT = 52`,
  checked against a regex inventory of `TF_EXPORT` in `cpp/src/*.cpp`,
  plus a check that every declared ctypes symbol resolves in the built
  library;
- `tests/test_native_abi_boundary.py` — `assert len(source_exports) ==
  52`, and a partition assertion `with_arrays + handle_only + test_only ==
  52`;
- `tests/test_native_storage_allocation.py` — `EXPECTED_TF_EXPORTS = 52`,
  checked against the **built library's** export table.

### 2.14 Tests that lock checkpoint version 2 and accepted versions (1, 2)

`native_checkpoint._FORMAT_VERSION == 2` and
`_SUPPORTED_FORMAT_VERSIONS == (1, 2)`, asserted in
`tests/test_native_phase_h_closure.py`
(`test_the_checkpoint_format_did_not_move`) and exercised throughout
`tests/test_native_checkpoint.py` and
`tests/test_native_checkpoint_v2.py`, including a corruption matrix that
rejects unknown versions.

### 2.15 Exact-resume tests that must eventually work per dtype

`tests/test_native_phase_c.py`, `test_native_phase_d.py`,
`test_native_phase_e.py`, `test_native_phase_f.py`,
`test_native_phase_g.py`, and `test_native_phase_g_closure.py`, together
with the example-backed proofs in
`examples/native_checkpoint_resume.py`,
`examples/native_cnn_training.py`,
`examples/native_classification_training.py`,
`examples/native_normalization_training.py`, and
`examples/native_dropout_training.py`. Each proves that an interrupted
run reloaded into a **fresh** model/optimizer/generator set reproduces the
loss suffix, every parameter, every buffer, every moment, every step
counter, the generator state, and the final outputs by exact equality.
Phase I must leave every one of them passing unchanged at float64 and add
an independent float32 counterpart.

### 2.16 Metadata that is integer-valued without being an integer tensor

Recorded so no Phase-I milestone mistakes one for a tensor-dtype
requirement:

- shapes, strides, offsets, `ndim`, element counts, and the H3/H7 layout
  arrays;
- conv2d/pooling geometry (kernel, stride, padding, output extents);
- cross-entropy class **targets** (a host `int64` array, independently
  copied, revalidated in C++);
- the softmax/log-softmax `(outer, axis_length, inner)` decomposition and
  the reduction code (`0 = mean`, `1 = sum`);
- optimizer **step counters**;
- generator `seed` and `calls` (`uint64`, carried as canonical decimal
  strings in the checkpoint manifest because a `uint64` above `2**53`
  cannot survive a JSON double);
- MaxPool2d **winner indices** — flat plane offsets *encoded as float64
  values*, exact while `H*W <= 2**53` (`_MAX_EXACT_WINNER_PLANE`), and
  revalidated element by element in the backward.

**None of these is an integer tensor, and Phase I adds no integer tensor
dtype.**

### 2.17 Phase-H optimizations sensitive to element size, alignment, pointer type, or dispatch

Every one of them, and this is the list a Phase-I milestone must check
itself against:

| Phase-H result | Sensitivity |
|---|---|
| H1 uninitialized output allocation | The per-kernel audit ("this kernel writes every destination element") is a property of the *traversal*, not the element type, so it carries over — but the poison pattern must exist in both widths, and the zero-initialized default must still be all-zero **bytes**. |
| H2 matmul `i`-`k`-`j` row sweep over four destination rows | Written in terms of elements, not bytes; a narrower element changes how many rows fit in cache but not the correctness or the accumulation order. The predicate reads layout metadata only. |
| H3 metadata normalization and per-view layout arrays | Dtype-independent — the layout arrays are `int64` counts. |
| H4 optimizer step (per-step scalar holder, exact Python reciprocal) | The exact-substitution proof is stated over IEEE-754 **binary64** values; a float32 parameter needs its own statement (§15.3). |
| H5 `contiguous_copy` as the value-transfer primitive, with `tf::copy_prefers_contiguous` | The identity map performs no arithmetic, which is *why* it is bit-preserving; that argument is width-independent and must be restated per dtype rather than inherited. |
| H6 `sum_contiguous_blocks` with per-output order preserved | Order is a traversal property; the accumulator width follows the element width by §10. |
| H7 trusted `_LAYOUT_POINTER` bindings and memoized pointer conversion | Layout only — untouched. The **checked** `ndpointer` bindings are where dtype enters (§7.3). |
| H8 templated `template <class Op>` traversals and collapsed plans | The direct extension point: the template gains a scalar parameter (§8.2). The plan structs are `int64` metadata and are unchanged. |
| H9 conv2d row-sweep/gather traversals and their predicates | Element-based; predicates read geometry only. |

No Phase-H optimization depends on a pointer value, an alignment, or a
CPU-feature probe, because §4.2 forbids that — which is precisely why
none of them breaks when the element type changes.

### 2.18 Whether stable-backend code could be affected

**No, and a test proves it.** `tensorforge`, `tensorforge.nn`,
`tensorforge.optim`, `tensorforge.data`, and `tensorforge.serialization`
never import `tensorforge.backends.cpp`;
`test_importing_the_stable_framework_does_not_load_the_native_backend`
runs a subprocess and asserts no `backends.cpp` module is loaded. The
stable line's own dtype behavior is NumPy's and is not part of this
phase.

The one adjacency worth naming: `tensorforge.backends.native_backend`
and `tensorforge.backends.registry` expose the explicit
`get_backend("native")` object, which forwards `dtype`/`device` to the
`NativeTensorCore` factories. That object is opt-in, is never selected
implicitly, and will inherit float32 through the same explicit argument —
it introduces no coupling and gains no dispatch.

### 2.19 Baseline measurements this contract is written against

- Full Python suite: **6415 passed, 0 failed, 0 skipped** (native library
  built).
- Native CTest inventory: **17** dependency-free C++ tests.
- Production `tf_*` exports: **52** in source, matching the built library.
- `CHECKPOINT_VERSION == 2`, `SUPPORTED_CHECKPOINT_VERSIONS == (1, 2)`.

---

## 3. The internal dtype model

### 3.1 One authority per side of the boundary, and no third

Phase I introduces exactly **two** dtype authorities — one in C++, one in
Python — and they agree by construction because the ABI codes are the
same integers.

**C++** (`cpp/include/tf_internal.h`, beside `TfStatus`):

```c
enum TfDtype {
    TF_DTYPE_FLOAT64 = 0,
    TF_DTYPE_FLOAT32 = 1,
};
```

with, in `namespace tf`:

```cpp
enum class Dtype : std::int32_t { Float64 = 0, Float32 = 1 };

bool dtype_from_code(std::int32_t code, Dtype& out) noexcept;  // total
std::size_t dtype_item_size(Dtype dtype) noexcept;             // 8 or 4
const char* dtype_name(Dtype dtype) noexcept;                  // for messages
```

**Python** (`src/tensorforge/backends/cpp.py`, beside `normalize_dtype`):
one code table, one item-size table, one NumPy-type table, and nothing
else anywhere in the repository.

### 3.2 Required properties

- **Exactly two values**, float32 and float64, and no way to construct a
  third.
- **Stable numeric codes** suitable for the C ABI. `0` and `1` are frozen
  in the same sense the `TfStatus` codes are: their meaning never
  changes, and a hypothetical future dtype (not in Phase I) takes `2`.
  float64 is `0` so that the compatibility wrappers of §6.4 pass a code
  they could equally have defaulted to.
- **Validated conversion** from an ABI code to an internal value.
  `dtype_from_code` is total, `noexcept`, allocation-free, and returns
  `false` for every unknown code — never a default, never a clamp, never
  an assertion.
- **Canonical item size** from `dtype_item_size`, the single place a
  width is written down in C++. No kernel, no export, no test helper, and
  no build file may spell `sizeof(double)` or `8` as a storage width
  again.
- **Canonical public name** from `dtype_name` / the Python table:
  `"float64"` and `"float32"`, matching `SUPPORTED_DTYPES` exactly.
- **Rejection of unknown codes** at the ABI boundary with
  `TF_ERROR_INVALID` → Python `ValueError`, naming the offending code and
  the supported set.

### 3.3 What is deliberately *not* built

- No public Python dtype **object** — no `tensorforge.float32`, no
  `NativeDtype` class, no `np.dtype` acceptance, no `"f4"` aliases. The
  public form is the canonical string, exactly as it has been since
  v1.21 (§25).
- No dtype **registry** that a user can extend.
- No dtype **inference** from an input array (§9.4).
- No item-size table in any kernel, wrapper, test helper, or benchmark.
  A test that needs a width imports the one authority.

### 3.4 Rejected alternatives

| Alternative | Why rejected |
|---|---|
| A string dtype carried in C++ | A string in a hot dispatch is exactly what §22 forbids, and string comparison at an ABI boundary invites locale, encoding, and lifetime problems for no benefit. |
| `sizeof(T)` at each use site | Reintroduces the scattered size table the contract exists to prevent; a single wrong site is a silent out-of-bounds rather than a compile error. |
| Reusing NumPy's type-number codes | Couples the C ABI to another project's numbering, which the repository has no reason to inherit and cannot control. |
| A dtype *enum class* crossing the ABI directly | The ABI is plain C; an `int32_t` code with a validated conversion is the form every other ABI value (status, reduction code) already takes. |

---

## 4. Dtype-tagged storage

### 4.1 The representation

```cpp
struct Storage {
    void*   data;   // untyped; the dtype tag says how to read it
    int64_t size;   // LOGICAL ELEMENT COUNT — meaning unchanged
    Dtype   dtype;  // the authority for this buffer's element type
};
```

Requirements:

- **Untyped data pointer.** No `double*` member, no union of typed
  pointers, no second pointer. A union would let the tag and the pointer
  disagree; a `void*` cannot.
- **Logical element count**, unchanged in meaning and unchanged in name,
  so `tf_storage_size` keeps returning exactly what it returns today and
  no caller anywhere reinterprets it.
- **Physical byte size** is *derivable* as `size * dtype_item_size(dtype)`
  and is **not stored**. Storing it would create a second source of truth
  that can disagree with the first; the derivation is one multiplication
  of two values the struct already holds. (The allocation path computes
  it once, checked, before allocating; the destruction path recomputes it
  only if the deallocation form needs it, which the form chosen below
  does not.)
- **Checked `numel × itemsize` arithmetic, with the overflow proved
  rather than assumed.** Before allocating, the creation body proves
  `size <= INT64_MAX / item_size` and, on a platform where `size_t` is
  narrower, `size * item_size <= SIZE_MAX`. An overflow is
  `TF_ERROR_INVALID` with a message naming the count and the dtype —
  never an allocation attempt, never a silent wrap, never UB. This is a
  *new* failure mode: today the implicit `new double[count]` sizing has
  no checked equivalent, so I1 adds the check rather than inheriting one.
- **Correct dtype-aware construction and destruction, with C++17 object
  lifetimes explicitly begun.** This is the part it is easiest to get
  subtly wrong, so it is specified rather than left to the
  implementation.

  The kernels do not merely dereference their operands, they **index**
  them: `data[i]` and `data + i` across the whole allocation. In C++17
  pointer arithmetic is defined only *within a single array object*
  ([expr.add]/4), and an object that is not an array element behaves as a
  one-element array. Two plausible models therefore fail:

  - **`unsigned char[n]` plus a reinterpreting cast.** This begins the
    lifetime of an array of `unsigned char` and of no `float` or `double`
    at all, so every typed access is to an object that never existed.
    C++20 added implicit object creation for exactly this pattern
    ([intro.object]/10, P0593); C++17 did not, and the project may not
    rely on a rule it does not build under.
  - **Raw storage plus a per-element placement-new loop.** This does begin
    `count` floating-point lifetimes — but as `count` *separate scalar
    objects*. Adjacent scalars do not become elements of one array merely
    because their storage is contiguous, so `data[i]` past the first still
    walks outside its array object.

  The model that actually supports the indexing is the ordinary one: an
  **array new-expression**, which creates a genuine `float[count]` or
  `double[count]` object. The element type is a runtime property, so the
  choice is one dtype dispatch into a templated body, and the array
  pointer is type-erased into `void*` only **after** the array exists:

  ```cpp
  template <class T>
  Storage* create_typed_storage(int64_t size, Dtype dtype,
                                bool zero_initialize) {
      static_assert(std::is_trivially_destructible<T>::value, "...");
      const std::size_t count = static_cast<std::size_t>(size);
      std::unique_ptr<T[]> data(                      // type-correct RAII
          zero_initialize ? new (std::nothrow) T[count]()
                          : new (std::nothrow) T[count]);
      if (!data) { /* TF_ERROR_ALLOC */ return nullptr; }
      Storage* storage = new (std::nothrow) Storage{data.get(), size, dtype};
      if (storage == nullptr) { /* TF_ERROR_ALLOC */ return nullptr; }
      data.release();
      return storage;
  }

  // destruction: one central, dtype-matched switch
  void destroy_storage_data(Storage* storage) noexcept {
      switch (storage->dtype) {
          case Dtype::Float32: delete[] static_cast<float*>(storage->data);
                               return;
          case Dtype::Float64: delete[] static_cast<double*>(storage->data);
                               return;
      }
  }
  ```

  Properties this form has, each load-bearing:

  - **The kernels' pointer arithmetic becomes valid.** One array object
    spans the whole allocation, so `data[i]` is defined for every
    `i` in `[0, size)` — which is the entire reason for the design.
  - **The dtype tag is the sole authority in both directions.** It
    selects the typed accessor that recovers `T*` and it selects the
    matching `delete[]`. The allocation form and the deallocation form
    cannot disagree, because the same immutable field chooses both.
  - **Destruction is centralized.** One `switch`, in one function, with
    no `default:` label — so a future dtype without a deleter is a
    compile-time warning, and a tag holding neither value would fall
    through *without* deleting (a leak a sanitizer reports) rather than
    running a wrong `delete[]` (undefined behavior). Declining to guess
    is the safer failure.
  - **Zero-initialization is a value-initialized array.** `new T[n]()`
    zero-initializes every element, and IEEE-754 zero is **positive**
    zero at both widths — so H1's zero-initialized default is preserved
    exactly, for both dtypes, with no per-dtype fill pass. Proved by
    value *and* by sign bit.
  - **Uninitialized really is uninitialized.** `new T[n]` default-
    initializes a scalar array, which writes nothing, so H1's saving is
    intact.
  - **Metadata failure is covered by `std::unique_ptr<T[]>`.** It releases
    through `delete[]` on the exact `T*` that was allocated, so the
    "array allocated, metadata allocation fails, array leaks" scenario
    cannot happen and cannot free wrongly.
  - **Alignment** comes from the array new-expression itself, which
    allocates suitably for its own element type. Checked at runtime for
    every creator, dtype, and size.
  - **No per-element destruction, licensed rather than assumed.** `float`
    and `double` are trivially destructible, `static_assert`ed beside the
    allocation, so `delete[]` runs no destructor pass. A dtype needing
    destruction could not be added without that assertion firing.
  - **Validation is written once.** Size, checked `numel × itemsize`, and
    fault injection live in the single non-templated caller, so the two
    instantiations cannot drift apart on any of it. The byte count is a
    *validation* rather than a sizing input — the array new-expression
    computes its own size.
- **Allocation-failure atomicity**, unchanged: the `unique_ptr` owns the
  buffer until the `Storage` node has successfully adopted it, so the
  "data allocated, then metadata allocation fails, buffer leaks" scenario
  remains impossible. A failed creation sets the thread-local error and
  returns `nullptr`, having published nothing.
- **No mismatched allocation/deallocation type**, proved by having
  exactly one allocation form and exactly one deallocation form.

### 4.2 Zero-sized storage

`size <= 0` is **rejected today** (`"storage size must be positive"`) and
stays rejected. Phase I introduces no zero-element storage and no
zero-element tensor: the native shape rules require positive dimensions,
so no empty core is constructible. The typed creators validate `size > 0`
with the identical message so a caller cannot tell the two constructors
apart by their errors. A milestone that wanted zero-sized storage would
be a capability change and is not in this phase.

### 4.3 Elements versus bytes — where the line is

**Shapes, strides, tensor offsets, view arithmetic, element counts,
bounds checks, and every ABI layout argument stay in logical elements.**
Byte conversion happens in exactly two places:

1. the storage **allocation** boundary (`size × itemsize`, checked); and
2. the raw-memory **addressing** boundary inside a kernel, which is not a
   conversion at all — the kernel casts `void*` to `T*` once at the top
   and then indexes in elements, exactly as it does today.

No kernel computes a byte offset. No Python code computes a byte size.
`memcpy` is not introduced. This rule is what makes the whole existing
layout, bounds-checking, and dispatch apparatus carry over unexamined.

### 4.4 Rejected alternatives

| Alternative | Why rejected |
|---|---|
| Two typed pointers (`double* d; float* f;`) | The tag can disagree with which pointer is valid; doubles the null-checking surface for no benefit. |
| A `std::variant` / tagged union of buffers | Same disagreement risk, plus a C++ abstraction in a struct that crosses an opaque C boundary. |
| A separate `Float32Storage` handle type | Two handle types means every export needs two overloads or a runtime type test — precisely the per-operation duplication §6.5 rejects. |
| Storing byte capacity in the struct | A second source of truth for a value that is one multiplication away, and a field a corrupted or mismatched writer could make inconsistent. |
| `std::aligned_alloc` / over-aligned buffers | Would be an alignment-dependent optimization, which §4.2 of `CLAUDE.md` forbids dispatching on, and buys nothing without SIMD — which is rejected. |

---

## 5. Tensor dtype ownership

### 5.1 The single authority

**A tensor's dtype is its storage's dtype.** There is exactly one dtype
field per buffer, in `tf::Storage` on the C++ side and on
`NativeStorage._dtype` on the Python side, and every layer above reads it
rather than copying it:

- `NativeTensorView` has **no** dtype field and must never gain one.
- `NativeTensorCore.dtype` delegates to `self._storage.dtype`.
- `NativeTensor.dtype` delegates to its core.
- `NativeParameter.dtype` delegates to its tensor.

This is already true today and Phase I's job is to *keep* it true while
the field becomes able to hold two values. Independently mutable tensor
and storage tags that could disagree are forbidden; so is caching the
dtype on a view or a core.

### 5.2 Consequences that follow for free

- **All views of one storage have the same dtype**, because there is one
  field and views borrow the storage that holds it.
- **A view operation never casts or reinterprets.** `reshape`,
  `transpose`, `T`, and `narrow` produce new views over the same storage
  with new element-valued layout metadata; none of them touches the
  buffer, the dtype, or a byte.
- **The dtype is immutable for a storage's lifetime.** It is assigned
  once at creation, before the handle is published, and no export, method,
  or loader may change it. Changing a tensor's dtype in place would be a
  cast, which does not exist.
- **The dtype is readable after `close()`**, exactly as `shape` and the
  existing dtype tag are, because it is Python-side metadata that
  outlives the handle.

### 5.3 What a caller may rely on

If two `NativeTensor`s were produced by any operation from operands of
dtype `D`, they have dtype `D`. If an operation's operands do not all
have the same dtype, it raises before allocating anything (§9). If a
tensor reports dtype `D`, `to_numpy()` returns a NumPy array whose dtype
is exactly `D` — no widening on the way out, ever.

---

## 6. C ABI strategy

### 6.1 The principle

**The dtype travels with the data, not with the call.** A handle already
identifies a storage; a storage will carry its dtype; therefore every
handle-based export already has everything it needs. The ABI grows only
where a *new* storage is **constructed**, because that is the one moment
at which the dtype is not yet knowable from any argument.

### 6.2 The two new exports

```c
void* tf_storage_create_typed(int64_t size, int32_t dtype_code);
void* tf_storage_create_uninitialized_typed(int64_t size, int32_t dtype_code);
```

**Semantics.** Allocate `size` elements of the dtype named by
`dtype_code` and return an opaque `Storage` handle, or `nullptr` with the
calling thread's error slot set. The first zero-initializes; the second
leaves the contents indeterminate and is legal only for a destination
whose kernel provably writes every element (the H1 audit table, extended
per dtype in I1).

**Guard and error contract.** Both are `TF_GUARD_BEGIN` /
`TF_GUARD_END(nullptr)` and both join `_CHECKED_KERNELS`, so they clear
the thread-local error on entry and report failure exactly the way
`tf_storage_create` reports it today — a null handle plus a status code —
and take the identical `errcheck` hook. No second failure convention is
introduced.

**Validation order, which is part of the contract:**

1. `dtype_code` converts to a known `Dtype` — else `TF_ERROR_INVALID`,
   `"unknown dtype code <n>; supported codes are 0 (float64) and 1
   (float32)"`.
2. `size > 0` — else `TF_ERROR_INVALID`, `"storage size must be
   positive"`, **byte-for-byte the message the existing creators
   produce**.
3. `size × item_size` is representable — else `TF_ERROR_INVALID` naming
   the count and dtype.
4. the test-only allocation fault-injection hook.
5. the allocation itself — failure is `TF_ERROR_ALLOC`, `"could not
   allocate native storage"`.

The dtype check comes first because the item size the overflow check
needs comes from it. For the compatibility wrappers of §6.4 the dtype is
known by construction, so **their** first observable failure is still
`size <= 0` with the identical message — existing behavior is preserved
to the letter.

**Output-handle behavior.** Success returns a fully constructed,
independently owned handle destroyed exactly once by the unchanged
`tf_storage_destroy`. Failure returns `nullptr` and publishes nothing —
no partially built node, no orphaned buffer, no mutated global.

**Zero-size behavior.** Rejected, as §4.2.

**Ownership.** Identical to the existing creators: the caller owns the
handle; the Python wrapper's `close()` is the contract and `__del__` is
only a fallback.

**Compatibility.** They add capability; they change nothing. They are the
**only** two production symbols Phase I adds, taking the library from 52
to **54**.

### 6.3 Where the code comes from on the Python side

`NativeStorage.__init__` already validates and normalizes `dtype` before
allocating. It gains one lookup — the canonical string to its ABI code —
and passes the code to the typed creator. There is no branching on dtype
anywhere else in the constructor, and no other Python object learns an
ABI code.

### 6.4 Existing float64 creators stay, unchanged

`tf_storage_create(size)` and `tf_storage_create_uninitialized(size)` are
**not removed, not renamed, not deprecated, and not behaviorally
altered.** They become thin compatibility wrappers that delegate to the
shared creation body with `Dtype::Float64`, which is what they already
effectively do. Every existing caller — including the C++ CTest
executables, which construct real `tf::Storage` handles — keeps working
without an edit.

Whether the Python wrapper continues to *call* them is a separate,
smaller question, answered at I1: it will call the typed creators
uniformly, because a single path is easier to prove correct than two, and
the untyped exports remain for ABI compatibility and are exercised by
their own tests. **The exports do not go away merely because the primary
caller moves.**

### 6.5 Why two creators are sufficient, and why per-operation float32 exports are rejected

Sufficiency follows from §2.9: 42 of 52 exports are handle-only, and of
the ten that touch a raw buffer, three carry a handle beside it (§7.3)
and seven are handle-free float64 utilities that stay float64 (§7.2). So
after the storage carries a dtype, **every dtype-general operation can
read its operands' dtype from an argument it already receives.** The only
gap is construction, and two constructors close it.

Per-operation float32 exports (`tf_core_add_f32`, `tf_core_matmul_f32`,
…) are **rejected**, for reasons that compound:

1. **Surface.** It would take the ABI from 52 to roughly 90–100 symbols,
   each needing a declaration, an `errcheck` registration, an inventory
   entry, and a test.
2. **It relocates the dtype from the data to the call site.** Python
   would have to choose a symbol per call from the operand's dtype —
   which is a dispatch decision made *outside* the runtime, in the layer
   §22 requires to stay thin.
3. **It violates the §4.2 dispatch shape** that every optimized path in
   this repository already follows: one unchanged export, one hidden
   predicate, one retained reference. Two exports per operation is two
   public paths.
4. **It doubles the drift surface.** Every future kernel change would
   have to land twice, and nothing structural would stop the two from
   diverging.
5. **It buys nothing measurable.** The dispatch it removes is one
   `switch` per exported call, executed once per operation over an entire
   tensor — invisible beside the Python-plus-ctypes floor Phase H already
   measured at ~7–12 µs per call.

### 6.6 Symbols that are *not* added

For the avoidance of doubt, Phase I adds no `tf_storage_dtype`,
`tf_storage_cast`, `tf_storage_bytes`, `tf_dtype_item_size`,
`tf_dtype_name`, `tf_storage_copy_from_f32`, `tf_storage_materialize_f32`,
or any per-dtype compute symbol. Python knows the dtype because it
created the storage; a query export would be a second authority for a
value the wrapper already owns.

---

## 7. The raw-buffer boundary

### 7.1 The division

The ABI splits cleanly into two classes, and Phase I makes the split
explicit rather than implicit:

- **Dtype-general handle-based paths** — every export that receives at
  least one storage handle. The handle's dtype is the authority. This is
  45 of the eventual 54 symbols (42 today plus the two new creators, plus
  the three transfer exports of §7.3 which bear a handle).
- **Float64-specific raw-buffer utility paths** — the seven handle-free
  kernels of §2.8: `tf_elementwise_add`, `tf_elementwise_subtract`,
  `tf_elementwise_multiply`, `tf_elementwise_divide`, `tf_relu`,
  `tf_matmul`, `tf_matmul_tiled`.

### 7.2 The seven raw kernels stay float64, deliberately

They receive *only* `const double*`, `double*`, and an element count.
There is no handle, therefore no dtype tag, therefore nothing to dispatch
on: making them dtype-general would require either new symbols (rejected,
§6.5) or a dtype-code parameter, which would change existing signatures
in a way that is not merely a retype (§7.3) but a genuine argument-count
change — an ABI break.

They are also not needed for native float32 training. They are the
reference/benchmark set: no `NativeTensorCore`, `NativeTensor`, module,
loss, optimizer, or checkpoint path calls them. The complete native
float32 training stack runs entirely through handle-based `tf_core_*`
exports.

**The limitation is explicit and testable.** A registry declares it:

```python
RAW_KERNEL_DTYPES = ("float64",)
```

introduced in `src/tensorforge/backends/cpp.py` beside `RAW_KERNELS`, and
reported by `backend_info()` as `"raw_kernel_dtypes"`.

**Where and when it is introduced: milestone I2, not I0 — and it landed
there.** I0 declared no registry, because a contract-only tuple would
advertise a distinction that was not yet observable: every dtype was
float64, so the tuple would have been indistinguishable from
`SUPPORTED_DTYPES` and would have carried no information. I2 established
the typed transfer boundary and is therefore the first milestone at which
"this path is float64-only and that one is not" is a true statement about
the running code. Adding it was a deliberate capability-registry change,
called out as such in the I2 exit gate.

As delivered, the limitation is proved as three separate facts rather than
asserted as one tuple, because they are genuinely different claims:

1. **The C ABI positions are float64 and reject anything else.** Every raw
   kernel's array arguments keep the float64 checked `ndpointer` binding,
   so a float32 buffer cannot reach one.
2. **The Python wrappers convert rather than compute narrow.** They are the
   same explicit host-to-native conversion boundary `from_array` is, and
   have been since v0.x: a float32 input is converted to float64 on the way
   in and the result is **float64**. No float32 arithmetic happens at any
   width in any of them.
3. **No per-dtype raw wrapper or export exists**, and none may be added.

`backend_info()` reports `raw_kernel_dtypes` beside `supported_dtypes` so
neither can be read off the other. The two tuples are equal today and are
not the same statement: this one is a permanent property of seven
handle-free kernels, the other is a public promise that moves at I9.

### 7.3 The three transfer exports become dtype-general

`tf_storage_copy_from`, `tf_storage_copy_to`, and
`tf_storage_materialize` each carry a storage handle *and* a host buffer.
The handle supplies the dtype; the host buffer's declared type widens:

```c
void tf_storage_copy_from(void* handle, const void* src);
void tf_storage_copy_to(const void* handle, void* dst);
void tf_storage_materialize(const void* handle, void* dst,
                            const int64_t* shape, const int64_t* strides,
                            int64_t offset, int64_t ndim);
```

**This is a source-level retype, not an ABI change.** On every supported
platform a `double*` and a `void*` occupy the same argument slot and are
passed identically; `extern "C"` means there is no mangling to change;
the symbol names, argument counts, argument order, and return types are
untouched; and the export inventory does not grow. A previously compiled
caller would link and run identically. This is the **one** place Phase I
alters a declared parameter type on an existing export, it is recorded
here rather than discovered later, and no test pins a C parameter
spelling.

On the Python side the checked binding is chosen from the storage's
dtype: `_CHECKED_F64_ARRAY`, joined by a `_CHECKED_F32_ARRAY` at I2. The
choice is made from **data** — the storage tag — never from an
environment variable, a global, or a call-site flag, so it is not a
dispatch control. The `ndpointer` check keeps doing exactly what it always
did: verifying at every call that the caller-facing buffer really is a
NumPy array of exactly the expected dtype, byte order included, and
C-contiguous.

**Where that check lives, as delivered at I2, and why it moved.** A ctypes
`argtypes` slot holds exactly one type, and these three positions must
accept either dtype depending on a *runtime* value — so the binding could
not stay in the slot. The declaration became `ctypes.c_void_p`, which is
the precise expression of the C parameter rather than a looser one, and
the per-dtype binding moved one layer out to `cpp._host_pointer(array,
dtype)`, which is called at every transfer site:

- it runs `ndpointer.from_param` — literally the function ctypes would have
  run had the binding stayed in the slot, so there is **one**
  implementation of the check and it cannot drift from a second one;
- it selects that binding from `_CHECKED_HOST_ARRAYS[storage.dtype]`, so
  the choice is still made from data and from nothing else;
- a wrong buffer raises `TypeError` and the native call is never made;
- it returns `ndarray.ctypes.data_as(c_void_p)`, which attaches the owning
  array to the pointer, so the buffer cannot be freed while the pointer
  exists — the same lifetime property the trusted layout pointers rely on,
  and the reason `POINTER(...).from_address(...)` is not used.

The observable consequence, recorded so the ABI inventory tests are not
mistaken for a regression: the checked-array-position tally falls from 25
to 22 and the handle-only export column rises from 28 to 30, while the
export count stays at 54 and no check is lost.

**The host pointer carries no dtype, and cannot be made to.** The ABI
receives an address and nothing else, so it is structurally incapable of
proving that the buffer behind it holds the element type the storage does.
That is stated rather than checked in C: the storage handle's immutable tag
is authoritative, C++ dispatches from it, Python validates the NumPy dtype
before the call, and a **direct foreign caller remains responsible for
satisfying the contract** — exactly as it already is for the layout arrays
it passes. No byte-count argument and no dtype argument were added, and no
host dtype is ever guessed.

The two `const int64_t* targets` positions of the cross-entropy exports
are **unchanged**: class labels are host metadata, not tensor data, and
stay `int64` regardless of the logits' dtype.

### 7.4 Scalar `double` parameters are unchanged, and their conversion is specified

`tf_storage_fill(handle, double value)`, `tf_storage_scale(handle, double
factor)`, and `tf_core_dropout_forward(..., double p)` keep their
signatures. The scalar crosses as the widest binary floating-point type
the ABI has, and the kernel converts it **once, before its loop**, to the
storage's element type:

```cpp
const T v = static_cast<T>(value);
for (int64_t i = 0; i < n; ++i) data[i] = v;
```

**Converting a scalar argument is not casting a tensor.** A scalar is a
hyperparameter or a fill value supplied by the caller in Python's only
float type; a tensor is data with a dtype the runtime owns. Filling a
float32 tensor with `0.1` produces `float(0.1)` in every element, which
is the only sensible reading of the request, and it is documented rather
than silent. The no-cast rule of §9 governs native-tensor-to-native-tensor
conversion, which never happens.

The one place this needs an explicit numerical statement is `mean`, whose
scale factor is `1/count`: the reciprocal is computed once in binary64
(correctly rounded) and narrowed once to the element type before the
multiply loop, so the result is deterministic, identical on every
platform, and independent of `count`'s magnitude. Computing `1.0f/count`
in float32 instead would differ by up to one ULP for some counts; the
chosen form is stated so no milestone can quietly pick the other.

---

## 8. Dispatch strategy

### 8.1 One narrow dispatch per operation

Every dtype-general exported function has **exactly one** dispatch point:
at the top of the export body, *after* argument validation and *before*
any compute, it reads the operands' dtype from their handles, confirms
they agree, and calls one instantiation of a templated kernel. Below that
point nothing branches on dtype — not the plan builder, not the row
kernel, not the odometer carry, not the accumulator, not the predicate.

```cpp
TF_EXPORT void tf_core_add(const void* a, const void* b, void* dst, ...) {
    TF_GUARD_BEGIN
    // ... existing validation, unchanged ...
    switch (dtype_of(a)) {                      // ONE dispatch
        case Dtype::Float64: core_add<double>(...); break;
        case Dtype::Float32: core_add<float>(...);  break;
    }
    TF_GUARD_END_VOID()
}
```

### 8.2 How the templates extend

The H8 traversals are already templates over the *operation*; Phase I
adds the *scalar type* as a second parameter, and the operation structs
become templated on it:

```cpp
template <class T, class Op>
inline void unary_row(const T* src, T* dst, std::int64_t n, std::int64_t stride);

template <class T, class Op>
inline void binary_row(const T* a, const T* b, T* dst, std::int64_t n,
                       std::int64_t a_stride, std::int64_t b_stride);

struct AddOp { template <class T> static inline T apply(T x, T y) { return x + y; } };
```

Requirements on the extension:

- **Explicit `float` and `double` instantiation** for every kernel and
  every retained reference path. Both dtypes take the *same* code, so
  they cannot drift.
- **No virtual dispatch per element**, no function-pointer indirection
  added, no `std::function`, no runtime `Op` value.
- **No string-based dispatch anywhere**, and emphatically not in a hot
  loop.
- **No duplicated full kernels** unless measurement proves duplication
  necessary — and if it ever does, the duplication is recorded as a
  decision with its measurement, not slipped in.
- The retained generic reference paths of `CLAUDE.md` §4.2 are
  instantiated for both dtypes, so every optimized path keeps its oracle
  **per dtype**.
- The `exp`/`log` exclusion from the templated traversals (H8's decision,
  because they are library functions with no correctly-rounded
  guarantee) is preserved: they keep their retained paths, templated on
  the scalar type only, calling `std::exp`/`std::log` on the element type
  so a float32 tensor uses the float overload rather than computing in
  double and narrowing.

### 8.3 Where dispatch belongs, and where it must not happen

**Legitimate dispatch points — four, and only four:**

| Point | Frequency | What it decides |
|---|---|---|
| ABI entry | once per exported call | which template instantiation runs |
| Storage-transfer boundary | once per transfer | which host element type the buffer holds |
| Python operation entry | once per call | which NumPy dtype / checked `ndpointer` binding to use |
| Checkpoint serialization boundary | once per array | which NumPy dtype to write or expect |

**Forbidden:** inside any element loop; inside a row kernel; per plan
axis; per accumulation step; per window in conv2d or pooling; per class
in softmax or cross-entropy; per parameter inside an optimizer step (the
step dispatches once per parameter *tensor*, which is per operation, not
per element); per array element inside a checkpoint loop; and anywhere a
Python call site would have to ask "which dtype am I?" more than once per
operation.

### 8.4 Python-side dispatch

The Python layer holds one table lookup per operation at most: the
storage's dtype string to its NumPy type, or to its checked `ndpointer`
binding, used when the call actually crosses a host buffer. Handle-only
calls — which is nearly all of them — pass no dtype at all and therefore
do no Python-side dtype work whatsoever. This is the property that keeps
H3's and H7's boundary work intact.

---

## 9. No casting, no promotion, no mixed dtype

### 9.1 The rule

**Operations require matching input dtypes.** There is no implicit
promotion, no implicit narrowing, no explicit cast operation, and no
mixed-dtype arithmetic. float32 + float64 raises; it does not become
float64 and does not become float32. This is not a new rule — it is
`docs/native_dtype_device_metadata_design.md` §8, written in v1.20 for
exactly this moment, and Phase I is the milestone that makes it able to
fire.

### 9.2 Where mixed dtype is rejected

Every one of these sites **already exists** and already performs a dtype
equality check that is vacuously true today. Phase I's obligation is to
keep each one, make each one reachable, and test each one:

| Layer | Site |
|---|---|
| Binary elementwise, matmul | `NativeTensorCore._require_matching_metadata` |
| Convolution | `NativeConv2d.forward` (`input.dtype != weight.dtype`) |
| Linear | `NativeLinear.forward` |
| Normalization | `NativeLayerNorm.forward`, the shared BatchNorm implementation (`tensor.dtype != input.dtype`) |
| Loss functions | `NativeMSELoss` (prediction vs target), cross-entropy (logits, saved probabilities, upstream) |
| Parameter ↔ gradient | `NativeSGD.step`, `NativeAdam.step` (`grad.dtype != parameter.dtype`) |
| Optimizer state | `NativeAdam` moment validation (`state.dtype != parameter.dtype`), `load_state_dict` metadata validation |
| Parameter mutation | `NativeParameter.copy_value_`, `_replace_core` |
| Module state loading | `NativeModule.load_state_dict` (`value.dtype != destination.dtype`) |
| Staged state transactions | `_native_state` staging validation |
| Checkpoint loading | model entries, optimizer parameter metadata, and (new at I8) optimizer moment entries |
| Autograd accumulation | `NativeTensor` gradient accumulation (`g_core.dtype != core.dtype`) |

Sites that must be *added* rather than kept: the five hard
`dtype != "float64"` gates of §2.3 become dtype-general acceptance
(any supported dtype) with the operand-agreement check unchanged, and the
new dispatch points of §8.1 confirm operand agreement in C++ as a
defence-in-depth revalidation at the trust boundary — the same principle
that already makes the C++ side revalidate every cross-entropy target
index rather than trusting Python.

### 9.3 Rejection happens before anything is allocated or mutated

A mixed-dtype call must fail **before output allocation and before any
state mutation**, and this is achievable at every site above because each
already validates before allocating. The requirement is stated as a
testable property: after a rejected mixed-dtype operation, native live
storage is exactly what it was, every operand is unchanged and open, no
parameter version has moved, no optimizer state exists that did not
exist, no generator call has been consumed, and no graph node has been
created.

### 9.4 Host-array ingress is not a tensor cast

`from_array(values, dtype=...)` converts a host object — a Python list, a
NumPy array of any dtype — into native storage of the **requested**
dtype. That is the explicit host-to-native conversion boundary and it has
always converted; float32 changes only which target it converts to.

Two rules make it unambiguous:

- **The dtype is never inferred from the input array.** `dtype=None`
  means `"float64"`, exactly as today, so a float32 NumPy array handed to
  `from_array` without a dtype produces a **float64** native tensor.
  Inference would silently change the meaning of existing code the day
  someone passed a float32 array.
- **Egress reproduces the storage dtype exactly.** `to_numpy()` on a
  float32 tensor returns a float32 array. It never widens on the way out,
  because a widened result would silently claim precision the tensor does
  not have.

### 9.5 What is not added

No `astype`, no `to`, no `.float()`, no `.double()`, no `cast`, no
`same_kind` rules, no `result_type`, no promotion table. The v1.20
contract listed explicit casting as a future design item and it stays
future: Phase I gives the runtime two dtypes, not a way to move between
them. A tensor's dtype is fixed at construction and the only way to get
the other one is to construct it.

---

## 10. Accumulation policy and numerical contracts

### 10.1 The rule

**float32 operations accumulate in float32 and produce float32 outputs
and gradients.** No hidden float64 accumulator is introduced anywhere:
not in matmul, not in `sum`/`mean`, not in any convolution direction, not
in softmax / log-softmax / cross-entropy, not in LayerNorm or BatchNorm
statistics, not in an optimizer moment update.

Why, stated so no later milestone relitigates it:

1. A wider hidden accumulator makes a float32 result depend on an
   invisible policy no part of the public contract states.
2. It is mixed precision — which is AMP, which §1.2 excludes.
3. It breaks the "one traversal, two instantiations" symmetry that keeps
   the dtypes from drifting, because the accumulator type would no longer
   follow the element type.
4. It would make the float32 exact-resume proof depend on that policy
   rather than on IEEE-754.
5. It would tempt a later milestone to "improve agreement with float64",
   which is a comparison §10.4 forbids making a contract out of at all.

The only exceptions are the ones that already exist and are already
documented as scalar work rather than accumulation: §7.4's scalar
conversion, and §15.3's optimizer scalar coefficients.

### 10.2 Operation order determinism

**Per-output accumulation order is preserved exactly, per dtype.** The
traversal is literally the same source with a different scalar type, so
every ordering statement Phases D, E, F, and H made carries over
structurally — but each is *restated* for float32 and re-proved, never
inherited by assertion. Nothing anywhere reassociates arithmetic, uses
FMA, fast-math, an intrinsic, `restrict`, a tree/pairwise/parallel
reduction, or a horizontal vector reduction, at either width.

Subnormals are a specific float32 concern: binary32 subnormals are far
more reachable than binary64 ones. There is **no flush-to-zero and no
denormals-are-zero mode**, no `-ffast-math`, and no `/fp:fast`, so
subnormal float32 values behave exactly as IEEE-754 specifies. A
milestone may not add an architecture flag that changes this.

### 10.3 Where exact bit equality is required

- **float64 results are bit-identical to pre-Phase-I.** This is the
  strongest numerical requirement of the phase. Every float64 operation,
  every float64 training step, every float64 checkpoint, and every
  existing float64 exact-resume proof must produce the same bits after
  the generalization as before it. Tested by committed bit-pattern
  comparison and by every pre-existing test continuing to pass unchanged.
- **Value transfer is bit-preserving at both widths.** A
  `contiguous_copy`, a `copy_value_`, a state-dict transfer, and a
  checkpoint round trip reproduce the source's bits exactly — including
  `-0.0`, both signs of signalling NaN, and every NaN payload. For
  float32 that means all 23 payload bits, asserted through
  `ndarray.view(np.uint32)`. The reason is H5's and it is width
  independent: a transfer performs no arithmetic, so it has no operand
  roles to choose between. An *operation* that happens to copy
  (`zeros + x`) follows IEEE arithmetic instead and therefore does
  normalize `-0.0` and quiet a signalling NaN — at both widths.
- **Elementwise float32 against a NumPy float32 oracle**: bit-identical
  whenever at most one operand is NaN; `subtract` bit-identical
  everywhere; two-NaN payloads outside the contract, exactly as at
  float64.
- **matmul, reductions, and conv2d at float32**: every non-NaN result
  bit-identical to the retained reference path at the same dtype; NaN
  positions identical and quiet; signed zeros proved as raw bit patterns;
  at most one NaN per destination agrees including payload; two-or-more
  payloads outside the contract.
- **Dropout at float32**: the keep/drop decision sequence is bit-exactly
  the same as at float64 for the same key (§14.2), and the mask values
  are exactly `0.0f` and `float(1/(1-p))`.

### 10.4 Where tolerance is appropriate — and where a comparison is forbidden

- **Appropriate:** float32 results against an independent NumPy **float32**
  oracle for composed or transcendental work; finite-difference gradient
  checks at float32, with a step size and tolerance chosen for binary32
  and stated in the test rather than inherited from the float64 tests;
  `exp`/`log` at float32 against a **one-ULP** finite bound, for the same
  reason the float64 contract uses one — libm differs between toolchains.
- **Forbidden as a contract:** comparing a float32 result to a float64
  result at *any* tolerance and calling the agreement a requirement.
  float32 does not have to reproduce float64 and never will in general. A
  test may compare the two to illustrate a magnitude, but no contract, no
  gate, and no closure claim may rest on it.

### 10.5 How stable math stays stable at float32

Softmax, log-softmax, and cross-entropy are stable because of the
**maximum shift** and the **fused log-sum-exp**, not because of the
width. Both survive verbatim at float32 and must be computed in float32:

- the per-row maximum is a float32 comparison scan;
- `exp(x - max)` is `std::exp` on `float`;
- the log-normalizer is accumulated in float32;
- `log_softmax` remains its own fused kernel and is **never**
  `softmax().log()` — the reason is even stronger at float32, where the
  smallest normal probability is ~1.18e-38 and underflow arrives far
  sooner.

The honest statement Phase I must make and test: float32 classification
has a smaller dynamic range than float64, so the *magnitudes* at which
these kernels saturate differ. What does **not** differ is that neither
kernel produces a NaN or an infinity for any finite input for which the
float64 kernel does not, because the shift is what guarantees that and
the shift is width-independent.

---

## 11. Autograd dtype invariants

Phase I changes no autograd *structure*. The graph, the node ownership,
the `graph_resources` contract, the retain/release rules, the
parameter-version rules, and the failure semantics are exactly what
`docs/native_autograd_design.md` and the later phase contracts define.
What it adds is a dtype invariant at every point where a tensor is
produced:

1. **Output gradients have the dtype of the tensor they are a gradient
   of.** `grad.dtype == tensor.dtype` at every node, which is already the
   stated `NativeTensor` contract.
2. **Leaf gradients match their parameter's dtype.** A float32 parameter
   accumulates a float32 gradient; a float64 parameter a float64 one; a
   model may legitimately contain both, and each subgraph stays internally
   consistent.
3. **Saved tensors and saved resources retain the graph's dtype** — the
   Dropout multiplier mask, the BatchNorm eval snapshots, and the
   cross-entropy saved probabilities. The MaxPool2d winner buffer is the
   one exception and it is not a numeric operand (§13.3).
4. **Backward-created temporaries use the graph dtype.** Every constant a
   backward materializes — `0.5` for `sqrt`, `-1.0` for `reciprocal`,
   `1/count` for `mean`, `eps`, `1 - momentum`, the ones tensor — is
   already built with `dtype=<the operand's dtype>` at every existing
   site. Phase I keeps that and forbids introducing any literal-float64
   constant into a backward.
5. **Mixed-dtype gradient accumulation is rejected**, before the
   accumulation and before any allocation, by the existing check.
6. **Released saved resources remain safely released.** Dtype does not
   change the release-exactly-once contract, the retain-under-
   `retain_graph` behavior, the survival across a failed retryable
   backward, or the abandoned-graph `close()` path.
7. **Failure paths leak no dtype-specific storage.** A backward that
   fails after allocating float32 temporaries closes every one of them,
   and live native storage returns exactly to baseline — tested with the
   deterministic allocation-failure hook at both widths.
8. **Graph ownership and lifetime rules are unchanged** in every other
   respect. No milestone may use "dtype support" as cover for a graph
   change.

---

## 12. Parameters, modules, and initialization

### 12.1 Constructors gain a dtype argument, defaulting to float64

At milestone **I7** (not before), each of these accepts `dtype` and
defaults it to `"float64"`:

`NativeParameter`, `NativeLinear`, `NativeConv2d`, `NativeLayerNorm`,
`NativeBatchNorm1d`, `NativeBatchNorm2d`. `NativeReLU`, `NativeFlatten`,
`NativeSequential`, `NativeDropout`, `NativeMSELoss`, and
`NativeCrossEntropyLoss` hold no parameters or buffers of their own and
therefore take **no** dtype argument — they inherit the dtype of whatever
flows through them, and giving them one would create a second authority
that could disagree with the data.

The tensor factories `NativeTensor.zeros`, `.full`, and `.from_array`
already have the argument; only their NumPy conversion becomes
dtype-aware.

### 12.2 Validation and propagation

- Validation is `normalize_dtype`, the same function, producing the same
  `TypeError` for a non-string and the same `ValueError` naming the value
  and the supported set. No constructor invents its own dtype validation.
- Validation happens **before any native allocation**, so a bad dtype
  never creates storage a constructor then abandons — the rule
  `NativeLinear.__init__` already follows for its other arguments.
- A container module (`NativeSequential`) does not force a dtype on its
  children: each child was constructed with its own, and the first
  mismatched forward raises at the site of the mismatch with both dtypes
  named. Silently unifying them would be promotion.
- `state_dict()` remains `{name: NativeTensor}` and carries dtype
  implicitly through those tensors; `load_state_dict` validates dtype per
  entry against the live destination (it already does).

### 12.3 Initialization

`NativeLinear` and `NativeConv2d` draw fan-in uniform initial values with
a **local** `numpy.random.default_rng(seed)` into a float64 host array
and hand it to `NativeParameter`. Phase I keeps the host draw exactly as
it is and converts once at the ingress boundary:

**A float32 layer with seed *S* has weights equal to `float32(the float64
draw with seed S)`.** The underlying random stream is identical across
dtypes, the seed→values relationship is exactly specified, and a float32
model and a float64 model built with the same seed start from the same
values to within one rounding. Drawing directly in float32 (a different
NumPy stream) is **rejected**: it would silently make the two dtypes
start from unrelated points and would make the seed contract
dtype-dependent for no benefit.

The bound `1/sqrt(in_features)` is computed in Python (binary64) and
enters the draw before conversion, so it is not itself rounded twice.

### 12.4 Public naming and existing call sites

Every existing call site that omits `dtype` keeps producing float64
tensors, parameters, buffers, and modules, with byte-identical values.
This is a hard compatibility requirement of the phase, tested directly
rather than assumed.

---

## 13. Persistent buffers and normalization state

### 13.1 Buffers use the module's dtype

A module's persistent buffers are constructed at the module's dtype and
stay there:

- BatchNorm `running_mean` and `running_var`;
- the affine `weight` and `bias` of LayerNorm and BatchNorm (parameters,
  not buffers, but the same rule);
- the graph-safe **eval snapshots** BatchNorm takes of its running
  statistics;
- any other genuinely numeric registered tensor.

The running-statistics update remains the Phase-F **atomic two-buffer
transaction**, unchanged in structure: both new buffers are fully
computed and validated before either replaces a live one, and a failure
leaves both originals in place with identities and versions untouched.
Dtype adds one validation to that transaction — the replacement's dtype
must equal the original's — and changes nothing else.

### 13.2 Identity, aliasing, and restoration are unchanged

Buffer identity, alias behavior, `state_dict()` naming, the
`persistent=False` exclusion from serialization, in-place restoration on
load, and the rule that loading buffer state moves **no** parameter
version and stales **no** graph all survive Phase I exactly as Phase F
defined them.

### 13.3 Integer-like metadata stays metadata

Counters and indices that are not tensors do not become tensors, and this
phase adds no integer tensor dtype. Two decisions are recorded explicitly
because they are the ones a careless milestone would get wrong:

- **Cross-entropy targets stay a host `int64` array.** They are class
  indices, copied independently, revalidated in C++ against
  `num_classes`. They are unaffected by the logits' dtype.
- **The MaxPool2d winner buffer stays float64 at every tensor dtype.**
  This is a deliberate, load-bearing decision. The winners are *flat
  plane offsets encoded as floating-point values*, exact only while the
  plane fits the mantissa: `2**53` for binary64 and only `2**24` for
  binary32. Making the winner buffer follow the tensor's dtype would
  silently cut the largest poolable plane from about 9.0e15 elements to
  16,777,216 — a capability regression disguised as a dtype feature.
  Keeping it float64 preserves `_MAX_EXACT_WINNER_PLANE == 2**53`
  unchanged for both dtypes and preserves the existing exactness proof
  verbatim.

  The consequence is that `tf_core_maxpool2d_backward` legitimately
  receives a float64 winner handle beside float32 gradient handles. That
  is **not** a mixed-dtype operation: the winner buffer is private index
  metadata, never a tensor operand, never a public tensor, never in a
  `state_dict()`, and never returned to a caller. The backward validates
  the gradient handles against the graph dtype and the winner handle
  against float64, separately, and the dispatch of §8.1 is on the
  gradient dtype alone.

  The rejected alternative — a real integer winner buffer — would be
  strictly better engineering and is **out of scope**: it requires an
  integer storage dtype, which §1.2 excludes. It is recorded here as the
  natural first item for any future integer-tensor phase.

---

## 14. RNG and Dropout

### 14.1 The generator is untouched

The algorithm (`tensorforge.splitmix64`), `ALGORITHM_VERSION == 1`, the
seed representation, the call index, the reserve → commit / abandon call
transaction, the `uint64` bounds, the registration model, the
`generator_state_dict()` / `load_generator_state_dict()` pair, the
process-wide state-transaction guard, and the universal lock order are
**all unchanged by Phase I**. Generator state is dtype-independent and
must never gain a dtype field.

**I0 changes no RNG code. No Phase-I milestone changes the derivation.**

### 14.2 The draw stays binary64; only the written values are dtype-dependent

`dropout_uniform(bits)` produces a `double` on `[0, 1)` as
`(bits >> 11) * 2**-53`, and it **stays exactly that** for both dtypes.
The comparison against `p` is therefore made on the same value at both
widths, which gives a strong and deliberately chosen property:

**For one `(seed, call_index, element count)` key, a float32 Dropout and
a float64 Dropout drop exactly the same elements.**

Only the values written differ: the kept multiplier is
`static_cast<T>(1.0 / (1.0 - p))`, computed once in binary64 and narrowed
once, and the dropped multiplier is `T(0)`.

Deriving the uniform in float32 instead is **rejected**: it would make
the drop pattern dtype-dependent for the same key, so a float32 and a
float64 run of the same model would diverge in *structure* rather than
only in rounding, and every committed known-answer vector would need a
second, unrelated form.

### 14.3 What must remain true at float32

- The output and the private multiplier mask are float32 tensors when the
  input is float32.
- Generator state is dtype-independent: the same generator may drive
  float32 and float64 Dropout and its `calls` counter advances
  identically.
- A **failed** operation consumes no generator call, at either dtype.
- **Evaluation** consumes no generator call and returns the input object
  itself, at either dtype.
- `p == 0` is identity, at either dtype.
- The exact-resume proof includes the **next generated mask** after
  resume, compared by exact equality at the tensor's own dtype.
- Shared-generator alias topology survives a checkpoint restore
  unchanged.

---

## 15. Optimizer state

### 15.1 State matches the parameter

Every native optimizer's per-parameter state has the **dtype of that
parameter**:

- `NativeSGD` — no persistent per-parameter tensor state today; its
  gradient/parameter dtype agreement check is the requirement.
- `NativeAdam` — `m` and `v` are allocated at the parameter's dtype, at
  initialization, at replacement, and after a load.
- Step counters stay **Python integers**, per parameter. They are
  metadata, not tensors, and they must not become tensors.
- Parameter groups do not exist on the native line and Phase I does not
  add them.

### 15.2 Validation and atomicity

- **State initialization** validates dtype/shape/device against the
  parameter before allocating either moment.
- **State replacement** and **state loading** validate every entry —
  count, positional shape, dtype, device — in complete passes **before**
  any mutation. Phase H's H4 optimizer step keeps its four-complete-
  validation-passes-then-commit structure, with dtype joining shape and
  device in those passes rather than adding a fifth pass.
- **Mixed dtype is rejected**: a float32 parameter with a float64
  gradient, a float64 moment, or a float64 checkpoint entry raises,
  before any state changes.
- **Allocation-failure atomicity**: a failure part-way through
  constructing state closes everything it allocated, leaves the optimizer
  exactly as it was, and moves no parameter version. Tested with the
  deterministic fault-injection hook at both dtypes.
- The commit boundary stays one `copy_value_` and one version increment
  per updated parameter, with shared parameters deduplicated to one slot,
  one update, one increment.

### 15.3 Scalar coefficients, and the H4 exactness proof at float32

`lr`, `betas`, and `eps` are Python floats — hyperparameters, not
accumulators. The H4 per-step scalar holder computes each coefficient
once per step in Python (binary64) and materializes it as a rank-0 native
tensor **at the parameter's dtype**, which the existing code already does
by reading `parameter.dtype`.

H4's exact-substitution proof — that evaluating the bias-correction
reciprocal in Python is literally `1.0 / x` on the same IEEE-754
**binary64** value as the native `reciprocal` kernel — is stated at
binary64 and does **not** transfer by assertion. Phase I must restate it
for float32: the coefficient is computed in binary64 and narrowed once to
binary32 at materialization, so the value entering the float32 arithmetic
is `float(1.0 / x)`. A float32 milestone must prove that this equals what
the float32 `reciprocal` kernel would produce for the same input, or —
if it does not for some inputs — compute the coefficient the way the
kernel does and record the finding. Either outcome is acceptable; silently
assuming the binary64 proof carries over is not.

The alternative of computing coefficients in float32 throughout is
**rejected**: bias corrections like `1 - beta2**t` lose precision quickly
in binary32 for large `t`, and a hyperparameter is exactly the kind of
scalar §7.4 says should be computed at the widest available precision and
narrowed once.

### 15.4 Phase-H efficiency is preserved

The per-step scalar holder, the once-per-step construction, the
release-at-last-use discipline, the reduced per-parameter allocation
count, and the measured peak-transient reduction all survive. A milestone
that reintroduces per-parameter scalar construction in the name of dtype
support has regressed H4 and must say so.

---

## 16. Checkpoint format version 3

**I0 designs it. I0 does not implement or activate it.** The constants
stay at `_FORMAT_VERSION = 2` and `_SUPPORTED_FORMAT_VERSIONS = (1, 2)`
until milestone **I8**.

### 16.1 The constants after I8

```python
_FORMAT                     = "tensorforge.native_checkpoint"  # unchanged
_FORMAT_VERSION             = 3
_SUPPORTED_FORMAT_VERSIONS  = (1, 2, 3)
```

The format **name** never moves — that is the rule G5 established and it
holds. Every new save writes `3`, whether or not the model contains a
float32 tensor: the version describes the **schema**, not the content,
exactly as v2 is written for generator-free models.

### 16.2 What version 3 adds

The manifest already carries a `"dtype"` field on every model entry and
on every optimizer parameter-metadata entry, and both are already
validated against the live destination. Version 3 therefore changes less
than it might appear:

1. **Model entries** — `"dtype"` may now be `"float32"`. The archive
   array's NumPy dtype must equal the declared dtype exactly, in native
   byte order.
2. **Optimizer parameter metadata** — `"dtype"` may now be `"float32"`,
   unchanged in structure.
3. **Optimizer moments** — the one structural change. In v2, `"m"` and
   `"v"` are bare lists of archive names, with shape and dtype implied
   positionally by the `"parameters"` list. In v3 each becomes a list of
   entry objects with the same shape as a model entry:

   ```json
   {"array": "optimizer::m::000000", "shape": [4, 3],
    "dtype": "float32", "device": "cpu"}
   ```

   **Rejected alternative:** inferring a moment's dtype positionally from
   `parameters[i]["dtype"]`. It works only while the two lists are
   consistent, which is precisely what a malformed archive violates, and
   it leaves `_read_arrays` with no per-array expected dtype to check
   against. An explicit entry makes a mismatch a rejection rather than a
   plausible read.
4. **Persistent buffers** need no new field: they are model state
   entries and already carry `"dtype"`.
5. **Gradients are not serialized** — they never have been, at any
   version, and Phase I does not start.
6. **The `"generators"` section is unchanged.** Generator state is
   dtype-independent: `algorithm`, `algorithm_version`, `seed` and
   `calls` as canonical decimal `uint64` strings, plus the complete alias
   topology. No dtype field is added to it, and adding one would be
   wrong.
7. **`"metadata"` is unchanged**, an opaque JSON-serializable dict.

### 16.3 Validation before mutation

The v3 loader's ordering, every step of which completes before the next
begins and **all** of which completes before anything live is touched:

1. path and archive readable; manifest entry present; manifest is
   `uint8` 1-D; JSON parses;
2. `"format"` equals the format name;
3. `"format_version"` is in `_SUPPORTED_FORMAT_VERSIONS`;
4. the manifest key set is **exactly** the set for that version (v1
   without `"generators"`, v2/v3 with it) — a v1 archive carrying a
   generator section is rejected, not half-read;
5. every section's structure and key set;
6. every declared dtype is a known, supported dtype string — an unknown
   or unsupported dtype is a rejection naming the value and the supported
   set;
7. every referenced array exists, exactly once, with no unreferenced
   extras, and reads without pickle;
8. every array's NumPy dtype equals its declared dtype in native byte
   order, and its shape equals its declared shape (so `numel × itemsize`
   agreement is a derived cross-check, not a stored field);
9. structural agreement with the **live** model: key set, per-key shape,
   dtype, and device;
10. structural agreement with the live optimizer: type, state format
    version, parameter count, per-position shape/dtype/device, step-count
    count;
11. generator topology, validated strictly in both directions against a
    real `named_generators()` traversal.

Only then: stage every replacement tensor, take rollback snapshots at the
commit boundary, commit under the single whole-checkpoint rollback guard,
and deliver.

### 16.4 What version 3 rejects

- an unknown or unsupported dtype code or string;
- a dtype/payload disagreement in either direction (declared float32 with
  a float64 array, or the reverse), and any non-native byte order;
- a dtype/shape/payload-size disagreement;
- a checkpoint whose dtype differs from the live model's, at any entry —
  the API contract is exact equality, and a mismatch is a `ValueError`
  naming the key, the checkpoint's dtype, and the model's. **No cast, no
  `map_location`, no device transfer, no "closest match".**
- a v1 or v2 archive declaring or containing anything but float64
  (§16.5);
- everything v2 already rejects: unknown versions, wrong format name,
  extra or missing archive entries, pickled entries, duplicate
  references, a generator topology that does not match.

### 16.5 Versions 1 and 2 are float64-only formats, permanently

A v1 or v2 archive is **defined** to be float64. Loading one:

- requires every declared `"dtype"` to be `"float64"` and every archive
  array to be float64;
- **never guesses that a payload is float32**, under any circumstance —
  not from an array's dtype, not from a byte length, not from a shape;
- keeps every v1/v2 rule Phase G established, including v1's
  generator-free requirement and its rejection for a generator-bearing
  model.

A v1/v2 archive can therefore only be loaded into a float64 model, and
that is the correct behavior: the format predates float32 and has no way
to say otherwise. There is **no upgrade in place** and no "latest wins" —
the loader dispatches on the version, exactly as it does today.

### 16.6 Alias, identity, and transaction guarantees are unchanged

Restoration remains **in place**: parameters, buffers, and generators keep
their identities and every sharing relationship; a shared parameter
deduplicates to one slot, one update, one version increment; loading
buffer or generator state moves no parameter version and stales no graph;
a full load replaces parameters and therefore correctly stales an earlier
graph through the parameter rule. The whole load is one transaction under
one rollback guard spanning the model, optimizer, and generator commits.
External process or interpreter death remains the only documented
exception to whole-checkpoint atomicity.

---

## 17. Serialization encoding

### 17.1 Deterministic dtype identifiers

The **serialized** dtype identifier is the canonical string —
`"float64"`, `"float32"` — not the numeric ABI code. Reasons, in order:

1. the manifest is JSON and already carries `"dtype"` as a string, which
   `_MODEL_ENTRY_KEYS` and the optimizer parameter metadata both
   validate; using a different form for the same concept in the same file
   would be gratuitous;
2. a string is self-describing in an archive a human may inspect;
3. it is decoupled from the ABI numbering, so a future ABI code change
   (which will not happen — the codes are frozen — but the decoupling is
   free) cannot invalidate an archive.

The numeric codes of §3.1 are for the C ABI and appear nowhere in a file.

### 17.2 Payload encoding

- **Container**: the existing uncompressed `npz` written with
  `numpy.savez` (never `savez_compressed`, never pickle), one array per
  entry, plus the `uint8` JSON manifest. Unchanged.
- **Byte order**: **native**, recorded by the NPY descriptor NumPy writes
  (`<f8` / `<f4` on a little-endian machine). The loader compares the
  read array's dtype against the canonical native dtype, so a foreign
  byte order fails the comparison and is rejected — the behavior v2
  already has for float64, generalized. TensorForge does not claim
  cross-endian portability and does not silently byte-swap.
- **Element count** comes from the declared `"shape"`; the array's actual
  shape must equal it.
- **Expected payload byte length** is `numel × itemsize` and is a
  **derived cross-check**, not a stored field (§16.3 step 8). Storing it
  would be a second source of truth that a malformed writer could make
  disagree with the shape.
- **Zero elements**: not constructible in the native runtime (shapes must
  be positive), so no zero-element payload can be produced; a manifest
  declaring one is rejected by shape validation. This is stated rather
  than left implicit.

### 17.3 Value fidelity

The save/load round trip is **bit-preserving** at both widths, and this
is a contract, not a hope:

- **signed zero** — `-0.0` survives as `-0.0`;
- **infinities** — both signs survive;
- **NaNs** — quiet and signalling NaNs survive with **all payload bits
  intact**: 52 for binary64, 23 for binary32. NaN payload preservation
  *is* part of the transfer contract, unlike the arithmetic contracts of
  §10.3, because a transfer performs no arithmetic;
- **subnormals** — survive exactly, which matters more at binary32.

Tested by raw bit patterns (`.view(np.uint64)` / `.view(np.uint32)`),
never by `np.allclose` and never by an equality that NaN would silently
pass or fail.

The path this rests on: `snapshot.to_numpy()` materializes through the
native materialization kernel, which assigns elements without arithmetic;
`numpy.savez` writes those bytes; the loader reads them back and
`NativeTensor.from_array` copies them into fresh storage through
`tf_storage_copy_from`, again without arithmetic. Every step is an
assignment.

### 17.4 Malformed-payload behavior

Every malformation is a **`ValueError` naming the archive entry, what was
expected, and what was found**, raised before any live state is touched:
a wrong dtype, a wrong shape, a missing array, an unreferenced extra, a
duplicate reference, an entry that cannot be read without pickle, a
truncated or corrupt archive, a manifest that is not `uint8` 1-D, a
manifest that is not JSON, an unknown format name, an unsupported
version, a wrong key set for the version, an unknown dtype string, or a
dtype the runtime does not support. The existing corruption-matrix test
style extends to cover the dtype cases.

---

## 18. Determinism and exact resume

### 18.1 Two independent proofs

Phase-I closure requires exact deterministic resume proved **separately**
for float32 and float64. Each is self-consistent and neither is compared
to the other.

- The **float64** proof is the existing one: every pre-Phase-I resume
  test and example keeps passing, unchanged, with byte-identical results.
- The **float32** proof is new and integrated, over a network of
  approximately this shape:

```text
Conv2d → BatchNorm2d → ReLU → MaxPool2d → Flatten
       → Linear → BatchNorm1d → ReLU → LayerNorm → Dropout
       → Linear → CrossEntropy → NativeAdam
```

so that it carries all four TensorForge-owned state families at once:
parameters, persistent buffers, a registered generator, and optimizer
moments — plus all four graph-owned saved-resource families (Dropout
mask, MaxPool2d winners, BatchNorm eval snapshots, cross-entropy saved
probabilities).

### 18.2 What the interrupted/resumed comparison must reproduce

An interrupted run, checkpointed and reloaded into **completely fresh**
model, optimizer, and generator objects, must reproduce the uninterrupted
run by **exact equality** on:

- the resumed **loss suffix**;
- every **parameter**;
- **gradients** where contractually relevant — gradients are not
  checkpointed, so the claim is that the first resumed step *produces*
  equal gradients, compared directly at that step;
- every **persistent buffer**;
- every **optimizer moment**;
- every **step counter**;
- the **generator state** (`algorithm`, `algorithm_version`, `seed`,
  `calls`);
- the **shared-generator alias topology**;
- the **next Dropout mask** generated after resume;
- the final **logits**, **predictions**, and **evaluation output**.

As in every phase since C, the external loop position is carried as
explicit validated metadata rather than claimed as automatic checkpoint
state, and reproducibility is exact **for the state TensorForge
captures** — Python's `random`, NumPy's global RNG, data-loader position,
batch order, and scheduler state are not captured and full-program
determinism is not claimed.

### 18.3 Exact resume is not float64 agreement

**A float32 run does not need to produce the same numbers as a float64
run, and no part of the proof may depend on it.** "Exact" here means
*bitwise identical between the interrupted-and-resumed run and the
uninterrupted run at the same dtype*. Two distinct claims, kept distinct:

| Claim | Status |
|---|---|
| float32 resumed == float32 uninterrupted, bitwise | **required** |
| float64 resumed == float64 uninterrupted, bitwise | **required** (already true) |
| float64 after Phase I == float64 before Phase I, bitwise | **required** |
| float32 ≈ float64 to some tolerance | **not a contract**, never asserted |

---

## 19. Stable / native separation

Unchanged, and Phase I may not weaken any of it:

- The stable framework never imports the native backend; importing
  `tensorforge` must not load the C++ library, and the subprocess test
  that proves it must keep passing.
- `stable_framework_integration` stays `False` in `backend_info()`.
- **No automatic migration** of stable `Tensor`s into native tensors, in
  either direction, at any dtype.
- **No implicit backend dispatch** and **no environment-variable backend
  or dtype selector.** Which dtype a tensor has is a property of how it
  was constructed, full stop.
- **No change to default public behavior**: existing code that omits
  `dtype` gets float64 everywhere, in the native line and in the stable
  line alike.
- **No stable-API dtype change caused by Phase I.** The stable
  NumPy-backed `Tensor` is feature-frozen at v3.0 and this phase does not
  touch it, its dtype behavior, or its serialization.
- Native modules stay separate classes (`NativeLinear`, not a `Linear`
  backend flag), and float32 support is a native-line capability that
  says nothing about the stable line.

---

## 20. Error handling and failure atomicity

The C ABI error contract is unchanged: no C++ exception may escape an
exported function; fallible functions clear the thread-local slot on
entry and record a `TfStatus` plus message on failure; `TF_ERROR_ALLOC` →
`MemoryError`, `TF_ERROR_INVALID` → `ValueError`, `TF_ERROR_RUNTIME` →
`RuntimeError`.

Dtype-aware failure behavior, each row of which is a required test:

| Failure | Raised as | Required post-condition |
|---|---|---|
| Unknown dtype code at the ABI | `ValueError` | no allocation, no handle, nothing published |
| Unsupported dtype string in Python | `ValueError` | rejected before any native call |
| Non-string dtype in Python | `TypeError` | rejected before any native call |
| Overflow in `numel × itemsize` | `ValueError` | no allocation attempted |
| Allocation failure (real or injected) | `MemoryError` | live storage exactly at baseline; no partial node |
| Invalid raw host buffer (wrong dtype, non-contiguous, not an ndarray) | `TypeError`/`ValueError` from `ndpointer` | native call never made |
| Payload-length or shape mismatch on load | `ValueError` | nothing live mutated |
| Mixed dtype in any operation | `ValueError` naming both dtypes | no output allocated, no operand touched, no version moved, no generator call consumed |
| Mixed dtype in module/optimizer/checkpoint load | `ValueError` | nothing staged is committed; identities and versions unchanged |
| Invalid output handle (null, closed) | `ValueError`/`RuntimeError` | destination byte-for-byte unchanged |
| Partial optimizer-state construction | `MemoryError`/`ValueError` | everything allocated is closed; optimizer exactly as before |
| Partial checkpoint deserialization | `ValueError` | whole-checkpoint rollback; live state exactly as before |
| Exception during Python wrapper construction | propagates | every native resource the attempt created is closed |
| Backward failure after temporaries exist | propagates | every temporary closed; saved resources still retained; live storage at baseline |

**The governing rule:** no failure may publish a partially constructed
object or mutate destination state before validation is complete. A
rejecting export writes nothing.

---

## 21. Ownership, lifetime, and concurrency

Phase I introduces **no new ownership framework** and changes no
ownership rule. The existing model holds unchanged:

- a `NativeTensorCore` owns its `NativeStorage`; a `NativeTensorView`
  borrows; a chained view keeps the whole chain reachable;
- every operation allocates a fresh owning contiguous output aliasing
  neither operand;
- cleanup is explicit — `close()` is the contract, `__del__` is a
  fallback, and any failure closes everything it allocated;
- graph-owned saved resources are released exactly once with the graph
  history, retained under `retain_graph=True`, kept alive across a failed
  retryable backward, freed by an abandoned graph's `close()`, and closed
  immediately by a no-grad forward;
- a registered buffer is never a rereadable graph operand;
- loaders preserve every identity and sharing relationship and restore in
  place;
- a parameter's version counter moves once per committed mutation.

What dtype adds, and only this:

| Area | Dtype effect |
|---|---|
| Storage destruction | one allocation form, one deallocation form, so the type can never mismatch (§4.1) |
| Shared views | none — one dtype field per storage, so views cannot disagree |
| Graph-saved resources | each carries the graph dtype, except the winner buffer (§13.3) |
| Parameter / gradient lifetimes | unchanged; the dtype check is one more pre-mutation validation |
| Persistent buffers | unchanged; dtype joins shape/device in the replacement validation |
| Optimizer state replacement | unchanged; dtype joins the complete validation passes |
| Checkpoint claim/construct/publish/deliver | unchanged; dtype joins pre-commit validation |
| Generator synchronization | none — generators have no dtype |
| Concurrent reads | unchanged; the dtype tag is immutable after construction, so concurrent readers see one value |
| Concurrent mutation | still forbidden, exactly as before; Phase I adds no thread-safety claim |
| Exactly-once release | unchanged, and proved per dtype in the lifecycle tests |

The honest scoping of Phase G is carried forward verbatim: transactions
are **per module**, one whole training step is not globally
transactional, ordinary training mutation does not take the process-wide
state-replacement lock, and thread-safe concurrent training snapshots are
not offered. Participating state-replacement operations serialize with
respect to each other in the universal lock order (the private
process-wide guard first, then every unique generator lock in global
`id()` order, never the reverse).

---

## 22. Performance preservation

**Float64 performance must not regress.** Phase H's measured results are
the baseline Phase I inherits and must protect.

Requirements:

- **No dtype string parsing inside a hot loop**, or inside any loop, or
  anywhere in C++ at all.
- **One narrow dispatch per operation**, at ABI entry (§8.1).
- **Templates, not runtime polymorphism**: compile-time typed kernels,
  explicitly instantiated for `float` and `double`, with no added
  indirection.
- **Contiguous fast paths preserved** — the flat, index-free kernels keep
  their shape at both dtypes.
- **Broadcasting traversal preserved** — the zero-stride read model and
  the H8 collapsed plans are `int64` metadata and are untouched.
- **Optimized matmul access preserved** — H2's `i`-`k`-`j` row sweep and
  its predicate, at both dtypes.
- **Optimized reduction structure preserved** — H6's contiguous-block
  factorization and its predicate, with per-output order intact.
- **Optimized convolution preserved** — H9's three traversals, their
  predicates, and their allocation behavior; conv2d allocates no
  workspace today and must not start.
- **Optimizer allocation reductions preserved** — H4's per-step scalar
  holder and its allocation counts (§15.4).
- **Bit-preserving copy paths generalized without regressing float64** —
  H5's `contiguous_copy` primitive and `tf::copy_prefers_contiguous`.
- **H1's uninitialized-output audit re-derived per dtype**, with a poison
  pattern for each width, and every negative control kept.
- **No allocation count or memory peak may rise** for a float64 workload.

**Measurement discipline**, from §9 of `CLAUDE.md` and the hard-won
lessons of Phase H: correctness gated before timing; **no speed asserted
anywhere**, no threshold, no budget, no CI job that fails on a number,
**no result file of any kind**; `native_only` cases publish no ratio;
alternating pre/post rounds in separate subprocesses; every case proved
bit-identical before either side is timed; low round counts lie, so
21–25 rounds and never a 7–9-round figure quoted as evidence; the
machine's control band stated; whole-translation-unit code-layout effects
published rather than chased; and the ~7–12 µs Python-plus-ctypes floor
below ~1,000 elements named as an architectural floor.

**Phase I benchmarks float32 and float64 separately**, never as a ratio
of one to the other, because a float32/float64 speed ratio is a property
of memory bandwidth on one machine and not a project claim. The expected
and honest shape of the result is that float32 helps where the work is
bandwidth-bound and is neutral where it is not; whatever is measured is
what gets published, including neutral and negative findings.

**I0 modifies no benchmark code.** Benchmark work belongs to I10.

---

## 23. Build and platform requirements

Phase-I closure requires, with zero project compiler, linker, and CMake
warnings in each configuration:

- **Windows Release** — out-of-source, outside the repository, full CTest
  suite green;
- **Windows Debug** — same, written elsewhere so the active runtime stays
  the Release DLL;
- **Linux CI-equivalent** — `uv sync --group cpp`, `uv run python
  cpp/build.py`, the smoke check, the quick benchmark, then `uv run
  pytest`;
- **Linux Clang ASan and UBSan** — instrumentation proved present, the
  full native CTest suite, the native Python suites with **zero**
  diagnostics, and a **negative control** proving the instrumentation can
  fail;
- **LeakSanitizer lifecycle** — native live storage returns **exactly**
  to baseline, remaining process-exit allocations contain **no
  TensorForge frame**, and **no suppression file is added**;
- an **identical production export inventory across every supported
  platform**: exactly 54 `tf_*` symbols in source and in each built
  library.

Additional Phase-I rules:

- `TF_SANITIZE` and `TF_BUILD_TESTS` remain the **only** build options. A
  third — including any `TF_DTYPE`-flavoured one — is a milestone
  decision this phase does not take.
- **No CUDA compiler requirement**, no CUDA toolkit, no `.cu` file.
- **No mandatory `-march` / `/arch` flag**, and no `-ffast-math`,
  `/fp:fast`, or `-funsafe-math`. float32 makes an architecture flag
  *more* tempting and it is *more* forbidden: it would change results
  per machine and destroy every determinism claim in §18.
- **No platform-specific dtype semantics.** `float` is IEEE-754 binary32
  and `double` is binary64 on every supported platform; a toolchain where
  that is not true is not supported, and the C++ side asserts
  `std::numeric_limits<float>::is_iec559` and the same for `double` at
  compile time.
- The C++ test executables that compile kernel sources directly must be
  extended to cover both instantiations; the CTest inventory grows from
  **17** by whatever the milestones actually add, and the number is
  reported rather than predicted.

---

## 24. ABI compatibility table

| Group | Symbols | Before Phase I | After Phase I | Compatibility |
|---|---|---|---|---|
| Untyped storage creators | `tf_storage_create`, `tf_storage_create_uninitialized` | float64, zero-init / uninit | **unchanged**; delegate to the shared body with `Dtype::Float64` | Fully compatible; same signature, same validation order, same messages, same failure convention |
| **New typed creators** | `tf_storage_create_typed`, `tf_storage_create_uninitialized_typed` | — | `(int64_t size, int32_t dtype_code)` → handle or null | **Additive.** 52 → **54** |
| Storage lifecycle / query | `tf_storage_destroy`, `tf_storage_size` | handle-only | **unchanged**; `size` still means elements | Fully compatible |
| Scalar mutators | `tf_storage_fill`, `tf_storage_scale` | `(handle, double)` | **signature unchanged**; scalar narrowed once to the element type (§7.4) | Fully compatible |
| Host transfer | `tf_storage_copy_from`, `tf_storage_copy_to`, `tf_storage_materialize` | handle + `double*` | handle + `void*`; element type from the handle | **Binary compatible**; source-level retype only (§7.3) |
| Handle-based compute | all 33 `tf_core_*` | float64 | dtype from the handle; one dispatch at entry | Fully compatible; no signature change |
| Raw float64 utilities | `tf_elementwise_*` (4), `tf_relu`, `tf_matmul`, `tf_matmul_tiled` | float64 | **float64, permanently**; `RAW_KERNEL_DTYPES == ("float64",)` at I2 | Fully compatible; limitation explicit and testable |
| Cross-entropy targets | the `const int64_t*` positions | int64 host metadata | **unchanged** | Fully compatible |
| Layout metadata | every `const int64_t*` shape/stride position | element counts | **unchanged** | Fully compatible |
| Error / introspection | `tf_last_error_code`, `tf_last_error_message`, `tf_clear_error` | — | **unchanged**; dtype errors use `TF_ERROR_INVALID` | Fully compatible |
| Test-only hook | `tf_test_arm_alloc_failure`, `tf_fault_injection_available` | — | **unchanged**; no second such hook is added | Fully compatible |
| Opaque handles | `void*` `Storage` | `{double*, int64_t}` | `{void*, int64_t, Dtype}` | Opaque to callers; no caller has ever inspected it |
| dtype codes | — | — | `0 = float64`, `1 = float32`, **frozen** | New, additive, stable |
| Windows visibility | `__declspec(dllexport)` on `TF_EXPORT` | 52 | **54**, and nothing else | Unchanged mechanism |
| Linux visibility | hidden default + `__attribute__((visibility("default")))` | 52 | **54**, and nothing else | Unchanged mechanism |

**Why two new creators are sufficient**: §6.5. **Why per-operation
float32 exports are rejected**: §6.5. **Why the source inventory and the
built export table must agree**: they already do, and three separate
tests check it; those tests move from 52 to 54 at milestone I1 and never
again during the phase.

---

## 25. Public Python compatibility

### 25.1 Canonical form

The dtype argument accepts exactly:

- `"float64"` — the default meaning of `None`;
- `"float32"` — after the milestone that enables it;
- `None` — meaning `"float64"`.

**No aliases.** Not `np.float32`, not `numpy.dtype("float32")`, not
`"f4"`, not `"single"`, not `"double"`, not `float`, not `32`. The
repository's convention since v1.21 is a canonical validated string, and
a permissive front door is exactly how a "dtype" silently becomes a
NumPy-coupled type object.

### 25.2 Validation, defaults, and errors

- Validation is `normalize_dtype(dtype)`: `None` → `"float64"`; a
  non-string → **`TypeError`** naming the value; a string outside
  `SUPPORTED_DTYPES` → **`ValueError`** naming the value and the
  supported set. Pure Python, never touching the compiled library.
- **The default is `"float64"` at every constructor, factory, module, and
  parameter, forever.** Existing user code that omits `dtype` behaves
  byte-identically.
- The dtype is **never inferred** from an input array (§9.4).

### 25.3 Exposure

- `dtype` is a **read-only property** on `NativeStorage`,
  `NativeTensorCore`, `NativeTensor`, and `NativeParameter`, readable
  after `close()`, exactly as today. There is no setter and no in-place
  dtype change.
- Modules that own parameters or buffers expose a read-only `dtype`
  property reporting the dtype they were constructed with.
- `repr()` gains the dtype **only where it is currently absent and
  genuinely useful**, and any change to a repr is called out in its
  milestone rather than slipped in — reprs are read by tests and by
  humans reproducing bugs.
- `backend_info()` reports `supported_dtypes` from the registry as it
  does today. The flat `"dtype": "float64"` key is a *default* statement,
  not a capability statement; the milestone that enables float32 must
  decide explicitly whether it keeps naming the default or is removed,
  and must not leave it silently wrong.

### 25.4 State loading validates dtype

`load_state_dict`, the optimizer loaders, and `load_native_checkpoint`
each validate dtype per entry against the live destination and raise
`ValueError` naming the key and both dtypes on a mismatch — before
staging is committed, and without casting.

---

## 26. Testing strategy

Layered, with the appropriate comparison stated per layer. **Exact** means
`==` on values; **bitwise** means comparison of raw IEEE-754 bit patterns
through a `uint32`/`uint64` view; **tolerance** means `np.allclose` with
a stated, dtype-appropriate tolerance.

| Layer | What is tested | Comparison |
|---|---|---|
| dtype codes | round trip code ↔ dtype, unknown codes rejected, codes frozen at 0/1 | exact |
| item size | one authority, 8 and 4, no second table anywhere in the tree | exact |
| storage round trip | create/copy_from/to_numpy at both dtypes, all-zero default | **bitwise** |
| zero-size storage | rejected identically by all four creators | exact (message) |
| overflow | `numel × itemsize` rejected before allocation | exact |
| allocation failure | injected failure at both dtypes; live storage at baseline | exact |
| views | reshape/transpose/T/narrow preserve dtype; no view has its own tag | exact |
| non-contiguous tensors | strided/offset/negative-stride reads at both dtypes | **bitwise** |
| bit-preserving copies | `-0.0`, ±inf, quiet and signalling NaN with full payloads | **bitwise** |
| elementwise | against NumPy at the same dtype; the one-NaN rule; two-NaN carve-out | **bitwise** (finite, ≤1 NaN) |
| broadcasting | zero-stride model at both dtypes; plan path vs retained odometer | **bitwise** |
| reductions | per-output order; signed zeros; block path vs retained odometer | **bitwise** |
| matmul | row sweep vs retained generic; accumulation order; NaN class | **bitwise** (non-NaN) |
| exp / log | against NumPy at the same dtype | **one ULP** |
| autograd | dtype of every gradient and temporary; mixed-dtype rejection | exact |
| gradcheck | float32 finite differences with a binary32-appropriate step and tolerance | **tolerance** |
| CNN kernels | three directions, both traversals, both dtypes | **bitwise** vs the retained path |
| pooling | forward/backward; winner buffer stays float64; plane bound unchanged | **bitwise** / exact |
| normalization | LayerNorm and both BatchNorm shapes; buffer dtype; the two-buffer transaction | **bitwise** / exact |
| stable math | softmax, log-softmax; the maximum shift at float32; no NaN the float64 path does not produce | **tolerance** vs NumPy float32 |
| classification | fused cross-entropy forward/backward; int64 targets unchanged | **tolerance** vs NumPy float32 |
| modules | dtype-aware constructors; float64 default; seed→value relation across dtypes | exact |
| persistent buffers | dtype, identity, aliasing, state-dict round trip | exact |
| Dropout | same drop pattern across dtypes for one key; mask values; call accounting | **bitwise** / exact |
| optimizer state | moment dtype; validation passes; atomicity under injected failure | **bitwise** / exact |
| checkpoint v1/v2 | float64-only; float32 declaration or payload rejected; never guessed | exact |
| checkpoint v3 | every entry's dtype; moment entries; round trip | **bitwise** |
| malformed checkpoints | the corruption matrix extended with dtype cases | exact (message) |
| exact resume | both dtypes, independently, per §18.2 | **exact / bitwise** |
| ABI inventories | 54 in source and in the built library; declared == exported | exact |
| registry boundaries | `SUPPORTED_DTYPES`, `UNSUPPORTED`, `RAW_KERNEL_DTYPES` | exact |
| stable/native isolation | the subprocess import test; `stable_framework_integration is False` | exact |
| failure cleanup | every row of §20 | exact |
| concurrent state | the existing serialization tests, unchanged, at both dtypes | exact |
| cross-platform | Windows Release/Debug, Linux, per §23 | build + CTest green |
| sanitizers | ASan/UBSan/LSan per §23, with the negative control | zero diagnostics |
| benchmarks | float32 and float64 characterized separately, correctness-gated | **no assertion** |
| closure docs | every status surface accurate; no result file; no committed number | exact |

**Float64 regression is the highest-priority test surface of the phase**:
every pre-Phase-I test must keep passing **unchanged**, and a milestone
that needs to edit one has changed float64 behavior and must stop and say
so. Loosening a test to accommodate float32 is forbidden; float32 gets
its own tests.

---

## 27. Rollout discipline

### 27.1 Four distinct states, kept distinct

| State | Meaning |
|---|---|
| **Internal implementation availability** | The code path exists and is reachable from Python. Says nothing about support. |
| **Tested subsystem availability** | That subsystem's float32 behavior is proved by test at its own layer. |
| **Public supported status** | `"float32"` has left `UNSUPPORTED` and joined `SUPPORTED_DTYPES`. A promise. |
| **Phase-I closure** | The whole ladder has landed, both resume proofs pass, and every platform gate is green. |

### 27.2 The rule

**float32 is not publicly declared supported until the complete native
training stack, optimizer state, checkpoint v3, exact resume, hardening,
and platform validation exist.** Intermediate milestones legitimately
implement internal float32 pieces while the public registry still lists
float32 as unsupported — exactly the pattern Phase G used for `dropout`,
which stayed in `UNSUPPORTED` from G3 through G9 while the operation and
the module both existed, and left only at the G10 closure.

### 27.3 The exact milestone

**The public registry changes at I9, and at no other milestone.** At I9,
and only after integrated float32 training and the exact float32
resume proof both pass:

```python
SUPPORTED_DTYPES = ("float64", "float32")
UNSUPPORTED      = ("cuda", "amp")
```

Before I9, `SUPPORTED_DTYPES == ("float64",)` and `UNSUPPORTED ==
("float32", "cuda", "amp")` — which means that during I1–I8 the internal
float32 paths are exercised by tests that construct storage and tensors
through the private/typed entry points rather than by passing
`dtype="float32"` through `normalize_dtype`. Each such milestone states
in its exit gate exactly which entry point its tests use and why the
public boundary has not moved.

Rejected alternative: moving `"float32"` out of `UNSUPPORTED` early "so
tests can use the public API". That would publish a capability that does
not yet survive a checkpoint, which is the precise mistake Phase G's
`dropout` handling was designed to avoid.

---

## 28. Non-goals restated as boundaries

Nothing in Phase I may:

- add a dtype other than float32 and float64;
- add a device, or anything device-shaped;
- add casting, promotion, or mixed-dtype arithmetic;
- add `map_location` or any load-time conversion;
- add an integer, boolean, or complex tensor;
- add a public performance control, path selector, threshold setter,
  dispatch tracer, profiling counter, or "which path ran" query;
- add an environment variable that changes any behavior;
- add a build option;
- add a dependency;
- add SIMD, threading, OpenMP, BLAS, oneDNN, Eigen, im2col, general
  operator fusion, fast-math, cache blocking, a memory pool, or a scratch
  arena;
- add a second production poison, profiling, or allocation-content hook;
- weaken a validation, an error type, or an error message in the name of
  dtype support or speed;
- couple the stable line to the native line;
- add a timing assertion, a committed benchmark number, or a result file;
- claim CUDA, AMP, float16, bfloat16, integer tensors, data loaders, or
  distributed training exist.

---

## 29. Milestone ladder — I0 through I11

Each milestone lists its entry condition, scope, tests, documentation,
invariants, exclusions, and exit gate. The ladder is **evidence-driven**:
if repository reality contradicts a milestone's premise, that milestone
is narrowed, reordered, or dropped, and the revision is recorded here
rather than rewritten away — the precedent Phase H set three times.

### I0 — Repository reconciliation and dtype architecture contract

- **Entry:** Phase H complete and merged; working tree clean; 52 exports;
  checkpoint version 2; 6415 tests passing.
- **Scope:** inspect and record current reality (§2); create this
  contract; define the dtype model, storage model, ABI strategy,
  raw-buffer boundary, dispatch, no-cast rules, accumulation, autograd,
  module, buffer, RNG, optimizer, checkpoint, determinism, isolation,
  failure, ownership, performance, platform, compatibility, testing,
  rollout, and milestone boundaries.
- **Tests:** durable contract guardrails asserting the document's
  load-bearing decisions and the unchanged runtime boundaries (§30).
- **Docs:** this file; Phase-I status on the roadmap, project summary,
  support matrix, architecture, backend experiments, README, and
  `CLAUDE.md`; an in-progress planning entry in the release history.
- **Invariants:** no production Python, C++, ABI, build, CI, example, or
  benchmark change.
- **Exclusions:** any float32 runtime behavior.
- **Exit gate:** registries unchanged; checkpoint constants unchanged; 52
  exports; Phase H still complete; full suite green with only the new
  contract tests added.
- **Commit message:** `Define Phase I dtype generalization architecture`

### I1 — Internal dtype model and dtype-tagged storage foundation

- **Entry:** I0 merged.
- **Scope:** the C++ `TfDtype` enum, `dtype_from_code`,
  `dtype_item_size`, `dtype_name`; the Python code/item-size/NumPy
  tables; the dtype-tagged `Storage` with checked byte sizing and byte
  allocation; `tf_storage_create_typed` and
  `tf_storage_create_uninitialized_typed`; the untyped creators become
  compatibility wrappers.
- **Tests:** dtype-code round trip and rejection; item-size authority
  with a tree-wide check that no second size table exists; typed creation
  at both dtypes; zero-size and overflow rejection; injected allocation
  failure; all-zero-byte default equals `+0.0` at both widths; the
  untyped creators' messages and behavior byte-identical to before; **new
  export counts of 54** in all three inventory tests; a new dependency-free
  CTest for the typed creators.
- **Docs:** support matrix and backend experiments record the two new
  symbols and 52 → 54.
- **Invariants:** float64 behavior bit-identical; `SUPPORTED_DTYPES`
  still `("float64",)`.
- **Exclusions:** no operation is dtype-general yet.
- **Exit gate:** 54 exports in source and in the built library on every
  platform built; full suite green.
- **Commit message:** `Add dtype-tagged native storage foundation`

#### I1 as delivered

Landed as specified, with four implementation decisions recorded here —
three because the contract left the judgement open, and one (item 2)
because the first implementation got it wrong and the correction is worth
inheriting rather than rediscovering:

1. **The dtype model is defined inline in `tf_internal.h`** rather than in
   a new translation unit. The signatures are exactly §3.1's. A new
   `.cpp` would have had to join all seventeen existing CTest target
   source lists, and inline definitions make "one authority" structural
   rather than conventional — there is one definition, in the header every
   compute unit already includes. `VISIBILITY_INLINES_HIDDEN` keeps them
   out of the ABI.
2. **Storage owns a genuine typed array, and it took two corrections to
   get there.** Both wrong turns are recorded because each is a natural
   thing to reach for and neither is visibly wrong at the call site.

   The first implementation allocated `unsigned char[bytes]`, held it
   behind `void*`, and let the typed accessors reinterpret it — which
   acquires storage but creates no `float` or `double` at all, and is only
   well defined under C++20's implicit-object-creation rule that C++17
   does not have.

   The second replaced that with raw `::operator new` plus a per-element
   placement-new loop. That *does* begin `count` floating-point lifetimes,
   which fixes the first problem — but as `count` **separate scalar
   objects**, and the kernels index their operands across the whole
   allocation. Under [expr.add]/4 pointer arithmetic is defined only
   within one array object, and adjacent scalars do not become one array
   because their storage is contiguous, so `data[i]` past the first was
   still leaving its array object.

   The shipped model is the ordinary one: a real array new-expression
   (`new T[count]()` / `new T[count]`) chosen by one dtype dispatch into a
   templated body, owned across the metadata allocation by
   `std::unique_ptr<T[]>`, type-erased into `void*` only after the array
   exists, and released by one central dtype-matched `delete[]` switch
   (§4.1). The immutable dtype tag selects the array type, the typed
   accessor, and the deleter — so none of the three can disagree.
3. **Every float64 kernel reaches its buffer through one accessor pair**,
   `tf::storage_f64`, and the float32 rejection is one shared
   `tf::require_float64` call at each export boundary — 31 `tf_core_*`
   exports plus `fill`, `scale`, `copy_from`, `copy_to`, and
   `materialize`. No compute kernel was templated or otherwise
   generalized; that is I3 onward. The internal kernels keep their
   `double*` / `const double*` signatures unchanged.
4. **The four unguarded storage primitives stay unguarded and unhooked.**
   `tf_storage_fill`, `tf_storage_scale`, `tf_storage_copy_from`, and
   `tf_storage_copy_to` reject a float32 handle and record
   `TF_ERROR_INVALID`, but they do **not** clear the error slot on entry
   and do **not** join `_CHECKED_KERNELS`. H7 deliberately kept them
   hookless so they cost one native call rather than two, and no
   production Python path can reach them with a float32 handle while the
   public registry rejects the dtype. A direct C ABI caller — the only
   way to obtain one — reads the rejection through `tf_last_error_code`,
   which is the ordinary convention for an unhooked export. A failed call
   leaves the code in the slot and the next guarded call clears it on
   entry, so it can never be misattributed.

One **new failure mode** arrived exactly as §4.1 predicted: a byte-count
overflow is now `TF_ERROR_INVALID`, rejected by arithmetic before any
allocator is asked, where the implicit `new double[count]` sizing it
replaced could only discover the problem by throwing (`TF_ERROR_ALLOC` or
`TF_ERROR_RUNTIME`). The C++ storage-allocation CTest was advanced to
assert **both** modes separately — an overflowing count and a
representable-but-unsatisfiable one — rather than being loosened to accept
either.

The CTest inventory moved **17 → 18** with `test_dtype_storage`, which
links every kernel translation unit because the float32-rejection proof
has to cover each one's own validation front end.

### I2 — Typed array transfer, views, and materialization

- **Entry:** I1 merged.
- **Scope:** float32/float64 ingress and egress; the three transfer
  exports become dtype-general (§7.3); `_CHECKED_F32_ARRAY`; contiguous
  and non-contiguous transfer; materialization; views carry dtype through
  unchanged; bit-preserving copy paths generalized; failure cleanup;
  `RAW_KERNEL_DTYPES` introduced and reported by `backend_info()`.
- **Tests:** round trips at both dtypes proved **bitwise** including
  `-0.0`, ±inf, and full NaN payloads; strided/offset/negative-stride
  materialization; view dtype invariance; wrong-dtype host buffers
  rejected by `ndpointer`; the raw-kernel float64 boundary asserted; H5's
  copy contract restated per dtype.
- **Docs:** the raw-buffer/handle division and the registry.
- **Invariants:** float64 bit-identical; export count stays 54.
- **Exclusions:** no compute kernel is dtype-general yet.
- **Exit gate:** both dtypes move data in and out bit-exactly; full suite
  green.
- **Commit message:** `Add typed native array transfer boundaries`

#### I2 as delivered

Landed as specified, with six implementation decisions recorded here
because the contract left each judgement open.

1. **The identity copy was generalized; nothing else that computes was.**
   `tf_core_contiguous_copy` is the one handle-based compute-shaped export
   I2 touched, and it was touched because it is not really arithmetic: it
   is the runtime's **value-transfer primitive** (H5), the
   storage-to-storage twin of `tf_storage_materialize`, and what every
   Policy-B copy-then-compute path, `NativeFlatten`, `NativeParameter`
   construction, and the differentiable `contiguous_copy` are built on.
   Every other `tf_core_*` export keeps `tf::require_float64`.

   `tf_storage_fill` and `tf_storage_scale` were **deliberately left
   behind**, and the temptation to take them was real — the dispatch is two
   lines. They perform numerical assignment and arithmetic on the buffer,
   and a scalar narrowed once into a float32 buffer (§7.4) is a decision
   with its own numerical statement that is not this milestone's to make.

2. **Templates, not a twin.** The H8 unary traversals (`tf::unary_row`,
   `tf::unary_plan_walk`) and the retained generic odometer (`core_unary`)
   gained a scalar type parameter that is **deduced from their pointer
   arguments**, so every pre-Phase-I call site compiles unchanged and
   `T = double` is the pre-I2 code statement for statement. The alternative
   — a structurally identical float32 twin of the three-tier copy dispatch
   — was rejected as exactly the duplicated kernel §8.2 forbids: two
   traversals that must stay identical forever, with nothing structural
   keeping them so.

   Only `tf::IdentityOp::apply` became a member template. That is
   load-bearing rather than tidy: a fixed `double apply(double)` reached
   with a float operand would convert float → double → float around the
   "copy", and while that round trip is exact for every finite value and
   every quiet NaN payload, it **quiets a signalling NaN** — breaking the
   value-transfer contract in precisely the case the contract exists for.
   The other functors stay `double`-only until I3.

3. **`memcpy` was not introduced, and the transfers stay element
   assignments.** A byte copy is the obvious way to make a contiguous
   transfer representation-preserving, and §4.3 forbids it: byte arithmetic
   outside the allocation boundary is what this phase keeps out of the
   kernels, and it is what lets the whole layout and bounds-checking
   apparatus carry over unexamined. The contract's own argument is used
   instead — a same-type assignment performs no arithmetic, so it has no
   operand roles to choose between and nothing to round (§10.3) — and it is
   **proved by test rather than asserted**, at both widths, over seventeen
   IEEE-754 classes per dtype including both signed zeros, both
   infinities, subnormals, quiet NaN payloads, and signalling NaNs of both
   signs, compared as raw `uint32`/`uint64` bit patterns and never by value.

   Scope, stated honestly: that proof is a measurement on the toolchains
   validated here (MSVC x64, GCC/Clang x86-64), not a language guarantee.
   C++ does not promise that copying a signalling NaN leaves it signalling;
   an x87 code path would quiet it. TensorForge builds x86-64 with SSE2,
   where the copy is a register move, and the CTest and the Python suite
   both fail loudly if that ever stops being true.

4. **The per-dtype `ndpointer` check moved from the argtypes slot to the
   call site**, because one slot cannot describe two dtypes and the choice
   must be made from a runtime value. It is the same check, run by the same
   function, selected from the storage's tag — see §7.3 for the full
   reasoning and for the inventory numbers it moves.

5. **Internal float32 construction is three private constructors and one
   private flag**, not a public bypass: `NativeStorage._typed`,
   `NativeStorage._typed_from_array`, `NativeTensorCore._typed_from_array`,
   and a keyword-only `_trusted_dtype` on `NativeStorage.__init__` that
   validates against `_DTYPE_CODES` instead of `SUPPORTED_DTYPES`. The
   private H1 allocators (`NativeStorage._uninitialized` and
   `NativeTensorCore._uninitialized`) inherit that trust, because every one
   of their call sites passes `dtype=<operand>.dtype` — a canonical tag read
   off a storage that was validated when it was created — and an
   operation's freshly allocated output must be able to match its operand's
   dtype without asking permission the operand already has.

   One consequence had to be repaired rather than accepted:
   `NativeTensorCore.full` is a **public** constructor that reaches the
   private allocator, and `tf_storage_fill` is float64-only *and unhooked*,
   so an unvalidated `dtype="float32"` would have allocated storage, been
   silently rejected by an export that records into the error slot without
   raising, and returned an uninitialized tensor. `full` therefore calls
   `normalize_dtype` explicitly, restoring exactly the rejection it always
   had. Public behavior is unchanged at every constructor.

6. **`copy_from` keeps converting; the C ABI never does.** The two are
   different boundaries and I2 keeps them different.
   `NativeStorage.copy_from` and `from_array` are the explicit
   host-to-native conversion boundary (§9.4) and have converted since v0.8;
   they now convert to *the storage's* dtype instead of unconditionally to
   float64, and the dtype is still never inferred from the input. The raw
   transfer boundary beneath them converts nothing: a host buffer whose
   element type disagrees with the storage is rejected with `TypeError`
   before the native call is made.

The CTest inventory moved **18 → 19** with `test_typed_transfer`, which
compiles `storage.cpp` and `elementwise.cpp` directly so it reaches
`tf::copy_prefers_contiguous` beside the four generalized exports, and
proves the contract by bit pattern at both dtypes over thirteen view
layouts — scalar, 1-D, non-unit stride, reversed, 2-D contiguous,
transposed, narrowed-with-offset, both broadcast (stride-0) forms, unit
extent, and two rank-3 chains.

Public capability did not move: float64 CPU only, `float32` still in
`UNSUPPORTED`, checkpoint version 2 with (1, 2) accepted, and **54**
exports — I2 added none.

### I3 — Elementwise, broadcast, and unary dtype execution

- **Entry:** I2 merged.
- **Scope:** `add`, `subtract`, `multiply`, `relu`, `relu_backward`,
  `sqrt`, `reciprocal`, `exp`, `log`, and the identity/copy map at both
  dtypes; the H8 templates gain their scalar parameter; contiguous fast
  paths and the retained odometers instantiated for both; one dispatch
  per export; mixed-dtype rejection reachable and tested.
- **Tests:** per-family numerical contracts restated for float32; plan
  path versus retained odometer bit-identical per dtype; broadcasting;
  `exp`/`log` at one ULP; every mixed-dtype rejection site; CTest
  extended.
- **Docs:** the numerical contract table per dtype.
- **Invariants:** float64 bit-identical; no new export.
- **Exit gate:** full suite green; H8's float64 measurements re-checked
  as neutral.
- **Commit message:** `Generalize native elementwise dtype execution`

### I4 — Reductions, matmul, views, and core autograd

- **Entry:** I3 merged.
- **Scope:** `sum`, `mean`, `matmul`, `narrow_backward`, and broadcast
  backward at both dtypes; H2's and H6's optimized paths and predicates
  instantiated for both; the accumulation policy of §10 enforced; core
  autograd dtype invariants.
- **Tests:** per-output order per dtype; signed zeros as bit patterns;
  matmul's four-part contract restated for float32; float32 finite
  differences with a stated step and tolerance; float64 regression
  protection.
- **Docs:** accumulation policy as implemented.
- **Invariants:** float64 bit-identical; no new export.
- **Exit gate:** full suite green.
- **Commit message:** `Generalize native reductions and matmul dtypes`

### I5 — CNN and pooling dtype support

- **Entry:** I4 merged.
- **Scope:** all three Conv2d directions and both MaxPool2d directions at
  both dtypes; H9's traversals and predicates instantiated for both; **the
  winner buffer fixed at float64** per §13.3, with the plane bound
  unchanged.
- **Tests:** conv2d's per-destination order contract restated for
  float32; the optimized-versus-retained comparison per dtype; the
  winner buffer proved float64 at a float32 pool and the `2**53` bound
  proved unchanged; no integer tensor API introduced.
- **Docs:** the winner-buffer decision and its rationale.
- **Invariants:** float64 bit-identical; no new export; no workspace.
- **Exit gate:** full suite green.
- **Commit message:** `Add float32 native CNN kernel support`

### I6 — Stable math and classification dtype support

- **Entry:** I5 merged.
- **Scope:** softmax, log-softmax, and fused cross-entropy at both
  dtypes; the maximum shift and log-sum-exp computed at the element type;
  saved probabilities carry the graph dtype; targets stay int64 host
  metadata.
- **Tests:** float32 stability at magnitudes where a naive form would
  overflow or underflow; no NaN or infinity the float64 path does not
  also produce for the same finite input; the saved-probability backward
  contract per dtype; the int64 target boundary unchanged.
- **Docs:** the float32 stability statement, honestly scoped.
- **Invariants:** float64 bit-identical; no new export.
- **Exit gate:** full suite green.
- **Commit message:** `Add float32 native classification support`

### I7 — Modules, parameters, buffers, initialization, and Dropout

- **Entry:** I6 merged.
- **Scope:** dtype-aware constructors with float64 defaults on
  `NativeParameter`, `NativeLinear`, `NativeConv2d`, `NativeLayerNorm`,
  and both BatchNorm shapes; initialization per §12.3; persistent buffers
  and saved statistics at the module dtype; Dropout output and mask
  dtype; the generator algorithm unchanged; `state_dict` dtype
  validation.
- **Tests:** the seed→value relation across dtypes; buffer dtype,
  identity, and aliasing; the two-buffer transaction with a dtype
  validation; the same drop pattern for one key across dtypes; call
  accounting on failure, in eval, and at `p == 0`; container modules
  raising on a mismatched child.
- **Docs:** the constructor surface and the default rule.
- **Invariants:** float64 behavior and values byte-identical; float32
  still publicly unsupported.
- **Exit gate:** full suite green.
- **Commit message:** `Integrate float32 native modules and buffers`

### I8 — Optimizer state and checkpoint version 3

- **Entry:** I7 merged.
- **Scope:** float32 `NativeSGD` and `NativeAdam`; moments at the
  parameter dtype; §15.3's scalar-coefficient statement resolved with
  evidence; checkpoint **version 3** per §16 and §17; accepted versions
  become `(1, 2, 3)`; v1/v2 strict float64 compatibility; transactional
  validation; no casting.
- **Tests:** moment dtype and validation passes; atomicity under injected
  failure at both dtypes; v3 round trip proved **bitwise**; v1/v2
  float64-only and never guessed to be float32; the corruption matrix
  extended with dtype cases; alias and identity preservation unchanged.
- **Docs:** the format-version history and the v3 schema.
- **Invariants:** the format **name** unchanged; float64 checkpoints
  written before Phase I still load; float32 still publicly unsupported.
- **Exit gate:** `CHECKPOINT_VERSION == 3`,
  `SUPPORTED_CHECKPOINT_VERSIONS == (1, 2, 3)`; full suite green.
- **Commit message:** `Add dtype-aware native checkpoint version 3`

### I9 — Public float32 integration and exact-resume proof

- **Entry:** I8 merged.
- **Scope:** the integrated float32 training example over the §18.1
  network; the exact interrupted/resumed equality proof; **the public
  registry change** (§27.3); the Dropout next-mask proof; optimizer,
  buffer, generator, and alias restoration; a float64 integrated
  regression run beside it.
- **Tests:** every item of §18.2 at float32, by exact equality; the
  float64 integrated proof unchanged; the registry values asserted in
  their new form; `UNSUPPORTED` proved to still reject `cuda`, `amp`, and
  every dtype outside the two.
- **Docs:** the support matrix moves float32 out of the unsupported
  section; every status surface updated in the same milestone.
- **Invariants:** no ABI change; 54 exports; no new dependency.
- **Exit gate:** both resume proofs pass; full suite green.
- **Commit message:** `Prove deterministic native float32 training resume`

### I10 — Cross-cutting hardening and benchmarking

- **Entry:** I9 merged.
- **Scope:** mixed-dtype rejection across every layer of §9.2; the
  malformed-checkpoint matrix; allocation and failure cleanup at both
  dtypes; the four saved-resource families coexisting in one float32
  graph; concurrency and lifetime checks; stable/native isolation; ABI
  and export inventories; float32 and float64 benchmark characterization.
- **Tests:** the adversarial matrix; live-storage baselines across
  repeated lifecycle loops; no timing gate anywhere.
- **Docs:** measured results published with spread, control band, and
  every neutral and negative finding.
- **Invariants:** no capability added; no result file written; no CI
  timing threshold.
- **Exit gate:** full suite green; benchmarks correctness-gated and
  assertion-free.
- **Commit message:** `Harden and benchmark native float32 support`

### I11 — Cross-platform validation and Phase-I closure

- **Entry:** I10 merged.
- **Scope:** Windows Release and Debug; Linux CI-equivalent; Clang
  ASan/UBSan with the negative control; LeakSanitizer lifecycle; all
  examples; both exact-resume proofs; export inventory; final registry
  and checkpoint truth; documentation reconciliation across the support
  matrix, roadmap, architecture, project summary, release history,
  README, and `CLAUDE.md`; closure tests.
- **Tests:** a Phase-I closure guardrail module in the shape of
  `tests/test_native_phase_h_closure.py`.
- **Invariants:** exactly **54** exports on every platform; no generated
  artifact tracked; no committed benchmark number.
- **Exit gate:** the full matrix green and recorded with observed
  results.
- **Commit message:** `Complete Phase I native float32 support`

### 29.1 Ladder adjustments made at I0

**None.** The recommended I0–I11 structure survived the repository
inspection unchanged. Three findings were absorbed *into* milestones
rather than reshaping the ladder, and are recorded here so they are not
mistaken for late discoveries:

1. the three transfer exports need a source-level retype rather than new
   symbols, which keeps the two-new-export plan intact (§7.3, I2);
2. the MaxPool2d winner buffer must stay float64, which is an I5 decision
   rather than a new milestone (§13.3);
3. `RAW_KERNEL_DTYPES` lands at I2 rather than I0, because a
   contract-only registry would carry no information while every dtype is
   float64 (§7.2).

---

## 30. What I0's guardrails assert

`tests/test_native_phase_i.py` is the durable contract module. It asserts
**values and structure**, not wording, so ordinary prose improvements do
not require rewriting it, and it derives its premises from the live
registry, the live source, and real files wherever possible.

It pins that this document exists and is linked; that it records the
current float64-only reality and the float32-and-float64 target; that it
plans **exactly two** new ABI exports, named, with a final count of
**54**; that existing exports remain compatible and per-operation float32
duplication is explicitly rejected; that storage is dtype-tagged with one
central item-size authority; that shapes, strides, and offsets stay
element-measured; that there is no casting and no promotion; that
mixed-dtype is rejected across the named layers; that float32 accumulates
in float32; that checkpoint version 3 is planned with versions 1, 2, and
3 accepted and versions 1 and 2 defined as float64-only; that
stable/native isolation holds; that CUDA, AMP, integer tensors, and data
loading stay outside the phase; that the I0–I11 ladder appears once each
in order; that the public registry changes at **I9**; that exact resume
is required separately per dtype; and that Phase-H performance
preservation is required.

It also pins the **unchanged** runtime: `SUPPORTED_DTYPES ==
("float64",)`, `SUPPORTED_DEVICES == ("cpu",)`, `UNSUPPORTED ==
("float32", "cuda", "amp")`, `_FORMAT_VERSION == 2`,
`_SUPPORTED_FORMAT_VERSIONS == (1, 2)`, **52** exports in source and in
the built library, Phase H still complete, and no I0 change to any
production Python module, C++ source or header, build file, CI workflow,
example, or benchmark.
