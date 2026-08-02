# Native CNN architecture design (Phase D contract)

This is the **design-and-contract** document for the experimental native
C++ CPU line's convolutional stack — **Phase D**. It is a milestone-zero
(D0) deliverable: it locks the architecture, layouts, argument contracts,
ownership rules, C ABI shape, source organization, testing strategy, and
milestone sequence **before any numerical CNN code is written**.

This document defines the **complete Phase-D contract** — layouts,
argument contracts, ownership rules, C ABI shape, source organization,
testing strategy, and the full milestone sequence. **Implementation
status is not restated in this introduction; it is tracked per milestone
in the milestone map (§18) and the completion criteria (§19)**, and the
backend capability registry (`tensorforge.backends.cpp.UNSUPPORTED`,
mirrored in [native_support_matrix.md](native_support_matrix.md)) is the
**single source of truth** for exactly which surfaces are live at any
moment. Phase D began as a milestone-zero (D0) deliverable that locked
this architecture *before any numerical CNN code was written*; **every
D0–D12 milestone has since landed** (each marked in §18) and the
completion criteria in §19 are all met, so **Phase D is complete**. This
document therefore reads as the phase's contract *and* its record: where
implementation refined a D0 decision, the refinement is stated at the
milestone that made it.

Two later phases refined the Phase-D kernels without changing a Phase-D
decision, and both are recorded in their own contracts rather than
restated here: **Phase H milestone H9**
([native_cpu_performance_design.md](native_cpu_performance_design.md)
§16.9) added an optimized traversal beside each retained Phase-D Conv2d
loop behind the unchanged exports, and **Phase I milestone I5**
([native_dtype_float32_design.md](native_dtype_float32_design.md) §29)
made the Conv2d and MaxPool2d kernels templates over the element type —
`T = double` is the Phase-D code statement for statement, the kernel
definitions now live in `tf_conv2d_internal.h` / `tf_pooling_internal.h`,
and the **winner buffer stays float64 at every value dtype** with the
§12 `2**53` exactness bound unchanged, exactly as §12 specifies it.

The stable Python framework (`tensorforge.nn.Conv2d`,
`tensorforge.nn.MaxPool2d`, `tensorforge.nn.Flatten`) is the **numerical
and public-behavior reference**. Where the native architecture must
differ (ownership, lifetime, the fused-primitive/autograd split), the
difference is stated and justified. No implementation code is copied from
any other framework.

Read alongside:
[native_tensor_wrapper_design.md](native_tensor_wrapper_design.md) (the
`NativeTensor` wrapper and ownership model),
[native_autograd_design.md](native_autograd_design.md) (the
Python-managed reverse-mode graph),
[native_abi_error_contract.md](native_abi_error_contract.md) (the
exception-safe C ABI status contract), and
[backend_experiments.md](backend_experiments.md) (the whole native line).

---

## 0. Invariants Phase D must preserve

Phase D changes nothing about these existing guarantees:

- Stable `Tensor` and `NativeTensor` remain **separate systems**; there is
  no implicit conversion or dispatch between them.
- Native autograd stays **Python-managed** at the `NativeTensor` layer;
  the `NativeTensorCore` runtime and the C++ kernels stay
  **autograd-unaware**.
- The native runtime targets **CPU float64** only.
- Native storage **ownership and lifetime are explicit** (`owns_core`,
  `close()`); owning cores free storage, borrowing views do not.
- Views retain **shape / strides / offset / storage** semantics.
- `NativeParameter` preserves **object identity**; value replacement
  increments its monotonic **value version**; graphs record expected
  versions and detect **stale graphs** through version checks.
- `state_dict()` returns **independent snapshots**; state and checkpoint
  loading are **atomic** (validate → stage → commit → rollback on
  failure); native checkpoints stay **pickle-free**; buffers stay
  **separate from parameters**.
- Failed operations must **not partially mutate** caller-visible state.
- Every fallible native export uses the existing **thread-local status /
  `errcheck` contract**; **no C++ exception crosses the C ABI**.
- Existing numerical behavior is **unchanged**.

---

## 1. Phase-D scope

### In scope

| # | Deliverable |
|---|---|
| 1 | `NativeFlatten` (composed from existing reshape/view autograd; no new kernel) |
| 2 | CPU Conv2d **forward** kernel |
| 3 | Conv2d **input**-gradient kernel |
| 4 | Conv2d **weight**-gradient kernel |
| 5 | Conv2d **bias**-gradient (reuses existing `sum` reductions; no new kernel) |
| 6 | `NativeTensor.conv2d(...)` autograd integration (a new fused primitive) |
| 7 | `NativeConv2d` module |
| 8 | CPU MaxPool2d **forward** kernel + winner-index contract |
| 9 | MaxPool2d **backward** (scatter) kernel |
| 10 | `NativeTensor.maxpool2d(...)` autograd integration |
| 11 | `NativeMaxPool2d` module |
| 12 | Deterministic native CNN **training + checkpoint-resume** proof |
| 13 | Phase-D documentation, tests, benchmarks, and completion guardrails |

### Explicitly excluded from Phase D

Dilation; groups / depthwise convolution; transposed convolution;
adaptive pooling; average pooling; global pooling; channels-last layout;
float32/float16/bfloat16; CUDA; mixed precision (AMP); BatchNorm;
Dropout; classification losses (unless already required by the approved
D11 proof); **im2col** performance optimization; BLAS-specific
convolution; threaded convolution; and any stable/native automatic
dispatch. These stay in the support matrix's unsupported/future section.

The first implementation of every kernel favors **correctness,
readability, sanitizer safety, and explicit indexing** over speed —
direct nested-loop convolution, not im2col+GEMM.

---

## 2. Tensor and kernel layouts

All CNN tensors are **row-major contiguous float64/cpu**, matching the
stable framework's conventions verified in `src/tensorforge/nn/conv.py`
and `src/tensorforge/nn/pool.py`.

| Tensor | Layout | Shape | Dim order |
|---|---|---|---|
| Conv2d / MaxPool2d input | **NCHW** | `(N, C, H, W)` | batch, channel, height, width |
| Conv2d / MaxPool2d output | **NCHW** | `(N, O, out_h, out_w)` (pool: `(N, C, …)`) | same |
| Conv2d weight | **OIHW** | `(out_channels, in_channels, kh, kw)` | out-ch, in-ch, kh, kw |
| Conv2d bias | 1-D | `(out_channels,)` | — |

This is verified against the stable Conv2d, whose weight is
`np.random.randn(out_channels, in_channels, kh, kw)` and whose forward
contraction is `einsum("nckl,ockl->no", window, weight)` — i.e. output
channel `o`, input channel `c`, kernel row `k`, kernel col `l`. So
`weight[o, c, k, l]` (OIHW) and input `x[n, c, h, w]` (NCHW) is the
established convention; Phase D adopts it unchanged.

**Cross-correlation, not flipped convolution.** The stable Conv2d
multiplies each window by the kernel **without flipping** it
(`window * weight` summed directly). Native Conv2d performs the same
**cross-correlation**. This is the deliberate, documented convention
(the standard deep-learning "convolution"); Phase D does not introduce a
mathematically flipped variant.

**Output contiguity.** Every Phase-D kernel writes a **freshly
allocated row-major contiguous** output — exactly like every existing
native kernel (`relu`, `sum`, `narrow_backward`, `matmul` all allocate
`NativeTensorCore.zeros(...)` and write into it). Outputs never alias
inputs and never carry a strided or offset layout.

**How layout crosses the C ABI.** Because Phase-D kernels operate only on
**contiguous** NCHW/OIHW tensors (see §5), they do **not** take general
shape/stride/offset arrays the way the odometer kernels do. Instead each
kernel takes the storage handles plus the **integer dimensions** it needs
(`N, C, H, W, O, kh, kw, sh, sw, ph, pw, out_h, out_w`) and computes
row-major offsets internally. The Python `NativeTensorCore` wrapper
guarantees contiguity (materializing when necessary) and passes only the
`offset` of each operand alongside the dimensions. This keeps the kernel
signatures explicit and their index arithmetic self-contained and
sanitizer-checkable.

---

## 3. Supported argument forms

Reuse the stable `_pair()` validator's *semantics* (int or 2-tuple of
ints, bool rejected, minimum bound enforced) — re-expressed in the
native package's strict style. The stable helper lives at
`tensorforge.nn.conv._pair`; the native layer will not import the stable
frontend (the two lines never cross-import), so an equivalent local
`_pair`/`_spatial_pair` helper is introduced in the native package.

### 3.1 `NativeFlatten`

**Decision: fixed batch-preserving flatten only** — no `start_dim` /
`end_dim`. This matches the stable `Flatten` exactly
(`x.reshape(x.data.shape[0], -1)`), which is the reference, and it keeps
D1 to a pure reshape composition. Adding `start_dim`/`end_dim` would be
new public surface with no consumer in the planned CNN proof, so it is
deferred (and noted here as the natural extension if a future model needs
it).

Contract:

- **Rank requirement:** input rank **≥ 2** (a batch axis plus at least
  one feature axis), mirroring the stable `ndim < 2` rejection. Rank 0/1
  raise `ValueError`.
- **Negative-dimension normalization:** not applicable — there is no
  `dim` argument. (If `start_dim`/`end_dim` are ever added, they will
  normalize negatives against `ndim` the way `_normalize_axis` already
  does.)
- **Scalar handling:** scalars (rank 0) are rejected by the rank rule.
- **Empty-dimension behavior:** the native runtime forbids zero-size
  dimensions everywhere (`_as_shape` rejects non-positive dims), so
  Flatten inherits that — every axis is ≥ 1, and the flattened feature
  count is the product of the trailing axes.
- **View vs copy — refined in D1 to an owning result.** The forward uses
  a reshape *view* internally — directly for a contiguous input, and after
  a `contiguous_copy()` for a non-contiguous one (`reshape` requires
  contiguity, the same rule the runtime already enforces) — but the
  module's **output always owns its storage** (a final `contiguous_copy()`
  materializes the reshaped view). The original D0 sketch returned the
  bare reshape *view* for a contiguous input; D1 implementation showed
  that is unsafe for a composable module: `NativeTensor.reshape` returns a
  **borrowing** view valid only while its source storage stays open, and
  inside a `NativeSequential` each layer's input is a transient dropped as
  the loop rebinds — so a bare view dangles (storage freed) the moment the
  next layer runs, reproducibly, in the no-grad/eval path. Returning an
  independent owning tensor (as `NativeReLU`/`NativeLinear` already do)
  removes that hazard while still touching the contiguity rule only
  through existing ops and copying data at most once for a contiguous
  input.
- **Backward:** entirely the existing `reshape` (and, on the
  non-contiguous path, `contiguous_copy`) backward — the inverse reshape
  of the upstream gradient. `NativeFlatten` adds **no** new backward and
  **no** new kernel.

### 3.2 Conv2d (`NativeConv2d` constructor + `NativeTensor.conv2d`)

Constructor signature (mirroring stable `Conv2d`):
`NativeConv2d(in_channels, out_channels, kernel_size, stride=1,
padding=0, bias=True, *, seed=None, requires_grad=True)`.

| Argument | Accepted forms | Validation |
|---|---|---|
| `in_channels` | positive int | real int, non-bool, > 0 |
| `out_channels` | positive int | real int, non-bool, > 0 |
| `kernel_size` | int, or `(kh, kw)` | ints, non-bool, each **≥ 1** |
| `stride` | int, or `(sh, sw)` | ints, non-bool, each **≥ 1** (zero/negative rejected) |
| `padding` | int, or `(ph, pw)` | ints, non-bool, each **≥ 0** (negative rejected) |
| `bias` | bool | real bool |
| `seed` | int or `None` | like `NativeLinear` (local RNG) |
| `requires_grad` | bool | real bool |

**Constructor validation** runs entirely **before any native
allocation** (the project's standard order — see `NativeLinear`), so a
bad argument never leaks parameter storage.

**Runtime tensor validation** (in `forward` / `NativeTensor.conv2d`):

- input is an **open `NativeTensor`** (stable `Tensor`, NumPy arrays,
  lists, scalars rejected); dtype/device match the weight
  (`float64`/`cpu`).
- input rank is exactly **4** (NCHW); `input.shape[1] == in_channels`
  (channel compatibility) — else `ValueError` naming both.
- weight rank exactly 4 with shape `(out_channels, in_channels, kh, kw)`;
  bias (if present) shape `(out_channels,)`.
- **No dilation, no groups** — not in the signature at all.
- **Zero-sized dimensions:** unsupported (runtime forbids them); every
  extent is ≥ 1.
- Kernel/stride must **fit** the padded input (§4); an output that would
  be ≤ 0 raises before allocation.

### 3.3 MaxPool2d (`NativeMaxPool2d` constructor + `NativeTensor.maxpool2d`)

Constructor signature (mirroring stable `MaxPool2d`):
`NativeMaxPool2d(kernel_size, stride=None, padding=0)`.

| Argument | Accepted forms | Validation |
|---|---|---|
| `kernel_size` | int, or `(kh, kw)` | ints, non-bool, each **≥ 1** |
| `stride` | int, `(sh, sw)`, or **`None`** | `None` ⇒ **stride = kernel_size** (PyTorch/stable convention: non-overlapping windows); explicit values each ≥ 1 |
| `padding` | int, or `(ph, pw)` | ints, non-bool, each ≥ 0 |

**Padding in Phase D: included** (stable MaxPool2d supports it, and
including it keeps native/stable parity honest). Windows conceptually see
`-inf` outside the input (§10), so a padded cell never wins.

Runtime validation mirrors Conv2d: open 4-D NCHW `NativeTensor`,
float64/cpu, the window must fit the padded input, output extents ≥ 1
before allocation. MaxPool2d has **no parameters**.

**Status (D10 — implemented as specified).** `NativeMaxPool2d(kernel_size,
stride=None, padding=0)` ships in
`src/tensorforge/experimental/native_maxpool2d.py`, exported from the
experimental package and listed in `NATIVE_MODULES` (removed from
`UNSUPPORTED`). It normalizes all three arguments through `_spatial_pair`
(`stride=None` ⇒ the normalized `kernel_size`) into two-element tuples and
delegates its forward entirely to `NativeTensor.maxpool2d`; it is
parameter-free, buffer-free, holds no winner state between calls, and
contributes no state-dictionary or checkpoint keys. See §18 (D10) for the
full contract.

This is the **smallest coherent contract** that supports a real native
CNN training proof (Conv → ReLU → MaxPool → Flatten → Linear → MSE).

---

## 4. Output-shape formulas

For both Conv2d and MaxPool2d, per spatial axis:

```
out_h = floor((H + 2*ph - kh) / sh) + 1
out_w = floor((W + 2*pw - kw) / sw) + 1
```

(Identical to the stable implementations, which use Python integer
`//`.)

- **Floor behavior:** integer floor division `//` on Python ints.
- **Padding is symmetric per axis:** `ph` is added to *both* top and
  bottom, `pw` to *both* left and right (stable pads
  `((ph, ph), (pw, pw))`). No asymmetric padding in Phase D.
- **Asymmetric dimensions:** two-element tuples give independent
  `(kh, kw)`, `(sh, sw)`, `(ph, pw)`, so rectangular inputs and
  rectangular kernels are fully supported.
- **Validation when the kernel does not fit / output ≤ 0:** if `out_h ≤
  0` or `out_w ≤ 0`, raise `ValueError` (naming kernel, stride, padding,
  and input) **before any native allocation** — exactly the stable guard.
- **Overflow safety:** all shape arithmetic is done in **Python ints**
  (arbitrary precision) in the wrapper, *before* the count is handed to
  `NativeTensorCore.zeros(...)`; no intermediate can overflow int64 in the
  shape-math stage. The kernel receives already-validated dimensions.
- **Validation happens before allocation:** the wrapper computes and
  checks `out_h, out_w` (and channel compatibility) first, then allocates
  the output; a failed check allocates nothing.

### Worked examples

| Case | H×W | kernel | stride | padding | out_h × out_w |
|---|---|---|---|---|---|
| No padding, stride 1 | 5×5 | 3×3 | 1 | 0 | 3×3 |
| Symmetric padding (same) | 5×5 | 3×3 | 1 | 1 | 5×5 |
| Stride > 1 | 7×7 | 3×3 | 2 | 0 | 3×3 |
| Rectangular input | 6×10 | 3×3 | 1 | 0 | 4×8 |
| Rectangular kernel | 8×8 | 2×4 | 1 | 0 | 7×5 |
| Tuple stride & padding | 8×8 | 3×3 | (2,1) | (1,0) | 4×6 |
| Invalid (kernel too big) | 3×3 | 5×5 | 1 | 0 | **raises** (`out ≤ 0`) |
| Invalid (stride overshoots) | 4×4 | 4×4 | 3 | 0 | 1×1 (valid) |

---

## 5. Strided / non-contiguous input policy

The `NativeTensorCore` view model (verified in `backends/cpp.py`) allows
arbitrary strides and offsets, and the existing elementwise/matmul/
reduction kernels read strided views directly through an odometer. A
4-D convolution or pooling loop that also had to honor arbitrary
strides/offsets for input, weight, *and* upstream gradient would be far
harder to write correctly and to sanitize.

**Decision: Policy B — the Python/Core wrapper makes an explicit
contiguous copy of any non-contiguous operand before the kernel runs.**
The kernels themselves consume only **row-major contiguous** storage plus
an `offset`.

Applied per operand:

| Operand | Kernel input | Wrapper guarantee |
|---|---|---|
| Conv2d input | contiguous NCHW | `contiguous_copy()` if not already contiguous |
| Conv2d weight | contiguous OIHW | `contiguous_copy()` if not already contiguous |
| Conv2d upstream gradient | contiguous NCHW | copied if a user handed a strided gradient |
| MaxPool2d input | contiguous NCHW | copied if not contiguous |
| MaxPool2d upstream gradient | contiguous NCHW | copied if strided |

Why Policy B:

- **Correct view behavior:** the caller's view is untouched; only a
  private materialized copy feeds the kernel. This is exactly how the
  existing reshape/reduction backwards already materialize strided
  upstream gradients via `_native_copy`.
- **Explicit ownership:** each contiguous copy is fresh **owning** storage
  the wrapper closes as soon as the kernel returns; nothing borrows a
  transient.
- **Predictable backward semantics:** gradients live at the **logical
  shape**, so a transposed/narrowed/offset parent still differentiates
  correctly (the same principle already used for `narrow`/`reshape`
  backward).
- **No hidden stable/native conversion:** the copy is native→native
  (`contiguous_copy`); NumPy never enters the compute or gradient path.
- **Clear performance expectation:** a non-contiguous operand costs one
  extra native copy — documented, not hidden. im2col and BLAS remain out
  of scope, so this is honest and simple.
- **Failure atomicity:** a copy either fully succeeds (owning storage) or
  raises (allocation failure via the status contract) before the kernel
  touches any output.

**Output-layout guarantee (separate from input support):** regardless of
input layout, every Conv2d/MaxPool2d **output and every gradient** is a
**fresh row-major contiguous owning** `NativeTensor`. Input strided
*support* is "copy-then-compute"; output *layout* is always contiguous.

---

## 6. Conv2d forward contract

**Mathematical operation (cross-correlation):**

```
out[n, o, i, j] = bias[o]
               + Σ_c Σ_{p<kh} Σ_{q<kw}  x_pad[n, c, i*sh + p, j*sw + q] * weight[o, c, p, q]
```

where `x_pad` is `x` zero-padded by `ph`/`pw` on each spatial side.

- **Accumulation order:** deterministic nested loops in
  `c → p → q` order (row-major over the kernel), summed into a double
  accumulator per output cell. Floating-point sums are order-sensitive,
  so parity with the stable einsum is to a **tolerance** (`atol≈1e-9`…
  `1e-12` for the small test tensors), not bit-for-bit — the same honesty
  the reductions doc already states.
- **Bias application:** conceptually initializes each output cell to
  `bias[o]` (or 0 when `bias=False`). Implementation may add bias in the
  kernel *or* leave the kernel bias-free and add the `(out_channels,)`
  bias through the existing broadcast `add` at the `NativeTensor` level.
  **Decision:** the **kernel takes an optional bias handle** (nullable);
  when null, no bias is added. This keeps forward a single fused op and
  avoids a second allocation. (A null bias handle is the documented
  null-handle path in §13.)
- **Padding semantics:** zero padding, symmetric per axis; padding cells
  contribute `0 * weight = 0`. The kernel indexes the *logical* padded
  grid but reads the real (unpadded, contiguous) input, skipping cells
  that fall in the pad region — no padded copy is materialized.
- **Output allocation ownership:** the wrapper allocates
  `NativeTensorCore.zeros((N, O, out_h, out_w))` (fresh owning contiguous)
  and passes it as the output handle; the kernel only writes.
- **Input validation order:** dtype/device match → rank/channel/shape
  checks → output-shape computation and fit check → **allocate** →
  call kernel. A failure at any pre-allocation step allocates nothing.
- **C ABI signature strategy:** handles for input, weight, (nullable)
  bias, output, plus the integer dimensions
  `N, C, H, W, O, kh, kw, sh, sw, ph, pw, out_h, out_w` and the input
  offset. No shape/stride arrays (contiguous by Policy B).
- **Error/status behavior:** wrapped in `TF_GUARD_BEGIN/END_VOID`; an
  allocation inside the kernel that fails throws `std::bad_alloc`, is
  caught, recorded as `TF_ERROR_ALLOC`, and surfaces in Python as
  `MemoryError` via the existing `errcheck` hook.
- **First reference implementation:** **direct nested-loop
  convolution** (six loops: `n, o, i, j` outer, `c, p, q` inner). No
  im2col, no blocking, no threads.
- **Determinism:** identical inputs give identical outputs on a given
  build (fixed loop order, no parallel reduction).
- **The four layers and their responsibilities:**
  1. **Raw C++ kernel** (`tf_core_conv2d_forward`): pure arithmetic over
     contiguous storage; autograd-unaware; exception-guarded.
  2. **`NativeTensorCore.conv2d_forward(...)`**: validates shapes,
     enforces contiguity (Policy B), allocates the output, marshals the
     ctypes call. Forward-only; no graph.
  3. **`NativeTensor.conv2d(...)`**: builds the autograd graph node
     (§7–§8) when an operand requires grad; otherwise returns a plain
     forward tensor.
  4. **`NativeConv2d`**: the module holding the weight/bias parameters
     and calling `input.conv2d(weight, bias, ...)` (§9).

Conv2d is a **new fused primitive** at the `NativeTensor` layer (like
`relu`, `matmul`, `narrow`), **not** a composition of existing ops — it
cannot be expressed from current native ops without im2col+matmul, which
is out of scope. This contrasts with `NativeLinear`, whose forward is
pure existing ops and needs no new backward.

---

## 7. Conv2d backward contract

Three independent gradients, each with its own rule; **not** collapsed
into one future milestone (D4 does input-grad; D5 does weight- and
bias-grad).

Shared: upstream gradient `g = grad_out` has shape `(N, O, out_h,
out_w)`; it is made contiguous by Policy B before any kernel runs.

### 7.1 Gradient w.r.t. input — `∂L/∂x`

- **Metadata required:** the weight values, `(N,C,H,W)`, kernel, stride,
  padding.
- **Output shape:** `(N, C, H, W)` — the parent input's shape.
- **Rule:** `grad_x_pad[n, c, i*sh+p, j*sw+q] += g[n, o, i, j] *
  weight[o, c, p, q]`, summed over `o, i, j`; then the pad border is
  dropped (gradient landing on padding is discarded — matching the stable
  slice `grad_padded[..., ph:ph+h, pw:pw+w]`).
- **Distinct kernel:** yes — `tf_core_conv2d_input_backward`.
- **Allocation ownership:** wrapper allocates fresh owning contiguous
  `(N,C,H,W)` zeros; kernel scatters into it. Result is **always
  contiguous**.
- **Accumulation:** `+=` into the zero output (overlapping windows
  accumulate), deterministic `o→i→j→p→q` order.
- **This callback runs only when `input` requires grad**, and when it
  runs it **rereads the weight's forward value** — so a direct-parameter
  weight is version-guarded **only in that case** (§8).
- **Status (D4 — implemented, internal):** the numerical scatter-add ships
  as `tf::conv2d_input_backward_contiguous` in `cpp/src/conv2d.cpp`
  (declared in `cpp/include/tf_conv2d_internal.h`) — a hidden C++ symbol
  exercised only by the `conv2d_input_backward` CTest, **not** reachable
  from Python. It **zero-initializes the whole grad-input span itself**
  before the deterministic `n → o → i → j → c → p → q` `+=` accumulation,
  reads `grad_output`/`weight` without mutation, and allocates nothing.
  **D6 exposed it** through the exported guarded `tf_core_conv2d_input_backward`
  wrapper, `NativeTensorCore.conv2d_input_backward`, and the
  `NativeTensor.conv2d` input-grad callback (which records the weight's
  version iff the input requires grad).

### 7.2 Gradient w.r.t. weight — `∂L/∂W`

- **Metadata required:** the input values (padded logically), `g`,
  kernel, stride, padding.
- **Output shape:** `(O, C, kh, kw)` — the weight's shape (OIHW).
- **Rule:** `grad_w[o, c, p, q] += g[n, o, i, j] * x_pad[n, c, i*sh+p,
  j*sw+q]`, summed over `n, i, j` (matches stable
  `einsum("no,nckl->ockl", g, window)`).
- **Distinct kernel:** yes — `tf_core_conv2d_weight_backward`.
- **Allocation ownership:** wrapper allocates fresh owning contiguous
  `(O,C,kh,kw)` zeros. **Always contiguous.**
- **This callback runs only when `weight` requires grad**, and when it
  runs it **rereads the input's forward value** — so a direct-parameter
  input is version-guarded **only in that case** (§8).
- **Status (D5 — implemented, internal):** the numerical accumulation ships
  as `tf::conv2d_weight_backward_contiguous` in `cpp/src/conv2d.cpp`
  (declared in `cpp/include/tf_conv2d_internal.h`) — a hidden C++ symbol
  exercised only by the `conv2d_weight_backward` CTest, **not** reachable
  from Python. It **zero-initializes the whole grad-weight span itself**
  before the deterministic `n → o → i → j → c → p → q` `+=` accumulation,
  reads `grad_output`/`input` without mutation, reads no bias, and allocates
  nothing. **D6 exposed it** through the exported guarded
  `tf_core_conv2d_weight_backward` wrapper,
  `NativeTensorCore.conv2d_weight_backward`, and the `NativeTensor.conv2d`
  weight-grad callback (which records the input's version iff the weight
  requires grad).

### 7.3 Gradient w.r.t. bias — `∂L/∂b`

- **Output shape:** `(O,)`.
- **Rule:** `grad_b[o] = Σ_{n,i,j} g[n, o, i, j]` — i.e.
  `g.sum(axis=(0, 2, 3))`.
- **Distinct kernel:** **no.** This composes from the **existing native
  `sum` reductions** (`g.sum(axis=0)` then reduce the two spatial axes),
  reusing tested kernels — no new C++ code. Reads only `g` (no bias
  value), so it is **not** version-sensitive.
- **Locked sequence (D5, validated):** `NativeTensorCore.sum` reduces one
  axis at a time, so `∂L/∂b` over `g = (N, O, oh, ow)` is the deterministic
  chain **`g.sum(axis=0)` → `(O, oh, ow)`, then `.sum(axis=1)` → `(O, ow)`,
  then `.sum(axis=1)` → `(O,)`** (`keepdims=False` throughout). Each step is
  a fresh **owning** contiguous core; the two intermediates are **closed**
  after use, only the final `(O,)` result survives — so a failed
  intermediate reduction (e.g. an allocation failure) raises through the
  existing status contract with no leak, and the future D6 `backward()`
  snapshot/rollback restores grads (§7 shared properties). Because the chain
  reads only `g`, a **bias-only backward records no input/weight version**
  and never raises stale-graph on an input/weight mutated after forward
  (§8). **Status (D5):** proved reachable and correct via existing
  reductions in `tests/test_native_conv2d_gradient_contract.py` — **no
  dedicated bias kernel and no bias C ABI symbol.** **D6 wired it in:** the
  `NativeTensor.conv2d` bias-grad callback runs exactly this sequence
  (closing both intermediates), reading only the upstream — so a bias-only
  backward records no input/weight version and is unaffected by input/weight
  mutation after forward.

### Shared backward properties

- **Determinism:** fixed loop orders; parity to a tolerance (float
  order-sensitivity), verified by finite differences.
- **Gradient-check strategy:** central finite differences on each of `x`,
  `W`, `b` independently (the project's standard, as in
  `NativeLinear`/`NativeMSELoss` tests), plus exact hand-computed small
  cases and stable-framework parity.
- **Only a subset of parents requiring grad:** each gradient kernel/
  reduction runs **only if** that parent `requires_grad` — `if
  x._requires_grad: … ; if weight._requires_grad: … ; if bias is not None
  and bias._requires_grad: …`. Non-requiring parents cost nothing.
- **Failure behavior:** each contribution is fully computed into fresh
  owning storage before it is accumulated; an allocation failure raises
  (`MemoryError`) and the overall `backward()` snapshot/rollback (already
  in `NativeTensor.backward`) restores every node's grad — no partial
  commit.

---

## 8. Conv2d autograd ownership and saved state

The Python autograd node (built by `NativeTensor._from_op`) stores:

- **Parents:** `(input, weight)` when `bias=False`; `(input, weight,
  bias)` when bias is present — exactly the operands whose gradients may
  be needed.
- **No re-materialized tensor values are duplicated.** The backward
  reads the parents' **current values** through their live cores
  (`input`, `weight`), the same way `multiply`/`matmul` backward read
  their operands. Nothing copies the full input or weight into the node.
- **Metadata captured by value** (small, cheap): `(N,C,H,W)`, `(O,)`,
  `(kh,kw)`, `(sh,sw)`, `(ph,pw)`, `out_h`, `out_w`, and whether bias is
  present. These are plain Python ints held in the closure.
- **Forward output / temporaries:** **not retained.** Unlike
  `sqrt`/`reciprocal` (which save the forward *output*), Conv2d backward
  needs only the parents' values and the upstream gradient, so the
  forward output is not kept alive by the node.
- **Version snapshots / stale-graph validation — conditional on which
  callbacks will run.** A value's version is recorded **only when a
  backward callback that actually rereads that value will execute**, so
  the graph never fails merely because an operand nothing rereads was
  mutated after forward. Concretely:
  - **input-grad** rereads the **weight** value and runs iff `input`
    requires grad ⇒ record the **weight**'s version **iff
    `input._requires_grad`**.
  - **weight-grad** rereads the **input** value and runs iff `weight`
    requires grad ⇒ record the **input**'s version **iff
    `weight._requires_grad`**.
  - **bias-grad** rereads neither input nor weight ⇒ a **bias-only**
    backward records **no** version and must **not** raise if input or
    weight changed after forward.
  - If **both** input and weight require grad, **both** relevant version
    snapshots are recorded (weight's for input-grad, input's for
    weight-grad).

  How the future D6 builder constructs the version-read set — the value
  operands whose current value an *active* callback will reread:

  ```
  value_read_operands = []
  if input._requires_grad:    # input-grad will reread the weight value
      value_read_operands.append(weight)
  if weight._requires_grad:   # weight-grad will reread the input value
      value_read_operands.append(input)
  expected_versions = _versioned_value_reads("conv2d", value_read_operands)
  ```

  `_versioned_value_reads` keeps an entry **only** for operands carrying a
  `_version` slot — i.e. **direct `NativeParameter`s** — preserving the
  existing TensorForge rule that versioning applies only where the native
  autograd architecture supports a directly versioned object (a plain
  `NativeTensor` input, or an intermediate result, records nothing).
  `backward()` validates every recorded version **before** running any
  callback or touching any gradient (existing machinery), so a mutated
  direct-parameter operand whose value an active callback rereads raises
  the deterministic stale-graph `RuntimeError`. **Bias is never recorded**
  (its gradient reads no bias value).
- **Ownership / lifetime of saved metadata:** the ints live in the
  closure and are released when the graph is freed (one-shot
  `backward`) or GC'd — no native storage is owned by the Conv2d node
  beyond what the parents already own.
- **Shared parameters:** a parameter used as both operands (not typical
  for Conv2d, but possible via weight sharing across layers) accumulates
  correctly — `_accumulate_grad` sums contributions, and version
  recording is idempotent (the existing `multiply(a, a)` precedent).
- **Repeated backward:** governed by the existing one-shot/`retain_graph`
  policy — nothing Conv2d-specific.
- **Gradient accumulation into leaves:** standard `_accumulate_grad`
  (native `add`), un-broadcast not needed (Conv2d gradients already have
  the parent's exact shape; bias-grad is produced at `(O,)` directly).
- **Subset of parents requiring grad:** the node is built via `_from_op`,
  which sets `requires_grad` to the OR of parents; the closure guards
  each contribution by the parent's own `_requires_grad`.

Graph ownership stays **in Python** — no graph state moves into C++.

---

## 9. `NativeConv2d` module contract

(Designed here; **implemented in D7**, not D0.)

- **Parameter shapes:** `weight` is `(out_channels, in_channels, kh,
  kw)` (OIHW); `bias` is `(out_channels,)` when `bias=True`, else the
  attribute reads as `None` (the `NativeLinear` pattern).
- **Registration:** `weight` then `bias` (deterministic order), through
  the inherited `NativeModule` assignment-registration — so
  `parameters()`/`named_parameters()`/`state_dict()` are ordered
  `["weight", "bias"]` (nested e.g. `"0.weight"` in a `NativeSequential`).
- **Bias-disabled behavior:** nothing registered under `"bias"`;
  `state_dict()` has only `"weight"`; loading a biased state into a
  bias-free layer reports `"bias"` as **unexpected**, and the reverse as
  **missing**, under the existing v3.3 strict rules.
- **Initialization:** deterministic fan-in uniform, matching
  `NativeLinear`'s style but with the **conv fan-in** the stable Conv2d
  uses: `fan_in = in_channels * kh * kw`, `bound = 1/sqrt(fan_in)`,
  sampled from `[-bound, +bound]` with a **local**
  `numpy.random.default_rng(seed)` (global RNG untouched; NumPy only as
  host-side data prep feeding `NativeParameter`). The stable Conv2d uses
  `randn / sqrt(fan_in)` (Gaussian); the native line's established
  convention is **uniform** fan-in bounds (as `NativeLinear` chose over
  the stable Gaussian), so `NativeConv2d` follows the native convention
  while using the **same fan-in count** as stable. This difference is
  intentional and documented (both are valid fan-in scalings; the native
  line avoids the stable global-RNG `randn`).
- **Object identity:** parameters are `NativeParameter`s discovered by
  identity; no `__eq__`/`__hash__` overrides.
- **State-dict keys / checkpoint / loading:** `weight`/`bias` snapshot
  and load through the existing v3.3 `state_dict`/`load_state_dict` and
  v3.14 checkpoint paths unchanged — independent owning snapshots, atomic
  validate→stage→commit, identity/registration/`requires_grad`/grad
  preserved.
- **`requires_grad`:** default `True`; `False` freezes both parameters
  (registered, traversable, in `state_dict`, but no gradient) — the
  `NativeLinear` frozen semantics.
- **Representation string:** yes (existing modules define `__repr__`):
  `NativeConv2d(in_channels=…, out_channels=…, kernel_size=…, stride=…,
  padding=…, bias=…)`.
- **Input-channel validation:** `forward` rejects a 4-D input whose
  `shape[1] != in_channels` (and non-4-D input) with a clear
  `ValueError`.
- **`NativeSequential` interaction:** `NativeConv2d` is a `NativeModule`,
  so it drops into a `NativeSequential` slot; its state keys nest under
  the slot name (`"0.weight"`, `"0.bias"`). Forward is pure composition —
  `NativeSequential` adds no node.

---

## 10. MaxPool2d forward contract

- **Window semantics:** for each output cell `(n, c, i, j)`, take the
  max over the `kh×kw` window `x_pad[n, c, i*sh:i*sh+kh,
  j*sw:j*sw+kw]`.
- **Padded positions participate in winner selection** (they are **not**
  skipped). The conceptual window spans the padded grid; stable
  `np.pad(..., constant_values=-np.inf)` then runs `argmax` over the full
  flattened `(kh*kw)` window, padded cells included. The native forward
  scans the same `kh×kw` window in the **same row-major order** (`p`
  outer, `q` inner), treating out-of-bounds positions as the conceptual
  `-inf`, so its selection matches stable position-for-position.
- **Value used outside the input:** `-inf` (not `0`) — a `0`-pad would
  corrupt the max for all-negative activations. This is why padding
  participates yet a padded cell **loses to any finite in-bounds value**.
- **When a `-1` (padding-winner) sentinel is possible:** exactly when the
  window's maximum is `-inf` **and** the first row-major position
  achieving it is a padded cell — i.e. **every in-bounds value in the
  window is `-inf`** (all in-bounds values equal the padding value). With
  finite input data this never happens; it requires genuine `-inf` input
  or an all-padding window (below).
- **Windows with no in-bounds positions are allowed, not rejected** —
  matching stable, whose padding validator (`_pair(..., minimum=0)`)
  imposes no kernel-relative cap. Such a window is entirely `-inf`: the
  output cell is `-inf` and the winner is `-1` (gradient dropped in
  backward). This is a **deliberate parity choice with stable** (a
  stricter `ph < kh`, `pw < kw` rule *would* preclude all-padding windows
  and confine `-1` to genuine `-inf` data, but native does not impose it —
  bit-parity with the stable reference is preferred over adding a new
  restriction the stable line does not have).
- **Deterministic tie-breaking:** **first occurrence in row-major window
  traversal** — identical to stable's `argmax`, which returns the first
  maximum. Realized by a **strict `>`** update (keep the current winner
  unless a strictly greater value appears), so an equal-valued later cell
  never displaces the earlier one. This covers the tie between a valid
  input equal to the conceptual padding value (`-inf`) and a padded cell:
  whichever comes **first** in row-major order wins (a real `-inf` cell →
  its offset; a padded `-inf` cell → `-1`).
- **All valid values `-inf`:** the window maximum is `-inf`; the tie rule
  above selects the **first** row-major position (real or padded),
  deterministically — a real first position yields its offset (gradient
  flows to that real `-inf` cell, as in stable), a padded first position
  yields `-1`.
- **NaN behavior — deliberate, documented divergence from stable.** With
  the strict-`>` scan a `NaN` never displaces an earlier maximum
  (`NaN > best` is always false), so the **first non-NaN maximum** is
  selected. NumPy's `argmax` instead **propagates the first NaN** (it
  reports the NaN's index), so a window containing a `NaN` would make
  stable emit `NaN` with the NaN's position as winner. Native does **not**
  replicate that: it keeps the kernel a simple branchless `>` comparison,
  and **float64 `NaN` inputs are outside the supported/tested contract**
  (the CNN proof and parity tests use finite data). This is the deliberate
  native choice and its reason; it is not a promised guarantee.
- **Forward output and backward winner selection use the SAME rule.** The
  forward kernel computes the output value and the saved winner in **one
  pass** with the single strict-`>` traversal, and backward consumes that
  exact saved winner (it never re-selects). So forward and backward are
  **always mutually consistent** — even for `-inf`/`NaN`/tie inputs where
  parity-with-stable is not guaranteed, internal consistency still is.
- **Output allocation:** fresh owning contiguous `(N, C, out_h, out_w)`
  zeros written by the kernel.
- **Saved winner representation:** a parallel **winner buffer** of shape
  `(N, C, out_h, out_w)` — see §12.
- **Winner index coordinate system:** each winner stores the **flat
  offset into that `(n,c)` plane's `H*W` real (unpadded) input**, or the
  **sentinel `-1`** when the max fell on a padding cell. This makes the
  backward a pure scatter-add with **no** stride/padding recomputation and
  correctly drops padding gradient.
- **Ownership / lifetime of saved indices:** the winner buffer is a
  private owning native buffer held by the autograd node's backward
  closure; released when the graph is freed or GC'd (the same lifetime as
  `sqrt`/`reciprocal`'s saved forward output). It is **internal** — never
  surfaced as a public `NativeTensor`, never given a public dtype.
- **C ABI boundary:** `tf_core_maxpool2d_forward` takes input handle,
  output handle, **winner-buffer handle**, and the integer dims; it writes
  both outputs. Exception-guarded.
- **Layer responsibilities:** kernel = max + winner recording; Core
  wrapper = contiguity/allocation/marshalling; `NativeTensor.maxpool2d` =
  graph node holding the winner buffer; `NativeMaxPool2d` = the module.
- **Status (D8 — implemented, raw + Core forward).** The first two layers
  ship: `tf::maxpool2d_forward_contiguous` (internal, `cpp/src/pooling.cpp`,
  declared in `cpp/include/tf_pooling_internal.h`) and the exported guarded
  `tf_core_maxpool2d_forward` wrapper, reached from Python through
  `NativeTensorCore.maxpool2d_forward` / the private
  `_maxpool2d_forward_with_winners`. The third and fourth layers
  (`NativeTensor.maxpool2d` in D9, `NativeMaxPool2d` in D10) do not exist
  yet, so the registry still lists `maxpool2d` and `NativeMaxPool2d` as
  unsupported while `TENSOR_CORE_OPS` carries the layer-qualified
  `maxpool2d_forward`. **The implemented selection rule, exactly as
  specified above:** one row-major (`p` outer, `q` inner) scan per window;
  a padded position is a real candidate holding the conceptual `-inf` with
  winner `-1`; the first non-NaN candidate seeds the scan and only a
  **strictly greater** later candidate replaces it, so ties keep the first
  occurrence; an in-bounds first maximum saves `ih * W + iw`, a padded
  first maximum saves `-1`; a completely padded window yields `-inf` and
  `-1`. **NaN fallback (D8 refinement of the unsupported-NaN policy):** a
  NaN never wins, and a window whose candidates are *all* NaN (only
  reachable with no padded position, since padding is `-inf`) falls back
  deterministically to its **first candidate** — that candidate's value and
  its winner — so the output value and the saved winner always come from the
  same selected candidate even there. No NumPy/PyTorch NaN parity is
  claimed or tested.

---

## 11. MaxPool2d backward contract

- **Gradient scatter:** for each output cell, read its winner; if `≥ 0`,
  **add** the upstream gradient `g[n,c,i,j]` to that input element; if
  `-1` (padding won), **drop** it.
- **Overlapping-window accumulation:** with `stride < kernel`, an input
  element can win several windows; contributions **accumulate** (`+=`),
  which is correct.
- **Tie behavior:** the winner recorded at forward (first occurrence)
  receives the whole window's gradient; other equal cells get nothing —
  matching stable.
- **Saved-index validation:** the backward kernel safely trusts the
  winner buffer produced by its own forward (a private, immutable internal
  invariant — §12), reading each winner as `-1` or an exact in-range
  integer offset and **never silently rounding** a non-integral value; the
  wrapper still checks the winner buffer's shape equals the output shape
  and the upstream shape equals the output shape before calling.
- **Upstream gradient shape validation:** `g.shape == (N, C, out_h,
  out_w)` — else `ValueError`.
- **Output gradient shape:** `(N, C, H, W)` — the input's shape.
- **Allocation ownership:** fresh owning contiguous `(N,C,H,W)` zeros;
  kernel scatters. **Always contiguous.**
- **Failure atomicity:** contribution computed into fresh storage before
  accumulation; allocation failure raises and `backward()`'s snapshot
  rolls back.
- **Version / stale-graph checks:** MaxPool2d backward reads **only** the
  saved winners and the upstream — **never the input values** — so it
  records **no expected parameter version** (there is no value-sensitive
  read; input to a pool is rarely a parameter anyway). Consistent with the
  directive to **avoid recomputing winners** during backward.
- **Rereads input values?** No — relies **entirely** on saved winners.
- **Input not requiring grad:** if `input` does not require grad, no node
  is built and no winner buffer is retained (plain forward tensor).
- **Status (D9 — implemented as specified).** The scatter ships as the
  internal `tf::maxpool2d_backward_contiguous` plus the checked
  `tf_core_maxpool2d_backward` wrapper, reached from Python through
  `NativeTensorCore.maxpool2d_backward` and the single input-gradient
  callback of the `NativeTensor.maxpool2d` node (§18, D9). It takes **no
  kernel/stride/padding argument at any layer** — the saved winners are the
  whole routing story, so no window geometry is reconstructed and no input
  value is reread. The saved-index validation §12 demands is enforced at
  the checked boundary rather than assumed: every entry must be `-1.0` or
  an exact integral offset in `[0, H*W - 1]`, and a violation raises
  `ValueError` before grad_input is touched — never a silent rounding.
  Because backward is value-independent, the node records **no** expected
  parameter version, and the winner buffer's lifetime is bound to the graph
  history (released by a one-shot `backward()` or `close()`, retained under
  `retain_graph=True`, kept alive across a failed retryable backward).

---

## 12. Saved winner-index representation

**Chosen representation:** a dedicated **internal float64 native buffer**
(`NativeTensorCore`, owning, contiguous, shape `(N, C, out_h, out_w)`)
holding **flat `H*W`-plane offsets** (or `-1` sentinel for padding
winners), produced by the forward kernel and consumed by the backward
kernel.

**Value domain (locked).** Every stored winner is either:

- an **integral, non-negative flat offset** into the `(n, c)` plane's
  unpadded `H*W` grid, in the range `0 … H*W - 1`; **or**
- the sentinel **`-1`**, which is the **only** negative value the buffer
  may ever hold.

No other value is legal. There are no fractional indices, no other
sentinels, and no NaN/inf in the winner buffer.

**Float64 exactness bound (validated before allocation or kernel
execution).** IEEE float64 represents every integer in
`[-(2^53), 2^53]` exactly, so a winner is exact iff
`H*W - 1 ≤ 2^53 - 1`, i.e. **`H*W ≤ 2^53`** (and `-1` is trivially
exact). The Core wrapper **proves this before allocating the output or
the winner buffer and before calling the forward kernel**: it checks
`H * W <= 2**53` (Python arbitrary-precision arithmetic, so the product
itself cannot overflow during the check) and raises `ValueError` on
violation. The check is effectively unreachable on real hardware — a
plane that large is astronomically bigger than any allocatable tensor —
but it is **explicit and cheap**, so the "indices are exact" invariant is
guaranteed by construction, never merely assumed.

**Backward must not silently round.** The backward kernel reads each
winner as a float64 and treats it as `-1` (skip) or a non-negative
integer offset. It relies on the forward-established invariant (the
buffer is private and immutable — see below — so nothing between forward
and backward can perturb it); the design mandates that the **forward
kernel writes only exact integers or `-1`**, and the backward **truncates
by exact integer value, never rounds a non-integral float**. A value that
is neither `-1` nor an exact in-range integer would be a corrupted
internal invariant and must surface as an error, not be silently
rounded into a wrong scatter target.

Why float64 (not a new integer dtype, not per-window local indices):

- **Exactness:** guaranteed by the `H*W ≤ 2^53` bound above; `-1` is
  exact too. No precision risk within the validated domain.
- **Zero new dtype surface:** it reuses the existing float64
  `NativeStorage`/`NativeTensorCore` allocation, lifetime, and
  allocation-failure machinery. It does **not** introduce the future
  public integer `NativeTensor` dtype prematurely — the buffer is
  **internal**, never handed to users, never given a dtype tag, never
  routed through `normalize_dtype`.
- **Flat plane offset (vs per-window local index):** storing the absolute
  `H*W` offset (plus `-1` sentinel) makes backward a **pure scatter-add**
  needing no stride/padding arithmetic, and cleanly encodes "padding won
  → no input gradient." Per-window local indices would force the backward
  kernel to recompute `rows/cols` from `(i, j, sh, sw, ph, pw)`; the flat
  offset is simpler and sanitizer-friendlier.

Mechanics:

- **Allocation / deallocation:** allocated by the Core wrapper as
  `NativeTensorCore.zeros((N,C,out_h,out_w))` (owning); the forward kernel
  overwrites it. Deallocated when the autograd node's closure is released.
- **C++/Python ownership boundary:** C++ only *writes* the buffer handed
  to it; Python owns allocation and release — the standard boundary.
- **Lifetime until backward:** held by the backward closure of the
  `maxpool2d` graph node; alive exactly as long as the graph is (one-shot
  free or GC), matching `sqrt`/`reciprocal`'s saved output.
- **Public or internal:** **internal, private, and immutable from the
  public Python API.** It is owned solely by the `maxpool2d` node's
  backward closure, never returned from a public method, never exposed as
  a `NativeTensor`, and never appears in `state_dict`, checkpoints,
  `backend_info`, or `__all__` — so no public caller can read, mutate, or
  alias it between forward and backward.
- **Testing:** validated indirectly through backward correctness
  (finite-difference gradient checks, hand-computed scatters, overlapping
  windows, tie positions, padding-winner drop) and through a lifetime
  test (a closed/freed graph releases it; a retained graph keeps it).
- **Allocation-failure propagation:** the buffer allocation is an ordinary
  native allocation — a failure surfaces as `MemoryError` through the
  existing status contract, before the forward output is returned, so a
  failed pool forward yields no half-built node.

*Alternative considered and deferred:* a dedicated internal **int64**
buffer with its own `tf_pool_indices_alloc/free` C ABI family. Rejected
for D8 because it adds a new allocation family and a Python owner wrapper
for no correctness gain — the float64 buffer is exact and reuses
everything. If a future milestone introduces a real integer native dtype,
the winner buffer is the natural first internal adopter.

**Status (D8 — implemented as specified).** The winner buffer is allocated
by the Core wrapper as a second `NativeTensorCore.zeros((N, C, out_h,
out_w))` (owning, contiguous, offset 0) *after* the output and overwritten
entirely by the forward kernel. The exactness bound is proved **twice**:
`NativeTensorCore._maxpool2d_forward_with_winners` checks `H * W <= 2**53`
in Python arbitrary-precision arithmetic before allocating anything or
copying a non-contiguous input, and `tf_core_maxpool2d_forward` re-proves it
in overflow-checked int64 so a direct ABI caller cannot bypass the Python
check; either violation raises `ValueError`. The kernel writes only `-1.0`
or an exact non-negative integral offset. The public
`NativeTensorCore.maxpool2d_forward` **closes** the winner buffer before
returning the pooled values, so D8 exposes no winner object at all; the
private `_maxpool2d_forward_with_winners` returns the `(output, winners)`
pair that D9's backward closure will own and close exactly once. The buffer
is absent from `state_dict`, checkpoints, parameter/buffer traversal,
`backend_info`, `__all__`, and every public method name (guarded by
`tests/test_native_maxpool2d_core.py`), and `SUPPORTED_DTYPES` still reads
`("float64",)` — no index dtype was introduced.

**Status (D9 — the lifetime half).** D9 made the buffer graph-owned
without exposing it: `NativeTensor.maxpool2d` passes it to `_from_op` as a
**graph resource**, so the node closes it exactly once when its history is
released at either deterministic point (a one-shot `backward()`'s cleanup,
or an explicit `close()`; a merely dropped graph reaches `close()` via the
`__del__` refcount/GC fallback), keeps it under `retain_graph=True`, keeps
it across a failed retryable backward, closes it immediately when no parent
requires grad, and — if graph construction raises — closes both it and the
pooled output before re-raising. It is still never a public `NativeTensor`, never
returned from a public method, and never traversed as a parameter or
buffer; the *only* consumer is the backward callback, through the private
Core method. Its integrity is no longer merely assumed either — the D9 C
ABI validates every entry before scattering (§11 status).

---

## 13. C ABI design

Proposed exported function **families** (exact symbol names finalized at
implementation time, following the `tf_core_*` convention). All are
`TF_EXPORT void`, wrapped in `TF_GUARD_BEGIN/END_VOID`, and registered in
`_CHECKED_KERNELS` so the `errcheck` hook maps failures to Python
exceptions.

| Family | Purpose | Key parameters (categories) |
|---|---|---|
| `tf_core_conv2d_forward` | Conv2d forward | in, weight, **nullable** bias, out handles; dims `N,C,H,W,O,kh,kw,sh,sw,ph,pw,out_h,out_w`; offsets |
| `tf_core_conv2d_input_backward` | `∂L/∂x` | grad_out, weight, grad_in handles; same dims; offsets |
| `tf_core_conv2d_weight_backward` | `∂L/∂W` | grad_out, in, grad_weight handles; same dims; offsets |
| *(bias grad)* | `∂L/∂b` | **no new export** — composed from existing `tf_core_sum` |
| `tf_core_maxpool2d_forward` | MaxPool2d forward | in, out, **winner** handles; dims `N,C,H,W,kh,kw,sh,sw,ph,pw,out_h,out_w`; offset |
| `tf_core_maxpool2d_backward` | MaxPool2d backward (scatter) | grad_out, winner, grad_in handles; same dims; offsets |
| *(winner alloc/free)* | winner buffer | **no new export** — reuses `tf_storage_create`/`tf_storage_destroy` via `NativeTensorCore.zeros` |

Contract details:

- **Parameter categories:** opaque `void*` storage handles (via
  `as_storage`), `const int64_t` dimensions, and `int64_t` offsets. **No**
  shape/stride arrays — Policy B guarantees contiguity, so kernels compute
  row-major offsets themselves.
- **Return / status convention:** `void` return; success/failure signaled
  through the thread-local error slot (`TF_OK` / `TF_ERROR_ALLOC` /
  `TF_ERROR_INVALID` / `TF_ERROR_RUNTIME`) exactly as today.
- **Output ownership:** the **caller (Python)** owns every output/gradient
  buffer; kernels only write into caller-allocated storage.
- **Error behavior:** any `std::exception` (or unknown throw) is caught by
  the guard, recorded, and turned into the matching Python exception; the
  kernel returns without unwinding across the ABI.
- **Null-handle behavior:** only the Conv2d **bias** handle is nullable
  (null ⇒ no bias). All other handles are required; a null required handle
  is a caller bug prevented at the Python layer (the wrapper never passes
  null there). The kernel may additionally treat a null required handle
  defensively by recording `TF_ERROR_INVALID`.
- **Allocation-failure behavior:** internal allocations (e.g. an odometer
  counter, if any) go through `tf::make_counter` / honor
  `tf::should_fail_alloc`, so the **test-only fault-injection** hook can
  force `TF_ERROR_ALLOC` deterministically.
- **Validation location:** **shape/argument validation lives in Python**
  (the Core wrapper), where errors are clearest and no storage is
  allocated on the failure path; kernels assume validated, contiguous
  inputs and focus on arithmetic. (This mirrors the existing kernels: the
  wrapper validates, the kernel computes.)
- **C++ exception guards:** every new export uses `TF_GUARD_*`; **no C++
  exception may cross the boundary**.
- **Test-only fault injection:** the Phase-D failure tests arm
  `tf_test_arm_alloc_failure(nth)` and assert the forward/backward calls
  raise `MemoryError` with no partial mutation — reusing the existing
  fault-injection ABI.

The existing **thread-local error channel** and **`TF_GUARD_BEGIN/END`
macros** are preserved and used unchanged.

---

## 14. C++ source organization

New translation units under the existing split layout (`cpp/src/*.cpp`,
globbed by `CMakeLists.txt`'s `file(GLOB …)` — a new `.cpp` is picked up
automatically):

- **`cpp/src/conv2d.cpp`** — the internal compute kernels and (from D3)
  the exported wrappers. **D2 shipped** the internal
  `tf::conv2d_forward_contiguous` (pure arithmetic, hidden symbol),
  declared in the focused internal header
  **`cpp/include/tf_conv2d_internal.h`** (non-ABI). **D3 added** the
  exported, exception-guarded `tf_core_conv2d_forward` wrapper into this
  same file (validation + Storage handles + the thread-local error
  contract); the backward kernels (`tf_core_conv2d_input_backward`,
  `tf_core_conv2d_weight_backward`) join in D4/D5.
- **`cpp/src/pooling.cpp`** — `tf_core_maxpool2d_forward`,
  `tf_core_maxpool2d_backward`. **D8 shipped** the internal
  `tf::maxpool2d_forward_contiguous` (pure arithmetic, hidden symbol,
  declared in the focused internal header
  **`cpp/include/tf_pooling_internal.h`**) *and*, in the same file, the
  exported guarded `tf_core_maxpool2d_forward` wrapper; the backward
  scatter kernel and `tf_core_maxpool2d_backward` join in D9. Its
  file-local checked-arithmetic/offset helpers are deliberately kept
  separate from conv2d.cpp's rather than promoted to a shared header, so
  each compute unit stays self-contained (below).

The internal compute kernels are exercised by dependency-free CTest
binaries under **`cpp/tests/`** (e.g. `test_conv2d_forward.cpp`), built
only when `-DTF_BUILD_TESTS=ON`; each compiles the internal source it
tests directly, because the shared library exports only `TF_EXPORT`
symbols.

Guidelines:

- **No monolithic kernel file** — conv and pooling stay separate, as
  elementwise/reduction/matmul already are.
- **Autograd stays in Python** — these files hold pure arithmetic only.
- **No internal indexing helper is exported** — any shared row-major
  offset helper (e.g. `nchw_offset(n,c,h,w, C,H,W)`) is a **file-local
  `static`/anonymous-namespace inline**, not a `TF_EXPORT`.
- **No duplicated validation / shape math** — shape/output-size math lives
  in Python (`backends/cpp.py`, reusing `numel`, `row_major_strides`,
  and a new `conv_output_shape` helper); the kernels receive final
  dimensions.
- **Optional shared spatial helper:** if conv and pooling end up sharing a
  small `tf::conv_output_dim(size, kernel, stride, pad)` inline, it may go
  in `tf_internal.h` under `namespace tf` (internal, never ABI). Preferred
  default: keep such helpers file-local until a second user actually
  appears, avoiding premature shared surface.

---

## 15. Python source organization

Following the existing conventions (small focused modules, one concept per
file):

| Layer | Location |
|---|---|
| ctypes symbol registration (argtypes, `_CHECKED_KERNELS`) | `src/tensorforge/backends/cpp.py` (`_load_library`) |
| Output-shape helper (`conv_output_shape`, `_spatial_pair`) | `src/tensorforge/backends/cpp.py` (pure Python, no library) |
| `NativeTensorCore` forward/backward wrappers | `src/tensorforge/backends/cpp.py` (`NativeTensorCore` methods) |
| `NativeTensor.conv2d` / `.maxpool2d` graph ops | `src/tensorforge/experimental/native_tensor.py` |
| `NativeConv2d` | `src/tensorforge/experimental/native_conv2d.py` (new) |
| `NativeMaxPool2d` | `src/tensorforge/experimental/native_maxpool2d.py` (new) |
| `NativeFlatten` | `src/tensorforge/experimental/native_flatten.py` (new) |
| Public exports (`__all__`) | `src/tensorforge/experimental/__init__.py` |
| Backend capability registry | `src/tensorforge/backends/cpp.py` (`AUTOGRAD_OPS`, `NATIVE_MODULES`, `UNSUPPORTED`, `backend_info`) — updated **as each feature lands**, never before |

The autograd ops (`conv2d`, `maxpool2d`) join `AUTOGRAD_OPS`/
`TENSOR_CORE_OPS` when D6/D9 land; the modules join `NATIVE_MODULES` when
D7/D10/D1 land; `conv2d`/`maxpool2d`/`flatten` leave `UNSUPPORTED` only as
their milestones complete. The guardrail test
(`tests/test_cpp_backend_info.py`) keeps these lists honest.

---

## 16. Testing contract

Tolerances: `np.allclose(atol=1e-9)` for hand-computed and parity cases;
finite-difference gradient checks with `eps≈1e-6` and `atol≈1e-5`
(central differences), matching `NativeLinear`/`NativeMSELoss` tests. All
example/proof code uses **fixed seeds**. Native tests **skip** when the
backend is unbuilt (existing pattern).

### `tests/test_native_flatten.py`

Forward shapes (4-D→2-D, already-2-D preserved); owning result on both the
contiguous and non-contiguous paths (independent of the input's lifetime —
see §3.1); rank < 2 rejected; backward reshape correctness;
`NativeSequential` integration (Pool→Flatten→Linear).

### `tests/test_native_conv2d.py` (forward)

Hand-computed 1-channel example; multi-input-channel; multi-output-
channel; bias / no bias; padding; stride > 1; rectangular input;
rectangular kernel; tuple stride/padding; **non-contiguous input under
Policy B** (transposed/narrowed input gives the same result as its
contiguous copy); **stable-framework parity** (`tensorforge.nn.Conv2d` on
the same data, to tolerance); invalid shapes (wrong channels, non-4-D,
kernel doesn't fit); allocation failure (`tf_test_arm_alloc_failure` ⇒
`MemoryError`, no partial state); C ABI error propagation.

### `tests/test_native_conv2d_backward.py`

Central finite differences for `x`, `W`, `b` **independently**;
hand-computed small gradients; shared-parameter accumulation; only a
subset of parents requiring grad (freeze weight, check input grad only,
etc.); **stale parameter mutation** (mutate weight after forward ⇒
stale-graph `RuntimeError`); non-contiguous upstream gradient under Policy
B; failure atomicity (armed allocation failure mid-backward leaves grads
rolled back).

### `tests/test_native_maxpool2d.py`

Hand-computed forward; overlapping windows (`stride < kernel`); tie
behavior (first-occurrence winner); padding (`-inf`, padded cell never
wins, padding-winner gradient dropped); stride; backward scatter
correctness; repeated winner positions accumulate; **stable-framework
parity**; saved-winner lifetime (retained vs freed graph); allocation
failure; stale-graph behavior where applicable.

### `tests/test_native_cnn_training.py` (module + integration)

Parameter registration; state dicts; checkpoint round trips
(save→load→resume); `NativeSequential` traversal with Conv/Pool/Flatten
slots; deterministic CNN training (fixed seeds); resume equivalence
(exact continuation); existing Phase A–C regression coverage stays green.
**As implemented (D11):** the loss curve is deliberately *not* asserted
monotonic — Adam overshoots early on this task — so the guardrails are a
final loss and a final/initial ratio each far below the observed values,
plus "predictions moved closer" and "both trainable layers changed".

### `tests/test_native_phase_d.py` (cross-cutting, D12)

The completion file, complementing the per-component suites with the
invariants that span several Phase-D components at once: the full module
stack (shapes, order, state keys, dtype/device, non-contiguous input);
one graph producing every gradient; graph-resource release and
`retain_graph` across the mixed graph; shared modules/inputs accumulating
with identity-deduplicated traversal; the Conv2d and MaxPool2d versioning
contracts meeting in one backward; state/checkpoint integration for a
model containing both operations (including exact `NativeAdam`/`NativeSGD`
resume and Phase-C compatibility); cross-layer failure atomicity under
injected allocation failure; resource lifetime across the whole stack;
and the final capability boundary, including that the support matrix and
the backend inventories agree.

Every new test protects a real behavioral guarantee, not an arbitrary
string. **D12 also replaced the milestone-era documentation guardrails**
— which pinned transient wording, banned legitimate feature tokens from
whole sections, or asserted that a not-yet-written file was absent — with
durable semantic checks: the docs, the public exports, and the backend
registry must *agree*, the artifacts they reference must exist, and
genuinely absent capabilities must stay in `UNSUPPORTED`.

---

## 17. Benchmark contract

Small, **honest**, no performance assertions (the project's rule —
`benchmarks/` never asserts a speedup). A new
`benchmarks/benchmark_native_cnn.py` (or an addition to the existing
autograd benchmark) will separately time and label:

- **Conv2d forward** (native vs the stable NumPy reference).
- **Conv2d forward + backward** (all three gradients).
- **MaxPool2d forward + backward**.
- **End-to-end tiny CNN step** (Conv → ReLU → Pool → Flatten → Linear →
  MSE, forward + backward + optimizer step).
- A **stable NumPy reference** comparison where meaningful (correctness
  gate + wall-clock, no assertion that native is faster).

Reported modes distinguish **reference/correctness**, **native call
overhead**, **kernel execution**, and **end-to-end training** — the same
four-way honesty as the Phase-B autograd benchmark.

**Status (D12 — delivered as `benchmarks/benchmark_native_cnn.py`).**
Three cases (`conv2d`, `maxpool2d`, `cnn`) × five modes:
`forward_native` (no graph — native execution plus wrapper/ctypes
overhead), `forward_graph` (the same forward with grad-tracking operands,
isolating graph-construction cost), `forward_backward_fresh` (fresh graph
plus one default `backward()`, including the winner-buffer release),
`training_step` (the full D11 iteration: forward, loss, backward,
`step()`, `zero_grad()`, cleanup — `cnn` only), and `stable_forward` (the
stable `tensorforge.nn` equivalent on the same shapes). Warm-up plus
repeated timed batches, median/min/max per iteration, a correctness gate
before any timing, `--smoke`/`--json`/`--case`/`--mode` CLI, and no
timing assertion anywhere — the report states explicitly that both lines
are naive implementations and that the comparison is neither a speed
claim nor a cross-framework, GPU, or scalability result.
`tests/test_native_cnn_benchmark.py` proves it runs and reports the
expected fields without pinning a single duration.

---

## 18. Milestone dependency map (D0–D12)

Each milestone is small, tested, documented, and independently
mergeable. Milestones are **not** merged to reduce the count.

### D0 — Native CNN architecture contract *(this document)*
- **Scope:** design/contract only — this doc, roadmap Phase-D section,
  support-matrix upcoming section, doc guardrails.
- **Excludes:** all numerical code.
- **Files:** `docs/native_cnn_design.md`; edits to `docs/roadmap.md`,
  `docs/native_support_matrix.md`, `README.md`, `tests/test_docs.py`.
- **Tests:** documentation guardrails.
- **Acceptance:** design locked; full suite green; nothing advertised as
  implemented.
- **Dependencies:** none. **Risks:** ambiguity leaking into later
  milestones (mitigated by this doc's specificity).

### D1 — `NativeFlatten` — **implemented**
- **Scope:** batch-preserving flatten via existing `reshape`/
  `contiguous_copy` autograd; new module + exports.
- **Excludes:** `start_dim`/`end_dim`; any kernel.
- **Files:** `native_flatten.py`; `experimental/__init__.py`;
  `tests/test_native_flatten.py`; support-matrix + `NATIVE_MODULES`
  update.
- **Tests:** §16 Flatten group.
- **Acceptance:** forward/backward correct; **owning result** (the D1
  refinement of the D0 view/copy rule — see §3.1); Sequential integration.
- **Dependencies:** existing reshape autograd. **Risks:** contiguity edge
  cases (mitigated by copy-then-reshape) — resolved.
- **Status:** **done.** `NativeFlatten` ships as a parameter-free,
  buffer-free module Python-composed from `reshape`/`contiguous_copy`;
  no new C++ kernel, C ABI symbol, ctypes declaration, autograd
  primitive, checkpoint schema, dtype, or dispatch. Convolution and
  pooling remain unimplemented.

### D2 — Conv2d CPU forward kernel — **implemented**
- **Scope:** the **internal** CPU float64 compute kernel only —
  `tf::conv2d_forward_contiguous` (direct nested loops, nullable bias,
  symmetric zero padding by skipping out-of-bounds coordinates). **Not
  exported**: the `extern "C" tf_core_conv2d_forward` wrapper,
  fault-injection/ABI error path, ctypes registration, and Core/Python
  validation + allocation are **D3**, not D2.
- **Excludes:** any backward; the C ABI export; the wrapper/module/
  autograd; anything reachable from Python.
- **Files:** `cpp/src/conv2d.cpp` (definition),
  `cpp/include/tf_conv2d_internal.h` (internal, non-ABI declaration),
  `cpp/tests/test_conv2d_forward.cpp` (dependency-free CTest binary,
  built via the new `TF_BUILD_TESTS` CMake option; it compiles
  `conv2d.cpp` directly since the symbol is hidden).
- **Tests:** hand-computed C++ cases (single/multi channel, multi
  out-channel, bias, padding, stride, rectangular, combined, batch,
  negatives/fractions, null-bias, immutability, determinism) plus a
  stable-`tensorforge.nn.Conv2d` parity case (values generated once,
  compared to `1e-9`). No Python at runtime.
- **Acceptance:** correct against hand-computed values and stable parity;
  Release **and** Debug builds warning-clean.
- **Dependencies:** D0. **Risks:** index arithmetic bugs (mitigated by
  explicit signed-`int64_t` indexing and skip-on-out-of-bounds) —
  resolved. Full ASan/UBSan validation remains the D12 checkpoint.
- **Status:** **done** (internal compute kernel). Conv2d is still
  **unreachable from Python**; the C ABI export is D3.

### D3 — Conv2d `NativeTensorCore` wrapper + ctypes — **implemented**
- **Scope:** the exported C ABI wrapper `tf_core_conv2d_forward`
  (exception-guarded, self-validating; contiguous storage is a *caller
  precondition* — no stride metadata crosses the ABI, so it bounds-checks
  spans but does not inspect logical contiguity); its ctypes argtypes +
  `errcheck` registration; and `NativeTensorCore.conv2d_forward` (Policy-B
  copy-then-compute contiguity, output allocation, full shape validation).
- **Excludes:** autograd graph; module; any backward.
- **Files:** `cpp/src/conv2d.cpp` (the exported wrapper joins the D2
  internal kernel); `cpp/CMakeLists.txt` (the D2 CTest now also compiles
  `error.cpp`, since the wrapper references the thread-local error slot);
  `backends/cpp.py` (argtypes, `_CHECKED_KERNELS`, `_spatial_pair`,
  `conv_output_shape`, `NativeTensorCore.conv2d_forward`, `TENSOR_CORE_OPS`
  gains `conv2d_forward`); `tests/test_native_conv2d_core.py`.
- **Tests:** hand-computed forward, bias/no-bias, multi-channel, batch,
  padding, stride, rectangular, tuple stride/padding, stable parity,
  determinism; output shape/stride/offset/ownership/metadata contract;
  Policy-B non-contiguous input/weight/bias parity and temporary-copy
  closure; the full validation surface; raw-ABI rejection (null handle,
  output-dim mismatch, storage-span overflow) as `ValueError`; injected
  allocation failure as an atomic `MemoryError`; capability separation.
- **Acceptance:** forward-only `NativeTensorCore` conv works end to end.
  **Done.**
- **Dependencies:** D2.

**Actual exported C ABI signature (D3).** A `void` function signalling
success/failure only through the thread-local error slot (no C++ exception
crosses the boundary), registered in `_CHECKED_KERNELS`:

```c
void tf_core_conv2d_forward(
    const void* input_handle,  int64_t input_offset,
    const void* weight_handle, int64_t weight_offset,
    const void* bias_handle,   int64_t bias_offset,   // bias_handle == NULL => no bias
    void*       output_handle,                         // caller-allocated, offset 0
    int64_t batch, int64_t in_channels,
    int64_t input_height, int64_t input_width,
    int64_t out_channels,
    int64_t kernel_height, int64_t kernel_width,
    int64_t stride_height, int64_t stride_width,
    int64_t pad_height,    int64_t pad_width,
    int64_t output_height, int64_t output_width);
```

The wrapper validates at the boundary before running the D2 noexcept
kernel: null required handles, non-positive extents/kernel/stride, negative
padding/offsets, an output shape disagreeing with the floor formula, and —
using overflow-checked int64 arithmetic — any storage span that would fall
outside its allocation are all rejected with `TF_ERROR_INVALID`
(`ValueError` in Python). No shape/stride arrays cross the ABI: the raw
contract is **contiguous storage plus a per-operand offset**. Because no
stride metadata is supplied, the ABI **cannot and does not inspect logical
contiguity** — row-major NCHW/OIHW/1-D layout is a *caller precondition* it
trusts, and it interprets each `(handle, offset, dims)` span as canonical
contiguous data. What it independently guarantees is that that span lies
inside the allocation (plus every other metadata check above); ensuring the
data is genuinely contiguous is the Core layer's job (Policy B copies any
non-contiguous operand before the call). The division of responsibility is
therefore explicit: **`NativeTensorCore.conv2d_forward` accepts
non-contiguous Core operands and makes owning contiguous copies; the raw C
ABI only ever sees — and only ever promises to bounds-check — contiguous
spans.**

**Actual Core API (D3).**
`NativeTensorCore.conv2d_forward(weight, bias=None, *, stride=1,
padding=0)` — `self` is the `(N, C, H, W)` NCHW input, `weight` is
`(O, C, kh, kw)` OIHW, `bias` is an optional `(O,)` core. `stride`/
`padding` are an int or a 2-element `(height, width)` pair (bools
rejected, kernel/stride ≥ 1, padding ≥ 0). It validates dtype/device,
ranks, channel compatibility, and bias length; computes and checks
`out_h, out_w` in **Python ints** (arbitrary precision — the shape math
cannot overflow) before allocating anything; then, per **Policy B**,
materializes any non-contiguous input/weight/bias into a private owning
contiguous copy (offset 0) that is **closed as soon as the native call
returns**, passing already-contiguous operands (offset included) straight
through. The result is a fresh **owning** row-major contiguous
`(N, O, out_h, out_w)` CPU float64 core, valid after every input is closed
and independent of their storage. Failure at any stage allocates no output
and leaks no temporary. It is **forward-only and autograd-unaware**: the
differentiable `NativeTensor.conv2d` primitive (D6) and the `NativeConv2d`
module (D7) are later milestones and stay unsupported, so the backend
registry advertises `conv2d` as unsupported while listing the Core-level
`conv2d_forward` in `TENSOR_CORE_OPS`.

### D4 — Conv2d input-gradient kernel — **implemented (internal)**
- **Scope:** the **internal** CPU float64 input-gradient compute kernel
  only — `tf::conv2d_input_backward_contiguous` (direct nested-loop
  scatter-add, the adjoint of the forward cross-correlation; symmetric
  zero padding by skipping out-of-bounds coordinates; **zero-initializes
  its own output span** so the caller need not pre-zero it; deterministic
  `n → o → i → j → c → p → q` accumulation into the `(N,C,H,W)` grad-input
  span). Bias does not affect the input gradient, so the kernel neither
  receives nor reads a bias.
- **Excludes — deferred:** the `extern "C" tf_core_conv2d_input_backward`
  export, ctypes registration, the `NativeTensorCore` backward wrapper,
  fresh-storage allocation, error mapping, and any autograd wiring are
  **D6** (which exposes all Conv2d backward C ABI/Core surface and the
  `NativeTensor.conv2d` graph after D5 finishes the remaining gradient
  kernels). D4 adds **nothing reachable from Python**. Also excludes the
  weight- and bias-gradients (D5).
- **Files:** `cpp/src/conv2d.cpp` (definition, joining the forward kernel
  and the D3 exported wrapper), `cpp/include/tf_conv2d_internal.h`
  (internal, non-ABI declaration + the zero-init/accumulation/precondition
  contract), `cpp/tests/test_conv2d_input_backward.cpp` (dependency-free
  CTest binary compiling `conv2d.cpp` + `error.cpp` directly),
  `cpp/CMakeLists.txt` (the new `conv2d_input_backward` CTest target).
- **Internal signature:**
  `void tf::conv2d_input_backward_contiguous(const double* grad_output,
  const double* weight, double* grad_input, int64_t batch, in_channels,
  input_height, input_width, out_channels, kernel_height, kernel_width,
  stride_height, stride_width, pad_height, pad_width, output_height,
  output_width) noexcept` — reads `grad_output` (NCHW) and `weight` (OIHW)
  without mutation, writes only the `grad_input` (NCHW) span, allocates
  nothing.
- **Tests:** 19 dependency-free C++ cases — hand-computed 1×1/2×2 scatter,
  overlapping-window accumulation, multi-input-channel, multi-output-
  channel accumulation, batch, stride (untouched positions stay zero),
  symmetric padding, rectangular input, rectangular kernel, tuple
  asymmetric stride/padding, combined stride+padding, negative/fractional
  values, output zero-init from garbage, input immutability, determinism,
  stable-`tensorforge.nn.Conv2d` parity (embedded, `1e-9`), central
  finite-difference validation against the internal forward objective
  (`eps 1e-5`, `atol 1e-6`), and no-bias equivalence.
- **Acceptance:** `∂L/∂x` correct against hand-computed values, stable
  parity, and finite differences; Release **and** Debug CTests warning-
  clean. **Done** (internal kernel). **Dependencies:** D2, D3.

### D5 — Conv2d weight- and bias-gradient — **implemented (internal weight kernel; bias-grad reduction sequence locked & validated)**
- **Scope:** the **internal** CPU float64 weight-gradient compute kernel
  `tf::conv2d_weight_backward_contiguous` (direct nested-loop accumulation
  pairing each upstream value with the input pixel it multiplied in the
  forward; symmetric zero padding by skipping out-of-bounds coordinates;
  **zero-initializes its own `(O,C,kh,kw)` output span**; deterministic
  `n → o → i → j → c → p → q` accumulation), **plus** the locked, validated
  **bias-gradient reduction sequence** — computed with **no dedicated C++
  kernel** by reusing the existing native `sum` reduction (see §7.3).
- **Excludes — deferred:** the `extern "C" tf_core_conv2d_weight_backward`
  export, ctypes registration, the `NativeTensorCore` backward wrapper,
  fresh-storage allocation, error mapping, and the `NativeTensor.conv2d`
  autograd node are all **D6**. D5 adds **nothing reachable from Python**
  as a Conv2d backward operation (the bias-grad proof reuses the existing
  public `sum` op and adds no new capability).
- **Files:** `cpp/src/conv2d.cpp` (weight-gradient definition),
  `cpp/include/tf_conv2d_internal.h` (internal, non-ABI declaration + the
  zero-init/accumulation/precondition contract),
  `cpp/tests/test_conv2d_weight_backward.cpp` (dependency-free CTest),
  `cpp/CMakeLists.txt` (the new `conv2d_weight_backward` CTest target),
  `tests/test_native_conv2d_gradient_contract.py` (the bias-grad
  reduction-sequence proof). **No `backends/cpp.py` change** — no Python
  surface, no inventory change.
- **Internal weight-gradient signature:**
  `void tf::conv2d_weight_backward_contiguous(const double* grad_output,
  const double* input, double* grad_weight, int64_t batch, in_channels,
  input_height, input_width, out_channels, kernel_height, kernel_width,
  stride_height, stride_width, pad_height, pad_width, output_height,
  output_width) noexcept` — reads `grad_output` (NCHW) and `input` (NCHW)
  without mutation, writes only the `grad_weight` (OIHW) span, allocates
  nothing, reads no bias.
- **Bias-gradient sequence (locked for D6):** over grad_output
  `(N, O, oh, ow)`, `sum(axis=0) → (O, oh, ow)`, then
  `sum(axis=1) → (O, ow)`, then `sum(axis=1) → (O,)` — three deterministic
  single-axis native `sum` reductions (`keepdims=False`), each a fresh
  owning contiguous core, the two intermediates closed after use. Reads
  **only** `grad_output` (never input or weight), so a bias-only backward
  records **no** input/weight version snapshot (§8).
- **Tests:** 21 dependency-free C++ weight-gradient cases (hand-computed
  1×1/2×2, batch and output-position accumulation, multi-input/output
  channels, stride, symmetric padding, rectangular input/kernel, asymmetric
  stride/padding, combined stride+padding, negative/fractional, output
  zero-init from garbage, immutability, determinism, stable parity `1e-9`,
  central finite differences `eps 1e-5`/`atol 1e-6`, bias independence,
  no-bias equivalence, and padding-only boundary contributions — the last
  group cross-checked against an independent explicit-zero padded
  materialization oracle), plus the Python bias-grad reduction proof
  (hand-computed, multi-batch/channel, negative/fractional, stable-Conv2d
  parity, grad_output immutability, intermediate release, no-new-capability).
- **Acceptance:** `∂L/∂W` correct against hand-computed values, stable
  parity, and finite differences; `∂L/∂b` reduction sequence produces the
  correct `(O,)` values; Release **and** Debug CTests warning-clean.
  **Done** (internal weight kernel + bias reduction contract).
  **Dependencies:** D4.

### D6 — Conv2d backward exposure + `NativeTensor` autograd integration — **implemented**
- **Scope:** the exported, exception-guarded **backward C ABI wrappers**
  `tf_core_conv2d_input_backward` / `tf_core_conv2d_weight_backward` (over
  the D4/D5 internal kernels); their ctypes + `errcheck` registration; the
  **Core backward methods** `NativeTensorCore.conv2d_input_backward` /
  `conv2d_weight_backward` (Policy-B contiguity, output allocation, full
  shape validation); the **bias gradient composed from existing native
  `sum` reductions** (no C ABI symbol, no kernel); and the differentiable
  **`NativeTensor.conv2d(...)`** fused primitive — graph node, deterministic
  parent set, conditional version recording (§8), and the Python-managed
  backward closure calling the Core backward ops (input/weight) and the
  reduction chain (bias).
- **Excludes — deferred:** the `NativeConv2d` module (D7) and all of
  pooling (D8–D10). No bias-gradient C ABI symbol is added.
- **Files:** `cpp/src/conv2d.cpp` (the two exported backward wrappers, over
  the existing internal kernels); `backends/cpp.py` (argtypes,
  `_CHECKED_KERNELS`, the two Core backward methods; `TENSOR_CORE_OPS` gains
  `conv2d_input_backward`/`conv2d_weight_backward`, `AUTOGRAD_OPS` gains
  `conv2d`, `UNSUPPORTED` swaps the bare `conv2d` op for the still-absent
  `NativeConv2d` module); `experimental/native_tensor.py`
  (`NativeTensor.conv2d`); `tests/test_native_conv2d_backward_core.py`,
  `tests/test_native_conv2d_autograd.py`.
- **Exported ABI signatures (D6).** Both `void`, guarded, in
  `_CHECKED_KERNELS`, taking **no stride arrays** (contiguous caller
  precondition, span-bounds validated exactly like the D3 forward wrapper):
  ```c
  void tf_core_conv2d_input_backward(
      const void* grad_output_handle, int64_t grad_output_offset,
      const void* weight_handle,      int64_t weight_offset,
      void*       grad_input_handle,   // caller-allocated (N,C,H,W), offset 0
      int64_t N, C, H, W, O, kh, kw, sh, sw, ph, pw, out_h, out_w);
  void tf_core_conv2d_weight_backward(
      const void* grad_output_handle, int64_t grad_output_offset,
      const void* input_handle,       int64_t input_offset,
      void*       grad_weight_handle,  // caller-allocated (O,C,kh,kw), offset 0
      int64_t N, C, H, W, O, kh, kw, sh, sw, ph, pw, out_h, out_w);
  ```
- **Core backward APIs (D6).**
  `NativeTensorCore.conv2d_input_backward(weight, *, input_shape, stride=1,
  padding=0)` (receiver = grad_output → grad_input `(N,C,H,W)`) and
  `NativeTensorCore.conv2d_weight_backward(input, *, weight_shape, stride=1,
  padding=0)` (receiver = grad_output → grad_weight `(O,C,kh,kw)`). Each
  validates ranks/channels/spatial relationships and the recomputed
  grad_output shape before allocating, copies any non-contiguous operand
  (Policy B) into a private core closed after the call, returns a fresh
  **owning** row-major contiguous CPU-float64 result, closes the output on
  failure, and never mutates its inputs.
- **`NativeTensor.conv2d` API (D6).** `conv2d(weight, bias=None, *,
  stride=1, padding=0)` — `self` NCHW, `weight` OIHW NativeTensor, `bias`
  `None` or a rank-1 NativeTensor; int/2-tuple stride & padding (bools
  rejected); CPU float64 open NativeTensors only (no stable `Tensor`, no
  implicit conversion). Forward reuses `conv2d_forward`; returns a fresh
  owning `(N, O, out_h, out_w)` NativeTensor.
- **Graph ownership & parent set.** One Python-managed node via `_from_op`
  with parents `(input, weight)` or `(input, weight, bias)` (deterministic
  order). `requires_grad` is the OR of parents; when none requires grad the
  op returns a plain forward leaf with no graph. Only small ints
  (`input_shape`, `weight_shape`, stride, padding) live in the closure — **no
  forward output or full operand values are duplicated into Python**; the
  backward rereads the parents' live values, and graph ownership stays in
  Python (never moves into C++). Repeated-backward / one-shot-free /
  `retain_graph` follow the existing engine unchanged.
- **Backward callbacks.** Input-grad runs iff `input.requires_grad` (rereads
  the weight value, via `conv2d_input_backward`); weight-grad runs iff
  `weight.requires_grad` (rereads the input value, via
  `conv2d_weight_backward`); bias-grad runs iff bias present and
  `bias.requires_grad` (reads only the upstream: `sum(0).sum(1).sum(1) →
  (O,)`, both intermediates closed, only the final core kept). Each
  contribution is fresh owning storage accumulated via `_accumulate_grad`;
  a non-contiguous upstream is copied by the Core layer.
- **Conditional version tracking (§8).** Built through the existing
  `_versioned_value_reads`: `weight` is recorded **iff `input.requires_grad`**
  (input-grad rereads it), `input` is recorded **iff `weight.requires_grad`**
  (weight-grad rereads it), and a **bias-only** backward records nothing — so
  mutating input or weight after forward never makes a bias-only backward
  raise, while a direct-parameter value an active callback rereads raises the
  deterministic stale-graph `RuntimeError`. Bias is never versioned.
- **Failure rollback.** The existing `backward()` snapshot/rollback engine
  is reused unchanged: gradients are staged against a per-node snapshot, so a
  mid-pass allocation failure (`MemoryError` via the status contract) rolls
  back every leaf gradient, commits nothing partially, frees no graph, leaves
  inputs open/unchanged, and a subsequent backward succeeds with the native
  error slot clear. The raw ABI never allocates or frees caller storage.
- **Tests:** 29 Core-backward cases (stable parity for input/weight grad;
  multi-channel/batch/padding/stride/rectangular; non-contiguous operands;
  output ownership/layout; validation; allocation failure; and direct-ctypes
  ABI checks — null handle, negative offset, invalid dims, output-shape
  mismatch, undersized span → `ValueError`) plus 33 autograd cases (forward +
  input/weight/bias gradient parity; finite differences; int/tuple
  stride/padding; all `requires_grad` combinations; no-grad graph avoidance;
  parent ordering; shared-graph accumulation; scalar loss through reductions;
  conditional version tracking incl. bias-only-ignores-mutation; explicit
  non-scalar / wrong-shape / non-contiguous gradient validation; failure
  rollback; lifetime / `zero_grad` / one-shot-free).
- **Acceptance:** end-to-end differentiable native conv correct against
  stable parity and finite differences; Release **and** Debug CTests
  warning-clean. **Done.** **Dependencies:** D5.

### D7 — `NativeConv2d` module — **Implemented**
- **Scope:** module (§9) — parameters, init, registration, state/
  checkpoint, repr, frozen support, `NativeSequential` fit.
- **Files:** `native_conv2d.py`; `experimental/__init__.py`;
  `NATIVE_MODULES` + support-matrix update;
  `tests/test_native_conv2d_module.py`.
- **Tests:** registration/state/checkpoint/Sequential + forward parity.
- **Acceptance:** trainable conv layer. **Dependencies:** D6.
- **Status (D7 — implemented):** `NativeConv2d(in_channels, out_channels,
  kernel_size, stride=1, padding=0, bias=True, *, seed=None,
  requires_grad=True)` ships in
  `src/tensorforge/experimental/native_conv2d.py`, exported from the
  experimental package and listed in `NATIVE_MODULES` (removed from
  `UNSUPPORTED`). Every argument is validated **before any native
  allocation**: `in_channels`/`out_channels` are real positive ints (bools
  rejected → `TypeError`, non-positive → `ValueError`);
  `kernel_size`/`stride` (≥ 1) and `padding` (≥ 0) are normalized to
  two-element `(height, width)` tuples through the native `_spatial_pair`
  helper (int or 2-pair, bools/malformed/non-int rejected); `bias` and
  `requires_grad` are real bools; `seed` is an int or `None`. **Normalized
  attributes:** `in_channels`, `out_channels`, `kernel_size`, `stride`,
  `padding` (spatial ones as 2-tuples), `weight`, `bias`. **Parameter
  shapes:** `weight` is `(out_channels, in_channels, kh, kw)` (OIHW), `bias`
  is `(out_channels,)` when enabled else the attribute reads `None`;
  registration order is `weight` then `bias` (deterministic
  `["weight", "bias"]` keys, nested as `"0.weight"`/`"0.bias"` in a
  `NativeSequential`). No buffers. **Initialization:** deterministic uniform
  fan-in — `fan_in = in_channels * kh * kw`, `bound = 1/sqrt(fan_in)`,
  sampled from `[-bound, +bound]` via a **local**
  `numpy.random.default_rng(seed)` (global RNG untouched; equal seeds →
  identical values; no graph built; versions start at 0). **Forward**
  validates an open 4-D NCHW input with `shape[1] == in_channels` and
  matching float64/cpu, then delegates to `input.conv2d(self.weight,
  self.bias, stride=self.stride, padding=self.padding)` — no numerical or
  autograd logic is duplicated, and the module adds no graph node.
  Non-contiguous inputs ride the existing Conv2d Policy-B path. **Autograd,
  state, checkpoint, and optimizers** are inherited unchanged: input/weight/
  bias gradients through the D6 backward, `requires_grad=False` freezes both
  parameters (input still differentiates), stale-graph detection inherited,
  `state_dict()` independent snapshots with atomic identity-preserving
  `load_state_dict()`, the existing pickle-free checkpoint format (no schema
  change), and both `NativeSGD`/`NativeAdam`. Verified by
  `tests/test_native_conv2d_module.py`.

### D8 — MaxPool2d forward + winner-index contract — **implemented**
- **Scope:** the internal CPU float64 forward compute kernel
  `tf::maxpool2d_forward_contiguous` (hidden symbol, declared in the new
  focused internal header `cpp/include/tf_pooling_internal.h`); the
  exported, exception-guarded `tf_core_maxpool2d_forward` wrapper; its
  ctypes/`errcheck` registration; and the Core forward
  `NativeTensorCore.maxpool2d_forward` plus the private
  `_maxpool2d_forward_with_winners` helper that keeps the **saved winner
  buffer** (§12).
- **Excludes — deferred:** the pooling **backward** scatter kernel, a
  backward C ABI symbol, `NativeTensor.maxpool2d`, MaxPool2d autograd
  (all **D9**), and `NativeMaxPool2d` (**D10**). D8 adds no public winner
  API, no integer dtype, and no CNN training work.
- **Files:** `cpp/src/pooling.cpp`, `cpp/include/tf_pooling_internal.h`,
  `cpp/tests/test_maxpool2d_forward.cpp`, `cpp/CMakeLists.txt` (the new
  `maxpool2d_forward` CTest target), `backends/cpp.py` (argtypes,
  `_CHECKED_KERNELS`, the Core methods, `TENSOR_CORE_OPS` gains
  `maxpool2d_forward`), `tests/test_native_maxpool2d_core.py`.
- **Internal kernel signature:**
  `void tf::maxpool2d_forward_contiguous(const double* input, double*
  output, double* winners, int64_t batch, channels, input_height,
  input_width, kernel_height, kernel_width, stride_height, stride_width,
  pad_height, pad_width, output_height, output_width) noexcept` — direct
  nested loops in `n → c → i → j → p → q` order, reads the input without
  mutation, writes only the caller-owned output and winner spans (both
  fully defined, no pre-initialization needed), allocates nothing, and
  materializes no padded input (out-of-bounds coordinates are computed in
  signed `int64_t`).
- **Exported ABI signature (D8).** A `void` function signalling
  success/failure only through the thread-local error slot, registered in
  `_CHECKED_KERNELS`:
  ```c
  void tf_core_maxpool2d_forward(
      const void* input_handle, int64_t input_offset,
      void*       output_handle,    // caller-allocated (N,C,oh,ow), offset 0
      void*       winners_handle,   // caller-allocated (N,C,oh,ow), offset 0
      int64_t batch, int64_t channels,
      int64_t input_height,  int64_t input_width,
      int64_t kernel_height, int64_t kernel_width,
      int64_t stride_height, int64_t stride_width,
      int64_t pad_height,    int64_t pad_width,
      int64_t output_height, int64_t output_width);
  ```
  No handle is nullable and **no stride arrays cross the ABI** — the raw
  contract is canonical contiguous storage plus the input offset, exactly
  like the Conv2d wrappers, so it bounds-checks each span but cannot and
  does not inspect *logical* contiguity (that stays the Core layer's
  Policy-B responsibility). It rejects with `TF_ERROR_INVALID`
  (`ValueError` in Python): null handles, non-positive extents/kernel/
  stride, negative padding or offset, an output shape disagreeing with the
  floor formula, a plane with `H*W > 2^53` (re-proving the winner-exactness
  bound in its own overflow-checked fixed-width arithmetic), shape products
  that overflow int64, and any input/output/winner span falling outside its
  allocation. It allocates and frees nothing and mutates only the
  caller-provided output/winner storage.
- **Core API and result representation (D8).**
  `NativeTensorCore.maxpool2d_forward(*, kernel_size, stride=None,
  padding=0)` returns the pooled **owning** contiguous
  `(N, C, out_h, out_w)` core alone (it closes the winner buffer
  deterministically); the private
  `NativeTensorCore._maxpool2d_forward_with_winners(...)` returns the
  `(output, winners)` pair — the representation D9's backward closure will
  hold. `stride=None` means `stride = kernel_size`; `kernel_size`/`stride`
  are int or 2-pairs ≥ 1 and `padding` int or 2-pair ≥ 0 (bools and
  malformed pairs rejected). Validation — open receiver, rank exactly 4,
  float64/cpu metadata, argument forms, `H*W ≤ 2^53`, the floor output
  shape with both extents ≥ 1, and int64-representable element counts —
  runs entirely in Python ints **before any allocation**. Non-contiguous
  input takes one explicit owning contiguous copy (Policy B) closed as soon
  as the native call returns; an already-contiguous input (even at a
  non-zero offset) is passed straight through.
- **Allocation order and cleanup.** Output first, winner buffer second,
  then the native call. A failure at any point closes every object already
  allocated (winner then output) and the temporary contiguous copy, returns
  no partial result, leaves the input open and unchanged, restores the
  native error slot for the next call, and returns live native storage to
  its baseline — never relying on garbage collection.
- **Winner ownership/lifetime.** Fresh owning row-major contiguous
  `(N, C, out_h, out_w)` float64 storage at offset 0, valid independently
  of the input's lifetime, closed exactly once by whoever holds it (the
  public Core method, or D9's backward closure). It never appears in
  `state_dict`, checkpoints, parameter/buffer traversal, `backend_info`, or
  any public NativeTensor API, and advertises no new dtype.
- **Tests:** 25 dependency-free C++ cases (simple/multi-window/multi-
  channel/batch, rectangular input and kernel, stride > 1, separate h/w
  stride, symmetric and separate h/w padding, combined stride+padding,
  negative and fractional values, unique maximum, equal-value tie,
  padding-vs-real-`-inf` tie, all-`-inf` valid values, completely padded
  windows, repeated winner offsets across overlapping windows, input
  immutability, full output/winner initialization from garbage,
  determinism, exact stable-`tensorforge.nn.MaxPool2d` parity for output
  *and* converted winners, winner exactness at a 200×200 plane's offset
  39999, and a focused NaN case) plus 83 Python Core cases (forward
  correctness and stable parity, output/winner contracts, Policy B,
  the validation surface, raw-ABI rejection, allocation/native-call failure
  atomicity with a live-storage baseline, and capability separation).
- **Acceptance:** correct pooled values **and** winners; Release **and**
  Debug CTests warning-clean. **Done** (raw + Core, forward-only).
  **Dependencies:** D3 (shares the Core patterns).

### D9 — MaxPool2d backward + autograd integration — **implemented**
- **Scope:** the internal CPU float64 scatter-add kernel
  `tf::maxpool2d_backward_contiguous`; the exported, exception-guarded
  `tf_core_maxpool2d_backward` wrapper (which **validates every winner
  value**); its ctypes/`errcheck` registration; the Core backward
  `NativeTensorCore.maxpool2d_backward`; the differentiable
  **`NativeTensor.maxpool2d`** graph node; and the smallest general
  mechanism the winner buffer needed to live exactly as long as the graph
  does (`_from_op(..., graph_resources=...)` +
  `NativeTensor._release_graph_resources`).
- **Excludes — deferred:** `NativeMaxPool2d` (**D10**), the CNN training
  proof (**D11**). No module, parameters, buffers, public
  `return_indices`, public winner tensor, integer dtype, or
  checkpoint-schema change.
- **Files:** `cpp/src/pooling.cpp`, `cpp/include/tf_pooling_internal.h`,
  `cpp/tests/test_maxpool2d_backward.cpp`, `cpp/CMakeLists.txt` (the new
  `maxpool2d_backward` CTest target), `backends/cpp.py` (argtypes,
  `_CHECKED_KERNELS`, the Core backward method, `TENSOR_CORE_OPS` gains
  `maxpool2d_backward`, `AUTOGRAD_OPS` gains `maxpool2d`, `UNSUPPORTED`
  swaps the bare `maxpool2d` op for the still-absent `NativeMaxPool2d`
  module), `experimental/native_tensor.py`,
  `experimental/native_parameter.py` (the `_from_op` override forwards the
  new keyword), `tests/test_native_maxpool2d_backward_core.py`,
  `tests/test_native_maxpool2d_autograd.py`.
- **Internal kernel signature:**
  `void tf::maxpool2d_backward_contiguous(const double* grad_output, const
  double* winners, double* grad_input, int64_t batch, channels,
  input_height, input_width, output_height, output_width) noexcept` — it
  **zero-initializes the whole grad_input span itself**, then scatters in
  deterministic `n → c → oh → ow` order: a `-1` winner drops its gradient,
  any other winner is converted (`ih = winner / W`, `iw = winner % W`) and
  accumulated with `+=` so overlapping windows sum. It reads no input
  value, recomputes no maximum, takes no kernel/stride/padding argument,
  allocates nothing, and mutates neither grad_output nor winners.
- **Exported ABI signature (D9).** `void`, guarded, in `_CHECKED_KERNELS`,
  with **no stride arrays and no window geometry**:
  ```c
  void tf_core_maxpool2d_backward(
      const void* grad_output_handle, int64_t grad_output_offset,
      const void* winners_handle,     int64_t winners_offset,
      void*       grad_input_handle,   // caller-allocated (N,C,H,W), offset 0
      int64_t batch, int64_t channels,
      int64_t input_height, int64_t input_width,
      int64_t output_height, int64_t output_width);
  ```
  Beyond the shared metadata checks (null handles, negative offsets,
  non-positive extents, `H*W ≤ 2^53`, overflow-checked shape products, and
  every span inside its allocation) it **validates each winner value**
  before the kernel runs: an entry must be exactly `-1.0`, or finite,
  non-negative, exactly integral (`floor(v) == v`), and `≤ H*W - 1`.
  Anything else — fractional, NaN, ±inf, below `-1`, or out of range — is
  rejected with `TF_ERROR_INVALID` (`ValueError`) and **nothing is rounded,
  truncated, or written**; grad_input is left untouched. It allocates and
  frees nothing.
- **Core API (D9).** `NativeTensorCore.maxpool2d_backward(winners, *,
  input_shape)` — the receiver is `grad_output` `(N, C, out_h, out_w)`,
  `winners` is the private saved-winner core of the same shape, and
  `input_shape` normalizes to a 4-tuple of positive ints whose `(N, C)`
  must match. It validates rank, float64/cpu metadata, shape agreement,
  and the `H*W ≤ 2^53` bound before allocating; copies a non-contiguous
  grad_output *or* winner core under Policy B (each temporary closed as
  soon as the call returns); returns a fresh **owning** row-major
  contiguous `(N, C, H, W)` core; and closes that output if the native
  call fails.
- **`NativeTensor.maxpool2d` API (D9).** `maxpool2d(*, kernel_size,
  stride=None, padding=0)` — keyword-only, over an open 4-D NCHW CPU
  float64 receiver, with the D8 argument contract unchanged
  (`stride=None` ⇒ `stride = kernel_size`; bools and malformed pairs
  rejected). Forward calls the private
  `_maxpool2d_forward_with_winners` Core path; the result is a fresh
  owning `(N, C, out_h, out_w)` NativeTensor. Exactly **one parent —
  `(input,)`** — and the closure captures only the private winner core and
  the input shape: no input values, no forward output, no graph state in
  C++.
- **Saved-winner graph ownership (the load-bearing D9 contract).** The
  winner core is handed to `_from_op` as a **graph resource**. When a graph
  is built the node owns it and releases it *exactly once*, at the same
  deterministic point the history is released. There are exactly **two
  deterministic release points**: the cleanup step of a successful default
  `backward()`, and an explicit `close()`. A graph object that is merely
  dropped reaches `close()` through `__del__`, which is the refcount/GC
  **fallback** — a safety net, not a guarantee this design leans on.
  When **no** parent requires grad, `_from_op` closes the buffer
  immediately — a no-grad forward leaves nothing behind — and if graph
  construction itself raises, `maxpool2d` closes both the winner buffer and
  the pooled output core before re-raising, so a failed `_from_op` leaks
  neither (ownership never transferred, and every close is idempotent, so
  nothing is closed twice). `retain_graph=True` skips the release, so the
  buffer stays valid for another pass; a **failed** backward never reaches
  the cleanup step, so the buffer survives and the graph stays retryable;
  and a repeated backward after a one-shot free raises the existing
  freed-graph `RuntimeError` without double-closing (the resource tuple is
  cleared before closing). This is the smallest reusable mechanism that
  fits the existing lifecycle — no autograd redesign, and no other
  operation is affected.
- **No version read (deliberate contrast with Conv2d).** Backward reads
  only the saved winners and the upstream, so `maxpool2d` records **no**
  expected version and is absent from `_versioned_value_reads`. Mutating a
  directly versioned input after the forward pass neither raises
  stale-graph nor changes gradient routing — the winners recorded at
  forward time still decide.
- **Failure rollback and retry.** The existing snapshot/rollback engine is
  reused unchanged: a mid-pass failure (an injected allocation failure, or
  a corrupted winner rejected at the ABI) restores every node's gradient
  reference, commits nothing partially, frees no graph, keeps the winner
  buffer alive, leaves the native error slot clear, and a subsequent
  backward through the same graph succeeds and accumulates normally.
- **Tests:** 21 dependency-free C++ cases (simple/multi-window scatter,
  overlapping accumulation, channels, batch, rectangular, stride through
  saved winners, padding sentinel, all-padding, repeated offsets, negative
  and fractional upstreams, zero-init from garbage, grad_output and winner
  immutability, determinism, forward→backward integration, exact
  stable-framework gradient parity, checked-wrapper rejection of seven
  malformed winner classes, and both offset boundaries) plus 52 Core
  backward and 55 autograd Python cases.
- **Acceptance:** differentiable native pooling correct against the stable
  reference; Release **and** Debug CTests warning-clean. **Done.**
  **Dependencies:** D8.

### D10 — `NativeMaxPool2d` module — **implemented**
- **Scope:** the parameter-free module (like `NativeReLU`) wrapping the
  D8/D9 `maxpool2d` operation, its public experimental export, and the
  capability-inventory update. **No** new kernel, C ABI symbol, ctypes
  declaration, pooling numerics, autograd callback, public winner access,
  parameter/buffer, `return_indices`, or checkpoint-schema change.
- **Excludes — deferred:** the deterministic end-to-end native CNN
  training + checkpoint-resume proof (**D11**) and the Phase-D
  cross-cutting/completion work (**D12**).
- **Files:** `experimental/native_maxpool2d.py` (new),
  `experimental/__init__.py` (import, `__all__`, docstring),
  `backends/cpp.py` (`NATIVE_MODULES` gains `NativeMaxPool2d`,
  `UNSUPPORTED` drops it), `tests/test_native_maxpool2d_module.py`.
- **Final constructor:** `NativeMaxPool2d(kernel_size, stride=None,
  padding=0)` — no other arguments in Phase D (no `dilation`,
  `ceil_mode`, `return_indices`, adaptive/average pooling, `device`,
  `dtype`, `requires_grad`, or `seed`).
- **Validation and normalized attributes:** `kernel_size`/`stride`
  (each ≥ 1) and `padding` (≥ 0) go through the native `_spatial_pair`
  helper (int or 2-element pair; bools, malformed lengths, and
  non-integer members rejected with `ValueError`); `stride=None` resolves
  to the normalized `kernel_size` (non-overlapping windows). The stored
  `kernel_size` / `stride` / `padding` attributes are always two-element
  `(height, width)` int tuples. Validation runs before any module state
  exists, and the module allocates **no native storage at all**, so a
  rejected argument cannot leak any.
- **Forward delegation:** `forward(input)` validates an open 4-D NCHW
  `NativeTensor` (rank error named by the module; the stable `Tensor`,
  NumPy arrays, lists, scalars, and closed tensors rejected) and returns
  `input.maxpool2d(kernel_size=…, stride=…, padding=…)` — the fresh owning
  tensor the operation produced. `NativeTensorCore` is never called
  directly, and **no** output-shape math, winner generation, tie/padding/
  `-inf`/NaN behavior, backward scatter, winner lifetime, or versioning
  logic is duplicated: all of it stays in D8/D9.
- **Parameterless and stateless:** no parameters, no buffers, no custom
  backward, no persistent forward state, and **no winner storage held by
  the module** — each forward's winners belong to that call's output
  graph, so repeated forwards produce independent graphs and independent
  winner resources. `parameters()`, `named_parameters()`, `buffers()`,
  `named_buffers()`, and `state_dict()` are empty; the layer contributes
  **no keys** to a parent module's state dictionary or to a checkpoint,
  and `train()`/`eval()` never change pooling numerics.
- **State and checkpoints:** `load_state_dict({})` succeeds; unexpected
  keys follow the existing strict/non-strict rules. Architecture lives in
  the constructor, never in tensor state. The existing v3.14 checkpoint
  path handles models containing the module unchanged (`format_version`
  1, pickle-free, no MaxPool-specific serializer, parameter identities
  preserved on load).
- **Sequential/optimizer integration:** drops into a `NativeSequential`
  beside `NativeConv2d`/`NativeReLU`/`NativeFlatten`/`NativeLinear`;
  parameter order and hierarchical names skip the pooling slot entirely
  (`"0.weight"`, `"0.bias"`, `"4.weight"`, `"4.bias"` in the Conv→ReLU→
  Pool→Flatten→Linear stack), and `NativeSGD`/`NativeAdam` ignore it
  naturally because it owns nothing trainable. Reusing one pooling
  instance in several slots is safe (identity-deduplicated traversal).
- **Tests:** 93 focused cases — constructor/attribute normalization and
  rejection, repr, empty parameter/buffer/state surface, forward
  correctness and stable parity, inherited autograd (ties, padding
  sentinel, overlapping accumulation, no version snapshot, retain/freed
  history, no-grad avoidance), Sequential composition, checkpoint and
  optimizer round trips, ownership/failure paths, and the capability
  split.
- **Acceptance:** pooling module in a `NativeSequential`. **Done.**
  **Dependencies:** D9.

### D11 — Deterministic native CNN training + checkpoint-resume proof — **implemented**
- **Scope:** `examples/native_cnn_training.py` — a small deterministic
  Conv→ReLU→Pool→Flatten→Linear→MSE CNN trained end to end through the
  native stack, plus an exact uninterrupted-versus-resumed equivalence
  proof over one existing pickle-free checkpoint, and the integration
  tests. **No** new kernel, C ABI symbol, ctypes declaration, tensor
  operation, loss, optimizer, dataset/dataloader abstraction, checkpoint
  schema, or capability-inventory entry: D11 is an *integration proof*,
  not new API surface.
- **Excludes — deferred:** the Phase-D cross-cutting guardrails,
  benchmarks, ASan/UBSan validation, documentation reconciliation, and the
  Phase-D completion declaration are all **D12**.
- **Files:** `examples/native_cnn_training.py` (new),
  `tests/test_native_cnn_training.py` (new), plus documentation updates.
  **No source module changed.**
- **Canonical model.** `NativeSequential(NativeConv2d(1, 2, 2, seed=0),
  NativeReLU(), NativeMaxPool2d(2), NativeFlatten(), NativeLinear(8, 1,
  seed=1))` — NCHW float64/cpu throughout: `(8, 1, 6, 6)` → conv
  `(8, 2, 5, 5)` → pool `(8, 2, 2, 2)` (the floor formula drops the odd
  row/column) → flatten `(8, 8)` → `(8, 1)`. Trainable parameters are
  exactly `["0.weight", "0.bias", "4.weight", "4.bias"]`; pooling, ReLU,
  and flatten contribute none.
- **Deterministic task.** Eight fixed 6×6 single-channel images (frozen
  Python literals, every value a half or a quarter so they are exactly
  representable) whose target is the strength of the strongest
  *bright-to-dark vertical edge*: `0.25 × max over every 2×2 window of
  (top-left + bottom-left − top-right − bottom-right)`, floored at 0. That
  rule is precisely a 2×2 convolution, a ReLU, and a maximum over
  positions, so it **requires the convolutional path** — the maximum is not
  a linear function of the pixels, and the flat image (target exactly 0)
  plus the dark-to-bright image keep it from degenerating into a pixel sum.
  Targets are frozen literals too, re-derived in the tests from the
  documented rule. Nothing is generated, shuffled, augmented, or loaded,
  and no random number is drawn during training.
- **Configuration.** `NativeAdam(lr=0.05)` — the canonical optimizer,
  because its persistent moments and per-parameter step counts make the
  resume proof meaningful — for `TOTAL_STEPS = 40`, split at
  `SPLIT_STEP = 15`. `NativeSGD` gets a smaller smoke case (two valid CNN
  steps with a decreasing loss), not a second convergence suite.
- **Training loop.** A fresh graph per step: forward → `NativeMSELoss` →
  record the scalar → `backward()` → `optimizer.step()` → close that
  step's prediction/loss tensors → `optimizer.zero_grad()`. The one-shot
  backward releases each step's graph *and* max-pooling's private winner
  buffer, so no graph is retained, no stale graph is reused after a step,
  parameter identities stay stable, and versions advance exactly once per
  step.
- **Loss-reduction proof.** Deterministic curve `0.771306 → 0.011085`
  (98.6% reduction; non-monotonic early, as Adam overshoots). The
  guardrails allow a final loss up to `0.10` and a final/initial ratio up
  to `0.10` — a ~7–9× margin over the observed values — plus "predictions
  moved at least 3× closer to the targets" and "both Conv2d and Linear
  parameters changed". Every trainable parameter receives a finite,
  meaningfully nonzero gradient on the **first** backward (shapes checked
  against the parameters), and input gradients work when the input
  requires grad.
- **Uninterrupted vs. resumed equivalence.** Path A trains 40 steps.
  Path B trains 15, saves model **and** optimizer state through
  `save_native_checkpoint` (with metadata) into a temporary directory,
  loads it into a **completely fresh** model/optimizer pair, and continues
  to 40. The two agree **exactly** (`==`, not a tolerance — the CPU
  float64 kernels are deterministic: fixed loop orders, no parallel
  reduction, no fast-math, and nothing random happens between checkpoint
  and resume): prefix losses, resumed suffix losses step-for-step, final
  loss, final predictions, every parameter value, every optimizer state
  entry, and the parameter ordering. The fresh model's parameter
  identities survive the load, and the loaded optimizer references those
  fresh parameters — it retains no object from the saved run.
- **Checkpoint boundaries.** The archive holds **only** persistent state:
  the four trainable parameters and the optimizer's own state. Its file
  list and manifest contain no pooling winners, no graph history, no
  gradients, and no transient outputs (asserted by substring scan);
  `format` and `format_version` (1) are unchanged, loading stays
  `allow_pickle=False` and atomic, and no CNN-specific serializer exists.
  Loading restores graph-free leaves, so training resumes by building
  fresh graphs.
- **Lifetime and failure paths.** With a warm-up and `gc.collect()`, the
  live native-storage count is **exactly constant** across repeated steps
  (no winner-buffer or graph accumulation); a checkpoint round trip leaks
  nothing; an injected allocation failure mid-step leaves every parameter
  value and version untouched, leaks no winner buffer, clears the native
  error slot, and a later step succeeds; loading into an incompatible
  architecture is rejected without mutating it, and the original
  checkpoint still loads afterwards.
- **Tests:** 36 focused integration cases, including the example smoke
  test (runs `main()`, checks it reports learning and resume equivalence,
  writes nothing into the repository, and needs no network).
- **Acceptance:** end-to-end native CNN training proof. **Done.**
  **Dependencies:** D1, D7, D10. The proof stays on **MSE** — no
  classification scope leaked in.

### D12 — Phase-D cross-cutting tests, benchmarks, docs, completion — **implemented (phase closed)**
- **Scope:** cross-cutting guardrails (`tests/test_native_phase_d.py`),
  benchmarks (§17), the **ASan/UBSan validation checkpoint**, the
  documentation/status reconciliation, and the Phase-D completion
  declaration. **No** new kernel, C ABI symbol, ctypes declaration,
  operation, module, loss, optimizer feature, dtype, or checkpoint schema:
  D12 is hardening and certification only.
- **Files:** `tests/test_native_phase_d.py`,
  `benchmarks/benchmark_native_cnn.py`,
  `tests/test_native_cnn_benchmark.py` (new); documentation updates across
  `native_cnn_design.md`, `native_support_matrix.md`, `roadmap.md`,
  `backend_experiments.md`, `architecture.md`, `project_summary.md`,
  `release_history.md`, `README.md`, `CLAUDE.md`, and the experimental
  package docstring; and the replacement of the milestone-era
  documentation guardrails in `tests/test_docs.py` (plus the three
  per-component files that asserted a not-yet-written artifact's absence)
  with durable semantic checks.
- **Cross-cutting tests (`tests/test_native_phase_d.py`).** 40 cases
  covering only what the per-component suites cannot: the complete
  Conv→ReLU→Pool→Flatten→Linear stack (shapes, deterministic parameter
  order, state keys, float64/cpu, contiguous *and* non-contiguous input);
  one graph producing every expected gradient plus the input gradient;
  graph-resource release and `retain_graph` behavior across the mixed
  graph; shared modules and shared inputs accumulating across branches
  with identity-deduplicated traversal; the two versioning contracts
  meeting in one place (Conv2d input-grad detects a stale weight,
  weight-grad detects a stale versioned input, bias-only backward ignores
  both, MaxPool2d records nothing and still routes by saved winners);
  state/checkpoint integration for a model containing both operations
  (independent snapshots, atomic invalid load, exact `NativeAdam` **and**
  `NativeSGD` resume, no transient state serialized, Phase-C
  compatibility); cross-layer failure atomicity under injected allocation
  failure at four successive points; resource lifetime (no accumulation
  across steps, winner lifetime, graph-construction failure, explicit
  close); and the final capability boundary.
- **Benchmarks (§17).** `benchmarks/benchmark_native_cnn.py` with five
  modes — `forward_native`, `forward_graph`, `forward_backward_fresh`,
  `training_step`, and a `stable_forward` reference — over `conv2d`,
  `maxpool2d`, and the end-to-end `cnn` case, with warm-up, repeated timed
  batches, median/min/max reporting, a correctness gate before timing, and
  `--smoke`/`--json` modes. It asserts no speed and claims no
  cross-framework, GPU, production, or scalability result;
  `tests/test_native_cnn_benchmark.py` proves it runs and reports the
  expected fields without pinning any timing.
- **Sanitizer validation.** Clang 18.1.3 on Linux (WSL2 Ubuntu), CMake
  3.28.3, `-DCMAKE_BUILD_TYPE=Debug -DTF_SANITIZE=address,undefined
  -DTF_BUILD_TESTS=ON` into a build directory outside the tree.
  Instrumentation was confirmed in the built library (`nm -D` shows 19
  `__asan*` and 12 `__ubsan*` symbols). With
  `ASAN_OPTIONS=halt_on_error=1:abort_on_error=1:detect_stack_use_after_return=1`
  and `UBSAN_OPTIONS=halt_on_error=1:print_stacktrace=1`: all 5 native
  CTests pass (with `detect_leaks=1`, i.e. LeakSanitizer active), the 634
  Phase-D Python tests pass, the 1602-test native-focused selection
  passes, the full Python suite passes, and the CNN/checkpoint-resume/MLP
  examples and the CNN benchmark smoke all run clean — **zero ASan and
  zero UBSan diagnostics attributable to TensorForge**. Python-level runs
  preload the ASan runtime (`LD_PRELOAD`) because the interpreter itself
  is not instrumented.
  **LeakSanitizer scope, stated honestly.** The fully instrumented native
  CTest binaries report **no leaks at all**. Running LSan over the Python
  process *with* the instrumented library does report ~1.36 MB in ~1315
  allocations, but **not one leak stack touches `_tensorforge_cpp.so` or
  any `tf::`/`tf_core_`/`tf_storage_` symbol**: every allocation site is
  CPython/NumPy module-and-type initialization
  (`_PyObject_Malloc`, `PyType_GenericAlloc`, `_PyType_FromMetaclass_impl`,
  `initumath`, `PyUFunc_FromFuncAndDataAndSignatureAndIdentity`, …) that a
  non-instrumented interpreter never frees at shutdown. That output is
  therefore unusable as a project-level leak gate, and the project's leak
  contract remains what it has always been: the deterministic live-storage
  counters and explicit-cleanup tests, which assert an exact return to
  baseline. **No suppression file was added** — nothing was hidden.
- **Acceptance:** all §19 criteria met; Phase D marked complete.
  **Done.** **Dependencies:** D1–D11.

---

## 19. Phase-D completion criteria — all met

Phase D was to be complete only when **all** of the following held. Each
is checked against reality at the D12 completion checkpoint:

- ✅ `NativeFlatten` works through existing reshape/autograd behavior
  (D1; refined to an **owning** output — §3.1).
- ✅ Conv2d forward and all required gradients (input, weight, bias) work
  (D2–D6; bias gradient composed from existing `sum` reductions).
- ✅ `NativeConv2d` registers and restores parameters correctly (D7).
- ✅ MaxPool2d forward and backward are deterministic (D8–D9), driven by
  the private saved winners with the locked tie/padding/`-inf` rules.
- ✅ `NativeMaxPool2d` integrates with `NativeSequential` (D10).
- ✅ Stable/native numerical **parity** tests pass — exactly for pooling
  (selection copies values) and to tolerance for convolution.
- ✅ Finite-difference **gradient checks** pass (D4/D5 C++ CTests, D6
  autograd tests).
- ✅ Stale-graph / versioning behavior remains correct, including the
  deliberate asymmetry between Conv2d (conditional version reads) and
  MaxPool2d (none) — cross-checked together in
  `tests/test_native_phase_d.py`.
- ✅ Allocation failures produce Python **exceptions** (`MemoryError` via
  the status contract) at every Phase-D layer.
- ✅ Failure paths **leak nothing** and **partially mutate nothing** —
  verified with the deterministic live-storage counters.
- ✅ Native CNN training is **deterministic** (D11).
- ✅ Checkpoint **resume reproduces** uninterrupted training *exactly*
  (D11: losses, predictions, parameters, and optimizer state).
- ✅ Existing **Phase A–C tests keep passing** (full suite green).
- ✅ Support documentation marks **only completed** features as supported,
  and the doc guardrails are now semantic agreement checks between the
  docs and the backend inventories rather than wording pins.
- ✅ **ASan/UBSan** validation run at the Phase-D completion checkpoint
  (D12, Clang 18 on Linux): no diagnostic attributable to TensorForge;
  LeakSanitizer clean over the instrumented native CTests.
- ✅ **No** CUDA, dtype expansion, normalization, dropout, or
  classification-stack scope leaked into Phase D — `SUPPORTED_DTYPES` is
  still `("float64",)`, `SUPPORTED_DEVICES` still `("cpu",)`, and every
  excluded name is still in `UNSUPPORTED`.

**Phase D is therefore complete.** The backend registry, support matrix,
roadmap, and README all present the same shipped surface: `NativeFlatten`,
the full Conv2d line, the full MaxPool2d line, and the deterministic
training + exact checkpoint-resume proof, with the phase's deliberate
exclusions still marked unsupported. The next native phase (**Phase E**)
has not started; nothing in this document should be read as a claim about
it.

---

## 20. Post-Phase-D: how Conv2d executes after Phase H, milestone H9

**Everything above still describes the contract.** H9 changed *how* the
three convolution kernels walk memory and **nothing else** — no C ABI
symbol, no signature, no supported option, no layout rule, no error, no
allocation policy, and no autograd behaviour. It is recorded here because
§14's description of `conv2d.cpp` is otherwise incomplete, not because any
statement above became false.

Each of `tf_core_conv2d_forward`, `tf_core_conv2d_input_backward`, and
`tf_core_conv2d_weight_backward` now dispatches, inside the **unchanged**
export, between two compute paths:

- `tf::conv2d_forward_generic`, `tf::conv2d_input_backward_generic`, and
  `tf::conv2d_weight_backward_generic` — **the D2/D4/D5 direct loops,
  retained verbatim.** They are the shipped generic reference paths,
  reachable through ordinary production dispatch for every geometry that
  fails a predicate, and the oracle every optimized result is compared
  against. The `n, o, i, j` outer / `c, p, q` inner nest and the
  deterministic `c → p → q` order described in §6 and §7 are exactly these
  functions.
- `tf::conv2d_forward_row_sweep`, `tf::conv2d_input_backward_gather`, and
  `tf::conv2d_weight_backward_gather` — the H9 traversals, which replace
  the short kernel-tap inner loop with a sweep along one contiguous
  spatial row.

The choice is made by three hidden predicates from the integer geometry
the export already receives:
`min(input_width, output_width) >= 4` for all three, plus **unit stride in
both axes** for the input gradient. They are total, pure, allocation-free,
and functions of that geometry alone — never a pointer value, an
alignment, a clock, an environment variable, or a CPU-feature probe — and
**a false answer selects the generic path and is never an error**. There
is no path selector, block-size setter, dispatch tracer, or "which path
ran" query anywhere in the ABI or the Python layer.

**Policy B is untouched and load-bearing.** The C ABI still takes
contiguous storage only, and the Core layer still materializes any
non-contiguous operand into a private copy closed as soon as the call
returns. H9 is a *geometry* optimization, not a layout one, and it
broadened layout support by nothing.

**§6's and §7's accumulation orders are preserved exactly**, per
destination, on both paths — separately proved for all three directions in
`docs/native_cpu_performance_design.md` §16.9.5. The bias is still added
exactly once per output element as the accumulator's seed; padded taps are
still skipped rather than materialized; and the two gradients still define
their whole destination span. What changed is only *how* the same taps are
enumerated.

`cpp/tests/test_conv2d_execution.cpp` (CTest #17) compiles `conv2d.cpp` in
so it can drive the predicates and **both** paths per direction directly,
and proves them bit-identical across the geometry matrix, the signed-zero
matrix, and the NaN matrix. The full H9 record — attribution, candidates,
rejected alternatives, the numerical contract, the H1 proof, and the
measurements — is in
[native_cpu_performance_design.md](native_cpu_performance_design.md)
§16.9.
