# Native CNN architecture design (Phase D contract)

This is the **design-and-contract** document for the experimental native
C++ CPU line's convolutional stack — **Phase D**. It is a milestone-zero
(D0) deliverable: it locks the architecture, layouts, argument contracts,
ownership rules, C ABI shape, source organization, testing strategy, and
milestone sequence **before any numerical CNN code is written**.

Nothing in Phase D is implemented yet. No `NativeConv2d`,
`NativeMaxPool2d`, or `NativeFlatten` exists; no convolution or pooling
kernel exists; the backend capability registry still lists `conv2d`,
`maxpool2d`, and `flatten` as **unsupported** (see
[native_support_matrix.md](native_support_matrix.md) and
`tensorforge.backends.cpp.UNSUPPORTED`). This document describes what
those milestones **will** build, not what exists.

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
- **View vs copy:** result is a **view when the input is contiguous**
  (`NativeTensor.reshape` returns a borrowing view), and a **fresh owning
  copy** when the input is non-contiguous (Flatten first materializes via
  `contiguous_copy()` because `reshape` requires contiguity — the same
  rule the runtime already enforces). This is the only place Flatten
  touches the contiguity rule, and it does so through existing ops.
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

### 7.3 Gradient w.r.t. bias — `∂L/∂b`

- **Output shape:** `(O,)`.
- **Rule:** `grad_b[o] = Σ_{n,i,j} g[n, o, i, j]` — i.e.
  `g.sum(axis=(0, 2, 3))`.
- **Distinct kernel:** **no.** This composes from the **existing native
  `sum` reductions** (`g.sum(axis=0)` then reduce the two spatial axes),
  reusing tested kernels — no new C++ code. Reads only `g` (no bias
  value), so it is **not** version-sensitive.

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

- **`cpp/src/conv2d.cpp`** — `tf_core_conv2d_forward`,
  `tf_core_conv2d_input_backward`, `tf_core_conv2d_weight_backward`.
- **`cpp/src/pooling.cpp`** — `tf_core_maxpool2d_forward`,
  `tf_core_maxpool2d_backward`.

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

Forward shapes (4-D→2-D, already-2-D pass-through); view when contiguous /
copy when non-contiguous; rank < 2 rejected; backward reshape correctness;
`NativeSequential` integration (Conv/Pool→Flatten→Linear).

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

Parameter registration (`["weight", "bias"]`); state dicts; checkpoint
round trips (save→load→resume); `NativeSequential` traversal with
Conv/Pool/Flatten slots; deterministic CNN training (fixed seed,
monotonic loss); resume equivalence (bit-identical continuation);
existing Phase A–C regression coverage stays green.

Every new test protects a real behavioral guarantee, not an arbitrary
string.

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

### D1 — `NativeFlatten`
- **Scope:** batch-preserving flatten via existing `reshape`/
  `contiguous_copy` autograd; new module + exports.
- **Excludes:** `start_dim`/`end_dim`; any kernel.
- **Files:** `native_flatten.py`; `experimental/__init__.py`;
  `tests/test_native_flatten.py`; support-matrix + `NATIVE_MODULES`
  update.
- **Tests:** §16 Flatten group.
- **Acceptance:** forward/backward correct; view/copy rule honored;
  Sequential integration.
- **Dependencies:** existing reshape autograd. **Risks:** contiguity edge
  cases (mitigated by copy-then-reshape).

### D2 — Conv2d CPU forward kernel
- **Scope:** `tf_core_conv2d_forward` (C++) + fault-injection safety;
  direct nested loops; nullable bias.
- **Excludes:** any backward; the wrapper/module/autograd.
- **Files:** `cpp/src/conv2d.cpp`; a kernel-level smoke test.
- **Tests:** kernel-level numeric checks (via a temporary low-level
  harness) + ABI error path.
- **Acceptance:** correct against hand-computed values; sanitizer-clean.
- **Dependencies:** D0. **Risks:** index arithmetic bugs (mitigated by
  explicit indexing + ASan/UBSan).

### D3 — Conv2d `NativeTensorCore` wrapper + ctypes
- **Scope:** argtypes registration; `NativeTensorCore.conv2d_forward`;
  Policy-B contiguity; output allocation; shape validation.
- **Excludes:** autograd graph; module.
- **Files:** `backends/cpp.py`; `tests/test_native_tensor_core.py`
  additions.
- **Tests:** forward parity with stable; non-contiguous input; invalid
  shapes; allocation failure.
- **Acceptance:** forward-only `NativeTensorCore` conv works end to end.
- **Dependencies:** D2.

### D4 — Conv2d input-gradient kernel
- **Scope:** `tf_core_conv2d_input_backward` + its Core wrapper.
- **Excludes:** weight/bias grad; autograd wiring.
- **Files:** `cpp/src/conv2d.cpp`; `backends/cpp.py`; tests.
- **Tests:** input-grad finite differences (forward-only harness).
- **Acceptance:** `∂L/∂x` correct. **Dependencies:** D3.

### D5 — Conv2d weight- and bias-gradient
- **Scope:** `tf_core_conv2d_weight_backward` + Core wrapper; **bias grad
  via existing `sum` reductions** (no kernel).
- **Excludes:** autograd graph node.
- **Files:** `cpp/src/conv2d.cpp`; `backends/cpp.py`; tests.
- **Tests:** weight/bias-grad finite differences.
- **Acceptance:** `∂L/∂W`, `∂L/∂b` correct. **Dependencies:** D4.

### D6 — Conv2d `NativeTensor` autograd integration
- **Scope:** `NativeTensor.conv2d(...)` fused primitive — graph node,
  parent set, version recording (§8), backward closure calling D4/D5.
- **Excludes:** the module.
- **Files:** `native_tensor.py`; `AUTOGRAD_OPS`/`TENSOR_CORE_OPS` update;
  `tests/test_native_conv2d_backward.py`.
- **Tests:** §16 backward group (finite diff, subset-grad, stale
  mutation, atomicity).
- **Acceptance:** end-to-end differentiable native conv.
- **Dependencies:** D5. **Risks:** version/stale-graph correctness
  (mitigated by reusing the audited `_versioned_value_reads` path).

### D7 — `NativeConv2d` module
- **Scope:** module (§9) — parameters, init, registration, state/
  checkpoint, repr, frozen support, `NativeSequential` fit.
- **Files:** `native_conv2d.py`; `experimental/__init__.py`;
  `NATIVE_MODULES` + support-matrix update; `tests/test_native_conv2d.py`.
- **Tests:** registration/state/checkpoint/Sequential + forward parity.
- **Acceptance:** trainable conv layer. **Dependencies:** D6.

### D8 — MaxPool2d forward + winner-index contract
- **Scope:** `tf_core_maxpool2d_forward`; winner buffer (§12);
  `NativeTensorCore.maxpool2d_forward`.
- **Excludes:** backward; autograd; module.
- **Files:** `cpp/src/pooling.cpp`; `backends/cpp.py`; tests.
- **Tests:** forward hand-computed, ties, padding, stride, winner buffer
  contents.
- **Acceptance:** correct pooled values + winners. **Dependencies:** D3
  (shares Core patterns). **Risks:** tie/padding winner encoding
  (mitigated by the sentinel design).

### D9 — MaxPool2d backward + autograd integration
- **Scope:** `tf_core_maxpool2d_backward` (scatter); `NativeTensor.
  maxpool2d(...)` node holding the winner buffer.
- **Files:** `cpp/src/pooling.cpp`; `native_tensor.py`; `AUTOGRAD_OPS`
  update; `tests/test_native_maxpool2d.py`.
- **Tests:** §16 MaxPool2d group (scatter, overlap, ties, lifetime,
  atomicity).
- **Acceptance:** differentiable native pooling. **Dependencies:** D8.

### D10 — `NativeMaxPool2d` module
- **Scope:** parameter-free module (like `NativeReLU`) wrapping
  `maxpool2d`.
- **Files:** `native_maxpool2d.py`; `experimental/__init__.py`;
  `NATIVE_MODULES` + support-matrix update; tests.
- **Acceptance:** pooling module in a `NativeSequential`.
- **Dependencies:** D9.

### D11 — Deterministic native CNN training + checkpoint-resume proof
- **Scope:** `examples/native_cnn_training.py` — a small deterministic
  Conv→ReLU→Pool→Flatten→Linear→MSE CNN trained through the native stack;
  checkpoint save + resume equivalence; integration tests.
- **Files:** `examples/native_cnn_training.py`;
  `tests/test_native_cnn_training.py`; support-matrix "proven" row.
- **Tests:** deterministic loss trajectory; bit-identical resume; NumPy
  tripwire over the run.
- **Acceptance:** end-to-end native CNN training proof.
- **Dependencies:** D1, D7, D10. **Risks:** any loss beyond MSE would
  pull in classification scope — kept to MSE unless the approved proof
  explicitly needs otherwise.

### D12 — Phase-D cross-cutting tests, benchmarks, docs, completion
- **Scope:** cross-cutting guardrails (`tests/test_native_phase_d.py`),
  benchmarks (§17), final support-matrix/roadmap/README updates, and the
  **ASan/UBSan validation checkpoint**.
- **Files:** `tests/test_native_phase_d.py`; `benchmarks/…`; doc updates.
- **Tests:** phase-completion invariants; benchmark correctness gate.
- **Acceptance:** all §19 criteria met; Phase D marked complete.
- **Dependencies:** D1–D11.

---

## 19. Phase-D completion criteria

Phase D is complete only when **all** hold:

- `NativeFlatten` works through existing reshape/autograd behavior.
- Conv2d forward and all required gradients (input, weight, bias) work.
- `NativeConv2d` registers and restores parameters correctly.
- MaxPool2d forward and backward are deterministic.
- `NativeMaxPool2d` integrates with `NativeSequential`.
- Stable/native numerical **parity** tests pass (to tolerance).
- Finite-difference **gradient checks** pass.
- Stale-graph / versioning behavior remains correct.
- Allocation failures produce Python **exceptions** (`MemoryError` via the
  status contract).
- Failure paths **leak nothing** and **partially mutate nothing**.
- Native CNN training is **deterministic**.
- Checkpoint **resume reproduces** uninterrupted training.
- Existing **Phase A–C tests keep passing**.
- Support documentation marks **only completed** features as supported.
- **ASan/UBSan** validation is run at the Phase-D completion checkpoint.
- **No** CUDA, dtype expansion, normalization, dropout, or
  classification-stack scope has leaked into Phase D.

Until every milestone above lands, the backend registry, support matrix,
and README continue to present convolution and pooling as **not
implemented**.
