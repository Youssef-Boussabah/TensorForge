# Native support matrix

The canonical, authoritative statement of what the **experimental
native C++ CPU line** supports today, as of Advanced C++ v3.16 —
**Phase D (the native CNN stack) is complete**, closing the
Phase A (native CPU runtime) → Phase B (native autograd) → Phase C
(native training stack) → Phase D arc in code. The stable Python
framework's features (see
[architecture.md](architecture.md)) are **not** listed here — a feature
appears as supported only if the native stack itself provides it.
Everything below is float64/cpu only, explicit, and experimental; see
[backend_experiments.md](backend_experiments.md) for the full story and
[native_autograd_design.md](native_autograd_design.md) for the autograd
design.

**Phase status.** Phase A — **complete** (runtime, ownership, shapes/
strides/offsets/views, broadcasting, reductions, float64/cpu metadata).
Phase B — **complete** (Python-managed reverse-mode autograd, graph
lifetime, view and broadcasting gradients, parameter-version
stale-graph safety). Phase C — **complete** (parameters, modules,
Linear/ReLU/Sequential, MSE loss, `sqrt`/`reciprocal` optimizer
primitives, SGD, Adam, optimizer state snapshots, checkpoint files,
deterministic training and in-memory/file resume, and the failure/
lifetime/ownership guardrails). Phase D — **complete** (the native CNN
stack, milestones D0–D12: the locked architecture contract
([native_cnn_design.md](native_cnn_design.md)); `NativeFlatten`; the
differentiable `conv2d` operation with input/weight/bias gradients and the
trainable `NativeConv2d` module; the `maxpool2d` operation with its private
saved winners, backward scatter, and the `NativeMaxPool2d` module; the
deterministic end-to-end CNN training + exact checkpoint-resume proof
(`examples/native_cnn_training.py`); cross-cutting Phase-D integration
tests; honest CNN benchmarks (`benchmarks/benchmark_native_cnn.py`); and
ASan/UBSan validation of the whole stack). The next native phase is
**Phase E**, which has **not started** — see
[roadmap.md](roadmap.md). Everything Phase D deliberately excluded
remains unsupported and is named in the "Unsupported or future" section
below, which stays the single place this document lists capabilities the
native line does not have.

## Runtime and metadata

| Capability | Status | Notes |
|---|---|---|
| Layered runtime | Supported | `NativeStorage` → `NativeTensorView` → `NativeTensorCore` → `NativeTensor`, each layer explicit |
| `NativeStorage` ownership | Supported | Explicit allocate/free of native memory; positive sizes only |
| Shape / strides / offsets | Supported | Full strided layouts, including non-contiguous and offset views |
| Contiguity tracking | Supported | `contiguous` reported on every tensor |
| dtype metadata | Supported | `"float64"` only — validated, never promoted or cast |
| device metadata | Supported | `"cpu"` only — validated, never transferred |
| Lifetime rules | Supported | Explicit `close()` / `with` blocks; closed tensors reject reads clearly; `owns_core` distinguishes owning from borrowing views |

## Forward operations on NativeTensor

| Operation | Forward | Differentiable | Notes |
|---|---|---|---|
| `add` | Yes | Yes | NumPy-style broadcasting; gradients un-broadcast back |
| `subtract` | Yes | Yes | Broadcasting; right operand's gradient negated |
| `multiply` | Yes | Yes | Broadcasting; each gradient reads the other operand |
| `relu` | Yes | Yes | Fused native `relu_backward` mask kernel |
| `sqrt` | Yes | Yes | v3.11 optimizer math primitive; backward `1/(2·sqrt(x))` from the **saved forward output** — IEEE: negatives → NaN, signed zeros preserved |
| `reciprocal` | Yes | Yes | v3.11 optimizer math primitive; backward `−1/x²` from the **saved forward output** — IEEE: ±0 → ±inf, ±inf → ±0, NaN propagates |
| `matmul` | Yes | Yes | 2-D only, no batching/broadcasting |
| `sum` | Yes | Yes | All elements or one axis; `keepdims` |
| `mean` | Yes | Yes | All elements or one axis; `keepdims` |
| `reshape` | Yes | Yes | Borrowing metadata-only view; contiguous sources only |
| `transpose` / `T` | Yes | Yes | Borrowing view; inverse-permutation backward |
| `narrow` | Yes | Yes | Borrowing view; native scatter backward |
| `contiguous_copy` | Yes | Yes | Owning materialization; pass-through backward |
| `maxpool2d` | Yes | Yes | D8–D9: fused NCHW window-maximum primitive (int/tuple `kernel_size`/`stride`/`padding`, `stride=None` ⇒ non-overlapping); backward scatters through the **private winner buffer the forward saved** — never rereading the input or recomputing a maximum — so it records **no** version snapshot and survives input mutation; the winner buffer is graph-owned state released with the graph history. **The pooling *module* (D10) is separate and not yet implemented.** |
| `conv2d` | Yes | Yes | D6: fused NCHW/OIHW cross-correlation primitive (int/tuple stride & padding, optional bias); input/weight gradients via native backward kernels, bias gradient via existing `sum`; conditional stale-value versioning. **The trainable convolution *module* (D7) is separate and not yet implemented.** |

## Autograd engine

| Capability | Status | Notes |
|---|---|---|
| Reverse-mode `backward()` | Supported | Python-managed graph over autograd-unaware native kernels |
| Broadcasting gradients | Supported | Native un-broadcast reduction |
| View gradients | Supported | reshape/transpose/narrow/contiguous_copy backwards |
| Narrow scatter backward | Supported | Dedicated native kernel |
| Gradient accumulation | Supported | Multiple paths sum; leaves retain `.grad` |
| One-shot graph release | Supported | Default `backward()` frees the traversed graph deterministically |
| `retain_graph=True` | Supported | Repeated passes accumulate until `zero_grad()` |
| Stale parameter-version detection | Supported | Mutated-after-forward parameters raise before any gradient changes (v3.7) — recorded only where backward reads a direct parent's current value |
| Saved-forward-result backwards | Supported | `sqrt`/`reciprocal` backward reads the recorded output, never the parent — parameter mutation after forward leaves those edges valid (v3.11) |
| Failure rollback | Supported | A failed pass commits no partial gradients and frees nothing |
| Double backward / higher-order | Not supported | No graph is built through backward math |

## Training stack

| Component | Status | Notes |
|---|---|---|
| `NativeParameter` | Supported | Graph-free trainable leaf; value versioning; controlled `copy_value_` mutation |
| `NativeModule` | Supported | Registration by assignment, recursive identity-deduplicated cycle-safe traversal, train/eval, `zero_grad()` |
| Buffers | Supported | v3.15: `register_buffer(name, tensor, persistent=True)`, `buffers()` / `named_buffers()`; NativeTensor-backed non-`Parameter` persistent state (infrastructure for future BatchNorm/RNG state — no algorithm yet); identity-deduplicated, cycle-safe traversal; persistent buffers join `state_dict`/`load_state_dict` and checkpoints, non-persistent buffers are never serialized |
| `state_dict` / `load_state_dict` | Supported | In-memory, parameters and persistent buffers, atomic validate-then-commit with rollback (buffer identity preserved on restore) |
| `NativeLinear` | Supported | Seeded deterministic init; strictly 2-D input |
| `NativeReLU` | Supported | Parameter-free activation module |
| `NativeFlatten` | Supported | D1 (Phase D): parameter-free, buffer-free batch-preserving flatten `(N, …) → (N, features)`, Python-composed from the existing `reshape`/`contiguous_copy` ops and their autograd — no new kernel, no custom backward; returns an independent owning result so it composes safely in `NativeSequential` |
| `NativeSequential` | Supported | Ordered container with contiguous integer-string slots |
| `NativeMSELoss` | Supported | `"mean"` / `"sum"` reductions; exact shapes, no broadcasting |
| `NativeSGD` | Supported | Minimal `value ← value − lr·grad`; identity-deduplicated; two-phase mutation-atomic `step()`; `zero_grad()`; in-memory `state_dict`/`load_state_dict` (v3.13: lr + positional parameter metadata) |
| `NativeAdam` | Supported | Adaptive optimizer (v3.12): validated `lr`/`betas`/`eps`; persistent optimizer-owned native m/v moments and per-parameter step counts; bias correction via `sqrt`/`reciprocal` (no division); graph-free staged updates committed through `copy_value_`; skipped frozen/`grad=None` parameters never age state; explicit state lifetime — `close()` releases the moments; in-memory `state_dict`/`load_state_dict` (v3.13) |
| Optimizer state (in-memory) | Supported | v3.13: one versioned schema (format 1, exact optimizer type tag), ordered positional shape/dtype/device parameter metadata — no object ids, names, values, or gradients — caller-owned independent NativeTensor m/v snapshots and per-parameter step counts (NativeAdam), exact validation with no casting or device movement, staged atomic loading that never touches parameter values, versions, gradients, or retained graphs; deterministic in-memory training continuation with the module state contract |
| Checkpoint files / resume | Supported | v3.14: `save_native_checkpoint` / `load_native_checkpoint` — one pickle-free NPZ archive (format `"tensorforge.native_checkpoint"`, version 1) holding the model state, optionally one native optimizer's v3.13 state, and JSON-compatible metadata; UTF-8/JSON uint8 manifest, indexed float64 array entries, strict full-archive validation before any live mutation, strict optimizer presence/type matching, atomic temporary-file replacement, `allow_pickle=False` loading, deterministic bit-identical file resume (`examples/native_checkpoint_resume.py`); no scheduler or random-state capture, no `map_location` |
| End-to-end MLP training | Proven | `examples/native_mlp_training.py`: 25 deterministic steps, monotonic 99.5% loss reduction |
| End-to-end CNN training + checkpoint resume | Proven | D11: `examples/native_cnn_training.py` — convolution → activation → pooling → flatten → linear over eight fixed 6×6 images, learning a spatial edge-strength target with `NativeMSELoss` and `NativeAdam(lr=0.05)`; 40 deterministic steps, loss 0.771306 → 0.011085 (98.6% reduction); a run interrupted at step 15, checkpointed (model **and** optimizer state) and resumed into a fresh model/optimizer pair reproduces the uninterrupted run **exactly** — losses, predictions, parameter values, and optimizer state. Adds no kernel, operation, loss, optimizer, or checkpoint schema |

## Unsupported or future (native line)

None of the following exists on the native stack today. Several exist
in the stable Python framework — that does not make them native.

- `divide` as a NativeTensor operation (a raw ctypes `elementwise_divide`
  kernel exists at the kernel layer, but no tensor op and no backward;
  `reciprocal` + `multiply` compose what the training stack needs)
- `exp`, `log`, `tanh`, `sigmoid`, `softmax`
- scheduler state, random-state capture/restoration, or dataloader
  state in native checkpoints; `map_location`, partial or name-remapped
  loading, checkpoint merging, sharding, compression, or encryption
- weight decay, AdamW, AMSGrad, parameter groups, per-parameter
  learning rates, or schedulers on the native optimizers
- native classification: `softmax`, cross-entropy, or any classification
  loss/metric (the native line trains regression through `NativeMSELoss`)
- native normalization (BatchNorm/LayerNorm), dropout, or a native RNG
- additional native activations/math beyond `relu`/`sqrt`/`reciprocal`
- CUDA / GPU execution
- float32 / float16 / bfloat16, dtype promotion or casting, AMP
- Transformers / text models
- distributed training
- integration or implicit dispatch into the stable `tensorforge.Tensor`

## Phase D — the native CNN stack, **complete**

Every milestone below has shipped and is validated; this section is the
per-milestone record, not a plan. The native CNN stack's **architecture
contract is locked** in
[native_cnn_design.md](native_cnn_design.md) (milestone **D0**). **D1
(`NativeFlatten`) has shipped**, **D2 shipped the internal convolution
forward compute kernel** (`tf::conv2d_forward_contiguous`, a hidden C++
symbol), and **D3 has shipped the forward-only convolution *layer***: the
exported, exception-guarded C ABI wrapper `tf_core_conv2d_forward`, its
ctypes/`errcheck` registration, and `NativeTensorCore.conv2d_forward` (a
Python-reachable, forward-only, autograd-unaware Core method). **D4/D5
shipped the internal convolution input/weight-gradient kernels**, and **D6
completed the differentiable native convolution operation**: the exported
guarded backward C ABI wrappers (`tf_core_conv2d_input_backward`,
`tf_core_conv2d_weight_backward`), the Core backward methods
(`NativeTensorCore.conv2d_input_backward`/`conv2d_weight_backward`), the
bias gradient composed from the existing native `sum` reduction (no
dedicated kernel), and the Python-managed **`NativeTensor.conv2d`** autograd
primitive (input/weight/bias gradients, conditional stale-value version
tracking, failure rollback). **D7 has shipped the trainable `NativeConv2d`
module** — an OIHW weight / optional `(O,)` bias `NativeParameter` layer
with deterministic uniform conv fan-in initialization, 4-D NCHW input
validation, and backward supplied entirely by the D6 `conv2d` autograd (no
new kernel, C ABI symbol, or custom module backward); it registers in
`NATIVE_MODULES` and exports from `tensorforge.experimental`. **D8 has
shipped the forward-only pooling *layer*:** the internal compute kernel
`tf::maxpool2d_forward_contiguous` (a hidden C++ symbol in the new
`cpp/src/pooling.cpp`), the exported guarded C ABI wrapper
`tf_core_maxpool2d_forward` with its ctypes/`errcheck` registration, and
`NativeTensorCore.maxpool2d_forward` — a Python-reachable, autograd-unaware
Core method that also produces the **private saved-winner buffer** (an
internal float64 buffer of flat plane offsets with a `-1` padding sentinel,
validated exact against `H*W ≤ 2^53`) the D9 backward consumes. **D9 has
completed the differentiable pooling operation**: the internal scatter-add
`tf::maxpool2d_backward_contiguous`, the exported guarded
`tf_core_maxpool2d_backward` wrapper (which **validates every winner value**
— `-1` or an exact in-range integer — before scattering, never rounding),
`NativeTensorCore.maxpool2d_backward`, and the Python-managed
**`NativeTensor.maxpool2d`** autograd node whose single input-gradient
callback uses only the saved winners (no input reread, no recomputed
maximum, **no version snapshot**), with the winner buffer owned by the
graph history and released exactly when it is. **D10 has shipped the
`NativeMaxPool2d` module** — a parameter-free, buffer-free layer that
normalizes `kernel_size`/`stride`/`padding` to two-element tuples
(`stride=None` ⇒ non-overlapping windows) and delegates its forward
entirely to that operation: no new kernel, C ABI symbol, ctypes
declaration, custom backward, parameter, buffer, or checkpoint schema, no
`return_indices`, and no winner state held between calls. It registers in
`NATIVE_MODULES`, exports from `tensorforge.experimental`, contributes no
state-dictionary or checkpoint keys, and composes in a `NativeSequential`
beside `NativeConv2d`/`NativeReLU`/`NativeFlatten`/`NativeLinear`. **D11
has shipped the deterministic end-to-end proof**
(`examples/native_cnn_training.py`): the full Conv→ReLU→Pool→Flatten→
Linear stack learns a genuinely spatial regression target (the strongest
bright-to-dark vertical edge of eight fixed 6×6 images) with
`NativeMSELoss` and `NativeAdam(lr=0.05)` over 40 steps — loss
0.771306 → 0.011085 — and a run interrupted at step 15, checkpointed
(model **and** optimizer state) and resumed into a completely fresh
model/optimizer pair reproduces the uninterrupted run **exactly**: loss
history, final predictions, every parameter value, and every optimizer
state entry. It adds no kernel, ABI symbol, operation, loss, optimizer, or
checkpoint schema. **D12 closed the phase**: cross-cutting Phase-D
integration tests (`tests/test_native_phase_d.py` — the full module stack,
end-to-end autograd, shared graphs, the two versioning contracts meeting in
one backward, state/checkpoint integration, cross-layer failure atomicity,
resource lifetime, and the capability boundary), honest CNN benchmarks
(`benchmarks/benchmark_native_cnn.py` — conv/pool forward, forward+backward,
end-to-end training step, and a stable-framework reference, with no speed
claims), **ASan/UBSan validation** of the whole native CNN stack under
Clang on Linux, a LeakSanitizer pass over the native CTests, documentation
reconciliation, and the replacement of the milestone-era documentation
guardrails with durable semantic ones. D12 added no kernel, C ABI symbol,
ctypes declaration, operation, module, or schema.

| Capability | Milestone | Status |
|---|---|---|
| `NativeFlatten` (batch-preserving; existing reshape/copy autograd) | D1 | **Implemented** |
| Internal convolution forward compute kernel (C++, not exposed) | D2 | **Implemented (internal)** |
| Convolution forward C ABI export (`tf_core_conv2d_forward`) — exception-guarded; self-validates handles/dims/offsets/output-shape/overflow/span-bounds; contiguous storage is a caller precondition (no stride metadata crosses the ABI, so it never inspects logical contiguity) | D3 | **Implemented (raw kernel)** |
| Convolution forward Core wrapper (`NativeTensorCore.conv2d_forward`) — ctypes, Policy-B copy, output allocation, Python forward access | D3 | **Implemented (Core, forward-only)** |
| Internal convolution input-gradient compute kernel (`tf::conv2d_input_backward_contiguous`, C++, not exposed) | D4 | **Implemented (internal)** |
| Internal convolution weight-gradient compute kernel (`tf::conv2d_weight_backward_contiguous`, C++, not exposed) | D5 | **Implemented (internal)** |
| Convolution input/weight-gradient C ABI export (`tf_core_conv2d_input_backward`, `tf_core_conv2d_weight_backward`) + Core wrappers (`NativeTensorCore.conv2d_input_backward`/`conv2d_weight_backward`) | D6 | **Implemented (raw + Core)** |
| Convolution bias-gradient numerical path (reuse of the existing native `sum` reduction: `g.sum(0).sum(1).sum(1) → (O,)`; no dedicated kernel, no C ABI symbol) | D5–D6 | **Implemented (existing-reduction composition, wired into the autograd node)** |
| Convolution `NativeTensor` autograd op — differentiable `NativeTensor.conv2d(weight, bias=None, *, stride, padding)` | D6 | **Implemented** |
| `NativeConv2d` module — trainable OIHW weight / optional `(O,)` bias layer over the D6 `conv2d` autograd (deterministic uniform conv fan-in init, 4-D NCHW validation; no new kernel/ABI/backward) | D7 | **Implemented** |
| Internal max-pooling forward compute kernel (`tf::maxpool2d_forward_contiguous`, C++, not exposed) — pooled values and winners in one pass | D8 | **Implemented (internal)** |
| Max-pooling forward C ABI export (`tf_core_maxpool2d_forward`) — exception-guarded; self-validates handles/dims/offset/output-shape/winner-exactness/overflow/span-bounds; contiguous storage is a caller precondition (no stride metadata crosses the ABI) | D8 | **Implemented (raw kernel)** |
| Max-pooling forward Core wrapper (`NativeTensorCore.maxpool2d_forward`, plus the private with-winners helper) — ctypes, Policy-B copy, output + private winner-buffer allocation, failure-atomic cleanup | D8 | **Implemented (Core, forward-only)** |
| Internal max-pooling backward compute kernel (`tf::maxpool2d_backward_contiguous`, C++, not exposed) — zero-initializing scatter-add driven only by the saved winners | D9 | **Implemented (internal)** |
| Max-pooling backward C ABI export (`tf_core_maxpool2d_backward`) — exception-guarded; validates handles/dims/offsets/spans **and every winner value** (`-1` or an exact in-range integer, never rounded); takes no kernel/stride/padding | D9 | **Implemented (raw kernel)** |
| Max-pooling backward Core wrapper (`NativeTensorCore.maxpool2d_backward`) — ctypes, Policy-B copies, fresh owning grad_input, failure-atomic cleanup | D9 | **Implemented (Core)** |
| Pooling `NativeTensor` autograd op — differentiable `NativeTensor.maxpool2d(*, kernel_size, stride, padding)`; single `(input,)` parent, graph-owned saved winners, no version snapshot | D9 | **Implemented** |
| `NativeMaxPool2d` module — parameter-free, buffer-free layer over the D8/D9 pooling operation (normalized `(h, w)` `kernel_size`/`stride`/`padding`, `stride=None` ⇒ non-overlapping); no new kernel/ABI/backward, no parameters, buffers, winner state, or state-dict keys | D10 | **Implemented** |
| Deterministic native CNN training + checkpoint-resume proof (`examples/native_cnn_training.py`: Conv→ReLU→Pool→Flatten→Linear + `NativeMSELoss` + `NativeAdam`, 40 steps, loss 0.771306 → 0.011085; interrupted-and-resumed training matches the uninterrupted run exactly — losses, predictions, parameters, and optimizer state) | D11 | **Implemented (proven)** |
| Phase-D cross-cutting integration tests (`tests/test_native_phase_d.py`), CNN benchmarks (`benchmarks/benchmark_native_cnn.py`), ASan/UBSan validation, LeakSanitizer pass over the native CTests, documentation reconciliation, and durable capability guardrails | D12 | **Implemented (phase closed)** |

Locked design decisions (see the design doc for the full contract):
**NCHW** activations, **OIHW** convolution weights, **cross-correlation**
(not flipped); floor output-shape formulas with symmetric per-axis
padding; **copy-then-compute** for non-contiguous inputs (kernels consume
contiguous storage only); convolution as a **new fused `NativeTensor`
primitive** with a Python-managed backward (input/weight kernels + bias
via existing `sum` reductions); max-pool winners saved in an **internal
float64 buffer** of flat input offsets (with a `-1` padding sentinel);
new C ABI families `tf_core_conv2d_*` / `tf_core_maxpool2d_*` under the
existing status/guard contract; and new C++ units `cpp/src/conv2d.cpp`
and `cpp/src/pooling.cpp`. Still float64/cpu only; no dilation, groups,
transposed/average/adaptive/global pooling, channels-last, float32,
CUDA, AMP, BatchNorm, Dropout, im2col, or BLAS/threaded convolution.

## How to build and verify

The native backend is built with CMake (`cpp/CMakeLists.txt`), wrapped by
the cross-platform `cpp/build.py` (which falls back to a direct compiler
invocation — `g++`/`clang++`/`ziglang` — when CMake is unavailable, as on
CI). Every fallible native export is exception-guarded so no C++ exception
crosses the ABI; native failures surface as `MemoryError` / `ValueError` /
`RuntimeError` (see docs/native_abi_error_contract.md). All commands are
verified against this repository:

```
uv sync                                                # dependencies
uv sync --group cpp                                    # only if no system C++ compiler
uv run python cpp/build.py                             # build the native backend (Release)
uv run python cpp/build.py --debug                     # unoptimized debug build
uv run python scripts/smoke_cpp_backend.py             # hard-failing smoke check
uv run python examples/native_tensor_demo.py           # runtime and views
uv run python examples/native_autograd_demo.py         # native backward
uv run python examples/native_mlp_training.py          # end-to-end training proof
uv run python examples/native_checkpoint_resume.py     # save, restore, resume bit-for-bit
uv run python examples/native_cnn_training.py          # end-to-end CNN training + resume proof
uv run python benchmarks/benchmark_native_autograd.py --smoke
uv run python benchmarks/benchmark_native_cnn.py --smoke   # CNN characterization
uv run pytest                                          # full suite (native tests skip if unbuilt)
```

Sanitizer validation (Linux/WSL with Clang; MSVC supports neither
`-fsanitize=undefined` nor this option form):

```
cmake -S cpp -B build/phase-d-sanitizers -DCMAKE_BUILD_TYPE=Debug \
      -DCMAKE_CXX_COMPILER=clang++ -DTF_SANITIZE=address,undefined \
      -DTF_BUILD_TESTS=ON
cmake --build build/phase-d-sanitizers
ctest --test-dir build/phase-d-sanitizers --output-on-failure
```
