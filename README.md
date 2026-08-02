# TensorForge

A from-scratch deep learning and ML systems framework in two
lines: a **stable Python framework** built on NumPy — PyTorch-style
autograd, modules, optimizers, checkpointing, and CNN support, complete
as of v3.0 — and an **experimental native C++ CPU backend**, living in
its own `tensorforge.backends` / `tensorforge.experimental` namespaces,
with its own explicit tensor runtime, Python-managed
native autograd, and a native training stack that already trains an MLP
end to end. The two lines stay strictly separate: `NativeTensor` never
masquerades as `tensorforge.Tensor`, nothing dispatches implicitly, and
the native stack is reached only by explicit import.

Everything is implemented by hand and kept readable: the autograd
engines, the layers, the optimizers, the losses, the C++ kernels. If
you want to see what `loss.backward()` actually does — in NumPy or
through a ctypes boundary into native code — this repo shows you.

**Framework internals covered:** reverse-mode autograd, how neural
network modules and parameters fit together, what optimizers actually
do, regularization (Dropout) and normalization (BatchNorm, LayerNorm),
checkpoint/resume mechanics down to the RNG, the internals of
convolution and pooling, and native-backend mechanics: an owning
storage/view/tensor runtime, strided kernels behind a C ABI, a
Python-managed autograd graph over them, parameter versioning with
stale-graph detection, and honest benchmarks.

## Features — stable Python framework

**Core engine**
- Tensor with reverse-mode autograd: `+ - * / ** @`, `sum`, `mean`,
  `reshape`, `exp`, `log`, `tanh`, `sigmoid`, `relu`, `softmax`, with
  broadcasting-aware gradients — all verified against finite differences

**Models (`tensorforge.nn`)**
- `Parameter`, `Module`, `Linear`, `ReLU`/`Sigmoid`/`Tanh`, `Dropout`,
  `BatchNorm1d`, `LayerNorm`, `Conv2d`, `MaxPool2d`, `Flatten`,
  `Sequential`
- Train/eval mode (`model.train()` / `model.eval()`), frozen
  parameters, model summaries, parameter counting
- Losses: `mse_loss`, numerically stable `cross_entropy` and
  `binary_cross_entropy` (both on raw logits)
- Metrics: `accuracy`, `binary_accuracy`, and eval-safe
  `evaluate_classifier` / `evaluate_binary_classifier`

**Training (`tensorforge.optim`, `tensorforge.data`)**
- `SGD`, `Adam`, `StepLR` scheduler, `clip_grad_norm` /
  `clip_grad_value`
- `batches` mini-batch iterator, `train_test_split` validation holdout

**Persistence**
- `save_parameters` / `load_parameters` for weights, and
  `save_checkpoint` / `load_checkpoint` with optimizer state, optional
  scheduler state, optional RNG state (for dropout-safe, bit-exact
  resume), and metadata — all plain `.npz`, no pickle

## Features — experimental native C++ backend

The explicit experimental native line — merged into `main` and reached
only through `tensorforge.experimental` and `tensorforge.backends`, never
by the stable framework — carries a complete native CPU training line:

- **Native runtime**: `NativeStorage` → `NativeTensorView` →
  `NativeTensorCore` → `NativeTensor` — explicit ownership and
  lifetime, shapes/strides/offsets, borrowing views, contiguity
  tracking, NumPy-style broadcasting, sum/mean reductions, and
  float64/cpu dtype/device metadata over ctypes-loaded C++ kernels.
- **Native autograd (Phase B, complete)**: a Python-managed
  reverse-mode graph over autograd-unaware kernels — the differentiable
  operations Phase B shipped (`add`, `subtract`, `multiply`, `relu`,
  `sqrt`, `reciprocal`, `matmul`, `sum`, `mean`, `reshape`,
  `transpose`/`T`, `narrow`, `contiguous_copy`), joined by the Phase-D
  `conv2d` and `maxpool2d` primitives below and the Phase-E
  `exp`/`log`/`softmax`, plus broadcasting and view
  gradients, a native scatter backward for `narrow`, one-shot graph
  release with `retain_graph` opt-in, and failure rollback. The backend's
  `AUTOGRAD_OPS` registry is the exact, current list.
- **Native training stack (Phase C, complete)**: `NativeParameter`
  (value versioning and a controlled mutation path with stale-graph
  detection), `NativeModule` with atomic `state_dict`/
  `load_state_dict`, `NativeLinear`, `NativeReLU`, `NativeSequential`,
  `NativeMSELoss`, the minimal `NativeSGD` optimizer, and the
  adaptive `NativeAdam` optimizer (persistent native moment state,
  per-parameter bias correction, explicit state lifetime) — both with
  in-memory `state_dict`/`load_state_dict`, plus pickle-free native
  checkpoint files (`save_native_checkpoint` /
  `load_native_checkpoint`: model + optional optimizer state + JSON
  metadata, atomic writes, strict validation, deterministic
  bit-identical file resume).
- **A native MLP training proof**: `examples/native_mlp_training.py`
  trains a 2→8→ReLU→1 MLP for 25 deterministic native SGD steps with a
  monotonic 99.5% loss reduction — model, loss, gradients, and updates
  all native.
- **A native CNN training proof**: `examples/native_cnn_training.py`
  trains Conv2d→ReLU→MaxPool2d→Flatten→Linear on eight fixed 6×6 images
  for 40 deterministic native Adam steps (98.6% loss reduction), then
  checkpoints mid-run and resumes into a fresh model/optimizer pair that
  reproduces the uninterrupted run exactly.
- **A native classification proof**:
  `examples/native_classification_training.py` trains the same layer
  stack as a three-class classifier over **raw logits** into
  `NativeCrossEntropyLoss` on twelve fixed 6×6 images, for 40
  deterministic native Adam steps (loss 1.159638 → 0.000101, reporting
  accuracy 0.3333 → 1.0000 via `native_accuracy`), then checkpoints at
  step 15 and resumes into a fresh model/optimizer pair that reproduces
  the remaining losses, parameters, optimizer state, logits,
  predictions, and accuracy exactly.
- **A native normalized-training proof**:
  `examples/native_normalization_training.py` trains a
  `Linear → BatchNorm1d → ReLU → LayerNorm → Linear` regressor — running
  **both** normalization families in every forward, with `NativeBatchNorm1d`
  the only stateful module — for 24 deterministic native Adam steps with
  `NativeMSELoss` (98.9% loss reduction), then checkpoints at step 10 and
  resumes into a fresh model/optimizer pair that reproduces the remaining
  losses, every parameter, the NativeAdam state, the **BatchNorm running
  statistics**, the final training-step prediction, and the final
  **evaluation-mode** output exactly (format version 1, training flags
  runtime-only).
- **A native stochastic-training proof**:
  `examples/native_dropout_training.py` trains a
  `Linear → BatchNorm1d → ReLU → Dropout → LayerNorm → Linear` classifier
  over raw logits with `NativeCrossEntropyLoss` and native Adam — the
  smallest model carrying **all four** TensorForge-owned state families at
  once (parameters, persistent BatchNorm buffers, a registered
  `NativeGenerator`, and Adam moments) — then checkpoints after 7 completed
  steps, releases that model, and resumes into a completely fresh
  model/optimizer/generator set built from a *different* Dropout seed,
  reproducing the whole loss sequence, every parameter, both running
  statistics, the full optimizer state, the generator's seed and call
  counter, the final training logits, and the final evaluation output
  **exactly** (checkpoint format version 2). External loop progress travels
  as explicit, validated metadata — a checkpoint captures TensorForge-owned
  state, not a data loader, a shuffle order, a scheduler, Python's `random`,
  or NumPy's global RNG.

The exact operation-by-operation status lives in the
[native support matrix](docs/native_support_matrix.md).

## Quickstart

Requires Python ≥ 3.13 and [uv](https://docs.astral.sh/uv/):

```
uv sync
uv run pytest
uv run python examples/train_mlp_with_dropout.py
```

The stable API looks the way you'd expect:

```python
from tensorforge import Tensor, Adam
from tensorforge.nn import Linear, ReLU, Sequential, cross_entropy

x = Tensor(4.0, requires_grad=True)
(x * x).backward()
print(x.grad)  # 8.0

model = Sequential(Linear(2, 16), ReLU(), Linear(16, 3))
optimizer = Adam(model.parameters(), lr=0.01)

loss = cross_entropy(model(Tensor(inputs)), targets)
optimizer.zero_grad()
loss.backward()
optimizer.step()
```

### Native quickstart

Build the experimental backend once, then run the native line:

```
uv run python cpp/build.py                        # uv sync --group cpp first if you have no C++ compiler
uv run python scripts/smoke_cpp_backend.py        # hard-failing backend check
uv run python examples/native_tensor_demo.py      # the native runtime and views
uv run python examples/native_autograd_demo.py    # native backward
uv run python examples/native_mlp_training.py     # end-to-end native training
uv run python examples/native_checkpoint_resume.py # save, restore, resume bit-for-bit
uv run python examples/native_cnn_training.py     # end-to-end native CNN training + resume
uv run python examples/native_classification_training.py  # native classification + exact resume
uv run python examples/native_normalization_training.py   # native BatchNorm+LayerNorm training + exact resume
uv run python examples/native_dropout_training.py         # native Dropout training + exact STOCHASTIC resume
uv run python benchmarks/benchmark_native_autograd.py --smoke
uv run python benchmarks/benchmark_native_cnn.py --smoke  # CNN characterization
uv run python benchmarks/benchmark_native_classification.py --smoke        # classification characterization
uv run python benchmarks/benchmark_native_classification.py --smoke --json # machine-readable JSON
uv run python benchmarks/benchmark_native_normalization.py --smoke         # normalization characterization
uv run python benchmarks/benchmark_native_normalization.py --smoke --json  # machine-readable JSON
uv run python benchmarks/benchmark_native_dropout.py --smoke               # Dropout characterization
uv run python benchmarks/benchmark_native_dropout.py --smoke --json        # machine-readable JSON
uv run python benchmarks/benchmark_native_cpu_performance.py --smoke       # Phase-H CPU baseline
uv run python benchmarks/benchmark_native_cpu_performance.py --workload matmul
uv run python benchmarks/benchmark_native_cpu_performance.py --profile matmul_square_contiguous
```

The native API mirrors the stable one, explicitly:

```python
from tensorforge.experimental import (
    NativeLinear, NativeMSELoss, NativeReLU, NativeSequential,
    NativeSGD, NativeTensor,
)

model = NativeSequential(NativeLinear(2, 8, seed=0), NativeReLU(),
                         NativeLinear(8, 1, seed=1))
optimizer = NativeSGD(model.parameters(), lr=0.1)

loss = NativeMSELoss()(model(NativeTensor.from_array(inputs)),
                       NativeTensor.from_array(targets))
loss.backward()
optimizer.step()
optimizer.zero_grad()
```

## Examples

Six runnable, seeded stable-framework examples forming a progression —
each one introduces a single idea:

```
uv run python examples/train_linear_regression.py    # the bare training loop
uv run python examples/train_xor.py                  # why hidden layers exist
uv run python examples/train_multiclass.py           # real classification + mini-batches
uv run python examples/train_binary_classification.py  # logits and stable losses
uv run python examples/train_mlp_with_dropout.py     # train mode vs eval mode
uv run python examples/train_tiny_cnn.py             # convolution and pooling
```

What each one teaches, and what to expect: [docs/examples.md](docs/examples.md).
The native examples and demos are listed in the native quickstart above.

## Documentation

- [docs/project_summary.md](docs/project_summary.md) — the whole project in two minutes
- [docs/architecture.md](docs/architecture.md) — how both framework lines fit together
- [docs/autograd.md](docs/autograd.md) — the stable autograd engine, explained
- [docs/training.md](docs/training.md) — training loops, train/eval mode, saving
- [docs/examples.md](docs/examples.md) — the stable examples and what they teach
- [docs/roadmap.md](docs/roadmap.md) — what's done and what's next
- [docs/release_history.md](docs/release_history.md) — how the project grew, by version
- [docs/native_support_matrix.md](docs/native_support_matrix.md) — exactly what the native stack supports
- [docs/backend_experiments.md](docs/backend_experiments.md) — the experimental C++ backend line
- [docs/dispatch_design.md](docs/dispatch_design.md) — how backends might eventually meet the Tensor
- [docs/native_tensor_wrapper_design.md](docs/native_tensor_wrapper_design.md) — Stage-2 design for the native tensor wrapper
- [docs/native_contiguous_fast_path_design.md](docs/native_contiguous_fast_path_design.md) — design for a contiguous elementwise fast path in the native runtime
- [docs/native_broadcasting_design.md](docs/native_broadcasting_design.md) — design for NumPy-style broadcasting in the native elementwise runtime
- [docs/native_reductions_design.md](docs/native_reductions_design.md) — design for native sum/mean reductions in the native runtime
- [docs/native_dtype_device_metadata_design.md](docs/native_dtype_device_metadata_design.md) — design for explicit dtype/device metadata in the native runtime
- [docs/native_autograd_design.md](docs/native_autograd_design.md) — design for native reverse-mode autograd over NativeTensor (Phase B)
- [docs/native_autograd_benchmarks.md](docs/native_autograd_benchmarks.md) — characterization benchmark for the native autograd stack (Phase B)
- [docs/native_cnn_design.md](docs/native_cnn_design.md) — architecture contract for the native CNN stack (Phase D)
- [docs/native_classification_design.md](docs/native_classification_design.md) — architecture contract for the native classification stack (Phase E — complete: E0–E10 shipped)
- [docs/native_normalization_design.md](docs/native_normalization_design.md) — architecture contract for the native normalization stack (Phase F — **complete**: F0, F1, F2 (`NativeLayerNorm`), F3 (`NativeBatchNorm1d`), F4 (`NativeBatchNorm2d`), F5 (state/checkpoint/graph-safety hardening), F6 (a deterministic normalized training example with exact resume), F7 (the honest benchmark characterization), F8 (the cross-cutting integration and semantic guardrails), and F9 (the phase closure — validation and documentation only) have all shipped)
- [docs/native_rng_dropout_design.md](docs/native_rng_dropout_design.md) — architecture contract for native RNG and Dropout (Phase G — **complete**: milestone G0, the design lock, milestone G1, `NativeGenerator` and module generator-state ownership, milestone G2, the stateless `dropout_forward` **Core** kernel and its C ABI, milestone G3, the differentiable `NativeTensor.dropout(p, *, generator)` with its graph-owned saved mask and generator call transaction, milestone G4, the `NativeDropout` module and its public export, milestone G5, native checkpoint **format version 2** — persisted generator state with its shared-generator alias topology, strict topology validation, version-1 compatibility rules, and the whole-checkpoint load transaction — and milestone G6, the RNG/graph/ownership/checkpoint hardening that added no capability, and milestone G7, the deterministic stochastic training example and its exact checkpoint resume (no capability), and milestone G8, the honest benchmark characterization `benchmarks/benchmark_native_dropout.py` (also no capability — correctness gated before timing, no speed asserted), and milestone G9, the cross-cutting integration suite `tests/test_native_phase_g.py` (integration evidence only — no capability, and no runtime file changed), and milestone G10, the phase closure — the Release/Debug/sanitizer validation matrix, the documentation reconciliation, and the single registry line that finally removed `dropout` from `UNSUPPORTED` — are all complete, so end-to-end **exact stochastic training resume is demonstrated** and native Dropout is now supported on the experimental native float64 CPU line)
- [docs/native_cpu_performance_design.md](docs/native_cpu_performance_design.md) — architecture contract for native CPU performance and runtime efficiency (Phase H — the **current** phase, **begun, with H0, H1, H2, H3, H4, H5, and H6 complete**: the design lock, the unified baseline harness `benchmarks/benchmark_native_cpu_performance.py`, its contract tests, and documentation reconciliation. H0 is architecture, profiling, and baseline work — **no performance optimization has shipped**, no numerical capability, dtype, device, export, registry value, or checkpoint version changed, and the proposed H1–H11 ladder is explicitly evidence-driven and conditional, so a milestone whose premise the measurement does not support is narrowed, reordered, or dropped. **Milestone H1 — the output-allocation contract — has since shipped**: redundant zero-initialization removed from output storage a kernel provably overwrites in full, behind one new C ABI symbol, bit-identical, with the zero-initializing path still the default, `sum` and `narrow_backward` explicitly rejected, completeness proved by deterministic poison tests, and no capability, dtype, device, export, registry value, or checkpoint version changed. **Milestone H2 — native matmul memory access — has since shipped too**: the production matmul's loop order swapped from `i`-`j`-`k` to `i`-`k`-`j` over four destination rows at a time, with **cache blocking measured and rejected**, the pre-H2 triple loop retained verbatim as the shipped generic reference path, metadata-driven dispatch between them inside the kernel, a four-part numerical contract rather than a blanket bit-identity claim (identical accumulation order, bit identity on every non-NaN result, NaN-class equivalence, and NaN payload bits deliberately outside the contract), H1's uninitialized-output contract preserved on both paths, **no exported C ABI symbol added** (still 52), and no capability, dtype, device, registry value, or checkpoint version changed. **Milestone H3 — native metadata and dispatch efficiency — has since shipped as well**, and is **Python-only**: one normalization boundary replacing the four redundant re-validations every `shape_info` call used to perform, private `_checked` primitives for the derived strides/count/contiguity, a private already-validated view constructor sharing one bounds-checking `_bind` with the public one, and lazy read-only per-view `int64` layout arrays whose immutability makes staleness impossible by construction. Every rejection, message, and ordering is preserved, nothing global was introduced, and no public API — cache control, statistic, profiling counter, or dispatch selector — was added. Measured: `_as_int_tuple` calls per MLP training step 815 → 149, view construction 3.2×, an MLP step 1.43×, a CNN step 1.29×, a normalized step 1.51× — and **no measurable change on large kernel-bound matmul or elementwise work**, reported as such. Still 52 exported symbols; no capability, dtype, device, registry value, or checkpoint version changed. **Milestone H4 — native optimizer step efficiency — has since shipped as well**, also **Python-only** and the first Phase-H milestone whose subject is a *training-stack* component rather than the tensor runtime: the step's scalar coefficients built once per step instead of once per parameter (in a private per-step holder that is never stored on the optimizer, so no scalar survives a step or reaches `state_dict()`, a checkpoint, or `close()`), the bias-correction reciprocal evaluated in Python as an **exact substitution** for the native kernel — which literally is `1.0 / x` on the same IEEE-754 binary64 value, proved over 20,000+ values on raw bit patterns — and every temporary released at its last use. Bit-identical against a pre-H4 composition **retained in the test suite and executed natively**; the two-phase stage/commit contract, the single `copy_value_` per updated parameter, the version counting, and the gradient-retention rule are all exactly what they were. Measured by alternating pre/post subprocess rounds: `NativeAdam.step()` 1.58× at (128, 128), 1.48× on a four-parameter MLP with a 256² weight, a large MLP training step 1.23×, a normalized step 1.13×, and the gap against `tensorforge.optim.Adam` 23.8× → 19.7× — with a (512, 512) parameter, the Dropout training step, and `NativeSGD` all reported as **neutral**, and the machine's control-case noise band stated at 0.84×–1.26×. Peak live transient bytes during an Adam step fell 2.6–3.0× and per-parameter allocations 27 → 17, so the time was not bought with memory. Six alternatives were measured and rejected, including scalar materialization (faster small, slower large) and a persistent scalar cache (the forbidden hidden scratch tensor). Still 52 exported symbols; no public API, capability, dtype, device, registry value, or checkpoint version changed**Milestone H5 — native copy and mutation-transfer efficiency — has since shipped**, and is the first Phase-H milestone since H2 to change C++ though **not the ABI** (still exactly 52 exported `tf_*` symbols): the native line's value-transfer primitive `_native_copy` moved from `zeros(shape) + core` — two allocations, a zero-fill pass, and an elementwise-addition pass — to the E3.1 native identity gather `contiguous_copy()`, at one uninitialized allocation and one pass, across all **ten** of its call sites, with `_broadcast_back` **rejected** because it is a genuine broadcast rather than a copy. Over a fixed 18-pattern IEEE-754 sweep **exactly three** patterns moved — the addition normalized `-0.0` and quieted both signs of signaling NaN — while no NaN payload differed under either spelling, so **H2's matmul payload carve-out does not generalize to copies**; the rule H5 states is that a value transfer reproduces its source's bits while an operation follows IEEE arithmetic. One C++ change inside the unchanged export: a metadata-driven second *traversal* (`tf::copy_prefers_contiguous`, hidden visibility, total, pure, no environment variable or CPU probe) that sweeps a row-major source flat and falls back to the retained odometer, bit-identical **by construction** because the identity map performs no arithmetic, proved by a new dependency-free CTest (13 → 14). Nothing became in-place, so every alias and overlap arrangement, parameter identity, storage replacement, version counting, gradient accumulation, state-transaction atomicity, and exact resume are unchanged. Measured: the traversal alone 2.5–5.5× on contiguous sources and 0.94–1.02× on transposed ones (the unchanged odometer, the design's own control); `copy_value_` 2.14× at (512, 512), optimizer `state_dict()` 2.40×, `load_state_dict()` 1.69×, `NativeSGD` 1.15–1.31× — with **`NativeAdam.step()`, every training step, the BatchNorm running update, and copies below ~16 K elements all reported as neutral**. Allocations fell everywhere and no measured peak rose. The harness gained two cases (26 → 28) and the ladder was **reordered**, moving reduction execution to H6. Still 52 exported symbols; no public API, capability, dtype, device, registry value, or checkpoint version changed. **Milestone H6 — native reduction execution efficiency — has since shipped**, the third Phase-H milestone to change C++ and, like H2 and H5, **not the ABI** (still exactly 52 exported `tf_*` symbols). Reductions were the last core family always paying the generic strided indexing cost, and the pre-H6 kernel was re-measured and **decomposed** rather than trusted: at (256, 256) axis 0 a `core.sum` costs 99.7 µs of which the raw native call is **94.8 µs — 95 %**, with the entire Python wrapper at about 5 µs — the opposite of H3's finding, and unambiguously a compiled-loop problem. H6 reused the dispatch shape H2 and H5 each proved: one hidden metadata predicate (`tf::reduce_prefers_contiguous_blocks` — total, pure, allocation-free, a function of layout metadata alone, never a pointer value, alignment, clock, environment variable, or CPU probe, and a false answer is a fallback rather than an error), inside the unchanged `tf_core_sum` export, with the pre-H6 odometer **retained as the shipped generic reference path** — the only path that can address a transposed, narrowed, non-unit-strided, or broadcast source at all. The optimized path is a flat walk over an `outer × mid × inner` factorization, accepted when the source is row-major and the reduced axes form one contiguous run; stride collapsing is implicit and bounded rather than a general layout compiler, nothing is cached, and `keepdims` needs no special case. **Per-output accumulation order is preserved exactly** and the source traversal order is not even reordered, with no reassociation, FMA, Kahan, pairwise, tree, parallel, or horizontal-vector reduction anywhere; signed zeros are proved as raw bit patterns; and the **NaN rule is H6's own, measured rather than inherited from H2** — bit-identical whenever at most one NaN enters an accumulation (every case that occurs in practice), with payloads outside the contract only when two or more NaNs meet in one cell, after four accumulation spellings (including one accumulating through memory exactly as the odometer does) all diverged from the odometer identically, so parity was unavailable at any spelling. H1's rejection of this destination **stands and is confirmed**: both traversals read it, so it stays zero-initialized, and H6 adds no poison test because it introduces no uninitialized destination. Measured against a pre-H6 library on identical `ctypes` calls with outputs proved bit-identical before timing (control band 0.90–1.03×): full reductions up to **3.96×**, 2-D axis reductions **3.24–6.37×**, and — unpredicted — 3-D/4-D reductions **8.60–10.94×**, because the odometer's carry loop scales with rank; `mean` 4.11×, the convolution bias gradient **1.46×**, softmax backward 1.14×, LayerNorm forward 1.16×, and the NumPy gap on contiguous reductions closed from roughly 8–13× to **1.67–3.75×**. Reported just as honestly: **every training step is neutral** (0.99–1.03×), so H6 does not make training faster; **normalization is mostly neutral**, which narrows H7 rather than motivating it; **tiny reductions are neutral**; and a real, repeatable **~10 % regression on 2-D transposed axis-0 fallbacks** is published, with the cause isolated to whole-translation-unit code layout rather than to the extracted call. Memory moved not at all and it is asserted: exactly one allocation per `sum` on both paths. The harness gained three cases (28 → 31) and one dependency-free CTest (14 → 15). Still 52 exported symbols; no public API, capability, dtype, device, registry value, or checkpoint version changed)

- [docs/native_dtype_float32_design.md](docs/native_dtype_float32_design.md) — architecture contract for native dtype generalization and float32 CPU support (Phase I — the **current** phase, with **milestones I0, I1, and I2 complete and I3–I11 not started**. I0 shipped this contract, its guardrail tests, and nothing else: no dtype, no storage change, no kernel, no C ABI symbol, no ctypes declaration, no `NativeTensorCore` method, no `NativeTensor` operation, no module, no optimizer, no export, no registry change, and no checkpoint-format change. **I1 delivered the dtype model and dtype-tagged storage**: frozen ABI codes with one item-size and one canonical-name authority, an owned, runtime-selected `float[]` or `double[]` array behind a type-erased `void*` plus one dtype tag, created with checked `numel × itemsize` so the kernels' pointer arithmetic is valid C++17 over one array object, and the two typed creation exports — **52 → 54 `tf_*` symbols**, the count for the whole phase, with the untyped creators kept unchanged as thin float64 compatibility wrappers and the CTest inventory moving 17 → 18. **I2 delivered the typed transfer boundary and added no export**: the three exports that carry a storage handle *and* a raw host buffer (`tf_storage_copy_from`, `tf_storage_copy_to`, `tf_storage_materialize`) became dtype-general through a **source-level retype** of their host positions from `double*` to `void*` — same symbols, same argument slots, same calling convention, so a previously compiled caller would link and run identically — and `tf_core_contiguous_copy`, the runtime's value-transfer primitive, became dtype-preserving and dtype-strict over its unchanged three-tier traversal, rejecting a mixed float32/float64 pair before anything is written. Transfer is **bit-preserving at both widths** (signed zeros, both infinities, subnormals, quiet NaN payloads, and signalling NaNs, proved as raw IEEE-754 bit patterns rather than by value), `RAW_KERNEL_DTYPES == ("float64",)` records that the seven handle-free raw utility kernels have no dtype to dispatch on and stay float64, and the CTest inventory moved 18 → **19**. **No public capability moved: the native runtime is still float64 CPU only, `float32` is still listed as unsupported — allocatable and movable through the C ABI and computed on by no operation, since every kernel not yet generalized rejects a float32 handle before touching memory — and the native checkpoint format is still version 2 with versions 1 and 2 accepted.** The contract locks the internal dtype model and its frozen ABI codes; dtype-tagged storage measured in logical elements with checked `numel × itemsize` byte arithmetic; storage as the single dtype authority, so every view of one buffer agrees and no view op casts; the ABI strategy — handle-based exports dispatch internally from the storage dtype, so **exactly two** new production symbols are planned (`tf_storage_create_typed` and `tf_storage_create_uninitialized_typed`, 52 → **54**) and per-operation float32 exports are explicitly **rejected**; the split between dtype-general handle-based paths and the seven float64-only raw-buffer utility kernels; one narrow dispatch per operation over templated `float`/`double` kernels; **no casting, no promotion, and no mixed-dtype arithmetic**, rejected before any allocation or mutation; float32 accumulating in float32 with no hidden float64 accumulator; the autograd, module, buffer, RNG/Dropout, and optimizer-state dtype invariants; dtype-aware checkpoint **version 3** with versions 1, 2, and 3 accepted and versions 1 and 2 defined as float64-only formats that are never guessed to be float32; exact deterministic resume proved **separately** for float32 and float64 and never as agreement between them; the preservation of every Phase-H float64 optimization and measurement discipline; the cross-platform and sanitizer gates; and the I0–I11 ladder, in which the public support registry changes at **I9** and at no earlier milestone)

## Limitations

Honest expectations:

- Not production-ready — clarity and correctness take priority over
  performance everywhere, in both lines.
- The stable framework is NumPy on CPU. The native line is an
  experimental C++ **CPU** backend: float64/cpu only, no CUDA backend
  yet, no dtype promotion or casting, and no implicit dispatch into
  `tensorforge.Tensor`.
- The native CNN stack (Phase D), the native classification stack
  (Phase E), the native normalization stack (Phase F), and native RNG and
  Dropout (Phase G) are all
  complete — but "complete" means *these* capabilities
  work and are validated, not that the native line is finished. All three
  normalization modules
  (`NativeLayerNorm`, `NativeBatchNorm1d`, `NativeBatchNorm2d`) shipped
  as modules composed from existing operations with no kernel, F5
  proved their state/checkpoint/ownership/graph-safety contracts by
  exhaustive test, F6 shipped a deterministic normalized training
  example with exact checkpoint resume
  (`examples/native_normalization_training.py`), F7 shipped the
  honest benchmark characterization
  (`benchmarks/benchmark_native_normalization.py`), F8 shipped the
  cross-cutting integration and semantic guardrails
  (`tests/test_native_phase_f.py`), and F9 closed the phase under
  Release/Debug builds and Clang ASan/UBSan/LeakSanitizer. Phase G then
  added native RNG and Dropout — `NativeGenerator`, the stateless Dropout
  Core, the differentiable `NativeTensor.dropout`, and `NativeDropout` —
  closing at G10 under the same validation matrix. What the
  native line still does **not** have: data loaders, native integer
  tensors, further dtypes or devices, CUDA, AMP, a generic random-number
  API, `Dropout2d`/`Dropout3d`, and any implicit
  dispatch into `tensorforge.Tensor`. Native checkpoints capture
  parameters, persistent buffers, optimizer state, and generator state,
  but **no** data-loader position, shuffle or epoch state, scheduler
  state, Python `random`, or NumPy global RNG, and the classification
  loss supports
  `"mean"`/`"sum"` only — no `reduction="none"`, class weights,
  `ignore_index`, label smoothing, or soft targets. See the
  [native support matrix](docs/native_support_matrix.md).
- Both lines' convolution and pooling use deliberately naive loops (the
  stable `Conv2d`/`MaxPool2d` and the native kernels alike: no im2col,
  BLAS, threading, or SIMD).
- Benchmarks are hardware-specific characterizations with no universal
  speed claims; the naive native kernels can lose to NumPy's BLAS.
- No real datasets and no external ML libraries; every example runs on
  small synthetic data.

## Status

**v3.0 — the stable Python framework line is complete**, covered by the
test suite and documented. **The experimental native line — merged into
`main`, and reached only through the explicit `tensorforge.backends` and
`tensorforge.experimental` namespaces — has completed Phases A–G**:
Phase A (native CPU runtime),
Phase B (native autograd), Phase C (the native training stack),
Phase D (the native CNN stack), Phase E (native classification and
stable math), and Phase F (native normalization and stateful buffers)
are all complete, and so is Phase G (native RNG and Dropout). **Phase H —
native CPU performance and runtime efficiency — is complete (H0–H10)**,
and is the latest completed phase:
every shipped training workload is **1.50×–3.89× faster than it was at
the H0 baseline**, with bit-identical results, and **no numerical
capability, dtype, device, export, registry value, or checkpoint version
moved at any milestone**. **Phase I — native dtype generalization and
float32 CPU support — is the latest phase; milestones I0 through I7
are complete and I8–I11 are not started.** I0 was the design lock and
documentation reconciliation, shipping
[docs/native_dtype_float32_design.md](docs/native_dtype_float32_design.md)
and its guardrail tests and nothing else. **I1 built the dtype
foundation**: the C++ dtype model with frozen ABI codes and one item-size
authority, dtype-tagged storage allocated with checked
`numel × itemsize` byte arithmetic, and the two typed creation exports —
taking the library from 52 to **54** `tf_*` symbols, the count for the
whole phase. **I2 built the typed transfer boundary and added no export**:
the three exports carrying a storage handle *and* a raw host buffer became
dtype-general through a source-level retype of their host positions to
`void*` (same symbols, same slots, same calling convention), and
`tf_core_contiguous_copy` — the value-transfer primitive — became
dtype-preserving and dtype-strict, so float32 values move in, out, through
any view layout, and storage-to-storage **bit for bit**, signalling NaNs
and signed zeros included. **I3 made float32 computable and added no
export**: `add`, `subtract`, `multiply`, `relu`, `relu_backward`, `sqrt`,
`reciprocal`, `exp`, and `log` dispatch once from the storage tag into
templated `float`/`double` kernels, with NumPy-style broadcasting, all
three Phase-H traversal tiers instantiated for both widths from one source,
dtype-preserving outputs, and mixed dtype rejected before any allocation.
**I4 made float32 accumulate, and added no export**: `sum`, `mean`,
`matmul`, and `narrow_backward` are dtype-general, with H6's
contiguous-block factorization, H2's `i`-`k`-`j` row sweep, and the retained
generic paths beside them all instantiated for both widths from one source
and both metadata predicates untouched; the two scalar storage primitives
narrow their `double` argument **once, before the loop**; and private
float32 `NativeTensor` graphs differentiate end to end with every gradient,
temporary, and constant at the graph's dtype. It is also where "float32
accumulates in float32" became a *measured* claim rather than a structural
one — on `1.0` plus eight copies of `2**-24`, binary32 stays at exactly
`1.0` while binary64-then-narrow lands four ULPs higher, and TensorForge is
asserted equal to the first and unequal to the second on every shipped path.
**I5 made float32 convolve and pool, and added no export**: all three
Conv2d directions and both MaxPool2d directions dispatch once from the
storage tag into templated kernels, H9's traversals and geometry predicates
are one source at both widths, Conv2d accumulates in the element type with
the witness proved in every direction on both paths, private float32
graphs differentiate through convolution and pooling, and the MaxPool2d
winner buffer stays **private float64 at every value dtype** with its
`2**53` exact-plane bound unchanged — so a float32 pool over a plane beyond
float32's `2**24` exact-integer range still records its winner offsets
exactly.
**I6 made float32 classify, and added no export**: softmax, log-softmax,
and the fused cross-entropy forward and backward dispatch once from the
storage tag into templated kernels, with the maximum scan, the shift, the
exponentials, the normalizing sum, the log-normalizer, the batch-loss
accumulator, and every backward contribution at the element type; the saved
probabilities carry the graph dtype; log-softmax is still its own fused
log-sum-exp kernel and never `softmax().log()`; the backward still reads
those saved probabilities and cannot even name the logits; and the class
**targets stay host `int64` metadata at every width**, so no integer tensor
dtype was introduced. It is also where the float32 stability statement
picked up its one honest qualification: the maximum shift guarantees no
*exponent* overflows, but a slice whose **spread** exceeds the element
type's largest finite value makes the shift itself overflow to `-inf` — a
correctly-rounded IEEE result for a value with no representation at that
width, reachable at binary64 too past ~1.8e308. `softmax` is unaffected and
still exact; `log_softmax` and `cross_entropy` report `-inf`/`+inf` as
values, never errors. The counterexample is recorded and tested rather than
papered over with a widened intermediate.
**I7 made float32 a module dtype, and added no export**: six state-owning
constructors — `NativeParameter`, `NativeLinear`, `NativeConv2d`,
`NativeLayerNorm`, `NativeBatchNorm1d`, and `NativeBatchNorm2d` — take a
keyword-only `dtype` accepting exactly the two widths and defaulting to
float64, through one shared validator; affine parameters, both BatchNorm
running buffers, the evaluation snapshots, and every scalar a composed
normalization forward materializes are at the module's dtype; the atomic
two-buffer running-statistics transaction gained one dtype validation and
nothing else; and **initialization did not move** — the same local
`default_rng(seed)` stream in the same order, so a float32 layer with seed
*S* holds exactly `float32(the float64 draw with seed S)`. Dropout was the
last family out: the export keeps its exact ABI shape, and the **random
derivation is untouched**, so one `(seed, call_index, element count)` key
drops exactly the same elements at both widths and only the two multiplier
values differ — the kept one being the binary64 reciprocal narrowed once.
**The native runtime is still declared float64 CPU only**: `float32`
remains listed as unsupported. Every *numerical* family is now
dtype-general and the experimental state-owning modules can be constructed
at float32, but no public tensor constructor produces a float32 tensor, no
optimizer accepts a float32 parameter, and a version-2 checkpoint refuses
to save a float32 model rather than writing an archive it could never read
back. Those gaps are milestones I8 and I9, and the registry moves at I9.
The native checkpoint format is still version 2 with versions 1 and 2
accepted. The
contract's **exactly two** new C ABI symbols for the whole phase
(52 → **54**) are now spent; it rejects per-operation float32 exports, forbids casting,
promotion, and mixed-dtype arithmetic, requires float32 to accumulate in
float32, designs checkpoint version 3 without activating it, requires
exact deterministic resume to be proved separately for each dtype rather
than as agreement between them, and moves the public support registry at
milestone **I9** and at no earlier one. Phase C shipped
parameters, modules, state dictionaries,
Linear/ReLU/Sequential, MSE loss, parameter versioning with stale-graph
safety, `sqrt`/`reciprocal` optimizer primitives, SGD and adaptive Adam,
in-memory optimizer state snapshots, pickle-free native checkpoint files,
end-to-end deterministic MLP training, and deterministic in-memory and
file resume — with cross-cutting failure, lifetime, and ownership
guardrails. **Phase D** added every native CNN layer — `NativeFlatten`,
the differentiable convolution layer (`NativeConv2d`) over the native
`conv2d` operation, and the pooling layer (`NativeMaxPool2d`) over the
native `maxpool2d` operation with its private saved winners — plus the
end-to-end training + checkpoint-resume proof
(`examples/native_cnn_training.py`: 40 deterministic steps, 98.6% loss
reduction, and a checkpoint-interrupted run that reproduces the
uninterrupted one exactly), cross-cutting integration tests, honest CNN
benchmarks, and ASan/UBSan validation of the whole native stack.
**Phase E — Native Classification and Stable Math — is complete**
(milestones E0–E10): its architecture contract is locked
([docs/native_classification_design.md](docs/native_classification_design.md),
milestone E0) and milestones **E1–E4 shipped the differentiable native
`exp`, `log`, and the fused stable `softmax` and `log_softmax`** — C++
kernels, self-validating guarded C ABI, `NativeTensorCore` and
`NativeTensor` layers. `exp` and `log` are the phase's two backward
archetypes: `exp` reads its saved output and records no parameter
version, while `log` rereads the live input (`upstream × reciprocal(x)`,
no division operation added) and version-guards a direct parameter so a
post-forward mutation fails before any gradient moves. `softmax` and
`log_softmax` are the phase's two fused probability transforms — a
maximum-shift kernel and a log-sum-exp kernel over any axis behind a
contiguous-only ABI, each with a saved-output backward composed from
existing Core operations rather than a dedicated kernel. `log_softmax`
is deliberately **never** `softmax().log()`: it forms no probability and
performs no division, so it stays accurate where the composed form
collapses to `-inf`. **E5 shipped the fused `cross_entropy` Core
contract** — a single kernel producing the stable scalar loss *and* the
private saved probabilities per row (never `-log(p[target])`), a
backward that reads only those saved probabilities, the copied `int64`
targets, and a native one-element upstream (**it never rereads the
logits**), and strict targets that reject `bool` and floating-point
labels and are copied so caller mutation cannot reach the kernel. **E6
shipped the differentiable `NativeTensor.cross_entropy(targets,
reduction="mean")`** over that contract, adding no kernel and no
numerical change: one scalar-output autograd node whose private saved
probabilities are **graph-owned** — retained under `retain_graph=True`,
released exactly once with the graph history, closed immediately when no
gradient is required — and whose backward never rereads the logits, so
no parameter version is recorded and mutating the logits after the
forward leaves the gradient correct for the forward that ran.
**E7 completed the public surface**: `NativeCrossEntropyLoss`, a
stateless module whose whole forward delegates to that operation, and
`native_accuracy`, a deliberately **reporting-only** helper — no kernel,
no Core method, no autograd node — that materializes once through the
explicit public `to_numpy()` boundary, takes a NumPy argmax, and returns
a plain `float` without building a graph or touching any gradient.
**E8 proved the assembled stack end to end without adding anything to
it**: `examples/native_classification_training.py` trains a
`NativeConv2d(1, 4, 3)` → `NativeReLU` → `NativeMaxPool2d(2)` →
`NativeFlatten` → `NativeLinear(16, 3)` classifier — no softmax layer,
raw logits straight into `NativeCrossEntropyLoss` — on twelve fixed 6×6
images in three classes for 40 deterministic `NativeAdam(lr=0.05)` steps
(loss 1.159638 → 0.000101, reporting accuracy 0.3333 → 1.0000), then
interrupts at step 15, checkpoints model **and** optimizer state through
the existing pickle-free path (format **version 1**, unchanged), and
resumes into a **fresh** model/optimizer pair that reproduces the
uninterrupted run **exactly** — the whole remaining loss suffix, every
parameter, both Adam moment buffers and step counters, the final logits,
the predictions, and the accuracy. It is an integration proof on one
fixed task: no speed and no generalization is claimed.
**E9 added the characterization benchmark**,
`benchmarks/benchmark_native_classification.py` — seven cases (`exp`,
`log`, `softmax`, `log_softmax`, cross-entropy forward, cross-entropy
backward, and one complete classification training step), each with a
correctness gate that runs **before** any timing, each labelled with the
reference it actually used (`stable_tensorforge`, `numpy` where the
stable line has no direct operation, or `native_only`), and each reported
as a median with min/max/spread over repeated `time.perf_counter_ns`
measurements taken after warm-up with setup and cleanup outside the
timer. It has `--smoke` and `--json` modes and writes no result file.
Observed ratios are **local characterizations, not guarantees**: no test
asserts a speed, no timing number is committed as a promise, and there is
no CI performance gate. **E10 closed the phase with no new numerical
capability**: cross-cutting integration tests
(`tests/test_native_phase_e.py`), Release **and** Debug native builds
(10/10 CTests each, zero warnings), Clang AddressSanitizer and
UndefinedBehaviorSanitizer validation of the whole classification stack
with zero diagnostics attributable to TensorForge, a practical
LeakSanitizer pass finding no native leak, the full Python regression
suite, and documentation reconciliation across every status surface.
Phase E expanded nothing beyond float64/CPU and added no implicit
stable/native dispatch.

**Phase F — Native Normalization and Stateful Buffers — is complete
(F0–F9).**
Its architecture
contract is locked in
[docs/native_normalization_design.md](docs/native_normalization_design.md)
(milestone **F0**, complete: design and repository reconciliation only,
adding no numerical behavior), as is **F1** (a private atomic
native-buffer state transaction, the `load_state_dict` refactor onto it,
and the `STATE_SUPPORT` persistent-buffer reconciliation — state
management and capability reporting only, no normalization mathematics),
and **F2** (`NativeLayerNorm` — the first native normalization module:
stateless, differentiable through the mean and the population variance,
and composed entirely from existing native operations with `sqrt(var +
eps)` ordering and no kernel, ABI symbol, `NativeTensorCore` method,
custom backward, or `NativeTensor` normalization operation;
`"NativeLayerNorm"` is now in `NATIVE_MODULES` and the exports, and
`"layernorm"` has left `UNSUPPORTED`), and **F3**
(`NativeBatchNorm1d` — the **first stateful native numerical module**:
`(N, C)` batch normalization with differentiable training statistics
(gradients flow through the batch mean *and* the population variance),
persistent native `running_mean`/`running_var` buffers advanced
graph-free by one **atomic two-buffer transaction** over the F1
primitive (both identities preserved, replaced cores closed exactly
once, no parameter version moved), and evaluation from **graph-safe
immutable snapshots** of those buffers, so a later training step, or a
buffer-only state or checkpoint load, can never change an earlier eval
graph's gradient — while a *full* checkpoint load that also replaces
`gamma`/`beta` still stales that graph through the unchanged
parameter-version rule, which is correct and deliberately unweakened; again composed from existing operations, so again no kernel,
ABI symbol, `NativeTensorCore` method, custom backward, or
`NativeTensor.batch_norm` operation, and the native checkpoint format
stays at version 1; `"NativeBatchNorm1d"` is now in `NATIVE_MODULES` and
the exports, while `"batchnorm"` stayed in `UNSUPPORTED`), and **F4**
(`NativeBatchNorm2d` — NCHW `(N, C, H, W)` batch normalization reducing
over **N, H, and W**, so each channel gets one population mean and one
population variance over `N * H * W` values. It is built on the **same**
shared private implementation as the 1-D shape and declares only its
rank, its reduction axes, its `(1, C, 1, 1)` broadcast layout, and the
channels-last permutation its rank-1 `gamma`/`beta` need — the
activation is transposed for the affine step, never the parameters, so
the existing direct-parameter stale-value guard is preserved exactly.
Running buffers stay `(C,)`; again no kernel, ABI symbol,
`NativeTensorCore` method, custom backward, or `NativeTensor.batch_norm`
operation, and the checkpoint format stays version 1;
`"NativeBatchNorm2d"` is now in `NATIVE_MODULES` and the exports, and
with both shapes live **`batchnorm` has left `UNSUPPORTED`**, which now
reads exactly `("dropout", "float32", "cuda", "amp")`).
**That completes the numerical normalization *module* surface.** **F5 is
complete**: the exhaustive state/checkpoint, ownership, and graph-safety
hardening — a focused `tests/test_native_normalization_state.py` plus
narrow additions to the generic buffer and checkpoint suites — proves the
design's §7–§10 contracts by executable test (canonical dotted buffer
keys, independent state snapshots, strict/non-strict loads, exact
never-casting metadata validation, mixed parameter/buffer transaction
atomicity, buffer identity across state and checkpoint loads, exact
eval-output reproduction, the buffer-only-versus-full stale-graph
distinction, the save/corrupt-load failure boundaries, eval-graph snapshot
safety under `retain_graph` and a failed retryable backward, and the
live-storage baselines); it is **tests and documentation only — no
numerical behavior, no new capability, and the checkpoint format stays
version 1**. **F6 is complete**: `examples/native_normalization_training.py`
trains a `Linear → BatchNorm1d → ReLU → LayerNorm → Linear` regressor for
24 deterministic `NativeAdam` steps with `NativeMSELoss` (a 98.9% loss
reduction), then checkpoints at step 10 and resumes into a completely
fresh model/optimizer pair that reproduces the remaining losses, every
parameter, the NativeAdam state, the **BatchNorm running statistics**, the
final training-step prediction, and the final **evaluation-mode** output
exactly — proving two uninterrupted runs are bit-identical and the
interrupted resume is exact (format version 1 unchanged, training flags
runtime-only). It **adds no capability**: one example and its integration
test, no operation, kernel, or schema change. **F7 is complete**:
`benchmarks/benchmark_native_normalization.py` characterizes the stack
with nine cases — the LayerNorm forward and backward, the BatchNorm1d
training forward, evaluation forward, and backward, the BatchNorm2d
training forward, evaluation forward, and backward, and one complete
F6-style normalized training step. Every case is **correctness-gated
before any timing** (a failed gate exits nonzero and publishes nothing);
six are measured against `stable_tensorforge` equivalents on the same
inputs, epsilon, momentum, affine values, running state, initial
parameters, and optimizer hyperparameters, while the three BatchNorm2d
cases are labelled `native_only` for timing because the stable line has
**no public `BatchNorm2d`** to compare against — they still carry a
rigorous correctness oracle (an explicit NumPy NCHW population-statistics
formula, and for the backward the stable `BatchNorm1d` on the equivalent
`(N*H*W, C)` sample matrix transformed back to NCHW). Medians are
reported with min, max, and spread after warm-up, `--smoke` and `--json`
modes exist, **no result file is written, no speed is asserted, no timing
number is committed, and no CI job asserts a duration** — measurement
only, adding no capability. **F8 is complete**:
`tests/test_native_phase_f.py` proves the cross-cutting interactions no
single-module suite can — one integrated `Conv2d → BatchNorm2d → ReLU →
MaxPool2d → Flatten → Linear → BatchNorm1d → ReLU → LayerNorm → Linear`
classifier over **raw logits** and the fused loss, trained by
`NativeAdam` and resumed **exactly** from one version-1 checkpoint (the
loss suffix, every parameter, the NativeAdam state, **all four**
running-statistic buffers, the final training logits, and the final
evaluation-mode logits, predictions, and accuracy); BatchNorm eval
snapshots, MaxPool2d winners, and cross-entropy probabilities coexisting
in one eval graph and releasing exactly once; buffer-only mutation
leaving an earlier eval graph valid while a full checkpoint load or an
affine `copy_value_` correctly stales it through the unchanged parameter
rule; the versioning archetypes; shared and frozen parameters; a
non-contiguous NCHW input through the whole stack; strict stable/native
separation; and each failure boundary tested **honestly** — BatchNorm
transactions are per module, so one whole training step is *not*
presented as globally transactional. Tests and documentation only,
adding no capability.
Milestone **F9 is complete** — the phase closure: fresh Windows Release
**and** Debug builds each passing the full existing 10-test CTest suite
with zero project compiler, linker, or CMake warnings, and the active
runtime proved to stay the Release DLL; a fresh Clang 18.1.3
ASan+UBSan build in WSL2 Ubuntu 24.04 whose instrumentation is *proved*
rather than assumed (22 `__asan*` and 13 `__ubsan*` dynamic symbols, and
a library that refuses to load without the sanitizer runtime); 10/10
sanitized native CTests with leak detection enabled; **1,968** sanitized
normalization-focused Python tests with zero ASan and zero UBSan
diagnostics; the F6 example reproducing its exact resume and the F7
benchmark passing all nine correctness gates under the sanitized
library; and a practical LeakSanitizer lifecycle whose native
live-storage counter returned **exactly** to baseline, its remaining
process-exit allocations identified honestly as CPython/NumPy shutdown
retention with no TensorForge frame and no suppression file —
**validation and documentation only, adding no numerical capability**.
So **Phase F is complete**, and no
normalization *operation*, kernel, or C ABI symbol exists at all.
Closing Phase F closes that phase, not the project: the native line is
still experimental, still float64/CPU only, and still not
production-ready.

**Phase G — Native RNG and Dropout — is complete.** All eleven milestones,
G0 through G10, have landed. **G0**
locked the architecture
contract in
[docs/native_rng_dropout_design.md](docs/native_rng_dropout_design.md) —
Python-managed generator state (an explicit 64-bit seed and call counter
with an algorithm identifier), stateless native random kernels, inverted
Dropout with a graph-owned multiplier mask, one generator call consumed
per *successful* stochastic forward and none on any failure — behind a
lock-protected, token-validated reservation protocol so no two callers
can ever receive the same call index — generator state registered on
`NativeModule`, native checkpoint **version 2** that records the
generator *alias topology* (which layers share a stream, not just the
saved states) with a locked version-1 compatibility rule, and
whole-checkpoint transaction atomicity in which any ordinary synchronous
commit failure rolls back parameters, buffers, optimizer state, and
generator state together. **G0 was design, documentation, and guardrails
only.**

**G1** then shipped the state itself: `NativeGenerator` — a pure-Python
value holder carrying the algorithm identity, a 64-bit seed, and a
counter of *committed* stochastic calls, with atomic
`state()`/`load_state()`/`reseed()`/`reset()`, identity (never value)
semantics, no copying and no `close()`, and a lock-protected
token-validated call transaction — plus generators as a **fourth**
`NativeModule` registration category beside parameters, buffers, and
child modules (`register_generator`, `generators()`,
`named_generators()`, `generator_state_dict()`,
`load_generator_state_dict()`). **G1 generates no random values by
itself.**

**G2** then shipped the stateless Dropout-forward **Core**: the exact
`tensorforge.splitmix64` derivation in unsigned 64-bit arithmetic
(`mix64(seed + GOLDEN*(call+1))`, then `mix64(stream + GOLDEN*(i+1))`,
then `(bits >> 11) * 2**-53` compared against `p`), an inverted-Dropout
float64 CPU kernel in `cpp/src/random.cpp`, the guarded
`tf_core_dropout_forward` export, and
`NativeTensorCore.dropout_forward(p, *, seed, call_index)` with the
private `_dropout_forward_with_mask` that also returns the multiplier
mask. It is **stateless**: the whole random key arrives as two explicit
integers, and the Core touches no `NativeGenerator` — no reservation, no
commit, no counter movement — while no C++ translation unit holds random
state of any kind. The keep/drop pattern is keyed by the **logical**
row-major element index, so a transposed, narrowed, or nonzero-offset
view gets the same mask as a contiguous tensor of the same shape.
Committed known-answer vectors are asserted identically from C++ and
Python.

**G3** then shipped the differentiable operation over that Core:

```python
from tensorforge.experimental import NativeGenerator, NativeTensor

g = NativeGenerator(1234)
x = NativeTensor.from_array(values, requires_grad=True)
y = x.dropout(0.25, generator=g)        # g.calls: 0 -> 1
y.backward(gradient=upstream)           # grad = upstream * the saved mask
```

The generator is **required and keyword-only** — there is no default,
process-global, or module-global stream, no implicit per-call generator,
and no NumPy or Python `random` fallback. One successful stochastic
forward consumes exactly one call; `p == 0` returns the input object
itself and consumes none; every failure before the commit cancels the
reservation, so the next forward reuses the same index; and backward
consumes none, ever. Backward reads only the upstream gradient and the
**graph-owned multiplier mask** the forward saved — the third member of
the saved-state family beside MaxPool2d's winners and cross-entropy's
probabilities — so it never rereads the input, never redraws, and never
touches the generator, which is why mutating the input or reseeding the
generator afterwards leaves an existing graph's gradient exactly as it
was. There is no Dropout backward kernel: the gradient is the existing
native `multiply`. The whole registry footprint is one name, `dropout`,
in `AUTOGRAD_OPS`.

**G4** then shipped the public module over that operation:

```python
from tensorforge.experimental import NativeDropout, NativeGenerator

layer = NativeDropout(0.5, seed=1234)   # owns its generator
shared = NativeGenerator(7)             # ...or two layers share one stream
a, b = NativeDropout(0.5, generator=shared), NativeDropout(0.5, generator=shared)

layer.train();  y = layer(x)            # stochastic; calls 0 -> 1
layer.eval();   assert layer(x) is x    # identity; consumes nothing
```

`NativeDropout(p=0.5, seed=None, generator=None)` — `seed` and
`generator` are **mutually exclusive**; without an explicit generator the
module creates and owns one, and with one it registers **that exact
object**, never a copy. The generator is first-class registered state:
it appears in `named_generators()` and `generator_state_dict()` and is
deliberately absent from `state_dict()`, which stays tensor-only. Training
delegates to the G3 operation, so a successful forward consumes exactly
one call and a failed one none; evaluation returns the input object
itself, so any number of eval forwards leaves **no gap in the stream**;
and `p == 0` is identity in both modes. The module owns no native storage.

**G5** closed the persistence gap G4 left open: the native checkpoint
format is now **version 2** (the format *name* is unchanged), and it
carries a `generators` manifest section holding every registered
generator's `algorithm`, `algorithm_version`, `seed`, and `calls` — seeds
and counters as canonical decimal strings, because a `uint64` above
`2**53` cannot survive a JSON double — plus the complete
**alias topology**: every registered path mapped to its canonical
generator, so *shared versus independent* streams are restored, not just
the numbers. Generator state adds **no array** to the archive. A load
restores each generator **in place**, preserving object identity and
every sharing relationship, and validates the archive's topology strictly
in both directions against a real `named_generators()` traversal — a
missing path, an extra path, or a shared-versus-independent difference
fails before anything changes. A version-1 archive still loads into a
model with **no** generators and is **rejected** for one that has them,
naming them: no seed and no counter is ever fabricated. A load is one
transaction over the whole archive — model, buffers, optimizer, and
generators commit under a single rollback guard, so any synchronous
failure (a deliverable `KeyboardInterrupt` included) restores all four
together rather than leaving a mixed checkpoint. It is also
**serializable**, not merely deadlock-free: every participating state
replacement — the checkpoint load, `load_state_dict`,
`load_generator_state_dict`, both optimizers' state loads — and the
checkpoint save snapshot run under one private shared `RLock`, with
generator locks taken under it in the global `id()` order. Two concurrent
loads therefore produce one archive's state followed by the other's,
never model state from one beside optimizer or generator state from the
other. Ordinary training mutation deliberately does not take that guard,
so thread-safe concurrent training snapshots are not claimed.

Milestone **G6 is complete** — hardening, and no new capability.
`tests/test_native_phase_g_hardening.py` attacks the finished G1–G5
surface: the reservation transition matrix, the exact `uint64` boundary,
forced concurrent interleavings (bounded joins, no sleeps), the
deterministic Core's structural key properties beside its committed
vectors, every pre-commit and post-commit failure position of the call
transaction across four exception classes, all four graph-owned
saved-resource families in one graph, a 76-case checkpoint corruption
matrix, whole-transaction rollback at every commit position, save-seam
destination atomicity, and repeated success-and-failure lifecycle loops
measured against a real native live-storage baseline. It changed no C++,
ABI, ctypes, Core method, operation, module, export, schema field, or
registry value, and added no benchmark and no example; it found and fixed
exactly one runtime defect — a cleanup-failure `__context__` chain that
could become cyclic — with a dedicated regression guard.

Milestone **G7 is complete** — the end-to-end exact stochastic resume,
and **no new capability**. `examples/native_dropout_training.py` trains
`NativeLinear(4, 8)` -> `NativeBatchNorm1d(8)` -> `NativeReLU` ->
`NativeDropout(p=0.5, seed=20240707)` -> `NativeLayerNorm(8)` ->
`NativeLinear(8, 3)` over raw logits with `NativeCrossEntropyLoss` and
`NativeAdam` on a fixed twelve-sample three-class task computed from an
explicit formula, in three fixed batches on a schedule that is a **pure
function of the training step**. It carries all four TensorForge-owned
state families at once — parameters, persistent BatchNorm running
buffers, a registered `NativeGenerator`, and NativeAdam moments with
per-parameter step counters — so an incomplete restore diverges
immediately. Two uninterrupted runs are bit-identical; an interrupted run
checkpointed after 7 **completed** steps (deliberately mid-cycle in the
batch schedule), whose model, optimizer, and generator are **released
before the resume begins**, reloads into a completely fresh set built
with a *different* Dropout seed and reproduces the uninterrupted run by
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
module's next Dropout output against `NativeTensorCore.dropout_forward`
at the exact restored `(seed, call_index)`, advancing `calls` by exactly
one. **External loop progress is carried explicitly**, as validated JSON
metadata (`{"training_step": ..., "next_batch_index": ...}`), because
checkpoint v2 captures TensorForge-owned state and **not** data-loader
position, batch order, shuffle state, epoch counters, scheduler state,
Python's `random`, or NumPy's global RNG — a missing or inconsistent
field raises rather than silently restarting from step 0.
Reproducibility is exact **for the state actually captured**;
full-program determinism is not claimed. The whole milestone is one
example, one test module, and documentation: **no** C++, C ABI symbol,
ctypes declaration, Core method, autograd operation, module, export,
schema field, checkpoint version, benchmark, or registry value changed.

Milestone **G8 is complete** — the honest benchmark characterization,
and **no new capability**. `benchmarks/benchmark_native_dropout.py`
measures thirty-five cases in eight families: the stateless Core against
an **exact, bit-for-bit** vectorized NumPy implementation of the same
locked derivation, size scaling from a rank-0 scalar to a
six-figure-element tensor, four physical layouts over one logical shape
(contiguous, transposed, narrowed non-contiguous, and offset-contiguous),
a five-value probability sweep at three layers, the no-grad /
differentiable / backward-only / forward-plus-backward operation layers,
the module's training and identity paths, and one complete Dropout
training step. Correctness is gated **before** timing everywhere — the
committed known-answer vectors first pin the harness's reference and then
the native kernel — and the `NativeTensor` and `NativeDropout` cases are
labelled `native_only`, publishing no ratio, because no NumPy expression
has their generator transaction, ownership, or graph. `--smoke`
(`--quick`), `--json`, and `--json-out` are supported; **no result file
is written unless a destination is named**, and there is no speed
assertion, no committed timing number, and no CI timing threshold
anywhere. Results are a machine-specific snapshot, not a performance
contract, and **nothing was optimized to improve a number** — G8 changed
no runtime file.

Milestone **G9 is complete** — the cross-cutting Phase-G integration
suite, and **no new capability**. `tests/test_native_phase_g.py` builds one
test-only model that carries every registered state family at once —
`NativeConv2d` -> `NativeBatchNorm2d` -> `NativeReLU` -> `NativeMaxPool2d`
-> `NativeDropout` -> `NativeFlatten` -> `NativeLinear` ->
`NativeBatchNorm1d` -> `NativeReLU` -> `NativeLayerNorm` ->
`NativeDropout` -> `NativeLinear` over raw logits with
`NativeCrossEntropyLoss` — with the two Dropout layers sharing **one**
registered generator, and proves the interactions no single-module suite
can: all four saved-resource families (Dropout masks, MaxPool2d winners,
BatchNorm eval snapshots, and cross-entropy probabilities) alive in one
graph and released exactly once; deterministic training and **exact**
version-2 resume into a completely fresh model, optimizer, and generator
set, with a negative control that diverges; the generator-topology matrix
(shared, independent, equal-valued-but-distinct, renamed, missing, extra)
with every mismatch rejected **before** any state family changes;
evaluation consuming no generator call anywhere and training resuming at
the exact next index; `p == 0` through the whole model; non-contiguous
NCHW and strided views; the whole-checkpoint transaction rolled back at
every commit position; four deterministic concurrency cases proving the
participating state transactions serialize; a Phase A–F regression
matrix; and native live storage returning exactly to baseline across
success and failure cycles. It changed **no** runtime file and found no
defect.

Milestone **G10 is complete, and it closed the phase.** The validation
matrix ran with observed results: fresh Windows **Release** and **Debug**
builds (Visual Studio 17 2022, MSVC 19.44.35228.0), each passing
**11/11 native CTests** with zero project compiler, linker, and CMake
warnings, and the active runtime proved to remain the Release DLL. A fresh
Clang 18.1.3 `-DTF_SANITIZE=address,undefined` build in WSL2 Ubuntu
24.04.4 with **instrumentation proved rather than assumed** — 22 `__asan*`
and 14 `__ubsan*` dynamic symbols beside the 51 exported `tf_*` symbols,
and a library that refuses to load without the sanitizer runtime — then
**11/11 sanitized CTests** with leak detection on, **3,166 sanitized
Python tests** across 43 suites, the G7 example reproducing its exact
resume, and the G8 benchmark passing every correctness gate, all with
**zero ASan and zero UBSan diagnostics**. A practical LeakSanitizer
lifecycle returned native live storage **exactly to baseline (0 → 0)**,
and not one of its remaining process-exit allocations names a TensorForge
frame — **no suppression file was added**.

Only after all of that did the single registry line move: at **G10**, and
not one milestone earlier, `dropout` **left** the unsupported list, which
now reads exactly `("float32", "cuda", "amp")`. The claim is deliberately narrow — **native Dropout is
supported in TensorForge's experimental native float64 CPU backend**. That
is not a stable-framework claim (`tensorforge.nn.Dropout` is its own
separate NumPy implementation), not float32, CUDA, or AMP support, not a
generic random-number API, and not a production-readiness or speed claim.
Reproducibility is exact for the state actually captured; Python's
`random`, NumPy's global RNG, data-loader position, and scheduler state
are **not** captured and full-program determinism is not claimed. Ordinary
concurrent *training* is not claimed thread-safe either: the
serializability guarantee covers the participating state transactions, not
an optimizer `step()` racing a forward.

**Phase H — Native CPU Performance and Runtime Efficiency — is complete
(H0–H10), and is the latest *completed* phase.** H10 re-measured the whole phase against a reconstructed and verified H0 baseline (52 cases, **zero checksum mismatches** — every figure compares implementations that produced bit-identical results), resolved the acceleration gate as three documented rejections with measurements (SIMD, threading/OpenMP, BLAS), assessed `tf_core_narrow_backward` and the small-operation boundary floor and implemented neither, ran the full Release/Debug/Linux/sanitizer/lifecycle matrix, and closed the phase. **Every shipped training workload is 1.50×–3.89× faster than at H0**, matmul 4.71×, Conv2d kernels 2.59×–4.64×, reductions 3.78×–5.06×, with no allocation count or memory peak raised anywhere — and across the whole phase **no capability, dtype, device, registry value, public API, checkpoint field, or checkpoint version moved**, with exactly **one** C ABI symbol added (`tf_storage_create_uninitialized`, at H1): 51 → **52**.

Reported as honestly as the wins: the controls held (the unchanged
raw-buffer matmul 0.99×, NumPy 1.03×, storage allocation 0.98×, Dropout
1.00×), and **`to_numpy` at 0.95× is the one reproducible regression** —
its compiled traversal is byte-identical source measuring 0.975×–1.008×,
so what changed is that H3's and H7's much cheaper wrapper no longer
hides it. Remaining limitations are stated rather than smoothed over: the
gap to a tuned multi-threaded BLAS is 3.6×–9.3× and widens with size;
convolution is entirely scalar; a small operation still costs a few
microseconds, because **60 % of that is the owning allocation and 19 % is
building the result's Python ownership objects, against 12 % for the
ctypes crossing** — an architectural floor, not a defect. Every number is
a local characterization of one machine, asserted by no test.

H0 is an
architecture, profiling, and baseline milestone: it locked the contract
in
[docs/native_cpu_performance_design.md](docs/native_cpu_performance_design.md),
shipped `benchmarks/benchmark_native_cpu_performance.py` — a unified
measurement harness that separates the layers a caller actually pays for
(NumPy, the stable line, the raw-buffer kernels, `NativeTensorCore`,
`NativeTensor` with and without a graph, backward, an optimizer step, and
a whole training step) across twelve workload families, gates correctness
**before** timing everywhere, publishes no ratio where no honest
equivalent exists, and writes no result file — and its contract tests.

**Nothing was made faster.** Every kernel is exactly the deliberately
plain reference loop Phase G left behind; no numerical capability, dtype,
device, export, capability-registry value, or checkpoint version changed,
and the checkpoint format remains version 2 with versions 1 and 2
supported. What H0 produced is *evidence*, and it is deliberately
surprising in places: the largest measured factors are an allocator
behavior and a memory access pattern rather than raw arithmetic, the
Python-side per-call metadata path costs several times the ctypes
boundary it wraps, and the `NativeTensor` wrapper and its autograd graph
node are measurably **not** a bottleneck — a negative result that rules
out a family of plausible optimizations.

**Milestone H1 — the output-allocation contract — has since shipped**, and
it is the first Phase-H change to production code. Every native operation
allocates a fresh owning output, and native storage was value-initialized
on construction — a full write pass over a buffer most kernels then
overwrite completely. H1 added one C ABI symbol,
`tf_storage_create_uninitialized`, identical to the zero-initializing
constructor in size validation, allocation-failure handling, error state,
ownership, destruction, and live-storage accounting, and differing only in
the buffer's initial contents. The zero-initializing path is still the
default: there is no global allocator policy, no environment variable, no
heuristic, no memory pool, no scratch arena, and **no public
empty-tensor API**. Each call site opts in explicitly against a
per-kernel audit table, and two operations are deliberately **rejected** —
`sum`/`mean`, which accumulates into its output, and `narrow_backward`,
whose untouched zeros *are* the gradient.

Because ASan and UBSan do not detect uninitialized-value reads (that is
MemorySanitizer's job, and MSan needs an instrumented libc and CPython,
which this project does not have), completeness is proved separately, by
**deterministic poison tests**: an uninitialized allocation is filled with
a quiet NaN or a nontrivial finite pattern, and no poison may survive into
a result. The poison is applied **entirely by test infrastructure wrapped
around the allocator** — the real constructor allocates, the test fills
the returned storage through the ordinary fill primitive, and that same
storage goes to the real operation — so **the shipped library and the
installed Python backend contain no poison control at all**: no exported
hook, no thread-local flag, no environment variable, no global mode.
Negative controls prove the detector can actually fail. H1 is
bit-identical — every enabled operation and a complete training run are
compared element-wise against the zero-initializing allocator — and no
capability, dtype, device, registry value, or checkpoint version changed;
`tf_storage_create_uninitialized` is its only added export, taking the
library from 51 exported `tf_*` symbols to 52. The measured result is reported honestly rather than as a headline: isolated, the zero-fill is enormous and scales with the buffer (about 52x at 2 MB, 119x at 8 MB, 552x at 32 MB, and *negative* below roughly 16,000 elements, where it sits inside the noise). End to end it is much smaller and often inconclusive — clearly real for large memory-bound elementwise work (about 1.5-1.8x on an 8 MB output), small and variable for normalization and Adam, and with no measurable effect on Conv2d, the MLP step, or matmul, whose arithmetic dwarfs its allocation. Those inconclusive and negative rows are published as such.

**Milestone H2 — native matmul memory access — has since shipped**, the
first Phase-H milestone to change how a numerical kernel executes. It
swapped the production matmul's loop order from `i`-`j`-`k` to
`i`-`k`-`j` over four destination rows at a time, so the innermost loop
walks a *row* of the right operand and a row of the output sequentially
instead of walking a column. **Cache blocking, which the milestone title
anticipated, was measured against 22 blocked variants and rejected** — an
unblocked full-width row sweep was faster at every non-trivial size — so
H2 shipped the simpler superior design and recorded the negative blocking
result. The pre-H2 triple loop is **retained verbatim as the shipped
generic reference path**, still reachable through ordinary production
dispatch, and the choice between the two is made inside the kernel from
the stride metadata it already receives: a right operand whose column
stride is 1, with a non-empty inner dimension and at least 8 result
columns, takes the row sweep; a transposed right operand, a narrow
result, or an empty inner dimension takes the generic path — which is the
loop order that case already suits, so the fallback is a design choice
rather than a gap. Dispatch is metadata-driven, deterministic, total,
side-effect free, and independent of pointer values, alignment, timing,
environment variables, and CPU-feature probes; a failed precondition is
never an error. **H2 added no exported C ABI symbol** — the library still
exports exactly 52 `tf_*` symbols — and there is no kernel selector,
block-size setter, benchmark hook, dispatch tracer, or public dispatch
control of any kind; the two kernels and the predicate are
hidden-visibility C++ that the native test reaches only by compiling the
source in. The numerical agreement between the two paths is stated in **four
parts** rather than as a blanket claim, because a blanket claim would be
an overclaim. (1) **Accumulation order is preserved exactly** — same
starting zero, same products, same ascending `k`. (2) **Every non-NaN
result is bit-identical**, asserted as raw IEEE-754 bit patterns rather
than tolerances across shapes, layouts, signed zeros, infinities,
denormals, the largest finite magnitudes, both gradients, `NativeLinear`,
both optimizers, deterministic training, and exact checkpoint resume —
which covers every committed loss trajectory and every resume proof in
the project, since all of them run on finite data. (3) **NaN-class
equivalence holds**: NaNs appear in exactly the same positions on both
paths and are always quiet, and neither path produces a signaling NaN.
(4) **NaN payload bits are deliberately outside TensorForge's numerical
contract** and may differ between the paths. Ten source-level
formulations were measured while trying to close (4) — compound versus
explicit assignment, named locals, `__restrict`, disabled inner-loop
vectorization, and two stack-accumulator tile shapes — and all ten
`i`-`k`-`j` spellings behaved identically; the only structure that
reproduces the reference's payloads is the `i`-`j`-`k` order H2 exists to
replace, so payload parity is unavailable short of abandoning the
optimization. Measured: MSVC Release differs on 162 of 208 results in a
NaN-saturated matrix, MSVC Debug and Clang on none. H1's uninitialized-output
contract still holds on both paths, for a different reason on each — the
generic path never reads the destination, and the row sweep's `k == 0`
pass assigns every element of every row before anything accumulates into
it — proved by poison tests over both paths with both patterns plus a
negative control. The measured result is reported honestly: roughly
4.1-4.7x at 384 cubed, 4.2-4.5x at 128 cubed, about 4-6.8x on
`NativeLinear` forward, 1.7-2.5x on its backward (only one of its two
matmuls qualifies, by design), 2.0-2.4x on a 128x256 MLP Adam step, and
**no measurable effect below roughly 32 cubed or on a small MLP step**,
where a fixed ~10 microsecond per-call Python cost dominates and control
cases whose compiled code did not change at all vary by 0.50-1.44x. No
capability, dtype, device, registry value, checkpoint field, or
checkpoint version moved.

**Milestone H3 — native metadata and dispatch efficiency — has since
shipped**, and unlike H1 and H2 it is **Python-only**: no C++, no C ABI
symbol, no ctypes declaration, and no kernel changed, so the library
still exports exactly **52** `tf_*` symbols. H3 attacked the fixed
per-operation cost B3 measured at 18.6-22.6 microseconds, of which only
about 1.9 was the ctypes boundary and the rest was Python-side shape and
stride work. The measured cause was redundant *re-validation*: one
`shape_info` call ran `_as_int_tuple` **four** times over a tuple that
was fully validated after the first pass, and computed the row-major
strides **twice**, while `NativeTensorCore.zeros` validated the caller's
shape a second complete time by calling `numel(shape)` and then
constructing a view from the same raw shape. Instrumented call counts put
that at **815** `_as_int_tuple` calls per MLP training step and 604 per
`NativeAdam` step. H3 introduced **one normalization boundary** — the
private `_normalized_layout`, performing exactly the checks `shape_info`
always performed, in the same order and with the same messages, and
normalizing the shape once — with the derived quantities computed by
private `_checked` primitives that validate nothing *because there is
nothing left to validate*. Each public helper (`row_major_strides`,
`numel`, `reduce_shape`, `broadcast_shapes`) is now its own validation
followed by the matching primitive, so the two can never disagree.
`NativeTensorView` gained a private `_from_validated` constructor that
skips **only** that normalization; both constructors funnel through one
shared `_bind` that still performs the storage open check and the full
reachable-offset bounds check, and the element count and contiguity flag
are **derived inside** the private constructor rather than passed to it,
so no caller can supply an inconsistent pair — which is why H3 has a
separate private constructor rather than the misusable `validated=True`
flag. Views also memoize their `int64` shape/stride arrays for the
strided C ABI, **lazily** and **read-only**. That memoization cannot go
stale: a view's layout is assigned exactly once, in `_bind`, and every
layout-changing operation (`reshape`, `transpose`, `T`, `narrow`)
returns a *new* view, so no invalidation is ever required and none
exists. Nothing global was introduced — no shape cache, no stride
interning, no weak-reference machinery, no thread-local state — and
**no validation was removed**: every rejection still happens, with the
same exception type, the same message, and the same shape-then-strides-
then-offset ordering. Measured: `shape_info` 2.6-4.5x faster, view
construction 3.2x, `_as_int_tuple` calls per MLP step **815 -> 149** and
per CNN step **815 -> 150**; end to end, a one-element allocation 2.1x, a
`reshape` 3.1x, a view chain 2.4x, a small `add` 1.56x, `NativeAdam` on a
small MLP 1.42x, a **whole MLP training step 1.43x**, a **CNN training
step 1.29x**, and a **normalized training step 1.51x**, which cut the
Adam step's gap against the stable line from 39.8x to 31.9x. Reported
just as honestly: **large kernel-bound work shows no measurable change in
either direction** — 384-cubed, 512-cubed and 128-cubed matmul, 256-
squared elementwise, and 128-squared reduction all sit inside their own
run-to-run spread, so H2's large-matmul result is intact. The layout-
array cache is the weakest of the three changes and was kept on measured
merit, not principle: isolated, it saves 0.6-1.5 microseconds per
*strided* small operation and nothing at all on large ones or on a
contiguous training step, and even a deliberately cold-cache measurement
is no slower than pre-H3. One methodology finding is published rather
than buried: at the harness's default 11 repetitions a case appeared to
regress 35%, and at 201 repetitions the same case measured 1.19x
*faster* — so no default-repetition figure is quoted as H3 evidence.
Object footprint is unchanged for a cold view (byte-identical) and
+328 bytes for one that actually takes a strided path; in a full MLP step
only **5 of 134** views ever populate it, 1,560 bytes in total. All
instrumentation was test-local or benchmark-local monkeypatching and
subprocess A/B runs against a retained pre-H3 copy of the package — **no
production counter, environment-variable profiler, or installed tracing
mode exists**, and H3 added no public API of any kind: no cache control,
statistic, reset, profiling counter, or dispatch selector. No capability,
dtype, device, registry value, checkpoint field, or checkpoint version
moved.

**Milestone H5 — native copy and mutation-transfer efficiency — has
since shipped**, and it is the first Phase-H milestone since H2 to change
C++ — though **not the ABI**: the library still exports exactly **52**
`tf_*` symbols. H5 replaced the native line's **value-transfer
primitive**. `_native_copy` was `zeros(shape) + core` — two allocations,
a full zero-fill pass, and a full elementwise-addition pass — and is now
the E3.1 native identity gather, `NativeTensorCore.contiguous_copy()`:
one uninitialized allocation (H1) and one pass. The composition predates
that gather and was simply never migrated to it. A complete inventory
found **ten** call sites of the one helper — `NativeParameter.copy_value_`
staging, both `state_dict()` snapshot paths, both `load_state_dict()`
staging paths, both BatchNorm running-statistic commits, and the
reshape/transpose/unbroadcast gradient materializations — and every one
of them is a **pure value transfer**: an independent contiguous
materialization of some tensor's current value, wanting no arithmetic.
All ten were enabled. `_broadcast_back`'s `zeros(x_shape) + upstream` was
**rejected** because it is not a copy at all but a genuine broadcast
expansion, which `contiguous_copy` cannot express; `sum`/`mean` and
`narrow_backward` keep their zeroed destinations for H1's unchanged
reasons.

The semantic question H4 refused to decide in passing was decided here,
by measurement over a fixed 18-pattern IEEE-754 sweep. **Exactly three**
patterns behaved differently under the two spellings: the addition
normalized `-0.0` to `+0.0` and quieted both signs of signaling NaN,
while the gather preserves all three. Everything else — `±0`, `±inf`,
quiet NaNs of either sign and **any payload**, denormals, the smallest
normal, the largest finite magnitudes — was already identical, so no NaN
payload differed at all (with one NaN operand and one zero, x86-64's
`ADDSD` returns that operand's NaN). **H2's matmul NaN-payload carve-out
does not generalize to copies**: it exists because two NaN operands meet
in an accumulation, and a copy performs no arithmetic. The pre-H5
behavior was **accidental and inconsistent**, not contracted — three
other value-copy paths (`NativeParameter(source)` construction,
`detach()`, and the `to_numpy()`/`from_array` boundary) always used the
gather and always preserved `-0.0`, while `copy_value_` documented the
same thing and did not deliver it. H5 states the narrowest coherent rule:
**a value transfer reproduces its source's bits exactly; an operation —
`zeros + x` included — follows IEEE arithmetic.** No operation's
arithmetic changed anywhere, and the whole pre-H5 suite passes unchanged
apart from the guardrails that pinned the old composition by name.

Swapping the composition alone would have **regressed** the common case,
so H5's one C++ change is a second **traversal** inside the unchanged
`tf_core_contiguous_copy` export. `zeros.add(core)` on a contiguous
source takes a flat pointer loop, while the gather always walked the
generic odometer — the only unary export without the contiguous fast path
every other one has — and a naive swap measured **0.48x** at 16,384
elements. The export now picks its traversal from the layout metadata it
already receives, exactly as H2's matmul picks its kernel:
`tf::copy_prefers_contiguous` is hidden-visibility C++ in a new internal
header, total, pure, allocation-free, and a function of metadata alone —
never of a pointer value, an alignment, a clock, an environment variable,
or a CPU-feature probe — testing exact equality against the row-major
strides implied by the shape, which is the same definition
`NativeTensorView` uses, so the two layers agree by construction. A false
answer falls back to the retained odometer and is never an error. **No
numerical carve-out is needed, and that is the difference from H2**: both
traversals evaluate `dst[out] = src[pos]` over the same logical elements
in the same destination order and differ only in how `pos` is computed,
so they are bit-identical *by construction* — proved directly at the C++
level by a new dependency-free CTest, taking the suite from 13 to 14.
There is no copy-mode selector, overlap-mode flag, traversal tracer, or
public dispatch control of any kind.

Nothing became less safe, because nothing became in-place: every call
site still **stages** an independent materialization and only then adopts
it. The overlapping arrangements the runtime can construct —
`copy_value_(self)`, a source that is a view of the destination's own
storage, a square parameter's own transpose, sibling views, duplicate
parameters across optimizers — are each tested and each correct, and no
`memcpy` is used anywhere. Parameter identity, storage replacement,
gradient retention by identity and value, the one version increment per
commit, the F1 state transaction, checkpoint atomicity, and exact resume
are all exactly what they were; gradient *accumulation* still adds rather
than assigns. H1's full-write contract is proved on both traversals by
poison injected purely by test infrastructure around the allocator, with
a negative control showing the detector can fail.

Measured by alternating pre/post **subprocess** rounds against a retained
pre-H5 composition, with a control band of **0.96x-1.05x**, and — for
the C++ half — by building a **pre-H5 library** and driving both through
identical `ctypes` calls on identical data, outputs proved bit-identical
before either was timed. The traversal alone: **2.5x-5.5x** on contiguous
sources from 16 K elements up (5.53x at 512 squared, 5.53x on 4-D NCHW,
5.46x on an offset view), 1.29-1.62x on small ones, and **0.94x-1.02x on
transposed and last-axis-narrowed sources**, which take the *unchanged*
odometer and are the design's own control. End to end: `copy_value_`
**2.14x** at (512, 512) and 1.26x at (128, 128), optimizer `state_dict()`
2.40x and `load_state_dict()` 1.69x, module `load_state_dict()` 1.37x,
`NativeSGD.step()` 1.15-1.31x. Reported just as honestly:
**`NativeAdam.step()` is neutral** (0.98x-1.06x — the commit copy is one
of about seventeen buffers and the arithmetic dominates), **every
training step is neutral** (0.95x-1.07x), the **BatchNorm running update
is neutral** (0.98x), and **copies below ~16 K elements are neutral**
(0.93x-1.01x), because a `contiguous_copy` call converts two `int64`
layout arrays at the ctypes boundary at **~1.1 us each** — a cost
measured, attributed, and left to a later dispatch milestone rather than
paid for by weakening H3's validation. Two methodology findings are
published rather than buried: at 7 alternating rounds the small copies
read 0.78x-0.94x and looked like a regression, while at 21 rounds the
same cases read 0.93x-1.01x (the same lesson H3 recorded); and the
largest single ratio, **7.9x-10.5x at 512-640 KB**, is a **512 KB
allocator cliff on this machine**, not a loop-speed result — the pre-H5
composition makes two large allocations and zero-fills one, so it crosses
that threshold at half the size and pays it twice. The durable statements
are ~2.1x at 1-2 MB and neutrality below 384 KB.

Memory moved with time, never against it: **no measured peak rose**, and
the pure-transfer paths halved. `copy_value_` at (512, 512) went 2
allocations to **1** and 4,194,304 to **2,097,152** peak bytes; module
`state_dict()` and `load_state_dict()` 4 to **2** allocations with peak
bytes halved; optimizer `state_dict()` 16 to **8**; `NativeSGD.step()`
5 to **4** with peak 393,216 to **262,152**; and `NativeAdam.step()` went
**17 to 16** allocations per parameter (H4 took it 27 to 17), removing a
whole-parameter zero-fill pass from every committed update. The harness
gained two cases, 26 to **28**: `row_major_materialization`, the
flat-traversal twin of the existing transposed-source case, so the two
traversals are separated rather than averaged; and
`parameter_value_commit`, `native_only` with **no ratio**, because the
stable line mutates a `Parameter` by rebinding `.data`, which is a
different operation. The ladder was **reordered** here — reduction
execution, drafted as H5, moved to H6 — and no public API, capability,
dtype, device, registry value, checkpoint field, or checkpoint version
moved.

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
0.99x, MLP large 1.03x, normalized 1.03x, CNN 1.01x, Dropout control 1.02x,
all inside the control band — so **H6 does not make training faster**, and
no reading should be quoted as if it did; a reduction is a small share of a
step whose cost is the optimizer and the large matmuls. **Normalization is
mostly neutral** too: BatchNorm1d training forward 1.04x, eval 0.98x,
backward 1.02x, BatchNorm2d training forward 1.06x, eval 1.00x, LayerNorm
backward 1.01x, with only LayerNorm forward and BatchNorm2d backward
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
parameters, BatchNorm buffers, and Adam moments produced a **bit-identical**
allocation and live-count profile before and after H6, which also confirms
that profile's oscillation is CPython's collector rather than a leak either
version introduced.

The harness gained three cases, 28 to **31**, following H5's
separate-rather-than-average precedent: `reduction_last_axis` (the suffix
form LayerNorm's mean and both softmax backwards actually reduce over),
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
and this is the result — the **native Dropout step 1.32x, the normalized
step 1.31x, `NativeAdam` at (32, 32) 1.31x, the CNN step 1.30x, the MLP step
1.28x**, `NativeLayerNorm` forward 1.23x, `NativeBatchNorm1d` eval
1.23x, `NativeAdam` at (128, 128) 1.14x, `NativeSGD` 1.13x, the large MLP step 1.08x. **H7 is the
first Phase-H milestone to move every training step** — H4 moved them
1.09x-1.23x and H5 and H6 were neutral on all of them — because the cost
is paid per *call* and a step makes hundreds of them.

Reported just as honestly: **large kernel-bound work is neutral**, exactly
as the attribution predicts. 256-cubed matmul **0.99x** and 8-cubed matmul
1.00x are controls that take no array at all, so **H2's result is
structurally untouched**; contiguous 16x16 `add` 1.05x is the third
array-free control; and 512x512 `copy` 1.02x, 256x256 `to_numpy` 1.04x,
512x512 full `sum` 1.06x, 256x256 broadcast `multiply` 1.08x, and the large
MLP step 1.08x are all at or inside the band. **H7 did not make matmul faster**,
and no reading should say otherwise.

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

The ladder ran **H0–H10 and ended there**, revised on evidence three
times: reordered at H5, a milestone **dropped** at H7 when the
measurement did not support its premise, and a slot **reassigned** at H9.
Allocation pooling, SIMD, threading/OpenMP, and BLAS were **all finally
rejected at H10, with measurements** — elementwise, matmul, and reduction
are already auto-vectorized, a CNN step's native calls have a 1.20 µs
median, and a BLAS matmul is not bit-identical, which would break every
exact-resume proof. The criteria that would reopen each are recorded
rather than an answer invented. Every number in that document is a local
characterization of one machine, reported with its spread, and asserted
by no test — there is no CI timing threshold anywhere in this
repository.

**Milestone H7 — native Python/C ABI boundary efficiency — shipped next**,
and it is **Python-only**: no C++, no exported symbol, no kernel, no
traversal, no arithmetic, still exactly **52** exported `tf_*` symbols.
**The ladder was revised there, and the revision is recorded rather than
retrofitted**: H0's H7 slot was *composed-module cost*, explicitly
conditional on a re-measurement after H1, H3, and H6. H6 measured the
normalization modules almost entirely neutral, so the condition was **not
met** and that milestone was **dropped on evidence**, its proposal
preserved verbatim in the design rather than deleted. The slot was
refilled from the same measurements: H3, H5, and H6 had each ended by
deferring one identically named cost to "a later dispatch milestone". Of
57 array argument positions, the **32 layout-metadata positions across 13
exports** moved from the checked `numpy.ctypeslib.ndpointer` binding —
measured at **~2.1 us per array per call** — to
`ctypes.POINTER(ctypes.c_int64)`, fed by exactly two private producers,
while the **25 data positions stay checked**. ctypes still type-checks
every call, and the **length/rank invariant `ndpointer` never checked**
became checkable for the first time; `POINTER.from_address` was measured
faster and **rejected** for owning nothing. Measured against a retained
pre-H7 `cpp.py` on the same Release DLL, bit-identical before timing: tiny
operations 1.3x–1.9x and **every training step 1.08x–1.32x — the first
Phase-H milestone to move them all** — with large kernel-bound work
neutral and three array-free control cases confirming it. Harness 31 to
**34** cases; no ABI change; no capability move.

**Milestone H8 — native elementwise traversal and composed allocation
efficiency — has since shipped**, the fourth Phase-H milestone to change
C++ and, like H2, H5, and H6, **not the ABI**: still exactly **52**
exported `tf_*` symbols.

H8 entered with **two** candidate tracks and an explicit instruction not
to force both into production. **Track A — elementwise traversal — was
confirmed and is the milestone. Track B — composed normalization
allocation — was confirmed only as a memory result and is reported as
timing-neutral.**

The cost was decomposed rather than assumed. The odometer costs
**1.60x–6.42x** the flat loop on identical contiguous data; **all
broadcasting is on the odometer**, because there is no broadcast fast path
at all; and a guarded standalone split the odometer's cost four ways at
`(256, 256)` contiguous `add` — the shipped odometer-plus-function-pointer
at **123.5 us**, templating alone **81.3 us (1.52x)**, collapsing alone
**63.6 us (1.94x)**, and **both together 11.5 us (10.7x)**. Neither change
is worth much alone and together they are worth an order of magnitude,
because only their combination lets the compiler emit a vector loop. The
same run showed the **existing flat contiguous kernel was itself hobbled**
by its indirect call, at 21.0 us against 11.7 us.

H8 therefore reused the dispatch shape H2, H5, and H6 each proved — one
hidden metadata builder, inside the existing export, no new symbol, the
pre-milestone traversal retained. A **plan** is an *operation-local
normalized descriptor*: built on the stack, used by one call, dropped,
with nothing cached, interned, or shared between calls. It applies exactly
two sequence-preserving transformations — **unit axes are dropped**, and
**adjacent axes are merged** when
`stride[outer] == stride[inner] * extent(inner)` holds for *every* operand
at once. Axes are never reordered, split, or transposed, the bound is a
fixed **4 axes**, and **this is not a layout compiler**. The builders are
total, pure, allocation-free, and a function of layout metadata alone, and
a rejection is a **fallback, never an error**; `core_unary` and
`core_binary` are retained **verbatim** as the shipped generic reference
paths. **`exp` and `log` are deliberately excluded** — IEEE-754 does not
specify them, so a vectorizing toolchain would be free to return different
bits, and the templated traversal measured **1.05x** on both, inside the
noise.

**The numerical contract is H8's own, in four parts.** Every result in
which at most one operand is a NaN is **bit-identical** to the pre-H8
kernel's — zero differing results across 15 op × layout combinations; NaN
positions are identical and every NaN the arithmetic produces is quiet;
**subtraction is bit-identical everywhere**, two-NaN pairs included,
because it is not commutative; and for **addition and multiplication with
two NaN operands** the surviving payload is outside the contract. **That
last part predates H8, and H8 narrows it**: the pre-H8 library's own flat
kernel and its own odometer already disagreed on **30 of 196** such pairs,
while post-H8 only a transposed operand differs, on **5 of 196**. It is a
*different* qualification from H2's and H6's, which concerned NaNs meeting
inside an accumulation.

**Track B** shipped the one composed-allocation change the evidence
supported: BatchNorm builds its `(1 - momentum, momentum)` pair **once per
forward** instead of once per buffer, and each blend releases its
temporaries at last use. A `NativeBatchNorm1d` training forward goes
**25 → 23** allocations with **peak live storages 25 → 17**, and
`NativeBatchNorm2d` **30 → 28** and **30 → 22**. **Its timing effect is
neutral**, and it is reported as a memory result rather than a speed one.
Four alternatives were rejected with reasons, including adopting the blend
result into the running-state transaction — which would have moved
numerical work inside the staging phase and changed a failure ordering F5
and F8 prove by test.

Measured against a pre-H8 library on identical `ctypes` calls, every case
bit-identical before timing, control band **0.97x–1.08x**: row-broadcast
`multiply` **10.58x**, strided same-shape `add` **9.67x**, NCHW-statistic
`multiply` **7.15x**, column broadcast **6.70x**, scalar broadcast
**6.31x**, transposed `add` 2.63x, strided `relu` 2.51x, `sqrt` 2.03x,
`relu_backward` 1.86x, contiguous `add`/`multiply` 1.76x. End to end,
over 11 alternating subprocess rounds with all 31 cases bit-identical
first: **`NativeAdam.step()` 2.01x**, **`NativeBatchNorm1d` eval forward
1.40x**, **`NativeBatchNorm2d` eval forward 1.36x**, **`NativeBatchNorm2d`
training forward 1.33x**, **`NativeLayerNorm` forward 1.30x**, **the large
MLP training step 1.19x**, and **the normalized training step 1.08x**.
**This is the milestone that finally moved the normalization modules** —
which H6 measured as almost entirely neutral, and which is precisely why
H0's composed-module H7 was dropped and this one entered.

Reported just as honestly: **small normalization shapes are neutral**
(`NativeBatchNorm1d` training at `(32, 16)` **0.98x**), **the CNN step is
neutral (0.99x)**, the `exp`/`log` controls read 0.97x–1.07x exactly as
the deliberate exclusion predicts, and one control is **published rather
than buried** — 256-cubed `matmul` reads **0.93x–0.96x**, isolated by a
25-round run to that one size while 64, 128, and 384 cubed are neutral and
an identical-code twin reads 0.969x on the same case. `matmul.cpp` is
byte-identical source; `elementwise.cpp`'s object code grew 127 KB to
188 KB, moving every function's placement in the image, which is the same
whole-translation-unit code-layout effect H6 documented and the
machine-specific tuning the design rejects chasing.

**Memory: Track A moved none, and the odometer's heap-allocated counter is
now removed on every plannable layout** — a strided elementwise call makes
**one** allocation where it previously made two. The harness gained four
cases, 34 to **38**, and the native CTests 15 to **16**. **No exported C
ABI symbol, no new translation unit, and no public control of any kind**,
and no SIMD, threading, OpenMP, BLAS, memory pool, scratch workspace,
general fusion, or fast-math. No public API, capability, dtype, device,
registry value, checkpoint field, or checkpoint version moved.

**Milestone H9 — native Conv2d execution efficiency — has since shipped**,
the fifth Phase-H milestone to change C++ and, like H2/H5/H6/H8, **not the
ABI**: still exactly **52** exported `tf_*` symbols. **It was not in H0's
ladder** — that slot held SIMD/threading/BLAS, *conditional and presumed
rejected*; none qualified, so the acceleration decision moved to H10's
decision gate and the slot went to the last large compute family still
running its unmodified Phase-D implementation. Measurement decided it:
the Python wrapper is a fixed ≈ 8–12 µs — **66 %** of a toy `(4,1,6,6)`
convolution but **≈ 0 %** at `(16,8,32,32)` — so the compiled traversal is
essentially **100 %** of any real convolution. Each of the three exports
now ships two paths behind one unchanged symbol: the **Phase-D direct loop
retained verbatim** as the generic reference, and one optimized traversal
— a forward row sweep, and gathers for both gradients — chosen by a hidden
predicate from the integer geometry alone
(`min(input_width, output_width) >= 4`, plus **unit stride** for the input
gradient, whose downward kernel walk inverts one-for-one only there). **A
false answer is a fallback, never an error**, and **per-destination
accumulation order is preserved exactly in all three directions**, each
separately proved. Measured against a pre-H9 library on identical `ctypes`
calls, bit-identical before timing: kernels **1.97×–8.37×**, `NativeConv2d`
forward+backward **3.09×–3.13×**, and **a CNN training step 1.86×** —
**the first Phase-H milestone to move one**. Honestly reported: small
convolutions are neutral (1.06×), the strided input gradient falls back by
design (1.04×), and no control regressed (band **0.97×–1.07×**). Memory is
**byte-identical**; there is no scratch, workspace, im2col, or padded copy
anywhere. The harness gained three cases, 38 to **41**, and the native
CTests 16 to **17**. No public API, capability, dtype, device, registry
value, checkpoint field, or checkpoint version moved, and no convolution
option was added.

float32 CPU support is **Phase I's** subject and is designed but not
implemented; more activations/math, data loaders, native integer tensors,
further dtypes beyond float32/float64, and CUDA/GPU experiments remain
future work beyond Phase I. See
[docs/roadmap.md](docs/roadmap.md) and
[docs/release_history.md](docs/release_history.md).

TensorForge is a from-scratch look at how a deep learning framework
works under the hood — not a PyTorch replacement. Start reading at
`src/tensorforge/tensor.py`, then cross the ctypes boundary at
`src/tensorforge/backends/cpp.py`.
