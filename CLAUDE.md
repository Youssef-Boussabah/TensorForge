# TensorForge — project instructions

## What this is

TensorForge is a from-scratch deep learning framework built in
Python + NumPy — a serious ML systems project covering PyTorch-style
framework internals, inspired by Daedalus ML but not a copy. It was
developed milestone by milestone (v0.1 … v3.0), each one small,
tested, and readable. The Python framework line is complete as of
v3.0; work continues on advanced branches (the experimental C++
native line has completed Phase A — CPU runtime, Phase B — native
autograd, Phase C — the native training stack, and Phase D — the
native CNN stack, through Advanced C++ v3.16; Phase E — native
classification and stable math — is *complete* (E0–E10), its contract
locked
in `docs/native_classification_design.md` with milestones E1–E4 (the
differentiable native `exp`, `log`, and the fused stable `softmax` and
`log_softmax`), E5 (the fused `cross_entropy` **Core** contract —
`NativeTensorCore.cross_entropy_forward`/`cross_entropy_backward`), and
E6 (the differentiable `NativeTensor.cross_entropy(targets,
reduction="mean")`, one autograd node with graph-owned saved
probabilities and no logits reread), and E7 (the stateless
`NativeCrossEntropyLoss` module delegating to that operation, and the
reporting-only `native_accuracy` — explicit `to_numpy()` + NumPy
argmax, no graph, in the new `NATIVE_METRICS` inventory), and E8 (the
deterministic classification training and exact checkpoint-resume proof
— `examples/native_classification_training.py`: a three-class native
CNN classifier over raw logits, 40 `NativeAdam(lr=0.05)` steps, loss
1.159638 → 0.000101, accuracy 0.3333 → 1.0000, interrupted at step 15
and resumed into a fresh model/optimizer pair that matches exactly;
example, tests, and docs only — no new capability), and E9 (the honest
characterization benchmark
`benchmarks/benchmark_native_classification.py`: seven cases, each
correctness-gated before timing, each labelled with the reference it
used, medians with spread after warm-up, `--smoke`/`--json` modes, and
**no speed assertion or timing threshold anywhere**), and E10 (phase
closure: `tests/test_native_phase_e.py` cross-cutting integration,
Release and Debug builds with 10/10 CTests each, Clang ASan/UBSan and
LeakSanitizer validation, and documentation reconciliation — no new
numerical capability) all shipped. **Phase F — Native Normalization and
Stateful Buffers — is the current phase and is *in progress*:**
milestone F0 is complete (the architecture contract in
`docs/native_normalization_design.md` plus repository reconciliation —
**no numerical behavior**), F1 is complete (the private atomic
native-buffer state transaction in
`src/tensorforge/experimental/_native_state.py`, `load_state_dict`
refactored onto it, and `persistent_buffers` added to `STATE_SUPPORT` —
state management and capability reporting only), and F2 is complete
(`NativeLayerNorm` — the first native normalization module: stateless
(no buffers, identical in train and eval), differentiable through the
mean and the population variance, **composed entirely from existing
native operations** — `mean`/`subtract`/`multiply`/`add`/`sqrt`/
`reciprocal`, `sqrt(var + eps)`, no Bessel correction — adding no C++
code, kernel, C ABI symbol, ctypes declaration, `NativeTensorCore`
method, custom backward, functional helper, or `NativeTensor.layer_norm`
operation; `weight`/`bias` `NativeParameter`s only when
`elementwise_affine=True`; `"NativeLayerNorm"` in `NATIVE_MODULES` and
the exports, and `"layernorm"` removed from `UNSUPPORTED`), and F3 is
complete (`NativeBatchNorm1d` in
`src/tensorforge/experimental/native_batchnorm.py` — the **first
stateful native numerical module**: strictly `(N, C)` batch
normalization, again composed entirely from existing native operations,
adding no C++ code, kernel, C ABI symbol, ctypes declaration,
`NativeTensorCore` method, custom backward, functional helper, or
`NativeTensor.batch_norm` operation. Training normalizes with this
batch's **differentiable** population statistics (`sqrt(var + eps)`, no
Bessel correction, gradients through the mean *and* the variance) and
advances the persistent native `running_mean`/`running_var` buffers by
`(1 - momentum)*running + momentum*batch` from the *same* batch
statistics — computed **graph-free** via `detach()` and committed as one
**atomic two-buffer transaction** through the F1 primitive, preserving
both Python identities, closing each replaced core exactly once, and
moving **no** parameter version. Evaluation reads **independent owning
graph-free `(1, C)` snapshots** of those buffers, so no registered
buffer is ever a rereadable graph operand and a later training step, a
buffer-only `load_state_dict()`, or a buffer-only
`load_native_checkpoint()` cannot change an earlier eval graph's
gradient (a *full* checkpoint load also replaces `gamma`/`beta`, so the
unchanged v3.7 parameter-version guard correctly stales that graph — a
parameter contract, never a buffer effect); the snapshots ride the existing `graph_resources`
contract and release exactly once with the graph history. `gamma`/`beta`
always exist (no `affine=False`, `track_running_stats`, or
`num_batches_tracked`); state order is `gamma`, `beta`, `running_mean`,
`running_var`; the checkpoint format stays version 1;
`"NativeBatchNorm1d"` is in `NATIVE_MODULES` and the exports, while
`"batchnorm"` stayed in `UNSUPPORTED`), and F4 is complete
(`NativeBatchNorm2d`, the second public class in the same file — NCHW
`(N, C, H, W)` batch normalization reducing over **N, H, and W**, so
each channel gets one population mean and variance over `N * H * W`
values. It is built on the **same** private `_NativeBatchNorm` and
declares *only* `_INPUT_NDIM = 4`, `_REDUCTION_AXES = (0, 2, 3)`,
`_TRAILING_DIMS = 2`, `_LAYOUT`, and `_CHANNELS_LAST = (0, 2, 3, 1)` —
every method is inherited by function identity. The one shared piece F4
added is the channelwise affine: rank-1 `gamma`/`beta` broadcast from
the *trailing* axis, so the **activation** is transposed to
channels-last for the affine step and back again (then materialized
contiguous) rather than reshaping the parameters — which keeps `gamma` a
direct versioned `multiply` operand and preserves the existing
stale-parameter guard exactly. Statistics are `(1, C, 1, 1)`, running
buffers stay `(C,)`, the checkpoint format stays version 1, and
`"NativeBatchNorm2d"` is in `NATIVE_MODULES` and the exports; with both
shapes live `"batchnorm"` has **left** `UNSUPPORTED`, which now reads
exactly `("dropout", "float32", "cuda", "amp")`). **That completes the
numerical normalization *module* surface — not Phase F.** **F5 is
complete** (the exhaustive state, checkpoint, ownership, and graph-safety
hardening — a focused `tests/test_native_normalization_state.py` plus
narrow additions to the generic buffer and checkpoint suites, proving
§7–§10 of the design by executable test rather than by prose: canonical
dotted buffer keys, independent state snapshots, strict/non-strict loads,
exact never-casting metadata validation, mixed parameter/buffer
transaction atomicity, buffer identity across state and checkpoint loads,
exact eval-output reproduction, the buffer-only-versus-full stale-graph
distinction, the save/corrupt-load failure boundaries, eval-graph snapshot
safety under `retain_graph` and a failed retryable backward, and the
live-storage baselines; **tests and documentation only — no numerical
behavior and no new public capability**, with the exports, every
capability registry, and the version-1 checkpoint format all exactly what
F4 left, and no production behavior changed). **F6 is complete** (the
deterministic normalized training and exact checkpoint-resume proof —
`examples/native_normalization_training.py`: a `NativeNormalizedRegressor`
(`Linear → BatchNorm1d → ReLU → LayerNorm → Linear`, both normalization
families in every forward, BatchNorm the only stateful module) trained for
24 deterministic `NativeAdam` steps with `NativeMSELoss` (98.9% loss
reduction), with two uninterrupted runs proved bit-identical and an
interrupted run resumed into a **fresh** model/optimizer pair that
reproduces the remaining loss suffix, every parameter, the NativeAdam
state, both BatchNorm `running_mean`/`running_var`, the final
training-step prediction, and the final **evaluation-mode** output exactly
— checkpoint format version 1 unchanged, training flags runtime-only;
**one example and its integration test, adding no capability, operation,
kernel, schema field, benchmark, or export, and changing no production
behavior**). Milestones F7–F9 (a benchmark characterization, cross-cutting
integration, and closure) have **not started** — so there is no
normalization benchmark, no Phase-F integration file, and no phase
closure, and no normalization operation, kernel, C ABI symbol, or custom
backward exists at all. F7 is next.
Dropout/RNG, data loaders, native integer tensors, further
dtypes/devices, CPU optimization, and CUDA experiments are
future work beyond Phase F.
Position the project as serious and systems-focused — never
"educational", "toy", or "mini" — while staying honest: not
production-ready, not a PyTorch replacement.

## Tech stack

- Python ≥ 3.13, NumPy, pytest — nothing else.
- Managed with `uv` (`uv run …` for everything).
- Never introduce PyTorch, TensorFlow, JAX, sklearn, pandas, or
  matplotlib. NumPy is the only numeric dependency.

## Layout

- `src/tensorforge/tensor.py` — Tensor + reverse-mode autograd. Ops are
  either primitives (eager NumPy forward + `_backward` closure holding
  the local derivative) or derived (compositions that get gradients for
  free). Gradients accumulate via `_accumulate_grad`, which also
  un-broadcasts.
- `src/tensorforge/nn/` — Parameter, Module, Linear, activations,
  Dropout, BatchNorm1d, LayerNorm, Conv2d, MaxPool2d, Flatten,
  Sequential, losses (`mse_loss`, `cross_entropy`,
  `binary_cross_entropy`), metrics (`accuracy`, `binary_accuracy`,
  `evaluate_classifier`, `evaluate_binary_classifier` — the evaluators
  measure with the model temporarily in eval mode and restore it).
  Modules have train/eval mode: `model.train()` / `model.eval()`
  recurse through children; Dropout and BatchNorm1d change behavior.
  Modules can declare non-trainable buffers via `self._buffers =
  ("attr", ...)` (e.g. BatchNorm running stats); `state_dict()` /
  `load_state_dict()` cover parameters *and* buffers.
- `src/tensorforge/optim/` — SGD, Adam. Plain classes: `step()` skips
  `None` grads and frozen params, `zero_grad()` sets grads to `None`.
  Also `StepLR` (multiplies `optimizer.lr` by gamma every step_size
  epochs) and `clip_grad_norm` / `clip_grad_value` (clip gradients in
  place before `optimizer.step()`).
- `src/tensorforge/data.py` — `batches` mini-batch iterator.
- `examples/` — runnable scripts, each with `train(...)` returning
  stats and a `main()` that prints, guarded by `__main__`.
- `tests/` — pytest suite; every feature has tests.
- `docs/` — project summary, architecture, autograd, training,
  examples, roadmap, release history, and the native-line design
  contracts (`native_cnn_design.md` for Phase D,
  `native_classification_design.md` for Phase E,
  `native_normalization_design.md` for Phase F — F0–F6 shipped). When a milestone changes the
  public API or the examples, update the matching docs file (and
  README links) in the same milestone.
- `.github/workflows/tests.yml` — minimal CI: install uv, build the
  experimental C++ backend, hard-failing kernel smoke check, then
  pytest.
- `cpp/` + `src/tensorforge/backends/` — the experimental C++ backend
  (post-v3.0 line; `cpp/src/classification.cpp` holds the Phase-E
  classification kernels). Plain C-ABI kernels loaded via ctypes; built with
  `uv run python cpp/build.py` (`uv sync --group cpp` first if no
  compiler). Never imported by the main framework; importing the
  wrapper is always safe (lazy load) — check `cpp.is_available()` /
  `cpp.backend_info()`; kernels raise ImportError at call time when
  unbuilt, and the backend tests skip. `benchmarks/cpp_backend.py`
  compares kernels against NumPy honestly (no performance assertions
  anywhere), and `benchmarks/benchmark_native_cnn.py` characterizes the
  Phase-D CNN stack the same way. `scripts/smoke_cpp_backend.py` is the
  hard-failing smoke check CI runs after building. Dependency-free C++
  CTests live in `cpp/tests/` and build only with `-DTF_BUILD_TESTS=ON`;
  sanitizer validation uses Clang on Linux
  (`-DTF_SANITIZE=address,undefined`), which MSVC does not support.

## Commands

- Run tests: `uv run pytest`
- Run examples:
  - `uv run python examples/train_linear_regression.py`
  - `uv run python examples/train_xor.py`
  - `uv run python examples/train_multiclass.py`
  - `uv run python examples/train_binary_classification.py`
  - `uv run python examples/train_mlp_with_dropout.py`
  - `uv run python examples/train_tiny_cnn.py`

## Style rules

- Keep code simple and readable — clarity beats cleverness.
- Match the existing style: NumPy-only internals, small modules, one
  concept per file.
- Comments explain math/autograd reasoning, not obvious Python.
- Losses and metrics stay simple: losses are Tensor expressions or
  fused ops with custom backward; metrics are plain NumPy returning
  Python floats, outside autograd.
- Examples use fixed seeds so output is reproducible, and follow the
  `train()` + `main()` pattern so tests can import `train`.
- Tests use `np.allclose` with sensible tolerances (e.g. `atol=1e-6`);
  training tests assert learning without fragile exact-loss values.

## Workflow rules for Claude Code

- Inspect existing code before editing; find where a concept lives and
  follow its pattern.
- Keep changes scoped to the requested milestone. No unrelated
  features, no drive-by refactors, no framework rewrites.
- If a requested feature already exists, verify it against the spec and
  add tests/documentation instead of reimplementing it.
- Preserve all previous tests. Never loosen a test just to pass.
- Run `uv run pytest` (and any requested manual checks) before
  reporting success; report the actual observed output.
- Do not use git: no commits, no pushes, no `git` commands. The user
  handles version control.
- Final responses report: files changed, what was implemented, tests
  added, the exact pytest result, manual check outputs, and any notes
  or limitations.

## Current notes

- This machine has a permissions quirk: directories created by one
  process often cannot be deleted by a later one. Consequences already
  handled: pytest's cache is redirected to `.cache/pytest` (pyproject)
  and `conftest.py` gives each test session a fresh unique basetemp so
  tmp_path never needs to wipe an old directory. Don't try to delete
  `.pytest_cache/`, `.cache/pytest-tmp/`, or `%TEMP%/pytest-of-*`.
- Two example-test import styles coexist: `tests/test_examples.py`
  inserts `examples/` into `sys.path`; newer tests import
  `examples.<name>` as a namespace package from the repo root.
- Root package exports: `Tensor`, `Parameter`, `Dropout`,
  `BatchNorm1d`, `LayerNorm`, `Conv2d`, `MaxPool2d`, `Flatten`,
  `cross_entropy`,
  `binary_cross_entropy`, `accuracy`, `binary_accuracy`,
  `evaluate_classifier`, `evaluate_binary_classifier`, `SGD`, `Adam`,
  `StepLR`, `clip_grad_norm`, `clip_grad_value`, `batches`,
  `train_test_split`, `save_parameters`, `load_parameters`,
  `save_checkpoint`, `load_checkpoint`, `count_parameters`,
  `model_summary` (locked in by `tests/test_public_api.py`).
  Checkpoints = weights + optimizer state + optional scheduler state
  + optional RNG state (`rng_state=True` / `restore_rng_state=True`,
  covers unseeded Dropout) + JSON metadata; parameters = weights only.
