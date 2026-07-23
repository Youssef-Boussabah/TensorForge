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
Stateful Buffers — is the current phase and is *designed only*:**
milestone F0 is complete (the architecture contract in
`docs/native_normalization_design.md` plus repository reconciliation —
**no numerical behavior**), and milestones F1–F9 (an atomic
native-buffer state transaction, `NativeLayerNorm`, `NativeBatchNorm1d`,
`NativeBatchNorm2d`, state/checkpoint and graph-safety hardening, a
deterministic normalized training run with exact resume, a benchmark
characterization, cross-cutting integration, and closure) have **not
started** — so `batchnorm` and `layernorm` remain in the backend
registry's `UNSUPPORTED` tuple and no normalization module, operation,
kernel, or C ABI symbol exists.
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
  `native_normalization_design.md` for Phase F — designed only). When a milestone changes the
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
