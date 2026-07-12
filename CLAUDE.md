# TensorForge — project instructions

## What this is

TensorForge is a from-scratch deep learning framework built in
Python + NumPy — a serious ML systems project covering PyTorch-style
framework internals, inspired by Daedalus ML but not a copy. It was
developed milestone by milestone (v0.1 … v3.0), each one small,
tested, and readable. The Python framework line is complete as of
v3.0; work continues on advanced branches (the experimental C++
native line has completed Phase A — CPU runtime, Phase B — native
autograd, and Phase C — the native training stack, through Advanced
C++ v3.15; the native CNN stack and CUDA experiments are future work).
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
  examples, roadmap, release history. When a milestone changes the
  public API or the examples, update the matching docs file (and
  README links) in the same milestone.
- `.github/workflows/tests.yml` — minimal CI: install uv, build the
  experimental C++ backend, hard-failing kernel smoke check, then
  pytest.
- `cpp/` + `src/tensorforge/backends/` — the experimental C++ backend
  (post-v3.0 line). Plain C-ABI kernels loaded via ctypes; built with
  `uv run python cpp/build.py` (`uv sync --group cpp` first if no
  compiler). Never imported by the main framework; importing the
  wrapper is always safe (lazy load) — check `cpp.is_available()` /
  `cpp.backend_info()`; kernels raise ImportError at call time when
  unbuilt, and the backend tests skip. `benchmarks/cpp_backend.py`
  compares kernels against NumPy honestly (no performance assertions
  anywhere). `scripts/smoke_cpp_backend.py` is the hard-failing smoke
  check CI runs after building.

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
