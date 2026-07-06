# TensorForge — project instructions

## What this is

TensorForge is a mini deep learning framework built from scratch in
Python + NumPy — an educational take on PyTorch, inspired by Daedalus ML
but not a copy. It is developed milestone by milestone (v0.1, v0.2, …),
each one small, tested, and readable.

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
  Sequential, losses (`mse_loss`, `cross_entropy`), metrics (`accuracy`).
- `src/tensorforge/optim/` — SGD, Adam. Plain classes: `step()` skips
  `None` grads, `zero_grad()` sets grads to `None`.
- `src/tensorforge/data.py` — `batches` mini-batch iterator.
- `examples/` — runnable scripts, each with `train(...)` returning
  stats and a `main()` that prints, guarded by `__main__`.
- `tests/` — pytest suite; every feature has tests.

## Commands

- Run tests: `uv run pytest`
- Run examples:
  - `uv run python examples/train_linear_regression.py`
  - `uv run python examples/train_xor.py`
  - `uv run python examples/train_multiclass.py`

## Style rules

- Keep code simple, readable, and educational — clarity beats cleverness.
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

- `pytest` cache is redirected to `.cache/pytest` (see pyproject) —
  the default `.pytest_cache/` at the repo root has broken permissions
  on this machine; don't try to delete it.
- Two example-test import styles coexist: `tests/test_examples.py`
  inserts `examples/` into `sys.path`; newer tests import
  `examples.<name>` as a namespace package from the repo root.
- Root package exports: `Tensor`, `Parameter`, `cross_entropy`,
  `accuracy`, `SGD`, `Adam`, `batches` (locked in by
  `tests/test_public_api.py`).
