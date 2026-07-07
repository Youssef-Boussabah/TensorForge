# TensorForge — project summary

A two-minute overview for anyone landing on this repository.

## What TensorForge is

TensorForge is an educational deep learning framework built from
scratch in Python + NumPy. It reimplements the core machinery of a
framework like PyTorch — automatic differentiation, neural network
modules, optimizers, checkpointing, basic CNN support — in small,
readable code. NumPy is the only dependency. It was built milestone by
milestone (v0.1 through v3.0), each one tested and documented.

## What TensorForge implements

- A `Tensor` with reverse-mode autograd: arithmetic, matmul, reshape,
  reductions, and the common nonlinearities, with broadcasting-aware
  gradients verified against finite differences.
- A module system: `Parameter`, `Module`, `Linear`, `Sequential`,
  activations, `Dropout`, `BatchNorm1d`, `LayerNorm`, `Conv2d`,
  `MaxPool2d`, `Flatten`, with train/eval mode, model summaries, and
  parameter freezing.
- Losses (MSE, cross-entropy, binary cross-entropy — the latter two as
  numerically stable fused ops), metrics, and eval-safe evaluation
  helpers.
- Training tools: SGD, Adam, StepLR scheduling, gradient clipping,
  mini-batching, and train/validation splitting.
- Persistence: save/load for weights, and full checkpoints carrying
  optimizer state, scheduler state, JSON metadata, and optionally the
  RNG state — so training resumes bit-for-bit, dropout included.

## What makes it educational

Every design decision favors readability. Operations carry short
comments explaining their local derivative; tricky pieces (stable
losses, batchnorm statistics, max-pool gradient routing) explain *why*
they work, not just what they do. Where possible, layers are built by
composing existing autograd ops so their gradients need no new code —
and where a fused op is necessary, finite-difference tests prove the
hand-written backward correct.

## Example learning path

The six runnable examples form a progression, each introducing one
idea: linear regression (the bare training loop), XOR (why hidden
layers exist), a 3-class spiral (real classification, mini-batches,
validation), binary classification (logits and stable losses), a
dropout MLP (train mode vs eval mode), and a tiny CNN (convolution,
pooling, and flattening on synthetic images). All are seeded,
dependency-free, and finish in seconds.

## Testing and reliability

370+ pytest tests cover every feature: known-value checks against
hand-computed math, finite-difference gradient verification, exact
resume-equivalence tests for checkpointing, and guardrail tests that
keep docs, examples, and the public API from drifting apart. The suite
runs in a few seconds.

## Current limitations

Educational, not production-ready. NumPy on CPU only; `Conv2d` and
`MaxPool2d` use deliberately naive loops. No real datasets, no
external ML libraries. There is no C++ backend and no CUDA backend —
both are future experiments, not current features.

## What comes after v3.0

v3.0 closes the Python framework line. Future work moves to advanced
branches: a C++ backend experiment and CUDA/GPU experiments, each a
substantial project of its own. See
[release_history.md](release_history.md) for the full arc.
