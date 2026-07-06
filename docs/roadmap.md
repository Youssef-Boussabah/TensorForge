# Roadmap

## Where the project is

The core educational framework is complete: you can define a model,
train it, regularize it, evaluate it honestly, save it, and resume it —
all from readable NumPy code. The v2.x milestones are about expanding
and polishing rather than filling gaps.

## What's been built

**v0.x — foundations.** The Tensor and autograd engine (elementwise
ops, matmul, exp/log/tanh/sigmoid/relu/softmax with broadcasting-aware
gradients), the module system (Parameter, Module, Linear, activations,
Sequential), SGD, MSE and cross-entropy losses, and the first
examples: linear regression, XOR, and the multi-class spiral.

**v1.x — training basics and evaluation.** The accuracy metric, Adam,
mini-batching, gradient checking against finite differences,
save/load parameters, model summaries and parameter counting, frozen
parameters, train/validation splitting, evaluation helpers, binary
cross-entropy with a binary classification example, and checkpoints
that capture optimizer state so training can resume exactly.

**v2.x — deeper framework features.** Train/eval mode, Dropout (and an
example that uses it properly), eval-safe evaluators, BatchNorm1d with
module buffers, gradient clipping, the StepLR scheduler, and scheduler
state in checkpoints — completing the training-resume story.

## Practical next steps

Roughly in order of usefulness:

- **Conv2d + Flatten** — the first step beyond flat vectors.
- **A tiny CNN-style example** — something image-shaped, still
  synthetic and dependency-free.
- **More docs** — deeper walkthroughs of individual layers as the
  framework grows.
- **Backend planning** — a C++ or GPU backend has always been the
  long-term daydream. It would be a large project and is deliberately
  parked until the Python framework has nothing left to teach.

## What this project is not

TensorForge is not production software and doesn't try to compete with
PyTorch or any real framework. It trades performance for readability
at every opportunity — that's the point. If it helps someone
understand what `loss.backward()` actually does, it has succeeded.
