# Release history

A short summary of how TensorForge grew, milestone line by milestone.
Details live in the docs and the test suite; this is the map.

## v0.x — Autograd and basics

The Tensor and the reverse-mode autograd engine: elementwise ops,
matmul, exp/log/tanh/sigmoid/relu/softmax, broadcasting-aware
gradients. The module system (Parameter, Module, Linear, activations,
Sequential), SGD, MSE and cross-entropy losses, and the first three
examples — linear regression, XOR, and the multi-class spiral.

## v1.x — Training stack and framework utilities

Everything a real training loop needs: the accuracy metric, Adam,
mini-batching, gradient checking against finite differences, save/load
parameters, model summaries and parameter counting, frozen parameters,
train/validation splitting, evaluation helpers, binary cross-entropy
with a binary classification example, and checkpoints that capture
optimizer state so training can resume exactly.

## v2.x — Regularization, CNN support, and release readiness

Train/eval mode and Dropout (with an example that measures honestly in
eval mode), eval-safe evaluators, BatchNorm1d with module buffers,
gradient clipping, the StepLR scheduler and scheduler checkpointing,
the docs set, image-shaped input (Conv2d, MaxPool2d, Flatten, and the
tiny CNN example), LayerNorm, optional RNG state in checkpoints for
bit-exact dropout resume, and this release-readiness pass.

## v3.0 — Portfolio Release

The completed Python framework line: consistent docs, a clean landing
page, a project summary, CI, and guardrail tests that keep docs and
code from drifting apart.

## Advanced branches — after v3.0

The experimental line, separate from the finished Python framework.
**C++ v0.1** built the first proof: a tiny compiled elementwise-add
kernel called from Python through ctypes. **v0.1.1** made CI build and
hard-verify the compiled backend. **v0.2** grew the kernel family to
add/subtract/multiply/divide/ReLU. **v0.3** added a naive 2-D matmul
kernel. **v0.4** added honest benchmarks against NumPy — which NumPy
mostly wins, as expected. **v0.5** cleaned up the backend API with
introspection helpers (`is_available`, `list_kernels`,
`backend_info`) and lazy library loading. **v0.6** added
`matmul_tiled`, a cache-blocking optimization experiment benchmarked
against the naive reference and NumPy. **v0.7** added the
shape/stride metadata layer. **v0.8** added `NativeStorage`, a
C++-owned float64 buffer. **v0.9** bound the two together as
`NativeTensorView`, with native contiguous materialization as its
first operation. **v1.0** composed storage + view into
`NativeTensorCore`, the first native tensor runtime object. **v1.1**
gave it metadata-only view operations — reshape, transpose/T, narrow —
sharing storage without copying. **v1.2** made the runtime
self-contained for simple compute: relu/add/subtract/multiply as
native kernels reading strided views directly. **v1.3** completed the
compute set with TensorCore matmul over strided views (see
[backend_experiments.md](backend_experiments.md)). CUDA/GPU
experiments remain future work.
