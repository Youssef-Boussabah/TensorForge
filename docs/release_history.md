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
compute set with TensorCore matmul over strided views. **v1.4**
upgraded the benchmarks into a suite covering NumPy, the raw-buffer
kernels, and the TensorCore runtime side by side — overheads
included, honestly. **v1.5** added the backend dispatch design
([dispatch_design.md](dispatch_design.md)) and Stage 1 of it: an
explicit backend API (`get_backend("numpy"|"native")`) with no
implicit routing. **v1.6** polished that API's conversion contract —
`to_numpy` as the explicit exit boundary matching `tensor_from_array`,
consistent operand errors, and tested exact-shape native behavior
against NumPy's broadcasting (see
[backend_experiments.md](backend_experiments.md)). **v1.7** is
design-only: it writes the Stage-2 plan for a future forward-only
native tensor wrapper over `NativeTensorCore` — purpose, non-goals,
ownership/lifetime, the inherited conversion contract, a minimal API
sketch, error/shape behavior, a testing plan, and a staged v1.8–v1.11
sequence, all in
[native_tensor_wrapper_design.md](native_tensor_wrapper_design.md); no
code ships. **v1.8** builds the shell of that wrapper —
`tensorforge.experimental.NativeTensor`, a forward-only layer over
`NativeTensorCore` with constructors (`from_array`/`zeros`/`full`),
metadata, `to_numpy`, and an explicit ownership/lifetime story
(idempotent `close()`, context manager, closed tensors reject access) —
and nothing more: no compute ops, no view ops, no autograd, not
`tensorforge.Tensor`. **v1.9** gave that wrapper its forward-only
compute methods — `relu`, `add`, `subtract`, `multiply`, `matmul` —
each delegating to the native kernel and returning a new owning
`NativeTensor`, preserving the runtime's exact-shape/2-D behavior (no
broadcasting), clear `TypeError` naming `NativeTensor` for bad operands,
and `RuntimeError` on closed tensors; no operator overloads, no view
ops, still not `tensorforge.Tensor`. **v1.10** added the metadata-only
view ops — `reshape`, `transpose`, `T`, `narrow` return borrowing
wrappers that share the parent's storage, and `contiguous_copy` returns
a fresh owning wrapper; compute runs over strided views directly,
closing a view spares the owner, and closing the owner invalidates the
views' data access. Still forward-only, no operator overloads, not
`tensorforge.Tensor`. **v1.11** made the wrapper demonstrable — a small,
deterministic runnable example (`examples/native_tensor_demo.py`)
touring construction, compute, views, `contiguous_copy`, and explicit
lifetime, plus a metadata-only `repr` — and documented it as the
experimental, forward-only, isolated tensor it is (no autograd, no
dispatch, no CUDA, no operator overloads, no performance claims).
**v1.12** extended the benchmark suite to characterize the wrapper:
`add`/`relu`/`matmul`, their strided-view forms, and `contiguous_copy`
are timed across four layers — NumPy, the raw-buffer kernels,
`NativeTensorCore`, and `NativeTensor` — so the wrapper's thin
ownership/lifetime/conversion overhead is visible beside the bare
runtime, with correctness checked before timing and no performance
assertions anywhere. **v1.13** is design-only: acting on the v1.12
finding that the elementwise cost is the generic shape/stride odometer
traversal (not the wrapper), it specifies a contiguous fast path for the
native `relu`/`add`/`subtract`/`multiply` kernels — flat index-free loops
for contiguous inputs, the odometer retained for strided views, living in
the `NativeTensorCore`/native-kernel layer so `NativeTensor` inherits it,
bit-for-bit equivalent with unchanged semantics
([native_contiguous_fast_path_design.md](native_contiguous_fast_path_design.md));
no code ships. **v1.14** implements that fast path: flat, index-free
kernels (`tf_core_relu_contiguous` and add/subtract/multiply variants)
sit beside the generic odometer kernels, and `NativeTensorCore.relu` /
`_binary_core_op` pick them when every operand is row-major contiguous
(nonzero offsets and scalars included), falling back to the retained
odometer path for any strided view — proven bit-for-bit equal to it and
to NumPy. `NativeTensor` inherited the change with no wrapper edits, and
no broadcasting, reductions, autograd, `Tensor` integration, or CUDA
were added; performance is left to the benchmark suite, not claimed.
**v1.15** closes that optimization loop with an honest benchmark impact
report (in [backend_experiments.md](backend_experiments.md)): on a local
run of the existing suite, contiguous elementwise `add`/`relu` on
`NativeTensorCore` and `NativeTensor` moved to roughly raw-buffer-C++
speed (~1.5× NumPy at 1000×1000), while the non-contiguous view rows
stayed on the generic odometer path (~2.5–3.5×) — the exact
contiguous-vs-strided spread the design predicted. Matmul and
`contiguous_copy` were unchanged (out of v1.14 scope), and `NativeTensor`
kept tracking `NativeTensorCore` closely, reinforcing that the wrapper is
thin. No kernels or runtime behavior changed, numbers are
hardware-dependent, and no test asserts a speedup. **v1.16** is
design-only: with Phase A1 complete, it writes the design for native
broadcasting — lifting the native elementwise ops
(`add`/`subtract`/`multiply`) from exact-shape-only to NumPy-style
broadcasting (scalar↔tensor, same-rank size-1 stretching, left-padding
with leading 1s) via a zero-stride read model that never materializes an
expanded operand and keeps a freshly allocated contiguous output. The
v1.14 fast path is preserved for the same-shape contiguous case;
broadcasting lives in the `NativeTensorCore`/native-kernel layer so
`NativeTensor` will inherit it with no wrapper change; errors stay
explicit (a mismatch names both shapes, no silent NumPy fallback); and
the autograd implication (a broadcast forward read is a sum-reduction on
the backward pass) is noted for later, not built
([native_broadcasting_design.md](native_broadcasting_design.md)). No code
ships; implementation is v1.17. **v1.17** implements that design:
`NativeTensorCore.add`/`subtract`/`multiply` now broadcast NumPy-style
(scalar↔tensor, same-rank size-1 stretching, left-padding with leading
1s). A pure `broadcast_shapes` helper infers the output shape (raising a
`ValueError` naming both shapes when incompatible) and a `_broadcast_strides`
helper feeds the **existing** generic odometer kernel zero strides on
stretched axes — so no new C++ kernel was added, nothing is materialized,
and the same-shape v1.14 fast path and generic odometer are untouched.
Output stays freshly allocated row-major contiguous; `NativeTensor` (and
the explicit native backend) inherited broadcasting with no wrapper edit;
results match NumPy exactly, including transposed/narrowed/nonzero-offset
operands. No reductions, autograd, `Tensor` integration, CUDA, dtype
promotion, operator overloads, or matmul broadcasting were added.
**v1.18** is design-only: with Phase A2 complete, it writes the design for
native reductions — `sum`/`mean` first (`max`/`argmax`/`min`/`product`
deferred), with NumPy-style `axis=None`/integer/negative-axis and
`keepdims` semantics and a scatter-accumulate traversal that is the **dual
of broadcasting** (broadcasting reads through zero strides; a reduction
writes through zero strides, so the existing odometer machinery drives it,
reads any contiguous/transposed/narrowed/nonzero-offset input directly
without materializing, and writes a freshly allocated row-major contiguous
output). It commits to honest floating-point behavior (order-sensitive
sums, plain deterministic loop, NumPy comparison to a **tolerance** rather
than bit-for-bit, no Kahan/pairwise/SIMD in first scope) and records the
autograd relationship — broadcasting's backward is a reduction over the
broadcast axes, so reductions are a prerequisite for native autograd — but
builds none of it. `NativeTensor` will inherit `sum`/`mean` by delegation
with no wrapper change; no autograd, `Tensor` integration, CUDA, dtype
promotion, operator overloads, tuple-axis, or distributed reductions come
with it ([native_reductions_design.md](native_reductions_design.md)). No
code ships; implementation is v1.19. CUDA/GPU experiments remain future
work.
