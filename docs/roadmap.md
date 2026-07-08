# Roadmap

## Where the project is

**v3.0 is the completed Python framework release.** You
can define a model (including CNNs), train it, regularize it,
normalize it, evaluate it honestly, save it, and resume it bit-for-bit
— all from readable NumPy code, all tested, all documented. For the
version-by-version story, see [release_history.md](release_history.md);
for a two-minute overview, see [project_summary.md](project_summary.md).

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
module buffers, gradient clipping, the StepLR scheduler, scheduler
state in checkpoints — completing the training-resume story — and
image-shaped input: Conv2d, MaxPool2d, Flatten, and a tiny CNN
example; LayerNorm as the batch-independent normalization; optional
RNG state in checkpoints for bit-exact dropout resume; and a
release-readiness pass over docs and guardrail tests.

## Practical next steps

The Python line is done; what remains is expansion on its own terms:

- **Advanced branches** — the C++ backend experiment now has
  elementwise kernels, naive and cache-tiled 2-D matmuls, an
  introspection API, and a native runtime prototype: shape/stride
  metadata, a C++-owned NativeStorage buffer, NativeTensorView binding
  the two with native contiguous materialization, and NativeTensorCore
  composing it all into the first native tensor runtime object with
  metadata-only view operations (reshape, transpose, narrow), native
  compute over strided views (elementwise ops and matmul), a
  benchmark suite measuring NumPy, raw-buffer kernels, and the
  TensorCore runtime side by side, and a backend dispatch design plus
  Stage 1 of it — an explicit `get_backend("numpy"|"native")` API with
  no implicit routing and a polished conversion contract
  (`tensor_from_array` in, `to_numpy` out; see
  [backend_experiments.md](backend_experiments.md) and
  [dispatch_design.md](dispatch_design.md)). Stage 2 — a forward-only
  native tensor wrapper over `NativeTensorCore` — is now **designed**
  (purpose, non-goals, ownership/lifetime, conversion contract, minimal
  API, testing plan, and a staged build sequence; see
  [native_tensor_wrapper_design.md](native_tensor_wrapper_design.md)),
  and it is **now feature-complete as a forward-only wrapper**:
  `tensorforge.experimental.NativeTensor`, a forward-only wrapper over
  `NativeTensorCore` with constructors, metadata, `to_numpy`, an
  explicit ownership/lifetime story (v1.8), forward compute — `relu`,
  `add`, `subtract`, `multiply`, `matmul` with exact-shape/2-D behavior
  and no broadcasting (v1.9) — and metadata-only view ops: `reshape`,
  `transpose`, `T`, `narrow` returning borrowing wrappers, plus
  `contiguous_copy` returning an owning one (v1.10). No autograd, no
  operator overloads, not `tensorforge.Tensor`. It is now demonstrable
  too: a small deterministic example
  (`examples/native_tensor_demo.py`), a metadata-only `repr`, and a
  wrapper overview in the docs (v1.11); and honestly characterized —
  the benchmark suite times the wrapper's ops (strided views and
  `contiguous_copy` included) across NumPy, the raw-buffer kernels,
  `NativeTensorCore`, and `NativeTensor`, overheads included and with no
  performance assertions (v1.12). Acting on that finding — the
  elementwise cost is the generic shape/stride odometer traversal in the
  native runtime, not the wrapper — the contiguous elementwise fast path
  is now **designed** (v1.13): flat, index-free loops for contiguous
  `relu`/`add`/`subtract`/`multiply`, the odometer kept for strided
  views, placed in the `NativeTensorCore`/native-kernel layer so
  `NativeTensor` inherits it, bit-for-bit equivalent
  ([native_contiguous_fast_path_design.md](native_contiguous_fast_path_design.md)).
  The next step there is **Advanced C++ v1.14 — NativeTensorCore
  contiguous elementwise fast path**: implementing that design and
  proving it equal to the generic path. CUDA/GPU experiments are still
  entirely future work. The Python framework stays the reference
  implementation.
- **The Daedalus-class native roadmap** — the longer arc the advanced
  branch is building toward, in phases, each landing only when the
  previous is tested and documented:
  - **Phase A — native CPU runtime.** A1: the contiguous elementwise
    fast path (design done in v1.13, implementation v1.14). A2:
    broadcasting for elementwise ops. A3: reductions (sum/mean/max). A4:
    dtype and device metadata beyond float64-CPU-only.
  - **Then** native autograd, a native training stack, the CUDA runtime,
    an AMP / Tensor Core path, Transformer / text examples, distributed
    / DDP, and a final benchmark / profiling / docs polish.
- **A larger synthetic image example** — more classes, bigger images,
  still dependency-free.
- **More docs** — deeper walkthroughs of individual layers, if the
  framework grows further.

## What this project is not

TensorForge is not production software and doesn't try to compete with
PyTorch or any real framework. It trades performance for readability
at every opportunity — that's the point. If it helps someone
understand what `loss.backward()` actually does, it has succeeded.
