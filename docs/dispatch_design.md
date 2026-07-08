# Backend dispatch design

How TensorForge's backends should eventually meet its Tensor — and
why, for now, they deliberately don't. This document is the design
record for the staged path; Stage 1 (the explicit backend API) is
implemented.

## The layers as they exist

1. **NumPy-backed Python Tensor frontend** — `tensorforge.Tensor`,
   the stable, complete framework line: autograd, modules, optimizers,
   checkpointing. All math is NumPy.
2. **Raw-buffer C++ backend** — compiled kernels over contiguous
   NumPy arrays (`tensorforge.backends.cpp.elementwise_add`, ...),
   converted at the call boundary. Proof of mechanism.
3. **NativeTensorCore runtime** — C++-owned storage, shape/stride
   metadata, views, and native kernels that compute over strided
   layouts without NumPy touching the data.
4. **Future CUDA backend** — does not exist. A separate branch,
   eventually, following the same staged discipline.

## Why explicit selection comes before Tensor integration

The frontend is finished and trusted: 500+ tests, checkpoint
compatibility, examples that train. Any implicit wiring of the native
backend into `Tensor` puts all of that at risk for zero user benefit —
the native kernels are reference loops, not a speedup. So the first
public step is the opposite of dispatch: an **explicit backend API**
(`get_backend("numpy")`, `get_backend("native")`) where the caller
names the backend, receives a backend object, and everything that
happens after is visibly that backend's behavior. Nothing selects a
backend on the user's behalf.

## Why no silent switching

A framework that silently routes an operation to a different backend
must guarantee bit-compatible results, identical error behavior,
identical view/aliasing semantics, and identical performance
intuitions — or users end up debugging the router. TensorForge's
native backend intentionally differs from NumPy today (no
broadcasting, float64 only, explicit close semantics). Until those
differences are either eliminated or formally specified, switching
silently would convert every difference into a latent bug.

**Rule: no implicit fallback during experimental backend mode.** If a
native operation cannot run, it raises; it does not quietly compute
the answer with NumPy. A later stage may define explicit, opt-in
fallback — never before the semantics are pinned down. (Corollary:
the NumPy backend object follows NumPy semantics, e.g. broadcasting,
while the native backend requires exact shapes. The explicit API does
not paper over this Stage-1 asymmetry; aligning or specifying it is
Stage 2/3 work.)

## Why NativeTensorCore is not tensorforge.Tensor

`Tensor` carries autograd machinery: `requires_grad`, `grad`, graph
edges, `backward()`. `NativeTensorCore` carries none of that — it is
storage + layout + forward kernels. Bolting one onto the other
prematurely would force answers to every risk below at once. They
stay separate until each answer exists.

## How they might eventually meet

The plausible shape: `Tensor` gains an internal *data provider*
abstraction, where today's NumPy array is one provider and a
NativeTensorCore is another. Forward ops route through the provider;
autograd stays in Python and orchestrates providers. Conversion is
explicit (`Tensor.to_backend("native")`-style), copies are visible,
and mixing providers in one graph is initially an error. That design
is Stage 4 material — sketched here, decided there.

## Why dispatch is risky (the checklist any future stage must answer)

- **Autograd correctness** — every native forward op needs a backward
  story; gradients must match the NumPy path bit-for-bit or within a
  specified tolerance, verified by the existing gradcheck discipline.
- **Dtype mismatch** — native is float64-only; Tensor is float64
  today, but any future dtype work multiplies the matrix.
- **Device mismatch** — CPU-only today; CUDA later means every op
  must define what happens when operands live in different places.
- **Memory ownership/lifetime** — NativeTensorCore has explicit
  close() and owner/borrower semantics; Tensor has garbage-collected
  NumPy arrays. Mixing the two lifetimes needs rules, not luck.
- **Fallback behavior** — see the rule above: none, until specified.
- **View semantics** — native views share storage and can be
  invalidated by an owner's close; NumPy views share memory with
  different aliasing rules. Autograd assumes value semantics in
  several places.
- **Broadcasting** — NumPy broadcasts; native rejects. A dispatch
  layer must pick one contract per op and enforce it everywhere.
- **Optimizer state location** — Adam's moments are NumPy arrays
  updated in Python; native parameters would need state on the same
  side as the data or pay conversion costs every step.
- **Checkpoint compatibility** — checkpoints are `.npz` of NumPy
  arrays. Native tensors must serialize through the same format or
  version it explicitly.

## Staged path

- **Stage 1 — explicit backend API only** (this milestone): named
  backend objects with a small common surface; no Tensor contact.
- **Stage 2 — experimental native forward tensors**: a thin
  Tensor-like forward-only wrapper over NativeTensorCore, exercised in
  isolation.
- **Stage 3 — conversion APIs**: explicit, copy-visible bridges
  between NumPy-backed Tensors and native tensors.
- **Stage 4 — native autograd design**: the provider abstraction and
  the backward story, on paper before in code.
- **Stage 5 — optional native training demo**: one small model
  training end-to-end on the native path, still opt-in.
- **Stage 6 — future CUDA branch**: the same staged discipline,
  different device.

Each stage lands only when the previous one is tested and documented.

## Non-goals for this milestone

No Tensor integration. No autograd on the native path. No CUDA. No
performance promises — the native backend remains a runtime
correctness experiment, and NumPy remains the reference
implementation.
