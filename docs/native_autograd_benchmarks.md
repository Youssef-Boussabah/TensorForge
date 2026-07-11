# Native autograd benchmarks (Advanced C++ v2.5)

This is the **characterization** milestone for the native autograd stack:
a reproducible harness that measures where time goes in
`tensorforge.experimental.NativeTensor`'s forward + reverse-mode autograd,
and one honest, hardware-specific snapshot of its output. It does **not**
optimize anything, and it makes **no** cross-framework claims.

The harness lives at
[`benchmarks/benchmark_native_autograd.py`](../benchmarks/benchmark_native_autograd.py);
its behavior (not its speed) is tested in
`tests/test_native_autograd_benchmark.py`. It sits beside the older
kernel-vs-NumPy suite ([backend_experiments.md](backend_experiments.md),
`benchmarks/cpp_backend.py`), which it deliberately does not duplicate:
that one compares native kernels against NumPy; this one characterizes the
*autograd* layers on top of them.

## Architecture being characterized

- **C++ kernels do the numerical primitives** (elementwise ops, matmul,
  reductions, the fused `relu_backward`, the `narrow_backward` scatter).
- **Python manages the autograd graph** on `NativeTensor`: construction
  (`_from_op`), reverse-topological traversal, native gradient
  accumulation, one-shot cleanup, `retain_graph`.
- **Every measured time therefore includes the Python wrapper and the
  ctypes boundary** as well as the native compute. That boundary cost is
  not noise to be subtracted away — it is a real part of what running this
  stack costs today, so the harness measures it honestly rather than
  pretending it away.

## Benchmark cases

Each case is a small but representative workload ending in a scalar loss
(so `backward()` seeds `1.0` and leaf gradients have the leaves' shapes).
In the grad-tracking modes every operand is a requiring leaf, so backward
exercises each operand's rule.

| case | forward | what it stresses |
|------|---------|------------------|
| `elementwise` | `x.multiply(scale).add(bias).relu().mean()` (all same shape) | the contiguous elementwise fast path + `relu`/`mean` backward, no broadcasting |
| `broadcast` | `x.multiply(scale).add(bias).mean()` with `scale` `(N,)`, `bias` `(M,1)` | genuine broadcasting forward and the `unbroadcast` reduction on the way back |
| `reduction` | `x.mean(axis=1).sum()` over a 3-D `(A,B,C)` tensor | reduction forward and the broadcast-back of `sum`/`mean` backward |
| `matmul` | `x.matmul(w).add(bias).relu().mean()` (strictly 2-D) | the naive triple-loop matmul forward and its two matmul backward products |
| `view_chain` | `x.T` → `narrow` → `contiguous_copy` → `reshape` → `mean` | the view backward rules (transpose inverse, `narrow` scatter, `contiguous_copy` identity, reshape) composed |

Shapes are configurable through `--smoke` (tiny, for tests/CI) vs the
default full sizes; the selected shapes are recorded in the output
metadata.

## Benchmark modes

Four modes separate, as far as the current architecture permits, what each
layer costs. For each mode only the per-iteration work is timed; input
tensors are created outside the timed loop.

- **`forward_native`** — inputs do **not** require grad, so the ops build
  no autograd graph. Measures native forward numerical execution plus the
  wrapper/ctypes cost, with no graph construction at all.
- **`forward_graph`** — the same forward result with grad-tracking inputs,
  so the Python graph *is* constructed (parents, backward closures, op
  names). `backward()` is **not** called. The gap from `forward_native`
  characterizes graph-construction overhead.
- **`forward_backward_fresh`** — a **fresh** graph each iteration plus one
  default `backward()` (`retain_graph=False`). Includes forward, graph
  construction, the native backward primitives, reverse traversal,
  gradient accumulation, **and** graph cleanup. Leaf gradients are cleared
  each iteration so accumulated history does not distort the workload.
- **`backward_retained`** — one graph is built **outside** the timed loop,
  then `backward(retain_graph=True)` is called repeatedly (grads cleared
  each iteration). This **isolates repeated backward over a fixed graph**:
  it removes forward computation and graph rebuilding from the measured
  loop.

### Why `forward_native` and `forward_graph` differ

They run the identical forward math. The only difference is that
`forward_graph`'s inputs require grad, so each op additionally records a
graph node (allocates the parent tuple, captures a backward closure, sets
the op name). Their difference is therefore the Python graph-construction
overhead, isolated from any backward compute.

### Why `forward_backward_fresh` includes construction and cleanup

It is the closest mode to "one training step's worth of autograd" (minus
an optimizer): building the graph, running backward through it, and then
releasing it under the default one-shot policy. Bundling construction,
backward, and cleanup into one measured unit is deliberate — that whole
unit is what a fresh-graph step pays.

### What `backward_retained` isolates — and why it is not training

It measures repeated backward over a **single, already-built graph**. That
deliberately excludes forward computation and graph rebuilding, so it is
**not** equivalent to ordinary training, where every step builds a fresh
forward graph from fresh inputs. Read `backward_retained` as "the cost of
the backward pass alone over this graph shape," never as a training-step
estimate. (At the end of a retained case the references simply leave scope
and are garbage-collected; the harness does not alter graph-lifetime
behavior for benchmarking.)

## Timing methodology

- `time.perf_counter_ns()` (never `time.time()`), standard library only.
- Configurable **warmup**, **iterations** (calls timed per sample), and
  **repeats** (number of samples). Each sample times a batch of
  `iterations` calls and divides by `iterations`; the harness keeps all
  `repeats` samples.
- The **median** per-iteration time is the primary statistic; **min** and
  **max** report the spread. The single fastest run is never reported
  alone. Optional throughput is iterations per second (`1 / median`).
- Before timing each case/mode the harness runs the workload once as a
  **correctness gate**: it checks the output shape, checks that the output
  is finite, and — for backward modes — that every leaf gradient exists,
  has the leaf's shape, and is finite. Gradients are cleared before timing.
  NumPy appears only to *inspect* copied values (`to_numpy`); it never
  performs a benchmarked native computation.
- No synchronization is used: the CPU backend is synchronous, so there is
  nothing to wait for (and no fake CUDA-style barrier is added).

## Why there are no speed assertions

Benchmark durations are machine-, build-, and load-dependent. A test that
asserted "native beats X", "backward under N ms", "matmul reaches T
FLOP/s", or "results within P%" would flake and, worse, would misrepresent
a measurement tool as a performance guarantee. The tests therefore check
only that the harness runs, validates its inputs, and produces
schema-correct data. The harness records timings; it renders no verdict.

## Hardware-specific results — one snapshot

The table below is a single run on one machine. **Do not generalize it** to
other hardware, other builds, or other loads; re-run the harness to
characterize your own machine. Every value is copied verbatim from the
benchmark output — nothing is estimated, smoothed, or cleaned up.

Command:

```
uv run python benchmarks/benchmark_native_autograd.py --warmup 5 --iterations 40 --repeats 7
```

Environment (from the run's metadata):

- platform: `Windows-11-10.0.26200-SP0`
- machine: `AMD64`
- processor: `Intel64 Family 6 Model 170 Stepping 4, GenuineIntel`
- python: `3.13.14`
- backend: `NativeTensorCore` (float64 / cpu)
- warmup / iterations / repeats: `5 / 40 / 7`

Median per-iteration time (min / max spread across the 7 repeats):

| case | shape | forward_native | forward_graph | forward_backward_fresh | backward_retained |
|------|-------|---------------:|--------------:|-----------------------:|------------------:|
| elementwise | M=512,N=512 | 3.21 ms | 3.52 ms | 8.97 ms | 4.83 ms |
| broadcast   | M=512,N=512 | 2.70 ms | 2.71 ms | 7.57 ms | 4.63 ms |
| reduction   | A=48,B=48,C=48 | 286.64 us | 238.69 us | 1.24 ms | 878.46 us |
| matmul      | M=96,K=96,N=96 | 629.97 us | 767.24 us | 2.35 ms | 1.49 ms |
| view_chain  | M=256,N=256 | 300.99 us | 257.96 us | 1.64 ms | 1.15 ms |

### Cautious interpretation

Only observations the numbers above actually support, for this machine:

- **Adding a backward pass is the dominant cost.** In every case
  `forward_backward_fresh` is roughly 2.5×–5× the matching
  `forward_native` (e.g. elementwise 3.21 ms → 8.97 ms; matmul 630 us →
  2.35 ms) — expected, since backward runs additional native primitives
  (a second/third matmul, reduction broadcast-backs, scatter),
  accumulation, traversal, and cleanup.
- **`backward_retained` is below `forward_backward_fresh` everywhere**
  (e.g. elementwise 4.83 ms vs 8.97 ms; matmul 1.49 ms vs 2.35 ms),
  because it removes forward recomputation and graph rebuilding from the
  measured loop. This confirms the modes isolate what they claim to — not
  that retained backward is "faster" in any training-relevant sense.
- **Backward cost varies substantially by workload**, from ~880 us
  (reduction) to ~4.8 ms (elementwise) for the retained pass — the graph
  shape and the per-op backward work differ a lot between cases.
- **Graph construction overhead is small relative to compute at these
  sizes.** `forward_graph` sits close to `forward_native` for the
  elementwise and broadcast cases; the clearest construction gap here is
  matmul (630 us → 767 us). For the cheapest cases the difference is within
  the min/max spread, so it should not be over-read.
- **matmul spends a larger share of its time in numerical compute** than
  the tiny reduction/view cases — the naive single-threaded triple loop
  dominates, which is also why its construction gap is the most visible.
- On the `--smoke` shapes (single-digit dimensions) every mode collapses to
  tens of microseconds regardless of the math, i.e. **wrapper and ctypes
  boundary cost dominate very small tensors** — which is exactly why the
  full sizes above exist.

These are cautious, machine-specific observations. There is **no** claim
that TensorForge is faster than NumPy, PyTorch, TensorFlow, or anything
else; **no** production-performance claim; **no** universal scaling,
throughput, or GPU implication; and **no** speedup that was not directly
measured.

## How to run

```
# tiny, fast (tests/CI):
uv run python benchmarks/benchmark_native_autograd.py --smoke

# the complete benchmark (default full shapes):
uv run python benchmarks/benchmark_native_autograd.py

# with explicit sampling (as used for the snapshot above):
uv run python benchmarks/benchmark_native_autograd.py --warmup 5 --iterations 40 --repeats 7

# a single case or a single mode:
uv run python benchmarks/benchmark_native_autograd.py --case matmul
uv run python benchmarks/benchmark_native_autograd.py --mode forward_backward_fresh

# machine-readable JSON (raw samples included; stdout is pure JSON):
uv run python benchmarks/benchmark_native_autograd.py --json > results.json
```

Unknown cases/modes and non-positive warmup/iterations/repeats are
rejected with a clear error. The JSON payload carries the full metadata
and, per record, the raw per-repeat samples alongside the median/min/max —
so a run can be re-analyzed without re-timing.

## Using this harness for future optimization work

This is a **characterization baseline**, not a target. Future native-runtime
work (a faster matmul, a fused kernel, lower wrapper overhead) should:

- re-run the **same** cases and modes so the comparison is like-for-like;
- keep the modes' meanings fixed — `forward_native` stays graph-free,
  `forward_graph` builds the graph without backward, `forward_backward_fresh`
  bundles construction + backward + cleanup, `backward_retained` isolates
  repeated backward — so a later "before/after" reads honestly;
- record its own hardware snapshot rather than editing this one; and
- add speed *measurements*, never speed *assertions*.

Rewriting what a mode measures would silently break the historical meaning
of every past snapshot, so extend the harness (new cases/modes) rather than
redefining the existing ones.
