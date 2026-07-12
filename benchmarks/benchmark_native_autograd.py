"""Characterization benchmark for the native autograd stack (v2.5).

This measures the current ``tensorforge.experimental.NativeTensor``
autograd stack; it does not try to make it faster. Nothing here asserts a
speedup, and no result is compared against NumPy, PyTorch, or any other
framework — the point is an honest, reproducible snapshot of where time
goes on *one* machine.

The architecture it characterizes: the C++ kernels do the numerical
primitives, Python manages the autograd graph (construction, reverse
traversal, native gradient accumulation, one-shot cleanup), and every
measured time therefore includes the Python wrapper and the ctypes
boundary as well as the native compute.

Four modes separate, as far as the current architecture permits, what
each layer costs:

- ``forward_native`` — inputs do not require grad, so the ops build no
  autograd graph: native forward numerical execution plus wrapper cost.
- ``forward_graph`` — the same forward result with grad-tracking inputs,
  so the Python graph is constructed; backward is *not* called. The gap
  from ``forward_native`` characterizes graph-construction overhead.
- ``forward_backward_fresh`` — a fresh graph plus one default
  ``backward()`` (``retain_graph=False``): forward, graph construction,
  native backward primitives, reverse traversal, gradient accumulation,
  and graph cleanup, all included. Leaf grads are cleared each iteration.
- ``backward_retained`` — one graph built *outside* the timed loop, then
  ``backward(retain_graph=True)`` repeatedly (grads cleared each
  iteration). This isolates repeated backward over a fixed graph; it is
  **not** equivalent to training with fresh inputs, which rebuilds the
  forward graph every step.

Build the backend first:

    uv run python cpp/build.py

Then, for example:

    uv run python benchmarks/benchmark_native_autograd.py            # all cases/modes
    uv run python benchmarks/benchmark_native_autograd.py --smoke    # tiny, fast
    uv run python benchmarks/benchmark_native_autograd.py --case matmul
    uv run python benchmarks/benchmark_native_autograd.py --mode forward_backward_fresh
    uv run python benchmarks/benchmark_native_autograd.py --json > results.json

See docs/native_autograd_benchmarks.md for what each case and mode means
and why results must not be generalized across machines.
"""

import argparse
import json
import platform
import statistics
import time
from datetime import datetime, timezone

import numpy as np

from tensorforge.backends import cpp
from tensorforge.experimental import NativeTensor

BENCHMARK_NAME = "native_autograd"
BENCHMARK_VERSION = "2.5"

# The four measured modes, in reporting order.
MODES = (
    "forward_native",
    "forward_graph",
    "forward_backward_fresh",
    "backward_retained",
)

# Normal defaults and the (tiny) --smoke defaults. --smoke also selects the
# small shape variants below; explicit --warmup/--iterations/--repeats always
# win over either default set.
DEFAULTS = {"warmup": 3, "iterations": 20, "repeats": 5}
SMOKE_DEFAULTS = {"warmup": 1, "iterations": 3, "repeats": 3}


# ---------------------------------------------------------------------------
# Benchmark cases.
#
# Each case builds its input leaves once (NumPy only creates the input data —
# never the benchmarked compute) and a ``forward`` that runs the native op
# chain. Every chain ends in a scalar loss, so backward seeds 1.0 and the
# leaf gradients have the leaves' shapes. A case's requiring modes flip
# requires_grad on every leaf, so backward exercises each operand's rule.
# ---------------------------------------------------------------------------


def _rng(seed):
    return np.random.default_rng(seed)


def _elementwise_inputs(cfg, requires_grad):
    """Same-shape (contiguous, no broadcast) elementwise operands."""
    m, n = cfg["M"], cfg["N"]
    r = _rng(11)
    x = NativeTensor.from_array(r.normal(size=(m, n)), requires_grad=requires_grad)
    scale = NativeTensor.from_array(r.normal(size=(m, n)), requires_grad=requires_grad)
    bias = NativeTensor.from_array(r.normal(size=(m, n)), requires_grad=requires_grad)
    leaves = [x, scale, bias] if requires_grad else []
    return {"operands": (x, scale, bias), "leaves": leaves}


def _elementwise_forward(inp):
    x, scale, bias = inp["operands"]
    return x.multiply(scale).add(bias).relu().mean()


def _broadcast_inputs(cfg, requires_grad):
    """Operands that require genuine broadcasting: a (N,) scale stretched
    over the rows and a (M, 1) bias stretched over the columns."""
    m, n = cfg["M"], cfg["N"]
    r = _rng(22)
    x = NativeTensor.from_array(r.normal(size=(m, n)), requires_grad=requires_grad)
    scale = NativeTensor.from_array(r.normal(size=(n,)), requires_grad=requires_grad)
    bias = NativeTensor.from_array(r.normal(size=(m, 1)), requires_grad=requires_grad)
    leaves = [x, scale, bias] if requires_grad else []
    return {"operands": (x, scale, bias), "leaves": leaves}


def _broadcast_forward(inp):
    x, scale, bias = inp["operands"]
    return x.multiply(scale).add(bias).mean()


def _reduction_inputs(cfg, requires_grad):
    """A multidimensional (3-D) contiguous tensor for a reduction chain."""
    a, b, c = cfg["A"], cfg["B"], cfg["C"]
    r = _rng(33)
    x = NativeTensor.from_array(r.normal(size=(a, b, c)), requires_grad=requires_grad)
    leaves = [x] if requires_grad else []
    return {"operands": (x,), "leaves": leaves}


def _reduction_forward(inp):
    (x,) = inp["operands"]
    return x.mean(axis=1).sum()


def _matmul_inputs(cfg, requires_grad):
    """2-D matmul operands plus a broadcast bias (matching NativeTensor's
    strictly-2-D matmul)."""
    m, k, n = cfg["M"], cfg["K"], cfg["N"]
    r = _rng(44)
    x = NativeTensor.from_array(r.normal(size=(m, k)), requires_grad=requires_grad)
    w = NativeTensor.from_array(r.normal(size=(k, n)), requires_grad=requires_grad)
    bias = NativeTensor.from_array(r.normal(size=(n,)), requires_grad=requires_grad)
    leaves = [x, w, bias] if requires_grad else []
    return {"operands": (x, w, bias), "leaves": leaves}


def _matmul_forward(inp):
    x, w, bias = inp["operands"]
    return x.matmul(w).add(bias).relu().mean()


def _view_chain_inputs(cfg, requires_grad):
    """A single leaf; the chain exercises transpose/narrow/contiguous_copy/
    reshape backward together."""
    m, n = cfg["M"], cfg["N"]
    r = _rng(55)
    x = NativeTensor.from_array(r.normal(size=(m, n)), requires_grad=requires_grad)
    leaves = [x] if requires_grad else []
    keep = max(1, n // 2)
    return {"operands": (x,), "leaves": leaves, "keep": keep, "rows": m}


def _view_chain_forward(inp):
    (x,) = inp["operands"]
    keep, rows = inp["keep"], inp["rows"]
    # transpose -> narrow -> contiguous_copy -> reshape -> mean. reshape needs
    # a contiguous source, so the narrowed (strided) view goes through
    # contiguous_copy first. The owning copy is bound to a name so the
    # borrowing reshape view stays valid through mean() in forward-only mode
    # (there is no graph holding it alive there).
    contiguous = x.T.narrow(0, 0, keep).contiguous_copy()
    flat = contiguous.reshape((keep * rows,))
    return flat.mean()


CASES = {
    "elementwise": {
        "description": "same-shape elementwise chain: x*scale + bias, relu, mean",
        "make_inputs": _elementwise_inputs,
        "forward": _elementwise_forward,
        "shapes": {"full": {"M": 512, "N": 512}, "smoke": {"M": 4, "N": 5}},
        "out_shape": (),
    },
    "broadcast": {
        "description": "broadcast elementwise + reduction: x*(N,) + (M,1), mean",
        "make_inputs": _broadcast_inputs,
        "forward": _broadcast_forward,
        "shapes": {"full": {"M": 512, "N": 512}, "smoke": {"M": 4, "N": 5}},
        "out_shape": (),
    },
    "reduction": {
        "description": "reduction chain over a 3-D contiguous tensor: mean(axis=1), sum",
        "make_inputs": _reduction_inputs,
        "forward": _reduction_forward,
        "shapes": {"full": {"A": 48, "B": 48, "C": 48}, "smoke": {"A": 3, "B": 4, "C": 2}},
        "out_shape": (),
    },
    "matmul": {
        "description": "2-D matmul chain: x @ w + bias, relu, mean",
        "make_inputs": _matmul_inputs,
        "forward": _matmul_forward,
        "shapes": {"full": {"M": 96, "K": 96, "N": 96}, "smoke": {"M": 3, "K": 4, "N": 2}},
        "out_shape": (),
    },
    "view_chain": {
        "description": "view chain: transpose, narrow, contiguous_copy, reshape, mean",
        "make_inputs": _view_chain_inputs,
        "forward": _view_chain_forward,
        "shapes": {"full": {"M": 256, "N": 256}, "smoke": {"M": 4, "N": 6}},
        "out_shape": (),
    },
}


# ---------------------------------------------------------------------------
# Timing and correctness.
# ---------------------------------------------------------------------------


def measure_ns(run_once, warmup, iterations, repeats):
    """Return ``repeats`` per-iteration seconds samples for ``run_once``.

    Each sample times a batch of ``iterations`` calls with
    ``time.perf_counter_ns()`` and divides by ``iterations`` — so a sample
    is the mean per-iteration time within one batch, and the returned list
    has exactly ``repeats`` samples (never only the single fastest run).
    Only ``run_once`` is timed; any setup lives outside this call.
    """
    for _ in range(warmup):
        run_once()
    samples = []
    for _ in range(repeats):
        start = time.perf_counter_ns()
        for _ in range(iterations):
            run_once()
        elapsed_ns = time.perf_counter_ns() - start
        samples.append(elapsed_ns / iterations / 1e9)
    return samples


def _verify(case_name, mode, out, leaves):
    """Correctness gate run once before timing: the output must have the
    case's (scalar) shape and be finite; for backward modes every leaf must
    have a gradient of the leaf's shape with finite values. NumPy is used
    only to *inspect* copied values (``to_numpy``), never to compute the
    benchmarked result."""
    expected = CASES[case_name]["out_shape"]
    if out.shape != expected:
        raise AssertionError(
            f"{case_name}/{mode}: output shape {out.shape} != expected {expected}"
        )
    if not np.all(np.isfinite(out.to_numpy())):
        raise AssertionError(f"{case_name}/{mode}: output is not finite")
    if mode in ("forward_backward_fresh", "backward_retained"):
        for i, leaf in enumerate(leaves):
            if leaf.grad is None:
                raise AssertionError(f"{case_name}/{mode}: leaf {i} has no gradient")
            if leaf.grad.shape != leaf.shape:
                raise AssertionError(
                    f"{case_name}/{mode}: leaf {i} grad shape {leaf.grad.shape} "
                    f"!= {leaf.shape}"
                )
            if not np.all(np.isfinite(leaf.grad.to_numpy())):
                raise AssertionError(f"{case_name}/{mode}: leaf {i} grad not finite")


def _measure_case_mode(case_name, mode, warmup, iterations, repeats, smoke):
    """Build the inputs, run the correctness gate, then time the mode and
    return one result record."""
    spec = CASES[case_name]
    cfg = spec["shapes"]["smoke" if smoke else "full"]
    forward = spec["forward"]
    requires_grad = mode != "forward_native"
    inputs = spec["make_inputs"](cfg, requires_grad)
    leaves = inputs["leaves"]

    if mode in ("forward_native", "forward_graph"):
        def run_once():
            return forward(inputs)
    elif mode == "forward_backward_fresh":
        def run_once():
            for leaf in leaves:
                leaf.zero_grad()
            out = forward(inputs)
            out.backward()
            return out
    elif mode == "backward_retained":
        # Build the graph once, outside the timed loop; the inner loop only
        # re-runs backward over it (grads cleared each iteration).
        retained = forward(inputs)

        def run_once():
            for leaf in leaves:
                leaf.zero_grad()
            retained.backward(retain_graph=True)
            return retained
    else:  # pragma: no cover - modes are validated upstream
        raise ValueError(f"unknown mode {mode!r}")

    # Correctness once, then clear grads so accumulation does not distort
    # the timed workload.
    out = run_once()
    _verify(case_name, mode, out, leaves)
    for leaf in leaves:
        leaf.zero_grad()

    samples = measure_ns(run_once, warmup, iterations, repeats)
    median = statistics.median(samples)
    return {
        "case": case_name,
        "mode": mode,
        "shape": dict(cfg),
        "warmup": warmup,
        "iterations": iterations,
        "repeats": repeats,
        "samples_s": samples,
        "median_s": median,
        "min_s": min(samples),
        "max_s": max(samples),
        "iters_per_s": (1.0 / median) if median > 0 else None,
        "units": "seconds_per_iteration",
    }


# ---------------------------------------------------------------------------
# Orchestration, metadata, and validation.
# ---------------------------------------------------------------------------


def _positive_int(value, name):
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be a positive int, got {value!r}")
    if value < 1:
        raise ValueError(f"{name} must be a positive int, got {value!r}")
    return value


def _resolve(requested, allowed, label):
    """Validate a requested case/mode selection against ``allowed``. ``None``
    means all (in ``allowed`` order); an unknown value raises ValueError."""
    if requested is None:
        return list(allowed)
    resolved = []
    for item in requested:
        if item not in allowed:
            raise ValueError(
                f"unknown {label} {item!r}; choose from {tuple(allowed)}"
            )
        resolved.append(item)
    return resolved


def _metadata(cases, modes, warmup, iterations, repeats, smoke):
    info = cpp.backend_info()
    variant = "smoke" if smoke else "full"
    return {
        "benchmark": BENCHMARK_NAME,
        "version": BENCHMARK_VERSION,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor() or "unknown",
        "python_version": platform.python_version(),
        "native_backend": {
            "name": info["name"],
            "tensor_core": info["tensor_core"],
            "available": info["available"],
            "autograd_integration": info["autograd_integration"],
        },
        "dtype": "float64",
        "device": "cpu",
        "warmup": warmup,
        "iterations": iterations,
        "repeats": repeats,
        "smoke": bool(smoke),
        "cases": list(cases),
        "modes": list(modes),
        "shapes": {name: dict(CASES[name]["shapes"][variant]) for name in cases},
    }


def run_benchmark(cases=None, modes=None, warmup=DEFAULTS["warmup"],
                  iterations=DEFAULTS["iterations"], repeats=DEFAULTS["repeats"],
                  smoke=False):
    """Run the selected cases/modes and return ``{"metadata", "results"}``.

    ``cases``/``modes`` default to all (validated against the registries;
    unknown values raise ValueError). ``warmup``/``iterations``/``repeats``
    must be positive ints. Raises RuntimeError if the native backend is not
    built. No timing threshold is ever applied — this only measures.
    """
    if not cpp.is_available():
        raise RuntimeError(
            "The experimental C++ backend is not built.\n" + cpp.build_instructions()
        )
    selected_cases = _resolve(cases, tuple(CASES), "case")
    selected_modes = _resolve(modes, MODES, "mode")
    warmup = _positive_int(warmup, "warmup")
    iterations = _positive_int(iterations, "iterations")
    repeats = _positive_int(repeats, "repeats")

    results = []
    for case_name in selected_cases:
        for mode in selected_modes:
            results.append(
                _measure_case_mode(case_name, mode, warmup, iterations, repeats, smoke)
            )
    return {
        "metadata": _metadata(
            selected_cases, selected_modes, warmup, iterations, repeats, smoke
        ),
        "results": results,
    }


# ---------------------------------------------------------------------------
# Human-readable report.
# ---------------------------------------------------------------------------


def _format_duration(seconds):
    if seconds < 1e-3:
        return f"{seconds * 1e6:8.2f} us"
    if seconds < 1.0:
        return f"{seconds * 1e3:8.2f} ms"
    return f"{seconds:8.3f} s "


def _shape_str(shape):
    return ",".join(f"{k}={v}" for k, v in shape.items())


def format_report(payload):
    """A clean, aligned human-readable report of a ``run_benchmark`` payload.
    Contains every case and mode name; carries no speed verdict."""
    meta = payload["metadata"]
    lines = [
        f"TensorForge native autograd benchmark v{meta['version']}",
        f"  platform  : {meta['platform']}",
        f"  machine   : {meta['machine']}",
        f"  processor : {meta['processor']}",
        f"  python    : {meta['python_version']}",
        f"  backend   : {meta['native_backend']['tensor_core']} "
        f"({meta['dtype']}/{meta['device']})",
        f"  timestamp : {meta['timestamp']}",
        f"  warmup/iterations/repeats : "
        f"{meta['warmup']}/{meta['iterations']}/{meta['repeats']}"
        + ("   [smoke]" if meta["smoke"] else ""),
        "",
    ]
    header = (
        f"{'case':<12} {'mode':<22} {'shape':<18} "
        f"{'median':>12} {'min':>12} {'max':>12} {'iters/s':>10}"
    )
    lines.append(header)
    lines.append("-" * len(header))
    previous = None
    for rec in payload["results"]:
        if previous is not None and rec["case"] != previous:
            lines.append("")
        previous = rec["case"]
        ips = f"{rec['iters_per_s']:>10.0f}" if rec["iters_per_s"] else f"{'n/a':>10}"
        lines.append(
            f"{rec['case']:<12} {rec['mode']:<22} {_shape_str(rec['shape']):<18} "
            f"{_format_duration(rec['median_s']):>12} "
            f"{_format_duration(rec['min_s']):>12} "
            f"{_format_duration(rec['max_s']):>12} {ips}"
        )
    lines.append("")
    lines.append(
        "Median per-iteration time; min/max show spread across repeats. No"
    )
    lines.append(
        "speed assertions and no cross-framework claims -- one hardware-specific"
    )
    lines.append(
        "snapshot. See docs/native_autograd_benchmarks.md for interpretation."
    )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI.
# ---------------------------------------------------------------------------


def build_parser():
    parser = argparse.ArgumentParser(
        description="Characterize the native autograd stack (measurement only)."
    )
    parser.add_argument(
        "--case", choices=tuple(CASES), default=None,
        help="run a single case (default: all)",
    )
    parser.add_argument(
        "--mode", choices=MODES, default=None,
        help="run a single mode (default: all applicable)",
    )
    parser.add_argument("--warmup", type=int, default=None, help="warmup iterations")
    parser.add_argument(
        "--iterations", type=int, default=None, help="measured iterations per sample"
    )
    parser.add_argument("--repeats", type=int, default=None, help="number of samples")
    parser.add_argument(
        "--json", action="store_true", help="emit machine-readable JSON only"
    )
    parser.add_argument(
        "--smoke", action="store_true",
        help="tiny shapes and counts, for tests/CI",
    )
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    defaults = SMOKE_DEFAULTS if args.smoke else DEFAULTS
    warmup = args.warmup if args.warmup is not None else defaults["warmup"]
    iterations = args.iterations if args.iterations is not None else defaults["iterations"]
    repeats = args.repeats if args.repeats is not None else defaults["repeats"]
    cases = [args.case] if args.case else None
    modes = [args.mode] if args.mode else None
    try:
        payload = run_benchmark(
            cases=cases, modes=modes, warmup=warmup,
            iterations=iterations, repeats=repeats, smoke=args.smoke,
        )
    except (ValueError, RuntimeError) as error:
        parser.error(str(error))  # writes to stderr, exits 2 — stdout stays clean
    if args.json:
        print(json.dumps(payload))
    else:
        print(format_report(payload))


if __name__ == "__main__":
    main()
