"""Characterization benchmark for the native CNN stack (Phase D, D12).

This measures the completed native convolution/pooling line — the
``NativeTensor.conv2d`` and ``NativeTensor.maxpool2d`` primitives and the
full ``NativeConv2d → NativeReLU → NativeMaxPool2d → NativeFlatten →
NativeLinear`` training step. It does **not** try to make anything faster,
and nothing here asserts a speedup: the kernels are deliberately direct
nested loops (correctness-first, no im2col, no BLAS, no threading, no
SIMD), so this is an honest, reproducible snapshot of where time goes on
*one* machine — not a competitive comparison, not a scalability claim, and
not a statement about GPUs or any other framework.

The architecture it characterizes: the C++ kernels do the numerical work,
Python manages the autograd graph and the module/optimizer layers, and
every measured time therefore includes the Python wrapper and the ctypes
boundary as well as the native compute.

Five modes separate, as far as the architecture permits, what each layer
costs:

- ``forward_native`` — operands do not require grad, so no autograd graph
  is built: native forward execution plus wrapper/ctypes overhead. For the
  pooling case this still allocates and fills the private winner buffer,
  which the forward kernel produces in the same pass.
- ``forward_graph`` — the same forward with grad-tracking operands, so the
  Python graph node is constructed (and, for pooling, its saved winners
  are retained); backward is *not* called. The gap from
  ``forward_native`` characterizes graph-construction overhead.
- ``forward_backward_fresh`` — a fresh graph plus one default
  ``backward()``: forward, graph construction, the native backward
  kernels, reverse traversal, gradient accumulation, and graph cleanup
  (including the winner-buffer release), all included.
- ``training_step`` — the end-to-end cost of one real iteration of the D11
  proof: forward, loss, backward, ``optimizer.step()``, ``zero_grad()``,
  and the per-step tensor cleanup. Only the ``cnn`` case supports it.
- ``stable_forward`` — the **stable NumPy framework's** equivalent forward
  (``tensorforge.nn``) on the same shapes, as a reference point. Both
  lines are naive loop implementations; this comparison says which
  *implementation* is faster on this machine today, and nothing more.

Build the backend first:

    uv run python cpp/build.py

Then, for example:

    uv run python benchmarks/benchmark_native_cnn.py             # all cases/modes
    uv run python benchmarks/benchmark_native_cnn.py --smoke     # tiny, fast
    uv run python benchmarks/benchmark_native_cnn.py --case conv2d
    uv run python benchmarks/benchmark_native_cnn.py --mode training_step
    uv run python benchmarks/benchmark_native_cnn.py --json > results.json
"""

import argparse
import json
import platform
import statistics
import time
from datetime import datetime, timezone

import numpy as np

from tensorforge.backends import cpp
from tensorforge.experimental import (
    NativeAdam,
    NativeConv2d,
    NativeFlatten,
    NativeLinear,
    NativeMaxPool2d,
    NativeMSELoss,
    NativeReLU,
    NativeSequential,
    NativeTensor,
)

BENCHMARK_NAME = "native_cnn"
BENCHMARK_VERSION = "1.0"

MODES = (
    "forward_native",
    "forward_graph",
    "forward_backward_fresh",
    "training_step",
    "stable_forward",
)

# Deterministic shapes. "full" stays small on purpose — the kernels are
# direct nested loops, so a realistic image size would take minutes and
# tell us nothing new about where the time goes.
DEFAULTS = {"warmup": 2, "iterations": 5, "repeats": 5}
SMOKE_DEFAULTS = {"warmup": 1, "iterations": 2, "repeats": 2}


def _rng(seed):
    return np.random.default_rng(seed)


# ---------------------------------------------------------------------------
# Case builders. Each returns {"leaves": [...], "close": [...]} plus the
# operands its forward needs; timing never includes this setup.
# ---------------------------------------------------------------------------


def _conv2d_inputs(cfg, requires_grad):
    rng = _rng(0)
    x = NativeTensor.from_array(
        rng.standard_normal((cfg["n"], cfg["c"], cfg["h"], cfg["w"])),
        requires_grad=requires_grad,
    )
    weight = NativeTensor.from_array(
        rng.standard_normal((cfg["o"], cfg["c"], cfg["k"], cfg["k"])),
        requires_grad=requires_grad,
    )
    bias = NativeTensor.from_array(
        rng.standard_normal(cfg["o"]), requires_grad=requires_grad
    )
    return {"x": x, "weight": weight, "bias": bias,
            "leaves": [x, weight, bias] if requires_grad else []}


def _conv2d_forward(inp):
    return inp["x"].conv2d(inp["weight"], inp["bias"], stride=1, padding=0).sum()


def _conv2d_stable(cfg):
    from tensorforge.nn import Conv2d
    from tensorforge.tensor import Tensor

    rng = _rng(0)
    layer = Conv2d(cfg["c"], cfg["o"], cfg["k"], stride=1, padding=0, bias=True)
    layer.weight.data = rng.standard_normal(
        (cfg["o"], cfg["c"], cfg["k"], cfg["k"])
    )
    layer.bias.data = rng.standard_normal(cfg["o"])
    x = Tensor(rng.standard_normal((cfg["n"], cfg["c"], cfg["h"], cfg["w"])))
    return lambda: layer(x)


def _maxpool2d_inputs(cfg, requires_grad):
    x = NativeTensor.from_array(
        _rng(1).standard_normal((cfg["n"], cfg["c"], cfg["h"], cfg["w"])),
        requires_grad=requires_grad,
    )
    return {"x": x, "k": cfg["k"], "leaves": [x] if requires_grad else []}


def _maxpool2d_forward(inp):
    return inp["x"].maxpool2d(kernel_size=inp["k"], stride=inp["k"]).sum()


def _maxpool2d_stable(cfg):
    from tensorforge.nn import MaxPool2d
    from tensorforge.tensor import Tensor

    layer = MaxPool2d(cfg["k"], stride=cfg["k"])
    x = Tensor(_rng(1).standard_normal((cfg["n"], cfg["c"], cfg["h"], cfg["w"])))
    return lambda: layer(x)


def _cnn_inputs(cfg, requires_grad):
    """The D11 model and data. ``requires_grad`` only controls the *input*;
    the layer parameters always require grad (they are parameters), so
    ``forward_native`` here means "no grad-tracking input", which is how the
    stack is actually evaluated."""
    features = cfg["o"] * ((cfg["h"] - cfg["k"] + 1) // cfg["p"]) * (
        (cfg["w"] - cfg["k"] + 1) // cfg["p"]
    )
    model = NativeSequential(
        NativeConv2d(cfg["c"], cfg["o"], cfg["k"], seed=0),
        NativeReLU(),
        NativeMaxPool2d(cfg["p"]),
        NativeFlatten(),
        NativeLinear(features, cfg["out"], seed=1),
    )
    rng = _rng(2)
    x = NativeTensor.from_array(
        rng.standard_normal((cfg["n"], cfg["c"], cfg["h"], cfg["w"])),
        requires_grad=requires_grad,
    )
    y = NativeTensor.from_array(rng.standard_normal((cfg["n"], cfg["out"])))
    return {
        "model": model, "x": x, "y": y,
        "loss_fn": NativeMSELoss(),
        "optimizer": NativeAdam(model.parameters(), lr=0.01),
        "leaves": list(model.parameters()) + ([x] if requires_grad else []),
    }


def _cnn_forward(inp):
    return inp["loss_fn"](inp["model"](inp["x"]), inp["y"])


def _cnn_stable(cfg):
    from tensorforge.nn import Conv2d, Flatten, Linear, MaxPool2d, ReLU, Sequential
    from tensorforge.tensor import Tensor

    features = cfg["o"] * ((cfg["h"] - cfg["k"] + 1) // cfg["p"]) * (
        (cfg["w"] - cfg["k"] + 1) // cfg["p"]
    )
    model = Sequential(
        Conv2d(cfg["c"], cfg["o"], cfg["k"]), ReLU(), MaxPool2d(cfg["p"]),
        Flatten(), Linear(features, cfg["out"]),
    )
    x = Tensor(_rng(2).standard_normal((cfg["n"], cfg["c"], cfg["h"], cfg["w"])))
    return lambda: model(x)


CASES = {
    "conv2d": {
        "make_inputs": _conv2d_inputs,
        "forward": _conv2d_forward,
        "stable": _conv2d_stable,
        "modes": ("forward_native", "forward_graph", "forward_backward_fresh",
                  "stable_forward"),
        "shapes": {
            "full": {"n": 4, "c": 3, "h": 16, "w": 16, "o": 4, "k": 3},
            "smoke": {"n": 1, "c": 1, "h": 6, "w": 6, "o": 2, "k": 2},
        },
    },
    "maxpool2d": {
        "make_inputs": _maxpool2d_inputs,
        "forward": _maxpool2d_forward,
        "stable": _maxpool2d_stable,
        "modes": ("forward_native", "forward_graph", "forward_backward_fresh",
                  "stable_forward"),
        "shapes": {
            "full": {"n": 4, "c": 4, "h": 16, "w": 16, "k": 2},
            "smoke": {"n": 1, "c": 2, "h": 6, "w": 6, "k": 2},
        },
    },
    "cnn": {
        "make_inputs": _cnn_inputs,
        "forward": _cnn_forward,
        "stable": _cnn_stable,
        "modes": ("forward_native", "forward_graph", "forward_backward_fresh",
                  "training_step", "stable_forward"),
        "shapes": {
            "full": {"n": 8, "c": 1, "h": 12, "w": 12, "o": 4, "k": 3,
                     "p": 2, "out": 1},
            "smoke": {"n": 4, "c": 1, "h": 6, "w": 6, "o": 2, "k": 2,
                      "p": 2, "out": 1},
        },
    },
}


def measure(run_once, warmup, iterations, repeats):
    """Return ``repeats`` per-iteration seconds samples for ``run_once``.

    Each sample times a batch of ``iterations`` calls with
    ``time.perf_counter_ns()`` and divides by ``iterations``, so a sample is
    the mean per-iteration time within one batch and the result has exactly
    ``repeats`` samples — never only the single fastest run. Only
    ``run_once`` is timed; setup lives outside. CPU execution is
    synchronous, so no explicit synchronization is needed or performed."""
    for _ in range(warmup):
        run_once()
    samples = []
    for _ in range(repeats):
        start = time.perf_counter_ns()
        for _ in range(iterations):
            run_once()
        samples.append((time.perf_counter_ns() - start) / iterations / 1e9)
    return samples


def _verify(case_name, mode, out, leaves):
    """Correctness gate run once before timing: the output must be finite,
    and in ``forward_backward_fresh`` every leaf must carry a finite
    gradient of its own shape. ``training_step`` ends with ``zero_grad()``
    by construction, so its gradients are gated through the same case's
    ``forward_backward_fresh`` mode instead. NumPy only *inspects* copied
    values, never computes the measured result."""
    values = out.to_numpy() if hasattr(out, "to_numpy") else out.data
    if not np.all(np.isfinite(values)):
        raise AssertionError(f"{case_name}/{mode}: output is not finite")
    if mode == "forward_backward_fresh":
        for index, leaf in enumerate(leaves):
            if leaf.grad is None:
                raise AssertionError(
                    f"{case_name}/{mode}: leaf {index} has no gradient"
                )
            if leaf.grad.shape != leaf.shape:
                raise AssertionError(
                    f"{case_name}/{mode}: leaf {index} grad shape "
                    f"{leaf.grad.shape} != {leaf.shape}"
                )
            if not np.all(np.isfinite(leaf.grad.to_numpy())):
                raise AssertionError(
                    f"{case_name}/{mode}: leaf {index} grad not finite"
                )


def _measure_case_mode(case_name, mode, warmup, iterations, repeats, smoke):
    """Build the operands, run the correctness gate, then time the mode and
    return one result record."""
    spec = CASES[case_name]
    if mode not in spec["modes"]:
        raise ValueError(f"case {case_name!r} does not support mode {mode!r}")
    cfg = spec["shapes"]["smoke" if smoke else "full"]

    if mode == "stable_forward":
        run_once = spec["stable"](cfg)
        leaves = []
        out = run_once()
    else:
        forward = spec["forward"]
        inputs = spec["make_inputs"](cfg, mode != "forward_native")
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
        else:  # training_step — the full D11 iteration
            optimizer = inputs["optimizer"]

            def run_once():
                out = forward(inputs)
                out.backward()
                optimizer.step()
                optimizer.zero_grad()
                return out
        out = run_once()

    _verify(case_name, mode, out, leaves)
    for leaf in leaves:
        if hasattr(leaf, "zero_grad"):
            leaf.zero_grad()

    samples = measure(run_once, warmup, iterations, repeats)
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


def _positive_int(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive int, got {value!r}")
    return value


def _resolve(requested, allowed, label):
    if requested is None:
        return tuple(allowed)
    selected = tuple(requested)
    for item in selected:
        if item not in allowed:
            raise ValueError(
                f"unknown {label} {item!r}; choose from {tuple(allowed)}"
            )
    return selected


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
            "native_autograd": info["native_autograd"],
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
                  iterations=DEFAULTS["iterations"],
                  repeats=DEFAULTS["repeats"], smoke=False):
    """Run the selected cases/modes and return ``{"metadata", "results"}``.

    ``cases``/``modes`` default to all (validated against the registries;
    unknown values raise ValueError). A mode a case does not declare is
    skipped rather than failing, so ``--mode training_step`` runs only the
    end-to-end case. Raises RuntimeError if the native backend is not built.
    No timing threshold is ever applied — this only measures."""
    if not cpp.is_available():
        raise RuntimeError(
            "The experimental C++ backend is not built.\n"
            + cpp.build_instructions()
        )
    selected_cases = _resolve(cases, tuple(CASES), "case")
    selected_modes = _resolve(modes, MODES, "mode")
    warmup = _positive_int(warmup, "warmup")
    iterations = _positive_int(iterations, "iterations")
    repeats = _positive_int(repeats, "repeats")

    results = []
    for case_name in selected_cases:
        for mode in selected_modes:
            if mode not in CASES[case_name]["modes"]:
                continue
            results.append(
                _measure_case_mode(case_name, mode, warmup, iterations,
                                   repeats, smoke)
            )
    if not results:
        raise ValueError(
            f"no case in {selected_cases} supports any mode in {selected_modes}"
        )
    return {
        "metadata": _metadata(selected_cases, selected_modes, warmup,
                              iterations, repeats, smoke),
        "results": results,
    }


def _format_duration(seconds):
    if seconds < 1e-3:
        return f"{seconds * 1e6:8.2f} us"
    if seconds < 1.0:
        return f"{seconds * 1e3:8.2f} ms"
    return f"{seconds:8.3f} s "


def _shape_str(shape):
    return ",".join(f"{k}={v}" for k, v in shape.items())


def format_report(payload):
    """A clean, aligned human-readable report of a ``run_benchmark``
    payload. Contains every case and mode name; carries no speed verdict."""
    meta = payload["metadata"]
    lines = [
        f"TensorForge native CNN benchmark v{meta['version']}",
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
        f"{'case':<11} {'mode':<22} {'shape':<34} "
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
            f"{rec['case']:<11} {rec['mode']:<22} {_shape_str(rec['shape']):<34} "
            f"{_format_duration(rec['median_s']):>12} "
            f"{_format_duration(rec['min_s']):>12} "
            f"{_format_duration(rec['max_s']):>12} {ips}"
        )
    lines.append("")
    lines.append(
        "Median per-iteration time; min/max show spread across repeats. The"
    )
    lines.append(
        "native kernels are deliberately direct nested loops (no im2col, BLAS,"
    )
    lines.append(
        "threading, or SIMD), so `stable_forward` is a reference point between"
    )
    lines.append(
        "two naive implementations on this machine -- not a speed claim, not a"
    )
    lines.append(
        "cross-framework comparison, and not a scalability result."
    )
    return "\n".join(lines)


def build_parser():
    parser = argparse.ArgumentParser(
        description="Characterize the native CNN stack (measurement only)."
    )
    parser.add_argument("--case", choices=tuple(CASES), default=None,
                        help="run a single case (default: all)")
    parser.add_argument("--mode", choices=MODES, default=None,
                        help="run a single mode (default: all applicable)")
    parser.add_argument("--warmup", type=int, default=None,
                        help="warmup iterations")
    parser.add_argument("--iterations", type=int, default=None,
                        help="measured iterations per sample")
    parser.add_argument("--repeats", type=int, default=None,
                        help="number of samples")
    parser.add_argument("--json", action="store_true",
                        help="emit machine-readable JSON only")
    parser.add_argument("--smoke", action="store_true",
                        help="tiny shapes and counts, for tests/CI")
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    defaults = SMOKE_DEFAULTS if args.smoke else DEFAULTS
    warmup = args.warmup if args.warmup is not None else defaults["warmup"]
    iterations = (args.iterations if args.iterations is not None
                  else defaults["iterations"])
    repeats = args.repeats if args.repeats is not None else defaults["repeats"]
    try:
        payload = run_benchmark(
            cases=[args.case] if args.case else None,
            modes=[args.mode] if args.mode else None,
            warmup=warmup, iterations=iterations, repeats=repeats,
            smoke=args.smoke,
        )
    except (ValueError, RuntimeError) as error:
        parser.error(str(error))  # stderr, exit 2 — stdout stays clean
    if args.json:
        print(json.dumps(payload))
    else:
        print(format_report(payload))


if __name__ == "__main__":
    main()
