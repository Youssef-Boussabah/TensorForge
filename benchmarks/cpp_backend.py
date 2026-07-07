"""Honest benchmarks: the experimental C++ backend vs NumPy.

This script exists for honest measurement, not marketing. It compares
three implementations of each operation:

- NumPy (the baseline; optimized C, and BLAS for matmul);
- the raw-buffer C++ kernels (naive single-threaded loops over
  contiguous NumPy arrays, converted at the call boundary);
- the NativeTensorCore runtime kernels, which read native storage
  through shape/stride metadata — their timings therefore include the
  Python wrapper, output-storage allocation, and strided traversal,
  which is exactly the overhead worth seeing. View rows feed
  non-contiguous (transposed) inputs straight to the native kernels.

Expect NumPy to win, often dramatically for matmul. Nothing here
asserts a speedup; numbers are hardware-dependent.

Build the backend first:

    uv run python cpp/build.py

Then run:

    uv run python benchmarks/cpp_backend.py          # default sizes
    uv run python benchmarks/cpp_backend.py --quick  # fast smoke run
"""

import argparse
import statistics
import time

import numpy as np


def _load_backend():
    from tensorforge.backends import cpp

    if not cpp.is_available():
        raise SystemExit(
            "The experimental C++ backend is not available.\n"
            + cpp.build_instructions()
        )
    return cpp


def measure(fn, repeats, warmup=2):
    """Median wall-clock seconds over ``repeats`` runs, after warmup."""
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        times.append(time.perf_counter() - start)
    return statistics.median(times)


def _format_time(seconds):
    if seconds < 1e-3:
        return f"{seconds * 1e6:8.1f} us"
    if seconds < 1.0:
        return f"{seconds * 1e3:8.2f} ms"
    return f"{seconds:8.2f} s "


def _as_numpy(result):
    """Normalize a benchmark result for correctness checking: tensor
    cores materialize, NumPy arrays pass through."""
    if hasattr(result, "to_numpy"):
        return result.to_numpy()
    return np.asarray(result)


def build_suite(cpp, quick):
    """Build the benchmark plan: a list of groups, one per
    (operation, size). Each group has a timed NumPy ``baseline`` and
    ``implementations`` — (name, timed_fn, reference_fn) triples.
    Every implementation carries its own correctness reference because
    view cases legitimately compute transposed results.
    """
    core = cpp.NativeTensorCore
    rng = np.random.default_rng(0)
    elementwise_shapes = [(250, 400)] if quick else [(25, 40), (1_000, 1_000)]
    matmul_sizes = [48] if quick else [96, 192]

    groups = []

    for shape in elementwise_shapes:
        x = rng.normal(size=shape)
        y = rng.normal(size=shape)
        cx, cy = core.from_array(x), core.from_array(y)
        label = f"({shape[0]}x{shape[1]})"
        groups.append({
            "operation": "add",
            "shape": label,
            "baseline": lambda x=x, y=y: x + y,
            "implementations": [
                ("cpp raw buffer",
                 lambda x=x, y=y: cpp.elementwise_add(x, y),
                 lambda x=x, y=y: x + y),
                ("tensor core",
                 lambda cx=cx, cy=cy: cx.add(cy),
                 lambda x=x, y=y: x + y),
                ("tensor core (view)",
                 lambda cx=cx, cy=cy: cx.T.add(cy.T),
                 lambda x=x, y=y: x.T + y.T),
            ],
        })
        groups.append({
            "operation": "relu",
            "shape": label,
            "baseline": lambda x=x: np.maximum(x, 0.0),
            "implementations": [
                ("cpp raw buffer",
                 lambda x=x: cpp.relu(x),
                 lambda x=x: np.maximum(x, 0.0)),
                ("tensor core",
                 lambda cx=cx: cx.relu(),
                 lambda x=x: np.maximum(x, 0.0)),
                ("tensor core (view)",
                 lambda cx=cx: cx.T.relu(),
                 lambda x=x: np.maximum(x.T, 0.0)),
            ],
        })

    for n in matmul_sizes:
        m1 = rng.normal(size=(n, n))
        m2 = rng.normal(size=(n, n))
        cm1, cm2 = core.from_array(m1), core.from_array(m2)
        groups.append({
            "operation": "matmul",
            "shape": f"({n}x{n}) @ ({n}x{n})",
            "baseline": lambda m1=m1, m2=m2: m1 @ m2,
            "implementations": [
                ("cpp raw naive",
                 lambda m1=m1, m2=m2: cpp.matmul(m1, m2),
                 lambda m1=m1, m2=m2: m1 @ m2),
                ("cpp raw tiled",
                 lambda m1=m1, m2=m2: cpp.matmul_tiled(m1, m2),
                 lambda m1=m1, m2=m2: m1 @ m2),
                ("tensor core",
                 lambda cm1=cm1, cm2=cm2: cm1.matmul(cm2),
                 lambda m1=m1, m2=m2: m1 @ m2),
                ("tensor core (T view)",
                 lambda cm1=cm1, cm2=cm2: cm1.T.matmul(cm2),
                 lambda m1=m1, m2=m2: m1.T @ m2),
            ],
        })

    return groups


def run_suite(quick=False, cpp=None):
    """Run every group; return result rows. Each group produces one
    ``numpy`` baseline row (ratio 1.0) plus one row per implementation
    with its time and its ratio against that baseline. Correctness is
    verified against each implementation's own reference before any
    timing — a fast wrong kernel is not interesting."""
    if cpp is None:
        cpp = _load_backend()
    repeats = 5 if quick else 20

    rows = []
    for group in build_suite(cpp, quick):
        baseline_fn = group["baseline"]
        baseline_time = measure(baseline_fn, repeats)
        rows.append({
            "operation": group["operation"],
            "shape": group["shape"],
            "implementation": "numpy",
            "time_s": baseline_time,
            "ratio": 1.0,
        })
        for name, timed_fn, reference_fn in group["implementations"]:
            if not np.allclose(_as_numpy(timed_fn()), reference_fn(), atol=1e-10):
                raise AssertionError(
                    f"{group['operation']} {group['shape']} [{name}]: "
                    f"result disagrees with the reference"
                )
            impl_time = measure(timed_fn, repeats)
            rows.append({
                "operation": group["operation"],
                "shape": group["shape"],
                "implementation": name,
                "time_s": impl_time,
                "ratio": impl_time / baseline_time,
            })
    return rows


def print_report(rows):
    header = (
        f"{'operation':<10} {'shape':<20} {'implementation':<21} "
        f"{'time':>12} {'vs numpy':>10}"
    )
    print(header)
    print("-" * len(header))
    previous_group = None
    for row in rows:
        group = (row["operation"], row["shape"])
        if previous_group is not None and group != previous_group:
            print()
        previous_group = group
        print(
            f"{row['operation']:<10} {row['shape']:<20} "
            f"{row['implementation']:<21} {_format_time(row['time_s'])} "
            f"{row['ratio']:>9.1f}x"
        )
    print()
    print("ratio > 1 means slower than NumPy. That is expected: the C++")
    print("kernels are naive, single-threaded reference loops, while")
    print("NumPy uses optimized C and BLAS. 'tensor core' rows include")
    print("wrapper, output allocation, and strided-traversal overhead;")
    print("small arrays also pay ctypes call + conversion overhead.")
    print("Results are hardware-dependent.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--quick", action="store_true",
        help="small sizes and few repeats, for smoke checks",
    )
    args = parser.parse_args()
    print_report(run_suite(quick=args.quick))


if __name__ == "__main__":
    main()
