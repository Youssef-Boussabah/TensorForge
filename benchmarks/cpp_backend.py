"""Honest benchmarks: the experimental C++ backend vs NumPy.

This script exists to teach, not to win. The C++ kernels are naive,
single-threaded, textbook loops; NumPy dispatches to heavily optimized
C — and, for matmul, to a BLAS library. Expect NumPy to be faster,
often dramatically so for matmul. What the numbers show:

- per-call overhead (ctypes + conversion) dominates on small arrays;
- on large elementwise arrays the naive loop can get within range of
  NumPy, because both are ultimately memory-bound;
- for matmul, BLAS's blocking/SIMD/threading crushes the triple loop.

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


def build_cases(cpp, quick):
    """Return (name, shape_label, numpy_fn, cpp_fn) benchmark cases."""
    rng = np.random.default_rng(0)
    small_n = 1_000
    large_n = 100_000 if quick else 1_000_000
    mat_n = 48 if quick else 192

    cases = []
    for n in (small_n, large_n):
        a = rng.normal(size=n)
        b = rng.normal(size=n)
        cases.append((
            "elementwise_add", f"({n:,})",
            lambda a=a, b=b: a + b,
            lambda a=a, b=b: cpp.elementwise_add(a, b),
        ))
    for n in (small_n, large_n):
        x = rng.normal(size=n)
        cases.append((
            "relu", f"({n:,})",
            lambda x=x: np.maximum(x, 0.0),
            lambda x=x: cpp.relu(x),
        ))
    for n in (mat_n // 2, mat_n):
        m1 = rng.normal(size=(n, n))
        m2 = rng.normal(size=(n, n))
        cases.append((
            "matmul", f"({n}x{n}) @ ({n}x{n})",
            lambda m1=m1, m2=m2: m1 @ m2,
            lambda m1=m1, m2=m2: cpp.matmul(m1, m2),
        ))
    return cases


def run_benchmarks(quick=False, cpp=None):
    """Run all cases; return a list of result-row dicts."""
    if cpp is None:
        cpp = _load_backend()
    repeats = 5 if quick else 20

    rows = []
    for name, shape, numpy_fn, cpp_fn in build_cases(cpp, quick):
        # Correctness first: a fast wrong kernel is not interesting.
        if not np.allclose(numpy_fn(), cpp_fn(), atol=1e-10):
            raise AssertionError(f"{name} {shape}: C++ result disagrees with NumPy")
        numpy_time = measure(numpy_fn, repeats)
        cpp_time = measure(cpp_fn, repeats)
        rows.append({
            "operation": name,
            "shape": shape,
            "numpy_s": numpy_time,
            "cpp_s": cpp_time,
            "ratio": cpp_time / numpy_time,
        })
    return rows


def print_report(rows):
    header = f"{'operation':<17} {'shape':<20} {'numpy':>12} {'c++':>12} {'c++/numpy':>10}"
    print(header)
    print("-" * len(header))
    for row in rows:
        print(
            f"{row['operation']:<17} {row['shape']:<20} "
            f"{_format_time(row['numpy_s'])} {_format_time(row['cpp_s'])} "
            f"{row['ratio']:>9.1f}x"
        )
    print()
    print("ratio > 1 means the C++ backend is slower than NumPy.")
    print("That is expected: these kernels are naive, educational,")
    print("single-threaded loops, while NumPy uses optimized C and BLAS.")
    print("Small arrays also pay ctypes call + conversion overhead.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--quick", action="store_true",
        help="small sizes and few repeats, for smoke checks",
    )
    args = parser.parse_args()
    print_report(run_benchmarks(quick=args.quick))


if __name__ == "__main__":
    main()
