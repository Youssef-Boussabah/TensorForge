"""Lightweight tests for the backend benchmark helpers.

No timing assertions — benchmark durations are machine-dependent and
would flake. These tests only prove the script's logic runs and its
results have the right shape. They skip when the backend is unbuilt,
like the other backend tests.
"""

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from benchmarks.cpp_backend import build_cases, measure, run_benchmarks

try:
    from tensorforge.backends import cpp
    _IMPORT_ERROR = None
except ImportError as error:
    cpp = None
    _IMPORT_ERROR = str(error)

needs_backend = pytest.mark.skipif(
    cpp is None,
    reason=f"experimental C++ backend not built: {_IMPORT_ERROR}",
)


def test_measure_times_the_function():
    calls = []
    result = measure(lambda: calls.append(1), repeats=3, warmup=2)
    assert isinstance(result, float)
    assert result >= 0.0
    assert len(calls) == 5  # warmup runs + timed runs


@needs_backend
def test_quick_benchmarks_produce_result_rows():
    rows = run_benchmarks(quick=True, cpp=cpp)
    assert len(rows) == len(build_cases(cpp, quick=True))
    assert {row["operation"] for row in rows} == {"elementwise_add", "relu", "matmul"}
    for row in rows:
        assert set(row) == {"operation", "shape", "numpy_s", "cpp_s", "ratio"}
        assert row["numpy_s"] > 0.0
        assert row["cpp_s"] > 0.0
        assert row["ratio"] > 0.0  # no assertion about who wins — that's the point
