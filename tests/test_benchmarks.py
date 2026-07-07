"""Lightweight tests for the backend benchmark suite helpers.

No timing assertions and no speed assumptions — benchmark durations
are machine-dependent and any expectation about who wins would flake
(and would also miss the point: the suite measures, it doesn't
advocate). These tests only prove the plan and result structure. They
skip when the backend is unbuilt, like the other backend tests.
Importing the benchmark module must never execute benchmarks.
"""

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from benchmarks.cpp_backend import _format_time, build_suite, measure, run_suite

from tensorforge.backends import cpp  # importing never raises (lazy load)

needs_backend = pytest.mark.skipif(
    not cpp.is_available(),
    reason="experimental C++ backend not built; " + cpp.build_instructions(),
)

ROW_KEYS = {"operation", "shape", "implementation", "time_s", "ratio"}


def test_measure_times_the_function():
    calls = []
    result = measure(lambda: calls.append(1), repeats=3, warmup=2)
    assert isinstance(result, float)
    assert result >= 0.0
    assert len(calls) == 5  # warmup runs + timed runs


def test_format_time_is_readable():
    assert "us" in _format_time(5e-6)
    assert "ms" in _format_time(5e-3)
    assert "s" in _format_time(5.0)


@needs_backend
def test_quick_plan_covers_all_operations_and_tensor_core():
    groups = build_suite(cpp, quick=True)
    assert {group["operation"] for group in groups} == {"add", "relu", "matmul"}
    for group in groups:
        assert callable(group["baseline"])
        names = [name for name, _, _ in group["implementations"]]
        assert "cpp raw buffer" in names or "cpp raw naive" in names
        assert any("tensor core" in name for name in names)
        assert any("view" in name.lower() for name in names)  # non-contiguous case
    matmul_names = [
        name
        for group in groups
        if group["operation"] == "matmul"
        for name, _, _ in group["implementations"]
    ]
    assert "cpp raw tiled" in matmul_names


@needs_backend
def test_quick_suite_rows_are_structured_consistently():
    rows = run_suite(quick=True, cpp=cpp)
    assert rows, "the suite produced no rows"
    implementations = set()
    for row in rows:
        assert set(row) == ROW_KEYS
        assert row["time_s"] > 0.0
        assert row["ratio"] > 0.0  # deliberately no assumption about who wins
        implementations.add(row["implementation"])
    assert "numpy" in implementations
    assert "tensor core" in implementations
    # Every group leads with a numpy baseline row at ratio exactly 1.
    for row in rows:
        if row["implementation"] == "numpy":
            assert row["ratio"] == pytest.approx(1.0)


@needs_backend
def test_every_group_has_a_numpy_row():
    rows = run_suite(quick=True, cpp=cpp)
    groups = {(row["operation"], row["shape"]) for row in rows}
    numpy_groups = {
        (row["operation"], row["shape"])
        for row in rows
        if row["implementation"] == "numpy"
    }
    assert groups == numpy_groups
