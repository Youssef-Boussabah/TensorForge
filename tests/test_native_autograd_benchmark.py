"""Tests for the native autograd benchmark harness (Advanced C++ v2.5).

These validate the benchmark's *behavior* — registration, schema, CLI,
metadata, and that the correctness gate runs — never its speed. There are
no timing/threshold assertions of any kind: benchmark durations are
machine-dependent and any expectation about who wins (or a fixed time
budget) would flake and miss the point (the harness measures, it does not
advocate). Tests skip when the backend is unbuilt, like the other native
tests. Importing the benchmark module must never execute a benchmark.
"""

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from benchmarks.benchmark_native_autograd import (  # noqa: E402
    CASES,
    MODES,
    _verify,
    build_parser,
    format_report,
    main,
    run_benchmark,
)

from tensorforge.backends import cpp  # noqa: E402  (importing never raises)

needs_backend = pytest.mark.skipif(
    not cpp.is_available(),
    reason="experimental C++ backend not built; " + cpp.build_instructions(),
)

REQUIRED_CASES = {"elementwise", "broadcast", "reduction", "matmul", "view_chain"}
REQUIRED_MODES = {
    "forward_native",
    "forward_graph",
    "forward_backward_fresh",
    "backward_retained",
}
RECORD_KEYS = {
    "case", "mode", "shape", "warmup", "iterations", "repeats",
    "samples_s", "median_s", "min_s", "max_s", "iters_per_s", "units",
}
META_KEYS = {
    "benchmark", "version", "timestamp", "platform", "machine", "processor",
    "python_version", "native_backend", "dtype", "device", "warmup",
    "iterations", "repeats", "smoke", "cases", "modes", "shapes",
}


# -- registration ------------------------------------------------------


def test_native_autograd_benchmark_required_cases_registered():
    assert REQUIRED_CASES <= set(CASES)


def test_native_autograd_benchmark_required_modes_registered():
    assert set(MODES) == REQUIRED_MODES


def test_native_autograd_benchmark_import_does_not_run():
    # Importing the module (done at collection time above) must not have
    # executed a benchmark; the registries are plain data.
    assert callable(run_benchmark)
    assert all("forward" in m or "backward" in m for m in MODES)


# -- smoke run + schema ------------------------------------------------


@needs_backend
def test_native_autograd_benchmark_smoke_runs():
    payload = run_benchmark(smoke=True)
    assert set(payload) == {"metadata", "results"}
    assert payload["results"], "smoke run produced no records"
    # all cases x all modes
    assert len(payload["results"]) == len(CASES) * len(MODES)


@needs_backend
def test_native_autograd_benchmark_record_fields():
    for rec in run_benchmark(smoke=True)["results"]:
        assert set(rec) == RECORD_KEYS
        assert rec["units"] == "seconds_per_iteration"
        assert rec["case"] in CASES
        assert rec["mode"] in MODES


@needs_backend
def test_native_autograd_benchmark_metadata_fields():
    meta = run_benchmark(smoke=True)["metadata"]
    assert META_KEYS <= set(meta)
    assert meta["benchmark"] == "native_autograd"
    assert meta["dtype"] == "float64"
    assert meta["device"] == "cpu"
    assert meta["smoke"] is True
    assert set(meta["cases"]) == set(CASES)
    assert set(meta["modes"]) == set(MODES)
    assert set(meta["shapes"]) == set(CASES)


@needs_backend
def test_native_autograd_benchmark_sample_count_matches_repeats():
    payload = run_benchmark(smoke=True, repeats=4)
    for rec in payload["results"]:
        assert rec["repeats"] == 4
        assert len(rec["samples_s"]) == 4


@needs_backend
def test_native_autograd_benchmark_timings_positive():
    for rec in run_benchmark(smoke=True)["results"]:
        assert all(sample > 0.0 for sample in rec["samples_s"])
        assert rec["median_s"] > 0.0
        assert rec["min_s"] > 0.0
        assert rec["max_s"] > 0.0
        assert rec["iters_per_s"] > 0.0


@needs_backend
def test_native_autograd_benchmark_median_between_min_and_max():
    for rec in run_benchmark(smoke=True)["results"]:
        assert rec["min_s"] <= rec["median_s"] <= rec["max_s"]


# -- selection ---------------------------------------------------------


@needs_backend
def test_native_autograd_benchmark_single_case_runs():
    payload = run_benchmark(cases=["matmul"], smoke=True)
    assert {rec["case"] for rec in payload["results"]} == {"matmul"}
    assert {rec["mode"] for rec in payload["results"]} == set(MODES)


@needs_backend
def test_native_autograd_benchmark_single_mode_runs():
    payload = run_benchmark(modes=["forward_native"], smoke=True)
    assert {rec["mode"] for rec in payload["results"]} == {"forward_native"}
    assert {rec["case"] for rec in payload["results"]} == set(CASES)


# -- validation --------------------------------------------------------


@needs_backend
def test_native_autograd_benchmark_rejects_non_positive_counts():
    for kwargs in ({"warmup": 0}, {"iterations": 0}, {"repeats": 0},
                   {"iterations": -3}, {"repeats": -1}):
        with pytest.raises(ValueError):
            run_benchmark(smoke=True, **kwargs)


@needs_backend
def test_native_autograd_benchmark_rejects_unknown_case():
    with pytest.raises(ValueError, match="unknown case"):
        run_benchmark(cases=["nope"], smoke=True)


@needs_backend
def test_native_autograd_benchmark_rejects_unknown_mode():
    with pytest.raises(ValueError, match="unknown mode"):
        run_benchmark(modes=["nope"], smoke=True)


# -- JSON + human-readable output --------------------------------------


@needs_backend
def test_native_autograd_benchmark_json_parses(capsys):
    main(["--smoke", "--json", "--case", "reduction"])
    out = capsys.readouterr().out
    payload = json.loads(out)  # pure JSON, no human text mixed in
    assert payload["metadata"]["benchmark"] == "native_autograd"
    assert {rec["case"] for rec in payload["results"]} == {"reduction"}


@needs_backend
def test_native_autograd_benchmark_report_contains_case_and_mode_names():
    text = format_report(run_benchmark(smoke=True))
    for case in CASES:
        assert case in text
    for mode in MODES:
        assert mode in text


@needs_backend
def test_native_autograd_benchmark_cli_human_readable(capsys):
    main(["--smoke", "--case", "elementwise", "--mode", "forward_native"])
    out = capsys.readouterr().out
    assert "elementwise" in out
    assert "forward_native" in out


@needs_backend
def test_native_autograd_benchmark_cli_rejects_bad_iterations():
    with pytest.raises(SystemExit):
        main(["--smoke", "--iterations", "0"])


# -- correctness gate + no speed verdict -------------------------------


@needs_backend
def test_native_autograd_benchmark_correctness_gate_runs_in_smoke():
    # The backward modes' records only exist if the gate (which checks each
    # leaf gradient exists, has the right shape, and is finite) passed while
    # running them in smoke mode.
    payload = run_benchmark(
        modes=["forward_backward_fresh", "backward_retained"], smoke=True
    )
    modes_seen = {rec["mode"] for rec in payload["results"]}
    assert modes_seen == {"forward_backward_fresh", "backward_retained"}


@needs_backend
def test_native_autograd_benchmark_correctness_gate_rejects_bad_output():
    # Direct check that the gate is real: a non-scalar output is rejected.
    from tensorforge.experimental import NativeTensor

    out = NativeTensor.from_array([1.0, 2.0])  # shape (2,), not the scalar ()
    with pytest.raises(AssertionError):
        _verify("elementwise", "forward_native", out, [])
    out.close()


@needs_backend
def test_native_autograd_benchmark_records_carry_no_speed_verdict():
    # No pass/fail on speed anywhere: records hold only timing data and
    # schema fields (no "passed"/"verdict"/"faster" field).
    for rec in run_benchmark(smoke=True)["results"]:
        assert set(rec) == RECORD_KEYS
        for forbidden in ("passed", "verdict", "faster", "ok", "threshold"):
            assert forbidden not in rec


@needs_backend
def test_native_autograd_benchmark_does_not_change_autograd_semantics():
    from tensorforge.experimental import NativeTensor

    run_benchmark(smoke=True)  # run the whole harness first

    # NativeTensor autograd still behaves exactly as v2.4 defined it:
    x = NativeTensor.from_array([1.0, 2.0, 3.0], requires_grad=True)
    out = x.multiply(x).sum()
    out.backward()  # default retain_graph=False -> one-shot free
    assert pytest.approx([2.0, 4.0, 6.0]) == list(x.grad.to_numpy())
    with pytest.raises(RuntimeError, match="freed"):
        out.backward()  # graph was released, exactly as before
    x.close()


# -- parser ------------------------------------------------------------


def test_native_autograd_benchmark_parser_has_expected_options():
    parser = build_parser()
    options = {action.dest for action in parser._actions}
    for name in ("case", "mode", "warmup", "iterations", "repeats", "json", "smoke"):
        assert name in options
