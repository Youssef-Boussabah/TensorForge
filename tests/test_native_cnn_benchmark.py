"""Tests for the native CNN benchmark harness (Phase D, milestone D12).

These validate the benchmark's *behavior* — registration, schema, mode
applicability, correctness gating, and the CLI — with **no timing or
threshold assertions of any kind**: benchmark durations are
hardware-specific measurements, never pass/fail criteria. Importing the
module must not run anything.

Selector: python -m pytest -q -k native_cnn_benchmark
"""

import json

import pytest

from tensorforge.backends import cpp
from benchmarks import benchmark_native_cnn as bench

needs_native = pytest.mark.skipif(
    not cpp.is_available(),
    reason="experimental C++ backend not built; " + cpp.build_instructions(),
)

SMOKE = {"warmup": 1, "iterations": 1, "repeats": 2, "smoke": True}


def test_registries_cover_the_required_phase_d_surface():
    # The milestone requires conv forward, conv forward+backward, pooling
    # forward, pooling forward+backward, an end-to-end training step, and an
    # optional stable-framework reference.
    assert set(bench.CASES) == {"conv2d", "maxpool2d", "cnn"}
    assert bench.MODES == (
        "forward_native", "forward_graph", "forward_backward_fresh",
        "training_step", "stable_forward",
    )
    for name in ("conv2d", "maxpool2d"):
        modes = bench.CASES[name]["modes"]
        assert "forward_native" in modes and "forward_backward_fresh" in modes
        assert "stable_forward" in modes
        assert "training_step" not in modes      # only the end-to-end case
    assert "training_step" in bench.CASES["cnn"]["modes"]


def test_every_case_declares_deterministic_smoke_and_full_shapes():
    for name, spec in bench.CASES.items():
        assert set(spec["shapes"]) == {"full", "smoke"}, name
        for variant, shape in spec["shapes"].items():
            assert shape, (name, variant)
            assert all(isinstance(v, int) and v > 0 for v in shape.values())
        # Smoke shapes must be no larger than the full ones.
        for key, value in spec["shapes"]["smoke"].items():
            assert value <= spec["shapes"]["full"][key], (name, key)


@needs_native
def test_smoke_run_produces_the_expected_schema():
    payload = bench.run_benchmark(**SMOKE)
    meta = payload["metadata"]
    assert meta["benchmark"] == "native_cnn"
    assert meta["dtype"] == "float64" and meta["device"] == "cpu"
    assert meta["smoke"] is True
    assert set(meta["cases"]) == set(bench.CASES)
    for record in payload["results"]:
        assert record["case"] in bench.CASES
        assert record["mode"] in bench.CASES[record["case"]]["modes"]
        assert record["units"] == "seconds_per_iteration"
        assert len(record["samples_s"]) == SMOKE["repeats"]
        assert record["min_s"] <= record["median_s"] <= record["max_s"]
        assert all(sample > 0 for sample in record["samples_s"])
    # Every required measurement is present.
    produced = {(r["case"], r["mode"]) for r in payload["results"]}
    for required in (
        ("conv2d", "forward_native"), ("conv2d", "forward_backward_fresh"),
        ("maxpool2d", "forward_native"),
        ("maxpool2d", "forward_backward_fresh"),
        ("cnn", "training_step"), ("cnn", "stable_forward"),
    ):
        assert required in produced, required


@needs_native
def test_single_case_and_single_mode_selection():
    payload = bench.run_benchmark(cases=["maxpool2d"], **SMOKE)
    assert {r["case"] for r in payload["results"]} == {"maxpool2d"}
    payload = bench.run_benchmark(modes=["training_step"], **SMOKE)
    # Only the end-to-end case declares that mode; the others are skipped
    # rather than failing.
    assert {(r["case"], r["mode"]) for r in payload["results"]} == {
        ("cnn", "training_step")
    }


def test_unknown_case_or_mode_is_rejected():
    with pytest.raises(ValueError, match="unknown case"):
        bench.run_benchmark(cases=["nope"], **SMOKE)
    with pytest.raises(ValueError, match="unknown mode"):
        bench.run_benchmark(modes=["nope"], **SMOKE)


@pytest.mark.parametrize("bad", [0, -1, True, 1.5, "3"])
def test_non_positive_counts_are_rejected(bad):
    with pytest.raises(ValueError):
        bench.run_benchmark(warmup=bad, smoke=True)


@needs_native
def test_report_and_json_are_well_formed():
    payload = bench.run_benchmark(cases=["maxpool2d"], **SMOKE)
    report = bench.format_report(payload)
    assert "TensorForge native CNN benchmark" in report
    assert "maxpool2d" in report
    # The report must carry no speed verdict.
    lowered = report.lower()
    for banned in ("faster", "speedup", "pytorch", "gpu", "production"):
        assert banned not in lowered, banned
    assert "not a speed claim" in lowered
    assert json.loads(json.dumps(payload))["metadata"]["benchmark"] == "native_cnn"


@needs_native
def test_cli_smoke_and_json(capsys):
    bench.main(["--smoke", "--case", "conv2d", "--iterations", "1",
                "--repeats", "2", "--warmup", "1"])
    assert "conv2d" in capsys.readouterr().out
    bench.main(["--smoke", "--case", "conv2d", "--iterations", "1",
                "--repeats", "2", "--warmup", "1", "--json"])
    payload = json.loads(capsys.readouterr().out)
    assert payload["metadata"]["cases"] == ["conv2d"]


def test_cli_rejects_unknown_selections():
    with pytest.raises(SystemExit):
        bench.main(["--case", "nope"])


@needs_native
def test_correctness_gate_rejects_a_broken_measurement(monkeypatch):
    # The gate is real: a forward that returns a non-finite result must
    # fail before any timing happens.
    import numpy as np

    from tensorforge.experimental import NativeTensor

    original = bench.CASES["maxpool2d"]["forward"]

    def broken(inp):
        result = original(inp)
        result.close()
        return NativeTensor.from_array(np.array([np.inf]))

    monkeypatch.setitem(bench.CASES["maxpool2d"], "forward", broken)
    with pytest.raises(AssertionError, match="not finite"):
        bench.run_benchmark(cases=["maxpool2d"], modes=["forward_native"],
                            **SMOKE)


def test_importing_the_module_runs_nothing(capsys):
    import importlib

    importlib.reload(bench)
    assert capsys.readouterr().out == ""
