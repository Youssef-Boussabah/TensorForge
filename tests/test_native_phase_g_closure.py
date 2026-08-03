"""Phase-G closure guardrails (Advanced C++ milestone G10).

The durable replacement for Phase G's milestone-era *absence* checks. Every
earlier guard was written in the negative — "no surface claims Dropout is
supported", "``dropout`` is still in ``UNSUPPORTED``", "G10 has not begun" —
because none of those things had happened yet. G10 ran the closure matrix
in docs/native_rng_dropout_design.md §18 and moved the boundary, so those
premises have all expired. What replaces them is this file: the positive,
post-closure form of the same contract, described in §21 of the design.

The rules these tests protect:

* the phase is closed, and its ladder is whole (G0-G10, once each, in
  order, all complete, with nothing later claimed);
* the final capability tuple is **exact**, and no name appears in both an
  implemented inventory and ``UNSUPPORTED``;
* each Dropout name sits in exactly one layer-appropriate inventory,
  exactly once;
* the checkpoint contract is pinned (format name, version 2, versions
  ``(1, 2)``, both generator state-support names);
* every Phase-G deliverable still exists;
* **the claim stays narrow** — experimental native float64 CPU only, never
  the stable framework, float32, CUDA, AMP, a generic RNG API,
  ``Dropout2d``/``Dropout3d``, production readiness, or universal speed;
* the closure evidence is recorded rather than merely asserted; and
* no machine-specific artifact is committed.

They deliberately test *values and structure* rather than wording, so
ordinary prose improvements do not require rewriting them.
"""
import json
import re
from pathlib import Path

import pytest

import tensorforge
from tensorforge.backends import cpp

REPO_ROOT = Path(__file__).resolve().parent.parent
PHASE_G_DESIGN = REPO_ROOT / "docs" / "native_rng_dropout_design.md"

# The surfaces that state *current* capability status. Per-milestone
# historical records elsewhere deliberately preserve superseded wording and
# are not scanned.
STATUS_SURFACES = (
    "README.md",
    "CLAUDE.md",
    "docs/roadmap.md",
    "docs/project_summary.md",
    "docs/native_support_matrix.md",
    "docs/backend_experiments.md",
    "docs/architecture.md",
)

# The boundary **Phase G's closure left**, written out once so a drift in
# either direction is a single obvious diff rather than a scatter of edits.
# This is a historical record and does not move again: it is what G10
# produced, and it is asserted below as a *difference* from the live tuple
# rather than as the live tuple, because a later phase legitimately moved
# one of these names.
G10_UNSUPPORTED = ("float32", "cuda", "amp")

# What ``"float32"`` did after Phase G closed, and where. Phase I milestone
# I9 moved it into ``SUPPORTED_DTYPES`` once integrated float32 training and
# the exact float32 resume proof both passed. That is a Phase-I event, and
# recording it here is what keeps this file's Phase-G claims true instead of
# quietly rewriting them.
MOVED_AFTER_PHASE_G = {"float32"}
CURRENT_UNSUPPORTED = tuple(name for name in G10_UNSUPPORTED
                            if name not in MOVED_AFTER_PHASE_G)


def _read(relative):
    return (REPO_ROOT / relative).read_text(encoding="utf-8")


def _design_section(heading):
    """The design document's top-level section under ``heading``."""
    text = PHASE_G_DESIGN.read_text(encoding="utf-8")
    for chunk in re.split(r"\n## ", text):
        if chunk.split("\n", 1)[0].strip().lower().startswith(
            heading.lower()
        ):
            return chunk
    raise AssertionError(f"the Phase-G design has no {heading!r} section")


def _ladder():
    return _design_section("19. Milestone ladder")


# ==========================================================================
# 1. The phase is closed, and the ladder is whole
# ==========================================================================

def test_the_milestone_ladder_runs_g0_to_g10_once_each_in_order():
    """§21.1. Every milestone appears exactly once, in order, and the
    ladder stops at G10 — no Phase-G milestone beyond the closure."""
    rows = re.findall(r"^\|\s*G(\d+)\s*\|", _ladder(), re.M)
    assert rows == [str(index) for index in range(11)], rows


def test_every_ladder_row_is_marked_complete():
    """§21.1. A closed phase has no open milestone left."""
    ladder = _ladder()
    for index in range(11):
        row = re.search(rf"^\|\s*G{index}\s*\|[^|]*\|([^|]*)\|",
                        ladder, re.M)
        assert row is not None, index
        status = re.sub(r"[*`]", "", row.group(1)).strip().lower()
        assert status.startswith("complete"), (index, status)
        assert "not started" not in status, index


def test_no_surface_claims_a_phase_g_milestone_beyond_g10():
    """§21.1. G10 is the last **Phase-G** milestone. Nothing may claim a
    G11 exists or has begun, because none does.

    Phase H is a different matter and is deliberately not banned here:
    it opened later, at milestone H0, and naming it is now accurate. What
    Phase H may not claim is that it delivered a capability, which the H0
    guardrails check against the live registry — a stronger check than a
    phase-name scan, and not this test's subject.

    Mentions carrying their own negation — the design's own "no ``G11``
    or later Phase-G milestone is claimed anywhere" — are the honest form
    and pass, so the guard bans the *claim*, not the token."""
    negations = re.compile(
        r"\b(no|not|never|none|neither|nor|without|absent|beyond|future"
        r"|would|will|if)\b", re.I,
    )
    subjects = re.compile(r"\bG1[1-9]\b")
    for surface in STATUS_SURFACES + ("docs/native_rng_dropout_design.md",):
        text = _read(surface)
        offenders = [
            match.group(0) for match in subjects.finditer(text)
            if not negations.search(
                text[max(0, match.start() - 80):match.end() + 40]
            )
        ]
        assert offenders == [], (surface, offenders[:3])

    # ...and the ladder itself has no row beyond G10, negation or not.
    assert not re.search(r"^\|\s*G1[1-9]\s*\|", _ladder(), re.M)


def test_the_design_records_the_closure_and_its_ordering_rule():
    """§21.8. The design keeps *both* halves: the rule that the boundary
    moved last, and the observed results that justified moving it."""
    # Whitespace-normalized: these phrases wrap across lines in the
    # source, and a reflow must not fail the test.
    closure = re.sub(r"\s+", " ",
                     _design_section("18. Phase-closure requirements")).lower()
    # The ordering rule that governed the move.
    assert "gate on the capability boundary" in closure
    assert "only when" in closure or "closes the phase only" in closure
    # ...and the evidence, by subject rather than by wording.
    for evidence in ("release", "debug", "ctest", "asan", "ubsan",
                     "leaksanitizer", "baseline", "suppression"):
        assert evidence in closure, evidence


# ==========================================================================
# 2. The final capability boundary
# ==========================================================================

def test_the_unsupported_tuple_is_phase_gs_minus_what_a_later_phase_moved():
    """§21.2. The single most important assertion in this file, stated so
    it stays true as the project continues: the live tuple is exactly what
    Phase G's closure left **minus the names a later phase legitimately
    moved**, in order, with nothing added and ``dropout`` gone.

    Written as a difference rather than as a literal because the two
    claims are different and both matter. "G10 left
    ``("float32", "cuda", "amp")``" is history and must not be rewritten;
    "the runtime today lists ``("cuda", "amp")``" is current truth. Pinning
    only the second would lose the Phase-G record; pinning only the first
    would have failed the moment I9 landed, which is the failure this
    split turns into a documented fact."""
    assert cpp.UNSUPPORTED == CURRENT_UNSUPPORTED
    assert cpp.backend_info()["unsupported"] == CURRENT_UNSUPPORTED
    assert "dropout" not in cpp.UNSUPPORTED
    # Nothing was *added* to the boundary after Phase G — every live name
    # is one G10 already listed.
    assert set(cpp.UNSUPPORTED) <= set(G10_UNSUPPORTED)
    assert set(G10_UNSUPPORTED) - set(cpp.UNSUPPORTED) == MOVED_AFTER_PHASE_G
    # Each remaining name is there exactly once and is genuinely absent.
    for name in CURRENT_UNSUPPORTED:
        assert cpp.UNSUPPORTED.count(name) == 1, name


def test_phase_g_moved_one_capability_name_and_no_dtype_or_device():
    """§21.2. **Phase G's** closure moved ``dropout`` and nothing else —
    in particular it moved no dtype and no device, which is why
    ``"float32"`` was still listed unsupported when this file was written.

    The dtype registry moved later, at Phase I milestone **I9**, once
    integrated float32 training and the exact float32 resume proof both
    passed. Asserting the attribution rather than the old literal is what
    keeps the Phase-G claim honest: the device set really did not move, and
    the default really is still float64."""
    assert "dropout" not in cpp.UNSUPPORTED
    assert cpp.SUPPORTED_DEVICES == ("cpu",)
    # float64 was Phase G's only dtype and is still the *default*; float32
    # joined it at I9 and is an addition, never a replacement.
    assert cpp.SUPPORTED_DTYPES[0] == "float64"
    assert set(cpp.SUPPORTED_DTYPES) == {"float64", "float32"}
    info = cpp.backend_info()
    assert info["supported_dtypes"] == cpp.SUPPORTED_DTYPES
    assert info["supported_devices"] == ("cpu",)
    assert info["dtype"] == "float64"
    assert info["device"] == "cpu"


def test_no_name_is_both_unsupported_and_implemented():
    """§21.3. The deliberate G3-G9 ``dropout`` overlap is gone, and a new
    one would be a regression rather than a repeat of that decision."""
    implemented = (
        set(cpp.RAW_KERNELS)
        | set(cpp.TENSOR_CORE_KERNELS)
        | set(cpp.TENSOR_CORE_OPS)
        | set(cpp.AUTOGRAD_OPS)
        | set(cpp.NATIVE_MODULES)
        | set(cpp.NATIVE_LOSSES)
        | set(cpp.NATIVE_METRICS)
        | set(cpp.NATIVE_OPTIMIZERS)
        | set(cpp.STATE_SUPPORT)
    )
    assert implemented & set(cpp.UNSUPPORTED) == set()


def test_the_unsupported_capabilities_really_are_unreachable():
    """§21.2, from behavior rather than from the tuple: asking for a
    still-unsupported dtype or device is refused.

    ``"float32"`` is deliberately **not** in this list any more. It was
    when Phase G closed, and it was reachable-and-refused for exactly the
    right reason then; Phase I milestone I9 made it real, so listing it
    here would now assert the opposite of the truth. The names below are
    the ones that remain genuinely absent, and ``float16``/``bfloat16``
    keep the "two dtypes, not any dtype" boundary honest."""
    from tensorforge.experimental import NativeTensor

    if not cpp.is_available():
        pytest.skip("backend not built")
    for dtype in ("float16", "bfloat16", "int64", "complex64"):
        with pytest.raises((ValueError, TypeError)):
            NativeTensor.zeros((2, 2), dtype=dtype)
    with pytest.raises((ValueError, TypeError)):
        NativeTensor.zeros((2, 2), device="cuda")
    # ...and the one that stopped being unsupported really works now.
    tensor = NativeTensor.zeros((2, 2), dtype="float32")
    try:
        assert tensor.dtype == "float32"
    finally:
        tensor.close()


# ==========================================================================
# 3. Each Dropout name in exactly one layer-appropriate inventory
# ==========================================================================

def test_each_dropout_name_appears_once_in_the_right_inventory():
    """§21.4. The three-way Core / operation / module split conv2d and
    maxpool2d established, applied to Dropout and pinned in both
    directions."""
    # The Core wrapper (G2), layer-qualified.
    assert cpp.TENSOR_CORE_OPS.count("dropout_forward") == 1
    assert "dropout_forward" not in cpp.AUTOGRAD_OPS
    assert "dropout_forward" not in cpp.NATIVE_MODULES
    # The differentiable operation (G3), bare.
    assert cpp.AUTOGRAD_OPS.count("dropout") == 1
    assert "dropout" not in cpp.TENSOR_CORE_OPS
    assert "dropout" not in cpp.RAW_KERNELS
    assert "dropout" not in cpp.TENSOR_CORE_KERNELS
    assert "dropout" not in cpp.NATIVE_MODULES
    # The module (G4).
    assert cpp.NATIVE_MODULES.count("NativeDropout") == 1
    for inventory in (cpp.AUTOGRAD_OPS, cpp.TENSOR_CORE_OPS,
                      cpp.RAW_KERNELS, cpp.NATIVE_LOSSES,
                      cpp.NATIVE_METRICS, cpp.NATIVE_OPTIMIZERS,
                      cpp.STATE_SUPPORT):
        assert "NativeDropout" not in inventory


def test_the_dropout_c_abi_surface_is_exactly_one_symbol():
    """§21.4. One guarded export, and deliberately **no** backward kernel:
    inverted Dropout's gradient is the existing ``multiply`` over the
    saved mask (design §7.5)."""
    checked = [name for name in cpp._CHECKED_KERNELS if "dropout" in name]
    assert checked == ["tf_core_dropout_forward"], checked
    assert "tf_core_dropout_backward" not in cpp._CHECKED_KERNELS
    assert "dropout_backward" not in cpp.TENSOR_CORE_OPS


@pytest.mark.skipif(not cpp.is_available(), reason="backend not built")
def test_every_advertised_dropout_name_maps_to_something_real():
    """§21.4, from reality rather than from the tuples."""
    from tensorforge.experimental import NativeDropout, NativeTensor

    assert hasattr(cpp.NativeTensorCore, "dropout_forward")
    assert hasattr(NativeTensor, "dropout")
    assert callable(NativeDropout)


# ==========================================================================
# 4. The checkpoint contract
# ==========================================================================

def test_the_checkpoint_format_and_versions_are_pinned():
    """§21.5. G5 moved the version to 2 and G10 moved nothing: the format
    *name* never changes, and version 1 stays loadable."""
    from tensorforge.experimental import native_checkpoint

    assert native_checkpoint._FORMAT == "tensorforge.native_checkpoint"
    assert native_checkpoint._FORMAT_VERSION == 3
    assert native_checkpoint._SUPPORTED_FORMAT_VERSIONS == (1, 2, 3)


def test_both_generator_state_capabilities_stay_reported():
    """§21.5. The in-memory surface (G1) and the file surface (G5) are two
    separate names on purpose, and both survive the closure."""
    assert "generator_state" in cpp.STATE_SUPPORT
    assert "checkpoint_generator_state" in cpp.STATE_SUPPORT
    assert cpp.STATE_SUPPORT.count("generator_state") == 1
    assert cpp.STATE_SUPPORT.count("checkpoint_generator_state") == 1
    assert cpp.backend_info()["state_support"] == cpp.STATE_SUPPORT


@pytest.mark.skipif(not cpp.is_available(), reason="backend not built")
def test_a_real_archive_still_declares_format_version_two(tmp_path):
    """§21.5, proved against an actual saved archive rather than against
    the constant — the closure must not have changed what is written."""
    import numpy as np

    from tensorforge.experimental import (
        NativeDropout, NativeSequential, save_native_checkpoint,
    )

    model = NativeSequential(NativeDropout(p=0.25, seed=7))
    path = tmp_path / "closure.npz"
    save_native_checkpoint(str(path), model)

    with np.load(str(path), allow_pickle=False) as archive:
        # The manifest crosses as a 1-D uint8 array of UTF-8 JSON.
        manifest = json.loads(
            archive["manifest"].tobytes().decode("utf-8")
        )
    assert manifest["format"] == "tensorforge.native_checkpoint"
    assert manifest["format_version"] == 3
    # The generator section exists and carries the alias topology, not
    # just the states.
    generators = manifest["generators"]
    assert generators is not None
    assert set(generators) == {"keys", "entries", "aliases"}
    assert generators["keys"] == ["0.generator"]
    # Every canonical name is self-mapped, so sharing is recoverable from
    # the archive rather than inferred.
    assert generators["aliases"] == {"0.generator": "0.generator"}
    assert set(generators["entries"]) == set(generators["keys"])
    entry = generators["entries"]["0.generator"]
    assert entry["algorithm"] == "tensorforge.splitmix64"
    assert entry["algorithm_version"] == 1
    # uint64 fields cross as canonical decimal strings, never as floats:
    # a value above 2**53 is not representable in an IEEE double.
    assert isinstance(entry["seed"], str) and entry["seed"] == "7"
    assert isinstance(entry["calls"], str) and entry["calls"] == "0"


# ==========================================================================
# 5. Every Phase-G deliverable still exists
# ==========================================================================

@pytest.mark.parametrize("relative", [
    "docs/native_rng_dropout_design.md",
    "src/tensorforge/experimental/native_generator.py",
    "src/tensorforge/experimental/native_dropout.py",
    "src/tensorforge/experimental/native_checkpoint.py",
    "src/tensorforge/experimental/_native_checkpoint_transaction.py",
    "src/tensorforge/experimental/_native_state_lock.py",
    "cpp/src/random.cpp",
    "cpp/include/tf_random_internal.h",
    "cpp/tests/test_dropout_forward.cpp",
    "examples/native_dropout_training.py",
    "benchmarks/benchmark_native_dropout.py",
    "tests/test_native_phase_g.py",
    "tests/test_native_phase_g_hardening.py",
])
def test_the_phase_g_deliverables_are_present(relative):
    """§21.6. The closure is meaningless if a deliverable it validated has
    since been deleted."""
    assert (REPO_ROOT / relative).is_file(), relative


def test_the_dropout_ctest_is_registered():
    """§21.6. The C++ known-answer test the closure ran 11/11 of is still
    wired into CTest."""
    cmake = _read("cpp/CMakeLists.txt")
    assert "add_test(NAME dropout_forward" in cmake
    assert "test_dropout_forward.cpp" in cmake


def test_the_benchmark_keeps_its_quick_mode_and_characterization_contract():
    """§21.6. G8 is measurement only, and must stay that way: a quick
    mode, no automatic result file, and no asserted speed."""
    source = _read("benchmarks/benchmark_native_dropout.py")
    assert "--quick" in source
    assert "--smoke" in source
    lowered = source.lower()
    # It says what it is...
    assert "characterization" in lowered
    # ...and never asserts a duration.
    assert not re.search(r"assert\s+[\w.\[\]\"']*\b(elapsed|duration|ns|"
                         r"seconds|median)\b[^\n]{0,40}[<>]", source), (
        "the benchmark asserts a timing threshold"
    )


def test_the_example_defines_no_public_training_api():
    """§21.6. G7 shipped a proof, not a framework surface."""
    import tensorforge.experimental as experimental

    for absent in ("run_training", "run_resume_proof", "train_step",
                   "build_model", "NativeDropoutClassifier"):
        assert not hasattr(experimental, absent), absent
        assert not hasattr(tensorforge, absent), absent


# ==========================================================================
# 6. The claim stays narrow
# ==========================================================================

def test_the_phase_g_exports_are_experimental_only():
    """§21.7. ``NativeGenerator`` and ``NativeDropout`` are reachable from
    ``tensorforge.experimental`` and from nowhere else — the root package
    is the stable framework and gained nothing."""
    import tensorforge.experimental as experimental

    for name in ("NativeGenerator", "NativeDropout"):
        assert name in experimental.__all__, name
        assert hasattr(experimental, name), name
        assert not hasattr(tensorforge, name), name
        assert name not in getattr(tensorforge, "__all__", ()), name


def test_the_stable_framework_keeps_its_own_separate_dropout():
    """§21.7. The narrow claim depends on this: stable ``Dropout`` is a
    different implementation that Phase G never touched, and the two
    lines share no object."""
    import tensorforge.nn as nn
    from tensorforge.experimental import NativeDropout

    assert hasattr(nn, "Dropout")
    assert hasattr(tensorforge, "Dropout")
    assert nn.Dropout is not NativeDropout
    assert not issubclass(NativeDropout, nn.Dropout)
    # Stable Dropout is pure NumPy and knows nothing about the backend.
    stable_source = _read("src/tensorforge/nn/dropout.py")
    for native in ("NativeGenerator", "NativeDropout", "backends.cpp",
                   "dropout_forward"):
        assert native not in stable_source, native
    # ...and the registry still says the two lines are unwired.
    assert cpp.backend_info()["stable_framework_integration"] is False


def test_no_generic_random_api_is_claimed_or_present():
    """§21.7. Phase G shipped one deterministic Dropout stream behind an
    explicit generator — never ``rand``/``randn``/sampling, and never a
    global stream or ``manual_seed``."""
    import tensorforge.experimental as experimental
    from tensorforge.experimental import NativeGenerator, NativeTensor

    for absent in ("rand", "randn", "randint", "bernoulli", "normal",
                   "uniform", "manual_seed", "seed", "default_generator",
                   "get_rng_state", "set_rng_state"):
        assert not hasattr(experimental, absent), absent
        assert not hasattr(NativeTensor, absent), absent
        assert absent not in cpp.AUTOGRAD_OPS, absent
        assert absent not in cpp.TENSOR_CORE_OPS, absent
        assert absent not in cpp.RAW_KERNELS, absent
    # The generator produces no value on its own; the derivation is in
    # C++ behind the Core.
    generator = NativeGenerator(seed=1)
    for absent in ("random", "next", "draw", "uniform", "bits"):
        assert not hasattr(generator, absent), absent


def test_no_extra_dropout_rank_is_claimed_or_present():
    """§21.7. ``Dropout2d``/``Dropout3d`` and stochastic depth are
    explicit non-goals and do not exist."""
    import tensorforge.experimental as experimental

    for absent in ("NativeDropout2d", "NativeDropout3d",
                   "NativeAlphaDropout", "NativeStochasticDepth"):
        assert not hasattr(experimental, absent), absent
        assert absent not in cpp.NATIVE_MODULES, absent
    for absent in ("dropout2d", "dropout3d", "stochastic_depth"):
        assert absent not in cpp.AUTOGRAD_OPS, absent
        assert absent not in cpp.TENSOR_CORE_OPS, absent


def test_no_status_surface_overclaims_the_boundary():
    """§21.7. The names still in ``UNSUPPORTED`` must not be described as
    supported anywhere, and no surface may claim the phase made the native
    line production-ready or universally faster.

    Spans carrying their own negation ("float16 is not supported") are the
    honest form and pass.

    **``float32`` left this scan at Phase I milestone I9**, on exactly the
    terms ``dropout`` left it at G10: it stopped being an over-claim
    because it stopped being absent. Phase G's boundary was
    ``("float32", "cuda", "amp")`` and that record is preserved above in
    ``G10_UNSUPPORTED``; the live tuple is ``("cuda", "amp")``, so banning
    "float32 is supported" would now force every status surface to
    under-report the runtime — the mirror of the failure this guards.
    ``float16``, ``bfloat16``, CUDA, GPU, AMP, and mixed precision stay,
    which is what keeps "two dtypes" from eroding into "any dtype"."""
    assert "float32" not in cpp.UNSUPPORTED, (
        "float32 is still unsupported, so it belongs in the scan below"
    )
    claim = re.compile(
        r"(float16|bfloat16|CUDA|GPU|AMP|mixed precision)"
        r"[^.]{0,60}\b(is|are|now)\s+"
        r"(supported|implemented|shipped|available)"
        r"|\b(production[- ]ready|production ready)\b"
        r"|(always|universally|consistently)\s+faster",
        re.I,
    )
    negations = re.compile(
        r"\b(not|never|no|none|neither|nor|un\w+|without|future|beyond"
        r"|remain\w*|stay\w*|still|planned|would|will)\b", re.I,
    )
    for surface in STATUS_SURFACES:
        text = _read(surface)
        offenders = [
            match.group(0) for match in claim.finditer(text)
            if not negations.search(
                text[max(0, match.start() - 70):match.end() + 30]
            )
        ]
        assert offenders == [], (surface, offenders[:3])
    # The negative control the I9 edit requires: removing a name from a
    # regex changes what "no offenders" can mean, so the parser is shown to
    # still fire on the sentences it must catch...
    for detected in ("CUDA is supported", "float16 is now available",
                     "bfloat16 is implemented", "AMP is shipped",
                     "mixed precision is supported",
                     "the native line is production-ready",
                     "it is universally faster"):
        assert claim.search(detected), detected
    # ...and to stay silent on the one I9 made true.
    for allowed in ("float32 is supported",
                    "float32 and float64 are supported on the CPU"):
        assert claim.search(allowed) is None, allowed


def test_the_dropout_claim_is_scoped_to_the_experimental_native_line():
    """§21.7. Saying "Dropout is supported" is now accurate, but only in
    one scope. Wherever a surface makes the claim, the surrounding text
    must qualify it — experimental, native, float64, or CPU."""
    claim = re.compile(r"[Dd]ropout[^.]{0,60}\b(is|are|now)\s+"
                       r"(supported|available)", re.I)
    scope = re.compile(r"experimental|native|float64|CPU", re.I)
    found_anywhere = False
    for surface in STATUS_SURFACES:
        text = _read(surface)
        for match in claim.finditer(text):
            found_anywhere = True
            window = text[max(0, match.start() - 300):match.end() + 300]
            assert scope.search(window), (surface, match.group(0))
    assert found_anywhere, (
        "no status surface states the shipped Dropout capability at all"
    )


# ==========================================================================
# 7. Hygiene: no machine-specific artifact is committed
# ==========================================================================

@pytest.mark.parametrize("relative", [
    "benchmark_results",
    "benchmark_results.json",
    "cpp/build",
    "asan.log",
    "ubsan.log",
    "lsan.log",
    "sanitizer.log",
    "suppressions.txt",
    "lsan_suppressions.txt",
])
def test_no_closure_artifact_is_tracked(relative):
    """§21.9. The closure produced builds, sanitizer runs, and benchmark
    output; none of it belongs in the repository. The leak contract in
    particular rests on **no suppression file** existing."""
    import subprocess

    result = subprocess.run(
        ["git", "ls-files", "--error-unmatch", relative],
        cwd=REPO_ROOT, capture_output=True, text=True,
    )
    assert result.returncode != 0, f"{relative} is tracked in git"


def test_no_compiled_library_or_result_file_sits_in_a_source_directory():
    """§21.9. Built libraries and machine-specific results must never be
    tracked, whatever a local working tree happens to contain."""
    import subprocess

    tracked = subprocess.run(
        ["git", "ls-files"], cwd=REPO_ROOT,
        capture_output=True, text=True, check=True,
    ).stdout.splitlines()
    for path in tracked:
        lowered = path.lower()
        assert not lowered.endswith((".so", ".dll", ".dylib", ".pyd",
                                     ".obj", ".o", ".lib", ".pdb")), path
        assert "benchmark_results" not in lowered, path
        assert not re.search(r"(^|/)build/", lowered), path
