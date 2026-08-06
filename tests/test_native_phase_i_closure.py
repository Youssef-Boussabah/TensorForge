"""Phase-I closure guardrails (native dtype generalization, milestone I11).

The durable replacement for Phase I's milestone-era *pending* checks, in
the shape ``tests/test_native_phase_h_closure.py`` established. Every
earlier guard about this phase carried a premise that expires at closure —
"I11 is not started", "the phase is active rather than closed", "no surface
may say Phase I is done". I11 ran the cross-platform matrix and closed the
phase, so those premises are gone. What replaces them is not silence: the
boundary Phase I stopped at is now a **permanent** rule rather than a
temporary one, and this module is where it is enforced.

The rules these tests protect:

* the phase is closed, and its ladder is whole (I0-I11, once each, in
  order, every one marked complete, with no I12 and nothing left open);
* **the dtype phase broadened exactly one thing** — ``"float32"`` moved
  from ``UNSUPPORTED`` into ``SUPPORTED_DTYPES`` at I9 — and float64 is
  still the default, the device set never moved, and ``RAW_KERNEL_DTYPES``
  is still the separate and permanently narrower statement it always was;
* the export surface is exactly **54**, which is Phase H's 52 plus the two
  typed storage creators I1 added, and the source inventory agrees with the
  built library;
* **no casting, promotion, dtype inference, global default, device
  transfer, or ``map_location`` exists**, in any form, and none of them may
  be claimed by a status surface either;
* checkpoint format version 3 with ``(1, 2, 3)`` accepted, versions 1 and 2
  permanently float64-only, and the in-memory optimizer state still at
  version 1;
* the I9 exact-resume proof and the I10 hardening, corruption, and
  benchmark-contract evidence are all still present and cannot be deleted
  or weakened unnoticed;
* the frozen benchmark-immutability guard stays **shallow-clone safe** — no
  historical Git object, no ``fetch-depth: 0``, and a workflow Phase I
  never touched;
* no generated artifact is tracked and no benchmark number is committed.

Two properties of this module are deliberate and load-bearing.

**It runs anywhere.** A full local clone, a GitHub Actions depth-1
checkout, a CRLF working tree, and an environment with no ``.git`` at all
must all give the same verdict, because the one CI failure this phase
already produced came from a guard that could not read its own baseline.
Nothing here reaches for a historical Git object, and the one test that
uses ``git ls-files`` degrades to a documented skip rather than a false
pass.

**Every parser has a negative control.** A structural check that cannot
fail proves nothing, so each scanner below is a pure function over text and
is driven, in ``test_*_detects_*``, against deliberately broken input. The
controls operate on temporary strings and temporary directories only: no
repository file is written, moved, or restored.

They test *values and structure* rather than wording, so ordinary prose
improvements do not require rewriting them. Nothing here asserts a
character count, a paragraph order, or a benchmark number.
"""
import ast
import inspect
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from tensorforge.backends import cpp

REPO_ROOT = Path(__file__).resolve().parent.parent
PHASE_I_DESIGN = REPO_ROOT / "docs" / "native_dtype_float32_design.md"
AGENT_INSTRUCTIONS = REPO_ROOT / "CLAUDE.md"
WORKFLOW = REPO_ROOT / ".github" / "workflows" / "tests.yml"

# Surfaces that state *current* status. Per-milestone historical records
# deliberately preserve superseded wording and are not scanned; the
# release history is a chronology for the same reason.
STATUS_SURFACES = (
    "README.md",
    "docs/roadmap.md",
    "docs/project_summary.md",
    "docs/native_support_matrix.md",
    "docs/backend_experiments.md",
    "docs/architecture.md",
    "src/tensorforge/experimental/__init__.py",
)

# ---------------------------------------------------------------------------
# The final boundary. Written out once, as literals, so a drift in either
# direction is a single obvious diff — and split by *what kind of fact each
# one is*, because Phase I's most repeated lesson is that three different
# dtype rows answer three different questions.
# ---------------------------------------------------------------------------

# The capability.
FINAL_DTYPES = ("float64", "float32")
FINAL_DEVICES = ("cpu",)
FINAL_UNSUPPORTED = ("cuda", "amp")
# The default an omitted dtype selects — decided explicitly at I9 and kept
# because it is accurate, not merely because it did not move.
FINAL_DEFAULT_DTYPE = "float64"
# A permanent limitation of the seven handle-free raw utility kernels,
# which take only ``double*`` and an element count and so have no dtype to
# dispatch on. Never the overall support statement.
FINAL_RAW_KERNEL_DTYPES = ("float64",)

# What Phase I inherited, pinned as history so the delta is checkable.
PHASE_H_DTYPES = ("float64",)
PHASE_H_UNSUPPORTED = ("float32", "cuda", "amp")
PHASE_H_EXPORT_COUNT = 52
# ...and the phase's one and only public capability change, at I9.
PHASE_I_MOVED_DTYPES = frozenset({"float32"})
PUBLIC_SUPPORT_MILESTONE = "I9"

# The ABI. Two symbols, added at I1, and I2-I11 added none.
PHASE_I_ADDED_EXPORTS = (
    "tf_storage_create_typed",
    "tf_storage_create_uninitialized_typed",
)
FINAL_EXPORT_COUNT = PHASE_H_EXPORT_COUNT + len(PHASE_I_ADDED_EXPORTS)  # 54

# Symbols added by **later phases**, after Phase I closed — the exact
# counterpart of ``POST_PHASE_I_EXAMPLES`` below, and for the same reason.
# Phase K, milestone K3 added the argmax forward.
POST_PHASE_I_EXPORTS = {"tf_core_argmax": "K3", "tf_core_index_select": "K4"}
CURRENT_EXPORT_COUNT = FINAL_EXPORT_COUNT + len(POST_PHASE_I_EXPORTS)  # 56

# Frozen ABI dtype codes and the one item-size authority's answers. Written
# here independently of the module under test, so a silent renumbering
# fails rather than propagating.
DTYPE_CODE_FLOAT64 = 0
DTYPE_CODE_FLOAT32 = 1
ITEM_SIZES = {"float64": 8, "float32": 4}

# Serialization.
FINAL_CHECKPOINT_FORMAT = "tensorforge.native_checkpoint"
FINAL_CHECKPOINT_VERSION = 3
FINAL_CHECKPOINT_VERSIONS = (1, 2, 3)
FLOAT64_ONLY_CHECKPOINT_VERSIONS = (1, 2)
FINAL_OPTIMIZER_STATE_VERSION = 1

# Inventories, **as Phase I closed on them**. These are historical: they
# record what I11 left, not what the tree happens to hold today.
FINAL_CTEST_COUNT = 24
FINAL_EXAMPLE_COUNT = 15

# Native CTests added *after* Phase I closed, each mapped to the milestone
# that shipped it — the same split ``POST_PHASE_I_EXAMPLES`` uses below.
# Phase K, milestone K1 added the int64 storage target, milestone K3 the
# argmax one, and milestone K4 the index_select one.
POST_PHASE_I_CTESTS = {"dtype_int64_storage": "K1", "argmax": "K3",
                       "index_select": "K4"}
CURRENT_CTEST_COUNT = FINAL_CTEST_COUNT + len(POST_PHASE_I_CTESTS)  # 27
MILESTONES = tuple(f"I{index}" for index in range(12))   # I0 ... I11

# Examples added *after* Phase I closed, each mapped to the milestone that
# shipped it. Keeping the split explicit is what stops later growth from
# being absorbed into I11's record: Phase I closed at fifteen examples and
# always will have, whatever the tree grows to afterwards.
POST_PHASE_I_EXAMPLES = {
    "native_minibatch_training.py": "J6",
}
CURRENT_EXAMPLE_COUNT = FINAL_EXAMPLE_COUNT + len(POST_PHASE_I_EXAMPLES)

# The one example I9 added, and the one benchmark I10 added.
I9_EXAMPLE = "examples/native_float32_training.py"
I10_BENCHMARK = "benchmarks/benchmark_native_dtype.py"
INHERITED_BENCHMARK_COUNT = 7

# Benchmarks added by **later phases**, after Phase I closed — the exact
# counterpart of ``POST_PHASE_I_EXAMPLES`` above. Each names the milestone
# that shipped it, so Phase I's own benchmark delta stays exactly I10's
# one and a later addition is attributed rather than absorbed.
POST_PHASE_I_BENCHMARKS = {
    "benchmark_native_data_pipeline.py": "J8",
}
CURRENT_BENCHMARK_COUNT = (INHERITED_BENCHMARK_COUNT + 1
                           + len(POST_PHASE_I_BENCHMARKS))

# The evidence I9 and I10 left, which closure must not let disappear. Each
# entry is (path, minimum test count) — a floor rather than an equality, so
# adding coverage is free while deleting it is not.
REQUIRED_EVIDENCE = (
    ("tests/test_native_float32_public.py", 40),
    ("tests/test_native_float32_training.py", 50),
    ("tests/test_native_float32_state.py", 40),
    ("tests/test_native_float32_hardening.py", 60),
    ("tests/test_native_float32_checkpoint_corruption.py", 15),
    ("tests/test_native_dtype_benchmark.py", 35),
    ("tests/test_native_phase_i.py", 200),
)


def _read(relative):
    return (REPO_ROOT / relative).read_text(encoding="utf-8")


def _flat(text):
    """Whitespace-flattened, emphasis-stripped text, so a claim split
    across lines or wrapped in markdown still reads as one sentence."""
    return re.sub(r"\s+", " ", re.sub(r"[*`]", "", text))


def _design():
    return PHASE_I_DESIGN.read_text(encoding="utf-8")


def _source_exports():
    names = set()
    for source in sorted((REPO_ROOT / "cpp" / "src").glob("*.cpp")):
        text = source.read_text(encoding="utf-8")
        names.update(re.findall(r"TF_EXPORT[^;{]*?\b(tf_[a-z0-9_]+)\s*\(",
                                text, re.S))
    return names


def _tracked_files():
    """Every tracked path, or a documented skip where Git cannot answer.

    The one place this module talks to Git at all. It asks about the
    *index*, which a depth-1 checkout has in full — unlike a historical
    blob, which is what the guard this phase already had to repair was
    reaching for."""
    try:
        done = subprocess.run(["git", "ls-files"], cwd=REPO_ROOT,
                              capture_output=True, text=True)
    except OSError:                                   # pragma: no cover
        pytest.skip("git is unavailable, so the tracked set cannot be read")
    if done.returncode != 0:                          # pragma: no cover
        pytest.skip("this tree has no git index to read")
    return done.stdout.splitlines()


# ===========================================================================
# 1. The milestone ladder — whole, ordered, and closed
# ===========================================================================
#
# Parsed structurally rather than matched as prose, and every parser here
# is a pure function over text so the negative controls below can feed it
# deliberately broken ladders.

_LADDER_HEADING = re.compile(r"^### (I\d+) — (.*)$", re.M)


def _ladder_text(text):
    """The §29 ladder body, from the section heading to §29.1."""
    if "## 29." not in text:
        raise AssertionError("the design has no milestone-ladder section")
    ladder = text.split("## 29.", 1)[1]
    return ladder.split("### 29.1", 1)[0]


def _ladder_rows(text):
    """``[(milestone, heading_tail)]`` in document order."""
    return _LADDER_HEADING.findall(_ladder_text(text))


def _ladder_problems(text):
    """Every way the ladder can be wrong, as a list of reasons.

    Returned rather than raised so one call can report all of them, and so
    the negative controls can assert *which* fault was detected instead of
    merely that something failed."""
    problems = []
    rows = _ladder_rows(text)
    names = [name for name, _ in rows]

    if len(names) != len(set(names)):
        duplicated = sorted({n for n in names if names.count(n) > 1})
        problems.append(f"duplicated milestone(s): {duplicated}")
    if sorted(set(names), key=lambda n: int(n[1:])) != list(MILESTONES):
        missing = [n for n in MILESTONES if n not in names]
        extra = [n for n in names if n not in MILESTONES]
        if missing:
            problems.append(f"missing milestone(s): {missing}")
        if extra:
            problems.append(f"unexpected milestone(s): {extra}")
    if names != sorted(names, key=lambda n: int(n[1:])):
        problems.append(f"out of order: {names}")

    # Every row marked complete, in the heading, in the form milestone I10
    # established. A phase with an unmarked row is not closed.
    for name, tail in rows:
        if not re.search(r"\*\*complete\*\*", tail, re.I):
            problems.append(f"{name} is not marked complete")
        if re.search(r"not started|in progress|pending", tail, re.I):
            problems.append(f"{name} is still marked open")
    return problems


def test_the_milestone_ladder_runs_i0_to_i11_once_each_in_order():
    """I0 through I11, each exactly once, ordered, and every one marked
    complete. This is the single assertion that says the phase is done."""
    problems = _ladder_problems(_design())
    assert problems == [], problems


def test_the_design_marks_the_phase_complete_rather_than_active():
    """The status line, which is what every other surface is reconciled
    against. Through I10 it had to name a first *unstarted* milestone;
    at closure there is none, and saying so is the point."""
    text = _flat(_design())
    status = re.search(r"Phase-I status:(.{0,240})", text, re.I)
    assert status, "the design does not state its milestone status"
    claim = status.group(1)
    assert re.search(r"\bI0\b.{0,80}?\bI11\b.{0,60}?complete", claim, re.I), (
        f"the status line does not record I0-I11 complete: {claim!r}"
    )
    assert not re.search(r"not started", claim, re.I), (
        f"the status line still names an unstarted milestone: {claim!r}"
    )
    # ...and the phase itself, stated as a phase rather than as a run of
    # milestone numbers.
    assert re.search(
        r"Phase I is (?:now )?(?:\w+ ){0,3}?(complete|closed)", text, re.I
    ), "the design does not state that Phase I is closed"


def test_the_design_names_no_milestone_beyond_i11():
    """An I12 would be a phase this document does not define."""
    for match in re.finditer(r"\bI1[2-9]\b", _design()):
        raise AssertionError(f"the design names {match.group(0)}")


# One authority for "this surface still says the phase is open", used by
# the scan and by its own negative control so the two cannot drift apart.
_STALE_STATUS = re.compile(
    r"Phase[- ]I\b[^.;]{0,90}?\b(is not complete|is not closed|"
    r"is active|remains active|has not been closed|is the current phase|"
    r"is in progress|is active rather than closed)\b"
    r"|\bI11\b[^.;]{0,70}?\b(not started|is next|is not started|pending)\b"
    r"|\b(what remains|remaining|still to come|next)\b[^.;]{0,40}?\bI11\b"
    r"|\bI0[-–—]I10\b[^.;]{0,30}?\bcomplete\b"
    r"|\bI0 through I10\b[^.;]{0,30}?\bcomplete\b",
    re.I)


def test_no_status_surface_still_calls_phase_i_unfinished():
    """The failure this exists for: a surface advanced in one paragraph
    and left saying "I11 is not started" in another.

    Time-scoped sentences are excused, because "Phase I was active at I10"
    is history and history stays accurate."""
    stale = _STALE_STATUS
    history = re.compile(
        r"\b(was|were|had|until|through|before|at I\d|by I\d|then|earlier|"
        r"previously|stayed|remained|originally|drafted|no longer|used to|"
        r"since|history|historical)\b", re.I)
    for surface in STATUS_SURFACES + ("CLAUDE.md",):
        text = _flat(_read(surface))
        offenders = [
            match.group(0) for match in stale.finditer(text)
            if not history.search(
                text[max(0, match.start() - 140):match.end() + 50])
        ]
        assert offenders == [], f"{surface}: {offenders[:3]}"


@pytest.mark.parametrize("surface", STATUS_SURFACES)
def test_every_status_surface_marks_phase_i_complete(surface):
    text = _flat(_read(surface))
    assert "Phase I" in text, f"{surface} does not name Phase I"
    assert re.search(
        r"Phase I[^.;]{0,140}?\b(is|are)\s+(now\s+)?(complete|closed)"
        r"|Phase I[^.;]{0,140}?\bcomplete \(I0[-–—]I11\)"
        r"|\bI0[-–—]I11\b[^.;]{0,70}?\b(complete|landed|closed)"
        r"|\bI0 through I11\b[^.;]{0,70}?\b(complete|landed|closed)",
        text, re.I), f"{surface} does not mark Phase I complete"


def test_phase_h_is_still_recorded_complete_and_was_not_reopened():
    """Closing one phase must not disturb the one before it."""
    design = _flat((REPO_ROOT / "docs"
                    / "native_cpu_performance_design.md").read_text(
                        encoding="utf-8"))
    assert re.search(r"Phase-H status:\s*complete", design, re.I)
    for surface in STATUS_SURFACES:
        text = _flat(_read(surface))
        assert not re.search(
            r"Phase H[^.;]{0,80}?\b(is not complete|has not been closed|"
            r"is the current phase|is active)\b", text, re.I), surface


# --- the ladder parser's negative controls --------------------------------

def _ladder_document(rows):
    """A synthetic design fragment carrying exactly ``rows``."""
    body = "\n".join(f"### {name} — Something — **complete**\n\nbody\n"
                     for name in rows)
    return f"## 29. Milestone ladder\n\n{body}\n### 29.1 Adjustments\n"


def test_the_ladder_parser_detects_every_kind_of_drift():
    """A parser that cannot fail proves nothing. Each fault the closure
    claim depends on is produced deliberately and shown to be caught, on
    temporary strings only — no repository file is touched."""
    # 0. The positive control: a whole, ordered, complete ladder passes.
    assert _ladder_problems(_ladder_document(MILESTONES)) == []

    # 1. One milestone omitted.
    problems = _ladder_problems(
        _ladder_document([n for n in MILESTONES if n != "I5"]))
    assert any("missing" in reason and "I5" in reason for reason in problems), (
        problems)

    # 2. One milestone duplicated.
    problems = _ladder_problems(
        _ladder_document(list(MILESTONES) + ["I7"]))
    assert any("duplicated" in reason for reason in problems), problems

    # 3. Two milestones swapped.
    swapped = list(MILESTONES)
    swapped[3], swapped[4] = swapped[4], swapped[3]
    problems = _ladder_problems(_ladder_document(swapped))
    assert any("out of order" in reason for reason in problems), problems

    # 4. An invented I12.
    problems = _ladder_problems(
        _ladder_document(list(MILESTONES) + ["I12"]))
    assert any("unexpected" in reason and "I12" in reason
               for reason in problems), problems

    # 5. I11 left not started — the specific drift closure exists to stop.
    document = _ladder_document(MILESTONES).replace(
        "### I11 — Something — **complete**",
        "### I11 — Something — **not started**")
    problems = _ladder_problems(document)
    assert any("I11" in reason for reason in problems), problems

    # 6. An earlier milestone falsely reverted.
    document = _ladder_document(MILESTONES).replace(
        "### I4 — Something — **complete**", "### I4 — Something")
    problems = _ladder_problems(document)
    assert "I4 is not marked complete" in problems, problems

    # 7. No ladder section at all.
    with pytest.raises(AssertionError):
        _ladder_problems("# a document with no section 29\n")

    # Nothing above touched the repository: the live ladder still passes.
    assert _ladder_problems(_design()) == []


def test_the_phase_complete_scan_detects_a_surface_left_active():
    """The prose scan's own control, since "no offenders" is only
    meaningful when the pattern can find one. Deliberately driven through
    the **same** compiled pattern the scan uses, so a weakening there
    fails here."""
    stale = _STALE_STATUS
    for caught in (
        "Phase I is active rather than closed",
        "Phase I is not complete",
        "milestones I0-I10 complete, and I11 is not started",
        "what remains is I11 (cross-platform validation)",
    ):
        assert stale.search(caught), caught
    # ...and the accurate closure sentences are not caught.
    for allowed in (
        "Phase I is complete",
        "Phase I is closed: milestones I0 through I11 have all landed",
        "I11 closed the phase",
    ):
        assert not stale.search(allowed), allowed


# ===========================================================================
# 2. The final registries
# ===========================================================================

def test_the_public_registries_are_exactly_phase_is_final_values():
    """The capability, stated as literals and as the delta from what the
    phase inherited, so both facts survive: Phase I moved **one** name, and
    the tuple it left is exactly this one."""
    assert cpp.SUPPORTED_DTYPES == FINAL_DTYPES
    assert cpp.SUPPORTED_DEVICES == FINAL_DEVICES
    assert cpp.UNSUPPORTED == FINAL_UNSUPPORTED
    assert cpp.RAW_KERNEL_DTYPES == FINAL_RAW_KERNEL_DTYPES

    # Exactly one name moved, in exactly one direction, and nothing else
    # joined either tuple.
    assert set(FINAL_DTYPES) - set(PHASE_H_DTYPES) == PHASE_I_MOVED_DTYPES
    assert set(PHASE_H_UNSUPPORTED) - set(FINAL_UNSUPPORTED) == (
        PHASE_I_MOVED_DTYPES)
    assert set(FINAL_UNSUPPORTED) <= set(PHASE_H_UNSUPPORTED)
    # The device set never moved at all — a dtype phase grants no device.
    assert FINAL_DEVICES == ("cpu",)

    # Order is contractual: float64 first, because it is the default.
    assert cpp.SUPPORTED_DTYPES[0] == FINAL_DEFAULT_DTYPE
    # Supported and unsupported cannot overlap, in either direction.
    assert not set(cpp.SUPPORTED_DTYPES) & set(cpp.UNSUPPORTED)
    assert "float32" not in cpp.UNSUPPORTED
    # The raw-kernel registry is a *narrower* statement, permanently, and
    # is a strict subset of what the runtime supports.
    assert set(cpp.RAW_KERNEL_DTYPES) < set(cpp.SUPPORTED_DTYPES)


def test_the_three_dtype_rows_answer_three_different_questions():
    """``backend_info()`` reports a capability, a default, and a
    limitation, and none of them may be reported as another."""
    info = cpp.backend_info()
    assert info["supported_dtypes"] == FINAL_DTYPES        # the capability
    assert info["dtype"] == FINAL_DEFAULT_DTYPE            # the default
    assert info["raw_kernel_dtypes"] == FINAL_RAW_KERNEL_DTYPES  # the limit
    assert info["supported_devices"] == FINAL_DEVICES
    assert info["device"] == "cpu"
    assert info["unsupported"] == FINAL_UNSUPPORTED
    # The default is genuinely what an omitted dtype selects.
    assert cpp.normalize_dtype(None) == FINAL_DEFAULT_DTYPE
    assert cpp.normalize_dtype("float64") == "float64"
    assert cpp.normalize_dtype("float32") == "float32"


def test_the_numpy_reference_backend_keeps_its_own_narrower_truth():
    """Phase I is a native-line phase. The NumPy reference backend has its
    own supported-dtype statement, and it did **not** move with the native
    one — a float32 native tensor implies nothing about this backend."""
    from tensorforge.backends.numpy_backend import NumpyBackend

    info = NumpyBackend().backend_info()
    assert info["supported_dtypes"] == ("float64",)
    assert info["supported_devices"] == ("cpu",)
    assert info["dtype"] == "float64"
    # Deliberately different from the native registry, and stated as a
    # difference so a future edit that "helpfully" syncs them fails here.
    assert info["supported_dtypes"] != cpp.SUPPORTED_DTYPES
    # Its conversion boundary is still float64 in and float64 out.
    backend = NumpyBackend()
    assert backend.tensor_from_array(
        np.zeros(3, dtype=np.float32)).dtype == np.float64
    assert backend.to_numpy(np.zeros(3, dtype=np.float32)).dtype == np.float64
    assert backend.zeros((2,)).dtype == np.float64


def test_no_dtype_beyond_the_two_is_reachable():
    """The registry claim checked against behavior rather than trusted:
    two dtypes, not "any dtype"."""
    for absent in ("float16", "bfloat16", "float128", "int64", "int32",
                   "complex64", "bool"):
        with pytest.raises((ValueError, TypeError)):
            cpp.normalize_dtype(absent)
    for device in ("cuda", "cuda:0", "gpu", "mps"):
        with pytest.raises((ValueError, TypeError)):
            cpp.normalize_device(device)


# ===========================================================================
# 3. Public construction — both dtypes, float64 still the default
# ===========================================================================

_PUBLIC_STORAGE_BUILDERS = (
    "NativeStorage",
    "NativeTensorCore.from_array",
    "NativeTensorCore.zeros",
    "NativeTensorCore.full",
)
_PUBLIC_TENSOR_BUILDERS = (
    "NativeTensor.from_array",
    "NativeTensor.zeros",
    "NativeTensor.full",
)


def _public_builders():
    """Every public constructor, as zero-argument factories keyed by name,
    for each dtype and for an omitted dtype."""
    from tensorforge.experimental import NativeTensor

    values = np.zeros((2, 2), dtype=np.float64)

    def at(dtype):
        keywords = {} if dtype is None else {"dtype": dtype}
        return {
            "NativeStorage": lambda: cpp.NativeStorage(4, **keywords),
            "NativeTensorCore.from_array":
                lambda: cpp.NativeTensorCore.from_array(values, **keywords),
            "NativeTensorCore.zeros":
                lambda: cpp.NativeTensorCore.zeros((2, 2), **keywords),
            "NativeTensorCore.full":
                lambda: cpp.NativeTensorCore.full((2, 2), 1.5, **keywords),
            "NativeTensor.from_array":
                lambda: NativeTensor.from_array(values, **keywords),
            "NativeTensor.zeros":
                lambda: NativeTensor.zeros((2, 2), **keywords),
            "NativeTensor.full":
                lambda: NativeTensor.full((2, 2), 1.5, **keywords),
        }
    return at


@pytest.mark.skipif(not cpp.is_available(),
                    reason="the native library is not built")
@pytest.mark.parametrize("dtype", FINAL_DTYPES)
@pytest.mark.parametrize(
    "name", _PUBLIC_STORAGE_BUILDERS + _PUBLIC_TENSOR_BUILDERS)
def test_every_public_constructor_builds_both_dtypes(name, dtype):
    built = _public_builders()(dtype)[name]()
    try:
        assert built.dtype == dtype, name
    finally:
        built.close()


@pytest.mark.skipif(not cpp.is_available(),
                    reason="the native library is not built")
@pytest.mark.parametrize(
    "name", _PUBLIC_STORAGE_BUILDERS + _PUBLIC_TENSOR_BUILDERS)
def test_an_omitted_dtype_still_means_float64(name):
    built = _public_builders()(None)[name]()
    try:
        assert built.dtype == FINAL_DEFAULT_DTYPE, name
    finally:
        built.close()


@pytest.mark.skipif(not cpp.is_available(),
                    reason="the native library is not built")
def test_the_dtype_is_never_inferred_from_the_input_array():
    """The rule that keeps ingress an explicit conversion rather than a
    cast: a float32 host array with no ``dtype`` still gives float64."""
    from tensorforge.experimental import NativeTensor

    narrow = np.zeros((2, 2), dtype=np.float32)
    for build in (lambda: cpp.NativeTensorCore.from_array(narrow),
                  lambda: NativeTensor.from_array(narrow)):
        built = build()
        try:
            assert built.dtype == FINAL_DEFAULT_DTYPE
        finally:
            built.close()
    # ...and an int64 array is converted at the boundary rather than
    # becoming an integer tensor.
    built = cpp.NativeTensorCore.from_array(np.arange(4, dtype=np.int64))
    try:
        assert built.dtype == FINAL_DEFAULT_DTYPE
    finally:
        built.close()


@pytest.mark.skipif(not cpp.is_available(),
                    reason="the native library is not built")
def test_to_numpy_returns_the_tensors_own_width_and_never_widens():
    from tensorforge.experimental import NativeTensor

    expected = {"float64": np.float64, "float32": np.float32}
    for dtype, scalar in expected.items():
        tensor = NativeTensor.full((2, 3), 0.5, dtype=dtype)
        try:
            produced = tensor.to_numpy()
            assert produced.dtype == np.dtype(scalar), dtype
        finally:
            tensor.close()


def test_no_cast_promotion_or_device_transfer_api_exists():
    """The absence, stated over the live objects rather than promised in
    prose. None of these may be added without leaving the phase."""
    import tensorforge.experimental as experimental

    banned = ("astype", "to", "cast", "float", "double", "half", "cuda",
              "cpu", "type_as", "promote", "to_dtype", "as_dtype",
              "map_location", "set_default_dtype", "get_default_dtype")
    targets = [cpp.NativeStorage, cpp.NativeTensorCore, cpp.NativeTensorView,
               experimental.NativeTensor, experimental.NativeParameter]
    for target in targets:
        for name in banned:
            assert not hasattr(target, name), (target.__name__, name)
    # ...and no module-level global default either.
    for module in (cpp, experimental):
        for name in ("set_default_dtype", "get_default_dtype",
                     "default_dtype", "DEFAULT_DTYPE", "set_default_device"):
            assert not hasattr(module, name), (module.__name__, name)


def test_the_private_typed_constructors_stayed_private():
    """They grant no width the public constructors do not, and they exist
    because "this dtype came from a live storage" and "this dtype came from
    a caller" are different trust statements. Neither may be exported."""
    import tensorforge.experimental as experimental

    for name in ("_typed", "_typed_from_array", "_typed_full",
                 "_typed_zeros", "_from_core"):
        assert name not in experimental.__all__, name
        assert not hasattr(experimental, name.lstrip("_")), name
    # They are still present where the runtime needs them, so this is a
    # visibility rule rather than a removal.
    assert hasattr(cpp.NativeStorage, "_typed")
    assert hasattr(cpp.NativeTensorCore, "_typed_from_array")
    assert hasattr(experimental.NativeTensor, "_typed_zeros")


# ===========================================================================
# 4. The ABI and the export inventory
# ===========================================================================

def test_the_source_exports_exactly_fifty_four_symbols():
    """Stated as arithmetic rather than as a bare number, so the facts stay
    separable: Phase H closed at 52, Phase I added exactly two, and every
    symbol beyond that belongs to a named later milestone."""
    exports = _source_exports()
    assert len(exports) == CURRENT_EXPORT_COUNT, sorted(exports)
    for name in PHASE_I_ADDED_EXPORTS:
        assert name in exports, name
    for name, milestone in POST_PHASE_I_EXPORTS.items():
        assert name in exports, (name, milestone)
    phase_i = exports - set(POST_PHASE_I_EXPORTS)
    assert len(phase_i) == FINAL_EXPORT_COUNT, sorted(phase_i)
    assert len(phase_i - set(PHASE_I_ADDED_EXPORTS)) == PHASE_H_EXPORT_COUNT


def test_the_two_phase_i_symbols_are_the_only_typed_creators():
    """Per-operation float32 exports were explicitly rejected (§6.5), and
    the untyped creators stayed as float64 compatibility wrappers."""
    exports = _source_exports()
    typed = {name for name in exports if name.endswith("_typed")}
    assert typed == set(PHASE_I_ADDED_EXPORTS), sorted(typed)
    assert "tf_storage_create" in exports
    assert "tf_storage_create_uninitialized" in exports


def test_no_per_dtype_cast_or_query_export_was_added():
    """The shapes a dtype phase is most tempted to add, banned by name."""
    banned = re.compile(
        r"_f32$|_f64$|_float32$|_float64$|_typed_[a-z]+_(add|mul|matmul)"
        r"|cast|convert|astype|promote|dtype_of|get_dtype|query_dtype"
        r"|set_dtype|default_dtype", re.I)
    for name in sorted(_source_exports()):
        assert not banned.search(name), name
    # ...and no optimizer, checkpoint, or module export either: the whole
    # of that stack is Python over the handle-based ABI.
    forbidden = re.compile(
        r"^tf_(optimizer|adam|sgd|checkpoint|module|linear|conv2d_module|"
        r"batchnorm|layernorm|save|load)", re.I)
    for name in sorted(_source_exports()):
        assert not forbidden.search(name), name


@pytest.mark.skipif(not cpp.is_available(),
                    reason="the native library is not built")
def test_the_built_library_exports_exactly_what_the_source_declares():
    library = cpp._require_library()
    for name in sorted(_source_exports()):
        assert hasattr(library, name), name


def test_the_declared_ctypes_surface_matches_the_source_exports():
    declared = set(re.findall(r"library\.(tf_[a-z0-9_]+)\s*\.",
                              _read("src/tensorforge/backends/cpp.py")))
    declared |= set(re.findall(r"getattr\(library,\s*\"(tf_[a-z0-9_]+)\"",
                               _read("src/tensorforge/backends/cpp.py")))
    missing = declared - _source_exports()
    assert not missing, f"declared but not exported: {sorted(missing)}"


def test_the_seven_raw_kernels_kept_their_float64_only_signatures():
    """§7.2, permanently. These take only ``double*`` and an element
    count, so they have no dtype to dispatch on — which is why
    ``RAW_KERNEL_DTYPES`` is a separate row and not a gap."""
    assert cpp.RAW_KERNELS == (
        "elementwise_add", "elementwise_subtract", "elementwise_multiply",
        "elementwise_divide", "relu", "matmul", "matmul_tiled",
    )
    assert len(cpp.RAW_KERNELS) == INHERITED_BENCHMARK_COUNT
    source = "\n".join(
        path.read_text(encoding="utf-8")
        for path in sorted((REPO_ROOT / "cpp" / "src").glob("*.cpp")))
    for symbol in ("tf_elementwise_add", "tf_relu", "tf_matmul",
                   "tf_matmul_tiled"):
        signature = re.search(
            rf"TF_EXPORT[^;{{]*?\b{symbol}\s*\(([^)]*)\)", source, re.S)
        assert signature, symbol
        arguments = signature.group(1)
        assert "void*" not in arguments, (symbol, arguments)
        assert "double" in arguments, (symbol, arguments)


def test_the_three_transfer_exports_kept_their_argument_shape():
    """I2 retyped their host positions from ``double*`` to ``void*`` — a
    declaration change, not an ABI change. Same symbols, same argument
    counts and order, so a previously compiled caller links unchanged."""
    source = "\n".join(
        path.read_text(encoding="utf-8")
        for path in sorted((REPO_ROOT / "cpp" / "src").glob("*.cpp")))
    expected = {
        "tf_storage_copy_from": 2,
        "tf_storage_copy_to": 2,
        "tf_storage_materialize": 6,
    }
    for symbol, count in expected.items():
        signature = re.search(
            rf"TF_EXPORT[^;{{]*?\b{symbol}\s*\(([^)]*)\)", source, re.S)
        assert signature, symbol
        arguments = [a.strip() for a in signature.group(1).split(",")]
        assert len(arguments) == count, (symbol, arguments)
        # The host position carries no dtype; the storage tag is the
        # authority, which is exactly why it is ``void*``.
        assert any("void" in argument for argument in arguments), (
            symbol, arguments)


def test_the_frozen_dtype_codes_and_the_one_item_size_authority():
    """Codes 0 and 1, sizes 8 and 4, and exactly one place in C++ that may
    spell a storage width."""
    codes = "\n".join(
        path.read_text(encoding="utf-8")
        for path in sorted((REPO_ROOT / "cpp" / "include").glob("*.h")))
    assert re.search(r"TF_DTYPE_FLOAT64\s*=\s*0", codes)
    assert re.search(r"TF_DTYPE_FLOAT32\s*=\s*1", codes)
    # The Python side agrees with the frozen codes.
    table = getattr(cpp, "_DTYPE_CODES", None)
    if table is not None:
        assert table["float64"] == DTYPE_CODE_FLOAT64
        assert table["float32"] == DTYPE_CODE_FLOAT32
    # One item-size authority: ``dtype_item_size`` is defined once.
    definitions = 0
    for path in sorted((REPO_ROOT / "cpp").rglob("*.h")):
        text = path.read_text(encoding="utf-8")
        definitions += len(re.findall(
            r"inline\s+[\w:]+\s+dtype_item_size\s*\(", text))
    for path in sorted((REPO_ROOT / "cpp" / "src").glob("*.cpp")):
        text = path.read_text(encoding="utf-8")
        definitions += len(re.findall(
            r"^[\w:]+\s+dtype_item_size\s*\(", text, re.M))
    assert definitions == 1, (
        f"{definitions} definitions of dtype_item_size; there must be one")
    assert ITEM_SIZES == {"float64": 8, "float32": 4}


# ===========================================================================
# 5. The CTest inventory
# ===========================================================================

def _registered_ctests():
    cmake = _read("cpp/CMakeLists.txt")
    return re.findall(r"add_test\s*\(\s*NAME\s+(\w+)", cmake)


def test_the_ctest_inventory_is_exactly_twenty_four_unique_targets():
    """Phase I closed at 24, and every target the live tree carries beyond
    that belongs to a named later milestone."""
    names = _registered_ctests()
    assert len(names) == CURRENT_CTEST_COUNT, names
    for name, milestone in POST_PHASE_I_CTESTS.items():
        assert name in names, (name, milestone)
    assert len([n for n in names
                if n not in POST_PHASE_I_CTESTS]) == FINAL_CTEST_COUNT, names
    assert len(set(names)) == len(names), "a CTest name is registered twice"
    # Every registered test has a source file, and every source file is
    # registered — so a target cannot be added or orphaned unnoticed.
    sources = {path.stem for path in
               sorted((REPO_ROOT / "cpp" / "tests").glob("test_*.cpp"))}
    assert sources == {f"test_{name}" for name in names}, (
        sorted(sources), sorted(names))


@pytest.mark.parametrize("name", (
    "dtype_storage", "typed_transfer", "dtype_elementwise",
    "dtype_reduction_matmul", "dtype_cnn", "dtype_classification",
    "dtype_dropout",
))
def test_every_phase_i_ctest_is_still_registered(name):
    """The seven typed targets I1-I7 added, one per milestone that shipped
    a kernel. Deleting one would silently drop a dtype's proof."""
    assert name in _registered_ctests(), name


def test_the_build_still_offers_exactly_two_options():
    cmake = _read("cpp/CMakeLists.txt")
    options = set(re.findall(r"^\s*option\s*\(\s*(\w+)", cmake, re.M))
    assert options == {"TF_BUILD_TESTS", "TF_SANITIZE"}, options
    assert not re.search(r"/arch:AVX|-mavx|-march=native|-ffast-math"
                         r"|/fp:fast|-funsafe-math", cmake)


# ===========================================================================
# 6. Checkpoint and optimizer state
# ===========================================================================

def test_the_checkpoint_constants_are_phase_is_final_values():
    from tensorforge.experimental import native_checkpoint

    assert native_checkpoint._FORMAT == FINAL_CHECKPOINT_FORMAT
    assert native_checkpoint._FORMAT_VERSION == FINAL_CHECKPOINT_VERSION
    assert (native_checkpoint._SUPPORTED_FORMAT_VERSIONS
            == FINAL_CHECKPOINT_VERSIONS)
    accepted = native_checkpoint._SUPPORTED_FORMAT_VERSIONS
    assert list(accepted) == sorted(accepted)
    assert accepted[-1] == native_checkpoint._FORMAT_VERSION
    # Nothing the phase inherited was dropped.
    assert set(FLOAT64_ONLY_CHECKPOINT_VERSIONS).issubset(set(accepted))


def test_the_in_memory_optimizer_state_version_did_not_move():
    from tensorforge.experimental import native_optimizer_state

    assert (native_optimizer_state.FORMAT_VERSION
            == FINAL_OPTIMIZER_STATE_VERSION)


def test_a_fourth_checkpoint_version_is_rejected():
    from tensorforge.experimental import native_checkpoint

    assert 4 not in native_checkpoint._SUPPORTED_FORMAT_VERSIONS
    assert 0 not in native_checkpoint._SUPPORTED_FORMAT_VERSIONS


def test_the_loader_validates_metadata_with_the_savers_own_authority():
    """I10's one production repair, pinned so it cannot regress: the same
    ``_validated_metadata`` runs on both sides."""
    from tensorforge.experimental import native_checkpoint

    assert hasattr(native_checkpoint, "_validated_metadata")
    tree = ast.parse(inspect.getsource(native_checkpoint))
    users = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for inner in ast.walk(node):
                if (isinstance(inner, ast.Call)
                        and getattr(inner.func, "id", None)
                        == "_validated_metadata"):
                    users.add(node.name)
    assert "save_native_checkpoint" in users, sorted(users)
    assert "load_native_checkpoint" in users, sorted(users)


def _executable_source(text):
    """``text`` with every docstring removed, so a rule is measured
    against what runs rather than against what is explained.

    Needed here for the same reason the benchmark guard needs it: this
    module's prose legitimately states "no ``map_location``, no cast", and
    a raw substring scan would fail on the very file that gets it right."""
    tree = ast.parse(text)
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef,
                             ast.AsyncFunctionDef)) and node.body:
            first = node.body[0]
            if (isinstance(first, ast.Expr)
                    and isinstance(first.value, ast.Constant)
                    and isinstance(first.value.value, str)):
                node.body = node.body[1:] or [ast.Pass()]
    return ast.unparse(tree)


def test_no_pickle_map_location_or_upgrade_in_place_exists():
    source = _read("src/tensorforge/experimental/native_checkpoint.py")
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            names = [alias.name for alias in node.names]
            if isinstance(node, ast.ImportFrom):
                names.append(node.module or "")
            for name in names:
                assert "pickle" not in (name or ""), name
    code = _executable_source(source)
    for banned in ("map_location", "allow_pickle=True", ".astype("):
        assert banned not in code, banned
    # ...and the archive really is opened with pickling refused.
    assert "allow_pickle=False" in code


# ===========================================================================
# 7. The exact-resume proof, and the I10 hardening evidence
# ===========================================================================

def test_the_exact_resume_proof_still_exists_and_covers_both_dtypes():
    """Not reimplemented here: the I9 proof is a whole example plus a
    focused suite, and duplicating it would create a second authority that
    could disagree. What closure owns is that it cannot be deleted or
    quietly narrowed."""
    example = REPO_ROOT / I9_EXAMPLE
    assert example.is_file(), I9_EXAMPLE

    sys.path.insert(0, str(REPO_ROOT / "examples"))
    try:
        import native_float32_training as proof
    finally:
        sys.path.pop(0)

    # Both dtypes, run independently, and compared only against themselves.
    assert set(proof.RUN_DTYPES) == set(FINAL_DTYPES), proof.RUN_DTYPES
    # Every state family the resume must reproduce, by name.
    for claim in (
        "suffix_matches", "split_gradients_match", "gradients_nonempty",
        "parameters_match", "buffers_match", "moments_match",
        "counters_match", "optimizer_matches", "generator_matches",
        "topology_matches", "final_train_logits_match", "predictions_match",
        "final_eval_matches", "metadata_validated", "identities_preserved",
        "fresh_started_different", "dtypes_match",
    ):
        assert claim in proof.REQUIRED, claim
    # The interrupt is real: the split is neither the first nor the last
    # step, so a resume genuinely has to restore something.
    assert 0 < proof.SPLIT_STEP < proof.TOTAL_STEPS


def test_the_exact_resume_suite_still_exercises_every_required_claim():
    """The focused suite that drives the proof, held to its shape rather
    than to a count: it must parametrize over both dtypes and over the
    example's own required-claim list."""
    source = _read("tests/test_native_float32_training.py")
    assert "REQUIRED" in source
    assert re.search(r"@pytest\.mark\.parametrize\(\s*\"claim\",\s*REQUIRED",
                     source), "the suite no longer drives every claim"
    assert "test_the_resume_reproduces_every_state_family_exactly" in source
    assert "test_the_first_resumed_step_produces_equal_gradients" in source
    assert "test_the_next_dropout_mask_after_a_resume_is_identical" in source
    # The negative controls that make the proof non-vacuous.
    assert "test_a_resume_that_ignores_the_metadata_diverges" in source
    assert "test_a_generator_that_was_not_restored_produces_a_different_mask" \
        in source
    assert "test_the_two_dtypes_are_genuinely_different_runs" in source
    # Bit comparison, never a tolerance — asserted through the suite's own
    # guard rather than by scanning for the token, because the suite's
    # prose and that guard's banned-token list both legitimately spell
    # ``allclose`` while no comparison uses one.
    assert "test_the_example_makes_no_cross_dtype_numerical_claim" in source
    for banned in ('"allclose("', '"isclose("', '"atol="', '"rtol="'):
        assert banned in source, (
            f"the suite's tolerance guard no longer bans {banned}")
    # ...and the example itself really is free of them, measured over what
    # executes rather than over what it explains.
    example = _executable_source(_read(I9_EXAMPLE))
    for banned in ("allclose(", "isclose(", "atol=", "rtol="):
        assert banned not in example, banned


@pytest.mark.parametrize("relative,minimum", REQUIRED_EVIDENCE)
def test_the_phase_i_evidence_files_are_all_present(relative, minimum):
    """A floor rather than an equality: adding coverage is free, deleting
    it is not."""
    path = REPO_ROOT / relative
    assert path.is_file(), relative
    count = len(re.findall(r"^def test_", path.read_text(encoding="utf-8"),
                           re.M))
    assert count >= minimum, f"{relative} has {count} tests, expected >= {minimum}"


def test_the_i10_hardening_evidence_retains_its_named_subjects():
    """The specific coverage I10 recorded, asserted by subject so a
    rewrite that quietly drops one is caught."""
    hardening = _read("tests/test_native_float32_hardening.py")
    for subject in ("relu_backward", "conv2d", "cross_entropy",
                    "sequential", "copy_value_", "load_state_dict",
                    "batchnorm", "maxpool2d_backward"):
        assert subject in hardening.lower(), subject
    # The C ABI proved a *second* authority, both ways, each with its own
    # negative control.
    assert "_require_matching_metadata" in hardening
    assert re.search(r"negative control|non-vacuous", hardening, re.I)
    # All four saved-resource families in one float32 graph.
    for family in ("dropout", "winner", "snapshot", "probabilit"):
        assert family in hardening.lower(), family


def test_the_checkpoint_corruption_matrix_still_carries_its_cases():
    """Structure rather than a literal count: the matrix keeps its own
    floor assertion, so pinning a second number here would create a rival
    authority that could disagree with it."""
    corruption = _read("tests/test_native_float32_checkpoint_corruption.py")
    # Both dtypes, driven independently.
    assert re.search(r'BOTH_DTYPES\s*=\s*\("float64",\s*"float32"\)',
                     corruption)
    assert 'parametrize("dtype", BOTH_DTYPES)' in corruption
    # All five case families still contribute to the one matrix.
    for family in ("archive_and_manifest_cases", "model_section_cases",
                   "optimizer_section_cases", "generator_section_cases",
                   "metadata_cases"):
        assert f"{family}(" in corruption, family
    # The matrix keeps a floor on its own size, so it cannot silently
    # shrink to a handful of cases.
    floor = re.search(r"assert len\(cases\) >= (\d+)", corruption)
    assert floor and int(floor.group(1)) >= 110, (
        "the corruption matrix lost its size floor")
    # The complete-world fingerprint after every rejection, and its own
    # non-vacuity control.
    assert "fingerprint" in corruption
    assert "assert_unchanged" in corruption
    # Genuine v1, v2, and v3 archives still load in the same module, so the
    # matrix cannot pass by rejecting everything.
    assert 'parametrize("version", (1, 2, 3))' in corruption


def test_no_benchmark_test_asserts_a_duration_or_writes_a_result_file():
    """The project's oldest performance rule, re-proved at closure over
    every benchmark-facing suite."""
    suspicious = re.compile(
        r"assert\s+[^\n]{0,80}?\b(elapsed|duration|seconds|perf_counter|"
        r"speedup|ratio)\b[^\n]{0,40}?[<>]=?\s*(?!0(\.0+)?\b)[0-9]", re.I)
    for name in ("tests/test_native_dtype_benchmark.py",
                 "tests/test_native_cpu_performance_benchmark.py",
                 "tests/test_benchmarks.py"):
        text = _read(name)
        for match in suspicious.finditer(text):
            line = text[:match.start()].count("\n") + 1
            raise AssertionError(
                f"{name}:{line} asserts a duration: {match.group(0)!r}")
    # The I10 harness writes nothing, in any mode.
    tree = ast.parse(_read(I10_BENCHMARK))
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        name = getattr(node.func, "attr", getattr(node.func, "id", ""))
        assert name not in {"write_text", "write_bytes", "savez", "savetxt",
                            "to_csv"}, f"{I10_BENCHMARK} writes a file"
        if name == "open":
            for argument in node.args[1:]:
                if isinstance(argument, ast.Constant) and isinstance(
                        argument.value, str):
                    assert "w" not in argument.value, I10_BENCHMARK
                    assert "a" not in argument.value, I10_BENCHMARK
    # ...and no CLI option names a destination. ``--json`` is deliberately
    # allowed and deliberately not an exception: it is a *stdout* format
    # flag, so it takes no path and produces no artifact. The banned shape
    # is an option that receives one — ``--json-out PATH`` and its family.
    harness = _read(I10_BENCHMARK)
    for option in re.findall(r'add_argument\(\s*"(--[\w-]+)"[^)]*\)',
                             harness, re.S):
        assert not re.search(r"(out|output|file|path|report|save|dest)$",
                             option), f"{I10_BENCHMARK} offers {option}"
    assert '"--json", action="store_true"' in harness, (
        "--json is no longer a plain stdout format flag")


# ===========================================================================
# 8. The shallow-clone benchmark guard
# ===========================================================================
#
# Phase I already produced one false local-only guard, so closure asserts
# CI portability structurally rather than trusting it.

def _phase_i_module():
    sys.path.insert(0, str(REPO_ROOT / "tests"))
    try:
        import test_native_phase_i as module
    finally:
        sys.path.pop(0)
    return module


def test_the_benchmark_guard_is_frozen_content_rather_than_git_history():
    """The correction this phase had to make, pinned so it cannot be
    undone: seven inherited digests, one approved addition, and no
    historical object anywhere in the call chain."""
    module = _phase_i_module()

    assert len(module.I0_BENCHMARK_DIGESTS) == INHERITED_BENCHMARK_COUNT
    assert set(module.I10_ADDED_BENCHMARKS) == {I10_BENCHMARK}
    assert not (set(module.I0_BENCHMARK_DIGESTS)
                & set(module.I10_ADDED_BENCHMARKS))
    for relative, digest in module.I0_BENCHMARK_DIGESTS.items():
        assert relative.startswith("benchmarks/") and relative.endswith(".py")
        assert len(digest) == 64 and set(digest) <= set("0123456789abcdef")
    # Every digest distinct, so no two entries were pasted from one file.
    assert len(set(module.I0_BENCHMARK_DIGESTS.values())) == (
        INHERITED_BENCHMARK_COUNT)

    # The call chain, parsed rather than grepped: the guard's own prose
    # legitimately discusses ``git show`` and ``ls-tree``, so a raw
    # substring scan would trip on the sentences explaining their absence.
    def executable_source(function):
        import textwrap

        tree = ast.parse(textwrap.dedent(inspect.getsource(function)))
        for node in ast.walk(tree):
            if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef,
                                 ast.AsyncFunctionDef)) and node.body:
                first = node.body[0]
                if (isinstance(first, ast.Expr)
                        and isinstance(first.value, ast.Constant)
                        and isinstance(first.value.value, str)):
                    node.body = node.body[1:] or [ast.Pass()]
        return ast.unparse(tree)

    chain = "".join(executable_source(function) for function in (
        module.test_the_phase_h_benchmark_harness_is_untouched,
        module._benchmark_digests,
        module._classify_benchmarks,
        module._content_digest,
        module._normalized,
    ))
    for banned in ("git", "subprocess", "ls-tree", "rev-parse", "I0_COMMIT",
                   "_changed_since", "urlopen", "socket", "open(", "write"):
        assert banned not in chain, (
            f"the benchmark guard reached for {banned!r}")
    # It carries no skip at all, so it cannot degrade into a vacuous pass.
    assert "skip" not in inspect.getsource(
        module.test_the_phase_h_benchmark_harness_is_untouched)


def test_the_benchmark_guard_still_passes_with_git_unreachable():
    """The behavioural half of the CI-portability proof, proved
    non-vacuous first — a run in which git stayed reachable would assert
    nothing."""
    import os

    module = _phase_i_module()
    saved = os.environ.get("PATH")
    try:
        os.environ["PATH"] = ""
        try:
            probe = subprocess.run(["git", "--version"],
                                   capture_output=True, env={"PATH": ""})
            reachable = probe.returncode == 0
        except OSError:
            reachable = False
        assert not reachable, (
            "git stayed reachable, so this half of the proof is vacuous")
        module.test_the_phase_h_benchmark_harness_is_untouched()
    finally:
        if saved is None:                             # pragma: no cover
            os.environ.pop("PATH", None)
        else:
            os.environ["PATH"] = saved


def test_the_benchmark_guard_still_detects_drift_without_any_repository():
    """The no-``.git`` behavioural control: the guard classifies a
    temporary directory it has never seen, so its verdict comes from
    content rather than from the checkout."""
    module = _phase_i_module()
    import tempfile

    with tempfile.TemporaryDirectory() as raw:
        staging = Path(raw) / "benchmarks"
        staging.mkdir()
        for relative in module.I0_BENCHMARK_DIGESTS:
            name = relative.split("/", 1)[1]
            (staging / name).write_bytes((REPO_ROOT / relative).read_bytes())
        (staging / "benchmark_native_dtype.py").write_bytes(
            (REPO_ROOT / I10_BENCHMARK).read_bytes())

        # Clean, with no .git anywhere above it.
        assert not (Path(raw) / ".git").exists()
        modified, deleted, unexpected, approved = module._classify_benchmarks(
            module._benchmark_digests(staging))
        assert (modified, deleted, unexpected) == ([], [], [])
        assert approved == sorted(module.I10_ADDED_BENCHMARKS)

        # ...and a single appended byte is still caught there.
        victim = staging / "benchmark_native_cpu_performance.py"
        victim.write_bytes(victim.read_bytes() + b"\n# drift\n")
        modified, _, _, _ = module._classify_benchmarks(
            module._benchmark_digests(staging))
        assert modified == ["benchmarks/benchmark_native_cpu_performance.py"]


def test_the_workflow_is_unchanged_and_needs_no_history():
    """No ``fetch-depth: 0`` was added, and nothing in CI asks for a
    complete clone."""
    workflow = WORKFLOW.read_text(encoding="utf-8")
    assert "actions/checkout" in workflow
    assert "fetch-depth" not in workflow, (
        "the workflow now configures clone depth; the guards must not need it")
    assert "threshold" not in workflow.lower()
    # No closure test may reintroduce a historical-object lookup.
    closure = Path(__file__).read_text(encoding="utf-8")
    tree = ast.parse(closure)
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        arguments = [a for a in node.args if isinstance(a, ast.List)]
        for argument in arguments:
            literals = [element.value for element in argument.elts
                        if isinstance(element, ast.Constant)]
            if "git" in literals:
                assert not ({"show", "ls-tree", "cat-file", "rev-list"}
                            & set(literals)), literals


def test_the_two_history_reading_guards_still_skip_rather_than_pass():
    """The asymmetry, re-recorded at closure: the two whole-tree guards
    genuinely need history and degrade to a documented skip, while the
    benchmark guard needs none."""
    module = _phase_i_module()
    source = inspect.getsource(module._changed_since)
    assert "pytest.skip" in source
    for guard in (module.test_the_phase_changed_no_ci_or_dependency_file,
                  module.
                  test_the_phase_touched_only_the_python_modules_its_scope_names):
        assert "_changed_since" in inspect.getsource(guard), guard.__name__


# ===========================================================================
# 9. Examples and benchmarks
# ===========================================================================

def test_the_example_inventory_still_carries_phase_is_fifteen():
    """Phase I closed at **fifteen** examples, and I9's is one of them.

    The tree may hold more now — later phases ship their own — so the
    equality is stated as "fifteen, plus exactly the examples later
    milestones added, each named". That keeps I11's record historically
    exact while still failing on an unannounced example."""
    examples = sorted((REPO_ROOT / "examples").glob("*.py"))
    names = [path.name for path in examples if path.name != "__init__.py"]
    assert "native_float32_training.py" in names
    assert set(POST_PHASE_I_EXAMPLES) <= set(names), sorted(
        set(POST_PHASE_I_EXAMPLES) - set(names))
    assert len(names) == CURRENT_EXAMPLE_COUNT, names
    assert len(names) - len(POST_PHASE_I_EXAMPLES) == FINAL_EXAMPLE_COUNT


def test_the_float32_example_uses_only_public_construction():
    """I9 switched its one ingress helper to the public constructor, and
    the example is the phase's public-facing proof — a private ``_typed``
    call there would make the promise untested.

    Parsed rather than grepped, so the module's prose may explain the
    private constructors at any length while the executable half may not
    call one."""
    source = _read(I9_EXAMPLE)
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute):
            assert not node.attr.startswith("_typed"), node.attr
            assert node.attr != "_from_core", node.attr
        if isinstance(node, ast.Name):
            assert not node.id.startswith("_typed"), node.id
    # ...and it really does use the public one.
    assert "NativeTensor.from_array" in source
    assert re.search(r"from_array\([^)]*dtype=", source)


def test_every_example_is_a_tracked_source_file():
    tracked = set(_tracked_files())
    for path in sorted((REPO_ROOT / "examples").glob("*.py")):
        relative = f"examples/{path.name}"
        assert relative in tracked, relative


def test_the_benchmark_inventory_is_the_inherited_set_plus_one():
    """Phase I's own delta is **exactly one** benchmark, I10's. Later
    phases' harnesses are named individually rather than folded into a
    bumped literal, so that claim about Phase I stays checkable."""
    names = sorted(path.name for path in
                   (REPO_ROOT / "benchmarks").glob("*.py"))
    assert len(names) == CURRENT_BENCHMARK_COUNT, names
    assert "benchmark_native_dtype.py" in names
    assert "benchmark_native_cpu_performance.py" in names
    for name in POST_PHASE_I_BENCHMARKS:
        assert name in names, name
    # Phase I's own contribution, stated apart from every later one.
    inherited_plus_i10 = [name for name in names
                          if name not in POST_PHASE_I_BENCHMARKS]
    assert len(inherited_plus_i10) == INHERITED_BENCHMARK_COUNT + 1


def test_the_phase_h_harness_case_inventory_is_still_pinned_as_history():
    """Phase H's harness inventory is pinned by test as "the H0 set". If a
    dtype axis had been added to it, every published Phase-H number would
    silently mean something else."""
    pinned = _read("tests/test_native_cpu_performance_benchmark.py")
    assert re.search(r"H0", pinned)
    assert re.search(r"I10_ADDED|later phase|added", pinned, re.I)


# ===========================================================================
# 10. Stable / native isolation
# ===========================================================================

def test_importing_the_stable_framework_does_not_load_the_native_backend():
    code = (
        "import sys\n"
        "import tensorforge, tensorforge.nn, tensorforge.optim\n"
        "loaded = [m for m in sys.modules if m.endswith('backends.cpp')]\n"
        "assert not loaded, loaded\n"
        "print('isolated')\n"
    )
    done = subprocess.run([sys.executable, "-c", code], cwd=REPO_ROOT,
                          capture_output=True, text=True)
    assert done.returncode == 0, done.stderr
    assert "isolated" in done.stdout


def test_the_backend_still_reports_no_stable_integration():
    assert cpp.backend_info()["stable_framework_integration"] is False


def test_the_stable_public_api_gained_no_dtype_surface():
    """A dtype phase in the native line must leave the stable line's
    public API exactly as it found it."""
    import tensorforge

    for name in ("NativeTensor", "NativeLinear", "NativeAdam",
                 "save_native_checkpoint", "normalize_dtype"):
        assert not hasattr(tensorforge, name), name
    for name in ("set_default_dtype", "float32", "float64"):
        assert not hasattr(tensorforge, name), name


def test_no_environment_variable_steers_the_backend_or_the_dtype():
    tree = ast.parse(_read("src/tensorforge/backends/cpp.py"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and node.attr in {"getenv",
                                                             "environ"}:
            raise AssertionError("backends/cpp.py consults the environment")
        if isinstance(node, ast.Name) and node.id == "environ":
            raise AssertionError("backends/cpp.py consults the environment")


# ===========================================================================
# 11. The unsupported boundaries, stated as claims rather than as tokens
# ===========================================================================
#
# Context-aware: an accurate sentence saying CUDA is *not* supported must
# pass, and only a claim that it *is* may fail.

_OVERCLAIMS = (
    ("CUDA or a GPU backend is supported",
     r"\b(CUDA|GPU|ROCm|Metal)\b[^.;]{0,60}?\b(is|are|now)\s+"
     r"(supported|available|implemented|shipped|working|enabled)\b"
     r"|\bsupports?\s+(CUDA|GPU)\b"),
    ("AMP or mixed precision is supported",
     r"\b(AMP|mixed[- ]precision|autocast)\b[^.;]{0,60}?\b(is|are|now)\s+"
     r"(supported|available|implemented|shipped|working|enabled)\b"
     r"|\bsupports?\s+(AMP|mixed[- ]precision)\b"),
    ("a dtype beyond float32 and float64 is supported",
     r"\b(float16|bfloat16|float128|complex64|int(8|16|32|64) tensors?)\b"
     r"[^.;]{0,60}?\b(is|are|now)\s+"
     r"(supported|available|implemented|shipped|working)\b"
     r"|\bsupports?\s+(float16|bfloat16|integer tensors)\b"),
    ("casting or promotion exists",
     r"\b(casting|dtype promotion|type promotion|automatic conversion)\b"
     r"[^.;]{0,60}?\b(is|are|now)\s+"
     r"(supported|available|implemented|shipped|performed|applied)\b"
     r"|\b(astype|map_location)\b[^.;]{0,40}?\b(is|are)\s+(supported|"
     r"available|accepted)\b"),
    ("device movement exists",
     r"\b(device transfer|device movement|\.to\(|\.cuda\(\))\b"
     r"[^.;]{0,60}?\b(is|are|now)\s+(supported|available|implemented)\b"),
    ("the raw utility kernels support float32",
     r"\braw (utility )?kernels?\b[^.;]{0,60}?\b(support|accept|are)\b"
     r"[^.;]{0,30}?\bfloat32\b"),
    ("the stable framework dispatches to the native backend",
     r"\bstable\b[^.;]{0,60}?\b(automatically|implicitly)\s+"
     r"(uses|dispatches|selects|falls back)\b"),
)

_NEGATIONS = re.compile(
    r"\b(no|not|never|neither|nor|none|without|absent|unsupported|"
    r"planned|future|beyond|until|once|when|will|would|before|yet|"
    r"rejected|refused|forbidden|outside)\b", re.I)


def _overclaims(text):
    """Every overclaim in ``text``, as ``(label, matched span)``."""
    found = []
    for label, pattern in _OVERCLAIMS:
        for match in re.finditer(pattern, text, re.I):
            window = text[max(0, match.start() - 70):match.end() + 30]
            if not _NEGATIONS.search(window):
                found.append((label, match.group(0)))
    return found


@pytest.mark.parametrize("surface", STATUS_SURFACES + ("CLAUDE.md",))
def test_no_status_surface_overclaims_an_unsupported_boundary(surface):
    offenders = _overclaims(_flat(_read(surface)))
    assert offenders == [], f"{surface}: {offenders[:3]}"


def test_the_overclaim_scanner_detects_what_it_claims_to():
    """The scanner's non-vacuity control: each sentence below must be
    caught, and each accurate one must not be."""
    for caught in (
        "CUDA is supported on the native backend",
        "the runtime supports GPU",
        "AMP is now available",
        "mixed-precision is supported",
        "float16 is supported",
        "the backend supports integer tensors",
        "casting is supported between the two dtypes",
        "raw utility kernels accept float32",
    ):
        assert _overclaims(caught), caught
    for allowed in (
        "float32 is supported",
        "float32 and float64 are supported on the CPU",
        "CUDA and AMP remain unsupported",
        "there is no casting and no promotion",
        "float16 and bfloat16 are not supported",
        "the raw utility kernels stay float64 only",
        "device movement does not exist and none may be added",
        "AMP is outside the phase",
    ):
        assert _overclaims(allowed) == [], (allowed, _overclaims(allowed))


def test_the_unsupported_names_are_genuinely_unreachable():
    """The registry claim checked against behavior."""
    if not cpp.is_available():                        # pragma: no cover
        pytest.skip("the native library is not built")
    with pytest.raises((ValueError, TypeError)):
        cpp.NativeTensorCore.zeros((2, 2), device="cuda")
    for dtype in ("float16", "bfloat16", "int64", "complex64"):
        with pytest.raises((ValueError, TypeError)):
            cpp.NativeTensorCore.from_array(
                np.zeros((2, 2), dtype=np.float64), dtype=dtype)


@pytest.mark.skipif(not cpp.is_available(),
                    reason="the native library is not built")
def test_mixed_dtype_is_still_rejected_before_anything_is_allocated():
    """The rule that makes "no casting" observable rather than asserted."""
    wide = cpp.NativeTensorCore.from_array(np.ones((2, 2)), dtype="float64")
    narrow = cpp.NativeTensorCore.from_array(np.ones((2, 2)), dtype="float32")
    try:
        for left, right in ((wide, narrow), (narrow, wide)):
            with pytest.raises(ValueError):
                left.add(right)
            with pytest.raises(ValueError):
                left.matmul(right)
    finally:
        narrow.close()
        wide.close()


# ===========================================================================
# 12. Repository hygiene
# ===========================================================================

_ARTIFACT_SUFFIXES = (".so", ".dll", ".dylib", ".pyd", ".obj", ".o", ".lib",
                      ".pdb", ".exp", ".a", ".npz", ".npy", ".coverage",
                      ".swp", ".bak", ".orig", ".rej")


def test_no_generated_artifact_is_tracked():
    for path in _tracked_files():
        lowered = path.lower()
        assert not lowered.endswith(_ARTIFACT_SUFFIXES), path
        assert not re.search(r"(^|/)(build|dist|\.venv|venv|node_modules|"
                             r"__pycache__|\.pytest_cache|\.cache)/",
                             lowered), path
        assert "benchmark_result" not in lowered, path
        assert not re.search(r"(^|/)(asan|ubsan|lsan)[-_.]", lowered), path


def test_no_checkpoint_archive_or_result_file_is_tracked():
    for path in _tracked_files():
        lowered = path.lower()
        assert not lowered.endswith((".npz", ".ckpt", ".pt", ".pth")), path
        assert not re.search(r"(results?|timings?|measurements?)\.(json|csv|txt)$",
                             lowered), path


def test_every_tracked_text_file_is_valid_utf8_without_a_bom():
    """Committed bytes, so a CRLF checkout cannot create a false
    positive: line endings are deliberately not asserted."""
    textual = (".py", ".md", ".txt", ".toml", ".yml", ".yaml", ".cfg",
               ".ini", ".h", ".cpp", ".json")
    for path in _tracked_files():
        if not path.endswith(textual):
            continue
        data = (REPO_ROOT / path).read_bytes()
        assert not data.startswith(b"\xef\xbb\xbf"), f"{path} carries a BOM"
        try:
            data.decode("utf-8")
        except UnicodeDecodeError as error:           # pragma: no cover
            raise AssertionError(f"{path} is not valid UTF-8: {error}")


def test_no_compiled_library_sits_in_a_tracked_source_directory():
    """The built DLL/SO lands in ``src/tensorforge/backends/``; it must
    never be committed from there."""
    tracked = set(_tracked_files())
    for suffix in (".dll", ".so", ".dylib", ".pyd"):
        for path in (REPO_ROOT / "src" / "tensorforge" / "backends").glob(
                f"*{suffix}"):
            relative = path.relative_to(REPO_ROOT).as_posix()
            assert relative not in tracked, relative


# ===========================================================================
# 13. The agent instructions
# ===========================================================================

def test_claude_md_records_the_final_phase_i_truth():
    """Facts and pointers, never a phrasing or a length."""
    text = AGENT_INSTRUCTIONS.read_text(encoding="utf-8")
    for value in FINAL_DTYPES + FINAL_DEVICES + FINAL_UNSUPPORTED:
        assert value in text, value
    # Both export counts, because both are current facts of a different
    # kind: 54 is what the library exports, 52 is what Phase H closed at.
    assert str(FINAL_EXPORT_COUNT) in text
    assert str(PHASE_H_EXPORT_COUNT) in text
    assert re.search(rf"version\s*\**\s*{FINAL_CHECKPOINT_VERSION}\b", text)
    for version in FLOAT64_ONLY_CHECKPOINT_VERSIONS:
        assert str(version) in text, version
    for document in ("docs/native_dtype_float32_design.md",
                     "docs/native_support_matrix.md",
                     "docs/release_history.md",
                     "docs/roadmap.md"):
        assert document in text, document
    # The phase's status, and the boundary it stopped at.
    assert re.search(r"Phase I[^.]{0,80}complete", text, re.I)
    assert re.search(r"RAW_KERNEL_DTYPES", text)


def test_claude_md_stayed_inside_the_project_memory_budget():
    """A soft structural bound: closure adds status, not a transcript.

    The ceiling is the 150,000-character project-memory limit, and it is
    a ceiling rather than a target — an active phase may grow the file
    with operational detail an implementer genuinely needs. What it may
    not grow with is milestone history."""
    size = len(AGENT_INSTRUCTIONS.read_text(encoding="utf-8"))
    assert size < 150_000, (
        f"CLAUDE.md has grown to {size} characters; milestone history "
        f"belongs in docs/, not in project memory")


def test_the_design_is_linked_from_the_readme_and_the_instructions():
    for surface in ("README.md", "CLAUDE.md"):
        assert "docs/native_dtype_float32_design.md" in _read(surface), surface
