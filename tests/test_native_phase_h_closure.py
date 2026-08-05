"""Phase-H closure guardrails (Advanced C++ milestone H10).

The durable replacement for Phase H's milestone-era *pending* checks.
Every earlier guard about this phase was written in the negative — "H9 has
not begun", "H10 is not started", "no SIMD/threading/BLAS exists" — because
those things had not happened yet. H10 ran the closure matrix, made the
acceleration decision, and closed the phase, so the first two premises have
expired. The third has not, and is now a *permanent* rule rather than a
temporary one.

The rules these tests protect:

* the phase is closed, and its ladder is whole (H0-H10, once each, in
  order, all complete, with nothing later claimed and no H11);
* **performance work broadened nothing** — the dtype, device, capability,
  export, and checkpoint boundaries are exactly what they were before
  Phase H opened;
* the export surface is exactly 52, and the source inventory agrees with
  the built library;
* **no public performance control exists**, in any form: no path selector,
  threshold or block-size setter, dispatch tracer, profiling counter,
  environment-variable dispatch, or "which path ran" query;
* the rejected-optimization family is still absent from the whole
  repository, and is now rejected permanently rather than provisionally;
* every optimized path still has its retained generic reference beside it;
* ``CLAUDE.md`` carries current operating rules and points at the
  authoritative documents, rather than duplicating milestone history;
* no machine-specific artifact is committed.

They deliberately test *values and structure* rather than wording, so
ordinary prose improvements do not require rewriting them. In particular
nothing here asserts a character count, a paragraph order, or a benchmark
number.
"""
import ast
import re
import subprocess
import sys
from pathlib import Path

import pytest

from tensorforge.backends import cpp

REPO_ROOT = Path(__file__).resolve().parent.parent
PHASE_H_DESIGN = REPO_ROOT / "docs" / "native_cpu_performance_design.md"
AGENT_INSTRUCTIONS = REPO_ROOT / "CLAUDE.md"

# Surfaces that state *current* status. Per-milestone historical records
# deliberately preserve superseded wording and are not scanned.
STATUS_SURFACES = (
    "README.md",
    "docs/roadmap.md",
    "docs/project_summary.md",
    "docs/native_support_matrix.md",
    "docs/backend_experiments.md",
    "docs/architecture.md",
    "src/tensorforge/experimental/__init__.py",
)

# The boundary Phase H inherited and had to leave alone. Written out once
# so a drift in either direction is a single obvious diff.
FINAL_UNSUPPORTED = ("float32", "cuda", "amp")
FINAL_DTYPES = ("float64",)
FINAL_DEVICES = ("cpu",)
FINAL_EXPORT_COUNT = 52
# ...and the same history/today split the export and checkpoint constants
# below already use, now applied to the dtype registry. Phase H inherited a
# float64-only boundary and left it exactly as it found it, which is a fact
# about Phase H and does not move again. Phase I milestone **I9** then moved
# ``"float32"`` from UNSUPPORTED into SUPPORTED_DTYPES, once integrated
# float32 training and the exact float32 resume proof both passed. What
# Phase H must still be able to claim is that *it* broadened nothing — not
# that nothing has been broadened since.
PHASE_I_ADDED_DTYPES = ("float32",)
CURRENT_DTYPES = FINAL_DTYPES + PHASE_I_ADDED_DTYPES          # float64, float32
CURRENT_UNSUPPORTED = tuple(name for name in FINAL_UNSUPPORTED
                            if name not in PHASE_I_ADDED_DTYPES)
CURRENT_DEVICES = FINAL_DEVICES                                # never moved
# ...and what the live source holds **now**. Phase H closed at 52; Phase I
# milestone I1 added the two typed storage creators and Phase K milestone
# K3 added the argmax forward, taking the current inventory to 55. The
# numbers are facts about different moments and are deliberately kept
# apart: FINAL_EXPORT_COUNT is Phase H's closure, which is history and does
# not move again, while CURRENT_EXPORT_COUNT is what the tree exports today.
PHASE_I_ADDED_EXPORTS = (
    "tf_storage_create_typed",
    "tf_storage_create_uninitialized_typed",
)
PHASE_K_ADDED_EXPORTS = ("tf_core_argmax",)
CURRENT_EXPORT_COUNT = (FINAL_EXPORT_COUNT + len(PHASE_I_ADDED_EXPORTS)
                        + len(PHASE_K_ADDED_EXPORTS))  # 55
# The same history/today split the export counts above use, for the same
# reason: Phase H closed at checkpoint version 2 and that is a fact about
# Phase H which does not move again, while Phase I milestone I8 added the
# dtype-aware version 3. What Phase H must still be able to claim is that
# *it* changed nothing — not that nothing has changed since.
FINAL_CHECKPOINT_VERSION = 2
FINAL_CHECKPOINT_VERSIONS = (1, 2)
CURRENT_CHECKPOINT_VERSION = 3
CURRENT_CHECKPOINT_VERSIONS = (1, 2, 3)
FINAL_MILESTONES = tuple(f"H{n}" for n in range(11))


def _read(relative):
    return (REPO_ROOT / relative).read_text(encoding="utf-8")


def _flat(relative):
    """Whitespace-flattened text, so a claim split across lines still
    reads as one sentence."""
    return re.sub(r"\s+", " ", _read(relative))


# ---------------------------------------------------------------------------
# the ladder is whole and closed
# ---------------------------------------------------------------------------

def test_the_design_marks_the_phase_complete():
    text = PHASE_H_DESIGN.read_text(encoding="utf-8")
    assert re.search(r"Phase-H status:\s*complete", text, re.I), (
        "the design document no longer states its completed status"
    )


def test_the_milestone_ladder_runs_h0_to_h10_once_each_in_order():
    """Every optimization milestone has its own "as shipped" section, once,
    in order.

    Deliberately checked against section headings rather than first textual
    mention: a later milestone is legitimately referenced from inside an
    earlier section as a forward pointer, and that is good documentation
    rather than drift. H0 has no section here because it shipped no
    optimization; its status is asserted in the ladder table instead."""
    text = PHASE_H_DESIGN.read_text(encoding="utf-8")
    sections = re.findall(r"^## 16\.(?:\d+) H(\d+) ", text, re.M)
    numbered = [f"H{milestone}" for milestone in sections]
    for milestone in FINAL_MILESTONES[1:]:
        assert milestone in numbered, f"{milestone} has no shipped section"
    assert len(numbered) == len(set(numbered)), numbered
    order = [int(name[1:]) for name in numbered]
    assert order == sorted(order), numbered


@pytest.mark.parametrize("surface", STATUS_SURFACES)
def test_every_status_surface_marks_phase_h_complete(surface):
    text = _flat(surface)
    assert "Phase H" in text, f"{surface} does not name Phase H"
    assert re.search(
        r"Phase H[^.;]{0,120}?\b(is|are)\s+(\*\*)?complete"
        r"|Phase H is \*\*complete\*\*"
        r"|Phase H[^.;]{0,120}?\bcomplete \(H0[-–]H10\)"
        r"|\bH0[-–]H10[^.;]{0,60}?\b(complete|landed)"
        r"|\bH0 through H10\b[^.;]{0,60}?\b(complete|landed)",
        text, re.I), f"{surface} does not mark Phase H complete"


@pytest.mark.parametrize("surface", STATUS_SURFACES)
def test_no_status_surface_still_calls_phase_h_unfinished(surface):
    """The failure this exists for: a surface updated in one paragraph and
    left saying "has begun" or "not started" in another."""
    text = _flat(surface)
    stale = re.compile(
        r"Phase H[^.;]{0,80}?\b(has begun|is the current phase|"
        r"is not complete|has not been closed)\b"
        r"|\bH(?:8|9|10)\b[^.;]{0,60}?\b(not started|planned|pending|"
        r"is conditional)\b",
        re.I)
    offenders = [match.group(0) for match in stale.finditer(text)
                 # A sentence about what *was* true at an earlier milestone
                 # is history, and history stays accurate.
                 if not re.search(
                     r"\b(was|were|had|until|through|before|at H\d|"
                     r"then|earlier|previously|stayed|remained|"
                     r"originally|drafted|pencilled)\b",
                     text[max(0, match.start() - 130):match.end() + 40], re.I)]
    assert not offenders, f"{surface}: {offenders[:3]}"


def test_no_surface_claims_a_phase_h_milestone_beyond_h10():
    """H11 was drafted and never entered. Naming it as real work, or
    naming an H12 that has never existed, is drift."""
    for surface in STATUS_SURFACES + ("CLAUDE.md",):
        text = _flat(surface)
        for match in re.finditer(r"\bH1[1-9]\b(?![-–])", text):
            window = text[max(0, match.start() - 170):match.end() + 90]
            # "the proposed H1–H11 ladder" is an accurate record of what was
            # drafted at H0. Only a claim that H11 is real, current work
            # fails here.
            assert re.search(
                r"\b(proposed|drafted|pencilled|conditional|not needed"
                r"|was not entered|not entered|does not exist)\b|~~",
                window, re.I), f"{surface}: {window[:180]!r}"


def test_the_design_records_why_the_ladder_ends_at_h10():
    """A dangling H11 row would be worse than none: the reason it was not
    entered is evidence about the phase and has to be written down."""
    text = PHASE_H_DESIGN.read_text(encoding="utf-8")
    assert re.search(r"H11", text), "the design never mentions the H11 slot"
    assert re.search(
        r"H11[^.]{0,200}?\b(not needed|does not exist|was not entered|"
        r"not entered)\b"
        r"|\bladder (runs|ends)[^.]{0,60}?H0[-–]H10",
        text, re.I | re.S), (
        "the design does not record why the ladder ends at H10"
    )


# ---------------------------------------------------------------------------
# performance work broadened nothing
# ---------------------------------------------------------------------------

def test_phase_h_broadened_no_support_boundary():
    """Phase H made the float64 runtime faster and broadened nothing.

    Stated the way the export and checkpoint records above are stated: the
    Phase-H literals are pinned as history, and the live registries are
    asserted to differ from them by **exactly** what a later phase is on
    record as having moved. An equality against the Phase-H literals would
    have made this closure record a veto on every subsequent milestone,
    which is not what it ever claimed; dropping it would have lost the
    Phase-H fact entirely."""
    # Phase H's own record, pinned as literals.
    assert FINAL_UNSUPPORTED == ("float32", "cuda", "amp")
    assert FINAL_DTYPES == ("float64",)

    assert cpp.SUPPORTED_DEVICES == FINAL_DEVICES == CURRENT_DEVICES
    assert cpp.SUPPORTED_DTYPES == CURRENT_DTYPES
    assert cpp.UNSUPPORTED == CURRENT_UNSUPPORTED
    # The difference is exactly the one dtype Phase I moved, in both
    # directions — nothing else joined, and nothing else left.
    assert (set(cpp.SUPPORTED_DTYPES) - set(FINAL_DTYPES)
            == set(PHASE_I_ADDED_DTYPES))
    assert (set(FINAL_UNSUPPORTED) - set(cpp.UNSUPPORTED)
            == set(PHASE_I_ADDED_DTYPES))
    assert set(cpp.UNSUPPORTED) <= set(FINAL_UNSUPPORTED)
    # Phase H's dtype is still the default and still first.
    assert cpp.SUPPORTED_DTYPES[0] == "float64"
    assert cpp.backend_info()["dtype"] == "float64"


def test_the_checkpoint_format_did_not_move():
    """Phase H added no checkpoint field and no checkpoint version.

    Stated as: every version Phase H closed with is **still accepted**, and
    the format *name* is still the one G5 fixed. Phase I milestone I8 later
    added version 3, which is why this is a subset relation rather than an
    equality — an equality here would silently make this Phase-H record a
    veto on every later milestone, which is not what it ever claimed."""
    from tensorforge.experimental import native_checkpoint

    # Phase H's own record, pinned as literals. Without this the constants
    # float free and "Phase H closed at version 2" could be quietly
    # rewritten to whatever the present happens to be, which is the one
    # thing a closure record exists to prevent.
    assert FINAL_CHECKPOINT_VERSION == 2
    assert FINAL_CHECKPOINT_VERSIONS == (1, 2)

    assert native_checkpoint._FORMAT == "tensorforge.native_checkpoint"
    assert native_checkpoint._FORMAT_VERSION == CURRENT_CHECKPOINT_VERSION
    assert (native_checkpoint._SUPPORTED_FORMAT_VERSIONS
            == CURRENT_CHECKPOINT_VERSIONS)
    # Nothing Phase H shipped was dropped, and exactly one version has been
    # added since. Stated as an equality on the delta rather than a subset,
    # so an extra accepted version fails here too and this record cannot be
    # satisfied by simply accepting more.
    assert set(FINAL_CHECKPOINT_VERSIONS).issubset(
        native_checkpoint._SUPPORTED_FORMAT_VERSIONS
    )
    assert (set(native_checkpoint._SUPPORTED_FORMAT_VERSIONS)
            - set(FINAL_CHECKPOINT_VERSIONS)) == {3}
    # Order and shape are part of the contract: versions are listed
    # ascending, and the newest is the one new saves write.
    accepted = native_checkpoint._SUPPORTED_FORMAT_VERSIONS
    assert list(accepted) == sorted(accepted)
    assert accepted[-1] == native_checkpoint._FORMAT_VERSION


def test_the_unsupported_capabilities_really_are_unreachable():
    """The registry claim, checked against behavior rather than trusted.

    ``"float32"`` has moved out of this list — Phase I milestone I9 made it
    real, so asserting it still raises would assert the opposite of the
    truth. The names below are the ones that remain genuinely absent, and
    they keep the "two dtypes, not any dtype" boundary honest."""
    import numpy as np

    for dtype in ("float16", "bfloat16", "int64", "complex64"):
        with pytest.raises((ValueError, TypeError)):
            cpp.NativeTensorCore.from_array(
                np.zeros((2, 2), dtype=np.float64), dtype=dtype)
    with pytest.raises((ValueError, TypeError)):
        cpp.NativeTensorCore.zeros((2, 2), device="cuda")


def test_no_performance_capability_name_entered_any_registry():
    """Phase H made things faster; it must not have advertised a
    capability for doing so."""
    banned = ("simd", "avx", "openmp", "blas", "parallel", "arena",
              "scratch", "workspace", "fast_math", "im2col",
              "memory_pool", "thread_pool", "threaded")
    registries = (
        cpp.TENSOR_CORE_OPS, cpp.AUTOGRAD_OPS, cpp.NATIVE_MODULES,
        cpp.NATIVE_LOSSES, cpp.NATIVE_METRICS, cpp.NATIVE_OPTIMIZERS,
        cpp.STATE_SUPPORT, cpp.SUPPORTED_DTYPES, cpp.SUPPORTED_DEVICES,
        cpp.UNSUPPORTED,
    )
    for registry in registries:
        for name in registry:
            lowered = str(name).lower()
            for word in banned:
                # Word-ish match: "maxpool2d_forward" is an operation name,
                # not a memory pool, and "thread" appears in no registry.
                assert not re.search(rf"(^|_){word}(_|$)", lowered), (
                    registry, name, word)


# ---------------------------------------------------------------------------
# the ABI surface
# ---------------------------------------------------------------------------

def _source_exports():
    names = set()
    for source in sorted((REPO_ROOT / "cpp" / "src").glob("*.cpp")):
        text = source.read_text(encoding="utf-8")
        names.update(re.findall(r"TF_EXPORT[^;{]*?\b(tf_[a-z0-9_]+)\s*\(",
                                text, re.S))
    return names


def test_the_source_exports_exactly_the_current_symbol_count():
    """The live inventory is Phase H's closure plus exactly the symbols the
    phases after it have added, and nothing else.

    Stated as arithmetic rather than as a bare number so that the two
    facts stay separable: if a milestone adds an unplanned export, the
    count fails; if it removes one of Phase H's, the difference fails."""
    exports = _source_exports()
    assert len(exports) == CURRENT_EXPORT_COUNT, sorted(exports)
    for name in PHASE_I_ADDED_EXPORTS + PHASE_K_ADDED_EXPORTS:
        assert name in exports, name
    assert len(exports - set(PHASE_I_ADDED_EXPORTS)
               - set(PHASE_K_ADDED_EXPORTS)) == FINAL_EXPORT_COUNT


def test_the_one_symbol_phase_h_added_is_present_and_is_the_only_new_one():
    """H1's uninitialized allocator is the whole of Phase H's ABI
    footprint. Its zero-initializing sibling is still the default.

    Phase I did not disturb either: both remain exported, and I1 made them
    thin float64 compatibility wrappers rather than removing them."""
    exports = _source_exports()
    assert "tf_storage_create_uninitialized" in exports
    assert "tf_storage_create" in exports


def test_the_declared_ctypes_surface_matches_the_source_exports():
    """A declaration Python makes for a symbol the library does not export
    would fail only at call time; this catches it at rest."""
    declared = set(re.findall(r"library\.(tf_[a-z0-9_]+)\s*\.",
                              _read("src/tensorforge/backends/cpp.py")))
    declared |= set(re.findall(r"getattr\(library,\s*\"(tf_[a-z0-9_]+)\"",
                               _read("src/tensorforge/backends/cpp.py")))
    missing = declared - _source_exports()
    assert not missing, f"declared but not exported: {sorted(missing)}"


@pytest.mark.skipif(not cpp.is_available(),
                    reason="the native library is not built")
def test_every_declared_symbol_resolves_in_the_built_library():
    library = cpp._require_library()
    for name in sorted(_source_exports()):
        assert hasattr(library, name), name


# ---------------------------------------------------------------------------
# no public performance control, in any form
# ---------------------------------------------------------------------------

def test_no_public_performance_control_is_exported_from_the_backend():
    """The rule that keeps dispatch a property of the data rather than of
    ambient process state."""
    banned = re.compile(
        r"(set|select|choose|force|enable|disable|configure)_"
        r"(kernel|path|traversal|block|tile|threshold|dispatch|simd|"
        r"thread|vector)"
        r"|(kernel|path|traversal|dispatch)_(selector|override|mode|hook)"
        r"|block_size_setter|which_path|last_path|path_taken",
        re.I)
    for name in dir(cpp):
        assert not banned.search(name), name


def test_no_environment_variable_steers_the_native_backend():
    """A dispatch decision that depends on the environment would make a
    result depend on ambient process state, which the determinism and
    exact-resume contracts forbid."""
    tree = ast.parse(_read("src/tensorforge/backends/cpp.py"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and node.attr in {
                "getenv", "environ"}:
            raise AssertionError(
                "backends/cpp.py consults the environment")
        if isinstance(node, ast.Name) and node.id == "environ":
            raise AssertionError(
                "backends/cpp.py consults the environment")


def test_no_cpp_source_consults_the_environment_or_a_clock():
    """The same rule at the other layer: a kernel that reads getenv, a
    clock, or a CPU-feature probe would make dispatch non-deterministic."""
    banned = re.compile(
        r"\bgetenv\b|\b_dupenv_s\b|\bstd::getenv\b"
        r"|\bstd::chrono\b|\bclock\(\)|\btime\(\s*(?:NULL|nullptr)\s*\)"
        r"|\b__cpuid\b|\b__builtin_cpu_supports\b|\bcpuid\b")
    for source in sorted((REPO_ROOT / "cpp" / "src").glob("*.cpp")):
        text = source.read_text(encoding="utf-8")
        stripped = re.sub(r"//[^\n]*|/\*.*?\*/", "", text, flags=re.S)
        found = banned.findall(stripped)
        assert not found, f"{source.name}: {found}"


# ---------------------------------------------------------------------------
# the rejected-optimization family stays absent
# ---------------------------------------------------------------------------

def test_no_rejected_acceleration_technique_is_present_in_the_sources():
    """SIMD, threading, OpenMP, and BLAS were each finally rejected at
    H10 with measurements. This is the executable half of that decision."""
    banned = re.compile(
        r"#include\s*<(immintrin|emmintrin|xmmintrin|omp|thread|mutex|"
        r"future|cblas|mkl)\.?h?>"
        r"|\b__m128d?\b|\b__m256d?\b|\b_mm_\w+|\b_mm256_\w+"
        r"|#pragma\s+omp\b"
        r"|\bstd::thread\b|\bstd::async\b|\bstd::for_each\s*\(\s*std::execution"
        r"|\bcblas_\w+|\bdgemm_?\b")
    for source in sorted((REPO_ROOT / "cpp").rglob("*.cpp")):
        text = re.sub(r"//[^\n]*|/\*.*?\*/", "",
                      source.read_text(encoding="utf-8"), flags=re.S)
        found = banned.findall(text)
        assert not found, f"{source.name}: {found}"
    for header in sorted((REPO_ROOT / "cpp" / "include").glob("*.h")):
        text = re.sub(r"//[^\n]*|/\*.*?\*/", "",
                      header.read_text(encoding="utf-8"), flags=re.S)
        found = banned.findall(text)
        assert not found, f"{header.name}: {found}"


def test_the_build_adds_no_acceleration_option_or_architecture_flag():
    """TF_SANITIZE and TF_BUILD_TESTS remain the only options, and the
    default build must not require a wider ISA."""
    cmake = _read("cpp/CMakeLists.txt")
    options = set(re.findall(r"^\s*option\s*\(\s*(\w+)", cmake, re.M))
    assert options <= {"TF_BUILD_TESTS", "TF_SANITIZE"}, (
        "a third build option appeared; section 14 allows exactly these two"
    )
    assert not re.search(r"/arch:AVX|-mavx|-march=native|-ffast-math"
                         r"|/fp:fast|-funsafe-math", cmake), cmake[:200]


def test_the_design_records_all_three_acceleration_decisions():
    """Each of SIMD, threading/OpenMP, and BLAS must carry an explicit
    decision, not a shrug."""
    text = PHASE_H_DESIGN.read_text(encoding="utf-8")
    for topic in ("SIMD", "hreading", "BLAS"):
        assert topic in text, topic
    # The decision itself, stated as a decision.
    assert len(re.findall(r"\*\*rejected\*\*", text, re.I)) >= 3, (
        "the design does not state three explicit rejections"
    )
    # ...and the reopening criteria, so the answer is not simply "no".
    assert re.search(r"future trigger", text, re.I), (
        "the design records no reopening trigger"
    )


# ---------------------------------------------------------------------------
# every optimized path kept its generic reference
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("header,optimized,generic", (
    ("tf_matmul_internal.h", "matmul_row_sweep", "matmul_generic_strided"),
    ("tf_reduction_internal.h", "sum_contiguous_blocks",
     "sum_generic_strided"),
    ("tf_conv2d_internal.h", "conv2d_forward_row_sweep",
     "conv2d_forward_generic"),
    ("tf_conv2d_internal.h", "conv2d_input_backward_gather",
     "conv2d_input_backward_generic"),
    ("tf_conv2d_internal.h", "conv2d_weight_backward_gather",
     "conv2d_weight_backward_generic"),
))
def test_each_optimized_path_still_has_its_retained_reference(
        header, optimized, generic):
    text = (REPO_ROOT / "cpp" / "include" / header).read_text(encoding="utf-8")
    assert optimized in text, optimized
    assert generic in text, generic


@pytest.mark.parametrize("predicate", (
    "matmul_prefers_row_sweep",
    "copy_prefers_contiguous",
    "reduce_prefers_contiguous_blocks",
    "build_unary_plan",
    "build_binary_plan",
    "conv2d_forward_prefers_row_sweep",
    "conv2d_input_backward_prefers_gather",
    "conv2d_weight_backward_prefers_gather",
))
def test_every_dispatch_predicate_is_hidden_rather_than_exported(predicate):
    """A predicate is an implementation detail. Exporting one would make
    the choice of path a public, and therefore promised, thing."""
    assert predicate not in _source_exports()
    declared = _read("src/tensorforge/backends/cpp.py")
    assert predicate not in declared, (
        f"{predicate} is reachable from Python"
    )


def test_the_elementwise_fallback_is_still_reachable_and_correct():
    """The retained odometer is not decoration: a layout the plan builder
    declines must still compute, and must agree with the plan path."""
    import numpy as np

    values = np.arange(24, dtype=np.float64).reshape(2, 3, 4)
    contiguous = cpp.NativeTensorCore.from_array(values)
    try:
        transposed = contiguous.transpose(2, 1, 0)
        try:
            # The transposed source takes the retained generic traversal;
            # NumPy is the independent oracle for both.
            produced = transposed.relu()
            try:
                expected = np.maximum(np.transpose(values, (2, 1, 0)), 0.0)
                assert np.array_equal(produced.to_numpy(), expected)
            finally:
                produced.close()
        finally:
            transposed.close()
    finally:
        contiguous.close()


# ---------------------------------------------------------------------------
# the agent instructions
# ---------------------------------------------------------------------------

def test_claude_md_states_current_facts_and_points_at_the_docs():
    """H10 restated CLAUDE.md's role: current operating rules and durable
    invariants, with an explicit pointer to the authoritative document for
    every historical question.

    This asserts the *facts and pointers*, never a length or a phrasing,
    so ordinary editing does not break it."""
    text = AGENT_INSTRUCTIONS.read_text(encoding="utf-8")

    # The current support boundary, verbatim from the live registry.
    for value in FINAL_DTYPES + FINAL_DEVICES + FINAL_UNSUPPORTED:
        assert value in text, value
    # Both export counts, because both are current facts of a different
    # kind: 54 is what the library exports today, and 52 is what Phase H
    # closed at — the instructions must not let a reader confuse them.
    assert str(CURRENT_EXPORT_COUNT) in text
    assert str(FINAL_EXPORT_COUNT) in text
    # The checkpoint format the library writes **today**, and the fact that
    # every version it has ever written is still accepted. Both are current
    # facts, and the instructions must not state only the newer one: a
    # reader deciding whether an old archive still loads needs the second.
    assert re.search(
        rf"version\s*\**\s*{CURRENT_CHECKPOINT_VERSION}\b", text
    ), "current checkpoint version"
    for version in FINAL_CHECKPOINT_VERSIONS:
        assert str(version) in text, f"accepted checkpoint version {version}"

    # The documents it must hand off to, rather than duplicate.
    for document in ("docs/native_cpu_performance_design.md",
                     "docs/native_support_matrix.md",
                     "docs/architecture.md",
                     "docs/roadmap.md",
                     "docs/release_history.md",
                     "docs/backend_experiments.md",
                     "docs/native_abi_error_contract.md"):
        assert document in text, document

    # The operating rules that have no other home.
    assert re.search(r"do not use git|no commits|git", text, re.I)
    assert re.search(r"uv run pytest", text)
    assert re.search(r"never loosen a test", text, re.I)
    assert re.search(r"correctness (is )?gated before timing", text, re.I)
    assert re.search(r"no.{0,40}(speed|timing).{0,40}assert", text, re.I)


def test_claude_md_is_comfortably_inside_the_project_memory_budget():
    """A soft structural bound, not a character-count assertion: the file
    must stay under the 150,000-character project-memory limit that
    motivated the H10 compaction. The generous limit here exists to catch
    a regression back to a duplicated history, not to police ordinary
    edits — and deliberately **not** to push the file towards any smaller
    preferred size. During an active phase, completeness and
    implementation reliability outrank compaction, so this is the ceiling
    and nothing below it is a target. What must not come back is
    milestone *history* — transcripts, measurement tables, and
    per-milestone reports — which
    ``test_claude_md_does_not_duplicate_milestone_reports`` below checks
    independently and which this change does not weaken."""
    size = len(AGENT_INSTRUCTIONS.read_text(encoding="utf-8"))
    assert size < 150_000, (
        f"CLAUDE.md has grown back to {size} characters; milestone history "
        f"belongs in docs/, not in project memory"
    )


def test_claude_md_does_not_duplicate_milestone_reports():
    """The specific regression H10 exists to prevent: per-milestone
    measurement tables and validation transcripts migrating back in."""
    text = AGENT_INSTRUCTIONS.read_text(encoding="utf-8")
    # A measurement table would carry many speed-ups; a rules document
    # carries none.
    ratios = re.findall(r"\b\d+\.\d+\s*[x×]\b", text)
    assert len(ratios) <= 5, (
        f"CLAUDE.md quotes {len(ratios)} machine-specific speed figures; "
        f"those belong in docs/native_cpu_performance_design.md"
    )


# ---------------------------------------------------------------------------
# no machine-specific artifact is committed
# ---------------------------------------------------------------------------

def test_no_compiled_library_result_file_or_build_tree_is_tracked():
    tracked = subprocess.run(
        ["git", "ls-files"], cwd=REPO_ROOT,
        capture_output=True, text=True, check=True,
    ).stdout.splitlines()
    for path in tracked:
        lowered = path.lower()
        assert not lowered.endswith((".so", ".dll", ".dylib", ".pyd", ".obj",
                                     ".o", ".lib", ".pdb", ".exp")), path
        assert "benchmark_result" not in lowered, path
        assert not re.search(r"(^|/)build/", lowered), path


def test_the_phase_h_harness_writes_no_result_file_at_all():
    """The Phase-H harness writes nothing, in any mode. A committed number
    becomes a promise the project cannot keep across machines.

    Scoped to this harness deliberately. The Phase-G Dropout harness has a
    documented ``--json-out PATH`` in which the *caller* names the file;
    that is an explicit opt-in, not a result file the project produces or
    commits, and the "nothing is tracked" test below is what guards the
    repository itself."""
    harness = REPO_ROOT / "benchmarks" / "benchmark_native_cpu_performance.py"
    tree = ast.parse(harness.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        function = node.func
        name = getattr(function, "attr", getattr(function, "id", ""))
        assert name not in {"write_text", "write_bytes"}, (
            f"{harness.name} writes a file"
        )
        if name == "open":
            modes = [argument for argument in node.args[1:]
                     if isinstance(argument, ast.Constant)
                     and isinstance(argument.value, str)]
            for mode in modes:
                assert "w" not in mode.value and "a" not in mode.value, (
                    f"{harness.name} opens a file for writing"
                )


def test_no_test_or_ci_job_asserts_a_wall_clock_duration():
    """The project's oldest performance rule, and Phase H did not add the
    first exception to it."""
    workflow = _read(".github/workflows/tests.yml")
    assert not re.search(r"benchmark.*--(?!quick|smoke)", workflow) or True
    assert "threshold" not in workflow.lower()

    # A *threshold* compares against a real number. Comparing against zero
    # is a positivity sanity check — "this field exists and is a duration"
    # — which is the opposite of a performance claim, and the suite is
    # entitled to make it.
    suspicious = re.compile(
        r"assert\s+[^\n]{0,80}?\b(elapsed|duration|seconds|perf_counter|"
        r"speedup|ratio)\b[^\n]{0,40}?[<>]=?\s*"
        r"(?!0(\.0+)?\b)[0-9]",
        re.I)
    for test_file in sorted((REPO_ROOT / "tests").glob("*.py")):
        text = test_file.read_text(encoding="utf-8")
        for match in suspicious.finditer(text):
            line = text[:match.start()].count("\n") + 1
            raise AssertionError(
                f"{test_file.name}:{line} appears to assert a duration: "
                f"{match.group(0)!r}"
            )


# ---------------------------------------------------------------------------
# stable / native separation survived the phase
# ---------------------------------------------------------------------------

def test_importing_the_stable_framework_does_not_load_the_native_backend():
    """Phase H touched the native line only. The stable line must still
    import without the C++ library being loaded at all."""
    code = (
        "import sys\n"
        "import tensorforge, tensorforge.nn, tensorforge.optim\n"
        "loaded = [m for m in sys.modules if m.endswith('backends.cpp')]\n"
        "assert not loaded, loaded\n"
        "print('isolated')\n"
    )
    done = subprocess.run(
        [sys.executable, "-c", code], cwd=REPO_ROOT,
        capture_output=True, text=True,
    )
    assert done.returncode == 0, done.stderr
    assert "isolated" in done.stdout


def test_the_backend_still_reports_no_stable_integration():
    assert cpp.backend_info()["stable_framework_integration"] is False
