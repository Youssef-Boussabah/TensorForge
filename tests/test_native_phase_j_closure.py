"""Phase-J closure guardrails (deterministic native data pipeline, J9).

The durable replacement for Phase J's milestone-era *pending* checks, in the
shape ``tests/test_native_phase_h_closure.py`` established and
``tests/test_native_phase_i_closure.py`` refined. Every earlier guard about
this phase carried a premise that expires at closure — "J9 has not started",
"the closure module is absent", "no surface may call Phase J complete", "the
phase is newly approved and in progress". J9 ran the cross-platform matrix
and closed the phase, so those premises are gone. What replaces them is not
silence: the boundary Phase J stopped at is now a **permanent** rule rather
than a temporary one, and this module is where it is enforced.

The rules these tests protect:

* the phase is closed, and its ladder is whole (J0-J9, once each, in order,
  every one marked complete, with no J10 and nothing left open);
* **the data-pipeline phase broadened nothing** — it added three public
  Python names and *zero* dtypes, devices, C ABI exports, CTests,
  checkpoint versions, state versions, and dependencies, and every §3
  capability row is exactly what a completed Phase I handed it;
* the three public names are ``NativeTensorDataset`` (J1),
  ``NativeBatchSampler`` (J2), and ``NativeDataLoader`` (J3), each landed by
  exactly one milestone, and **J0 and J4-J9 added none**, leaving
  ``tensorforge.experimental.__all__`` at **25**;
* the export surface is still exactly **54**, and the source inventory
  agrees with the built library;
* the four state authorities keep their exact formats and versions —
  checkpoint 3 with ``(1, 2, 3)`` accepted, optimizer state 1, loader state
  1, sampler state 1 — with **no cross-object atomicity**, no loader
  discovery, and no import edge in either direction between the checkpoint
  and the pipeline;
* the J1-J8 evidence — dataset, sampler, loader transaction, loader state,
  checkpoint metadata workflow, training example, hardening matrix, and
  benchmark — is all still present and cannot be deleted or weakened
  unnoticed;
* **concurrency stays a documented boundary rather than a tested safety
  claim**: no Phase-J module holds a lock, thread, queue, future, or async
  primitive, and none joins the process-wide state-replacement lock order;
* no generated artifact is tracked, no benchmark number is committed, and
  no timing gate exists anywhere.

Three properties of this module are deliberate and load-bearing.

**It runs anywhere.** A full local clone, a GitHub Actions depth-1
checkout, a CRLF working tree, and an environment with no ``.git`` at all
must all give the same verdict. Nothing here reaches for a historical Git
object — no ``git show``, ``git diff``, ``ls-tree``, ``cat-file``, or
commit SHA — and the one helper that talks to Git at all asks about the
*index*, which a shallow checkout has in full, and degrades to a documented
skip rather than a false pass.

**Every parser has a negative control.** A structural check that cannot
fail proves nothing, so each scanner below is a pure function over text and
is driven, in ``test_*_detects_*`` / ``test_*_can_actually_fail``, against
deliberately broken input. The controls operate on temporary strings and
temporary directories only: no repository file is written, moved, or
restored.

**Prohibited-token scans read code, not prose.** A closure module whose job
includes proving "no lock, no worker, no prefetch" would fail on the very
sentences that document the prohibition, so every such scan strips
docstrings and string literals through the AST first and looks at
identifiers.

They test *values and structure* rather than wording, so ordinary prose
improvements do not require rewriting them. Nothing here asserts a
character count, a paragraph order, a total suite size, or a benchmark
number.
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
PHASE_J_DESIGN = REPO_ROOT / "docs" / "native_data_pipeline_design.md"
AGENT_INSTRUCTIONS = REPO_ROOT / "CLAUDE.md"
WORKFLOW = REPO_ROOT / ".github" / "workflows" / "tests.yml"

# Surfaces that state *current* status. Per-milestone historical records
# deliberately preserve superseded wording and are not scanned; the release
# history is a chronology for the same reason, and the design document
# carries its own status line, checked on its own below.
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
# one is*, because Phase J's defining property is that it moved none of them.
# ---------------------------------------------------------------------------

# The capability, exactly as a completed Phase I handed it over.
FINAL_DTYPES = ("float64", "float32")
FINAL_DEVICES = ("cpu",)
FINAL_UNSUPPORTED = ("cuda", "amp")
# The default an omitted dtype selects.
FINAL_DEFAULT_DTYPE = "float64"
# A permanent limitation of the seven handle-free raw utility kernels, which
# take only ``double*`` and an element count and so have no dtype to dispatch
# on. Never the overall support statement.
FINAL_RAW_KERNEL_DTYPES = ("float64",)

# What Phase J inherited, pinned as history so the delta is checkable. These
# are the same tuples: that identity *is* the phase's capability claim.
PHASE_I_DTYPES = ("float64", "float32")
PHASE_I_DEVICES = ("cpu",)
PHASE_I_UNSUPPORTED = ("cuda", "amp")
PHASE_I_EXPORT_COUNT = 54
# Phase I's one public capability change, kept attributed so a reader cannot
# mistake float32 for something the data pipeline granted.
FLOAT32_SUPPORT_PHASE = "I"
FLOAT32_SUPPORT_MILESTONE = "I9"

# The ABI. Phase J planned no new export and added none, at any milestone.
PHASE_J_ADDED_EXPORTS = ()
FINAL_EXPORT_COUNT = PHASE_I_EXPORT_COUNT + len(PHASE_J_ADDED_EXPORTS)  # 54

# Symbols and CTests added by **later phases**, after Phase J closed, each
# mapped to the milestone that shipped it. Phase J's own record does not
# move; the live tree's is derived from it plus these, so an unrecorded
# addition still fails an exact equality. Phase K, milestone K1 added the
# int64 storage CTest, milestone K3 the argmax export and its CTest, and
# milestone K4 the index_select export and its CTest.
POST_PHASE_J_EXPORTS = {"tf_core_argmax": "K3", "tf_core_index_select": "K4"}
# ...and the one example a later phase added, named for the same reason.
POST_PHASE_J_EXAMPLES = {"native_integer_indexing.py": "K6"}
# Benchmarks a later phase added, named and subtracted the same way, so
# Phase J's own closing benchmark count stays historically exact.
POST_PHASE_J_BENCHMARKS = {"benchmark_native_integer.py": "K8"}
POST_PHASE_J_CTESTS = {"dtype_int64_storage": "K1", "argmax": "K3",
                       "index_select": "K4"}
CURRENT_EXPORT_COUNT = FINAL_EXPORT_COUNT + len(POST_PHASE_J_EXPORTS)  # 56

# Serialization — four separate authorities, none of which moved.
FINAL_CHECKPOINT_FORMAT = "tensorforge.native_checkpoint"
FINAL_CHECKPOINT_VERSION = 3
FINAL_CHECKPOINT_VERSIONS = (1, 2, 3)
FINAL_OPTIMIZER_STATE_VERSION = 1
FINAL_LOADER_STATE_FORMAT = "tensorforge.native_data_loader"
FINAL_LOADER_STATE_VERSION = 1
FINAL_LOADER_STATE_VERSIONS = (1,)
FINAL_LOADER_STATE_FIELDS = ("format", "format_version", "sampler")
FINAL_SAMPLER_STATE_FORMAT = "tensorforge.native_sampler"
FINAL_SAMPLER_STATE_VERSION = 1
FINAL_SAMPLER_STATE_VERSIONS = (1,)

# Inventories, **as Phase J closed on them**.
FINAL_CTEST_COUNT = 24
FINAL_EXAMPLE_COUNT = 16
FINAL_BENCHMARK_COUNT = 9
FINAL_EXPERIMENTAL_EXPORTS = 25

MILESTONES = tuple(f"J{index}" for index in range(10))   # J0 ... J9

# The three public names, and the one milestone that shipped each. Every
# other Phase-J milestone — J0, and J4 through J9 — added none, which is
# derived below rather than restated.
PHASE_J_CLASSES = {
    "NativeTensorDataset": "J1",
    "NativeBatchSampler": "J2",
    "NativeDataLoader": "J3",
}
ZERO_API_MILESTONES = tuple(name for name in MILESTONES
                            if name not in set(PHASE_J_CLASSES.values()))

# The production modules the phase added, and the private names inside them
# that must never be exported.
PHASE_J_MODULES = ("native_dataset.py", "native_sampler.py",
                   "_native_permutation.py", "native_data_loader.py")
PERMANENTLY_PRIVATE = ("_native_permutation", "_NativeBatchIterator",
                       "_deliver_batch", "_BatchTransaction",
                       "_validate_state", "_assign_state", "_claim_batch",
                       "_publish_pending", "_commit_pending",
                       "_rollback_pending", "_complete_pending")

# The one artifact each of J6 and J8 shipped. J0-J5, J7, and J9 shipped none.
J6_EXAMPLE = "examples/native_minibatch_training.py"
J8_BENCHMARK = "benchmarks/benchmark_native_data_pipeline.py"

# The evidence J1-J8 left, which closure must not let disappear. Each entry
# is (path, minimum test count) — a floor rather than an equality, so adding
# coverage is free while deleting it is not. The floors are set below the
# counts observed at closure on purpose: this guards deletion, not growth.
REQUIRED_EVIDENCE = (
    ("tests/test_native_dataset.py", 80),            # J1
    ("tests/test_native_sampler.py", 100),           # J2
    ("tests/test_native_data_loader.py", 70),        # J3
    ("tests/test_native_loader_state.py", 70),       # J4
    ("tests/test_native_data_checkpoint.py", 35),    # J5
    ("tests/test_native_minibatch_training.py", 60), # J6
    ("tests/test_native_data_hardening.py", 85),     # J7
    ("tests/test_native_data_benchmark.py", 85),     # J8
    ("tests/test_native_phase_j.py", 60),            # the contract module
)

# The benchmark's identity, written here independently of the harness so a
# silent renaming fails rather than propagating.
BENCHMARK_NAME = "tensorforge.native_data_pipeline"
BENCHMARK_VERSION = "1.0"
BENCHMARK_SCHEMA_VERSION = 1
BENCHMARK_WORKLOADS = ("dataset_indexing", "batch_planning",
                       "permutation_construction",
                       "host_to_native_materialization", "loader_delivery")
BENCHMARK_CASE_COUNT = 20
BENCHMARK_REFERENCES = ("numpy", "native_only")


def _read(relative):
    return (REPO_ROOT / relative).read_text(encoding="utf-8")


def _flat(text):
    """Whitespace-flattened, emphasis-stripped text, so a claim split across
    lines or wrapped in markdown still reads as one sentence.

    Line endings collapse with the rest of the whitespace, which is what
    makes every prose scan below identical on a CRLF checkout."""
    return re.sub(r"\s+", " ", re.sub(r"[*`]", "", text))


def _design():
    return PHASE_J_DESIGN.read_text(encoding="utf-8")


def _source_exports():
    names = set()
    for source in sorted((REPO_ROOT / "cpp" / "src").glob("*.cpp")):
        text = source.read_text(encoding="utf-8")
        names.update(re.findall(r"TF_EXPORT[^;{]*?\b(tf_[a-z0-9_]+)\s*\(",
                                text, re.S))
    return names


def _git_lines(*arguments):
    """One read-only Git query, or a documented skip where Git cannot
    answer.

    The only place this module talks to Git at all, and it asks exclusively
    about the *working tree and index*, which a depth-1 checkout has in
    full — unlike a historical blob, which no guard here may reach for."""
    try:
        done = subprocess.run(["git", *arguments], cwd=REPO_ROOT,
                              capture_output=True, text=True)
    except OSError:                                   # pragma: no cover
        pytest.skip("git is unavailable, so the tree cannot be inspected")
    if done.returncode != 0:                          # pragma: no cover
        pytest.skip("this tree has no git index to read")
    return done.stdout.splitlines()


def _tracked_files():
    """Every tracked path."""
    return _git_lines("ls-files")


def _untracked_unignored_files():
    """Every file present but neither tracked nor ignored.

    This — rather than "every file on disk" — is the honest scope for an
    artifact guard. A path ``.gitignore`` covers is by construction outside
    the repository's content and cannot be committed by accident; a
    developer's local scratch build is exactly what that mechanism exists
    for, and failing on one would make this a machine-specific guard that
    passes in CI and fails on a workstation. What must never appear is an
    artifact that is **not** ignored, because that one is a single
    ``git add`` away from becoming repository content."""
    return _git_lines("ls-files", "--others", "--exclude-standard")


class _LiveStorages:
    """The ids of every open ``NativeStorage``, tracked by wrapping the
    constructor and ``close``.

    The project's deterministic instrumentation for native-allocation
    lifetime, used unchanged since Phase C. **There is no public counter
    and closure adds none** — a live-storage query would be exactly the
    kind of runtime introspection §4.1 forbids. Installed and restored
    explicitly rather than through ``monkeypatch``, matching J7, so no
    later ``undo()`` can silently disarm it and leave every assertion
    below vacuously true."""

    def __init__(self):
        self.open_ids = set()

    def __enter__(self):
        self._init = cpp.NativeStorage.__init__
        self._close = cpp.NativeStorage.close
        original_init, original_close = self._init, self._close
        open_ids = self.open_ids

        def tracked_init(inner, *args, **kwargs):
            original_init(inner, *args, **kwargs)
            open_ids.add(id(inner))

        def tracked_close(inner):
            original_close(inner)
            open_ids.discard(id(inner))

        cpp.NativeStorage.__init__ = tracked_init
        cpp.NativeStorage.close = tracked_close
        return self

    def __exit__(self, *exception):
        cpp.NativeStorage.__init__ = self._init
        cpp.NativeStorage.close = self._close
        return False

    def __len__(self):
        return len(self.open_ids)


def _code_identifiers(text):
    """Every identifier that *executes* in ``text``.

    Names, attributes, keyword-argument names, function/class names, and
    imported module and alias names — and **no string content at all**, so a
    module may explain a prohibition at any length while remaining unable to
    perform it. Keyword names matter specifically: a substring scan that
    reads only ``ast.Name`` misses ``f(_trusted_dtype=True)`` entirely."""
    found = set()
    for node in ast.walk(ast.parse(text)):
        if isinstance(node, ast.Name):
            found.add(node.id)
        elif isinstance(node, ast.Attribute):
            found.add(node.attr)
        elif isinstance(node, ast.keyword) and node.arg:
            found.add(node.arg)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef,
                               ast.ClassDef)):
            found.add(node.name)
        elif isinstance(node, ast.arg):
            found.add(node.arg)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                found.update(alias.name.split("."))
                if alias.asname:
                    found.add(alias.asname)
        elif isinstance(node, ast.ImportFrom):
            found.update((node.module or "").split("."))
            for alias in node.names:
                found.add(alias.name)
                if alias.asname:
                    found.add(alias.asname)
    return found


def _async_constructs(text):
    """The async statement forms, which carry no identifier to scan for."""
    kinds = set()
    for node in ast.walk(ast.parse(text)):
        if isinstance(node, ast.AsyncFunctionDef):
            kinds.add("async def")
        elif isinstance(node, ast.Await):
            kinds.add("await")
        elif isinstance(node, (ast.AsyncFor, ast.AsyncWith)):
            kinds.add("async for/with")
    return kinds


# ===========================================================================
# 1. The milestone ladder — whole, ordered, and closed
# ===========================================================================
#
# Parsed structurally rather than matched as prose, and every parser here is
# a pure function over text so the negative controls below can feed it
# deliberately broken ladders.

_LADDER_HEADING = re.compile(r"^### (J\d+) — (.*)$", re.M)


def _ladder_text(text):
    """The §23 ladder body, from the section heading to §23.1."""
    if "## 23." not in text:
        raise AssertionError("the design has no milestone-ladder section")
    ladder = text.split("## 23.", 1)[1]
    return ladder.split("### 23.1", 1)[0]


def _ladder_rows(text):
    """``[(milestone, heading_tail)]`` in document order."""
    return _LADDER_HEADING.findall(_ladder_text(text))


def _ladder_problems(text):
    """Every way the closed ladder can be wrong, as a list of reasons.

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

    # Every row marked complete, in the heading. A phase with an unmarked or
    # still-open row is not closed, and at closure there is no exception.
    for name, tail in rows:
        if not re.search(r"\*\*complete\*\*", tail, re.I):
            problems.append(f"{name} is not marked complete")
        if re.search(r"not started|in progress|pending", tail, re.I):
            problems.append(f"{name} is still marked open")
    return problems


def test_the_milestone_ladder_runs_j0_to_j9_once_each_in_order():
    """J0 through J9, each exactly once, ordered, and every one marked
    complete. This is the single assertion that says the phase is done."""
    problems = _ladder_problems(_design())
    assert problems == [], problems


def test_the_ladder_carries_exactly_ten_rows_and_no_more():
    rows = _ladder_rows(_design())
    assert len(rows) == 10, [name for name, _ in rows]
    assert [name for name, _ in rows] == list(MILESTONES)


def test_the_design_marks_the_phase_complete_rather_than_active():
    """The status line, which is what every other surface is reconciled
    against. Through J8 it had to name a first *unstarted* milestone; at
    closure there is none, and saying so is the point."""
    text = _flat(_design())
    status = re.search(r"Phase-J status:(.{0,240})", text, re.I)
    assert status, "the design does not state its milestone status"
    claim = status.group(1)
    assert re.search(r"\bJ0\b.{0,80}?\bJ9\b.{0,60}?complete", claim, re.I), (
        f"the status line does not record J0-J9 complete: {claim!r}"
    )
    assert not re.search(r"not started", claim, re.I), (
        f"the status line still names an unstarted milestone: {claim!r}"
    )
    # ...and the phase itself, stated as a phase rather than as a run of
    # milestone numbers.
    assert re.search(
        r"Phase J is (?:now )?(?:\w+ ){0,3}?(complete|closed)", text, re.I
    ), "the design does not state that Phase J is closed"


def test_the_design_names_no_milestone_beyond_j9():
    """A J10 would be a milestone this document does not define, and the
    phase after Phase J requires a separate approval rather than a
    paragraph."""
    for match in re.finditer(r"\bJ1[0-9]\b", _design()):
        raise AssertionError(f"the design names {match.group(0)}")


def test_phase_i_is_still_recorded_complete_and_was_not_reopened():
    """Closing one phase must not disturb the one before it, and Phase I is
    where float32 came from."""
    design = _flat((REPO_ROOT / "docs"
                    / "native_dtype_float32_design.md").read_text(
                        encoding="utf-8"))
    assert re.search(r"Phase I is (?:now )?(?:\w+ ){0,3}?(complete|closed)",
                     design, re.I)
    for surface in STATUS_SURFACES:
        text = _flat(_read(surface))
        assert not re.search(
            r"Phase I[^.;]{0,80}?\b(is not complete|has not been closed|"
            r"is the current phase|is active)\b", text, re.I), surface


# --- the ladder parser's negative controls --------------------------------

def _ladder_document(rows):
    """A synthetic design fragment carrying exactly ``rows``."""
    body = "\n".join(f"### {name} — Something — **complete**\n\nbody\n"
                     for name in rows)
    return f"## 23. Milestone ladder\n\n{body}\n### 23.1 Adjustments\n"


def test_the_ladder_parser_detects_every_kind_of_drift():
    """A parser that cannot fail proves nothing. Each fault the closure
    claim depends on is produced deliberately and shown to be caught, on
    temporary strings only — no repository file is touched."""
    # 0. The positive control: a whole, ordered, complete ladder passes.
    assert _ladder_problems(_ladder_document(MILESTONES)) == []

    # 1. J9 omitted — the specific drift a closure ladder must not have.
    problems = _ladder_problems(
        _ladder_document([n for n in MILESTONES if n != "J9"]))
    assert any("missing" in reason and "J9" in reason
               for reason in problems), problems

    # 2. A middle milestone omitted.
    problems = _ladder_problems(
        _ladder_document([n for n in MILESTONES if n != "J5"]))
    assert any("missing" in reason and "J5" in reason
               for reason in problems), problems

    # 3. One milestone duplicated.
    problems = _ladder_problems(_ladder_document(list(MILESTONES) + ["J7"]))
    assert any("duplicated" in reason for reason in problems), problems

    # 4. Two milestones swapped.
    swapped = list(MILESTONES)
    swapped[3], swapped[4] = swapped[4], swapped[3]
    problems = _ladder_problems(_ladder_document(swapped))
    assert any("out of order" in reason for reason in problems), problems

    # 5. An invented J10.
    problems = _ladder_problems(_ladder_document(list(MILESTONES) + ["J10"]))
    assert any("unexpected" in reason and "J10" in reason
               for reason in problems), problems

    # 6. J9 reverted to not started — the drift closure exists to stop.
    document = _ladder_document(MILESTONES).replace(
        "### J9 — Something — **complete**",
        "### J9 — Something — **not started**")
    problems = _ladder_problems(document)
    assert any("J9" in reason for reason in problems), problems

    # 7. J8 reverted to not started — the milestone before closure.
    document = _ladder_document(MILESTONES).replace(
        "### J8 — Something — **complete**",
        "### J8 — Something — **not started**")
    problems = _ladder_problems(document)
    assert any("J8" in reason for reason in problems), problems

    # 8. An earlier milestone left with no completion marker at all.
    document = _ladder_document(MILESTONES).replace(
        "### J4 — Something — **complete**", "### J4 — Something")
    assert "J4 is not marked complete" in _ladder_problems(document)

    # 9. No ladder section at all.
    with pytest.raises(AssertionError):
        _ladder_problems("# a document with no section 23\n")

    # Nothing above touched the repository: the live ladder still passes.
    assert _ladder_problems(_design()) == []


def test_every_ladder_row_still_states_its_scope_and_exit_gate():
    """A row that names a milestone without saying what it delivers or how
    it was judged is a title, not a contract — including J9's own."""
    ladder = _ladder_text(_design())
    rows = re.split(r"^### J\d+ — ", ladder, flags=re.M)[1:]
    assert len(rows) == len(MILESTONES)
    for index, body in enumerate(rows):
        assert "**Scope:**" in body, f"J{index} states no scope"
        assert "**Exit gate:**" in body, f"J{index} states no exit gate"


# ===========================================================================
# 2. Phase-complete status, across every current surface
# ===========================================================================

# One authority for "this surface still says the phase is open", used by the
# scan and by its own negative control so the two cannot drift apart.
_STALE_STATUS = re.compile(
    r"Phase[- ]J\b[^.;]{0,90}?\b(is not complete|is not closed|is active|"
    r"remains active|has not been closed|is the current phase|"
    r"is in progress|is active rather than closed|is not finished|"
    r"is nearly complete|is incomplete|is unfinished|is newly approved|"
    r"is a newly approved|is awaiting)\b"
    r"|\bJ9\b[^.;]{0,70}?\b(not started|is next|has not started|pending|"
    r"is unstarted|remains)\b"
    r"|\b(what remains|remaining|still to come|next|awaiting)\b"
    r"[^.;]{0,40}?\bJ9\b"
    r"|\bJ0[-–—]J8\b[^.;]{0,40}?\bcomplete\b"
    r"|\bJ0 through J8\b[^.;]{0,40}?\bcomplete\b"
    r"|\bJ8 is the latest\b"
    r"|\bclosure (remains|is still) (pending|a promise|unstarted)\b"
    r"|\bdata[- ]pipeline benchmark has not started\b",
    re.I)

# Sentences whose tense makes them accurate history rather than a stale
# claim. Deliberately narrower than "any past-tense verb anywhere": the
# marker has to sit in the same neighbourhood as the match.
_HISTORY = re.compile(
    r"\b(was|were|had|until|before|at J\d|by J\d|then|earlier|previously|"
    r"stayed|remained|originally|drafted|no longer|used to|history|"
    r"historical|during|while|as of J\d|through J8,)\b", re.I)


def _stale_status(text):
    """Every stale status claim in ``text``, as matched spans."""
    return [match.group(0) for match in _STALE_STATUS.finditer(text)
            if not _HISTORY.search(text[max(0, match.start() - 140):
                                        match.end() + 50])]


@pytest.mark.parametrize("surface", STATUS_SURFACES + ("CLAUDE.md",))
def test_no_status_surface_still_calls_phase_j_unfinished(surface):
    """The failure this exists for: a surface advanced in one paragraph and
    left saying "J9 has not started" in another.

    Time-scoped sentences are excused, because "Phase J was active at J8" is
    history and history stays accurate."""
    offenders = _stale_status(_flat(_read(surface)))
    assert offenders == [], f"{surface}: {offenders[:3]}"


# The completion claim, which must attach to the **phase** rather than to a
# milestone inside it: "Phase J is approved and J0 is complete" is a J0-era
# sentence and may not satisfy this. The inner negative lookahead is what
# enforces that — no milestone token may sit between the subject and the
# completion word.
_PHASE_COMPLETE = re.compile(
    r"Phase J\b(?:(?!\bJ\d)[^.;]){0,140}?\b(is|are)\s+(now\s+)?"
    r"(complete|closed)"
    r"|Phase J\b[^.;]{0,60}?\bcomplete \(J0[-–—]J9\)"
    r"|\bJ0[-–—]J9\b[^.;]{0,70}?\b(complete|landed|closed)"
    r"|\bJ0 through J9\b[^.;]{0,70}?\b(complete|landed|closed)",
    re.I)


@pytest.mark.parametrize("surface", STATUS_SURFACES)
def test_every_status_surface_marks_phase_j_complete(surface):
    text = _flat(_read(surface))
    assert "Phase J" in text, f"{surface} does not name Phase J"
    assert _PHASE_COMPLETE.search(text), (
        f"{surface} does not mark Phase J complete")


@pytest.mark.parametrize("surface", STATUS_SURFACES + ("CLAUDE.md",))
def test_no_status_surface_invents_a_phase_after_j(surface):
    """Closure is not permission to *invent* the next phase. A J10, an
    unapproved successor, or a committed promise about one would be a
    roadmap entry nobody approved.

    The J10 half is permanent: this ladder ran J0-J9 and ended there. The
    phase-name half is a moving sentinel. It banned ``Phase K``, which was
    accurate protection right up until it stopped being: Phase K (native
    integer tensors and indexing) was **separately approved after this
    phase closed**, and its K0 architecture contract now exists, so naming
    it is a status update rather than an invention — and keeping the ban
    would force every status surface to under-report the project. ``Phase
    L`` takes its place. What may not be claimed for Phase K is a
    *capability*, which tests/test_native_phase_k.py checks against the
    live registry, the live source, and the built library."""
    text = _read(surface)
    for match in re.finditer(r"\bJ1[0-9]\b", text):
        raise AssertionError(f"{surface} names {match.group(0)}")
    flat = _flat(text)
    for pattern in (r"\bPhase L\b",
                    r"\bthe next phase (is|will be)\b[^.;]{0,40}?\b\w"):
        offender = re.search(pattern, flat, re.I)
        assert offender is None, f"{surface}: {offender.group(0)!r}"


def test_the_phase_complete_scan_detects_a_surface_left_active():
    """The prose scans' own control, since "no offenders" is only meaningful
    when the pattern can find one. Deliberately driven through the **same**
    compiled patterns the scans use, so a weakening there fails here."""
    for caught in (
        "Phase J is active rather than closed",
        "Phase J is not complete",
        "Phase J is in progress",
        "Phase J is nearly complete",
        "Phase J is newly approved",
        "milestones J0-J8 complete, and J9 is not started",
        "J0 through J8 complete",
        "what remains is J9 (integration and closure)",
        "J9 has not started",
        "J8 is the latest completed milestone",
        "the phase closure remains a promise",
        "the data-pipeline benchmark has not started",
    ):
        assert _stale_status(caught), caught
    # ...and the accurate closure sentences are not caught.
    for allowed in (
        "Phase J is complete",
        "Phase J is closed: milestones J0 through J9 have all landed",
        "J9 closed the phase",
        "Phase J was newly approved after Phase I closed at I11",
        "milestones J0 through J9 complete",
        "at J8 the benchmark landed; J9 closed the phase",
    ):
        assert _stale_status(allowed) == [], (allowed, _stale_status(allowed))
    # The completion pattern must recognise the closure sentences and must
    # not be satisfied by an open one.
    for recognised in (
        "Phase J is complete",
        "Phase J is now closed",
        "Phase J complete (J0–J9)",
        "milestones J0–J9 are complete",
        "J0 through J9 have all landed",
    ):
        assert _PHASE_COMPLETE.search(recognised), recognised
    for rejected in (
        "Phase J is approved and J0 is complete",
        "milestones J0 through J8 have landed",
    ):
        assert not _PHASE_COMPLETE.search(rejected), rejected


# ===========================================================================
# 3. The final support boundary — Phase J moved nothing
# ===========================================================================

def test_the_public_registries_are_exactly_what_phase_i_handed_over():
    """The capability, stated as literals and as the delta from what the
    phase inherited. Phase J's delta is **empty**, in every direction."""
    assert cpp.SUPPORTED_DTYPES == FINAL_DTYPES
    assert cpp.SUPPORTED_DEVICES == FINAL_DEVICES
    assert cpp.UNSUPPORTED == FINAL_UNSUPPORTED
    assert cpp.RAW_KERNEL_DTYPES == FINAL_RAW_KERNEL_DTYPES

    # Nothing moved in or out of any registry across the whole phase.
    assert set(FINAL_DTYPES) == set(PHASE_I_DTYPES)
    assert set(FINAL_DEVICES) == set(PHASE_I_DEVICES)
    assert set(FINAL_UNSUPPORTED) == set(PHASE_I_UNSUPPORTED)
    assert FINAL_DTYPES == PHASE_I_DTYPES          # order is contractual too

    # Order is contractual: float64 first, because it is the default.
    assert cpp.SUPPORTED_DTYPES[0] == FINAL_DEFAULT_DTYPE
    # Supported and unsupported cannot overlap, in either direction.
    assert not set(cpp.SUPPORTED_DTYPES) & set(cpp.UNSUPPORTED)
    # The raw-kernel registry is a *narrower* statement, permanently.
    assert set(cpp.RAW_KERNEL_DTYPES) < set(cpp.SUPPORTED_DTYPES)


def test_the_three_dtype_rows_still_answer_three_different_questions():
    """``backend_info()`` reports a capability, a default, and a limitation,
    and none of them may be reported as another."""
    info = cpp.backend_info()
    assert info["supported_dtypes"] == FINAL_DTYPES        # the capability
    assert info["dtype"] == FINAL_DEFAULT_DTYPE            # the default
    assert info["raw_kernel_dtypes"] == FINAL_RAW_KERNEL_DTYPES  # the limit
    assert info["supported_devices"] == FINAL_DEVICES
    assert info["unsupported"] == FINAL_UNSUPPORTED
    assert info["stable_framework_integration"] is False
    assert cpp.normalize_dtype(None) == FINAL_DEFAULT_DTYPE
    assert cpp.normalize_dtype("float64") == "float64"
    assert cpp.normalize_dtype("float32") == "float32"


def test_float32_is_attributed_to_phase_i_rather_than_to_the_pipeline():
    """Historical attribution, kept explicit so a reader of the closed phase
    cannot conclude the data pipeline granted a dtype. Phase I's own design
    owns that claim; Phase J's says it moved nothing."""
    phase_i = _flat((REPO_ROOT / "docs"
                     / "native_dtype_float32_design.md").read_text(
                         encoding="utf-8"))
    assert re.search(rf"\b{FLOAT32_SUPPORT_MILESTONE}\b", phase_i)
    design = _flat(_design())
    # The pipeline design states the registries stay put, naming each one.
    for registry in ("SUPPORTED_DTYPES", "SUPPORTED_DEVICES", "UNSUPPORTED",
                     "RAW_KERNEL_DTYPES"):
        assert registry in design, registry
    assert re.search(r"grants no dtype, no device", design, re.I), (
        "the design no longer states its empty capability delta")


def test_no_dtype_or_device_beyond_the_final_boundary_is_reachable():
    """The registry claim checked against behavior rather than trusted."""
    for absent in ("float16", "bfloat16", "float128", "int64", "int32",
                   "complex64", "bool"):
        with pytest.raises((ValueError, TypeError)):
            cpp.normalize_dtype(absent)
    for device in ("cuda", "cuda:0", "gpu", "mps"):
        with pytest.raises((ValueError, TypeError)):
            cpp.normalize_device(device)


@pytest.mark.skipif(not cpp.is_available(),
                    reason="the native library is not built")
@pytest.mark.parametrize("dtype", FINAL_DTYPES)
def test_the_pipeline_delivers_batches_at_each_supported_dtype(dtype):
    """The one place the closure module runs the pipeline itself: both
    supported widths reach a delivered batch, the target stays host
    ``int64`` at each, and everything the caller receives is closed."""
    from tensorforge.experimental import (NativeBatchSampler, NativeDataLoader,
                                          NativeTensorDataset)

    features = np.arange(24, dtype=np.float64).reshape(8, 3)
    targets = np.array([0, 1, 2, 0, 1, 2, 0, 1], dtype=np.int64)
    dataset = NativeTensorDataset(features, targets, dtype=dtype)
    try:
        sampler = NativeBatchSampler(dataset, batch_size=4, shuffle=True,
                                     seed=7)
        loader = NativeDataLoader(sampler)
        try:
            batch_features, batch_targets = next(iter(loader))
            try:
                assert batch_features.dtype == dtype
                assert batch_features.device == "cpu"
                assert batch_targets.dtype == np.dtype(np.int64)
                assert not batch_targets.flags.writeable
                assert batch_targets.flags.owndata
            finally:
                batch_features.close()
        finally:
            loader.close()
    finally:
        dataset.close()


@pytest.mark.skipif(not cpp.is_available(),
                    reason="the native library is not built")
def test_the_dataset_dtype_is_chosen_and_never_inferred():
    """The rule that keeps ingress an explicit conversion rather than a
    cast, re-proved at the pipeline's own boundary: a float32 host array
    with no ``dtype`` still gives a float64 dataset."""
    from tensorforge.experimental import NativeTensorDataset

    narrow = np.zeros((4, 2), dtype=np.float32)
    targets = np.zeros(4, dtype=np.int64)
    dataset = NativeTensorDataset(narrow, targets)
    try:
        assert dataset.dtype == FINAL_DEFAULT_DTYPE
    finally:
        dataset.close()


# ===========================================================================
# 4. The final public surface
# ===========================================================================

def test_the_phase_added_exactly_three_public_names():
    import tensorforge.experimental as experimental

    for name, milestone in PHASE_J_CLASSES.items():
        assert hasattr(experimental, name), (name, milestone)
        assert name in experimental.__all__, name
        assert experimental.__all__.count(name) == 1, name
    assert len(PHASE_J_CLASSES) == 3
    # ...and the milestones that added none really did add none: every
    # Phase-J milestone except J1, J2, and J3.
    assert ZERO_API_MILESTONES == ("J0", "J4", "J5", "J6", "J7", "J8", "J9")


def test_the_experimental_export_surface_is_exactly_twenty_five():
    import tensorforge.experimental as experimental

    assert len(experimental.__all__) == FINAL_EXPERIMENTAL_EXPORTS
    assert len(set(experimental.__all__)) == len(experimental.__all__)
    for name in experimental.__all__:
        assert hasattr(experimental, name), name


def test_no_fourth_pipeline_name_arrived_and_the_helpers_stayed_private():
    import tensorforge.experimental as experimental

    for invented in ("NativeDataset", "NativeSampler", "NativeLoader",
                     "NativeBatchLoader", "NativeDataIterator",
                     "NativeBatchIterator", "NativeCollate", "NativeWorker",
                     "NativePrefetcher", "NativeTransform",
                     "NativeDistributedSampler", "native_batches",
                     "native_data_loader_from_checkpoint"):
        assert not hasattr(experimental, invented), invented
        assert invented not in experimental.__all__, invented
    # The private names, none of which may be exported. They exist, so this
    # is a visibility rule rather than a removal.
    for name in PERMANENTLY_PRIVATE:
        assert name not in experimental.__all__, name
        assert not hasattr(experimental, name.lstrip("_")), name
    # The submodules are ordinary package attributes, not exports.
    for module in ("native_dataset", "native_sampler", "native_data_loader",
                   "_native_permutation"):
        assert module not in experimental.__all__, module


def test_the_private_loader_internals_are_still_where_the_evidence_needs_them():
    """``_deliver_batch`` is a **test seam**, never a hook: it stays a
    module-level private function so J7 can patch it, and it stays out of
    every public surface."""
    from tensorforge.experimental import native_data_loader

    assert hasattr(native_data_loader, "_deliver_batch")
    assert hasattr(native_data_loader, "_NativeBatchIterator")
    assert not hasattr(native_data_loader, "deliver_batch")
    # The iterator is reachable only through ``iter(loader)``.
    assert "_NativeBatchIterator" not in getattr(
        native_data_loader, "__all__", [])


def test_no_phase_j_name_entered_the_stable_public_api():
    import tensorforge

    for name in tuple(PHASE_J_CLASSES) + ("NativeTensor", "NativeAdam",
                                          "save_native_checkpoint"):
        assert not hasattr(tensorforge, name), name
    stable = _read("tests/test_public_api.py")
    for name in PHASE_J_CLASSES:
        assert name not in stable, name


def test_the_closure_milestone_added_no_public_name_of_its_own():
    """J9's whole delta is evidence and status. A closure module that
    exported something would be capability wearing a test's clothes."""
    import tensorforge.experimental as experimental

    closure = Path(__file__).read_text(encoding="utf-8")
    identifiers = _code_identifiers(closure)
    for forbidden in ("NativeClosure", "closure_report", "phase_j_status"):
        assert forbidden not in identifiers, forbidden
    assert not hasattr(experimental, "NativeClosure")
    assert len(experimental.__all__) == FINAL_EXPERIMENTAL_EXPORTS


# ===========================================================================
# 5. The final C ABI
# ===========================================================================

def test_the_source_exports_exactly_fifty_four_symbols():
    """Stated as arithmetic rather than as a bare number, so the facts stay
    separable: Phase I closed at 54, Phase J added none, and every symbol
    the live source carries beyond that belongs to a named later
    milestone."""
    exports = _source_exports()
    assert len(exports) == CURRENT_EXPORT_COUNT, sorted(exports)
    assert PHASE_J_ADDED_EXPORTS == ()
    for name, milestone in POST_PHASE_J_EXPORTS.items():
        assert name in exports, (name, milestone)
    assert len(exports - set(POST_PHASE_J_EXPORTS)) == FINAL_EXPORT_COUNT
    assert len(exports - set(POST_PHASE_J_EXPORTS)) == PHASE_I_EXPORT_COUNT


def test_no_pipeline_shaped_c_abi_symbol_was_added():
    """The shapes a data-pipeline phase is most tempted to add, banned by
    name. The whole pipeline is Python over the existing handle-based ABI."""
    banned = re.compile(
        r"^tf_(dataset|sampler|loader|batch|shuffle|permutation|gather|"
        r"index_select|collate|prefetch|worker|epoch|cursor)", re.I)
    for name in sorted(_source_exports()):
        assert not banned.search(name), name


@pytest.mark.skipif(not cpp.is_available(),
                    reason="the native library is not built")
def test_the_built_library_exports_exactly_what_the_source_declares():
    library = cpp._require_library()
    for name in sorted(_source_exports()):
        assert hasattr(library, name), name


def test_the_declared_ctypes_surface_matches_the_source_exports():
    source = _read("src/tensorforge/backends/cpp.py")
    declared = set(re.findall(r"library\.(tf_[a-z0-9_]+)\s*\.", source))
    declared |= set(re.findall(r"getattr\(library,\s*\"(tf_[a-z0-9_]+)\"",
                               source))
    missing = declared - _source_exports()
    assert not missing, f"declared but not exported: {sorted(missing)}"


def test_no_phase_j_module_declares_a_ctypes_symbol_of_its_own():
    """``backends/cpp.py`` is the only module that may import ``ctypes``,
    and the pipeline reaches the runtime through the wrapper rather than
    around it."""
    package = REPO_ROOT / "src" / "tensorforge" / "experimental"
    for name in PHASE_J_MODULES:
        identifiers = _code_identifiers(
            (package / name).read_text(encoding="utf-8"))
        assert "ctypes" not in identifiers, name
        assert not {n for n in identifiers if n.startswith("tf_")}, name


# ===========================================================================
# 6. The native inventories — CTests, sources, headers, and build options
# ===========================================================================

def _registered_ctests():
    cmake = _read("cpp/CMakeLists.txt")
    return re.findall(r"add_test\s*\(\s*NAME\s+(\w+)", cmake)


def test_the_ctest_inventory_is_exactly_twenty_four_unique_targets():
    """Phase J registered none, and every target the live tree carries
    beyond its 24 belongs to a named later milestone."""
    names = _registered_ctests()
    for name, milestone in POST_PHASE_J_CTESTS.items():
        assert name in names, (name, milestone)
    assert len([n for n in names
                if n not in POST_PHASE_J_CTESTS]) == FINAL_CTEST_COUNT, names
    assert len(set(names)) == len(names), "a CTest name is registered twice"
    # Every registered test has a source file, and every source file is
    # registered — so a target cannot be added or orphaned unnoticed.
    sources = {path.stem for path in
               sorted((REPO_ROOT / "cpp" / "tests").glob("test_*.cpp"))}
    assert sources == {f"test_{name}" for name in names}, (
        sorted(sources), sorted(names))


def test_no_ctest_names_the_data_pipeline():
    """Phase J added no C++ at all, so no CTest may be about it."""
    for name in _registered_ctests():
        assert not re.search(r"dataset|sampler|loader|batch|pipeline", name,
                             re.I), name


def test_the_compiled_source_inventory_is_declared_and_complete():
    """Every compiled production source is declared in CMake, every declared
    source exists, and nothing unexpected is compiled."""
    cmake = _read("cpp/CMakeLists.txt")
    declared = set(re.findall(r"src/(\w+\.cpp)", cmake))
    present = {path.name for path in (REPO_ROOT / "cpp" / "src").glob("*.cpp")}
    assert declared == present, (sorted(declared), sorted(present))
    for name in declared:
        assert (REPO_ROOT / "cpp" / "src" / name).is_file(), name
    # Test-only sources stay test-only: none of them lives beside the
    # production sources.
    assert not [p for p in (REPO_ROOT / "cpp" / "src").glob("test_*.cpp")]
    # No generated source entered the tree.
    for path in (REPO_ROOT / "cpp").rglob("*.cpp"):
        assert ".generated" not in path.name, path
    # Headers are all hand-written internals; none is a build product.
    headers = {path.name for path in
               (REPO_ROOT / "cpp" / "include").glob("*.h")}
    assert headers, "the include directory is empty"
    for name in headers:
        assert name.startswith("tf_"), name


def test_the_build_still_offers_exactly_two_options_and_no_arch_flag():
    cmake = _read("cpp/CMakeLists.txt")
    options = set(re.findall(r"^\s*option\s*\(\s*(\w+)", cmake, re.M))
    assert options == {"TF_BUILD_TESTS", "TF_SANITIZE"}, options
    assert not re.search(r"/arch:AVX|-mavx|-march=native|-ffast-math"
                         r"|/fp:fast|-funsafe-math", cmake)


def test_no_build_output_can_become_repository_content():
    """Builds and logs belong outside the tree. What this guards is that
    none of them can be **committed**: every build-shaped directory name is
    covered by ``.gitignore``, and no build output is tracked or sitting
    unignored in the working tree.

    Scoped to ignored-ness rather than to existence on purpose. A local
    scratch build directory is what ``.gitignore`` is for, and a guard that
    failed on one would pass in CI and fail on a workstation — a
    machine-specific check, which is the failure mode this repository has
    already had to repair once."""
    ignore = _read(".gitignore")
    for pattern in ("build/", "dist/", "__pycache__", ".venv",
                    "src/tensorforge/backends/_tensorforge_cpp.*",
                    "src/tensorforge/backends/*.lib"):
        assert pattern in ignore, pattern
    tracked = set(_tracked_files())
    for path in tracked:
        assert not re.match(r"(build|cmake-build|out|asan|ubsan)/", path), path
        assert "CMakeCache.txt" not in path, path
    for path in _untracked_unignored_files():
        lowered = path.lower()
        assert "cmakecache" not in lowered, path
        assert not re.match(r"(build|cmake-build|out|asan|ubsan|lsan)/",
                            lowered), path


# ===========================================================================
# 7. Examples and benchmarks
# ===========================================================================

def test_the_example_inventory_closed_at_sixteen():
    """Phase J's own example delta is **exactly one**, J6's.

    The tree may hold more now — later phases ship their own — so the
    equality is stated as "sixteen, plus exactly the examples later
    milestones added, each named". That keeps J9's record historically
    exact while still failing on an unannounced example."""
    names = sorted(path.name for path in (REPO_ROOT / "examples").glob("*.py")
                   if path.name != "__init__.py")
    for name, milestone in POST_PHASE_J_EXAMPLES.items():
        assert name in names, (name, milestone)
    assert len([name for name in names
                if name not in POST_PHASE_J_EXAMPLES]) == (
        FINAL_EXAMPLE_COUNT), names
    assert Path(J6_EXAMPLE).name in names
    # No second pipeline example arrived under another name.
    pipeline = [n for n in names
                if re.search(r"minibatch|data_pipeline|dataloader|sampler", n)]
    assert pipeline == [Path(J6_EXAMPLE).name], pipeline


def test_the_benchmark_inventory_closed_at_nine():
    """Phase J's own benchmark delta is **exactly one**, J8's.

    Benchmarks a later phase added are named in ``POST_PHASE_J_BENCHMARKS``
    and subtracted the same way later examples are, so what this asserts
    stays a fact about **Phase J's close** rather than drifting into a
    claim about today."""
    names = sorted(path.name for path in
                   (REPO_ROOT / "benchmarks").glob("*.py")
                   if path.name != "__init__.py")
    for name, milestone in POST_PHASE_J_BENCHMARKS.items():
        assert name in names, (name, milestone)
    assert len([name for name in names
                if name not in POST_PHASE_J_BENCHMARKS]) == (
        FINAL_BENCHMARK_COUNT), names
    assert Path(J8_BENCHMARK).name in names
    pipeline = [n for n in names if "data_pipeline" in n]
    assert pipeline == [Path(J8_BENCHMARK).name], pipeline


def test_j9_added_no_example_and_no_benchmark():
    """The closure milestone's artifact delta is zero in both inventories,
    which is what the two equalities above already encode — restated here as
    the claim rather than as a consequence."""
    examples = {path.name for path in (REPO_ROOT / "examples").glob("*.py")}
    benchmarks = {path.name for path in (REPO_ROOT / "benchmarks").glob("*.py")}
    for name in sorted(examples | benchmarks):
        assert "closure" not in name.lower(), name
        assert "phase_j" not in name.lower(), name


def test_every_example_and_benchmark_is_tracked_source():
    tracked = set(_tracked_files())
    for directory in ("examples", "benchmarks"):
        for path in sorted((REPO_ROOT / directory).glob("*.py")):
            relative = f"{directory}/{path.name}"
            assert relative in tracked, relative


def test_no_benchmark_result_file_or_results_directory_exists():
    """The oldest performance rule in the project: a committed number
    becomes a promise the project cannot keep across machines."""
    for pattern in ("*.json", "*.csv", "*.npz", "*.txt"):
        for path in (REPO_ROOT / "benchmarks").glob(pattern):
            raise AssertionError(f"a benchmark artifact exists: {path.name}")
    for name in ("benchmark_results", "results", "timings", "measurements"):
        assert not (REPO_ROOT / name).exists(), name
        assert not (REPO_ROOT / "benchmarks" / name).exists(), name
    for path in _tracked_files():
        lowered = path.lower()
        assert "benchmark_result" not in lowered, path
        assert not re.search(
            r"(results?|timings?|measurements?|baseline)\.(json|csv|txt|md)$",
            lowered), path


def test_no_benchmark_suite_asserts_a_duration_or_writes_a_result_file():
    """Re-proved at closure over every benchmark-facing suite, including
    J8's."""
    suspicious = re.compile(
        r"assert\s+[^\n]{0,80}?\b(elapsed|duration|seconds|perf_counter|"
        r"speedup|ratio)\b[^\n]{0,40}?[<>]=?\s*(?!0(\.0+)?\b)[0-9]", re.I)
    for name in ("tests/test_native_data_benchmark.py",
                 "tests/test_native_dtype_benchmark.py",
                 "tests/test_native_cpu_performance_benchmark.py",
                 "tests/test_benchmarks.py"):
        text = _read(name)
        for match in suspicious.finditer(text):
            line = text[:match.start()].count("\n") + 1
            raise AssertionError(
                f"{name}:{line} asserts a duration: {match.group(0)!r}")
    # The J8 harness writes nothing, in any mode.
    tree = ast.parse(_read(J8_BENCHMARK))
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        name = getattr(node.func, "attr", getattr(node.func, "id", ""))
        assert name not in {"write_text", "write_bytes", "savez", "savetxt",
                            "to_csv", "mkdir"}, f"{J8_BENCHMARK} writes"
        if name == "open":
            for argument in node.args[1:]:
                if isinstance(argument, ast.Constant) and isinstance(
                        argument.value, str):
                    assert "w" not in argument.value, J8_BENCHMARK
                    assert "a" not in argument.value, J8_BENCHMARK


def test_the_j8_cli_offers_no_destination_save_baseline_or_compare_mode():
    """``--json`` is a *stdout* format flag and is deliberately allowed; the
    banned shape is an option that receives a path or names a stored
    baseline."""
    harness = _read(J8_BENCHMARK)
    options = re.findall(r'add_argument\(\s*"(--[\w-]+)"', harness)
    assert options, "the harness declares no options at all"
    # Anchored on a word boundary, so ``--profile`` — a case selector that
    # takes no path — is not mistaken for a destination ending in "file".
    destination = re.compile(
        r"(?:^--|[-_])(out|output|outfile|file|path|report|save|dest|"
        r"baseline|compare|record|store|results?)$")
    for option in options:
        assert not destination.search(option), (
            f"{J8_BENCHMARK} offers {option}")
    assert '"--json", action="store_true"' in harness, (
        "--json is no longer a plain stdout format flag")


def test_no_ci_job_gates_on_a_number():
    workflow = WORKFLOW.read_text(encoding="utf-8")
    lowered = workflow.lower()
    for banned in ("threshold", "budget", "regression", "speedup",
                   "upload-artifact", "benchmark-results"):
        assert banned not in lowered, banned


# ===========================================================================
# 8. The four state authorities, and the couplings that do not exist
# ===========================================================================

def test_the_checkpoint_constants_did_not_move():
    from tensorforge.experimental import native_checkpoint

    assert native_checkpoint._FORMAT == FINAL_CHECKPOINT_FORMAT
    assert native_checkpoint._FORMAT_VERSION == FINAL_CHECKPOINT_VERSION
    assert (native_checkpoint._SUPPORTED_FORMAT_VERSIONS
            == FINAL_CHECKPOINT_VERSIONS)
    accepted = native_checkpoint._SUPPORTED_FORMAT_VERSIONS
    assert list(accepted) == sorted(accepted)
    assert accepted[-1] == native_checkpoint._FORMAT_VERSION
    assert 4 not in accepted and 0 not in accepted


def test_the_optimizer_state_version_did_not_move():
    from tensorforge.experimental import native_optimizer_state

    assert (native_optimizer_state.FORMAT_VERSION
            == FINAL_OPTIMIZER_STATE_VERSION)
    assert 2 != native_optimizer_state.FORMAT_VERSION


def test_the_loader_state_schema_is_exactly_j4s_three_key_wrapper():
    from tensorforge.experimental import native_data_loader

    assert native_data_loader._FORMAT == FINAL_LOADER_STATE_FORMAT
    assert native_data_loader._FORMAT_VERSION == FINAL_LOADER_STATE_VERSION
    assert (native_data_loader._SUPPORTED_FORMAT_VERSIONS
            == FINAL_LOADER_STATE_VERSIONS)
    assert native_data_loader._STATE_FIELDS == FINAL_LOADER_STATE_FIELDS
    assert 2 not in native_data_loader._SUPPORTED_FORMAT_VERSIONS
    # No epoch, cursor, seed, shuffle, batch-size, or drop-last field may be
    # duplicated at the root: the loader owns none of them.
    for owned_by_the_sampler in ("epoch", "cursor", "seed", "shuffle",
                                 "batch_size", "drop_last", "dataset"):
        assert owned_by_the_sampler not in native_data_loader._STATE_FIELDS, (
            owned_by_the_sampler)


def test_the_sampler_state_schema_did_not_move_either():
    from tensorforge.experimental import native_sampler

    assert native_sampler._FORMAT == FINAL_SAMPLER_STATE_FORMAT
    assert native_sampler._FORMAT_VERSION == FINAL_SAMPLER_STATE_VERSION
    assert (native_sampler._SUPPORTED_FORMAT_VERSIONS
            == FINAL_SAMPLER_STATE_VERSIONS)
    assert 2 not in native_sampler._SUPPORTED_FORMAT_VERSIONS


@pytest.mark.skipif(not cpp.is_available(),
                    reason="the native library is not built")
def test_the_loader_state_is_the_wrapper_plus_the_untouched_sampler_state():
    """The shape is contractual: three keys, and ``sampler`` is *exactly*
    the sampler's own version-1 state rather than a re-spelling of it."""
    from tensorforge.experimental import (NativeBatchSampler, NativeDataLoader,
                                          NativeTensorDataset)

    features = np.zeros((6, 2), dtype=np.float64)
    targets = np.zeros(6, dtype=np.int64)
    dataset = NativeTensorDataset(features, targets)
    try:
        sampler = NativeBatchSampler(dataset, batch_size=3)
        loader = NativeDataLoader(sampler)
        try:
            state = loader.state_dict()
            assert set(state) == set(FINAL_LOADER_STATE_FIELDS)
            assert state["format"] == FINAL_LOADER_STATE_FORMAT
            assert state["format_version"] == FINAL_LOADER_STATE_VERSION
            assert state["sampler"] == sampler.state_dict()
            # Pure and fresh: a second call shares nothing with the first.
            again = loader.state_dict()
            assert again == state
            assert again is not state
            assert again["sampler"] is not state["sampler"]
        finally:
            loader.close()
    finally:
        dataset.close()


def test_the_checkpoint_and_the_pipeline_import_each_other_in_neither_direction():
    """The absence that keeps loader state *caller-managed*: no import edge,
    so neither module can grow a default for the other."""
    package = REPO_ROOT / "src" / "tensorforge" / "experimental"
    checkpoint = _code_identifiers(
        (package / "native_checkpoint.py").read_text(encoding="utf-8"))
    for pipeline in ("native_dataset", "native_sampler", "native_data_loader",
                     "_native_permutation", "NativeTensorDataset",
                     "NativeBatchSampler", "NativeDataLoader"):
        assert pipeline not in checkpoint, pipeline
    for name in PHASE_J_MODULES:
        identifiers = _code_identifiers(
            (package / name).read_text(encoding="utf-8"))
        for checkpoint_name in ("native_checkpoint", "save_native_checkpoint",
                                "load_native_checkpoint"):
            assert checkpoint_name not in identifiers, (name, checkpoint_name)


def test_the_checkpoint_grew_no_loader_field_and_no_discovery():
    """The archive captures no loader position, and nothing reconstructs a
    dataset or restores a loader on a caller's behalf."""
    from tensorforge.experimental import native_checkpoint

    identifiers = _code_identifiers(
        _read("src/tensorforge/experimental/native_checkpoint.py"))
    for absent in ("data_loader", "dataloader", "loader_state", "sampler",
                   "dataset", "epoch", "cursor", "discover_loader",
                   "rebuild_dataset"):
        assert absent not in identifiers, absent
    for absent in ("discover_loader", "restore_loader", "rebuild_dataset",
                   "default_loader", "LOADER_KEY", "DATA_LOADER"):
        assert not hasattr(native_checkpoint, absent), absent


def test_no_cross_object_atomicity_is_offered_or_claimed():
    """``load_native_checkpoint`` is atomic over model, optimizer, and
    generators; ``loader.load_state_dict`` over loader and sampler. Nothing
    joins them, and the design says so."""
    from tensorforge.experimental import native_checkpoint

    for combined in ("load_native_checkpoint_with_loader",
                     "restore_everything", "load_training_state",
                     "atomic_restore"):
        assert not hasattr(native_checkpoint, combined), combined
    design = _flat(_design())
    assert re.search(r"cross-object atomicity[^.;]{0,80}?\bnot\b", design,
                     re.I), "the design no longer disclaims cross-object atomicity"
    assert re.search(r"nothing rolls back", design, re.I), design[:0]


def test_no_loader_discovery_or_registry_exists_anywhere():
    import tensorforge.experimental as experimental

    for absent in ("DATA_LOADER_REGISTRY", "LOADER_REGISTRY", "loaders",
                   "register_loader", "discover_loader", "default_loader",
                   "get_loader", "DATASET_REGISTRY", "register_dataset"):
        assert not hasattr(experimental, absent), absent
        assert absent not in experimental.__all__, absent
    package = REPO_ROOT / "src" / "tensorforge" / "experimental"
    for name in PHASE_J_MODULES:
        identifiers = _code_identifiers(
            (package / name).read_text(encoding="utf-8"))
        for absent in ("register_loader", "discover_loader", "LOADER_REGISTRY",
                       "DATASET_REGISTRY"):
            assert absent not in identifiers, (name, absent)


# ===========================================================================
# 9. Dataset contract evidence (J1)
# ===========================================================================

def test_the_dataset_evidence_retains_its_named_subjects():
    """Not reimplemented here: J1's suite is the authority, and duplicating
    it would create a second one that could disagree. What closure owns is
    that its named subjects cannot quietly disappear."""
    evidence = _read("tests/test_native_dataset.py").lower()
    for subject in ("ndarray", "dtype", "contiguous", "fingerprint",
                    "sha256", "int64", "identity", "close", "alias",
                    "duplicate", "read_only", "writeable", "endian",
                    "feature_batch", "target_batch"):
        assert subject in evidence, subject
    # The two rejections the empty-dataset and target-range rules exist for.
    for rule in ("empty", "negative", "representab"):
        assert rule in evidence, rule


@pytest.mark.skipif(not cpp.is_available(),
                    reason="the native library is not built")
def test_the_dataset_snapshot_is_copied_and_its_identity_is_deterministic():
    """Two structural facts the whole resume proof rests on, checked live:
    caller mutation after construction reaches nothing, and two equal
    datasets produce one identity while different data does not."""
    from tensorforge.experimental import NativeTensorDataset

    features = np.arange(12, dtype=np.float64).reshape(6, 2)
    targets = np.array([0, 1, 0, 1, 0, 1], dtype=np.int64)
    first = NativeTensorDataset(features.copy(), targets.copy())
    second = NativeTensorDataset(features.copy(), targets.copy())
    try:
        assert first.identity() == second.identity()
        assert first.identity() is not second.identity()
        # Mutating the caller's arrays afterwards changes nothing.
        mutated = features.copy()
        third = NativeTensorDataset(mutated, targets.copy())
        try:
            before = third.identity()
            mutated[0, 0] = 1234.5
            assert third.identity() == before
        finally:
            third.close()
        # Different data, different fingerprint.
        other = features.copy()
        other[0, 0] = 99.0
        fourth = NativeTensorDataset(other, targets.copy())
        try:
            assert fourth.fingerprint != first.fingerprint
        finally:
            fourth.close()
    finally:
        second.close()
        first.close()


@pytest.mark.skipif(not cpp.is_available(),
                    reason="the native library is not built")
def test_the_dataset_owns_no_native_storage_between_calls():
    """Holding a dataset leaves the native live-storage count untouched;
    only a materialized batch allocates, and the caller owns that."""
    from tensorforge.experimental import NativeTensorDataset

    with _LiveStorages() as live:
        dataset = NativeTensorDataset(np.zeros((4, 2)),
                                      np.zeros(4, dtype=np.int64))
        try:
            assert len(live) == 0
            batch = dataset.feature_batch((0, 1))
            try:
                assert len(live) > 0
            finally:
                batch.close()
            assert len(live) == 0
        finally:
            dataset.close()
        assert len(live) == 0


# ===========================================================================
# 10. Sampler contract evidence (J2)
# ===========================================================================

def test_the_sampler_evidence_retains_its_named_subjects():
    evidence = _read("tests/test_native_sampler.py").lower()
    for subject in ("splitmix64", "fisher", "epoch", "cursor", "drop_last",
                    "batch_size", "shuffle", "seed", "permutation",
                    "state_dict", "load_state_dict", "identity",
                    "reference", "unbiased"):
        assert subject in evidence, subject


def test_the_sampler_reference_vectors_are_still_committed_and_exact():
    """The committed §8.9 vectors, in the design and in the evidence. A
    derivation with no known answer is not deterministic, it is merely
    repeatable."""
    design = _design()
    # ``| length | seed | epoch | permutation |``, with the seed written in
    # decimal or hexadecimal and the permutation in backticks.
    vectors = re.findall(
        r"\|\s*(\d+)\s*\|\s*`([^`]+)`\s*\|\s*(\d+)\s*\|\s*"
        r"`\[([0-9,\s]*)\]`\s*\|", design)
    assert vectors, "the design carries no reference-vector table"
    for length, _seed, _epoch, body in vectors:
        order = [int(value) for value in body.split(",") if value.strip()]
        assert sorted(order) == list(range(int(length))), (length, order)
    # Lengths 1, 2, 5, and 8 all appear, so the table cannot shrink to one
    # convenient case, and so do seed 0, a nontrivial large seed, and the
    # accepted upper bound.
    lengths = {int(length) for length, _s, _e, _b in vectors}
    assert {1, 2, 5, 8} <= lengths, sorted(lengths)
    seeds = {seed for _l, seed, _e, _b in vectors}
    assert "0" in seeds
    assert "0xFFFFFFFFFFFFFFFF" in seeds, sorted(seeds)
    assert any(seed.startswith("0x") and seed != "0xFFFFFFFFFFFFFFFF"
               for seed in seeds), sorted(seeds)
    # Epoch 0 and a later epoch both appear, so the epoch key is exercised.
    assert {int(epoch) for _l, _s, epoch, _b in vectors} >= {0, 7}
    # The derivation's own known answers are committed beside the orders.
    assert "splitmix64_mix" in design and "epoch_key" in design
    # ...and the empty case is stated as a rejection rather than a vector.
    assert re.search(r"length == 0` has \*\*no vector\*\*", design)


@pytest.mark.skipif(not cpp.is_available(),
                    reason="the native library is not built")
def test_the_sampler_is_still_a_planner_that_owns_nothing():
    """Pure planning: no native allocation, no ``close()``, and an order
    that is a pure function of ``(seed, epoch, length)``."""
    from tensorforge.experimental import NativeBatchSampler, NativeTensorDataset

    with _LiveStorages() as live:
        dataset = NativeTensorDataset(np.zeros((8, 2)),
                                      np.zeros(8, dtype=np.int64))
        try:
            sampler = NativeBatchSampler(dataset, batch_size=3, shuffle=True,
                                         seed=5)
            assert not hasattr(sampler, "close")
            assert len(live) == 0
            # Repeatable and side-effect free: inspection consumes nothing.
            first = sampler.plan()
            assert sampler.plan() == first
            assert sampler.epoch_permutation(0) == sampler.epoch_permutation(0)
            assert sampler.cursor == 0 and sampler.epoch == 0
            # A second sampler with the same key gives the same order, and a
            # different seed does not — so the order is a pure function of
            # the key rather than of the object.
            twin = NativeBatchSampler(dataset, batch_size=3, shuffle=True,
                                      seed=5)
            assert twin.epoch_permutation(0) == sampler.epoch_permutation(0)
            other = NativeBatchSampler(dataset, batch_size=3, shuffle=True,
                                       seed=6)
            assert other.epoch_permutation(0) != sampler.epoch_permutation(0)
            # ...and a later epoch differs from epoch 0.
            assert sampler.epoch_permutation(1) != sampler.epoch_permutation(0)
            assert len(live) == 0
        finally:
            dataset.close()
        assert len(live) == 0


def test_the_sampler_has_no_public_advance_or_reset():
    """Position moves only through a delivered batch or a state load. A
    public advance would let a caller skip a batch nothing recorded."""
    from tensorforge.experimental import NativeBatchSampler

    for banned in ("advance", "reset", "step", "next", "set_epoch",
                   "set_cursor", "skip", "seek", "close"):
        assert not hasattr(NativeBatchSampler, banned), banned
    # ``epoch`` and ``cursor`` are read-only properties.
    for name in ("epoch", "cursor", "batches_per_epoch", "remaining"):
        attribute = getattr(NativeBatchSampler, name)
        assert isinstance(attribute, property), name
        assert attribute.fset is None, name


def test_the_permutation_derivation_is_uncoupled_from_the_live_generator():
    """No second RNG surface: the sampler reuses the locked derivation and
    never reaches for a ``NativeGenerator``, ``random``, or NumPy's global
    stream."""
    package = REPO_ROOT / "src" / "tensorforge" / "experimental"
    for name in ("_native_permutation.py", "native_sampler.py"):
        identifiers = _code_identifiers(
            (package / name).read_text(encoding="utf-8"))
        # No live generator object, no second RNG algorithm, and no
        # entropy, clock, or global stream anywhere.
        for banned in ("NativeGenerator", "random", "secrets", "default_rng",
                       "RandomState", "Generator", "time", "getrandbits",
                       "urandom", "uuid"):
            assert banned not in identifiers, (name, banned)
    # The sampler's one import from the generator module is the shared
    # unsigned-64-bit *validator*, which is a validation helper rather than
    # a random surface — so the uncoupling is stated precisely rather than
    # by banning the module wholesale.
    sampler = (package / "native_sampler.py").read_text(encoding="utf-8")
    imported = set()
    for node in ast.walk(ast.parse(sampler)):
        if isinstance(node, ast.ImportFrom) and node.module == \
                "native_generator":
            imported.update(alias.name for alias in node.names)
    assert imported <= {"UINT64_MAX", "_validate_uint64"}, sorted(imported)
    # ...and NumPy's or Python's own shuffling entry points are never
    # called. Scoped to the *receiver* rather than to the method name,
    # because ``_perm.permutation(...)`` is this phase's own derivation and
    # banning the bare word would ban the thing being proved.
    def receiver_root(node):
        while isinstance(node, ast.Attribute):
            node = node.value
        return node.id if isinstance(node, ast.Name) else ""

    for name in ("_native_permutation.py", "native_sampler.py"):
        source = (package / name).read_text(encoding="utf-8")
        for node in ast.walk(ast.parse(source)):
            if not isinstance(node, ast.Attribute):
                continue
            if node.attr in {"shuffle", "permutation", "choice", "randint",
                             "seed", "random"}:
                root = receiver_root(node)
                assert root not in {"np", "numpy", "random", "rng"}, (
                    name, root, node.attr)


# ===========================================================================
# 11. Loader transaction evidence (J3)
# ===========================================================================

def test_the_loader_evidence_retains_its_named_subjects():
    evidence = _read("tests/test_native_data_loader.py").lower()
    for subject in ("iter", "stopiteration", "supersed", "close",
                    "_deliver_batch", "rollback", "epoch", "cursor",
                    "retry", "same", "storage"):
        assert subject in evidence, subject


def test_the_design_still_specifies_the_five_phase_handoff():
    """§9.4, which the whole exact-resume claim rests on."""
    section = _flat(_design().split("### 9.4", 1)[1].split("\n## 10.", 1)[0])
    lowered = section.lower()
    for phase in ("claim", "construct", "publish", "commit", "rollback"):
        assert phase in lowered, phase
    assert "five phases" in lowered, "the handoff is no longer five phases"
    # The invariant, in §9.4's own words.
    assert re.search(
        r"no committed sampler position ever advances for a batch the "
        r"caller did not receive", lowered), (
        "§9.4 no longer states the delivery invariant")
    for claim in ("epoch", "cursor", "close", "retry", "serial"):
        assert claim in lowered, claim
    # ...and the same invariant in its "if and only if" spelling, which the
    # rest of the document and every status surface quote.
    assert len(re.findall(r"if and only if", _flat(_design()), re.I)) >= 2


@pytest.mark.skipif(not cpp.is_available(),
                    reason="the native library is not built")
def test_a_failed_delivery_consumes_nothing_and_a_retry_repeats_the_batch():
    """The phase's central invariant, re-proved at closure through the same
    private seam J7 uses: the committed position advances **if and only if**
    a batch was delivered.

    J7 owns the exhaustive matrix; what closure owns is that this one
    guarantee cannot regress unnoticed."""
    from tensorforge.experimental import (NativeBatchSampler, NativeDataLoader,
                                          NativeTensorDataset)
    from tensorforge.experimental import native_data_loader

    features = np.arange(32, dtype=np.float64).reshape(8, 4)
    targets = np.array([0, 1, 2, 3, 0, 1, 2, 3], dtype=np.int64)
    original = native_data_loader._deliver_batch
    with _LiveStorages() as live:
        dataset = NativeTensorDataset(features, targets)
        try:
            sampler = NativeBatchSampler(dataset, batch_size=3, shuffle=True,
                                         seed=11)
            loader = NativeDataLoader(sampler)
            try:
                iterator = iter(loader)
                expected = sampler.next_batch_indices()
                before = (sampler.epoch, sampler.cursor)

                def failing(record):
                    raise RuntimeError("deliberate delivery failure")

                native_data_loader._deliver_batch = failing
                try:
                    with pytest.raises(RuntimeError):
                        next(iterator)
                finally:
                    native_data_loader._deliver_batch = original

                # Nothing was consumed, nothing leaked, and the same batch
                # is still the next one.
                assert (sampler.epoch, sampler.cursor) == before
                assert len(live) == 0
                assert sampler.next_batch_indices() == expected

                batch_features, batch_targets = next(iterator)
                try:
                    assert (sampler.epoch, sampler.cursor) == (before[0],
                                                               before[1] + 1)
                    assert batch_features.shape[0] == len(expected)
                    assert list(batch_targets) == [int(targets[i])
                                                   for i in expected]
                finally:
                    batch_features.close()
            finally:
                loader.close()
        finally:
            native_data_loader._deliver_batch = original
            dataset.close()
        assert len(live) == 0


@pytest.mark.skipif(not cpp.is_available(),
                    reason="the native library is not built")
def test_a_delivered_batch_belongs_to_the_caller_and_the_loader_never_closes_it():
    """Closing the loader must not reach a batch it already handed over."""
    from tensorforge.experimental import (NativeBatchSampler, NativeDataLoader,
                                          NativeTensorDataset)

    with _LiveStorages() as live:
        dataset = NativeTensorDataset(
            np.arange(16, dtype=np.float64).reshape(8, 2),
            np.zeros(8, dtype=np.int64))
        try:
            loader = NativeDataLoader(NativeBatchSampler(dataset,
                                                         batch_size=4))
            batch_features, _ = next(iter(loader))
            loader.close()
            # Still alive and still readable after the loader closed.
            assert len(live) > 0
            assert batch_features.to_numpy().shape == (4, 2)
            batch_features.close()
            assert len(live) == 0
        finally:
            dataset.close()
        assert len(live) == 0


def test_the_transaction_serial_and_participation_token_are_never_reused():
    """Two contracted counters that legitimately advance on a *failed*
    attempt, because neither is ever reused. The evidence asserts it
    explicitly rather than excluding it quietly."""
    hardening = _read("tests/test_native_data_hardening.py").lower()
    assert "serial" in hardening
    assert "token" in hardening
    assert re.search(r"never reused|not reused|non-reuse|reuse", hardening)


def test_the_loader_state_snapshot_refuses_an_in_flight_transaction():
    """§9.5: no snapshot may observe a skipped-but-undelivered position, and
    the refusal comes from the *sampler's* existing guard rather than a
    second authority in the loader."""
    from tensorforge.experimental import native_data_loader, native_sampler

    # The guard lives on the sampler's own snapshot...
    sampler_snapshot = inspect.getsource(
        native_sampler.NativeBatchSampler.state_dict)
    assert "_require_no_transaction" in sampler_snapshot, (
        "the sampler's state_dict no longer refuses mid-transaction")
    # ...and the loader reaches it by delegating rather than restating it,
    # taking the sampler's snapshot *first* so nothing is built on refusal.
    loader_snapshot = inspect.getsource(
        native_data_loader.NativeDataLoader.state_dict)
    body = loader_snapshot.split('"""')[-1]
    assert "self._sampler.state_dict()" in body, (
        "the loader's state_dict no longer delegates to the sampler")
    assert body.index("self._sampler.state_dict()") < body.index("_FORMAT"), (
        "the loader builds its wrapper before the guard can refuse")
    # ...and the loader defines no rival guard of its own.
    identifiers = _code_identifiers(
        _read("src/tensorforge/experimental/native_data_loader.py"))
    for rival in ("_transaction_in_flight", "_check_transaction",
                  "_require_no_batch"):
        assert rival not in identifiers, rival


def test_the_loader_load_state_dict_delegates_the_whole_nested_validation():
    """§12.5: the loader validates its own three keys and hands the nest to
    ``NativeBatchSampler._validate_state``. Never a restated nested rule,
    and never the sampler's public loader."""
    from tensorforge.experimental import native_data_loader

    source = inspect.getsource(
        native_data_loader.NativeDataLoader.load_state_dict)
    assert "_validate_state" in source
    assert "_assign_state" in source
    assert "load_state_dict(" not in source.replace(
        "def load_state_dict(", ""), (
        "the loader now calls the sampler's public loader")


def test_no_public_advance_iterator_hook_collate_or_worker_was_added():
    from tensorforge.experimental import NativeDataLoader

    for banned in ("advance", "reset", "collate", "collate_fn", "transform",
                   "num_workers", "workers", "prefetch", "pin_memory",
                   "on_batch", "hook", "register_hook", "__len__",
                   "set_epoch", "skip"):
        assert not hasattr(NativeDataLoader, banned), banned
    # ``__len__`` specifically: a loader whose length looks knowable would
    # invite a caller to precompute a schedule the transaction cannot honour.
    assert not hasattr(NativeDataLoader, "__len__")


# ===========================================================================
# 12. Exact resume evidence (J4-J6)
# ===========================================================================

def test_the_training_example_still_exists_and_proves_every_state_family():
    """Not reimplemented here: J6's proof is a whole example plus a focused
    suite. What closure owns is that it cannot be deleted or narrowed."""
    example = REPO_ROOT / J6_EXAMPLE
    assert example.is_file(), J6_EXAMPLE

    sys.path.insert(0, str(REPO_ROOT / "examples"))
    try:
        import native_minibatch_training as proof
    finally:
        sys.path.pop(0)

    # Both dtypes, run independently and compared only against themselves.
    assert set(proof.RUN_DTYPES) == set(FINAL_DTYPES), proof.RUN_DTYPES
    # Every state family the resume must reproduce, by name.
    for claim in ("parameters_match", "buffers_match", "moments_match",
                  "counters_match", "optimizer_matches", "generator_matches",
                  "topology_matches", "loss_sequence_matches",
                  "suffix_losses_match", "feature_batches_match",
                  "target_batches_match", "whole_index_sequence_matches",
                  "final_loader_state_matches", "evaluation_matches",
                  "identities_preserved", "fresh_started_different"):
        assert claim in proof.REQUIRED, claim
    # The negative control that makes the proof non-vacuous: a resume that
    # omits the loader state must diverge.
    for control in ("omitted_next_batch_differs", "omitted_indices_differ",
                    "omitted_losses_differ", "omitted_parameters_differ",
                    "omitted_evaluation_differs"):
        assert control in proof.REQUIRED, control
    # The interrupt is genuinely mid-epoch: not the first step, not the
    # last, and not an epoch boundary.
    assert 0 < proof.SPLIT_STEP < proof.TOTAL_STEPS
    assert proof.SPLIT_STEP % proof.BATCHES_PER_EPOCH != 0
    # The only cross-dtype claim carries no dtype at all.
    for claim in proof.REQUIRED_CROSS_DTYPE:
        assert re.search(r"index|permutation|position|batch", claim), claim


def test_the_resume_suite_still_drives_every_required_claim():
    source = _read("tests/test_native_minibatch_training.py")
    assert "REQUIRED" in source
    # Every one of the example's four claim lists is driven, so a family
    # cannot be quietly dropped from the proof.
    for family in ("REQUIRED", "REQUIRED_TRAINING", "REQUIRED_SCHEDULE"):
        assert re.search(
            rf"@pytest\.mark\.parametrize\(\s*\"check\",\s*{family}\b",
            source), f"the suite no longer drives {family}"
    assert "REQUIRED_CROSS_DTYPE" in source, (
        "the suite no longer covers the cross-dtype claims")
    # Bit comparison, never a tolerance — measured over what executes rather
    # than over what the module explains.
    example = _code_identifiers(_read(J6_EXAMPLE))
    for banned in ("allclose", "isclose", "approx", "atol", "rtol",
                   "assert_almost_equal", "assert_allclose"):
        assert banned not in example, banned


def test_the_checkpoint_metadata_workflow_evidence_is_still_present():
    """J5 proved the workflow and changed no production code; its whole diff
    was one test module. Closure keeps that module, and keeps the ordering
    it fixed."""
    evidence = _read("tests/test_native_data_checkpoint.py")
    lowered = evidence.lower()
    for subject in ("save_native_checkpoint", "load_native_checkpoint",
                    "state_dict", "load_state_dict", "metadata"):
        assert subject in evidence, subject
    # The three delivery boundaries, all contractual.
    for boundary in ("failed", "epoch", "next"):
        assert boundary in lowered, boundary
    # ``training`` / ``data_loader`` / ``next_step`` are **caller
    # conventions**: no production constant may spell one.
    package = REPO_ROOT / "src" / "tensorforge" / "experimental"
    for path in sorted(package.glob("*.py")):
        identifiers = _code_identifiers(path.read_text(encoding="utf-8"))
        for convention in ("DATA_LOADER_KEY", "TRAINING_KEY", "NEXT_STEP_KEY"):
            assert convention not in identifiers, (path.name, convention)


def test_the_checkpoint_module_still_preserves_metadata_without_interpreting_it():
    """It validates JSON-compatibility only, invents no default for an
    absent loader state, and calls no loader method."""
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


def test_the_loader_state_evidence_covers_restoration_and_rejection():
    evidence = _read("tests/test_native_loader_state.py").lower()
    for subject in ("state_dict", "load_state_dict", "format_version",
                    "mid-epoch", "identity", "precedence", "reject",
                    "fresh"):
        assert subject in evidence, subject


# ===========================================================================
# 13. Hardening evidence (J7)
# ===========================================================================

def test_the_hardening_matrix_still_protects_every_named_subject():
    """Structural evidence and a floor rather than a brittle list of every
    test name: the matrix must keep its subjects, not its exact shape."""
    hardening = _read("tests/test_native_data_hardening.py")
    lowered = hardening.lower()
    for subject in ("malformed", "precedence", "fingerprint",
                    "live_storages", "monkeypatch", "baseexception",
                    "reentran", "abandon", "rollback", "gather",
                    "allocation", "transfer", "target", "publication",
                    "commit", "seam", "serial", "token", "arm_alloc_failure"):
        assert subject in lowered, subject
    # The four Phase-2 failures stay four distinct injections.
    for injection in ("gather", "alloc", "transfer", "target"):
        assert injection in lowered, injection
    # Every injection and every parser has a non-vacuity control.
    assert re.search(r"non-vacuit|negative control|can_actually", lowered)
    # The fault-injection arm is disarmed in a ``finally`` *and* an autouse
    # fixture, so a failing test cannot leave it armed.
    assert "autouse=True" in hardening
    assert "finally" in hardening


def test_the_hardening_matrix_starts_no_thread_and_claims_no_race_is_safe():
    """Concurrency stays a documented boundary, never a tested safety
    claim — including in the module that attacks everything else."""
    identifiers = _code_identifiers(
        _read("tests/test_native_data_hardening.py"))
    for banned in ("Thread", "threading", "ThreadPoolExecutor",
                   "ProcessPoolExecutor", "multiprocessing", "asyncio",
                   "Queue", "Future"):
        assert banned not in identifiers, banned


def test_the_fault_injection_hook_is_the_one_documented_exception():
    """The deterministic thread-local allocation-failure arm is part of the
    export count and is inert until armed. There is exactly one, and J9 did
    not add a second."""
    exports = _source_exports()
    injection = {name for name in exports
                 if re.search(r"fault|inject|poison|arm_", name)}
    assert injection == {"tf_test_arm_alloc_failure",
                         "tf_fault_injection_available"}, sorted(injection)


# ===========================================================================
# 14. Benchmark evidence (J8)
# ===========================================================================

def _benchmark_module():
    sys.path.insert(0, str(REPO_ROOT / "benchmarks"))
    try:
        import benchmark_native_data_pipeline as module
    finally:
        sys.path.pop(0)
    return module


def test_the_benchmark_identity_and_schema_are_unchanged():
    module = _benchmark_module()

    assert module.BENCHMARK_NAME == BENCHMARK_NAME
    assert module.BENCHMARK_VERSION == BENCHMARK_VERSION
    assert module.SCHEMA_VERSION == BENCHMARK_SCHEMA_VERSION
    # Module constants, never package exports: there is no benchmark
    # registry inside ``tensorforge``.
    import tensorforge.experimental as experimental

    for name in ("BENCHMARK_NAME", "SCHEMA_VERSION", "CASES", "WORKLOADS"):
        assert not hasattr(experimental, name), name


def test_the_benchmark_answers_four_separate_questions_plus_one_composition():
    module = _benchmark_module()

    assert module.WORKLOADS == BENCHMARK_WORKLOADS
    assert len(module.CASES) == BENCHMARK_CASE_COUNT, sorted(module.CASES)
    assert module.DTYPES == FINAL_DTYPES
    assert set(module.REFERENCE_TYPES) == set(BENCHMARK_REFERENCES)
    # Every case belongs to exactly one workload, and every workload has at
    # least one case — so the four questions cannot silently merge.
    covered = {workload: 0 for workload in module.WORKLOADS}
    for case in module.CASES.values():
        covered[case["workload"]] += 1
    assert all(count > 0 for count in covered.values()), covered
    # The fifth family is a composition and is deliberately the smallest.
    assert covered["loader_delivery"] == 1
    # Both dtypes are measured, separately, and every case says so.
    for name, case in module.CASES.items():
        assert set(case["dtypes"]) <= set(FINAL_DTYPES), name


def test_only_the_host_only_cases_publish_a_ratio():
    """Never divide a native case that allocates by a NumPy case that does
    not. Planning, permutation, materialization, and delivery are all
    ``native_only`` and publish no ratio at all."""
    module = _benchmark_module()

    for name, case in module.CASES.items():
        if case["workload"] == "dataset_indexing":
            assert case["reference_type"] == "numpy", name
            assert case["native_only"] is False, name
            assert case["ratio_meaning"], name
        else:
            assert case["reference_type"] == "native_only", name
            assert case["native_only"] is True, name
            assert not case["ratio_meaning"], name


def test_the_two_dtypes_are_never_divided_ranked_or_averaged():
    """The one allowed cross-dtype claim is the index and permutation
    sequence, which carries no dtype at all."""
    identifiers = _code_identifiers(_read(J8_BENCHMARK))
    for banned in ("dtype_ratio", "dtype_speedup", "rank_dtypes",
                   "average_dtypes", "compare_dtypes"):
        assert banned not in identifiers, banned
    module = _benchmark_module()
    assert module.DISCLAIMER, "the harness no longer publishes its disclaimer"
    disclaimer = _flat(" ".join(module.DISCLAIMER)
                       if isinstance(module.DISCLAIMER, (tuple, list))
                       else str(module.DISCLAIMER)).lower()
    assert "machine" in disclaimer or "not a contract" in disclaimer


def test_the_benchmark_gates_correctness_before_timing_with_no_tolerance():
    module = _benchmark_module()

    assert set(module.GATES) >= {"host_gather_bits", "target_batch_exact",
                                 "plan_exact", "permutation_exact",
                                 "feature_batch_bits", "delivery_transaction"}
    identifiers = _code_identifiers(_read(J8_BENCHMARK))
    for banned in ("allclose", "isclose", "approx", "assert_allclose",
                   "atol", "rtol"):
        assert banned not in identifiers, banned
    # Cold and warm permutation construction stay separate cases and are
    # never averaged.
    assert module.CACHE_STATES == ("cold", "warm")
    cold = [n for n in module.CASES if n.startswith("permutation_cold")]
    warm = [n for n in module.CASES if n == "permutation_cache_hit"]
    assert len(cold) >= 2 and len(warm) == 1, (cold, warm)


def test_no_public_cache_or_benchmark_control_exists():
    import tensorforge.experimental as experimental
    from tensorforge.experimental import NativeBatchSampler

    for banned in ("clear_cache", "cache_clear", "set_cache", "warm_cache",
                   "invalidate_cache", "enable_benchmark", "set_threshold"):
        assert not hasattr(NativeBatchSampler, banned), banned
        assert not hasattr(experimental, banned), banned
        assert not hasattr(cpp, banned), banned


def test_the_benchmark_shipped_no_optimization_and_no_production_change():
    """J8's whole diff is a harness, a test module, inventory edits, and
    documentation. The claim is checkable structurally: no Phase-J
    production module imports or mentions the harness."""
    package = REPO_ROOT / "src" / "tensorforge" / "experimental"
    for path in sorted(package.glob("*.py")):
        identifiers = _code_identifiers(path.read_text(encoding="utf-8"))
        assert "benchmark_native_data_pipeline" not in identifiers, path.name
        for banned in ("perf_counter", "perf_counter_ns", "monotonic",
                       "process_time"):
            assert banned not in identifiers, (path.name, banned)


# ===========================================================================
# 15. Lifecycle and the concurrency boundary
# ===========================================================================

# Identifiers that would make concurrency a feature rather than a documented
# boundary. Scanned over *code*, never over prose, so the modules and the
# documents may explain the prohibition at any length.
_CONCURRENCY_NAMES = (
    "threading", "Thread", "Lock", "RLock", "Semaphore", "Condition",
    "Event", "Barrier", "Queue", "SimpleQueue", "Future",
    "ThreadPoolExecutor", "ProcessPoolExecutor", "concurrent",
    "multiprocessing", "asyncio", "aiofiles", "Process", "Pool",
    "_native_state_lock", "prefetch", "num_workers",
)


def _concurrency_offenders(text):
    """Every concurrency identifier that executes in ``text``."""
    identifiers = _code_identifiers(text)
    found = sorted(name for name in _CONCURRENCY_NAMES if name in identifiers)
    return found + sorted(_async_constructs(text))


@pytest.mark.parametrize("module", PHASE_J_MODULES)
def test_no_phase_j_module_contains_a_lock_thread_queue_or_async_primitive(
        module):
    package = REPO_ROOT / "src" / "tensorforge" / "experimental"
    offenders = _concurrency_offenders(
        (package / module).read_text(encoding="utf-8"))
    assert offenders == [], f"{module}: {offenders}"


def test_the_concurrency_scanner_can_actually_fail():
    """The scanner's non-vacuity control, on temporary strings only. It must
    catch code that imports or uses a primitive, and must **not** catch a
    module that merely documents the prohibition at length."""
    caught = (
        "import threading\nlock = threading.Lock()\n",
        "from queue import Queue\nq = Queue()\n",
        "import asyncio\n",
        "async def deliver():\n    pass\n",
        "async def go():\n    await something()\n",
        "from concurrent.futures import ThreadPoolExecutor\n",
        "def build(num_workers=4):\n    return num_workers\n",
        "from tensorforge.experimental import _native_state_lock\n",
    )
    for source in caught:
        assert _concurrency_offenders(source), source
    allowed = (
        '"""No lock, no thread, no queue, no Future, no prefetch, and no\n'
        'async iteration exists in this module; num_workers is not a\n'
        'parameter and threading.Lock is never imported."""\n'
        'BANNED = ("threading", "Lock", "Queue", "prefetch", "num_workers")\n',
        "# threading.Lock() would be a second authority\ndef plan():\n"
        "    return ()\n",
    )
    for source in allowed:
        assert _concurrency_offenders(source) == [], source


def test_no_test_in_the_phase_j_evidence_starts_a_thread():
    """"No test starts a thread, and none may." A test that did would be
    claiming a race is safe."""
    for name in ("tests/test_native_dataset.py", "tests/test_native_sampler.py",
                 "tests/test_native_data_loader.py",
                 "tests/test_native_loader_state.py",
                 "tests/test_native_data_checkpoint.py",
                 "tests/test_native_minibatch_training.py",
                 "tests/test_native_data_hardening.py",
                 "tests/test_native_data_benchmark.py",
                 "tests/test_native_phase_j.py"):
        offenders = _concurrency_offenders(_read(name))
        assert offenders == [], f"{name}: {offenders}"
    # ...including this module.
    assert _concurrency_offenders(
        Path(__file__).read_text(encoding="utf-8")) == []


def test_the_design_keeps_reentrancy_and_unsupported_concurrency_distinct():
    section = _flat(_design().split("\n## 16.", 1)[1].split("\n## 17.", 1)[0])
    lowered = section.lower()
    assert "reentran" in lowered
    assert "thread" in lowered
    assert re.search(r"not thread-safe|no thread safety|is not thread safe",
                     lowered), section[:400]
    assert re.search(r"external|caller", lowered), (
        "the design no longer places locking with the caller")


def test_close_exists_exactly_where_something_is_owned():
    """The dataset and the loader own something and close; the sampler owns
    nothing and has no ``close`` at all."""
    from tensorforge.experimental import (NativeBatchSampler, NativeDataLoader,
                                          NativeTensorDataset)

    for owner in (NativeTensorDataset, NativeDataLoader):
        assert hasattr(owner, "close"), owner.__name__
        assert hasattr(owner, "__enter__") and hasattr(owner, "__exit__")
    assert not hasattr(NativeBatchSampler, "close")
    assert not hasattr(NativeBatchSampler, "__enter__")


@pytest.mark.skipif(not cpp.is_available(),
                    reason="the native library is not built")
def test_close_is_idempotent_and_the_finalizer_is_only_a_fallback():
    from tensorforge.experimental import (NativeBatchSampler, NativeDataLoader,
                                          NativeTensorDataset)

    with _LiveStorages() as live:
        dataset = NativeTensorDataset(np.zeros((4, 2)),
                                      np.zeros(4, dtype=np.int64))
        loader = NativeDataLoader(NativeBatchSampler(dataset, batch_size=2))
        loader.close()
        loader.close()                   # idempotent
        assert loader.closed
        dataset.close()
        dataset.close()
        assert dataset.closed
        assert len(live) == 0
    # ``__del__`` exists as a fallback, but ``close`` is the contract.
    assert hasattr(type(loader), "close")


@pytest.mark.skipif(not cpp.is_available(),
                    reason="the native library is not built")
def test_a_whole_iteration_returns_live_storage_exactly_to_baseline():
    """One epoch, every delivered batch closed by the caller, and the count
    back where it started — no GC-timing assertion anywhere."""
    from tensorforge.experimental import (NativeBatchSampler, NativeDataLoader,
                                          NativeTensorDataset)

    with _LiveStorages() as live:
        dataset = NativeTensorDataset(
            np.arange(30, dtype=np.float64).reshape(10, 3),
            np.arange(10, dtype=np.int64) % 3)
        try:
            loader = NativeDataLoader(
                NativeBatchSampler(dataset, batch_size=4, shuffle=True,
                                   seed=3))
            try:
                delivered = 0
                for batch_features, _ in iter(loader):
                    delivered += 1
                    batch_features.close()
                assert delivered == 3           # ceil(10 / 4)
                assert len(live) == 0
                # One iterator is one epoch: the position is canonicalized
                # to the next epoch's start rather than left past the end.
                assert loader.sampler.epoch == 1
                assert loader.sampler.cursor == 0
            finally:
                loader.close()
        finally:
            dataset.close()
        assert len(live) == 0


# ===========================================================================
# 16. Stable / native isolation
# ===========================================================================

def test_importing_the_stable_framework_does_not_load_the_native_backend():
    code = (
        "import sys\n"
        "import tensorforge, tensorforge.nn, tensorforge.optim, tensorforge.data\n"
        "loaded = [m for m in sys.modules if m.endswith('backends.cpp')]\n"
        "assert not loaded, loaded\n"
        "assert not [m for m in sys.modules if 'experimental' in m]\n"
        "print('isolated')\n"
    )
    done = subprocess.run([sys.executable, "-c", code], cwd=REPO_ROOT,
                          capture_output=True, text=True)
    assert done.returncode == 0, done.stderr
    assert "isolated" in done.stdout


def test_the_backend_still_reports_no_stable_integration():
    assert cpp.backend_info()["stable_framework_integration"] is False


@pytest.mark.skipif(not cpp.is_available(),
                    reason="the native library is not built")
def test_a_stable_tensor_is_refused_as_dataset_features():
    """No implicit conversion in either direction: the dataset takes
    ``numpy.ndarray`` exactly."""
    import tensorforge
    from tensorforge.experimental import NativeTensorDataset

    stable = tensorforge.Tensor(np.zeros((4, 2)))
    with pytest.raises(TypeError):
        NativeTensorDataset(stable, np.zeros(4, dtype=np.int64))
    with pytest.raises(TypeError):
        NativeTensorDataset(np.zeros((4, 2)), stable)
    # ...and a plain Python list is not an ndarray either.
    with pytest.raises(TypeError):
        NativeTensorDataset([[0.0, 1.0]] * 4, np.zeros(4, dtype=np.int64))


def test_no_phase_j_module_imports_the_stable_line():
    package = REPO_ROOT / "src" / "tensorforge" / "experimental"
    for name in PHASE_J_MODULES:
        source = (package / name).read_text(encoding="utf-8")
        for node in ast.walk(ast.parse(source)):
            if isinstance(node, ast.ImportFrom):
                module = node.module or ""
                assert not re.match(r"tensorforge\.(nn|optim|data)\b", module), (
                    name, module)
                assert module != "tensorforge", (name, module)
            if isinstance(node, ast.Import):
                for alias in node.names:
                    assert not re.match(r"tensorforge\.(nn|optim|data)\b",
                                        alias.name), (name, alias.name)


def test_the_stable_batches_helper_was_not_wrapped_or_replaced():
    """``tensorforge.data.batches`` is the stable line's own iterator and
    stays exactly where it is; nothing in the native line wraps it."""
    import tensorforge.data as stable_data

    assert hasattr(stable_data, "batches")
    package = REPO_ROOT / "src" / "tensorforge" / "experimental"
    for name in PHASE_J_MODULES:
        identifiers = _code_identifiers(
            (package / name).read_text(encoding="utf-8"))
        assert "batches" not in identifiers, name


def test_no_environment_variable_steers_the_pipeline():
    package = REPO_ROOT / "src" / "tensorforge" / "experimental"
    for name in PHASE_J_MODULES:
        tree = ast.parse((package / name).read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute) and node.attr in {"getenv",
                                                                 "environ"}:
                raise AssertionError(f"{name} consults the environment")
            if isinstance(node, ast.Name) and node.id in {"environ", "getenv"}:
                raise AssertionError(f"{name} consults the environment")


def test_no_global_or_default_dataset_sampler_or_loader_exists():
    import tensorforge.experimental as experimental

    for banned in ("DEFAULT_DATASET", "DEFAULT_LOADER", "DEFAULT_SAMPLER",
                   "default_dataset", "default_loader", "default_sampler",
                   "current_loader", "set_default_loader"):
        assert not hasattr(experimental, banned), banned


# ===========================================================================
# 17. Unsupported side capabilities, stated as claims rather than as tokens
# ===========================================================================
#
# Context-aware: an accurate sentence saying workers are *not* supported must
# pass, and only a claim that they *are* may fail.

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
    ("the pipeline is thread-safe or concurrent",
     r"\b(thread[- ]safe|concurrent|parallel|multi[- ]threaded)\b"
     r"[^.;]{0,60}?\b(is|are|now)\s+"
     r"(supported|available|implemented|guaranteed)\b"
     r"|\b(loader|sampler|dataset|pipeline)s?\b[^.;]{0,40}?\bis thread[- ]safe\b"),
    ("workers, prefetch, or async iteration exist",
     r"\b(worker processes|worker threads|prefetch(ing)?|"
     r"asynchronous iteration|background workers)\b"
     r"[^.;]{0,60}?\b(is|are|now)\s+"
     r"(supported|available|implemented|shipped|enabled)\b"
     r"|\bsupports?\s+(workers|prefetch(ing)?|multiprocessing)\b"),
    ("loader discovery or automatic restoration exists",
     r"\b(loader discovery|automatic loader|automatically (restores?|"
     r"discovers?|reconstructs?))\b[^.;]{0,60}?\b(is|are|now)\s+"
     r"(supported|available|implemented|shipped)\b"
     r"|\bthe checkpoint (automatically )?(restores?|discovers?) the "
     r"(data )?loader\b"),
    ("the stable framework dispatches to the native backend",
     r"\bstable\b[^.;]{0,60}?\b(automatically|implicitly)\s+"
     r"(uses|dispatches|selects|falls back)\b"),
)

_NEGATIONS = re.compile(
    r"\b(no|not|never|neither|nor|none|without|absent|unsupported|"
    r"planned|future|beyond|until|once|when|will|would|before|yet|"
    r"rejected|refused|forbidden|outside|cannot|must)\b", re.I)


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
        "AMP is now available",
        "float16 is supported",
        "casting is supported between the two dtypes",
        "the loader is thread-safe",
        "worker processes are supported",
        "TensorForge supports prefetching",
        "the checkpoint automatically restores the data loader",
    ):
        assert _overclaims(caught), caught
    for allowed in (
        "float32 and float64 are supported on the CPU",
        "CUDA and AMP remain unsupported",
        "there is no casting and no promotion",
        "the loader is not thread-safe and external locking is the caller's job",
        "worker processes are not supported and none may be added",
        "prefetching does not exist",
        "no automatic loader discovery exists, in either direction",
        "device movement does not exist and none may be added",
    ):
        assert _overclaims(allowed) == [], (allowed, _overclaims(allowed))


def test_the_absent_capabilities_are_genuinely_unreachable():
    """The absence, stated over the live objects rather than promised in
    prose."""
    import tensorforge.experimental as experimental
    from tensorforge.experimental import (NativeBatchSampler, NativeDataLoader,
                                          NativeTensorDataset)

    banned = ("astype", "to", "cast", "float", "double", "half", "cuda",
              "cpu", "type_as", "promote", "map_location", "pin_memory",
              "share_memory", "memmap", "download", "from_url", "from_path",
              "from_files", "stream")
    for target in (NativeTensorDataset, NativeBatchSampler, NativeDataLoader):
        for name in banned:
            assert not hasattr(target, name), (target.__name__, name)
    for name in ("set_default_dtype", "get_default_dtype", "default_dtype",
                 "DEFAULT_DTYPE", "set_default_device"):
        assert not hasattr(experimental, name), name
        assert not hasattr(cpp, name), name


@pytest.mark.skipif(not cpp.is_available(),
                    reason="the native library is not built")
def test_no_device_argument_is_accepted_anywhere_in_the_pipeline():
    from tensorforge.experimental import (NativeBatchSampler, NativeDataLoader,
                                          NativeTensorDataset)

    features = np.zeros((4, 2), dtype=np.float64)
    targets = np.zeros(4, dtype=np.int64)
    with pytest.raises(TypeError):
        NativeTensorDataset(features, targets, device="cpu")
    dataset = NativeTensorDataset(features, targets)
    try:
        with pytest.raises(TypeError):
            NativeBatchSampler(dataset, batch_size=2, device="cpu")
        sampler = NativeBatchSampler(dataset, batch_size=2)
        with pytest.raises(TypeError):
            NativeDataLoader(sampler, device="cpu")
        # ...and no dtype on the classes that own no dtype-bearing state.
        with pytest.raises(TypeError):
            NativeBatchSampler(dataset, batch_size=2, dtype="float32")
        with pytest.raises(TypeError):
            NativeDataLoader(sampler, dtype="float32")
    finally:
        dataset.close()


def test_no_external_dependency_was_introduced():
    """NumPy is still the only numeric dependency, and the C++ backend still
    needs only a C++17 compiler."""
    project = _read("pyproject.toml")
    declared = re.search(r"dependencies\s*=\s*\[(.*?)\]", project, re.S)
    assert declared, "pyproject declares no dependency list"
    body = declared.group(1).lower()
    for banned in ("torch", "tensorflow", "jax", "sklearn", "scikit",
                   "pandas", "matplotlib", "pybind11", "eigen", "onednn",
                   "pillow", "requests", "aiohttp"):
        assert banned not in body, banned
    assert "numpy" in body


# ===========================================================================
# 18. Workflow and shallow-clone safety
# ===========================================================================

def test_the_workflow_still_builds_smoke_tests_benchmarks_and_runs_pytest():
    workflow = WORKFLOW.read_text(encoding="utf-8")
    for step in ("uv run python cpp/build.py",
                 "uv run python scripts/smoke_cpp_backend.py",
                 "uv run python benchmarks/cpp_backend.py --quick",
                 "uv run pytest"):
        assert step in workflow, step
    assert "actions/checkout" in workflow


def test_the_workflow_needs_no_clone_depth_and_no_history():
    workflow = WORKFLOW.read_text(encoding="utf-8")
    assert "fetch-depth" not in workflow, (
        "the workflow now configures clone depth; the guards must not need it")


def test_no_closure_guard_reaches_for_a_historical_git_object():
    """Parsed rather than grepped, because this module's own prose
    legitimately names ``git show`` and ``ls-tree`` while explaining their
    absence."""
    closure = Path(__file__).read_text(encoding="utf-8")
    tree = ast.parse(closure)
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        for argument in node.args:
            if not isinstance(argument, ast.List):
                continue
            literals = [element.value for element in argument.elts
                        if isinstance(element, ast.Constant)]
            if "git" in literals:
                assert not ({"show", "ls-tree", "cat-file", "rev-list",
                             "diff", "log", "rev-parse", "merge-base"}
                            & set(literals)), literals
    # No commit SHA is spelled anywhere in the module either.
    assert not re.search(r"\b[0-9a-f]{40}\b", closure), (
        "the closure module names a commit object")
    assert not re.search(r"\b[0-9a-f]{7,12}\b\s*(?:commit|SHA)", closure, re.I)


def test_the_tracked_set_helper_degrades_rather_than_passing_falsely():
    """The one Git call in the module asks about the working tree and index
    and skips when it cannot be read — never returning an empty set that
    would make every tracked-file guard vacuously true."""
    source = inspect.getsource(_git_lines)
    assert "pytest.skip" in source
    assert source.count("pytest.skip") == 2, (
        "one of the two failure paths no longer skips")
    for historical in ("show", "ls-tree", "cat-file", "rev-list", "log",
                       "diff", "rev-parse"):
        assert f'"{historical}"' not in source, historical
    # Both callers ask only about the index or the working tree.
    for caller in (_tracked_files, _untracked_unignored_files):
        assert "ls-files" in inspect.getsource(caller), caller.__name__
    # ...and the helper is genuinely non-vacuous here: this very file is in
    # one of the two sets, so neither can be silently empty.
    relative = Path(__file__).relative_to(REPO_ROOT).as_posix()
    assert relative in set(_tracked_files()) | set(
        _untracked_unignored_files()), relative


def test_the_current_tree_inventory_checks_need_no_repository_at_all(tmp_path):
    """The behavioural half of the portability proof: every inventory this
    module pins is read from files, so it gives the same verdict in a
    directory that has never been a clone."""
    staging = tmp_path / "cpp" / "src"
    staging.mkdir(parents=True)
    for source in sorted((REPO_ROOT / "cpp" / "src").glob("*.cpp")):
        (staging / source.name).write_bytes(source.read_bytes())
    assert not (tmp_path / ".git").exists()

    names = set()
    for source in sorted(staging.glob("*.cpp")):
        names.update(re.findall(r"TF_EXPORT[^;{]*?\b(tf_[a-z0-9_]+)\s*\(",
                                source.read_text(encoding="utf-8"), re.S))
    assert len(names) == CURRENT_EXPORT_COUNT
    assert names == _source_exports()

    # ...and a deliberately mutated copy is still caught there, so the
    # reader is proved able to notice a change.
    victim = next(iter(sorted(staging.glob("*.cpp"))))
    victim.write_text(victim.read_text(encoding="utf-8")
                      + "\nTF_EXPORT void tf_invented_symbol(void) {}\n",
                      encoding="utf-8")
    mutated = set()
    for source in sorted(staging.glob("*.cpp")):
        mutated.update(re.findall(r"TF_EXPORT[^;{]*?\b(tf_[a-z0-9_]+)\s*\(",
                                  source.read_text(encoding="utf-8"), re.S))
    assert len(mutated) == CURRENT_EXPORT_COUNT + 1
    assert "tf_invented_symbol" in mutated


def test_every_prose_scan_is_line_ending_agnostic():
    """A CRLF working tree must give the same verdict as an LF one, which is
    why every scan runs over ``_flat`` output."""
    sample = "Phase J\r\nis complete.\r\n"
    assert _flat(sample) == _flat(sample.replace("\r\n", "\n"))
    assert _PHASE_COMPLETE.search(_flat(sample))
    assert _stale_status(_flat("Phase J\r\nis not complete.\r\n"))


# ===========================================================================
# 19. Repository hygiene — no generated artifact entered
# ===========================================================================

_ARTIFACT_SUFFIXES = (".so", ".dll", ".dylib", ".pyd", ".obj", ".o", ".lib",
                      ".pdb", ".exp", ".a", ".npz", ".npy", ".ckpt", ".pt",
                      ".pth", ".coverage", ".swp", ".bak", ".orig", ".rej",
                      ".ilk", ".idb", ".core", ".log")


def test_no_generated_artifact_is_tracked():
    for path in _tracked_files():
        lowered = path.lower()
        assert not lowered.endswith(_ARTIFACT_SUFFIXES), path
        assert not re.search(r"(^|/)(build|dist|\.venv|venv|node_modules|"
                             r"__pycache__|\.pytest_cache|\.cache|"
                             r"cmake-build[^/]*|tf-j\d[^/]*)/",
                             lowered), path
        assert not re.search(r"(^|/)(asan|ubsan|lsan|tsan)[-_.]", lowered), path
        assert "suppress" not in lowered, path


def test_no_generated_artifact_sits_unignored_in_the_working_tree_either():
    """Builds, sanitizer output, CTest logs, compiler transcripts, copied
    libraries, checkpoints, and core dumps all live **outside** the
    repository — and where one is produced inside it, ``.gitignore`` must
    cover it so it can never become content.

    Scoped to untracked-**and-unignored** paths, which is the set that can
    actually be committed. An ignored local build tree is what that
    mechanism is for, and failing on one would make this guard
    machine-specific rather than durable."""
    for relative in _untracked_unignored_files():
        lowered = relative.lower()
        assert not lowered.endswith(
            (".obj", ".o", ".lib", ".pdb", ".exp", ".ilk", ".idb", ".a",
             ".so", ".dll", ".dylib", ".pyd", ".npz", ".npy", ".ckpt", ".pt",
             ".pth", ".core", ".log")), relative
        assert "cmakecache" not in lowered, relative
        assert not re.search(r"(^|/)(asan|ubsan|lsan|tsan)[-_.]",
                             lowered), relative
        assert not re.search(r"suppressions?(\.txt|\.supp)?$",
                             lowered), relative
        assert not re.search(r"(^|/)core(\.\d+)?$", lowered), relative
        assert "benchmark_result" not in lowered, relative
    # The built library lands beside the wrapper; it must never be tracked,
    # and it must be ignored so it cannot be added by accident.
    tracked = set(_tracked_files())
    for suffix in (".dll", ".so", ".dylib", ".pyd"):
        for path in (REPO_ROOT / "src" / "tensorforge" / "backends").glob(
                f"*{suffix}"):
            relative = path.relative_to(REPO_ROOT).as_posix()
            assert relative not in tracked, relative
            assert relative not in _untracked_unignored_files(), relative


def test_no_temporary_cpp_source_or_generated_manifest_entered():
    """Scoped to the tracked and unignored ``cpp/`` tree. A local
    ``cpp/build/`` produced by ``cpp/build.py`` is ignored by construction
    and holds exactly the object files and dependency manifests a build is
    supposed to produce; failing on it would make this guard
    machine-specific rather than a statement about repository content."""
    candidates = [path for path in _tracked_files()
                  + _untracked_unignored_files()
                  if path.startswith("cpp/")]
    assert candidates, "the cpp/ tree reads as empty; the guard is vacuous"
    for relative in candidates:
        name = Path(relative).name.lower()
        assert not name.startswith("tmp"), relative
        assert not name.endswith((".i", ".ii", ".s", ".d", ".o", ".obj")), (
            relative)
        assert "generated" not in name, relative


def test_every_tracked_text_file_is_valid_utf8_without_a_bom():
    """Committed bytes, so a CRLF checkout cannot create a false positive:
    line endings are deliberately not asserted."""
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


# ===========================================================================
# 20. Evidence floors, and the agent instructions
# ===========================================================================

@pytest.mark.parametrize("relative,minimum", REQUIRED_EVIDENCE)
def test_the_phase_j_evidence_files_are_all_present(relative, minimum):
    """A floor rather than an equality: adding coverage is free, deleting it
    is not. Exact totals are deliberately never made permanent."""
    path = REPO_ROOT / relative
    assert path.is_file(), relative
    count = len(re.findall(r"^def test_", path.read_text(encoding="utf-8"),
                           re.M))
    assert count >= minimum, (
        f"{relative} has {count} tests, expected >= {minimum}")


def test_the_evidence_floor_reader_can_actually_fail(tmp_path):
    """The floor parser's negative control, on a temporary file only."""
    good = tmp_path / "test_sample.py"
    good.write_text("def test_a():\n    pass\n\n\ndef test_b():\n    pass\n",
                    encoding="utf-8")
    counted = len(re.findall(r"^def test_", good.read_text(encoding="utf-8"),
                             re.M))
    assert counted == 2
    thin = tmp_path / "test_thin.py"
    thin.write_text('"""def test_looks_like_one but is prose."""\n',
                    encoding="utf-8")
    assert len(re.findall(r"^def test_", thin.read_text(encoding="utf-8"),
                          re.M)) == 0


def test_claude_md_records_the_final_phase_j_truth():
    """Facts and pointers, never a phrasing or a length."""
    text = AGENT_INSTRUCTIONS.read_text(encoding="utf-8")
    for value in FINAL_DTYPES + FINAL_DEVICES + FINAL_UNSUPPORTED:
        assert value in text, value
    assert "RAW_KERNEL_DTYPES" in text
    for number in (FINAL_EXPORT_COUNT, FINAL_CTEST_COUNT, FINAL_EXAMPLE_COUNT,
                   FINAL_BENCHMARK_COUNT, FINAL_EXPERIMENTAL_EXPORTS):
        assert str(number) in text, number
    assert re.search(rf"version\s*\**\s*{FINAL_CHECKPOINT_VERSION}\b", text)
    assert FINAL_LOADER_STATE_FORMAT in text
    assert FINAL_SAMPLER_STATE_FORMAT in text
    assert FINAL_CHECKPOINT_FORMAT in text
    for document in ("docs/native_data_pipeline_design.md",
                     "docs/native_support_matrix.md",
                     "docs/release_history.md",
                     "docs/roadmap.md"):
        assert document in text, document
    # The phase's status, and the boundary it stopped at.
    assert re.search(r"Phase J[^.]{0,80}complete", text, re.I)
    assert _PHASE_COMPLETE.search(_flat(text)), (
        "CLAUDE.md does not mark Phase J complete")


def test_claude_md_stayed_inside_the_project_memory_budget():
    """A soft structural bound: closure adds status, not a transcript.

    The ceiling is the 150,000-character project-memory limit, and it is a
    ceiling rather than a target — an active phase may grow the file with
    operational detail an implementer genuinely needs. What it may not grow
    with is milestone history. Measured LF-normalized so a CRLF checkout
    cannot change the verdict."""
    text = AGENT_INSTRUCTIONS.read_text(encoding="utf-8")
    size = len(text.replace("\r\n", "\n"))
    assert size < 150_000, (
        f"CLAUDE.md has grown to {size} characters; milestone history "
        f"belongs in docs/, not in project memory")


def test_claude_md_keeps_the_durable_operating_rules_a_successor_needs():
    """Closure cleanup removes milestone transcripts, never operating
    rules. Each subject below is checked as a *fact or pointer*, not as a
    phrasing, so the file may be reorganised freely."""
    text = AGENT_INSTRUCTIONS.read_text(encoding="utf-8")
    flat = _flat(text)
    for subject in (
        # Identity, layout, and workflow.
        "TensorForge", "uv run pytest", "cpp/build.py",
        "src/tensorforge/backends/cpp.py", "experimental",
        # The separation and the loading rule.
        "stable_framework_integration", "lazily",
        # The numerical contracts a successor must not weaken.
        "bit-identical", "one-ULP",
        # Ownership and the delivery transaction.
        "close()", "if and only if",
        # Benchmarks and validation.
        "characterization", "sanitizer", "Release", "Debug",
        # Version control.
        "Git",
    ):
        assert subject in text, subject
    for pattern in (
        r"never commit|user performs every Git-writing",
        r"no result file|writes no result file|no committed benchmark",
        r"external locking|not thread-safe",
        r"caller-managed|caller-owned|caller managed",
    ):
        assert re.search(pattern, flat, re.I), pattern


def test_the_design_is_linked_from_the_readme_and_the_instructions():
    for surface in ("README.md", "CLAUDE.md"):
        assert "docs/native_data_pipeline_design.md" in _read(surface), surface


def test_the_exit_gate_section_still_exists_and_is_resolved():
    """§24.2 is the phase's own gate. At closure every row is answered, and
    the one row that cannot be answered locally is stated as the external
    confirmation it is rather than checked off silently."""
    gate = _design().split("### 24.2", 1)
    assert len(gate) == 2, "the design has no J9 exit gate"
    body = gate[1].split("\n---", 1)[0]
    rows = re.findall(r"^- \[([ x])\]", body, re.M)
    assert rows, "the J9 exit gate has no rows"
    # Nothing left unmarked except an explicitly external gate.
    unresolved = [line for line in body.splitlines()
                  if line.startswith("- [ ]")]
    for line in unresolved:
        assert re.search(r"GitHub Actions|post-commit|external", line, re.I), (
            f"an unresolved local gate row remains: {line!r}")
