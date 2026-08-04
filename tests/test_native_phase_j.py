"""Phase-J contract guardrails (deterministic native data pipeline).

**Phase J is newly approved, and milestones J0 through J6 have landed.**
J0 was an architecture, contract, documentation, and status milestone: it
shipped ``docs/native_data_pipeline_design.md``, this module, and
documentation, and **no runtime behavior at all**. **J1** shipped the
first runtime — ``NativeTensorDataset``, the finite host-backed dataset —
**J2** the second — ``NativeBatchSampler``, the deterministic order and
batch planner, over the permanently private ``_native_permutation``
derivation — and **J3** the third and last of §3.1's names,
``NativeDataLoader``, with its private ``_NativeBatchIterator`` and the
transactional batch delivery behind it. Each of those three added exactly
one new public experimental name. **J4** added **none**: it gave the
existing loader its own in-memory ``state_dict``/``load_state_dict`` and
exact mid-epoch restoration, and left ``__all__`` at 25. **J5** added no
public name *and no production code at all*: it proved that the loader
state a caller already had survives a real version-3 archive as ordinary
metadata and restores an exact continuation into entirely fresh objects,
leaving the checkpoint module untouched. **J6** added no public name
either, and no production code: it is
``examples/native_minibatch_training.py`` — the first end-user program to
train through the pipeline — and its exact interrupted-versus-resumed
proof, taking the example inventory from 15 to **16**. None of the six
added C++, a C ABI symbol, a benchmark, or a checkpoint or
optimizer-state change. Their own behavior is covered by
``tests/test_native_dataset.py``, ``tests/test_native_sampler.py``,
``tests/test_native_data_loader.py``,
``tests/test_native_loader_state.py``,
``tests/test_native_data_checkpoint.py``, and
``tests/test_native_minibatch_training.py``; what lives here is the
*phase* boundary.

**J7 through J9 have not started**, so the adversarial hardening matrix,
the benchmark, and the closure module are still asserted **absent**
below. A caller can serialize where a loader stopped, carry it through an
archive, restore it exactly, and now read a worked example doing exactly
that — but nothing discovers a loader for them, no benchmark ships, and
Phase J is not complete.

Three kinds of fact live here, and keeping them apart is the point of the
module:

* **What the contract says** — a property of the design document, which
  spans the whole phase and does not move as milestones land. These
  assertions are **section-scoped** and require **combinations** of
  architectural terms rather than the presence of one vague word: a
  document that merely contains the string "shuffle" passes nothing here.
* **What the repository is now** — the live registries, the live source,
  the built library, and real files, at J0.
* **What is still a promise** — everything J1 onward will do, asserted as
  *absent*, so a later milestone cannot be mistaken for an earlier one and
  so a placeholder class cannot appear without failing a test.

They deliberately test *values and structure* rather than wording, so
ordinary prose improvements do not require rewriting them. Nothing here
asserts a character count, a paragraph order, an error message, or a
benchmark number.

**Every parser in this module has a negative control.** A checker that
silently stopped matching would pass forever, so each one is driven against
text it must reject as well as text it must accept. The controls operate on
temporary strings only; no repository file is read for mutation, written,
moved, or restored.
"""
import ast
import re
from pathlib import Path

import pytest

from tensorforge.backends import cpp

REPO_ROOT = Path(__file__).resolve().parent.parent
PHASE_J_DESIGN_NAME = "native_data_pipeline_design.md"
PHASE_J_DESIGN = REPO_ROOT / "docs" / PHASE_J_DESIGN_NAME

# ---------------------------------------------------------------------------
# The boundary Phase J inherits from a completed Phase I, and must not move.
# Written here independently of the modules under test, so a silent change
# fails rather than propagating.
# ---------------------------------------------------------------------------
J0_DTYPES = ("float64", "float32")
J0_DEVICES = ("cpu",)
J0_UNSUPPORTED = ("cuda", "amp")
J0_RAW_KERNEL_DTYPES = ("float64",)
J0_DEFAULT_DTYPE = "float64"

J0_CHECKPOINT_FORMAT = "tensorforge.native_checkpoint"
J0_CHECKPOINT_VERSION = 3
J0_CHECKPOINT_VERSIONS = (1, 2, 3)
J0_OPTIMIZER_STATE_VERSION = 1

J0_EXPORT_COUNT = 54
J0_CTEST_COUNT = 24
J0_EXAMPLE_COUNT = 15
J0_BENCHMARK_COUNT = 8

# The artifacts Phase J has shipped so far, each mapped to the milestone
# that added it. J0-J5 added none; **J6** added exactly one example. The
# current count is derived rather than restated, so an unannounced example
# fails the equality below instead of being absorbed into a bumped literal.
PHASE_J_EXAMPLES = {"native_minibatch_training.py": "J6"}
PHASE_J_BENCHMARKS = {}                                  # J8 owns the first
CURRENT_EXAMPLE_COUNT = J0_EXAMPLE_COUNT + len(PHASE_J_EXAMPLES)
CURRENT_BENCHMARK_COUNT = J0_BENCHMARK_COUNT + len(PHASE_J_BENCHMARKS)

MILESTONES = tuple(f"J{index}" for index in range(10))   # J0 ... J9

# The three eventual public names, and the milestone that adds each.
PLANNED_CLASSES = {
    "NativeTensorDataset": "J1",
    "NativeBatchSampler": "J2",
    "NativeDataLoader": "J3",
}

# Split by what has actually landed. ``LANDED_CLASSES`` must exist and be
# exported; ``UNLANDED_CLASSES`` must not exist at all. A milestone moves a
# name from the second set to the first and nowhere else, so the two
# together are always exactly ``PLANNED_CLASSES``.
LANDED_CLASSES = {"NativeTensorDataset", "NativeBatchSampler",
                  "NativeDataLoader"}                            # J1, J2, J3
UNLANDED_CLASSES = set()                                         # J4 onward
assert LANDED_CLASSES | UNLANDED_CLASSES == set(PLANNED_CLASSES)

# The Phase-J modules under ``src/tensorforge/experimental``, split the
# same way. ``_native_permutation`` is J2's derivation helper and is
# **permanently private**: it exists, and it must never be exported.
LANDED_MODULES = ("native_dataset.py", "native_sampler.py",
                  "_native_permutation.py", "native_data_loader.py")
UNLANDED_MODULES = ()

# The private names J3 added inside ``native_data_loader.py``. They exist
# and must **never** be exported: a caller receives an iterator from
# ``iter(loader)`` and never constructs one, and the delivery seam is a
# test seam rather than a hook.
PRIVATE_LOADER_NAMES = ("_NativeBatchIterator", "_deliver_batch")

# The milestones complete right now, and the ones still promised. The
# ladder parser below is driven from these rather than from a hard-coded
# "only J0", so landing a milestone is a one-line change here and a
# document edit — never a loosened checker.
COMPLETE_MILESTONES = ("J0", "J1", "J2", "J3", "J4", "J5", "J6")
UNSTARTED_MILESTONES = tuple(name for name in MILESTONES
                             if name not in COMPLETE_MILESTONES)

# The locked derivation Phase J reuses rather than replacing. These are the
# Phase-G constants, spelled out here so a change to either side fails.
GOLDEN = 0x9E3779B97F4A7C15
SPLITMIX_MULTIPLIERS = (0xBF58476D1CE4E5B9, 0x94D049BB133111EB)
SAMPLER_DOMAIN = 0x54465F53414D504C


def _read(relative):
    return (REPO_ROOT / relative).read_text(encoding="utf-8")


def _design():
    return PHASE_J_DESIGN.read_text(encoding="utf-8")


def _flat(text):
    """Whitespace-flattened, emphasis-stripped text, so a claim split
    across lines or wrapped in markdown still reads as one sentence."""
    return re.sub(r"\s+", " ", re.sub(r"[*`]", "", text))


def _section(text, number):
    """The body of top-level section ``number``, up to the next one.

    Section-scoped rather than whole-document, so an assertion about §8
    cannot be satisfied by a sentence that happens to appear in §2."""
    marker = f"\n## {number}."
    assert marker in text, f"the design has no section {number}"
    body = text.split(marker, 1)[1]
    following = re.search(r"\n## \d+\.", body)
    return body[:following.start()] if following else body


def _requires(haystack, *terms):
    """Every term present, case-insensitively, in one flattened body."""
    flat = _flat(haystack).lower()
    return [term for term in terms if term.lower() not in flat]


# ===========================================================================
# 1. The design document exists and is reachable
# ===========================================================================

def test_the_phase_j_design_exists_and_is_not_empty():
    assert PHASE_J_DESIGN.is_file(), f"missing docs/{PHASE_J_DESIGN_NAME}"
    assert len(_design().strip()) > 10_000, (
        "the Phase-J contract is too short to be an implementation contract"
    )


def test_the_phase_j_design_is_linked_from_the_readme_and_the_doc_map():
    """A contract nobody can find is not a contract. The README links every
    document, and CLAUDE.md's documentation map names the authority for
    each question."""
    for surface in ("README.md", "CLAUDE.md"):
        assert f"docs/{PHASE_J_DESIGN_NAME}" in _read(surface), surface


def test_the_design_names_the_phase_and_its_subject():
    heading = _design().splitlines()[0]
    missing = _requires(heading, "Phase J")
    assert not missing, heading
    assert _requires(_design()[:4000], "data pipeline", "mini-batch") == []


# ===========================================================================
# 2. Phase J is newly approved, after a completed Phase I
# ===========================================================================
#
# The failure this section exists for: a document that reads Phase J as
# though it had always been on the roadmap, which would make "Phase I closed
# without a successor" retroactively false.

def test_the_design_presents_phase_j_as_newly_approved_after_phase_i():
    head = _flat(_design()[:6000])
    assert re.search(r"newly\s+approved", head, re.I), (
        "the design does not say Phase J is newly approved"
    )
    assert re.search(r"Phase I remains complete", head, re.I), head[:400]
    # ...and it says explicitly that the phase was approved after Phase I
    # closed, rather than having been planned all along.
    assert re.search(r"without committing to a successor|approved afterwards",
                     head, re.I), (
        "the design does not record that Phase I closed without a successor"
    )


def test_the_design_states_exactly_which_runtime_exists():
    """The header carries the phase's status, and from J1 on that status
    has two halves: what landed, and what still has not. Both are
    required, so a future milestone cannot quietly drop the second.

    Driven from ``COMPLETE_MILESTONES`` rather than from a hard-coded
    sentence, so the check keeps meaning the same thing as milestones
    land instead of being rewritten into agreement each time.
    """
    head = _flat(_design()[:6000])
    landed = ", ".join(COMPLETE_MILESTONES[:-1]) + f", and {COMPLETE_MILESTONES[-1]}"
    next_up = UNSTARTED_MILESTONES[0]
    last = UNSTARTED_MILESTONES[-1]
    assert re.search(r"J0 added no runtime behavior", head, re.I), head[:400]
    assert re.search(r"[Rr]untime capability began at \*{0,2}J1", head), head[:400]
    assert re.search(rf"{landed} complete; {next_up} through {last} not "
                     rf"started", head, re.I), head[:800]
    # The absent half, named rather than implied: what the *next*
    # milestones will add must still be spelled out as missing. J4 landed
    # the loader state schema, its two methods, and exact in-memory
    # mid-epoch restoration; J5 landed the caller-managed
    # checkpoint-metadata workflow; J6 landed the training example and its
    # exact resume proof. Each moved out of this list and into the presence
    # checks below in the milestone that shipped it, rather than being
    # softened.
    for absent in ("automatic loader discovery", "hardening matrix",
                   "benchmark"):
        assert re.search(rf"no {absent}", head, re.I), absent
    # ``head`` is emphasis-stripped, so these are the flattened spellings.
    for present in ("loader state schema", "loader state_dict",
                    "exact in-memory mid-epoch loader restoration",
                    "caller-managed checkpoint-metadata workflow",
                    "real version-3 archive",
                    "deterministic native mini-batch training example",
                    "exact interrupted-versus-uninterrupted training"):
        assert present.lower() in head.lower(), present
    # ...and J5's defining negative: the archive did not grow.
    assert re.search(r"capture set did not grow", head, re.I), head[:1200]
    assert re.search(rf"{next_up} is\s+(the\s+)?next", head, re.I), head[:800]


# The one over-claim pattern, shared by the design scan and the status-surface
# scan below so the two cannot drift apart. Two arms, because the claim has two
# ordinary shapes: "<subject> is supported", and "<something> supports
# <subject>" — the second is invisible to the first.
_RUNTIME_CLAIM = re.compile(
    r"(NativeTensorDataset|NativeBatchSampler|NativeDataLoader|"
    r"data loader|dataloader|mini-?batching|shuffled training)"
    r"[^.;]{0,60}?\b(is|are|now)\s+"
    r"(supported|implemented|shipped|available|complete)\b"
    r"|\bsupports?\s+(native\s+)?(mini-?batching|data loading|data loaders?)\b",
    re.I)


def test_the_design_does_not_claim_a_phase_j_runtime_capability():
    """The document may describe what J1-J9 *will* do at any length; it may
    not say any of it exists. Spans carrying their own future or negative
    marker are the honest form and pass."""
    claim = _RUNTIME_CLAIM
    future = re.compile(
        r"\b(not|never|no|will|would|eventual\w*|planned|future|once|until|"
        r"when|yet|begins?|claim\w*|at J[1-9])\b", re.I)
    text = _flat(_design())
    offenders = [
        match.group(0)
        for match in claim.finditer(text)
        if not future.search(text[max(0, match.start() - 140):
                                  match.end() + 30])
    ]
    assert offenders == [], offenders


def test_the_runtime_claim_scanner_can_actually_fail():
    """Negative control for the scanner above: it must catch the sentences
    it exists to catch, and pass the ones the document has to be able to
    write."""
    claim = _RUNTIME_CLAIM
    for detected in (
        "NativeDataLoader is available",
        "the data loader is supported",
        "native mini-batching is now implemented",
        "deterministic shuffled training is implemented",
        "NativeTensorDataset is shipped",
        "TensorForge now supports native mini-batching",
    ):
        assert claim.search(detected), detected
    for accurate in (
        "NativeDataLoader is not implemented",
        "the loader will be available at J3",
        "no Phase-J runtime API is exported yet",
    ):
        match = claim.search(accurate)
        future = re.compile(r"\b(not|never|no|will|yet)\b", re.I)
        assert match is None or future.search(accurate), accurate


# ===========================================================================
# 3. The milestone ladder
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
    """Every way the J0 ladder can be wrong, as a list of reasons.

    Returned rather than raised so one call reports all of them, and so the
    negative controls can assert *which* fault was detected instead of
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

    # The landed rows are marked complete and no other row is: a row
    # claimed complete before it landed is the over-claim this parser
    # exists to catch, and a landed row left open is the under-claim.
    for name, tail in rows:
        complete = re.search(r"\*\*complete\*\*", tail, re.I) is not None
        unstarted = re.search(r"\*\*not started\*\*", tail, re.I) is not None
        if name in COMPLETE_MILESTONES:
            if not complete:
                problems.append(f"{name} is not marked complete")
            if unstarted:
                problems.append(f"{name} is marked not started")
        else:
            if complete:
                problems.append(f"{name} is marked complete before it landed")
            if not unstarted:
                problems.append(f"{name} is not marked not started")
    return problems


def test_the_milestone_ladder_runs_j0_to_j9_once_each_in_order():
    problems = _ladder_problems(_design())
    assert problems == [], problems


def test_exactly_the_landed_milestones_are_marked_complete():
    rows = dict(_ladder_rows(_design()))
    for name in COMPLETE_MILESTONES:
        assert re.search(r"\*\*complete\*\*", rows[name], re.I), rows[name]
    for name in UNSTARTED_MILESTONES:
        assert re.search(r"\*\*not started\*\*", rows[name], re.I), rows[name]
    # J7 is the next implementation milestone, and nothing beyond it may be
    # claimed under a J6 heading.
    assert COMPLETE_MILESTONES == ("J0", "J1", "J2", "J3", "J4", "J5", "J6")
    assert UNSTARTED_MILESTONES == ("J7", "J8", "J9")


def test_the_ladder_checker_can_actually_fail():
    """The negative control. Each mutation below must be *detected*, and
    the reason must name the fault — otherwise "no problems" would mean
    "the parser stopped matching" rather than "the ladder is right"."""
    text = _design()
    ladder = _ladder_text(text)
    assert _ladder_problems(text) == []

    def problems_for(mutated_ladder):
        return _ladder_problems(text.replace(ladder, mutated_ladder))

    # A missing milestone.
    dropped = ladder.replace("### J5 — ", "### JX5 — ", 1)
    assert any("missing" in reason for reason in problems_for(dropped)), dropped[:0]
    # A duplicated milestone.
    duplicated = ladder.replace("### J4 — ", "### J3 — ", 1)
    assert any("duplicated" in reason for reason in problems_for(duplicated))
    # An invented milestone one past the end of the ladder.
    invented = ladder + "\n### J10 — Something else — **not started**\n"
    assert any("unexpected" in reason for reason in problems_for(invented))
    # A row falsely claimed complete before it landed. Driven from the
    # first *unstarted* milestone rather than a hard-coded one, so landing
    # a milestone moves this control with the ladder instead of quietly
    # leaving it pointed at a row that has since shipped.
    next_up = UNSTARTED_MILESTONES[0]
    overclaimed = re.sub(rf"(### {next_up} — [^\n]*?)\*\*not started\*\*",
                         r"\1**complete**", ladder, count=1)
    assert overclaimed != ladder, f"the {next_up} row was not found"
    assert any("marked complete before it landed" in reason
               for reason in problems_for(overclaimed))
    # J0 left open.
    unopened = ladder.replace(
        "### J0 — Architecture and API contract — **complete**",
        "### J0 — Architecture and API contract — **not started**")
    assert any("J0 is not marked complete" in reason
               for reason in problems_for(unopened))
    # A landed row under-claimed. The mirror of the over-claim above, and
    # the one each status edit could have got wrong: a milestone that has
    # shipped must not still read "not started". Both landed runtime rows
    # are driven, so neither edit can silently stop being checked.
    for row in ("### J1 — Host-backed dataset foundation",
                "### J2 — Deterministic sampler",
                "### J3 — Native mini-batch loader",
                "### J4 — Loader state and mid-epoch resume",
                "### J5 — Native checkpoint metadata integration",
                "### J6 — Deterministic mini-batch training example"):
        understated = ladder.replace(f"{row} — **complete**",
                                     f"{row} — **not started**")
        assert understated != ladder, f"the {row!r} row was not found"
        milestone = row.split()[1]
        assert any(f"{milestone} is not marked complete" in reason
                   for reason in problems_for(understated))
    # Two rows swapped.
    swapped = ladder.replace("### J7 — ", "@@SEVEN@@ ", 1)
    swapped = swapped.replace("### J8 — ", "### J7 — ", 1)
    swapped = swapped.replace("@@SEVEN@@ ", "### J8 — ", 1)
    assert any("out of order" in reason for reason in problems_for(swapped))


def test_every_ladder_row_states_its_scope_and_exit_gate():
    """A row that names a milestone without saying what it delivers or how
    it is judged is a title, not a contract."""
    ladder = _ladder_text(_design())
    rows = re.split(r"^### J\d+ — ", ladder, flags=re.M)[1:]
    assert len(rows) == len(MILESTONES)
    for index, body in enumerate(rows):
        assert "**Scope:**" in body, f"J{index} states no scope"
        assert "**Exit gate:**" in body, f"J{index} states no exit gate"


def test_the_ladder_records_whether_it_was_adjusted_at_j0():
    """Phase H and Phase I both revised their ladders on evidence and
    recorded it. Phase J must state the outcome of that check either
    way rather than leaving it unasked."""
    adjustments = _design().split("### 23.1", 1)
    assert len(adjustments) == 2, "the design has no ladder-adjustment record"
    body = _flat(adjustments[1].split("\n## ", 1)[0])
    assert re.search(r"\bNone\b", body), body[:300]
    assert _requires(body, "verified at J0") == [], body[:300]


# ===========================================================================
# 4. The load-bearing design decisions, each in the section that owns it
# ===========================================================================

@pytest.mark.parametrize("section, terms", (
    # §3 — the public API surface.
    (3, ("NativeTensorDataset", "NativeBatchSampler", "NativeDataLoader",
         "tensorforge.experimental", "private")),
    # §4 — the dataset input contract.
    (4, ("numpy.ndarray", "float64", "float32", "int64", "rejected",
         "byte order", "read-only", "subclass")),
    # §5 — ownership and snapshot semantics.
    (5, ("snapshot", "copy", "aliasing", "close", "contiguous")),
    # §6 — dataset identity.
    (6, ("fingerprint", "SHA-256", "little-endian", "samples",
         "feature_dtype", "collision")),
    # §7 — the sampler.
    (7, ("batch_size", "drop_last", "cursor", "epoch", "pure function")),
    # §8 — the deterministic permutation.
    (8, ("tensorforge.splitmix64", "Fisher", "rejection", "seed", "epoch",
         "no new RNG algorithm")),
    # §9 — the iterator state machine and the batch-delivery transaction.
    (9, ("iterator", "supersed", "StopIteration", "commit", "claim",
         "publish", "rollback", "_deliver_batch", "exact-match")),
    # §10 — batch materialization and ownership.
    (10, ("NativeTensor", "int64", "read-only", "caller", "close")),
    # §11 — the state schemas.
    (11, ("format_version", "tensorforge.native_sampler",
          "tensorforge.native_data_loader", "JSON")),
    # §12 — validation and error ordering.
    (12, ("TypeError", "ValueError", "RuntimeError", "cannot fail")),
    # §13 — checkpoint metadata.
    (13, ("metadata", "version 3", "(1, 2, 3)", "atomic")),
    # §14 — the exact-resume contract.
    (14, ("fresh", "bit patterns", "no tolerance", "negative control")),
    # §15 — lifecycle.
    (15, ("close", "idempotent", "fallback", "rollback",
          "never closed by loader shutdown", "never refused")),
    # §16 — concurrency.
    (16, ("not thread-safe", "no lock", "undefined", "external locking")),
    # §17 — rollback.
    (17, ("unchanged", "retry", "MemoryError", "same batch",
          "consumes a logical batch")),
    # §18 — stable/native isolation.
    (18, ("tensorforge.data", "no implicit", "stable_framework_integration")),
    # §19 — dtype boundaries.
    (19, ("float64", "float32", "never", "int64", "device")),
))
def test_the_design_resolves_its_load_bearing_decisions(section, terms):
    """Section-scoped and combination-based: a document that contains one
    of these words somewhere passes nothing here."""
    missing = _requires(_section(_design(), section), *terms)
    assert not missing, f"§{section} does not resolve: {missing}"


def test_the_section_scoping_can_actually_fail():
    """Negative control for ``_section``/``_requires``: a term that belongs
    to one section must not be findable in another, and a missing term must
    be reported."""
    design = _design()
    # A real term of §8, absent from §18.
    assert _requires(_section(design, 8), "Fisher") == []
    assert _requires(_section(design, 18), "Fisher") == ["Fisher"]
    # An invented term is missing from every section.
    assert _requires(_section(design, 3), "quaternion") == ["quaternion"]
    # And an unknown section is an error rather than an empty pass.
    with pytest.raises(AssertionError):
        _section(design, 97)


def test_the_design_chooses_exactly_three_eventual_public_names():
    """The public surface is a contract. The design must name the three
    classes and say which milestone adds each, so J1-J3 cannot quietly add
    a fourth."""
    body = _section(_design(), 3)
    for name, milestone in PLANNED_CLASSES.items():
        assert name in body, name
        row = re.search(rf"\|\s*`{name}`\s*\|\s*(J\d+)\s*\|", body)
        assert row, f"{name} has no milestone row"
        assert row.group(1) == milestone, (name, row.group(1), milestone)
    assert _requires(body, "Three names, and no more") == []


def test_the_design_keeps_the_permutation_helpers_private():
    body = _section(_design(), 3)
    assert "_native_permutation" in body
    assert "_NativeBatchIterator" in body
    assert _requires(body, "never exported") == []


# ===========================================================================
# 4A. The batch-delivery transaction
# ===========================================================================
#
# This is the guarantee the whole phase's exact-resume claim rests on: the
# committed sampler position advances **if and only if** the caller received
# the batch. An earlier draft of this contract accepted an asynchronous
# window in which a commit could outlive a failed delivery, permanently
# skipping a batch. That was a material defect against TensorForge's
# exact-resume and transactional-state contracts, and it was corrected at J0
# rather than deferred to J3 or J7.
#
# Every assertion below is scoped to the section that owns the decision, and
# the negative control at the end proves the scan fails if the document is
# changed back.

# The forbidden shapes: any sentence conceding that a batch may be lost,
# skipped, or permanently consumed without delivery. Assembled as one
# pattern so the design scan and the negative control cannot drift apart.
_LOST_BATCH_CONCESSION = re.compile(
    r"(batch|position|cursor)[^.;]{0,80}?"
    r"\b(is|are|may be|can be|could be|will be|might be)\b[^.;]{0,40}?"
    r"\b(lost|skipped|dropped|permanently consumed|not re-delivered|"
    r"consumed and not used)\b"
    r"|\b(accepted|honest|unavoidable|irreducible|residual)\s+"
    r"(gap|window)\b"
    r"|\bbatch (consumed and not used|lost|skipped|permanently consumed)\b"
    r"|\basynchronous exception[^.;]{0,80}?"
    r"\b(lose|loses|skip|skips|consume|consumes)\b"
    r"|the committed position is retained after (a )?failed deliver",
    re.I)


def test_the_design_specifies_the_batch_handoff_as_a_transaction():
    """The five phases must all be named in the section that owns them, so
    J3 inherits a transaction rather than a sequence of steps."""
    body = _section(_design(), 9)
    for phase in ("Claim", "Construct", "Publish", "Commit and deliver",
                  "Failed delivery", "Successful delivery"):
        assert re.search(rf"Phase \d+ — {re.escape(phase)}", body), phase
    # ...and it is presented as a transaction, on the stated precedent.
    assert _requires(body, "transaction", "NativeGenerator") == []


def test_the_design_states_that_a_failed_delivery_changes_no_position():
    """The load-bearing sentence. A failed delivery must leave epoch and
    cursor exactly as they were."""
    body = _flat(_section(_design(), 9))
    assert re.search(
        r"No committed sampler position ever advances for a batch the "
        r"caller did not receive", body, re.I), body[:400]
    assert re.search(r"restore.{0,60}pre-delivery", body, re.I)
    assert re.search(r"epoch and cursor are \*{0,2}exactly\*{0,2} their "
                     r"pre-delivery values", _flat(_section(_design(), 9)),
                     re.I) or "exactly their pre-delivery values" in body
    # The same guarantee restated where the rollback table lives.
    rollback = _flat(_section(_design(), 17))
    assert "Every row leaves the cursor and epoch unchanged" in rollback


def test_the_design_states_that_no_batch_is_consumed_before_delivery():
    nine = _flat(_section(_design(), 9))
    assert re.search(r"no logical batch position was consumed", nine, re.I)
    assert re.search(r"Only a successfully delivered batch is ever consumed",
                     nine, re.I)
    seventeen = _flat(_section(_design(), 17))
    assert re.search(r"consumed exactly once", seventeen, re.I)


def test_the_design_states_that_the_undelivered_batch_is_closed():
    """Ownership of an undelivered batch must be released by the rollback,
    not left to garbage collection."""
    nine = _flat(_section(_design(), 9))
    assert re.search(r"[Cc]lose the undelivered feature .?NativeTensor",
                     nine)
    assert re.search(r"release the host target", nine, re.I)
    # ...and §15 must put that ownership on the iterator and keep delivered
    # batches out of every close path.
    fifteen = _flat(_section(_design(), 15))
    assert re.search(r"undelivered pending batches are iterator-owned",
                     fifteen, re.I)
    assert re.search(r"never closed by loader shutdown", fifteen, re.I)


def test_the_design_states_that_retry_yields_the_same_batch():
    nine = _flat(_section(_design(), 9))
    assert re.search(
        r"retry is valid and returns the exact same batch indices and the "
        r"exact same values", nine, re.I), nine[:400]
    seventeen = _flat(_section(_design(), 17))
    assert "retry, same batch" in seventeen.lower()


def test_the_design_names_a_private_delivery_seam_to_be_tested_later():
    """The seam has to exist, be private, be named, and be identified as a
    test seam rather than a public hook."""
    body = _section(_design(), 9)
    assert "_deliver_batch" in body
    flat = _flat(body)
    assert re.search(r"private", flat, re.I)
    assert re.search(r"tested deliberately|can be tested", flat, re.I)
    # Not a hook, and not a public callback.
    assert re.search(r"no public callback", flat, re.I)
    # It is listed among the names that must never be exported.
    assert "_deliver_batch" in _section(_design(), 3)
    # ...and the testing strategy says how it will be exercised.
    testing = _flat(_section(_design(), 21))
    assert "_deliver_batch" in testing
    assert re.search(r"monkeypatch", testing, re.I)
    assert re.search(r"non-vacuity control", testing, re.I)


def test_the_design_forbids_a_snapshot_observing_an_undelivered_position():
    nine = _flat(_section(_design(), 9))
    assert re.search(
        r"state snapshot must never be able to observe a "
        r"skipped-but-undelivered position", nine, re.I), nine[:400]
    # The mechanism: refusal, not a best-effort answer.
    assert re.search(r"refused rather than", nine, re.I)
    # And the checkpoint section carries the consequence.
    thirteen = _flat(_section(_design(), 13))
    assert re.search(
        r"checkpoint metadata cannot capture a skipped-but-undelivered "
        r"position", thirteen, re.I), thirteen[-800:]


def test_the_design_states_the_checkpoint_consequences_of_the_transaction():
    body = _flat(_section(_design(), 13))
    assert re.search(r"state_dict\(\) always describes the exact next batch",
                     body, re.I)
    assert re.search(r"failed delivery followed by a checkpoint resumes from "
                     r"the same candidate batch", body, re.I)
    assert re.search(r"successful delivery followed by a checkpoint resumes "
                     r"from the\s+following batch", body, re.I)
    # Cross-object atomicity is still explicitly disclaimed.
    assert re.search(r"no cross-object atomicity", body, re.I)


def test_the_design_distinguishes_the_four_abandonment_positions():
    """Abandoning before a request, failing during construction, failing
    during delivery, and abandoning after a successful delivery are four
    different events with four different outcomes."""
    body = _flat(_section(_design(), 9))
    for position in (
        r"Iterator abandoned \*{0,2}before\*{0,2} requesting a batch",
        r"Iterator abandoned \*{0,2}during\*{0,2} construction or delivery",
        r"Iterator abandoned \*{0,2}after\*{0,2} a successful delivery",
    ):
        assert re.search(position, body, re.I), position
    assert re.search(r"Failure in Phase 2 \(construct\)", body, re.I)
    assert re.search(r"Failure in Phase 4", body, re.I)


def test_the_design_resolves_every_operation_during_a_transaction():
    """§9.5 must give a definite answer for each named operation, so J3 has
    no discretion left."""
    body = _flat(_section(_design(), 9))
    for operation in ("state_dict", "load_state_dict", "close",
                      "iter(loader)", "__next__", "dataset.close"):
        assert operation in body, operation
    # Refusal is the documented answer for the ambiguous ones...
    assert body.count("RuntimeError") >= 4
    # ...and close is explicitly never refused, because it is the recovery.
    fifteen = _flat(_section(_design(), 15))
    assert re.search(r"close\(\) is never refused", fifteen, re.I)


def test_the_design_keeps_reentrancy_and_concurrency_distinct():
    body = _flat(_section(_design(), 16))
    assert re.search(r"claim guards reentrancy, not concurrency", body, re.I)
    assert re.search(r"exact-match", body, re.I)
    assert re.search(r"external locking is required", body, re.I)


def test_the_design_concedes_no_lost_or_skipped_batch():
    """The defect this section exists for. No sentence anywhere in the
    contract may concede that a batch can be lost, skipped, or permanently
    consumed without delivery."""
    offenders = [match.group(0)
                 for match in _LOST_BATCH_CONCESSION.finditer(_flat(_design()))]
    assert offenders == [], offenders


def test_the_lost_batch_scanner_can_actually_fail():
    """The negative control this correction requires. Mutating the design
    back to conceding a lost batch must be **detected** — otherwise "no
    offenders" would mean the pattern stopped matching rather than that the
    contract is sound."""
    # Each of these is a form the earlier, defective draft actually used or
    # could have used. Every one must be caught.
    for conceded in (
        "the batch is not re-delivered",
        "the committed position is retained after a failed delivery",
        "the asynchronous window is an accepted gap",
        "so the worst case is one batch consumed and not used",
        "a batch may be lost in that window",
        "the position can be skipped if delivery fails",
        "an asynchronous exception can consume a batch the caller never saw",
        "this is an irreducible window",
    ):
        assert _LOST_BATCH_CONCESSION.search(conceded), conceded

    # ...and the sentences the corrected contract must be able to write are
    # not caught, or the document could not state its own guarantee.
    for accurate in (
        "no committed sampler position ever advances for a batch the caller "
        "did not receive",
        "a failed delivery consumes no logical batch position",
        "the rollback restores the exact pre-delivery epoch and cursor",
        "a state snapshot must never be able to observe a "
        "skipped-but-undelivered position",
    ):
        assert _LOST_BATCH_CONCESSION.search(accurate) is None, accurate

    # And the mutation test proper: splice a concession into a copy of the
    # real document and prove the whole-document scan flags it. Operates on
    # a string only — no repository file is written.
    mutated = _flat(_design()).replace(
        "Only a successfully delivered batch is ever consumed.",
        "The batch is not re-delivered.", 1)
    assert _LOST_BATCH_CONCESSION.search(mutated), (
        "splicing a concession into the design did not trip the scanner"
    )


# ===========================================================================
# 5. The deterministic derivation
# ===========================================================================

def test_the_design_reuses_the_locked_splitmix64_derivation():
    """Phase J must not introduce a second RNG algorithm. It reuses the
    Phase-G finalizer and constants, which are asserted here against the
    C++ header rather than against the design's own prose."""
    header = _read("cpp/include/tf_random_internal.h")
    for constant in (GOLDEN,) + SPLITMIX_MULTIPLIERS:
        spelled = f"0x{constant:016X}"
        assert spelled in header, f"{spelled} is not the C++ constant"
        assert spelled in _design(), f"the design does not pin {spelled}"
    body = _section(_design(), 8)
    assert _requires(body, "tensorforge.splitmix64", "no new RNG algorithm",
                     "key schedule") == []


def test_the_design_separates_the_sampler_stream_from_the_dropout_stream():
    body = _section(_design(), 8)
    assert f"0x{SAMPLER_DOMAIN:016X}" in body, "no domain constant is pinned"
    # ...and it is honest about what the separation is not.
    assert _requires(body, "not a cryptographic separation") == []


def test_the_design_deliberately_does_not_couple_the_sampler_to_a_generator():
    """Premise, from the live class: NativeGenerator really does expose no
    bit derivation, so coupling would require inventing one."""
    from tensorforge.experimental import NativeGenerator

    generator = NativeGenerator(1)
    for numerical in ("random", "rand", "randn", "bits", "next", "uniform",
                      "bernoulli", "mask", "permutation", "shuffle"):
        assert not hasattr(generator, numerical), numerical
    body = _section(_design(), 8)
    assert _requires(body, "not coupled", "NativeGenerator") == [] or \
        _requires(body, "deliberately not coupled") == []
    assert "does not hold one" in _flat(body)


def test_the_design_specifies_an_unbiased_bounded_integer_rule():
    body = _section(_design(), 8)
    assert _requires(body, "modulo", "bias", "reject") == []
    assert "limit = (1 << 64) - ((1 << 64) % bound)" in body, (
        "the bounded-integer rule is not spelled out"
    )


def test_the_design_fixes_the_fisher_yates_direction():
    body = _section(_design(), 8)
    assert re.search(r"Fisher[-–—]Yates,?\s*\*{0,2}DOWNWARD|downward",
                     body, re.I), "the sweep direction is not fixed"
    assert "range(length - 1, 0, -1)" in body


def test_the_design_carries_pseudocode_that_could_be_implemented_directly():
    """Not a wording check: the pseudocode block must parse as Python, so
    a later milestone cannot inherit something merely suggestive."""
    body = _section(_design(), 8)
    blocks = re.findall(r"```\n(.*?)```", body, re.S)
    assert blocks, "§8 carries no pseudocode block"
    implementable = [block for block in blocks if "def permutation" in block]
    assert implementable, "§8 has no permutation pseudocode"
    for block in implementable:
        ast.parse(block)          # raises SyntaxError if it is not real code


# ===========================================================================
# 6. The reference vectors
# ===========================================================================
#
# The vectors are the specification. They are parsed out of the document
# and checked for coverage and internal consistency — not recomputed, since
# no runtime implementation exists at J0 to recompute them with.

_PERMUTATION_ROW = re.compile(
    r"^\|\s*(\d+)\s*\|\s*([^|]+?)\s*\|\s*(\d+)\s*\|\s*`\[([0-9, ]*)\]`\s*\|$",
    re.M)


def _permutation_vectors(text):
    """``[(length, seed_text, epoch, [indices])]`` from §8.9's table."""
    body = _section(text, 8)
    rows = []
    for length, seed, epoch, indices in _PERMUTATION_ROW.findall(body):
        rows.append((int(length), seed.strip(), int(epoch),
                     [int(value) for value in indices.split(",") if value.strip()]))
    return rows


def test_the_reference_vectors_cover_the_required_combinations():
    rows = _permutation_vectors(_design())
    assert rows, "§8.9 carries no permutation vectors"
    lengths = {length for length, _, _, _ in rows}
    epochs = {epoch for _, _, epoch, _ in rows}
    seeds = {seed for _, seed, _, _ in rows}
    # Length 1, length 2, a small odd length, and a small even length.
    for required in (1, 2, 5, 8):
        assert required in lengths, f"no vector at length {required}"
    # Epoch 0 and a later epoch.
    assert 0 in epochs and any(epoch > 0 for epoch in epochs), epochs
    # Seed 0 and a nontrivial large seed near the accepted upper bound.
    assert any(seed.strip("`") == "0" for seed in seeds), seeds
    assert any("0xFEDCBA9876543210" in seed for seed in seeds), seeds
    assert any("0xFFFFFFFFFFFFFFFF" in seed for seed in seeds), seeds


def test_every_reference_vector_is_a_real_permutation():
    """Internal consistency: each listed sequence must be a permutation of
    ``range(length)``. This catches a transcription error in the document
    without needing an implementation."""
    for length, seed, epoch, indices in _permutation_vectors(_design()):
        assert sorted(indices) == list(range(length)), (length, seed, epoch,
                                                        indices)


def test_the_vector_parser_can_actually_fail():
    """Negative control: the parser must read real rows, and must reject a
    row whose sequence is not a permutation."""
    rows = _permutation_vectors(_design())
    assert len(rows) >= 20, len(rows)
    broken = "| 5 | `0` | 0 | `[1, 1, 3, 4, 2]` |"
    parsed = _PERMUTATION_ROW.findall(broken)
    assert parsed, "the row parser does not read a well-formed row"
    indices = [int(value) for value in parsed[0][3].split(",")]
    assert sorted(indices) != list(range(5)), "the control row is not broken"


def test_the_design_states_the_empty_dataset_case_rather_than_a_vector():
    """Length 0 has no vector *because* an empty dataset is rejected. The
    document must say so, in the section that owns the input contract and
    in the vector table."""
    assert _requires(_section(_design(), 4), "Empty datasets are rejected",
                     "shape[0] == 0") == []
    assert "no vector" in _flat(_section(_design(), 8)).lower()


def test_the_design_commits_sequential_and_batch_plan_vectors():
    body = _section(_design(), 8)
    assert "Sequential order" in body
    assert "Complete batch plans" in body
    for plan in ("`[[7, 5, 4], [0, 1, 3], [6, 2]]`",
                 "`[[1, 0], [3, 4], [2]]`",
                 "`[[0, 1], [2, 3], [4]]`"):
        assert plan in body, plan


# ===========================================================================
# 7. State schemas and the checkpoint workflow
# ===========================================================================

def test_the_state_schemas_are_json_compatible_by_construction():
    """Every field the design defines must pass the checkpoint's own
    metadata validator. Checked against the real validator on a literal
    built from the documented schema, so this tracks the runtime rather
    than the prose."""
    from tensorforge.experimental import native_checkpoint

    documented = {
        "format": "tensorforge.native_data_loader",
        "format_version": 1,
        "sampler": {
            "format": "tensorforge.native_sampler",
            "format_version": 1,
            "dataset": {
                "samples": 800,
                "feature_shape": [1, 6, 6],
                "feature_dtype": "float32",
                "fingerprint": "0" * 64,
            },
            "seed": 2**64 - 1,
            "shuffle": True,
            "batch_size": 16,
            "drop_last": False,
            "epoch": 3,
            "cursor": 27,
        },
    }
    normalized = native_checkpoint._validated_metadata(
        {"training": {"next_step": 12, "data_loader": documented}},
        "metadata", frozenset())
    # Round-trips unchanged: no coercion, and booleans stay booleans.
    assert normalized == {"training": {"next_step": 12,
                                       "data_loader": documented}}
    inner = normalized["training"]["data_loader"]["sampler"]
    assert inner["shuffle"] is True and inner["drop_last"] is False
    assert inner["seed"] == 2**64 - 1


def test_the_design_states_both_schema_format_tags_and_versions():
    body = _section(_design(), 11)
    assert '"tensorforge.native_sampler"' in body
    assert '"tensorforge.native_data_loader"' in body
    assert _requires(body, "format_version", "exactly 1") == []
    # The permutation is derived, not serialized.
    assert _requires(body, "permutation is derivable",
                     "so it is not serialized") == []


def test_the_design_forbids_payload_in_state():
    body = _flat(_section(_design(), 11)).lower()
    for forbidden in ("nativetensor", "numpy arrays", "bytes",
                      "dataset content"):
        assert forbidden in body, forbidden


def test_the_checkpoint_workflow_is_caller_managed_and_not_claimed_atomic():
    body = _section(_design(), 13)
    assert _requires(
        body,
        "caller-supplied metadata",
        "does not interpret",
        "There is no cross-object atomicity",
        "caller's responsibility",
    ) == []
    # ...and it explicitly adds nothing to the checkpoint.
    assert _requires(body, "No automatic loader discovery",
                     "No new checkpoint root field", "No version 4") == []


def test_the_design_keeps_the_checkpoint_capture_guardrail_true():
    """The long-standing promise is that the archive captures no data-loader
    position. Phase J must not weaken it; it must say the caller carries
    the position through metadata instead."""
    body = _flat(_section(_design(), 13))
    assert re.search(
        r"the native checkpoint captures no data-loader position", body, re.I
    ), body[-1200:]


# ===========================================================================
# 8. The unchanged runtime — asserted against reality, never against prose
# ===========================================================================

def test_the_capability_registries_did_not_move():
    assert cpp.SUPPORTED_DTYPES == J0_DTYPES
    assert cpp.SUPPORTED_DEVICES == J0_DEVICES
    assert cpp.UNSUPPORTED == J0_UNSUPPORTED
    assert cpp.RAW_KERNEL_DTYPES == J0_RAW_KERNEL_DTYPES


def test_float64_is_still_the_default_and_no_dtype_is_inferred():
    assert cpp.normalize_dtype(None) == J0_DEFAULT_DTYPE
    info = cpp.backend_info()
    assert info["dtype"] == J0_DEFAULT_DTYPE
    assert info["supported_dtypes"] == J0_DTYPES
    assert info["raw_kernel_dtypes"] == J0_RAW_KERNEL_DTYPES
    assert info["stable_framework_integration"] is False


@pytest.mark.skipif(not cpp.is_available(),
                    reason="the native library is not built")
def test_an_omitted_dtype_still_means_float64_even_for_a_float32_input():
    """The rule Phase J's dataset contract inherits verbatim (§19.3),
    asserted against the live constructor rather than the document."""
    import numpy as np

    from tensorforge.experimental import NativeTensor

    values = np.arange(6, dtype=np.float32).reshape(2, 3)
    tensor = NativeTensor.from_array(values)
    try:
        assert tensor.dtype == "float64"
        assert tensor.to_numpy().dtype == np.float64
    finally:
        tensor.close()


def test_the_checkpoint_and_optimizer_state_versions_did_not_move():
    from tensorforge.experimental import (
        native_checkpoint, native_optimizer_state,
    )

    assert native_checkpoint._FORMAT == J0_CHECKPOINT_FORMAT
    assert native_checkpoint._FORMAT_VERSION == J0_CHECKPOINT_VERSION
    assert (native_checkpoint._SUPPORTED_FORMAT_VERSIONS
            == J0_CHECKPOINT_VERSIONS)
    assert native_optimizer_state.FORMAT_VERSION == J0_OPTIMIZER_STATE_VERSION


def _source_exports():
    names = set()
    for source in sorted((REPO_ROOT / "cpp" / "src").glob("*.cpp")):
        text = source.read_text(encoding="utf-8")
        names.update(re.findall(r"TF_EXPORT[^;{]*?\b(tf_[a-z0-9_]+)\s*\(",
                                text, re.S))
    return names


def test_the_source_still_exports_exactly_fifty_four_symbols():
    names = _source_exports()
    assert len(names) == J0_EXPORT_COUNT, sorted(names)
    # And no data-pipeline symbol appeared: the phase plans none.
    forbidden = re.compile(
        r"^tf_(dataset|sampler|loader|batch|shuffle|permut|gather)", re.I)
    for name in sorted(names):
        assert not forbidden.search(name), name


@pytest.mark.skipif(not cpp.is_available(),
                    reason="the native library is not built")
def test_the_built_library_exports_exactly_what_the_source_declares():
    library = cpp._require_library()
    for name in sorted(_source_exports()):
        assert hasattr(library, name), name


def test_the_ctest_inventory_is_still_exactly_twenty_four():
    cmake = _read("cpp/CMakeLists.txt")
    names = re.findall(r"add_test\s*\(\s*NAME\s+(\w+)", cmake)
    assert len(names) == J0_CTEST_COUNT, names
    assert len(set(names)) == len(names), "a CTest name is registered twice"
    sources = {path.stem for path in
               sorted((REPO_ROOT / "cpp" / "tests").glob("test_*.cpp"))}
    assert sources == {f"test_{name}" for name in names}


def test_the_example_and_benchmark_inventories_moved_by_exactly_j6s_example():
    """Phase J started at 15 examples and 8 benchmarks. **J6** added one
    example and nothing else; J8's benchmark has not started.

    Driven from ``PHASE_J_EXAMPLES`` rather than from a bumped literal, so
    each artifact stays attributed to the milestone that shipped it and an
    unannounced one fails the exact equality."""
    examples = [path.name for path in (REPO_ROOT / "examples").glob("*.py")
                if path.name != "__init__.py"]
    benchmarks = [path.name for path in (REPO_ROOT / "benchmarks").glob("*.py")
                  if path.name != "__init__.py"]
    assert len(examples) == CURRENT_EXAMPLE_COUNT == 16, sorted(examples)
    assert len(benchmarks) == CURRENT_BENCHMARK_COUNT == 8, sorted(benchmarks)
    assert set(PHASE_J_EXAMPLES) <= set(examples), sorted(PHASE_J_EXAMPLES)
    assert PHASE_J_BENCHMARKS == {}
    # Every *other* artifact name is still free of Phase-J vocabulary, so a
    # second example or J8's benchmark cannot arrive unnoticed.
    for name in examples + benchmarks:
        if name in PHASE_J_EXAMPLES:
            continue
        assert "data_pipeline" not in name and "minibatch" not in name, name


def test_the_j6_example_and_its_proof_exist_and_nothing_later_does():
    """J6's two files moved from absence to presence, and only those two.
    J7's hardening module, J8's benchmark and its test, and J9's closure
    module are all still absent."""
    assert (REPO_ROOT / "examples" / "native_minibatch_training.py").is_file()
    assert (REPO_ROOT / "tests"
            / "test_native_minibatch_training.py").is_file()
    for later in ("tests/test_native_data_hardening.py",
                  "tests/test_native_data_benchmark.py",
                  "tests/test_native_phase_j_closure.py",
                  "benchmarks/benchmark_native_data_pipeline.py"):
        assert not (REPO_ROOT / later).exists(), later
    # The example is an example: it adds no public name and no production
    # module, and its model class stays an implementation detail of it.
    import tensorforge.experimental as experimental

    assert len(experimental.__all__) == 25
    assert "NativeMiniBatchClassifier" not in experimental.__all__
    for path in (REPO_ROOT / "src").rglob("*.py"):
        assert "NativeMiniBatchClassifier" not in path.read_text(
            encoding="utf-8"), path.name


def test_phase_i_is_still_complete_and_is_the_latest_completed_phase():
    assert (REPO_ROOT / "docs" / "native_dtype_float32_design.md").is_file()
    assert (REPO_ROOT / "tests" / "test_native_phase_i_closure.py").is_file()
    head = _flat(_design()[:6000])
    assert re.search(r"Phase I remains complete \(I0[-–—]I11\)", head), head[:600]
    assert re.search(r"latest \*{0,2}completed\*{0,2} phase", head), head[:600]


# ===========================================================================
# 9. Presence and absence — exactly J1's runtime exists, and no placeholder
# ===========================================================================

def test_exactly_the_landed_phase_j_classes_exist():
    import tensorforge.experimental as experimental

    for name in LANDED_CLASSES:
        assert hasattr(experimental, name), (
            f"{name} landed at {PLANNED_CLASSES[name]} but is not exported"
        )
        assert name in experimental.__all__, name
    for name in UNLANDED_CLASSES:
        assert not hasattr(experimental, name), (
            f"{name} exists, but its milestone has not landed"
        )
        assert name not in experimental.__all__, name
    # All three of §3.1's names have now landed, so the absence half of
    # this split is carried by the *closed* surface instead: **three
    # names, and no more**. A fourth Phase-J public name is the over-claim
    # this check now guards against, and J4-J9 add none.
    assert LANDED_CLASSES == set(PLANNED_CLASSES)
    assert UNLANDED_CLASSES == set()
    for invented in ("NativeDataset", "NativeSampler", "NativeLoader",
                     "NativeBatchLoader", "NativeDataIterator",
                     "NativeCollate", "NativeWorker", "NativePrefetcher",
                     "native_batches", "NativeBatchIterator"):
        assert not hasattr(experimental, invented), invented
        assert invented not in experimental.__all__, invented
    # The submodules themselves are ordinary package attributes and are
    # **not** exports: reaching one requires naming the private module.
    for module in ("native_dataset", "native_sampler", "native_data_loader",
                   "_native_permutation"):
        assert module not in experimental.__all__, module


def test_only_the_landed_phase_j_modules_exist_under_src():
    package = REPO_ROOT / "src" / "tensorforge" / "experimental"
    for name in LANDED_MODULES:
        assert (package / name).is_file(), f"{name} is missing"
    for name in UNLANDED_MODULES:
        assert not (package / name).exists(), f"{name} exists before its milestone"
    # J2's derivation helper exists but stays permanently private: a public
    # bit-generation API would be a second RNG surface beside
    # NativeGenerator, which §20 forbids.
    import tensorforge.experimental as private_check

    assert "_native_permutation" not in private_check.__all__
    for helper in ("splitmix64_mix", "epoch_key", "draw_bits", "bounded",
                   "permutation", "sample_order", "batch_plan"):
        assert helper not in private_check.__all__, helper
    # ...and no module anywhere under src/ defines an unlanded class, which
    # would mean a stub landed outside the expected filenames.
    definitions = {name: [] for name in PLANNED_CLASSES}
    for path in (REPO_ROOT / "src").rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        for name in PLANNED_CLASSES:
            if re.search(rf"^\s*class {name}\b", text, re.M):
                definitions[name].append(path.name)
    for name in UNLANDED_CLASSES:
        assert definitions[name] == [], (
            f"{definitions[name]} defines {name} before its milestone"
        )
    # Each landed class is defined exactly once, in its contracted module.
    assert definitions["NativeTensorDataset"] == ["native_dataset.py"], (
        definitions["NativeTensorDataset"]
    )
    assert definitions["NativeBatchSampler"] == ["native_sampler.py"], (
        definitions["NativeBatchSampler"]
    )
    assert definitions["NativeDataLoader"] == ["native_data_loader.py"], (
        definitions["NativeDataLoader"]
    )
    # J3's iterator and delivery seam exist, in exactly one module, and
    # stay permanently private: a caller receives an iterator from
    # ``iter(loader)`` and never constructs one, and the seam takes no
    # user-supplied callable and is no public hook.
    for private in PRIVATE_LOADER_NAMES:
        assert private not in private_check.__all__, private
        assert not hasattr(private_check, private), private
    owners = {"_NativeBatchIterator": [], "_deliver_batch": []}
    for path in sorted(package.glob("*.py")):
        text = path.read_text(encoding="utf-8")
        if re.search(r"^\s*class _NativeBatchIterator\b", text, re.M):
            owners["_NativeBatchIterator"].append(path.name)
        if re.search(r"^\s*def _deliver_batch\b", text, re.M):
            owners["_deliver_batch"].append(path.name)
    assert owners == {"_NativeBatchIterator": ["native_data_loader.py"],
                      "_deliver_batch": ["native_data_loader.py"]}, owners


# The public experimental surface as each milestone left it. Each entry is
# **the delta that milestone added**, so the running total stays an exact
# equality in both directions and no addition can be misattributed to an
# earlier milestone than the one that shipped it.
J0_EXPORT_INVENTORY = frozenset({
    "NativeTensor", "NativeGenerator", "NativeParameter",
    "NativeParameterRegistry", "NativeModule", "NativeLinear",
    "NativeReLU", "NativeFlatten", "NativeConv2d", "NativeMaxPool2d",
    "NativeSequential", "NativeLayerNorm", "NativeBatchNorm1d",
    "NativeBatchNorm2d", "NativeDropout", "NativeMSELoss",
    "NativeCrossEntropyLoss", "native_accuracy", "NativeSGD",
    "NativeAdam", "save_native_checkpoint", "load_native_checkpoint",
})
J1_ADDITION = frozenset({"NativeTensorDataset"})
J2_ADDITION = frozenset({"NativeBatchSampler"})
J3_ADDITION = frozenset({"NativeDataLoader"})


def test_each_landed_milestone_added_exactly_one_public_name():
    """The J1, J2, and J3 exit gates, over the live inventory: ``__all__``
    grew by **exactly one** name per milestone, and by that name.

    22 at J0, 23 at J1, 24 at J2, 25 at J3 — asserted as a running exact
    equality rather than as a subset, so an unannounced export fails here.
    """
    import tensorforge.experimental as experimental

    live = set(experimental.__all__)
    assert len(experimental.__all__) == len(live), "duplicate export"
    assert len(J0_EXPORT_INVENTORY) == 22
    assert len(J0_EXPORT_INVENTORY | J1_ADDITION) == 23
    assert len(J0_EXPORT_INVENTORY | J1_ADDITION | J2_ADDITION) == 24
    assert len(J0_EXPORT_INVENTORY | J1_ADDITION | J2_ADDITION
               | J3_ADDITION) == 25
    assert live == (J0_EXPORT_INVENTORY | J1_ADDITION | J2_ADDITION
                    | J3_ADDITION)
    assert live - J0_EXPORT_INVENTORY == (J1_ADDITION | J2_ADDITION
                                          | J3_ADDITION)
    assert len(live) == 25


def test_the_dataset_still_plans_orders_and_groups_nothing():
    """J2's planning surface belongs to the **sampler**, not to the
    dataset. A cursor, an epoch, a shuffle, or a state schema arriving
    inside ``NativeTensorDataset`` would be one milestone's work landing
    under another's name."""
    from tensorforge.experimental import NativeTensorDataset

    for name in ("state_dict", "load_state_dict", "epoch", "cursor",
                 "shuffle", "seed", "batch_size", "drop_last", "plan",
                 "epoch_permutation", "next_batch_indices",
                 "batches_per_epoch", "remaining", "sampler", "loader"):
        assert not hasattr(NativeTensorDataset, name), name
    source = (REPO_ROOT / "src" / "tensorforge" / "experimental"
              / "native_dataset.py").read_text(encoding="utf-8")
    code = "\n".join(
        line for line in source.splitlines()
        if not line.lstrip().startswith("#")
    )
    tree = ast.parse(source)
    defined = {node.name for node in ast.walk(tree)
               if isinstance(node, (ast.FunctionDef, ast.ClassDef))}
    for forbidden in ("permutation", "_shuffle", "_advance", "_next_epoch"):
        assert not any(forbidden in name for name in defined), forbidden
    assert "import random" not in code
    assert "np.random" not in code


def test_the_sampler_stays_a_planner_and_j5_onward_has_no_runtime():
    """The sampler is a **planner**: it owns a position, but it never
    iterates, never materializes, and exposes no public advance — only
    J3's loader may consume a batch position, and only through the private
    transaction primitives. J5-J9's vocabulary stays absent everywhere."""
    from tensorforge.experimental import NativeBatchSampler, NativeDataLoader

    for name in ("__iter__", "__next__", "advance", "step", "reset",
                 "next_epoch", "advance_epoch", "advance_cursor", "consume",
                 "deliver", "_deliver_batch", "close", "closed",
                 "feature_batch", "target_batch", "materialize", "collate",
                 "prefetch", "num_workers", "pin_memory", "loader"):
        assert not hasattr(NativeBatchSampler, name), name
    # J4 landed the loader's own **in-memory** state, so its two methods
    # and its private format tag are asserted *present* — and nothing
    # beside them arrived. J5 owns the checkpoint workflow, which has no
    # runtime at all; a stub that existed only to fail until then would be
    # the over-claim this module exists to prevent.
    for name in ("__len__", "__next__", "save", "load", "state",
                 "load_state", "restore", "num_workers", "collate_fn",
                 "prefetch", "pin_memory", "transform"):
        assert not hasattr(NativeDataLoader, name), name
    assert hasattr(NativeDataLoader, "state_dict")
    assert hasattr(NativeDataLoader, "load_state_dict")
    from tensorforge.experimental import native_data_loader

    assert native_data_loader._FORMAT == "tensorforge.native_data_loader"
    assert native_data_loader._FORMAT_VERSION == 1
    assert native_data_loader._SUPPORTED_FORMAT_VERSIONS == (1,)
    assert len(native_data_loader._STATE_FIELDS) == 3
    # The sampler's own schema and version did not move underneath it.
    from tensorforge.experimental import native_sampler

    assert native_sampler._FORMAT == "tensorforge.native_sampler"
    assert native_sampler._FORMAT_VERSION == 1
    package = REPO_ROOT / "src" / "tensorforge" / "experimental"
    # The checkpoint stays uncoupled in both directions.
    checkpoint = (package / "native_checkpoint.py").read_text(encoding="utf-8")
    for pipeline in ("native_dataset", "native_sampler", "native_data_loader",
                     "_native_permutation"):
        assert pipeline not in checkpoint, pipeline
    loader_source = (package / "native_data_loader.py").read_text(
        encoding="utf-8")
    assert "native_checkpoint" not in loader_source
    # The sampler consults no Python or NumPy random source, and no
    # generator: its order is a pure function of (seed, epoch, length).
    for module in ("native_sampler.py", "_native_permutation.py"):
        tree = ast.parse((package / module).read_text(encoding="utf-8"))
        named = {node.id for node in ast.walk(tree)
                 if isinstance(node, ast.Name)}
        named |= {node.attr for node in ast.walk(tree)
                  if isinstance(node, ast.Attribute)}
        for forbidden in ("random", "secrets", "NativeGenerator",
                          "_reserve_call", "time"):
            assert forbidden not in named, (module, forbidden)


def test_the_checkpoint_metadata_workflow_is_proved_and_still_uncoupled():
    """J5's landed half asserted as **presence**, and its defining negative
    asserted as **absence**.

    J5 shipped no production code, so what must be present is the proof
    module and the two existing APIs it composes — not a new name. What
    must stay absent is every form of coupling: a loader argument on
    either checkpoint entry point, a loader registry or traversal on the
    module system, and any production constant spelling a caller
    convention, which is what would turn a recommendation into a schema.
    """
    import inspect

    from tensorforge.experimental import (
        NativeDataLoader, NativeModule, load_native_checkpoint,
        save_native_checkpoint,
    )

    assert (REPO_ROOT / "tests" / "test_native_data_checkpoint.py").is_file()
    for half in (save_native_checkpoint, load_native_checkpoint,
                 NativeDataLoader.state_dict,
                 NativeDataLoader.load_state_dict):
        assert callable(half), half
    # Neither entry point grew a loader argument, and the loader grew no
    # checkpoint convenience.
    assert list(inspect.signature(save_native_checkpoint).parameters) == [
        "path", "model", "optimizer", "metadata"]
    assert list(inspect.signature(load_native_checkpoint).parameters) == [
        "path", "model", "optimizer"]
    for forbidden in ("save", "load", "checkpoint", "save_checkpoint",
                      "load_checkpoint", "to_metadata", "from_metadata"):
        assert not hasattr(NativeDataLoader, forbidden), forbidden
    # No discovery, no registry, no traversal.
    for forbidden in ("loaders", "named_loaders", "data_loaders",
                      "register_loader", "datasets", "named_datasets",
                      "samplers", "named_samplers"):
        assert not hasattr(NativeModule, forbidden), forbidden

    package = REPO_ROOT / "src" / "tensorforge" / "experimental"
    tree = ast.parse((package / "native_checkpoint.py").read_text(
        encoding="utf-8"))
    literals = {node.value for node in ast.walk(tree)
                if isinstance(node, ast.Constant)
                and isinstance(node.value, str)}
    # Negative control for the literal scanner: it must find the constants
    # the module really does define, or the absences below prove nothing.
    assert "tensorforge.native_checkpoint" in literals
    assert "manifest" in literals
    for convention in ("data_loader", "training", "next_step",
                       "tensorforge.native_data_loader",
                       "tensorforge.native_sampler"):
        assert convention not in literals, convention


def test_the_stable_public_api_did_not_gain_a_name():
    import tensorforge

    for name in PLANNED_CLASSES:
        assert not hasattr(tensorforge, name), name
    # The stable mini-batch iterator is untouched and stays stable-only.
    assert hasattr(tensorforge, "batches")
    assert "batches" in tensorforge.__all__


def test_importing_tensorforge_still_does_not_import_the_native_line():
    """The isolation Phase J must not break. Asserted structurally: the
    stable package's modules name no experimental module."""
    stable = REPO_ROOT / "src" / "tensorforge"
    offenders = []
    for path in sorted(stable.glob("*.py")):
        text = path.read_text(encoding="utf-8")
        if re.search(r"^\s*(from|import)\s+.*\bexperimental\b", text, re.M):
            offenders.append(path.name)
    assert offenders == [], offenders


def test_the_absence_checks_can_actually_fail():
    """Negative control: the class scanner must find a class when one is
    present, so "no offenders" is evidence rather than a dead regex."""
    planted = "class NativeDataLoader:\n    pass\n"
    for name in PLANNED_CLASSES:
        pattern = re.compile(rf"^\s*class {name}\b", re.M)
        assert bool(pattern.search(planted)) == (name == "NativeDataLoader")
    # ...and the same scanner really does find the classes that *have*
    # landed, so the presence half of the split is evidence too — each in
    # its own contracted module and in no other.
    package = REPO_ROOT / "src" / "tensorforge" / "experimental"
    dataset = (package / "native_dataset.py").read_text(encoding="utf-8")
    sampler = (package / "native_sampler.py").read_text(encoding="utf-8")
    assert re.search(r"^\s*class NativeTensorDataset\b", dataset, re.M)
    assert re.search(r"^\s*class NativeBatchSampler\b", sampler, re.M)
    assert not re.search(r"^\s*class NativeBatchSampler\b", dataset, re.M)
    assert not re.search(r"^\s*class NativeTensorDataset\b", sampler, re.M)
    assert not re.search(r"^\s*class NativeDataLoader\b", sampler, re.M)
    import tensorforge.experimental as experimental
    # ...and the export check really is reading a populated inventory.
    assert len(experimental.__all__) >= 20, experimental.__all__
    assert "NativeTensor" in experimental.__all__


# ===========================================================================
# 10. Non-goals, and the boundaries Phase J may not erode
# ===========================================================================

@pytest.mark.parametrize("non_goal", (
    "native integer tensors", "embeddings", "sparse tensors", "CUDA", "AMP",
    "float16", "bfloat16", "dtype casting", "dtype promotion",
    "device movement", "map_location", "multiprocessing workers",
    "prefetch threads", "asynchronous iteration", "pinned memory",
    "distributed sampling", "network datasets", "streaming",
    "infinite datasets", "memory mapping", "checkpoint version 4",
    "optimizer-state version 2", "another RNG algorithm",
    "implicit stable/native conversion", "timing assertions",
    "performance gates", "external dependencies",
))
def test_the_design_lists_every_required_non_goal(non_goal):
    assert non_goal.lower() in _flat(_section(_design(), 20)).lower(), non_goal


def test_the_design_plans_no_new_export_and_says_what_would_change_that():
    body = _section(_design(), 22)
    assert _requires(body, "54", "no new dependency", "No C++") == []
    # The only reopening condition is stated, and it is not J8's to take.
    assert _requires(body, "separately", "approved",
                     "characterization only", "measured") == []


def test_no_status_surface_claims_a_phase_j_capability():
    """The prose half of the boundary, over the surfaces that must agree.
    A span carrying its own future or negative marker is the honest form
    and passes."""
    claim = _RUNTIME_CLAIM
    future = re.compile(
        r"\b(not|never|no|will|would|planned|future|once|until|when|yet|"
        r"begins?|eventual\w*|approved|design\w*|claim\w*)\b", re.I)
    surfaces = ("README.md", "CLAUDE.md", "docs/roadmap.md",
                "docs/project_summary.md", "docs/native_support_matrix.md",
                "docs/architecture.md", "docs/backend_experiments.md",
                "src/tensorforge/experimental/__init__.py")
    for surface in surfaces:
        text = _flat(_read(surface))
        offenders = [
            match.group(0) for match in claim.finditer(text)
            if not future.search(text[max(0, match.start() - 90):
                                      match.end() + 30])
        ]
        assert offenders == [], (surface, offenders[:3])


def test_no_status_surface_calls_phase_j_complete():
    complete = re.compile(
        r"Phase.J\b[^.;]{0,70}?\b(is|are|was|has been)\s+"
        r"(complete|completed|finished|closed)\b", re.I)
    scoped = re.compile(r"\bJ0\b|\bmilestone\b|\bwill\b|\bnot\b", re.I)
    for surface in ("README.md", "CLAUDE.md", "docs/roadmap.md",
                    "docs/project_summary.md",
                    "docs/native_support_matrix.md",
                    "docs/architecture.md", "docs/backend_experiments.md",
                    "src/tensorforge/experimental/__init__.py"):
        text = _flat(_read(surface))
        for match in complete.finditer(text):
            window = text[max(0, match.start() - 60):match.end() + 30]
            assert scoped.search(window), (surface, match.group(0))


def test_the_completion_scanner_can_actually_fail():
    """Negative control for both prose scanners above."""
    claim = _RUNTIME_CLAIM
    complete = re.compile(
        r"Phase.J\b[^.;]{0,70}?\b(is|are|was|has been)\s+"
        r"(complete|completed|finished|closed)\b", re.I)
    for detected in ("TensorForge now supports native mini-batching",
                     "NativeDataLoader is available",
                     "the data loader is implemented"):
        assert claim.search(detected), detected
    for detected in ("Phase J is complete", "Phase J has been finished"):
        assert complete.search(detected), detected
    # ...and the accurate sentences this phase must be able to write.
    for accurate in ("Phase J is approved and J0 is complete",
                     "Phase J milestone J0 is complete"):
        match = complete.search(accurate)
        window_ok = re.search(r"\bJ0\b|\bmilestone\b", accurate, re.I)
        assert match is None or window_ok, accurate
