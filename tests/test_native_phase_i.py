"""Phase-I contract guardrails (native dtype generalization).

**Phase I is complete: milestones I0 through I11 have all landed.** This
module is the *contract* half of the phase's guardrails — what the design
says, and whether the runtime has moved where the ladder says it should.
The *closure* half, which asserts that the phase is honestly finished and
cannot drift back, is ``tests/test_native_phase_i_closure.py``.

(The paragraphs below were written milestone by milestone and are kept as
the record of how the phase was built; they describe each milestone at the
moment it landed rather than restating the finished state.)

I0 was a design-and-reconciliation milestone: it shipped
``docs/native_dtype_float32_design.md``, this module, and documentation,
and **no runtime behavior at all**. I1 built the foundation the rest of the
phase stands on — the C++ dtype model, dtype-tagged storage, and the two
typed creation exports that take the library from 52 symbols to 54. I2 made
float32 *movable*: the three transfer exports and the identity copy became
dtype-general and bit-preserving. I3 made it *computed on*, by the
elementwise and unary Core family. I4 extended that to the reduction,
matmul, and view-backward families and to the private Core autograd graph
composed from them — and, because accumulation is where a hidden wider
accumulator would finally show, added the runtime witness I3 recorded as
unavailable to it. I5 extended the same treatment to the CNN stack: all
three Conv2d directions and both MaxPool2d directions execute at both
dtypes through H9's unchanged traversals and predicates, private float32
graphs differentiate through convolution and pooling, and the MaxPool2d
winner buffer stays **float64 at every value dtype** with its ``2**53``
plane bound unchanged (design §13.3).

None of the five moved a **public** capability: float32 is allocatable,
transferable, and now arithmetically usable through the C ABI and the
private typed constructors, and it is still not a supported TensorForge
dtype.

Three kinds of fact therefore live here, and keeping them apart is the
point of the module:

* **What the contract says** — a property of the design document, which
  spans the whole phase and does not move as milestones land.
* **What the repository is now** — the live registries, the live source,
  and the built library, at I4.
* **What is still a promise** — everything I5 onward will do, asserted as
  *absent* so a later milestone cannot be mistaken for an earlier one.

These tests therefore protect two different things at once, and the split
matters:

* **What the contract says.** The load-bearing dtype, storage, ABI,
  dispatch, autograd, module, RNG, optimizer, checkpoint, determinism,
  isolation, and performance decisions must actually be written down, in
  the section that owns each of them, so a later milestone inherits an
  unambiguous design instead of re-deriving one. These assertions are
  **section-scoped** and require **combinations** of architectural terms
  rather than the presence of one vague word — a document that merely
  contains the string "float32" passes nothing here.
* **What the repository still is.** The registries, the checkpoint
  constants, the export count, and Phase H's completion are asserted
  against the **live** module, the **live** source, and the **built**
  library — never against prose.

They deliberately test *values and structure* rather than wording, so
ordinary prose improvements do not require rewriting them. Nothing here
asserts a character count, a paragraph order, or a benchmark number.
"""
import ctypes
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from tensorforge.backends import cpp

REPO_ROOT = Path(__file__).resolve().parent.parent
PHASE_I_DESIGN = REPO_ROOT / "docs" / "native_dtype_float32_design.md"

# The boundary Phase I inherited. The **public dtype registry** half of it
# does not move until milestone I9, so those stay exactly as Phase H left
# them right through I1-I8 even as internal float32 capability appears
# beneath them.
I0_DTYPES = ("float64",)
I0_DEVICES = ("cpu",)
I0_UNSUPPORTED = ("float32", "cuda", "amp")
I0_EXPORT_COUNT = 52
# ...and what milestone **I9** moved them to. This is the phase's one and
# only public registry change (design §27.3), and the same history/today
# split the checkpoint constants below already use applies here: the I0_*
# values are what I1-I8 ran against and are history that does not move
# again, while the I9_* values are what the runtime promises today. The
# device set is deliberately identical in both — a dtype milestone grants
# no device.
I9_DTYPES = ("float64", "float32")
I9_DEVICES = I0_DEVICES
I9_UNSUPPORTED = ("cuda", "amp")
# The checkpoint constants are the one inherited value with a *different*
# milestone: design §16.1 puts them at **I8**, not I9. They are recorded
# here as the pre-I8 baseline and asserted against the post-I8 values
# below, so the move is a stated fact rather than a relaxed assertion.
I0_CHECKPOINT_VERSION = 2
I0_CHECKPOINT_VERSIONS = (1, 2)
I8_CHECKPOINT_VERSION = 3
I8_CHECKPOINT_VERSIONS = (1, 2, 3)

# What I1 added, and the only thing it added to the ABI. The count is
# arithmetic over the inherited baseline so the two cannot drift apart.
I1_EXPORT_COUNT = I0_EXPORT_COUNT + 2  # 54, and it does not move again

# The ABI dtype codes, frozen. Written here independently of the module
# under test so a silent renumbering fails rather than propagating.
DTYPE_CODE_FLOAT64 = 0
DTYPE_CODE_FLOAT32 = 1

# The merged I0 commit. I1's change surface is measured against this, so
# the assertion stays meaningful after I1 is itself committed.
I0_COMMIT = "39d416aa51abc00976a771ecbc7a334545c25e59"

# What the phase plans, which is a property of the contract rather than of
# the runtime. Written out once so a drift in either direction is one diff.
PLANNED_NEW_EXPORTS = (
    "tf_storage_create_typed",
    "tf_storage_create_uninitialized_typed",
)
FINAL_EXPORT_COUNT = 54
FINAL_CHECKPOINT_VERSION = 3
MILESTONES = tuple(f"I{n}" for n in range(12))
PUBLIC_SUPPORT_MILESTONE = "I9"

# The exact constructor surface milestone I7 opened (design §12.1): six
# state-owning classes gained a **keyword-only** ``dtype`` defaulting to
# ``"float64"``, and nothing else did. Written here as a closed set, because
# the interesting failure is in both directions — a class that quietly gains
# one is as much a contract break as a class that loses one. The stateless
# modules, the losses, the metric, the generator, and the optimizers are
# deliberately absent: they own no dtype-bearing numeric state of their own,
# so a dtype argument there would be a second authority that could disagree
# with the data (the optimizers' state follows their parameters, at I8).
I7_DTYPE_CONSTRUCTORS = frozenset({
    "NativeParameter",
    "NativeLinear",
    "NativeConv2d",
    "NativeLayerNorm",
    "NativeBatchNorm1d",
    "NativeBatchNorm2d",
})

# What a *later* phase legitimately added on the same rule, named separately
# so the Phase-I statement above stays exactly what Phase I shipped. The
# assertions below compare against the union, so they remain exact equalities
# in both directions — a class that quietly gains a ``dtype`` argument still
# fails, and so does one that loses it.
#
# ``NativeTensorDataset`` (Phase J, milestone J1) belongs here rather than
# among the absences: it **does** own dtype-bearing numeric state — its
# feature snapshot is materialized at the chosen dtype, which every batch it
# produces then carries — so it takes the argument through the same shared
# ``_native_dtype.normalize_module_dtype`` validator, keyword-only, defaulting
# to ``None`` meaning float64, and infers nothing from the input array.
POST_PHASE_I_DTYPE_CONSTRUCTORS = frozenset({
    "NativeTensorDataset",
})
DTYPE_CONSTRUCTORS = I7_DTYPE_CONSTRUCTORS | POST_PHASE_I_DTYPE_CONSTRUCTORS


def _constructors_with_a_dtype_argument():
    """Every exported ``tensorforge.experimental`` class whose constructor
    accepts ``dtype``, as a set of names."""
    import inspect

    import tensorforge.experimental as experimental

    found = set()
    for name in experimental.__all__:
        obj = getattr(experimental, name)
        if not inspect.isclass(obj):
            continue
        try:
            signature = inspect.signature(obj)
        except (TypeError, ValueError):  # pragma: no cover
            continue
        if "dtype" in signature.parameters:
            found.add(name)
            # Keyword-only at every one of them, so no positional shape
            # changed and no existing call site can be reinterpreted.
            assert (signature.parameters["dtype"].kind
                    is inspect.Parameter.KEYWORD_ONLY), name
            assert signature.parameters["dtype"].default is None, name
    return found

# Surfaces that state *current* status and therefore had to be reconciled
# when the phase opened. Per-milestone historical records deliberately
# preserve superseded wording and are not scanned.
STATUS_SURFACES = (
    "README.md",
    "docs/roadmap.md",
    "docs/project_summary.md",
    "docs/native_support_matrix.md",
    "docs/backend_experiments.md",
    "docs/architecture.md",
)


def _assert_the_public_registry_is_i9s():
    """The live public registry, asserted as **I9's** values.

    Every per-milestone exit gate in this file calls this, and the reason
    it is a shared helper is the reason it exists at all. Those gates were
    written as "milestone IN moved no public capability", which they
    proved by pinning the I0 literals — sound while the registry had not
    moved, and impossible to keep once it did. A test that runs only
    against the current tree cannot tell "I3 did not move this" from "I9
    did", so the honest thing each gate can still assert is that the
    registry is at the phase's *one* recorded change and no other, with the
    attribution recorded in prose beside it.

    What is genuinely preserved, and is checked here in both directions:
    the change is exactly ``"float32"`` moving from ``UNSUPPORTED`` into
    ``SUPPORTED_DTYPES``; nothing else joined either tuple; the device set
    never moved; and float64 is still first and still the default."""
    assert cpp.SUPPORTED_DTYPES == I9_DTYPES
    assert cpp.SUPPORTED_DEVICES == I9_DEVICES == I0_DEVICES
    assert cpp.UNSUPPORTED == I9_UNSUPPORTED
    # Exactly one name moved, and it moved in one direction.
    assert set(cpp.SUPPORTED_DTYPES) - set(I0_DTYPES) == {"float32"}
    assert set(I0_UNSUPPORTED) - set(cpp.UNSUPPORTED) == {"float32"}
    assert set(cpp.UNSUPPORTED) <= set(I0_UNSUPPORTED)
    # The default did not move with it.
    assert cpp.SUPPORTED_DTYPES[0] == "float64"
    assert cpp.normalize_dtype(None) == "float64"
    assert cpp.backend_info()["dtype"] == "float64"


def _read(relative):
    return (REPO_ROOT / relative).read_text(encoding="utf-8")


def _flat(text):
    """Whitespace-flattened, emphasis-stripped text, so a claim split
    across lines or wrapped in markdown still reads as one sentence."""
    return re.sub(r"\s+", " ", re.sub(r"[*`]", "", text))


def _design():
    return PHASE_I_DESIGN.read_text(encoding="utf-8")


def _raw_sections():
    """``{heading_number: body}`` for every ``## N.`` section of the
    design, so an assertion can be scoped to the section that actually
    owns its subject rather than to the whole file.

    Scoping is the point: "no casting" appearing anywhere in a 1,500-line
    document proves nothing, while "no casting" inside the section on
    casting is the decision itself.
    """
    text = _design()
    sections = {}
    matches = list(re.finditer(r"^## (\d+)\. (.+)$", text, re.M))
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        sections[int(match.group(1))] = text[match.end():end]
    return sections


def _raw_section(number):
    """A section's text as written — used wherever punctuation carries the
    meaning (``void*``, ``tf_*`` symbols, tuple literals)."""
    body = _raw_sections().get(number)
    assert body, f"the design has no section {number}"
    return body


def _section(number):
    """A section flattened for prose matching: emphasis and code markers
    removed and whitespace collapsed, so a claim split across lines or
    wrapped in markdown still reads as one sentence."""
    return _flat(_raw_section(number))


def _subsection(number, minor):
    """One ``### N.M`` block of a section, for the assertions that must not
    see a neighbouring subsection's text — notably the ABI section, whose
    *rejected* forms are deliberately spelled out beside its accepted
    ones."""
    body = _raw_section(number)
    marks = list(re.finditer(rf"^### {number}\.(\d+) ", body, re.M))
    for index, mark in enumerate(marks):
        if int(mark.group(1)) != minor:
            continue
        end = marks[index + 1].start() if index + 1 < len(marks) else len(body)
        return body[mark.end():end]
    raise AssertionError(f"the design has no section {number}.{minor}")


def _all_of(haystack, *needles):
    """Every needle present, case-insensitively — the combination form
    these tests use instead of single-keyword searches."""
    lowered = haystack.lower()
    missing = [needle for needle in needles if needle.lower() not in lowered]
    return missing


def _cpp_code_only(text):
    """C++ with ``//`` comments removed, so a rule is measured against what
    compiles rather than against what is explained.

    Both source guardrails below need this: each states its rule in a
    comment beside the code, and a naive substring search would find the
    prose and fail on the very file that gets it right.
    """
    return "\n".join(re.sub(r"//.*", "", line) for line in text.splitlines())


def _source_exports():
    names = set()
    for source in sorted((REPO_ROOT / "cpp" / "src").glob("*.cpp")):
        text = source.read_text(encoding="utf-8")
        names.update(re.findall(r"TF_EXPORT[^;{]*?\b(tf_[a-z0-9_]+)\s*\(",
                                text, re.S))
    return names


# ---------------------------------------------------------------------------
# the contract exists, is linked, and is scoped to this phase
# ---------------------------------------------------------------------------

def test_the_phase_i_design_exists_and_is_not_a_stub():
    assert PHASE_I_DESIGN.is_file(), "docs/native_dtype_float32_design.md is missing"
    text = _design()
    # A contract this phase depends on cannot be a page of headings.
    assert len(text) > 20_000, (
        f"the Phase-I contract is only {len(text)} characters; it is meant "
        f"to be implementation-grade"
    )
    assert re.search(r"^## 29\.", text, re.M), "the milestone ladder section is gone"


def test_the_design_is_linked_from_the_readme_and_the_agent_instructions():
    for surface in ("README.md", "CLAUDE.md"):
        assert "docs/native_dtype_float32_design.md" in _read(surface), surface


def test_the_design_states_its_milestone_status_and_what_is_unshipped():
    """The status line, parsed rather than matched as a phrase.

    **Retired at I11, in one half only.** Through I10 this required the
    line to name *both* a completed run and a first unstarted milestone,
    which was right while one existed. At closure none does, and demanding
    one would demand a false sentence — the exact failure this file exists
    to prevent, pointed the other way. So the "unstarted" half is now
    conditional on there being an unstarted milestone at all, while the
    arithmetic that makes the two halves meet exactly is kept for as long
    as both are present. Everything else is unchanged, including the
    cross-checks against runtime reality below, which are what stop a
    status line from running ahead of the code."""
    text = _flat(_design())
    status = re.search(r"Phase-I status:(.{0,200})", text, re.I)
    assert status, "the design does not state its milestone status"
    claim = status.group(1)
    # Structural, and genuinely milestone-agnostic: the line must name a
    # first milestone and a last completed one.
    completed = re.search(
        r"\bI0\b.{0,80}?\bI(\d+)\b.{0,40}?complete", claim, re.I
    )
    assert completed, (
        f"the status line does not record a completed run starting at I0: "
        f"{claim!r}"
    )
    last_complete = int(completed.group(1))
    final = len(MILESTONES) - 1
    unstarted = re.search(
        r"complete.{0,120}?\bI(\d+)\b.{0,80}?not started", claim, re.I
    )
    if last_complete < final:
        # The phase is still running, so the line must say which milestones
        # are not, and the two halves must meet exactly: no milestone
        # claimed twice and none left unaccounted for.
        assert unstarted, (
            f"the status line does not record which milestones are "
            f"unstarted: {claim!r}"
        )
        first_unstarted = int(unstarted.group(1))
        assert first_unstarted == last_complete + 1, (
            f"the status line leaves a gap or an overlap between complete "
            f"({last_complete}) and unstarted ({first_unstarted}): {claim!r}"
        )
    else:
        # The phase is closed: the whole ladder is complete, so there is
        # nothing left to call unstarted and the line must not invent one.
        assert last_complete == final, claim
        assert unstarted is None, (
            f"the status line claims the ladder is complete and still names "
            f"an unstarted milestone: {claim!r}"
        )
    assert f"I{final}" in claim, (
        f"the status line does not run to the end of the ladder: {claim!r}"
    )
    # ...and the claim is checked against **runtime reality**, so a status
    # line cannot be advanced ahead of the code. I8 is defined as the
    # milestone that moves the checkpoint format to 3 (§16.1), which makes
    # the live constant an independent witness for the I7/I8 boundary. The
    # old assertion pinned a literal milestone number and so caught this
    # for free; parsing the split structurally would not, and dropping the
    # check with it would have been a real weakening.
    from tensorforge.experimental import native_checkpoint

    if last_complete >= 8:
        assert native_checkpoint._FORMAT_VERSION == 3, (
            f"the status line claims I{last_complete} complete, but the "
            f"checkpoint format is still version "
            f"{native_checkpoint._FORMAT_VERSION}; I8 is the milestone "
            f"that moves it to 3"
        )
        assert native_checkpoint._SUPPORTED_FORMAT_VERSIONS == (1, 2, 3)
    else:
        assert native_checkpoint._FORMAT_VERSION == 2, (
            f"the status line claims only I{last_complete} complete, but "
            f"the checkpoint format has already moved to version "
            f"{native_checkpoint._FORMAT_VERSION}"
        )
    # The public registry is I9's, and the same cross-check applies to it —
    # in **both** directions, so a status line can neither run ahead of the
    # registry nor lag behind it.
    if last_complete >= 9:
        assert "float32" in cpp.SUPPORTED_DTYPES, (
            f"the status line claims I{last_complete} complete, but "
            f"SUPPORTED_DTYPES is still {cpp.SUPPORTED_DTYPES}; I9 is the "
            f"milestone that moves it"
        )
        assert "float32" not in cpp.UNSUPPORTED
    else:
        assert cpp.SUPPORTED_DTYPES == I0_DTYPES, (
            f"the status line claims only I{last_complete} complete, but "
            f"the public dtype registry has already moved to "
            f"{cpp.SUPPORTED_DTYPES}"
        )
        assert cpp.UNSUPPORTED == I0_UNSUPPORTED
    # ...and that I0 itself shipped no behavior, which is a historical
    # fact about I0 and stays true however far the phase progresses. A
    # phase that has begun but claims delivery is the exact drift here.
    assert re.search(r"I0 adds no runtime behavior", text, re.I), (
        "the design does not state that I0 adds no runtime behavior"
    )
    # What the design must say about the public boundary depends on which
    # side of I9 the phase is on, and it must say one of them explicitly —
    # silence is the drift this catches. Before I9 it records the boundary
    # as unmoved; from I9 it records the move and the values it moved to.
    if last_complete >= 9:
        assert re.search(
            r"SUPPORTED_DTYPES (?:reads|is|=) ?=? ?\(\"float64\", "
            r"\"float32\"\)|SUPPORTED_DTYPES = \(\"float64\", \"float32\"\)",
            text,
        ), "the design does not record the moved public dtype registry"
    else:
        assert re.search(
            r"SUPPORTED_DTYPES still reads \(\"float64\",\)", text
        ), "the design no longer records the unchanged public dtype registry"


def test_the_design_records_the_current_float64_only_reality():
    """Requirement: the contract is written against verified reality, not
    against an imagined starting point."""
    reality = _raw_section(2)
    missing = _all_of(
        reality,
        "double* data",          # the physical buffer as it is today
        "int64_t size",          # measured in logical elements
        "float64",
        "52",                    # the export count it starts from
        "version 2",             # the checkpoint format it starts from
        "np.float64",            # the hardcoded Python boundary
        "opaque handles",        # the fact the whole ABI plan rests on
    )
    assert not missing, f"the reality section omits {missing}"
    # The reality report must separate what is verified from what is a
    # future decision, or it is just more design.
    assert re.search(r"verified|read out of the tree", reality, re.I)


def test_the_design_records_the_float32_and_float64_target():
    scope = _section(1)
    missing = _all_of(scope, "float32", "float64", "cpu")
    assert not missing, f"the scope section omits {missing}"
    # Exactly two real-number dtypes, and the excluded ones named.
    dtype_model = _section(3)
    assert re.search(r"exactly two", dtype_model, re.I), (
        "the dtype model does not state that there are exactly two dtypes"
    )


# ---------------------------------------------------------------------------
# the ABI plan
# ---------------------------------------------------------------------------

def test_the_design_plans_exactly_two_new_abi_exports_by_name():
    abi = _section(6)
    for symbol in PLANNED_NEW_EXPORTS:
        assert symbol in abi, f"section 6 does not name {symbol}"
    assert re.search(r"exactly two|only.{0,20}two", abi, re.I), (
        "section 6 does not state that exactly two exports are planned"
    )
    # No third symbol may be smuggled in as a plan. Scoped to the
    # subsection that *declares* the new exports, because the neighbouring
    # subsections deliberately spell out the forms that are rejected
    # (tf_core_add_f32) and the queries that are not added
    # (tf_storage_dtype) — naming those is the contract working, not
    # drift.
    declared = _subsection(6, 2)
    proposed = set(re.findall(r"\btf_[a-z0-9_]+\b", declared))
    unknown = proposed - _source_exports() - set(PLANNED_NEW_EXPORTS)
    assert not unknown, (
        f"section 6.2 names symbols that neither exist nor are planned: "
        f"{sorted(unknown)}"
    )


def test_the_design_lists_the_symbols_it_deliberately_does_not_add():
    """The other half of "exactly two": a contract that only says what it
    adds invites a third symbol as an implementation detail."""
    absent = _subsection(6, 6)
    for query in ("tf_storage_dtype", "tf_storage_cast"):
        assert query in absent, f"section 6.6 does not rule out {query}"
    assert re.search(r"adds no|not added|no per-dtype", absent, re.I)


def test_the_design_records_the_final_export_count_of_fifty_four():
    text = _flat(_design())
    assert re.search(r"52\s*(?:→|->|to)\s*\*{0,2}54", text), (
        "the design does not record the 52 -> 54 export growth"
    )
    compatibility = _section(24)
    assert "54" in compatibility, "the ABI table does not carry the final count"


def test_the_design_records_that_existing_exports_stay_compatible():
    compatibility = _section(24)
    missing = _all_of(
        compatibility,
        "tf_storage_create",
        "compatible",
        "unchanged",
        "tf_storage_copy_from",
        "opaque",
    )
    assert not missing, f"the ABI compatibility table omits {missing}"
    # The old creators are kept, not replaced.
    creators = _section(6)
    assert re.search(
        r"not removed, not renamed|remain|stay",
        creators, re.I), "section 6 does not keep the existing creators"


def test_the_design_rejects_per_operation_float32_abi_duplication():
    abi = _section(6)
    window = re.search(
        r"per-operation float32 exports are (?:explicitly )?\*{0,2}rejected"
        r"|reject(?:ed|s)?[^.]{0,120}per-operation float32",
        abi, re.I)
    assert window, "section 6 does not reject per-operation float32 exports"
    # And it must say *why*, with more than one reason.
    assert re.search(r"tf_core_add_f32|tf_core_matmul_f32", abi), (
        "section 6 does not name the rejected duplicated form"
    )
    assert re.search(r"dispatch shape|call site|surface", abi, re.I), (
        "section 6 rejects duplication without recording its reasons"
    )


def test_the_design_divides_the_abi_into_handle_and_raw_buffer_paths():
    raw = _section(7)
    missing = _all_of(
        raw,
        "handle",
        "raw",
        "float64",
        "tf_elementwise_add",
        "tf_matmul_tiled",
        "RAW_KERNEL_DTYPES",
    )
    assert not missing, f"the raw-buffer section omits {missing}"
    # The registry is a *future* declaration, and the design must say when.
    assert re.search(r"\bI2\b", raw), (
        "the raw-buffer section does not say which milestone introduces "
        "the registry"
    )


# ---------------------------------------------------------------------------
# storage, dtype authority, and dispatch
# ---------------------------------------------------------------------------

def test_the_design_records_dtype_tagged_storage_and_one_item_size_authority():
    storage = _raw_section(4)
    missing = _all_of(
        storage,
        "void*",          # untyped data pointer
        "dtype",          # the tag
        "itemsize",       # checked byte arithmetic
        "overflow",       # ...and that it is checked
        "zero-initial",   # the preserved H1 default
    )
    assert not missing, f"the storage section omits {missing}"
    model = _section(3)
    assert re.search(r"item.?size", model, re.I), (
        "the dtype model does not own the item-size authority"
    )
    assert re.search(r"single|one|canonical", model, re.I)


def test_the_design_keeps_shape_stride_and_offset_in_logical_elements():
    storage = _section(4)
    missing = _all_of(storage, "logical element", "stride", "offset", "byte")
    assert not missing, f"the storage section omits {missing}"
    assert re.search(
        r"(shapes?, strides?|strides?, (tensor )?offsets?)[^.]{0,200}"
        r"logical elements"
        r"|logical elements[^.]{0,200}(shape|stride|offset)",
        storage, re.I), (
        "the storage section does not state that layout stays measured in "
        "logical elements"
    )


def test_the_design_makes_storage_the_single_dtype_authority():
    ownership = _section(5)
    missing = _all_of(ownership, "storage", "authority", "view")
    assert not missing, f"the dtype-ownership section omits {missing}"
    # The two failure modes it exists to prevent.
    assert re.search(r"no dtype field|has no dtype|never gain", ownership, re.I), (
        "the ownership section does not forbid a second dtype tag on views"
    )
    assert re.search(r"never cast|does not cast|no.{0,20}cast", ownership, re.I), (
        "the ownership section does not state that a view never casts"
    )


def test_the_design_locks_one_narrow_typed_dispatch_per_operation():
    dispatch = _section(8)
    missing = _all_of(dispatch, "template", "float", "double", "instantiat")
    assert not missing, f"the dispatch section omits {missing}"
    assert re.search(r"one dispatch|exactly one|one narrow", dispatch, re.I), (
        "the dispatch section does not require one dispatch per operation"
    )
    # The forbidden forms, which are the whole reason the section exists.
    for banned in ("virtual", "string"):
        assert re.search(rf"no {banned}|never.{{0,30}}{banned}",
                         dispatch, re.I), (
            f"the dispatch section does not rule out {banned} dispatch"
        )
    assert re.search(r"must not|forbidden", dispatch, re.I), (
        "the dispatch section does not say where dispatch may not happen"
    )


# ---------------------------------------------------------------------------
# numerical and semantic rules
# ---------------------------------------------------------------------------

def test_the_design_records_no_casting_and_no_promotion():
    casting = _section(9)
    missing = _all_of(casting, "no implicit promotion", "cast", "astype")
    assert not missing, f"the no-casting section omits {missing}"
    assert re.search(r"no\b[^.]{0,40}\bastype|astype[^.]{0,60}(not|no)\b",
                     casting, re.I), (
        "the section does not rule out an explicit cast operation"
    )
    # Both directions, explicitly — silent narrowing is the easy mistake.
    assert re.search(r"narrow", casting, re.I), (
        "the section does not forbid silent narrowing"
    )


def test_the_design_records_mixed_dtype_rejection_across_the_layers():
    casting = _section(9)
    for layer in ("matmul", "convolution", "normalization", "optimizer",
                  "checkpoint", "gradient"):
        assert re.search(layer, casting, re.I), (
            f"the mixed-dtype rejection table omits {layer}"
        )
    # Rejection must be ordered, not merely present.
    assert re.search(
        r"before[^.]{0,80}(allocat|mutat)", casting, re.I), (
        "the section does not require rejection before allocation or "
        "mutation"
    )


def test_the_design_records_the_float32_accumulation_policy():
    accumulation = _section(10)
    assert re.search(
        r"float32[^.]{0,80}accumulate[^.]{0,40}float32", accumulation, re.I), (
        "the accumulation section does not state the float32 policy"
    )
    assert re.search(r"no hidden float64|hidden[^.]{0,30}accumulator",
                     accumulation, re.I), (
        "the section does not forbid a hidden wider accumulator"
    )
    # The three comparison regimes must be distinguished, or a later
    # milestone will pick whichever is convenient.
    missing = _all_of(accumulation, "bit-identical", "tolerance", "ULP")
    assert not missing, f"the accumulation section omits {missing}"
    # And the comparison that is explicitly not a contract.
    assert re.search(r"float32[^.]{0,120}(not a contract|forbidden)"
                     r"|forbidden[^.]{0,120}float64",
                     accumulation, re.I), (
        "the section does not forbid making float32-vs-float64 agreement a "
        "contract"
    )


# ---------------------------------------------------------------------------
# checkpointing
# ---------------------------------------------------------------------------

def test_the_design_records_checkpoint_version_three_and_its_compatibility():
    checkpoint = _section(16)
    assert re.search(r"version 3|_FORMAT_VERSION\s*=\s*3", checkpoint, re.I), (
        "the checkpoint section does not name version 3"
    )
    assert re.search(r"\(1,\s*2,\s*3\)", checkpoint), (
        "the checkpoint section does not record the accepted version tuple"
    )
    # The format *name* must not move, which is the G5 precedent.
    assert re.search(r"tensorforge\.native_checkpoint", checkpoint), (
        "the checkpoint section does not keep the format name"
    )
    # And v3 must be designed rather than shipped at I0.
    assert re.search(r"I8", checkpoint), (
        "the checkpoint section does not say which milestone activates v3"
    )


def test_the_design_defines_versions_one_and_two_as_float64_only():
    checkpoint = _section(16)
    assert re.search(
        r"(v1|version)[^.]{0,60}(and|/)[^.]{0,20}(v2|2)[^.]{0,80}float64"
        r"|float64-only",
        checkpoint, re.I), (
        "the checkpoint section does not define v1/v2 as float64-only"
    )
    assert re.search(r"never[^.]{0,60}guess", checkpoint, re.I), (
        "the checkpoint section does not forbid guessing a v1/v2 payload "
        "to be float32"
    )
    # No load-time conversion of any kind.
    assert "map_location" in checkpoint, (
        "the checkpoint section does not rule out map_location"
    )


def test_the_design_specifies_a_deterministic_serialization_encoding():
    encoding = _section(17)
    missing = _all_of(
        encoding,
        "byte order",
        "shape",
        "NaN",
        "signed zero",
        "infinit",
        "bit-preserving",
    )
    assert not missing, f"the encoding section omits {missing}"
    # Payload length must be derived rather than stored, which is the
    # decision that keeps the manifest single-sourced.
    assert re.search(r"derived", encoding, re.I), (
        "the encoding section does not say how the payload length is "
        "established"
    )


# ---------------------------------------------------------------------------
# determinism, isolation, and performance
# ---------------------------------------------------------------------------

def test_the_design_requires_exact_resume_separately_for_each_dtype():
    determinism = _section(18)
    assert re.search(r"separate", determinism, re.I), (
        "the determinism section does not require separate per-dtype proofs"
    )
    # The state families the proof has to carry.
    for item in ("parameter", "buffer", "moment", "step counter",
                 "generator", "mask", "logits", "evaluation"):
        assert re.search(item, determinism, re.I), (
            f"the resume requirement omits {item}"
        )
    # ...and the distinction that stops the proof degenerating into a
    # float32-matches-float64 comparison.
    assert re.search(
        r"does not need to produce the same numbers|never required to "
        r"(agree|reproduce)|not a contract",
        determinism, re.I), (
        "the determinism section does not distinguish exact resume from "
        "float64 agreement"
    )


def test_the_design_records_stable_native_isolation():
    isolation = _section(19)
    missing = _all_of(
        isolation,
        "stable",
        "native",
        "implicit",
        "stable_framework_integration",
    )
    assert not missing, f"the isolation section omits {missing}"
    # Hyphenated or spaced, an environment-variable selector must be
    # ruled out by name.
    assert re.search(r"environment.variable", isolation, re.I), (
        "the isolation section does not rule out an environment-variable "
        "selector"
    )


def test_the_design_requires_phase_h_performance_preservation():
    performance = _section(22)
    missing = _all_of(
        performance,
        "contiguous fast path",
        "broadcast",
        "matmul",
        "reduction",
        "convolution",
        "optimizer",
        "bit-preserving",
    )
    assert not missing, f"the performance section omits {missing}"
    # The two rules that keep dtype support from becoming a slow path.
    assert re.search(r"no dtype string[^.]{0,40}(hot )?loop"
                     r"|string parsing", performance, re.I), (
        "the performance section does not ban dtype string parsing in hot "
        "loops"
    )
    # And the project's oldest benchmark rule, restated rather than dropped.
    assert re.search(r"no (speed|timing)[^.]{0,40}assert"
                     r"|asserted by no test|no result file",
                     performance, re.I), (
        "the performance section does not restate the benchmark rules"
    )


# ---------------------------------------------------------------------------
# scope boundaries
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("excluded", (
    "CUDA", "AMP", "float16", "bfloat16", "integer tensors",
    "data loaders", "distributed", "pybind11", "BLAS", "OpenMP",
    "memory pool",
))
def test_the_design_excludes_the_out_of_scope_subjects(excluded):
    """Each must appear in the non-goals, not merely somewhere."""
    scope = _section(1)
    boundaries = _section(28)
    combined = f"{scope} {boundaries}"
    assert excluded.lower() in combined.lower(), (
        f"{excluded!r} is not recorded as out of scope"
    )


def test_the_design_keeps_integer_metadata_from_becoming_integer_tensors():
    buffers = _section(13)
    missing = _all_of(buffers, "winner", "float64", "metadata")
    assert not missing, f"the buffer section omits {missing}"
    assert re.search(r"no integer tensor|adds no integer", buffers, re.I), (
        "the buffer section does not rule out an integer tensor dtype"
    )


def test_the_design_leaves_the_generator_algorithm_alone():
    rng = _section(14)
    missing = _all_of(rng, "splitmix64", "unchanged", "call")
    assert not missing, f"the RNG section omits {missing}"
    assert re.search(r"dtype-independent|no dtype field", rng, re.I), (
        "the RNG section does not state that generator state has no dtype"
    )


def test_the_design_ties_optimizer_state_dtype_to_its_parameter():
    optimizer = _section(15)
    missing = _all_of(optimizer, "NativeAdam", "NativeSGD", "moment",
                      "step counter")
    assert not missing, f"the optimizer section omits {missing}"
    assert re.search(r"step counters? (stay|remain)[^.]{0,40}"
                     r"(Python )?integer|metadata, not tensors",
                     optimizer, re.I), (
        "the optimizer section does not keep step counters as metadata"
    )


def test_the_design_states_the_autograd_dtype_invariants():
    autograd = _section(11)
    missing = _all_of(autograd, "grad", "leaf", "saved", "temporar")
    assert not missing, f"the autograd section omits {missing}"
    assert re.search(r"reject", autograd, re.I), (
        "the autograd section does not reject mixed-dtype accumulation"
    )


def test_the_design_defines_dtype_aware_failure_atomicity():
    failure = _section(20)
    for failure_mode in ("unknown dtype", "overflow", "allocation failure",
                         "mixed dtype", "partial"):
        assert re.search(failure_mode, failure, re.I), (
            f"the failure matrix omits {failure_mode}"
        )
    assert re.search(r"partially constructed|publish", failure, re.I), (
        "the failure section does not forbid publishing a partial object"
    )


def test_the_design_keeps_the_existing_ownership_model():
    ownership = _section(21)
    assert re.search(r"no new ownership framework|introduces no new",
                     ownership, re.I), (
        "the ownership section does not disclaim a new ownership model"
    )
    for subject in ("view", "graph", "buffer", "generator", "concurrent"):
        assert re.search(subject, ownership, re.I), (
            f"the ownership section omits {subject}"
        )


def test_the_design_states_the_public_python_dtype_form():
    public = _section(25)
    missing = _all_of(public, "float64", "float32", "None", "read-only")
    assert not missing, f"the public-compatibility section omits {missing}"
    assert re.search(r"no alias", public, re.I), (
        "the public section does not rule out dtype aliases"
    )
    assert re.search(r"default", public, re.I)


def test_the_design_lays_out_a_layered_test_matrix():
    testing = _section(26)
    for layer in ("storage", "views", "elementwise", "matmul", "autograd",
                  "checkpoint", "resume", "sanitiz", "benchmark"):
        assert re.search(layer, testing, re.I), (
            f"the test matrix omits {layer}"
        )
    # The comparison regimes must be assigned, not left to taste.
    missing = _all_of(testing, "bitwise", "tolerance", "exact")
    assert not missing, f"the test matrix omits {missing}"


def test_the_design_requires_both_platforms_and_the_sanitizers():
    platform = _section(23)
    missing = _all_of(platform, "Windows", "Linux", "ASan", "UBSan",
                      "LeakSanitizer")
    assert not missing, f"the platform section omits {missing}"
    assert re.search(r"no CUDA compiler|no CUDA toolkit", platform, re.I), (
        "the platform section does not exclude a CUDA compiler requirement"
    )


# ---------------------------------------------------------------------------
# the milestone ladder and the rollout point
# ---------------------------------------------------------------------------

def test_the_milestone_ladder_runs_i0_to_i11_once_each_in_order():
    text = _design()
    headings = re.findall(r"^### (I\d+) — ", text, re.M)
    assert headings == list(MILESTONES), (
        f"the ladder is {headings}, expected {list(MILESTONES)}"
    )


def test_every_milestone_carries_an_exit_gate_and_a_commit_message():
    text = _design()
    ladder = text.split("## 29.", 1)[1].split("## 30.", 1)[0]
    blocks = re.split(r"^### I\d+ — ", ladder, flags=re.M)[1:]
    assert len(blocks) == len(MILESTONES), len(blocks)
    for milestone, block in zip(MILESTONES, blocks):
        assert re.search(r"\*\*Exit gate:\*\*", block), (
            f"{milestone} has no exit gate"
        )
        assert re.search(r"\*\*Commit message:\*\*", block), (
            f"{milestone} has no suggested commit message"
        )


def test_the_ladder_claims_no_milestone_beyond_i11():
    """An I12 would be a phase this document does not define."""
    text = _design()
    for match in re.finditer(r"\bI1[2-9]\b", text):
        raise AssertionError(f"the design names {match.group(0)}")


def test_the_design_names_i9_as_the_public_support_milestone():
    rollout = _section(27)
    assert PUBLIC_SUPPORT_MILESTONE in rollout, (
        "the rollout section does not name the milestone that enables "
        "public float32 support"
    )
    assert re.search(
        rf"{PUBLIC_SUPPORT_MILESTONE}\b[^.]{{0,200}}"
        r"(SUPPORTED_DTYPES|registry|public)"
        rf"|registry changes at \*{{0,2}}{PUBLIC_SUPPORT_MILESTONE}",
        rollout, re.I | re.S), (
        "the rollout section does not tie the registry change to I9"
    )
    # The four states must be distinguished, or "implemented" will be read
    # as "supported" by the first milestone that ships a kernel.
    for state in ("internal", "tested", "public", "closure"):
        assert re.search(state, rollout, re.I), (
            f"the rollout section does not distinguish the {state} state"
        )
    # And the target registry values are written down.
    assert re.search(r'\("float64",\s*"float32"\)', rollout), (
        "the rollout section does not record the final SUPPORTED_DTYPES"
    )
    assert re.search(r'\("cuda",\s*"amp"\)', rollout), (
        "the rollout section does not record the final UNSUPPORTED"
    )


# ---------------------------------------------------------------------------
# I0 changed no runtime: the live registries, constants, and exports
# ---------------------------------------------------------------------------

def test_the_runtime_registries_are_exactly_what_phase_i_inherited():
    _assert_the_public_registry_is_i9s()


def test_float32_is_genuinely_reachable_and_nothing_beyond_it_is():
    """The registry claim, checked against behavior rather than trusted.

    Through I8 this asserted the opposite — that float32 was genuinely
    *un*reachable — because a contract document must not quietly enable
    anything. **I9 enabled it deliberately**, so the same test now has to
    prove the promise is real rather than merely written down, and that the
    boundary moved by exactly one dtype and not to "any dtype"."""
    import numpy as np

    values = np.zeros((2, 2), dtype=np.float64)
    for build in (
        lambda: cpp.NativeTensorCore.from_array(values, dtype="float32"),
        lambda: cpp.NativeTensorCore.zeros((2, 2), dtype="float32"),
        lambda: cpp.NativeTensorCore.full((2, 2), 1.0, dtype="float32"),
        lambda: cpp.NativeStorage(4, dtype="float32"),
    ):
        if not cpp.is_available():
            break
        built = build()
        try:
            assert built.dtype == "float32"
        finally:
            built.close()
    assert cpp.normalize_dtype("float32") == "float32"
    # ...and the default did not move with it.
    assert cpp.normalize_dtype(None) == "float64"
    # ...and no third dtype came along for the ride.
    for absent in ("float16", "bfloat16", "int64", "complex64"):
        with pytest.raises(ValueError):
            cpp.normalize_dtype(absent)


def test_no_dtype_capability_name_entered_any_registry():
    banned = ("float32", "float16", "bfloat16", "typed", "cast", "astype",
              "promote", "promotion", "dtype_code", "amp")
    registries = (
        cpp.RAW_KERNELS, cpp.TENSOR_CORE_KERNELS, cpp.TENSOR_CORE_OPS,
        cpp.AUTOGRAD_OPS, cpp.NATIVE_MODULES, cpp.NATIVE_LOSSES,
        cpp.NATIVE_METRICS, cpp.NATIVE_OPTIMIZERS, cpp.STATE_SUPPORT,
    )
    for registry in registries:
        for name in registry:
            lowered = str(name).lower()
            for word in banned:
                assert word not in lowered, (registry, name, word)


def test_the_checkpoint_constants_moved_exactly_once_and_only_at_i8():
    """The format **name** never moves; the version moved once, at I8.

    Both halves matter. Every version the phase inherited is still
    accepted — a pre-Phase-I float64 archive still loads — and the only
    value added is 3. Nothing was dropped, renumbered, or reordered."""
    from tensorforge.experimental import native_checkpoint

    assert native_checkpoint._FORMAT == "tensorforge.native_checkpoint"
    assert native_checkpoint._FORMAT_VERSION == I8_CHECKPOINT_VERSION
    assert (native_checkpoint._SUPPORTED_FORMAT_VERSIONS
            == I8_CHECKPOINT_VERSIONS)
    # The inherited versions survive, in order, with exactly one addition.
    assert (native_checkpoint._SUPPORTED_FORMAT_VERSIONS[:len(
        I0_CHECKPOINT_VERSIONS)] == I0_CHECKPOINT_VERSIONS)
    assert (set(I8_CHECKPOINT_VERSIONS) - set(I0_CHECKPOINT_VERSIONS)
            == {I8_CHECKPOINT_VERSION})
    # ...and the phase's planned end state is where I8 landed.
    assert I8_CHECKPOINT_VERSION == FINAL_CHECKPOINT_VERSION
    # The in-memory optimizer state schema is a different thing and did
    # not move with the file format (design §15, §16.2).
    from tensorforge.experimental import native_optimizer_state
    assert native_optimizer_state.FORMAT_VERSION == 1


def test_the_production_export_count_is_now_fifty_four():
    """I1 added the two typed creators and nothing else. Stated as
    arithmetic over Phase H's 52 so that an unplanned addition and a
    silent removal both fail, rather than cancelling out."""
    exports = _source_exports()
    assert len(exports) == I1_EXPORT_COUNT, sorted(exports)
    for planned in PLANNED_NEW_EXPORTS:
        assert planned in exports, f"{planned} is missing from the source"
    assert len(exports - set(PLANNED_NEW_EXPORTS)) == I0_EXPORT_COUNT
    # 54 is the count for the **whole** phase: no later milestone adds a
    # symbol, so any per-operation or per-dtype export is a contract
    # violation wherever it appears.
    assert not [name for name in exports
                if name.endswith(("_f32", "_f64", "_float32", "_float64"))]


@pytest.mark.skipif(not cpp.is_available(),
                    reason="the native library is not built")
def test_the_built_library_exports_exactly_what_the_source_does():
    library = cpp._require_library()
    for name in sorted(_source_exports()):
        assert hasattr(library, name), name
    for planned in PLANNED_NEW_EXPORTS:
        assert hasattr(library, planned), (
            f"the built library does not export {planned}, which I1 adds"
        )


def test_no_dtype_query_or_casting_symbol_was_added():
    """The two creators are sufficient *because* the dtype travels with
    the data. A query export would be a second authority for a value the
    Python wrapper already owns, and the phase adds none — at any
    milestone."""
    exports = _source_exports()
    for absent in ("tf_storage_dtype", "tf_dtype_size", "tf_dtype_item_size",
                   "tf_dtype_name", "tf_storage_bytes", "tf_storage_cast",
                   "tf_storage_astype", "tf_dtype_promote"):
        assert absent not in exports, absent


def test_the_typed_creators_and_the_i2_raw_kernel_registry_are_declared():
    declared = _read("src/tensorforge/backends/cpp.py")
    for planned in PLANNED_NEW_EXPORTS:
        assert planned in declared, (
            f"{planned} is not declared in the ctypes layer; I1 declares it"
        )
    # The raw-kernel dtype registry was an I2 deliverable precisely because
    # a contract-only tuple would have advertised a distinction that was
    # not yet observable (design section 7.2): before I2 every dtype was
    # float64, so the tuple would have been indistinguishable from
    # SUPPORTED_DTYPES. I2 made the distinction real, so the registry lands
    # here — and it is a *different* fact from the public promise.
    assert cpp.RAW_KERNEL_DTYPES == ("float64",)
    assert cpp.backend_info()["raw_kernel_dtypes"] == cpp.RAW_KERNEL_DTYPES
    # The two tuples happen to be equal today and are **not** the same
    # statement: this one is a permanent property of seven handle-free
    # kernels, the other is a public promise that moves at I9. They are
    # reported as separate keys so neither can be read off the other, and
    # they are declared separately in the source rather than aliased.
    assert "RAW_KERNEL_DTYPES = (" in declared
    info = cpp.backend_info()
    assert "raw_kernel_dtypes" in info and "supported_dtypes" in info
    # ...and no raw kernel gained a per-dtype wrapper to go with it.
    assert not [name for name in cpp.RAW_KERNELS
                if name.endswith(("_f32", "_f64"))]
    # Internal representability is wider than both — that gap is the whole
    # point of the registry existing now rather than at I0.
    assert set(cpp._DTYPE_CODES) == {"float64", "float32"}
    assert set(cpp.RAW_KERNEL_DTYPES) < set(cpp._DTYPE_CODES)


def test_the_cpp_storage_struct_is_dtype_tagged():
    """The premise of the whole phase, now delivered: one untyped owned
    pointer, a logical element count whose meaning did not move, and one
    dtype tag."""
    header = _read("cpp/include/tf_internal.h")
    struct = re.search(r"struct Storage\s*\{(.+?)\};", header, re.S)
    assert struct, "tf::Storage is no longer declared in the shared header"
    body = struct.group(1)
    assert re.search(r"\bvoid\*\s+data;", body), (
        "the storage buffer is not an untyped pointer"
    )
    assert re.search(r"\bint64_t\s+size;", body), (
        "the storage element count is gone or renamed"
    )
    assert re.search(r"\bDtype\s+dtype", body), (
        "the storage carries no dtype tag"
    )
    # A union or a second typed pointer would let the tag and the buffer
    # disagree about what the memory holds, which is exactly what the
    # single untyped pointer exists to prevent.
    assert "double*" not in body and "float*" not in body, (
        "a typed pointer survives inside Storage"
    )
    assert "union" not in body, "Storage holds a union of typed pointers"
    # The dtype enum and its frozen codes.
    assert re.search(rf"TF_DTYPE_FLOAT64\s*=\s*{DTYPE_CODE_FLOAT64}\b", header)
    assert re.search(rf"TF_DTYPE_FLOAT32\s*=\s*{DTYPE_CODE_FLOAT32}\b", header)


def test_storage_owns_a_genuine_typed_array_under_cpp17():
    """The allocator must create a real ``float[]`` or ``double[]`` array
    object — not bytes, and not a run of separate scalars.

    The kernels *index* their operands (``data[i]``, ``data + i``) across
    the whole allocation, and in C++17 pointer arithmetic is only defined
    **within one array object** ([expr.add]/4). Two plausible-looking
    models fail that requirement:

    * ``new unsigned char[n]`` plus a reinterpreting cast — begins the
      lifetime of an array of ``unsigned char`` and of no floating object
      at all. C++20's implicit object creation ([intro.object]/10, P0593)
      would rescue it; C++17 has no such rule.
    * raw storage plus a per-element placement-new loop — begins ``count``
      lifetimes, but as ``count`` *separate scalar objects*. A scalar that
      is not an array element behaves as a one-element array, so indexing
      past the first still leaves its array object, however contiguous the
      storage happens to be.

    The model that actually supports the indexing is an ordinary array
    new-expression, dispatched once by dtype. This guardrail asserts that
    *mechanism* semantically — whitespace-tolerant, not tied to variable
    names — and asserts the absence of both rejected models.
    """
    source = _cpp_code_only(_read("cpp/src/storage.cpp"))

    # 1. A genuine array new-expression, in both initialization forms:
    #    value-initialized for the zeroing creators (which is what makes
    #    every element positive zero) and default-initialized for the
    #    uninitialized ones (which writes nothing at all).
    assert re.search(r"new\s*\(\s*std::nothrow\s*\)\s*T\s*\[[^\]]+\]\s*\(\s*\)",
                     source), (
        "storage.cpp does not create a value-initialized T[] array"
    )
    assert re.search(r"new\s*\(\s*std::nothrow\s*\)\s*T\s*\[[^\]]+\]\s*(?!\s*\()",
                     source), (
        "storage.cpp does not create a default-initialized T[] array"
    )

    # 2. The array type is chosen by dtype, and the element type is
    #    exactly float or exactly double — nothing else is instantiated.
    assert re.search(r"create_typed_storage\s*<\s*float\s*>", source), (
        "storage.cpp never instantiates the allocation body for float"
    )
    assert re.search(r"create_typed_storage\s*<\s*double\s*>", source), (
        "storage.cpp never instantiates the allocation body for double"
    )

    # 3. Type-correct array ownership across the metadata allocation.
    assert re.search(r"std::unique_ptr\s*<\s*T\s*\[\s*\]\s*>", source), (
        "storage.cpp does not hold the array in a unique_ptr<T[]>, so a "
        "metadata-allocation failure would leak it or free it wrongly"
    )

    # 4. Destruction is a centralized, dtype-matched delete[] — one
    #    switch, both dtypes, each applied to its own element type.
    destroy = re.search(r"void\s+destroy_storage_data\s*\(.*?\n\}", source,
                        re.S)
    assert destroy, "storage.cpp has no central destroy_storage_data"
    body = destroy.group(0)
    assert re.search(r"delete\s*\[\s*\]\s*static_cast\s*<\s*float\s*\*\s*>",
                     body), "float32 storage is not released as float[]"
    assert re.search(r"delete\s*\[\s*\]\s*static_cast\s*<\s*double\s*\*\s*>",
                     body), "float64 storage is not released as double[]"
    # ...and nowhere else duplicates it.
    assert len(re.findall(r"delete\s*\[\s*\]", source)) == 2, (
        "a delete[] exists outside the central destroy_storage_data switch"
    )

    # 5. Neither rejected model is present.
    assert not re.search(r"new\s*(?:\([^)]*\)\s*)?unsigned\s+char\s*\[",
                         source), (
        "storage.cpp allocates a byte array; an unsigned-char array "
        "lifetime is not a float or double array lifetime under C++17"
    )
    assert not re.search(r"::\s*operator\s+new\s*\(", source), (
        "storage.cpp allocates a raw block again; separately constructed "
        "scalars are not one array object and cannot be indexed across"
    )
    assert not re.search(r"::\s*new\s*\(", source), (
        "storage.cpp placement-constructs elements again; that creates "
        "separate scalars rather than a single array object"
    )
    for banned in ("malloc", "calloc", "realloc", "std::memset", "memset("):
        assert banned not in source, (
            f"storage.cpp uses {banned}, which creates no array object"
        )

    # 6. Storage::data stays type-erased, and the dtype tag stays the sole
    #    authority that selects the array type.
    header = _cpp_code_only(_read("cpp/include/tf_internal.h"))
    struct = re.search(r"struct Storage\s*\{(.+?)\};", header, re.S)
    assert struct and re.search(r"\bvoid\*\s+data;", struct.group(1)), (
        "Storage::data is no longer a type-erased void*"
    )

    # 7. The absence of a per-element destruction pass is licensed by an
    #    assertion rather than by assumption.
    assert re.search(r"static_assert\s*\(\s*std::is_trivially_destructible",
                     source), (
        "storage.cpp does not assert that its element types are trivially "
        "destructible, which is what licenses having no destructor pass"
    )


def test_there_is_exactly_one_item_size_authority_in_the_cpp_tree():
    """No kernel, export, test helper, or build file may spell a storage
    width again: everything that needs one calls ``dtype_item_size``.

    The check is deliberately narrow — a *storage width* is
    ``sizeof(double)`` or ``sizeof(float)`` used as an element size — and
    the one definition site is allowed to spell it, because that is what
    being the authority means."""
    code_only = _cpp_code_only
    header = _read("cpp/include/tf_internal.h")
    authority = re.search(
        r"inline std::size_t dtype_item_size\(.*?\n\}", header, re.S)
    assert authority, "dtype_item_size is not defined in the shared header"
    for relative in sorted(
            p.relative_to(REPO_ROOT).as_posix()
            for p in (REPO_ROOT / "cpp" / "src").glob("*.cpp")):
        text = code_only(_read(relative))
        assert "sizeof(double)" not in text, relative
        assert "sizeof(float)" not in text, relative
    # ...and the header itself spells a width only inside the authority.
    # The compile-time platform assertions are excluded deliberately: they
    # assert that ``sizeof(double) == 8``, which is a statement about the
    # toolchain rather than a lookup of an element's width, and they are
    # what make the authority's answers trustworthy in the first place.
    outside = "\n".join(
        line for line in
        code_only(header.replace(authority.group(0), "")).splitlines()
        if "static_assert" not in line
    )
    assert "sizeof(double)" not in outside
    assert "sizeof(float)" not in outside


def test_phase_h_is_still_complete_and_untouched():
    design = (REPO_ROOT / "docs" / "native_cpu_performance_design.md")
    text = design.read_text(encoding="utf-8")
    assert re.search(r"Phase-H status:\s*complete", text, re.I)
    # Its one added symbol, and the boundary it left behind.
    exports = _source_exports()
    assert "tf_storage_create_uninitialized" in exports
    assert "tf_storage_create" in exports


# ---------------------------------------------------------------------------
# I0's change surface
# ---------------------------------------------------------------------------

# A file no Phase-I milestone touches. If ``git diff`` claims this one
# changed, git's view of the working tree cannot be trusted to answer a
# "what did this milestone change?" question — see ``_changed_since``.
_UNTOUCHED_SENTINEL = "LICENSE"


def _changed_since(base):
    """Every path that differs from ``base``, tracked or not.

    ``git diff`` reports tracked modifications only, so a *new* file
    dropped into src/ or cpp/ would slip past it. The untracked listing is
    what closes that, and it is the more likely mistake of the two.

    Skips rather than fails on three environment facts, none of which is a
    defect in the tree under test: git missing, the base commit missing,
    and — the one that actually bites here — a git whose line-ending
    configuration disagrees with how the working tree was checked out. A
    CRLF checkout read by a git with ``core.autocrlf`` unset reports
    *every* text file as modified, which is exactly what happens when this
    suite is run from WSL against a Windows checkout. The sentinel below
    detects that: if a file the phase never touches comes back "changed",
    the diff is measuring line endings rather than edits, and it can
    answer nothing.
    """
    try:
        ancestry = subprocess.run(
            ["git", "merge-base", "--is-ancestor", base, "HEAD"],
            cwd=REPO_ROOT, capture_output=True, text=True,
        )
    except (OSError, subprocess.SubprocessError):  # pragma: no cover
        pytest.skip("git is not available")
    if ancestry.returncode != 0:  # pragma: no cover
        pytest.skip(f"{base[:7]} is not an ancestor of HEAD")
    changed = subprocess.run(
        ["git", "diff", "--name-only", base, "--"],
        cwd=REPO_ROOT, capture_output=True, text=True, check=True,
    ).stdout.split()
    if _UNTOUCHED_SENTINEL in changed:  # pragma: no cover
        pytest.skip(
            f"git reports {_UNTOUCHED_SENTINEL} as changed, so it is "
            f"comparing line endings rather than edits in this environment"
        )
    changed += subprocess.run(
        ["git", "ls-files", "--others", "--exclude-standard"],
        cwd=REPO_ROOT, capture_output=True, text=True, check=True,
    ).stdout.split()
    return changed


# The example surface the phase is allowed to add, and the milestone that
# owns it. Everything else under ``examples/`` is off limits, which is what
# makes "the phase changed no existing example" checkable rather than
# promised: a modified float64 example would show up here as surely as an
# unplanned new one.
I9_ADDED_EXAMPLES = frozenset({"examples/native_float32_training.py"})

# Examples added by **later phases**, after Phase I closed at I11. Phase I's
# own example set is ``I9_ADDED_EXAMPLES`` and never grows; this second set
# exists only because the check below is a *cumulative* diff against the I0
# commit, which keeps seeing later work forever. Each entry names the
# milestone that shipped it, so a later phase's example is attributed rather
# than absorbed into Phase I's record — and every *other* path under
# ``examples/``, including an edit to an existing file, still fails.
POST_PHASE_I_EXAMPLES = frozenset({
    "examples/native_minibatch_training.py",          # Phase J, milestone J6
})
assert not (I9_ADDED_EXAMPLES & POST_PHASE_I_EXAMPLES)

# The runtime files I9 touches for **documentation only**. Each said
# "float64/cpu only" in a module docstring — accurate through I8, and wrong
# the moment the registry moved — so the sentence had to be corrected where
# a reader actually meets it. None of them gained a line of dtype logic,
# and ``test_the_i9_documentation_only_files_really_are_documentation_only``
# checks that rather than trusting this comment.
I9_DOCUMENTATION_ONLY = frozenset({
    "src/tensorforge/experimental/__init__.py",
    "src/tensorforge/experimental/native_relu.py",
    "src/tensorforge/experimental/native_flatten.py",
    "src/tensorforge/experimental/native_maxpool2d.py",
    "src/tensorforge/experimental/native_dropout.py",
    "src/tensorforge/experimental/native_cross_entropy_loss.py",
})

# The benchmark surface the phase is allowed to add, and the milestone that
# owns it. §22 assigns **all** benchmark work to I10, and I10 adds exactly
# one harness: a separate file, so that Phase H's instrument keeps its case
# inventory, its CLI, its tests, and — most importantly — the meaning of
# every number it ever published. Modifying an existing benchmark still
# fails here, which is what makes that separation checkable rather than
# merely intended.
I10_ADDED_BENCHMARKS = frozenset({"benchmarks/benchmark_native_dtype.py"})


def test_the_phase_changed_no_ci_or_dependency_file():
    """The phase's discipline, expressed as a cumulative diff assertion
    against the merged I0 commit rather than as a promise.

    The milestones legitimately change C++ sources, the CMake build, and
    the Python package — that is the phase. What they must *not* touch is
    the surface that would signal an unearned capability change or a
    changed environment: the CI workflow or the dependency set. Phase I
    adds no dependency and no build option, and neither of those has an
    exemption at any milestone.

    **Examples and benchmarks are the two qualified cases**, and both
    qualifications are enumerated rather than waived:

    - I1 through I8 added no example — those milestones were deliberately
      unable to write one, because no public constructor could produce a
      float32 tensor. **I9 adds exactly one**, the integrated exact-resume
      proof its scope calls for.
    - I0 through I9 changed no benchmark at all, because §22 assigns
      benchmark work to I10. **I10 adds exactly one**, and adds it as a new
      file rather than as a mode of the Phase-H harness.

    Any *other* change under ``examples/`` or ``benchmarks/`` — including
    an edit to an existing one — still fails here.

    The diff is **cumulative against the I0 commit**, so it keeps seeing
    later phases' work indefinitely. ``POST_PHASE_I_EXAMPLES`` names those
    additions explicitly, one entry per shipping milestone, rather than
    relaxing the rule to a prefix or a count: Phase I's own example set
    stays exactly I9's one, and an edit to any pre-existing example still
    fails.
    """
    forbidden = []
    for path in _changed_since(I0_COMMIT):
        if (path.startswith("examples/")
                and path not in I9_ADDED_EXAMPLES
                and path not in POST_PHASE_I_EXAMPLES):
            forbidden.append(path)
        if (path.startswith("benchmarks/")
                and path not in I10_ADDED_BENCHMARKS):
            forbidden.append(path)
        if path.startswith(".github/"):
            forbidden.append(path)
        if path in ("pyproject.toml", "uv.lock", "conftest.py"):
            forbidden.append(path)
    assert not forbidden, (
        f"the phase must not touch existing examples or benchmarks, CI, or "
        f"dependencies, but these changed: {forbidden}"
    )
    # ...and the files those exemptions were written for really exist, so
    # an exemption cannot outlive its subject.
    for allowed in (I9_ADDED_EXAMPLES | I10_ADDED_BENCHMARKS
                    | POST_PHASE_I_EXAMPLES):
        assert (REPO_ROOT / allowed).is_file(), allowed


# ---------------------------------------------------------------------------
# The Phase-H benchmark immutability guard, frozen rather than historical
#
# The question this guard answers — "did the phase modify a benchmark it
# inherited?" — was originally asked by reading each file's committed blob
# at ``I0_COMMIT``. That works locally and fails in CI, and the failure is
# instructive rather than incidental: ``actions/checkout@v4`` with no
# ``fetch-depth`` performs a **depth-1** clone, so the runner has the
# triggering commit and no history at all. ``git show I0:path`` then exits
# non-zero and ``git ls-tree I0`` exits 128, and a guard that cannot read
# its own baseline reports every inherited benchmark as newly added.
#
# The fix is to stop needing the object. The baseline is a property of one
# frozen commit, so it is *recorded* here as a content digest rather than
# re-derived from history on every run. The map below was produced from the
# genuine I0 commit during the I10 correction and verified three ways: each
# blob round-tripped through ``git hash-object`` to prove the bytes read are
# the bytes committed; each digest was recomputed from the current working
# tree; and ``git diff I0 HEAD -- benchmarks/`` was confirmed to list
# exactly the one file I10 adds.
#
# Why this guard can run everywhere while its two siblings still skip:
# ``_changed_since`` asks a **whole-tree** question ("what differs from
# I0?"), which has no answer at all without history, and whose CRLF
# sensitivity comes from git comparing a normalized index against a
# denormalized checkout. This guard asks a **fixed, enumerable** question
# about seven known files, so its baseline can be frozen and its one
# environment sensitivity — the checkout's line-ending style — is removed
# by normalizing both sides itself. Freezing the siblings would mean
# freezing a digest of the entire tree, which would have to be regenerated
# on every legitimate edit and would assert nothing. That asymmetry is why
# only this one is converted.

# Repository-relative path -> SHA-256 of the I0 committed content after
# CRLF-to-LF normalization. Derived from
# ``39d416aa51abc00976a771ecbc7a334545c25e59`` and immutable: a value here
# changes only if the *history* changes, never because a file did.
I0_BENCHMARK_DIGESTS = {
    "benchmarks/benchmark_native_autograd.py":
        "34f15260313dda83d1675858780ef135d57792c9a7726fa41653d5f90f15c26d",
    "benchmarks/benchmark_native_classification.py":
        "fac9c087358014af120bf5f8a227c908a84ce4f7dc3226cc5be60d77188e0f06",
    "benchmarks/benchmark_native_cnn.py":
        "853e43ad15ff3e4b0f9d5e7dab3277f1f606ea657f40cb92ab75f9333ab80e12",
    "benchmarks/benchmark_native_cpu_performance.py":
        "74bb8156166f556c72e5e15970edd7bde41bcfceb8a423f41b2703455c93cf72",
    "benchmarks/benchmark_native_dropout.py":
        "1389261c1e390391a785d1e7fd7fe671ee80864c4e1a96b19142e840caefc869",
    "benchmarks/benchmark_native_normalization.py":
        "5e70c4c67639dbd49ae821a2cbf7593d04951ea21d343e8a45f4aad236853e07",
    "benchmarks/cpp_backend.py":
        "317183a9239191bbd0a8b898de731a088b3fd9aaf8f4c44e3422a8d7764b922a",
}


def _normalized(data):
    """The repository's one normalization rule, and only it: CRLF to LF.

    Nothing else is forgiven — not whitespace, not encoding, not comments,
    not formatting, not a trailing newline — because every one of those is
    a real edit to a file this guard exists to freeze. The rule is needed
    at all because a Windows checkout stores CRLF for content git holds as
    LF, so an un-normalized digest would depend on the checkout rather than
    on the file."""
    return data.replace(b"\r\n", b"\n")


def _content_digest(data):
    return hashlib.sha256(_normalized(data)).hexdigest()


def _benchmark_digests(directory):
    """``{repository-relative path: digest}`` for every ``*.py`` under
    ``directory``, which is a real path so a negative control can point it
    at a temporary copy instead of the repository."""
    return {
        f"benchmarks/{path.name}": _content_digest(path.read_bytes())
        for path in sorted(Path(directory).glob("*.py"))
    }


def _classify_benchmarks(observed):
    """Split an observed digest map against the frozen baseline.

    Four independent findings, deliberately not collapsed into one: a
    modified inherited file, a deleted inherited file, an unexpected new
    file, and the approved I10 addition. Reporting them separately is what
    makes a failure say *which* invariant broke."""
    modified = sorted(
        relative for relative, digest in I0_BENCHMARK_DIGESTS.items()
        if relative in observed and observed[relative] != digest
    )
    deleted = sorted(
        relative for relative in I0_BENCHMARK_DIGESTS
        if relative not in observed
    )
    unexpected = sorted(
        relative for relative in observed
        if relative not in I0_BENCHMARK_DIGESTS
        and relative not in I10_ADDED_BENCHMARKS
    )
    approved = sorted(
        relative for relative in observed
        if relative in I10_ADDED_BENCHMARKS
    )
    return modified, deleted, unexpected, approved


def test_the_phase_h_benchmark_harness_is_untouched():
    """Phase H's harness is the instrument its ladder was chosen from and
    re-measured against, and its case inventory is pinned by test as "the
    H0 set". If I10 had added a dtype axis to it, every published Phase-H
    number would silently start meaning something else. So it is left
    exactly as it was, and I10's characterization lives in its own file.

    Answered entirely from **frozen content**: no git command, no
    historical object, no network, and nothing written. It therefore gives
    the same verdict on a full clone, on a shallow CI checkout, and on a
    CRLF working tree."""
    observed = _benchmark_digests(REPO_ROOT / "benchmarks")
    assert observed, "no benchmark harnesses found at all"

    modified, deleted, unexpected, approved = _classify_benchmarks(observed)

    assert modified == [], (
        f"the phase modified a benchmark it inherited from Phase H: "
        f"{modified}"
    )
    assert deleted == [], (
        f"the phase deleted a benchmark it inherited from Phase H: {deleted}"
    )
    assert unexpected == [], (
        f"the phase added a benchmark its contract does not permit: "
        f"{unexpected}"
    )
    # ...and the one addition the contract does permit is present, under
    # exactly the name the contract records.
    assert approved == sorted(I10_ADDED_BENCHMARKS), approved
    for relative in I10_ADDED_BENCHMARKS:
        assert (REPO_ROOT / relative).is_file(), relative
    # The separation stated the other way round as well: the new harness
    # is a **new** file, and the Phase-H one still exists beside it.
    assert (REPO_ROOT / "benchmarks"
            / "benchmark_native_cpu_performance.py").is_file()


def test_the_frozen_benchmark_baseline_covers_the_inherited_set():
    """The baseline is the whole inherited set, not a sample.

    Seven harnesses existed at I0 and every one has a digest, so a file
    quietly dropped from the map could not hide an edit."""
    assert len(I0_BENCHMARK_DIGESTS) == 7, sorted(I0_BENCHMARK_DIGESTS)
    for relative, digest in I0_BENCHMARK_DIGESTS.items():
        assert relative.startswith("benchmarks/") and relative.endswith(".py")
        assert len(digest) == 64 and set(digest) <= set("0123456789abcdef"), (
            relative
        )
    # The frozen set and the approved addition are disjoint: the I10
    # harness is deliberately *not* an inherited file, and freezing it
    # would pin a file the milestone is allowed to keep editing.
    assert not (set(I0_BENCHMARK_DIGESTS) & set(I10_ADDED_BENCHMARKS))
    # Every digest is distinct, so no two entries were pasted from one file.
    assert len(set(I0_BENCHMARK_DIGESTS.values())) == 7


def test_the_benchmark_guard_needs_no_git_history_at_runtime():
    """The CI-independence proof, asserted structurally over the guard's
    own source rather than promised in a comment.

    The failure this replaced was a depth-1 CI checkout: the runner has the
    triggering commit and no ancestors, so ``git show I0:path`` fails and
    ``git ls-tree I0`` exits 128. Nothing in the guard's call chain may
    reach for history again."""
    import ast
    import inspect
    import textwrap

    def executable_source(function):
        """``function``'s body with docstrings removed.

        Parsed rather than grepped, and for the same reason
        ``test_the_i9_documentation_only_files_really_are_documentation_only``
        parses: the prose here legitimately *discusses* ``git show`` and
        ``ls-tree``, so a raw substring scan would trip on the very
        sentences that explain why they are absent."""
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
        test_the_phase_h_benchmark_harness_is_untouched,
        _benchmark_digests,
        _classify_benchmarks,
        _content_digest,
        _normalized,
    ))
    for banned in ("git", "subprocess", "ls-tree", "rev-parse",
                   "I0_COMMIT", "_changed_since", "urlopen", "requests",
                   "socket", "open(", "write"):
        assert banned not in chain, (
            f"the benchmark guard reached for {banned!r}; it must answer "
            f"from frozen content alone"
        )
    # The behavioural half: with git made unreachable the guard still
    # passes. Proved non-vacuous first — a run in which git happened to
    # stay reachable would assert nothing.
    import os

    saved = os.environ.get("PATH")
    try:
        os.environ["PATH"] = ""
        try:
            probe = subprocess.run(["git", "--version"],
                                   capture_output=True, env={"PATH": ""})
            git_reachable = probe.returncode == 0
        except OSError:
            git_reachable = False
        assert not git_reachable, (
            "git stayed reachable, so this half of the proof is vacuous"
        )
        test_the_phase_h_benchmark_harness_is_untouched()
    finally:
        if saved is None:                       # pragma: no cover
            os.environ.pop("PATH", None)
        else:
            os.environ["PATH"] = saved

    # ...and for contrast, the *old* approach really would have failed
    # here: reading a blob out of the object database is exactly what a
    # depth-1 CI checkout cannot do. Simulated with an object that is
    # certainly absent, which is the same condition the runner is in.
    missing = subprocess.run(
        ["git", "show", f"{'0' * 40}:benchmarks/cpp_backend.py"],
        cwd=REPO_ROOT, capture_output=True)
    assert missing.returncode != 0, (
        "the simulated missing object resolved, so the contrast is vacuous"
    )
    absent_tree = subprocess.run(
        ["git", "ls-tree", "--name-only", "0" * 40, "benchmarks/"],
        cwd=REPO_ROOT, capture_output=True)
    assert absent_tree.returncode != 0


def test_the_frozen_benchmark_guard_detects_every_kind_of_drift(tmp_path):
    """The negative controls, all on **temporary** bytes and directories —
    no repository file is read for mutation, written, or restored.

    A guard that cannot fail proves nothing, so each of the eight failure
    modes it claims to catch is produced deliberately and shown to be
    caught."""
    def stage(sources):
        """A throwaway benchmarks/ directory with the given contents."""
        directory = tmp_path / f"bench{len(list(tmp_path.iterdir()))}"
        directory.mkdir()
        for name, data in sources.items():
            (directory / name).write_bytes(data)
        return directory

    # A faithful stand-in for the real inherited set: bytes whose digests
    # really are the frozen ones, reconstructed by reading the live files
    # (read-only) so the control exercises the true baseline.
    genuine = {}
    for relative in I0_BENCHMARK_DIGESTS:
        name = relative.split("/", 1)[1]
        genuine[name] = (REPO_ROOT / relative).read_bytes()
    genuine["benchmark_native_dtype.py"] = (
        REPO_ROOT / "benchmarks" / "benchmark_native_dtype.py").read_bytes()

    # 0. The positive control: unmodified, everything clean.
    modified, deleted, unexpected, approved = _classify_benchmarks(
        _benchmark_digests(stage(genuine)))
    assert (modified, deleted, unexpected) == ([], [], [])
    assert approved == sorted(I10_ADDED_BENCHMARKS)

    victim = "benchmark_native_cpu_performance.py"
    target = f"benchmarks/{victim}"

    # 1. One changed byte.
    one_byte = dict(genuine)
    body = bytearray(one_byte[victim])
    body[len(body) // 2] ^= 0x01
    one_byte[victim] = bytes(body)
    modified, _, _, _ = _classify_benchmarks(
        _benchmark_digests(stage(one_byte)))
    assert modified == [target], "a single changed byte slipped through"

    # 2. One added line.
    added_line = dict(genuine)
    added_line[victim] = added_line[victim] + b"\n# one added line\n"
    modified, _, _, _ = _classify_benchmarks(
        _benchmark_digests(stage(added_line)))
    assert modified == [target], "an added line slipped through"

    # 3. One removed line.
    removed_line = dict(genuine)
    lines = removed_line[victim].split(b"\n")
    removed_line[victim] = b"\n".join(lines[:5] + lines[6:])
    modified, _, _, _ = _classify_benchmarks(
        _benchmark_digests(stage(removed_line)))
    assert modified == [target], "a removed line slipped through"

    # 4. CRLF-only conversion is **equal** — the one thing forgiven, and
    #    the reason this guard survives a Windows checkout.
    # Normalize first: the live checkout may already be CRLF (it is on
    # Windows), and a naive replace would produce CRCRLF and measure
    # nothing.
    windows = {name: _normalized(data).replace(b"\n", b"\r\n")
               for name, data in genuine.items()}
    assert b"\r\r\n" not in windows[victim], "the control corrupted its input"
    assert b"\r\n" in windows[victim], "the control produced no CRLF at all"
    modified, deleted, unexpected, approved = _classify_benchmarks(
        _benchmark_digests(stage(windows)))
    assert (modified, deleted, unexpected) == ([], [], []), (
        "a pure CRLF checkout was mistaken for an edit"
    )
    assert approved == sorted(I10_ADDED_BENCHMARKS)
    # ...and CRLF plus a real edit is still caught, so the normalization
    # does not swallow the edit along with the line endings.
    windows_edited = dict(windows)
    windows_edited[victim] += b"\r\n# edited\r\n"
    modified, _, _, _ = _classify_benchmarks(
        _benchmark_digests(stage(windows_edited)))
    assert modified == [target]

    # 5. An inherited benchmark deleted.
    without = {name: data for name, data in genuine.items() if name != victim}
    _, deleted, _, _ = _classify_benchmarks(
        _benchmark_digests(stage(without)))
    assert deleted == [target], "a deleted inherited benchmark slipped through"

    # 6. An unexpected additional benchmark.
    intruder = dict(genuine)
    intruder["benchmark_surprise.py"] = b"# not approved\n"
    _, _, unexpected, _ = _classify_benchmarks(
        _benchmark_digests(stage(intruder)))
    assert unexpected == ["benchmarks/benchmark_surprise.py"]

    # 7. The approved I10 benchmark missing.
    absent = {name: data for name, data in genuine.items()
              if name != "benchmark_native_dtype.py"}
    _, _, _, approved = _classify_benchmarks(_benchmark_digests(stage(absent)))
    assert approved == [], "a missing I10 harness read as present"

    # 8. The approved benchmark replaced by another filename.
    renamed = {name: data for name, data in genuine.items()
               if name != "benchmark_native_dtype.py"}
    renamed["benchmark_native_dtypes.py"] = genuine[
        "benchmark_native_dtype.py"]
    _, _, unexpected, approved = _classify_benchmarks(
        _benchmark_digests(stage(renamed)))
    assert approved == []
    assert unexpected == ["benchmarks/benchmark_native_dtypes.py"]

    # 9. A wrong frozen digest must fail too, so the map itself is load
    #    bearing rather than decorative.
    poisoned = dict(I0_BENCHMARK_DIGESTS)
    poisoned[target] = "0" * 64
    observed = _benchmark_digests(stage(genuine))
    assert observed[target] != poisoned[target]

    # Nothing above touched the repository.
    assert _benchmark_digests(REPO_ROOT / "benchmarks") == {
        **I0_BENCHMARK_DIGESTS,
        "benchmarks/benchmark_native_dtype.py": _content_digest(
            (REPO_ROOT / "benchmarks"
             / "benchmark_native_dtype.py").read_bytes()),
    }


def test_the_two_history_reading_guards_are_deliberately_left_alone():
    """The asymmetry, recorded so it is not mistaken for an oversight.

    ``_changed_since`` asks what differs across the **whole tree** since
    I0. That question has no answer without history, and its CRLF
    sensitivity is inherent — git is comparing a normalized index against a
    denormalized checkout, which is a fact about the environment rather
    than about the tree. So those two guards skip, loudly, with a reason.

    The benchmark guard asks a fixed question about an enumerable set, so
    its baseline can be frozen and its only environment sensitivity removed
    by normalizing both sides. Converting the siblings the same way would
    mean freezing a digest of every tracked file, which would need
    regenerating on every legitimate edit and would therefore assert
    nothing at all."""
    import inspect

    for guard in (test_the_phase_changed_no_ci_or_dependency_file,
                  test_the_phase_touched_only_the_python_modules_its_scope_names):
        assert "_changed_since" in inspect.getsource(guard), guard.__name__
    # ...and _changed_since still owns the documented skip, so the two
    # guards degrade honestly rather than passing vacuously.
    source = inspect.getsource(_changed_since)
    assert "pytest.skip" in source
    assert "_UNTOUCHED_SENTINEL" in source
    assert _UNTOUCHED_SENTINEL == "LICENSE"
    # The benchmark guard, by contrast, contains no skip at all.
    assert "skip" not in inspect.getsource(
        test_the_phase_h_benchmark_harness_is_untouched)


def test_the_i9_documentation_only_files_really_are_documentation_only():
    """The five runtime files I9 touches for their docstrings, held to the
    claim rather than trusted with it.

    Each of them said "float64/cpu only" where a reader meets the module,
    which was accurate through I8 and wrong the moment the registry moved.
    Correcting a sentence is legitimate; smuggling dtype behavior in beside
    it is not — and "documentation only" is exactly the kind of claim that
    is easy to make and easy to break. So the check is structural: none of
    these files may name a dtype in **code**.

    Parsed rather than grepped, and the docstrings are removed first, so
    the prose is free to discuss float32 at any length while the executable
    body stays untouched."""
    import ast

    for relative in sorted(I9_DOCUMENTATION_ONLY):
        source = (REPO_ROOT / relative).read_text(encoding="utf-8")
        tree = ast.parse(source)
        # Drop every docstring, then look at what is left.
        for node in ast.walk(tree):
            if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef,
                                 ast.AsyncFunctionDef)) and node.body:
                first = node.body[0]
                if (isinstance(first, ast.Expr)
                        and isinstance(first.value, ast.Constant)
                        and isinstance(first.value.value, str)):
                    node.body = node.body[1:] or [ast.Pass()]
        code = ast.unparse(tree)
        for banned in ("float32", "float64", "dtype", "normalize_dtype",
                       "_typed", "SUPPORTED_DTYPES"):
            assert banned not in code, (relative, banned)


def test_the_phase_touched_only_the_python_modules_its_scope_names():
    """Within ``src/``, the phase is confined to the files its milestones
    name — and the set grows only when a milestone says it does.

    Through I3 this was a single file, the ctypes layer. I4 added
    ``native_tensor.py``, because core autograd is I4 work. **I7 adds the
    state-owning module surface**: the six constructors of design §12.1, the
    one shared dtype validator they route through, and the two places where
    dtype-aware state meets an older contract — Dropout's Python wrapper
    (already in ``cpp.py``) and the version-2 checkpoint boundary, which must
    *refuse* float32 rather than silently write an unreadable archive.

    ``native_sequential.py`` is in the set for documentation only: containers
    take no dtype and enforce none, and saying so is part of the milestone.

    **I8 adds the optimizer and checkpoint surface** it always owned: the
    two optimizer files, and ``native_checkpoint.py`` again — this time to
    *carry* float32 under format version 3 rather than to refuse it.
    ``native_optimizer_state.py`` stays out, because the in-memory state
    schema does not move (design §15, §16.2): float32 metadata becomes
    reachable through it without a single line changing.

    **I9 adds the public registry** — which lives in ``cpp.py``, already in
    the set — plus five files it touches for **documentation only**. Those
    five are listed separately below and held to a stricter rule than the
    rest: a test asserts they contain no dtype logic at all, so "docstring
    only" is checked rather than promised.
    """
    allowed = {
        "src/tensorforge/backends/cpp.py",
        "src/tensorforge/experimental/native_tensor.py",
        # I7: the state-owning constructor surface.
        "src/tensorforge/experimental/_native_dtype.py",
        "src/tensorforge/experimental/native_parameter.py",
        "src/tensorforge/experimental/native_linear.py",
        "src/tensorforge/experimental/native_conv2d.py",
        "src/tensorforge/experimental/native_layernorm.py",
        "src/tensorforge/experimental/native_batchnorm.py",
        "src/tensorforge/experimental/native_sequential.py",
        "src/tensorforge/experimental/native_checkpoint.py",
        # I8: the state-bearing dtype stack.
        "src/tensorforge/experimental/native_sgd.py",
        "src/tensorforge/experimental/native_adam.py",
    } | I9_DOCUMENTATION_ONLY
    # The diff runs from I0 to HEAD, so it necessarily also sees files a
    # *later* phase added. Those are named explicitly and excluded rather than
    # the Phase-I claim being loosened: what stays asserted is exactly "Phase I
    # touched only the files its scope names", and a new *Phase-I* file would
    # still fail here. ``native_dataset.py`` is Phase J milestone J1's, and
    # that it is no part of Phase I is asserted by tests/test_native_phase_j.py.
    LATER_PHASE_FILES = {
        "src/tensorforge/experimental/native_dataset.py",   # Phase J, J1
        "src/tensorforge/experimental/native_sampler.py",   # Phase J, J2
        # Phase J, J2 — the private permutation derivation. It owns no
        # dtype at all, which is why it is a *later phase's* file here and
        # nowhere in the dtype inventories above.
        "src/tensorforge/experimental/_native_permutation.py",
        # Phase J, J3 — the mini-batch loader. It owns no dtype either:
        # the batches it delivers carry the *dataset's*, so it takes no
        # ``dtype`` argument and appears in no dtype inventory above.
        "src/tensorforge/experimental/native_data_loader.py",
    }
    changed = [path for path in _changed_since(I0_COMMIT)
               if path.startswith("src/") and path not in LATER_PHASE_FILES]
    unexpected = [path for path in changed if path not in allowed]
    assert unexpected == [], (
        f"the phase changed more of the Python package than its scope "
        f"names: {unexpected}"
    )
    # Stated the other way round as well, so a future milestone that adds a
    # file cannot satisfy the rule above by accident: the in-memory
    # optimizer state schema is untouched, by name, and so is the stable
    # line.
    for forbidden in ("native_optimizer_state",
                      "tensorforge/nn/", "tensorforge/optim/",
                      "tensorforge/tensor.py", "tensorforge/data.py"):
        assert not any(forbidden in path for path in changed), forbidden
    from tensorforge.experimental import native_optimizer_state
    assert native_optimizer_state.FORMAT_VERSION == 1
    # ...and the checkpoint format moved exactly to its I8 value.
    from tensorforge.experimental import native_checkpoint
    assert native_checkpoint._FORMAT_VERSION == I8_CHECKPOINT_VERSION
    assert (native_checkpoint._SUPPORTED_FORMAT_VERSIONS
            == I8_CHECKPOINT_VERSIONS)


def test_phase_i_introduced_no_prohibited_external_reference():
    """The repository self-containment guardrail, re-asserted over the
    files these milestones actually add.

    The terms are assembled from fragments so this module never spells one
    literally, exactly as the repository-wide guardrail does.
    """
    name = "dae" + "dalus"
    owner = "johnson" + "kayati"
    terms = (name, owner, name + "-ml", "github.com/" + owner)
    for relative in ("docs/native_dtype_float32_design.md",
                     "tests/test_native_phase_i.py",
                     "cpp/tests/test_dtype_storage.cpp",
                     "cpp/include/tf_internal.h",
                     "cpp/src/storage.cpp"):
        lowered = _read(relative).lower()
        for term in terms:
            assert term not in lowered, f"{relative} names {term!r}"


def test_the_stable_framework_still_does_not_load_the_native_backend():
    """Phase I touches the native line only. The isolation the whole
    project rests on must still hold."""
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
    assert cpp.backend_info()["stable_framework_integration"] is False


# ---------------------------------------------------------------------------
# I1: the dtype model and dtype-tagged storage, as running code
#
# Everything below drives the **live** library. float32 is not a supported
# TensorForge dtype and cannot be reached through any public constructor,
# so these tests reach the typed creators the only way anything can before
# I9: through the private library handle, which is existing private
# binding infrastructure rather than a new test-only API.
# ---------------------------------------------------------------------------

needs_native = pytest.mark.skipif(
    not cpp.is_available(), reason="the native library is not built")


def _create_typed(size, code, zero_initialize=True):
    """Allocate raw native storage at an arbitrary dtype code.

    Deliberately not a helper on ``NativeStorage``: I1 adds **no** public
    float32 construction path, and a convenience wrapper on the wrapper
    class would be exactly that. The caller destroys the handle.
    """
    library = cpp._require_library()
    create = (library.tf_storage_create_typed if zero_initialize
              else library.tf_storage_create_uninitialized_typed)
    return create(int(size), int(code))


def test_the_python_dtype_tables_agree_with_the_frozen_abi_codes():
    """One authority per side of the boundary, agreeing by construction
    because the codes are the same integers."""
    assert cpp._DTYPE_CODES == {"float64": DTYPE_CODE_FLOAT64,
                                "float32": DTYPE_CODE_FLOAT32}
    assert cpp._DTYPE_ITEM_SIZES == {"float64": 8, "float32": 4}
    assert cpp._DTYPE_NUMPY["float64"] is np.float64
    assert cpp._DTYPE_NUMPY["float32"] is np.float32
    # The three tables describe the same set of dtypes, so none can gain a
    # value the others do not know about.
    assert (set(cpp._DTYPE_CODES) == set(cpp._DTYPE_ITEM_SIZES)
            == set(cpp._DTYPE_NUMPY))
    # ...and each item size is NumPy's own, so the width cannot drift.
    for name, size in cpp._DTYPE_ITEM_SIZES.items():
        assert np.dtype(cpp._DTYPE_NUMPY[name]).itemsize == size


def test_the_dtype_tables_stayed_private_when_the_promise_caught_up():
    """The internal capability legitimately ran ahead of the public one
    between I1 and I8 — that was the rollout rule — and **I9 is where the
    promise caught up**. The tables stay private either way.

    That is the durable half of the original claim. "Wider" was a fact
    about eight milestones, and it stopped being one when the registry
    moved; "private, and never a public dtype object" is a fact about the
    design and does not lapse. The two sets agreeing is asserted here
    rather than assumed, because a representation table that could hold a
    dtype the registry does not is exactly the drift this guards."""
    assert "float32" in cpp._DTYPE_CODES
    assert set(cpp._DTYPE_CODES) == set(cpp.SUPPORTED_DTYPES)
    assert "float32" not in cpp.UNSUPPORTED
    for name in ("_DTYPE_CODES", "_DTYPE_ITEM_SIZES", "_DTYPE_NUMPY"):
        assert name.startswith("_"), name
    # No public dtype object was introduced anywhere on the module.
    assert not [name for name in dir(cpp)
                if not name.startswith("_")
                and name.lower() in ("float32", "float64", "dtype",
                                     "nativedtype")]


@needs_native
def test_the_typed_creators_have_the_declared_abi_signature():
    library = cpp._require_library()
    for name in PLANNED_NEW_EXPORTS:
        function = getattr(library, name)
        assert function.argtypes == [ctypes.c_int64, ctypes.c_int32], name
        assert function.restype is ctypes.c_void_p, name
        # They report failure the way the untyped creators do, so they
        # take the same hook rather than a second convention.
        assert function.errcheck is not None, name
        assert name in cpp._CHECKED_KERNELS, name
    # The untyped pair is unchanged and still exported.
    for name in ("tf_storage_create", "tf_storage_create_uninitialized"):
        function = getattr(library, name)
        assert function.argtypes == [ctypes.c_int64], name
        assert function.restype is ctypes.c_void_p, name


@needs_native
def test_typed_storage_can_be_created_and_destroyed_at_both_dtypes():
    library = cpp._require_library()
    for code in (DTYPE_CODE_FLOAT64, DTYPE_CODE_FLOAT32):
        for zero in (True, False):
            handle = _create_typed(64, code, zero)
            assert handle, (code, zero)
            try:
                # The size is a **logical element count** at both widths:
                # a float32 storage of 64 elements reports 64, not 256.
                assert library.tf_storage_size(handle) == 64
            finally:
                library.tf_storage_destroy(handle)


@needs_native
def test_an_unknown_dtype_code_raises_value_error_and_allocates_nothing():
    for code in (-1, 2, 3, 99, 2 ** 31 - 1):
        for zero in (True, False):
            with pytest.raises(ValueError, match="dtype"):
                _create_typed(16, code, zero)


@needs_native
def test_a_non_positive_size_raises_value_error_at_both_dtypes():
    for code in (DTYPE_CODE_FLOAT64, DTYPE_CODE_FLOAT32):
        for size in (0, -1, -4096):
            with pytest.raises(ValueError, match="positive"):
                _create_typed(size, code)


@needs_native
def test_a_byte_count_overflow_raises_value_error_rather_than_memory_error():
    """The distinction matters: an unrepresentable byte count is an
    invalid *request*, rejected by arithmetic before any allocator is
    asked, not an allocation that failed."""
    for name, code in (("float64", DTYPE_CODE_FLOAT64),
                       ("float32", DTYPE_CODE_FLOAT32)):
        overflowing = (2 ** 63 - 1) // cpp._DTYPE_ITEM_SIZES[name] + 1
        with pytest.raises(ValueError) as caught:
            _create_typed(overflowing, code)
        assert name in str(caught.value)


@needs_native
def test_an_injected_allocation_failure_raises_memory_error_at_both_dtypes():
    library = cpp._require_library()
    for code in (DTYPE_CODE_FLOAT64, DTYPE_CODE_FLOAT32):
        for zero in (True, False):
            cpp._arm_alloc_failure(1)
            try:
                with pytest.raises(MemoryError):
                    _create_typed(32, code, zero)
            finally:
                cpp._arm_alloc_failure(0)
            # ...and the next real creation succeeds with a clear slot,
            # so a failure cannot contaminate a later call.
            handle = _create_typed(32, code, zero)
            assert handle
            library.tf_storage_destroy(handle)


@needs_native
def test_a_float32_handle_is_rejected_by_operations_that_are_still_float64():
    """I1 made float32 allocatable and generalized no operation; I2
    generalized exactly the transfer, materialization, and identity-copy
    boundaries; I3 generalized the elementwise and unary family. Everything
    else must still reject a float32 operand rather than walk it as float64
    — which would overrun the buffer by exactly a factor of two.

    The C++ CTest proves this across every compute translation unit; this
    is the Python-visible half, through the errcheck hook.
    """
    library = cpp._require_library()
    f32 = _create_typed(64, DTYPE_CODE_FLOAT32)
    assert f32
    destination = cpp.NativeStorage(64)
    try:
        handle = destination._require_open()
        shape = (ctypes.c_int64 * 2)(8, 8)
        strides = (ctypes.c_int64 * 2)(8, 1)
        axes = (ctypes.c_int64 * 2)(0, 1)
        # Still float64-only, and each is a different translation unit:
        # reduction, matmul, classification, pooling, random.
        with pytest.raises(ValueError, match="float32"):
            library.tf_core_sum(f32, handle, shape, strides, axes, 0, 2)
        with pytest.raises(ValueError, match="float32"):
            library.tf_core_matmul(f32, handle, handle, 8, 8, 8, 8, 1, 8, 1,
                                   0, 0)
        with pytest.raises(ValueError, match="float32"):
            library.tf_core_softmax_forward(f32, 0, handle, 8, 8, 1)
        # ...and in the destination position, which is the direction that
        # would corrupt memory rather than merely misread it.
        with pytest.raises(ValueError, match="float32"):
            library.tf_core_sum(handle, f32, shape, strides, axes, 0, 2)
        # ``tf_core_contiguous_copy`` is dtype-general from I2, and the
        # elementwise family from I3, so none of them is rejecting "float32"
        # any more — each rejects a **mixed** pair, which is the stronger and
        # permanent rule. The assertions advance to that truth rather than
        # being deleted.
        with pytest.raises(ValueError, match="same dtype"):
            library.tf_core_contiguous_copy(f32, handle, shape, strides, 0, 2)
        with pytest.raises(ValueError, match="same dtype"):
            library.tf_core_relu(f32, handle, shape, strides, 0, 2)
        with pytest.raises(ValueError, match="same dtype"):
            library.tf_core_add(f32, handle, handle, shape, strides,
                                strides, 0, 0, 2)
        with pytest.raises(ValueError, match="same dtype"):
            library.tf_core_relu(handle, f32, shape, strides, 0, 2)
        # The float64 destination was never touched by any of them.
        assert np.array_equal(destination.to_numpy(), np.zeros(64))
    finally:
        destination.close()
        library.tf_storage_destroy(f32)


@needs_native
def test_the_public_wrapper_allocates_through_the_typed_path_at_both_widths():
    """``NativeStorage`` routes through the typed creators uniformly, and
    the dtype it passes is the one it validated — so the Python tag and
    the C++ tag cannot disagree. Every observable behavior of the float64
    default is unchanged, which is I1's claim and stays true; since I9 the
    same one path also serves float32, which is the point of there being
    one path."""
    for dtype, numpy_dtype in (("float64", np.float64),
                               ("float32", np.float32)):
        storage = cpp.NativeStorage(8, dtype=dtype)
        try:
            assert storage.dtype == dtype
            assert storage.size == 8          # elements, never bytes
            # The zero-initializing default did not move.
            assert np.array_equal(storage.to_numpy(),
                                  np.zeros(8, dtype=numpy_dtype))
            assert storage.to_numpy().dtype == numpy_dtype
        finally:
            storage.close()
    # ...and an unrepresentable dtype is still refused at the public
    # boundary, before any native call is made.
    with pytest.raises(ValueError):
        cpp.NativeStorage(8, dtype="float16")


@needs_native
def test_the_untyped_creators_still_work_and_still_mean_float64():
    """They did not go away because their primary caller moved: they are
    still exported, still callable, and still float64."""
    library = cpp._require_library()
    for name in ("tf_storage_create", "tf_storage_create_uninitialized"):
        handle = getattr(library, name)(16)
        assert handle, name
        try:
            assert library.tf_storage_size(handle) == 16
            # A float64 primitive accepts it, which is the observable
            # proof of its dtype tag from Python.
            library.tf_storage_fill(handle, 2.5)
            out = np.empty(16, dtype=np.float64)
            # The host position is a ``void*`` from I2, so the buffer goes
            # through the same per-dtype checked binding every production
            # transfer uses.
            library.tf_storage_copy_to(handle, cpp._host_pointer(out, "float64"))
            assert np.array_equal(out, np.full(16, 2.5))
        finally:
            library.tf_storage_destroy(handle)
        # ...and their pre-existing rejection is byte-for-byte unchanged.
        with pytest.raises(ValueError, match="storage size must be positive"):
            getattr(library, name)(0)


@needs_native
def test_every_public_construction_path_reaches_float32_since_i9():
    """The rollout rule's other side, checked behaviorally across every
    public constructor rather than trusted from the registry.

    Through I8 this listed the same constructors and required each to
    **raise**; I9 is the milestone that made them work, so it now requires
    each to produce a genuinely float32 tensor. The list is deliberately
    unchanged so the two versions cover exactly the same surface."""
    values = np.zeros((2, 2), dtype=np.float64)
    for build in (
        lambda: cpp.NativeStorage(4, dtype="float32"),
        lambda: cpp.NativeStorage.from_array(values, dtype="float32"),
        lambda: cpp.NativeTensorCore.zeros((2, 2), dtype="float32"),
        lambda: cpp.NativeTensorCore.from_array(values, dtype="float32"),
    ):
        built = build()
        try:
            assert built.dtype == "float32"
            assert built.to_numpy().dtype == np.float32
        finally:
            built.close()
    assert cpp.normalize_dtype("float32") == "float32"


# ---------------------------------------------------------------------------
# I2: typed array transfer, views, and materialization, as running code
#
# Everything below drives the **live** library at both dtypes. float32 is
# still not a supported TensorForge dtype and no public constructor can
# produce one, so these reach it through the private typed constructors
# ``NativeStorage._typed`` / ``._typed_from_array`` and
# ``NativeTensorCore._typed_from_array`` — the private/typed entry points
# the rollout rule (design §27.2) requires an intermediate milestone to test
# through while the public registry stays exactly where it is.
#
# Comparison is by **raw IEEE-754 bit pattern**, never by value and never by
# tolerance. The three cases the transfer contract exists for — negative
# zero, NaN payloads, and signalling NaNs — are exactly the ones ``==``
# cannot see or silently launders.
# ---------------------------------------------------------------------------

# The seventeen representative classes, at each width, built from bits and
# never from arithmetic. Written out rather than derived so a change to
# either list is a visible diff.
F64_BIT_PATTERNS = (
    0x0000000000000000,  # +0.0
    0x8000000000000000,  # -0.0                     <- arithmetic normalizes
    0x3FF0000000000000,  # 1.0
    0xBFF0000000000000,  # -1.0
    0x7FF0000000000000,  # +inf
    0xFFF0000000000000,  # -inf
    0x7FF8000000000001,  # quiet NaN, payload 1
    0x7FF800000000000A,  # quiet NaN, payload A
    0xFFF8000000000001,  # negative quiet NaN
    0x7FF0000000000001,  # signalling NaN           <- arithmetic quiets
    0xFFF0000000000001,  # negative signalling NaN
    0x0000000000000001,  # smallest positive subnormal
    0x8000000000000001,  # -smallest subnormal
    0x000FFFFFFFFFFFFF,  # largest subnormal
    0x0010000000000000,  # smallest positive normal
    0x7FEFFFFFFFFFFFFF,  # largest finite
    0xFFEFFFFFFFFFFFFF,  # -largest finite
)
F32_BIT_PATTERNS = (
    0x00000000,  # +0.0
    0x80000000,  # -0.0                     <- arithmetic normalizes
    0x3F800000,  # 1.0
    0xBF800000,  # -1.0
    0x7F800000,  # +inf
    0xFF800000,  # -inf
    0x7FC00001,  # quiet NaN, payload 1
    0x7FC0000A,  # quiet NaN, payload A
    0xFFC00001,  # negative quiet NaN
    0x7F800001,  # signalling NaN           <- arithmetic quiets
    0xFF800001,  # negative signalling NaN
    0x00000001,  # smallest positive subnormal
    0x80000001,  # -smallest subnormal
    0x007FFFFF,  # largest subnormal
    0x00800000,  # smallest positive normal
    0x7F7FFFFF,  # largest finite
    0xFF7FFFFF,  # -largest finite
)

_DTYPE_BITS = {
    "float64": (F64_BIT_PATTERNS, np.uint64, np.float64),
    "float32": (F32_BIT_PATTERNS, np.uint32, np.float32),
}
BOTH_DTYPES = ("float64", "float32")


def _patterned(dtype, count):
    """``count`` host values of ``dtype`` drawn from that width's sweep,
    cycling.

    Built by reinterpreting an integer array, so no floating operation ever
    touches them — a signalling NaN produced arithmetically would already be
    quiet before the test began, and the test would prove nothing.
    """
    patterns, unsigned, floating = _DTYPE_BITS[dtype]
    raw = np.array([patterns[i % len(patterns)] for i in range(count)],
                   dtype=unsigned)
    return raw.view(floating)


def _bits(array, dtype):
    """``array``'s raw object representations, as unsigned integers."""
    _, unsigned, floating = _DTYPE_BITS[dtype]
    assert array.dtype == np.dtype(floating), (array.dtype, dtype)
    return np.ascontiguousarray(array).view(unsigned)


def _same_bits(got, expected, dtype):
    return np.array_equal(_bits(got, dtype), _bits(expected, dtype))


@needs_native
@pytest.mark.parametrize("dtype", BOTH_DTYPES)
def test_a_host_round_trip_preserves_every_bit_at_both_dtypes(dtype):
    """The core I2 claim. A transfer performs no arithmetic, so it has no
    operand roles to choose between and no rounding to do: every object
    representation reproduces exactly, including both signed zeros, both
    infinities, subnormals, quiet NaN payloads, and signalling NaNs."""
    for count in (1, 2, 17, 18, 1000):
        source = _patterned(dtype, count)
        storage = cpp.NativeStorage._typed(count, dtype)
        try:
            storage.copy_from(source)
            assert storage.dtype == dtype
            assert storage.size == count      # elements, never bytes
            out = storage.to_numpy()
            # Egress reproduces the storage dtype exactly — never widened.
            assert out.dtype == np.dtype(_DTYPE_BITS[dtype][2])
            assert out.shape == (count,)
            assert _same_bits(out, source, dtype), (dtype, count)
            # First, middle, and last named explicitly, so a failure
            # localizes rather than merely reporting "something differs".
            for index in (0, count // 2, count - 1):
                assert (_bits(out, dtype)[index]
                        == _bits(source, dtype)[index]), (dtype, count, index)
            # Repeated round trips erode nothing.
            for _ in range(3):
                storage.copy_from(out)
                out = storage.to_numpy()
            assert _same_bits(out, source, dtype), (dtype, count, "repeated")
        finally:
            storage.close()


@needs_native
@pytest.mark.parametrize("dtype", BOTH_DTYPES)
def test_non_contiguous_host_input_is_made_contiguous_without_conversion(dtype):
    """``copy_from`` may make a host array contiguous — that is the
    documented host-to-native boundary — but it must not change what the
    elements *are* while doing so."""
    floating = _DTYPE_BITS[dtype][2]
    wide = _patterned(dtype, 34).reshape(17, 2)
    strided = wide[:, 0]                      # stride 2, non-contiguous
    assert not strided.flags.c_contiguous
    storage = cpp.NativeStorage._typed(17, dtype)
    try:
        storage.copy_from(strided)
        out = storage.to_numpy()
        assert out.dtype == np.dtype(floating)
        assert _same_bits(out, np.ascontiguousarray(strided), dtype)
    finally:
        storage.close()


@needs_native
@pytest.mark.parametrize("dtype", BOTH_DTYPES)
def test_the_raw_transfer_boundary_rejects_a_wrong_dtype_host_buffer(dtype):
    """No implicit conversion happens at the C ABI. The wrapper converts,
    once, on the way in; the boundary itself checks and refuses."""
    other = "float32" if dtype == "float64" else "float64"
    storage = cpp.NativeStorage._typed(8, dtype)
    try:
        handle = storage._require_open()
        wrong = np.zeros(8, dtype=_DTYPE_BITS[other][2])
        with pytest.raises(TypeError):
            cpp._host_pointer(wrong, dtype)
        # ...so driving the export with it never happens: the pointer the
        # call would need is never produced.
        with pytest.raises(TypeError):
            storage._lib.tf_storage_copy_from(
                handle, cpp._host_pointer(wrong, dtype))
        # The storage is untouched by the refusal.
        assert _same_bits(storage.to_numpy(),
                          np.zeros(8, dtype=_DTYPE_BITS[dtype][2]), dtype)
    finally:
        storage.close()


@needs_native
@pytest.mark.parametrize("dtype", BOTH_DTYPES)
def test_views_preserve_dtype_storage_identity_and_element_metadata(dtype):
    """A view has no dtype of its own and never could: it borrows the
    storage that holds the one authority. Reshape, transpose, ``.T``,
    narrow, and a chain of them therefore cannot disagree with it, cannot
    copy, and cannot cast."""
    core = cpp.NativeTensorCore._typed_from_array(
        _patterned(dtype, 24).reshape(4, 6), dtype)
    views = []
    try:
        assert core.dtype == dtype
        assert core.storage.dtype == dtype
        chained = core.reshape((2, 12)).narrow(1, 2, 6).transpose(1, 0)
        for view in (core.reshape((2, 12)), core.transpose(1, 0), core.T,
                     core.narrow(0, 1, 2), core.narrow(1, 3, 3).T, chained):
            views.append(view)
            assert view.dtype == dtype
            # One storage object, shared — no copy, no second dtype field.
            assert view.storage is core.storage
            assert view._view._storage is core.storage
            # Shapes, strides, and offsets stay logical element counts.
            assert all(isinstance(n, int) for n in view.shape)
            assert all(isinstance(n, int) for n in view.strides)
            assert isinstance(view.offset, int)
            assert view.offset < core.storage.size
            # The view object itself carries no dtype attribute of its own.
            assert "_dtype" not in vars(view._view)
        # Closing one view leaves every other alias live, and none of them
        # destroys the shared storage.
        views[0].close()
        assert views[1].to_numpy().dtype == np.dtype(_DTYPE_BITS[dtype][2])
        assert core.dtype == dtype
    finally:
        for view in views:
            view.close()
        core.close()


@needs_native
@pytest.mark.parametrize("dtype", BOTH_DTYPES)
def test_view_materialization_is_bit_exact_at_both_dtypes(dtype):
    """Strided, offset, transposed, narrowed, and chained views all
    materialize into a host array of the storage's own dtype, with exact
    bits, in row-major order."""
    values = _patterned(dtype, 24).reshape(4, 6)
    core = cpp.NativeTensorCore._typed_from_array(values, dtype)
    try:
        cases = (
            ("identity", core, values),
            ("reshape", core.reshape((2, 12)), values.reshape(2, 12)),
            ("transpose", core.T, values.T),
            ("narrow rows", core.narrow(0, 1, 2), values[1:3]),
            ("narrow cols", core.narrow(1, 2, 3), values[:, 2:5]),
            ("chained", core.narrow(0, 1, 3).T.narrow(0, 1, 4),
             values[1:4].T[1:5]),
        )
        for name, view, expected in cases:
            try:
                got = view.to_numpy()
                assert got.dtype == np.dtype(_DTYPE_BITS[dtype][2]), name
                assert got.shape == np.asarray(expected).shape, name
                assert _same_bits(got, np.ascontiguousarray(expected),
                                  dtype), (dtype, name)
            finally:
                if view is not core:
                    view.close()
    finally:
        core.close()


@needs_native
@pytest.mark.parametrize("dtype", BOTH_DTYPES)
def test_the_identity_copy_preserves_dtype_and_bits_at_both_dtypes(dtype):
    """``contiguous_copy`` is the runtime's value-transfer primitive. Its
    output has the input's dtype, is canonical row-major contiguous, owns
    fresh storage that aliases nothing, and reproduces the source's bits
    exactly — on the contiguous fast path and on the strided traversals
    alike."""
    values = _patterned(dtype, 24).reshape(4, 6)
    core = cpp.NativeTensorCore._typed_from_array(values, dtype)
    try:
        for name, view, expected in (
            ("contiguous", core, values),
            ("transposed", core.T, values.T),
            ("narrowed", core.narrow(1, 1, 4), values[:, 1:5]),
            ("reshaped", core.reshape((6, 4)), values.reshape(6, 4)),
        ):
            copied = view.contiguous_copy()
            try:
                assert copied.dtype == dtype, name
                assert copied.storage.dtype == dtype, name
                assert copied.contiguous and copied.offset == 0, name
                assert copied.storage is not core.storage, name
                assert _same_bits(copied.to_numpy(),
                                  np.ascontiguousarray(expected),
                                  dtype), (dtype, name)
            finally:
                copied.close()
                if view is not core:
                    view.close()
        # ...and the source is untouched by any of it.
        assert _same_bits(core.to_numpy(), values, dtype)
    finally:
        core.close()


@needs_native
def test_a_mixed_dtype_identity_copy_is_rejected_before_any_write():
    """There is no casting and no promotion anywhere in the runtime, so a
    float32 source with a float64 destination is an invalid request rather
    than a conversion opportunity — and the refusal happens before a single
    element is written, in either direction."""
    library = cpp._require_library()
    f32 = cpp.NativeStorage._typed(20, "float32")
    f64 = cpp.NativeStorage(20)
    try:
        f64.copy_from(np.arange(20, dtype=np.float64))
        before64 = f64.to_numpy().copy()
        before32 = f32.to_numpy().copy()
        shape = (ctypes.c_int64 * 2)(4, 5)
        strides = (ctypes.c_int64 * 2)(5, 1)
        for source, destination in ((f32, f64), (f64, f32)):
            with pytest.raises(ValueError) as caught:
                library.tf_core_contiguous_copy(
                    source._require_open(), destination._require_open(),
                    shape, strides, 0, 2)
            message = str(caught.value)
            assert "float32" in message and "float64" in message
            assert "same dtype" in message
        # Neither buffer moved, in either direction.
        assert _same_bits(f64.to_numpy(), before64, "float64")
        assert _same_bits(f32.to_numpy(), before32, "float32")
    finally:
        f64.close()
        f32.close()


@needs_native
def test_the_raw_utility_kernels_compute_only_in_float64():
    """``RAW_KERNEL_DTYPES`` is a claim about running code, so it is checked
    against the kernels rather than read off the tuple.

    The seven raw kernels are handle-free — they take only ``const
    double*``, ``double*``, and an element count — so there is no dtype tag
    to dispatch on and no way to give them one without changing their
    argument count, which would be a real ABI break. Three separate facts
    make that concrete, and the distinction between them is the point:

    1. **The C ABI positions are float64 and reject anything else.** Every
       raw kernel's array arguments are bound with the float64 checked
       ``ndpointer``, so a float32 buffer never reaches one.
    2. **The Python wrappers convert rather than compute narrow.** They are
       the same explicit host-to-native conversion boundary
       ``from_array`` is, and have been since v0.x: a float32 input is
       converted to float64 on the way in and the **result is float64**. No
       float32 arithmetic happens anywhere, at any width, in any of them.
    3. **No per-dtype raw wrapper or export exists**, and none may be added.
    """
    library = cpp._require_library()
    # 1. The declared boundary is float64, and it refuses a float32 buffer.
    a32 = np.ones(4, dtype=np.float32)
    for name in ("tf_elementwise_add", "tf_elementwise_subtract",
                 "tf_elementwise_multiply", "tf_elementwise_divide",
                 "tf_relu", "tf_matmul", "tf_matmul_tiled"):
        for argtype in getattr(library, name).argtypes:
            if getattr(argtype, "__name__", "").startswith("ndpointer"):
                assert argtype is cpp._CHECKED_F64_ARRAY, name
                with pytest.raises(TypeError):
                    argtype.from_param(a32)

    # 2. The wrappers convert at the host boundary and return float64.
    a32sq = np.ones((2, 2), dtype=np.float32)
    for name in cpp.RAW_KERNELS:
        kernel = getattr(cpp, name)
        if name == "matmul_tiled":
            result = kernel(a32sq, a32sq, block_size=2)
        elif name == "matmul":
            result = kernel(a32sq, a32sq)
        elif name == "relu":
            result = kernel(a32sq)
        else:
            result = kernel(a32sq, a32sq)
        assert result.dtype == np.float64, name

    # 3. No per-dtype raw wrapper or export exists.
    assert not [name for name in dir(cpp)
                if name.endswith(("_f32", "_float32"))]
    assert cpp.RAW_KERNEL_DTYPES == ("float64",)


@needs_native
@pytest.mark.parametrize("dtype", BOTH_DTYPES)
def test_typed_storage_lifetime_returns_exactly_to_baseline(dtype):
    """I1's ownership guardrails, re-run over the paths I2 opened: repeated
    create / view / materialize / copy / close cycles must leave live native
    storage exactly where they found it, at both dtypes."""
    open_ids = set()
    original_init = cpp.NativeStorage.__init__
    original_close = cpp.NativeStorage.close

    def tracked_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        open_ids.add(id(self))

    def tracked_close(self):
        original_close(self)
        open_ids.discard(id(self))

    cpp.NativeStorage.__init__ = tracked_init
    cpp.NativeStorage.close = tracked_close
    try:
        baseline = len(open_ids)
        values = _patterned(dtype, 24).reshape(4, 6)
        for _ in range(50):
            core = cpp.NativeTensorCore._typed_from_array(values, dtype)
            view = core.T
            copied = view.contiguous_copy()
            assert _same_bits(copied.to_numpy(),
                              np.ascontiguousarray(values.T), dtype)
            copied.close()
            view.close()       # a borrowing view frees no shared storage
            core.close()
        assert len(open_ids) == baseline
    finally:
        cpp.NativeStorage.__init__ = original_init
        cpp.NativeStorage.close = original_close


@needs_native
@pytest.mark.parametrize("dtype", BOTH_DTYPES)
def test_a_failed_identity_copy_leaks_nothing_at_either_dtype(dtype):
    """A failure inside ``contiguous_copy`` closes everything it allocated,
    so live storage returns exactly to baseline and no caller can observe
    one lone result."""
    open_ids = set()
    original_init = cpp.NativeStorage.__init__
    original_close = cpp.NativeStorage.close

    def tracked_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        open_ids.add(id(self))

    def tracked_close(self):
        original_close(self)
        open_ids.discard(id(self))

    cpp.NativeStorage.__init__ = tracked_init
    cpp.NativeStorage.close = tracked_close
    try:
        core = cpp.NativeTensorCore._typed_from_array(
            _patterned(dtype, 24).reshape(4, 6), dtype)
        try:
            baseline = len(open_ids)
            # The output allocation itself fails.
            cpp._arm_alloc_failure(1)
            try:
                with pytest.raises(MemoryError):
                    core.T.contiguous_copy()
            finally:
                cpp._arm_alloc_failure(0)
            assert len(open_ids) == baseline
            # ...and the very next copy succeeds, so nothing latched.
            copied = core.contiguous_copy()
            assert copied.dtype == dtype
            copied.close()
            assert len(open_ids) == baseline
        finally:
            core.close()
    finally:
        cpp.NativeStorage.__init__ = original_init
        cpp.NativeStorage.close = original_close


@needs_native
def test_i2_moved_no_public_capability_at_all():
    """The exit gate, as one assertion block: internal float32 transfer
    exists and public float32 support does not."""
    from tensorforge.experimental import native_checkpoint

    _assert_the_public_registry_is_i9s()
    assert cpp.backend_info()["dtype"] == "float64"
    assert native_checkpoint._FORMAT_VERSION == I8_CHECKPOINT_VERSION
    assert (native_checkpoint._SUPPORTED_FORMAT_VERSIONS
            == I8_CHECKPOINT_VERSIONS)
    exports = _source_exports()
    assert len(exports) == I1_EXPORT_COUNT       # still 54; I2 adds none
    for absent in ("tf_storage_copy_from_typed", "tf_storage_copy_to_typed",
                   "tf_storage_materialize_typed",
                   "tf_core_contiguous_copy_f32", "tf_storage_dtype",
                   "tf_storage_cast"):
        assert absent not in exports, absent
    # I2 itself gave no constructor a dtype argument; the six that have one
    # are milestone I7's, and they are exactly the six.
    assert _constructors_with_a_dtype_argument() == DTYPE_CONSTRUCTORS


@needs_native
def test_the_float32_paths_are_exactly_the_families_landed_through_i7():
    """The precise statement that replaces I6's "float32 is computed on by
    transfer/copy, elementwise/unary, reductions, matmul, Conv2d, MaxPool2d,
    the classification stack, and view-backward".

    As of I7 **every numerical family in the native runtime is
    dtype-general**: storage, transfer, views, elementwise/unary execution,
    reductions, matmul, all three Conv2d directions, both MaxPool2d
    directions, softmax, log-softmax, fused cross-entropy, view backward,
    private Core autograd — and now Dropout, the last one out. There is no
    remaining float64-only compute path to enumerate, which is a real
    milestone and is stated as one.

    Above the Core layer, I7 opened exactly the state-owning constructors
    (§12.1) and nothing beyond them: ``NativeParameter`` builds float32
    state, and the six named modules construct at a dtype. What has **not**
    moved is the part that would make float32 a promise — the public
    registry, the public tensor constructors, the optimizers, and the
    checkpoint format. Those are milestones I8 and I9.

    This is the guardrail that keeps "float32 works" from ever being the
    honest summary: the set is enumerated in both directions.
    """
    library = cpp._require_library()
    core = cpp.NativeTensorCore._typed_from_array(
        _patterned("float32", 16).reshape(4, 4), "float32")
    other = cpp.NativeTensorCore._typed_from_array(
        np.ones((4, 4), dtype=np.float32), "float32")
    output = cpp.NativeStorage._typed(16, "float32")
    # The winner buffer a float32 pool consumes is **float64** — that is
    # the §13.3 decision, not a leftover: the value dtype dispatches, the
    # winner dtype is pinned.
    winners64 = cpp.NativeStorage(16)
    try:
        # -- consumed: transfer in, transfer out, materialize, identity copy
        assert core.to_numpy().dtype == np.float32
        copied = core.T.contiguous_copy()
        assert copied.dtype == "float32"
        copied.close()

        shape = (ctypes.c_int64 * 2)(4, 4)
        strides = (ctypes.c_int64 * 2)(4, 1)
        a = core.storage._require_open()
        b = other.storage._require_open()
        out = output._require_open()

        # -- consumed, as of I3: the elementwise and unary family. Each of
        # these must now *succeed* at float32 and leave the error slot clear.
        for call in (
            lambda: library.tf_core_relu(a, out, shape, strides, 0, 2),
            lambda: library.tf_core_relu_contiguous(a, out, 16, 0),
            lambda: library.tf_core_sqrt(a, out, shape, strides, 0, 2),
            lambda: library.tf_core_sqrt_contiguous(a, out, 16, 0),
            lambda: library.tf_core_reciprocal(a, out, shape, strides, 0, 2),
            lambda: library.tf_core_reciprocal_contiguous(a, out, 16, 0),
            lambda: library.tf_core_exp(a, out, shape, strides, 0, 2),
            lambda: library.tf_core_exp_contiguous(a, out, 16, 0),
            lambda: library.tf_core_log(a, out, shape, strides, 0, 2),
            lambda: library.tf_core_log_contiguous(a, out, 16, 0),
            lambda: library.tf_core_add(a, b, out, shape, strides, strides,
                                        0, 0, 2),
            lambda: library.tf_core_add_contiguous(a, b, out, 16, 0, 0),
            lambda: library.tf_core_subtract(a, b, out, shape, strides,
                                             strides, 0, 0, 2),
            lambda: library.tf_core_subtract_contiguous(a, b, out, 16, 0, 0),
            lambda: library.tf_core_multiply(a, b, out, shape, strides,
                                             strides, 0, 0, 2),
            lambda: library.tf_core_multiply_contiguous(a, b, out, 16, 0, 0),
            lambda: library.tf_core_relu_backward(a, b, out, shape, strides,
                                                  strides, 0, 0, 2),
            # -- consumed, as of I4: reductions, matmul, and the view
            # backward, plus the two scalar storage primitives the mean
            # scaling and the backward constants are built from.
            lambda: library.tf_core_matmul(a, b, out, 4, 4, 4, 4, 1, 4, 1,
                                           0, 0),
            lambda: library.tf_core_sum(a, out, shape, strides,
                                        (ctypes.c_int64 * 2)(0, 1), 0, 2),
            lambda: library.tf_core_narrow_backward(a, out, shape, strides,
                                                    strides, 0, 0, 2),
            # -- consumed, as of I5: all three Conv2d directions and both
            # MaxPool2d directions. The pooling calls carry a float64
            # winner buffer beside the float32 values, which is the locked
            # winner-dtype decision in action.
            # N, C, H, W, O, kh, kw, sh, sw, ph, pw, out_h, out_w
            lambda: library.tf_core_conv2d_forward(
                a, 0, b, 0, None, 0, out, 1, 1, 4, 4, 1, 2, 2, 1, 1, 0, 0,
                3, 3),
            lambda: library.tf_core_conv2d_input_backward(
                a, 0, b, 0, out, 1, 1, 4, 4, 1, 2, 2, 1, 1, 0, 0, 3, 3),
            lambda: library.tf_core_conv2d_weight_backward(
                a, 0, b, 0, out, 1, 1, 4, 4, 1, 2, 2, 1, 1, 0, 0, 3, 3),
            # N, C, H, W, kh, kw, sh, sw, ph, pw, out_h, out_w
            lambda: library.tf_core_maxpool2d_forward(
                a, 0, out, winners64._require_open(),
                1, 1, 4, 4, 2, 2, 2, 2, 0, 0, 2, 2),
            # The zero-initialized winner values are exact in-range plane
            # offsets, so the scatter is a legitimate call.
            lambda: library.tf_core_maxpool2d_backward(
                a, 0, winners64._require_open(), 0, out, 1, 1, 4, 4, 2, 2),
        ):
            library.tf_clear_error()
            call()
            assert library.tf_last_error_code() == 0

        # ``fill`` and ``scale`` are unhooked (H7) and now cannot fail at
        # all: with every dtype valid there is nothing left for either to
        # reject, so each leaves the error slot exactly as it found it.
        library.tf_clear_error()
        library.tf_storage_fill(out, 1.5)
        assert library.tf_last_error_code() == 0
        library.tf_storage_scale(out, 2.0)
        assert library.tf_last_error_code() == 0
        assert np.array_equal(output.to_numpy(),
                              np.full(16, 3.0, dtype=np.float32))
        assert output.to_numpy().dtype == np.float32

        # -- consumed, as of I6: the stable transforms and the fused
        # cross-entropy. The cross-entropy calls need their own destinations
        # (a scalar loss and a probability block) and an int64 target span,
        # which carries no dtype at either width.
        targets = np.zeros(4, dtype=np.int64)
        ce_loss = cpp.NativeStorage._typed(1, "float32")
        ce_probabilities = cpp.NativeStorage._typed(16, "float32")
        ce_grad = cpp.NativeStorage._typed(16, "float32")
        try:
            loss_handle = ce_loss._require_open()
            probabilities_handle = ce_probabilities._require_open()
            grad_handle = ce_grad._require_open()
            for call in (
                lambda: library.tf_core_softmax_forward(a, 0, out, 4, 4, 1),
                lambda: library.tf_core_log_softmax_forward(a, 0, out, 4, 4,
                                                            1),
                lambda: library.tf_core_cross_entropy_forward(
                    a, 0, targets, 4, loss_handle, probabilities_handle,
                    4, 4, 0),
                lambda: library.tf_core_cross_entropy_backward(
                    probabilities_handle, 0, targets, 4, loss_handle, 0,
                    grad_handle, 4, 4, 0),
            ):
                library.tf_clear_error()
                call()
                assert library.tf_last_error_code() == 0
            assert ce_loss.to_numpy().dtype == np.float32
            assert ce_probabilities.to_numpy().dtype == np.float32
            assert ce_grad.to_numpy().dtype == np.float32
        finally:
            ce_grad.close()
            ce_probabilities.close()
            ce_loss.close()

        # -- consumed, as of I7: Dropout, the last float64-only family.
        # Output and mask are separate destinations, so the aliasing rule
        # needs a second float32 block.
        mask32 = cpp.NativeStorage._typed(16, "float32")
        try:
            library.tf_clear_error()
            library.tf_core_dropout_forward(
                a, 0, out, mask32._require_open(), 16, 1, 0, 0.5)
            assert library.tf_last_error_code() == 0
            assert mask32.to_numpy().dtype == np.float32
            # Exactly two distinct multipliers, and the kept one is the
            # binary32 narrowing of the binary64 reciprocal.
            assert set(np.unique(mask32.to_numpy())) <= {
                np.float32(0.0), np.float32(1.0 / (1.0 - 0.5))}
        finally:
            mask32.close()

        # ...so there is no float64-only numerical export left to name. The
        # rejecting set is empty, asserted as a set rather than as prose.

        # Above the Core layer, exactly the I7 surface moved: a parameter
        # can be built at float32 — that is the milestone. Through I8 every
        # **public** tensor constructor still refused, because the registry
        # had not moved; **I9 moved it**, so they succeed now and the two
        # routes must agree. What I7 owns is the constructor argument, and
        # that is what is asserted here.
        from tensorforge.experimental import NativeParameter, NativeTensor
        for construct in (
            lambda: NativeTensor.from_array([1.0], dtype="float32"),
            lambda: NativeTensor.zeros((2,), dtype="float32"),
            lambda: NativeTensor.full((2,), 1.0, dtype="float32"),
            lambda: cpp.NativeTensorCore.from_array([1.0], dtype="float32"),
            lambda: cpp.NativeTensorCore.zeros((2,), dtype="float32"),
            lambda: cpp.NativeTensorCore.full((2,), 1.0, dtype="float32"),
        ):
            built = construct()
            try:
                assert built.dtype == "float32"
            finally:
                built.close()
        parameter = NativeParameter([1.0], dtype="float32")
        try:
            assert parameter.dtype == "float32"
        finally:
            parameter.close()
    finally:
        winners64.close()
        output.close()
        other.close()
        core.close()


# ---------------------------------------------------------------------------
# I3: elementwise, broadcast, and unary dtype execution, as running code
#
# Everything below drives the **live** library at both dtypes through the
# ``NativeTensorCore`` operations themselves. float32 is still not a supported
# TensorForge dtype and no public constructor can produce one, so these reach
# it through the private typed constructors the rollout rule (design §27.2)
# requires an intermediate milestone to test through.
#
# The comparison rules are **per operation**, and the split is the point:
#
#   * ``add``, ``subtract``, ``multiply``, ``relu``, ``relu_backward``,
#     ``sqrt``, and ``reciprocal`` are IEEE-754-specified and correctly
#     rounded, so they are compared **bit for bit** against an independent
#     NumPy oracle **at the same dtype**.
#   * ``exp`` and ``log`` are library functions with no correctly-rounded
#     guarantee, so they get a measured ULP bound — the same reason the
#     float64 contract uses one, restated at binary32.
#
# I2's transfer contract is **not** reused here. A transfer performs no
# arithmetic, so it preserves every object representation including a
# signalling NaN's signalling-ness; an *operation* follows IEEE arithmetic
# instead and therefore normalizes ``-0.0`` where the operation says so and
# quiets a signalling NaN. Those are different contracts and this section
# asserts the arithmetic one.
# ---------------------------------------------------------------------------

# Two measured bounds, neither of them a guess.
#
# Against a float32 reference formed by rounding the binary64 value **once**,
# TensorForge's float32 ``exp``/``log`` were measured at **1** representable
# step over 200,000 inputs on the toolchains validated here. Against NumPy's
# *own* float32 transcendentals the distance is **2**, because NumPy uses a
# separate SIMD kernel for float32 that is itself around two steps from
# correctly rounded — so the wider bound is a statement about NumPy, not
# about TensorForge, and the two are recorded separately rather than
# collapsed into the looser one.
F32_TRANSCENDENTAL_ULP = 1
F32_TRANSCENDENTAL_ULP_VS_NUMPY = 2

# The float64 half of the same statement, and not a new concession: it is
# the bound ``tests/test_native_abi_boundary.py`` has enforced since it was
# written, for the reason recorded there. ``exp`` and ``log`` are plain
# ``std::exp``/``std::log`` (cpp/src/elementwise.cpp), which no toolchain
# promises to round correctly, and they are deliberately excluded from the
# templated traversals so they stay on the retained odometer exclusively.
# Comparing TensorForge's libm against NumPy's bit-for-bit therefore tests
# the platform rather than TensorForge — measured here at 200,000 inputs,
# the shipped Windows UCRT and Linux glibc 2.39 results differ on 1,034
# ``exp`` elements and 149 ``log`` elements and **never by more than one
# representable step**, with each within one step of correctly rounded.
# One step is the tightest bound that evidence supports; zero is the
# platform-dependent claim that failed in Linux CI.
F64_TRANSCENDENTAL_ULP = 1


def _f32_ulp_distance(got, want):
    """Representable float32 steps between two finite arrays, elementwise.

    Sign-magnitude is reflected onto one monotone integer line, exactly as
    ``tests/test_native_abi_boundary.py`` does at float64, so the count is
    correct across zero and across the subnormal boundary.
    """
    def monotone(values):
        raw = np.ascontiguousarray(values, dtype=np.float32).view(np.int32)
        return np.where(raw < 0, np.int64(-(2 ** 31)) - raw.astype(np.int64),
                        raw.astype(np.int64))
    return np.abs(monotone(got) - monotone(want))


def _f64_ulp_distance(got, want):
    """Representable float64 steps between two finite arrays, elementwise.

    The same reflection as ``_f32_ulp_distance``, one width up, with the
    same semantics as ``tests/test_native_abi_boundary.py``'s scalar
    ``_ulp_distance``: neighbouring floats are 1 apart, a value is 0 from
    itself, and the count is exact across zero and across the
    denormal/normal boundary.

    The arithmetic is done in Python ``int`` rather than ``int64`` because
    the *difference* between two monotone float64 keys can reach
    ``2**64 - 2`` — representable as a Python integer and not as an
    ``int64`` — so a vectorized subtraction would silently wrap on exactly
    the pair a bound is most interested in. The arrays this runs on are
    small, and exactness is the point.
    """
    def keys(values):
        raw = np.ascontiguousarray(values, dtype=np.float64).view(np.int64)
        return [(-(2 ** 63) - bits) if bits < 0 else bits
                for bits in raw.reshape(-1).tolist()]
    return [abs(a - b) for a, b in zip(keys(got), keys(want))]


_ULP_DISTANCE = {"float32": _f32_ulp_distance, "float64": _f64_ulp_distance}


def _assert_transcendental(got, want, limit, label, dtype="float32"):
    """NaN by position, infinities and zeros by exact bits, everything else
    within ``limit`` steps. The tolerance deliberately never covers a zero:
    a distance cannot see a zero's sign, and that is precisely the kind of
    thing this suite exists to catch.

    ``dtype`` selects the width the steps are counted in, and both
    operands are required to be exactly that width — so a float32 result
    is never compared against a float64 reference, at either setting.
    """
    floating = _DTYPE_BITS[dtype][2]
    assert got.dtype == np.dtype(floating), (label, got.dtype)
    assert want.dtype == np.dtype(floating), (label, want.dtype)
    assert np.array_equal(np.isnan(got), np.isnan(want)), f"{label}: NaN places"
    special = np.isnan(got) | np.isinf(got) | np.isinf(want) | (got == 0) \
        | (want == 0)
    assert _same_bits(got[special], want[special], dtype), (
        f"{label}: a special value is not exactly reproduced")
    ordinary = ~special
    if not ordinary.any():
        return 0
    worst = int(max(_ULP_DISTANCE[dtype](got[ordinary], want[ordinary])))
    assert worst <= limit, f"{label}: {worst} ULP apart, over the {limit} bound"
    return worst


# -- the ULP helpers' own contract, at both widths --------------------------
#
# These need no backend: they pin the measuring instrument *before* anything
# is asserted with it, so "within one ULP" cannot quietly become "within
# whatever the helper happens to accept". A bound that is never shown able
# to fail is an unbounded tolerance wearing a number.

@pytest.mark.parametrize("dtype", BOTH_DTYPES)
@pytest.mark.parametrize("case", ("equal", "neighbour_up", "neighbour_down",
                                  "across_zero", "denormal_boundary",
                                  "signed_zeros_are_one_number"))
def test_the_ulp_distance_helpers_count_representable_steps(dtype, case):
    """Symmetric, exact at the denormal boundary, correct across zero, and
    blind to a zero's sign — which is why ``_assert_transcendental`` checks
    zeros by raw bits instead of by distance."""
    floating = _DTYPE_BITS[dtype][2]
    tiny = floating(np.finfo(floating).smallest_subnormal)
    smallest_normal = floating(np.finfo(floating).smallest_normal)
    pairs = {
        "equal": (floating(1.5), floating(1.5), 0),
        "neighbour_up": (floating(1.0),
                         np.nextafter(floating(1.0), floating(np.inf)), 1),
        "neighbour_down": (floating(-1.0),
                           np.nextafter(floating(-1.0), floating(-np.inf)), 1),
        "across_zero": (tiny, -tiny, 2),
        "denormal_boundary": (
            np.nextafter(smallest_normal, floating(0)), smallest_normal, 1),
        "signed_zeros_are_one_number": (floating(0.0), floating(-0.0), 0),
    }
    a, b, expected = pairs[case]
    distance = _ULP_DISTANCE[dtype]
    assert int(max(distance(np.array([a], dtype=floating),
                            np.array([b], dtype=floating)))) == expected
    assert int(max(distance(np.array([b], dtype=floating),
                            np.array([a], dtype=floating)))) == expected


@pytest.mark.parametrize("dtype", BOTH_DTYPES)
def test_the_transcendental_assertion_enforces_the_stated_boundary(dtype):
    """The negative control the one-ULP repair rests on. Exact equality
    passes, one step passes at a one-step budget, **two steps fail**, and no
    special value is ever absorbed by the tolerance."""
    floating = _DTYPE_BITS[dtype][2]

    def array(*values):
        return np.array(values, dtype=floating)

    one = array(1.0)
    up = np.nextafter(floating(1.0), floating(np.inf))
    two_up = np.nextafter(up, floating(np.inf))

    # Exact equality passes, and reports a zero-wide budget.
    assert _assert_transcendental(one, one, 1, "equal", dtype=dtype) == 0
    # One representable step passes at limit 1, and reports it honestly.
    assert _assert_transcendental(one, array(up), 1, "one step",
                                  dtype=dtype) == 1
    # Two steps do not. This is the assertion that keeps the bound a bound.
    with pytest.raises(AssertionError, match="2 ULP apart"):
        _assert_transcendental(one, array(two_up), 1, "two steps", dtype=dtype)
    # ...and one step does not pass at a zero-step budget either, so the
    # limit argument is genuinely load-bearing.
    with pytest.raises(AssertionError, match="1 ULP apart"):
        _assert_transcendental(one, array(up), 0, "exactness", dtype=dtype)

    # Specials are never covered by the tolerance, however small it is.
    with pytest.raises(AssertionError, match="NaN places"):
        _assert_transcendental(array(np.nan), one, 1, "nan place", dtype=dtype)
    with pytest.raises(AssertionError, match="NaN places"):
        _assert_transcendental(one, array(np.nan), 1, "nan place", dtype=dtype)
    with pytest.raises(AssertionError, match="special value"):
        _assert_transcendental(array(0.0), array(-0.0), 1, "signed zero",
                               dtype=dtype)
    with pytest.raises(AssertionError, match="special value"):
        _assert_transcendental(array(np.inf), array(-np.inf), 1, "inf sign",
                               dtype=dtype)
    with pytest.raises(AssertionError, match="special value"):
        _assert_transcendental(array(np.inf), array(np.finfo(floating).max),
                               1, "inf vs finite", dtype=dtype)
    # A NaN in the *same* place on both sides is fine, and is not measured.
    assert _assert_transcendental(array(np.nan, 1.0), array(np.nan, 1.0), 1,
                                  "matching nan", dtype=dtype) == 0


def test_the_transcendental_assertion_refuses_a_mixed_width_comparison():
    """A float32 result is never compared against a float64 reference, at
    either setting — the dtype claim is exact even where the value claim is
    a bound."""
    thirty_two = np.array([1.0], dtype=np.float32)
    sixty_four = np.array([1.0], dtype=np.float64)
    with pytest.raises(AssertionError):
        _assert_transcendental(thirty_two, sixty_four, 1, "mixed",
                               dtype="float32")
    with pytest.raises(AssertionError):
        _assert_transcendental(sixty_four, thirty_two, 1, "mixed",
                               dtype="float64")
    with pytest.raises(AssertionError):
        _assert_transcendental(thirty_two, thirty_two, 1, "wrong width",
                               dtype="float64")


def _core(values, dtype):
    """A NativeTensorCore holding ``values`` at ``dtype``, through the
    private typed constructor at float32 and the ordinary one at float64 —
    so the float64 half of every test below is the *public* path."""
    array = np.ascontiguousarray(values, dtype=_DTYPE_BITS[dtype][2])
    if dtype == "float64":
        return cpp.NativeTensorCore.from_array(array)
    return cpp.NativeTensorCore._typed_from_array(array, dtype)


def _sample(dtype, count, low=-9.0, high=9.0, seed=20260801):
    rng = np.random.default_rng(seed)
    return rng.uniform(low, high, count).astype(_DTYPE_BITS[dtype][2])


@needs_native
@pytest.mark.parametrize("dtype", BOTH_DTYPES)
@pytest.mark.parametrize("name", ("add", "subtract", "multiply"))
def test_binary_elementwise_matches_the_oracle_bitwise_at_both_dtypes(
        dtype, name):
    """The core I3 claim for the binary family: at either dtype the result
    is **exactly** what that dtype's arithmetic produces, on every traversal
    tier, compared as raw bit patterns against NumPy at the same width.

    Bit equality is the right contract here and not an over-claim: each of
    the three is a single correctly-rounded IEEE operation per destination
    element, with no accumulation, so there is nothing for an implementation
    to be legitimately different about.
    """
    floating = _DTYPE_BITS[dtype][2]
    left = _sample(dtype, 24, seed=1).reshape(4, 6)
    right = _sample(dtype, 24, seed=2).reshape(4, 6)
    oracle = {"add": np.add, "subtract": np.subtract,
              "multiply": np.multiply}[name]

    a = _core(left, dtype)
    b = _core(right, dtype)
    try:
        # Tier 1: both contiguous, same shape -> the flat fast-path kernel.
        out = getattr(a, name)(b)
        try:
            assert out.dtype == dtype
            assert out.to_numpy().dtype == np.dtype(floating)
            assert _same_bits(out.to_numpy(), oracle(left, right), dtype)
        finally:
            out.close()

        # Tier 2/3: a transposed operand cannot collapse, so the same
        # elements arrive through the plan and the odometer instead.
        at, bt = a.T, b.T
        try:
            out = getattr(at, name)(bt)
            try:
                assert out.dtype == dtype
                assert _same_bits(out.to_numpy(),
                                  np.ascontiguousarray(oracle(left.T, right.T)),
                                  dtype)
            finally:
                out.close()
        finally:
            bt.close()
            at.close()
    finally:
        b.close()
        a.close()


@needs_native
@pytest.mark.parametrize("dtype", BOTH_DTYPES)
def test_narrowed_and_offset_operands_compute_correctly(dtype):
    """A narrowed view carries a nonzero offset and a row stride wider than
    its own extent, which is the layout the plan builder must not merge and
    the odometer must address exactly."""
    left = _sample(dtype, 24, seed=3).reshape(4, 6)
    right = _sample(dtype, 24, seed=4).reshape(4, 6)
    a = _core(left, dtype)
    b = _core(right, dtype)
    try:
        na, nb = a.narrow(1, 1, 4), b.narrow(1, 2, 4)
        try:
            out = na.add(nb)
            try:
                assert out.dtype == dtype
                assert _same_bits(out.to_numpy(),
                                  (left[:, 1:5] + right[:, 2:6]).copy(), dtype)
            finally:
                out.close()
            out = na.multiply(nb)
            try:
                assert _same_bits(out.to_numpy(),
                                  (left[:, 1:5] * right[:, 2:6]).copy(), dtype)
            finally:
                out.close()
        finally:
            nb.close()
            na.close()
    finally:
        b.close()
        a.close()


@needs_native
@pytest.mark.parametrize("dtype", BOTH_DTYPES)
@pytest.mark.parametrize("left_shape,right_shape", [
    ((4, 6), (1, 6)),      # a stretched leading axis
    ((4, 6), (4, 1)),      # a stretched trailing axis
    ((4, 6), (6,)),        # left-padded rank
    ((4, 1), (1, 6)),      # both stretched, in different axes
    ((3, 1, 5), (1, 4, 5)),  # multiple broadcast axes, rank 3
    ((4, 6), ()),          # a rank-0 scalar operand
])
def test_broadcasting_works_at_both_dtypes(dtype, left_shape, right_shape):
    """NumPy-style broadcasting, unchanged in rule and unchanged in
    traversal — a stretched axis is read through a **zero stride**, so no
    expanded operand is materialized. I3 changes only the element type the
    zero-stride read produces."""
    floating = _DTYPE_BITS[dtype][2]
    left = _sample(dtype, int(np.prod(left_shape)) or 1,
                   seed=5).reshape(left_shape)
    right = _sample(dtype, int(np.prod(right_shape)) or 1,
                    seed=6).reshape(right_shape)
    a = _core(left, dtype)
    b = _core(right, dtype)
    try:
        for name, oracle in (("add", np.add), ("subtract", np.subtract),
                             ("multiply", np.multiply)):
            out = getattr(a, name)(b)
            try:
                expected = oracle(left, right)
                assert out.dtype == dtype
                assert out.shape == expected.shape
                assert out.to_numpy().dtype == np.dtype(floating)
                assert _same_bits(out.to_numpy(), expected, dtype), name
            finally:
                out.close()
    finally:
        b.close()
        a.close()


@needs_native
@pytest.mark.parametrize("dtype", BOTH_DTYPES)
def test_unary_operations_match_the_oracle_at_both_dtypes(dtype):
    """relu, sqrt, and reciprocal are IEEE-specified, so they are held to
    bit equality at both widths; exp and log get the measured ULP bound."""
    floating = _DTYPE_BITS[dtype][2]
    signed = _sample(dtype, 24, seed=7).reshape(4, 6)
    positive = np.abs(_sample(dtype, 24, seed=8).reshape(4, 6)) + floating(0.5)

    # ReLU is a **comparison-select**, ``x > 0 ? x : 0``, not ``max(x, 0)``:
    # the two agree on every finite value and disagree on NaN, where
    # ``np.maximum`` propagates and the select answers ``+0.0``. That
    # convention is the Python Tensor's ``(x > 0) * grad`` rule and predates
    # Phase I; the oracle is written to match the kernel rather than the
    # other way round.
    for values, cases in (
        (signed, (("relu", lambda v: np.where(v > floating(0), v,
                                              floating(0)).astype(floating)),)),
        (positive, (("sqrt", np.sqrt), ("reciprocal", lambda v: floating(1) / v))),
    ):
        core = _core(values, dtype)
        try:
            for name, oracle in cases:
                out = getattr(core, name)()
                try:
                    assert out.dtype == dtype
                    assert out.to_numpy().dtype == np.dtype(floating)
                    assert _same_bits(out.to_numpy(), oracle(values), dtype), name
                finally:
                    out.close()
                # ...and through a transposed view, which takes a different
                # traversal tier over the same logical elements.
                view = core.T
                try:
                    out = getattr(view, name)()
                    try:
                        assert _same_bits(out.to_numpy(),
                                          np.ascontiguousarray(oracle(values).T),
                                          dtype), f"{name} transposed"
                    finally:
                        out.close()
                finally:
                    view.close()
        finally:
            core.close()


@needs_native
def test_float64_transcendentals_stay_within_the_measured_ulp_bound():
    """The float64 half of exp/log is unchanged by the generalization — same
    kernel structure, same dtype in and out, same retained-odometer path —
    and it is held to the **measured** contract rather than to bit equality.

    ``exp`` and ``log`` are plain ``std::exp`` and ``std::log``
    (cpp/src/elementwise.cpp), and they remain the production operations at
    both widths; H8 deliberately kept them off the templated traversals for
    exactly this reason. Neither has a correctly-rounded cross-toolchain
    guarantee, so **comparing TensorForge's libm against NumPy's
    bit-for-bit tests the platform, not TensorForge** — it is the same
    invalid oracle the ABI-boundary module retired, and it is the one that
    failed in Linux CI while passing on the machine it was written on.

    The established contract, unchanged and not widened here, is: ordinary
    finite results within **one representable float64 step**, and specials
    exact — NaN in the same places, infinities matching by sign, and both
    signed zeros matching by raw bit pattern. The dtype claim stays exact
    at every point, because the dtype is a fact the implementation does
    specify.
    """
    values = _sample("float64", 64, seed=9)
    positive = np.abs(values) + 0.5
    for name, source, oracle in (("exp", values, np.exp),
                                 ("log", positive, np.log)):
        core = _core(source, "float64")
        try:
            out = core.exp() if name == "exp" else core.log()
            try:
                got = out.to_numpy()
                # Exact, because these are specified: the operation stays at
                # float64 in and float64 out, and narrows nothing.
                assert out.dtype == "float64"
                assert got.dtype == np.float64
                assert got.shape == source.shape
                _assert_transcendental(got, oracle(source),
                                       F64_TRANSCENDENTAL_ULP,
                                       f"float64 {name} vs numpy",
                                       dtype="float64")
            finally:
                out.close()
        finally:
            core.close()


@needs_native
def test_float32_transcendentals_are_within_the_measured_ulp_bounds():
    """``exp`` and ``log`` are library functions with no correctly-rounded
    IEEE guarantee, so the float32 contract is a bound rather than bit
    equality — exactly as the float64 contract is, and for the same reason:
    libm differs between toolchains.

    Two bounds, because they say different things. Against a float32
    reference rounded **once** from binary64 the distance is one step; against
    NumPy's own float32 kernel it is two, because that kernel is itself about
    two steps from correctly rounded. Neither is a comparison of a float32
    result to a float64 result — both references are float32 values.
    """
    values = _sample("float32", 4096, seed=10, low=-40.0, high=40.0)
    positive = np.abs(_sample("float32", 4096, seed=11, low=1e-20, high=1e20))
    for name, source, oracle in (("exp", values, np.exp),
                                 ("log", positive, np.log)):
        core = _core(source, "float32")
        try:
            out = core.exp() if name == "exp" else core.log()
            try:
                got = out.to_numpy()
                assert out.dtype == "float32"
                assert got.dtype == np.float32
                rounded_once = oracle(source.astype(np.float64)).astype(
                    np.float32)
                _assert_transcendental(got, rounded_once,
                                       F32_TRANSCENDENTAL_ULP,
                                       f"{name} vs a once-rounded reference")
                _assert_transcendental(got, oracle(source),
                                       F32_TRANSCENDENTAL_ULP_VS_NUMPY,
                                       f"{name} vs numpy float32")
            finally:
                out.close()
        finally:
            core.close()


@needs_native
@pytest.mark.parametrize("dtype", BOTH_DTYPES)
def test_special_values_follow_ieee_arithmetic_at_both_dtypes(dtype):
    """The arithmetic contract, which is **not** I2's transfer contract.

    A transfer reproduces a signalling NaN as a signalling NaN because it
    performs no arithmetic. An operation computes, so IEEE-754 applies: a
    NaN operand propagates as a **quiet** NaN, ``relu`` of ``-0.0`` is the
    branch's answer rather than a copy, and the infinities and signed zeros
    take the values the standard specifies. All of that is asserted as raw
    bits against NumPy at the same dtype, so nothing is laundered by ``==``.
    """
    floating = _DTYPE_BITS[dtype][2]
    # Ordinary values plus every special class, built from bits so no
    # arithmetic quiets a signalling NaN before the kernel sees it.
    specials = _patterned(dtype, 17)
    core = _core(specials, dtype)
    try:
        out = core.relu()
        try:
            # The comparison-select convention, at its only interesting
            # input: ``NaN > 0`` is false, so ReLU answers ``+0.0`` where
            # ``np.maximum`` would propagate the NaN. Both signed zeros
            # likewise answer ``+0.0`` — the select's ``else`` branch is a
            # typed zero, not a copy of the operand — and that is asserted
            # as raw bits, which is the only way to see it at all.
            assert _same_bits(
                out.to_numpy(),
                np.where(specials > floating(0), specials,
                         floating(0)).astype(floating), dtype)
        finally:
            out.close()
        out = core.reciprocal()
        try:
            got = out.to_numpy()
            # NumPy warns on divide-by-zero, overflow, and invalid; these
            # kernels produce the same IEEE *values* without warning, so the
            # oracle's warnings are suppressed rather than the values changed.
            with np.errstate(all="ignore"):
                want = floating(1) / specials
            # NaN positions and quietness first, then bits everywhere the
            # value is not a NaN (whose payload is outside the contract when
            # the operand was itself a NaN).
            assert np.array_equal(np.isnan(got), np.isnan(want))
            ordinary = ~np.isnan(got)
            assert _same_bits(got[ordinary], want[ordinary], dtype)
            # Every NaN an *operation* produces is quiet, signalling operand
            # included — the bit that distinguishes this from I2's copy.
            quiet_bit = (np.uint64(1) << np.uint64(51) if dtype == "float64"
                         else np.uint32(1) << np.uint32(22))
            produced = _bits(np.ascontiguousarray(got[np.isnan(got)]), dtype)
            assert bool(np.all((produced & quiet_bit) != 0))
        finally:
            out.close()
    finally:
        core.close()

    # ...and a binary operation over the same classes, likewise quiet.
    other = _core(np.ones(17, dtype=floating), dtype)
    core = _core(specials, dtype)
    try:
        for name, oracle in (("add", np.add), ("multiply", np.multiply)):
            out = getattr(core, name)(other)
            try:
                got = out.to_numpy()
                with np.errstate(all="ignore"):
                    want = oracle(specials, np.ones(17, dtype=floating))
                assert np.array_equal(np.isnan(got), np.isnan(want)), name
                ordinary = ~np.isnan(got)
                assert _same_bits(got[ordinary], want[ordinary], dtype), name
            finally:
                out.close()
    finally:
        core.close()
        other.close()


@needs_native
@pytest.mark.parametrize("dtype", BOTH_DTYPES)
def test_every_traversal_tier_produces_identical_bits(dtype):
    """H8's tier-parity statement, restated per dtype: the contiguous flat
    row, the collapsed plan, and the retained generic odometer are traversal
    choices and nothing else.

    All three are driven over the *same* logical elements — a contiguous
    tensor, a reshaped view of it that still collapses, and a rank-4 view
    the builder handles plus a transposed one it cannot merge — and every
    result is compared bit for bit against the first.
    """
    values = _sample(dtype, 48, seed=12)
    flat = _core(values, dtype)
    partner = _core(_sample(dtype, 48, seed=13), dtype)
    try:
        reference = flat.add(partner)
        try:
            expected = reference.to_numpy()
            # The same 48 elements through a rank-3 view (which collapses to
            # one axis) and a rank-4 one, then reshaped back.
            for shape in ((4, 12), (2, 3, 8), (2, 2, 3, 4)):
                a, b = flat.reshape(shape), partner.reshape(shape)
                try:
                    out = a.add(b)
                    try:
                        assert _same_bits(out.to_numpy().reshape(48), expected,
                                          dtype), shape
                    finally:
                        out.close()
                finally:
                    b.close()
                    a.close()
        finally:
            reference.close()

        # ...and the odometer tier, reached through a transposed operand,
        # against an independently transposed reference.
        square = _core(values[:36].reshape(6, 6), dtype)
        square_b = _core(_sample(dtype, 36, seed=14).reshape(6, 6), dtype)
        try:
            view = square.T
            try:
                out = view.add(square_b)
                try:
                    assert _same_bits(
                        out.to_numpy(),
                        np.ascontiguousarray(square.to_numpy().T)
                        + square_b.to_numpy(), dtype)
                finally:
                    out.close()
            finally:
                view.close()
        finally:
            square_b.close()
            square.close()
    finally:
        partner.close()
        flat.close()


@needs_native
def test_mixed_dtype_is_rejected_before_the_output_is_allocated():
    """Design §9.3, as a measurement rather than an inference: after a
    rejected mixed-dtype operation, **no** storage was allocated, live
    storage is exactly what it was, and both operands are unchanged and
    open.

    Python already knows the mismatch, so it raises in
    ``_require_matching_metadata`` before the output allocation — and C++
    revalidates independently at the trust boundary, which the C++ CTest
    proves by driving the ABI directly.
    """
    allocations = []
    open_ids = set()
    original_init = cpp.NativeStorage.__init__
    original_close = cpp.NativeStorage.close

    def tracked_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        allocations.append(self.dtype)
        open_ids.add(id(self))

    def tracked_close(self):
        original_close(self)
        open_ids.discard(id(self))

    f32 = cpp.NativeTensorCore._typed_from_array(
        np.arange(12, dtype=np.float32).reshape(3, 4), "float32")
    f64 = cpp.NativeTensorCore.from_array(
        np.arange(12, dtype=np.float64).reshape(3, 4))
    before32 = f32.to_numpy().copy()
    before64 = f64.to_numpy().copy()
    cpp.NativeStorage.__init__ = tracked_init
    cpp.NativeStorage.close = tracked_close
    try:
        baseline = len(open_ids)
        for left, right in ((f32, f64), (f64, f32)):
            for name in ("add", "subtract", "multiply", "relu_backward"):
                with pytest.raises(ValueError, match="matching dtype"):
                    getattr(left, name)(right)
        # Not one allocation happened, so nothing needed cleaning up.
        assert allocations == []
        assert len(open_ids) == baseline
    finally:
        cpp.NativeStorage.__init__ = original_init
        cpp.NativeStorage.close = original_close

    # Both operands survived, open and unchanged, in both directions.
    assert _same_bits(f32.to_numpy(), before32, "float32")
    assert _same_bits(f64.to_numpy(), before64, "float64")
    f64.close()
    f32.close()


@needs_native
@pytest.mark.parametrize("dtype", BOTH_DTYPES)
def test_repeated_elementwise_work_returns_live_storage_to_baseline(dtype):
    """Ownership is unchanged by dtype: every operation allocates a fresh
    owning output that aliases neither operand, an input view frees no
    shared storage, and closing an output does not close an input."""
    open_ids = set()
    original_init = cpp.NativeStorage.__init__
    original_close = cpp.NativeStorage.close

    def tracked_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        open_ids.add(id(self))

    def tracked_close(self):
        original_close(self)
        open_ids.discard(id(self))

    cpp.NativeStorage.__init__ = tracked_init
    cpp.NativeStorage.close = tracked_close
    try:
        baseline = len(open_ids)
        values = _sample(dtype, 24, seed=15).reshape(4, 6)
        partner = _sample(dtype, 24, seed=16).reshape(4, 6)
        for _ in range(40):
            a = _core(values, dtype)
            b = _core(partner, dtype)
            view = a.T
            summed = a.add(b)
            scaled = summed.multiply(b)
            rectified = view.relu()
            # The output owns storage the operands do not, and closing an
            # input leaves the result intact and readable.
            b.close()
            assert _same_bits(scaled.to_numpy(),
                              (values + partner) * partner, dtype)
            rectified.close()
            scaled.close()
            summed.close()
            view.close()
            a.close()
        assert len(open_ids) == baseline
    finally:
        cpp.NativeStorage.__init__ = original_init
        cpp.NativeStorage.close = original_close


@needs_native
@pytest.mark.parametrize("dtype", BOTH_DTYPES)
def test_a_failed_elementwise_output_allocation_leaks_nothing(dtype):
    """A failure in the output allocation closes everything it allocated, so
    live storage returns exactly to baseline and no caller can observe one
    lone half-built result — at either dtype."""
    open_ids = set()
    original_init = cpp.NativeStorage.__init__
    original_close = cpp.NativeStorage.close

    def tracked_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        open_ids.add(id(self))

    def tracked_close(self):
        original_close(self)
        open_ids.discard(id(self))

    cpp.NativeStorage.__init__ = tracked_init
    cpp.NativeStorage.close = tracked_close
    try:
        a = _core(_sample(dtype, 24, seed=17).reshape(4, 6), dtype)
        b = _core(_sample(dtype, 24, seed=18).reshape(4, 6), dtype)
        try:
            baseline = len(open_ids)
            for call in (lambda: a.add(b), lambda: a.T.relu(),
                         lambda: a.narrow(1, 1, 4).multiply(
                             b.narrow(1, 1, 4))):
                cpp._arm_alloc_failure(1)
                try:
                    with pytest.raises(MemoryError):
                        call()
                finally:
                    cpp._arm_alloc_failure(0)
            assert len(open_ids) == baseline
            # ...and the very next operation succeeds, so nothing latched.
            out = a.add(b)
            assert out.dtype == dtype
            out.close()
            assert len(open_ids) == baseline
        finally:
            b.close()
            a.close()
    finally:
        cpp.NativeStorage.__init__ = original_init
        cpp.NativeStorage.close = original_close


@needs_native
@pytest.mark.parametrize("dtype", BOTH_DTYPES)
def test_relu_backward_is_dtype_general_at_the_core_layer(dtype):
    """``tf_core_relu_backward`` is a forward-shaped numerical primitive, not
    graph machinery, so I3 generalizes it with the rest of the elementwise
    family.

    That is emphatically **not** float32 autograd: the C++ primitive is
    dtype-general and no public float32 backward graph can reach it, which
    the companion test asserts directly.
    """
    floating = _DTYPE_BITS[dtype][2]
    x = _sample(dtype, 24, seed=19).reshape(4, 6)
    upstream = _sample(dtype, 24, seed=20).reshape(4, 6)
    a = _core(x, dtype)
    b = _core(upstream, dtype)
    try:
        out = a.relu_backward(b)
        try:
            assert out.dtype == dtype
            # x == 0 blocks, matching the Python Tensor's (x > 0) * grad.
            expected = np.where(x > floating(0), upstream,
                                floating(0)).astype(floating)
            assert _same_bits(out.to_numpy(), expected, dtype)
        finally:
            out.close()
    finally:
        b.close()
        a.close()


@needs_native
def test_public_tensor_construction_opened_at_i9_and_not_before():
    """The boundary I3 drew, restated where **I9** moved it to.

    A dtype-general Core primitive is a kernel, and I7's state-owning
    constructors are an experimental module surface; **public** float32
    tensor construction is neither, and it did not open until the registry
    moved. Through I8 this test required all three ``NativeTensor``
    factories to refuse; I9 is the milestone that opened them, so it now
    requires all three to succeed — the same three, so the surface covered
    is identical and only the expected answer changed.

    ``NativeParameter`` is deliberately *not* in this list: it gained
    ``dtype`` at I7, which is a different milestone, and its own guardrail
    is the closed ``I7_DTYPE_CONSTRUCTORS`` set — which I9 did **not**
    widen, and that is asserted below.
    """
    import tensorforge.experimental as experimental

    values = np.zeros((2, 2), dtype=np.float32)
    for build in (
        lambda: experimental.NativeTensor.from_array(values, dtype="float32"),
        lambda: experimental.NativeTensor.zeros((2, 2), dtype="float32"),
        lambda: experimental.NativeTensor.full((2, 2), 1.0, dtype="float32"),
    ):
        built = build()
        try:
            assert built.dtype == "float32"
            assert built.to_numpy().dtype == np.float32
            # A public leaf, not a private one: it is an ordinary tensor.
            assert built.requires_grad is False
        finally:
            built.close()
    # ...and the dtype-argument surface is still exactly the closed I7 set:
    # opening the registry gave no *new* class a dtype argument.
    assert _constructors_with_a_dtype_argument() == DTYPE_CONSTRUCTORS


def test_the_float32_elementwise_path_holds_no_hidden_float64():
    """The structural half of design §10.1, checked semantically over the
    source rather than by whitespace.

    A runtime test cannot carry this claim on its own, and saying so is more
    honest than inventing one that appears to. For a **single** correctly
    rounded operation — which is every operation in the I3 family, since each
    destination element is one add, subtract, multiply, comparison, division,
    or square root — computing in binary64 and rounding once to binary32 is
    *provably* indistinguishable from computing in binary32: binary64 carries
    53 bits, comfortably more than the 2p+2 = 50 that would be needed for a
    double rounding to differ. So the observable float32 result is asserted
    to be exactly the binary32 result (the oracle tests above), and the
    absence of a widening intermediate is asserted here, where it is a
    property of the code.

    What this does rule out is the thing that *would* be observable once
    accumulation arrives: a double accumulator, a double constant that
    promotes an expression, or a ``static_cast<double>`` in a traversal.
    """
    header = (REPO_ROOT / "cpp" / "include"
              / "tf_elementwise_internal.h").read_text(encoding="utf-8")
    source = (REPO_ROOT / "cpp" / "src" / "elementwise.cpp").read_text(
        encoding="utf-8")

    def strip_comments(text):
        text = re.sub(r"/\*.*?\*/", "", text, flags=re.S)
        return re.sub(r"//[^\n]*", "", text)

    header_code = strip_comments(header)
    source_code = strip_comments(source)

    # 1. Every operation functor is templated on the element type, and none
    #    of them mentions ``double`` at all.
    functors = ("AddOp", "SubtractOp", "MultiplyOp", "ReluOp",
                "ReluBackwardOp", "SqrtOp", "ReciprocalOp", "IdentityOp")
    for functor in functors:
        body = header_code.split(f"struct {functor}", 1)
        assert len(body) == 2, functor
        body = body[1].split("};", 1)[0]
        assert "template <class T>" in body, functor
        assert "double" not in body, functor
        assert "float" not in body, functor
        # The constants are typed, so nothing promotes around a literal.
        assert not re.search(r"\b\d+\.\d+\b", body), functor

    # 2. The traversals take ``T*`` rather than a fixed width, in both the
    #    unary and the binary direction.
    for walker in ("unary_row", "binary_row", "unary_plan_walk",
                   "binary_plan_walk"):
        signature = header_code.split(f"inline void {walker}(", 1)[1]
        signature = signature.split(")", 1)[0]
        assert "double" not in signature, walker
        assert "float" not in signature, walker
        assert "T*" in signature, walker

    # 3. Nothing in the elementwise translation unit widens or narrows.
    for banned in ("static_cast<double>", "static_cast<float>", "(double)",
                   "(float)", "double accumulator", "long double"):
        assert banned not in source_code, banned
    # ...and the retained walkers are written over ``T``, not over double.
    for walker in ("core_unary_typed", "core_binary_typed"):
        signature = source_code.split(f"void {walker}(", 1)[1].split(")", 1)[0]
        assert "double" not in signature, walker
        assert "float" not in signature, walker

    # 4. The dispatch happens exactly once per exported call: each dtype
    #    helper holds one switch, and no export body holds a second.
    for helper in ("unary_by_dtype", "unary_contiguous_by_dtype",
                   "binary_by_dtype", "binary_contiguous_by_dtype"):
        body = source_code.split(f"void {helper}(", 1)[1].split("\n}\n", 1)[0]
        assert body.count("switch (") == 1, helper
        assert "Dtype::Float32" in body and "Dtype::Float64" in body, helper
        # No default label, so a future dtype without an instantiation is a
        # compile-time problem rather than a silent misread.
        assert "default:" not in body, helper

    # 5. Every generalized export performs at most one dtype switch of its
    #    own, and no element loop can contain one.
    for name in re.findall(r"TF_EXPORT void (tf_core_\w+)\(", source_code):
        body = source_code.split(f"TF_EXPORT void {name}(", 1)[1]
        body = body.split("TF_GUARD_END", 1)[0]
        assert body.count("switch (") <= 1, name


def test_the_elementwise_exports_use_the_matching_dtype_guard():
    """Structural: every export in the I3 family names
    ``require_matching_dtype`` and none of them still names
    ``require_float64`` — the two say opposite things about whether an
    operation has been generalized, so a leftover would be a real
    contradiction rather than a stylistic one."""
    source = (REPO_ROOT / "cpp" / "src" / "elementwise.cpp").read_text(
        encoding="utf-8")
    generalized = (
        "tf_core_relu", "tf_core_relu_contiguous",
        "tf_core_sqrt", "tf_core_sqrt_contiguous",
        "tf_core_reciprocal", "tf_core_reciprocal_contiguous",
        "tf_core_exp", "tf_core_exp_contiguous",
        "tf_core_log", "tf_core_log_contiguous",
        "tf_core_contiguous_copy",
        "tf_core_add", "tf_core_add_contiguous",
        "tf_core_subtract", "tf_core_subtract_contiguous",
        "tf_core_multiply", "tf_core_multiply_contiguous",
        "tf_core_relu_backward",
    )
    for name in generalized:
        body = source.split(f"TF_EXPORT void {name}(", 1)[1]
        body = body.split("\n}\n", 1)[0]
        assert "require_matching_dtype" in body, name
        assert "require_float64" not in body, name
    # ...and the export count did not move: generalization ships inside the
    # symbols Python already declares.
    assert len(_source_exports()) == I1_EXPORT_COUNT


@needs_native
def test_i3_moved_no_public_capability_at_all():
    """The exit gate, as one assertion block: internal float32 elementwise
    execution exists and public float32 support does not."""
    from tensorforge.experimental import native_checkpoint

    _assert_the_public_registry_is_i9s()
    assert cpp.RAW_KERNEL_DTYPES == ("float64",)
    assert cpp.backend_info()["dtype"] == "float64"
    assert native_checkpoint._FORMAT_VERSION == I8_CHECKPOINT_VERSION
    assert (native_checkpoint._SUPPORTED_FORMAT_VERSIONS
            == I8_CHECKPOINT_VERSIONS)
    exports = _source_exports()
    assert len(exports) == I1_EXPORT_COUNT       # still 54; I3 adds none
    for absent in ("tf_core_add_f32", "tf_core_relu_f32", "tf_core_add_float32",
                   "tf_core_multiply_f64", "tf_storage_dtype",
                   "tf_storage_cast", "tf_dtype_item_size"):
        assert absent not in exports, absent
    # No per-dtype sibling of any kind crept in.
    assert not [name for name in exports
                if name.endswith(("_f32", "_f64", "_float32", "_float64"))]


# ===========================================================================
# I4: reductions, matmul, views, and core autograd, as running code
#
# Everything below drives the **live** library at both dtypes. float32 is
# still not a supported TensorForge dtype and no public constructor can
# produce one, so these reach it through the private typed constructors the
# rollout rule (design §27.2) requires an intermediate milestone to test
# through.
#
# The comparison rules are, again, per family and stated rather than
# inherited:
#
#   * a reduction is compared against an **independent sequential oracle at
#     the same dtype** — never against a float64 result, which §10.4
#     forbids making a contract of;
#   * matmul is compared against the same-dtype textbook triple loop, and
#     H2's four parts are restated for float32 rather than assumed;
#   * ``narrow_backward`` assigns, so it is compared by raw bit pattern
#     including the zeros it does not write;
#   * gradients are compared analytically where a formula supports it and by
#     finite differences where it does not, with a binary32 step and stated
#     tolerances.
# ===========================================================================


def _live_storage_ids(monkeypatch):
    """The ids of every open NativeStorage, so an ownership claim can be
    proved against a real allocation count rather than trusting collection.
    The same fixture shape tests/test_native_abi_boundary.py uses."""
    open_ids = set()
    original_init = cpp.NativeStorage.__init__
    original_close = cpp.NativeStorage.close

    def tracked_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        open_ids.add(id(self))

    def tracked_close(self):
        original_close(self)
        open_ids.discard(id(self))

    monkeypatch.setattr(cpp.NativeStorage, "__init__", tracked_init)
    monkeypatch.setattr(cpp.NativeStorage, "close", tracked_close)
    return open_ids


def _tensor(values, dtype, requires_grad=False):
    """A NativeTensor over ``values`` at ``dtype``.

    float64 goes through the **public** constructor, so half of every test
    below exercises the shipped path. float32 goes through the private
    ``_from_core`` over a private typed core — the narrowest mechanism that
    already exists (it is how every op result is wrapped), with no new
    public surface, no bypass flag, and no change to what
    ``NativeTensor.from_array(dtype="float32")`` does, which is still raise.
    """
    from tensorforge.experimental import NativeTensor

    array = np.ascontiguousarray(values, dtype=_DTYPE_BITS[dtype][2])
    if dtype == "float64":
        return NativeTensor.from_array(array, requires_grad=requires_grad)
    tensor = NativeTensor._from_core(
        cpp.NativeTensorCore._typed_from_array(array, dtype))
    tensor._init_requires_grad(requires_grad)
    return tensor


def _sequential_sum(values, axis, dtype):
    """The reduction oracle: accumulate in row-major source order, one
    addition at a time, **at the element dtype**.

    Written with an explicit Python loop over 0-d NumPy scalars rather than
    ``ndarray.sum``, because NumPy's own reduction is free to pairwise-block
    its accumulation — which is exactly the reassociation TensorForge
    promises not to do, so ``np.sum`` is the wrong oracle for a bit-level
    claim at binary32. This is not a float64 comparison in disguise: every
    intermediate here is a ``numpy.float32`` (or ``float64``) scalar.
    """
    floating = _DTYPE_BITS[dtype][2]
    values = np.ascontiguousarray(values, dtype=floating)
    if axis is None:
        total = floating(0)
        for value in values.ravel(order="C"):
            total = floating(total + value)
        return np.array(total, dtype=floating)
    axis = axis % values.ndim
    moved = np.moveaxis(values, axis, -1)
    out = np.zeros(moved.shape[:-1], dtype=floating)
    for index in np.ndindex(*moved.shape[:-1]):
        total = floating(0)
        for value in moved[index]:
            total = floating(total + value)
        out[index] = total
    return out


def _sequential_matmul(a, b, dtype):
    """The matmul oracle: for each output cell, accumulate over ``k`` in
    ascending order at the element dtype. ``a @ b`` is **not** usable here —
    NumPy dispatches to a BLAS GEMM that blocks and reassociates."""
    floating = _DTYPE_BITS[dtype][2]
    a = np.ascontiguousarray(a, dtype=floating)
    b = np.ascontiguousarray(b, dtype=floating)
    m, n = a.shape
    p = b.shape[1]
    out = np.zeros((m, p), dtype=floating)
    # ``inf * 0`` and ``inf + -inf`` are *values* here, not failures — the
    # exceptional-value cases below rely on them — so NumPy's warnings are
    # silenced rather than the inputs being restricted.
    with np.errstate(all="ignore"):
        for i in range(m):
            for j in range(p):
                total = floating(0)
                for k in range(n):
                    total = floating(total + floating(a[i, k] * b[k, j]))
                out[i, j] = total
    return out


# ---------------------------------------------------------------------------
# 1. Reductions at both dtypes
# ---------------------------------------------------------------------------

@needs_native
@pytest.mark.parametrize("dtype", BOTH_DTYPES)
@pytest.mark.parametrize("axis,keepdims", [
    (None, False), (0, False), (0, True), (1, False), (1, True),
    (-1, False), (-2, True),
])
def test_sum_matches_the_same_dtype_sequential_oracle(dtype, axis, keepdims):
    """Every supported axis form, at both dtypes, against an independent
    sequential oracle **at the same width** — compared by raw bit pattern,
    because the accumulation order is the value's definition and a tolerance
    would not see a reassociation."""
    values = _sample(dtype, 24).reshape(4, 6)
    core = _core(values, dtype)
    try:
        out = core.sum(axis=axis, keepdims=keepdims)
        try:
            assert out.dtype == dtype
            expected = _sequential_sum(values, axis, dtype)
            if keepdims and axis is not None:
                expected = np.expand_dims(expected, axis % values.ndim)
            assert out.shape == expected.shape
            assert _same_bits(out.to_numpy(), expected, dtype), (
                dtype, axis, keepdims)
        finally:
            out.close()
    finally:
        core.close()


@needs_native
@pytest.mark.parametrize("dtype", BOTH_DTYPES)
def test_sum_reads_every_layout_at_both_dtypes(dtype):
    """The layouts the H6 predicate accepts and the ones it declines, at
    both widths: contiguous (block traversal), transposed and narrowed (the
    retained odometer). Each is compared against the sequential oracle for
    the values that layout actually presents, so the two shipped traversals
    are both proved rather than only the fast one."""
    values = _sample(dtype, 24).reshape(4, 6)
    owner = _core(values, dtype)
    try:
        cases = (
            ("contiguous", owner, values),
            ("transposed", owner.T, np.ascontiguousarray(values.T)),
            ("narrowed", owner.narrow(1, 2, 3),
             np.ascontiguousarray(values[:, 2:5])),
            ("narrowed rows", owner.narrow(0, 1, 2),
             np.ascontiguousarray(values[1:3, :])),
        )
        for label, view, host in cases:
            for axis in (None, 0, 1):
                out = view.sum(axis=axis)
                try:
                    assert out.dtype == dtype, label
                    assert _same_bits(out.to_numpy(),
                                      _sequential_sum(host, axis, dtype),
                                      dtype), (label, axis, dtype)
                finally:
                    out.close()
            if view is not owner:
                view.close()
    finally:
        owner.close()


@needs_native
@pytest.mark.parametrize("dtype", BOTH_DTYPES)
def test_sum_of_a_rank_three_source_at_both_dtypes(dtype):
    values = _sample(dtype, 24).reshape(2, 3, 4)
    core = _core(values, dtype)
    try:
        for axis in (None, 0, 1, 2, -1):
            for keepdims in (False, True):
                out = core.sum(axis=axis, keepdims=keepdims)
                try:
                    expected = _sequential_sum(values, axis, dtype)
                    if keepdims:
                        expected = (expected.reshape((1, 1, 1))
                                    if axis is None
                                    else np.expand_dims(expected, axis % 3))
                    assert out.dtype == dtype
                    assert out.shape == expected.shape
                    assert _same_bits(out.to_numpy(), expected, dtype)
                finally:
                    out.close()
    finally:
        core.close()


@needs_native
@pytest.mark.parametrize("dtype", BOTH_DTYPES)
def test_sum_signed_zeros_are_compared_as_raw_bits(dtype):
    """A run of ``-0.0`` sums to ``+0.0`` at both widths: the destination
    starts at the additive identity a zeroed buffer holds, and
    ``+0.0 + -0.0`` is ``+0.0``. Asserted on the bit pattern, because ``==``
    cannot see a zero's sign."""
    floating = _DTYPE_BITS[dtype][2]
    values = np.full((3, 4), floating(-0.0), dtype=floating)
    core = _core(values, dtype)
    try:
        for axis in (None, 0, 1):
            out = core.sum(axis=axis)
            try:
                produced = out.to_numpy()
                assert np.all(_bits(produced, dtype) == 0), (dtype, axis)
            finally:
                out.close()
    finally:
        core.close()


# ---------------------------------------------------------------------------
# 2. The float32 accumulation witness — the claim I3 could not make
# ---------------------------------------------------------------------------
#
# I3 recorded, and this milestone inherits, that "float32 is not secretly
# float64" could not rest on a runtime test *there*: every I3 operation
# produced each destination element with a single correctly-rounded IEEE
# operation, and computing in binary64 and rounding once is provably
# indistinguishable from computing in binary32 for those. A **sum** of three
# or more values is the first place the difference becomes observable, and
# I4 is where the structural argument acquires a behavioural partner.

# 1.0 followed by eight copies of 2**-24. In binary32 each addend is exactly
# half an ULP of 1.0, so round-to-nearest-even leaves the running total at
# exactly 1.0 forever. In binary64 they accumulate, and the single final
# narrowing lands four ULPs above. Deterministic, exactly representable, and
# independent of the toolchain and the machine.
_F32_ABSORPTION = np.array([1.0] + [2.0 ** -24] * 8, dtype=np.float32)


@needs_native
def test_the_float32_sum_witness_distinguishes_the_accumulation_policies():
    """The witness is a witness: sequential binary32 and narrowed-binary64
    genuinely disagree on this vector, so the assertions below cannot pass
    vacuously."""
    sequential = np.asarray(_sequential_sum(_F32_ABSORPTION, None, "float32"),
                            dtype=np.float32)
    widened = np.asarray(np.float32(np.float64(_F32_ABSORPTION).sum()),
                         dtype=np.float32)
    assert int(_bits(sequential, "float32").ravel()[0]) == 0x3F800000
    assert int(_bits(widened, "float32").ravel()[0]) == 0x3F800004
    assert not _same_bits(sequential, widened, "float32")


@needs_native
@pytest.mark.parametrize("layout", ("contiguous", "transposed"))
def test_float32_sum_accumulates_in_float32_not_in_a_widened_accumulator(
        layout):
    """TensorForge matches the sequential binary32 answer and **differs from
    the widened one**, on both shipped traversals.

    ``layout`` is what selects the traversal: a contiguous row-major source
    takes H6's block walk (whose local accumulator is the one that could
    most plausibly have been left ``double``), and a transposed view is
    declined by the predicate and takes the retained odometer. A hidden
    widening accumulator introduced on only one of them would fail exactly
    one of these two parametrizations, which is the plausible mistake.
    """
    sequential = np.asarray(_sequential_sum(_F32_ABSORPTION, None, "float32"),
                            dtype=np.float32)
    widened = np.asarray(np.float32(np.float64(_F32_ABSORPTION).sum()),
                         dtype=np.float32)
    if layout == "contiguous":
        core = _core(_F32_ABSORPTION, "float32")
        view = core
    else:
        # A (9, 1) column read through a transposed view: the same nine
        # values in the same order, on a layout the predicate declines.
        core = _core(_F32_ABSORPTION.reshape(1, 9), "float32")
        view = core.T
    try:
        out = view.sum()
        try:
            produced = out.to_numpy()
            assert _same_bits(produced, sequential, "float32"), (
                f"{layout}: float32 sum did not accumulate in float32")
            assert not _same_bits(produced, widened, "float32"), (
                f"{layout}: float32 sum produced the WIDENED result — there "
                f"is a hidden binary64 accumulator")
        finally:
            out.close()
    finally:
        if view is not core:
            view.close()
        core.close()


@needs_native
def test_float32_matmul_accumulates_in_float32_on_both_paths():
    """The same witness through matmul's ``k`` accumulator, on both shipped
    paths: ``p == 1`` is below MATMUL_MIN_COLUMNS and takes the retained
    generic kernel, ``p == 8`` takes the H2 row sweep."""
    sequential = np.asarray(_sequential_sum(_F32_ABSORPTION, None, "float32"),
                            dtype=np.float32)
    widened = np.asarray(np.float32(np.float64(_F32_ABSORPTION).sum()),
                         dtype=np.float32)
    ones = _core(np.ones((1, 9), dtype=np.float32), "float32")
    try:
        for p, label in ((1, "generic (p < 8)"), (8, "row sweep (p >= 8)")):
            column = _core(np.repeat(_F32_ABSORPTION.reshape(9, 1), p, axis=1),
                           "float32")
            try:
                out = ones.matmul(column)
                try:
                    produced = out.to_numpy()
                    for j in range(p):
                        cell = np.asarray(produced[0, j], dtype=np.float32)
                        assert _same_bits(cell, sequential, "float32"), label
                        assert not _same_bits(cell, widened, "float32"), (
                            f"{label}: the WIDENED result — there is a hidden "
                            f"binary64 accumulator")
                finally:
                    out.close()
            finally:
                column.close()
    finally:
        ones.close()


@needs_native
def test_the_mean_scale_factor_is_narrowed_once_before_the_loop():
    """Design §7.4, as a behavioural assertion rather than a comment.

    ``1/count`` is computed once in binary64 and narrowed **once** to the
    element type before the multiply loop. The distinction that makes this
    testable: for count == 3 and a sum of 5, ``float(5) * float(1/3)`` is
    1.6666667461395264 while ``float(double(5) * (1/3))`` is
    1.6666666269302368 — one representable step apart, so the two orders are
    separable. TensorForge must produce the first.
    """
    factor = 1.0 / 3.0
    specified = np.asarray(np.float32(5.0) * np.float32(factor),
                           dtype=np.float32)
    alternative = np.asarray(np.float32(np.float64(5.0) * factor),
                             dtype=np.float32)
    assert not _same_bits(specified, alternative, "float32"), (
        "the scale witness is not a witness")

    # 1 + 2 + 2 is exactly 5 at binary32, so the mean is exactly the scaling
    # step and no accumulation rounding is in the way.
    core = _core(np.array([[1.0, 2.0, 2.0]], dtype=np.float32), "float32")
    try:
        out = core.mean(axis=1)
        try:
            produced = np.asarray(out.to_numpy()[0], dtype=np.float32)
            assert _same_bits(produced, specified, "float32"), (
                "mean did not narrow the scale factor before the loop")
            assert not _same_bits(produced, alternative, "float32"), (
                "mean multiplied in binary64 and narrowed afterwards")
        finally:
            out.close()
    finally:
        core.close()


@needs_native
@pytest.mark.parametrize("dtype", BOTH_DTYPES)
@pytest.mark.parametrize("axis", (None, 0, 1))
def test_mean_is_the_sum_scaled_at_the_element_dtype(dtype, axis):
    """``mean`` is exactly ``sum`` times the once-narrowed reciprocal, at
    both widths, compared by bit pattern."""
    floating = _DTYPE_BITS[dtype][2]
    values = _sample(dtype, 24).reshape(4, 6)
    core = _core(values, dtype)
    try:
        count = values.size if axis is None else values.shape[axis]
        expected = (_sequential_sum(values, axis, dtype)
                    * floating(1.0 / count)).astype(floating)
        out = core.mean(axis=axis)
        try:
            assert out.dtype == dtype
            assert _same_bits(out.to_numpy(), expected, dtype), (dtype, axis)
        finally:
            out.close()
    finally:
        core.close()


# ---------------------------------------------------------------------------
# 3. Matmul at both dtypes
# ---------------------------------------------------------------------------

@needs_native
@pytest.mark.parametrize("dtype", BOTH_DTYPES)
@pytest.mark.parametrize("m,n,p", [
    (1, 1, 8), (4, 4, 8), (5, 3, 9), (7, 6, 8), (2, 9, 16),
    (9, 2, 8), (3, 3, 7), (3, 3, 8), (1, 5, 8), (6, 6, 6),
])
def test_matmul_matches_the_same_dtype_sequential_oracle(dtype, m, n, p):
    """Tall, wide, square, and small shapes, and both sides of the
    ``p >= 8`` predicate boundary, at both dtypes — compared by raw bit
    pattern against the same-dtype textbook triple loop.

    ``a @ b`` is deliberately not the oracle: NumPy dispatches matmul to a
    blocked BLAS GEMM whose accumulation order is not TensorForge's, so it
    would be the wrong reference for a bit-level claim at either width.
    """
    a = _sample(dtype, m * n, seed=41).reshape(m, n)
    b = _sample(dtype, n * p, seed=42).reshape(n, p)
    left = _core(a, dtype)
    right = _core(b, dtype)
    try:
        out = left.matmul(right)
        try:
            assert out.dtype == dtype
            assert out.shape == (m, p)
            assert _same_bits(out.to_numpy(), _sequential_matmul(a, b, dtype),
                              dtype), (dtype, m, n, p)
        finally:
            out.close()
    finally:
        right.close()
        left.close()


@needs_native
@pytest.mark.parametrize("dtype", BOTH_DTYPES)
def test_matmul_reads_non_contiguous_operands_at_both_dtypes(dtype):
    """A transposed right operand is exactly the case H2's predicate
    declines, so this drives the retained generic kernel at both widths; a
    transposed left operand keeps the row sweep. Neither is materialized
    first — the kernel addresses each source through its own strides."""
    a = _sample(dtype, 12, seed=51).reshape(3, 4)
    b = _sample(dtype, 32, seed=52).reshape(8, 4)
    c = _sample(dtype, 24, seed=53).reshape(3, 8)
    left = _core(a, dtype)
    right = _core(b, dtype)
    wide = _core(c, dtype)
    try:
        # (3, 4) @ (4, 8) with b transposed: b_stride1 != 1, generic path.
        out = left.matmul(right.T)
        try:
            assert out.dtype == dtype
            assert _same_bits(
                out.to_numpy(),
                _sequential_matmul(a, np.ascontiguousarray(b.T), dtype),
                dtype)
        finally:
            out.close()
        # (4, 3) @ (3, 8) with a transposed: b stays row-major, row sweep.
        out = left.T.matmul(wide)
        try:
            assert _same_bits(
                out.to_numpy(),
                _sequential_matmul(np.ascontiguousarray(a.T), c, dtype),
                dtype)
        finally:
            out.close()
    finally:
        wide.close()
        right.close()
        left.close()


@needs_native
@pytest.mark.parametrize("dtype", BOTH_DTYPES)
def test_matmul_signed_zeros_and_infinities_at_both_dtypes(dtype):
    """H2's contract parts 2 and 3, restated per dtype: signed zeros survive
    as raw bit patterns, infinities propagate, and every NaN a matmul
    produces is quiet."""
    floating = _DTYPE_BITS[dtype][2]
    quiet = _DTYPE_BITS[dtype][1](
        0x00400000 if dtype == "float32" else 0x0008000000000000)
    a = np.array([[-1.0, 0.0], [np.inf, 1.0]], dtype=floating)
    b = np.array([[0.0] * 8, [-0.0] * 8], dtype=floating)
    left = _core(a, dtype)
    right = _core(b, dtype)
    try:
        out = left.matmul(right)
        try:
            produced = out.to_numpy()
            expected = _sequential_matmul(a, b, dtype)
            assert np.array_equal(np.isnan(produced), np.isnan(expected))
            finite = ~np.isnan(produced)
            assert np.array_equal(
                _bits(np.ascontiguousarray(produced[finite]), dtype),
                _bits(np.ascontiguousarray(expected[finite]), dtype))
            nans = _bits(produced, dtype)[np.isnan(produced)]
            assert np.all((nans & quiet) != 0), "a matmul NaN was signalling"
        finally:
            out.close()
    finally:
        right.close()
        left.close()


# ---------------------------------------------------------------------------
# 4. narrow_backward at both dtypes
# ---------------------------------------------------------------------------

@needs_native
@pytest.mark.parametrize("dtype", BOTH_DTYPES)
@pytest.mark.parametrize("dim,start,length,original", [
    (0, 1, 2, (4, 3)), (1, 0, 2, (4, 3)), (1, 1, 2, (4, 3)),
    (0, 0, 4, (4, 3)),
])
def test_narrow_backward_scatters_at_both_dtypes(dtype, dim, start, length,
                                                 original):
    """The scatter writes only the narrowed region and leaves every other
    cell holding the zero the allocation gave it — and that zero *is* the
    gradient, which is why H1 rejected this destination from the
    uninitialized path. Compared by raw bit pattern, zeros included."""
    floating = _DTYPE_BITS[dtype][2]
    shape = tuple(length if axis == dim else size
                  for axis, size in enumerate(original))
    upstream = _sample(dtype, int(np.prod(shape))).reshape(shape)
    core = _core(upstream, dtype)
    try:
        out = core.narrow_backward(dim, start, original)
        try:
            assert out.dtype == dtype
            assert out.shape == original
            expected = np.zeros(original, dtype=floating)
            index = [slice(None)] * len(original)
            index[dim] = slice(start, start + length)
            expected[tuple(index)] = upstream
            assert _same_bits(out.to_numpy(), expected, dtype)
        finally:
            out.close()
    finally:
        core.close()


@needs_native
@pytest.mark.parametrize("dtype", BOTH_DTYPES)
def test_narrow_backward_assigns_rather_than_computes(dtype):
    """It is a transfer into a larger zeroed result, not an operation: a
    ``-0.0`` in the upstream stays negative and a signalling NaN stays
    signalling, at both widths. An arithmetic composition would normalize
    the first and quiet the second."""
    upstream = _patterned(dtype, 6).reshape(2, 3)
    core = _core(upstream, dtype)
    try:
        out = core.narrow_backward(0, 1, (4, 3))
        try:
            produced = out.to_numpy()
            assert np.array_equal(
                _bits(np.ascontiguousarray(produced[1:3]), dtype),
                _bits(upstream, dtype))
            untouched = np.ascontiguousarray(
                np.concatenate([produced[0:1], produced[3:4]]))
            assert np.all(_bits(untouched, dtype) == 0)
        finally:
            out.close()
    finally:
        core.close()


# ---------------------------------------------------------------------------
# 5. The private float32 autograd graph
# ---------------------------------------------------------------------------

@needs_native
def test_the_private_float32_graph_runs_forward_and_backward():
    """A composition touching every family I4 opened — elementwise,
    broadcasting, a reduction, and a matmul — differentiated end to end at
    float32, with every gradient checked analytically."""
    x = _tensor([[1.0, 2.0], [3.0, 4.0]], "float32", requires_grad=True)
    w = _tensor([[0.5], [-1.5]], "float32", requires_grad=True)
    bias = _tensor([[0.25]], "float32", requires_grad=True)
    try:
        out = x.matmul(w).add(bias).sum()
        try:
            out.backward()
        finally:
            out.close()
        # d/dx (x @ w + bias) summed = each row of w.T
        assert np.array_equal(x.grad.to_numpy(),
                              np.array([[0.5, -1.5], [0.5, -1.5]],
                                       dtype=np.float32))
        # d/dw = column sums of x
        assert np.array_equal(w.grad.to_numpy(),
                              np.array([[4.0], [6.0]], dtype=np.float32))
        # d/dbias = the number of broadcast positions
        assert np.array_equal(bias.grad.to_numpy(),
                              np.array([[2.0]], dtype=np.float32))
        for tensor in (x, w, bias):
            assert tensor.grad.dtype == "float32"
            assert tensor.grad.shape == tensor.shape
    finally:
        bias.close()
        w.close()
        x.close()


@needs_native
@pytest.mark.parametrize("dtype", BOTH_DTYPES)
@pytest.mark.parametrize("op", [
    "add", "subtract", "multiply", "relu", "sqrt", "reciprocal", "exp", "log",
    "sum", "mean", "matmul", "reshape", "transpose", "T", "narrow",
    "contiguous_copy", "broadcast",
])
def test_every_i4_operation_produces_a_gradient_of_the_graph_dtype(dtype, op):
    """Invariants 1 and 2 of design §11, over the whole I4 operation set at
    both widths: a gradient has the dtype of the tensor it is a gradient of,
    and a leaf's gradient matches the leaf. Nothing in a backward may
    introduce a tensor of another width — which, since the runtime never
    casts, would not merely be untidy but would raise at the first
    accumulation."""
    floating = _DTYPE_BITS[dtype][2]
    # Values chosen inside every domain in play at once: strictly positive
    # (so ``sqrt``/``log`` are differentiable and ``reciprocal`` is finite)
    # and away from 0 (so ``relu`` is not at its kink).
    x = _tensor(np.array([[1.5, 2.5], [3.5, 4.5]], dtype=floating), dtype,
                requires_grad=True)
    other = _tensor(np.array([[0.5, 1.5], [2.5, 3.5]], dtype=floating), dtype,
                    requires_grad=True)
    row = _tensor(np.array([[2.0, 3.0]], dtype=floating), dtype,
                  requires_grad=True)
    created = [x, other, row]
    try:
        if op in ("add", "subtract", "multiply", "matmul"):
            result = getattr(x, op)(other)
        elif op == "broadcast":
            result = x.multiply(row)          # (2, 2) * (1, 2)
        elif op == "reshape":
            result = x.reshape((4,))
        elif op == "transpose":
            result = x.transpose(1, 0)
        elif op == "T":
            result = x.T
        elif op == "narrow":
            result = x.narrow(0, 0, 1)
        elif op == "contiguous_copy":
            result = x.contiguous_copy()
        elif op in ("sum", "mean"):
            result = getattr(x, op)(axis=0)
        else:
            result = getattr(x, op)()
        created.append(result)
        assert result.dtype == dtype
        loss = result.sum()
        created.append(loss)
        loss.backward()
        assert x.grad is not None
        assert x.grad.dtype == dtype, op
        assert x.grad.shape == x.shape, op
        if op in ("add", "subtract", "multiply", "matmul"):
            assert other.grad.dtype == dtype, op
            assert other.grad.shape == other.shape, op
        if op == "broadcast":
            assert row.grad.dtype == dtype and row.grad.shape == (1, 2)
    finally:
        for tensor in reversed(created):
            tensor.close()


@needs_native
@pytest.mark.parametrize("dtype", BOTH_DTYPES)
def test_reduction_backward_broadcasts_back_at_the_graph_dtype(dtype):
    """sum and mean backward, over every axis form: the upstream is
    reshaped to the keepdims-compatible shape and expanded back to the input
    shape, at the graph's dtype, with the mean case scaled by the
    once-narrowed reciprocal."""
    floating = _DTYPE_BITS[dtype][2]
    values = np.arange(1.0, 13.0, dtype=floating).reshape(3, 4)
    for reduction in ("sum", "mean"):
        for axis, keepdims in ((None, False), (0, False), (1, True),
                               (-1, False)):
            x = _tensor(values, dtype, requires_grad=True)
            try:
                out = getattr(x, reduction)(axis=axis, keepdims=keepdims)
                try:
                    if out.numel == 1:
                        out.backward()
                    else:
                        loss = out.sum()
                        try:
                            loss.backward()
                        finally:
                            loss.close()
                finally:
                    out.close()
                count = (values.size if axis is None
                         else values.shape[axis % values.ndim])
                factor = (floating(1.0) if reduction == "sum"
                          else floating(1.0 / count))
                expected = np.full(values.shape, factor, dtype=floating)
                assert x.grad.dtype == dtype
                assert x.grad.shape == values.shape
                assert _same_bits(x.grad.to_numpy(), expected, dtype), (
                    reduction, axis, keepdims, dtype)
            finally:
                x.close()


@needs_native
@pytest.mark.parametrize("dtype", BOTH_DTYPES)
def test_view_backward_preserves_the_graph_dtype_and_owns_its_storage(dtype):
    """reshape, transpose, ``.T``, narrow, contiguous_copy, and a chain of
    them: each backward produces a gradient of the graph dtype, of the
    input's shape, in **independent owning** storage — reading the leaf
    after the gradient exists must still give the leaf's own value."""
    floating = _DTYPE_BITS[dtype][2]
    values = np.arange(1.0, 13.0, dtype=floating).reshape(3, 4)
    builders = {
        "reshape": lambda t: t.reshape((12,)),
        "transpose": lambda t: t.transpose(1, 0),
        "T": lambda t: t.T,
        "narrow": lambda t: t.narrow(1, 1, 2),
        "contiguous_copy": lambda t: t.contiguous_copy(),
        "chained": lambda t: t.contiguous_copy().T.narrow(1, 1, 2),
    }
    for label, build in builders.items():
        x = _tensor(values, dtype, requires_grad=True)
        try:
            view = build(x)
            try:
                loss = view.sum()
                try:
                    loss.backward()
                finally:
                    loss.close()
            finally:
                view.close()
            assert x.grad is not None, label
            assert x.grad.dtype == dtype, label
            assert x.grad.shape == values.shape, label
            assert x.grad.to_numpy().dtype == np.dtype(floating), label
            # The gradient owns independent storage: the leaf is unchanged.
            assert np.array_equal(x.to_numpy(), values), label
            # d(sum(view))/dx is 1 on every visited element and 0 elsewhere.
            if label == "narrow":
                expected = np.zeros_like(values)
                expected[:, 1:3] = floating(1.0)
            elif label == "chained":
                expected = np.zeros_like(values)
                expected[1:3, :] = floating(1.0)
            else:
                expected = np.ones_like(values)
            assert _same_bits(x.grad.to_numpy(), expected, dtype), label
        finally:
            x.close()


@needs_native
@pytest.mark.parametrize("dtype", BOTH_DTYPES)
def test_broadcast_backward_is_a_reduction_at_the_graph_dtype(dtype):
    """The adjoint of broadcasting really sums the stretched axes — it is
    not a copy — and it does so at the graph's dtype, for leading
    dimensions, singleton axes, scalars, and higher ranks."""
    floating = _DTYPE_BITS[dtype][2]
    cases = (
        ((2, 3), (1, 3), np.full((1, 3), 2.0)),
        ((2, 3), (2, 1), np.full((2, 1), 3.0)),
        ((2, 3), (3,), np.full((3,), 2.0)),
        ((2, 3), (), np.array(6.0)),
        ((4, 2, 3), (2, 1), np.full((2, 1), 12.0)),
    )
    for big_shape, small_shape, expected in cases:
        big = _tensor(np.ones(big_shape, dtype=floating), dtype,
                      requires_grad=True)
        small = _tensor(np.ones(small_shape, dtype=floating), dtype,
                        requires_grad=True)
        try:
            out = big.add(small)
            try:
                loss = out.sum()
                try:
                    loss.backward()
                finally:
                    loss.close()
            finally:
                out.close()
            assert small.grad.dtype == dtype
            # The gradient has the *operand's* shape, whatever normalization
            # the constructor applied to a rank-0 request.
            assert small.grad.shape == small.shape
            assert _same_bits(
                small.grad.to_numpy(),
                expected.astype(floating).reshape(small.shape), dtype), (
                big_shape, small_shape, dtype)
            assert big.grad.dtype == dtype
            assert big.grad.shape == tuple(big_shape)
        finally:
            small.close()
            big.close()


@needs_native
@pytest.mark.parametrize("dtype", BOTH_DTYPES)
def test_matmul_backward_keeps_every_operand_at_the_graph_dtype(dtype):
    """``dA = upstream @ B.T`` and ``dB = A.T @ upstream``, unchanged
    structurally, at both widths — with the transposes still metadata-only
    views and the accumulation still in the element type."""
    floating = _DTYPE_BITS[dtype][2]
    a_host = np.arange(1.0, 7.0, dtype=floating).reshape(2, 3)
    b_host = np.arange(1.0, 25.0, dtype=floating).reshape(3, 8)
    upstream_host = np.full((2, 8), 2.0, dtype=floating)
    a = _tensor(a_host, dtype, requires_grad=True)
    b = _tensor(b_host, dtype, requires_grad=True)
    upstream = _tensor(upstream_host, dtype)
    try:
        out = a.matmul(b)
        try:
            out.backward(gradient=upstream)
        finally:
            out.close()
        assert a.grad.dtype == dtype and b.grad.dtype == dtype
        assert _same_bits(
            a.grad.to_numpy(),
            _sequential_matmul(upstream_host,
                               np.ascontiguousarray(b_host.T), dtype), dtype)
        assert _same_bits(
            b.grad.to_numpy(),
            _sequential_matmul(np.ascontiguousarray(a_host.T), upstream_host,
                               dtype), dtype)
    finally:
        upstream.close()
        b.close()
        a.close()


# ---------------------------------------------------------------------------
# 6. Backward constants are built at the graph dtype
# ---------------------------------------------------------------------------

@needs_native
def test_no_backward_constant_is_created_at_a_hard_coded_float64():
    """Design §11.4, structurally.

    A backward materializes constants — ``0.5`` for sqrt, ``-1`` for
    reciprocal and for the negation subtract needs, ``1/count`` for mean,
    the ones seed, the zeros operand broadcast-back expands into. Every one
    of them must be built at the **operand's** dtype. The public ``full`` /
    ``zeros`` constructors normalize against the public registry and would
    therefore raise on a float32 graph, so a backward that used them would
    be float64-only by construction; the private typed constructors are what
    the autograd layer reaches for instead.

    Asserted over the source so the rule survives a future edit that adds a
    constant, not only over the graphs this suite happens to run.
    """
    code = _read("src/tensorforge/experimental/native_tensor.py")
    # The two public dtype-defaulting constructors appear **once each**, and
    # both inside the public constructor block at the top of the class,
    # where a caller's own ``dtype`` argument is what they normalize. Every
    # line after the lifetime gate — all of the compute, every backward
    # closure, and every module-level gradient helper — has none.
    compute = code.split("# -- lifetime gate", 1)[1]
    for banned in ("NativeTensorCore.full(", 'dtype="float64"',
                   "dtype='float64'"):
        assert banned not in compute, (
            f"a backward constant is built through {banned!r}, which pins it "
            f"to float64")
    # ``zeros`` survives in the compute half — broadcast-back genuinely
    # expands into one — but **only** in its dtype-trusting form, which
    # takes the operand's tag instead of normalizing against the public
    # registry. Every occurrence is checked, not merely the first.
    for fragment in compute.split("NativeTensorCore.zeros(")[1:]:
        assert "_trusted_dtype=True" in fragment.split(")", 1)[0], (
            "a compute path builds zeros through the public dtype gate, "
            "which pins it to float64")
    # The public constructors appear exactly once each, in the constructor
    # block above the gate, where a caller's own ``dtype`` is what they
    # normalize.
    assert code.count("cpp.NativeTensorCore.full(") == 1     # NativeTensor.full
    # ctor + broadcast-back + the I7 private ``_typed_zeros``, which passes
    # ``_trusted_dtype=True`` and is therefore counted by the loop above.
    assert code.count("cpp.NativeTensorCore.zeros(") == 3
    # ...and the typed constant constructor really is what the backwards use.
    assert compute.count("NativeTensorCore._typed_full(") >= 4


@needs_native
@pytest.mark.parametrize("dtype", BOTH_DTYPES)
def test_the_backward_seed_has_the_output_dtype(dtype):
    """Invariant 1 of design §11 at the entry point: the implicit
    ``d(out)/d(out) = 1`` seed is created at the **output's** dtype. A
    float64 seed for a float32 output would be rejected by the very first
    accumulation — so this passing is itself the proof it is not one."""
    floating = _DTYPE_BITS[dtype][2]
    x = _tensor(np.array([2.0, 3.0], dtype=floating), dtype,
                requires_grad=True)
    try:
        out = x.sum()
        try:
            out.backward()
        finally:
            out.close()
        assert x.grad.dtype == dtype
        assert np.array_equal(x.grad.to_numpy(), np.ones(2, dtype=floating))
    finally:
        x.close()


@needs_native
@pytest.mark.parametrize("dtype", BOTH_DTYPES)
def test_subtract_and_sqrt_backward_constants_land_at_the_graph_dtype(dtype):
    """The two backwards that materialize a *numeric* constant rather than a
    ones/zeros tensor: subtract's ``-1`` negation and sqrt's ``0.5``. Both
    must produce a gradient of the graph dtype with the analytically correct
    value, compared as bits."""
    floating = _DTYPE_BITS[dtype][2]
    a = _tensor(np.array([4.0, 9.0], dtype=floating), dtype,
                requires_grad=True)
    b = _tensor(np.array([1.0, 2.0], dtype=floating), dtype,
                requires_grad=True)
    try:
        out = a.subtract(b).sum()
        try:
            out.backward()
        finally:
            out.close()
        assert b.grad.dtype == dtype
        assert _same_bits(b.grad.to_numpy(),
                          np.full(2, -1.0, dtype=floating), dtype)
    finally:
        b.close()
        a.close()

    c = _tensor(np.array([4.0, 16.0], dtype=floating), dtype,
                requires_grad=True)
    try:
        out = c.sqrt().sum()
        try:
            out.backward()
        finally:
            out.close()
        assert c.grad.dtype == dtype
        # d(sqrt(x))/dx = 1/(2*sqrt(x)); at 4 and 16 that is 0.25 and 0.125,
        # both exact at either width.
        assert _same_bits(c.grad.to_numpy(),
                          np.array([0.25, 0.125], dtype=floating), dtype)
    finally:
        c.close()


# ---------------------------------------------------------------------------
# 7. Finite differences at float32
# ---------------------------------------------------------------------------
#
# The step and the tolerances are chosen for **binary32** and stated here
# rather than inherited from the float64 gradient checks, which use a far
# smaller step and a far tighter bound.
#
# Why 2**-11 (about 4.9e-4). A central difference has two competing error
# terms: truncation, which is O(h**2 * f'''), and cancellation, which is
# O(eps/h) where eps is the unit roundoff — 2**-24 at binary32 against
# 2**-53 at binary64. Balancing them puts the optimum near eps**(1/3),
# which is about 6e-3 for binary32; a slightly smaller step keeps the
# truncation term comfortably small for the mildly nonlinear functions used
# here while staying far above the cancellation floor. It is also an exact
# power of two, so the perturbation itself introduces no rounding error.
#
# Why these tolerances. With h = 2**-11 the cancellation term alone is about
# 2**-24 / 2**-11 = 2**-13, roughly 1.2e-4 scaled by the magnitude of f, so
# a relative tolerance of 2e-2 with an absolute floor of 2e-3 is a
# comfortable but non-vacuous band: it fails immediately for a gradient that
# is wrong by a factor, a sign, or a transposition, which is what an
# analytical-formula error actually looks like. The exact analytical checks
# above are what pin the last bits; this is what pins the *formula*. The
# negative control below proves the band can reject.
#
# The values are finite, well conditioned, strictly positive, and far from
# ReLU's kink and from the singular domains of reciprocal and log.

_F32_FD_STEP = np.float32(2.0 ** -11)
_F32_FD_RTOL = 2e-2
_F32_FD_ATOL = 2e-3


def _f32_scalar_value(build, host):
    """Evaluate a float32 graph forward-only and return its scalar value."""
    tensor = _tensor(host, "float32")
    try:
        result = build(tensor)
        try:
            return np.float32(result.to_numpy().reshape(()))
        finally:
            result.close()
    finally:
        tensor.close()


def _f32_numeric_gradient(build, host):
    """Central finite differences of ``build`` at ``host``, computed
    entirely in **binary32**: every perturbed input is a float32 array and
    every evaluation runs through the float32 native graph, so this is not a
    float64 comparison in disguise (design §10.4 forbids making one a
    contract)."""
    base = np.ascontiguousarray(host, dtype=np.float32)
    out = np.zeros_like(base)
    step = _F32_FD_STEP
    for index in np.ndindex(*base.shape):
        plus = base.copy()
        minus = base.copy()
        plus[index] = np.float32(plus[index] + step)
        minus[index] = np.float32(minus[index] - step)
        out[index] = np.float32(
            (_f32_scalar_value(build, plus) - _f32_scalar_value(build, minus))
            / np.float32(2.0 * step))
    return out


@needs_native
@pytest.mark.parametrize("label,build", [
    # elementwise + reduction
    ("multiply-sum", lambda t: t.multiply(t).sum()),
    # a chain through sqrt and mean
    ("sqrt-mean", lambda t: t.sqrt().mean()),
    # reciprocal and log, both strictly inside their domains here
    ("reciprocal-sum", lambda t: t.reciprocal().sum()),
    ("log-sum", lambda t: t.log().sum()),
    # exp, kept at small magnitudes so binary32 does not overflow
    ("exp-mean", lambda t: t.exp().mean()),
    # view chains feeding a reduction. Each intermediate is a *borrowing*
    # view of the leaf, or an owning copy consumed directly — a forward-only
    # evaluation builds no graph to keep an owning temporary alive, so a
    # chain that hung a second view off a dropped copy would read released
    # storage. That is the runtime's existing ownership contract, not
    # something dtype changed.
    ("transpose-narrow-sum", lambda t: t.T.narrow(0, 1, 2).sum()),
    ("contiguous-copy-sum", lambda t: t.T.contiguous_copy().sum()),
    ("reshape-narrow-sum", lambda t: t.reshape((6,)).narrow(0, 1, 4).sum()),
    # relu, well away from its kink at every sample point
    ("relu-multiply-sum", lambda t: t.relu().multiply(t).sum()),
    # a matmul against the tensor's own transpose, then a mean
    ("matmul-transpose-mean", lambda t: t.matmul(t.T).mean()),
    # broadcasting: a (2, 3) tensor against its own column mean
    ("broadcast-subtract-sum",
     lambda t: t.subtract(t.mean(axis=1, keepdims=True)).multiply(t).sum()),
])
def test_float32_finite_differences_agree_with_the_analytical_gradient(
        label, build):
    """The float32 gradients of representative differentiable compositions,
    checked against central finite differences computed **in binary32** with
    the step and tolerances stated above."""
    host = np.array([[0.75, 1.25, 1.75], [2.25, 0.5, 1.5]], dtype=np.float32)
    x = _tensor(host, "float32", requires_grad=True)
    try:
        out = build(x)
        try:
            out.backward()
        finally:
            out.close()
        analytical = x.grad.to_numpy().copy()
        assert x.grad.dtype == "float32"
        assert analytical.dtype == np.float32
    finally:
        x.close()

    numeric = _f32_numeric_gradient(build, host)
    assert np.allclose(analytical, numeric, rtol=_F32_FD_RTOL,
                       atol=_F32_FD_ATOL), (
        f"{label}: analytical {analytical!r} vs numeric {numeric!r}")


@needs_native
def test_the_float32_finite_difference_check_can_actually_fail():
    """The negative control that makes the test above load-bearing: with a
    deliberately wrong gradient — the same values reversed along an axis —
    the same tolerances reject it. A band that accepted everything would
    prove nothing."""
    host = np.array([[0.75, 1.25, 1.75], [2.25, 0.5, 1.5]], dtype=np.float32)
    numeric = _f32_numeric_gradient(lambda t: t.multiply(t).sum(), host)
    wrong = np.ascontiguousarray(numeric[:, ::-1])
    assert not np.allclose(wrong, numeric, rtol=_F32_FD_RTOL,
                           atol=_F32_FD_ATOL)


# ---------------------------------------------------------------------------
# 8. Mixed dtype is rejected before allocation or mutation
# ---------------------------------------------------------------------------

@needs_native
@pytest.mark.parametrize("op", ("add", "subtract", "multiply", "matmul"))
def test_a_mixed_dtype_core_operation_allocates_nothing(op, monkeypatch):
    """Design §9.3 as a testable property: after a rejected mixed-dtype
    operation, live native storage is exactly what it was, both operands are
    unchanged and open, and no output exists."""
    live = _live_storage_ids(monkeypatch)
    wide = _core(np.ones((4, 4)), "float64")
    narrow = _core(np.ones((4, 4), dtype=np.float32), "float32")
    try:
        baseline = len(live)
        for left, right in ((wide, narrow), (narrow, wide)):
            with pytest.raises(ValueError, match="matching dtype"):
                getattr(left, op)(right)
            assert len(live) == baseline, op
        assert np.array_equal(wide.to_numpy(), np.ones((4, 4)))
        assert np.array_equal(narrow.to_numpy(),
                              np.ones((4, 4), dtype=np.float32))
        assert narrow.dtype == "float32" and wide.dtype == "float64"
    finally:
        narrow.close()
        wide.close()


@needs_native
@pytest.mark.parametrize("call", ("sum", "matmul", "narrow_backward"))
def test_a_mixed_dtype_call_at_the_abi_leaves_the_destination_unchanged(call):
    """The C++ half of the same rule, reached directly so a mismatch Python
    would have caught first still gets proved at the trust boundary: the
    rejection is TF_ERROR_INVALID, and the destination is byte-for-byte what
    it was."""
    library = cpp._require_library()
    narrow = cpp.NativeStorage._typed(16, "float32")
    wide = cpp.NativeStorage(16, dtype="float64")
    try:
        wide.fill(-7.5)
        before = wide.to_numpy().copy()
        shape = (ctypes.c_int64 * 2)(4, 4)
        strides = (ctypes.c_int64 * 2)(4, 1)
        writes = (ctypes.c_int64 * 2)(0, 1)
        a = narrow._require_open()
        dst = wide._require_open()
        library.tf_clear_error()
        with pytest.raises(ValueError, match="same dtype"):
            if call == "sum":
                library.tf_core_sum(a, dst, shape, strides, writes, 0, 2)
            elif call == "matmul":
                library.tf_core_matmul(a, dst, dst, 4, 4, 4, 4, 1, 4, 1, 0, 0)
            else:
                library.tf_core_narrow_backward(a, dst, shape, strides,
                                                strides, 0, 0, 2)
        assert _same_bits(wide.to_numpy(), before, "float64")
        library.tf_clear_error()
    finally:
        wide.close()
        narrow.close()


@needs_native
def test_mixed_dtype_gradient_accumulation_is_rejected_before_it_mutates():
    """Invariant 5 of design §11: a contribution of the wrong dtype is
    refused before the accumulation and before any allocation, and the
    gradient already held is left exactly as it was."""
    x = _tensor([[1.0, 2.0]], "float32", requires_grad=True)
    try:
        first = _tensor([[1.0, 1.0]], "float32")
        x._accumulate_grad(first)
        before = x.grad.to_numpy().copy()
        wrong = _tensor([[1.0, 1.0]], "float64")
        try:
            with pytest.raises(ValueError, match="matching dtype"):
                x._accumulate_grad(wrong)
        finally:
            wrong.close()
        assert x.grad.dtype == "float32"
        assert np.array_equal(x.grad.to_numpy(), before)
    finally:
        x.close()


@needs_native
def test_a_float64_seed_is_refused_for_a_float32_output():
    """An explicit ``gradient`` of the wrong dtype is rejected by name,
    before any traversal — the seed is a gradient of the output, so it has
    the output's dtype by definition."""
    x = _tensor([[1.0, 2.0]], "float32", requires_grad=True)
    try:
        out = x.multiply(x)
        try:
            seed = _tensor([[1.0, 1.0]], "float64")
            try:
                with pytest.raises(ValueError, match="dtype/device"):
                    out.backward(gradient=seed)
            finally:
                seed.close()
            assert x.grad is None
        finally:
            out.close()
    finally:
        x.close()


# ---------------------------------------------------------------------------
# 9. Ownership and lifecycle at float32
# ---------------------------------------------------------------------------

@needs_native
def test_repeated_float32_graph_cycles_return_live_storage_to_baseline(
        monkeypatch):
    """The ownership claim, proved against a real allocation count rather
    than trusting collection: twenty-five forward/backward cycles over the
    I4 operation set leave live native storage exactly where it started."""
    live = _live_storage_ids(monkeypatch)
    host = np.array([[1.5, 2.5], [3.5, 4.5]], dtype=np.float32)
    weights = np.array([[0.5, 1.0], [1.5, 2.0]], dtype=np.float32)
    baseline = len(live)
    for _ in range(25):
        x = _tensor(host, "float32", requires_grad=True)
        w = _tensor(weights, "float32", requires_grad=True)
        try:
            product = x.matmul(w)
            scaled = product.sqrt()
            reduced = scaled.mean(axis=0)
            loss = reduced.sum()
            loss.backward()
            assert x.grad.dtype == "float32"
            for tensor in (loss, reduced, scaled, product):
                tensor.close()
            x.grad.close()
            w.grad.close()
        finally:
            w.close()
            x.close()
    assert len(live) == baseline


@needs_native
def test_a_failed_float32_backward_closes_every_temporary(monkeypatch):
    """Invariant 7 of design §11: a backward that fails after allocating
    float32 temporaries closes every one of them, live native storage
    returns exactly to baseline, and the retained graph survives so a retry
    still works."""
    live = _live_storage_ids(monkeypatch)
    x = _tensor([[1.0, 2.0], [3.0, 4.0]], "float32", requires_grad=True)
    w = _tensor([[1.0], [1.0]], "float32", requires_grad=True)
    try:
        out = x.matmul(w).sum()
        try:
            library = cpp._require_library()
            original = library.tf_core_matmul
            baseline = len(live)

            def exploding(*args, **kwargs):
                raise RuntimeError("injected")

            monkeypatch.setattr(library, "tf_core_matmul", exploding)
            try:
                with pytest.raises(RuntimeError, match="injected"):
                    out.backward(retain_graph=True)
            finally:
                monkeypatch.setattr(library, "tf_core_matmul", original)
            assert len(live) == baseline
            out.backward()
            assert x.grad.dtype == "float32"
            assert w.grad.dtype == "float32"
        finally:
            out.close()
    finally:
        w.close()
        x.close()


@needs_native
def test_retain_graph_still_holds_for_a_float32_graph():
    """``retain_graph=True`` keeps the history for a second pass and the
    gradients accumulate, at float32 exactly as at float64; the one-shot
    default still releases."""
    x = _tensor([[1.0, 2.0]], "float32", requires_grad=True)
    try:
        out = x.multiply(x).sum()
        try:
            out.backward(retain_graph=True)
            first = x.grad.to_numpy().copy()
            out.backward()
            assert np.array_equal(x.grad.to_numpy(), first * np.float32(2.0))
            assert x.grad.dtype == "float32"
            with pytest.raises(RuntimeError):
                out.backward()
        finally:
            out.close()
    finally:
        x.close()


@needs_native
def test_a_float32_result_owns_storage_that_aliases_no_operand():
    """Every operation allocates a fresh owning contiguous output that
    aliases neither operand — closing the result must leave the source
    readable and unchanged, at float32 as at float64."""
    values = np.arange(12.0, dtype=np.float32).reshape(3, 4)
    core = _core(values, "float32")
    try:
        producers = (
            lambda c: c.sum(axis=0),
            lambda c: c.mean(axis=1),
            lambda c: c.matmul(c.T),
            lambda c: c.narrow(0, 1, 2).narrow_backward(0, 1, (3, 4)),
        )
        for produce in producers:
            out = produce(core)
            assert out.storage is not core.storage
            assert out.dtype == "float32"
            out.close()
            assert np.array_equal(core.to_numpy(), values)
    finally:
        core.close()


# ---------------------------------------------------------------------------
# 10. What I4 did not move
# ---------------------------------------------------------------------------

@needs_native
def test_the_generalized_reduction_and_matmul_exports_carry_one_dispatch():
    """Design §8.1 for the exports I4 generalized: each does its dtype work
    exactly once, at ABI entry, and nothing beneath branches on dtype."""
    for relative, names in (
        ("cpp/src/reduction.cpp", ("tf_core_sum", "tf_core_narrow_backward")),
        ("cpp/src/matmul.cpp", ("tf_core_matmul",)),
    ):
        source = _read(relative)
        for name in names:
            body = source.split(f"TF_EXPORT void {name}(", 1)[1]
            body = body.split("\n}\n", 1)[0]
            assert "require_matching_dtype" in body, name
            assert "require_float64" not in body, name
            assert body.count("switch (tf::dispatch_dtype(") == 1, name
    # The two scalar primitives dispatch once each, and neither rejects a
    # dtype any more — there is nothing left for either to reject.
    storage = _read("cpp/src/storage.cpp")
    for name in ("tf_storage_fill", "tf_storage_scale"):
        body = storage.split(f"TF_EXPORT void {name}(", 1)[1]
        body = body.split("\n}\n", 1)[0]
        assert body.count("switch (tf::storage_dtype(") == 1, name
        assert "require_float64" not in body, name
    # ...and the export count did not move: generalization ships inside the
    # symbols Python already declares.
    assert len(_source_exports()) == I1_EXPORT_COUNT


@needs_native
def test_no_accumulator_in_the_generalized_kernels_is_hard_coded():
    """The structural half of "float32 accumulates in float32", over the two
    headers I4 templated. Its behavioural partner is the absorption witness
    above; neither alone would be enough — the witness proves the result,
    and this proves there is no width in the source that could make the
    result right on one path and wrong on another."""
    for relative in ("cpp/include/tf_reduction_internal.h",
                     "cpp/include/tf_matmul_internal.h"):
        code = _cpp_code_only(_read(relative))
        assert "double" not in code, f"{relative}: a binary64 local survived"
        assert "float" not in code, f"{relative}: a binary32 local was pinned"
        assert "static_cast<double>" not in code, relative
        # ...and nothing forbidden by CLAUDE.md §4.3 or design §10.2 appeared
        # while the kernels were being touched.
        for banned in ("immintrin", "_mm", "omp", "std::thread", "cblas_",
                       "std::fma", "restrict"):
            assert banned not in code, f"{relative}: {banned}"


@needs_native
def test_the_raw_utility_kernels_are_still_float64_only():
    """``RAW_KERNEL_DTYPES`` did not move at I4 and never will: the seven
    handle-free kernels take ``double*`` and an element count, so they have
    no dtype to dispatch on. No raw float32 matmul was added and neither raw
    matmul signature changed."""
    assert cpp.RAW_KERNEL_DTYPES == ("float64",)
    source = _flat(_read("cpp/src/matmul.cpp"))
    assert _all_of(source,
                   "TF_EXPORT void tf_matmul( const double* a, "
                   "const double* b, double* out, int64_t m, int64_t n, "
                   "int64_t p )",
                   "TF_EXPORT void tf_matmul_tiled( const double* a, "
                   "const double* b, double* out, int64_t m, int64_t n, "
                   "int64_t p, int64_t block )")
    exports = _source_exports()
    for absent in ("tf_matmul_f32", "tf_matmul_float32", "tf_relu_f32",
                   "tf_elementwise_add_f32"):
        assert absent not in exports, absent


@needs_native
def test_the_families_i8_owns_now_execute_at_float32():
    """The other direction of the boundary, advanced to the I8 line.

    MaxPool2d left the rejecting set at I5, the classification stack at I6,
    Dropout at I7, and **the optimizers at I8** — so nothing numerical
    rejects float32 any more, and this test proves the last two crossed
    rather than pretending the boundary is where it was.

    What is still shut is what milestone **I9** owns: the public registry.
    That is asserted as a rejection rather than as an absence, because "not
    supported" is only a safe state if the attempt actually fails."""
    from tensorforge.experimental import (
        NativeAdam, NativeParameter, NativeSGD, native_checkpoint,
    )

    assert _constructors_with_a_dtype_argument() == DTYPE_CONSTRUCTORS
    assert native_checkpoint._FORMAT_VERSION == I8_CHECKPOINT_VERSION
    assert (native_checkpoint._SUPPORTED_FORMAT_VERSIONS
            == I8_CHECKPOINT_VERSIONS)

    core = cpp.NativeTensorCore._typed_from_array(
        np.ones((1, 1, 4, 4), dtype=np.float32), "float32")
    try:
        pooled = core.maxpool2d_forward(kernel_size=2)
        assert pooled.dtype == "float32"
        assert pooled.shape == (1, 1, 2, 2)
        pooled.close()
        flat = core.reshape((4, 4))
        try:
            for method in ("softmax", "log_softmax"):
                out = getattr(flat, method)(axis=-1)
                assert out.dtype == "float32"
                out.close()
            result = flat.cross_entropy_forward([0, 1, 2, 3])
            assert result.loss.dtype == "float32"
            assert result.probabilities.dtype == "float32"
            result.close()
            # Dropout accepts it now — the I7 milestone, at the Core layer.
            dropped = flat.dropout_forward(0.5, seed=1, call_index=0)
            assert dropped.dtype == "float32"
            dropped.close()
        finally:
            flat.close()
    finally:
        core.close()

    # The I8 line. Both optimizers now *run* on a float32 parameter, and
    # they are proved by execution rather than by the absence of an error:
    # the value moves, the version moves once, and every piece of state the
    # optimizer owns comes back at the parameter's own width.
    parameter = NativeParameter(np.ones(4), dtype="float32")
    try:
        adam = NativeAdam([parameter], lr=0.1)
        try:
            state = adam.state_dict()
            try:
                assert [t.dtype for t in state["m"]] == ["float32"]
                assert [t.dtype for t in state["v"]] == ["float32"]
                assert state["parameters"][0]["dtype"] == "float32"
            finally:
                for snapshot in state["m"] + state["v"]:
                    snapshot.close()
            out = parameter.sum()
            try:
                out.backward()
            finally:
                out.close()
            assert parameter.grad.dtype == "float32"
            before = parameter.to_numpy().copy()
            adam.step()
            assert parameter.version == 1
            assert parameter.to_numpy().dtype == np.float32
            assert not np.array_equal(parameter.to_numpy(), before)
            assert adam.step_counts == (1,)
            assert all(t.dtype == "float32" for t in adam._m + adam._v)
        finally:
            adam.close()
    finally:
        parameter.close()

    parameter = NativeParameter(np.ones(4), dtype="float32")
    try:
        optimizer = NativeSGD([parameter], lr=0.1)
        out = parameter.sum()
        try:
            out.backward()
        finally:
            out.close()
        assert parameter.grad.dtype == "float32"
        grad_before = parameter.grad.to_numpy().copy()
        optimizer.step()
        # value - lr * grad, at float32, with the gradient retained.
        assert parameter.version == 1
        assert parameter.to_numpy().dtype == np.float32
        assert np.array_equal(parameter.to_numpy(),
                              np.full(4, 0.9, dtype=np.float32))
        assert np.array_equal(parameter.grad.to_numpy(), grad_before)
        assert optimizer.state_dict()["parameters"][0]["dtype"] == "float32"
    finally:
        parameter.close()

    # The I9 boundary, which moved **after** this milestone and not in
    # it: the public registry now admits float32. What this milestone
    # owns is the layer above, asserted directly rather than by the
    # absence of a public route that no longer is absent.
    _assert_the_public_registry_is_i9s()


@needs_native
def test_i4_moved_no_public_capability_at_all():
    """The exit gate, as one assertion block: internal float32 reduction,
    matmul, view-backward, and Core autograd exist, and public float32
    support does not."""
    from tensorforge.experimental import native_checkpoint

    _assert_the_public_registry_is_i9s()
    assert cpp.RAW_KERNEL_DTYPES == ("float64",)
    assert cpp.backend_info()["dtype"] == "float64"
    assert native_checkpoint._FORMAT_VERSION == I8_CHECKPOINT_VERSION
    assert (native_checkpoint._SUPPORTED_FORMAT_VERSIONS
            == I8_CHECKPOINT_VERSIONS)
    exports = _source_exports()
    assert len(exports) == I1_EXPORT_COUNT       # still 54; I4 adds none
    for absent in ("tf_core_sum_f32", "tf_core_matmul_f32",
                   "tf_core_narrow_backward_f32", "tf_storage_fill_f32",
                   "tf_storage_scale_typed", "tf_core_sum_typed",
                   "tf_storage_dtype", "tf_storage_cast"):
        assert absent not in exports, absent
    assert not [name for name in exports
                if name.endswith(("_f32", "_f64", "_float32", "_float64"))]


# ===========================================================================
# I5: Conv2d and MaxPool2d dtype execution, as running code
#
# Everything below drives the **live** library at both dtypes through the
# private typed constructors, exactly as the I2-I4 sections do. The
# comparison rules are per family and stated rather than inherited:
#
#   * every Conv2d direction is compared against an **independent
#     same-dtype reference** written here as explicit loops over 0-d NumPy
#     scalars of the element dtype, in the documented accumulation order —
#     never against a float64 result (design §10.4), and never against the
#     production kernel itself;
#   * MaxPool2d values and winners are compared against a scalar reference
#     that reproduces the exact production comparison sequence (first
#     candidate anchors, first non-NaN seeds, strict ``>`` keeps the first
#     occurrence of every tie);
#   * float32 accumulation is witnessed against the widened alternative in
#     every direction that accumulates;
#   * gradients are checked analytically where a formula supports it and by
#     binary32 finite differences (the stated I4 band) where it does not.
#
# The winner buffer is asserted **float64 at every value dtype**, and the
# ``2**53`` plane bound is asserted unchanged — the two halves of the
# §13.3 decision.
# ===========================================================================


def _conv_forward_reference(input_array, weight_array, bias_array,
                            stride, padding, floating):
    """The cross-correlation as explicit loops over ``floating`` scalars,
    seeding each destination with its bias and adding taps in ascending
    c -> p -> q order — the documented per-destination order."""
    n, c, h, w = input_array.shape
    o, _, kh, kw = weight_array.shape
    sh, sw = stride
    ph, pw = padding
    oh = (h + 2 * ph - kh) // sh + 1
    ow = (w + 2 * pw - kw) // sw + 1
    out = np.zeros((n, o, oh, ow), dtype=floating)
    for ni in range(n):
        for oi in range(o):
            for i in range(oh):
                for j in range(ow):
                    acc = (bias_array[oi] if bias_array is not None
                           else floating(0.0))
                    for ci in range(c):
                        for p in range(kh):
                            ih = i * sh + p - ph
                            if ih < 0 or ih >= h:
                                continue
                            for q in range(kw):
                                iw = j * sw + q - pw
                                if iw < 0 or iw >= w:
                                    continue
                                acc = floating(
                                    acc + floating(
                                        input_array[ni, ci, ih, iw]
                                        * weight_array[oi, ci, p, q]))
                    out[ni, oi, i, j] = acc
    return out


def _conv_input_backward_reference(grad_array, weight_array, input_shape,
                                   stride, padding, floating):
    """The scatter-add adjoint in the documented n -> o -> i -> j -> c ->
    p -> q order, accumulating in ``floating``."""
    n, c, h, w = input_shape
    o, _, kh, kw = weight_array.shape
    sh, sw = stride
    ph, pw = padding
    _, _, oh, ow = grad_array.shape
    out = np.zeros((n, c, h, w), dtype=floating)
    for ni in range(n):
        for oi in range(o):
            for i in range(oh):
                for j in range(ow):
                    g = grad_array[ni, oi, i, j]
                    for ci in range(c):
                        for p in range(kh):
                            ih = i * sh + p - ph
                            if ih < 0 or ih >= h:
                                continue
                            for q in range(kw):
                                iw = j * sw + q - pw
                                if iw < 0 or iw >= w:
                                    continue
                                out[ni, ci, ih, iw] = floating(
                                    out[ni, ci, ih, iw] + floating(
                                        g * weight_array[oi, ci, p, q]))
    return out


def _conv_weight_backward_reference(grad_array, input_array, weight_shape,
                                    stride, padding, floating):
    """The weight-gradient scatter in the same documented order."""
    n, c, h, w = input_array.shape
    o, _, kh, kw = weight_shape
    sh, sw = stride
    ph, pw = padding
    _, _, oh, ow = grad_array.shape
    out = np.zeros(weight_shape, dtype=floating)
    for ni in range(n):
        for oi in range(o):
            for i in range(oh):
                for j in range(ow):
                    g = grad_array[ni, oi, i, j]
                    for ci in range(c):
                        for p in range(kh):
                            ih = i * sh + p - ph
                            if ih < 0 or ih >= h:
                                continue
                            for q in range(kw):
                                iw = j * sw + q - pw
                                if iw < 0 or iw >= w:
                                    continue
                                out[oi, ci, p, q] = floating(
                                    out[oi, ci, p, q] + floating(
                                        g * input_array[ni, ci, ih, iw]))
    return out


def _maxpool_reference(input_array, kernel, stride, padding, floating):
    """MaxPool2d values and winners by the exact production comparison
    sequence: the window's first candidate anchors the all-NaN fallback,
    the first non-NaN candidate seeds the scan, and only a strictly
    greater value displaces the selection — so every tie keeps its first
    occurrence in row-major window order, padding participates as the
    element type's own -inf with winner -1, and a NaN never wins."""
    n, c, h, w = input_array.shape
    kh, kw = kernel
    sh, sw = stride
    ph, pw = padding
    oh = (h + 2 * ph - kh) // sh + 1
    ow = (w + 2 * pw - kw) // sw + 1
    values = np.zeros((n, c, oh, ow), dtype=floating)
    winners = np.zeros((n, c, oh, ow), dtype=np.float64)
    neg_inf = floating(-np.inf)
    for ni in range(n):
        for ci in range(c):
            for i in range(oh):
                for j in range(ow):
                    best = floating(0.0)
                    winner = -1.0
                    seen_any = False
                    seen_number = False
                    for p in range(kh):
                        ih = i * sh + p - ph
                        row_ok = 0 <= ih < h
                        for q in range(kw):
                            iw = j * sw + q - pw
                            ok = row_ok and 0 <= iw < w
                            cand = (input_array[ni, ci, ih, iw] if ok
                                    else neg_inf)
                            cand_winner = float(ih * w + iw) if ok else -1.0
                            if not seen_any:
                                seen_any = True
                                best = cand
                                winner = cand_winner
                            if cand != cand:
                                continue
                            if not seen_number or cand > best:
                                seen_number = True
                                best = cand
                                winner = cand_winner
                    values[ni, ci, i, j] = best
                    winners[ni, ci, i, j] = winner
    return values, winners


# One geometry sweep shared by the three Conv2d directions: both sides of
# the H9 swept-extent boundary, strides, padding, rectangles, batches, and
# channels.
_CONV_GEOMETRIES = (
    # (n, c, h, w, o, kh, kw, stride, padding, bias)
    (1, 1, 3, 3, 1, 2, 2, (1, 1), (0, 0), False),   # generic (ow = 2)
    (1, 1, 4, 5, 1, 2, 2, (1, 1), (0, 0), True),    # optimized (ow = 4)
    (2, 3, 6, 7, 4, 3, 2, (1, 1), (0, 0), True),    # batched rectangular
    (1, 2, 6, 6, 2, 3, 3, (1, 1), (1, 1), True),    # padded
    (1, 1, 7, 7, 1, 3, 3, (2, 2), (0, 0), False),   # strided
    (2, 2, 8, 8, 3, 3, 3, (2, 2), (1, 1), True),    # strided + padded
    (1, 1, 5, 9, 1, 1, 3, (1, 2), (0, 1), False),   # asymmetric forms
)


def _conv_case_arrays(geometry, dtype):
    n, c, h, w, o, kh, kw, stride, padding, bias = geometry
    floating = _DTYPE_BITS[dtype][2]
    input_array = _sample(dtype, n * c * h * w, seed=11).reshape(n, c, h, w)
    weight_array = _sample(dtype, o * c * kh * kw, seed=12).reshape(
        o, c, kh, kw)
    bias_array = _sample(dtype, o, seed=13) if bias else None
    sh, sw = stride
    ph, pw = padding
    oh = (h + 2 * ph - kh) // sh + 1
    ow = (w + 2 * pw - kw) // sw + 1
    grad_array = _sample(dtype, n * o * oh * ow, seed=14).reshape(
        n, o, oh, ow)
    return floating, input_array, weight_array, bias_array, grad_array


@needs_native
@pytest.mark.parametrize("dtype", BOTH_DTYPES)
@pytest.mark.parametrize("geometry", _CONV_GEOMETRIES)
def test_conv2d_forward_matches_the_same_dtype_reference(dtype, geometry):
    """Every geometry in the sweep, driven through the Core wrapper and
    compared **bitwise** against the independent same-dtype reference —
    the per-destination accumulation-order contract restated per dtype."""
    n, c, h, w, o, kh, kw, stride, padding, bias = geometry
    floating, input_array, weight_array, bias_array, _ = _conv_case_arrays(
        geometry, dtype)
    want = _conv_forward_reference(input_array, weight_array, bias_array,
                                   stride, padding, floating)
    x = _core(input_array, dtype)
    wt = _core(weight_array, dtype)
    bs = _core(bias_array, dtype) if bias else None
    try:
        out = x.conv2d_forward(wt, bs, stride=stride, padding=padding)
        try:
            assert out.dtype == dtype
            assert out.shape == want.shape
            assert _same_bits(out.to_numpy(), want, dtype), geometry
        finally:
            out.close()
    finally:
        if bs is not None:
            bs.close()
        wt.close()
        x.close()


@needs_native
@pytest.mark.parametrize("dtype", BOTH_DTYPES)
@pytest.mark.parametrize("geometry", _CONV_GEOMETRIES)
def test_conv2d_backwards_match_the_same_dtype_reference(dtype, geometry):
    """Both gradient directions over the same sweep, bitwise against their
    own same-dtype scatter references."""
    n, c, h, w, o, kh, kw, stride, padding, _ = geometry
    floating, input_array, weight_array, _, grad_array = _conv_case_arrays(
        geometry, dtype)
    want_input = _conv_input_backward_reference(
        grad_array, weight_array, (n, c, h, w), stride, padding, floating)
    want_weight = _conv_weight_backward_reference(
        grad_array, input_array, (o, c, kh, kw), stride, padding, floating)
    grad = _core(grad_array, dtype)
    wt = _core(weight_array, dtype)
    x = _core(input_array, dtype)
    try:
        grad_in = grad.conv2d_input_backward(
            wt, input_shape=(n, c, h, w), stride=stride, padding=padding)
        try:
            assert grad_in.dtype == dtype
            assert _same_bits(grad_in.to_numpy(), want_input, dtype), geometry
        finally:
            grad_in.close()
        grad_wt = grad.conv2d_weight_backward(
            x, weight_shape=(o, c, kh, kw), stride=stride, padding=padding)
        try:
            assert grad_wt.dtype == dtype
            assert _same_bits(grad_wt.to_numpy(), want_weight, dtype), geometry
        finally:
            grad_wt.close()
    finally:
        x.close()
        wt.close()
        grad.close()


@needs_native
@pytest.mark.parametrize("dtype", BOTH_DTYPES)
def test_conv2d_reads_non_contiguous_operands_at_both_dtypes(dtype):
    """Policy B at both widths: a transposed (non-contiguous) input
    produces bit-identical results to its contiguous equivalent, through
    the same private materialization the float64 path has always used —
    and the dtype survives the temporary copy."""
    floating = _DTYPE_BITS[dtype][2]
    base = _sample(dtype, 36, seed=15).reshape(1, 1, 6, 6)
    weight = _sample(dtype, 4, seed=16).reshape(1, 1, 2, 2)
    x = _core(base, dtype)
    wt = _core(weight, dtype)
    try:
        # Transpose the two spatial axes: a genuinely non-contiguous view,
        # compared against the reference over the transposed array.
        view = x.transpose(0, 1, 3, 2)
        assert not view.contiguous
        want = _conv_forward_reference(
            np.ascontiguousarray(base.transpose(0, 1, 3, 2)),
            weight, None, (1, 1), (0, 0), floating)
        out = view.conv2d_forward(wt, stride=1, padding=0)
        try:
            assert out.dtype == dtype
            assert _same_bits(out.to_numpy(), want, dtype)
        finally:
            out.close()
    finally:
        wt.close()
        x.close()


@needs_native
def test_the_float32_conv2d_witness_from_python():
    """The accumulation witness through the Core wrapper on both forward
    traversals: 1.0 followed by eight copies of 2**-24 stays exactly 1.0
    under sequential binary32 accumulation and lands higher when
    accumulated in binary64 and narrowed once. TensorForge must equal the
    first and differ from the second — a float32 Conv2d that is secretly
    float64 fails here by bit pattern."""
    tiny = np.float32(2.0 ** -24)
    sequential = np.float32(1.0)
    widened = 1.0
    for _ in range(8):
        sequential = np.float32(sequential + tiny)
        widened += float(tiny)
    narrowed_once = np.float32(widened)
    assert _bits(np.array([sequential]), "float32") != _bits(
        np.array([narrowed_once]), "float32")

    weight = np.ones((1, 1, 1, 9), dtype=np.float32)
    for width, label in ((9, "generic (ow=1)"), (12, "optimized (ow=4)")):
        values = np.full((1, 1, 1, width), tiny, dtype=np.float32)
        values[0, 0, 0, 0] = np.float32(1.0)
        x = _core(values, "float32")
        wt = _core(weight, "float32")
        try:
            out = x.conv2d_forward(wt, stride=1, padding=0)
            try:
                got = out.to_numpy()[0, 0, 0, 0]
                assert _bits(np.array([got]), "float32") == _bits(
                    np.array([sequential]), "float32"), label
                assert _bits(np.array([got]), "float32") != _bits(
                    np.array([narrowed_once]), "float32"), label
            finally:
                out.close()
        finally:
            wt.close()
            x.close()


@needs_native
def test_private_float32_conv2d_autograd_is_analytic():
    """A float32 conv2d graph with bias, differentiated end to end: with an
    all-ones weight and upstream, the input gradient is the per-cell window
    count, the weight gradient is the sum of the input cells each tap saw,
    and the bias gradient is the output count — all exactly representable,
    so the comparison is equality at float32."""
    host = np.arange(1.0, 17.0, dtype=np.float32).reshape(1, 1, 4, 4)
    x = _tensor(host, "float32", requires_grad=True)
    w = _tensor(np.ones((1, 1, 2, 2), dtype=np.float32), "float32",
                requires_grad=True)
    b = _tensor(np.zeros(1, dtype=np.float32), "float32", requires_grad=True)
    try:
        out = x.conv2d(w, b, stride=1, padding=0)
        try:
            assert out.dtype == "float32"
            loss = out.sum()
            try:
                loss.backward()
            finally:
                loss.close()
        finally:
            out.close()
        # Window counts for a 2x2 kernel at stride 1 on a 4x4 plane.
        counts = np.array([[1, 2, 2, 1],
                           [2, 4, 4, 2],
                           [2, 4, 4, 2],
                           [1, 2, 2, 1]], dtype=np.float32)
        assert x.grad.dtype == "float32"
        assert np.array_equal(x.grad.to_numpy()[0, 0], counts)
        # Each weight tap saw a 3x3 block of the input.
        want_w = np.empty((2, 2), dtype=np.float32)
        for p in range(2):
            for q in range(2):
                want_w[p, q] = host[0, 0, p:p + 3, q:q + 3].sum(
                    dtype=np.float32)
        assert w.grad.dtype == "float32"
        assert np.array_equal(w.grad.to_numpy()[0, 0], want_w)
        assert b.grad.dtype == "float32"
        assert np.array_equal(b.grad.to_numpy(), np.array([9.0],
                                                          dtype=np.float32))
    finally:
        b.close()
        w.close()
        x.close()


@needs_native
@pytest.mark.parametrize("dtype", BOTH_DTYPES)
def test_conv2d_and_maxpool_gradients_have_the_graph_dtype(dtype):
    """Design §11 invariants 1 and 2 over the I5 operations: every
    gradient — input, weight, bias, and pooling input — has the dtype of
    the tensor it is a gradient of, at both widths."""
    floating = _DTYPE_BITS[dtype][2]
    x = _tensor(_sample(dtype, 36, seed=17).reshape(1, 1, 6, 6), dtype,
                requires_grad=True)
    w = _tensor(_sample(dtype, 4, seed=18).reshape(1, 1, 2, 2), dtype,
                requires_grad=True)
    b = _tensor(np.zeros(1, dtype=floating), dtype, requires_grad=True)
    try:
        convolved = x.conv2d(w, b, stride=1, padding=1)
        pooled = convolved.maxpool2d(kernel_size=2)
        loss = pooled.sum()
        try:
            loss.backward()
        finally:
            loss.close()
            pooled.close()
            convolved.close()
        for leaf in (x, w, b):
            assert leaf.grad is not None
            assert leaf.grad.dtype == dtype
            assert leaf.grad.shape == leaf.shape
    finally:
        b.close()
        w.close()
        x.close()


@needs_native
@pytest.mark.parametrize("build", ["conv", "conv_pool"])
def test_float32_cnn_finite_differences(build):
    """The float32 conv2d input gradient (and the composed conv+pool
    gradient away from ties) against central finite differences computed
    in binary32 with the stated I4 step and tolerances. The pooling case
    uses inputs whose window maxima are unique and stable under the
    perturbation, so the finite difference is taken away from the
    nondifferentiable tie boundaries. The negative control shows the band
    rejects a deliberately wrong gradient."""
    # Distinct, well-separated values: every window maximum is unique with
    # margin far larger than 2 * step.
    host = (np.arange(16, dtype=np.float32).reshape(1, 1, 4, 4)
            * np.float32(0.5) + np.float32(0.25))
    w_host = np.array([[[[0.5, -0.25], [1.5, 0.75]]]], dtype=np.float32)

    # The analytical gradient: intermediates stay open until the backward
    # has consumed the graph, per the ordinary ownership contract.
    x = _tensor(host, "float32", requires_grad=True)
    w = _tensor(w_host, "float32")
    try:
        chain = [x.conv2d(w, stride=1, padding=0)]
        if build == "conv_pool":
            chain.append(chain[-1].maxpool2d(kernel_size=2, stride=1))
        chain.append(chain[-1].sum())
        try:
            chain[-1].backward()
        finally:
            for tensor in reversed(chain):
                tensor.close()
        analytical = x.grad.to_numpy().copy()
        assert analytical.dtype == np.float32
    finally:
        w.close()
        x.close()

    def scalar(value_host):
        # Forward-only: no graph is built, so eager closing is safe.
        t = _tensor(value_host, "float32")
        weight = _tensor(w_host, "float32")
        try:
            convolved = t.conv2d(weight, stride=1, padding=0)
            try:
                if build == "conv":
                    result = convolved.sum()
                else:
                    pooled = convolved.maxpool2d(kernel_size=2, stride=1)
                    try:
                        result = pooled.sum()
                    finally:
                        pooled.close()
            finally:
                convolved.close()
            try:
                return np.float32(result.to_numpy().reshape(()))
            finally:
                result.close()
        finally:
            weight.close()
            t.close()

    numeric = np.zeros_like(host)
    for index in np.ndindex(*host.shape):
        plus = host.copy()
        minus = host.copy()
        plus[index] = np.float32(plus[index] + _F32_FD_STEP)
        minus[index] = np.float32(minus[index] - _F32_FD_STEP)
        numeric[index] = np.float32(
            (scalar(plus) - scalar(minus)) / np.float32(2.0 * _F32_FD_STEP))
    assert np.allclose(analytical, numeric, rtol=_F32_FD_RTOL,
                       atol=_F32_FD_ATOL), (
        f"{build}: analytical {analytical!r} vs numeric {numeric!r}")
    # The negative control: the band rejects a deliberately wrong gradient.
    wrong = np.ascontiguousarray(analytical[..., ::-1])
    assert not np.allclose(wrong, numeric, rtol=_F32_FD_RTOL,
                           atol=_F32_FD_ATOL)


@needs_native
@pytest.mark.parametrize("dtype", BOTH_DTYPES)
def test_maxpool_forward_matches_the_exact_scalar_reference(dtype):
    """Values and winners over ordinary, overlapping, and padded windows,
    bitwise against the scalar reference — and the winner buffer is
    float64 at BOTH value dtypes, which is the §13.3 decision made
    observable."""
    for kernel, stride, padding, shape in (
        ((2, 2), (2, 2), (0, 0), (2, 2, 4, 4)),
        ((3, 2), (1, 1), (1, 0), (1, 2, 4, 5)),
        ((2, 2), (1, 1), (0, 0), (1, 1, 5, 5)),
    ):
        floating = _DTYPE_BITS[dtype][2]
        host = _sample(dtype, int(np.prod(shape)), seed=19).reshape(shape)
        want_values, want_winners = _maxpool_reference(
            host, kernel, stride, padding, floating)
        x = _core(host, dtype)
        try:
            out, winners = x._maxpool2d_forward_with_winners(
                kernel_size=kernel, stride=stride, padding=padding)
            try:
                assert out.dtype == dtype
                assert winners.dtype == "float64"
                assert _same_bits(out.to_numpy(), want_values, dtype)
                assert np.array_equal(winners.to_numpy(), want_winners)
            finally:
                winners.close()
                out.close()
        finally:
            x.close()


@needs_native
@pytest.mark.parametrize("dtype", BOTH_DTYPES)
def test_maxpool_tie_nan_and_signed_zero_semantics(dtype):
    """The exact selection rules, pinned beyond reference agreement: the
    first occurrence of an equal maximum wins (strict ``>``), a leading
    -0.0 is kept against a later +0.0, a NaN never wins and never
    displaces, an all-NaN window falls back to its first candidate, and
    +inf wins its window — identically at both widths."""
    floating = _DTYPE_BITS[dtype][2]
    host = np.zeros((1, 1, 4, 4), dtype=floating)
    host[0, 0, 0, 0] = 3.5; host[0, 0, 0, 1] = 3.5        # tie
    host[0, 0, 1, 0] = 1.0; host[0, 0, 1, 1] = -2.0
    host[0, 0, 0, 2] = -0.0; host[0, 0, 0, 3] = 0.0       # signed-zero tie
    host[0, 0, 1, 2] = -1.0; host[0, 0, 1, 3] = -1.0
    host[0, 0, 2, 0] = np.nan; host[0, 0, 2, 1] = -7.0    # NaN first
    host[0, 0, 3, 0] = -8.0; host[0, 0, 3, 1] = -9.0
    host[0, 0, 2, 2] = -np.inf
    host[0, 0, 2, 3] = np.finfo(floating).tiny
    host[0, 0, 3, 2] = np.inf; host[0, 0, 3, 3] = 5.0
    x = _core(host, dtype)
    try:
        out, winners = x._maxpool2d_forward_with_winners(kernel_size=2)
        try:
            values = out.to_numpy()
            offsets = winners.to_numpy()
            assert values[0, 0, 0, 0] == floating(3.5)
            assert offsets[0, 0, 0, 0] == 0.0          # first of the tie
            assert _bits(values[0, 0, 0, 1:2].copy(),
                         dtype) == _bits(np.array([-0.0], dtype=floating),
                                         dtype)
            assert offsets[0, 0, 0, 1] == 2.0          # -0.0 kept by strict >
            assert values[0, 0, 1, 0] == floating(-7.0)
            assert offsets[0, 0, 1, 0] == 9.0          # NaN never wins
            assert values[0, 0, 1, 1] == floating(np.inf)
            assert offsets[0, 0, 1, 1] == 14.0
        finally:
            winners.close()
            out.close()
        # The all-NaN window: output and winner still agree, at the
        # window's first candidate.
        nan_host = np.full((1, 1, 2, 2), np.nan, dtype=floating)
        core = _core(nan_host, dtype)
        try:
            out, winners = core._maxpool2d_forward_with_winners(kernel_size=2)
            try:
                assert np.isnan(out.to_numpy()[0, 0, 0, 0])
                assert winners.to_numpy()[0, 0, 0, 0] == 0.0
            finally:
                winners.close()
                out.close()
        finally:
            core.close()
    finally:
        x.close()


@needs_native
def test_maxpool_winner_offsets_stay_exact_beyond_float32_range():
    """The consequence the §13.3 decision buys: a float32 pool over a
    plane larger than float32's exact-integer range (2**24) still records
    its winner offset exactly, because the winner buffer is float64 and
    the bound is float64's 2**53. A winner buffer that followed the value
    dtype would round this offset."""
    width = 2 ** 24 + 2
    offset = width - 1                       # odd, above 2**24
    assert int(np.float32(offset)) != offset  # not float32-representable
    host = np.zeros((1, 1, 1, width), dtype=np.float32)
    host[0, 0, 0, offset] = np.float32(1.0)
    x = _core(host, "float32")
    try:
        out, winners = x._maxpool2d_forward_with_winners(
            kernel_size=(1, width))
        try:
            assert winners.dtype == "float64"
            recorded = winners.to_numpy()[0, 0, 0, 0]
            assert recorded == float(offset)          # exact, not rounded
            assert out.to_numpy()[0, 0, 0, 0] == np.float32(1.0)
        finally:
            winners.close()
            out.close()
    finally:
        x.close()
    assert cpp._MAX_EXACT_WINNER_PLANE == 2 ** 53


@needs_native
@pytest.mark.parametrize("dtype", BOTH_DTYPES)
def test_maxpool_backward_routes_and_accumulates_at_the_graph_dtype(dtype):
    """Unique windows reproduce each upstream value at its winner, in the
    graph dtype, bitwise against the scatter reference."""
    floating = _DTYPE_BITS[dtype][2]
    host = _sample(dtype, 16, seed=21).reshape(1, 1, 4, 4)
    x = _core(host, dtype)
    try:
        out, winners = x._maxpool2d_forward_with_winners(kernel_size=2)
        try:
            upstream_host = _sample(dtype, 4, seed=22).reshape(1, 1, 2, 2)
            upstream = _core(upstream_host, dtype)
            try:
                grad = upstream.maxpool2d_backward(
                    winners, input_shape=(1, 1, 4, 4))
                try:
                    assert grad.dtype == dtype
                    reference = _maxpool_reference(
                        host, (2, 2), (2, 2), (0, 0), floating)[1]
                    expected = np.zeros((1, 1, 4, 4), dtype=floating)
                    for i in range(2):
                        for j in range(2):
                            w_at = int(reference[0, 0, i, j])
                            expected[0, 0, w_at // 4, w_at % 4] = floating(
                                expected[0, 0, w_at // 4, w_at % 4]
                                + upstream_host[0, 0, i, j])
                    assert _same_bits(grad.to_numpy(), expected, dtype)
                finally:
                    grad.close()
            finally:
                upstream.close()
        finally:
            winners.close()
            out.close()
    finally:
        x.close()


@needs_native
def test_the_float32_maxpool_backward_witness():
    """Overlapping windows all selecting one cell: the cell's gradient is
    the witness sum, accumulated sequentially in binary32 — equal to the
    sequential result and unequal to the widened one, by bit pattern."""
    host = np.zeros((1, 1, 5, 5), dtype=np.float32)
    host[0, 0, 2, 2] = np.float32(100.0)     # wins all nine 3x3 windows
    upstream_host = np.full((1, 1, 3, 3), np.float32(2.0 ** -24),
                            dtype=np.float32)
    upstream_host[0, 0, 0, 0] = np.float32(1.0)
    sequential = np.float32(1.0)
    widened = 1.0
    for _ in range(8):
        sequential = np.float32(sequential + np.float32(2.0 ** -24))
        widened += 2.0 ** -24
    narrowed_once = np.float32(widened)
    x = _core(host, "float32")
    try:
        out, winners = x._maxpool2d_forward_with_winners(
            kernel_size=3, stride=1)
        try:
            upstream = _core(upstream_host, "float32")
            try:
                grad = upstream.maxpool2d_backward(
                    winners, input_shape=(1, 1, 5, 5))
                try:
                    got = grad.to_numpy()[0, 0, 2, 2]
                    assert _bits(np.array([got]), "float32") == _bits(
                        np.array([sequential]), "float32")
                    assert _bits(np.array([got]), "float32") != _bits(
                        np.array([narrowed_once]), "float32")
                    # Every unselected cell holds exactly +0.0.
                    rest = grad.to_numpy().copy()
                    rest[0, 0, 2, 2] = 0.0
                    assert _same_bits(rest, np.zeros_like(rest), "float32")
                finally:
                    grad.close()
            finally:
                upstream.close()
        finally:
            winners.close()
            out.close()
    finally:
        x.close()


@needs_native
@pytest.mark.parametrize("dtype", BOTH_DTYPES)
def test_a_non_float64_winner_buffer_is_rejected_before_allocation(
        dtype, monkeypatch):
    """The other half of §13.3: the winner buffer is validated as exactly
    float64 — never against the gradient's dtype — before any output is
    allocated, at both value dtypes, and a rejection leaves live storage
    exactly as it was."""
    live = _live_storage_ids(monkeypatch)
    upstream = _core(_sample(dtype, 4, seed=23).reshape(1, 1, 2, 2), dtype)
    fake_winners = cpp.NativeTensorCore._typed_from_array(
        np.zeros((1, 1, 2, 2), dtype=np.float32), "float32")
    try:
        baseline = len(live)
        with pytest.raises(ValueError, match="float64 winner"):
            upstream.maxpool2d_backward(fake_winners,
                                        input_shape=(1, 1, 4, 4))
        assert len(live) == baseline
    finally:
        fake_winners.close()
        upstream.close()


@needs_native
@pytest.mark.parametrize("dtype", BOTH_DTYPES)
def test_mixed_dtype_cnn_operations_allocate_nothing(dtype, monkeypatch):
    """Design §9.3 over the I5 operations: a mixed-dtype conv2d in any
    operand position is rejected before any allocation, and both operands
    stay open and unchanged."""
    other = "float64" if dtype == "float32" else "float32"
    live = _live_storage_ids(monkeypatch)
    x = _core(np.ones((1, 1, 4, 4), dtype=_DTYPE_BITS[dtype][2]), dtype)
    wrong_w = _core(np.ones((1, 1, 2, 2), dtype=_DTYPE_BITS[other][2]), other)
    wrong_b = _core(np.ones(1, dtype=_DTYPE_BITS[other][2]), other)
    right_w = _core(np.ones((1, 1, 2, 2), dtype=_DTYPE_BITS[dtype][2]), dtype)
    try:
        baseline = len(live)
        with pytest.raises(ValueError, match="matching dtype"):
            x.conv2d_forward(wrong_w)
        with pytest.raises(ValueError, match="matching dtype"):
            x.conv2d_forward(right_w, wrong_b)
        with pytest.raises(ValueError, match="matching dtype"):
            x.conv2d_input_backward(wrong_w, input_shape=(1, 1, 5, 5))
        with pytest.raises(ValueError, match="matching dtype"):
            x.conv2d_weight_backward(wrong_w, weight_shape=(1, 1, 2, 2))
        assert len(live) == baseline
        assert x.dtype == dtype and wrong_w.dtype == other
    finally:
        right_w.close()
        wrong_b.close()
        wrong_w.close()
        x.close()


@needs_native
def test_float32_maxpool_graph_resources_follow_the_saved_state_contract(
        monkeypatch):
    """The winner buffer at float32 rides the same graph_resources contract
    as at float64: float64 dtype beside float32 values, released exactly
    once by a one-shot backward, retained under retain_graph, and closed
    immediately by a no-grad forward."""
    host = np.arange(16, dtype=np.float32).reshape(1, 1, 4, 4)

    captured = {}
    original = cpp.NativeTensorCore._maxpool2d_forward_with_winners

    def capturing(self, **kwargs):
        out, winners = original(self, **kwargs)
        captured["winners"] = winners
        return out, winners

    monkeypatch.setattr(cpp.NativeTensorCore,
                        "_maxpool2d_forward_with_winners", capturing)

    # A no-grad forward releases the winners immediately.
    plain = _tensor(host, "float32")
    try:
        pooled = plain.maxpool2d(kernel_size=2)
        try:
            assert captured["winners"]._closed is True
        finally:
            pooled.close()
    finally:
        plain.close()

    # A one-shot backward releases them exactly once; retain_graph keeps
    # them for another pass first.
    x = _tensor(host, "float32", requires_grad=True)
    try:
        pooled = x.maxpool2d(kernel_size=2)
        winners = captured["winners"]
        assert winners.dtype == "float64"
        assert winners._closed is False
        loss = pooled.sum()
        try:
            loss.backward(retain_graph=True)
            assert winners._closed is False       # retained for another pass
            first = x.grad.to_numpy().copy()
            x.grad.close()
            x._grad = None
            loss.backward()
            assert winners._closed is True        # released exactly once
            assert np.array_equal(x.grad.to_numpy(), first)
            assert x.grad.dtype == "float32"
        finally:
            loss.close()
            pooled.close()
    finally:
        x.close()


@needs_native
def test_a_failed_float32_maxpool_backward_leaves_the_graph_retryable():
    """A backward that fails on its output allocation keeps the saved
    winners alive and the graph retryable, at float32 exactly as the
    Phase-D contract states at float64 — and the retry produces the
    correct gradient."""
    host = np.arange(16, dtype=np.float32).reshape(1, 1, 4, 4)
    x = _tensor(host, "float32", requires_grad=True)
    try:
        pooled = x.maxpool2d(kernel_size=2)
        winners = pooled._graph_resources[0]
        assert winners.dtype == "float64"
        loss = pooled.sum()
        try:
            cpp._arm_alloc_failure(1)
            try:
                with pytest.raises(MemoryError):
                    loss.backward()
            finally:
                cpp._arm_alloc_failure(0)
            assert winners._closed is False       # saved state survived
            assert x.grad is None
            loss.backward()                        # the retry succeeds
            assert winners._closed is True
            assert x.grad is not None and x.grad.dtype == "float32"
        finally:
            loss.close()
            pooled.close()
    finally:
        x.close()


@needs_native
def test_repeated_float32_cnn_cycles_return_live_storage_to_baseline(
        monkeypatch):
    """Fifteen conv2d + maxpool2d forward/backward cycles at float32 leave
    live native storage exactly at its baseline — outputs, gradients,
    Policy-B temporaries, and winner buffers all released."""
    live = _live_storage_ids(monkeypatch)
    host = _sample("float32", 36, seed=24).reshape(1, 1, 6, 6)
    weight = _sample("float32", 4, seed=25).reshape(1, 1, 2, 2)
    baseline = len(live)
    for _ in range(15):
        x = _tensor(host, "float32", requires_grad=True)
        w = _tensor(weight, "float32", requires_grad=True)
        try:
            convolved = x.conv2d(w, stride=1, padding=1)
            pooled = convolved.maxpool2d(kernel_size=2)
            loss = pooled.sum()
            loss.backward()
            assert x.grad.dtype == "float32"
            assert w.grad.dtype == "float32"
            for tensor in (loss, pooled, convolved):
                tensor.close()
            x.grad.close()
            w.grad.close()
        finally:
            w.close()
            x.close()
    assert len(live) == baseline


@needs_native
@pytest.mark.parametrize("dtype", BOTH_DTYPES)
def test_the_uninitialized_cnn_destinations_are_fully_written(
        dtype, monkeypatch):
    """The H1 audit re-proved per dtype through the Python seam: every CNN
    destination allocated uninitialized is completely overwritten by its
    kernel. The poison is applied by test infrastructure around the
    allocator — no production hook exists — and the negative control shows
    the detector can fail."""
    sentinel = float(2 ** 22 + 3)             # exact at both widths
    floating = _DTYPE_BITS[dtype][2]
    original = cpp.NativeTensorCore._uninitialized.__func__

    def poisoned(cls, shape, dtype="float64", device="cpu"):
        out = original(cls, shape, dtype=dtype, device=device)
        out._storage.fill(sentinel)
        return out

    monkeypatch.setattr(cpp.NativeTensorCore, "_uninitialized",
                        classmethod(poisoned))
    # Negative control: the poison really is visible on a buffer nothing
    # overwrites.
    control = cpp.NativeTensorCore._uninitialized((2, 2), dtype=dtype)
    try:
        assert np.all(control.to_numpy() == floating(sentinel))
    finally:
        control.close()

    host = _sample(dtype, 36, seed=26).reshape(1, 1, 6, 6)
    weight = _sample(dtype, 4, seed=27).reshape(1, 1, 2, 2)
    x = _core(host, dtype)
    wt = _core(weight, dtype)
    try:
        out = x.conv2d_forward(wt, stride=1, padding=1)
        try:
            assert not np.any(out.to_numpy() == floating(sentinel))
        finally:
            out.close()
        pooled, winners = x._maxpool2d_forward_with_winners(kernel_size=2)
        try:
            assert not np.any(pooled.to_numpy() == floating(sentinel))
            assert not np.any(winners.to_numpy() == sentinel)
        finally:
            winners.close()
            pooled.close()
    finally:
        wt.close()
        x.close()


# ---------------------------------------------------------------------------
# I5: what the generalized CNN source must (and must not) contain
# ---------------------------------------------------------------------------

def test_the_cnn_exports_carry_one_dispatch_each():
    """Design §8.1 over the five I5 exports, as source structure: each
    export runs the matching-dtype guard and exactly one dispatch switch,
    the pooling exports additionally pin the winner buffer to float64, and
    no switch has a ``default:`` label to hide behind."""
    conv = _cpp_code_only(_read("cpp/src/conv2d.cpp"))
    pool = _cpp_code_only(_read("cpp/src/pooling.cpp"))
    assert conv.count("tf::require_matching_dtype") == 3
    assert conv.count("switch (tf::dispatch_dtype") == 3
    assert pool.count("tf::require_matching_dtype") == 2
    assert pool.count("switch (tf::dispatch_dtype") == 2
    assert pool.count("require_winner_float64(") >= 3   # def + 1 per export
    for source in (conv, pool):
        assert "require_float64(" not in source.replace(
            "require_winner_float64(", "")
        assert "default:" not in source
    # The winner buffer is reached through the float64 accessor alone and
    # its bound is written once, as float64's 2**53.
    assert "storage_f64(winners_handle)" in pool
    assert "<< 53" in pool
    assert "<< 24" not in pool


def test_no_accumulator_in_the_cnn_kernels_is_hard_coded():
    """The structural half of the accumulation policy: the kernels take
    ``T*`` operands, accumulate in ``T``, and contain no widening cast and
    no double-typed accumulator — while the pooling header keeps its
    winner parameter spelled ``double*`` at every value dtype."""
    conv = _cpp_code_only(_read("cpp/include/tf_conv2d_internal.h"))
    pool = _cpp_code_only(_read("cpp/include/tf_pooling_internal.h"))
    for header, families in ((conv, ("conv2d_forward_generic",
                                     "conv2d_forward_row_sweep",
                                     "conv2d_input_backward_generic",
                                     "conv2d_input_backward_gather",
                                     "conv2d_weight_backward_generic",
                                     "conv2d_weight_backward_gather")),
                             (pool, ("maxpool2d_forward_contiguous",
                                     "maxpool2d_backward_contiguous"))):
        for family in families:
            assert f"void {family}(" in header, family
        assert "template <class T>" in header
        for banned in ("double acc", "double sum", "double bias_o",
                       "double g =", "double w_value", "double best_value"):
            assert banned not in header, banned
    assert "T acc = bias_o" in conv
    assert "T(0)" in conv
    # The winner side of the pooling kernels is deliberately NOT templated:
    # the buffer parameter is double at both widths, and the winner value
    # written is a double plane offset.
    assert "double* winners" in pool
    assert "T* winners" not in pool
    assert "double best_winner" in pool
    assert "static_cast<double>(ih * input_width + iw)" in pool
    # Nothing forbidden by CLAUDE.md §4.3 or design §10.2 appeared.
    for banned in ("immintrin", "_mm", "omp", "std::thread", "cblas_",
                   "std::fma", "restrict"):
        assert banned not in conv and banned not in pool, banned


def test_the_python_winner_allocation_is_pinned_to_float64():
    """The Python half of §13.3, as source structure: the winner buffer is
    allocated with an explicit ``dtype="float64"`` (never the input's),
    the backward validates the winner tag as exactly float64, and the
    ``2**53`` bound survives as the single plane authority."""
    source = _read("src/tensorforge/backends/cpp.py")
    assert '_MAX_EXACT_WINNER_PLANE = 2 ** 53' in source
    assert '(n, c, out_h, out_w), dtype="float64", device=self.device' \
        in source
    assert 'winners.dtype != "float64"' in source
    # All five §2.3 float64-only Core gates are gone now: I5 opened the two
    # pooling ones, I6 the two cross-entropy ones, and I7 Dropout's — the
    # last. The winner pin above is **not** one of them and is why this
    # assertion sits in this test: it is a deliberate, permanent decision
    # about index metadata, not a gate waiting to be opened.
    assert source.count('!= "float64" or self.device != "cpu"') == 0


@needs_native
def test_i5_moved_no_public_capability_at_all():
    """The exit gate, as one assertion block: internal float32 Conv2d and
    MaxPool2d exist in all their directions, and public float32 support
    does not."""
    from tensorforge.experimental import native_checkpoint

    _assert_the_public_registry_is_i9s()
    assert cpp.RAW_KERNEL_DTYPES == ("float64",)
    assert cpp.backend_info()["dtype"] == "float64"
    assert native_checkpoint._FORMAT_VERSION == I8_CHECKPOINT_VERSION
    assert (native_checkpoint._SUPPORTED_FORMAT_VERSIONS
            == I8_CHECKPOINT_VERSIONS)
    exports = _source_exports()
    assert len(exports) == I1_EXPORT_COUNT       # still 54; I5 adds none
    for absent in ("tf_core_conv2d_forward_f32", "tf_core_maxpool2d_f32",
                   "tf_core_conv2d_forward_typed", "tf_storage_winners",
                   "tf_core_maxpool2d_forward_f32"):
        assert absent not in exports, absent
    assert not [name for name in exports
                if name.endswith(("_f32", "_f64", "_float32", "_float64"))]
    # I5 gave no constructor a dtype argument. ``NativeConv2d`` got one at
    # I7 (it owns parameters); ``NativeMaxPool2d`` never will, because it
    # owns no numeric state and takes its dtype from the input.
    import inspect
    from tensorforge.experimental import NativeConv2d, NativeMaxPool2d
    assert "dtype" in inspect.signature(NativeConv2d).parameters
    assert "dtype" not in inspect.signature(NativeMaxPool2d).parameters
    assert _constructors_with_a_dtype_argument() == DTYPE_CONSTRUCTORS


# ===========================================================================
# I6: stable math and classification dtype execution, as running code
#
# Everything below drives the **live** library at both dtypes through the
# private typed constructors, exactly as the I2-I5 sections do. The
# comparison rules are per family and stated rather than inherited, and they
# differ from I5's in one structural way worth naming:
#
#   * softmax, log-softmax, and cross-entropy contain ``exp`` and ``log``,
#     which have **no correctly-rounded IEEE guarantee** — the reason H8
#     excluded them from the templated traversal and the reason the float64
#     transcendental contract is a ULP bound rather than bit equality. So
#     there are deliberately **two** references here, answering two different
#     questions:
#
#       1. an oracle that reproduces the kernel's exact traversal (the strict
#          ``>`` maximum scan, the shift, the accumulation order, the
#          in-place normalization, the fused log-sum-exp) but takes its
#          exponentials and logarithms from **TensorForge's own** ``exp`` and
#          ``log`` Core ops. Against that, every kernel is **bit-identical**
#          at both dtypes. That is the strong statement, and it is a
#          statement about the algorithm — with the one ingredient that has
#          no exactness guarantee factored out rather than glossed over.
#
#       2. an oracle in the same traversal that takes them from **NumPy**.
#          Against that, float32 sits within a small measured ULP bound and
#          float64 is exact. The bound is a statement about NumPy's separate
#          float32 SIMD transcendental kernel — the same two-bound split I3
#          measured and recorded for ``exp``/``log`` themselves.
#
#     Neither is a float32-versus-float64 comparison: both references are
#     evaluated at the dtype under test (design §10.4 forbids making the
#     cross-width comparison a contract).
#
#   * float32 accumulation is witnessed on the one place in this family where
#     the two candidate policies can differ at all — the batch-loss
#     accumulator — using per-row losses taken from the kernel itself, so no
#     assumption about ``expf``/``logf`` enters the witness.
#
#   * gradients are checked analytically where the closed form gives an exact
#     answer, and by binary32 finite differences (the stated I4 band)
#     everywhere else.
#
# The int64 target boundary is asserted **unchanged at both widths**, and the
# saved probabilities are asserted to carry the graph dtype — the two halves
# of what I6 does and does not move.
# ===========================================================================

# Measured on this machine at I6, and recorded separately for exactly the
# reason F32_TRANSCENDENTAL_ULP and F32_TRANSCENDENTAL_ULP_VS_NUMPY are: the
# tighter number is about TensorForge, the looser one is about NumPy. Both
# are float32-against-float32.
F32_CLASSIFICATION_ULP_VS_NUMPY = 4


def _tf_exp(values, dtype):
    """``exp`` through TensorForge's own Core op, at ``dtype``."""
    core = _core(values, dtype)
    try:
        out = core.exp()
        try:
            return out.to_numpy().copy()
        finally:
            out.close()
    finally:
        core.close()


def _tf_log(values, dtype):
    core = _core(values, dtype)
    try:
        out = core.log()
        try:
            return out.to_numpy().copy()
        finally:
            out.close()
    finally:
        core.close()


def _np_exp(values, dtype):
    floating = _DTYPE_BITS[dtype][2]
    return np.exp(np.ascontiguousarray(values, dtype=floating), dtype=floating)


def _np_log(values, dtype):
    floating = _DTYPE_BITS[dtype][2]
    return np.log(np.ascontiguousarray(values, dtype=floating), dtype=floating)


def _axis_slices(shape, axis):
    """Every (outer, inner) reduction slice of ``shape`` along ``axis``, as
    index tuples — the (outer, axis_length, inner) decomposition the ABI
    carries, expressed in NumPy indexing."""
    for prefix in np.ndindex(*shape[:axis]):
        for suffix in np.ndindex(*shape[axis + 1:]):
            yield tuple(prefix) + (slice(None),) + tuple(suffix)


def _softmax_oracle(values, axis, dtype, expf):
    """The kernel's own traversal, scalar, at ``dtype``: strict ``>`` maximum
    scan, shift, exponentials written into the destination while the total
    accumulates, then an in-place normalization."""
    floating = _DTYPE_BITS[dtype][2]
    source = np.ascontiguousarray(values, dtype=floating)
    axis = axis if axis >= 0 else source.ndim + axis
    out = np.empty_like(source)
    for index in _axis_slices(source.shape, axis):
        row = source[index]
        maximum = row[0]
        for value in row[1:]:
            if value > maximum:
                maximum = value
        shifted = expf(np.array([floating(v - maximum) for v in row],
                                dtype=floating), dtype)
        total = floating(0)
        for value in shifted:
            total = floating(total + value)
        out[index] = np.array([floating(v / total) for v in shifted],
                              dtype=floating)
    return out


def _log_softmax_oracle(values, axis, dtype, expf, logf):
    """The fused log-sum-exp traversal — never ``log(softmax(x))``."""
    floating = _DTYPE_BITS[dtype][2]
    source = np.ascontiguousarray(values, dtype=floating)
    axis = axis if axis >= 0 else source.ndim + axis
    out = np.empty_like(source)
    for index in _axis_slices(source.shape, axis):
        row = source[index]
        maximum = row[0]
        for value in row[1:]:
            if value > maximum:
                maximum = value
        shifted = np.array([floating(v - maximum) for v in row],
                           dtype=floating)
        exps = expf(shifted, dtype)
        total = floating(0)
        for value in exps:
            total = floating(total + value)
        denominator = logf(np.array([total], dtype=floating), dtype)[0]
        out[index] = np.array([floating(v - denominator) for v in shifted],
                              dtype=floating)
    return out


def _cross_entropy_oracle(logits, targets, reduction, dtype, expf, logf):
    """The fused forward: per-row maximum, shifted exponentials into the
    saved probabilities, an in-place normalization, then
    ``log(sum_exp) - (x[target] - m)`` accumulated in the element type and
    divided ONCE by the batch size for ``"mean"``."""
    floating = _DTYPE_BITS[dtype][2]
    source = np.ascontiguousarray(logits, dtype=floating)
    batch = source.shape[0]
    probabilities = np.empty_like(source)
    total = floating(0)
    for n in range(batch):
        row = source[n]
        maximum = row[0]
        for value in row[1:]:
            if value > maximum:
                maximum = value
        shifted = np.array([floating(v - maximum) for v in row],
                           dtype=floating)
        exps = expf(shifted, dtype)
        sum_exp = floating(0)
        for value in exps:
            sum_exp = floating(sum_exp + value)
        probabilities[n] = np.array([floating(v / sum_exp) for v in exps],
                                    dtype=floating)
        denominator = logf(np.array([sum_exp], dtype=floating), dtype)[0]
        total = floating(total + floating(denominator - shifted[targets[n]]))
    if reduction == "mean":
        return floating(total / floating(batch)), probabilities
    return total, probabilities


def _cross_entropy_backward_oracle(probabilities, targets, upstream,
                                   reduction, dtype):
    """``upstream * (p - onehot) / N``, in the kernel's exact order: read the
    saved probability, subtract the indicator, apply the mean scaling, then
    multiply by the upstream — never reassociated."""
    floating = _DTYPE_BITS[dtype][2]
    probabilities = np.ascontiguousarray(probabilities, dtype=floating)
    batch, classes = probabilities.shape
    out = np.empty_like(probabilities)
    count = floating(batch)
    scale = floating(upstream)
    for n in range(batch):
        for c in range(classes):
            contribution = probabilities[n, c]
            if c == targets[n]:
                contribution = floating(contribution - floating(1))
            if reduction == "mean":
                contribution = floating(contribution / count)
            out[n, c] = floating(scale * contribution)
    return out


_CLASSIFICATION_SHAPES = (
    ((1,), -1),                 # rank 1, the whole tensor is one slice
    ((5,), 0),                  # rank 1, positive axis
    ((4, 3), -1),               # last axis of a matrix
    ((4, 3), 0),                # FIRST axis: inner > 1
    ((2, 3, 4), 1),             # middle axis of a rank-3 tensor
    ((2, 3, 4), -1),            # last axis of a rank-3 tensor
    ((3, 1, 5), 1),             # a length-one reduction axis
    ((2, 2, 2, 2), -2),         # rank 4, a negative non-last axis
)


@needs_native
@pytest.mark.parametrize("dtype", BOTH_DTYPES)
@pytest.mark.parametrize("shape,axis", _CLASSIFICATION_SHAPES)
def test_softmax_matches_the_same_dtype_traversal_oracle(dtype, shape, axis):
    """Reference 1: bit-identical to an oracle that reproduces the kernel's
    traversal exactly and takes its exponentials from TensorForge's own
    ``exp``. What this pins is the maximum scan, the shift, the accumulation
    order, and the in-place normalization — at both widths."""
    host = _sample(dtype, int(np.prod(shape)), seed=31).reshape(shape)
    core = _core(host, dtype)
    try:
        out = core.softmax(axis=axis)
        try:
            assert out.dtype == dtype
            assert out.shape == shape
            got = out.to_numpy()
        finally:
            out.close()
    finally:
        core.close()
    assert _same_bits(got, _softmax_oracle(host, axis, dtype, _tf_exp), dtype)


@needs_native
@pytest.mark.parametrize("dtype", BOTH_DTYPES)
@pytest.mark.parametrize("shape,axis", _CLASSIFICATION_SHAPES)
def test_log_softmax_matches_the_same_dtype_traversal_oracle(dtype, shape,
                                                             axis):
    """The same statement for the fused log-sum-exp kernel."""
    host = _sample(dtype, int(np.prod(shape)), seed=32).reshape(shape)
    core = _core(host, dtype)
    try:
        out = core.log_softmax(axis=axis)
        try:
            assert out.dtype == dtype
            assert out.shape == shape
            got = out.to_numpy()
        finally:
            out.close()
    finally:
        core.close()
    assert _same_bits(
        got, _log_softmax_oracle(host, axis, dtype, _tf_exp, _tf_log), dtype)


@needs_native
def test_float64_classification_is_exact_against_the_numpy_oracle():
    """Reference 2 at float64: the width Phase E measured is bit-identical
    against a NumPy-transcendental oracle too, so the float64 contract is
    unmoved by the generalization."""
    host = _sample("float64", 24, seed=33).reshape(4, 6)
    core = _core(host, "float64")
    try:
        out = core.softmax(axis=-1)
        try:
            assert _same_bits(out.to_numpy(),
                              _softmax_oracle(host, -1, "float64", _np_exp),
                              "float64")
        finally:
            out.close()
        out = core.log_softmax(axis=-1)
        try:
            assert _same_bits(
                out.to_numpy(),
                _log_softmax_oracle(host, -1, "float64", _np_exp, _np_log),
                "float64")
        finally:
            out.close()
    finally:
        core.close()


@needs_native
def test_float32_classification_is_within_the_measured_numpy_ulp_bound():
    """Reference 2 at float32: within a small measured bound of a NumPy
    float32 oracle. The gap is NumPy's own float32 SIMD transcendental
    kernel, which I3 already measured at about two steps from correctly
    rounded — this is the same statement one composition further on, and it
    is float32-against-float32 throughout."""
    host = _sample("float32", 60, seed=34).reshape(6, 10)
    core = _core(host, "float32")
    try:
        out = core.softmax(axis=-1)
        try:
            _assert_transcendental(
                out.to_numpy(), _softmax_oracle(host, -1, "float32", _np_exp),
                F32_CLASSIFICATION_ULP_VS_NUMPY, "float32 softmax vs NumPy")
        finally:
            out.close()
        out = core.log_softmax(axis=-1)
        try:
            _assert_transcendental(
                out.to_numpy(),
                _log_softmax_oracle(host, -1, "float32", _np_exp, _np_log),
                F32_CLASSIFICATION_ULP_VS_NUMPY,
                "float32 log_softmax vs NumPy")
        finally:
            out.close()
    finally:
        core.close()


@needs_native
@pytest.mark.parametrize("dtype", BOTH_DTYPES)
@pytest.mark.parametrize("reduction", ("mean", "sum"))
@pytest.mark.parametrize("shape", ((1, 1), (1, 6), (5, 3), (4, 7)))
def test_cross_entropy_forward_matches_the_traversal_oracle(dtype, reduction,
                                                            shape):
    """The fused forward at both dtypes and both reductions: the scalar loss
    and the saved probabilities are each bit-identical to the same-traversal
    oracle, and both carry the logits' dtype."""
    floating = _DTYPE_BITS[dtype][2]
    batch, classes = shape
    host = _sample(dtype, batch * classes, seed=35).reshape(shape)
    targets = [(n * 3) % classes for n in range(batch)]
    want_loss, want_probabilities = _cross_entropy_oracle(
        host, targets, reduction, dtype, _tf_exp, _tf_log)
    core = _core(host, dtype)
    try:
        result = core.cross_entropy_forward(targets, reduction)
        try:
            assert result.loss.dtype == dtype
            assert result.loss.shape == ()
            assert result.probabilities.dtype == dtype
            assert result.probabilities.shape == shape
            assert _same_bits(result.loss.to_numpy().reshape(1),
                              np.array([want_loss], dtype=floating), dtype)
            assert _same_bits(result.probabilities.to_numpy(),
                              want_probabilities, dtype)
            # The saved probabilities alias neither the logits nor the loss.
            assert result.probabilities.storage is not core.storage
            assert result.probabilities.storage is not result.loss.storage
            # ...and the backward reads them, at the same dtype.
            for upstream_value in (1.0, -2.5, 0.125):
                upstream = _core(np.array(upstream_value, dtype=floating),
                                 dtype)
                try:
                    grad = result.probabilities.cross_entropy_backward(
                        result.targets, upstream, reduction)
                    try:
                        assert grad.dtype == dtype
                        assert _same_bits(
                            grad.to_numpy(),
                            _cross_entropy_backward_oracle(
                                want_probabilities, targets, upstream_value,
                                reduction, dtype), dtype)
                    finally:
                        grad.close()
                finally:
                    upstream.close()
        finally:
            result.close()
    finally:
        core.close()


@needs_native
def test_the_float32_batch_loss_accumulates_in_float32_from_python():
    """The one place in this family where a widened accumulator would be
    visible at all, witnessed through the shipped Core wrapper.

    Every other float32 property here is a single correctly-rounded IEEE
    operation per destination, where computing in binary64 and rounding once
    is *provably* indistinguishable from computing in binary32. Accumulation
    is not, so this is where the policy gets a behavioural proof rather than
    a structural one — and the per-row losses come from the kernel itself, so
    no assumption about ``expf``/``logf`` enters.

    Row 0 has a loss of exactly 200; every later row a loss of ~6.1e-6, below
    half a binary32 ULP of the running total. Sequential binary32 therefore
    absorbs all 199 of them and stays at exactly 200, while binary64
    accumulates them and lands ~1.2e-3 higher — two orders of magnitude above
    the narrowing step."""
    batch, classes = 200, 2
    host = np.zeros((batch, classes), dtype=np.float32)
    targets = [0] * batch
    host[0, 1] = np.float32(200.0)
    for n in range(1, batch):
        host[n, 1] = np.float32(12.0)
        targets[n] = 1
    core = _core(host, "float32")
    try:
        row_losses = []
        for n in range(batch):
            row = _core(host[n:n + 1], "float32")
            try:
                one = row.cross_entropy_forward([targets[n]], "sum")
                try:
                    row_losses.append(
                        np.float32(one.loss.to_numpy().reshape(())))
                finally:
                    one.close()
            finally:
                row.close()
        sequential = np.float32(0.0)
        widened = 0.0
        for value in row_losses:
            sequential = np.float32(sequential + value)
            widened += float(value)
        widened_narrowed = np.float32(widened)
        # Non-vacuous first: if the two policies agreed here the comparison
        # below would prove nothing.
        assert not _same_bits(np.array([sequential]),
                              np.array([widened_narrowed]), "float32")

        result = core.cross_entropy_forward(targets, "sum")
        try:
            got = np.float32(result.loss.to_numpy().reshape(()))
        finally:
            result.close()
        assert _same_bits(np.array([got]), np.array([sequential]), "float32")
        assert not _same_bits(np.array([got]), np.array([widened_narrowed]),
                              "float32")
    finally:
        core.close()


@needs_native
@pytest.mark.parametrize("dtype", BOTH_DTYPES)
def test_the_mean_reduction_divides_once_by_the_batch_at_the_element_dtype(
        dtype):
    """``mean`` is the ``sum`` divided ONCE by ``batch_size`` — never by
    ``num_classes``, never twice, and the division happens at the element
    type."""
    floating = _DTYPE_BITS[dtype][2]
    batch, classes = 6, 4
    host = _sample(dtype, batch * classes, seed=36).reshape(batch, classes)
    targets = [n % classes for n in range(batch)]
    core = _core(host, dtype)
    try:
        total = core.cross_entropy_forward(targets, "sum")
        mean = core.cross_entropy_forward(targets, "mean")
        try:
            summed = floating(total.loss.to_numpy().reshape(()))
            averaged = floating(mean.loss.to_numpy().reshape(()))
            assert _same_bits(np.array([averaged]),
                              np.array([floating(summed / floating(batch))]),
                              dtype)
            assert not _same_bits(
                np.array([averaged]),
                np.array([floating(summed / floating(classes))]), dtype)
        finally:
            mean.close()
            total.close()
    finally:
        core.close()


@needs_native
@pytest.mark.parametrize("dtype", BOTH_DTYPES)
def test_policy_b_views_reach_the_classification_kernels(dtype):
    """The contiguous-only C ABI is unchanged, so a strided view is
    materialized into a private contiguous copy before the call — at float32
    exactly as at float64. Transposed, narrowed, and chained views all produce
    the same result as the equivalent contiguous data."""
    host = _sample(dtype, 24, seed=37).reshape(4, 6)
    core = _core(host, dtype)
    try:
        views = {
            "transposed": (core.T, np.ascontiguousarray(host.T)),
            "narrowed": (core.narrow(1, 1, 4),
                         np.ascontiguousarray(host[:, 1:5])),
            "chained": (core.T.narrow(0, 2, 3),
                        np.ascontiguousarray(host.T[2:5])),
        }
        for label, (view, equivalent) in views.items():
            try:
                assert view.dtype == dtype
                out = view.softmax(axis=-1)
                try:
                    assert out.dtype == dtype and out.contiguous, label
                    assert _same_bits(
                        out.to_numpy(),
                        _softmax_oracle(equivalent, -1, dtype, _tf_exp),
                        dtype), label
                finally:
                    out.close()
                out = view.log_softmax(axis=-1)
                try:
                    assert _same_bits(
                        out.to_numpy(),
                        _log_softmax_oracle(equivalent, -1, dtype, _tf_exp,
                                            _tf_log), dtype), label
                finally:
                    out.close()
                # ...and the fused cross-entropy, which takes the same
                # Policy-B path.
                targets = [n % view.shape[1] for n in range(view.shape[0])]
                _, want = _cross_entropy_oracle(equivalent, targets, "mean",
                                                dtype, _tf_exp, _tf_log)
                result = view.cross_entropy_forward(targets, "mean")
                try:
                    assert result.loss.dtype == dtype
                    assert _same_bits(result.probabilities.to_numpy(), want,
                                      dtype), label
                finally:
                    result.close()
            finally:
                view.close()
    finally:
        core.close()


@needs_native
@pytest.mark.parametrize("dtype", BOTH_DTYPES)
def test_stability_holds_at_each_width_s_own_magnitudes(dtype):
    """The shift and the fusion carry the stability, not the width — proved at
    magnitudes chosen for the dtype under test rather than float64 magnitudes
    reused.

    Each case shows the *naive* form actually failing before showing the
    kernel succeeding; a stability witness on which the naive form happened to
    work would prove nothing."""
    floating = _DTYPE_BITS[dtype][2]
    # Where a naive unshifted exp() overflows: ~709 at binary64, ~88.7 at
    # binary32.
    offset = floating(800.0 if dtype == "float64" else 120.0)
    for sign in (floating(1.0), floating(-1.0)):
        base = np.array([[0.0, 1.0, 2.0, -1.0]], dtype=floating)
        shifted = (base + sign * offset).astype(floating)
        with np.errstate(over="ignore", under="ignore"):
            naive = np.exp(shifted, dtype=floating).sum(dtype=floating)
        assert (not np.isfinite(naive)) or naive == floating(0.0)

        core = _core(shifted, dtype)
        centered = _core(base, dtype)
        try:
            out = core.softmax(axis=-1)
            want = centered.softmax(axis=-1)
            try:
                # A large common offset cancels exactly: the shifted slice is
                # bit-identical to the centred one.
                assert _same_bits(out.to_numpy(), want.to_numpy(), dtype)
                assert np.all(np.isfinite(out.to_numpy()))
            finally:
                want.close()
                out.close()
            out = core.log_softmax(axis=-1)
            try:
                values = out.to_numpy()
                assert np.all(np.isfinite(values))
                assert np.all(values <= floating(0.0))
            finally:
                out.close()
        finally:
            centered.close()
            core.close()

    # A probability far below the element type's smallest normal: the naive
    # log(softmax(x)) collapses to -inf, the fused form stays finite and
    # exact. The gap is chosen per width for the same reason.
    gap = floating(1600.0 if dtype == "float64" else 240.0)
    row = np.array([[0.0, -gap]], dtype=floating)
    core = _core(row, dtype)
    try:
        probabilities = core.softmax(axis=-1)
        try:
            assert _same_bits(probabilities.to_numpy()[0, 1:2],
                              np.array([floating(0.0)]), dtype)
            with np.errstate(divide="ignore"):
                assert np.isneginf(
                    np.log(probabilities.to_numpy()[0, 1], dtype=floating))
        finally:
            probabilities.close()
        fused = core.log_softmax(axis=-1)
        try:
            values = fused.to_numpy()
            assert np.isfinite(values[0, 1])
            # sum_exp rounds to exactly 1 here, so the answer is exactly the
            # shifted logit and exactly +0.0 at the maximum.
            assert _same_bits(values[0],
                              np.array([floating(0.0), -gap], dtype=floating),
                              dtype)
        finally:
            fused.close()
        # The fused cross-entropy loss on the same row is exact where
        # -log(p[target]) would be +inf.
        result = core.cross_entropy_forward([1], "sum")
        try:
            assert _same_bits(result.loss.to_numpy().reshape(1),
                              np.array([gap], dtype=floating), dtype)
        finally:
            result.close()
    finally:
        core.close()


@needs_native
@pytest.mark.parametrize("dtype", BOTH_DTYPES)
def test_a_slice_spread_beyond_the_finite_range_is_recorded_not_hidden(dtype):
    """The one honest domain qualification of the float32 stability statement,
    asserted rather than glossed.

    The maximum shift guarantees no *exponent* overflows. It does not — and
    mathematically cannot — guarantee that the shifted value ``x - m`` is
    itself representable: a slice whose **spread** exceeds the element type's
    largest finite value makes the subtraction overflow to ``-inf``. That is
    the correctly-rounded IEEE-754 result for a quantity with no
    representation at that width; it is reachable at binary64 too, past
    ~1.8e308, so it is a dynamic-range fact and not a float32 defect.

    Recorded in both directions so no later milestone "fixes" it with a
    widened intermediate (which would be mixed precision, which §1.2 excludes)
    or a special case (which would break the traversal contract):

      * **softmax is unaffected** — the affected class gets exactly the
        mathematically correct probability ``+0.0``, and the maximum exactly
        ``1.0``;
      * **log-softmax reports -inf** and **cross-entropy +inf**, for true
        values below/above the representable range;
      * and all three stay **values**, not ABI errors.
    """
    floating = _DTYPE_BITS[dtype][2]
    huge = floating(3.0e38 if dtype == "float32" else 1.0e308)
    row = np.array([[huge, -huge]], dtype=floating)
    # Both inputs are finite; the spread between them is not.
    assert np.all(np.isfinite(row))
    with np.errstate(over="ignore"):
        assert not np.isfinite(floating(row[0, 1] - row[0, 0]))

    core = _core(row, dtype)
    try:
        probabilities = core.softmax(axis=-1)
        try:
            assert _same_bits(
                probabilities.to_numpy()[0],
                np.array([floating(1.0), floating(0.0)], dtype=floating),
                dtype)
        finally:
            probabilities.close()
        logs = core.log_softmax(axis=-1)
        try:
            values = logs.to_numpy()
            assert _same_bits(values[0, 0:1], np.array([floating(0.0)]), dtype)
            assert np.isneginf(values[0, 1])
        finally:
            logs.close()
        result = core.cross_entropy_forward([1], "sum")
        try:
            assert np.isposinf(result.loss.to_numpy().reshape(()))
        finally:
            result.close()
    finally:
        core.close()


@needs_native
@pytest.mark.parametrize("dtype", BOTH_DTYPES)
def test_exceptional_values_follow_plain_ieee_at_both_widths(dtype):
    """NaN first, NaN later, infinities, an all-``-inf`` slice, signed zeros,
    and subnormals — every one against the same-traversal oracle by raw bit
    pattern, and every one leaving the error slot clear, because a NaN in a
    *result* is a value rather than an ABI failure."""
    floating = _DTYPE_BITS[dtype][2]
    tiny = float(np.finfo(floating).smallest_subnormal)
    slices = [
        [np.nan, 1.0, 2.0, 3.0],
        [1.0, 2.0, np.nan, 3.0],
        [np.inf, 1.0, 2.0, 3.0],
        [-np.inf, 1.0, 2.0, 3.0],
        [-np.inf, -np.inf, -np.inf, -np.inf],
        [np.inf, -np.inf, 1.0, 2.0],
        [0.0, -0.0, 0.0, -0.0],
        [tiny, -tiny, 0.0, tiny],
        [np.nan, np.inf, -np.inf, 0.0],
    ]
    for values in slices:
        host = np.array([values], dtype=floating)
        # The *oracle* is scalar Python arithmetic over NumPy scalars, so
        # ``inf - inf`` and friends raise NumPy's invalid-operation warning
        # while producing exactly the IEEE result the kernel produces. The
        # warning is about the reference, not about TensorForge, and the
        # values it warns on are the whole point of these cases.
        core = _core(host, dtype)
        try:
            with np.errstate(all="ignore"):
                _softmax_want = _softmax_oracle(host, -1, dtype, _tf_exp)
                _log_want = _log_softmax_oracle(host, -1, dtype, _tf_exp,
                                                _tf_log)
                _ce_loss_want, _ce_probabilities_want = _cross_entropy_oracle(
                    host, [0], "sum", dtype, _tf_exp, _tf_log)
            out = core.softmax(axis=-1)
            try:
                assert _same_bits(out.to_numpy(), _softmax_want,
                                  dtype), values
            finally:
                out.close()
            out = core.log_softmax(axis=-1)
            try:
                assert _same_bits(out.to_numpy(), _log_want, dtype), values
            finally:
                out.close()
            # A NaN result is a value: the call succeeds, so nothing is
            # recorded in the thread-local error slot and no exception is
            # raised.
            result = core.cross_entropy_forward([0], "sum")
            try:
                assert _same_bits(result.probabilities.to_numpy(),
                                  _ce_probabilities_want, dtype), values
                assert _same_bits(result.loss.to_numpy().reshape(1),
                                  np.array([_ce_loss_want], dtype=floating),
                                  dtype), values
            finally:
                result.close()
        finally:
            core.close()


@needs_native
@pytest.mark.parametrize("dtype", BOTH_DTYPES)
def test_the_int64_target_boundary_is_unchanged_at_both_widths(dtype):
    """Targets carry no dtype and never gain one: they stay an independently
    owned host ``int64`` copy at float32 exactly as at float64, with the same
    strict accepted/rejected forms, the same boundary labels, and the same
    caller-independence."""
    floating = _DTYPE_BITS[dtype][2]
    batch, classes = 3, 4
    host = _sample(dtype, batch * classes, seed=38).reshape(batch, classes)
    core = _core(host, dtype)
    try:
        # Both boundary labels are accepted, and the copy is int64.
        result = core.cross_entropy_forward([0, classes - 1, 2], "mean")
        try:
            assert result.targets.dtype == np.int64
            assert result.targets.shape == (batch,)
            assert result.targets.flags["C_CONTIGUOUS"]
        finally:
            result.close()

        # The strict Phase-E rules are unchanged, and every one of them is
        # refused before anything is allocated.
        for bad, error in (
            ([0, 1, classes], ValueError),        # == num_classes
            ([0, 1, -1], ValueError),             # negative
            ([0, 1], ValueError),                 # wrong length
            ([0, 1, True], TypeError),            # bool is not a class index
            ([0, 1, 2.0], TypeError),             # integral float still fails
            ([0, 1, "2"], TypeError),             # a string is not an index
            ([[0], [1], [2]], TypeError),         # nested / rank 2
        ):
            with pytest.raises(error):
                core.cross_entropy_forward(bad, "mean")

        # Caller mutation after the forward cannot reach the backward: the
        # labels were copied.
        mutable = np.array([0, 1, 2], dtype=np.int64)
        result = core.cross_entropy_forward(mutable, "mean")
        try:
            before = result.targets.copy()
            mutable[:] = [3, 3, 3]
            assert np.array_equal(result.targets, before)
            upstream = _core(np.array(1.0, dtype=floating), dtype)
            try:
                saved = result.probabilities.to_numpy().copy()
                grad = result.probabilities.cross_entropy_backward(
                    result.targets, upstream, "mean")
                try:
                    assert grad.dtype == dtype
                    assert _same_bits(
                        grad.to_numpy(),
                        _cross_entropy_backward_oracle(saved, before, 1.0,
                                                       "mean", dtype), dtype)
                finally:
                    grad.close()
            finally:
                upstream.close()
        finally:
            result.close()
    finally:
        core.close()


# ---------------------------------------------------------------------------
# I6: private float32 classification autograd
# ---------------------------------------------------------------------------

@needs_native
@pytest.mark.parametrize("dtype", BOTH_DTYPES)
def test_softmax_backward_is_analytic_at_the_graph_dtype(dtype):
    """``dx = y * (g - sum(g * y, axis, keepdims=True))``. Under a plain
    ``sum`` seed every ``g`` is 1 and ``sum(y)`` is 1 along the axis, so the
    gradient is exactly zero — an exact analytical answer at both widths, and
    one a formula error could not accidentally produce."""
    floating = _DTYPE_BITS[dtype][2]
    host = np.array([[0.5, 1.5, -0.5], [2.0, -1.0, 0.25]], dtype=floating)
    for axis in (-1, 0):
        x = _tensor(host, dtype, requires_grad=True)
        try:
            out = x.softmax(axis=axis).sum()
            try:
                out.backward()
            finally:
                out.close()
            assert x.grad.dtype == dtype
            assert x.grad.shape == host.shape
            assert np.allclose(x.grad.to_numpy(), np.zeros_like(host),
                               atol=1e-6 if dtype == "float32" else 1e-12)
        finally:
            x.close()


@needs_native
@pytest.mark.parametrize("dtype", BOTH_DTYPES)
def test_log_softmax_backward_is_analytic_at_the_graph_dtype(dtype):
    """``dx = g - exp(y) * sum(g, axis, keepdims=True)``. Under a plain
    ``sum`` seed that is exactly ``1 - n * p``, where ``n`` is the axis length
    and ``p`` the softmax probabilities — checked against the probabilities
    the *softmax* kernel produces, at the graph dtype."""
    floating = _DTYPE_BITS[dtype][2]
    host = np.array([[0.5, 1.5, -0.5], [2.0, -1.0, 0.25]], dtype=floating)
    for axis in (-1, 0):
        x = _tensor(host, dtype, requires_grad=True)
        try:
            out = x.log_softmax(axis=axis).sum()
            try:
                out.backward()
            finally:
                out.close()
            assert x.grad.dtype == dtype
            got = x.grad.to_numpy().copy()
        finally:
            x.close()
        probabilities = _softmax_oracle(host, axis, dtype, _tf_exp)
        length = host.shape[axis]
        want = (np.ones_like(host) - floating(length) * probabilities)
        assert np.allclose(got, want,
                           rtol=1e-5 if dtype == "float32" else 1e-12,
                           atol=1e-6 if dtype == "float32" else 1e-12)


@needs_native
@pytest.mark.parametrize("dtype", BOTH_DTYPES)
@pytest.mark.parametrize("reduction", ("mean", "sum"))
def test_cross_entropy_backward_is_analytic_at_the_graph_dtype(dtype,
                                                               reduction):
    """``grad = upstream * (p - onehot) / N``, straight from the saved
    probabilities — bit-identical to the oracle at both widths, with the loss
    and the gradient both at the logits' dtype."""
    floating = _DTYPE_BITS[dtype][2]
    host = np.array([[0.5, 1.5, -0.5], [2.0, -1.0, 0.25]], dtype=floating)
    targets = [2, 0]
    x = _tensor(host, dtype, requires_grad=True)
    try:
        loss = x.cross_entropy(targets, reduction=reduction)
        try:
            assert loss.dtype == dtype
            assert loss.shape == ()
            loss.backward()
        finally:
            loss.close()
        assert x.grad.dtype == dtype
        got = x.grad.to_numpy().copy()
    finally:
        x.close()
    _, probabilities = _cross_entropy_oracle(host, targets, reduction, dtype,
                                             _tf_exp, _tf_log)
    want = _cross_entropy_backward_oracle(probabilities, targets, 1.0,
                                          reduction, dtype)
    assert _same_bits(got, want, dtype)


@needs_native
@pytest.mark.parametrize("label,build", [
    ("softmax-multiply-sum",
     lambda t: t.softmax(axis=-1).multiply(t).sum()),
    ("softmax-axis0-multiply-sum",
     lambda t: t.softmax(axis=0).multiply(t).sum()),
    ("log_softmax-multiply-sum",
     lambda t: t.log_softmax(axis=-1).multiply(t).sum()),
    ("log_softmax-axis0-multiply-sum",
     lambda t: t.log_softmax(axis=0).multiply(t).sum()),
    # A composed classification graph: the negative entropy of the softmax,
    # built from both transforms over one leaf.
    ("softmax-times-log_softmax",
     lambda t: t.softmax(axis=-1).multiply(t.log_softmax(axis=-1)).sum()),
    ("cross-entropy-mean", lambda t: t.cross_entropy([0, 2], "mean")),
    ("cross-entropy-sum", lambda t: t.cross_entropy([1, 0], "sum")),
    # cross-entropy over a transformed leaf, so the loss gradient flows back
    # through an earlier node too.
    ("relu-then-cross-entropy",
     lambda t: t.relu().cross_entropy([2, 1], "mean")),
])
def test_float32_classification_finite_differences(label, build):
    """The float32 gradients of the classification stack against central
    finite differences computed **in binary32**, with the step and tolerances
    stated at the I4 band above.

    The logits are finite, well conditioned, and deliberately away from the
    saturated regime the stability tests use: a saturated softmax has a
    numerically uninformative gradient, so an extreme witness would make this
    check vacuous rather than strict. Saturation is tested separately, for its
    own property."""
    host = np.array([[0.75, 1.25, -0.5], [0.25, -1.5, 1.0]], dtype=np.float32)
    x = _tensor(host, "float32", requires_grad=True)
    try:
        out = build(x)
        try:
            out.backward()
        finally:
            out.close()
        analytical = x.grad.to_numpy().copy()
        assert x.grad.dtype == "float32"
        assert analytical.dtype == np.float32
    finally:
        x.close()

    numeric = _f32_numeric_gradient(build, host)
    assert np.allclose(analytical, numeric, rtol=_F32_FD_RTOL,
                       atol=_F32_FD_ATOL), (
        f"{label}: analytical {analytical!r} vs numeric {numeric!r}")


@needs_native
def test_the_float32_classification_finite_difference_check_can_fail():
    """The negative control that makes the check above load-bearing: the same
    tolerances reject a deliberately wrong gradient."""
    host = np.array([[0.75, 1.25, -0.5], [0.25, -1.5, 1.0]], dtype=np.float32)
    numeric = _f32_numeric_gradient(
        lambda t: t.cross_entropy([0, 2], "mean"), host)
    wrong = np.ascontiguousarray(numeric[:, ::-1])
    assert not np.allclose(wrong, numeric, rtol=_F32_FD_RTOL,
                           atol=_F32_FD_ATOL)


@needs_native
@pytest.mark.parametrize("dtype", BOTH_DTYPES)
def test_saved_probabilities_carry_the_graph_dtype_and_stay_private(dtype):
    """The saved probabilities are the graph's, at the graph's dtype: never a
    public tensor, never a parameter or buffer, never in a ``state_dict`` or a
    checkpoint, aliasing neither the logits nor the loss."""
    floating = _DTYPE_BITS[dtype][2]
    host = np.array([[0.5, 1.5, -0.5], [2.0, -1.0, 0.25]], dtype=floating)
    x = _tensor(host, dtype, requires_grad=True)
    try:
        loss = x.cross_entropy([2, 0], reduction="mean")
        try:
            assert len(loss._graph_resources) == 1
            probabilities = loss._graph_resources[0]
            assert probabilities.dtype == dtype
            assert probabilities.shape == host.shape
            assert probabilities.storage is not x._core.storage
            assert probabilities.storage is not loss._core.storage
            # It is not reachable through any public surface.
            assert not [name for name in dir(loss)
                        if not name.startswith("_") and "probab" in name]
            assert "probab" not in repr(loss).lower()
        finally:
            loss.close()
    finally:
        x.close()


@needs_native
@pytest.mark.parametrize("dtype", BOTH_DTYPES)
def test_float32_saved_probabilities_follow_the_graph_resource_contract(
        dtype, monkeypatch):
    """The D9 ``graph_resources`` contract, unchanged, at both dtypes: closed
    immediately by a no-grad forward, retained under ``retain_graph``,
    released exactly once by a one-shot backward, and freed by an abandoned
    graph's ``close()``."""
    floating = _DTYPE_BITS[dtype][2]
    host = np.array([[0.5, 1.5, -0.5], [2.0, -1.0, 0.25]], dtype=floating)
    targets = [2, 0]

    captured = {}
    original = cpp.NativeTensorCore.cross_entropy_forward

    def capturing(self, targets_arg, reduction="mean"):
        result = original(self, targets_arg, reduction)
        captured["probabilities"] = result.probabilities
        return result

    monkeypatch.setattr(cpp.NativeTensorCore, "cross_entropy_forward",
                        capturing)

    # A no-grad forward closes the probabilities immediately.
    plain = _tensor(host, dtype)
    try:
        loss = plain.cross_entropy(targets, reduction="sum")
        try:
            assert captured["probabilities"]._closed is True
            assert loss.dtype == dtype
        finally:
            loss.close()
    finally:
        plain.close()

    # retain_graph keeps them; a one-shot backward then releases them once.
    x = _tensor(host, dtype, requires_grad=True)
    try:
        loss = x.cross_entropy(targets, reduction="mean")
        probabilities = loss._graph_resources[0]
        assert probabilities._closed is False
        try:
            loss.backward(retain_graph=True)
            assert probabilities._closed is False
            first = x.grad.to_numpy().copy()
            x.grad.close()
            x._grad = None
            loss.backward()
            assert probabilities._closed is True
            assert np.array_equal(x.grad.to_numpy(), first)
        finally:
            loss.close()
    finally:
        x.close()

    # An abandoned graph frees them through close().
    y = _tensor(host, dtype, requires_grad=True)
    try:
        loss = y.cross_entropy(targets, reduction="sum")
        probabilities = loss._graph_resources[0]
        assert probabilities._closed is False
        loss.close()
        assert probabilities._closed is True
    finally:
        y.close()


@needs_native
@pytest.mark.parametrize("dtype", BOTH_DTYPES)
def test_a_failed_float32_classification_backward_stays_retryable(dtype):
    """A backward that fails on its output allocation keeps the saved
    probabilities alive and the graph retryable, at both dtypes — and the
    retry produces the correct gradient."""
    floating = _DTYPE_BITS[dtype][2]
    host = np.array([[0.5, 1.5, -0.5], [2.0, -1.0, 0.25]], dtype=floating)
    x = _tensor(host, dtype, requires_grad=True)
    try:
        loss = x.cross_entropy([2, 0], reduction="mean")
        probabilities = loss._graph_resources[0]
        try:
            cpp._arm_alloc_failure(1)
            try:
                with pytest.raises(MemoryError):
                    loss.backward()
            finally:
                cpp._arm_alloc_failure(0)
            assert probabilities._closed is False      # saved state survived
            assert x.grad is None
            loss.backward()                            # the retry succeeds
            assert probabilities._closed is True
            assert x.grad is not None and x.grad.dtype == dtype
        finally:
            loss.close()
    finally:
        x.close()


@needs_native
@pytest.mark.parametrize("dtype", BOTH_DTYPES)
def test_a_failed_classification_forward_leaks_nothing_at_any_stage(
        dtype, monkeypatch):
    """The forward allocates the scalar loss and then the probability block. A
    failure at **either** stage must leave live native storage exactly at
    baseline and return no half-built result — and so must a failure during
    Python wrapper construction, after both native allocations succeeded and
    the kernel ran."""
    floating = _DTYPE_BITS[dtype][2]
    host = np.array([[0.5, 1.5, -0.5], [2.0, -1.0, 0.25]], dtype=floating)
    live = _live_storage_ids(monkeypatch)
    core = _core(host, dtype)
    try:
        baseline = set(live)
        # Stage 1: the loss allocation. Stage 2: the probability block.
        for attempt in (1, 2):
            cpp._arm_alloc_failure(attempt)
            try:
                with pytest.raises(MemoryError):
                    core.cross_entropy_forward([2, 0], "mean")
            finally:
                cpp._arm_alloc_failure(0)
            assert set(live) == baseline, attempt
    finally:
        core.close()

    # Wrapper construction: both native allocations succeed and the kernel
    # runs, then building the graph node raises. Both outputs must still be
    # released.
    x = _tensor(host, dtype, requires_grad=True)
    try:
        def exploding(*args, **kwargs):
            raise RuntimeError("wrapper construction failed")

        monkeypatch.setattr(type(x), "_from_op", exploding)
        node_baseline = set(live)
        with pytest.raises(RuntimeError, match="wrapper construction"):
            x.cross_entropy([2, 0], reduction="mean")
        assert set(live) == node_baseline
    finally:
        x.close()


@needs_native
@pytest.mark.parametrize("dtype", BOTH_DTYPES)
def test_repeated_float32_classification_cycles_return_storage_to_baseline(
        dtype, monkeypatch):
    """Fifteen softmax / log-softmax / cross-entropy forward-backward cycles
    leave live native storage exactly at its baseline: outputs, gradients,
    Policy-B temporaries, backward intermediates, and saved probabilities all
    released."""
    floating = _DTYPE_BITS[dtype][2]
    host = np.array([[0.5, 1.5, -0.5], [2.0, -1.0, 0.25]], dtype=floating)
    live = _live_storage_ids(monkeypatch)
    baseline = len(live)
    for _ in range(15):
        x = _tensor(host, dtype, requires_grad=True)
        try:
            # softmax, then log-softmax over a non-last axis, then the fused
            # cross-entropy, then a Policy-B (transposed) softmax. Every
            # intermediate is closed explicitly — the runtime's ownership
            # contract is that cleanup is explicit, never collection.
            probabilities = x.softmax(axis=-1)
            first = probabilities.sum()
            first.backward()
            for tensor in (first, probabilities):
                tensor.close()
            x.grad.close()
            x._grad = None

            logs = x.log_softmax(axis=0)
            second = logs.sum()
            second.backward()
            for tensor in (second, logs):
                tensor.close()
            x.grad.close()
            x._grad = None

            loss = x.cross_entropy([2, 0], "mean")
            loss.backward()
            loss.close()
            x.grad.close()
            x._grad = None

            transposed = x.T
            strided = transposed.softmax(axis=-1)
            fourth = strided.sum()
            fourth.backward()
            for tensor in (fourth, strided, transposed):
                tensor.close()
            x.grad.close()
            x._grad = None
        finally:
            x.close()
    assert len(live) == baseline


@needs_native
def test_mixed_dtype_classification_is_rejected_before_allocation(monkeypatch):
    """Design §9.3 for the four I6 exports: a mixed-dtype call fails before any
    output is allocated, leaves live native storage exactly as it was, leaves
    both operands open and unchanged, and writes nothing to any destination.

    Every participating handle position is driven independently — two for each
    transform, three for each cross-entropy direction — because each could
    have been left out of the guard on its own."""
    library = cpp._require_library()
    live = _live_storage_ids(monkeypatch)
    baseline = set(live)

    f32 = cpp.NativeStorage._typed(12, "float32")
    f64 = cpp.NativeStorage(12)
    loss32 = cpp.NativeStorage._typed(1, "float32")
    loss64 = cpp.NativeStorage(1)
    targets = np.zeros(3, dtype=np.int64)
    try:
        f64.fill(-12345.5)
        loss64.fill(-12345.5)
        a32, a64 = f32._require_open(), f64._require_open()
        l32, l64 = loss32._require_open(), loss64._require_open()
        after_setup = set(live)

        calls = [
            # transforms: source/destination, both directions
            lambda: library.tf_core_softmax_forward(a32, 0, a64, 1, 12, 1),
            lambda: library.tf_core_softmax_forward(a64, 0, a32, 1, 12, 1),
            lambda: library.tf_core_log_softmax_forward(a32, 0, a64, 1, 12, 1),
            lambda: library.tf_core_log_softmax_forward(a64, 0, a32, 1, 12, 1),
            # cross-entropy forward: logits / loss / probabilities
            lambda: library.tf_core_cross_entropy_forward(
                a32, 0, targets, 3, l64, a64, 3, 4, 0),
            lambda: library.tf_core_cross_entropy_forward(
                a64, 0, targets, 3, l32, a64, 3, 4, 0),
            lambda: library.tf_core_cross_entropy_forward(
                a64, 0, targets, 3, l64, a32, 3, 4, 0),
            # cross-entropy backward: probabilities / upstream / gradient
            lambda: library.tf_core_cross_entropy_backward(
                a32, 0, targets, 3, l64, 0, a64, 3, 4, 0),
            lambda: library.tf_core_cross_entropy_backward(
                a64, 0, targets, 3, l32, 0, a64, 3, 4, 0),
            lambda: library.tf_core_cross_entropy_backward(
                a64, 0, targets, 3, l64, 0, a32, 3, 4, 1),
        ]
        for index, call in enumerate(calls):
            with pytest.raises(ValueError, match="same dtype"):
                call()
            # No storage was created, and every float64 destination still
            # holds its sentinel byte for byte.
            assert set(live) == after_setup, index
            assert np.all(f64.to_numpy() == -12345.5), index
            assert np.all(loss64.to_numpy() == -12345.5), index
    finally:
        for storage in (loss64, loss32, f64, f32):
            storage.close()
    assert set(live) == baseline


@needs_native
def test_a_float64_upstream_is_refused_for_a_float32_classification_loss():
    """The autograd-level half of the same rule: a float32 loss handed a
    float64 seed raises before any gradient is mutated, through the existing
    accumulation check rather than a new one."""
    host = np.array([[0.5, 1.5, -0.5], [2.0, -1.0, 0.25]], dtype=np.float32)
    x = _tensor(host, "float32", requires_grad=True)
    try:
        loss = x.cross_entropy([2, 0], reduction="mean")
        try:
            seed = _tensor(np.array(1.0, dtype=np.float64), "float64")
            try:
                with pytest.raises(ValueError):
                    loss.backward(seed)
            finally:
                seed.close()
            assert x.grad is None
        finally:
            loss.close()
    finally:
        x.close()


@needs_native
@pytest.mark.parametrize("dtype", BOTH_DTYPES)
def test_the_uninitialized_classification_destinations_are_fully_written(
        dtype, monkeypatch):
    """The H1 audit re-derived per dtype through the Python seam: the softmax
    destination, the log-softmax destination, the scalar loss, the saved
    probabilities, and the logits gradient are each allocated uninitialized and
    each completely overwritten. The poison is applied by test infrastructure
    around the allocator — no production hook exists — and the negative control
    shows the detector can fail."""
    sentinel = float(2 ** 22 + 3)                # exact at both widths
    floating = _DTYPE_BITS[dtype][2]
    original = cpp.NativeTensorCore._uninitialized.__func__

    def poisoned(cls, shape, dtype="float64", device="cpu"):
        out = original(cls, shape, dtype=dtype, device=device)
        out._storage.fill(sentinel)
        return out

    monkeypatch.setattr(cpp.NativeTensorCore, "_uninitialized",
                        classmethod(poisoned))
    control = cpp.NativeTensorCore._uninitialized((2, 2), dtype=dtype)
    try:
        assert np.all(control.to_numpy() == floating(sentinel))
    finally:
        control.close()

    host = _sample(dtype, 12, seed=39).reshape(3, 4)
    core = _core(host, dtype)
    try:
        for method in ("softmax", "log_softmax"):
            out = getattr(core, method)(axis=-1)
            try:
                assert not np.any(out.to_numpy() == floating(sentinel)), method
            finally:
                out.close()
        result = core.cross_entropy_forward([0, 1, 3], "mean")
        try:
            assert not np.any(result.loss.to_numpy() == floating(sentinel))
            assert not np.any(
                result.probabilities.to_numpy() == floating(sentinel))
            upstream = _core(np.array(1.0, dtype=floating), dtype)
            try:
                grad = result.probabilities.cross_entropy_backward(
                    result.targets, upstream, "mean")
                try:
                    assert not np.any(grad.to_numpy() == floating(sentinel))
                finally:
                    grad.close()
            finally:
                upstream.close()
        finally:
            result.close()
    finally:
        core.close()


@needs_native
def test_the_classification_module_and_metric_need_no_dtype_authority():
    """``NativeCrossEntropyLoss`` stays a thin delegate to
    ``logits.cross_entropy(...)`` and ``native_accuracy`` stays a
    reporting-only helper over ``to_numpy()``. Neither gained a dtype
    argument, neither learned anything about dtype, and both simply work when
    handed a private float32 graph — which is a consequence of the *operation*
    being dtype-general, **not** public float32 module support."""
    import inspect

    from tensorforge.experimental import (NativeCrossEntropyLoss,
                                          native_accuracy)

    assert "dtype" not in inspect.signature(NativeCrossEntropyLoss).parameters
    assert "dtype" not in inspect.signature(native_accuracy).parameters
    source = inspect.getsource(NativeCrossEntropyLoss)
    assert "float64" not in source
    assert "logits.cross_entropy(targets, reduction=self.reduction)" in source

    host = np.array([[0.5, 1.5, -0.5], [2.0, -1.0, 0.25]], dtype=np.float32)
    x = _tensor(host, "float32", requires_grad=True)
    try:
        criterion = NativeCrossEntropyLoss(reduction="mean")
        loss = criterion(x, [2, 0])
        try:
            assert loss.dtype == "float32"
            loss.backward()
            assert x.grad.dtype == "float32"
        finally:
            loss.close()
        accuracy = native_accuracy(x, [2, 0])
        assert isinstance(accuracy, float)
        # Row 0's argmax is class 1, row 1's is class 0: one of the two
        # targets (2, 0) is hit.
        assert accuracy == 0.5
    finally:
        x.close()


# ---------------------------------------------------------------------------
# I6: what the generalized classification source must (and must not) contain
# ---------------------------------------------------------------------------

def test_the_classification_exports_carry_one_dispatch_each():
    """Design §8.1 over the four I6 exports, as source structure: each runs the
    matching-dtype guard and exactly one dispatch switch, and no
    ``require_float64`` survives in the unit."""
    source = _read("cpp/src/classification.cpp")
    for export in ("tf_core_softmax_forward", "tf_core_log_softmax_forward",
                   "tf_core_cross_entropy_forward",
                   "tf_core_cross_entropy_backward"):
        assert f'"{export}",' in source, export
    assert source.count("tf::require_matching_dtype(") == 4
    assert "require_float64" not in source
    assert source.count("switch (tf::dispatch_dtype(") == 4
    assert source.count("case tf::Dtype::Float32:") == 4
    assert source.count("case tf::Dtype::Float64:") == 4
    # No per-dtype duplicate kernel family, and no hand-rolled float64 cast.
    assert "storage_f64" not in source
    for banned in ("_f32(", "_f64(", "softmax_forward_f32"):
        assert banned not in source, banned


def test_no_classification_accumulator_is_hard_coded_to_a_width():
    """The behavioural half of the accumulation proof is the batch-loss
    witness; this is the structural half, and both are kept.

    Every value the four kernels compute is declared at the element type: the
    maximum, the shift, the exponentials, the normalizing sum, the
    log-normalizer, the row loss, the batch total, the mean divisor, the
    gradient contribution, and the upstream scale. A ``double`` anywhere in the
    templated kernels would make one instantiation silently wider than its
    element type."""
    header = _read("cpp/include/tf_classification_internal.h")
    body = header.split("constexpr int64_t kCrossEntropyReductionSum", 1)[1]
    # Comments discuss both widths by name — that is the documentation doing
    # its job. The claim is about the **code**, so the comments come out
    # first, exactly as the equivalent I4/I5 structural checks do.
    code = "\n".join(line.split("//", 1)[0] for line in body.splitlines())
    for banned in ("double", "float", "static_cast<double>",
                   "static_cast<float>", "= 0.0;", "= 1.0;", "-= 1.0"):
        assert banned not in code, banned
    for required in ("T maximum = ", "T total = T(0)", "T sum_exp = T(0)",
                     "const T log_denominator", "const T shifted_target",
                     "static_cast<T>(batch_size)", "contribution -= T(1)",
                     "const T count = static_cast<T>(batch_size)"):
        assert required in body, required
    # exp/log are called on the element type, so a float32 slice takes the
    # float overload rather than widening and narrowing back.
    assert "std::exp(shifted)" in body
    assert "std::log(sum_exp)" in body


def test_cross_entropy_backward_cannot_reach_the_logits():
    """The structural half of "backward never rereads the logits": the logits
    are not a parameter of the C kernel, not a parameter of the export, and not
    a parameter of the Core wrapper — so there is no path by which a later
    change could start reading them without changing a signature. No
    exponential or logarithm is evaluated there either."""
    header = _read("cpp/include/tf_classification_internal.h")
    kernel = header.split("cross_entropy_backward_contiguous", 1)[1]
    signature = kernel.split(") noexcept", 1)[0]
    # ``grad_logits`` is the write-only destination; every *readable*
    # parameter is named here, and none of them is the logits.
    assert "grad_logits" in signature
    assert signature.replace("grad_logits", "") .count("logits") == 0
    for readable in ("const T* probabilities", "const int64_t* targets",
                     "const T* upstream"):
        assert readable in signature, readable
    assert "std::exp" not in kernel
    assert "std::log" not in kernel

    source = _read("cpp/src/classification.cpp")
    export = source.split("TF_EXPORT void tf_core_cross_entropy_backward",
                          1)[1]
    export_signature = export.split(") {", 1)[0]
    assert "grad_logits_handle" in export_signature
    assert export_signature.replace("grad_logits_handle", "") \
        .count("logits") == 0


def test_every_one_of_the_five_python_float64_gates_is_gone():
    """Design §2.3 listed five explicit ``dtype != "float64"`` Core gates.
    I5 opened two (both pooling directions), I6 two more (both cross-entropy
    directions), and I7 the last one, Dropout's. None is left.

    Asserted by exact wording *and* by count, so a gate cannot be reworded
    back into existence, and so the count cannot pass while a differently
    spelled gate stands."""
    source = _read("src/tensorforge/backends/cpp.py")
    for gone in (
        "cross_entropy_forward requires float64/cpu logits",
        "cross_entropy_backward requires a float64/cpu probability",
        "maxpool2d_forward requires a float64/cpu input",
        "dropout_forward requires a float64/cpu input",
    ):
        assert gone not in source, gone
    assert source.count('!= "float64" or self.device != "cpu"') == 0
    # The one surviving ``!= "float64"`` in the file is the MaxPool2d winner
    # buffer's, which is §13.3's permanent pin rather than a §2.3 gate.
    assert source.count('!= "float64"') == 1
    assert 'winners.dtype != "float64"' in source


@needs_native
def test_i6_moved_no_public_capability_at_all():
    """The exit gate, as one assertion block: internal float32 softmax,
    log-softmax, and fused cross-entropy exist, and public float32 support does
    not."""
    import inspect

    import tensorforge.experimental as experimental
    from tensorforge.experimental import native_checkpoint

    _assert_the_public_registry_is_i9s()
    assert cpp.RAW_KERNEL_DTYPES == ("float64",)
    assert cpp.backend_info()["dtype"] == "float64"
    assert native_checkpoint._FORMAT_VERSION == I8_CHECKPOINT_VERSION
    assert (native_checkpoint._SUPPORTED_FORMAT_VERSIONS
            == I8_CHECKPOINT_VERSIONS)
    exports = _source_exports()
    assert len(exports) == I1_EXPORT_COUNT       # still 54; I6 adds none
    for absent in ("tf_core_softmax_forward_f32", "tf_core_softmax_typed",
                   "tf_core_cross_entropy_forward_f32",
                   "tf_core_cross_entropy_targets", "tf_storage_create_int64",
                   "tf_core_log_softmax_forward_f32"):
        assert absent not in exports, absent
    assert not [name for name in exports
                if name.endswith(("_f32", "_f64", "_float32", "_float64"))]
    # I6 itself gave no constructor a dtype argument; the six that have one
    # are milestone I7's, and they are exactly the six.
    assert _constructors_with_a_dtype_argument() == DTYPE_CONSTRUCTORS


# ===========================================================================
# I7: modules, parameters, buffers, initialization, normalization, and
#     Dropout, as running code
#
# Everything below drives the **live** library at both dtypes through the
# constructors design §12.1 opened. The split that matters, and the reason
# this section is long rather than one happy-path check per class:
#
#   * **Construction** is asserted in both directions — the six named
#     classes accept exactly two dtypes and default to float64, and nothing
#     else in ``experimental.__all__`` accepts one at all.
#   * **Initialization** is asserted as a *relation between dtypes*, not as
#     a value table: for one seed the float32 weights are exactly the
#     binary32 narrowing of the float64 weights, which is what makes the
#     seed contract dtype-independent (design §12.3).
#   * **float64 regression** is asserted **bitwise** everywhere it is
#     asserted at all. A milestone that moved a float64 value by one ULP
#     while adding float32 has broken the phase's hardest requirement, and
#     an ``allclose`` would not have noticed.
#   * **float32 correctness** is asserted against **same-dtype** references
#     (an independent NumPy float32 computation, or float32 finite
#     differences), never against the float64 result — design §10.4 forbids
#     making float32-matches-float64 a contract, because it is false by
#     construction for anything that accumulates.
#   * **Rejection** is asserted with its post-condition: after a refused
#     call, live native storage is at baseline, no version moved, no buffer
#     changed, and no generator call was consumed.
# ===========================================================================


def _module_dtype_state(module):
    """Every numeric state tensor a module owns, as ``{name: dtype}`` over
    parameters and buffers together — the thing that must be single-valued
    at the module's dtype."""
    found = {}
    for name, parameter in module.named_parameters():
        found[name] = parameter.dtype
    for name, buffer in module.named_buffers():
        found[name] = buffer.dtype
    return found


def _close_module(module):
    """Release every native object a module owns. Modules have no
    ``close()`` — lifetime stays with the tensors — so this is the shape the
    Phase-F suites use."""
    for parameter in module.parameters():
        parameter.close()
    for buffer in module.buffers():
        buffer.close()


def _release(built):
    """Release whatever a builder produced — a parameter or a module."""
    from tensorforge.experimental import NativeParameter

    if isinstance(built, NativeParameter):
        built.close()
    else:
        _close_module(built)


def _state_owning_builders():
    """One builder per class of the closed I7 set, each taking a ``dtype``
    keyword and nothing else the caller has to know about."""
    from tensorforge.experimental import (
        NativeBatchNorm1d, NativeBatchNorm2d, NativeConv2d, NativeLayerNorm,
        NativeLinear, NativeParameter,
    )

    return (
        ("NativeParameter",
         lambda **kw: NativeParameter(np.arange(6.0).reshape(2, 3), **kw)),
        ("NativeLinear", lambda **kw: NativeLinear(3, 4, seed=7, **kw)),
        ("NativeConv2d", lambda **kw: NativeConv2d(2, 3, 2, seed=7, **kw)),
        ("NativeLayerNorm", lambda **kw: NativeLayerNorm(4, **kw)),
        ("NativeBatchNorm1d", lambda **kw: NativeBatchNorm1d(4, **kw)),
        ("NativeBatchNorm2d", lambda **kw: NativeBatchNorm2d(4, **kw)),
    )


_BUILDER_IDS = [name for name, _ in _state_owning_builders()]


# ---------------------------------------------------------------------------
# I7.1 The constructor surface
# ---------------------------------------------------------------------------

@needs_native
def test_the_dtype_argument_surface_is_exactly_the_six_named_classes():
    """Design §12.1's list, asserted as a closed set in both directions and
    in the exact form it specifies: keyword-only, defaulting to ``None``
    (which means ``"float64"``).

    The absentees are the point. ``NativeReLU``, ``NativeFlatten``,
    ``NativeMaxPool2d``, ``NativeSequential``, ``NativeDropout``,
    ``NativeMSELoss``, ``NativeCrossEntropyLoss``, ``NativeGenerator``, and
    both optimizers own no dtype-bearing numeric state of their own, so a
    dtype argument there would be a **second authority** that could disagree
    with the data flowing through them."""
    import inspect

    import tensorforge.experimental as experimental

    assert _constructors_with_a_dtype_argument() == DTYPE_CONSTRUCTORS
    for name in ("NativeReLU", "NativeFlatten", "NativeMaxPool2d",
                 "NativeSequential", "NativeDropout", "NativeMSELoss",
                 "NativeCrossEntropyLoss", "NativeGenerator", "NativeSGD",
                 "NativeAdam"):
        obj = getattr(experimental, name)
        assert "dtype" not in inspect.signature(obj).parameters, name
    # Phase I adds no device, anywhere, at any of them.
    for name in experimental.__all__:
        obj = getattr(experimental, name)
        if not inspect.isclass(obj):
            continue
        try:
            signature = inspect.signature(obj)
        except (TypeError, ValueError):  # pragma: no cover
            continue
        assert "device" not in signature.parameters, name


@needs_native
@pytest.mark.parametrize("name,build", _state_owning_builders(),
                         ids=_BUILDER_IDS)
def test_every_state_owning_constructor_defaults_to_float64(name, build):
    """The hard compatibility requirement of the phase (design §12.4):
    omitting ``dtype`` produces float64 state, and explicit
    ``dtype="float64"`` produces **byte-identical** state — not merely
    equal, because a constructor that took a different path for the explicit
    form would be a second implementation waiting to drift."""
    default = build()
    explicit = build(dtype="float64")
    try:
        assert default.dtype == "float64"
        assert explicit.dtype == "float64"
        if hasattr(default, "to_numpy"):        # NativeParameter
            assert _same_bits(default.to_numpy(), explicit.to_numpy(),
                              "float64")
            return
        assert _module_dtype_state(default) == _module_dtype_state(explicit)
        state_a = default.state_dict()
        state_b = explicit.state_dict()
        try:
            assert list(state_a) == list(state_b)
            for key in state_a:
                assert _same_bits(state_a[key].to_numpy(),
                                  state_b[key].to_numpy(), "float64"), key
        finally:
            for snapshot in list(state_a.values()) + list(state_b.values()):
                snapshot.close()
    finally:
        _release(explicit)
        _release(default)


@needs_native
@pytest.mark.parametrize("name,build", _state_owning_builders(),
                         ids=_BUILDER_IDS)
def test_every_state_owning_constructor_builds_physical_float32(name, build):
    """``dtype="float32"`` produces state that is physically float32 — the
    storage tag *and* the host array NumPy sees on the way out, so this is
    not a label on a float64 buffer."""
    built = build(dtype="float32")
    try:
        assert built.dtype == "float32"
        if hasattr(built, "to_numpy"):          # NativeParameter
            assert built.to_numpy().dtype == np.float32
            return
        state = _module_dtype_state(built)
        assert state, name          # every one of these owns some state
        assert set(state.values()) == {"float32"}, state
        snapshots = built.state_dict()
        try:
            for key, snapshot in snapshots.items():
                assert snapshot.dtype == "float32", key
                assert snapshot.to_numpy().dtype == np.float32, key
        finally:
            for snapshot in snapshots.values():
                snapshot.close()
    finally:
        _release(built)


@needs_native
@pytest.mark.parametrize("name,build", _state_owning_builders(),
                         ids=_BUILDER_IDS)
def test_every_state_owning_constructor_rejects_everything_else(
    name, build, monkeypatch
):
    """The accepted set is exactly two strings plus ``None``. A non-string
    is a ``TypeError``; any other string is a ``ValueError``. There are no
    aliases and no case or whitespace tolerance, because a permissive front
    door is how a dtype silently becomes a NumPy-coupled type object
    (design §25.1).

    A rejected construction also allocates nothing: the dtype is validated
    before the first native allocation at every one of the six, so live
    storage is exactly what it was."""
    open_ids = _live_storage_ids(monkeypatch)
    baseline = len(open_ids)
    for bad in (np.float32, np.dtype("float32"), np.float64, float, 32, 4,
                True, False, b"float32", ["float32"], ("float32",)):
        with pytest.raises(TypeError):
            build(dtype=bad)
    for bad in ("Float32", "FLOAT32", " float32", "float32 ", "float 32",
                "f4", "f8", "single", "double", "float", "float16",
                "bfloat16", "int32", "int64", "complex64", "cuda", "amp",
                "cpu", ""):
        with pytest.raises(ValueError):
            build(dtype=bad)
    assert len(open_ids) == baseline, (
        f"{name} allocated native storage for a rejected dtype")
    # ...and ``None`` is accepted, meaning float64.
    built = build(dtype=None)
    try:
        assert built.dtype == "float64"
    finally:
        _release(built)


@needs_native
def test_the_module_dtype_validator_is_a_strict_delegate():
    """Design §12.2: *no constructor invents its own dtype validation*. The
    shared helper is a delegate over the internal normalizer, so the module
    surface and the storage layer cannot disagree about what a dtype is —
    asserted by comparing both functions' answers, and both functions'
    exception types and messages, over the same inputs."""
    from tensorforge.experimental import _native_dtype

    assert _native_dtype.MODULE_DTYPES == ("float64", "float32")
    for good in (None, "float64", "float32"):
        assert (_native_dtype.normalize_module_dtype(good)
                == cpp._normalize_internal_dtype(good))
    for bad in ("f4", "Float32", "cuda", "", np.float32, 4, True):
        with pytest.raises(BaseException) as helper:
            _native_dtype.normalize_module_dtype(bad)
        with pytest.raises(BaseException) as direct:
            cpp._normalize_internal_dtype(bad)
        assert type(helper.value) is type(direct.value), bad
        assert str(helper.value) == str(direct.value), bad
    # It is **not** the public validator and must not become one, even now
    # that the two accept the same set. Through I8 the difference was
    # visible — the public registry refused ``"float32"`` outright — and at
    # I9 it stopped being visible without stopping being real: they remain
    # separate functions measured against separate tables, and the delegate
    # answers "what may a module be constructed at?" rather than "what does
    # TensorForge support?".
    assert (_native_dtype.normalize_module_dtype
            is not cpp.normalize_dtype)
    assert set(_native_dtype.MODULE_DTYPES) == set(cpp.SUPPORTED_DTYPES)
    for dtype in _native_dtype.MODULE_DTYPES:
        assert (_native_dtype.normalize_module_dtype(dtype)
                == cpp.normalize_dtype(dtype))
    # ...and it is private: absent from the package's public surface.
    import tensorforge.experimental as experimental
    assert "normalize_module_dtype" not in experimental.__all__
    assert "_native_dtype" not in experimental.__all__


@needs_native
@pytest.mark.parametrize("dtype", BOTH_DTYPES)
def test_the_module_dtype_property_is_read_only_and_reports_the_state(dtype):
    """§25.3: modules that own parameters or buffers expose a read-only
    ``dtype``. It reports the constructed value and is never a setter, so a
    caller cannot relabel a module's state without changing it."""
    for name, build in _state_owning_builders():
        built = build(dtype=dtype)
        try:
            assert built.dtype == dtype, name
            if hasattr(built, "to_numpy"):
                continue        # NativeParameter's dtype is the storage's
            with pytest.raises(AttributeError):
                built.dtype = "float64"
            # ...and it agrees with every piece of state it owns.
            assert set(_module_dtype_state(built).values()) == {dtype}, name
        finally:
            _release(built)


# ---------------------------------------------------------------------------
# I7.2 NativeParameter
# ---------------------------------------------------------------------------

@needs_native
def test_a_float32_parameter_is_the_narrowed_host_array_exactly():
    """Host data crosses the **explicit host-to-native conversion boundary**
    (design §9.4), which has always converted whatever it was given. So a
    float64 host array becomes a float32 parameter by exactly one rounding,
    asserted by raw bit pattern against NumPy's own narrowing — not by
    tolerance, because "one rounding" is a bit-level claim.

    A float32 host array stays exactly itself: converting it again would be
    a second rounding, and there is nothing to round."""
    from tensorforge.experimental import NativeParameter

    host64 = np.linspace(-3.0, 3.0, 17, dtype=np.float64)
    narrowed = host64.astype(np.float32)
    parameter = NativeParameter(host64, dtype="float32")
    try:
        assert parameter.dtype == "float32"
        assert _same_bits(parameter.to_numpy(), narrowed, "float32")
    finally:
        parameter.close()
    host32 = np.linspace(-3.0, 3.0, 17, dtype=np.float32)
    parameter = NativeParameter(host32, dtype="float32")
    try:
        assert _same_bits(parameter.to_numpy(), host32, "float32")
    finally:
        parameter.close()
    # ...and the dtype is never inferred from the array: a float32 array
    # with no dtype argument still produces a float64 parameter (§9.4).
    parameter = NativeParameter(host32)
    try:
        assert parameter.dtype == "float64"
        assert _same_bits(parameter.to_numpy(),
                          host32.astype(np.float64), "float64")
    finally:
        parameter.close()


@needs_native
@pytest.mark.parametrize("dtype", BOTH_DTYPES)
def test_a_native_tensor_source_must_already_carry_the_requested_dtype(
    dtype, monkeypatch
):
    """A host array is data; a **native tensor** is a tensor, and there is
    no tensor cast in this runtime (§9.1/§9.5). So constructing a parameter
    from a live tensor of the other dtype is an invalid request rather than
    a conversion opportunity, in both directions — and the rejection
    allocates nothing."""
    from tensorforge.experimental import NativeParameter

    other = "float32" if dtype == "float64" else "float64"
    source = _tensor(np.arange(6.0).reshape(2, 3), dtype)
    open_ids = _live_storage_ids(monkeypatch)
    baseline = len(open_ids)
    try:
        with pytest.raises(ValueError, match="no casting"):
            NativeParameter(source, dtype=other)
        assert len(open_ids) == baseline
        # ...while the matching request is accepted and takes an
        # independent owning copy.
        parameter = NativeParameter(source, dtype=dtype)
        try:
            assert parameter.dtype == dtype
            assert _same_bits(parameter.to_numpy(), source.to_numpy(), dtype)
            assert parameter._core.storage is not source._core.storage
        finally:
            parameter.close()
    finally:
        source.close()


@needs_native
def test_a_float32_parameter_accumulates_a_float32_gradient():
    """Design §11.2: a leaf's gradient has the leaf's dtype. Proved through
    a real backward rather than by construction, and with the value checked
    too, so a gradient that were somehow allocated at the right dtype but
    filled from the wrong graph would still fail."""
    from tensorforge.experimental import NativeParameter

    parameter = NativeParameter(np.array([2.0, -3.0, 0.5]), dtype="float32")
    try:
        squared = parameter.multiply(parameter)
        total = squared.sum()
        try:
            total.backward()
        finally:
            total.close()
            squared.close()
        assert parameter.grad.dtype == "float32"
        assert parameter.grad.to_numpy().dtype == np.float32
        assert _same_bits(parameter.grad.to_numpy(),
                          np.array([4.0, -6.0, 1.0], dtype=np.float32),
                          "float32")
    finally:
        parameter.close()


@needs_native
def test_float32_parameter_mutation_keeps_identity_version_and_dtype():
    """``copy_value_`` is the one controlled mutation primitive, and its
    dtype rule is unchanged by I7: exact agreement, no cast. A refused
    mutation moves no version and changes no byte; an accepted one moves the
    version by exactly one and preserves the parameter's identity."""
    from tensorforge.experimental import NativeParameter

    parameter = NativeParameter(np.array([1.0, 2.0]), dtype="float32")
    identity = id(parameter)
    float64_source = _tensor(np.array([9.0, 9.0]), "float64")
    float32_source = _tensor(np.array([9.0, 9.0]), "float32")
    try:
        assert parameter.version == 0
        before = parameter.to_numpy().copy()
        with pytest.raises(ValueError, match="dtype/device mismatch"):
            parameter.copy_value_(float64_source)
        assert parameter.version == 0
        assert _same_bits(parameter.to_numpy(), before, "float32")
        assert id(parameter) == identity

        parameter.copy_value_(float32_source)
        assert parameter.version == 1
        assert parameter.dtype == "float32"
        assert id(parameter) == identity
        assert _same_bits(parameter.to_numpy(),
                          np.array([9.0, 9.0], dtype=np.float32), "float32")
    finally:
        float32_source.close()
        float64_source.close()
        parameter.close()


@needs_native
def test_float32_parameter_construction_failure_leaks_nothing(monkeypatch):
    """A failed construction leaves live native storage exactly at
    baseline — the deterministic allocation-failure hook, at the width the
    milestone added."""
    if not cpp.fault_injection_available():  # pragma: no cover
        pytest.skip("the build has no fault-injection hook")
    from tensorforge.experimental import NativeParameter

    open_ids = _live_storage_ids(monkeypatch)
    baseline = len(open_ids)
    cpp._arm_alloc_failure(1)
    try:
        with pytest.raises(MemoryError):
            NativeParameter(np.ones(8), dtype="float32")
    finally:
        cpp._arm_alloc_failure(0)
        cpp._require_library().tf_clear_error()
    assert len(open_ids) == baseline


# ---------------------------------------------------------------------------
# I7.3 Initialization (design §12.3)
# ---------------------------------------------------------------------------

def _init_builders():
    from tensorforge.experimental import NativeConv2d, NativeLinear

    return (
        ("NativeLinear",
         lambda seed, dtype: NativeLinear(5, 3, seed=seed, dtype=dtype)),
        ("NativeConv2d",
         lambda seed, dtype: NativeConv2d(2, 3, (2, 3), seed=seed,
                                          dtype=dtype)),
    )


_INIT_IDS = [name for name, _ in _init_builders()]


@needs_native
@pytest.mark.parametrize("name,build", _init_builders(), ids=_INIT_IDS)
def test_one_seed_gives_the_same_values_at_both_dtypes_to_one_rounding(
    name, build
):
    """The locked relation of design §12.3, stated as an equation and
    asserted as bits::

        weight_f32.bits == float32(weight_f64_draw_for_the_same_seed).bits

    This is what makes the seed contract dtype-**independent**: the host
    draw is the same ``numpy.random.default_rng(seed)`` stream, in the same
    order, at the same sizes, with the bound computed once in binary64. Only
    the ingress conversion differs. Asking NumPy for a float32 stream
    instead would have made two dtypes start from unrelated points, and the
    seed would then have meant something different at each width."""
    reference = build(11, "float64")
    narrowed = build(11, "float32")
    try:
        for key in ("weight", "bias"):
            wide = getattr(reference, key).to_numpy()
            narrow = getattr(narrowed, key).to_numpy()
            assert narrow.dtype == np.float32, key
            assert wide.dtype == np.float64, key
            assert _same_bits(narrow, wide.astype(np.float32), "float32"), key
    finally:
        _release(narrowed)
        _release(reference)


@needs_native
@pytest.mark.parametrize("name,build", _init_builders(), ids=_INIT_IDS)
@pytest.mark.parametrize("dtype", BOTH_DTYPES)
def test_initialization_is_repeatable_and_independent_of_earlier_layers(
    name, build, dtype
):
    """Each constructor owns a **local** generator, so one seed always gives
    one set of values — whatever was constructed before it, at whatever
    dtype. Changing a model's dtype therefore cannot shift any other layer's
    initialization, which is the property that makes a float32 and a
    float64 model comparable at all."""
    first = build(3, dtype)
    noise = build(99, "float32" if dtype == "float64" else "float64")
    second = build(3, dtype)
    try:
        for key in ("weight", "bias"):
            assert _same_bits(getattr(first, key).to_numpy(),
                              getattr(second, key).to_numpy(), dtype), key
    finally:
        _release(second)
        _release(noise)
        _release(first)


@needs_native
@pytest.mark.parametrize("name,build", _init_builders(), ids=_INIT_IDS)
def test_the_float64_initialization_is_byte_identical_to_the_host_draw(
    name, build
):
    """The float64 half of §12.4, against the host stream itself rather than
    against another TensorForge run: the values are exactly what
    ``default_rng(seed).uniform(-bound, bound, size=...)`` produced, in
    order, so nothing about the dtype work perturbed the draw, the bound, or
    the draw count."""
    import math

    module = build(5, "float64")
    try:
        if name == "NativeLinear":
            bound = 1.0 / math.sqrt(5)
            shapes = ((5, 3), (3,))
        else:
            bound = 1.0 / math.sqrt(2 * 2 * 3)
            shapes = ((3, 2, 2, 3), (3,))
        rng = np.random.default_rng(5)
        expected_weight = rng.uniform(-bound, bound, size=shapes[0])
        expected_bias = rng.uniform(-bound, bound, size=shapes[1])
        assert _same_bits(module.weight.to_numpy(), expected_weight, "float64")
        assert _same_bits(module.bias.to_numpy(), expected_bias, "float64")
    finally:
        _release(module)


# ---------------------------------------------------------------------------
# I7.4 NativeLinear and NativeConv2d at float32
# ---------------------------------------------------------------------------

@needs_native
@pytest.mark.parametrize("dtype", BOTH_DTYPES)
@pytest.mark.parametrize("bias", [True, False])
def test_native_linear_forward_and_backward_at_both_dtypes(dtype, bias):
    """The layer's whole numerical surface at one width, against an
    independent **same-dtype sequential** oracle compared bit for bit.

    A NumPy ``@`` would not do here: BLAS reassociates, so it would disagree
    with the kernel at float32 for reasons that have nothing to do with this
    milestone. The oracle below accumulates in source order at the element
    type, which is exactly what the contract says the kernel does."""
    from tensorforge.experimental import NativeLinear

    floating = _DTYPE_BITS[dtype][2]
    module = NativeLinear(4, 3, bias=bias, seed=17, dtype=dtype)
    values = _sample(dtype, 8).reshape(2, 4)
    x = _tensor(values, dtype, requires_grad=True)
    try:
        assert module.weight.dtype == dtype
        if bias:
            assert module.bias.dtype == dtype
        else:
            assert module.bias is None
        out = module(x)
        try:
            assert out.dtype == dtype
            assert out.shape == (2, 3)
            expected = _sequential_matmul(
                values.astype(floating),
                module.weight.to_numpy(), dtype)
            if bias:
                expected = floating(expected + module.bias.to_numpy())
            assert _same_bits(out.to_numpy(), expected, dtype)
            out.sum().backward()
        finally:
            out.close()
        assert x.grad.dtype == dtype
        assert module.weight.grad.dtype == dtype
        if bias:
            assert module.bias.grad.dtype == dtype
            # d(sum)/d(bias) is the batch count, exactly, at either width.
            assert _same_bits(module.bias.grad.to_numpy(),
                              np.full(3, 2.0, dtype=floating), dtype)
    finally:
        x.close()
        _release(module)


@needs_native
@pytest.mark.parametrize("dtype", BOTH_DTYPES)
def test_native_conv2d_forward_and_backward_at_both_dtypes(dtype):
    """The module wiring, not the kernel: I5 already proved all three Conv2d
    directions at both dtypes against independent references, so what is new
    here is that a *module's* parameters reach them at the module's dtype and
    that every gradient comes back at it. A non-contiguous input rides the
    existing Policy-B copy path, which must preserve the dtype too."""
    from tensorforge.experimental import NativeConv2d

    module = NativeConv2d(2, 3, 2, stride=1, padding=1, seed=5, dtype=dtype)
    values = _sample(dtype, 2 * 2 * 4 * 4).reshape(2, 2, 4, 4)
    x = _tensor(values, dtype, requires_grad=True)
    try:
        out = module(x)
        try:
            assert out.dtype == dtype
            assert out.shape == (2, 3, 5, 5)
            out.sum().backward()
        finally:
            out.close()
        assert x.grad.dtype == dtype
        assert module.weight.grad.dtype == dtype
        assert module.bias.grad.dtype == dtype
    finally:
        x.close()
        _release(module)

    # Policy B: a transposed (non-contiguous) input keeps the dtype through
    # the private contiguous copy the kernel needs.
    module = NativeConv2d(2, 2, 2, seed=5, dtype=dtype)
    base = _tensor(_sample(dtype, 2 * 2 * 4 * 4).reshape(2, 4, 4, 2), dtype)
    try:
        view = base.transpose((0, 3, 1, 2))
        try:
            assert not view.contiguous
            out = module(view)
            try:
                assert out.dtype == dtype
            finally:
                out.close()
        finally:
            view.close()
    finally:
        base.close()
        _release(module)


@needs_native
@pytest.mark.parametrize("dtype", BOTH_DTYPES)
def test_a_mismatched_input_is_refused_before_any_graph_or_gradient(
    dtype, monkeypatch
):
    """Design §9.2 and §9.3 at the module layer, for every state-owning
    module that validates an input dtype.

    The post-condition is the interesting half: after the refusal live
    native storage is at baseline, no parameter has a gradient, no
    parameter version moved, and — for BatchNorm — neither running buffer
    moved. A rejection that had already allocated an output or accumulated
    half a gradient would be a far worse failure than a wrong answer."""
    from tensorforge.experimental import (
        NativeBatchNorm1d, NativeBatchNorm2d, NativeConv2d, NativeLayerNorm,
        NativeLinear,
    )

    other = "float32" if dtype == "float64" else "float64"
    cases = (
        ("NativeLinear", lambda: NativeLinear(4, 3, seed=1, dtype=dtype),
         lambda: _sample(other, 8).reshape(2, 4)),
        ("NativeConv2d", lambda: NativeConv2d(2, 2, 2, seed=1, dtype=dtype),
         lambda: _sample(other, 2 * 2 * 4 * 4).reshape(2, 2, 4, 4)),
        ("NativeLayerNorm", lambda: NativeLayerNorm(4, dtype=dtype),
         lambda: _sample(other, 12).reshape(3, 4)),
        ("NativeBatchNorm1d", lambda: NativeBatchNorm1d(4, dtype=dtype),
         lambda: _sample(other, 12).reshape(3, 4)),
        ("NativeBatchNorm2d", lambda: NativeBatchNorm2d(2, dtype=dtype),
         lambda: _sample(other, 2 * 2 * 3 * 3).reshape(2, 2, 3, 3)),
    )
    for name, build, make_input in cases:
        module = build()
        x = _tensor(make_input(), other, requires_grad=True)
        buffers_before = {
            key: buffer.to_numpy().copy()
            for key, buffer in module.named_buffers()
        }
        versions = {key: parameter.version
                    for key, parameter in module.named_parameters()}
        open_ids = _live_storage_ids(monkeypatch)
        baseline = len(open_ids)
        try:
            with pytest.raises(ValueError) as error:
                module(x)
            assert dtype in str(error.value) and other in str(error.value), name
            assert len(open_ids) == baseline, name
            assert x.grad is None, name
            for key, parameter in module.named_parameters():
                assert parameter.grad is None, (name, key)
                assert parameter.version == versions[key], (name, key)
            for key, buffer in module.named_buffers():
                assert _same_bits(buffer.to_numpy(), buffers_before[key],
                                  dtype), (name, key)
        finally:
            monkeypatch.undo()
            x.close()
            _release(module)


@needs_native
def test_internally_inconsistent_module_state_is_refused():
    """The corruption direction: a module whose *own* state has drifted out
    of one dtype. Reachable only by substituting a parameter or a buffer by
    hand, which is exactly what a broken load or a careless caller would
    do — so the forward proves the invariant rather than assuming it."""
    from tensorforge.experimental import (
        NativeBatchNorm1d, NativeLayerNorm, NativeLinear, NativeParameter,
    )

    # weight/bias disagreeing inside one Linear.
    module = NativeLinear(3, 2, seed=1, dtype="float32")
    replacement = NativeParameter(np.zeros(2), dtype="float64")
    original_bias = module.bias
    x = _tensor(np.ones((2, 3), dtype=np.float32), "float32")
    try:
        module.bias = replacement
        with pytest.raises(ValueError):
            module(x)
    finally:
        module.bias = original_bias
        replacement.close()
        x.close()
        _release(module)

    # A LayerNorm whose affine weight has drifted.
    module = NativeLayerNorm(3, dtype="float32")
    replacement = NativeParameter(np.ones(3), dtype="float64")
    original_weight = module.weight
    x = _tensor(np.ones((2, 3), dtype=np.float32), "float32")
    try:
        module.weight = replacement
        with pytest.raises(ValueError):
            module(x)
    finally:
        module.weight = original_weight
        replacement.close()
        x.close()
        _release(module)

    # A BatchNorm whose running_var has drifted from its running_mean: the
    # module-coherence check fires before anything can be committed.
    module = NativeBatchNorm1d(3, dtype="float32")
    stray = _tensor(np.ones(3), "float64")
    original = module.running_var
    x = _tensor(np.ones((4, 3), dtype=np.float32), "float32")
    mean_before = module.running_mean.to_numpy().copy()
    try:
        module.register_buffer("running_var", stray, persistent=True)
        with pytest.raises(ValueError, match="module's dtype"):
            module(x)
        assert _same_bits(module.running_mean.to_numpy(), mean_before,
                          "float32")
    finally:
        module.register_buffer("running_var", original, persistent=True)
        stray.close()
        x.close()
        _release(module)


@needs_native
@pytest.mark.parametrize("dtype", BOTH_DTYPES)
def test_shared_and_frozen_float32_parameters_keep_their_semantics(dtype):
    """Nothing about dtype changes the identity rules: a parameter shared
    between two layers is one object with one gradient, and a frozen one
    stays registered and persisted while accumulating nothing."""
    from tensorforge.experimental import NativeLinear, NativeSequential

    floating = _DTYPE_BITS[dtype][2]
    first = NativeLinear(3, 3, bias=False, seed=2, dtype=dtype)
    second = NativeLinear(3, 3, bias=False, seed=3, dtype=dtype)
    second.weight = first.weight        # deliberate sharing
    model = NativeSequential(first, second)
    x = _tensor(np.ones((2, 3), dtype=floating), dtype)
    try:
        assert len(model.parameters()) == 1
        out = model(x)
        try:
            out.sum().backward()
        finally:
            out.close()
        assert first.weight is second.weight
        assert first.weight.grad.dtype == dtype
    finally:
        x.close()
        first.weight.close()

    frozen = NativeLinear(3, 2, seed=4, requires_grad=False, dtype=dtype)
    x = _tensor(np.ones((2, 3), dtype=floating), dtype, requires_grad=True)
    try:
        out = frozen(x)
        try:
            out.sum().backward()
        finally:
            out.close()
        assert frozen.weight.grad is None
        assert frozen.bias.grad is None
        assert x.grad.dtype == dtype
        assert set(frozen.state_dict()) == {"weight", "bias"}
        for snapshot in frozen.state_dict().values():
            assert snapshot.dtype == dtype
            snapshot.close()
    finally:
        x.close()
        _release(frozen)


@needs_native
def test_native_linear_gradients_pass_float32_finite_differences():
    """The formula, not the last bits: float32 central differences with the
    step and tolerances this module already justified for binary32, plus a
    negative control proving the band can actually reject."""
    from tensorforge.experimental import NativeLinear

    module = NativeLinear(3, 2, seed=8, dtype="float32")
    host = np.array([[0.5, -1.25, 2.0], [1.5, 0.75, -0.5]], dtype=np.float32)
    try:
        def build(tensor):
            return module(tensor).multiply(module(tensor)).sum()

        x = _tensor(host, "float32", requires_grad=True)
        try:
            out = build(x)
            try:
                out.backward()
            finally:
                out.close()
            analytical = x.grad.to_numpy()
        finally:
            x.close()
        numeric = _f32_numeric_gradient(build, host)
        assert np.allclose(analytical, numeric, rtol=_F32_FD_RTOL,
                           atol=_F32_FD_ATOL), (analytical, numeric)
        # Negative control: the band rejects a gradient that is wrong.
        assert not np.allclose(analytical * np.float32(1.5), numeric,
                               rtol=_F32_FD_RTOL, atol=_F32_FD_ATOL)
    finally:
        _release(module)


@needs_native
def test_a_bias_allocation_failure_leaves_no_live_weight_storage(monkeypatch):
    """§20's constructor row, at both widths and at every one of the four
    state-owning modules that allocate more than one object.

    ``NativeLinear`` had never been given the deterministic cleanup its
    younger siblings have, so a failed bias allocation abandoned the
    weight's storage to garbage collection. That is a real leak — a module
    the caller never receives and therefore can never close — and it is
    fixed here rather than documented."""
    if not cpp.fault_injection_available():  # pragma: no cover
        pytest.skip("the build has no fault-injection hook")
    from tensorforge.experimental import (
        NativeBatchNorm1d, NativeConv2d, NativeLayerNorm, NativeLinear,
    )

    from tensorforge.experimental import (
        native_batchnorm, native_conv2d, native_layernorm, native_linear,
    )

    builders = (
        ("NativeLinear", native_linear,
         lambda dtype: NativeLinear(4, 3, seed=1, dtype=dtype)),
        ("NativeConv2d", native_conv2d,
         lambda dtype: NativeConv2d(2, 3, 2, seed=1, dtype=dtype)),
        ("NativeLayerNorm", native_layernorm,
         lambda dtype: NativeLayerNorm(4, dtype=dtype)),
        ("NativeBatchNorm1d", native_batchnorm,
         lambda dtype: NativeBatchNorm1d(4, dtype=dtype)),
    )
    for name, module_namespace, build in builders:
        for dtype in BOTH_DTYPES:
            # Spy on the parameter constructor rather than counting live
            # storage: the half-built module is unreachable the moment
            # ``__init__`` re-raises, so a garbage-collected cleanup would
            # make a storage count pass without the constructor having done
            # anything. What must be true is that the constructor closed it
            # **itself**, deterministically, before propagating.
            created = []
            real = module_namespace.NativeParameter

            def spy(*args, **kwargs):
                parameter = real(*args, **kwargs)
                created.append(parameter)
                return parameter

            monkeypatch.setattr(module_namespace, "NativeParameter", spy)
            cpp._arm_alloc_failure(2)        # the second allocation fails
            try:
                with pytest.raises(MemoryError):
                    build(dtype)
            finally:
                cpp._arm_alloc_failure(0)
                cpp._require_library().tf_clear_error()
                monkeypatch.undo()
            assert len(created) == 1, (name, dtype)
            assert created[0].closed, (name, dtype)


# ---------------------------------------------------------------------------
# I7.5 Normalization at float32
#
# LayerNorm and both BatchNorm shapes are **compositions** of I3/I4
# operations, so I7 adds no kernel and no export to them; what it adds is
# that every operand, constant, buffer, and temporary is at the graph's own
# width. The references below are written at that width, in the layer's own
# operation order, so the comparison is same-dtype throughout.
# ---------------------------------------------------------------------------

def _layer_norm_reference(values, weight, bias, eps, axes, floating):
    """LayerNorm evaluated entirely in ``floating``, in the module's own
    order: sequential single-axis means, the population variance, epsilon
    **inside** the root, then the affine step. Every intermediate is
    narrowed back to the width, so this is not a float64 computation wearing
    a float32 hat."""
    def mean_over(array):
        out = array
        for axis in axes:
            out = out.mean(axis=axis, keepdims=True, dtype=floating)
        return out.astype(floating)

    values = values.astype(floating)
    mean = mean_over(values)
    centered = floating(values - mean)
    variance = mean_over(floating(centered * centered))
    inverse = floating(floating(1.0) / floating(np.sqrt(
        floating(variance + floating(eps)))))
    normalized = floating(centered * inverse)
    if weight is None:
        return normalized
    return floating(floating(normalized * weight) + bias)


@needs_native
@pytest.mark.parametrize("dtype", BOTH_DTYPES)
@pytest.mark.parametrize("affine", [True, False])
@pytest.mark.parametrize("normalized_shape", [4, (2, 4)])
def test_native_layer_norm_at_both_dtypes(dtype, affine, normalized_shape):
    """Forward and backward at one width, with the output compared against a
    same-dtype reference under an honest tolerance.

    The tolerance is *not* laziness: the composition runs ``sqrt`` and
    ``reciprocal``, and NumPy's reduction order is its own, so a bitwise
    claim here would be a claim about NumPy rather than about TensorForge.
    What is asserted exactly is what the contract states exactly — the
    dtype of every output, gradient, and parameter."""
    from tensorforge.experimental import NativeLayerNorm

    floating = _DTYPE_BITS[dtype][2]
    module = NativeLayerNorm(normalized_shape, elementwise_affine=affine,
                             dtype=dtype)
    shape = (3,) + (normalized_shape if isinstance(normalized_shape, tuple)
                    else (normalized_shape,))
    values = _sample(dtype, int(np.prod(shape))).reshape(shape)
    x = _tensor(values, dtype, requires_grad=True)
    try:
        out = module(x)
        try:
            assert out.dtype == dtype
            assert out.shape == shape
            k = len(shape) - 1 if not isinstance(normalized_shape, tuple) \
                else len(shape) - len(normalized_shape)
            axes = tuple(range(k, len(shape)))
            expected = _layer_norm_reference(
                values,
                module.weight.to_numpy() if affine else None,
                module.bias.to_numpy() if affine else None,
                module.eps, axes, floating)
            tolerance = 1e-12 if dtype == "float64" else 2e-6
            assert np.allclose(out.to_numpy(), expected, rtol=tolerance,
                               atol=tolerance), (dtype, affine)
            out.sum().backward()
        finally:
            out.close()
        assert x.grad.dtype == dtype
        if affine:
            assert module.weight.grad.dtype == dtype
            assert module.bias.grad.dtype == dtype
    finally:
        x.close()
        _release(module)


@needs_native
def test_a_non_affine_layer_norm_normalizes_whatever_dtype_it_is_given():
    """``elementwise_affine=False`` owns no numeric state, so its
    constructed ``dtype`` is a *report* and never an enforcement — the layer
    normalizes a float32 input and a float64 input alike, because there is
    nothing of its own for either to disagree with.

    Enforcing here would invent a second authority over data the module does
    not own, which is exactly what design §12.1 rejects for the stateless
    modules."""
    from tensorforge.experimental import NativeLayerNorm

    module = NativeLayerNorm(4, elementwise_affine=False, dtype="float32")
    assert module.dtype == "float32"
    assert module.parameters() == []
    assert module.state_dict() == {}
    for dtype in BOTH_DTYPES:
        floating = _DTYPE_BITS[dtype][2]
        x = _tensor(np.arange(8.0, dtype=floating).reshape(2, 4), dtype)
        try:
            out = module(x)
            try:
                assert out.dtype == dtype
            finally:
                out.close()
        finally:
            x.close()


@needs_native
def test_float32_layer_norm_gradients_pass_float32_finite_differences():
    """The formula at binary32, with the negative control that proves the
    band rejects."""
    from tensorforge.experimental import NativeLayerNorm

    module = NativeLayerNorm(4, dtype="float32")
    host = np.array([[0.5, -1.25, 2.0, 0.75],
                     [1.5, 0.25, -0.5, 1.75]], dtype=np.float32)
    try:
        def build(tensor):
            out = module(tensor)
            return out.multiply(out).sum()

        x = _tensor(host, "float32", requires_grad=True)
        try:
            value = build(x)
            try:
                value.backward()
            finally:
                value.close()
            analytical = x.grad.to_numpy()
        finally:
            x.close()
        numeric = _f32_numeric_gradient(build, host)
        assert np.allclose(analytical, numeric, rtol=_F32_FD_RTOL,
                           atol=_F32_FD_ATOL), (analytical, numeric)
        assert not np.allclose(analytical + np.float32(0.5), numeric,
                               rtol=_F32_FD_RTOL, atol=_F32_FD_ATOL)
    finally:
        _release(module)


@needs_native
@pytest.mark.parametrize("dtype", BOTH_DTYPES)
@pytest.mark.parametrize("kind", ["1d", "2d"])
def test_batch_norm_training_and_evaluation_at_both_dtypes(dtype, kind):
    """The whole stateful composition at one width: the training statistics,
    the normalized output, both running updates, and the evaluation path
    that reads snapshots instead.

    The running update is checked against a same-dtype reference of the
    documented formula, and the **buffer initialization** is checked
    exactly — zeros and ones are representable at every width, so there is
    nothing to round and a tolerance would be hiding something."""
    from tensorforge.experimental import NativeBatchNorm1d, NativeBatchNorm2d

    floating = _DTYPE_BITS[dtype][2]
    features = 3
    if kind == "1d":
        module = NativeBatchNorm1d(features, dtype=dtype)
        shape = (5, features)
        axes = (0,)
    else:
        module = NativeBatchNorm2d(features, dtype=dtype)
        shape = (2, features, 3, 3)
        axes = (0, 2, 3)
    values = _sample(dtype, int(np.prod(shape))).reshape(shape)
    x = _tensor(values, dtype, requires_grad=True)
    try:
        # Buffers start at exactly zeros and ones, at the module's dtype.
        assert module.running_mean.dtype == dtype
        assert module.running_var.dtype == dtype
        assert _same_bits(module.running_mean.to_numpy(),
                          np.zeros(features, dtype=floating), dtype)
        assert _same_bits(module.running_var.to_numpy(),
                          np.ones(features, dtype=floating), dtype)

        module.train()
        out = module(x)
        try:
            assert out.dtype == dtype
            assert out.shape == shape
            out.sum().backward()
        finally:
            out.close()
        assert x.grad.dtype == dtype
        assert module.gamma.grad.dtype == dtype
        assert module.beta.grad.dtype == dtype
        # ...and both buffers advanced, still at the module's dtype, to
        # (1 - m) * old + m * batch over the documented axes.
        wide = values.astype(floating)
        batch_mean = wide.mean(axis=axes, dtype=floating)
        batch_var = ((wide - wide.mean(axis=axes, keepdims=True,
                                       dtype=floating)) ** 2).mean(
            axis=axes, dtype=floating)
        momentum = floating(module.momentum)
        expected_mean = floating(
            floating(floating(1.0) - momentum) * floating(0.0)
            + momentum * batch_mean)
        expected_var = floating(
            floating(floating(1.0) - momentum) * floating(1.0)
            + momentum * batch_var)
        tolerance = 1e-12 if dtype == "float64" else 2e-6
        assert module.running_mean.dtype == dtype
        assert module.running_var.dtype == dtype
        assert np.allclose(module.running_mean.to_numpy(), expected_mean,
                           rtol=tolerance, atol=tolerance)
        assert np.allclose(module.running_var.to_numpy(), expected_var,
                           rtol=tolerance, atol=tolerance)

        # Evaluation reads snapshots, changes nothing, and stays at width.
        module.eval()
        frozen_mean = module.running_mean.to_numpy().copy()
        frozen_var = module.running_var.to_numpy().copy()
        out = module(x)
        try:
            assert out.dtype == dtype
        finally:
            out.close()
        assert _same_bits(module.running_mean.to_numpy(), frozen_mean, dtype)
        assert _same_bits(module.running_var.to_numpy(), frozen_var, dtype)
    finally:
        x.close()
        _release(module)


@needs_native
@pytest.mark.parametrize("dtype", BOTH_DTYPES)
def test_batch_norm_eval_snapshots_are_independent_at_the_graph_dtype(dtype):
    """§13.1 and §21 together: the evaluation graph holds **independent
    owning** copies of the running buffers at the graph's dtype, so a later
    training step cannot reach back into a graph that was already built —
    and the snapshots are released with the graph history, exactly once."""
    from tensorforge.experimental import NativeBatchNorm1d

    floating = _DTYPE_BITS[dtype][2]
    module = NativeBatchNorm1d(3, dtype=dtype)
    x = _tensor(_sample(dtype, 12).reshape(4, 3), dtype, requires_grad=True)
    try:
        module.eval()
        out = module(x)
        try:
            assert out.dtype == dtype
            before = out.to_numpy().copy()
            # Mutate the running buffers underneath the built graph.
            module.train()
            other = _tensor(_sample(dtype, 12).reshape(4, 3) + 5.0, dtype)
            try:
                module(other).close()
            finally:
                other.close()
            # The already-built eval graph is unaffected: its operands are
            # snapshots, not the live buffers.
            assert _same_bits(out.to_numpy(), before, dtype)
            out.sum().backward()
        finally:
            out.close()
        assert x.grad.dtype == dtype
        assert x.grad.to_numpy().dtype == floating
    finally:
        x.close()
        _release(module)


@needs_native
@pytest.mark.parametrize("dtype", BOTH_DTYPES)
def test_the_two_buffer_transaction_refuses_a_mismatched_replacement(dtype):
    """§7 of the milestone: the atomic two-buffer running-state transaction
    gains one dtype validation and nothing else.

    A replacement at the wrong width is refused **before either** live
    buffer changes, so the two never diverge — the failure mode the
    transaction exists to prevent, now reachable through dtype as well as
    through shape."""
    from tensorforge.experimental import NativeBatchNorm1d

    other = "float32" if dtype == "float64" else "float64"
    module = NativeBatchNorm1d(3, dtype=dtype)
    stray = _tensor(np.zeros(3), other)
    try:
        mean_before = module.running_mean.to_numpy().copy()
        var_before = module.running_var.to_numpy().copy()
        mean_id = id(module.running_mean)
        var_id = id(module.running_var)
        good = _tensor(np.full(3, 0.25), dtype)
        try:
            with pytest.raises(ValueError):
                module._commit_running_state(
                    module.running_mean, module.running_var, stray, good)
            with pytest.raises(ValueError):
                module._commit_running_state(
                    module.running_mean, module.running_var, good, stray)
        finally:
            good.close()
        assert _same_bits(module.running_mean.to_numpy(), mean_before, dtype)
        assert _same_bits(module.running_var.to_numpy(), var_before, dtype)
        assert id(module.running_mean) == mean_id
        assert id(module.running_var) == var_id
        # ...and the module still works afterwards.
        x = _tensor(_sample(dtype, 12).reshape(4, 3), dtype)
        try:
            module.train()
            module(x).close()
        finally:
            x.close()
        assert not _same_bits(module.running_mean.to_numpy(), mean_before,
                              dtype)
    finally:
        stray.close()
        _release(module)


@needs_native
def test_normalization_still_adds_no_kernel_export_or_numpy_compute():
    """The Phase-F architectural guarantee, re-asserted at the milestone
    that made normalization dtype-general: it is still **composition**.

    No normalization kernel, no normalization export, no ``NativeTensor``
    normalization operation, no ctypes import in either module, and no
    NumPy anywhere but the constructor's host-side data preparation."""
    import re

    from tensorforge import experimental
    from tensorforge.experimental import native_tensor

    for name in ("layer_norm", "batch_norm", "normalize", "fused_norm"):
        assert not hasattr(native_tensor.NativeTensor, name), name
        assert not hasattr(cpp.NativeTensorCore, name), name
        assert name not in experimental.__all__, name
    exports = _source_exports()
    assert not [name for name in exports if "norm" in name.lower()]
    assert len(exports) == I1_EXPORT_COUNT
    for relative in ("src/tensorforge/experimental/native_batchnorm.py",
                     "src/tensorforge/experimental/native_layernorm.py"):
        source = _read(relative)
        for line in re.findall(r"^\s*(?:from|import)\s+\S+.*$", source, re.M):
            assert "ctypes" not in line, (relative, line)
            assert "backends" not in line, (relative, line)
            assert "NativeTensorCore" not in line, (relative, line)
        assert not re.search(r"\bcpp\.\w", source), relative
        assert not re.search(r"\bNativeTensorCore\.\w", source), relative
        # The dtype-aware scalars are built through the tensor-level private
        # typed constructors, never through the public dtype-gated ones,
        # which would pin a float32 graph's constants to float64.
        body = source.split('"""', 2)[-1]
        assert "NativeTensor.full(" not in body, relative
        assert "NativeTensor.zeros(" not in body, relative


# ---------------------------------------------------------------------------
# I7.6 State dictionaries
# ---------------------------------------------------------------------------

@needs_native
@pytest.mark.parametrize("dtype", BOTH_DTYPES)
def test_state_dict_round_trips_at_both_dtypes_without_casting(dtype):
    """``state_dict()`` stays ``{name: NativeTensor}`` and carries dtype
    **implicitly**, through the tensors — no top-level manifest is added.
    The snapshots are independent owning copies, so closing them cannot
    touch the module and mutating the module cannot change them."""
    from tensorforge.experimental import NativeBatchNorm1d

    module = NativeBatchNorm1d(3, dtype=dtype)
    target = NativeBatchNorm1d(3, dtype=dtype)
    x = _tensor(_sample(dtype, 12).reshape(4, 3), dtype)
    try:
        module.train()
        module(x).close()           # move the buffers off their defaults
        snapshot = module.state_dict()
        try:
            assert list(snapshot) == ["gamma", "beta", "running_mean",
                                      "running_var"]
            for key, value in snapshot.items():
                assert value.dtype == dtype, key
                assert value.to_numpy().dtype == _DTYPE_BITS[dtype][2], key
            versions = {k: p.version for k, p in target.named_parameters()}
            identities = {k: id(t) for k, t in
                          list(target.named_parameters())
                          + list(target.named_buffers())}
            target.load_state_dict(snapshot)
            for key, value in snapshot.items():
                live = (dict(target.named_parameters())
                        | dict(target.named_buffers()))[key]
                assert live.dtype == dtype, key
                assert _same_bits(live.to_numpy(), value.to_numpy(), dtype), key
            # Identities preserved; parameter versions moved exactly once;
            # buffers carry no version and move none.
            for key, tensor in (list(target.named_parameters())
                                + list(target.named_buffers())):
                assert id(tensor) == identities[key], key
            for key, parameter in target.named_parameters():
                assert parameter.version == versions[key] + 1, key
        finally:
            for value in snapshot.values():
                value.close()
    finally:
        x.close()
        _release(target)
        _release(module)


@needs_native
@pytest.mark.parametrize("dtype", BOTH_DTYPES)
def test_cross_dtype_state_loading_is_refused_transactionally(dtype):
    """§25.4: the loader validates dtype per entry against the live
    destination and never casts — in **both** directions, and with the whole
    load rolled back when a single entry disagrees.

    The mixed case is the one that matters: three matching entries and one
    mismatched one must leave all four destinations exactly as they were.
    A loader that validated entry by entry as it committed would have
    written three of them."""
    from tensorforge.experimental import NativeBatchNorm1d

    other = "float32" if dtype == "float64" else "float64"
    source = NativeBatchNorm1d(3, dtype=dtype)
    target = NativeBatchNorm1d(3, dtype=other)
    x = _tensor(_sample(dtype, 12).reshape(4, 3), dtype)
    try:
        source.train()
        source(x).close()
        snapshot = source.state_dict()
        try:
            before = {k: t.to_numpy().copy() for k, t in
                      (list(target.named_parameters())
                       + list(target.named_buffers()))}
            versions = {k: p.version for k, p in target.named_parameters()}
            with pytest.raises(ValueError, match="dtype mismatch"):
                target.load_state_dict(snapshot)
            for key, tensor in (list(target.named_parameters())
                                + list(target.named_buffers())):
                assert _same_bits(tensor.to_numpy(), before[key], other), key
                assert tensor.dtype == other, key
            for key, parameter in target.named_parameters():
                assert parameter.version == versions[key], key
        finally:
            for value in snapshot.values():
                value.close()

        # ...and now the mixed case: one entry of four at the wrong width.
        same = NativeBatchNorm1d(3, dtype=dtype)
        try:
            snapshot = same.state_dict()
            stray = _tensor(np.zeros(3), other)
            try:
                snapshot["running_var"].close()
                snapshot["running_var"] = stray
                before = {k: t.to_numpy().copy() for k, t in
                          (list(source.named_parameters())
                           + list(source.named_buffers()))}
                versions = {k: p.version for k, p in source.named_parameters()}
                with pytest.raises(ValueError, match="dtype mismatch"):
                    source.load_state_dict(snapshot)
                for key, tensor in (list(source.named_parameters())
                                    + list(source.named_buffers())):
                    assert _same_bits(tensor.to_numpy(), before[key], dtype), key
                for key, parameter in source.named_parameters():
                    assert parameter.version == versions[key], key
            finally:
                stray.close()
                for key, value in snapshot.items():
                    if value is not stray:
                        value.close()
        finally:
            _release(same)
    finally:
        x.close()
        _release(target)
        _release(source)


# ---------------------------------------------------------------------------
# I7.7 The checkpoint boundary stays at version 2
# ---------------------------------------------------------------------------

@needs_native
def test_a_float32_model_round_trips_through_a_version_3_checkpoint(tmp_path):
    """§16-§17 at the I8 line: a float32 model saves and reloads **bitwise**
    under format version 3, which declares the dtype rather than implying it.

    The I7 refusal this replaces was correct for a version-2 archive and is
    still correct *for a version-2 archive* — that half is proved by the
    dedicated v1/v2 compatibility test. What changed is that there is now a
    version that can say "float32", so the save no longer has to refuse."""
    from tensorforge.experimental import (
        NativeLinear, load_native_checkpoint, native_checkpoint,
        save_native_checkpoint,
    )

    assert native_checkpoint._FORMAT_VERSION == I8_CHECKPOINT_VERSION
    assert (native_checkpoint._SUPPORTED_FORMAT_VERSIONS
            == I8_CHECKPOINT_VERSIONS)

    path = tmp_path / "float32.npz"
    source = NativeLinear(3, 2, seed=1, dtype="float32")
    restored = NativeLinear(3, 2, seed=9, dtype="float32")
    try:
        save_native_checkpoint(path, source)
        with np.load(path, allow_pickle=False) as archive:
            manifest = json.loads(
                archive["manifest"].tobytes().decode("utf-8")
            )
            # The payload is genuinely float32 — not float64 widened.
            for name in archive.files:
                if name != "manifest":
                    assert archive[name].dtype == np.float32, name
        assert manifest["format_version"] == 3
        for key in ("weight", "bias"):
            assert manifest["model"]["entries"][key]["dtype"] == "float32"

        load_native_checkpoint(path, restored)
        for key in ("weight", "bias"):
            assert _same_bits(getattr(restored, key).to_numpy(),
                              getattr(source, key).to_numpy(), "float32")
            assert getattr(restored, key).dtype == "float32"
    finally:
        _release(restored)
        _release(source)

    # A float64 model round-trips exactly as before, at the same version.
    reference = NativeLinear(3, 2, seed=1, dtype="float64")
    target64 = NativeLinear(3, 2, seed=2, dtype="float64")
    path64 = tmp_path / "float64.npz"
    try:
        save_native_checkpoint(path64, reference)
        load_native_checkpoint(path64, target64)
        for key in ("weight", "bias"):
            assert _same_bits(getattr(target64, key).to_numpy(),
                              getattr(reference, key).to_numpy(), "float64")
    finally:
        _release(target64)
        _release(reference)

    # Dtype is matched exactly, never converted: a float64 archive is
    # refused by a float32 model and a float32 archive by a float64 model,
    # each **before** anything is replaced. No cast, no map_location.
    for archive_path, module_dtype in ((path64, "float32"),
                                       (path, "float64")):
        wrong = NativeLinear(3, 2, seed=3, dtype=module_dtype)
        try:
            before = {k: p.to_numpy().copy()
                      for k, p in wrong.named_parameters()}
            versions = {k: p.version for k, p in wrong.named_parameters()}
            with pytest.raises(ValueError, match="dtype"):
                load_native_checkpoint(archive_path, wrong)
            for key, parameter in wrong.named_parameters():
                assert _same_bits(parameter.to_numpy(), before[key],
                                  module_dtype)
                assert parameter.version == versions[key]
        finally:
            _release(wrong)


# ---------------------------------------------------------------------------
# I7.8 Dropout at float32
#
# The C++ half is proved by cpp/tests/test_dtype_dropout.cpp, including the
# committed keep patterns at both widths and the narrow-once scale witness.
# What is proved here is the Python half: the operation, the module, the
# graph-owned mask, and — the part no C++ test can see — the generator call
# accounting, which must be identical at both widths on every path.
# ---------------------------------------------------------------------------

def _drop_pattern(mask_values):
    """The keep/drop pattern of a multiplier mask, as a tuple of bools."""
    return tuple(bool(value != 0) for value in mask_values.ravel())


@needs_native
@pytest.mark.parametrize("dtype", BOTH_DTYPES)
def test_the_dropout_operation_produces_output_and_mask_at_the_input_dtype(
    dtype
):
    """The Core contract at one width: both destinations are fresh owning
    contiguous storage at the **input's** dtype, the mask holds exactly two
    values, and the kept one is the binary64 reciprocal narrowed once."""
    floating = _DTYPE_BITS[dtype][2]
    core = _core(_sample(dtype, 24).reshape(4, 6), dtype)
    try:
        out, mask = core._dropout_forward_with_mask(0.25, seed=5,
                                                    call_index=2)
        try:
            assert out.dtype == dtype and mask.dtype == dtype
            assert out.to_numpy().dtype == floating
            assert mask.to_numpy().dtype == floating
            assert out.shape == core.shape and mask.shape == core.shape
            # Independent owning storage, aliasing neither the input nor
            # each other.
            assert out.storage is not mask.storage
            assert out.storage is not core.storage
            assert mask.storage is not core.storage
            scale = floating(1.0 / (1.0 - 0.25))
            values = set(np.unique(mask.to_numpy()).tolist())
            assert values <= {0.0, float(scale)}, values
            for value in mask.to_numpy().ravel():
                assert (_bits(np.array([value], dtype=floating), dtype)[0]
                        in (_bits(np.array([0.0], dtype=floating), dtype)[0],
                            _bits(np.array([scale], dtype=floating), dtype)[0]))
            # output == input * mask, exactly, at the element type.
            assert _same_bits(out.to_numpy(),
                              floating(core.to_numpy() * mask.to_numpy()),
                              dtype)
        finally:
            mask.close()
            out.close()
    finally:
        core.close()


@needs_native
@pytest.mark.parametrize("p", [0.1, 0.25, 0.5, 0.75, 0.9])
def test_one_random_key_drops_the_same_elements_at_both_dtypes(p):
    """Design §14.2's deliberate property, from Python: the uniform draw
    stays binary64 at every width, so a float32 Dropout and a float64
    Dropout with the same ``(seed, call_index, element count)`` key drop
    **exactly** the same elements.

    Asserted as a direct comparison of the two patterns, not as two
    independent agreements with a table, and over enough elements that a
    coincidence is not a plausible explanation."""
    count = 512
    patterns = {}
    for dtype in BOTH_DTYPES:
        floating = _DTYPE_BITS[dtype][2]
        core = _core(np.ones(count, dtype=floating), dtype)
        try:
            out, mask = core._dropout_forward_with_mask(p, seed=1234,
                                                        call_index=7)
            try:
                patterns[dtype] = _drop_pattern(mask.to_numpy())
            finally:
                mask.close()
                out.close()
        finally:
            core.close()
    assert patterns["float32"] == patterns["float64"]
    # ...and the pattern is not degenerate, so the equality means something.
    kept = sum(patterns["float64"])
    assert 0 < kept < count


@needs_native
@pytest.mark.parametrize("dtype", BOTH_DTYPES)
def test_the_float32_dropout_graph_backward_multiplies_by_the_saved_mask(
    dtype
):
    """The autograd half: the gradient is ``upstream * mask`` at the graph's
    dtype, over the private mask the forward saved — never a redraw."""
    from tensorforge.experimental import NativeGenerator

    floating = _DTYPE_BITS[dtype][2]
    generator = NativeGenerator(2024)
    x = _tensor(np.full(64, 2.0, dtype=floating), dtype, requires_grad=True)
    try:
        out = x.dropout(0.5, generator=generator)
        try:
            assert out.dtype == dtype
            observed = out.to_numpy()
            out.sum().backward()
        finally:
            out.close()
        assert x.grad.dtype == dtype
        # grad == mask, because the upstream seed is ones and the mask is
        # what the forward divided the kept values by.
        expected = floating(observed / floating(2.0))
        assert _same_bits(x.grad.to_numpy(), expected, dtype)
    finally:
        x.close()


@needs_native
@pytest.mark.parametrize("dtype", BOTH_DTYPES)
def test_dropout_call_accounting_is_identical_at_both_dtypes(dtype):
    """§16 of the milestone, at both widths and on every path.

    A successful stochastic forward consumes exactly one call; evaluation
    and ``p == 0`` return the input object itself and consume none; and a
    **failed** forward consumes none, leaving the same index free for the
    next one. The generator itself is dtype-free and must stay that way."""
    import inspect

    from tensorforge.experimental import NativeDropout, NativeGenerator

    floating = _DTYPE_BITS[dtype][2]
    assert "dtype" not in inspect.signature(NativeGenerator).parameters
    assert "dtype" not in inspect.signature(NativeDropout).parameters

    generator = NativeGenerator(99)
    module = NativeDropout(0.5, generator=generator)
    x = _tensor(np.ones(16, dtype=floating), dtype)
    try:
        assert generator.calls == 0
        module.train()
        out = module(x)
        try:
            assert out.dtype == dtype
            assert out is not x
        finally:
            out.close()
        assert generator.calls == 1

        # Evaluation: the caller's own object, nothing consumed.
        module.eval()
        assert module(x) is x
        assert generator.calls == 1

        # p == 0: identity, nothing consumed, whatever the mode.
        identity = NativeDropout(0.0, generator=generator)
        identity.train()
        assert identity(x) is x
        assert generator.calls == 1

        # A failed forward abandons its reservation.
        module.train()
        closed = _tensor(np.ones(4, dtype=floating), dtype)
        closed.close()
        with pytest.raises(RuntimeError):
            module(closed)
        assert generator.calls == 1

        # ...and the next successful forward takes the index the failure
        # did not spend.
        out = module(x)
        try:
            assert out.dtype == dtype
        finally:
            out.close()
        assert generator.calls == 2
        assert generator.algorithm == "tensorforge.splitmix64"
        assert generator.algorithm_version == 1
    finally:
        x.close()


@needs_native
@pytest.mark.parametrize("dtype", BOTH_DTYPES)
def test_the_float32_dropout_mask_rides_the_graph_resource_contract(
    dtype, monkeypatch
):
    """§21 at the new width: the saved mask is released **exactly once**
    with the graph history, retained under ``retain_graph=True``, released
    immediately by a no-grad forward, released by an abandoned graph's
    ``close()``, and never exposed anywhere. Live native storage returns to
    baseline after each shape."""
    from tensorforge.experimental import NativeGenerator

    floating = _DTYPE_BITS[dtype][2]
    generator = NativeGenerator(7)
    open_ids = _live_storage_ids(monkeypatch)

    # 1. no-grad forward: the mask is closed as soon as the forward returns,
    #    so only the output survives.
    x = _tensor(np.ones(32, dtype=floating), dtype)
    baseline = len(open_ids)
    out = x.dropout(0.5, generator=generator)
    assert len(open_ids) == baseline + 1        # the output, and nothing else
    out.close()
    assert len(open_ids) == baseline
    x.close()

    # 2. one-shot backward releases the mask exactly once.
    x = _tensor(np.ones(32, dtype=floating), dtype, requires_grad=True)
    baseline = len(open_ids)
    out = x.dropout(0.5, generator=generator)
    assert len(open_ids) == baseline + 2        # the output and the mask
    total = out.sum()
    total.backward()
    total.close()
    out.close()
    x.grad.close()
    x.close()
    assert len(open_ids) == baseline - 1        # x itself is closed too

    # 3. retain_graph keeps the mask for a second pass.
    x = _tensor(np.ones(32, dtype=floating), dtype, requires_grad=True)
    out = x.dropout(0.5, generator=generator)
    first = out.sum()
    first.backward(retain_graph=True)
    first.close()
    second = out.sum()
    second.backward()
    second.close()
    out.close()
    x.grad.close()
    x.close()

    # 4. an abandoned graph still frees the mask.
    baseline = len(open_ids)
    x = _tensor(np.ones(32, dtype=floating), dtype, requires_grad=True)
    out = x.dropout(0.5, generator=generator)
    out.close()
    x.close()
    assert len(open_ids) == baseline

    # 5. repeated cycles return to exactly the baseline.
    baseline = len(open_ids)
    for _ in range(5):
        x = _tensor(np.ones(32, dtype=floating), dtype, requires_grad=True)
        out = x.dropout(0.5, generator=generator)
        total = out.sum()
        total.backward()
        total.close()
        out.close()
        x.grad.close()
        x.close()
    assert len(open_ids) == baseline


@needs_native
@pytest.mark.parametrize("dtype", BOTH_DTYPES)
def test_a_corrupted_dropout_mask_dtype_is_refused_by_the_backward(dtype):
    """The graph-level mixed-dtype case: a saved mask that has drifted from
    the graph's dtype must be refused by the backward's multiply rather than
    walked at the wrong width. Reachable only by substituting the private
    resource, which is what makes it worth asserting."""
    from tensorforge.experimental import NativeGenerator

    floating = _DTYPE_BITS[dtype][2]
    other = "float32" if dtype == "float64" else "float64"
    generator = NativeGenerator(5)
    x = _tensor(np.ones(8, dtype=floating), dtype, requires_grad=True)
    stray = cpp.NativeTensorCore._typed_from_array(
        np.ones(8, dtype=_DTYPE_BITS[other][2]), other)
    try:
        out = x.dropout(0.5, generator=generator)
        try:
            # Substitute the mask the backward closure captured.
            closure = out._backward
            assert closure is not None
            saved = out._graph_resources
            assert len(saved) == 1 and saved[0].dtype == dtype
            replacement = _dropout_mask_swap(out, stray)
            total = out.sum()
            try:
                with pytest.raises(ValueError, match="matching dtype") as err:
                    total.backward()
                assert dtype in str(err.value) and other in str(err.value)
            finally:
                total.close()
            _dropout_mask_swap(out, replacement)
        finally:
            out.close()
    finally:
        stray.close()
        x.close()


def _dropout_mask_swap(node, replacement):
    """Swap the private mask a Dropout node's backward closure multiplies
    against, returning the one it replaced. Test-only surgery: the mask is
    private graph state with no public accessor, which is exactly why the
    corrupted case needs it."""
    cells = node._backward.__closure__
    for cell in cells:
        value = cell.cell_contents
        if isinstance(value, cpp.NativeTensorCore) and not value._closed:
            cell.cell_contents = replacement
            return value
    raise AssertionError("no mask found in the dropout backward closure")


# ---------------------------------------------------------------------------
# I7.9 Containers
# ---------------------------------------------------------------------------

@needs_native
def test_a_container_takes_no_dtype_and_raises_at_the_mismatched_child():
    """§12.2: a container does not force a dtype on its children and does
    not unify them — that would be promotion. A model may legitimately hold
    both widths; what it may not do is bridge between them silently.

    The failure is asserted to happen **at the mismatched child**, with both
    dtypes named, and to leave that child's own state untouched."""
    import inspect

    from tensorforge.experimental import (
        NativeLinear, NativeReLU, NativeSequential,
    )

    assert "dtype" not in inspect.signature(NativeSequential).parameters

    wide = NativeLinear(3, 3, seed=1, dtype="float64")
    narrow = NativeLinear(3, 2, seed=2, dtype="float32")
    model = NativeSequential(wide, NativeReLU(), narrow)
    x64 = _tensor(np.ones((2, 3)), "float64", requires_grad=True)
    try:
        versions = {k: p.version for k, p in model.named_parameters()}
        with pytest.raises(ValueError) as error:
            model(x64)
        message = str(error.value)
        assert "NativeLinear" in message
        assert "float32" in message and "float64" in message
        # The failing child mutated nothing, and neither did the earlier
        # ones: no gradient, no version movement.
        for key, parameter in model.named_parameters():
            assert parameter.grad is None, key
            assert parameter.version == versions[key], key
        assert x64.grad is None

        # The mirror case: a float32 input meets the float64 first child, so
        # the very first child raises.
        x32 = _tensor(np.ones((2, 3), dtype=np.float32), "float32")
        try:
            with pytest.raises(ValueError) as first_error:
                model(x32)
            assert "float32" in str(first_error.value)
        finally:
            x32.close()

        # ...and a consistent model of either width runs.
        for dtype in BOTH_DTYPES:
            floating = _DTYPE_BITS[dtype][2]
            a = NativeLinear(3, 3, seed=1, dtype=dtype)
            b = NativeLinear(3, 2, seed=2, dtype=dtype)
            consistent = NativeSequential(a, NativeReLU(), b)
            x = _tensor(np.ones((2, 3), dtype=floating), dtype)
            try:
                consistent.train()
                assert all(child.training for child in (a, b))
                consistent.eval()
                assert not any(child.training for child in (a, b))
                out = consistent(x)
                try:
                    assert out.dtype == dtype
                finally:
                    out.close()
            finally:
                x.close()
                _release(a)
                _release(b)
    finally:
        x64.close()
        _release(narrow)
        _release(wide)


@needs_native
def test_train_eval_and_generator_registration_are_unchanged_by_dtype():
    """A dtype-aware model still propagates ``train()``/``eval()`` through
    every child and still registers its generators as the fourth state
    category — absent from ``state_dict()``, present in
    ``named_generators()``, with the shared object's identity preserved."""
    from tensorforge.experimental import (
        NativeBatchNorm1d, NativeDropout, NativeGenerator, NativeLinear,
        NativeSequential,
    )

    generator = NativeGenerator(3)
    linear = NativeLinear(4, 4, seed=1, dtype="float32")
    norm = NativeBatchNorm1d(4, dtype="float32")
    drop = NativeDropout(0.25, generator=generator)
    model = NativeSequential(linear, norm, drop)
    x = _tensor(np.arange(16.0, dtype=np.float32).reshape(4, 4), "float32")
    try:
        assert dict(model.named_generators()) == {"2.generator": generator}
        assert model.generators() == [generator]
        state = model.state_dict()
        try:
            assert all(not key.endswith("generator") for key in state)
            assert set(value.dtype for value in state.values()) == {"float32"}
        finally:
            for value in state.values():
                value.close()

        model.train()
        assert linear.training and norm.training and drop.training
        out = model(x)
        try:
            assert out.dtype == "float32"
        finally:
            out.close()
        assert generator.calls == 1

        model.eval()
        assert not (linear.training or norm.training or drop.training)
        out = model(x)
        try:
            assert out.dtype == "float32"
        finally:
            out.close()
        assert generator.calls == 1     # eval consumes none, at float32 too
    finally:
        x.close()
        _release(norm)
        _release(linear)


# ---------------------------------------------------------------------------
# I7.10 float64 regression, bitwise
# ---------------------------------------------------------------------------

@needs_native
def test_the_float64_module_stack_is_bitwise_what_it_was():
    """The phase's hardest requirement (design §26): every pre-Phase-I
    float64 value is **byte-identical**.

    The whole pre-I7 surface is exercised in one place with committed bit
    patterns rather than left to the older suites alone, so a regression
    shows up as this test rather than as a puzzling failure three files
    away. Each expectation below is a value produced by the *shipped* path
    with no dtype argument anywhere."""
    from tensorforge.experimental import (
        NativeBatchNorm1d, NativeDropout, NativeGenerator, NativeLayerNorm,
        NativeLinear,
    )

    # Initialization: the host draw, untouched.
    linear = NativeLinear(4, 3, seed=42)
    try:
        import math
        bound = 1.0 / math.sqrt(4)
        rng = np.random.default_rng(42)
        assert _same_bits(linear.weight.to_numpy(),
                          rng.uniform(-bound, bound, size=(4, 3)), "float64")
        assert _same_bits(linear.bias.to_numpy(),
                          rng.uniform(-bound, bound, size=(3,)), "float64")
    finally:
        _release(linear)

    # LayerNorm forward: exact zeros and ones on a symmetric input, which
    # any change to the eps placement or the operand order would move.
    norm = NativeLayerNorm(4, elementwise_affine=False)
    x = _tensor(np.array([[1.0, 2.0, 3.0, 4.0]]), "float64")
    try:
        out = norm(x)
        try:
            values = out.to_numpy()
            assert values.dtype == np.float64
            assert _same_bits(values, -values[:, ::-1], "float64")
        finally:
            out.close()
    finally:
        x.close()

    # BatchNorm buffers: the documented defaults, exactly.
    bn = NativeBatchNorm1d(3)
    try:
        assert _same_bits(bn.running_mean.to_numpy(), np.zeros(3), "float64")
        assert _same_bits(bn.running_var.to_numpy(), np.ones(3), "float64")
        bn.train()
        data = _tensor(np.array([[0.0, 1.0, 2.0],
                                 [2.0, 3.0, 4.0]]), "float64")
        try:
            bn(data).close()
        finally:
            data.close()
        # (1 - 0.1) * 0 + 0.1 * batch_mean, and the population variance.
        assert _same_bits(bn.running_mean.to_numpy(),
                          np.array([0.1, 0.2, 0.30000000000000004]),
                          "float64")
        assert _same_bits(bn.running_var.to_numpy(),
                          np.array([1.0, 1.0, 1.0]), "float64")
    finally:
        _release(bn)

    # Dropout: the G2 committed vector, through the shipped module.
    generator = NativeGenerator(0)
    drop = NativeDropout(0.25, generator=generator)
    ones = _tensor(np.ones(12), "float64")
    try:
        drop.train()
        out = drop(ones)
        try:
            expected = np.where(
                np.array(list("111110111110")) == "1", 1.0 / 0.75, 0.0)
            assert _same_bits(out.to_numpy(), expected, "float64")
        finally:
            out.close()
        assert generator.calls == 1
    finally:
        ones.close()


@needs_native
def test_i7_moved_no_public_capability_at_all():
    """The exit gate, as one assertion block.

    What I7 delivered: dtype-aware state-owning constructors, float32
    parameters, buffers, normalization, and Dropout, and the last
    float64-only Core gate opened. What it deliberately did **not** deliver,
    and what the next milestones own:

      * **I8** — float32 optimizer state and checkpoint version 3, which
        landed there and not here;
      * **I9** — the public registry, which moved there and not here.

    The gap between implementation and promise was the phase's rollout
    discipline (§27), not an oversight, and stating it precisely is what
    kept "float32 is supported" from becoming the summary five milestones
    early. Both later milestones have since landed, so what this gate can
    still assert is that **I7 owns neither of them**: the constructor
    surface is exactly its six, and the registry and checkpoint constants
    are the ones I9 and I8 respectively are on record as setting."""
    from tensorforge.experimental import native_checkpoint

    _assert_the_public_registry_is_i9s()
    assert cpp.RAW_KERNEL_DTYPES == ("float64",)
    assert cpp.backend_info()["supported_dtypes"] == I9_DTYPES
    assert cpp.backend_info()["stable_framework_integration"] is False
    assert native_checkpoint._FORMAT_VERSION == I8_CHECKPOINT_VERSION
    assert (native_checkpoint._SUPPORTED_FORMAT_VERSIONS
            == I8_CHECKPOINT_VERSIONS)

    exports = _source_exports()
    assert len(exports) == I1_EXPORT_COUNT       # still 54; I7 adds none
    for absent in ("tf_core_dropout_forward_f32", "tf_core_dropout_typed",
                   "tf_core_dropout_backward", "tf_core_layer_norm",
                   "tf_core_batch_norm", "tf_storage_cast",
                   "tf_storage_dtype", "tf_parameter_create"):
        assert absent not in exports, absent
    assert not [name for name in exports
                if name.endswith(("_f32", "_f64", "_float32", "_float64"))]
    assert _constructors_with_a_dtype_argument() == DTYPE_CONSTRUCTORS


@needs_native
def test_the_dropout_export_kept_its_exact_abi_shape():
    """§12 of the milestone: the same symbol, the same argument count, the
    same order, the same types. The dtype travels on the handles, so nothing
    about the signature had to move — and a previously compiled caller
    therefore still links and still runs."""
    import ctypes

    library = cpp._require_library()
    signature = library.tf_core_dropout_forward.argtypes
    assert signature == [
        ctypes.c_void_p, ctypes.c_int64,     # input handle, input offset
        ctypes.c_void_p,                     # output handle
        ctypes.c_void_p,                     # mask handle
        ctypes.c_int64,                      # element count
        ctypes.c_uint64, ctypes.c_uint64,    # seed, call index
        ctypes.c_double,                     # p
    ]
    assert library.tf_core_dropout_forward.restype is None
    # The kernel is a template now, but the export is not: exactly one
    # dispatch, and no per-dtype symbol anywhere.
    source = _read("cpp/src/random.cpp")
    assert source.count("TF_EXPORT") == 1
    assert source.count("tf::dispatch_dtype(") == 1
    assert "tf::require_matching_dtype(" in source
    assert "tf::require_float64(" not in source
    # The random derivation itself is untouched: the uniform is still the
    # binary64 53-bit conversion, at every width (design §14.2).
    assert "static_cast<double>(bits >> 11) * 0x1p-53" in source
    header = _read("cpp/include/tf_random_internal.h")
    assert "static_cast<T>(1.0 / (1.0 - p))" in header
    assert "double dropout_uniform(std::uint64_t bits) noexcept;" in header
    # No dtype branch inside the element loop: the only ``switch`` in the
    # translation unit is the one dispatch above.
    assert source.count("switch (") == 1
