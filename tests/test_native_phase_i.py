"""Phase-I contract guardrails (native dtype generalization).

Milestones I0 through I4 are complete; I5 through I11 are not started.

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
unavailable to it.

None of the four moved a **public** capability: float32 is allocatable,
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
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from tensorforge.backends import cpp

REPO_ROOT = Path(__file__).resolve().parent.parent
PHASE_I_DESIGN = REPO_ROOT / "docs" / "native_dtype_float32_design.md"

# The boundary Phase I inherited. The **public** half of it does not move
# until milestone I9, so these stay exactly as Phase H left them right
# through I1-I8 even as internal float32 capability appears beneath them.
I0_DTYPES = ("float64",)
I0_DEVICES = ("cpu",)
I0_UNSUPPORTED = ("float32", "cuda", "amp")
I0_EXPORT_COUNT = 52
I0_CHECKPOINT_VERSION = 2
I0_CHECKPOINT_VERSIONS = (1, 2)

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
    text = _flat(_design())
    # The status line names which milestones are done and which are not.
    # Asserted as structure rather than as one phrasing, so the sentence
    # can be rewritten each milestone without rewriting this test — but it
    # must always say *both* halves.
    status = re.search(r"Phase-I status:(.{0,200})", text, re.I)
    assert status, "the design does not state its milestone status"
    claim = status.group(1)
    assert re.search(r"\bI0\b.*\bI1\b.*\bI2\b.*complete", claim, re.I), (
        f"the status line does not record I0, I1 and I2 as complete: {claim!r}"
    )
    assert re.search(r"I3\b.*\bI11\b.*not started", claim, re.I), (
        f"the status line does not record I3-I11 as unstarted: {claim!r}"
    )
    # ...and that I0 itself shipped no behavior, which is a historical
    # fact about I0 and stays true however far the phase progresses. A
    # phase that has begun but claims delivery is the exact drift here.
    assert re.search(r"I0 adds no runtime behavior", text, re.I), (
        "the design does not state that I0 adds no runtime behavior"
    )
    # The public boundary has not moved, and the design must keep saying
    # so until I9 — the milestone that is allowed to change it.
    assert re.search(r"SUPPORTED_DTYPES still reads \(\"float64\",\)", text), (
        "the design no longer records the unchanged public dtype registry"
    )


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
    assert cpp.SUPPORTED_DTYPES == I0_DTYPES
    assert cpp.SUPPORTED_DEVICES == I0_DEVICES
    assert cpp.UNSUPPORTED == I0_UNSUPPORTED


def test_float32_is_still_genuinely_unreachable():
    """The registry claim, checked against behavior rather than trusted —
    a contract document must not have quietly enabled anything."""
    import numpy as np

    with pytest.raises((ValueError, TypeError)):
        cpp.NativeTensorCore.from_array(
            np.zeros((2, 2), dtype=np.float64), dtype="float32")
    with pytest.raises((ValueError, TypeError)):
        cpp.NativeTensorCore.zeros((2, 2), dtype="float32")
    with pytest.raises(ValueError):
        cpp.normalize_dtype("float32")
    assert cpp.normalize_dtype(None) == "float64"


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


def test_the_checkpoint_constants_did_not_move():
    from tensorforge.experimental import native_checkpoint

    assert native_checkpoint._FORMAT_VERSION == I0_CHECKPOINT_VERSION
    assert (native_checkpoint._SUPPORTED_FORMAT_VERSIONS
            == I0_CHECKPOINT_VERSIONS)
    assert native_checkpoint._FORMAT == "tensorforge.native_checkpoint"


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


def test_i1_changed_no_example_benchmark_ci_or_dependency_file():
    """The milestone's discipline, expressed as a diff assertion against
    the merged I0 commit rather than as a promise.

    I1 legitimately changes C++ sources, the CMake build, and the ctypes
    layer — that is the milestone. What it must *not* touch is the
    surface that would signal a capability change: an example, a
    benchmark, the CI workflow, or the dependency set. Phase I adds no
    dependency and no build option, and benchmark work belongs to I10.
    """
    forbidden = []
    for path in _changed_since(I0_COMMIT):
        if path.startswith(("examples/", "benchmarks/", ".github/")):
            forbidden.append(path)
        if path in ("pyproject.toml", "uv.lock", "conftest.py"):
            forbidden.append(path)
    assert not forbidden, (
        f"I1 must not touch examples, benchmarks, CI, or dependencies, "
        f"but these changed: {forbidden}"
    )


def test_the_phase_touched_only_the_two_python_modules_its_scope_names():
    """Within ``src/``, the phase so far is confined to the module that
    owns the C ABI and the one that owns the native autograd graph.

    Through I3 this was a single file: the dtype foundation, the transfer
    boundaries, and the elementwise execution all live in the ctypes layer,
    and nothing above it participated. **I4 adds exactly one more**, and
    the milestone's scope is what names it — core autograd is I4 work, so
    the gradient dtype invariants of design §11 (a backward's constants
    built at the graph's dtype, the seed at the output's dtype, the
    broadcast-back operand at the upstream's) are edits to
    ``experimental/native_tensor.py`` and can be nowhere else.

    Everything else is still out: no parameter, module, optimizer,
    checkpoint, generator, loss, or stable-line file participates, and the
    stable framework is not coupled to the native line by any of it.
    """
    allowed = {
        "src/tensorforge/backends/cpp.py",
        "src/tensorforge/experimental/native_tensor.py",
    }
    changed = [path for path in _changed_since(I0_COMMIT)
               if path.startswith("src/")]
    unexpected = [path for path in changed if path not in allowed]
    assert unexpected == [], (
        f"the phase changed more of the Python package than its scope "
        f"names: {unexpected}"
    )
    # Stated the other way round as well, so a future milestone that adds a
    # file cannot satisfy the rule above by accident: the families I5-I8 own
    # are untouched, by name.
    for forbidden in ("native_parameter", "native_module", "native_linear",
                      "native_conv2d", "native_maxpool2d", "native_dropout",
                      "native_batchnorm", "native_layernorm", "native_sgd",
                      "native_adam", "native_checkpoint", "native_generator",
                      "native_cross_entropy_loss", "native_mse_loss"):
        assert not any(forbidden in path for path in changed), forbidden


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


def test_the_dtype_tables_are_private_and_wider_than_the_public_promise():
    """The internal capability legitimately runs ahead of the public one
    between I1 and I9 — that is the rollout rule — but the public surface
    must not leak it."""
    assert "float32" in cpp._DTYPE_CODES
    assert "float32" not in cpp.SUPPORTED_DTYPES
    assert "float32" in cpp.UNSUPPORTED
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
def test_the_public_wrapper_allocates_only_float64_through_the_typed_path():
    """``NativeStorage`` routes through the typed creators uniformly, and
    the dtype it passes is the one it validated — so the Python tag and
    the C++ tag cannot disagree. Every observable behavior is unchanged."""
    storage = cpp.NativeStorage(8)
    try:
        assert storage.dtype == "float64"
        assert storage.size == 8
        # The zero-initializing default did not move.
        assert np.array_equal(storage.to_numpy(), np.zeros(8))
        assert storage.to_numpy().dtype == np.float64
    finally:
        storage.close()
    # ...and float32 is still refused at the public boundary, before any
    # native call is made.
    with pytest.raises(ValueError):
        cpp.NativeStorage(8, dtype="float32")


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


def test_no_public_float32_construction_path_exists_anywhere():
    """The rollout rule, checked behaviorally across every public
    constructor rather than trusted from the registry."""
    values = np.zeros((2, 2), dtype=np.float64)
    for build in (
        lambda: cpp.NativeStorage(4, dtype="float32"),
        lambda: cpp.NativeStorage.from_array(values, dtype="float32"),
        lambda: cpp.NativeTensorCore.zeros((2, 2), dtype="float32"),
        lambda: cpp.NativeTensorCore.from_array(values, dtype="float32"),
        lambda: cpp.normalize_dtype("float32"),
    ):
        with pytest.raises((ValueError, TypeError)):
            build()


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
    import inspect

    import tensorforge.experimental as experimental
    from tensorforge.experimental import native_checkpoint

    assert cpp.SUPPORTED_DTYPES == I0_DTYPES
    assert cpp.SUPPORTED_DEVICES == I0_DEVICES
    assert cpp.UNSUPPORTED == I0_UNSUPPORTED
    assert cpp.backend_info()["dtype"] == "float64"
    assert native_checkpoint._FORMAT_VERSION == I0_CHECKPOINT_VERSION
    assert (native_checkpoint._SUPPORTED_FORMAT_VERSIONS
            == I0_CHECKPOINT_VERSIONS)
    exports = _source_exports()
    assert len(exports) == I1_EXPORT_COUNT       # still 54; I2 adds none
    for absent in ("tf_storage_copy_from_typed", "tf_storage_copy_to_typed",
                   "tf_storage_materialize_typed",
                   "tf_core_contiguous_copy_f32", "tf_storage_dtype",
                   "tf_storage_cast"):
        assert absent not in exports, absent
    # No module, parameter, optimizer, or loss constructor gained a dtype
    # argument — that is milestone I7, not this one.
    for name in experimental.__all__:
        obj = getattr(experimental, name)
        if not inspect.isclass(obj):
            continue
        try:
            signature = inspect.signature(obj)
        except (TypeError, ValueError):  # pragma: no cover
            continue
        assert "dtype" not in signature.parameters, name


@needs_native
def test_the_float32_paths_are_exactly_the_families_landed_through_i4():
    """The precise statement that replaces I3's "float32 is computed on by
    exactly the elementwise and unary family".

    As of I4, float32 is computed on by **transfer/copy, the elementwise and
    unary family, reductions, matmul, view-backward, and the two scalar
    storage primitives** — and by nothing else. Every later numerical family
    still rejects it, and rejects it as *float64-only*, with both operands at
    the same dtype, so this is not a mixed-dtype rejection in disguise.

    The families still out are exactly the ones I5-I8 own: conv2d (all three
    directions), maxpool (both), softmax, log-softmax, cross-entropy, and
    Dropout. Above the Core layer nothing moved at all — no parameter, no
    module, no optimizer, no checkpoint version, and no public constructor
    that produces a float32 tensor.

    This is the guardrail that keeps "some float32 works" from ever being
    the honest summary: the set is enumerated in both directions.
    """
    library = cpp._require_library()
    core = cpp.NativeTensorCore._typed_from_array(
        _patterned("float32", 16).reshape(4, 4), "float32")
    other = cpp.NativeTensorCore._typed_from_array(
        np.ones((4, 4), dtype=np.float32), "float32")
    output = cpp.NativeStorage._typed(16, "float32")
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

        # -- not consumed: every later numerical family, still float64-only.
        for call in (
            lambda: library.tf_core_softmax_forward(a, 0, out, 4, 4, 1),
            lambda: library.tf_core_log_softmax_forward(a, 0, out, 4, 4, 1),
            lambda: library.tf_core_dropout_forward(a, 0, out, out, 16,
                                                    1, 0, 0.5),
            # N, C, H, W, kh, kw, sh, sw, ph, pw, out_h, out_w
            lambda: library.tf_core_maxpool2d_forward(
                a, 0, out, out, 1, 1, 4, 4, 2, 2, 2, 2, 0, 0, 2, 2),
            # N, C, H, W, O, kh, kw, sh, sw, ph, pw, out_h, out_w
            lambda: library.tf_core_conv2d_forward(
                a, 0, b, 0, None, 0, out, 1, 1, 4, 4, 1, 2, 2, 1, 1, 0, 0,
                3, 3),
        ):
            with pytest.raises(ValueError, match="float64-only"):
                call()
        library.tf_clear_error()

        # Nothing above the Core layer moved: no public constructor makes a
        # float32 tensor, and no parameter does either.
        from tensorforge.experimental import NativeParameter, NativeTensor
        for construct in (
            lambda: NativeTensor.from_array([1.0], dtype="float32"),
            lambda: NativeTensor.zeros((2,), dtype="float32"),
            lambda: NativeTensor.full((2,), 1.0, dtype="float32"),
            lambda: cpp.NativeTensorCore.from_array([1.0], dtype="float32"),
            lambda: cpp.NativeTensorCore.zeros((2,), dtype="float32"),
            lambda: cpp.NativeTensorCore.full((2,), 1.0, dtype="float32"),
            lambda: NativeParameter([1.0], dtype="float32"),
        ):
            with pytest.raises((ValueError, TypeError)):
                construct()
    finally:
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


def _assert_transcendental(got, want, limit, label):
    """NaN by position, infinities and zeros by exact bits, everything else
    within ``limit`` steps. The tolerance deliberately never covers a zero:
    a distance cannot see a zero's sign, and that is precisely the kind of
    thing this suite exists to catch."""
    assert got.dtype == np.float32 and want.dtype == np.float32, label
    assert np.array_equal(np.isnan(got), np.isnan(want)), f"{label}: NaN places"
    special = np.isnan(got) | np.isinf(got) | np.isinf(want) | (got == 0) \
        | (want == 0)
    assert _same_bits(got[special], want[special], "float32"), (
        f"{label}: a special value is not exactly reproduced")
    ordinary = ~special
    if not ordinary.any():
        return 0
    worst = int(_f32_ulp_distance(got[ordinary], want[ordinary]).max())
    assert worst <= limit, f"{label}: {worst} ULP apart, over the {limit} bound"
    return worst


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
def test_float64_transcendentals_did_not_move():
    """The float64 half of exp/log is unchanged by the generalization: it is
    still bit-identical to NumPy's float64, which is what it was before I3.
    """
    values = _sample("float64", 64, seed=9)
    positive = np.abs(values) + 0.5
    for name, source, oracle in (("exp", values, np.exp),
                                 ("log", positive, np.log)):
        core = _core(source, "float64")
        try:
            out = core.exp() if name == "exp" else core.log()
            try:
                assert _same_bits(out.to_numpy(), oracle(source), "float64")
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
def test_i3_did_not_open_float32_autograd_modules_or_parameters():
    """The boundary I3 must not cross. A dtype-general Core primitive is a
    kernel; float32 autograd, parameters, modules, and optimizers are
    milestones I4 and I7, and none of them may be reachable yet."""
    import inspect

    import tensorforge.experimental as experimental

    # No public constructor produces a float32 tensor, so no float32 graph
    # can be built at all.
    values = np.zeros((2, 2), dtype=np.float32)
    for build in (
        lambda: experimental.NativeTensor.from_array(values, dtype="float32"),
        lambda: experimental.NativeTensor.zeros((2, 2), dtype="float32"),
        lambda: experimental.NativeTensor.full((2, 2), 1.0, dtype="float32"),
        lambda: experimental.NativeParameter(values, dtype="float32"),
    ):
        with pytest.raises((ValueError, TypeError)):
            build()
    # ...and no module, parameter, optimizer, or loss constructor gained a
    # dtype argument — that is milestone I7, not this one.
    for name in experimental.__all__:
        obj = getattr(experimental, name)
        if not inspect.isclass(obj):
            continue
        try:
            signature = inspect.signature(obj)
        except (TypeError, ValueError):  # pragma: no cover
            continue
        assert "dtype" not in signature.parameters, name


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

    assert cpp.SUPPORTED_DTYPES == I0_DTYPES
    assert cpp.SUPPORTED_DEVICES == I0_DEVICES
    assert cpp.UNSUPPORTED == I0_UNSUPPORTED
    assert cpp.RAW_KERNEL_DTYPES == ("float64",)
    assert cpp.backend_info()["dtype"] == "float64"
    assert native_checkpoint._FORMAT_VERSION == I0_CHECKPOINT_VERSION
    assert (native_checkpoint._SUPPORTED_FORMAT_VERSIONS
            == I0_CHECKPOINT_VERSIONS)
    with pytest.raises(ValueError):
        cpp.normalize_dtype("float32")
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
    assert code.count("cpp.NativeTensorCore.zeros(") == 2    # ctor + broadcast
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
def test_the_families_i5_through_i8_own_still_reject_float32():
    """The other direction of the I4 boundary: no module, parameter,
    optimizer, or checkpoint path accepts float32, and the later numerical
    families still refuse a float32 operand as float64-only."""
    import inspect

    import tensorforge.experimental as experimental
    from tensorforge.experimental import native_checkpoint

    for name in experimental.__all__:
        obj = getattr(experimental, name)
        if not inspect.isclass(obj):
            continue
        try:
            signature = inspect.signature(obj)
        except (TypeError, ValueError):  # pragma: no cover
            continue
        assert "dtype" not in signature.parameters, name
    assert native_checkpoint._FORMAT_VERSION == I0_CHECKPOINT_VERSION
    assert (native_checkpoint._SUPPORTED_FORMAT_VERSIONS
            == I0_CHECKPOINT_VERSIONS)
    # The Core operations I5 and I6 own still refuse a float32 operand. Some
    # refuse in Python ("requires a float64/cpu input") and some in C++
    # ("float64-only in the current runtime"); both name float64, and which
    # layer answers is not the point — that the operation is unavailable is.
    core = cpp.NativeTensorCore._typed_from_array(
        np.ones((1, 1, 4, 4), dtype=np.float32), "float32")
    try:
        with pytest.raises(ValueError, match="float64"):
            core.maxpool2d_forward(kernel_size=2)
        flat = core.reshape((4, 4))
        try:
            with pytest.raises(ValueError, match="float64"):
                flat.softmax(axis=-1)
            with pytest.raises(ValueError, match="float64"):
                flat.log_softmax(axis=-1)
            with pytest.raises(ValueError, match="float64"):
                flat.cross_entropy_forward([0, 1, 2, 3])
            with pytest.raises(ValueError, match="float64"):
                flat.dropout_forward(0.5, seed=1, call_index=0)
        finally:
            flat.close()
    finally:
        core.close()


@needs_native
def test_i4_moved_no_public_capability_at_all():
    """The exit gate, as one assertion block: internal float32 reduction,
    matmul, view-backward, and Core autograd exist, and public float32
    support does not."""
    from tensorforge.experimental import native_checkpoint

    assert cpp.SUPPORTED_DTYPES == I0_DTYPES
    assert cpp.SUPPORTED_DEVICES == I0_DEVICES
    assert cpp.UNSUPPORTED == I0_UNSUPPORTED
    assert cpp.RAW_KERNEL_DTYPES == ("float64",)
    assert cpp.backend_info()["dtype"] == "float64"
    assert native_checkpoint._FORMAT_VERSION == I0_CHECKPOINT_VERSION
    assert (native_checkpoint._SUPPORTED_FORMAT_VERSIONS
            == I0_CHECKPOINT_VERSIONS)
    with pytest.raises(ValueError):
        cpp.normalize_dtype("float32")
    exports = _source_exports()
    assert len(exports) == I1_EXPORT_COUNT       # still 54; I4 adds none
    for absent in ("tf_core_sum_f32", "tf_core_matmul_f32",
                   "tf_core_narrow_backward_f32", "tf_storage_fill_f32",
                   "tf_storage_scale_typed", "tf_core_sum_typed",
                   "tf_storage_dtype", "tf_storage_cast"):
        assert absent not in exports, absent
    assert not [name for name in exports
                if name.endswith(("_f32", "_f64", "_float32", "_float64"))]
