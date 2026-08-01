"""Phase-I contract guardrails (native dtype generalization).

Milestones I0, I1, I2, and I3 are complete; I4 through I11 are not started.

I0 was a design-and-reconciliation milestone: it shipped
``docs/native_dtype_float32_design.md``, this module, and documentation,
and **no runtime behavior at all**. I1 built the foundation the rest of the
phase stands on — the C++ dtype model, dtype-tagged storage, and the two
typed creation exports that take the library from 52 symbols to 54. I2 made
float32 *movable*: the three transfer exports and the identity copy became
dtype-general and bit-preserving. I3 made it *computed on*, by the
elementwise and unary Core family and by nothing else.

None of the three moved a **public** capability: float32 is allocatable,
transferable, and now arithmetically usable through the C ABI and the
private typed constructors, and it is still not a supported TensorForge
dtype.

Three kinds of fact therefore live here, and keeping them apart is the
point of the module:

* **What the contract says** — a property of the design document, which
  spans the whole phase and does not move as milestones land.
* **What the repository is now** — the live registries, the live source,
  and the built library, at I3.
* **What is still a promise** — everything I4 onward will do, asserted as
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


def test_i1_touched_only_the_backend_module_in_the_python_package():
    """Within ``src/``, the dtype foundation is confined to the one module
    that owns the C ABI. No tensor, autograd, module, optimizer,
    checkpoint, or stable-line file participates in I1, and the stable
    framework is not coupled to the native line by it."""
    changed = [path for path in _changed_since(I0_COMMIT)
               if path.startswith("src/")]
    assert changed == ["src/tensorforge/backends/cpp.py"] or not changed, (
        f"I1 changed more of the Python package than the ctypes layer: "
        f"{changed}"
    )


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
def test_the_float32_paths_are_exactly_the_i3_elementwise_and_unary_family():
    """The precise statement that replaces I2's "float32 is computed on by
    nothing".

    It is now computed on by **exactly** the I3 elementwise and unary Core
    family — add, subtract, multiply, ReLU and its backward, sqrt,
    reciprocal, exp, log — beside the transfer, materialization, and
    identity-copy infrastructure I2 opened. Every later numerical family
    still rejects it, and rejects it as *float64-only*, with both operands
    at the same dtype, so this is not a mixed-dtype rejection in disguise.

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
        ):
            library.tf_clear_error()
            call()
            assert library.tf_last_error_code() == 0

        # -- not consumed: every later numerical family, still float64-only.
        for call in (
            lambda: library.tf_core_matmul(a, b, out, 4, 4, 4, 4, 1, 4, 1,
                                           0, 0),
            lambda: library.tf_core_sum(a, out, shape, strides,
                                        (ctypes.c_int64 * 2)(0, 1), 0, 2),
            lambda: library.tf_core_narrow_backward(a, out, shape, strides,
                                                    strides, 0, 0, 2),
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
        # ...and the two float64-only storage primitives are deliberately
        # among them: they assign and multiply rather than transfer, so
        # broadening them is a later milestone's decision. They are
        # unhooked (H7), so their rejection is read from the error slot.
        library.tf_clear_error()
        library.tf_storage_fill(a, 1.0)
        assert library.tf_last_error_code() != 0
        library.tf_clear_error()
        library.tf_storage_scale(a, 2.0)
        assert library.tf_last_error_code() != 0
        library.tf_clear_error()
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
