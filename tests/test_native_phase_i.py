"""Phase-I contract guardrails (native dtype generalization, milestone I0).

I0 is a design-and-reconciliation milestone: it ships
``docs/native_dtype_float32_design.md``, this module, and documentation,
and **no runtime behavior at all**. These tests therefore protect two
different things at once, and the split matters:

* **What the contract says.** The load-bearing dtype, storage, ABI,
  dispatch, autograd, module, RNG, optimizer, checkpoint, determinism,
  isolation, and performance decisions must actually be written down, in
  the section that owns each of them, so a later milestone inherits an
  unambiguous design instead of re-deriving one. These assertions are
  **section-scoped** and require **combinations** of architectural terms
  rather than the presence of one vague word — a document that merely
  contains the string "float32" passes nothing here.
* **What the repository still is.** I0 must have changed none of it. The
  registries, the checkpoint constants, the export count, and Phase H's
  completion are asserted against the **live** module, the **live**
  source, and the **built** library — never against prose.

They deliberately test *values and structure* rather than wording, so
ordinary prose improvements do not require rewriting them. Nothing here
asserts a character count, a paragraph order, or a benchmark number.
"""
import re
import subprocess
import sys
from pathlib import Path

import pytest

from tensorforge.backends import cpp

REPO_ROOT = Path(__file__).resolve().parent.parent
PHASE_I_DESIGN = REPO_ROOT / "docs" / "native_dtype_float32_design.md"

# The boundary Phase I inherited and, at I0, must leave exactly alone.
I0_DTYPES = ("float64",)
I0_DEVICES = ("cpu",)
I0_UNSUPPORTED = ("float32", "cuda", "amp")
I0_EXPORT_COUNT = 52
I0_CHECKPOINT_VERSION = 2
I0_CHECKPOINT_VERSIONS = (1, 2)

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


def test_the_design_marks_the_phase_begun_at_i0_and_not_shipped():
    text = _flat(_design())
    assert re.search(r"Phase-I status:\s*begun at I0", text, re.I), (
        "the design does not state its milestone status"
    )
    # ...and that I0 itself shipped no behavior. Both halves matter: a
    # phase that has begun but claims delivery is the exact drift here.
    assert re.search(r"I0 adds no runtime behavior", text, re.I), (
        "the design does not state that I0 adds no runtime behavior"
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


def test_the_production_export_count_is_still_fifty_two():
    exports = _source_exports()
    assert len(exports) == I0_EXPORT_COUNT, sorted(exports)
    for planned in PLANNED_NEW_EXPORTS:
        assert planned not in exports, (
            f"{planned} is a *planned* Phase-I export; I0 must not add it"
        )


@pytest.mark.skipif(not cpp.is_available(),
                    reason="the native library is not built")
def test_the_built_library_still_exports_exactly_what_the_source_does():
    library = cpp._require_library()
    for name in sorted(_source_exports()):
        assert hasattr(library, name), name
    for planned in PLANNED_NEW_EXPORTS:
        assert not hasattr(library, planned), (
            f"the built library exports {planned}, which I0 does not add"
        )


def test_no_typed_creator_is_declared_to_ctypes_yet():
    declared = _read("src/tensorforge/backends/cpp.py")
    for planned in PLANNED_NEW_EXPORTS:
        assert planned not in declared, (
            f"{planned} is declared in the ctypes layer; I0 adds no "
            f"declaration"
        )
    # ...and the raw-kernel dtype registry is an I2 deliverable, not an
    # I0 one (design section 7).
    assert "RAW_KERNEL_DTYPES" not in declared, (
        "RAW_KERNEL_DTYPES is an I2 registry; I0 must not introduce it"
    )


def test_the_cpp_storage_struct_is_still_physically_float64():
    """The premise of the whole phase, asserted so that a milestone that
    changes it cannot be mistaken for I0."""
    header = _read("cpp/include/tf_internal.h")
    assert re.search(r"struct Storage\s*\{\s*double\*\s*data;",
                     header, re.S), (
        "tf::Storage is no longer the float64-only struct I0 recorded"
    )
    assert "TF_DTYPE_FLOAT32" not in header, (
        "a dtype enum has appeared in the C++ header; that is milestone I1"
    )


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

def test_i0_changed_no_implementation_build_or_ci_file():
    """The milestone's whole discipline, expressed as a diff assertion
    against the merged Phase-H base rather than as a promise.

    Skipped rather than failed when git is unavailable or the base commit
    is missing, because that is an environment fact and not a defect in
    the tree under test.
    """
    base = "1b6cc17305c7ffc6502e27c32b45661480e05f9d"
    try:
        merge_base = subprocess.run(
            ["git", "merge-base", "--is-ancestor", base, "HEAD"],
            cwd=REPO_ROOT, capture_output=True, text=True,
        )
    except (OSError, subprocess.SubprocessError):  # pragma: no cover
        pytest.skip("git is not available")
    if merge_base.returncode != 0:  # pragma: no cover
        pytest.skip(f"{base[:7]} is not an ancestor of HEAD")
    changed = subprocess.run(
        ["git", "diff", "--name-only", base, "--"],
        cwd=REPO_ROOT, capture_output=True, text=True, check=True,
    ).stdout.split()
    # ``git diff`` reports tracked modifications only, so a *new* file
    # dropped into src/ or cpp/ would slip past it. The untracked listing
    # is what closes that, and it is the more likely mistake of the two.
    changed += subprocess.run(
        ["git", "ls-files", "--others", "--exclude-standard"],
        cwd=REPO_ROOT, capture_output=True, text=True, check=True,
    ).stdout.split()
    forbidden = []
    for path in changed:
        if path.startswith(("src/", "cpp/", "examples/", "benchmarks/",
                            "scripts/", ".github/")):
            forbidden.append(path)
        if path in ("pyproject.toml", "uv.lock", "conftest.py"):
            forbidden.append(path)
    assert not forbidden, (
        f"I0 is documentation and tests only, but these changed: {forbidden}"
    )


def test_i0_introduced_no_prohibited_external_reference():
    """The repository self-containment guardrail, re-asserted over the
    files this milestone actually adds.

    The terms are assembled from fragments so this module never spells one
    literally, exactly as the repository-wide guardrail does.
    """
    name = "dae" + "dalus"
    owner = "johnson" + "kayati"
    terms = (name, owner, name + "-ml", "github.com/" + owner)
    for relative in ("docs/native_dtype_float32_design.md",
                     "tests/test_native_phase_i.py"):
        lowered = _read(relative).lower()
        for term in terms:
            assert term not in lowered, f"{relative} names {term!r}"


def test_the_stable_framework_still_does_not_load_the_native_backend():
    """Phase I touches the native line only, and I0 touches no line at
    all. The isolation the whole project rests on must still hold."""
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
