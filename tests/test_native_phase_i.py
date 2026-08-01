"""Phase-I contract guardrails (native dtype generalization).

Milestones I0 and I1 are complete; I2 through I11 are not started.

I0 was a design-and-reconciliation milestone: it shipped
``docs/native_dtype_float32_design.md``, this module, and documentation,
and **no runtime behavior at all**. I1 built the foundation the rest of the
phase stands on — the C++ dtype model, dtype-tagged storage, and the two
typed creation exports that take the library from 52 symbols to 54 — while
changing **no** public capability: float32 is allocatable through the C ABI
and is still not a supported TensorForge dtype.

Three kinds of fact therefore live here, and keeping them apart is the
point of the module:

* **What the contract says** — a property of the design document, which
  spans the whole phase and does not move as milestones land.
* **What the repository is now** — the live registries, the live source,
  and the built library, at I1.
* **What is still a promise** — everything I2 onward will do, asserted as
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
    assert re.search(r"\bI0\b.*\bI1\b.*complete", claim, re.I), (
        f"the status line does not record I0 and I1 as complete: {claim!r}"
    )
    assert re.search(r"I2\b.*\bI11\b.*not started", claim, re.I), (
        f"the status line does not record I2-I11 as unstarted: {claim!r}"
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


def test_the_typed_creators_are_declared_and_no_i2_registry_is():
    declared = _read("src/tensorforge/backends/cpp.py")
    for planned in PLANNED_NEW_EXPORTS:
        assert planned in declared, (
            f"{planned} is not declared in the ctypes layer; I1 declares it"
        )
    # ...and the raw-kernel dtype registry is still an I2 deliverable
    # (design section 7): a contract-only tuple would advertise a
    # distinction that is not yet observable.
    assert "RAW_KERNEL_DTYPES" not in declared, (
        "RAW_KERNEL_DTYPES is an I2 registry; I1 must not introduce it"
    )


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
    """I1 makes float32 allocatable and generalizes no operation, so every
    kernel must reject a float32 operand rather than walk it as float64 —
    which would overrun the buffer by exactly a factor of two.

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
        with pytest.raises(ValueError, match="float32"):
            library.tf_core_relu(f32, handle, shape, strides, 0, 2)
        with pytest.raises(ValueError, match="float32"):
            library.tf_core_contiguous_copy(f32, handle, shape, strides, 0, 2)
        with pytest.raises(ValueError, match="float32"):
            library.tf_core_add(f32, handle, handle, shape, strides,
                                strides, 0, 0, 2)
        # ...and in the destination position, which is the direction that
        # would corrupt memory rather than merely misread it.
        with pytest.raises(ValueError, match="float32"):
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
            library.tf_storage_copy_to(handle, out)
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
