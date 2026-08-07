"""Phase-K contract guardrails (native integer tensors and indexing).

**Phase K is newly approved, it was approved after Phase J closed, and K0
through K6 are the milestones that have landed.** K0 was an architecture,
contract, documentation, and status milestone: it shipped
``docs/native_integer_tensors_design.md``, this module, and the narrow
status reconciliation a newly approved phase requires — and **no runtime
behavior at all**.

**K1 was the first milestone with runtime, and deliberately the narrowest
one the phase could have**: the C++ side learned to *represent* ``int64``
(a third dtype enumerator, an allocation and destruction arm, and an
``Int64`` arm on the four transfer boundaries), and everything else it
added was a **barrier**. No public Python name, no C ABI symbol, no
registry movement, no checkpoint or optimizer-state or loader-state or
sampler-state change, no example, no benchmark, and no dependency.

**K2 is where the integer tensor became publicly constructible**, and it
landed atomically: the three Python dtype tables and the host binding
learned ``int64``; ``INDEX_DTYPES == ("int64",)`` appeared beside an
**unmoved** ``SUPPORTED_DTYPES``; the private ``_from_int64_array``
ingress arrived at the storage and core layers; the two wrapper gates
widened from "floating" to "floating **or** index"; and
``NativeTensor.from_int64_array`` / ``item()`` / ``tolist()`` became the
milestone's whole public delta. It added **no C ABI symbol** (still 54),
**no experimental export** (still 25), no CTest, no example, no benchmark,
and no version change of any kind. Every barrier it could meet had already
landed at K1, one milestone earlier — the ordering §32.1 exists to
guarantee.

**K3 shipped the phase's first operation and its first C ABI symbol,
native ``argmax``; K4 shipped its one index-*consuming* operation and its
second and final symbol, ``index_select``, forward only.** Together they
took the exports from 54 to the phase maximum of **56** and the native
CTests from 25 to **27**, added ``"argmax"`` and ``"index_select"`` to
``TENSOR_CORE_OPS`` and **neither** to ``AUTOGRAD_OPS``, and moved nothing
else.

**K5 is the compatibility proof, and it added zero production code.** Its
whole deliverable is ``tests/test_native_integer_compatibility.py``, which
drives the checkpoint, the in-memory optimizer state, the Phase-J loader
and sampler states, the Phase-J delivery contract,
``NativeCrossEntropyLoss``, ``native_accuracy``, and a real
interrupted-and-resumed training run, and shows that K1 through K4 left
every one of them exactly where they found it. No export, no public name,
no CTest, no example, no benchmark, no registry value, and no version
moved.

**K6 is the end-to-end integration example, and it added zero production
code too.** Its deliverable is ``examples/native_integer_indexing.py`` and
its owner ``tests/test_native_integer_indexing_example.py``: a
deterministic native classifier whose evaluation path takes a native
``argmax`` and an ``index_select`` of the predicted-class logits, with an
interrupted-and-resumed run reproducing the uninterrupted one exactly at
float64 and float32 independently. It is the **one** inventory K6 moves —
examples 16 -> **17** — and it moved nothing else: no C ABI symbol, no
public Python name, no CTest, no benchmark, no registry value, and no
version.

``SUPPORTED_DTYPES`` never gains ``int64`` at all, at any milestone, and
``normalize_dtype("int64")`` keeps raising forever. **K7 through K9 are
unstarted**: there is no hardening matrix, no benchmark, no closure, no
general ``gather``, ``scatter``, or embedding lookup, no ``index_select``
backward, no ``max`` or ``argmin``, no integer arithmetic or reduction, no
integer autograd, parameter, buffer, optimizer, or checkpoint entry, and no
casting or promotion.

Three kinds of fact live here, and keeping them apart is the point of the
module:

* **What the repository is** — the live registries, the live source, the
  real files, and the real inventories. Every inherited value is written
  down here *independently* of the module that defines it, so a silent
  change fails rather than propagating.
* **What the contract says** — a property of the design document, which
  spans the whole phase and does not move as milestones land. These
  assertions are **section-scoped** and require **combinations** of
  architectural terms rather than the presence of one vague word: a
  document that merely contains the string "argmax" passes nothing here.
* **What is deliberately absent** — the runtime Phase K has not built. The
  absence patterns are narrow on purpose: TensorForge has legitimate host
  ``int64`` class-label metadata, ``int64`` layout arrays, and an
  ``int64`` ctypes binding today, and none of those is a native integer
  tensor. Banning the string ``int64`` would fail on the very code the
  phase must not disturb.

They deliberately test *values and structure* rather than wording, so
ordinary prose improvements do not require rewriting them. Nothing here
asserts a total suite count, a benchmark number, an error message, a
paragraph order, or a line number.

**Every parser and scanner in this module has a negative control.** A
checker that silently stopped matching would pass forever, so each one is
driven against text it must reject as well as text it must accept. The
controls operate on temporary strings only; no repository file is read for
mutation, written, moved, or restored. Nothing here needs the network, a
Git ancestor, or a complete Git history, so the module is shallow-clone
safe.
"""
import ast
import hashlib
import inspect
import re
import textwrap
from pathlib import Path

import pytest

from tensorforge.backends import cpp

REPO_ROOT = Path(__file__).resolve().parent.parent
PHASE_K_DESIGN_NAME = "native_integer_tensors_design.md"
PHASE_K_DESIGN = REPO_ROOT / "docs" / PHASE_K_DESIGN_NAME

# ---------------------------------------------------------------------------
# The boundary Phase K inherits from a completed Phase J (which inherited it
# unmoved from a completed Phase I). Written here independently of the
# modules under test: if one of these values moves, this module fails rather
# than agreeing with whatever the source now says.
# ---------------------------------------------------------------------------
K0_DTYPES = ("float64", "float32")
K0_DEVICES = ("cpu",)
K0_UNSUPPORTED = ("cuda", "amp")
K0_RAW_KERNEL_DTYPES = ("float64",)
K0_DEFAULT_DTYPE = "float64"

K0_CHECKPOINT_FORMAT = "tensorforge.native_checkpoint"
K0_CHECKPOINT_VERSION = 3
K0_CHECKPOINT_VERSIONS = (1, 2, 3)
K0_OPTIMIZER_STATE_VERSION = 1
K0_LOADER_FORMAT = "tensorforge.native_data_loader"
K0_LOADER_VERSION = 1
K0_LOADER_VERSIONS = (1,)
K0_SAMPLER_FORMAT = "tensorforge.native_sampler"
K0_SAMPLER_VERSION = 1
K0_SAMPLER_VERSIONS = (1,)

K0_EXPORT_COUNT = 54
K0_EXPERIMENTAL_EXPORTS = 25
K0_CTEST_COUNT = 24
K0_EXAMPLE_COUNT = 16
K0_BENCHMARK_COUNT = 9

# The one example the phase adds, at K6, and the count it takes the
# inventory to. Written as K0's number plus a **named** addition rather than
# as a bumped literal, so K6's artifact stays attributed to the milestone
# that shipped it and an unrecorded example still fails an exact equality.
K6_EXAMPLES = {"native_integer_indexing.py": "K6"}
K6_EXAMPLE_COUNT = K0_EXAMPLE_COUNT + len(K6_EXAMPLES)      # 17

# What the live tree holds after K4, derived from K0's inventory plus the
# additions each milestone is on record for, so an unrecorded addition fails
# an exact equality rather than being absorbed into a bumped literal.
K4_EXPORT_COUNT = K0_EXPORT_COUNT + 2                       # 56
K4_CTEST_COUNT = K0_CTEST_COUNT + 3                         # 27
# K1's 32, plus one **export** per Phase-K operation with a floating role to
# guard: argmax's source at K3, and index_select's source and destination at
# K4 — two calls in one export, so the audit's per-export count moves by one.
K4_GUARDED_EXPORTS = 34

# The **only** public registry movement in the whole phase (design §33),
# written here independently of ``backends/cpp.py``. It appeared at K2, in
# the same commit as the public constructor it promises, and it gains no
# member at any later milestone.
K2_INDEX_DTYPES = ("int64",)
K2_ABI_INT64_CODE = 2
K2_INT64_ITEM_SIZE = 8

# The one inventory K1 moves, and the only one: the new native CTest that
# proves the int64 representation and drives the floating-role barrier
# table. Everything else in §33's row for K1 is identical to K0's.
K1_CTEST_COUNT = 25
K3_CTEST_COUNT = 26
assert K1_CTEST_COUNT == K0_CTEST_COUNT + 1
assert K3_CTEST_COUNT == K1_CTEST_COUNT + 1   # K3's argmax target, and only it
assert K4_CTEST_COUNT == K3_CTEST_COUNT + 1   # K4's index_select target

# The C++ dtype enumerators after K1. ``int64`` takes code 2 — the code the
# Phase-I header comment reserved for a future dtype — and neither floating
# code moves. Written here independently of the header.
K1_ABI_DTYPE_CODES = {"FLOAT64": 0, "FLOAT32": 1, "INT64": 2}
K1_SCOPED_DTYPES = {"Float64", "Float32", "Int64"}

# The exports that carry the K1 floating-role guard, and the ones that do
# not. Written down as a count rather than a list here because the *names*
# are audited by tests/test_native_integer_barriers.py against the live
# source; what this module pins is that the split exists and that its two
# halves add up to the unchanged export inventory.
K1_GUARDED_EXPORTS = 32
K1_GENERALIZED_TRANSFERS = (
    "tf_storage_copy_from", "tf_storage_copy_to", "tf_storage_materialize",
    "tf_core_contiguous_copy",
)

# The production C++ translation units at K0. An integer kernel would have
# to live somewhere, and "somewhere" is a new file or an existing one; this
# pins the file set and the export inventory pins the contents.
K0_CPP_SOURCES = (
    "classification.cpp", "conv2d.cpp", "elementwise.cpp", "error.cpp",
    "matmul.cpp", "pooling.cpp", "random.cpp", "reduction.cpp",
    "storage.cpp",
)
# ...and the one the phase adds, at K3, with the internal header beside it.
# Deliberately its own unit rather than a section of reduction.cpp: `argmax`
# searches for a position and writes a different dtype, which is not what
# `tf_core_sum` does (design §32, K3's layer list). **K4 extended that unit
# rather than adding a second one**, so the file set is K3's exactly and
# only the export count inside it moved.
K3_CPP_SOURCES = ("indexing.cpp",)
K3_CPP_HEADER = "tf_indexing_internal.h"
INDEXING_EXPORTS = ("tf_core_argmax", "tf_core_index_select")

# The whole Phase-K ladder, and the split that carries the phase. A
# milestone moves its identifier from the second tuple to the first and
# nowhere else, so the two together are always exactly ``MILESTONES``.
MILESTONES = tuple(f"K{index}" for index in range(10))      # K0 ... K9
COMPLETE_MILESTONES = ("K0", "K1", "K2", "K3", "K4", "K5", "K6", "K7")
UNSTARTED_MILESTONES = tuple(name for name in MILESTONES
                             if name not in COMPLETE_MILESTONES)
assert len(UNSTARTED_MILESTONES) == 2

# The milestones that ship **no production code at all**, and the module
# each one's proof lives in. K0 was architecture and guardrails; K5 is the
# compatibility proof; K6 is the end-to-end example and its owner; K7 is
# the adversarial hardening matrix. Written down because "this milestone
# landed" and "this milestone changed the package" are different facts, and
# the second is what the *package* inventories above are measured against.
# K6 adds one file under ``examples/``, which is a program written against
# the public API rather than production code — the design's own K6 row says
# so, and ``K6_EXAMPLES`` names it. K7 adds nothing outside ``tests/``.
ZERO_PRODUCTION_MILESTONES = {
    "K0": "tests/test_native_phase_k.py",
    "K5": "tests/test_native_integer_compatibility.py",
    "K6": "tests/test_native_integer_indexing_example.py",
    "K7": "tests/test_native_integer_hardening.py",
}

# The ordering the phase turns on (design §32.1): every reachability
# barrier lands at K1, and the first milestone at which an ``int64`` tensor
# can be constructed is K2. Written here independently of the document, so
# a ladder that reordered them would fail rather than be described.
BARRIER_MILESTONE = "K1"
FIRST_CONSTRUCTION_MILESTONE = "K2"
assert (MILESTONES.index(BARRIER_MILESTONE)
        < MILESTONES.index(FIRST_CONSTRUCTION_MILESTONE))

# The phase's public names, and the milestone that adds each. They are
# ``NativeTensor``/``NativeTensorCore`` methods rather than new classes, so
# ``experimental.__all__`` never moves — see design §23.2.
#
# The split is the point, and an entry moves from the second dict to the
# first when its milestone lands and never the other way (§37.2). K2
# shipped three names, all on ``NativeTensor``; ``NativeTensorCore``
# deliberately gained **no** public name, which is why the landed entries
# carry the class they landed on.
LANDED_TENSOR_METHODS = {
    "from_int64_array": "K2",
    "item": "K2",
    "tolist": "K2",
    "argmax": "K3",
    "index_select": "K4",
}
# Empty since K4: the phase's whole public delta has landed, and K5 through
# K9 add **no public name** (design §23.1). The map stays rather than being
# deleted, so a milestone that invented one would still have to move it here
# first.
PLANNED_TENSOR_METHODS = {}
# K3's one name is the phase's first that lands on **both** layers:
# ``NativeTensorCore.argmax`` is a public Core operation, unlike K2's
# construction door, whose Core and storage helpers stayed private. K4's
# ``index_select`` is the second and last.
CORE_METHODS_BY_MILESTONE = {"argmax": "K3", "index_select": "K4"}

# The C ABI delta, and its maximum. Phase K adds exactly two symbols and no
# milestone may exceed 56 (design §22.3). An entry moves from the planned
# map to the landed one when its milestone ships, and never back — and with
# K4 the planned map is empty, so the phase is at its committed ceiling.
LANDED_EXPORTS = {"tf_core_argmax": "K3", "tf_core_index_select": "K4"}
PLANNED_EXPORTS = {}
PHASE_K_MAX_EXPORTS = 56
assert (K0_EXPORT_COUNT + len(LANDED_EXPORTS) + len(PLANNED_EXPORTS)
        == PHASE_K_MAX_EXPORTS)
assert K0_EXPORT_COUNT + len(LANDED_EXPORTS) == K4_EXPORT_COUNT

# ``CLAUDE.md``'s only size policy. The project ceiling, not a new target.
CLAUDE_MD_CEILING = 150_000

# The status surfaces a newly approved phase has to reconcile. These are
# repository-owned documentation; ``src/tensorforge/experimental/__init__.py``
# is deliberately absent because K0 changes no production source at all.
STATUS_SURFACES = (
    "README.md",
    "CLAUDE.md",
    "docs/roadmap.md",
    "docs/project_summary.md",
    "docs/native_support_matrix.md",
    "docs/architecture.md",
    "docs/release_history.md",
)


# ---------------------------------------------------------------------------
# Helpers. Small, named, and each with a negative control below.
# ---------------------------------------------------------------------------

def _read(relative):
    return (REPO_ROOT / relative).read_text(encoding="utf-8")


def _design():
    return PHASE_K_DESIGN.read_text(encoding="utf-8")


def _flat(text):
    """Whitespace-flattened, emphasis-stripped text, so a claim split
    across lines or wrapped in markdown still reads as one sentence."""
    return re.sub(r"\s+", " ", re.sub(r"[*`]", "", text))


def _head():
    """The design's status header — everything before section 1.

    Bounded by the document's own first section rather than by a byte
    count, so a header that legitimately grows by one milestone's status
    does not push a required claim out of the window being checked."""
    text = _design()
    end = text.find("\n## 1.")
    assert end > 0, "the design has no section 1 to bound the header"
    return _flat(text[:end])


def _section(text, number):
    """The body of top-level section ``number``, up to the next one.

    Section-scoped rather than whole-document, so an assertion about the
    argmax contract cannot be satisfied by a sentence in the motivation."""
    marker = f"\n## {number}."
    assert marker in text, f"the design has no section {number}"
    body = text.split(marker, 1)[1]
    following = re.search(r"\n## \d+\.", body)
    return body[:following.start()] if following else body


def _missing(haystack, *terms):
    """The terms absent from one flattened body, case-insensitively."""
    flat = _flat(haystack).lower()
    return [term for term in terms if term.lower() not in flat]


def _module_source(relative):
    return (REPO_ROOT / relative).read_text(encoding="utf-8")


def _code_only(source):
    """Source with every docstring and string literal removed, through the
    AST rather than by regex.

    A substring ban that reads prose fails on the very sentence that
    documents the prohibition — ``native_metrics.py`` says "there is
    deliberately no native argmax" and must keep saying it. Reading code
    only is what makes an absence scan mean "the runtime does not do this"
    rather than "nobody wrote the word"."""
    tree = ast.parse(source)
    pieces = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            pieces.append(node.id)
        elif isinstance(node, ast.Attribute):
            pieces.append(node.attr)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef,
                               ast.ClassDef)):
            pieces.append(node.name)
        elif isinstance(node, ast.keyword) and node.arg:
            # Keyword-argument *names* are code too: without this a
            # ``_trusted_dtype=True`` or ``dtype="int64"`` is invisible.
            pieces.append(node.arg)
        elif isinstance(node, ast.arg):
            pieces.append(node.arg)
    return pieces


def _cpp_code_only(text):
    """C++ source with ``/* */`` and ``//`` comments removed.

    The C++ counterpart of ``_code_only``, and it exists for the same
    reason: an absence scan must mean "the runtime does not do this" rather
    than "nobody wrote the word", and every header here documents what it
    deliberately does not declare."""
    return re.sub(r"//[^\n]*", " ", re.sub(r"/\*.*?\*/", " ", text, flags=re.S))


def _defined_names(relative, class_name):
    """Every method name defined on ``class_name`` in a module."""
    tree = ast.parse(_module_source(relative))
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            return {child.name for child in node.body
                    if isinstance(child, (ast.FunctionDef,
                                          ast.AsyncFunctionDef))}
    raise AssertionError(f"{relative} defines no class {class_name}")


def _dict_literal_keys(relative, variable):
    """The string keys of a module-level dict literal, read from the AST."""
    tree = ast.parse(_module_source(relative))
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        targets = [t.id for t in node.targets if isinstance(t, ast.Name)]
        if variable in targets and isinstance(node.value, ast.Dict):
            return {key.value for key in node.value.keys
                    if isinstance(key, ast.Constant)}
    raise AssertionError(f"{relative} has no dict literal named {variable}")


def _source_exports():
    """The exported production ``tf_*`` symbol names, parsed from the C++
    sources — the source-level export inventory the repository already
    treats as authoritative."""
    names = set()
    for path in sorted((REPO_ROOT / "cpp" / "src").glob("*.cpp")):
        names.update(re.findall(r"TF_EXPORT[^;{]*?\b(tf_[a-z0-9_]+)\s*\(",
                                path.read_text(encoding="utf-8"), re.S))
    return names


# The over-claim scanner. Two arms, because the claim has two ordinary
# shapes — "<subject> is supported" and "<something> supports <subject>" —
# and the second is invisible to the first. Each entry is (label, pattern),
# and every one is driven against text it must catch *and* text it must not
# by ``test_the_overclaim_scanner_can_actually_fail``.
#
# Two deliberate narrownesses, both learned from real false positives:
#
# * the *subject* arm requires an explicit noun (``integer tensors``,
#   ``int64 dtype``, ``integer gradients``…). A bare ``index`` would match
#   the middle of ``index_select`` and turn every sentence about the
#   planned operation into an over-claim.
# * the *operation* arms do not treat "complete" as a landing word. A
#   sentence such as "the argmax and index_select contracts are complete"
#   is a statement about this **document**, and it is true; only
#   supported/available/implemented/shipped assert a runtime.
_SUBJECT = (r"(native\s+)?(int64|integer)[- ]?"
            r"(tensors?|dtype|storage|gradients?|parameters?|arithmetic)")
_BECAME = r"\b(is|are|was|were|now|has|have)\s+"
_LANDED = r"(supported|available|implemented|shipped|landed|working)\b"
_DONE = r"(complete|completed|closed|finished|done)\b"

_PHASE_K_OVERCLAIMS = (
    ("native integer tensors exist",
     _SUBJECT + r"[^.]{0,40}" + _BECAME + _LANDED),
    ("something supports integer tensors",
     r"\bsupports?\b[^.]{0,40}\b(int64|integer)[- ]?tensors?\b"),
    # ``argmax`` left this tuple at K3, which shipped it — the one entry
    # that has ever been removed, and it is removed rather than exempted
    # because a scanner that bans an accurate sentence is a scanner that
    # forces every status surface to lie. What remains banned is the claim
    # K3 did *not* earn: a ``max``, which §17.10 permanently declines.
    # Both spellings require a **landing verb**. A bare ``max_with_indices``
    # ban would fire on the very sentence that records the prohibition —
    # ``docs/native_support_matrix.md`` names it in its not-supported list —
    # which is the substring-ban failure mode this module exists to avoid.
    ("max is shipped beside argmax",
     r"\b(max_with_indices|max)\b[^.]{0,30}" + _BECAME + _LANDED),
    # ``index_select`` left this tuple at K4, which shipped it — the second
    # entry ever removed, and removed for ``argmax``'s reason rather than
    # exempted: a scanner that bans an accurate sentence forces every status
    # surface to lie. What remains banned is the general ``gather`` the
    # phase permanently declines (§18.1), and the embedding and scatter
    # §35 keeps outside it.
    ("a general gather or scatter exists",
     r"\b(gather|scatter|scatter[_ ]add|embedding)\b[^.]{0,40}"
     + _BECAME + _LANDED),
    # The sentinel advances one milestone as each lands, and only then:
    # it read K2-and-later while K1 was the newest, moved to K3 when K2
    # shipped, to K4 when K3 did, to K5 when K4 did, to K6 when K5 did, to
    # K7 when K6 did, and to K8 when K7 did. Keeping the old bound would
    # force every status surface to under-report the project, which is the
    # mirror of the failure this scanner exists to catch.
    ("a Phase-K milestone after K7 has landed",
     r"\bK(?:[89]|10)\b[^.]{0,30}" + _BECAME + r"(" + _LANDED + r"|"
     + _DONE + r")"),
    ("Phase K is finished",
     r"\bPhase K\s+(is|was|has been)\s+" + _DONE),
    ("a checkpoint version beyond 3",
     r"\bcheckpoint\b[^.]{0,40}\b(is|was|now|at|to|moved to|bumped to)\s+"
     r"version\s*[4-9]\b"),
    ("CUDA or AMP arrived",
     r"\b(CUDA|AMP)\b[^.]{0,30}" + _BECAME + _LANDED),
)

# Negated, planned, or explicitly future forms are the honest ones. The
# window is deliberately narrow and the tokens strict, so a disclaimer
# elsewhere in a paragraph cannot launder a real claim.
_NEGATED = re.compile(
    r"\b(no|not|never|nothing|none|cannot|without|future|beyond|planned|"
    r"unplanned|outside|remains?|remain|still|deliberately|excluded|"
    r"unsupported|unstarted|would|will|may|eventual\w*|once|only when|"
    r"absent|deferred|prohibited|rejected|refus\w*)\b", re.I)


def _overclaims(text):
    """Every over-claim in one body, ignoring negated and planned forms."""
    flat = _flat(text)
    found = []
    for label, pattern in _PHASE_K_OVERCLAIMS:
        for match in re.finditer(pattern, flat, re.I):
            window = flat[max(0, match.start() - 80):match.end() + 30]
            if not _NEGATED.search(window):
                found.append((label, match.group(0)))
    return found


# The external-provenance scanner. It is deliberately **generic**: naming
# the project it is meant to keep out would itself put that name in the
# repository, which is exactly what the rule forbids. So it looks for the
# *shape* of a provenance reference instead — a foreign repository URL, or
# porting/permission language — which catches any such reference rather
# than one particular one.
#
# It has **two** arms, and the first one is exact.
#
# **Exact tokens.** Some identifiers must never appear in this repository at
# all — a particular external project's name, its owner, its repository URL,
# and the source-path fragments unique to it. Writing any of them here to
# scan for them would put the very string in the tree that the rule forbids,
# so each is stored **encoded** — every stored character is **one codepoint
# above** the character it stands for, and decoding shifts it **down** by
# one — and rebuilt at import. Nothing readable is committed, and the second
# field of each row is a committed **full 64-character SHA-256 digest** of
# the intended token, so a typo in the encoded form fails loudly at import
# instead of silently disarming the scan. The banned set is lowercase and
# matched case-insensitively, so the path fragments below also cover every
# longer path containing them.
_ENCODED_BANNED = (
    ('ebfebmvt',
     "6afe40611ff94485d186f84f9433631893b90737bc4fa518a510f7b65c874bfe"),
    ('ebfebmvt.nm',
     "1c5226eb827d8042fe0fb7f892f3e4a43afac961dc9b17e25369056781b3670e"),
    ('kpiotpolbzbuj',
     "2294d267d94a81ee358011cb2f88439099477e6fcd4d8535568142c958c3b705"),
    ('iuuqt;00hjuivc/dpn0kpiotpolbzbuj0ebfebmvt.nm',
     "367146612f09bda6f527f4be9ffc855a47aa4c6ce46ad7d18d4cb2c8c908da98"),
    ('ebfebmvt0dpsf',
     "79afb7eaec7640b2c9b9b0cb9d6b8e4445ac18853a2b8851d6e0b0a704a65129"),
    ('ebfebmvt0bvuphsbe',
     "4e58c204fa905e8324d2941fe0db6987acf121ca65676429e309971870060af4"),
)

# The six rows above are, in order: the project name, the package /
# repository name, the owner, the **complete repository URL** (a decoded
# exact token in its own right, not merely an owner/repo substring caught
# incidentally inside some other URL), and two source-path fragments unique
# to that project.


def _decode(text):
    """Rebuild an exact token from its stored form.

    Each stored character is **one codepoint above** the character it
    stands for, so decoding shifts **down** by one. Nothing readable is
    committed, and every decoded token is checked against a committed
    **full 64-character SHA-256 digest** below, so a typo in the encoded
    form fails loudly at import instead of silently disarming the scan."""
    return "".join(chr(ord(char) - 1) for char in text)


def _banned_tokens():
    """The exact banned identifiers, each verified against its full digest."""
    tokens = []
    for encoded, digest in _ENCODED_BANNED:
        assert len(digest) == 64, "digests must be full SHA-256 values"
        token = _decode(encoded)
        actual = hashlib.sha256(token.encode()).hexdigest()
        assert actual == digest, "a banned token decoded to the wrong value"
        tokens.append(token)
    return tokens


BANNED_TOKENS = _banned_tokens()

_OWN_REPO = re.compile(r"github\.com/[\w.-]+/[Tt]ensor[Ff]orge\b", re.I)
_FOREIGN_URL = re.compile(r"https?://(?:www\.)?(?:github|gitlab|bitbucket)"
                          r"\.com/[\w.-]+/[\w.-]+", re.I)
# The nouns are deliberately restricted to ones that can only mean "another
# codebase". A bare "implementation" would fire on this repository's own
# honest sentences about its own two traversals ("borrowed from either
# implementation"), which say nothing about provenance.
_PROVENANCE = re.compile(
    r"\b(ported|adapted|copied)\s+(from|out of)\b[^.]{0,60}"
    r"\b(repo|repository|codebase|upstream|external project)\b"
    r"|\bwith (the )?(explicit |express )?permission\b"
    r"|\breference (implementation|repository|project)\s+(at|from)\b",
    re.I)


def _provenance_hits(text):
    """Every exact banned identifier, plus foreign-repository URLs and
    porting/permission language.

    The first arm is exact and is what the rule actually requires; the
    second is the generic net that catches *any* external-provenance
    reference, named or not."""
    flat = _flat(text)
    lowered = flat.lower()
    hits = [token for token in BANNED_TOKENS if token in lowered]
    hits.extend(url for url in _FOREIGN_URL.findall(flat)
                if not _OWN_REPO.search(url))
    hits.extend(match.group(0) for match in _PROVENANCE.finditer(flat))
    return hits


# Directories that are generated, cached, binary, VCS, virtual-environment,
# or build output. These are the **only** things skipped; everything else
# under the repository root is read.
_SKIPPED_DIRECTORIES = frozenset({
    ".git", ".hg", ".svn", ".venv", "venv", "env", "build", "dist",
    "__pycache__", ".pytest_cache", ".cache", ".mypy_cache", ".ruff_cache",
    ".tox", ".eggs", "node_modules", ".idea", "site-packages",
})


def _repository_text_files():
    """Every repository-owned **text** file, found exhaustively.

    Deliberately not an extension allow-list: a provenance reference can sit
    in a `CMakeLists.txt`, a `.cmake` module, a workflow `.yml`, a JSON
    config, a shell or PowerShell script, an extensionless dotfile, or a
    top-level `LICENSE`. The sweep therefore walks the whole tree, skips
    only the generated/binary/cache/VCS/virtualenv/build directories above,
    and decides text-versus-binary by **attempting a UTF-8 decode** rather
    than by guessing from a suffix. A file is ignored only when decoding
    proves it is not text.

    No file is exempt — this module included. No network, no Git history."""
    found = []
    for path in REPO_ROOT.rglob("*"):
        if not path.is_file():
            continue
        if _SKIPPED_DIRECTORIES & set(path.relative_to(REPO_ROOT).parts[:-1]):
            continue
        try:
            found.append((path, path.read_text(encoding="utf-8")))
        except UnicodeDecodeError:
            # The **only** reason a file is skipped: decoding proved it is
            # not UTF-8 text. An unreadable repository-owned file is a real
            # failure and must propagate — suppressing OSError here would
            # let a file the sweep cannot open pass as "nothing found".
            continue
    return found


# ===========================================================================
# 0. The helpers themselves — negative controls
# ===========================================================================
#
# "No offenders" only means something when the checker is known to be able
# to find one. Every helper above is driven here against text it must
# accept and text it must reject.

def test_the_section_parser_finds_a_section_and_rejects_a_missing_one():
    text = "# T\n\n## 1. One\nalpha\n\n## 2. Two\nbeta\n"
    assert "alpha" in _section(text, 1) and "beta" not in _section(text, 1)
    assert "beta" in _section(text, 2)
    with pytest.raises(AssertionError):
        _section(text, 3)


def test_the_flattener_joins_wrapped_and_emphasised_prose():
    assert _flat("**int64** is\n  not\tsupported") == "int64 is not supported"


def test_the_term_checker_can_actually_fail():
    assert _missing("alpha beta", "alpha", "beta") == []
    assert _missing("alpha", "beta", "gamma") == ["beta", "gamma"]


def test_the_cpp_comment_stripper_can_actually_fail():
    """Negative control for ``_cpp_code_only``, on temporary strings."""
    source = ("// TF_EXPORT void tf_core_probe(void);\n"
              "/* tf_core_probe again */\n"
              "TF_EXPORT void tf_core_real(void);\n")
    code = _cpp_code_only(source)
    assert "tf_core_probe" not in code, "prose leaked into the code view"
    assert "tf_core_real" in code and code.count("TF_EXPORT") == 1


def test_the_code_only_reader_ignores_prose_and_keeps_keywords():
    source = ('"""a docstring naming argmax."""\n'
              'def f(x):\n'
              '    return g(x, dtype="int64", trusted=True)\n')
    names = _code_only(source)
    assert "argmax" not in names, "prose leaked into the code view"
    assert "dtype" in names and "trusted" in names, (
        "keyword-argument names must be visible to an absence scan"
    )
    assert "g" in names and "f" in names


def test_the_overclaim_scanner_can_actually_fail():
    """The control every edit to ``_PHASE_K_OVERCLAIMS`` requires."""
    for caught in (
        "native int64 tensors are supported",
        "the backend supports integer tensors",
        "max is implemented",
        "max_with_indices is now available",
        "gather is available",
        "a general gather has landed",
        "scatter_add is implemented",
        "embedding is now supported",
        "Phase K is complete",
        "K8 has landed",
        "K9 is shipped",
        "the checkpoint is now at version 4",
        "CUDA is supported",
        "integer gradients are supported",
        "integer parameters are available",
    ):
        assert _overclaims(caught), caught
    # ...and every accurate sentence a K7 surface must be able to write.
    for allowed in (
        "int64 is not a supported native tensor dtype",
        "Phase K is newly approved and K0 through K7 are complete",
        "K0 through K7 are the only completed Phase-K milestones",
        "K8 through K9 are unstarted",
        "K4 is complete",
        "K5 is complete",
        "K5 landed and added zero production code",
        "K6 is complete",
        "K6 landed and added zero production code",
        "K7 is complete",
        "K7 landed and added zero production code",
        "the adversarial hardening matrix has landed",
        "the end-to-end integration example has landed",
        "the compatibility proof has landed",
        "a native argmax is implemented",
        "argmax is available at both floating dtypes",
        "index_select has landed, forward only",
        "index_select is implemented and consumes an int64 index tensor",
        "a future milestone may add the index_select backward",
        "max is deliberately not shipped",
        "a native max, max_with_indices, or argmin",
        "there is no max_with_indices and none is planned",
        "there is no general gather and none is planned",
        "no scatter or embedding exists",
        "integer gradients are prohibited",
        "integer parameters are rejected before allocation",
        "CUDA remains unsupported",
        "there is no checkpoint version 4",
        "Phase J is complete",
        "float32 is supported",
    ):
        assert _overclaims(allowed) == [], (allowed, _overclaims(allowed))


def test_the_provenance_scanner_catches_each_exact_identifier_alone():
    """The controls the exact arm exists for: each banned identifier must
    be caught **on its own**, embedded in otherwise ordinary prose, and in
    any casing. The strings are rebuilt at runtime, so none of them is
    committed to this file."""
    project, package, owner, url_path, path_core, path_autograd = \
        BANNED_TOKENS
    for label, token in (("project name", project),
                         ("package name", package),
                         ("owner name", owner),
                         ("repository URL", f"https://github.com/{url_path}"),
                         ("source-path fragment", f"include/{path_core}/x.h"),
                         ("second path fragment", f"include/{path_autograd}")):
        sentence = f"the kernel in {token} is worth a look"
        assert _provenance_hits(sentence), label
        assert _provenance_hits(sentence.upper()), (label, "upper case")
    # ...and each of those six is caught by the **exact** arm, not merely by
    # the generic URL net, so removing the generic arm would not hide them.
    for token in BANNED_TOKENS:
        assert token in _provenance_hits(f"x {token} y")


def test_the_generic_provenance_arm_can_actually_fail():
    """The second arm, driven against encoded controls so this module's own
    text stays clean and needs no self-exemption."""
    for encoded in (
        'tff!iuuqt;00hjuivc/dpn0tpnfpof0tpnf.svoujnf!gps!uif!psjhjobm',
        'uijt!lfsofm!xbt!qpsufe!gspn!uif!vqtusfbn!sfqptjupsz',
        'vtfe!xjui!uif!fyqmjdju!qfsnjttjpo!pg!uif!pxofs',
        'uif!sfgfsfodf!jnqmfnfoubujpo!bu!uibu!beesftt!epft!ju!ejggfsfoumz',
    ):
        sentence = _decode(encoded)
        assert _provenance_hits(sentence), sentence


def test_the_provenance_scanner_does_not_fire_on_ordinary_internal_prose():
    """The false positives that a blunter scanner produced on this
    repository's own honest sentences. ``borrowed implementation`` is the
    one that actually happened: two *internal* traversals, nothing to do
    with provenance."""
    for allowed in (
        "https://github.com/Youssef-Boussabah/TensorForge",
        "a value borrowed from either implementation is still exact",
        "the borrowed implementation is the shipped generic reference path",
        "a borrowing view never closes its parent's storage",
        "the design is derived from the measurements in this document",
        "permission is not required to read a public standard",
        "an external implementation may be a private comparative input",
        "this kernel was copied into contiguous storage before the call",
    ):
        assert _provenance_hits(allowed) == [], (allowed,
                                                 _provenance_hits(allowed))


# ===========================================================================
# 1. The design document exists, is reachable, and names its subject
# ===========================================================================

def test_the_phase_k_design_exists_and_is_not_a_sketch():
    assert PHASE_K_DESIGN.is_file(), f"missing docs/{PHASE_K_DESIGN_NAME}"
    assert len(_design().strip()) > 20_000, (
        "the Phase-K contract is too short to be an implementation contract"
    )


def test_the_phase_k_design_is_linked_from_the_readme_and_the_doc_map():
    """A contract nobody can find is not a contract. The README links every
    document, and CLAUDE.md's documentation map names the authority for
    each question."""
    for surface in ("README.md", "CLAUDE.md"):
        assert f"docs/{PHASE_K_DESIGN_NAME}" in _read(surface), surface


def test_the_design_names_the_phase_and_its_subject():
    heading = _design().splitlines()[0]
    assert _missing(heading, "Phase K") == [], heading
    assert _missing(_design()[:6000], "integer", "index") == []


# ===========================================================================
# 2. Status: newly approved after a completed Phase J, and zero runtime
# ===========================================================================

def test_the_design_presents_phase_k_as_newly_approved_after_phase_j():
    """The failure this guards: a document that reads Phase K as though it
    had always been on the roadmap, which would make "Phase J closed
    without a successor" retroactively false."""
    head = _head()
    assert re.search(r"newly\s+approved", head, re.I), (
        "the design does not say Phase K is newly approved"
    )
    assert re.search(r"Phase J[^.]{0,120}remains complete", head,
                     re.I), head[:600]
    assert re.search(r"without committing to a successor"
                     r"|approved afterwards", head, re.I), (
        "the design does not record that Phase J closed without a successor"
    )
    # ...and it denies each of the four ways history could be rewritten.
    for denial in ("not part of the Phase-I roadmap",
                   "not planned during Phase J",
                   "did not exist before"):
        assert denial.lower() in head.lower(), denial


def test_the_design_states_what_has_landed_and_what_has_not():
    head = _head()
    assert re.search(r"K0 adds no runtime behavior", head, re.I), head[:800]
    # Both shapes of the same claim: the enumeration the header used while
    # the completed set was short, and the range form it takes once the list
    # would be unreadable. The **bounds** are what is checked either way, so
    # a header naming the wrong last milestone still fails.
    enumerated = ", ".join(COMPLETE_MILESTONES[:-1]) + \
        f", and {COMPLETE_MILESTONES[-1]}"
    ranged = f"{COMPLETE_MILESTONES[0]} through {COMPLETE_MILESTONES[-1]}"
    assert re.search(rf"({enumerated}|{ranged}) are the only completed "
                     rf"Phase-K milestones", head, re.I), head[:1800]
    first_unstarted = UNSTARTED_MILESTONES[0]
    last = UNSTARTED_MILESTONES[-1]
    assert re.search(rf"{first_unstarted} through {last} are unstarted",
                     head, re.I), head[:1800]
    assert re.search(r"[Rr]untime capability begins at K1", head), head[:2200]
    # The claim that must be impossible to misread at every milestone
    # before K2, and after it: ``int64`` never joins the compute registry.
    assert re.search(r"int64 is not a supported TensorForge native tensor "
                     r"dtype", head, re.I), head[:2600]
    # ...and the exact scope of what K2 opened, in the header, because "one
    # public door" and "a supported dtype" are the two things a reader must
    # not conflate.
    assert re.search(r"NativeTensor\.from_int64_array", head), head[:3200]
    assert re.search(r"INDEX_DTYPES", head), head[:3200]


def test_the_design_header_records_the_inherited_boundary_unmoved():
    """The header carries the inherited compute boundary *and* says, in so
    many words, that no milestone has moved it.

    The wording names the milestones that have landed, so it advances as the
    ladder does — the alternation below accepts every form the claim has
    taken rather than freezing one. What it will never accept is the header
    simply dropping the claim, which is the failure this guards."""
    head = _head()
    for value in ("float64", "float32", "cpu", "cuda", "amp",
                  "tensorforge.native_checkpoint",
                  "tensorforge.native_data_loader",
                  "tensorforge.native_sampler"):
        assert value.lower() in head.lower(), value
    assert re.search(r"(moves|moved) none of (it|them)"
                     r"|neither K1 nor K2 moved any of it"
                     r"|no Phase-K milestone has moved any of it", head,
                     re.I), head[:3000]


# ===========================================================================
# 3. The live registries have not moved
# ===========================================================================

def test_the_supported_dtype_registry_is_exactly_what_phase_j_left():
    assert cpp.SUPPORTED_DTYPES == K0_DTYPES
    assert cpp.SUPPORTED_DEVICES == K0_DEVICES
    assert cpp.UNSUPPORTED == K0_UNSUPPORTED
    assert cpp.RAW_KERNEL_DTYPES == K0_RAW_KERNEL_DTYPES
    assert cpp.normalize_dtype(None) == K0_DEFAULT_DTYPE
    assert cpp.normalize_device(None) == "cpu"


def test_the_compute_registry_never_gained_int64_and_never_will():
    """The permanent half of taxonomy B, and the one claim that is
    identical at every Phase-K milestone: ``int64`` is a dtype a native
    tensor may *carry*, never one the kernels *compute* at.

    K2 added a **separate** row for it (asserted below) rather than
    widening this one, so the sentences ``normalize_dtype`` prints and the
    set every generic constructor validates against did not move."""
    assert "int64" not in cpp.SUPPORTED_DTYPES
    assert "int64" not in cpp.RAW_KERNEL_DTYPES
    with pytest.raises(ValueError):
        cpp.normalize_dtype("int64")
    # ...and no registry other than the one the contract names has appeared.
    for absent in ("COMPUTE_DTYPES", "INTEGER_DTYPES", "TENSOR_DTYPES"):
        assert not hasattr(cpp, absent), absent


def test_the_index_registry_is_exactly_what_k2_promised():
    """The one public registry movement of the whole phase, asserted as an
    exact value and written here independently of the module under test."""
    assert cpp.INDEX_DTYPES == K2_INDEX_DTYPES
    assert not set(cpp.INDEX_DTYPES) & set(cpp.SUPPORTED_DTYPES)
    # Representable **and** promised — the Phase-I no-drift guarantee,
    # generalized to two registries rather than deleted (design §5.1).
    assert set(cpp._DTYPE_CODES) == set(K0_DTYPES) | set(K2_INDEX_DTYPES)
    assert cpp._normalize_internal_dtype("int64") == "int64"
    assert cpp._normalize_index_dtype("int64") == "int64"


def test_backend_info_reports_four_dtype_rows_and_no_derived_fifth():
    info = cpp.backend_info()
    assert info["dtype"] == K0_DEFAULT_DTYPE
    assert info["device"] == "cpu"
    assert info["supported_dtypes"] == K0_DTYPES
    assert info["index_dtypes"] == K2_INDEX_DTYPES
    assert info["supported_devices"] == K0_DEVICES
    assert info["raw_kernel_dtypes"] == K0_RAW_KERNEL_DTYPES
    assert info["unsupported"] == K0_UNSUPPORTED
    assert info["stable_framework_integration"] is False
    # The default is still the default: no omitted dtype selects an index
    # dtype, at any constructor.
    assert info["dtype"] not in info["index_dtypes"]
    # The union is stated in prose, never materialized as a fifth key that
    # could drift from the two it derives from (design §5.1).
    for absent in ("compute_dtypes", "integer_dtypes", "tensor_dtypes",
                   "all_dtypes", "dtypes"):
        assert absent not in info, absent


def test_no_operation_inventory_grew_an_unplanned_entry():
    """The absence half of the inventory claim, and it narrowed by exactly
    one name at K3 and one more at K4 rather than being loosened.

    ``"argmax"`` and ``"index_select"`` are legitimate ``TENSOR_CORE_OPS``
    members, so each is asserted **present there and absent everywhere
    else** below — including ``AUTOGRAD_OPS``, which neither joins at any
    milestone. Every other banned name is still banned in every inventory,
    ``TENSOR_CORE_KERNELS`` stays frozen, and no inventory gained an integer
    *dtype* entry."""
    for inventory in (cpp.RAW_KERNELS, cpp.TENSOR_CORE_KERNELS,
                      cpp.TENSOR_CORE_OPS, cpp.AUTOGRAD_OPS,
                      cpp.NATIVE_MODULES, cpp.NATIVE_LOSSES,
                      cpp.NATIVE_METRICS, cpp.NATIVE_OPTIMIZERS,
                      cpp.STATE_SUPPORT):
        for banned in ("argmin", "gather",
                       "scatter", "embedding", "int64", "integer", "cast",
                       "astype", "promote"):
            assert not [name for name in inventory
                        if banned in name.lower()], (banned, inventory)
    # ``max`` is banned as a **whole** name rather than as a substring: it
    # is a member of nothing, and §17.10 keeps it that way, while
    # ``maxpool2d_forward`` legitimately contains it.
    for inventory in (cpp.RAW_KERNELS, cpp.TENSOR_CORE_KERNELS,
                      cpp.TENSOR_CORE_OPS, cpp.AUTOGRAD_OPS):
        for banned in ("max", "amax", "max_with_indices"):
            assert banned not in inventory, (banned, inventory)
    # The two names the phase added, each in exactly one inventory and
    # exactly once.
    for landed in ("argmax", "index_select"):
        assert cpp.TENSOR_CORE_OPS.count(landed) == 1, landed
        assert landed not in cpp.AUTOGRAD_OPS, landed
        assert landed not in cpp.TENSOR_CORE_KERNELS, landed
        assert landed not in cpp.RAW_KERNELS, landed
        assert landed not in cpp.NATIVE_METRICS, landed
    assert cpp.NATIVE_METRICS == ("native_accuracy",)


# ===========================================================================
# 4. The inventories have not moved
# ===========================================================================

def test_the_source_export_inventory_is_k0_plus_the_landed_symbols():
    """K0's 54 plus exactly the symbols the landed milestones are on record
    for — two, at K3 and K4 — which is the phase maximum of 56."""
    exports = _source_exports()
    assert len(exports) == K4_EXPORT_COUNT, sorted(exports)
    for name, milestone in LANDED_EXPORTS.items():
        assert name in exports, f"{name} landed at {milestone}"
    assert len(exports - set(LANDED_EXPORTS)) == K0_EXPORT_COUNT
    assert len(exports) <= PHASE_K_MAX_EXPORTS
    # The ceiling is reached, so it is now also the floor of what a further
    # symbol would break: the phase's remaining milestones add none.
    assert len(exports) == PHASE_K_MAX_EXPORTS
    assert PLANNED_EXPORTS == {}


def test_no_unplanned_export_exists():
    exports = _source_exports()
    for name, milestone in PLANNED_EXPORTS.items():
        assert name not in exports, f"{name} belongs to {milestone}"
    # The only ``index``-named export is K4's, and no ``gather`` exists.
    assert [name for name in exports if "index" in name] == \
        ["tf_core_index_select"]
    assert not [name for name in exports if "gather" in name]
    # ...and the shapes K3 and K4 were most tempted to add beside their two
    # symbols.
    for banned in ("tf_core_max", "tf_core_argmin", "tf_core_max_with_indices",
                   "tf_core_argmax_backward", "tf_core_index_select_backward",
                   "tf_core_scatter", "tf_core_scatter_add",
                   "tf_core_embedding", "tf_storage_dtype"):
        assert banned not in exports, banned


@pytest.mark.skipif(not cpp.is_available(),
                    reason="the native backend is not built")
def test_the_built_library_exports_the_same_inventory():
    """The source inventory and the built library must agree — the standing
    ABI-discipline rule, re-asserted at the phase boundary.

    This is also the stale-artifact guard: a library built before K4 would
    export 55 (before K3, 54) and would fail here rather than quietly
    satisfying the Python tests that call the new symbols."""
    storage_tests = pytest.importorskip("test_native_storage_allocation")
    _, names = storage_tests.exported_names(cpp._LIBRARY_PATH)
    if names is None:
        pytest.skip("this image format is not parsed here")
    exported = sorted(name for name in names if name.startswith("tf_"))
    assert len(exported) == K4_EXPORT_COUNT, exported
    assert set(exported) == _source_exports()
    for name in LANDED_EXPORTS:
        assert name in exported, name


def test_the_experimental_export_list_is_still_twenty_five():
    import tensorforge.experimental as experimental

    assert len(experimental.__all__) == K0_EXPERIMENTAL_EXPORTS
    assert len(set(experimental.__all__)) == K0_EXPERIMENTAL_EXPORTS
    for name in experimental.__all__:
        assert hasattr(experimental, name), name


def test_the_ctest_example_and_benchmark_inventories_match_the_ladder():
    """Four milestones have moved an inventory, each by exactly one artifact
    — K1's int64 storage CTest, K3's argmax CTest, K4's index_select CTest,
    and **K6's** integration example — and nothing else. Benchmarks belong
    to K8 and have not moved."""
    cmake = _read("cpp/CMakeLists.txt")
    assert len(re.findall(r"^\s*add_test\(", cmake, re.M)) == K4_CTEST_COUNT
    assert len(list((REPO_ROOT / "cpp" / "tests").glob("*.cpp"))) == \
        K4_CTEST_COUNT
    for name in ("test_dtype_int64_storage.cpp", "test_argmax.cpp",
                 "test_index_select.cpp"):
        assert (REPO_ROOT / "cpp" / "tests" / name).is_file(), name
    for target in ("dtype_int64_storage", "argmax", "index_select"):
        assert f"add_test(NAME {target} " in cmake, target
    examples = [path.name for path in (REPO_ROOT / "examples").glob("*.py")]
    for name, milestone in K6_EXAMPLES.items():
        assert name in examples, (name, milestone)
    assert len(examples) == K6_EXAMPLE_COUNT, sorted(examples)
    assert len([name for name in examples
                if name not in K6_EXAMPLES]) == K0_EXAMPLE_COUNT
    assert len(list((REPO_ROOT / "benchmarks").glob("*.py"))) == \
        K0_BENCHMARK_COUNT


def test_the_production_cpp_translation_units_are_k0s_plus_the_indexing_unit():
    """K0's nine, plus the one K3 adds and its internal header — and **K4
    added no second one**, which is the claim this now also carries.

    The file set is pinned because an integer kernel has to live somewhere,
    and the export inventory pins the contents. K3's unit is listed by name
    rather than absorbed, so a *tenth* file appearing without a milestone
    still fails."""
    present = tuple(sorted(path.name for path in
                           (REPO_ROOT / "cpp" / "src").glob("*.cpp")))
    assert present == tuple(sorted(K0_CPP_SOURCES + K3_CPP_SOURCES))
    headers = {path.name for path in (REPO_ROOT / "cpp" / "include").glob("*.h")}
    assert K3_CPP_HEADER in headers
    # The internal header carries no ABI declaration: both exports are
    # defined, with their TF_EXPORT markers, in the .cpp beside it.
    header_code = _cpp_code_only(_read(f"cpp/include/{K3_CPP_HEADER}"))
    assert "TF_EXPORT" not in header_code
    for name in INDEXING_EXPORTS:
        assert name not in header_code, name
    unit = _read("cpp/src/indexing.cpp")
    for name in INDEXING_EXPORTS:
        assert f"TF_EXPORT void {name}(" in unit, name
    assert _cpp_code_only(unit).count("TF_EXPORT") == len(INDEXING_EXPORTS)


# ===========================================================================
# 5. The versions have not moved
# ===========================================================================

def test_the_checkpoint_format_and_versions_are_unmoved():
    from tensorforge.experimental import native_checkpoint

    assert native_checkpoint._FORMAT == K0_CHECKPOINT_FORMAT
    assert native_checkpoint._FORMAT_VERSION == K0_CHECKPOINT_VERSION
    assert native_checkpoint._SUPPORTED_FORMAT_VERSIONS == \
        K0_CHECKPOINT_VERSIONS
    assert native_checkpoint._FLOAT64_ONLY_VERSIONS == (1, 2)


def test_the_optimizer_loader_and_sampler_state_versions_are_unmoved():
    from tensorforge.experimental import (native_data_loader,
                                          native_optimizer_state,
                                          native_sampler)

    assert native_optimizer_state.FORMAT_VERSION == \
        K0_OPTIMIZER_STATE_VERSION
    assert native_data_loader._FORMAT == K0_LOADER_FORMAT
    assert native_data_loader._FORMAT_VERSION == K0_LOADER_VERSION
    assert native_data_loader._SUPPORTED_FORMAT_VERSIONS == K0_LOADER_VERSIONS
    assert native_sampler._FORMAT == K0_SAMPLER_FORMAT
    assert native_sampler._FORMAT_VERSION == K0_SAMPLER_VERSION
    assert native_sampler._SUPPORTED_FORMAT_VERSIONS == K0_SAMPLER_VERSIONS


def test_no_future_version_constant_or_integer_field_was_reserved():
    """A reserved constant is a promise. K0 makes none."""
    from tensorforge.experimental import native_checkpoint

    assert 4 not in native_checkpoint._SUPPORTED_FORMAT_VERSIONS
    code = _code_only(_module_source(
        "src/tensorforge/experimental/native_checkpoint.py"))
    for banned in ("int64", "integer", "index_dtype", "_FORMAT_VERSION_4",
                   "_NEXT_FORMAT_VERSION"):
        assert banned not in code, banned


# ===========================================================================
# 6. The integer runtime is absent
# ===========================================================================
#
# Narrow on purpose. TensorForge legitimately has host ``int64`` class-label
# metadata, ``int64`` layout arrays, and an ``int64`` ctypes binding today;
# none is a native integer tensor, and a scan that rejected them would
# reject the code Phase K must not disturb.

def test_the_two_dtype_authorities_agree_again_after_k2():
    """The K1 asymmetry, closed exactly where the ladder said it would be.

    At K1 the C++ side **represented** three dtypes while the Python side
    **knew** two, and that gap was the whole of that milestone's safety
    argument: the representation existed so the barriers could be proved
    against it, and it was unreachable from Python because ``_DTYPE_CODES``
    had no entry. **K2 closed the gap in one commit**, together with
    ``INDEX_DTYPES`` and the public constructor, so the promise and the
    capability were never out of step in either direction.

    Both halves are asserted as exact sets, so a *fourth* dtype appearing on
    either side would fail here rather than be absorbed."""
    # The Python half: three now, with the tables in step with each other
    # and with the two registries.
    expected = set(K0_DTYPES) | set(K2_INDEX_DTYPES)
    assert set(cpp._DTYPE_CODES) == expected
    assert set(cpp._DTYPE_ITEM_SIZES) == expected
    assert set(cpp._DTYPE_NUMPY) == expected
    assert set(cpp._CHECKED_HOST_ARRAYS) == expected
    assert _dict_literal_keys("src/tensorforge/backends/cpp.py",
                              "_DTYPE_CODES") == expected
    assert set(cpp._DTYPE_CODES) == (set(cpp.SUPPORTED_DTYPES)
                                     | set(cpp.INDEX_DTYPES))
    assert cpp._DTYPE_CODES["int64"] == K2_ABI_INT64_CODE
    assert cpp._DTYPE_ITEM_SIZES["int64"] == K2_INT64_ITEM_SIZE

    # The C++ half: three, with the two floating codes exactly where they
    # were and int64 on the code the Phase-I comment reserved.
    header = _read("cpp/include/tf_internal.h")
    enum_body = header.split("enum TfDtype {", 1)[1].split("};", 1)[0]
    codes = dict(re.findall(r"TF_DTYPE_(\w+)\s*=\s*(\d+)", enum_body))
    assert {name: int(value) for name, value in codes.items()} == \
        K1_ABI_DTYPE_CODES
    scoped = header.split("enum class Dtype", 1)[1].split("};", 1)[0]
    assert set(re.findall(r"^\s*(\w+)\s*=", scoped, re.M)) == K1_SCOPED_DTYPES
    # ...and the role predicate exists, because the enumerator is only safe
    # to add if every float-only export can ask about it.
    assert "dtype_is_floating" in header
    assert "require_floating" in header


def test_no_generic_constructor_accepts_int64_at_any_milestone():
    """The property the whole contract is built around (design §5.4,
    §5.5), driven rather than described: **no existing generic constructor
    changed what it accepts, at any Phase-K milestone**.

    Under taxonomy B every one of these validates through
    ``normalize_dtype``, whose accepted set never moves, so there is no
    milestone at which one of them could have been narrowed and was not.
    The converting and uninitialized private routes are here for the same
    reason — each stays floating-only permanently, for a reason of its
    own — and the two routes K2 *did* open are deliberately absent from
    this list and asserted separately."""
    for build in (
        lambda: cpp.NativeStorage(4, dtype="int64"),
        lambda: cpp.NativeStorage.from_array([1, 2], dtype="int64"),
        lambda: cpp.NativeStorage._uninitialized(4, dtype="int64"),
        lambda: cpp.NativeStorage._typed_from_array([1, 2], "int64"),
        lambda: cpp.NativeTensorCore.from_array([1, 2], dtype="int64"),
        lambda: cpp.NativeTensorCore.zeros((2,), dtype="int64"),
        lambda: cpp.NativeTensorCore.zeros((2,), dtype="int64",
                                           _trusted_dtype=True),
        lambda: cpp.NativeTensorCore.full((2,), 1, dtype="int64"),
        lambda: cpp.NativeTensorCore._uninitialized((2,), dtype="int64"),
        lambda: cpp.NativeTensorCore._typed_from_array([1, 2], "int64"),
        lambda: cpp.NativeTensorCore._typed_full((2,), 1, "int64"),
    ):
        with pytest.raises(ValueError):
            build()


def test_no_integer_runtime_module_exists():
    experimental = REPO_ROOT / "src" / "tensorforge" / "experimental"
    modules = {path.name for path in experimental.glob("*.py")}
    for banned in ("native_int64", "native_integer", "native_index",
                   "native_argmax", "native_gather", "native_embedding",
                   "_native_index", "_native_integer"):
        assert not [name for name in modules if name.startswith(banned)], \
            banned


def test_exactly_one_public_integer_constructor_exists_and_no_index_operation():
    """The K2 public delta, in both directions and at all three layers.

    Present: ``from_int64_array``, ``item``, and ``tolist`` on
    ``NativeTensor``, plus the underscore-private ``_from_int64_array`` at
    the core and storage layers. Absent: any **public** integer constructor
    at either lower layer — which is what makes "one public door" literal —
    and every operation no Phase-K milestone has shipped."""
    tensor_methods = _defined_names(
        "src/tensorforge/experimental/native_tensor.py", "NativeTensor")
    core_methods = _defined_names("src/tensorforge/backends/cpp.py",
                                  "NativeTensorCore")
    storage_methods = _defined_names("src/tensorforge/backends/cpp.py",
                                     "NativeStorage")
    for name, milestone in LANDED_TENSOR_METHODS.items():
        assert name in tensor_methods, f"{name} landed at {milestone}"
        # ...and never on the storage layer, at any milestone.
        assert name not in storage_methods, (name, "NativeStorage")
        if name not in CORE_METHODS_BY_MILESTONE:
            # K2's three names are NativeTensor's alone: the Core's K2 row
            # reads "no public name", and its integer ingress stays private.
            assert name not in core_methods, (name, "NativeTensorCore")
    # K3's and K4's names are on **both** public layers, which is the row
    # §23.1 gives each and the one respect in which they differ from K2's.
    for name, milestone in CORE_METHODS_BY_MILESTONE.items():
        assert name in core_methods, f"{name} landed at {milestone}"
        assert name in tensor_methods, f"{name} landed at {milestone}"
        assert name not in storage_methods, (name, "NativeStorage")
    for name, milestone in PLANNED_TENSOR_METHODS.items():
        assert name not in tensor_methods, f"{name} belongs to {milestone}"
        assert name not in core_methods, f"{name} belongs to {milestone}"
    # The private ingress helpers exist, are private, and are at both
    # lower layers — the shape §8.1 requires.
    for names, layer in ((core_methods, "NativeTensorCore"),
                         (storage_methods, "NativeStorage")):
        assert "_from_int64_array" in names, layer
    for absent in ("int64", "as_int64", "to_int64"):
        assert absent not in storage_methods, absent
    # ...and the module-level factory shape for a later milestone is absent,
    # as is the ``max`` §17.10 permanently declines.
    module = _code_only(_module_source(
        "src/tensorforge/experimental/native_tensor.py"))
    for banned in ("gather", "scatter", "embedding", "argmin",
                   "max_with_indices"):
        assert banned not in module, banned
    for banned in ("max", "amax"):
        assert banned not in tensor_methods and banned not in core_methods, \
            banned


def test_no_casting_or_promotion_operation_exists():
    tensor_methods = _defined_names(
        "src/tensorforge/experimental/native_tensor.py", "NativeTensor")
    core_methods = _defined_names("src/tensorforge/backends/cpp.py",
                                  "NativeTensorCore")
    for banned in ("astype", "cast", "to", "type", "long", "int", "float",
                   "double", "promote", "as_type"):
        assert banned not in tensor_methods, banned
        assert banned not in core_methods, banned


def test_no_integer_kernel_exists_in_the_native_sources():
    """Read the C++ as code-ish text: the exports are the surface, and the
    templated kernels are named. A comment mentioning int64 metadata is
    legitimate and is not what this looks for.

    **The list shrank at K1, at K3, and again at K4, in one direction
    only.** ``Dtype::Int64``, ``TF_DTYPE_INT64``, and ``require_floating``
    moved from "absent" to "present" at K1, ``argmax`` did at K3, and
    ``index_select`` did at K4, because those milestones ship them — and
    each is asserted *present* below rather than merely no longer banned,
    the §37.2 rule that an entry moves between the two lists and nothing is
    loosened to let it. What stays banned is what stays absent: an integer
    **arithmetic** kernel, and the general gather, scatter, and embedding
    §18.1 keeps outside the phase. No integer addition, reduction, or
    comparison exists at any Phase-K milestone — nor does the ``max``
    §17.10 declines."""
    sources = {}
    for path in sorted((REPO_ROOT / "cpp" / "src").glob("*.cpp")):
        sources[path.name] = _cpp_code_only(
            path.read_text(encoding="utf-8"))
    for name, code in sources.items():
        for banned in ("embedding", "argmin", "max_with_indices",
                       "int64_add", "int64_sum", "int64_multiply"):
            assert banned not in code, (name, banned)
    # Both Phase-K operations live in exactly one translation unit, and
    # nowhere else.
    for operation in ("argmax", "index_select"):
        carriers = [name for name, code in sources.items()
                    if operation in code]
        assert carriers == list(K3_CPP_SOURCES), (operation, carriers)
    # ``gather`` and ``scatter`` are deliberately **not** banned: Conv2d has
    # carried ``conv2d_prefers_gather`` and a scatter-shaped backward since
    # Phase D, and neither is an index operation. Banning them would reject
    # code Phase K must not disturb — the narrowness this module exists for.
    assert "prefers_gather" in sources["conv2d.cpp"]
    # The present half, so the shrinking is proved to be a move rather than
    # a hole: the representation lives in storage.cpp, the transfers carry
    # it, and the barrier is applied in every compute unit.
    assert "Dtype::Int64" in sources["storage.cpp"]
    assert "std::int64_t" in sources["storage.cpp"]
    assert "Dtype::Int64" in sources["elementwise.cpp"]   # contiguous_copy
    for unit in ("elementwise.cpp", "matmul.cpp", "reduction.cpp",
                 "classification.cpp", "conv2d.cpp", "pooling.cpp",
                 "random.cpp", "storage.cpp", "indexing.cpp"):
        assert "require_floating" in sources[unit], unit
    # K3's export asks the **index-role** guard on its destination and asks
    # neither require_floating nor require_matching_dtype about it — either
    # would reject every valid call (design §22.8). K4's asks require_index
    # on its separate index handle and require_matching_dtype across its
    # floating source/destination pair, which is the one place in the phase
    # that guard is used (§22.9) — so the unit carries both, and the tests
    # that separate the two exports' bodies live in
    # tests/test_native_index_select.py.
    assert "require_index" in sources["indexing.cpp"]
    assert "require_matching_dtype" in sources["indexing.cpp"]
    guarded = set()
    for code in sources.values():
        guarded.update(re.findall(
            r'tf::require_floating\(\s*"(tf_[a-z0-9_]+)"', code))
    assert len(guarded) == K4_GUARDED_EXPORTS, sorted(guarded)
    assert set(LANDED_EXPORTS) <= guarded, sorted(LANDED_EXPORTS)
    assert len(guarded - set(LANDED_EXPORTS)) == K1_GUARDED_EXPORTS
    # ...and the four deliberately generalized transfer boundaries are NOT
    # guarded, because a floating-role guard there would refuse the
    # value-transfer primitive the phase is built on.
    for transfer in K1_GENERALIZED_TRANSFERS:
        assert transfer not in guarded, transfer


def test_the_existing_host_int64_metadata_is_untouched_and_still_legitimate():
    """The negative control for the absence scans above: the code they must
    **not** reject is proved to still be there. An absence scan that also
    deleted the class-label path would pass for the wrong reason."""
    module = _module_source("src/tensorforge/backends/cpp.py")
    assert "_CHECKED_I64_ARRAY" in module
    assert "_prepare_class_targets" in module
    assert "np.int64" in module
    dataset = _module_source("src/tensorforge/experimental/native_dataset.py")
    assert "target_batch" in dataset and "np.int64" in dataset


def test_cuda_and_amp_are_exactly_as_unsupported_as_they_were():
    assert cpp.UNSUPPORTED == K0_UNSUPPORTED
    for name in ("cuda", "amp"):
        assert name not in cpp.SUPPORTED_DEVICES
        assert name not in cpp.TENSOR_CORE_OPS
        assert name not in cpp.AUTOGRAD_OPS
        assert name not in cpp.NATIVE_MODULES


# ===========================================================================
# 7. Stable / native isolation
# ===========================================================================

def test_the_stable_public_api_is_unchanged_and_loads_no_native_library():
    import subprocess
    import sys

    # A fresh interpreter, so nothing this session already imported can
    # mask a stable-line import of the backend.
    program = (
        "import sys, tensorforge\n"
        "assert 'tensorforge.backends.cpp' not in sys.modules\n"
        "assert 'tensorforge.experimental' not in sys.modules\n"
        "from tensorforge import Tensor, nn, optim, data\n"
        "print(Tensor.__name__)\n"
    )
    result = subprocess.run([sys.executable, "-c", program],
                            capture_output=True, text=True,
                            cwd=str(REPO_ROOT))
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "Tensor"


def test_no_stable_module_imports_a_native_or_integer_module():
    stable = REPO_ROOT / "src" / "tensorforge"
    skipped = {"backends", "experimental"}
    for path in stable.rglob("*.py"):
        if set(path.relative_to(stable).parts) & skipped:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                assert "backends" not in node.module, path
                assert "experimental" not in node.module, path
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    assert "backends" not in alias.name, path
                    assert "experimental" not in alias.name, path


def test_the_design_keeps_the_isolation_rules_explicit():
    body = _section(_design(), 24)
    assert _missing(
        body, "tensorforge.Tensor", "unchanged", "serialization",
        "no implicit stable", "stable_framework_integration",
    ) == []


# ===========================================================================
# 8. The design resolves the architecture rather than listing options
# ===========================================================================
#
# Each entry is (section, [terms every one of which must appear in that
# section]). Section-scoped and combination-based: a document that merely
# contains one of these words somewhere passes nothing.

_REQUIRED_SECTIONS = {
    1: ("Phase K", "newly approved", "decided"),
    2: ("motivation", "index", "int64"),
    3: ("storage", "dispatch", "require_float64", "_DTYPE_CODES",
        "native_autograd_design", "native_abi_error_contract",
        "native_classification_design", "native_tensor_wrapper_design",
        "native_dtype_device_metadata_design"),
    4: ("floating compute dtype", "differentiable dtype", "index/result",
        "internally representable", "checkpoint-persistable"),
    5: ("SUPPORTED_DTYPES", "INDEX_DTYPES", "RAW_KERNEL_DTYPES",
        "Phase K adopts B", "remains the floating-compute registry",
        "NativeStorage.__init__", "NativeStorage.from_array",
        "NativeTensorCore.from_array", "NativeTensorCore.zeros",
        "NativeTensorCore.full", "NativeTensor.from_array",
        "normalize_module_dtype", "_validated_entry_dtype",
        "prohibited", "int32", "bool", "float16", "bfloat16", "deferred"),
    6: ("one extended NativeTensor", "separate public", "internal-only",
        "rejected", "integer autograd", "integer parameters",
        "optimizer ownership", "checkpoint"),
    7: ("storage owns", "views inherit", "trivially_destructible",
        "fresh owning contiguous"),
    8: ("from_int64_array", "numpy.ndarray", "byte order", "no dtype "
        "inference", "layout normalization", "to_numpy", "item", "tolist"),
    9: ("requires_grad", "before", "backward", "NativeParameter",
        "NativeSGD", "NativeAdam", "never receive"),
    10: ("parameters", "persistent buffers", "non-persistent",
         "prohibited", "deferred"),
    11: ("views and copies", "no integer addition", "argmax", "argmin",
         "index_select", "overflow"),
    12: ("no casting", "no promotion", "index operand is not an arithmetic "
         "operand", "bool"),
    13: ("two's-complement", "static_assert", "negative", "duplicate",
         "zero-sized", "deterministic"),
    14: ("negative indices reject", "rank exactly 1", "before the "
         "destination is allocated", "non-contiguous"),
    15: ("view", "copy", "narrow", "never called a view"),
    16: ("item", "tolist", "built-in int"),
    17: ("axis", "keepdims", "int64", "lowest", "NaN", "several NaNs",
         "signed zero", "no graph", "validation order",
         "max is not shipped", "row-major logical order",
         "increasing axis-index order"),
    18: ("index_select", "one axis", "bounds", "duplicate",
         "fresh owning contiguous", "never a view", "forward only",
         "deferred"),
    19: ("NativeTensor floating features", "numpy.int64 targets",
         "unchanged", "default-off", "deferred"),
    20: ("native_accuracy", "NativeCrossEntropyLoss", "host", "unchanged",
         "sequencing"),
    21: ("version 3", "(1, 2, 3)", "no checkpoint version change",
         "no version 4", "no version-4 constant", "metadata"),
    22: ("opaque handles", "ctypes", "no pybind11", "tf_core_argmax",
         "tf_core_index_select", "56", "require_floating",
         "require_matching_dtype", "exactly int64",
         "Validation order", "no index handle exists",
         "before the first destination element is written"),
    23: ("__all__", "25", "AUTOGRAD_OPS", "private"),
    24: ("stable", "unchanged", "stable_framework_integration"),
    25: ("CPU only", "CUDA", "AMP", "no device argument", "thread"),
    26: ("first error", "before allocation", "argmax", "index_select"),
    27: ("before allocation", "BaseException", "baseline", "distinct"),
    28: ("close()", "idempotent", "garbage collection", "caller closes"),
    29: ("std::int64_t", "8 bytes", "endian", "Windows", "Linux",
         "exact", "one-ULP"),
    30: ("negative control", "fingerprint", "AST", "no test starts a thread"),
    31: ("correctness is gated before timing", "no result file",
         "native_only", "never", "ratio"),
    32: ("K0", "K9", "closure", "window", "barrier", "reachability"),
    33: ("54", "56", "25", "SUPPORTED_DTYPES", "INDEX_DTYPES"),
    34: ("complete only when", "56", "25", "bit-identical", "ASan"),
    35: ("int32", "embedding", "scatter", "CUDA", "memory pool",
         "checkpoint version 4"),
    36: ("private comparative", "no code is copied", "independently "
         "justified"),
    37: ("negative control", "K0", "closure", "successor"),
}


@pytest.mark.parametrize("number", sorted(_REQUIRED_SECTIONS))
def test_every_required_design_topic_is_a_real_section(number):
    body = _section(_design(), number)
    assert len(body.strip()) > 200, (number, "section is a stub")
    missing = _missing(body, *_REQUIRED_SECTIONS[number])
    assert missing == [], (number, missing)


def test_the_design_has_no_gaps_in_its_section_numbering():
    numbers = [int(match) for match in
               re.findall(r"^## (\d+)\.", _design(), re.M)]
    assert numbers == sorted(numbers), numbers
    assert numbers == list(range(1, len(numbers) + 1)), numbers
    assert set(_REQUIRED_SECTIONS) <= set(numbers)


def test_the_object_model_decision_is_made_and_not_merely_surveyed():
    body = _flat(_section(_design(), 6))
    # All three candidates compared...
    assert _missing(body, "one extended NativeTensor",
                    "separate public integer", "internal-only") == []
    # ...one selected, in a sentence that says so.
    assert re.search(r"The decision.{0,80}one extended NativeTensor", body,
                     re.I | re.S), body[:600]
    # ...and each way the unified model could go wrong is answered with an
    # authority and a milestone rather than an assurance.
    for failure in ("Integer autograd", "Integer parameters",
                    "Optimizer ownership", "Mixed float/integer",
                    "Integer model state in a checkpoint"):
        assert failure.lower() in body.lower(), failure


def _barrier_rows():
    """(failure, milestone) for each row of the §6.5 barrier table."""
    rows = {}
    for line in _section(_design(), 6).splitlines():
        cells = [_flat(cell).strip() for cell in line.strip().strip("|").split("|")]
        if len(cells) == 3 and re.search(r"\bK\d\b", cells[2]):
            rows[cells[0]] = re.findall(r"\bK\d\b", cells[2])
    return rows


def test_the_barrier_table_parser_can_actually_fail():
    rows = _barrier_rows()
    assert rows, "the barrier table did not parse"
    assert any("autograd" in key.lower() for key in rows), sorted(rows)
    assert all(values for values in rows.values())


def test_every_barrier_lands_before_an_integer_tensor_can_exist():
    """The ordering invariant of design §32.1, checked structurally.

    The failure this exists for is a ladder that makes an integer tensor
    constructible in one milestone and gates a parameter, a buffer, an
    optimizer, a graph, a checkpoint, or a floating kernel in a later one.
    Every barrier's earliest milestone must strictly precede the first
    milestone at which construction is possible."""
    first = MILESTONES.index(FIRST_CONSTRUCTION_MILESTONE)
    for failure, milestones in _barrier_rows().items():
        earliest = min(MILESTONES.index(name) for name in milestones)
        assert earliest < first, (failure, milestones)
        assert MILESTONES[earliest] == BARRIER_MILESTONE, (failure, milestones)
    # ...and the table really does cover every reachability route.
    joined = _flat(_section(_design(), 6)).lower()
    for route in ("autograd", "parameter", "optimizer", "arithmetic",
                  "mixed float/integer", "buffer", "checkpoint",
                  "generic floating constructor"):
        assert route in joined, route


def test_the_window_proof_names_the_states_and_the_invariant():
    body = _flat(_section(_design(), 32))
    assert "32.1" in body
    for required in ("State after K0", "State after K1", "State after K2",
                     "no unsafe intermediate state"):
        assert required.lower() in body.lower(), required
    # The invariant is stated as an inequality over milestones, not as a
    # reassurance.
    assert re.search(r"milestone\(b\)\s*<\s*c", body), body[:400]


def test_the_taxonomy_choice_is_explicit_and_not_left_implicit():
    """Requirement: one taxonomy is chosen and named, not implied."""
    body = _flat(_section(_design(), 5))
    assert "Phase K adopts B" in body, body[:400]
    # Both candidates are stated before one is chosen.
    assert "SUPPORTED_DTYPES means every public native tensor dtype" in body
    assert "remains the floating-compute registry" in body
    # ...and the chosen one is spelled out in values.
    assert 'SUPPORTED_DTYPES stays ("float64", "float32") permanently' in body
    assert 'INDEX_DTYPES' in body


def test_the_design_resolves_every_affected_constructor_path():
    """Every path the correction named must appear in §5 with a resolution,
    so no generic constructor is left implicitly narrowed."""
    body = _flat(_section(_design(), 5))
    for path in ("NativeStorage.__init__", "NativeStorage.from_array",
                 "NativeStorage._typed", "NativeStorage._uninitialized",
                 "NativeStorage._typed_from_array",
                 "NativeTensorCore.from_array", "NativeTensorCore.zeros",
                 "NativeTensorCore.full", "NativeTensorCore._typed",
                 "NativeTensorCore._uninitialized",
                 "NativeTensorCore._typed_from_array",
                 "NativeTensorCore._typed_full",
                 "NativeTensor.from_array", "NativeTensor.from_int64_array",
                 "normalize_module_dtype", "_validated_entry_dtype",
                 "_narrowed_to_dtype"):
        assert path in body, path
    # The public storage question is answered, in the prohibiting direction.
    assert "prohibited" in body.lower()
    assert re.search(r"NativeStorage\(size, dtype=\"int64\"\)[^.]{0,60}"
                     r"prohibited", body, re.I | re.S), body[:600]


def test_the_ladder_cannot_move_int64_into_a_broad_registry():
    """`SUPPORTED_DTYPES` is identical in every row of the delta table, so
    no milestone can widen it — the structural half of taxonomy B."""
    supported = _delta_column("SUPPORTED_DTYPES")
    assert tuple(supported) == MILESTONES, tuple(supported)
    assert len(set(supported.values())) == 1, supported
    assert "int64" not in next(iter(supported.values()))
    # ...while the index registry appears exactly once, at K2, and never
    # before the first construction milestone.
    index = _delta_column("INDEX_DTYPES")
    for name in MILESTONES[:MILESTONES.index(FIRST_CONSTRUCTION_MILESTONE)]:
        assert "int64" not in index[name], (name, index[name])
    for name in MILESTONES[MILESTONES.index(FIRST_CONSTRUCTION_MILESTONE):]:
        assert "int64" in index[name], (name, index[name])


def test_the_argmax_nan_rule_is_exact_and_covers_every_case():
    """Ambiguous language such as "NaN propagates" is not a contract. The
    rule must state the traversal order and the exact returned index for
    every case, without this test freezing its prose."""
    body = _flat(_section(_design(), 17)).lower()
    for required in (
        "row-major logical order",          # full-reduction traversal
        "increasing axis-index order",      # axis-reduction traversal
        "several nans",                     # multiple NaNs
        "lowest",                           # first-occurrence / lowest index
        "+inf", "-inf",                     # NaN against either infinity
        "length 1",                         # degenerate run
        "identical",                        # contiguous vs non-contiguous
        "never inspects",                   # payload / signalling ignored
    ):
        assert required in body, required
    # The algorithm is given, not merely described.
    assert "best_index" in body and "isnan" in body
    # ...and each of the eight required cases has its own row.
    rows = [line for line in _section(_design(), 17).splitlines()
            if line.strip().startswith("|") and "NaN" in line]
    assert len(rows) >= 6, len(rows)


def test_the_initial_dtype_is_exactly_int64_and_the_others_are_deferred():
    body = _flat(_section(_design(), 5)).lower()
    assert "int64, and nothing else" in body
    for deferred in ("int32", "int16", "int8", "uint8", "bool", "complex",
                     "float16", "bfloat16"):
        assert deferred in body, deferred


def test_the_construction_contract_rejects_silent_casting():
    body = _flat(_section(_design(), 8)).lower()
    for rule in ("no dtype inference", "no numeric cast", "no truncation",
                 "no widening", "no reinterpretation"):
        assert rule in body, rule
    # ...and it distinguishes layout normalization from conversion.
    assert "layout normalization" in body
    assert "integer ingress converts nothing" in body


def _abi_contract(export):
    """The subsection of §22 that owns one export's self-validation."""
    body = _section(_design(), 22)
    start = body.index(f"Self-validation — `{export}`")
    following = re.search(r"\n### 22\.\d+", body[start:])
    return _flat(body[start:start + following.start()] if following
                 else body[start:])


def test_the_abi_contract_parser_separates_the_two_exports():
    """Negative control: the two contracts must be *different* bodies, so a
    single blanket paragraph cannot satisfy both."""
    argmax = _abi_contract("tf_core_argmax")
    select = _abi_contract("tf_core_index_select")
    assert argmax and select and argmax != select
    assert "tf_core_index_select" not in argmax.split("Required roles")[0]
    with pytest.raises(ValueError):
        _abi_contract("tf_core_not_an_export")


def test_the_argmax_abi_contract_has_its_own_roles_and_order():
    """`argmax` consumes floating and produces int64 **by design**, so a
    ``require_floating(destination)`` or a ``require_matching_dtype(source,
    destination)`` would reject every valid call. The contract must say so,
    and must not carry index validation it has no handle for."""
    body = _abi_contract("tf_core_argmax")
    lowered = body.lower()
    assert "require_floating" in body                  # on the source
    assert "exactly int64" in lowered                  # the destination role
    assert "alias" in lowered                          # aliasing rejected
    # The destination size is read from the **raw** section: ``_flat``
    # strips ``*``, which is the multiplication sign here.
    raw = _section(_design(), 22)
    assert "outer * inner" in raw                      # destination size
    # The two checks that must be *absent*, stated as absent.
    assert re.search(r"no\s+require_matching_dtype", lowered), body[:400]
    assert "never" in lowered and "require_floating" in lowered
    assert "no index handle exists" in lowered
    # ...and an ordered list that starts at null handles and ends at execute.
    assert "1. null handles" in body
    assert re.search(r"8\.\s*execute", body), body[-300:]


def test_the_index_select_abi_contract_has_its_own_roles_and_order():
    body = _abi_contract("tf_core_index_select")
    lowered = body.lower()
    assert "require_floating" in body
    assert "require_matching_dtype" in body            # source/destination only
    assert "exactly int64" in lowered                  # the index role
    assert "1. null handles" in body
    assert re.search(r"10\.\s*execute", body), body[-300:]
    # The bounds scan is complete and precedes every write.
    assert "before the first destination element is written" in lowered


def test_require_matching_dtype_never_crosses_a_role_boundary():
    """It may be used for the ``index_select`` floating source/destination
    pair and nowhere else — never across a floating/index boundary."""
    body = _flat(_section(_design(), 22)).lower()
    assert "used here and only here" in body
    assert "never applied across a floating/index role boundary" in body
    # The comparison table that makes the difference structural rather than
    # a matter of two paragraphs happening to differ.
    shared = _section(_design(), 22)
    assert "| `require_matching_dtype(src, dst)` |" in shared
    assert "| `require_floating(destination)` |" in shared


def test_only_native_tensor_gains_a_public_integer_constructor():
    """The Core and storage helpers are private; exactly one public name
    exists, and the public delta must not list the Core helper."""
    # Only the construction milestone's Core row — the later rows add
    # ``argmax`` and ``index_select``, which are legitimately public.
    core_rows = [
        line for line in _section(_design(), 23).splitlines()
        if line.strip().startswith(f"| {FIRST_CONSTRUCTION_MILESTONE} |")
        and "NativeTensorCore" in line
    ]
    assert core_rows, "the API plan has no construction-milestone Core row"
    for row in core_rows:
        flat = _flat(row)
        assert "_from_int64_array" in flat, flat
        assert "no public name" in flat.lower(), flat
    # The one public name is listed, on NativeTensor.
    api = _flat(_section(_design(), 23))
    assert "from_int64_array" in api
    # §5.4 and §8 agree: both lower helpers are underscore-private.
    for number in (5, 8):
        body = _flat(_section(_design(), number))
        assert "NativeStorage._from_int64_array" in body, number
        assert "NativeTensorCore._from_int64_array" in body, number
    # ...and nothing anywhere promises a *public* Core or storage door.
    whole = _flat(_design())
    for banned in ("NativeStorage.from_int64_array",
                   "NativeTensorCore.from_int64_array"):
        assert banned not in whole, banned


def test_the_design_makes_no_external_framework_nan_compatibility_claim():
    """K0 states a normative rule and claims parity with no other library —
    the unverified claim that was removed."""
    body = _flat(_section(_design(), 17))
    assert "TensorForge's normative rule" in body
    lowered = body.lower()
    assert "no compatibility claim" in lowered
    for banned in ("numpy.argmax", "what numpy does", "matches numpy",
                   "same as numpy", "agrees with numpy",
                   "is also what numpy"):
        assert banned not in lowered, banned


def test_the_abi_plan_states_a_maximum_and_a_per_milestone_delta():
    body = _flat(_section(_design(), 22))
    assert re.search(r"Total Phase-K ABI delta:\s*\+2", body), body[:400]
    assert re.search(r"Maximum:\s*56", body), body[:400]
    deltas = _flat(_section(_design(), 33))
    for milestone in MILESTONES:
        assert re.search(rf"\|\s*{milestone}\s*\|", deltas), milestone
    assert "56" in deltas and "54" in deltas


def test_the_phase_j_default_output_is_preserved_in_writing():
    body = _flat(_section(_design(), 19)).lower()
    assert "numpy.int64 targets" in body
    assert "no phase-k milestone modifies phase-j production code" in body


def test_the_exit_gate_and_non_goals_are_explicit():
    gate = _flat(_section(_design(), 34))
    assert re.search(r"complete only when", gate, re.I)
    assert len(re.findall(r"^\d+\.", _section(_design(), 34), re.M)) >= 15
    goals = _flat(_section(_design(), 35)).lower()
    for boundary in ("cuda", "amp", "embedding", "scatter", "memory pool",
                     "checkpoint version 4", "integer gradients"):
        assert boundary in goals, boundary


# ===========================================================================
# 9. The milestone ladder
# ===========================================================================

def _ladder_rows():
    """(identifier, cell) for each ``| **Kn** |`` row of the ladder table."""
    rows = {}
    for line in _section(_design(), 32).splitlines():
        match = re.match(r"\|\s*\*\*(K\d+)\*\*\s*\|(.*)\|\s*$", line)
        if match:
            rows[match.group(1)] = match.group(2)
    return rows


def test_the_ladder_parser_can_actually_fail():
    assert _ladder_rows(), "the ladder table did not parse"
    assert re.match(r"\|\s*\*\*(K\d+)\*\*\s*\|(.*)\|\s*$",
                    "| **K3** | Purpose | Unstarted. |")
    assert not re.match(r"\|\s*\*\*(K\d+)\*\*\s*\|(.*)\|\s*$",
                        "K3 | Purpose | Unstarted.")


def test_the_ladder_is_complete_unique_and_ordered():
    rows = _ladder_rows()
    assert tuple(rows) == MILESTONES, tuple(rows)
    assert len(set(rows)) == len(MILESTONES)


def test_every_ladder_row_carries_a_purpose_and_a_status():
    rows = _ladder_rows()
    for name, cell in rows.items():
        parts = [part.strip() for part in cell.split("|")]
        assert len(parts) >= 2, (name, cell)
        assert len(parts[0]) > 10, (name, "no purpose")
        status = _flat(parts[1]).lower()
        if name in COMPLETE_MILESTONES:
            assert "complete" in status, (name, status)
        else:
            assert "unstarted" in status, (name, status)


def test_every_zero_production_milestone_names_the_module_that_proves_it():
    """A milestone that ships no production code still has to be
    checkable, so each one names the module carrying its proof — and that
    module has to exist.

    The map is the honest place for the distinction the inventory rows
    depend on: "this milestone landed" and "this milestone changed the
    package" are different facts, and every count in this module measures
    the second."""
    rows = _ladder_rows()
    for milestone, module in ZERO_PRODUCTION_MILESTONES.items():
        assert milestone in COMPLETE_MILESTONES, milestone
        assert (REPO_ROOT / module).is_file(), module
        record = _flat(_milestone_record(milestone)).lower()
        # Both honest spellings the two records use — "zero runtime" for a
        # milestone that predated the runtime, "zero production code" for
        # one that adds none beside a runtime that exists — and nothing
        # vaguer than either.
        assert re.search(r"\b(zero|no)\s+(production\s+(code|source)"
                         r"|runtime\b)", record), milestone
        # The ladder row says the same thing in its own words.
        assert "complete" in _flat(rows[milestone]).lower(), milestone
    # Each row names its module, which is what makes the claim checkable.
    for milestone in ("K5", "K6", "K7"):
        assert ZERO_PRODUCTION_MILESTONES[milestone] in _flat(
            _milestone_record(milestone)), milestone
    # ...and the negative control: a milestone that *did* change the
    # package is not in the map, so the map is a claim rather than a list
    # of everything — and the phrase check really can fail.
    for shipped in ("K1", "K2", "K3", "K4"):
        assert shipped not in ZERO_PRODUCTION_MILESTONES, shipped
    pattern = re.compile(r"\b(zero|no)\s+(production\s+(code|source)"
                         r"|runtime\b)")
    assert pattern.search("this milestone adds zero production code")
    assert pattern.search("k0 adds no runtime behavior at all")
    assert not pattern.search("this milestone adds one export and a ctest")


def test_the_compatibility_module_is_k5s_and_adds_no_production_code():
    """K5's whole deliverable, asserted where a reader can find it: the
    module exists, the design assigns it to K5 in the ownership table, and
    the milestone's inventory row moves nothing at all."""
    module = ZERO_PRODUCTION_MILESTONES["K5"]
    assert (REPO_ROOT / module).is_file(), module
    ownership = _flat(_section(_design(), 30))
    assert "test_native_integer_compatibility.py" in ownership
    assert re.search(r"test_native_integer_compatibility\.py[^|]{0,40}K5",
                     ownership), ownership[:400]
    # Every column of K5's delta row is unchanged from K4's.
    for column, expected in (("C ABI", str(PHASE_K_MAX_EXPORTS)),
                             ("CTests", str(K4_CTEST_COUNT)),
                             ("Examples", str(K0_EXAMPLE_COUNT)),
                             ("Benchmarks", str(K0_BENCHMARK_COUNT))):
        cells = _delta_column(column)
        assert cells["K5"] == expected, (column, cells["K5"])
        assert cells["K5"] == cells["K4"], column
    # ...and it promises no public Python name.
    public = _delta_column("Public Python")
    assert not re.search(r"\bfrom_int64_array\b|\bargmax\b|\bindex_select\b",
                         public["K5"]), public["K5"]


def test_the_example_module_is_k6s_and_moves_only_the_example_inventory():
    """K6's whole deliverable, asserted where a reader can find it: the
    example and its owner exist, the design assigns the owner to K6 in the
    ownership table, and the milestone's delta row moves the **example**
    column and nothing else."""
    module = ZERO_PRODUCTION_MILESTONES["K6"]
    assert (REPO_ROOT / module).is_file(), module
    for name in K6_EXAMPLES:
        assert (REPO_ROOT / "examples" / name).is_file(), name
    ownership = _flat(_section(_design(), 30))
    assert "test_native_integer_indexing_example.py" in ownership
    assert re.search(r"test_native_integer_indexing_example\.py[^|]{0,40}K6",
                     ownership), ownership[:400]
    # Every column of K6's delta row is K5's, except Examples.
    for column, expected in (("C ABI", str(PHASE_K_MAX_EXPORTS)),
                             ("CTests", str(K4_CTEST_COUNT)),
                             ("Benchmarks", str(K0_BENCHMARK_COUNT))):
        cells = _delta_column(column)
        assert cells["K6"] == expected, (column, cells["K6"])
        assert cells["K6"] == cells["K5"], column
    examples = _delta_column("Examples")
    assert examples["K5"] == str(K0_EXAMPLE_COUNT), examples["K5"]
    assert examples["K6"] == str(K6_EXAMPLE_COUNT), examples["K6"]
    # ...and it promises no public Python name.
    public = _delta_column("Public Python")
    assert not re.search(r"\bfrom_int64_array\b|\bargmax\b|\bindex_select\b",
                         public["K6"]), public["K6"]


def test_the_hardening_module_is_k7s_and_moves_no_inventory_at_all():
    """K7's whole deliverable, asserted where a reader can find it: the
    module exists, the design assigns it to K7 in the ownership table, and
    every column of its delta row is K6's.

    K7 is the second milestone in a row to add **no** production code, and
    the third overall — so unlike K6 it does not even move the example
    count, which is what makes "every column is the previous one's" the
    right check rather than an approximation of one."""
    module = ZERO_PRODUCTION_MILESTONES["K7"]
    assert (REPO_ROOT / module).is_file(), module
    ownership = _flat(_section(_design(), 30))
    assert "test_native_integer_hardening.py" in ownership
    assert re.search(r"test_native_integer_hardening\.py[^|]{0,40}K7",
                     ownership), ownership[:400]
    for column, expected in (("C ABI", str(PHASE_K_MAX_EXPORTS)),
                             ("CTests", str(K4_CTEST_COUNT)),
                             ("Examples", str(K6_EXAMPLE_COUNT)),
                             ("Benchmarks", str(K0_BENCHMARK_COUNT))):
        cells = _delta_column(column)
        assert cells["K7"] == expected, (column, cells["K7"])
        assert cells["K7"] == cells["K6"], column
    # ...and it promises no public Python name.
    public = _delta_column("Public Python")
    assert not re.search(r"\bfrom_int64_array\b|\bargmax\b|\bindex_select\b",
                         public["K7"]), public["K7"]
    # The K8/K9 artifacts stay absent while K7 is the newest milestone.
    for absent in ("benchmarks/benchmark_native_integer.py",
                   "tests/test_native_integer_benchmark.py",
                   "tests/test_native_phase_k_closure.py"):
        assert not (REPO_ROOT / absent).exists(), absent


def test_the_ladder_has_a_closure_milestone_and_no_successor_promise():
    last = MILESTONES[-1]
    assert "closure" in _ladder_rows()[last].lower()
    design = _flat(_design())
    assert not re.search(r"\bK11\b", design), "the ladder invents a K11"
    assert not re.search(r"\bPhase L\b", design, re.I), (
        "the design names a phase after K"
    )


def _delta_column(header_term):
    """The per-milestone cells of one column of the delta table, keyed by
    milestone and located by its **header**, so the check reads the column
    it means rather than a position that could shift."""
    lines = [line for line in _section(_design(), 33).splitlines()
             if line.lstrip().startswith("|")]
    headers = [_flat(cell).strip() for cell in lines[0].strip("|").split("|")]
    index = next(position for position, name in enumerate(headers)
                 if header_term in name)
    cells = {}
    for line in lines[1:]:
        parts = [_flat(cell).strip() for cell in line.strip().strip("|").split("|")]
        if parts and re.fullmatch(r"K\d+", parts[0]):
            cells[parts[0]] = parts[index]
    return cells


def test_the_delta_column_reader_can_actually_fail():
    exports = _delta_column("C ABI")
    assert tuple(exports) == MILESTONES, tuple(exports)
    assert exports["K0"] == str(K0_EXPORT_COUNT)
    assert exports[MILESTONES[-1]] == str(PHASE_K_MAX_EXPORTS)
    with pytest.raises(StopIteration):
        _delta_column("a column this table does not have")


def test_no_public_promise_moves_before_its_proof():
    """Under taxonomy B the strongest form of this rule is available: the
    broad registry never moves at all, and the one registry that does
    appears in the same milestone as the public constructor it promises."""
    supported = _delta_column("SUPPORTED_DTYPES")
    assert len(set(supported.values())) == 1, supported
    public = _delta_column("Public Python")
    for name in MILESTONES[:MILESTONES.index(FIRST_CONSTRUCTION_MILESTONE)]:
        assert "int64" not in public[name], (name, public[name])
    assert "from_int64_array" in public[FIRST_CONSTRUCTION_MILESTONE]
    body = _flat(_section(_design(), 33))
    assert "No public promise moves before its proof" in body


# ===========================================================================
# 10. No surface over-claims
# ===========================================================================

@pytest.mark.parametrize("surface", STATUS_SURFACES +
                         (f"docs/{PHASE_K_DESIGN_NAME}",))
def test_no_surface_claims_an_integer_runtime_exists(surface):
    offenders = _overclaims(_read(surface))
    assert offenders == [], (surface, offenders[:3])


@pytest.mark.parametrize("surface", STATUS_SURFACES)
def test_every_status_surface_places_phase_k_after_a_complete_phase_j(surface):
    text = _flat(_read(surface))
    assert "Phase J" in text, f"{surface} does not name Phase J"
    assert re.search(r"Phase J[^.]{0,60}\b(is|was)\s+complete"
                     r"|Phase J[^.]{0,60}complete\b", text, re.I), surface
    assert "Phase K" in text, f"{surface} does not name Phase K"
    assert re.search(r"Phase K[^.]{0,80}newly approved", text, re.I), surface


@pytest.mark.parametrize("surface", STATUS_SURFACES)
def test_every_status_surface_says_k0_is_architecture_only(surface):
    text = _flat(_read(surface)).lower()
    assert "k0" in text, surface
    assert re.search(r"k0[^.]{0,120}(architecture|no runtime|zero.runtime)",
                     text), surface


# ===========================================================================
# 10a. Current-phase reconciliation — and the exception is gone
# ===========================================================================
#
# The failure this section exists for: a repository where Phase K has
# opened but a status surface still calls Phase J "the latest phase", so a
# reader cannot tell which phase is current. K0 could not repair
# ``src/tensorforge/experimental/__init__.py`` — that is production source,
# and K0 changed none — so exactly one surface was allowed to lag, with the
# repair assigned to K1 in writing.
#
# **K1 performed that repair, so the exemption is removed rather than
# retained.** The production module is now an ordinary editable status
# surface, held to the identical rule as every other one, and
# ``test_no_stale_latest_phase_exemption_survives`` proves that no scoped
# exception is left anywhere — an exemption that outlived its reason is
# exactly the shape of a guardrail that stopped guarding.

# Editable surfaces that state *current* status. Every one of these must
# name K as latest and J as latest completed.
EDITABLE_STATUS_SURFACES = (
    "README.md",
    "CLAUDE.md",
    "docs/roadmap.md",
    "docs/project_summary.md",
    "docs/native_support_matrix.md",
    "docs/architecture.md",
    "docs/backend_experiments.md",
    # Repaired at K1, the first Phase-K milestone that edits the package.
    "src/tensorforge/experimental/__init__.py",
)

# The surface K0 could not touch, and the milestone that repaired it.
REPAIRED_PRODUCTION_SURFACE = "src/tensorforge/experimental/__init__.py"
STALE_REPAIR_MILESTONE = "K1"

# Both orders of each claim: the sentence form ("Phase K … is the current
# phase") and the heading form ("The current phase — Phase K"). A document
# can carry either, and a guardrail that knew only one would accept a stale
# heading sitting over corrected body prose.
_LATEST_PHASE_FORM = re.compile(
    r"Phase ([A-K])\b[^.;]{0,80}?\bis the (?:latest|current) phase\b"
    r"|The (?:latest|current) (?:native )?phase [—-] Phase ([A-K])\b", re.I)
_LATEST_COMPLETED_FORM = re.compile(
    r"Phase ([A-K])\b[^.;]{0,60}?\bis the latest completed\b"
    r"|latest completed (?:native )?phase is Phase ([A-K])\b"
    r"|The latest completed (?:native )?phase [—-] Phase ([A-K])\b", re.I)


def _phase_letters(pattern, text):
    letters = set()
    for match in pattern.finditer(text):
        letters.update(group.upper() for group in match.groups() if group)
    return letters


def test_the_latest_phase_forms_can_actually_fail():
    """Negative control for both forms, on temporary strings."""
    assert _phase_letters(_LATEST_PHASE_FORM,
                          "Phase J is the latest phase") == {"J"}
    assert _phase_letters(_LATEST_PHASE_FORM,
                          "Phase K is the latest phase") == {"K"}
    assert _phase_letters(_LATEST_PHASE_FORM, "Phase K is complete") == set()
    assert _phase_letters(_LATEST_COMPLETED_FORM,
                          "Phase J is the latest completed phase") == {"J"}
    assert _phase_letters(_LATEST_COMPLETED_FORM,
                          "the latest completed phase is Phase I") == {"I"}
    assert _phase_letters(_LATEST_COMPLETED_FORM,
                          "Phase J is the latest phase") == set()


# The landed/unstarted claim every editable surface must carry, derived
# from the ladder split above rather than written out, so a milestone
# landing moves one tuple and the wording follows.
_LANDED_CLAIM = re.compile(
    rf"only {COMPLETE_MILESTONES[0]} through {COMPLETE_MILESTONES[-1]} "
    rf"have landed", re.I)
_UNSTARTED_CLAIM = re.compile(
    rf"{UNSTARTED_MILESTONES[0]} through {UNSTARTED_MILESTONES[-1]} are "
    rf"unstarted", re.I)


def test_the_landed_and_unstarted_claim_forms_can_actually_fail():
    """Negative controls for both, on temporary strings."""
    assert _LANDED_CLAIM.search("only K0 through K7 have landed")
    assert not _LANDED_CLAIM.search("only K0 through K6 have landed")
    assert _UNSTARTED_CLAIM.search("K8 through K9 are unstarted")
    assert not _UNSTARTED_CLAIM.search("K7 through K9 are unstarted")


@pytest.mark.parametrize("surface", EDITABLE_STATUS_SURFACES)
def test_every_editable_status_surface_names_k_as_the_current_phase(surface):
    text = _flat(_read(surface))
    assert _phase_letters(_LATEST_PHASE_FORM, text) == {"K"}, surface
    assert _LANDED_CLAIM.search(text), surface
    assert _UNSTARTED_CLAIM.search(text), surface


@pytest.mark.parametrize("surface", EDITABLE_STATUS_SURFACES)
def test_every_editable_status_surface_names_j_as_latest_completed(surface):
    text = _flat(_read(surface))
    assert _phase_letters(_LATEST_COMPLETED_FORM, text) == {"J"}, surface


@pytest.mark.parametrize("surface", EDITABLE_STATUS_SURFACES)
def test_no_editable_status_surface_calls_j_the_latest_phase(surface):
    """The stale claim itself, banned everywhere it can be repaired."""
    named = _phase_letters(_LATEST_PHASE_FORM, _flat(_read(surface)))
    assert named <= {"K"}, (surface, sorted(named))


def test_the_production_docstring_was_repaired_at_k1():
    """The repair K1 owns, asserted directly on the file it names.

    This is the same check the K0 exemption existed to defer, inverted:
    the module must now name Phase K as current and Phase J as the latest
    completed one, and it must not still say the thing the exemption
    covered."""
    text = _flat(_read(REPAIRED_PRODUCTION_SURFACE))
    assert _phase_letters(_LATEST_PHASE_FORM, text) == {"K"}, text[:200]
    assert _phase_letters(_LATEST_COMPLETED_FORM, text) == {"J"}
    assert not re.search(r"Phase J[^.;]{0,80}?\bis the latest phase\b", text,
                         re.I)
    # ...and it records what each landed milestone actually did, including
    # the halves a reader would otherwise get wrong.
    assert re.search(r"K1[^.]{0,200}(barrier|representation)", text, re.I)
    assert re.search(r"\bK2\b.{0,1500}from_int64_array", text, re.I), \
        "the module does not record K2's one public door"
    assert re.search(r"\bK3\b.{0,1500}argmax", text, re.I), \
        "the module does not record K3's one operation"
    assert "INDEX_DTYPES" in text
    # The absence half, which is what keeps "an integer tensor exists" from
    # being read as "int64 is supported", and what keeps "an argmax exists"
    # from being read as "a max exists".
    assert re.search(r"not a supported native tensor dtype", text, re.I)
    # ``_flat`` strips backticks, so the needles are the flattened forms.
    for stated in ("argmax", "integer arithmetic", "no max"):
        assert stated.lower() in text.lower(), stated


def test_no_production_module_carries_a_stale_latest_phase_claim():
    """Every module in the package, with **no** file exempt."""
    package = REPO_ROOT / "src" / "tensorforge"
    offenders = []
    for path in package.rglob("*.py"):
        if "__pycache__" in path.parts:
            continue
        named = _phase_letters(_LATEST_PHASE_FORM,
                               _flat(path.read_text(encoding="utf-8")))
        if named - {"K"}:
            offenders.append((path.relative_to(REPO_ROOT).as_posix(),
                              sorted(named)))
    assert offenders == [], offenders


def test_no_stale_latest_phase_exemption_survives():
    """The exemption itself is gone, in both places that carried it.

    An exemption that outlives its reason is a guardrail that has stopped
    guarding, so its **removal** is asserted rather than assumed: the
    scoped surface set in ``tests/test_docs.py`` must no longer exist, and
    no status surface may be skipped by this module either."""
    docs_source = _read("tests/test_docs.py")
    assert "_STALE_LATEST_PHASE_SURFACES" not in docs_source, (
        "the K0-only stale-docstring exemption is still in test_docs.py"
    )
    # The production module is now an ordinary editable status surface.
    assert REPAIRED_PRODUCTION_SURFACE in EDITABLE_STATUS_SURFACES
    # ...and test_docs.py checks it like every other surface.
    assert REPAIRED_PRODUCTION_SURFACE in docs_source


def test_the_design_records_the_production_docstring_repair_as_k1_work():
    """The repair was only defensible because a milestone owned it in
    writing. The design must still name the exact file, the reason K0 could
    not touch it, and the milestone that did — the record of *why* the
    exemption was legitimate outlives the exemption."""
    ladder = _design()
    start = ladder.index(f"### {STALE_REPAIR_MILESTONE} — ")
    following = re.search(r"\n### K\d+ — ", ladder[start + 5:])
    body = _flat(ladder[start:start + 5 + following.start()] if following
                 else ladder[start:])
    assert REPAIRED_PRODUCTION_SURFACE in body, body[:200]
    assert "latest phase" in body
    assert "no production source at all" in body
    # ...and it records that the exemption was **removed** rather than
    # carried forward, naming the file that held it.
    assert "tests/test_docs.py" in body
    assert re.search(r"removed rather than retained", body), body[:400]


# ===========================================================================
# 11. No external-project provenance reference
# ===========================================================================

def test_no_repository_text_carries_external_provenance():
    """The rule is unconditional: no external project's name, owner, URL,
    path, or permission discussion appears anywhere in this repository.

    The sweep is **exhaustive** rather than extension-filtered, and **no
    file is exempt — this module included**. Every banned token is stored
    encoded and every control string is decoded at runtime, so this module
    contains none of the text it scans for and needs no self-exemption."""
    offenders = []
    for path, text in _repository_text_files():
        hits = _provenance_hits(text)
        if hits:
            offenders.append((str(path.relative_to(REPO_ROOT)), hits[:2]))
    assert offenders == [], offenders


def test_the_provenance_sweep_is_exhaustive_and_not_extension_filtered():
    """Non-vacuity for the sweep itself.

    "No offenders" is only meaningful if the sweep actually read the tree,
    so the file set is checked for a build file, a configuration/workflow
    file, an extensionless top-level text file, the two root documents, the
    design, two production sources, and **this module**."""
    scanned = {path.relative_to(REPO_ROOT).as_posix()
               for path, _ in _repository_text_files()}
    names = {name.rsplit("/", 1)[-1] for name in scanned}
    for required in ("CMakeLists.txt", "README.md", "CLAUDE.md",
                     PHASE_K_DESIGN_NAME, "cpp.py", "storage.cpp",
                     Path(__file__).name):
        assert required in names, required
    # A configuration or workflow file, whichever this checkout carries.
    assert any(name.endswith((".yml", ".yaml", ".json", ".cfg", ".ini",
                              ".cmake", ".sh", ".ps1", ".toml"))
               for name in names), sorted(names)[:20]
    # An extensionless top-level text file (LICENSE, .gitignore, and the
    # like) — exactly what an extension allow-list would have dropped.
    assert any("." not in name or name.startswith(".") for name in names), (
        "the sweep found no extensionless file"
    )
    # ...and the skip list really is only directories, not a file filter.
    assert all(not (_SKIPPED_DIRECTORIES & set(name.split("/")[:-1]))
               for name in scanned)


def test_the_prohibited_token_digests_are_full_sha256_values():
    """A truncated digest is a weaker check than it looks. Every committed
    digest is the full 64-character hex value, and each one really does
    verify its own decoded token."""
    import hashlib as _hashlib

    assert len(_ENCODED_BANNED) == 6, len(_ENCODED_BANNED)
    for encoded, digest in _ENCODED_BANNED:
        assert len(digest) == 64, (encoded, len(digest))
        assert set(digest) <= set("0123456789abcdef"), digest
        token = _decode(encoded)
        assert _hashlib.sha256(token.encode()).hexdigest() == digest
    # The decoder shifts **down** one codepoint, which the comment states.
    assert _decode("bcd") == "abc"
    # The complete repository URL is one of the exact tokens in its own
    # right, not merely an owner/repo substring caught inside a URL.
    assert any(token.startswith("http") and token.count("/") >= 4
               for token in BANNED_TOKENS), BANNED_TOKENS


def test_the_design_states_the_external_reference_policy_generically():
    body = _flat(_section(_design(), 36))
    assert _missing(body, "private comparative", "No code is copied",
                    "independently justified") == []
    # The policy section must not itself name anything.
    assert _provenance_hits(body) == [], body[:400]


def test_the_sweep_only_suppresses_a_utf8_decoding_failure():
    """Source-level guardrail: `_repository_text_files` must classify a file
    as non-text **only** when UTF-8 decoding proves it, and must let every
    other error propagate.

    Suppressing ``OSError`` would let a repository-owned file the sweep
    cannot open pass silently as "nothing found" — the exact shape of a
    scanner that cannot fail. Read from the AST rather than by substring, so
    a comment mentioning ``OSError`` cannot satisfy it."""
    source = inspect.getsource(_repository_text_files)
    tree = ast.parse(textwrap.dedent(source))
    handlers = [handler for node in ast.walk(tree)
                if isinstance(node, ast.Try) for handler in node.handlers]
    assert handlers, "the sweep has no exception handler at all"
    caught = set()
    for handler in handlers:
        assert handler.type is not None, "a bare except would hide everything"
        names = (handler.type.elts if isinstance(handler.type, ast.Tuple)
                 else [handler.type])
        for name in names:
            assert isinstance(name, ast.Name), ast.dump(name)
            caught.add(name.id)
    assert caught == {"UnicodeDecodeError"}, sorted(caught)
    # ...and the docstring says exactly that, rather than "unreadable".
    assert "proves it is not text" in (_repository_text_files.__doc__ or "")
    assert "unreadable" not in (_repository_text_files.__doc__ or "").lower()


def test_the_roadmap_headings_match_the_corrected_body():
    """A stale heading over corrected prose is still a wrong document.

    The failure this prevents: the roadmap body says Phase K is current
    while the section headings still announce Phase J as *the latest phase*
    and Phase I as *the latest completed phase*."""
    headings = [line.strip() for line in _read("docs/roadmap.md").splitlines()
                if line.startswith("## ")]
    joined = " | ".join(headings)
    assert any(re.fullmatch(r"## The current phase [—-] Phase K.*", h)
               for h in headings), joined
    assert any(re.fullmatch(r"## The latest completed phase [—-] Phase J.*", h)
               for h in headings), joined
    # The two stale forms must be gone, in either direction.
    for stale in (r"## The latest phase [—-] Phase J",
                  r"## The latest completed phase [—-] Phase I"):
        assert not [h for h in headings if re.match(stale, h)], (stale, joined)
    # No heading may name a phase later than K as current or completed.
    for heading in headings:
        for letter in re.findall(r"Phase ([A-Z])\b", heading):
            assert letter <= "K", heading


def test_the_roadmap_current_status_paragraph_is_accurate():
    """The introductory status must state all five current facts."""
    text = _flat(_read("docs/roadmap.md"))
    assert re.search(r"Python line is (?:\*\*)?complete at v3\.0", text, re.I)
    assert re.search(r"Phases A through J", text)
    assert re.search(r"Phase K[^.;]{0,90}?(current|latest) phase", text, re.I)
    assert _LANDED_CLAIM.search(text)
    assert re.search(r"Phase J is the latest completed phase", text, re.I)
    # ...and the Phase-K section states the absence half.
    assert _UNSTARTED_CLAIM.search(text)
    assert re.search(r"design, documentation, and guardrails only", text, re.I)
    # The presence half K3 and K4 earned, and the absence half neither
    # touched.
    assert re.search(r"native\s+`?argmax`?", text, re.I)
    assert re.search(r"`?index_select`?", text, re.I)
    assert re.search(r"no general\s+`?gather`?", text, re.I)
    assert re.search(r"no\s+`?max`?\b", text, re.I)
    assert re.search(r"not a supported native tensor dtype", text, re.I)
    assert f"docs/{PHASE_K_DESIGN_NAME}" in _read("docs/roadmap.md") or \
        PHASE_K_DESIGN_NAME in _read("docs/roadmap.md")


def test_the_roadmap_heading_scanner_can_actually_fail():
    """Negative control for the heading test, on temporary strings."""
    stale = ["## The latest phase — Phase J, complete",
             "## The latest completed phase — Phase I, complete"]
    assert [h for h in stale
            if re.match(r"## The latest phase [—-] Phase J", h)]
    assert [h for h in stale
            if re.match(r"## The latest completed phase [—-] Phase I", h)]
    fresh = ["## The current phase — Phase K, K0 complete",
             "## The latest completed phase — Phase J, complete"]
    assert any(re.fullmatch(r"## The current phase [—-] Phase K.*", h)
               for h in fresh)
    assert not [h for h in fresh
                if re.match(r"## The latest phase [—-] Phase J", h)]


def test_the_k2_summary_does_not_call_all_python_construction_private():
    """K2 ships a **public** constructor over **private** helpers. A summary
    that calls all K2 Python construction private contradicts the milestone
    it summarizes, which is exactly the sentence this replaces."""
    body = _flat(_section(_design(), 33))
    # The corrected reachability sentence, in both halves.
    assert "raw private C ABI at K1" in body, body[:400]
    assert "public NativeTensor.from_int64_array constructor" in body, body[:400]
    assert "backed by private Storage/Core helpers at K2" in body, body[:400]
    # ...and the contradiction is absent.
    lowered = body.lower()
    for banned in ("only through private python constructors",
                   "private python constructors at k2"):
        assert banned not in lowered, banned
    # The delta table and the ladder agree that K2's public name exists.
    public = _delta_column("Public Python")
    assert "from_int64_array" in public[FIRST_CONSTRUCTION_MILESTONE]


# ===========================================================================
# 12. The K2 public delta — three public names, exactly one public door
# ===========================================================================
#
# The two claims are different and both are true, so neither may be written
# as the other:
#
#   * K2 adds **three** public ``NativeTensor`` method names —
#     ``from_int64_array``, ``item``, and ``tolist``;
#   * ``NativeTensor.from_int64_array`` is the **only public construction
#     or host-ingress door** — the one public API through which an
#     ``int64`` buffer can come into existence.
#
# ``item`` and ``tolist`` are dtype-general host *inspection*: they
# construct nothing, so they are names in the delta and are not doors.
# Collapsing the delta into "one public name" understates it, and it is
# what these guardrails exist to catch.

# A quantified claim about a public *name*. Deliberately narrow: "no public
# name" (K1's and K5–K9's honest rows) does not match, and neither does
# "no new public experimental name" — only a claim that some **single**
# public name is the whole of a delta.
_SINGULAR_PUBLIC_NAME = re.compile(
    r"\b(only|one|single|sole|exactly one|just one)\s+(new\s+)?public\s+"
    r"(python\s+)?names?\b", re.I)

# A *public* integer constructor at a layer that must only have a private
# one. The leading boundary keeps ``NativeStorage._from_int64_array`` —
# the real, private helper — from matching its own name.
_PUBLIC_LOWER_DOOR = re.compile(
    r"\b(NativeStorage|NativeTensorCore)\.from_int64_array\b")

# Public method names that would be a second construction door.
_DOOR_MARKERS = ("_from_int64_array", "_normalize_index_dtype")


def _public_integer_doors():
    """Every **public** method, at any of the three layers, that reaches
    the integer construction machinery.

    Read from the AST: a method is a door when its body calls the private
    integer ingress or the index registry gate. ``NativeStorage.copy_from``
    is deliberately not a door — it writes into storage that only a door
    could have created — so ``_exact_host_array`` is not a marker."""
    doors = set()
    for relative, class_names in (
        ("src/tensorforge/experimental/native_tensor.py", ("NativeTensor",)),
        ("src/tensorforge/backends/cpp.py",
         ("NativeTensorCore", "NativeStorage")),
    ):
        tree = ast.parse(_module_source(relative))
        for node in ast.walk(tree):
            if not isinstance(node, ast.ClassDef) or node.name not in class_names:
                continue
            for child in node.body:
                if not isinstance(child, (ast.FunctionDef,
                                          ast.AsyncFunctionDef)):
                    continue
                if child.name.startswith("_"):
                    continue
                for call in ast.walk(child):
                    if not isinstance(call, ast.Call):
                        continue
                    func = call.func
                    name = (func.attr if isinstance(func, ast.Attribute)
                            else func.id if isinstance(func, ast.Name)
                            else None)
                    if name in _DOOR_MARKERS:
                        doors.add(f"{node.name}.{child.name}")
    return doors


def test_the_public_delta_scanners_can_actually_fail():
    """Negative controls, on temporary strings, for all three arms."""
    # Arm 1: the singular-name claim is caught in every ordinary shape...
    for offender in ("It is the only public name K2 adds",
                     "K2 adds exactly one public name",
                     "the single public Python name",
                     "just one new public name"):
        assert _SINGULAR_PUBLIC_NAME.search(offender), offender
    # ...and the honest rows are not.
    for honest in ("K1 | none | no public name",
                   "Phase K adds no new public experimental name",
                   "K2 adds three public NativeTensor method names",
                   "the one public construction door",
                   "the only public construction or host-ingress door"):
        assert not _SINGULAR_PUBLIC_NAME.search(honest), honest
    # Arm 2: a public lower-layer door is caught, the private one is not.
    assert _PUBLIC_LOWER_DOOR.search("use NativeStorage.from_int64_array(x)")
    assert _PUBLIC_LOWER_DOOR.search("NativeTensorCore.from_int64_array")
    for private in ("NativeStorage._from_int64_array",
                    "NativeTensorCore._from_int64_array",
                    "NativeTensor.from_int64_array"):
        assert not _PUBLIC_LOWER_DOOR.search(private), private
    # Arm 3: the AST door-finder really finds a door, and really ignores a
    # private helper and a public method that is not one.
    source = ("class NativeTensor:\n"
              "    def from_int64_array(cls, v):\n"
              "        return cpp.NativeTensorCore._from_int64_array(v)\n"
              "    def _private_door(cls, v):\n"
              "        return cpp.NativeTensorCore._from_int64_array(v)\n"
              "    def to_numpy(self):\n        return 1\n")
    found = set()
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, ast.ClassDef):
            continue
        for child in node.body:
            if (isinstance(child, ast.FunctionDef)
                    and not child.name.startswith("_")
                    and any(isinstance(call, ast.Call)
                            and isinstance(call.func, ast.Attribute)
                            and call.func.attr in _DOOR_MARKERS
                            for call in ast.walk(child))):
                found.add(f"{node.name}.{child.name}")
    assert found == {"NativeTensor.from_int64_array"}, found


def test_the_design_never_describes_the_k2_delta_as_a_single_public_name():
    """Arm 1 over the authority itself. The design may say "one public
    **door**" as often as it likes; it may not say the delta is one public
    **name**, because `item` and `tolist` are in it too."""
    offenders = _SINGULAR_PUBLIC_NAME.findall(_flat(_design()))
    assert offenders == [], offenders


def _milestone_record(milestone):
    """The body of one ``### Kn — …`` milestone record, up to the next one.

    Subsection-scoped rather than section-scoped, so a claim about K2 must
    live in K2's record and cannot be satisfied by K3's."""
    text = _design()
    marker = f"\n### {milestone} —"
    assert marker in text, f"the design has no {milestone} record"
    body = text.split(marker, 1)[1]
    following = re.search(r"\n### K\d+ [—-]", body)
    return body[:following.start()] if following else body


def test_the_design_states_the_three_names_and_the_one_door_distinction():
    """The distinction is written down, in the delta table and in the
    milestone record, rather than merely not-contradicted."""
    api = _flat(_section(_design(), 23))
    for name in LANDED_TENSOR_METHODS:
        assert name in api, name
    lowered = api.lower()
    assert "construction door" in lowered or "host-ingress door" in lowered
    assert "host-inspection" in lowered or "host inspection" in lowered
    # The milestone record says the same thing, in its own words.
    record = _flat(_milestone_record(FIRST_CONSTRUCTION_MILESTONE))
    assert "three" in record.lower(), record[:200]
    for name in LANDED_TENSOR_METHODS:
        assert name in record, name
    assert "only public construction or host-ingress door" in record.lower()
    # ...and the exit gate carries it too, so closure cannot restate the
    # delta more narrowly than the milestone did.
    gate = _flat(_section(_design(), 34)).lower()
    assert "only public construction or host-ingress door" in gate
    assert "item()" in gate and "tolist()" in gate


def test_no_public_storage_or_core_integer_constructor_appears_anywhere():
    """Arm 2, structurally and in prose. The design must never promise a
    public lower-layer door; the source must never define one."""
    assert _PUBLIC_LOWER_DOOR.findall(_flat(_design())) == []
    core_methods = _defined_names("src/tensorforge/backends/cpp.py",
                                  "NativeTensorCore")
    storage_methods = _defined_names("src/tensorforge/backends/cpp.py",
                                     "NativeStorage")
    for methods, layer in ((core_methods, "NativeTensorCore"),
                           (storage_methods, "NativeStorage")):
        assert "_from_int64_array" in methods, layer      # the private one
        public = {name for name in methods if not name.startswith("_")}
        for name in public:
            lowered = name.lower()
            for banned in ("int64", "integer"):
                assert banned not in lowered, (layer, name)
            # ``index`` is banned as a **constructor** name rather than as a
            # substring: K4's ``index_select`` is an operation, not a door,
            # and the Core legitimately carries it (§23.1). What must never
            # appear is a lower-layer name that *constructs* from indices.
            if "index" in lowered:
                assert name == "index_select", (layer, name)
                assert layer == "NativeTensorCore", (layer, name)


def test_exactly_one_public_construction_door_exists():
    """Arm 3. A second public door would make "the one public API through
    which an `int64` buffer can come into existence" false, however
    carefully each door was written."""
    assert _public_integer_doors() == {"NativeTensor.from_int64_array"}, \
        sorted(_public_integer_doors())


def test_item_and_tolist_are_part_of_the_delta_and_are_not_removed():
    """The correction is a *wording* correction: nothing is deleted to make
    a shorter sentence true."""
    tensor_methods = _defined_names(
        "src/tensorforge/experimental/native_tensor.py", "NativeTensor")
    for name in ("item", "tolist", "from_int64_array"):
        assert name in tensor_methods, name
    k2_names = {name for name, milestone in LANDED_TENSOR_METHODS.items()
                if milestone == FIRST_CONSTRUCTION_MILESTONE}
    assert k2_names == {"from_int64_array", "item", "tolist"}


# ===========================================================================
# 13. No live surface explains behavior by an absent integer dtype
# ===========================================================================
#
# K2 gave the runtime a real `int64` index/result dtype, so two long-true
# *reasons* expired even though both *conclusions* stand:
#
#   * classification targets remain exact host-side label metadata under
#     the Phase-E contract — because K2 did not widen cross-entropy, not
#     because the runtime cannot express an integer;
#   * native `argmax` is absent — because no milestone has shipped one,
#     not because its result type is inexpressible.
#
# A clearly time-bound historical statement ("at Phase E the runtime had no
# integer dtype") is honest and is allowed; a present-tense runtime
# limitation is not. Scoped to production source and current-status
# documentation: `examples/` and `tests/` are held to their own modules'
# rules and are not rewritten here.

LIVE_REASON_SURFACES = (
    "src/tensorforge/backends/cpp.py",
    "src/tensorforge/experimental/native_tensor.py",
    "src/tensorforge/experimental/native_metrics.py",
    "src/tensorforge/experimental/native_cross_entropy_loss.py",
    "README.md",
    "CLAUDE.md",
    "docs/roadmap.md",
    "docs/project_summary.md",
    "docs/architecture.md",
    "docs/release_history.md",
    "docs/native_support_matrix.md",
    "docs/native_integer_tensors_design.md",
    "docs/native_classification_design.md",
    "docs/native_data_pipeline_design.md",
)

# The expired reason, in the shapes it is actually written in.
_ABSENT_INTEGER_DTYPE = r"no\s+(?:native\s+|public\s+)?(?:integer|int64)\s+dtype"

# The two conclusions it may no longer explain. Each arm reads both
# directions, because "targets … no integer dtype" and "no integer dtype …
# for targets" are the same claim written two ways.
_STALE_REASONS = (
    ("cross-entropy targets are host metadata because of an absent "
     "integer dtype",
     r"\b(targets?|labels?|cross[- ]entropy)\b[^.]{0,120}" +
     _ABSENT_INTEGER_DTYPE
     + r"|" + _ABSENT_INTEGER_DTYPE +
     r"[^.]{0,120}\b(targets?|labels?|cross[- ]entropy)\b"),
    ("native argmax is absent because of an absent integer dtype",
     r"\bargmax\b[^.]{0,120}" + _ABSENT_INTEGER_DTYPE
     + r"|" + _ABSENT_INTEGER_DTYPE + r"[^.]{0,120}\bargmax\b"),
)

# What makes a statement honestly historical. A milestone label alone is
# enough ("K0 adds no integer dtype" is a delta, not a limitation), as is
# an explicit past framing or a quotation of the sentence that expired.
_TIME_BOUND = re.compile(
    r"\b(at\s+Phase\s+[A-J]\b|before\s+Phase\s+K|before\s+K[0-9]|"
    r"prior\s+to\s+K[0-9]|until\s+K[0-9]|through\s+K[0-9]|up\s+to\s+K[0-9]|"
    r"K[0-9]\s+(adds?|added|recorded|is|was)|used\s+to|no\s+longer|"
    r"expired|historical|formerly|previously|was\s+accurate|"
    r"the\s+sentence\s+read|now\s+exists|has\s+changed|changed\s+at|"
    r"corrected)\b", re.I)


def _stale_reasons(text):
    """Every expired-reason sentence in one body, ignoring the ones an
    explicit time bound makes honest."""
    flat = _flat(text)
    found = []
    for label, pattern in _STALE_REASONS:
        for match in re.finditer(pattern, flat, re.I):
            window = flat[max(0, match.start() - 220):match.end() + 220]
            if not _TIME_BOUND.search(window):
                found.append((label, match.group(0)))
    return found


def test_the_stale_reason_scanner_can_actually_fail():
    """Negative control: the scanner catches each expired reason, in both
    directions, and clears the time-bound forms."""
    for offender in (
        "Targets are not native tensors (the runtime has no integer dtype)",
        "the runtime has no integer dtype, so targets stay host metadata",
        "there is no native argmax: the runtime has no integer dtype",
        "no integer dtype exists for an argmax to return",
    ):
        assert _stale_reasons(offender), offender
    for honest in (
        "At Phase E the runtime had no integer dtype, so targets are host "
        "metadata",
        "Before Phase K there was no integer dtype for an argmax to return",
        "K0 adds no integer dtype, no dtype code, and no kernel",
        "Until K2 the sentence read \"the runtime has no integer dtype\", "
        "which was accurate for argmax then",
        "Classification targets remain exact host-side label metadata "
        "under the Phase-E contract",
        "A native argmax left the not-supported list at K3, which shipped "
        "it; max is declined permanently",
    ):
        assert _stale_reasons(honest) == [], (honest, _stale_reasons(honest))
    # ...and the scanner is not simply inert: an unrelated absence sentence
    # is neither caught nor needed.
    assert _stale_reasons("The native runtime has no CUDA backend.") == []


@pytest.mark.parametrize("surface", LIVE_REASON_SURFACES)
def test_no_live_surface_blames_an_absent_integer_dtype(surface):
    found = _stale_reasons(_read(surface))
    assert found == [], (surface, found[:3])


def test_the_current_reasons_are_written_down_where_they_belong():
    """The other half: the correct reason really is stated, so the scan
    above passes because the text was fixed rather than deleted."""
    targets = "Classification targets remain exact host-side label metadata"
    for surface in ("src/tensorforge/backends/cpp.py",
                    "src/tensorforge/experimental/native_tensor.py",
                    "docs/native_support_matrix.md"):
        flat = _flat(_read(surface)).lower()
        assert targets.lower() in flat, surface
        assert "did not widen cross-entropy" in flat or \
               "not widen cross-entropy" in flat, surface
    # The support matrix no longer lists a native ``argmax`` as absent —
    # K3 shipped it — and names what stays absent instead. Both halves are
    # asserted, so the row cannot be fixed by deletion.
    argmax = _flat(_read("docs/native_support_matrix.md")).lower()
    assert "left this list at phase k, milestone k3" in argmax
    assert "remains intentionally absent until k3" not in argmax
    assert re.search(r"native max, max_with_indices, or argmin", argmax)
    # ...and the production metric surface states the post-K3 truth: the
    # operation exists, and this helper still does not use it.
    metrics = _flat(_read(
        "src/tensorforge/experimental/native_metrics.py")).lower()
    assert "a native argmax exists" in metrics
    assert "deliberate" in metrics
    for expired in ("absent because nobody has shipped it",
                    "there is deliberately no native argmax"):
        assert expired not in metrics, expired


# ===========================================================================
# 14. CLAUDE.md size policy — the existing ceiling, and nothing stricter
# ===========================================================================

def test_claude_md_stays_below_the_project_ceiling():
    text = _read("CLAUDE.md").replace("\r\n", "\n")
    assert len(text) < CLAUDE_MD_CEILING, len(text)


def test_the_claude_md_document_map_names_the_phase_k_authority():
    text = _read("CLAUDE.md")
    assert f"docs/{PHASE_K_DESIGN_NAME}" in text
    assert "Phase K" in text
