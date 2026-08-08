"""Phase-K closure guardrails (native integer tensors and indexing, K9).

The durable replacement for Phase K's milestone-era *pending* checks, in the
shape ``tests/test_native_phase_h_closure.py`` established and the Phase-I
and Phase-J closure modules refined. Every earlier guard about this phase
carried a premise that expires at closure — "K9 has not started", "the
closure module is absent", "no surface may call Phase K complete", "the
phase is newly approved and in progress". K9 ran the cross-platform matrix
and closed the phase, so those premises are gone. What replaces them is not
silence: the boundary Phase K stopped at is now a **permanent** rule rather
than a temporary one, and this module is where it is enforced.

The rules these tests protect:

* the phase is closed, and its ladder is whole (K0-K9, once each, in order,
  every one marked complete, with no K10 and nothing left open);
* **the phase's whole public delta is five ``NativeTensor`` methods and one
  registry row** — ``from_int64_array``/``item``/``tolist`` at K2,
  ``argmax`` at K3, ``index_select`` at K4, and ``INDEX_DTYPES ==
  ("int64",)`` at K2 — with ``SUPPORTED_DTYPES`` **unmoved**,
  ``normalize_dtype("int64")`` raising permanently, and
  ``tensorforge.experimental.__all__`` still at **25**;
* the export surface closed at exactly **56** — Phase J's 54 plus
  ``tf_core_argmax`` (K3) and ``tf_core_index_select`` (K4) — and the
  source inventory agrees with the built library;
* ``int64`` is an **index/result** dtype, never a compute dtype: no
  generic constructor accepts it, no kernel computes at it, no autograd
  node, parameter, buffer, optimizer slot, or checkpoint entry can carry
  it, and the four state authorities keep their exact formats and versions
  — checkpoint 3 with ``(1, 2, 3)`` accepted, optimizer state 1, loader
  state 1, sampler state 1, with no version 4 anywhere;
* the deliberately absent surface stays absent: no ``argmin``, ``max``, or
  ``max_with_indices``, no general ``gather``/``scatter``/``scatter_add``,
  no embedding, no ``__getitem__`` indexing, no ``index_select`` backward,
  no casting or promotion, no integer arithmetic, no CUDA, and no AMP;
* the K0-K8 evidence — contract, barriers, tensor, argmax, index_select,
  compatibility, example, hardening, and benchmark — is all still present
  and cannot be deleted or weakened unnoticed;
* closure is **not** permission to name a successor: no K10, and no
  invented phase after this one;
* **K9's chronology stays truthful**: it added no new *capability*, but it
  was not production-code-free — it carries **two** behaviour-preserving
  production repairs, the C++ switch-exhaustiveness change across seven
  translation units and the ``NativeTensorCore`` fresh-storage ownership
  guard in ``backends/cpp.py`` — so no current status surface may credit it
  with "zero production code", "no production code", or "proof only" while
  those files are part of it, and none may call either repair K9's *only*
  one;
* **§25's device contract is pinned by signature, not by prose**: the
  inherited floating constructors *do* carry a validated, CPU-only
  ``device="cpu"`` metadata parameter, the three Phase-K public methods
  carry no ``device`` at all, and the absolute claim "no ``device``
  argument exists anywhere" may not return.

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
is driven against deliberately broken input. The controls operate on
temporary strings and temporary directories only: no repository file is
written, moved, or restored.

**Prohibited-token scans read code, not prose.** A closure module whose job
includes proving "no gather, no scatter, no thread" would fail on the very
sentences that document the prohibition, so every such scan strips
docstrings, string literals, and C++ comments first and looks at
identifiers and definitions.

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
PHASE_K_DESIGN = REPO_ROOT / "docs" / "native_integer_tensors_design.md"
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
# The final boundary, written out once, as literals, so a drift in either
# direction is a single obvious diff — and split by *what kind of fact each
# one is*. Phase K's defining property is that it moved exactly one public
# registry (a new row, not a widened one), two exports, three CTests, one
# example, and one benchmark, and nothing else.
# ---------------------------------------------------------------------------

# The floating-compute capability, exactly as a completed Phase J handed it
# over. Phase K never moves it: taxonomy B (design §5.1) is permanent.
FINAL_SUPPORTED_DTYPES = ("float64", "float32")
FINAL_SUPPORTED_DEVICES = ("cpu",)
FINAL_UNSUPPORTED = ("cuda", "amp")
FINAL_DEFAULT_DTYPE = "float64"
# A permanent limitation of the seven handle-free raw utility kernels,
# which take only ``double*`` and an element count. Never the overall
# support statement.
FINAL_RAW_KERNEL_DTYPES = ("float64",)
# The one public registry Phase K added, at K2, in the same commit as the
# public constructor it promises. ``int64`` is an index/result dtype in its
# own row, and the union of this row and SUPPORTED_DTYPES is every dtype a
# native tensor can carry.
FINAL_INDEX_DTYPES = ("int64",)
ABI_INT64_CODE = 2
INT64_ITEM_SIZE = 8

# What Phase K inherited, pinned as history so the delta is checkable.
PHASE_J_EXPORT_COUNT = 54
PHASE_J_CTEST_COUNT = 24
PHASE_J_EXAMPLE_COUNT = 16
PHASE_J_BENCHMARK_COUNT = 9

# The ABI delta: two exports, each attributed to the milestone that shipped
# it, and 56 is the phase maximum the design commits to (§33).
PHASE_K_ADDED_EXPORTS = {"tf_core_argmax": "K3", "tf_core_index_select": "K4"}
FINAL_EXPORT_COUNT = PHASE_J_EXPORT_COUNT + len(PHASE_K_ADDED_EXPORTS)  # 56

# The CTest delta: three targets, likewise attributed.
PHASE_K_ADDED_CTESTS = {"dtype_int64_storage": "K1", "argmax": "K3",
                        "index_select": "K4"}
FINAL_CTEST_COUNT = PHASE_J_CTEST_COUNT + len(PHASE_K_ADDED_CTESTS)     # 27

# The artifact deltas: one example (K6) and one benchmark (K8).
PHASE_K_ADDED_EXAMPLES = {"native_integer_indexing.py": "K6"}
FINAL_EXAMPLE_COUNT = PHASE_J_EXAMPLE_COUNT + len(PHASE_K_ADDED_EXAMPLES)
PHASE_K_ADDED_BENCHMARKS = {"benchmark_native_integer.py": "K8"}
FINAL_BENCHMARK_COUNT = (PHASE_J_BENCHMARK_COUNT
                         + len(PHASE_K_ADDED_BENCHMARKS))               # 10

# The public Python surface: no new experimental export at any milestone,
# and the errcheck-hooked kernel list grew by exactly the two new exports.
FINAL_EXPERIMENTAL_EXPORTS = 25
FINAL_CHECKED_KERNELS = 38

# Serialization — four separate authorities, none of which moved.
FINAL_CHECKPOINT_FORMAT = "tensorforge.native_checkpoint"
FINAL_CHECKPOINT_VERSION = 3
FINAL_CHECKPOINT_VERSIONS = (1, 2, 3)
FINAL_OPTIMIZER_STATE_VERSION = 1
FINAL_LOADER_STATE_FORMAT = "tensorforge.native_data_loader"
FINAL_LOADER_STATE_VERSION = 1
FINAL_LOADER_STATE_VERSIONS = (1,)
FINAL_SAMPLER_STATE_FORMAT = "tensorforge.native_sampler"
FINAL_SAMPLER_STATE_VERSION = 1
FINAL_SAMPLER_STATE_VERSIONS = (1,)

MILESTONES = tuple(f"K{index}" for index in range(10))   # K0 ... K9

# The phase's public names — five NativeTensor methods, each landed by
# exactly one milestone, with ``experimental.__all__`` never moving.
PHASE_K_TENSOR_METHODS = {
    "from_int64_array": "K2",
    "item": "K2",
    "tolist": "K2",
    "argmax": "K3",
    "index_select": "K4",
}

# The operation registries as Phase K closed on them, pinned as exact
# tuples independently of ``backends/cpp.py`` so a drift in either
# direction fails here rather than propagating. ``argmax`` and
# ``index_select`` are the phase's two additions to TENSOR_CORE_OPS, and
# AUTOGRAD_OPS gained **nothing**.
FINAL_TENSOR_CORE_OPS = (
    "relu", "sqrt", "reciprocal", "exp", "log", "softmax", "log_softmax",
    "add", "subtract", "multiply", "matmul", "sum", "mean",
    "reshape", "transpose", "T", "narrow", "contiguous_copy",
    "conv2d_forward", "conv2d_input_backward", "conv2d_weight_backward",
    "maxpool2d_forward", "maxpool2d_backward",
    "cross_entropy_forward", "cross_entropy_backward", "dropout_forward",
    "argmax", "index_select",
)
FINAL_AUTOGRAD_OPS = (
    "add", "subtract", "multiply", "relu", "sum", "mean", "matmul",
    "reshape", "transpose", "T", "narrow", "contiguous_copy",
    "sqrt", "reciprocal", "conv2d", "maxpool2d",
    "exp", "log", "softmax", "log_softmax", "cross_entropy", "dropout",
)

# The evidence K0-K8 left, which closure must not let disappear. Each entry
# is (path, minimum test count) — a floor rather than an equality, so adding
# coverage is free while deleting it is not. The floors sit below the counts
# observed at closure on purpose: this guards deletion, not growth.
REQUIRED_EVIDENCE = (
    ("tests/test_native_phase_k.py", 80),                 # K0, the contract
    ("tests/test_native_integer_barriers.py", 35),        # K1
    ("tests/test_native_int64_tensor.py", 70),            # K2
    ("tests/test_native_argmax.py", 50),                  # K3
    ("tests/test_native_index_select.py", 55),            # K4
    ("tests/test_native_integer_compatibility.py", 40),   # K5
    ("tests/test_native_integer_indexing_example.py", 75),  # K6
    ("tests/test_native_integer_hardening.py", 70),       # K7
    ("tests/test_native_integer_benchmark.py", 110),      # K8
)

# Method names that must not exist on either tensor layer — the absent
# surface §35 makes permanent (or defers to a separately approved phase).
BANNED_TENSOR_METHODS = (
    "argmin", "max", "max_with_indices", "gather", "scatter", "scatter_add",
    "embedding", "take", "nonzero", "sort", "argsort", "topk", "top_k",
    "unique", "where", "searchsorted", "bincount", "cumsum",
    "__getitem__", "__setitem__", "astype", "to", "cast", "float", "double",
    "int", "long", "half", "cuda", "cpu", "type_as", "promote",
    "index_select_backward",
)

# The seven pre-existing float-only dispatch translation units K9's
# warning-cleanliness repair touched. K1's third dtype enumerator left
# their ``switch``es non-exhaustive — harmless at runtime, because
# ``tf::require_floating`` rejects an ``int64`` handle first, but 237
# ``-Wswitch`` diagnostics across 21 sites on a ``-Wall`` build — and K9
# wrote the unreachable ``Int64`` arm out at every site.
#
# They are named here so the chronology ban below keeps a **live premise**:
# "K9 was not production-code-free" is only worth enforcing while these
# files really do carry the repair, so the guard checks both halves.
K9_REPAIRED_TRANSLATION_UNITS = (
    "cpp/src/classification.cpp",
    "cpp/src/conv2d.cpp",
    "cpp/src/elementwise.cpp",
    "cpp/src/matmul.cpp",
    "cpp/src/pooling.cpp",
    "cpp/src/random.cpp",
    "cpp/src/reduction.cpp",
)

# §25's device contract, split into the two halves that are genuinely
# different facts. The first is **inherited** Phase-I-era dtype/device
# metadata: a defaulted, validated tag that accepts only ``"cpu"``,
# selects nothing, and transfers nothing. The second is Phase K's own
# surface, which carries no such parameter at all.
INHERITED_DEVICE_CONSTRUCTORS = (
    ("cpp.NativeStorage", "__init__"),
    ("cpp.NativeStorage", "from_array"),
    ("cpp.NativeTensorCore", "from_array"),
    ("cpp.NativeTensorCore", "zeros"),
    ("cpp.NativeTensorCore", "full"),
    ("NativeTensor", "from_array"),
    ("NativeTensor", "zeros"),
    ("NativeTensor", "full"),
)
PHASE_K_DEVICE_FREE_METHODS = (
    ("NativeTensor", "from_int64_array"),
    ("NativeTensor", "argmax"),
    ("NativeTensor", "index_select"),
    ("cpp.NativeTensorCore", "argmax"),
    ("cpp.NativeTensorCore", "index_select"),
)
# ...and the third fact, which is neither of those two: Phase K's own
# **private** ingress helpers inherit the same defaulted metadata tag. They
# are the reason "everything added since carries no device parameter" is
# false while "everything **public** added since carries none" is true, and
# they are pinned here so the true sentence keeps a checkable premise.
PRIVATE_DEVICE_INGRESS_HELPERS = (
    ("cpp.NativeStorage", "_from_int64_array"),
    ("cpp.NativeTensorCore", "_from_int64_array"),
)
FINAL_DEFAULT_DEVICE = "cpu"

# C ABI shapes an integer-indexing phase would be most tempted to add,
# banned by name. The phase's two exports are the whole of its ABI.
#
# The trailing boundary is load-bearing: without it ``max`` would match the
# **shipped** ``tf_core_maxpool2d_forward``/``_backward``, which are Phase
# D's pooling exports and have nothing to do with a scalar maximum. A
# banned name must be the whole operation name or be followed by ``_``.
BANNED_EXPORT_SHAPES = re.compile(
    r"^tf_(core_)?(argmin|max|gather|scatter|scatter_add|embedding|"
    r"index_put|take|sort|argsort|topk|nonzero|where|cumsum|cast|astype|"
    r"promote)(_|$)", re.I)


def _read(relative):
    return (REPO_ROOT / relative).read_text(encoding="utf-8")


def _flat(text):
    """Whitespace-flattened, emphasis-stripped text, so a claim split across
    lines or wrapped in markdown still reads as one sentence.

    Line endings collapse with the rest of the whitespace, which is what
    makes every prose scan below identical on a CRLF checkout."""
    return re.sub(r"\s+", " ", re.sub(r"[*`]", "", text))


def _design():
    return PHASE_K_DESIGN.read_text(encoding="utf-8")


def _section(text, number):
    """The body of top-level section ``number``, up to the next one."""
    marker = f"\n## {number}."
    assert marker in text, f"the design has no section {number}"
    body = text.split(marker, 1)[1]
    following = re.search(r"\n## \d+\.", body)
    return body[:following.start()] if following else body


def _source_exports():
    names = set()
    for source in sorted((REPO_ROOT / "cpp" / "src").glob("*.cpp")):
        text = source.read_text(encoding="utf-8")
        names.update(re.findall(r"TF_EXPORT[^;{]*?\b(tf_[a-z0-9_]+)\s*\(",
                                text, re.S))
    return names


def _cpp_code_only(text):
    """C++ source with ``/* */`` and ``//`` comments removed, so an absence
    scan means "the runtime does not do this" rather than "nobody wrote the
    word"."""
    return re.sub(r"//[^\n]*", " ", re.sub(r"/\*.*?\*/", " ", text,
                                           flags=re.S))


def _git_lines(*arguments):
    """One read-only Git query, or a documented skip where Git cannot
    answer. The only place this module talks to Git at all, and it asks
    exclusively about the *working tree and index*, which a depth-1
    checkout has in full — never a historical blob."""
    try:
        done = subprocess.run(["git", *arguments], cwd=REPO_ROOT,
                              capture_output=True, text=True)
    except OSError:                                   # pragma: no cover
        pytest.skip("git is unavailable, so the tree cannot be inspected")
    if done.returncode != 0:                          # pragma: no cover
        pytest.skip("this tree has no git index to read")
    return done.stdout.splitlines()


def _tracked_files():
    return _git_lines("ls-files")


def _untracked_unignored_files():
    """Every file present but neither tracked nor ignored — the set that is
    one ``git add`` away from becoming repository content."""
    return _git_lines("ls-files", "--others", "--exclude-standard")


def _class_method_names(relative, class_name):
    """Every method name defined on ``class_name`` in a module, read from
    the AST so prose cannot satisfy or trip the scan."""
    tree = ast.parse(_read(relative))
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            return {child.name for child in node.body
                    if isinstance(child, (ast.FunctionDef,
                                          ast.AsyncFunctionDef))}
    raise AssertionError(f"{relative} defines no class {class_name}")


def _code_identifiers(text):
    """Every identifier that *executes* in ``text`` — names, attributes,
    keyword-argument names, definitions, and import names — and no string
    content at all."""
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


def _int64_probe():
    """A fresh one-element int64 tensor, for boundary probes."""
    from tensorforge.experimental import NativeTensor
    return NativeTensor.from_int64_array(np.array([0], dtype=np.int64))


needs_backend = pytest.mark.skipif(not cpp.is_available(),
                                   reason="the native library is not built")


# ===========================================================================
# 1. The milestone ladder — whole, ordered, and closed
# ===========================================================================
#
# The ladder is §32's ``| **Kn** | purpose | status |`` table. Parsed
# structurally rather than matched as prose, and the parser is a pure
# function over text so the negative controls can feed it deliberately
# broken ladders.

_LADDER_ROW = re.compile(r"^\|\s*\*\*(K\d+)\*\*\s*\|(.*)\|\s*$", re.M)


def _ladder_rows(text):
    """``[(milestone, remaining cells)]`` in document order."""
    if "## 32." not in text:
        raise AssertionError("the design has no milestone-ladder section")
    ladder = text.split("## 32.", 1)[1]
    following = re.search(r"\n## \d+\.", ladder)
    if following:
        ladder = ladder[:following.start()]
    return _LADDER_ROW.findall(ladder)


def _ladder_problems(text):
    """Every way the closed ladder can be wrong, as a list of reasons."""
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

    # Every row marked complete, in its status cell. A phase with an
    # unmarked or still-open row is not closed, with no exception.
    for name, cells in rows:
        status = _flat(cells).lower()
        if "complete" not in status:
            problems.append(f"{name} is not marked complete")
        if re.search(r"unstarted|not started|in progress|pending", status):
            problems.append(f"{name} is still marked open")
    return problems


def test_the_milestone_ladder_runs_k0_to_k9_once_each_in_order():
    """K0 through K9, each exactly once, ordered, and every one marked
    complete. This is the single assertion that says the phase is done."""
    problems = _ladder_problems(_design())
    assert problems == [], problems


def test_the_ladder_carries_exactly_ten_rows_and_no_more():
    rows = _ladder_rows(_design())
    assert len(rows) == 10, [name for name, _ in rows]
    assert [name for name, _ in rows] == list(MILESTONES)


def test_every_milestone_record_exists_and_k9_records_the_closure():
    """Each ``### Kn —`` record section survives closure, and K9's is a
    real record rather than a placeholder: it must name the validation
    matrix it ran (builds, sanitizers, the cross-platform equality) and its
    own closure module."""
    text = _design()
    for name in MILESTONES:
        assert f"### {name} —" in text, f"no milestone record for {name}"
    record = text.split("### K9 —", 1)[1]
    following = re.search(r"\n### K\d+ [—-]|\n## \d+\.", record)
    body = _flat(record[:following.start()] if following else record)
    for fact in ("Release", "Debug", "CTest", "ASan", "UBSan",
                 "LeakSanitizer", "Windows", "Linux",
                 "test_native_phase_k_closure.py"):
        assert fact.lower() in body.lower(), fact
    assert re.search(r"no (new )?capability", body, re.I), body[:300]


def test_the_design_marks_the_phase_complete_rather_than_active():
    """The status line, which is what every other surface is reconciled
    against. Through K8 it had to name K9 as unstarted; at closure there is
    no unstarted milestone, and saying so is the point."""
    text = _flat(_design())
    status = re.search(r"Phase-K status:(.{0,240})", text, re.I)
    assert status, "the design does not state its milestone status"
    claim = status.group(1)
    assert re.search(r"\bK0\b.{0,80}?\bK9\b.{0,60}?complete", claim, re.I), (
        f"the status line does not record K0-K9 complete: {claim!r}")
    assert not re.search(r"unstarted|not started", claim, re.I), (
        f"the status line still names an unstarted milestone: {claim!r}")
    assert re.search(
        r"Phase K is (?:now )?(?:\w+ ){0,3}?(complete|closed)", text, re.I
    ), "the design does not state that Phase K is closed"


def test_the_design_names_no_milestone_beyond_k9():
    """A K10 would be a milestone this document does not define, and the
    phase after Phase K requires a separate approval, not a paragraph."""
    for match in re.finditer(r"\bK1[0-9]\b", _design()):
        raise AssertionError(f"the design names {match.group(0)}")


def test_phases_i_and_j_are_still_recorded_complete_and_not_reopened():
    """Closing one phase must not disturb the ones before it: Phase I is
    where float32 came from, Phase J is where the pipeline came from, and
    both closure modules must survive."""
    assert (REPO_ROOT / "tests" / "test_native_phase_i_closure.py").is_file()
    assert (REPO_ROOT / "tests" / "test_native_phase_j_closure.py").is_file()
    phase_j = _flat((REPO_ROOT / "docs"
                     / "native_data_pipeline_design.md").read_text(
                         encoding="utf-8"))
    assert re.search(r"Phase J is (?:now )?(?:\w+ ){0,3}?(complete|closed)",
                     phase_j, re.I)
    phase_i = _flat((REPO_ROOT / "docs"
                     / "native_dtype_float32_design.md").read_text(
                         encoding="utf-8"))
    assert re.search(r"Phase I is (?:now )?(?:\w+ ){0,3}?(complete|closed)",
                     phase_i, re.I)


# --- the ladder parser's negative controls --------------------------------

def _ladder_document(rows, status="**Complete.** Body."):
    """A synthetic design fragment carrying exactly ``rows``."""
    body = "\n".join(f"| **{name}** | Some purpose here | {status} |"
                     for name in rows)
    return f"## 32. Final milestone ladder\n\n{body}\n\n## 33. Next\n"


def test_the_ladder_parser_detects_every_kind_of_drift():
    """Each fault the closure claim depends on is produced deliberately and
    shown to be caught, on temporary strings only."""
    # 0. The positive control: a whole, ordered, complete ladder passes.
    assert _ladder_problems(_ladder_document(MILESTONES)) == []

    # 1. K9 omitted — the specific drift a closure ladder must not have.
    problems = _ladder_problems(
        _ladder_document([n for n in MILESTONES if n != "K9"]))
    assert any("missing" in reason and "K9" in reason
               for reason in problems), problems

    # 2. A middle milestone omitted.
    problems = _ladder_problems(
        _ladder_document([n for n in MILESTONES if n != "K5"]))
    assert any("missing" in reason and "K5" in reason
               for reason in problems), problems

    # 3. One milestone duplicated.
    problems = _ladder_problems(_ladder_document(list(MILESTONES) + ["K7"]))
    assert any("duplicated" in reason for reason in problems), problems

    # 4. Two milestones swapped.
    swapped = list(MILESTONES)
    swapped[3], swapped[4] = swapped[4], swapped[3]
    problems = _ladder_problems(_ladder_document(swapped))
    assert any("out of order" in reason for reason in problems), problems

    # 5. An invented K10.
    problems = _ladder_problems(_ladder_document(list(MILESTONES) + ["K10"]))
    assert any("unexpected" in reason and "K10" in reason
               for reason in problems), problems

    # 6. K9 reverted to unstarted — the drift closure exists to stop.
    document = _ladder_document(MILESTONES).replace(
        "| **K9** | Some purpose here | **Complete.** Body. |",
        "| **K9** | Some purpose here | *Unstarted.* |")
    problems = _ladder_problems(document)
    assert any("K9" in reason for reason in problems), problems

    # 7. An earlier milestone left with no completion marker at all.
    document = _ladder_document(MILESTONES).replace(
        "| **K4** | Some purpose here | **Complete.** Body. |",
        "| **K4** | Some purpose here | Body only. |")
    assert "K4 is not marked complete" in _ladder_problems(document)

    # 8. No ladder section at all.
    with pytest.raises(AssertionError):
        _ladder_problems("# a document with no section 32\n")

    # Nothing above touched the repository: the live ladder still passes.
    assert _ladder_problems(_design()) == []


# ===========================================================================
# 2. Phase-complete status, across every current surface
# ===========================================================================

# One authority for "this surface still says the phase is open", used by the
# scan and by its own negative control so the two cannot drift apart.
_STALE_STATUS = re.compile(
    r"Phase[- ]K\b[^.;]{0,90}?\b(is not complete|is not closed|is active|"
    r"remains active|has not been closed|is the current phase|"
    r"is in progress|remains in progress|is not finished|is incomplete|"
    r"is unfinished|is newly approved|is a newly approved|is awaiting)\b"
    r"|\bK9\b[^.;]{0,70}?\b(unstarted|not started|has not started|pending|"
    r"is next)\b"
    r"|\b(what remains|remaining|still to come|next|awaiting)\b"
    r"[^.;]{0,40}?\bK9\b"
    r"|\bonly K0 through K8 have landed\b"
    r"|\bK0[-–—]K8\b[^.;]{0,40}?\bcomplete\b"
    r"|\bK0 through K8\b[^.;]{0,40}?\b(are )?complete\b"
    r"|\bK8 is the (newest|latest)\b",
    re.I)

# Sentences whose tense makes them accurate history rather than a stale
# claim. Deliberately narrower than "any past-tense verb anywhere": the
# marker has to sit in the same neighbourhood as the match.
_HISTORY = re.compile(
    r"\b(was|were|had|until|before|at K\d|by K\d|then|earlier|previously|"
    r"stayed|remained|originally|drafted|no longer|used to|history|"
    r"historical|during|while|as of K\d|through K8,|read)\b", re.I)


def _stale_status(text):
    """Every stale status claim in ``text``, as matched spans."""
    return [match.group(0) for match in _STALE_STATUS.finditer(text)
            if not _HISTORY.search(text[max(0, match.start() - 140):
                                        match.end() + 50])]


@pytest.mark.parametrize("surface", STATUS_SURFACES + ("CLAUDE.md",))
def test_no_status_surface_still_calls_phase_k_unfinished(surface):
    """The failure this exists for: a surface advanced in one paragraph and
    left saying "K9 is unstarted" in another. Time-scoped sentences are
    excused, because "K9 was unstarted at K8" is history and history stays
    accurate."""
    offenders = _stale_status(_flat(_read(surface)))
    assert offenders == [], f"{surface}: {offenders[:3]}"


# The completion claim, which must attach to the **phase** rather than to a
# milestone inside it: "Phase K is approved and K0 is complete" is a K0-era
# sentence and may not satisfy this.
_PHASE_COMPLETE = re.compile(
    r"Phase K\b(?:(?!\bK\d)[^.;]){0,140}?\b(is|are)\s+(now\s+)?"
    r"(complete|closed)"
    r"|Phase K\b[^.;]{0,60}?\bcomplete \(K0[-–—]K9\)"
    r"|\bK0[-–—]K9\b[^.;]{0,70}?\b(complete|landed|closed)"
    r"|\bK0 through K9\b[^.;]{0,70}?\b(complete|landed|closed)",
    re.I)


@pytest.mark.parametrize("surface", STATUS_SURFACES)
def test_every_status_surface_marks_phase_k_complete(surface):
    text = _flat(_read(surface))
    assert "Phase K" in text, f"{surface} does not name Phase K"
    assert _PHASE_COMPLETE.search(text), (
        f"{surface} does not mark Phase K complete")


# Both orders of the latest-completed claim, shared with the negative
# control below.
_LATEST_COMPLETED_FORM = re.compile(
    r"Phase ([A-K])\b[^.;]{0,60}?\bis the latest completed\b"
    r"|latest completed (?:native )?phase is Phase ([A-K])\b"
    r"|The latest completed (?:native )?phase [—-] Phase ([A-K])\b", re.I)


def _latest_completed_letters(text, allow_history=False):
    """Every phase letter awarded the "latest completed" title in ``text``.

    ``allow_history`` skips a match whose own sentence scopes it to an
    earlier moment — "Phase I *was* the latest completed phase until Phase
    J closed", or an H-era milestone record saying "at this milestone Phase
    G *remains* the latest completed phase". Those are accurate history and
    the project's discipline is to repair rather than delete them. The
    escape is deliberately narrow: the marker has to sit in the same
    neighbourhood, and a bare present-tense claim carries none."""
    letters = set()
    for match in _LATEST_COMPLETED_FORM.finditer(text):
        if allow_history:
            window = text[max(0, match.start() - 130):match.end() + 90]
            if re.search(r"\b(was|were|until|before|at [A-K]\d|by [A-K]\d|"
                         r"remained|stayed|no longer|previously|"
                         r"closed after it|remains the latest completed)\b",
                         window, re.I):
                continue
        letters.update(group.upper() for group in match.groups() if group)
    return letters


@pytest.mark.parametrize("surface", STATUS_SURFACES + ("CLAUDE.md",))
def test_no_surface_calls_a_phase_before_k_the_latest_completed(surface):
    """At closure Phase K is the latest completed native phase. A surface
    may say so or say nothing; what none may do is award the title to an
    earlier phase as a claim about *now*.

    History is excused, because a per-milestone record saying "Phase G
    remained the latest completed phase at H4" is accurate and must stay
    that way — that is why the repository repairs such sentences rather
    than deleting them."""
    named = _latest_completed_letters(_flat(_read(surface)),
                                      allow_history=True)
    assert named <= {"K"}, (surface, sorted(named))


def test_the_latest_completed_claim_is_made_somewhere():
    """At least one current-status surface states the new fact outright,
    so the closure is discoverable rather than merely not-contradicted."""
    for surface in STATUS_SURFACES:
        if "K" in _latest_completed_letters(_flat(_read(surface))):
            return
    raise AssertionError(
        "no status surface names Phase K as the latest completed phase")


@pytest.mark.parametrize("surface", STATUS_SURFACES + ("CLAUDE.md",))
def test_no_status_surface_invents_a_phase_after_k(surface):
    """Closure is not permission to *invent* the next phase. A K10, an
    unapproved successor, or a committed promise about one would be a
    roadmap entry nobody approved. Phase L is the moving sentinel exactly
    as Phase K was for the Phase-J closure — if a Phase L is ever
    separately approved, this entry moves on the terms the J-closure
    documented."""
    text = _read(surface)
    for match in re.finditer(r"\bK1[0-9]\b", text):
        raise AssertionError(f"{surface} names {match.group(0)}")
    flat = _flat(text)
    for pattern in (r"\bPhase L\b",
                    r"\bthe next phase (is|will be)\b[^.;]{0,40}?\b\w"):
        offender = re.search(pattern, flat, re.I)
        assert offender is None, f"{surface}: {offender.group(0)!r}"


def test_the_phase_complete_scan_detects_a_surface_left_active():
    """The prose scans' own control, driven through the **same** compiled
    patterns the scans use, so a weakening there fails here."""
    for caught in (
        "Phase K is in progress",
        "Phase K is not complete",
        "Phase K is the current phase",
        "Phase K is newly approved",
        "Phase K remains in progress",
        "K9 is unstarted",
        "K9 has not started",
        "only K0 through K8 have landed",
        "K0 through K8 are complete",
        "K8 is the newest milestone",
        "what remains is K9",
    ):
        assert _stale_status(caught), caught
    # ...and the accurate closure sentences are not caught.
    for allowed in (
        "Phase K is complete",
        "Phase K is closed: milestones K0 through K9 have all landed",
        "K9 closed the phase",
        "Phase K was newly approved after Phase J closed at J9",
        "milestones K0 through K9 complete",
        "at K8 the benchmark landed; K9 closed the phase",
        "K9 was unstarted while K8 was the newest milestone",
    ):
        assert _stale_status(allowed) == [], (allowed, _stale_status(allowed))
    # The completion pattern recognises the closure sentences and is not
    # satisfied by an open one.
    for recognised in (
        "Phase K is complete",
        "Phase K is now closed",
        "Phase K complete (K0–K9)",
        "milestones K0–K9 are complete",
        "K0 through K9 have all landed",
    ):
        assert _PHASE_COMPLETE.search(recognised), recognised
    for rejected in (
        "Phase K is approved and K0 is complete",
        "milestones K0 through K8 have landed",
    ):
        assert not _PHASE_COMPLETE.search(rejected), rejected
    # The latest-completed form finds the letter and only the letter.
    assert _latest_completed_letters(
        "Phase K is the latest completed native phase") == {"K"}
    assert _latest_completed_letters(
        "Phase J is the latest completed phase") == {"J"}
    assert _latest_completed_letters("Phase K is complete") == set()
    # ...and the history escape excuses a scoped sentence without
    # excusing a bare present-tense claim, which is the whole point of it.
    for excused in (
        "Phase J was the latest completed phase until Phase K closed",
        "at H4, Phase G remains the latest completed phase",
        "Phase I was the latest completed phase before Phase J closed",
    ):
        assert _latest_completed_letters(excused, allow_history=True) == set(), (
            excused)
    for still_caught in (
        "the latest completed phase is Phase J",
        "Phase J is the latest completed phase",
    ):
        assert _latest_completed_letters(still_caught,
                                         allow_history=True) == {"J"}, (
            still_caught)


# ===========================================================================
# 2b. K9's own chronology — no new capability, but not proof-only
# ===========================================================================
#
# The failure this section exists for actually happened: the K9 candidate
# was written up on every surface as "zero production code" / "proof only",
# while the same candidate carried an executable C++ repair across seven
# translation units. Both halves of the truth are load-bearing and neither
# may be dropped:
#
#   * K9 added **no new capability** — no export, no public name, no
#     registry, no version, no CTest, no example, no benchmark, no
#     TENSOR_CORE_OPS or AUTOGRAD_OPS entry;
#   * K9 **did** include behaviour-preserving executable production
#     repairs — **two** of them, in two languages: the C++
#     switch-exhaustiveness change across seven translation units, and the
#     ``NativeTensorCore`` fresh-storage ownership guard in
#     ``backends/cpp.py``, whose unguarded window sat on the Phase-K
#     Policy-B materialization path.
#
# A surface that states only the first, in the form of the second, is
# wrong. So is one that states the second in the singular after the second
# repair landed — hence the two scans below. Both read *asserted* claims: a
# phrase inside quotes is a mention (this repository documents its own
# prohibitions in prose), and a phrase whose own clause negates it is a
# denial.

_ZERO_PRODUCTION_CLAIM = re.compile(
    r"zero production code|no production code|production[- ]code[- ]free|"
    r"proof[- ]only|added proof only|adding proof only|adds proof only|"
    r"added only proof|touched no production code", re.I)

# A negation sitting in the claim's **own clause**. The clause boundary
# matters and is not decoration: the wording that first got past an
# earlier draft of this scan was "...so it is **not** carried-over roadmap
# work. **K9 added no production code**", where the negation belongs to
# the *previous sentence*. So the prefix is cut at the last sentence or
# table-cell boundary before the phrase, and only what remains is read.
_DENIAL = re.compile(
    r"\b(not|never|neither|nor|rather than|instead of|no longer|"
    r"isn't|wasn't|cannot|must not|may not|stop|avoid|false)\b", re.I)


def _quoted(text, match):
    """True when the matched phrase is wrapped in quotes or backticks — a
    *mention* of the banned wording rather than a claim in it."""
    before = text[max(0, match.start() - 3):match.start()]
    after = text[match.end():match.end() + 3]
    return bool(re.search(r"[\"“”`']\s*$", before)
                and re.match(r"\s*[\"“”`']", after))


def _owning_milestone(text, position):
    """The nearest milestone token named before ``position``, or ``None``.

    "Is K9 mentioned nearby?" is the wrong question: a ladder row names K9
    in its first cell and makes the claim two cells later, while a
    chronology paragraph about J7 can easily sit within a few hundred
    characters of an unrelated K-milestone sentence. The *nearest
    preceding* milestone is the one the claim is about, so that is what
    this returns — across every phase letter, so a J- or H-era sentence
    is attributed to its own milestone rather than borrowed by this
    scan."""
    tokens = list(re.finditer(r"\b([A-K]\d)\b", text[:position]))
    if not tokens or position - tokens[-1].start() > 400:
        return None
    return tokens[-1].group(1)


def _k9_zero_production_claims(text):
    """Every asserted "K9 shipped no production code" claim in ``text``.

    A pure function over flattened text, so the negative control can feed
    it deliberately wrong surfaces without touching the repository."""
    offenders = []
    for match in _ZERO_PRODUCTION_CLAIM.finditer(text):
        if _owning_milestone(text, match.start()) != "K9":
            continue                    # a claim about some other milestone
        if _quoted(text, match):
            continue                    # named, not asserted
        clause = re.split(r"[.;:|]",
                          text[max(0, match.start() - 90):match.start()])[-1]
        if _DENIAL.search(clause):
            continue                    # the clause denies it
        offenders.append(match.group(0))
    return offenders


@pytest.mark.parametrize(
    "surface",
    STATUS_SURFACES + ("CLAUDE.md", "docs/native_integer_tensors_design.md",
                       "docs/release_history.md"))
def test_no_surface_credits_k9_with_zero_production_code(surface):
    """The permanent guardrail. K5 through K8 really were production-code
    free and their records say so; K9 was not, and no surface may say it
    was while the seven repaired translation units are part of it."""
    offenders = _k9_zero_production_claims(_flat(_read(surface)))
    assert offenders == [], f"{surface}: {offenders[:3]}"


def test_the_seven_repaired_translation_units_still_carry_the_repair():
    """The ban's live premise, checked rather than assumed.

    If the repair were ever reverted, the ban above would be enforcing a
    fact that had stopped being true — so the premise is verified here:
    each of the seven units exists and carries at least one explicit
    ``Int64`` dispatch arm, read from comment-stripped code so a comment
    saying "Int64 is handled elsewhere" cannot satisfy it.

    The *structural* regression — every dtype-dispatch switch exhaustive
    and no ``default:`` label anywhere — is K1's, and lives with the
    barriers in ``tests/test_native_integer_barriers.py``. This is the
    closure half: the files are named, present, and repaired."""
    for relative in K9_REPAIRED_TRANSLATION_UNITS:
        path = REPO_ROOT / relative
        assert path.is_file(), relative
        code = _cpp_code_only(path.read_text(encoding="utf-8"))
        arms = re.findall(r"case\s+tf::Dtype::Int64\s*:", code)
        assert arms, f"{relative} carries no explicit Int64 dispatch arm"
        assert "default:" not in code, (
            f"{relative} gained a default: label, which would hide the "
            f"next enumerator from the compiler")
    # ...and the phase's own unit, which established the idiom, is not one
    # of the seven: it was written exhaustive at K3.
    assert "cpp/src/indexing.cpp" not in K9_REPAIRED_TRANSLATION_UNITS


def test_at_least_one_surface_records_the_k9_repair_positively():
    """Not-contradicted is weaker than stated. Somewhere in the current
    documentation **both** repairs must be discoverable, or a reader
    auditing "what did K9 change?" would come away with half the answer."""
    for pattern, label in (
            (r"-Wswitch|Int64 arm|Int64. arms", "warning-cleanliness"),
            (r"fresh.storage ownership|_uninitialized|"
             r"NativeTensorCore\.from_array", "ownership/lifecycle")):
        for surface in STATUS_SURFACES + (
                "CLAUDE.md", "docs/native_integer_tensors_design.md"):
            text = _flat(_read(surface))
            if re.search(rf"K9\b.{{0,900}}?({pattern})", text, re.S) \
                    or re.search(rf"({pattern}).{{0,900}}?\bK9\b", text, re.S):
                break
        else:
            raise AssertionError(
                f"no current surface records K9's {label} repair")


# The **second** false chronology shape, which only became false when the
# second repair landed: "K9's one production repair". Banned separately
# from the zero-production claim above, because a surface can be perfectly
# honest that K9 changed production code and still undercount it.
# Longest alternatives first: ``exactly one`` must match as a whole rather
# than leaving "exactly" outside the span, or the quotation escape below
# would see a quote that does not bracket the match.
_SINGLE_REPAIR_CLAIM = re.compile(
    r"\b(?:exactly\s+one|a\s+single|its\s+one|the\s+one|only|one)\b"
    r"[^.;|]{0,60}?\b(?:production|executable|behaviou?r[- ]preserving)\b"
    r"[^.;|]{0,40}?\brepair\b"
    r"|\brepair\b[^.;|]{0,40}?\bis\s+(?:the\s+)?only\b", re.I)


def _inside_quotes(text, match):
    """True when the match lies **within** a quoted span, rather than being
    exactly bracketed by one.

    ``_quoted`` above asks the narrower question, which is right for a
    fixed phrase like "zero production code" but wrong here: the sentence
    this repository writes is `K9's one production repair`, where the
    opening quote sits before the possessive and the match begins two words
    later. A span search on the same clause answers it correctly."""
    before = text[max(0, match.start() - 120):match.start()]
    after = text[match.end():match.end() + 120]
    opening = re.search(r'["“](?=[^"”.;|]*$)', before)
    closing = re.match(r'[^"“.;|]*["”]', after)
    return bool(opening and closing)


def _k9_single_repair_claims(text):
    """Every asserted "K9 carries exactly one production repair" claim.

    Attribution is the zero-production scan's, on purpose: I10's genuinely
    single production repair is described in this repository in exactly
    those words, and a scan that could not tell the two milestones apart
    would force that true sentence to be rewritten."""
    offenders = []
    for match in _SINGLE_REPAIR_CLAIM.finditer(text):
        if _owning_milestone(text, match.start()) != "K9":
            continue
        if _inside_quotes(text, match):
            continue                    # named, not asserted
        clause = re.split(r"[.;:|]",
                          text[max(0, match.start() - 90):match.start()])[-1]
        if _DENIAL.search(clause):
            continue
        offenders.append(match.group(0)[:100])
    return offenders


@pytest.mark.parametrize(
    "surface",
    STATUS_SURFACES + ("CLAUDE.md", "docs/native_integer_tensors_design.md",
                       "docs/release_history.md"))
def test_no_surface_credits_k9_with_a_single_production_repair(surface):
    """The permanent guardrail for the count. K9 carries two repairs while
    both premises below hold, and no surface may say it carries one."""
    offenders = _k9_single_repair_claims(_flat(_read(surface)))
    assert offenders == [], f"{surface}: {offenders[:3]}"


def test_the_python_lifecycle_repair_is_present(monkeypatch):
    """The second ban's live premise, checked rather than assumed — the
    mirror of the seven-translation-unit check above.

    ``NativeTensorCore.from_array``, ``zeros``, and ``_uninitialized`` must
    each close their freshly allocated storage when publication raises,
    including under ``BaseException``. Proved **behaviourally** rather than
    by reading the source, and with the storage retained strongly so the
    result cannot come from ``__del__``."""
    if not cpp.is_available():
        pytest.skip("native backend not built")

    class Abort(BaseException):
        pass

    builders = (
        lambda: cpp.NativeTensorCore.from_array(np.zeros((2, 2))),
        lambda: cpp.NativeTensorCore.zeros((2, 2)),
        lambda: cpp.NativeTensorCore._uninitialized((2, 2)),
    )
    original_init = cpp.NativeStorage.__init__
    for build in builders:
        retained = []

        def retaining_init(storage, *args, **kwargs):
            original_init(storage, *args, **kwargs)
            retained.append(storage)

        def failing_view(storage, dims):
            raise Abort("injected")

        cpp.NativeStorage.__init__ = retaining_init
        monkeypatch.setattr(cpp, "_contiguous_view", failing_view)
        try:
            with pytest.raises(Abort):
                build()
        finally:
            cpp.NativeStorage.__init__ = original_init
            monkeypatch.undo()
        assert len(retained) == 1, retained
        assert retained[0]._handle is None, (
            "a fresh storage survived a failed publication: K9's "
            "lifecycle repair is not in force")
    # ...and the three named methods really are the ones that were repaired.
    for name in ("from_array", "zeros", "_uninitialized"):
        assert hasattr(cpp.NativeTensorCore, name), name


def test_the_zero_production_scan_detects_the_claim_it_bans():
    """The scanner's negative control, on temporary strings only.

    Every sentence below is one that was actually written about the K9
    candidate, or a close variant, and each must be caught. The positive
    controls afterwards prove the scan is not simply always-firing: an
    accurate K9 sentence, a *quoted* mention of the banned wording, and
    the same claims made about K5–K8 — where they are true — all pass."""
    for caught in (
        "K9 closed the phase and added proof only",
        "K9 closed the phase with that boundary intact, adding proof only",
        "K9 added no production code",
        "K9 is the closure milestone, and it added zero production code",
        # The ladder row as it was actually written, cells and all.
        "| **K9** | Cross-platform validation and Phase-K closure | "
        "**Complete.** Zero production code: no export, no public name, "
        "no CTest, no example, no benchmark. |",
        "K9 was production-code-free",
        "the closure milestone K9 adds proof only to the tree",
        "K9 touched no production code",
        # The wording an earlier draft of this scan let through: the only
        # negation in range belongs to the *previous* sentence.
        "Phase K was approved after Phase J closed, so it is not "
        "carried-over roadmap work. K9 added no production code: it is "
        "the cross-platform validation and closure milestone.",
    ):
        assert _k9_zero_production_claims(caught), caught
    for allowed in (
        # The corrected forms.
        "K9 added no new capability, and it was not a proof-only milestone",
        "K9 closed the phase and added no new capability — but it was not "
        "proof-only",
        "K9 added no new capability, but it was not production-code-free",
        # A quoted mention: this is how the prohibition itself is written.
        'never describe K9 as "zero production code" or "proof only"',
        'no surface may credit K9 with "no production code"',
        # K5 through K8, where the claim is true and must stay sayable.
        "K5 is the compatibility proof and added zero production code",
        "K7 added no production code either",
        "K8 is the benchmark milestone and added zero production code",
        # Unrelated milestone wording from other phases.
        "J9 changed no production code and added no export",
        "F6 is an integration proof only, with no numerical behavior",
    ):
        assert _k9_zero_production_claims(allowed) == [], (
            allowed, _k9_zero_production_claims(allowed))
    # The quotation escape is not a blanket escape: an assertive sentence
    # that merely happens to sit near a quote mark is still caught.
    assert _k9_zero_production_claims(
        'The "closure" milestone K9 added no production code at all')
    # Attribution is by *nearest preceding* milestone, which is what makes
    # a long ladder row and a mixed chronology both readable.
    assert _owning_milestone("| **K9** | purpose | body body body", 30) == "K9"
    assert _owning_milestone("K5 did this. K9 did that. Then more", 30) == "K9"
    assert _owning_milestone("K9 did that. K5 did this. Then more", 30) == "K5"
    assert _owning_milestone("no milestone is named here at all", 30) is None
    assert _owning_milestone("K9 far away." + " " * 500 + "claim", 520) is None
    # ...and the premise check is non-vacuous: a unit with its arms
    # stripped is reported by the same reader the live check uses.
    stripped = _cpp_code_only(
        "switch (tf::dispatch_dtype({a})) {\n"
        "    // case tf::Dtype::Int64: return;\n"
        "    case tf::Dtype::Float64: break;\n"
        "}\n")
    assert not re.findall(r"case\s+tf::Dtype::Int64\s*:", stripped)


def test_the_single_repair_scan_detects_the_claim_it_bans():
    """The count scanner's negative control, on temporary strings only.

    Every sentence below is one a surface could plausibly regress to once
    the second repair exists. The positive controls prove the scan is not
    always-firing: the corrected two-repair wording, a quoted mention of
    the banned phrasing, and — importantly — I10's genuinely single
    production repair, which is described in this repository in exactly
    these words and must stay sayable."""
    for caught in (
        "K9's one production repair is the -Wswitch change",
        "K9 carries exactly one production repair",
        "K9 closed the phase with a single production repair",
        "K9 added the only production repair of the phase",
        "K9 carries one behaviour-preserving executable repair",
        "| **K9** | closure | Complete. Its one production repair is the "
        "switch-exhaustiveness change. |",
    ):
        assert _k9_single_repair_claims(_flat(caught)), caught
    for allowed in (
        # The corrected forms.
        "K9 carries two behaviour-preserving production repairs",
        "K9's two production repairs are the C++ switch change and the "
        "NativeTensorCore ownership guard",
        # Quoted mentions: how the prohibition itself is written, both with
        # the quote bracketing the phrase and with it opening earlier.
        'no surface may say "K9\'s one production repair"',
        'K9: never describe it as carrying "exactly one production repair"',
        # Other milestones, where a single repair is the truth.
        "I10's one production repair, pinned so it cannot regress",
        "K5 found one real defect, repaired before the milestone landed",
        # A denial in the claim's own clause.
        "K9 does not carry only one production repair",
    ):
        assert _k9_single_repair_claims(_flat(allowed)) == [], (
            allowed, _k9_single_repair_claims(_flat(allowed)))


# ===========================================================================
# 2c. The device contract — §25, pinned by signature rather than by prose
# ===========================================================================
#
# §25 used to say "No `device` argument exists anywhere and none may be
# added", which was simply false about the tree it governs: the inherited
# floating constructors have carried a defaulted, validated `device="cpu"`
# metadata tag since Phase I. The true rule is narrower and checkable:
# **Phase K added no device argument and widened none**, and "cpu" is
# still the only value anything accepts.
#
# Both halves are asserted, because a guard that only checked the second
# would pass just as happily on a tree where the first had been quietly
# deleted — and then the old absolute sentence would be true again for the
# wrong reason.


def _resolve(owner, name):
    from tensorforge.experimental import NativeTensor

    holder = {"cpp.NativeStorage": cpp.NativeStorage,
              "cpp.NativeTensorCore": cpp.NativeTensorCore,
              "NativeTensor": NativeTensor}[owner]
    return getattr(holder, name)


@pytest.mark.parametrize("owner,name", INHERITED_DEVICE_CONSTRUCTORS)
def test_the_inherited_floating_constructors_do_carry_device_cpu(owner, name):
    """The half the old absolute sentence denied. Each inherited
    constructor takes ``device`` with a literal ``"cpu"`` default — a
    CPU-only metadata tag, not a selector."""
    parameters = inspect.signature(_resolve(owner, name)).parameters
    assert "device" in parameters, (owner, name, sorted(parameters))
    assert parameters["device"].default == FINAL_DEFAULT_DEVICE, (
        owner, name, parameters["device"].default)


@pytest.mark.parametrize("owner,name", PHASE_K_DEVICE_FREE_METHODS)
def test_no_phase_k_public_method_added_a_device_parameter(owner, name):
    """The half that is Phase K's own commitment: the construction door
    and both operations carry no ``device`` parameter at all, on either
    layer."""
    parameters = inspect.signature(_resolve(owner, name)).parameters
    assert "device" not in parameters, (owner, name, sorted(parameters))


def test_the_device_registry_and_validator_still_accept_only_cpu():
    """A tag with exactly one legal value is not a device system. The
    validator is the authority and it is asked directly."""
    assert cpp.SUPPORTED_DEVICES == FINAL_SUPPORTED_DEVICES
    assert cpp.backend_info()["supported_devices"] == FINAL_SUPPORTED_DEVICES
    assert cpp.normalize_device(None) == FINAL_DEFAULT_DEVICE
    assert cpp.normalize_device("cpu") == FINAL_DEFAULT_DEVICE
    for rejected in ("cuda", "cuda:0", "CUDA", "gpu", "mps", "xpu", "amp",
                     "cpu:0", "", 0, 1.0, ["cpu"]):
        with pytest.raises((ValueError, TypeError)):
            cpp.normalize_device(rejected)


@needs_backend
def test_no_device_movement_operation_exists_on_either_tensor_layer():
    """The tag moves nothing, and nothing moves it: no transfer method
    exists to be called, at any dtype including ``int64``."""
    from tensorforge.experimental import NativeTensor

    tensor_methods = _class_method_names(
        "src/tensorforge/experimental/native_tensor.py", "NativeTensor")
    core_methods = _class_method_names(
        "src/tensorforge/backends/cpp.py", "NativeTensorCore")
    for banned in ("to", "cpu", "cuda", "device_", "pin_memory", "to_device",
                   "map_location"):
        assert banned not in tensor_methods, banned
        assert banned not in core_methods, banned
    probe = _int64_probe()
    try:
        for banned in ("to", "cpu", "cuda", "pin_memory", "to_device"):
            assert not hasattr(probe, banned), banned
        # The read-only tag *does* exist and answers "cpu" at ``int64``
        # too. Asserted rather than ignored: it is the other half of why
        # the old absolute sentence was false, and a guard that only
        # checked the absences would pass on a tree where the property
        # had been deleted — and then the false sentence would be true
        # again for the wrong reason.
        assert probe.device == FINAL_DEFAULT_DEVICE
    finally:
        probe.close()


# The absolute sentence, as one authority shared by the scan and its own
# control, so a weakening in either place fails in the other. ``_flat``
# has already stripped the backticks by the time this runs.
_DEVICE_CLAIM = re.compile(
    r"\bno\s+device\s+(?:argument|parameter|keyword)s?\b([^.;]{0,60}?)"
    r"\banywhere\b", re.I)

# A scope marker that turns the sentence from a false absolute into an
# accurate statement about what a phase or milestone *added*. It must sit
# in the claim's own span or immediately before it — "no device argument
# **was added** anywhere" and "**Phase K adds** no device argument
# anywhere" are scoped, while "no device argument exists anywhere, and
# none may be added" is not: there the scope-shaped word arrives only
# after the absolute claim has already been made.
_DEVICE_SCOPE = re.compile(
    r"\b(added|adds|add|gains?|gained|new|introduced|widened|"
    r"Phase\s+[A-K]\b|[A-K]\d\b|milestone)\b", re.I)
# ...and the one scope that legitimately trails the claim: an explicit
# "in the phase" restriction.
_DEVICE_SCOPE_AFTER = re.compile(
    r"^[^.;]{0,30}?\bin (?:the|this) phase\b"
    r"|^[^.;]{0,30}?\bin Phase [A-K]\b", re.I)


def _absolute_device_claims(text):
    """Every **absolute** "no device argument anywhere" claim — the false
    sentence — leaving the phase- and milestone-scoped forms alone,
    because those are accurate and the repository must keep saying them."""
    offenders = []
    for match in _DEVICE_CLAIM.finditer(text):
        if _DEVICE_SCOPE.search(match.group(1)):
            continue
        if _DEVICE_SCOPE.search(text[max(0, match.start() - 45):
                                     match.start()]):
            continue
        if _DEVICE_SCOPE_AFTER.search(text[match.end():match.end() + 45]):
            continue
        offenders.append(match.group(0))
    return offenders


@pytest.mark.parametrize(
    "surface",
    ("CLAUDE.md", "docs/native_integer_tensors_design.md")
    + STATUS_SURFACES)
def test_the_absolute_no_device_argument_claim_cannot_return(surface):
    """The specific false sentence, banned by shape rather than by exact
    wording, on the two authorities that carried it and on every current
    status surface."""
    offenders = _absolute_device_claims(_flat(_read(surface)))
    assert offenders == [], f"{surface}: {offenders[:3]}"


def test_the_absolute_device_scan_can_actually_fail():
    """Negative control, through the same compiled pattern the scan uses."""
    for caught in (
        "No `device` argument exists anywhere and none may be added.",
        "no device parameter exists anywhere",
        "There is no `device` argument anywhere in this codebase.",
        "no `device` argument anywhere; no astype / to / .int()",
    ):
        assert _absolute_device_claims(_flat(caught)), caught
    for allowed in (
        # The corrected form.
        "Phase K adds no `device` argument, and none may be added.",
        # Phase-scoped history, which is accurate and stays.
        "No `device` argument anywhere in the phase.",
        "no `device` argument was added anywhere",
        "K9 added no device parameter anywhere",
        "no new `device` argument anywhere",
        # The real signatures, described honestly.
        "The inherited floating constructors carry a validated "
        "`device=\"cpu\"` metadata tag.",
    ):
        assert _absolute_device_claims(_flat(allowed)) == [], (
            allowed, _absolute_device_claims(_flat(allowed)))
    # The design must positively record the inherited parameter, so a
    # reader is told the true rule rather than merely not told the false
    # one.
    section = _flat(_section(_design(), 25))
    assert re.search(r"device=.?cpu", section), section[:400]
    assert "from_int64_array" in section
    assert "normalize_device" in section


@pytest.mark.parametrize("owner,name", PRIVATE_DEVICE_INGRESS_HELPERS)
def test_the_private_k2_ingress_helpers_do_carry_device_cpu(owner, name):
    """The third device fact, and the premise of the scoped sentence below.

    Phase K's two private ``int64`` ingress helpers inherit the same
    defaulted, ``normalize_device``-validated ``device="cpu"`` metadata tag
    the floating constructors carry. They are private, they are only ever
    called with the default, and the tag selects and transfers nothing —
    but they *do* have the parameter, which is exactly why the unqualified
    "everything added since carries no device parameter" is false.

    Pinned rather than removed on purpose: deleting the parameter to make a
    sentence true would be fixing the tree to match the prose."""
    parameters = inspect.signature(_resolve(owner, name)).parameters
    assert "device" in parameters, (owner, name, sorted(parameters))
    assert parameters["device"].default == FINAL_DEFAULT_DEVICE, (
        owner, name, parameters["device"].default)


# The second false device shape — an *unscoped* "everything added since
# carries no device parameter". It survived the absolute-claim scan above
# because it never says "anywhere", and it is false about precisely the two
# private helpers pinned overhead. The repaired sentence says
# "everything **public** added since", so the qualifier that makes it true
# must sit on the subject itself.
_EVERYTHING_DEVICE_CLAIM = re.compile(
    r"\bevery(?:thing|\s+\w+){0,1}?\b(?P<qualifier>[^.;]{0,30}?)"
    r"\b(?:added|since)\b[^.;]{0,220}?"
    r"\bno\b[^.;]{0,30}?\bdevice\b\s*(?:parameter|argument|keyword)s?\b",
    re.I | re.S)
_PUBLIC_SCOPE = re.compile(r"\bpublic(?:ly)?\b|\bnew\b", re.I)


def _unscoped_everything_device_claims(text):
    """Every "everything added since carries no ``device`` parameter" claim
    whose subject is **not** narrowed to the public surface (or to what was
    newly *added*, which is the other true reading).

    The qualifier is read from the span between the subject and
    ``added``/``since`` rather than from the whole sentence, because the
    false sentence itself contains the words "Phase-K public methods"
    further along — a whole-sentence search for "public" would have
    accepted the exact wording this exists to reject."""
    offenders = []
    for match in _EVERYTHING_DEVICE_CLAIM.finditer(text):
        if _PUBLIC_SCOPE.search(match.group("qualifier")):
            continue
        offenders.append(match.group(0)[:120])
    return offenders


@pytest.mark.parametrize(
    "surface",
    ("CLAUDE.md", "docs/native_integer_tensors_design.md")
    + STATUS_SURFACES)
def test_the_unscoped_everything_device_claim_cannot_return(surface):
    """The exact overbroad shape CLAUDE.md carried until the K9 repair,
    banned by shape on every surface that could carry it again."""
    offenders = _unscoped_everything_device_claims(_flat(_read(surface)))
    assert offenders == [], f"{surface}: {offenders[:3]}"


def test_the_everything_device_scan_can_actually_fail():
    """Negative control through the same compiled pattern, including the
    exact sentence the repository shipped and the minimal form the audit
    named."""
    for caught in (
        # The audit's minimal statement of the falsehood.
        "Everything added since K2 carries no device parameter.",
        # The sentence CLAUDE.md actually carried, flattened as the scan
        # sees it — note that it contains the word "public" *downstream*,
        # which is why the qualifier is read from the subject's own span.
        "Everything added since — including all three Phase-K public "
        "methods, from_int64_array, argmax, and index_select — carries no "
        "device parameter, and none may gain one.",
        "everything added since Phase I carries no device argument",
        "Every method added since K2 carries no device keyword",
    ):
        assert _unscoped_everything_device_claims(_flat(caught)), caught
    for allowed in (
        # The repaired form: the qualifier sits on the subject.
        "Everything public added since — including all three Phase-K "
        "public methods, from_int64_array, argmax, and index_select — "
        "carries no device parameter, and none may gain one.",
        "Everything public added since K2 carries no device parameter.",
        "Every public method added since K2 carries no device argument",
        # The "no *new* device argument" reading, which is also true.
        "Everything new added since K2 carries no device parameter",
        # The positive statement of the truth this scan protects.
        "The two private K2 ingress helpers do carry the inherited "
        "device=cpu metadata tag.",
    ):
        assert _unscoped_everything_device_claims(_flat(allowed)) == [], (
            allowed, _unscoped_everything_device_claims(_flat(allowed)))


def test_claude_md_records_the_private_ingress_helpers_device_tag():
    """Not-contradicted is weaker than stated: the operating rules must say
    positively that the two private helpers carry the inherited tag, or a
    reader is left with a rule that is merely no longer false rather than
    one that is true and complete."""
    text = _flat(_read("CLAUDE.md"))
    assert "NativeStorage._from_int64_array" in text
    assert "NativeTensorCore._from_int64_array" in text
    # ...in the device paragraph, beside the metadata tag itself.
    window = re.search(
        r"NativeStorage\._from_int64_array.{0,400}", text, re.S)
    assert window is not None
    assert re.search(r"device=.?cpu", window.group(0)), window.group(0)[:300]


# ===========================================================================
# 3. The final capability boundary — taxonomy B, permanently
# ===========================================================================

def test_the_public_registries_closed_exactly_where_the_design_says():
    """The capability, stated as literals. ``SUPPORTED_DTYPES`` is exactly
    what Phase I created and Phases J and K left; ``INDEX_DTYPES`` is the
    phase's one registry addition."""
    assert cpp.SUPPORTED_DTYPES == FINAL_SUPPORTED_DTYPES
    assert cpp.INDEX_DTYPES == FINAL_INDEX_DTYPES
    assert cpp.SUPPORTED_DEVICES == FINAL_SUPPORTED_DEVICES
    assert cpp.UNSUPPORTED == FINAL_UNSUPPORTED
    assert cpp.RAW_KERNEL_DTYPES == FINAL_RAW_KERNEL_DTYPES
    # Order is contractual: float64 first, because it is the default.
    assert cpp.SUPPORTED_DTYPES[0] == FINAL_DEFAULT_DTYPE
    # The two dtype registries are disjoint — an index dtype is never a
    # compute dtype — and neither overlaps the unsupported row.
    assert not set(cpp.SUPPORTED_DTYPES) & set(cpp.INDEX_DTYPES)
    assert not set(cpp.SUPPORTED_DTYPES) & set(cpp.UNSUPPORTED)
    assert not set(cpp.INDEX_DTYPES) & set(cpp.UNSUPPORTED)
    # The raw-kernel registry is a *narrower* statement, permanently.
    assert set(cpp.RAW_KERNEL_DTYPES) < set(cpp.SUPPORTED_DTYPES)
    # The no-drift guard, generalized at K2 rather than deleted: the
    # representation table is exactly the union of the two public rows.
    assert (set(cpp._DTYPE_CODES)
            == set(cpp.SUPPORTED_DTYPES) | set(cpp.INDEX_DTYPES))
    assert cpp._DTYPE_CODES["int64"] == ABI_INT64_CODE
    assert cpp._DTYPE_ITEM_SIZES["int64"] == INT64_ITEM_SIZE


def test_backend_info_reports_the_closed_boundary():
    info = cpp.backend_info()
    assert info["supported_dtypes"] == FINAL_SUPPORTED_DTYPES
    assert info["index_dtypes"] == FINAL_INDEX_DTYPES
    assert info["dtype"] == FINAL_DEFAULT_DTYPE
    assert info["raw_kernel_dtypes"] == FINAL_RAW_KERNEL_DTYPES
    assert info["supported_devices"] == FINAL_SUPPORTED_DEVICES
    assert info["unsupported"] == FINAL_UNSUPPORTED
    assert info["stable_framework_integration"] is False


def test_normalize_dtype_permanently_rejects_int64_and_every_other_width():
    assert cpp.normalize_dtype(None) == FINAL_DEFAULT_DTYPE
    assert cpp.normalize_dtype("float64") == "float64"
    assert cpp.normalize_dtype("float32") == "float32"
    with pytest.raises(ValueError):
        cpp.normalize_dtype("int64")
    for absent in ("int32", "int16", "int8", "uint8", "uint64", "bool",
                   "float16", "bfloat16", "float128", "complex64"):
        with pytest.raises((ValueError, TypeError)):
            cpp.normalize_dtype(absent)
    for device in ("cuda", "cuda:0", "gpu", "mps", "amp"):
        with pytest.raises((ValueError, TypeError)):
            cpp.normalize_device(device)


def test_the_index_registry_gate_is_the_one_the_door_asks():
    """``_normalize_index_dtype`` is the registry gate for the one public
    door: no default, ``None`` rejected, the registry value accepted."""
    assert cpp._normalize_index_dtype("int64") == "int64"
    with pytest.raises((ValueError, TypeError)):
        cpp._normalize_index_dtype(None)
    with pytest.raises((ValueError, TypeError)):
        cpp._normalize_index_dtype("float64")


@needs_backend
def test_no_generic_constructor_accepts_int64_after_closure():
    """The property the whole phase is built around, re-proved at the
    closed boundary: the one integer door is a **new** name, and no widened
    old one exists."""
    from tensorforge.experimental import NativeTensor

    values = np.zeros(3, dtype=np.int64)
    with pytest.raises((ValueError, TypeError)):
        cpp.NativeStorage(3, dtype="int64")
    with pytest.raises((ValueError, TypeError)):
        cpp.NativeStorage.from_array(values, dtype="int64")
    with pytest.raises((ValueError, TypeError)):
        cpp.NativeTensorCore.from_array(values, dtype="int64")
    with pytest.raises((ValueError, TypeError)):
        cpp.NativeTensorCore.zeros((3,), dtype="int64")
    with pytest.raises((ValueError, TypeError)):
        cpp.NativeTensorCore.full((3,), 1, dtype="int64")
    with pytest.raises((ValueError, TypeError)):
        NativeTensor.from_array(values, dtype="int64")
    with pytest.raises((ValueError, TypeError)):
        NativeTensor.zeros((3,), dtype="int64")
    with pytest.raises((ValueError, TypeError)):
        NativeTensor.full((3,), 1, dtype="int64")


@needs_backend
def test_the_one_public_door_still_works_and_is_exact():
    """The positive half the rejections above need: the door exists, takes
    exactly a native-order ``int64`` ndarray, and reproduces every value
    exactly — including the ones beyond float64's 2**53 exact range."""
    from tensorforge.experimental import NativeTensor

    exact = [-(2**63), -(2**53) - 1, -1, 0, 1, 2**53 + 1, 2**63 - 1]
    tensor = NativeTensor.from_int64_array(np.array(exact, dtype=np.int64))
    try:
        assert tensor.dtype == "int64"
        assert tensor.requires_grad is False
        assert tensor.to_numpy().tolist() == exact
        assert tensor.tolist() == exact
    finally:
        tensor.close()
    # ...and it converts nothing: a float array is rejected even when its
    # values are integral, because ingress is exact rather than a cast.
    with pytest.raises(TypeError):
        NativeTensor.from_int64_array(np.array([1.0, 2.0]))
    with pytest.raises(TypeError):
        NativeTensor.from_int64_array(np.array([1, 2], dtype=np.int32))
    with pytest.raises(TypeError):
        NativeTensor.from_int64_array([1, 2])


# ===========================================================================
# 4. The phase's public delta — five methods, one door, __all__ unmoved
# ===========================================================================

def test_the_experimental_export_surface_is_exactly_twenty_five():
    import tensorforge.experimental as experimental

    assert len(experimental.__all__) == FINAL_EXPERIMENTAL_EXPORTS
    assert len(set(experimental.__all__)) == FINAL_EXPERIMENTAL_EXPORTS
    for name in experimental.__all__:
        assert hasattr(experimental, name), name
    # No pipeline-, integer-, or indexing-shaped name arrived at closure.
    for absent in ("NativeIntTensor", "NativeIndexTensor", "IntegerTensor",
                   "from_int64_array", "argmax", "index_select"):
        assert absent not in experimental.__all__, absent


def test_the_five_tensor_methods_exist_and_are_attributed():
    """The phase's whole public delta, as methods on ``NativeTensor``. The
    map pins the attribution; the live class pins the existence."""
    from tensorforge.experimental import NativeTensor

    for name, milestone in PHASE_K_TENSOR_METHODS.items():
        assert hasattr(NativeTensor, name), (name, milestone)
    assert set(PHASE_K_TENSOR_METHODS.values()) == {"K2", "K3", "K4"}
    # Every name Phase K added carries **no `device` parameter at all**,
    # and the door carries no `dtype` either — the dtype is in its name,
    # so it cannot be omitted, mistyped, or contradicted.
    #
    # This is the sharp form of §25's rule. The pre-Phase-K floating
    # constructors do carry a defaulted, validated ``device="cpu"``
    # metadata tag inherited from Phase I; it accepts only ``"cpu"``,
    # transfers nothing, and is not a device selector. What matters at
    # closure is that **Phase K widened none of that and added none of
    # it**, which is what these exact signatures pin.
    door = inspect.signature(NativeTensor.from_int64_array)
    assert set(door.parameters) == {"values", "requires_grad"}, (
        sorted(door.parameters))
    for operation in ("argmax", "index_select"):
        parameters = set(
            inspect.signature(getattr(NativeTensor, operation)).parameters)
        assert "device" not in parameters, (operation, sorted(parameters))
        assert "dtype" not in parameters, (operation, sorted(parameters))
        # ...and the Core layer's mirror carries neither either.
        core = set(inspect.signature(
            getattr(cpp.NativeTensorCore, operation)).parameters)
        assert "device" not in core and "dtype" not in core, (operation,
                                                              sorted(core))
    # argmax and index_select are Core operations too; construction is not.
    assert hasattr(cpp.NativeTensorCore, "argmax")
    assert hasattr(cpp.NativeTensorCore, "index_select")
    # No device beyond the one supported tag is reachable, on any path.
    for absent in ("cuda", "cuda:0", "gpu", "mps", "amp", "xpu"):
        with pytest.raises((ValueError, TypeError)):
            cpp.normalize_device(absent)


def test_the_storage_and_core_ingress_helpers_stayed_private():
    """One door means one: the Storage and Core integer ingress routes are
    underscore-private, and no public ``from_int64*`` name exists on
    either layer."""
    for relative, class_name in (
            ("src/tensorforge/backends/cpp.py", "NativeStorage"),
            ("src/tensorforge/backends/cpp.py", "NativeTensorCore")):
        methods = _class_method_names(relative, class_name)
        public_integer = {name for name in methods
                          if "int64" in name and not name.startswith("_")}
        assert public_integer == set(), (class_name, sorted(public_integer))
        assert "_from_int64_array" in methods, class_name


def test_the_operation_registries_are_exactly_the_closed_tuples():
    """Pinned as exact equality against independent copies: ``argmax`` and
    ``index_select`` are TENSOR_CORE_OPS' two Phase-K entries, and
    AUTOGRAD_OPS gained nothing — an index has no derivative, and the
    ``index_select`` backward is a separately approved milestone that does
    not exist."""
    assert cpp.TENSOR_CORE_OPS == FINAL_TENSOR_CORE_OPS
    assert cpp.AUTOGRAD_OPS == FINAL_AUTOGRAD_OPS
    assert "argmax" not in cpp.AUTOGRAD_OPS
    assert "index_select" not in cpp.AUTOGRAD_OPS
    # The frozen five stay frozen.
    assert cpp.TENSOR_CORE_KERNELS == ("relu", "add", "subtract",
                                       "multiply", "matmul")


# ===========================================================================
# 5. The final C ABI — 56 symbols, source and library agreeing
# ===========================================================================

def test_the_source_exports_exactly_fifty_six_symbols():
    """Stated as arithmetic rather than as a bare number, so the facts stay
    separable: Phase J closed at 54, K3 and K4 added one each, and 56 is
    the maximum the design authorizes."""
    exports = _source_exports()
    assert len(exports) == FINAL_EXPORT_COUNT, sorted(exports)
    for name, milestone in PHASE_K_ADDED_EXPORTS.items():
        assert name in exports, (name, milestone)
    assert len(exports - set(PHASE_K_ADDED_EXPORTS)) == PHASE_J_EXPORT_COUNT


def test_no_banned_export_shape_exists():
    """The shapes an integer-indexing phase is most tempted to add, banned
    by name over the live export inventory."""
    for name in sorted(_source_exports()):
        assert not BANNED_EXPORT_SHAPES.search(name), name
    # ...and the ban itself is non-vacuous, in both directions. The
    # ``maxpool2d`` entries are the reason the pattern ends in a boundary:
    # they are Phase D's shipped pooling exports, not a scalar maximum.
    for caught in ("tf_core_argmin", "tf_core_gather", "tf_core_scatter",
                   "tf_core_scatter_add", "tf_core_max", "tf_core_embedding",
                   "tf_cast", "tf_core_max_with_indices", "tf_core_topk"):
        assert BANNED_EXPORT_SHAPES.search(caught), caught
    for allowed in ("tf_core_argmax", "tf_core_index_select",
                    "tf_core_matmul", "tf_core_maxpool2d_forward",
                    "tf_core_maxpool2d_backward"):
        assert not BANNED_EXPORT_SHAPES.search(allowed), allowed


@needs_backend
def test_the_built_library_exports_exactly_what_the_source_declares():
    library = cpp._require_library()
    for name in sorted(_source_exports()):
        assert hasattr(library, name), name


def test_the_checked_kernel_list_closed_at_thirty_eight():
    """The errcheck-hooked exports: Phase J's 36 plus the two Phase-K
    operations, each of which self-validates and reports through the
    thread-local error slot."""
    assert len(cpp._CHECKED_KERNELS) == FINAL_CHECKED_KERNELS
    assert len(set(cpp._CHECKED_KERNELS)) == FINAL_CHECKED_KERNELS
    for name in PHASE_K_ADDED_EXPORTS:
        assert name in cpp._CHECKED_KERNELS, name
    # Every checked kernel is a real export.
    missing = set(cpp._CHECKED_KERNELS) - _source_exports()
    assert not missing, sorted(missing)


def test_the_export_reader_can_actually_fail(tmp_path):
    """The inventory reader's negative control, on a temporary copy: it
    needs no repository, and it notices both an added and a removed
    symbol."""
    staging = tmp_path / "cpp" / "src"
    staging.mkdir(parents=True)
    for source in sorted((REPO_ROOT / "cpp" / "src").glob("*.cpp")):
        (staging / source.name).write_bytes(source.read_bytes())
    assert not (tmp_path / ".git").exists()

    def read(root):
        names = set()
        for source in sorted(root.glob("*.cpp")):
            names.update(re.findall(
                r"TF_EXPORT[^;{]*?\b(tf_[a-z0-9_]+)\s*\(",
                source.read_text(encoding="utf-8"), re.S))
        return names

    assert read(staging) == _source_exports()
    victim = next(iter(sorted(staging.glob("*.cpp"))))
    victim.write_text(victim.read_text(encoding="utf-8")
                      + "\nTF_EXPORT void tf_invented_symbol(void) {}\n",
                      encoding="utf-8")
    mutated = read(staging)
    assert len(mutated) == FINAL_EXPORT_COUNT + 1
    assert "tf_invented_symbol" in mutated


# ===========================================================================
# 6. The native test, source, example, and benchmark inventories
# ===========================================================================

def _registered_ctests():
    return re.findall(r"add_test\s*\(\s*NAME\s+(\w+)",
                      _read("cpp/CMakeLists.txt"))


def test_the_ctest_inventory_closed_at_twenty_seven():
    names = _registered_ctests()
    assert len(names) == FINAL_CTEST_COUNT, names
    assert len(set(names)) == len(names), "a CTest name is registered twice"
    for name, milestone in PHASE_K_ADDED_CTESTS.items():
        assert name in names, (name, milestone)
    assert len([n for n in names
                if n not in PHASE_K_ADDED_CTESTS]) == PHASE_J_CTEST_COUNT
    # Every registered test has a source file and vice versa.
    sources = {path.stem for path in
               sorted((REPO_ROOT / "cpp" / "tests").glob("test_*.cpp"))}
    assert sources == {f"test_{name}" for name in names}, (
        sorted(sources), sorted(names))


def test_the_indexing_unit_and_header_exist_and_no_other_cpp_arrived():
    """K3 added ``indexing.cpp`` and its internal header; K4 extended the
    same unit. No other production translation unit is Phase K's."""
    assert (REPO_ROOT / "cpp" / "src" / "indexing.cpp").is_file()
    assert (REPO_ROOT / "cpp" / "include"
            / "tf_indexing_internal.h").is_file()
    present = {path.name for path in (REPO_ROOT / "cpp" / "src").glob("*.cpp")}
    assert present == {"classification.cpp", "conv2d.cpp", "elementwise.cpp",
                       "error.cpp", "matmul.cpp", "pooling.cpp", "random.cpp",
                       "reduction.cpp", "storage.cpp", "indexing.cpp"}, (
        sorted(present))


def test_the_example_and_benchmark_inventories_closed_at_17_and_10():
    examples = sorted(path.name
                      for path in (REPO_ROOT / "examples").glob("*.py"))
    assert len(examples) == FINAL_EXAMPLE_COUNT, examples
    for name, milestone in PHASE_K_ADDED_EXAMPLES.items():
        assert name in examples, (name, milestone)
    benchmarks = sorted(path.name
                        for path in (REPO_ROOT / "benchmarks").glob("*.py"))
    assert len(benchmarks) == FINAL_BENCHMARK_COUNT, benchmarks
    for name, milestone in PHASE_K_ADDED_BENCHMARKS.items():
        assert name in benchmarks, (name, milestone)


@pytest.mark.parametrize("relative,minimum", REQUIRED_EVIDENCE)
def test_the_phase_k_evidence_files_are_all_present(relative, minimum):
    """A floor rather than an equality: adding coverage is free, deleting
    it is not. Exact totals are deliberately never made permanent."""
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
    assert len(re.findall(r"^def test_", good.read_text(encoding="utf-8"),
                          re.M)) == 2
    thin = tmp_path / "test_thin.py"
    thin.write_text('"""def test_looks_like_one but is prose."""\n',
                    encoding="utf-8")
    assert len(re.findall(r"^def test_", thin.read_text(encoding="utf-8"),
                          re.M)) == 0


# ===========================================================================
# 7. Serialization and state — nothing moved, and no integer entered
# ===========================================================================

def test_the_checkpoint_constants_did_not_move():
    module = __import__("tensorforge.experimental.native_checkpoint",
                        fromlist=["_FORMAT_VERSION"])
    assert module._FORMAT_VERSION == FINAL_CHECKPOINT_VERSION
    assert tuple(module._SUPPORTED_FORMAT_VERSIONS) == \
        FINAL_CHECKPOINT_VERSIONS
    assert tuple(module._FLOAT64_ONLY_VERSIONS) == (1, 2)
    source = _read("src/tensorforge/experimental/native_checkpoint.py")
    assert FINAL_CHECKPOINT_FORMAT in source
    # No version-4 constant is written, reserved, or accepted.
    assert 4 not in module._SUPPORTED_FORMAT_VERSIONS
    assert not re.search(r"_FORMAT_VERSION\s*=\s*4", source)


def test_the_optimizer_loader_and_sampler_state_versions_did_not_move():
    from tensorforge.experimental import (NativeBatchSampler,
                                          NativeDataLoader,
                                          NativeTensorDataset)

    loader_module = sys.modules[NativeDataLoader.__module__]
    assert loader_module._FORMAT == FINAL_LOADER_STATE_FORMAT
    assert loader_module._FORMAT_VERSION == FINAL_LOADER_STATE_VERSION
    assert tuple(loader_module._SUPPORTED_FORMAT_VERSIONS) == \
        FINAL_LOADER_STATE_VERSIONS
    sampler_module = sys.modules[NativeBatchSampler.__module__]
    assert sampler_module is not loader_module, (
        "the loader and sampler states must stay two authorities")
    assert sampler_module._FORMAT == FINAL_SAMPLER_STATE_FORMAT
    assert sampler_module._FORMAT_VERSION == FINAL_SAMPLER_STATE_VERSION
    assert tuple(sampler_module._SUPPORTED_FORMAT_VERSIONS) == \
        FINAL_SAMPLER_STATE_VERSIONS
    checkpoint_module = __import__(
        "tensorforge.experimental.native_checkpoint",
        fromlist=["_STATE_FORMAT_VERSION"])
    assert checkpoint_module._STATE_FORMAT_VERSION == \
        FINAL_OPTIMIZER_STATE_VERSION
    assert NativeTensorDataset is not None


@needs_backend
def test_no_integer_parameter_buffer_optimizer_or_grad_state_exists():
    """The §6.5 barriers, re-proved at the closed boundary against a real
    integer tensor. Every rejection leaves live storage where it was —
    proved here by the probe still being open and closable afterwards."""
    from tensorforge.experimental import (NativeAdam, NativeModule,
                                          NativeParameter, NativeSGD,
                                          NativeTensor)

    probe = _int64_probe()
    try:
        with pytest.raises((TypeError, ValueError)):
            NativeParameter(probe)
        module = type("Holder", (NativeModule,), {})()
        for persistent in (True, False):
            with pytest.raises((TypeError, ValueError)):
                module.register_buffer("b", probe, persistent=persistent)
        with pytest.raises((TypeError, ValueError)):
            NativeSGD([probe], lr=0.1)
        with pytest.raises((TypeError, ValueError)):
            NativeAdam([probe], lr=0.1)
        with pytest.raises((TypeError, ValueError)):
            NativeTensor.from_int64_array(np.array([1], dtype=np.int64),
                                          requires_grad=True)
        with pytest.raises((TypeError, ValueError, RuntimeError)):
            probe.backward()
        assert probe.requires_grad is False
        assert probe.grad is None
        # The probe survived every rejection intact.
        assert probe.to_numpy().tolist() == [0]
    finally:
        probe.close()


@needs_backend
def test_every_floating_operation_still_refuses_an_integer_operand():
    """No integer arithmetic, reduction, or unary math — the compute
    boundary is exactly what Phase I established."""
    from tensorforge.experimental import NativeTensor

    probe = _int64_probe()
    partner = NativeTensor.from_array(np.array([1.0]))
    try:
        for operation in ("add", "subtract", "multiply", "matmul"):
            with pytest.raises((TypeError, ValueError)):
                getattr(probe, operation)(probe)
            with pytest.raises((TypeError, ValueError)):
                getattr(partner, operation)(probe)
        for operation in ("sum", "mean", "relu", "sqrt", "reciprocal",
                          "exp", "log"):
            with pytest.raises((TypeError, ValueError)):
                getattr(probe, operation)()
        # argmax takes a floating source, never an integer one...
        with pytest.raises((TypeError, ValueError)):
            probe.argmax()
        # ...and index_select's source must float while its index must not.
        with pytest.raises((TypeError, ValueError)):
            probe.index_select(0, probe)
        with pytest.raises((TypeError, ValueError)):
            partner.index_select(0, partner)
    finally:
        partner.close()
        probe.close()


@needs_backend
def test_the_two_operations_keep_their_role_contracts():
    """argmax: floating source in, fresh owning int64 leaf out, first
    maximum wins. index_select: floating source plus rank-1 int64 index,
    duplicates and order preserved, a gradient-tracking source rejected
    with a message naming detach()."""
    from tensorforge.experimental import NativeTensor

    source = NativeTensor.from_array(
        np.array([[1.0, 3.0, 3.0], [7.0, -1.0, 5.0]]))
    tracked = NativeTensor.from_array(np.array([[2.0, 1.0]]),
                                      requires_grad=True)
    index = NativeTensor.from_int64_array(np.array([2, 0, 2],
                                                   dtype=np.int64))
    produced = []
    try:
        best = source.argmax(axis=1)
        produced.append(best)
        assert best.dtype == "int64"
        assert best.requires_grad is False
        assert best.to_numpy().tolist() == [1, 0]   # first maximum wins
        # Even a gradient-tracking input yields a plain leaf.
        tracked_best = tracked.argmax()
        produced.append(tracked_best)
        assert tracked_best.requires_grad is False

        selected = source.index_select(1, index)
        produced.append(selected)
        assert selected.dtype == "float64"
        assert selected.to_numpy().tolist() == [[3.0, 1.0, 3.0],
                                                [5.0, 7.0, 5.0]]
        # A tracking source is rejected, naming detach() — never silently
        # detached, because that would be a silent gradient hole.
        with pytest.raises((TypeError, ValueError)) as failure:
            tracked.index_select(1, index)
        assert "detach" in str(failure.value)
        # Out-of-range and negative indices reject rather than wrap.
        for bad in (3, -1):
            bad_index = NativeTensor.from_int64_array(
                np.array([bad], dtype=np.int64))
            try:
                with pytest.raises((ValueError, IndexError)):
                    source.index_select(1, bad_index)
            finally:
                bad_index.close()
    finally:
        for tensor in produced:
            tensor.close()
        index.close()
        tracked.close()
        source.close()


@needs_backend
def test_the_pipeline_still_delivers_floating_features_and_host_targets():
    """Phase J's delivery contract, untouched by the whole phase: a
    floating ``NativeTensor`` feature batch and a **read-only host**
    ``numpy.ndarray`` target batch of dtype int64 — never a native label
    tensor, and no option to request one."""
    from tensorforge.experimental import (NativeBatchSampler,
                                          NativeDataLoader,
                                          NativeTensor,
                                          NativeTensorDataset)

    features = np.arange(12, dtype=np.float64).reshape(4, 3)
    targets = np.array([0, 1, 0, 1], dtype=np.int64)
    dataset = NativeTensorDataset(features, targets)
    try:
        loader = NativeDataLoader(NativeBatchSampler(dataset, batch_size=2))
        batch_features, batch_targets = next(iter(loader))
        try:
            assert isinstance(batch_features, NativeTensor)
            assert batch_features.dtype in FINAL_SUPPORTED_DTYPES
            assert isinstance(batch_targets, np.ndarray)
            assert batch_targets.dtype == np.int64
            assert not batch_targets.flags.writeable
        finally:
            batch_features.close()
    finally:
        dataset.close()
    # No native-label option exists on any of the three constructors.
    for target, banned in ((NativeTensorDataset, "native_targets"),
                           (NativeDataLoader, "native_labels"),
                           (NativeBatchSampler, "native_targets")):
        assert banned not in inspect.signature(target).parameters, (
            target.__name__, banned)


@needs_backend
def test_native_accuracy_still_computes_without_the_two_operations():
    """§20.3's deliberate reconciliation: the metric keeps its host
    ``to_numpy`` round trip, so it must work with both native operations
    disabled — proved by patching them to raise, restored explicitly."""
    from tensorforge.experimental import NativeTensor, native_accuracy

    def poisoned(*_args, **_kwargs):                  # pragma: no cover
        raise AssertionError("native_accuracy called a native indexing "
                             "operation")

    logits = NativeTensor.from_array(np.array([[2.0, 1.0], [0.5, 3.0]]))
    original_argmax = NativeTensor.argmax
    original_select = NativeTensor.index_select
    NativeTensor.argmax = poisoned
    NativeTensor.index_select = poisoned
    try:
        value = native_accuracy(logits, np.array([0, 1], dtype=np.int64))
        assert value == 1.0
    finally:
        NativeTensor.argmax = original_argmax
        NativeTensor.index_select = original_select
        logits.close()


# ===========================================================================
# 8. The absent surface, structurally
# ===========================================================================

def test_no_banned_method_is_defined_on_either_tensor_layer():
    """Read from the AST of the two defining modules, so a docstring
    explaining "there is no gather" cannot trip it and a definition cannot
    hide from it."""
    tensor_methods = _class_method_names(
        "src/tensorforge/experimental/native_tensor.py", "NativeTensor")
    core_methods = _class_method_names(
        "src/tensorforge/backends/cpp.py", "NativeTensorCore")
    for banned in BANNED_TENSOR_METHODS:
        assert banned not in tensor_methods, banned
        assert banned not in core_methods, banned
    # ...and the reader itself is non-vacuous: the methods that do exist
    # are found by the same reader.
    for present in PHASE_K_TENSOR_METHODS:
        assert present in tensor_methods, present
    assert "argmax" in core_methods and "index_select" in core_methods


def test_the_method_reader_can_actually_fail(tmp_path):
    """Negative control on a temporary module: a planted banned definition
    is found, and prose is not."""
    planted = tmp_path / "sample.py"
    planted.write_text(
        '"""gather is documented here but not defined."""\n'
        "class Sample:\n"
        "    def gather(self):\n"
        "        pass\n", encoding="utf-8")
    tree = ast.parse(planted.read_text(encoding="utf-8"))
    names = {child.name for node in ast.walk(tree)
             if isinstance(node, ast.ClassDef) and node.name == "Sample"
             for child in node.body if isinstance(child, ast.FunctionDef)}
    assert "gather" in names
    prose_only = tmp_path / "prose.py"
    prose_only.write_text('"""gather scatter embedding argmin."""\n'
                          "class Sample:\n    pass\n", encoding="utf-8")
    tree = ast.parse(prose_only.read_text(encoding="utf-8"))
    names = {child.name for node in ast.walk(tree)
             if isinstance(node, ast.ClassDef)
             for child in node.body if isinstance(child, ast.FunctionDef)}
    assert "gather" not in names


def test_no_integer_kernel_or_casting_primitive_exists_in_the_cpp_sources():
    """Over comment-stripped C++ code: the indexing unit holds the two
    shipped traversals and no arithmetic at int64, and no unit anywhere
    declares a cast, promote, or integer-arithmetic export."""
    for source in sorted((REPO_ROOT / "cpp" / "src").glob("*.cpp")):
        code = _cpp_code_only(source.read_text(encoding="utf-8"))
        for banned in ("tf_core_cast", "tf_core_astype", "tf_core_promote",
                       "tf_core_int_add", "tf_core_argmin",
                       "tf_core_gather", "tf_core_scatter",
                       "tf_core_embedding", "tf_core_index_select_backward",
                       "tf_core_max_with_indices"):
            assert banned not in code, (source.name, banned)
    # The comment stripper is proved able to fail elsewhere; here its
    # positive control: the two real exports are visible through it.
    indexing = _cpp_code_only((REPO_ROOT / "cpp" / "src"
                               / "indexing.cpp").read_text(encoding="utf-8"))
    assert "tf_core_argmax" in indexing
    assert "tf_core_index_select" in indexing


def test_the_cpp_comment_stripper_can_actually_fail():
    source = ("// tf_core_probe in a comment\n"
              "/* tf_core_probe again */\n"
              "TF_EXPORT void tf_core_real(void);\n")
    code = _cpp_code_only(source)
    assert "tf_core_probe" not in code
    assert "tf_core_real" in code


def test_no_concurrency_primitive_entered_with_the_phase():
    """The synchronous, externally locked boundary stands: the phase's C++
    unit uses no thread, mutex, atomic, or future, and the two Python
    operations introduce no concurrency identifier."""
    indexing = _cpp_code_only((REPO_ROOT / "cpp" / "src"
                               / "indexing.cpp").read_text(encoding="utf-8"))
    for banned in ("std::thread", "std::mutex", "std::atomic",
                   "std::future", "std::async", "openmp", "#pragma omp"):
        assert banned not in indexing.lower(), banned
    from tensorforge.experimental import NativeTensor

    for operation in ("argmax", "index_select", "from_int64_array"):
        attribute = inspect.getattr_static(NativeTensor, operation)
        function = getattr(attribute, "__func__", attribute)
        names = _code_identifiers(
            "def _probe():\n"
            + re.sub(r"^", "    ", inspect.getsource(function),
                     flags=re.M))
        for banned in ("Thread", "Lock", "RLock", "Queue", "Future",
                       "threading", "asyncio", "concurrent"):
            assert banned not in names, (operation, banned)


def test_the_concurrency_scanner_can_actually_fail():
    planted = ("def f():\n"
               "    import threading\n"
               "    lock = threading.Lock()\n"
               "    return lock\n")
    names = _code_identifiers(planted)
    assert "threading" in names and "Lock" in names
    innocent = '"""threading.Lock() is documented as absent."""\n'
    assert "threading" not in _code_identifiers(innocent)


# ===========================================================================
# 9. Stable / native isolation
# ===========================================================================

def test_importing_the_stable_framework_does_not_load_the_native_backend():
    code = (
        "import sys\n"
        "import tensorforge, tensorforge.nn, tensorforge.optim, "
        "tensorforge.data\n"
        "loaded = [m for m in sys.modules if m.endswith('backends.cpp')]\n"
        "assert not loaded, loaded\n"
        "assert not [m for m in sys.modules if 'experimental' in m]\n"
        "print('isolated')\n"
    )
    done = subprocess.run([sys.executable, "-c", code], cwd=REPO_ROOT,
                          capture_output=True, text=True)
    assert done.returncode == 0, done.stderr
    assert "isolated" in done.stdout


@needs_backend
def test_a_stable_tensor_is_refused_by_the_integer_door():
    """No implicit conversion in either direction, including at the one
    door Phase K opened."""
    import tensorforge
    from tensorforge.experimental import NativeTensor

    stable = tensorforge.Tensor(np.zeros(3))
    with pytest.raises(TypeError):
        NativeTensor.from_int64_array(stable)


def test_no_environment_variable_steers_the_integer_stack():
    """The indexing paths consult no environment: dispatch stays a function
    of layout and geometry metadata alone."""
    indexing = _cpp_code_only((REPO_ROOT / "cpp" / "src"
                               / "indexing.cpp").read_text(encoding="utf-8"))
    assert "getenv" not in indexing
    from tensorforge.experimental import NativeTensor

    for operation in ("argmax", "index_select", "from_int64_array"):
        attribute = inspect.getattr_static(NativeTensor, operation)
        function = getattr(attribute, "__func__", attribute)
        names = _code_identifiers(
            "def _probe():\n"
            + re.sub(r"^", "    ", inspect.getsource(function), flags=re.M))
        assert "environ" not in names and "getenv" not in names, operation


# ===========================================================================
# 10. Benchmark governance at the closed boundary
# ===========================================================================

def test_the_k8_benchmark_offers_no_save_baseline_or_compare_mode():
    """Characterization, never a promise: the CLI cannot write a result
    file, commit a baseline, or gate on a duration."""
    source = _read("benchmarks/benchmark_native_integer.py")
    tree = ast.parse(source)
    option_strings = {node.value for node in ast.walk(tree)
                      if isinstance(node, ast.Constant)
                      and isinstance(node.value, str)
                      and node.value.startswith("--")}
    for banned in ("--save", "--output", "--baseline", "--compare",
                   "--record", "--write"):
        assert banned not in option_strings, banned
    assert "--smoke" in option_strings and "--json" in option_strings


def test_no_ci_job_gates_on_a_number_and_the_workflow_still_validates():
    workflow = WORKFLOW.read_text(encoding="utf-8")
    for step in ("uv run python cpp/build.py",
                 "uv run python scripts/smoke_cpp_backend.py",
                 "uv run pytest"):
        assert step in workflow, step
    assert "actions/checkout" in workflow
    assert "fetch-depth" not in workflow
    for banned in ("--fail-on-slow", "--assert-duration", "timeout-minutes:"
                   " 1\n"):
        assert banned not in workflow, banned


# ===========================================================================
# 11. Repository hygiene — closure left no artifact behind
# ===========================================================================

def test_no_generated_artifact_sits_unignored_in_the_working_tree():
    """Builds, sanitizer output, CTest logs, witness records, and review
    packages all live **outside** the repository. Scoped to
    untracked-and-unignored paths, which is the set that can actually be
    committed by accident."""
    for relative in _untracked_unignored_files():
        lowered = relative.lower()
        assert not lowered.endswith(
            (".obj", ".o", ".lib", ".pdb", ".exp", ".ilk", ".idb", ".a",
             ".so", ".dll", ".dylib", ".pyd", ".npz", ".npy", ".ckpt",
             ".pt", ".pth", ".core", ".log", ".zip")), relative
        assert "cmakecache" not in lowered, relative
        assert not re.search(r"(^|/)(asan|ubsan|lsan|tsan)[-_.]",
                             lowered), relative
        assert not re.search(r"suppressions?(\.txt|\.supp)?$",
                             lowered), relative
        assert "witness" not in lowered, relative
        assert "benchmark_result" not in lowered, relative


def test_no_sanitizer_suppression_file_exists_anywhere_in_the_tree():
    """Zero diagnostics means something only when nothing is suppressed."""
    for path in _tracked_files():
        lowered = path.lower()
        assert "suppress" not in lowered, path
        assert not re.search(r"(^|/)(asan|ubsan|lsan)[-_.]", lowered), path


# ===========================================================================
# 12. Shallow-clone and line-ending safety — this module's own discipline
# ===========================================================================

def test_no_closure_guard_reaches_for_a_historical_git_object():
    """Parsed rather than grepped, because this module's own prose
    legitimately names ``git show`` and ``ls-tree`` while explaining their
    absence."""
    closure = Path(__file__).read_text(encoding="utf-8")
    for node in ast.walk(ast.parse(closure)):
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
    assert not re.search(r"\b[0-9a-f]{40}\b", closure), (
        "the closure module names a commit object")


def test_the_git_helper_degrades_rather_than_passing_falsely():
    source = inspect.getsource(_git_lines)
    assert source.count("pytest.skip") == 2, (
        "one of the two failure paths no longer skips")
    for caller in (_tracked_files, _untracked_unignored_files):
        assert "ls-files" in inspect.getsource(caller), caller.__name__
    relative = Path(__file__).relative_to(REPO_ROOT).as_posix()
    assert relative in set(_tracked_files()) | set(
        _untracked_unignored_files()), relative


def test_every_prose_scan_is_line_ending_agnostic():
    sample = "Phase K\r\nis complete.\r\n"
    assert _flat(sample) == _flat(sample.replace("\r\n", "\n"))
    assert _PHASE_COMPLETE.search(_flat(sample))
    assert _stale_status(_flat("Phase K\r\nis not complete.\r\n"))


def test_this_module_asserts_no_timing_and_no_tolerance():
    """The closure module's own discipline, checked against its executable
    code rather than its prose."""
    names = _code_identifiers(Path(__file__).read_text(encoding="utf-8"))
    for banned in ("allclose", "approx", "isclose", "perf_counter",
                   "monotonic", "process_time", "Thread", "sleep"):
        assert banned not in names, banned


# ===========================================================================
# 13. The instructions and the document map
# ===========================================================================

def test_claude_md_records_the_final_phase_k_truth():
    """Facts and pointers, never a phrasing or a length."""
    text = AGENT_INSTRUCTIONS.read_text(encoding="utf-8")
    for value in (FINAL_SUPPORTED_DTYPES + FINAL_INDEX_DTYPES
                  + FINAL_SUPPORTED_DEVICES + FINAL_UNSUPPORTED):
        assert value in text, value
    assert "INDEX_DTYPES" in text
    for number in (FINAL_EXPORT_COUNT, FINAL_CTEST_COUNT,
                   FINAL_EXAMPLE_COUNT, FINAL_BENCHMARK_COUNT,
                   FINAL_EXPERIMENTAL_EXPORTS):
        assert str(number) in text, number
    assert "docs/native_integer_tensors_design.md" in text
    assert _PHASE_COMPLETE.search(_flat(text)), (
        "CLAUDE.md does not mark Phase K complete")


def test_claude_md_stays_below_the_project_ceiling():
    text = AGENT_INSTRUCTIONS.read_text(encoding="utf-8")
    size = len(text.replace("\r\n", "\n"))
    assert size < 150_000, (
        f"CLAUDE.md has grown to {size} characters; milestone history "
        f"belongs in docs/, not in project memory")


def test_the_design_is_linked_from_the_readme_and_the_instructions():
    for surface in ("README.md", "CLAUDE.md"):
        assert "docs/native_integer_tensors_design.md" in _read(surface), (
            surface)


def test_the_release_history_records_the_closure_as_chronology():
    """The chronology gains K9 as an event; it is not a status surface and
    its earlier entries keep their era's wording."""
    history = _flat(_read("docs/release_history.md"))
    assert re.search(r"\bK9\b", history), "release history never reaches K9"
    assert re.search(r"K9\b[^.;]{0,200}?(closure|closed|cross-platform)",
                     history, re.I)
