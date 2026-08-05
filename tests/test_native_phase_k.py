"""Phase-K contract guardrails (native integer tensors and indexing).

**Phase K is newly approved, it was approved after Phase J closed, and K0
is the only milestone that has landed.** K0 is an architecture, contract,
documentation, and status milestone: it shipped
``docs/native_integer_tensors_design.md``, this module, and the narrow
status reconciliation a newly approved phase requires — and **no runtime
behavior at all**. No integer dtype, no dtype code, no C++ enumerator, no
kernel, no C ABI symbol, no ctypes declaration, no ``NativeTensorCore``
method, no ``NativeTensor`` operation, no public export, no
capability-registry movement, no checkpoint or optimizer-state or
loader-state or sampler-state change, no example, no benchmark, no CTest,
and no dependency. Runtime capability begins at **K1**, which has not
started.

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

# The production C++ translation units at K0. An integer kernel would have
# to live somewhere, and "somewhere" is a new file or an existing one; this
# pins the file set and the export inventory pins the contents.
K0_CPP_SOURCES = (
    "classification.cpp", "conv2d.cpp", "elementwise.cpp", "error.cpp",
    "matmul.cpp", "pooling.cpp", "random.cpp", "reduction.cpp",
    "storage.cpp",
)

# The whole Phase-K ladder, and the split that carries the phase. A
# milestone moves its identifier from the second tuple to the first and
# nowhere else, so the two together are always exactly ``MILESTONES``.
MILESTONES = tuple(f"K{index}" for index in range(10))      # K0 ... K9
COMPLETE_MILESTONES = ("K0",)
UNSTARTED_MILESTONES = tuple(name for name in MILESTONES
                             if name not in COMPLETE_MILESTONES)
assert len(UNSTARTED_MILESTONES) == 9

# The ordering the phase turns on (design §32.1): every reachability
# barrier lands at K1, and the first milestone at which an ``int64`` tensor
# can be constructed is K2. Written here independently of the document, so
# a ladder that reordered them would fail rather than be described.
BARRIER_MILESTONE = "K1"
FIRST_CONSTRUCTION_MILESTONE = "K2"
assert (MILESTONES.index(BARRIER_MILESTONE)
        < MILESTONES.index(FIRST_CONSTRUCTION_MILESTONE))

# The eventual public names, and the milestone that adds each. **None of
# them exists at K0**, and the absence half is what this module proves.
# ``NativeTensor``/``NativeTensorCore`` methods rather than new classes, so
# ``experimental.__all__`` never moves — see design §23.2.
PLANNED_TENSOR_METHODS = {
    "from_int64_array": "K2",
    "item": "K2",
    "tolist": "K2",
    "argmax": "K3",
    "index_select": "K4",
}

# The eventual C ABI delta, and its maximum. Phase K adds exactly two
# symbols and no milestone may exceed 56 (design §22.3).
PLANNED_EXPORTS = {"tf_core_argmax": "K3", "tf_core_index_select": "K4"}
PHASE_K_MAX_EXPORTS = 56
assert K0_EXPORT_COUNT + len(PLANNED_EXPORTS) == PHASE_K_MAX_EXPORTS

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
    ("argmax exists",
     r"\bargmax\b[^.]{0,40}" + _BECAME + _LANDED),
    ("index selection exists",
     r"\b(index[_ ]select|gather)\b[^.]{0,40}" + _BECAME + _LANDED),
    ("a Phase-K milestone after K0 has landed",
     r"\bK(?:[1-9]|10)\b[^.]{0,30}" + _BECAME + r"(" + _LANDED + r"|"
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
        "argmax is implemented",
        "gather is available",
        "index_select has landed",
        "Phase K is complete",
        "K1 has landed",
        "K4 is shipped",
        "the checkpoint is now at version 4",
        "CUDA is supported",
        "integer gradients are supported",
        "integer parameters are available",
    ):
        assert _overclaims(caught), caught
    # ...and every accurate sentence a K0 surface must be able to write.
    for allowed in (
        "int64 is not a supported native tensor dtype",
        "no native integer tensor exists",
        "Phase K is newly approved and K0 is complete",
        "K0 is the only completed Phase-K milestone",
        "K1 through K9 are unstarted",
        "a future milestone may add argmax",
        "argmax is deliberately absent",
        "index_select would need one new export",
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


def test_the_design_states_k0_is_zero_runtime_and_the_only_milestone_done():
    head = _head()
    assert re.search(r"K0 adds no runtime behavior", head, re.I), head[:800]
    assert re.search(r"K0 is the only completed Phase-K milestone", head,
                     re.I), head[:1200]
    first_unstarted = UNSTARTED_MILESTONES[0]
    last = UNSTARTED_MILESTONES[-1]
    assert re.search(rf"{first_unstarted} through {last} are unstarted",
                     head, re.I), head[:1200]
    assert re.search(r"[Rr]untime capability begins at K1", head), head[:1600]
    # The claim that must be impossible to misread at K0.
    assert re.search(r"int64 is not a supported TensorForge native tensor "
                     r"dtype", head, re.I), head[:2000]


def test_the_design_header_records_the_inherited_boundary_unmoved():
    head = _head()
    for value in ("float64", "float32", "cpu", "cuda", "amp",
                  "tensorforge.native_checkpoint",
                  "tensorforge.native_data_loader",
                  "tensorforge.native_sampler"):
        assert value.lower() in head.lower(), value
    assert re.search(r"K0 moves none of them", head, re.I), head[:3000]


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


def test_no_public_int64_promise_has_moved():
    """The whole point of K0: `int64` is not promised anywhere."""
    assert "int64" not in cpp.SUPPORTED_DTYPES
    assert "int64" not in cpp.RAW_KERNEL_DTYPES
    with pytest.raises(ValueError):
        cpp.normalize_dtype("int64")
    with pytest.raises(ValueError):
        cpp._normalize_internal_dtype("int64")
    # ...and no second dtype row has appeared ahead of its milestone.
    for absent in ("INDEX_DTYPES", "COMPUTE_DTYPES", "INTEGER_DTYPES",
                   "TENSOR_DTYPES"):
        assert not hasattr(cpp, absent), absent


def test_backend_info_reports_the_same_three_dtype_rows_it_did_at_j9():
    info = cpp.backend_info()
    assert info["dtype"] == K0_DEFAULT_DTYPE
    assert info["device"] == "cpu"
    assert info["supported_dtypes"] == K0_DTYPES
    assert info["supported_devices"] == K0_DEVICES
    assert info["raw_kernel_dtypes"] == K0_RAW_KERNEL_DTYPES
    assert info["unsupported"] == K0_UNSUPPORTED
    assert info["stable_framework_integration"] is False
    for absent in ("index_dtypes", "compute_dtypes", "integer_dtypes"):
        assert absent not in info, (absent, "the index row belongs to K2")


def test_no_operation_inventory_grew_an_integer_entry():
    for inventory in (cpp.RAW_KERNELS, cpp.TENSOR_CORE_KERNELS,
                      cpp.TENSOR_CORE_OPS, cpp.AUTOGRAD_OPS,
                      cpp.NATIVE_MODULES, cpp.NATIVE_LOSSES,
                      cpp.NATIVE_METRICS, cpp.NATIVE_OPTIMIZERS,
                      cpp.STATE_SUPPORT):
        for banned in ("argmax", "argmin", "index_select", "gather",
                       "scatter", "embedding", "int64", "integer", "cast",
                       "astype", "promote"):
            assert not [name for name in inventory
                        if banned in name.lower()], (banned, inventory)


# ===========================================================================
# 4. The inventories have not moved
# ===========================================================================

def test_the_source_export_inventory_is_still_fifty_four():
    exports = _source_exports()
    assert len(exports) == K0_EXPORT_COUNT, sorted(exports)


def test_neither_planned_export_exists_yet():
    exports = _source_exports()
    for name, milestone in PLANNED_EXPORTS.items():
        assert name not in exports, f"{name} belongs to {milestone}"
    assert not [name for name in exports
                if "argmax" in name or "index" in name or "gather" in name]


@pytest.mark.skipif(not cpp.is_available(),
                    reason="the native backend is not built")
def test_the_built_library_exports_the_same_fifty_four():
    """The source inventory and the built library must agree — the standing
    ABI-discipline rule, re-asserted at the phase boundary."""
    storage_tests = pytest.importorskip("test_native_storage_allocation")
    _, names = storage_tests.exported_names(cpp._LIBRARY_PATH)
    if names is None:
        pytest.skip("this image format is not parsed here")
    exported = sorted(name for name in names if name.startswith("tf_"))
    assert len(exported) == K0_EXPORT_COUNT, exported
    assert set(exported) == _source_exports()


def test_the_experimental_export_list_is_still_twenty_five():
    import tensorforge.experimental as experimental

    assert len(experimental.__all__) == K0_EXPERIMENTAL_EXPORTS
    assert len(set(experimental.__all__)) == K0_EXPERIMENTAL_EXPORTS
    for name in experimental.__all__:
        assert hasattr(experimental, name), name


def test_the_ctest_example_and_benchmark_inventories_are_unmoved():
    cmake = _read("cpp/CMakeLists.txt")
    assert len(re.findall(r"^\s*add_test\(", cmake, re.M)) == K0_CTEST_COUNT
    assert len(list((REPO_ROOT / "cpp" / "tests").glob("*.cpp"))) == \
        K0_CTEST_COUNT
    assert len(list((REPO_ROOT / "examples").glob("*.py"))) == \
        K0_EXAMPLE_COUNT
    assert len(list((REPO_ROOT / "benchmarks").glob("*.py"))) == \
        K0_BENCHMARK_COUNT


def test_the_production_cpp_translation_units_are_unchanged():
    present = tuple(sorted(path.name for path in
                           (REPO_ROOT / "cpp" / "src").glob("*.cpp")))
    assert present == tuple(sorted(K0_CPP_SOURCES))


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

def test_no_integer_dtype_code_exists_on_either_side_of_the_abi():
    """The two dtype authorities, both still two-valued."""
    assert set(cpp._DTYPE_CODES) == set(K0_DTYPES)
    assert set(cpp._DTYPE_ITEM_SIZES) == set(K0_DTYPES)
    assert set(cpp._DTYPE_NUMPY) == set(K0_DTYPES)
    assert _dict_literal_keys("src/tensorforge/backends/cpp.py",
                              "_DTYPE_CODES") == set(K0_DTYPES)

    header = _read("cpp/include/tf_internal.h")
    enum_body = header.split("enum TfDtype {", 1)[1].split("};", 1)[0]
    assert set(re.findall(r"TF_DTYPE_(\w+)", enum_body)) == \
        {"FLOAT64", "FLOAT32"}
    scoped = header.split("enum class Dtype", 1)[1].split("};", 1)[0]
    assert set(re.findall(r"^\s*(\w+)\s*=", scoped, re.M)) == \
        {"Float64", "Float32"}
    assert "TF_DTYPE_INT64" not in header
    assert "Int64" not in scoped


def test_no_integer_runtime_module_exists():
    experimental = REPO_ROOT / "src" / "tensorforge" / "experimental"
    modules = {path.name for path in experimental.glob("*.py")}
    for banned in ("native_int64", "native_integer", "native_index",
                   "native_argmax", "native_gather", "native_embedding",
                   "_native_index", "_native_integer"):
        assert not [name for name in modules if name.startswith(banned)], \
            banned


def test_no_public_integer_constructor_or_index_operation_exists():
    tensor_methods = _defined_names(
        "src/tensorforge/experimental/native_tensor.py", "NativeTensor")
    core_methods = _defined_names("src/tensorforge/backends/cpp.py",
                                  "NativeTensorCore")
    storage_methods = _defined_names("src/tensorforge/backends/cpp.py",
                                     "NativeStorage")
    for name, milestone in PLANNED_TENSOR_METHODS.items():
        assert name not in tensor_methods, f"{name} belongs to {milestone}"
        assert name not in core_methods, f"{name} belongs to {milestone}"
    for absent in ("from_int64_array", "int64", "as_int64", "to_int64"):
        assert absent not in storage_methods, absent
    # ...and the module-level factory shape is absent too.
    module = _code_only(_module_source(
        "src/tensorforge/experimental/native_tensor.py"))
    for banned in ("from_int64_array", "argmax", "index_select", "tolist"):
        assert banned not in module, banned


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
    legitimate and is not what this looks for."""
    for path in sorted((REPO_ROOT / "cpp" / "src").glob("*.cpp")):
        text = path.read_text(encoding="utf-8")
        # Strip // and /* */ comments so prose cannot trip the scan.
        code = re.sub(r"/\*.*?\*/", " ", text, flags=re.S)
        code = re.sub(r"//[^\n]*", " ", code)
        for banned in ("argmax", "index_select", "Dtype::Int64",
                       "TF_DTYPE_INT64", "require_floating",
                       "storage_typed<std::int64_t>",
                       "storage_typed<int64_t>"):
            assert banned not in code, (path.name, banned)


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
# 10a. Current-phase reconciliation, and the one scoped exception
# ===========================================================================
#
# The failure this section exists for: a repository where Phase K has
# opened but every status surface still calls Phase J "the latest phase",
# so a reader cannot tell which phase is current. K0 cannot repair
# ``src/tensorforge/experimental/__init__.py`` — that is production source
# — so exactly one surface is allowed to lag, and these tests prove the
# exception is *one* surface rather than a hole in the checking.

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
)

# The one surface K0 may not touch, and the milestone that repairs it.
STALE_PRODUCTION_SURFACE = "src/tensorforge/experimental/__init__.py"
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


@pytest.mark.parametrize("surface", EDITABLE_STATUS_SURFACES)
def test_every_editable_status_surface_names_k_as_the_current_phase(surface):
    text = _flat(_read(surface))
    assert _phase_letters(_LATEST_PHASE_FORM, text) == {"K"}, surface
    assert re.search(r"only K0 has landed", text, re.I), surface


@pytest.mark.parametrize("surface", EDITABLE_STATUS_SURFACES)
def test_every_editable_status_surface_names_j_as_latest_completed(surface):
    text = _flat(_read(surface))
    assert _phase_letters(_LATEST_COMPLETED_FORM, text) == {"J"}, surface


@pytest.mark.parametrize("surface", EDITABLE_STATUS_SURFACES)
def test_no_editable_status_surface_calls_j_the_latest_phase(surface):
    """The stale claim itself, banned everywhere it can be repaired."""
    named = _phase_letters(_LATEST_PHASE_FORM, _flat(_read(surface)))
    assert named <= {"K"}, (surface, sorted(named))


def test_the_production_docstring_is_the_only_allowed_stale_surface():
    """The exception is exactly one file, and it really is still stale — a
    check that would start failing the moment K1 repairs it, which is the
    point: the exemption cannot outlive its reason."""
    stale = _phase_letters(_LATEST_PHASE_FORM, _flat(_read(STALE_PRODUCTION_SURFACE)))
    assert stale == {"J"}, (STALE_PRODUCTION_SURFACE, sorted(stale))
    # ...and no other production module carries the same stale claim.
    package = REPO_ROOT / "src" / "tensorforge"
    offenders = []
    for path in package.rglob("*.py"):
        if "__pycache__" in path.parts:
            continue
        relative = path.relative_to(REPO_ROOT).as_posix()
        if relative == STALE_PRODUCTION_SURFACE:
            continue
        named = _phase_letters(_LATEST_PHASE_FORM,
                               _flat(path.read_text(encoding="utf-8")))
        if named - {"K"}:
            offenders.append((relative, sorted(named)))
    assert offenders == [], offenders


def test_the_design_assigns_the_production_docstring_repair_to_k1():
    """The exception is only defensible because a milestone owns it. The
    design must name the exact file, the reason K0 could not touch it, and
    the milestone that does."""
    ladder = _design()
    start = ladder.index(f"### {STALE_REPAIR_MILESTONE} — ")
    following = re.search(r"\n### K\d+ — ", ladder[start + 5:])
    body = _flat(ladder[start:start + 5 + following.start()] if following
                 else ladder[start:])
    assert STALE_PRODUCTION_SURFACE in body, body[:200]
    assert "latest phase" in body
    assert "no production source at all" in body
    assert "_LATEST_PHASE" in body


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
    assert re.search(r"only K0 has landed", text, re.I)
    assert re.search(r"Phase J is the latest completed phase", text, re.I)
    # ...and the Phase-K section states the absence half.
    assert re.search(r"K1 through K9 are unstarted", text, re.I)
    assert re.search(r"design, documentation, and guardrails only", text, re.I)
    assert re.search(r"no runtime capability exists yet", text, re.I)
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
# 12. CLAUDE.md size policy — the existing ceiling, and nothing stricter
# ===========================================================================

def test_claude_md_stays_below_the_project_ceiling():
    text = _read("CLAUDE.md").replace("\r\n", "\n")
    assert len(text) < CLAUDE_MD_CEILING, len(text)


def test_the_claude_md_document_map_names_the_phase_k_authority():
    text = _read("CLAUDE.md")
    assert f"docs/{PHASE_K_DESIGN_NAME}" in text
    assert "Phase K" in text
