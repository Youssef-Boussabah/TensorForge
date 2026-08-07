"""The K6 end-to-end native integer indexing example, and its exact
interrupted-versus-uninterrupted proof.

Milestone **K6** is the first end-user program in which the native runtime's
**integer** side carries real work: a deterministic classifier trains over
the Phase-J pipeline, and at fixed evaluation points its logits become
native ``int64`` prediction indices through ``NativeTensor.argmax`` (K3)
which are then **consumed** by ``NativeTensor.index_select`` (K4) over a
detached, graph-free copy of the same logits. It adds **no runtime
capability**: no kernel, no C ABI export, no module, no checkpoint field or
version, no public package export, and no executable line of ``src/``. Its
whole diff is ``examples/native_integer_indexing.py``, this module, the
narrow inventory and status edits the landing requires, and documentation.

What is asserted here:

* the example exists, imports without training, and runs as a script,
  leaving no file behind and claiming no timing;
* its executable code uses **public APIs only** — every private runtime
  seam the design names is proved absent by an **AST** scan (not a
  substring ban, which the module's own prose would defeat), with a
  negative control proving the scanner can find a planted violation;
* the host dataset is deterministic, non-degenerate, built at an explicit
  dtype from exactly representable binary fractions, and identical in
  logical value across the two dtypes;
* the committed batch plan is pinned as **literal expected values** written
  on the test side, and the interruption lands **strictly inside** an epoch
  with batches still owed and evaluations on **both** sides of it;
* the ``argmax`` results really are ``int64`` plain leaves that own fresh
  contiguous storage, and the ``index_select`` results really are
  source-dtype, owning, contiguous, and graph-free;
* ``index_select`` is **axis selection, not a per-row gather** — the whole
  ``(batch, batch)`` result is recomputed here from the recorded logits and
  checked column by column, the diagonal is checked against each example's
  own predicted-class logit, and duplicate predicted classes are proved to
  produce identical columns in their original order;
* the resumed run reproduces the uninterrupted one **exactly** — every
  prediction index by exact integer equality, every floating value by raw
  IEEE-754 bits;
* the negative controls really fail: omitting the loader restoration
  diverges, the bit helper separates signed zeros and adjacent values, the
  storage tracker notices a deliberately retained tensor, and the training
  claims are backed by state that actually moved;
* native live storage returns **exactly** to its baseline;
* the inventories move by exactly one example and nothing else.

**Every equality here is exact.** Prediction indices are compared as
Python integers and are never converted to a floating value; floating
values are compared over raw bit patterns — a ``uint32`` view at float32
and a ``uint64`` view at float64. Never a tolerance, never ``allclose``,
never ``pytest.approx``, and **never a numeric comparison between the two
dtypes**: each is proved only against itself. Whether the two widths happen
to predict the same classes is reported by the example as an *observation*
and is deliberately not required here.

**K7, K8, and K9 are not started and are not anticipated here.** The
adversarial injection matrix is K7's, the benchmark is K8's, and the phase
closure is K9's; this module contains none of them, and asserts that none
of their artifacts exists.
"""
import ast
import gc
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from tensorforge.backends import cpp
from tensorforge.experimental import (
    NativeBatchSampler,
    NativeDataLoader,
    NativeGenerator,
    NativeModule,
    NativeTensor,
    NativeTensorDataset,
    native_data_loader,
    native_sampler,
)
from tensorforge.experimental import native_checkpoint, native_optimizer_state

from examples.native_integer_indexing import (
    BATCHES_PER_EPOCH,
    BATCH_SIZE,
    CLASS_AXIS,
    DEFAULT_LR,
    DROP_LAST,
    EVAL_STEPS,
    EXERCISED_EPOCHS,
    FEATURES,
    FRESH_BATCH_SIZE,
    FRESH_LR,
    FRESH_SAMPLER_SEED,
    FRESH_SHUFFLE,
    HIDDEN,
    INDEX_DTYPE,
    LOADER_KEY,
    NEXT_STEP_KEY,
    NUM_CLASSES,
    REQUIRED,
    REQUIRED_CROSS_DTYPE,
    REQUIRED_INDEXING,
    REQUIRED_SCHEDULE,
    REQUIRED_TRAINING,
    RUN_DTYPES,
    SAMPLER_SEED,
    SAMPLES,
    SHUFFLE,
    SPLIT_STEP,
    TOTAL_STEPS,
    TRAINING_KEY,
    NativeIndexingClassifier,
    advance_loader,
    bits,
    build_dataset,
    build_features,
    build_loader,
    build_loss,
    build_model,
    build_optimizer,
    build_targets,
    cross_dtype_facts,
    evaluate_indexing,
    failed_checks,
    failed_cross_dtype_checks,
    host_arrays,
    index_values,
    main,
    model_facts,
    optimizer_facts,
    run_dtype_proof,
    run_omitted_loader_control,
    run_uninterrupted,
    train_steps,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
EXAMPLE_RELATIVE = "examples/native_integer_indexing.py"
EXAMPLE = REPO_ROOT / EXAMPLE_RELATIVE

needs_native = pytest.mark.skipif(
    not cpp.is_available(), reason="the experimental C++ backend is not built"
)

pytestmark = needs_native


# ===========================================================================
# The committed schedule — literal expected values, written here
# ===========================================================================
#
# These are the *specification* of what the example's constants mean, not a
# transcription of what it happens to produce. They are written on the test
# side as literals so that a change to the seed, the batch size, the sweep
# direction, the domain separator, or the epoch schedule fails here instead
# of quietly redefining the proof.

EXPECTED_PERMUTATIONS = (
    (17, 5, 15, 0, 7, 20, 11, 21, 2, 19, 16, 10,
     22, 23, 1, 18, 13, 9, 8, 6, 4, 12, 14, 3),
    (19, 7, 4, 22, 6, 12, 10, 2, 21, 14, 3, 13,
     5, 18, 11, 17, 0, 1, 9, 8, 23, 20, 16, 15),
    (19, 14, 1, 17, 16, 4, 20, 12, 23, 7, 11, 22,
     13, 10, 6, 2, 0, 15, 3, 18, 5, 8, 9, 21),
)

EXPECTED_PLANS = (
    ((17, 5, 15, 0, 7, 20), (11, 21, 2, 19, 16, 10),
     (22, 23, 1, 18, 13, 9), (8, 6, 4, 12, 14, 3)),
    ((19, 7, 4, 22, 6, 12), (10, 2, 21, 14, 3, 13),
     (5, 18, 11, 17, 0, 1), (9, 8, 23, 20, 16, 15)),
    ((19, 14, 1, 17, 16, 4), (20, 12, 23, 7, 11, 22),
     (13, 10, 6, 2, 0, 15), (3, 18, 5, 8, 9, 21)),
)

# Ten steps: all four batches of epoch 0, all four of epoch 1, then the
# first two of epoch 2.
EXPECTED_INDEX_SEQUENCE = (
    EXPECTED_PLANS[0] + EXPECTED_PLANS[1] + EXPECTED_PLANS[2][:2]
)

EXPECTED_POSITIONS_BEFORE = (
    (0, 0), (0, 1), (0, 2), (0, 3),
    (1, 0), (1, 1), (1, 2), (1, 3),
    (2, 0), (2, 1),
)
EXPECTED_POSITIONS_AFTER = (
    (0, 1), (0, 2), (0, 3), (1, 0),
    (1, 1), (1, 2), (1, 3), (2, 0),
    (2, 1), (2, 2),
)

# The saved position, and the batch the archive therefore describes.
EXPECTED_SPLIT_POSITION = (1, 1)
EXPECTED_NEXT_BATCH_AT_SPLIT = (10, 2, 21, 14, 3, 13)
EXPECTED_FINAL_POSITION = (2, 2)

# The inventories after K6: one more example than Phase J closed with, and
# nothing else moved. 56 exports and 27 CTests are the phase maximum K4
# reached; `__all__` stays at 25 for the whole phase.
EXPECTED_EXAMPLE_COUNT = 17
EXPECTED_BENCHMARK_COUNT = 9
EXPECTED_EXPERIMENTAL_EXPORTS = 25
EXPECTED_ABI_EXPORTS = 56
EXPECTED_CTESTS = 27

# The raw-bit view each dtype is read through, on the test side, so the
# reconstruction below never borrows the example's own table.
_BIT_VIEW = {"float64": np.uint64, "float32": np.uint32}
_HOST_VIEW = {"float64": np.float64, "float32": np.float32}


def from_bits(values, dtype, shape):
    """Rebuild a host array from the raw bit patterns a proof recorded.

    The test's own inverse of ``bits()``, written independently so the
    column and diagonal checks below are recomputed from the recorded data
    rather than read back out of the example's booleans."""
    raw = np.asarray(values, dtype=_BIT_VIEW[dtype])
    return raw.view(_HOST_VIEW[dtype]).reshape(shape)


# ===========================================================================
# Shared fixtures — the proof is expensive, so it runs once per module
# ===========================================================================

@pytest.fixture(scope="module")
def proofs():
    """``{dtype: run_dtype_proof(dtype)}`` — the complete proof at each
    dtype, computed once. Plain Python values only."""
    return {dtype: run_dtype_proof(dtype) for dtype in RUN_DTYPES}


@pytest.fixture(scope="module")
def uninterrupted():
    """One uninterrupted float64 run, for the schedule and state checks that
    do not need the resume half."""
    return run_uninterrupted("float64")


@pytest.fixture()
def live_storages(monkeypatch):
    """The ids of every open ``NativeStorage`` — a real live-allocation
    count, so a lifecycle test can prove the count returns exactly to its
    baseline instead of trusting collection."""
    open_ids = set()
    original_init = cpp.NativeStorage.__init__
    original_close = cpp.NativeStorage.close

    def tracked_init(instance, *args, **kwargs):
        original_init(instance, *args, **kwargs)
        open_ids.add(id(instance))

    def tracked_close(instance):
        original_close(instance)
        open_ids.discard(id(instance))

    monkeypatch.setattr(cpp.NativeStorage, "__init__", tracked_init)
    monkeypatch.setattr(cpp.NativeStorage, "close", tracked_close)
    return open_ids


# ===========================================================================
# 1. The module, the CLI, and what running it leaves behind
# ===========================================================================

def test_the_example_exists_and_is_a_tracked_python_file():
    assert EXAMPLE.is_file(), EXAMPLE_RELATIVE
    source = EXAMPLE.read_text(encoding="utf-8")
    assert source.strip(), "the example is empty"
    ast.parse(source)          # it is real, parseable Python


def test_importing_the_example_runs_no_training():
    """Importing must define, never execute. Asserted structurally: the
    module body holds nothing but imports, constants, definitions, and the
    ``__main__`` guard — so there is no call at module level that could
    train, allocate, write, or print."""
    tree = ast.parse(EXAMPLE.read_text(encoding="utf-8"))
    allowed = (ast.Import, ast.ImportFrom, ast.Assign, ast.AnnAssign,
               ast.FunctionDef, ast.ClassDef, ast.Expr)
    guards = 0
    for node in tree.body:
        if isinstance(node, ast.If):
            guards += 1
            test = ast.unparse(node.test).replace('"', "'")
            assert test == "__name__ == '__main__'", test
            continue
        assert isinstance(node, allowed), ast.dump(node)[:120]
        if isinstance(node, ast.Expr):
            # The only bare expression permitted at module level is the
            # docstring.
            assert isinstance(node.value, ast.Constant), ast.dump(node)[:120]
        if isinstance(node, ast.Assign):
            # A module-level constant may not be produced by a call.
            for call in ast.walk(node.value):
                assert not isinstance(call, ast.Call), ast.unparse(node)[:120]
    assert guards == 1, "the example has no single __main__ guard"


def test_the_public_helpers_are_importable_callables():
    for helper in (train_steps, main, run_uninterrupted, run_dtype_proof,
                   evaluate_indexing, build_model, build_loss,
                   build_optimizer, build_dataset, build_loader,
                   host_arrays, index_values):
        assert callable(helper), helper


def test_the_main_guard_calls_main():
    source = EXAMPLE.read_text(encoding="utf-8")
    assert 'if __name__ == "__main__":' in source
    assert source.rstrip().endswith("main()")


def test_the_script_runs_successfully_and_reports_both_dtypes():
    """The CLI contract, end to end in a **subprocess** so nothing this
    session did can influence it."""
    result = subprocess.run(
        [sys.executable, str(EXAMPLE)],
        cwd=str(REPO_ROOT), capture_output=True, text=True, timeout=1800,
    )
    assert result.returncode == 0, result.stdout[-4000:] + result.stderr[-4000:]
    out = result.stdout
    for dtype in RUN_DTYPES:
        assert (f"exact native integer indexing resume at {dtype}: yes"
                in out), out[-3000:]
    assert "FAILED" not in out, out[-3000:]
    # The claims the milestone owes a reader, as stable fragments rather
    # than a frozen transcript.
    for fragment in (
        "native argmax produced int64 predictions",
        "index_select",
        "per-row gather",
        "read-only host int64 target",
        "version 3",
        "no cross-object atomicity",
        "no timing or performance is claimed or measured anywhere",
        "native argmax + index_select evaluation with exact interrupted",
    ):
        assert fragment in out, (fragment, out[-3000:])
    # The lifecycle line is checked as an *equality between the two numbers*
    # rather than as a literal, so it stays a real claim without turning the
    # console text into an exact contract.
    lifecycle = re.search(
        r"live native storage baseline / final:\s*(\d+)\s*/\s*(\d+)", out)
    assert lifecycle, out[-2000:]
    assert lifecycle.group(1) == lifecycle.group(2), lifecycle.group(0)


def test_running_the_script_leaves_no_file_behind():
    """Checkpoints live in a temporary directory that is removed
    automatically: no ``.npz``, cache, result, or report file may survive."""
    watched = ("", "examples", "tests", "benchmarks", "docs")
    before = {name: {path.name for path in (REPO_ROOT / name).iterdir()}
              for name in watched}
    result = subprocess.run(
        [sys.executable, str(EXAMPLE)],
        cwd=str(REPO_ROOT), capture_output=True, text=True, timeout=1800,
    )
    assert result.returncode == 0, result.stderr[-3000:]
    for name in watched:
        after = {path.name for path in (REPO_ROOT / name).iterdir()}
        new = after - before[name] - {"__pycache__"}
        assert new == set(), (name, sorted(new))
    assert not list(REPO_ROOT.glob("*.npz"))
    assert not list((REPO_ROOT / "examples").glob("*.npz"))


# The shapes a *timing claim* takes. Deliberately not a substring ban on
# "timing" or "performance": the example says in prose that it measures
# neither, and a naive scan would fail on exactly the sentence that makes
# the promise.
_TIMING_CLAIM = re.compile(
    r"\b\d+(?:\.\d+)?\s*(?:ns|us|ms|s|sec|secs|seconds?|milliseconds?)\b"
    r"|\bspeed-?ups?\b|\b\d+(?:\.\d+)?\s*x\s+(?:faster|slower)\b"
    r"|\bfaster\b|\bslower\b|\bthroughput\b|\bsamples?\s*/\s*s(?:ec)?\b"
    r"|\belapsed\b|\bwall[- ]clock\b",
    re.I)

_TIMING_IDENTIFIERS = ("time", "timeit", "perf_counter", "monotonic",
                       "process_time", "clock", "default_timer", "timer")


def test_the_example_claims_and_measures_no_timing():
    """An integration example is not a benchmark. K8 owns performance
    characterization, and nothing here may assert, print, or even measure a
    duration."""
    offenders = _TIMING_CLAIM.findall(EXAMPLE.read_text(encoding="utf-8"))
    assert offenders == [], offenders
    names = code_identifiers(EXAMPLE_RELATIVE)
    for forbidden in _TIMING_IDENTIFIERS:
        assert forbidden not in names, forbidden


def test_the_timing_scanner_can_actually_fail():
    """Negative control: the scanner must catch real timing claims, and must
    pass the sentences the example has to be able to write."""
    for detected in ("the argmax took 12.5 ms",
                     "a 3.4x faster index_select",
                     "measured throughput was high",
                     "elapsed time per epoch",
                     "2 seconds per epoch"):
        assert _TIMING_CLAIM.search(detected), detected
    for accurate in ("no timing or performance is claimed or measured "
                     "anywhere",
                     "this is an integration proof, not a benchmark",
                     "no performance is claimed at either dtype"):
        assert _TIMING_CLAIM.search(accurate) is None, accurate


# ===========================================================================
# 2. Public-API discipline — an AST scan, with a negative control
# ===========================================================================

def code_identifiers(relative):
    """Every identifier a module's **executable code** names.

    A source-text scan would be wrong here: the example explains at length
    what it deliberately does *not* do, so a prose mention of
    ``_deliver_batch`` inside its docstring would fail a substring check
    that is supposed to be about behavior. Reading the AST asks the question
    that was meant — docstrings and comments carry no identifier.

    **Keyword-argument names are collected too**, and that is load-bearing
    rather than thorough: several of the private seams K6 forbids are only
    ever *reachable* as a keyword — ``_trusted_dtype=True`` is passed, never
    named — so a scanner that read only ``Name`` and ``Attribute`` nodes
    would be blind to exactly the constructs it exists to catch."""
    tree = ast.parse((REPO_ROOT / relative).read_text(encoding="utf-8"))
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            names.add(node.id)
        elif isinstance(node, ast.Attribute):
            names.add(node.attr)
        elif isinstance(node, ast.keyword):
            if node.arg:
                names.add(node.arg)
        elif isinstance(node, (ast.ClassDef, ast.FunctionDef,
                               ast.AsyncFunctionDef)):
            names.add(node.name)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                names.add(alias.name)
                names.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            names.add(node.module or "")
            names.update(alias.name for alias in node.names)
    return names


def assigned_attributes(relative):
    """Every attribute name the module **assigns to**, so "it never mutates
    private runtime state" is a statement about writes rather than reads."""
    tree = ast.parse((REPO_ROOT / relative).read_text(encoding="utf-8"))
    written = set()
    for node in ast.walk(tree):
        targets = []
        if isinstance(node, ast.Assign):
            targets = node.targets
        elif isinstance(node, (ast.AugAssign, ast.AnnAssign)):
            targets = [node.target]
        for target in targets:
            for inner in ast.walk(target):
                if isinstance(inner, ast.Attribute):
                    written.add(inner.attr)
    return written


# Every private runtime seam the K6 contract forbids the example to touch:
# the private typed and integer constructors, the index-registry gate and
# the private dtype validators, the permutation derivation, the whole
# five-phase delivery transaction, the state-assignment and validation
# seams, the committed position fields, the private checkpoint/loader/
# sampler constants, and the fault-injection hook K7 owns.
PROHIBITED_PRIVATE_NAMES = (
    "_typed_from_array", "_typed_zeros", "_typed_full", "_from_core",
    "_trusted_dtype", "_normalize_internal_dtype", "normalize_module_dtype",
    "_from_int64_array", "_normalize_index_dtype", "_require_index_dtype",
    "_is_index_dtype", "_is_tensor_dtype", "_require_tensor_dtype",
    "_require_axis_int", "_is_axis_int", "_validated_entry_dtype",
    "_canonical_persisted_dtype", "_validated_persisted_dtype",
    "_native_permutation", "_deliver_batch", "_NativeBatchIterator",
    "_claim_batch", "_publish_pending", "_commit_pending",
    "_rollback_pending", "_complete_pending", "_release_undelivered",
    "_assign_state", "_validate_state", "_snapshot_state", "_next_position",
    "_begin_iteration", "_end_iteration", "_iteration_is_active",
    "_has_transaction", "_require_no_transaction",
    "_require_no_active_iteration", "_matching_transaction",
    "_epoch", "_cursor", "_txn_serial", "_superseded", "_to_yield",
    "_validate_dataset_identity", "_validated_indices", "_fingerprint",
    "_hash_values", "_prepare_class_targets", "_validated_metadata",
    "_FORMAT", "_FORMAT_VERSION", "_SUPPORTED_FORMAT_VERSIONS",
    "_STATE_FIELDS", "_DTYPE_CODES", "_DTYPE_NUMPY", "_DTYPE_ITEM_SIZES",
    "_require_library", "_reserve_call", "_commit_call", "_abandon_call",
    "_native_state_lock", "state_transaction", "_validate_uint64",
    "_require_exact_int", "_require_exact_keys",
    "tf_core_argmax", "tf_core_index_select",
    "tf_test_arm_alloc_failure", "fault_injection_available",
)


def test_the_example_names_no_prohibited_private_runtime_api():
    """K6 composes public behavior; it creates none. Every seam above is a
    private runtime detail, and executable example code may not read, call,
    patch, or assign one."""
    names = code_identifiers(EXAMPLE_RELATIVE)
    offenders = sorted(name for name in PROHIBITED_PRIVATE_NAMES
                       if name in names)
    assert offenders == [], offenders


def test_the_example_assigns_no_private_runtime_state():
    """The committed position moves through delivery and through a validated
    state load — never through an assignment. Asserted over **writes**
    specifically, so a read-only inspection is not confused with a
    mutation."""
    written = assigned_attributes(EXAMPLE_RELATIVE)
    for forbidden in ("_epoch", "_cursor", "_seed", "_shuffle", "_batch_size",
                      "_drop_last", "_sampler", "_dataset", "_iterator",
                      "_transaction", "_txn_serial", "_features", "_targets",
                      "_closed", "_token", "_to_yield", "_superseded",
                      "_calls", "_grad", "_version", "_storage", "_view",
                      "_core"):
        assert forbidden not in written, forbidden


def test_the_private_api_scanner_can_actually_fail():
    """Negative control for both scans above. A deliberately forbidden
    construct must be **detected** — otherwise "no offenders" would mean the
    scanner stopped matching rather than that the example is clean."""
    planted = (
        "import numpy\n"
        "def go(loader, tensor, values):\n"
        "    loader.sampler._epoch = 3\n"
        "    loader.sampler._cursor = 0\n"
        "    native_data_loader._deliver_batch(record)\n"
        "    cpp.NativeStorage._from_int64_array(values)\n"
        "    return NativeTensor._from_core(tensor, _trusted_dtype=True)\n"
    )
    tree = ast.parse(planted)
    named = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            named.add(node.id)
        elif isinstance(node, ast.Attribute):
            named.add(node.attr)
        elif isinstance(node, ast.keyword) and node.arg:
            # The keyword arm is what catches ``_trusted_dtype=True``, which
            # is never spelled as a Name or an Attribute anywhere.
            named.add(node.arg)
    for expected in ("_epoch", "_cursor", "_deliver_batch", "_from_core",
                     "_from_int64_array", "_trusted_dtype"):
        assert expected in named, expected
    assert sorted(name for name in PROHIBITED_PRIVATE_NAMES
                  if name in named) != []

    written = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                for inner in ast.walk(target):
                    if isinstance(inner, ast.Attribute):
                        written.add(inner.attr)
    assert {"_epoch", "_cursor"} <= written

    # ...and the real scanners, run on the real file, disagree with the
    # planted one — so they are reading the example and not a constant.
    assert "_epoch" not in code_identifiers(EXAMPLE_RELATIVE)
    assert "_epoch" not in assigned_attributes(EXAMPLE_RELATIVE)
    # The scanner is not vacuously empty: it really does see the example's
    # own public vocabulary, including both Phase-K operations.
    real = code_identifiers(EXAMPLE_RELATIVE)
    for present in ("NativeDataLoader", "next_batch_indices", "state_dict",
                    "load_state_dict", "save_native_checkpoint",
                    "load_native_checkpoint", "argmax", "index_select",
                    "detach", "tolist", "to_numpy"):
        assert present in real, present


def test_the_example_imports_only_public_surfaces():
    """The two modules the example is allowed to reach — the public
    experimental package and the public backend reporting surface — plus the
    standard library and NumPy. No private module, in particular, is
    imported."""
    tree = ast.parse(EXAMPLE.read_text(encoding="utf-8"))
    modules = set()
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            modules.add(node.module or "")
            if node.module == "tensorforge.experimental":
                imported.update(alias.name for alias in node.names)
    assert modules == {"gc", "os", "tempfile", "numpy",
                       "tensorforge.backends", "tensorforge.experimental"}, (
        sorted(modules))
    # Every name taken from the experimental package is an exported one.
    import tensorforge.experimental as experimental

    assert imported, "the example imports nothing from the public package"
    assert imported <= set(experimental.__all__), sorted(
        imported - set(experimental.__all__))


def test_the_example_adds_no_public_export():
    """K6's export delta is zero. The example's own model class is an
    example implementation detail and must never become a public name."""
    import tensorforge
    import tensorforge.experimental as experimental

    assert len(experimental.__all__) == EXPECTED_EXPERIMENTAL_EXPORTS
    assert len(set(experimental.__all__)) == len(experimental.__all__)
    for invented in ("NativeIndexingClassifier", "native_integer_indexing",
                     "evaluate_indexing", "native_argmax", "gather",
                     "NativeIndexer"):
        assert invented not in experimental.__all__, invented
        assert invented not in tensorforge.__all__, invented
    assert not hasattr(experimental, "NativeIndexingClassifier")
    assert not hasattr(tensorforge, "NativeIndexingClassifier")


# ===========================================================================
# 3. The NumPy boundary — host work is allowed, native compute is not
#    replaced
# ===========================================================================

# Everything the example is allowed to ask NumPy for: building the host
# dataset at an explicit width, reading raw bits back after an explicit
# ``to_numpy()``, and reporting. Every entry is a host-boundary or
# inspection operation; none of them is arithmetic — and ``argmax`` is
# deliberately **not** in the set, because the whole point of K6 is that the
# prediction indices come from the *native* operation.
ALLOWED_NUMPY_ATTRIBUTES = {
    "asarray", "ascontiguousarray", "arange", "diagonal", "ndarray",
    "float64", "float32", "uint64", "uint32", "int64",
}


def numpy_attributes(relative):
    """Every ``np.<name>`` the module's executable code reaches for."""
    tree = ast.parse((REPO_ROOT / relative).read_text(encoding="utf-8"))
    used = set()
    for node in ast.walk(tree):
        if (isinstance(node, ast.Attribute)
                and isinstance(node.value, ast.Name)
                and node.value.id == "np"):
            used.add(node.attr)
    return used


def test_numpy_never_replaces_the_native_computation():
    """The standing tripwire, in its source-structure half: NumPy may build
    the host dataset, read bits back, and report — it may not compute a
    forward, an activation, a cross entropy, a backward, a parameter update,
    or **an argmax**."""
    used = numpy_attributes(EXAMPLE_RELATIVE)
    assert used, "the scanner found no numpy use at all — it is not reading"
    unexpected = sorted(used - ALLOWED_NUMPY_ATTRIBUTES)
    assert unexpected == [], unexpected
    for arithmetic in ("argmax", "argmin", "take", "argsort", "sort", "dot",
                       "matmul", "einsum", "exp", "log", "maximum", "mean",
                       "var", "std", "sqrt", "tanh", "clip", "where",
                       "power", "divide", "multiply", "add", "subtract",
                       "random", "default_rng", "seed", "shuffle",
                       "permutation", "softmax"):
        assert arithmetic not in used, arithmetic


def test_the_numpy_scanner_can_actually_fail():
    """Negative control: a planted NumPy forward and a planted NumPy argmax
    must both be detected."""
    planted = ("import numpy as np\n"
               "def forward(x, w):\n"
               "    return np.argmax(np.matmul(x, w), axis=1)\n")
    tree = ast.parse(planted)
    used = {node.attr for node in ast.walk(tree)
            if isinstance(node, ast.Attribute)
            and isinstance(node.value, ast.Name) and node.value.id == "np"}
    assert {"matmul", "argmax"} <= used
    assert sorted(used - ALLOWED_NUMPY_ATTRIBUTES) == ["argmax", "matmul"]


def test_the_evaluation_really_runs_the_two_native_integer_operations(
        monkeypatch):
    """The tripwire's **runtime** half: one real evaluation is proved to
    drive ``NativeTensor.argmax`` and ``NativeTensor.index_select``. If a
    host round trip had replaced either, the corresponding counter would
    still read zero.

    The counters are the non-vacuity control on themselves: each is asserted
    zero before the call and nonzero after."""
    counts = {"argmax": 0, "index_select": 0, "detach": 0}
    originals = {name: getattr(NativeTensor, name) for name in counts}

    def counted(name):
        original = originals[name]

        def wrapper(self, *args, **kwargs):
            counts[name] += 1
            return original(self, *args, **kwargs)

        return wrapper

    for name in counts:
        monkeypatch.setattr(NativeTensor, name, counted(name))

    dataset = build_dataset("float64")
    loader = model = None
    try:
        loader, _sampler = build_loader(dataset)
        model = build_model("float64")
        assert counts == {"argmax": 0, "index_select": 0, "detach": 0}
        iterator = iter(loader)
        try:
            features, _targets = next(iterator)
        finally:
            iterator.close()
        logits = model(features)
        try:
            record = evaluate_indexing(logits, "float64")
        finally:
            logits.close()
            features.close()
        assert counts["argmax"] == 1, counts
        assert counts["index_select"] == 1, counts
        assert counts["detach"] == 1, counts
        assert record["prediction_dtype"] == INDEX_DTYPE
    finally:
        if loader is not None:
            loader.close()
        dataset.close()
        if model is not None:
            for parameter in model.parameters():
                parameter.close()


# ===========================================================================
# 4. The deterministic host dataset
# ===========================================================================

def test_the_host_dataset_is_deterministic_and_independently_contained():
    """Repeated construction returns equal logical values in independent
    containers — no shared list, no shared buffer, no cached array."""
    first_features, first_targets = build_features(), build_targets()
    second_features, second_targets = build_features(), build_targets()
    assert first_features == second_features
    assert first_targets == second_targets
    assert first_features is not second_features
    assert all(a is not b for a, b in zip(first_features, second_features))
    first_features[0][0] = 99.0
    assert build_features()[0][0] != 99.0


def test_the_host_dataset_has_the_contracted_dimensions_and_classes():
    features, targets = build_features(), build_targets()
    assert len(features) == SAMPLES == 24
    assert all(len(row) == FEATURES == 5 for row in features)
    assert len(targets) == SAMPLES
    assert set(targets) == set(range(NUM_CLASSES)) == {0, 1, 2, 3}
    assert NUM_CLASSES >= 3, "the task must have at least three classes"
    # Every class occurs several times, so no class is a singleton.
    for label in range(NUM_CLASSES):
        assert targets.count(label) == SAMPLES // NUM_CLASSES == 6
    # ...and the task is not degenerate: the rows are genuinely different.
    distinct = {tuple(row) for row in features}
    assert len(distinct) == SAMPLES, len(distinct)


def test_every_feature_value_is_an_exact_binary_fraction():
    """Every value is a multiple of one eighth, so it is representable
    **exactly** in binary32 and binary64 alike — which is why the same
    literals can seed both runs without either being a rounded copy."""
    for row in build_features():
        for value in row:
            scaled = value * 8.0
            assert scaled == int(scaled), value
            assert float(np.float32(value)) == value, value


def test_the_host_arrays_physically_match_the_requested_dtype():
    for dtype in RUN_DTYPES:
        features, targets = host_arrays(dtype)
        assert features.dtype == np.dtype(dtype)
        assert targets.dtype == np.int64
        assert targets.ndim == 1
        assert features.shape == (SAMPLES, FEATURES)


def test_the_two_dtypes_carry_the_same_logical_dataset():
    """The float32 and float64 host arrays hold the same *values* — which is
    what makes the cross-dtype batch-index claim meaningful — while staying
    physically distinct arrays at distinct widths."""
    wide, wide_targets = host_arrays("float64")
    narrow, narrow_targets = host_arrays("float32")
    assert wide.dtype != narrow.dtype
    assert np.array_equal(wide_targets, narrow_targets)
    # Exact, not approximate: every value is representable at both widths,
    # so the float32 array widened back is bit-for-bit the float64 one.
    assert bits(narrow.astype(np.float64), "float64") == bits(wide, "float64")


def test_the_dataset_is_constructed_with_an_explicit_dtype():
    """The dtype is chosen, never inferred. A float32 host array with the
    argument omitted would give a **float64** dataset, and the example never
    relies on that."""
    for dtype in RUN_DTYPES:
        dataset = build_dataset(dtype)
        try:
            assert dataset.dtype == dtype
            assert dataset.samples == SAMPLES
            assert dataset.feature_shape == (FEATURES,)
            assert dataset.device == "cpu"
        finally:
            dataset.close()
    features, targets = host_arrays("float32")
    inferred = NativeTensorDataset(features, targets)
    try:
        assert inferred.dtype == "float64"
    finally:
        inferred.close()


def test_the_dataset_construction_reads_no_file_clock_or_random_source():
    """Structural: the example names no filesystem read, no network, no
    clock, and no global random stream anywhere."""
    names = code_identifiers(EXAMPLE_RELATIVE)
    for forbidden in ("random", "default_rng", "RandomState", "urlopen",
                      "requests", "urllib", "socket", "download", "read_csv",
                      "load", "loadtxt", "genfromtxt", "getenv", "environ",
                      "open"):
        assert forbidden not in names, forbidden


# ===========================================================================
# 5. The sampler and the loader — the committed plan
# ===========================================================================

def test_the_loader_is_shuffled_at_the_committed_configuration():
    dataset = build_dataset("float64")
    loader = None
    try:
        loader, sampler = build_loader(dataset)
        assert sampler.shuffle is True is SHUFFLE
        assert sampler.seed == SAMPLER_SEED
        assert sampler.batch_size == BATCH_SIZE == 6
        assert sampler.drop_last is False is DROP_LAST
        assert sampler.epoch == 0 and sampler.cursor == 0
        assert sampler.batches_per_epoch == BATCHES_PER_EPOCH == 4
        assert isinstance(loader, NativeDataLoader)
        assert isinstance(sampler, NativeBatchSampler)
    finally:
        if loader is not None:
            loader.close()
        dataset.close()


def test_the_committed_permutations_and_plans_are_exactly_the_expected_ones():
    """The literal expected values above are the specification. A change to
    the seed, the batch size, the sweep direction, or the key schedule fails
    here rather than silently redefining what the proof proves."""
    dataset = build_dataset("float64")
    loader = None
    try:
        loader, sampler = build_loader(dataset)
        for epoch, expected in enumerate(EXPECTED_PERMUTATIONS):
            assert sampler.epoch_permutation(epoch) == expected, epoch
            assert sampler.plan(epoch) == EXPECTED_PLANS[epoch], epoch
    finally:
        if loader is not None:
            loader.close()
        dataset.close()


def test_the_exercised_orders_are_non_identity_and_mutually_distinct():
    """Non-vacuity, proved from the committed plan rather than from
    probability: shuffling really reorders, and two exercised epochs really
    differ."""
    identity = tuple(range(SAMPLES))
    for order in EXPECTED_PERMUTATIONS:
        assert sorted(order) == list(identity)      # a real permutation
        assert order != identity
    assert len(set(EXPECTED_PERMUTATIONS)) == len(EXPECTED_PERMUTATIONS)
    assert len(EXPECTED_PERMUTATIONS) == EXERCISED_EPOCHS == 3


def test_the_split_is_genuinely_mid_epoch_with_batches_still_owed():
    assert 0 < SPLIT_STEP < TOTAL_STEPS
    assert SPLIT_STEP != TOTAL_STEPS - 1, "the split must not be the last step"
    assert SPLIT_STEP % BATCHES_PER_EPOCH != 0, "the split is an epoch boundary"
    assert EXPECTED_POSITIONS_BEFORE[SPLIT_STEP] == EXPECTED_SPLIT_POSITION
    epoch, cursor = EXPECTED_SPLIT_POSITION
    assert cursor > 0, "at least one batch must have been delivered"
    assert BATCHES_PER_EPOCH - cursor >= 1, "batches must remain in the epoch"
    assert TOTAL_STEPS - SPLIT_STEP > 1, "the resumed suffix must be multi-step"
    # The run crosses at least one epoch boundary.
    assert len({epoch for epoch, _ in EXPECTED_POSITIONS_BEFORE}) >= 2


def test_the_evaluation_schedule_straddles_the_interruption():
    """K6's own scheduling requirement: the integer evaluation runs on both
    sides of the checkpoint, so the resumed run has to reproduce indexing it
    did not itself compute the first half of."""
    assert sorted(EVAL_STEPS) == list(EVAL_STEPS), "eval steps must be sorted"
    assert len(set(EVAL_STEPS)) == len(EVAL_STEPS)
    assert all(0 <= step < TOTAL_STEPS for step in EVAL_STEPS)
    assert [step for step in EVAL_STEPS if step < SPLIT_STEP]
    assert [step for step in EVAL_STEPS if step >= SPLIT_STEP]


def test_the_delivered_rows_match_the_publicly_planned_indices():
    """The batch a caller can *predict* through the public planning API is
    the batch the loader then delivers — indices, and the actual feature
    rows behind them."""
    dataset = build_dataset("float64")
    loader = None
    try:
        loader, sampler = build_loader(dataset)
        host, _targets = host_arrays("float64")
        iterator = iter(loader)
        try:
            for expected in EXPECTED_PLANS[0]:
                indices = sampler.next_batch_indices()
                assert indices == expected
                features, targets = next(iterator)
                try:
                    assert tuple(features.shape) == (BATCH_SIZE, FEATURES)
                    assert bits(features.to_numpy(), "float64") == bits(
                        host[list(indices)], "float64")
                    assert targets.tolist() == [index % NUM_CLASSES
                                                for index in indices]
                finally:
                    features.close()
        finally:
            iterator.close()
        assert (sampler.epoch, sampler.cursor) == (1, 0)
    finally:
        if loader is not None:
            loader.close()
        dataset.close()


def test_every_delivered_target_batch_stays_read_only_host_int64(
        uninterrupted):
    """The Phase-J delivery contract, unchanged by Phase K: the targets are
    a read-only host ``numpy.ndarray`` of dtype ``int64``, never a native
    tensor, and never what ``argmax`` produced."""
    for step in uninterrupted["steps"]:
        targets = step["targets"]
        assert targets["dtype"] == "int64"
        assert targets["shape"] == (BATCH_SIZE,)
        assert targets["c_contiguous"] is True
        assert targets["owndata"] is True
        assert targets["writeable"] is False
        assert targets["is_ndarray"] is True
        assert targets["values"] == [index % NUM_CLASSES
                                     for index in step["indices"]]


def test_every_delivered_feature_batch_is_owning_contiguous_and_floating(
        uninterrupted):
    for step in uninterrupted["steps"]:
        features = step["features"]
        assert features["dtype"] == "float64"
        assert features["shape"] == (BATCH_SIZE, FEATURES)
        assert features["device"] == "cpu"
        assert features["contiguous"] is True
        assert features["owns_core"] is True
        assert features["requires_grad"] is False
        assert len(features["bits"]) == BATCH_SIZE * FEATURES


# ===========================================================================
# 6. The model, loss, and optimizer
# ===========================================================================

def test_the_model_is_a_real_public_native_classifier():
    model = build_model("float64")
    try:
        assert isinstance(model, NativeModule)
        names = [name for name, _ in model.named_parameters()]
        assert names == ["hidden.weight", "hidden.bias",
                         "output.weight", "output.bias"], names
        # Deliberately no buffers and no registered generator: K6's subject
        # is the indexing, and the example never claims state it lacks.
        assert list(model.named_buffers()) == []
        assert list(model.named_generators()) == []
        weight = dict(model.named_parameters())["output.weight"]
        assert tuple(weight.shape) == (HIDDEN, NUM_CLASSES)
        assert tuple(dict(model.named_parameters())["hidden.weight"].shape) == (
            FEATURES, HIDDEN)
    finally:
        for parameter in model.parameters():
            parameter.close()


def test_the_loss_is_the_native_fused_cross_entropy():
    from tensorforge.experimental import NativeCrossEntropyLoss

    criterion = build_loss()
    assert isinstance(criterion, NativeCrossEntropyLoss)
    assert isinstance(criterion, NativeModule)


def test_the_optimizer_is_native_adam_with_nontrivial_state(uninterrupted):
    state = uninterrupted["optimizer"]
    assert state["optimizer"] == "NativeAdam"
    assert state["format_version"] == 1
    assert state["lr"] == DEFAULT_LR
    assert len(state["m"]) == len(state["v"]) == 4
    assert state["step_counts"] == [TOTAL_STEPS] * 4
    assert all(moment["device"] == "cpu" for moment in state["m"])
    assert all(moment["dtype"] == "float64" for moment in state["m"])


# ===========================================================================
# 7. The uninterrupted run
# ===========================================================================

def test_the_uninterrupted_run_takes_exactly_one_batch_per_step(
        uninterrupted):
    steps = uninterrupted["steps"]
    assert len(steps) == TOTAL_STEPS
    assert [step["step"] for step in steps] == list(range(TOTAL_STEPS))
    for index, step in enumerate(steps):
        assert (step["epoch_before"], step["cursor_before"]) == (
            EXPECTED_POSITIONS_BEFORE[index])
        assert (step["epoch_after"], step["cursor_after"]) == (
            EXPECTED_POSITIONS_AFTER[index])


def test_the_uninterrupted_batch_sequence_is_the_committed_one(uninterrupted):
    assert tuple(uninterrupted["index_sequence"]) == EXPECTED_INDEX_SEQUENCE
    assert tuple(uninterrupted["position_sequence"]) == (
        EXPECTED_POSITIONS_BEFORE)
    assert tuple(uninterrupted["epoch_permutations"]) == EXPECTED_PERMUTATIONS
    assert (uninterrupted["loader"]["epoch"],
            uninterrupted["loader"]["cursor"]) == EXPECTED_FINAL_POSITION


def test_the_uninterrupted_run_advances_every_state_family(uninterrupted):
    initial = uninterrupted["initial"]
    assert uninterrupted["parameters"] != initial["parameters"]
    assert uninterrupted["optimizer"]["step_counts"] == [TOTAL_STEPS] * 4
    assert any(any(pattern != 0 for pattern in moment["bits"])
               for moment in uninterrupted["optimizer"]["m"])
    losses = [tuple(step["loss_bits"]) for step in uninterrupted["steps"]]
    assert len(set(losses)) > 1, "the loss sequence is constant"
    assert uninterrupted["gradients_cleared"] is True


def test_the_completed_run_returns_only_plain_python(uninterrupted):
    """No live native object, model, optimizer, loader, dataset, sampler, or
    generator may survive in a completed proof's result."""
    _assert_plain_python(uninterrupted)


def _assert_plain_python(value, path="record"):
    forbidden = (NativeTensor, NativeModule, NativeDataLoader,
                 NativeBatchSampler, NativeTensorDataset, NativeGenerator,
                 np.ndarray)
    if isinstance(value, dict):
        for key, item in value.items():
            _assert_plain_python(item, f"{path}[{key!r}]")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _assert_plain_python(item, f"{path}[{index}]")
    else:
        assert not isinstance(value, forbidden), (path, type(value))
        assert isinstance(value, (int, float, str, bool, type(None))), (
            path, type(value))


# ===========================================================================
# 8. The integer evaluation path — argmax
# ===========================================================================

def test_every_prediction_tensor_is_an_owning_contiguous_int64_leaf(proofs):
    """§17.3 and §17.9, observed on the real results: a fresh owning
    contiguous ``int64`` tensor that is a plain leaf even though its source
    was a live gradient-tracking forward output."""
    for dtype in RUN_DTYPES:
        for step, record in proofs[dtype]["evaluations"].items():
            assert record["prediction_dtype"] == INDEX_DTYPE, (dtype, step)
            assert record["prediction_dtype"] not in cpp.SUPPORTED_DTYPES
            assert record["prediction_dtype"] in cpp.INDEX_DTYPES
            assert record["prediction_shape"] == (BATCH_SIZE,), (dtype, step)
            assert record["prediction_numel"] == BATCH_SIZE
            assert record["prediction_device"] == "cpu"
            assert record["prediction_contiguous"] is True
            assert record["prediction_owns_core"] is True
            assert record["prediction_requires_grad"] is False
            assert record["prediction_is_leaf"] is True
            assert record["prediction_grad_is_none"] is True


def test_every_prediction_index_is_an_exact_in_range_python_int(proofs):
    for dtype in RUN_DTYPES:
        for step, record in proofs[dtype]["evaluations"].items():
            predictions = record["predictions"]
            assert len(predictions) == BATCH_SIZE, (dtype, step)
            for value in predictions:
                assert type(value) is int, (dtype, step, type(value))
                assert 0 <= value < NUM_CLASSES, (dtype, step, value)
            assert record["predictions_in_range"] is True
            assert record["predictions_are_exact_ints"] is True


def test_each_prediction_really_names_a_maximum_of_its_own_row(proofs):
    """An independent check on the *values*, recomputed here from the
    recorded logits rather than read out of the example.

    Deliberately stated as "the selected value is a maximum of its row"
    rather than "it equals ``numpy.argmax``": §20.3 declines to claim the
    two tie rules are equivalent, and this form is true under either while
    still failing on a wrong index."""
    for dtype in RUN_DTYPES:
        for step, record in proofs[dtype]["evaluations"].items():
            logits = from_bits(record["logit_bits"], dtype,
                               record["logit_shape"])
            for row, index in enumerate(record["predictions"]):
                row_values = logits[row]
                assert row_values[index] == max(row_values), (dtype, step, row)


def test_the_predictions_come_from_the_native_operation_not_a_host_round_trip(
        proofs):
    """The recorded prediction dtype is the native index dtype, and the
    example's own NumPy vocabulary contains no ``argmax`` at all — so the
    indices cannot have come from a host reduction."""
    assert "argmax" not in numpy_attributes(EXAMPLE_RELATIVE)
    assert "argmax" in code_identifiers(EXAMPLE_RELATIVE)
    for dtype in RUN_DTYPES:
        assert proofs[dtype]["indexing"]["every_prediction_is_int64"] is True


# ===========================================================================
# 9. The integer evaluation path — index_select, and what it is not
# ===========================================================================

def test_every_selection_is_a_fresh_owning_graph_free_source_dtype_copy(
        proofs):
    """§18.4 and §18.8, observed on the real results."""
    for dtype in RUN_DTYPES:
        for step, record in proofs[dtype]["evaluations"].items():
            assert record["selected_dtype"] == dtype, (dtype, step)
            assert record["detached_dtype"] == dtype
            assert record["detached_requires_grad"] is False
            assert record["selected_device"] == "cpu"
            assert record["selected_contiguous"] is True
            assert record["selected_owns_core"] is True
            assert record["selected_requires_grad"] is False
            assert record["selected_is_leaf"] is True
            assert record["selected_grad_is_none"] is True


def test_the_selection_is_axis_selection_and_not_a_per_row_gather(proofs):
    """The shape claim, stated as the thing it actually is.

    ``index_select(1, predictions)`` selects the **same ordered index vector
    along the class axis for every row**, so a ``(batch, classes)`` source
    and a ``(batch,)`` index give a ``(batch, batch)`` result — not a
    ``(batch,)`` per-row gather, which is a different operation TensorForge
    does not have."""
    for dtype in RUN_DTYPES:
        for step, record in proofs[dtype]["evaluations"].items():
            assert record["logit_shape"] == (BATCH_SIZE, NUM_CLASSES)
            assert record["selected_shape"] == (BATCH_SIZE, BATCH_SIZE), (
                dtype, step, record["selected_shape"])
            assert record["selected_shape"] != (BATCH_SIZE,), "a gather shape"
            assert record["selected_shape"][CLASS_AXIS] == len(
                record["predictions"])
            assert record["class_axis_length_is_prediction_count"] is True
            assert record["result_is_square_batch"] is True
            assert len(record["selected_bits"]) == BATCH_SIZE * BATCH_SIZE


def test_every_selected_column_is_the_whole_source_class_column(proofs):
    """Recomputed here from the recorded bits, column by column, rather than
    trusting the example's boolean: column *j* of the result must be the
    entire source column ``predictions[j]``, bit for bit."""
    for dtype in RUN_DTYPES:
        for step, record in proofs[dtype]["evaluations"].items():
            logits = from_bits(record["logit_bits"], dtype,
                               record["logit_shape"])
            selected = from_bits(record["selected_bits"], dtype,
                                 record["selected_shape"])
            for position, index in enumerate(record["predictions"]):
                assert bits(np.ascontiguousarray(selected[:, position]),
                            dtype) == bits(
                    np.ascontiguousarray(logits[:, index]), dtype), (
                    dtype, step, position)
            assert record["columns_match_source_columns"] is True


def test_the_diagonal_is_each_examples_own_predicted_class_logit(proofs):
    """The relation that makes this composition useful, recomputed here:
    ``selected[row, row] == logits[row, predictions[row]]``, compared as raw
    bits."""
    for dtype in RUN_DTYPES:
        for step, record in proofs[dtype]["evaluations"].items():
            logits = from_bits(record["logit_bits"], dtype,
                               record["logit_shape"])
            selected = from_bits(record["selected_bits"], dtype,
                                 record["selected_shape"])
            diagonal = np.ascontiguousarray(
                np.asarray([selected[row][row]
                            for row in range(BATCH_SIZE)], dtype=logits.dtype))
            expected = np.ascontiguousarray(
                np.asarray([logits[row][record["predictions"][row]]
                            for row in range(BATCH_SIZE)], dtype=logits.dtype))
            assert bits(diagonal, dtype) == bits(expected, dtype), (dtype, step)
            assert bits(diagonal, dtype) == record["diagonal_bits"]
            assert record["diagonal_is_predicted_logits"] is True


def test_duplicate_predictions_are_preserved_in_order(proofs):
    """Duplicates really occur, and where an index repeats the result's
    columns are identical **and stay in their original positions** — which
    is the observable form of "duplicates and order are preserved"."""
    for dtype in RUN_DTYPES:
        duplicates_seen = 0
        for step, record in proofs[dtype]["evaluations"].items():
            predictions = record["predictions"]
            selected = from_bits(record["selected_bits"], dtype,
                                 record["selected_shape"])
            if len(set(predictions)) < len(predictions):
                duplicates_seen += 1
            for left in range(len(predictions)):
                for right in range(left + 1, len(predictions)):
                    left_column = bits(
                        np.ascontiguousarray(selected[:, left]), dtype)
                    right_column = bits(
                        np.ascontiguousarray(selected[:, right]), dtype)
                    if predictions[left] == predictions[right]:
                        assert left_column == right_column, (dtype, step)
                    else:
                        # Distinct indices need not give distinct columns in
                        # general, but the *positions* must still follow the
                        # index order, which the column check above pins.
                        assert left_column == bits(
                            np.ascontiguousarray(
                                from_bits(record["logit_bits"], dtype,
                                          record["logit_shape"])
                                [:, predictions[left]]), dtype)
            assert record["duplicate_columns_identical"] is True
        assert duplicates_seen > 0, (dtype, "no duplicate prediction occurred")
        # ...and duplicates are structural rather than lucky.
        assert BATCH_SIZE > NUM_CLASSES


def test_the_index_select_source_must_be_detached():
    """§18.9, exercised directly: a gradient-tracking source is **rejected**
    with a message naming ``detach()``, which is exactly why the example
    detaches instead of passing the live logits."""
    dataset = build_dataset("float64")
    loader = model = None
    try:
        loader, _sampler = build_loader(dataset)
        model = build_model("float64")
        iterator = iter(loader)
        try:
            features, _targets = next(iterator)
        finally:
            iterator.close()
        logits = model(features)
        predictions = logits.argmax(axis=CLASS_AXIS)
        try:
            assert logits.requires_grad is True
            with pytest.raises(ValueError) as excinfo:
                logits.index_select(CLASS_AXIS, predictions)
            assert "detach" in str(excinfo.value)
            # ...and the detached form works, which is the example's route.
            detached = logits.detach()
            try:
                selected = detached.index_select(CLASS_AXIS, predictions)
                selected.close()
            finally:
                detached.close()
        finally:
            predictions.close()
            logits.close()
            features.close()
    finally:
        if loader is not None:
            loader.close()
        dataset.close()
        if model is not None:
            for parameter in model.parameters():
                parameter.close()


# ===========================================================================
# 10. The interrupted run, the archive, and the restore ordering
# ===========================================================================

def test_the_snapshot_precedes_the_save_with_no_delivery_between(proofs):
    for dtype in RUN_DTYPES:
        proof = proofs[dtype]
        assert proof["journal_tail"] == [
            ("deliver", SPLIT_STEP - 1),
            ("loader_state_dict", SPLIT_STEP),
            ("save_checkpoint", SPLIT_STEP),
        ], proof["journal_tail"]
        assert proof["snapshot_immediately_precedes_save"] is True


def test_next_step_is_the_number_of_completed_steps(proofs):
    for dtype in RUN_DTYPES:
        proof = proofs[dtype]
        assert proof["next_step"] == SPLIT_STEP
        assert proof["next_step_is_split"] is True
        assert proof["one_batch_per_step"] is True
        metadata = proof["checkpoint_metadata"]
        assert metadata[TRAINING_KEY][NEXT_STEP_KEY] == SPLIT_STEP


def test_the_loader_state_describes_the_exact_next_batch(proofs):
    for dtype in RUN_DTYPES:
        proof = proofs[dtype]
        assert proof["next_batch_at_interruption"] == (
            EXPECTED_NEXT_BATCH_AT_SPLIT)
        assert proof["next_batch_at_interruption"] == (
            EXPECTED_INDEX_SEQUENCE[SPLIT_STEP])
        sampler_state = proof["checkpoint_metadata"][TRAINING_KEY][
            LOADER_KEY]["sampler"]
        assert (sampler_state["epoch"], sampler_state["cursor"]) == (
            EXPECTED_SPLIT_POSITION)


def test_the_archive_is_a_real_unchanged_version_three_checkpoint(proofs):
    """The loader state travels as **caller metadata** through the existing
    version-3 archive. No root field, no version 4, and no runtime constant
    spelling a caller convention."""
    assert native_checkpoint._FORMAT == "tensorforge.native_checkpoint"
    assert native_checkpoint._FORMAT_VERSION == 3
    assert native_checkpoint._SUPPORTED_FORMAT_VERSIONS == (1, 2, 3)
    assert native_optimizer_state.FORMAT_VERSION == 1
    assert native_data_loader._FORMAT == "tensorforge.native_data_loader"
    assert native_data_loader._FORMAT_VERSION == 1
    assert native_sampler._FORMAT == "tensorforge.native_sampler"
    assert native_sampler._FORMAT_VERSION == 1

    for dtype in RUN_DTYPES:
        state = proofs[dtype]["checkpoint_metadata"][TRAINING_KEY][LOADER_KEY]
        assert set(state) == {"format", "format_version", "sampler"}
        assert state["format"] == "tensorforge.native_data_loader"
        assert state["format_version"] == 1
        assert set(state["sampler"]) == {
            "format", "format_version", "dataset", "seed", "shuffle",
            "batch_size", "drop_last", "epoch", "cursor"}

    tree = ast.parse((REPO_ROOT / "src" / "tensorforge" / "experimental"
                      / "native_checkpoint.py").read_text(encoding="utf-8"))
    literals = {node.value for node in ast.walk(tree)
                if isinstance(node, ast.Constant)
                and isinstance(node.value, str)}
    assert "tensorforge.native_checkpoint" in literals      # the control
    for convention in (TRAINING_KEY, LOADER_KEY, NEXT_STEP_KEY):
        assert convention not in literals, convention


def test_the_fresh_restore_target_is_deliberately_different(proofs):
    """The proof cannot pass vacuously: every family of the restore target
    is proved to start somewhere else **before** the load."""
    assert FRESH_SAMPLER_SEED != SAMPLER_SEED
    assert FRESH_BATCH_SIZE != BATCH_SIZE
    assert FRESH_SHUFFLE is not SHUFFLE
    assert FRESH_LR != DEFAULT_LR
    for dtype in RUN_DTYPES:
        proof = proofs[dtype]
        assert proof["fresh_started_different"] is True
        assert proof["fresh_shares_no_identity"] is True
        assert proof["identities_preserved"] is True


def test_the_load_order_is_checkpoint_first_then_loader():
    """Structural: the example's restore path calls
    ``load_native_checkpoint`` before ``loader.load_state_dict``, and never
    the other way round."""
    tree = ast.parse(EXAMPLE.read_text(encoding="utf-8"))
    restore = next(node for node in ast.walk(tree)
                   if isinstance(node, ast.FunctionDef)
                   and node.name == "_restore_and_finish")
    order = []
    for node in ast.walk(restore):
        if isinstance(node, ast.Call):
            rendered = ast.unparse(node.func)
            if rendered == "load_native_checkpoint":
                order.append(("checkpoint", node.lineno))
            elif rendered.endswith(".load_state_dict"):
                order.append(("loader", node.lineno))
    assert [name for name, _ in order] == ["checkpoint", "loader"], order
    assert order[0][1] < order[1][1]


def test_the_restored_loader_adopted_every_saved_value(proofs):
    for dtype in RUN_DTYPES:
        proof = proofs[dtype]
        assert proof["loader_adopted_saved_state"] is True
        assert proof["loader_next_batch_matches_saved"] is True
        assert proof["next_batch_after_restore_matches"] is True


def test_configuration_is_adopted_from_the_state_not_from_the_constructor():
    """A restored sampler may legitimately report a different ``batch_size``
    than its constructor was given — the six configuration and position
    values are **adopted**, while the four dataset identity fields are
    validated against live reality and never adopted."""
    source_dataset = build_dataset("float64")
    target_dataset = build_dataset("float64")
    source = target = None
    try:
        source, source_sampler = build_loader(source_dataset)
        target, target_sampler = build_loader(
            target_dataset, seed=FRESH_SAMPLER_SEED,
            batch_size=FRESH_BATCH_SIZE, shuffle=FRESH_SHUFFLE)
        advance_loader(source, SPLIT_STEP)
        assert (source_sampler.epoch, source_sampler.cursor) == (
            EXPECTED_SPLIT_POSITION)
        state = source.state_dict()
        assert target_sampler.batch_size == FRESH_BATCH_SIZE
        target.load_state_dict(state)
        assert target_sampler.batch_size == BATCH_SIZE
        assert target_sampler.seed == SAMPLER_SEED
        assert target_sampler.shuffle is SHUFFLE
        assert (target_sampler.epoch, target_sampler.cursor) == (
            EXPECTED_SPLIT_POSITION)
        assert target_sampler.next_batch_indices() == (
            EXPECTED_NEXT_BATCH_AT_SPLIT)
        assert target.sampler is target_sampler
        assert target_sampler.dataset is target_dataset
    finally:
        for loader in (source, target):
            if loader is not None:
                loader.close()
        source_dataset.close()
        target_dataset.close()


# ===========================================================================
# 11. Exact equality — the whole comparison inventory
# ===========================================================================

@pytest.mark.parametrize("check", REQUIRED)
def test_every_required_exact_check_holds_at_every_dtype(proofs, check):
    for dtype in RUN_DTYPES:
        assert proofs[dtype][check] is True, (dtype, check)


@pytest.mark.parametrize("check", REQUIRED_INDEXING)
def test_every_indexing_non_vacuity_check_holds(proofs, check):
    for dtype in RUN_DTYPES:
        assert proofs[dtype]["indexing"][check] is True, (dtype, check)


@pytest.mark.parametrize("check", REQUIRED_TRAINING)
def test_every_training_non_vacuity_check_holds(proofs, check):
    for dtype in RUN_DTYPES:
        assert proofs[dtype]["training"][check] is True, (dtype, check)


@pytest.mark.parametrize("check", REQUIRED_SCHEDULE)
def test_every_schedule_non_vacuity_check_holds(proofs, check):
    for dtype in RUN_DTYPES:
        assert proofs[dtype]["schedule"][check] is True, (dtype, check)


def test_the_example_reports_no_failed_check(proofs):
    for dtype in RUN_DTYPES:
        assert failed_checks(proofs[dtype]) == [], dtype


def test_the_exact_comparison_inventory_is_complete():
    """Every row K6 owes is present in the gate by name, so a row cannot be
    dropped from the proof without failing here."""
    for row in ("next_batch_after_restore_matches", "suffix_indices_match",
                "whole_index_sequence_matches", "feature_batches_match",
                "target_batches_match", "parameters_match", "moments_match",
                "counters_match", "hyperparameters_match",
                "final_loader_state_matches", "loss_sequence_matches",
                "logits_match", "epoch_boundaries_match",
                "position_sequence_matches", "prediction_indices_match",
                "selected_bits_match", "diagonal_bits_match",
                "whole_evaluation_record_matches", "suffix_evaluations_match",
                "evaluation_steps_match"):
        assert row in REQUIRED, row


def test_the_float64_uninterrupted_and_resumed_runs_match_exactly(proofs):
    """The whole float64 claim in one place, stated as the milestone states
    it rather than only as a parametrized sweep."""
    proof = proofs["float64"]
    assert proof["dtype"] == "float64"
    assert failed_checks(proof) == []
    assert proof["whole_index_sequence_matches"] is True
    assert proof["loss_sequence_matches"] is True
    assert proof["parameters_match"] is True
    assert proof["moments_match"] is True
    assert proof["prediction_indices_match"] is True
    assert proof["whole_evaluation_record_matches"] is True


def test_the_float32_uninterrupted_and_resumed_runs_match_exactly(proofs):
    proof = proofs["float32"]
    assert proof["dtype"] == "float32"
    assert failed_checks(proof) == []
    assert proof["whole_index_sequence_matches"] is True
    assert proof["loss_sequence_matches"] is True
    assert proof["parameters_match"] is True
    assert proof["moments_match"] is True
    assert proof["prediction_indices_match"] is True
    assert proof["whole_evaluation_record_matches"] is True


def test_no_tolerance_is_used_anywhere_in_the_proof():
    """The standard every exact-resume proof from Phase C onward has met.
    Asserted over the AST, so the prose sentence promising it does not
    satisfy the check."""
    for relative in (EXAMPLE_RELATIVE,
                     "tests/test_native_integer_indexing_example.py"):
        names = code_identifiers(relative)
        for forbidden in ("allclose", "isclose", "approx", "almost_equal",
                          "assert_allclose", "rtol", "atol", "round",
                          "around", "rint"):
            assert forbidden not in names, (relative, forbidden)


def test_the_tolerance_scanner_can_actually_fail():
    """Negative control: a planted tolerance must be detected."""
    tree = ast.parse("import numpy as np\n"
                     "def check(a, b):\n"
                     "    return np.allclose(a, b, atol=1e-6)\n")
    named = {node.attr for node in ast.walk(tree)
             if isinstance(node, ast.Attribute)}
    named |= {node.arg for node in ast.walk(tree)
              if isinstance(node, ast.keyword)}
    assert "allclose" in named and "atol" in named


def test_no_index_is_ever_read_at_a_floating_width():
    """Indices are compared as integers, never converted. ``index_values``
    refuses a floating tensor outright, and the example's own evaluation
    record carries the prediction list as built-in ``int``s."""
    tensor = NativeTensor.from_array([[1.0, 2.0]], dtype="float64")
    try:
        with pytest.raises(TypeError):
            index_values(tensor)
    finally:
        tensor.close()
    indices = NativeTensor.from_int64_array(np.asarray([2, 0, 1],
                                                       dtype=np.int64))
    try:
        values = index_values(indices)
        assert values == [2, 0, 1]
        assert all(type(value) is int for value in values)
    finally:
        indices.close()


# ===========================================================================
# 12. The negative controls
# ===========================================================================

def test_omitting_the_loader_restoration_makes_the_run_diverge(proofs):
    """With ``loader.load_state_dict`` left out, the next batch, the whole
    remaining sequence, the losses, the parameters, **and the integer
    evaluations** must all differ — otherwise the positive proof would be
    passing without the loader restoration doing anything."""
    for dtype in RUN_DTYPES:
        proof = proofs[dtype]
        assert proof["omitted_next_batch_differs"] is True
        assert proof["omitted_indices_differ"] is True
        assert proof["omitted_losses_differ"] is True
        assert proof["omitted_parameters_differ"] is True
        assert proof["omitted_evaluations_differ"] is True


def test_the_omitted_control_leg_cleans_up_completely(live_storages):
    gc.collect()
    baseline = len(live_storages)
    run_omitted_loader_control("float64")
    gc.collect()
    assert len(live_storages) == baseline


def test_the_bit_helper_separates_signed_zeros():
    positive = np.array([0.0], dtype=np.float64)
    negative = np.array([-0.0], dtype=np.float64)
    assert positive[0] == negative[0]                    # equal as floats...
    assert bits(positive, "float64") != bits(negative, "float64")
    narrow_positive = np.array([0.0], dtype=np.float32)
    narrow_negative = np.array([-0.0], dtype=np.float32)
    assert bits(narrow_positive, "float32") != bits(narrow_negative,
                                                    "float32")


def test_the_bit_helper_separates_adjacent_values_at_both_widths():
    wide = np.array([1.0], dtype=np.float64)
    wide_next = np.nextafter(wide, np.float64(2.0))
    assert wide_next[0] != wide[0]
    assert bits(wide, "float64") != bits(wide_next, "float64")

    narrow = np.array([1.0], dtype=np.float32)
    narrow_next = np.nextafter(narrow, np.float32(2.0))
    assert narrow_next[0] != narrow[0]
    assert bits(narrow, "float32") != bits(narrow_next, "float32")
    assert (bits(narrow_next, "float32")[0] - bits(narrow, "float32")[0]) == 1


def test_the_bit_helper_refuses_a_wrong_width_array_and_converts_nothing():
    narrow = np.array([1.5], dtype=np.float32)
    wide = np.array([1.5], dtype=np.float64)
    with pytest.raises(TypeError):
        bits(narrow, "float64")
    with pytest.raises(TypeError):
        bits(wide, "float32")
    values = np.array([1.5, -2.25, 0.0], dtype=np.float64)
    assert bits(values, "float64") == values.view(np.uint64).tolist()


def test_the_bit_reconstruction_helper_round_trips():
    """The test's own inverse really is one — otherwise the column and
    diagonal recomputations above would be checking nothing."""
    for dtype in RUN_DTYPES:
        values = np.asarray([[1.5, -2.25, 0.0], [3.75, -0.0, 8.0]],
                            dtype=_HOST_VIEW[dtype])
        rebuilt = from_bits(bits(values, dtype), dtype, (2, 3))
        assert bits(rebuilt, dtype) == bits(values, dtype)
        assert rebuilt.shape == (2, 3)
        # ...and it separates the signed zeros it round-trips.
        assert bits(rebuilt, dtype)[4] == bits(values, dtype)[4]
        assert bits(rebuilt, dtype)[4] != bits(
            np.asarray([0.0], dtype=_HOST_VIEW[dtype]), dtype)[0]


def test_training_state_actually_changes(proofs):
    for dtype in RUN_DTYPES:
        training = proofs[dtype]["training"]
        assert training["parameters_moved"] is True
        assert training["moments_became_nonzero"] is True
        assert training["optimizer_state_was_empty_at_the_start"] is True
        assert training["step_counters_advanced"] is True
        assert training["loss_sequence_varies"] is True
        assert training["logits_changed_over_training"] is True
        assert training["optimizer_state_nonempty"] is True
        assert training["gradients_cleared"] is True


def test_the_schedule_controls_are_directly_asserted(proofs):
    for dtype in RUN_DTYPES:
        schedule = proofs[dtype]["schedule"]
        assert schedule["batches_per_epoch"] == BATCHES_PER_EPOCH
        assert schedule["split_position"] == EXPECTED_SPLIT_POSITION
        assert schedule["batches_left_in_active_epoch"] == 3
        assert schedule["epoch_boundaries_crossed"] == 2
        assert schedule["exercised_epochs"] == EXERCISED_EPOCHS == 3
        assert schedule["split_is_mid_epoch"] is True
        assert schedule["order_is_not_identity"] is True
        assert schedule["epochs_have_distinct_orders"] is True


def test_the_indexing_controls_are_directly_asserted(proofs):
    for dtype in RUN_DTYPES:
        indexing = proofs[dtype]["indexing"]
        assert indexing["evaluated_steps"] == list(EVAL_STEPS)
        assert indexing["evaluated_before_split"] == [
            step for step in EVAL_STEPS if step < SPLIT_STEP]
        assert indexing["evaluated_after_split"] == [
            step for step in EVAL_STEPS if step >= SPLIT_STEP]
        assert indexing["evaluations_on_both_sides"] is True
        assert indexing["duplicates_occurred"] is True
        assert indexing["duplicates_guaranteed_by_pigeonhole"] is True


# ===========================================================================
# 13. The two dtypes
# ===========================================================================

def test_each_dtype_is_proved_completely_and_only_against_itself(proofs):
    assert RUN_DTYPES == ("float64", "float32")
    assert cpp.SUPPORTED_DTYPES == RUN_DTYPES
    for dtype in RUN_DTYPES:
        proof = proofs[dtype]
        assert proof["dtype"] == dtype
        assert proof["all_state_at_run_dtype"] is True
        assert proof["all_state_on_cpu"] is True
        assert failed_checks(proof) == []


def test_only_the_dtype_independent_facts_are_required_across_dtypes(proofs):
    facts = cross_dtype_facts(proofs)
    assert failed_cross_dtype_checks(facts) == []
    for check in REQUIRED_CROSS_DTYPE:
        assert facts[check] is True, check
    assert facts["dtypes"] == list(RUN_DTYPES)
    # The prediction agreement is an observation, never a gate.
    assert "prediction_indices_agree" not in REQUIRED_CROSS_DTYPE
    assert isinstance(facts["prediction_indices_agree"], bool)


def test_the_batch_index_sequence_is_identical_across_dtypes(proofs):
    """The one numeric-looking cross-dtype equality this design really has:
    a permutation is a pure function of ``(seed, epoch, samples)`` and
    carries no dtype at all."""
    wide, narrow = (proofs[dtype] for dtype in RUN_DTYPES)
    assert wide["index_sequence"] == narrow["index_sequence"]
    assert tuple(wide["index_sequence"]) == EXPECTED_INDEX_SEQUENCE
    assert wide["epoch_permutations"] == narrow["epoch_permutations"]
    assert wide["position_sequence"] == narrow["position_sequence"]
    assert (wide["next_batch_at_interruption"]
            == narrow["next_batch_at_interruption"])
    assert wide["final_loader_position"] == narrow["final_loader_position"]
    assert wide["selection_shapes"] == narrow["selection_shapes"]


def test_no_floating_value_is_compared_across_dtypes(proofs):
    """The complement, asserted rather than assumed: the two dtypes really
    do produce **different** bits for the same task, so a cross-dtype
    numeric equality would be false — and nothing here asserts one."""
    wide, narrow = (proofs[dtype] for dtype in RUN_DTYPES)
    assert wide["uninterrupted_losses"] != narrow["uninterrupted_losses"]
    wide_selected = [record["selected_bits"]
                     for _, record in sorted(wide["evaluations"].items())]
    narrow_selected = [record["selected_bits"]
                       for _, record in sorted(narrow["evaluations"].items())]
    assert wide_selected != narrow_selected
    facts = cross_dtype_facts(proofs)
    for numeric in ("losses", "logits", "parameters", "moments", "loss_bits",
                    "logit_bits", "selected_bits", "diagonal"):
        assert not any(numeric in key for key in facts), numeric


# ===========================================================================
# 14. Ownership, cleanup, and the live-storage baseline
# ===========================================================================

def test_the_complete_proof_returns_storage_to_its_exact_baseline(
        live_storages):
    """Both dtypes, all three legs each. Explicit ``close()`` is the release
    mechanism; the collection below only settles the documented reference
    cycle the Python-managed autograd graph creates through its backward
    closures."""
    gc.collect()
    baseline = len(live_storages)
    for dtype in RUN_DTYPES:
        run_dtype_proof(dtype)
    gc.collect()
    assert len(live_storages) == baseline


def test_the_integer_evaluation_closes_every_temporary(live_storages):
    """The three objects ``evaluate_indexing`` creates — the ``argmax``
    result, the detached source, and the ``index_select`` result — are all
    released by the helper that created them."""
    dataset = build_dataset("float64")
    loader = model = None
    try:
        loader, _sampler = build_loader(dataset)
        model = build_model("float64")
        iterator = iter(loader)
        try:
            features, _targets = next(iterator)
        finally:
            iterator.close()
        logits = model(features)
        try:
            gc.collect()
            baseline = len(live_storages)
            for _ in range(3):
                evaluate_indexing(logits, "float64")
            gc.collect()
            assert len(live_storages) == baseline
        finally:
            logits.close()
            features.close()
    finally:
        if loader is not None:
            loader.close()
        dataset.close()
        if model is not None:
            for parameter in model.parameters():
                parameter.close()


def test_the_storage_tracker_can_actually_fail(live_storages):
    """Non-vacuity for every lifecycle assertion above: a deliberately
    retained tensor must be **seen**, and closing it must clear it. A
    tracker that had silently stopped recording would make every baseline
    equality pass for the wrong reason."""
    gc.collect()
    baseline = len(live_storages)
    leaked = NativeTensor.from_int64_array(np.asarray([1, 0], dtype=np.int64))
    try:
        gc.collect()
        assert len(live_storages) > baseline, "the tracker saw nothing"
    finally:
        leaked.close()
    gc.collect()
    assert len(live_storages) == baseline


def test_repeated_runs_return_to_baseline_each_time(live_storages):
    gc.collect()
    baseline = len(live_storages)
    for _ in range(2):
        run_uninterrupted("float32")
        gc.collect()
        assert len(live_storages) == baseline


def test_the_proof_helpers_return_no_live_object(proofs):
    for dtype in RUN_DTYPES:
        _assert_plain_python(proofs[dtype], f"proof[{dtype!r}]")


def test_the_example_closes_what_it_owns_and_never_a_delivered_batch():
    """Structural: the cleanup helper closes the loader, the dataset, the
    optimizer, and every parameter — and nothing in the example closes a
    *target* array, which is ordinary host memory."""
    source = EXAMPLE.read_text(encoding="utf-8")
    tree = ast.parse(source)
    closer = next(node for node in ast.walk(tree)
                  if isinstance(node, ast.FunctionDef)
                  and node.name == "_close_run")
    closed = {ast.unparse(node.func).rsplit(".", 1)[0]
              for node in ast.walk(closer)
              if isinstance(node, ast.Call)
              and ast.unparse(node.func).endswith(".close")}
    assert {"loader", "dataset", "optimizer", "parameter"} <= closed
    assert ".close()" in source
    assert not re.search(r"\btargets\.close\(\)", source)
    # The integer temporaries are closed in the evaluation helper itself.
    evaluator = next(node for node in ast.walk(tree)
                     if isinstance(node, ast.FunctionDef)
                     and node.name == "evaluate_indexing")
    evaluated = {ast.unparse(node.func).rsplit(".", 1)[0]
                 for node in ast.walk(evaluator)
                 if isinstance(node, ast.Call)
                 and ast.unparse(node.func).endswith(".close")}
    assert {"predictions", "detached", "selected"} <= evaluated, evaluated


# ===========================================================================
# 15. Inventories, and the K6 scope boundary
# ===========================================================================

def test_the_example_inventory_grew_by_exactly_one():
    examples = sorted(path.name for path in (REPO_ROOT / "examples").glob("*.py")
                      if path.name != "__init__.py")
    assert len(examples) == EXPECTED_EXAMPLE_COUNT, examples
    assert "native_integer_indexing.py" in examples
    # K6 is the phase's only example, and it is named rather than merely
    # counted so a second integer example cannot arrive unnoticed.
    integer = [name for name in examples
               if "integer" in name or "index" in name]
    assert integer == ["native_integer_indexing.py"], integer
    benchmarks = sorted(path.name
                        for path in (REPO_ROOT / "benchmarks").glob("*.py")
                        if path.name != "__init__.py")
    assert len(benchmarks) == EXPECTED_BENCHMARK_COUNT, benchmarks
    assert not [name for name in benchmarks
                if "integer" in name or "index" in name], benchmarks


def test_the_capability_boundary_did_not_move():
    assert cpp.SUPPORTED_DTYPES == ("float64", "float32")
    assert cpp.INDEX_DTYPES == ("int64",)
    assert cpp.SUPPORTED_DEVICES == ("cpu",)
    assert cpp.UNSUPPORTED == ("cuda", "amp")
    assert cpp.RAW_KERNEL_DTYPES == ("float64",)
    assert cpp.normalize_dtype(None) == "float64"
    with pytest.raises(ValueError):
        cpp.normalize_dtype("int64")
    info = cpp.backend_info()
    assert info["dtype"] == "float64"
    assert info["index_dtypes"] == ("int64",)
    assert info["stable_framework_integration"] is False


def test_the_abi_and_ctest_inventories_did_not_move():
    names = set()
    for source in sorted((REPO_ROOT / "cpp" / "src").glob("*.cpp")):
        names.update(re.findall(
            r"TF_EXPORT[^;{]*?\b(tf_[a-z0-9_]+)\s*\(",
            source.read_text(encoding="utf-8"), re.S))
    assert len(names) == EXPECTED_ABI_EXPORTS, sorted(names)
    assert {"tf_core_argmax", "tf_core_index_select"} <= names
    cmake = (REPO_ROOT / "cpp" / "CMakeLists.txt").read_text(encoding="utf-8")
    tests = re.findall(r"add_test\s*\(\s*NAME\s+(\w+)", cmake)
    assert len(tests) == EXPECTED_CTESTS, tests


def test_no_cpp_or_build_surface_mentions_the_new_example():
    """K6 changed no C++, no CMake, and no dependency file."""
    for relative in ("cpp/CMakeLists.txt", "cpp/build.py", "pyproject.toml",
                     ".github/workflows/tests.yml"):
        text = (REPO_ROOT / relative).read_text(encoding="utf-8")
        assert "integer_indexing" not in text, relative
    for path in sorted((REPO_ROOT / "cpp").rglob("*.cpp")):
        assert "integer_indexing" not in path.read_text(encoding="utf-8"), (
            path.name)


def test_the_later_phase_k_milestones_have_not_started():
    """K6 is the integration example. The benchmark (K8) and the closure
    (K9) are unstarted and neither artifact exists — and **this module**
    still contains none of the vocabulary a later milestone owns: it
    injects nothing, times nothing, and makes no phase-wide claim.

    K7's adversarial matrix landed after K6 and is therefore asserted
    **present** rather than absent. The entry moved instead of being
    deleted, so this stays a claim about the ladder: what K6 did not ship
    is still what a later milestone owns, and this module still does not
    do any of it."""
    assert (REPO_ROOT / "tests"
            / "test_native_integer_hardening.py").is_file()      # K7
    for absent in ("test_native_integer_benchmark.py",           # K8
                   "test_native_phase_k_closure.py"):            # K9
        assert not (REPO_ROOT / "tests" / absent).exists(), absent
    assert not (REPO_ROOT / "benchmarks"
                / "benchmark_native_integer.py").exists()
    names = code_identifiers("tests/test_native_integer_indexing_example.py")
    for hardening in ("tf_test_arm_alloc_failure", "fault_injection_available",
                      "_deliver_batch", "_claim_batch", "_rollback_pending",
                      "threading", "Thread", "asyncio", "Queue", "Pool"):
        assert hardening not in names, hardening


def test_no_absent_indexing_operation_appeared():
    """The operations Phase K deliberately does not have, asserted on the
    live classes rather than in prose."""
    for owner in (NativeTensor, cpp.NativeTensorCore):
        for absent in ("gather", "scatter", "scatter_add", "embedding",
                       "argmin", "max", "max_with_indices", "take",
                       "index_add", "index_put", "masked_select",
                       "index_select_backward", "__getitem__"):
            assert not hasattr(owner, absent), (owner.__name__, absent)
    for present in ("argmax", "index_select"):
        assert hasattr(NativeTensor, present), present
        assert hasattr(cpp.NativeTensorCore, present), present
    assert "argmax" in cpp.TENSOR_CORE_OPS
    assert "index_select" in cpp.TENSOR_CORE_OPS
    assert "argmax" not in cpp.AUTOGRAD_OPS
    assert "index_select" not in cpp.AUTOGRAD_OPS


def test_the_model_class_stays_an_example_implementation_detail():
    assert NativeIndexingClassifier.__module__ == (
        "examples.native_integer_indexing")
    assert issubclass(NativeIndexingClassifier, NativeModule)
    for path in sorted((REPO_ROOT / "src").rglob("*.py")):
        text = path.read_text(encoding="utf-8")
        assert "NativeIndexingClassifier" not in text, path.name


def test_the_reporting_helpers_are_importable_and_pure(uninterrupted):
    """``model_facts`` / ``optimizer_facts`` close every caller-owned
    snapshot they take, so calling them repeatedly cannot leak."""
    assert isinstance(model_facts, type(optimizer_facts))
    assert set(uninterrupted["parameters"]) == set(
        uninterrupted["initial"]["parameters"])
    assert uninterrupted["optimizer"]["format_version"] == 1
