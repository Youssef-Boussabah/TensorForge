"""Phase K, milestone K3 — native ``argmax``.

K3 is the milestone at which Phase K ships its first **operation** and its
first **C ABI symbol**. This module owns the split
``docs/native_integer_tensors_design.md`` §30.1 assigns it: everything about
``NativeTensor.argmax`` / ``NativeTensorCore.argmax`` and the
``tf_core_argmax`` export behind them.

Five claims, and they only mean something together:

1. **The value rule is exact, and it is TensorForge's own.** §17.5 is an
   algorithm, not an adjective: equal maxima give the **lowest** index,
   ``+0.0`` and ``-0.0`` tie, an all-``-inf`` run gives 0, the **first** NaN
   wins against every finite value and either infinity, and a length-1 run
   gives 0. Every expectation below is written from that contract rather
   than delegated to ``numpy.argmax``, and every case is driven at
   ``float32`` and ``float64`` **separately**.
2. **The result is an `int64` index tensor, and nothing else changed.** It
   owns fresh contiguous storage at offset 0, survives its input's
   ``close()``, aliases nothing, and is the caller's to close — while
   ``SUPPORTED_DTYPES``, every version, ``__all__``, and ``AUTOGRAD_OPS``
   are exactly what K2 left.
3. **It is never differentiable.** Not at the Core, not at the tensor
   layer, and **not even when the input requires gradients** — which is the
   one place ``argmax`` differs from every other operation on a
   gradient-tracking tensor, and is correct, because the derivative of an
   index with respect to a value does not exist.
4. **The C ABI is a second authority.** ``tf_core_argmax`` is driven
   directly through ``ctypes``, with the Python layer bypassed entirely, to
   prove it rejects a non-floating source and a non-``int64`` destination on
   its own — and, the claim no structural check can make, that a **floating
   source with an int64 destination succeeds**, which a
   ``require_floating`` or a ``require_matching_dtype`` on that destination
   would have rejected.
5. **Every failure position is distinct, and each leaves the world
   unchanged.** The Policy-B temporary, the destination allocation, the
   native call, and the wrapper publication are four injections, not one,
   each with a control proving it fired and each followed by a live-storage
   baseline check.

Discipline this module inherits (integer design §29.6, §30.2):

* **Exact equality only** for integers — Python ``int`` comparison or
  ``numpy.int64`` array equality. **No tolerance is used for an integer
  anywhere**, and there is nothing else here to compare.
* **Every rejection is followed by a complete before/after fingerprint of
  the observable world**, and the fingerprint has its own non-vacuity
  control proving each component can notice the change it exists for.
* **Every injected failure position is a distinct injection**, each with a
  control proving it can fire, and each followed by a live-storage baseline
  check. The tracker installs itself **outside** ``monkeypatch``, so a
  mid-test ``undo()`` cannot silently disarm it.
* **Abandonment is proved by explicit ``close()``.** No assertion here
  depends on garbage-collection timing; the injection tests retain their
  objects strongly and assert they were closed while still referenced.
* **Source scans read code, not prose** — docstrings and string literals
  are stripped through the AST first. Every scanner has a negative control.
* No test starts a thread, touches the network, needs a Git ancestor, or
  depends on a total suite count.
"""
import ast
import contextlib
import ctypes
import math
import re
from pathlib import Path

import numpy as np
import pytest

from tensorforge.backends import cpp
from tensorforge.experimental import (
    NativeGenerator,
    NativeModule,
    NativeParameter,
    NativeSGD,
    NativeTensor,
)

REPO_ROOT = Path(__file__).resolve().parent.parent

needs_native = pytest.mark.skipif(
    not cpp.is_available(), reason="experimental C++ backend not built"
)

# Written here independently of the modules under test, so a silent change
# fails rather than propagating.
FLOATING_DTYPES = ("float64", "float32")
INDEX_DTYPE = "int64"
K3_EXPORT = "tf_core_argmax"
# The **live** inventories, not K3's: these are repository-wide totals that
# a later milestone legitimately moves, and Phase K milestone K4 moved each
# of them by exactly one when it shipped ``tf_core_index_select``. What is
# K3's and stays K3's is that ``tf_core_argmax`` is *in* the inventory, and
# that is asserted separately below.
LIVE_EXPORT_COUNT = 56
LIVE_CTEST_COUNT = 27
LIVE_CHECKED_KERNELS = 38
EXPERIMENTAL_EXPORTS = 25

# The operations K3 did **not** ship, at any layer. ``max`` is permanent
# (§17.10); the rest are no milestone's. ``index_select`` left this tuple
# at **K4**, which shipped it — an entry moves between the present and
# absent lists when its milestone lands, and is never loosened away.
ABSENT_OPERATIONS = ("max", "amax", "max_with_indices", "argmin",
                     "gather", "scatter", "take", "topk",
                     "sort", "argsort", "nonzero", "where", "bincount",
                     "cumsum")


# ---------------------------------------------------------------------------
# Live-storage accounting.
#
# Installed **outside** ``monkeypatch`` on purpose (design §30.2): a mid-test
# ``undo()`` must not be able to disarm the tracker that proves a failure
# leaked nothing.
# ---------------------------------------------------------------------------

@contextlib.contextmanager
def live_storage_baseline():
    """Assert that native live storage returns exactly to baseline.

    Counts every ``NativeStorage`` this block constructs and every one it
    closes, by wrapping the **one** constructor and the **one** release
    point every allocation runs through."""
    live = set()
    original_init = cpp.NativeStorage.__init__
    original_close = cpp.NativeStorage.close

    def counting_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        live.add(id(self))

    def counting_close(self):
        original_close(self)
        live.discard(id(self))

    cpp.NativeStorage.__init__ = counting_init
    cpp.NativeStorage.close = counting_close
    try:
        yield live
        assert not live, f"{len(live)} native storages were never closed"
    finally:
        cpp.NativeStorage.__init__ = original_init
        cpp.NativeStorage.close = original_close


# ---------------------------------------------------------------------------
# The observable-world fingerprint.
# ---------------------------------------------------------------------------

class World:
    """A snapshot of everything a rejected or failed ``argmax`` must leave
    alone. Deliberately proportional to this operation's boundary: the
    operand it was handed, a parameter and its graph state, the registries
    it must not move, and both global RNGs."""

    def __init__(self, operand, parameter, module, optimizer, generator):
        self.operand = operand
        self.parameter = parameter
        self.module = module
        self.optimizer = optimizer
        self.generator = generator

    def fingerprint(self):
        operand, parameter = self.operand, self.parameter
        # A closed operand has no readable metadata, so the closed flag is
        # read first and the rest is skipped: a fingerprint that *raises*
        # after a close would be an instrument that cannot report the very
        # change it exists to notice.
        closed = operand.closed
        return (
            # the operand: identity, bits, layout, graph state
            id(operand),
            closed,
            None if closed else operand.to_numpy().tobytes(),
            None if closed else (operand.shape, operand.strides,
                                 operand.dtype, operand.device,
                                 operand.contiguous, operand.requires_grad,
                                 operand.is_leaf),
            None if closed or operand.grad is None
            else operand.grad.to_numpy().tobytes(),
            # a parameter: identity, value, version, gradient
            id(parameter),
            parameter.to_numpy().tobytes(),
            parameter.version,
            None if parameter.grad is None
            else parameter.grad.to_numpy().tobytes(),
            tuple(sorted(name for name, _ in self.module.named_parameters())),
            tuple(id(p) for p in self.optimizer.parameters()),
            # a registered generator, whose call counter nothing here spends
            self.generator.state(),
            # every registry a rejection must not move
            cpp.SUPPORTED_DTYPES, cpp.INDEX_DTYPES, cpp.SUPPORTED_DEVICES,
            cpp.UNSUPPORTED, cpp.RAW_KERNEL_DTYPES,
            tuple(sorted(cpp._DTYPE_CODES.items())),
            cpp.TENSOR_CORE_OPS, cpp.AUTOGRAD_OPS, cpp._CHECKED_KERNELS,
            # both global RNGs, which nothing here may consume
            np.random.get_state()[1][0],
        )


def _build_world(dtype="float64"):
    operand = NativeTensor.from_array(
        np.array([[1.0, 7.0, 3.0], [9.0, 9.0, 2.0]]), dtype=dtype)
    parameter = NativeParameter(np.array([[1.5, -2.5], [0.25, 4.0]]))
    module = NativeModule()
    module.weight = parameter
    generator = NativeGenerator(seed=11)
    module.register_generator("rng", generator)
    optimizer = NativeSGD([parameter], lr=0.1)
    return World(operand, parameter, module, optimizer, generator)


@contextlib.contextmanager
def unchanged_world(dtype="float64"):
    """Build the world, hand it over, and assert its fingerprint is
    byte-identical afterwards."""
    world = _build_world(dtype)
    before = world.fingerprint()
    try:
        yield world
        assert world.fingerprint() == before, "a rejection changed the world"
    finally:
        world.operand.close()
        world.parameter.close()


# ---------------------------------------------------------------------------
# Small helpers.
# ---------------------------------------------------------------------------

def _code_only(source):
    """Source with every docstring and string literal removed, through the
    AST rather than by regex — so a comment or docstring naming an absent
    operation cannot satisfy or break an absence scan."""
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
            pieces.append(node.arg)
        elif isinstance(node, ast.arg):
            pieces.append(node.arg)
    return pieces


def _source_exports():
    names = set()
    for path in sorted((REPO_ROOT / "cpp" / "src").glob("*.cpp")):
        names.update(re.findall(r"TF_EXPORT[^;{]*?\b(tf_[a-z0-9_]+)\s*\(",
                                path.read_text(encoding="utf-8"), re.S))
    return names


def _argmax_indices(tensor, **kwargs):
    """``tensor.argmax(**kwargs)`` as (shape, nested Python ints), with the
    result closed. Every assertion below compares exact integers."""
    result = tensor.argmax(**kwargs)
    try:
        assert result.dtype == INDEX_DTYPE
        return result.shape, result.tolist()
    finally:
        result.close()


def _from_values(values, dtype):
    return NativeTensor.from_array(np.asarray(values, dtype=np.float64),
                                   dtype=dtype)


NAN = float("nan")
INF = float("inf")


# ===========================================================================
# 0. The instruments can fail
# ===========================================================================

@needs_native
def test_the_fingerprint_can_notice_each_change_it_exists_for():
    """Non-vacuity for every ``unchanged_world`` assertion below."""
    # 1. a changed parameter value (and, with it, its version)
    world = _build_world()
    before = world.fingerprint()
    try:
        replacement = NativeTensor.from_array(np.zeros((2, 2)))
        try:
            world.parameter.copy_value_(replacement)
        finally:
            replacement.close()
        assert world.fingerprint() != before
    finally:
        world.operand.close()
        world.parameter.close()

    # 2. a gradient appearing on the parameter
    world = _build_world()
    before = world.fingerprint()
    try:
        grad = NativeTensor.from_array(np.ones((2, 2)))
        world.parameter._accumulate_grad(grad)
        assert world.fingerprint() != before
    finally:
        world.operand.close()
        world.parameter.close()

    # 3. the operand closing
    world = _build_world()
    before = world.fingerprint()
    try:
        world.operand.close()
        assert world.fingerprint() != before
    finally:
        world.parameter.close()

    # 4. the generator's committed-call counter advancing
    world = _build_world()
    before = world.fingerprint()
    try:
        source = NativeTensor.from_array(np.ones((4,)))
        try:
            dropped = source.dropout(0.5, generator=world.generator)
            dropped.close()
        finally:
            source.close()
        assert world.fingerprint() != before
    finally:
        world.operand.close()
        world.parameter.close()

    # 5. a registry moving (simulated on a copy, never on the real module)
    world = _build_world()
    before = world.fingerprint()
    try:
        original = cpp.TENSOR_CORE_OPS
        cpp.TENSOR_CORE_OPS = original + ("probe",)
        try:
            assert world.fingerprint() != before
        finally:
            cpp.TENSOR_CORE_OPS = original
        assert world.fingerprint() == before
    finally:
        world.operand.close()
        world.parameter.close()

    # 6. the global NumPy RNG being consumed
    world = _build_world()
    before = world.fingerprint()
    try:
        np.random.seed(1234)
        assert world.fingerprint() != before
    finally:
        world.operand.close()
        world.parameter.close()


@needs_native
def test_the_live_storage_tracker_can_actually_fail():
    """Non-vacuity for every baseline assertion: an unclosed tensor is
    reported, and a closed one is not."""
    with pytest.raises(AssertionError, match="never closed"):
        with live_storage_baseline():
            leaked = NativeTensor.from_array(np.array([1.0]))
    leaked.close()
    with live_storage_baseline():
        kept = NativeTensor.from_array(np.array([1.0]))
        kept.close()


def test_the_code_only_reader_ignores_prose():
    """Negative control for the absence scanner, on a temporary string."""
    names = _code_only('"""a docstring naming max and argmin."""\n'
                       'def f(x):\n    return g(x, keepdims=True)\n')
    assert "argmin" not in names and "max" not in names
    assert "keepdims" in names and "g" in names


def test_the_export_scanner_can_actually_fail():
    """Negative control, on a temporary string: an export really is found,
    so the inventory below is a measurement rather than an artifact."""
    found = re.findall(r"TF_EXPORT[^;{]*?\b(tf_[a-z0-9_]+)\s*\(",
                       "TF_EXPORT void tf_core_probe(const void* a) { x(a); }",
                       re.S)
    assert found == ["tf_core_probe"]
    assert re.findall(r"TF_EXPORT[^;{]*?\b(tf_[a-z0-9_]+)\s*\(",
                      "void tf_core_probe(void);", re.S) == []


# ===========================================================================
# 1. The public API and the inventories
# ===========================================================================

def test_both_layers_gained_argmax_and_only_argmax():
    assert hasattr(NativeTensor, "argmax")
    assert hasattr(cpp.NativeTensorCore, "argmax")
    assert not hasattr(cpp.NativeStorage, "argmax")
    for owner in (NativeTensor, cpp.NativeTensorCore, cpp.NativeStorage):
        for absent in ABSENT_OPERATIONS:
            assert not hasattr(owner, absent), (owner.__name__, absent)
    # K4's ``index_select`` is the one name that left ABSENT_OPERATIONS, and
    # it landed on exactly the two layers ``argmax`` did — asserted present
    # here rather than merely no longer banned, so the move is proved to be
    # a move (§37.2). Its own contract lives in
    # tests/test_native_index_select.py.
    assert hasattr(NativeTensor, "index_select")
    assert hasattr(cpp.NativeTensorCore, "index_select")
    assert not hasattr(cpp.NativeStorage, "index_select")


def test_the_signature_matches_the_repositorys_reduction_spelling():
    """`axis`/`keepdims`, in that order, with this repository's defaults —
    the same spelling ``sum`` and ``mean`` carry, not another framework's."""
    import inspect

    for owner in (NativeTensor, cpp.NativeTensorCore):
        signature = inspect.signature(owner.argmax)
        assert list(signature.parameters) == ["self", "axis", "keepdims"]
        assert signature.parameters["axis"].default is None
        assert signature.parameters["keepdims"].default is False
        # ...and it is exactly ``sum``'s and ``mean``'s.
        for sibling in ("sum", "mean"):
            assert list(inspect.signature(
                getattr(owner, sibling)).parameters) == \
                list(signature.parameters)


def test_argmax_joined_exactly_one_inventory():
    assert cpp.TENSOR_CORE_OPS.count("argmax") == 1
    assert "argmax" not in cpp.AUTOGRAD_OPS
    assert "argmax" not in cpp.TENSOR_CORE_KERNELS
    assert "argmax" not in cpp.RAW_KERNELS
    assert "argmax" not in cpp.NATIVE_METRICS
    assert "argmax" not in cpp.NATIVE_MODULES
    assert "argmax" not in cpp.NATIVE_LOSSES
    assert "argmax" not in cpp.STATE_SUPPORT
    # The deliberately frozen historical registry is exactly what it was.
    assert cpp.TENSOR_CORE_KERNELS == (
        "relu", "add", "subtract", "multiply", "matmul")
    assert cpp.backend_info()["tensor_core_ops"] == cpp.TENSOR_CORE_OPS
    assert cpp.backend_info()["autograd_ops"] == cpp.AUTOGRAD_OPS


def test_the_source_export_inventory_carries_the_argmax_symbol():
    exports = _source_exports()
    assert len(exports) == LIVE_EXPORT_COUNT, sorted(exports)
    assert K3_EXPORT in exports
    # K4's symbol is present beside it, and 56 is Phase K's committed
    # maximum — so a third indexing export would fail here.
    assert "tf_core_index_select" in exports
    for absent in ("tf_core_max", "tf_core_argmin",
                   "tf_core_max_with_indices", "tf_core_argmax_backward",
                   "tf_storage_dtype"):
        assert absent not in exports, absent


@needs_native
def test_the_built_library_exports_the_same_inventory():
    """Source and built library must agree, which is also the stale-artifact
    guard: a library built before K3 exports 54 (before K4, 55) and fails
    here rather than silently satisfying the tests that call the new
    symbols."""
    storage_tests = pytest.importorskip("test_native_storage_allocation")
    _, names = storage_tests.exported_names(cpp._LIBRARY_PATH)
    if names is None:
        pytest.skip("this image format is not parsed here")
    exported = sorted(name for name in names if name.startswith("tf_"))
    assert len(exported) == LIVE_EXPORT_COUNT, exported
    assert set(exported) == _source_exports()
    assert K3_EXPORT in exported


@needs_native
def test_the_export_is_declared_and_carries_the_error_hook():
    library = cpp._require_library()
    function = getattr(library, K3_EXPORT)
    assert function.argtypes == [
        ctypes.c_void_p, ctypes.c_int64, ctypes.c_void_p,
        ctypes.c_int64, ctypes.c_int64, ctypes.c_int64,
    ]
    assert function.restype is None
    assert K3_EXPORT in cpp._CHECKED_KERNELS
    assert cpp._CHECKED_KERNELS.count(K3_EXPORT) == 1
    assert len(cpp._CHECKED_KERNELS) == LIVE_CHECKED_KERNELS
    assert function.errcheck is not None


def test_the_argmax_ctest_target_is_registered_exactly_once():
    cmake = (REPO_ROOT / "cpp" / "CMakeLists.txt").read_text(encoding="utf-8")
    registered = re.findall(r"add_test\s*\(\s*NAME\s+(\w+)", cmake)
    assert len(registered) == LIVE_CTEST_COUNT, registered
    assert len(set(registered)) == len(registered)
    assert registered.count("argmax") == 1
    sources = {path.stem for path in
               (REPO_ROOT / "cpp" / "tests").glob("test_*.cpp")}
    assert sources == {f"test_{name}" for name in registered}
    assert (REPO_ROOT / "cpp" / "tests" / "test_argmax.cpp").is_file()


def test_the_indexing_unit_is_its_own_translation_unit_and_header():
    assert (REPO_ROOT / "cpp" / "src" / "indexing.cpp").is_file()
    assert (REPO_ROOT / "cpp" / "include"
            / "tf_indexing_internal.h").is_file()
    header = (REPO_ROOT / "cpp" / "include" / "tf_indexing_internal.h"
              ).read_text(encoding="utf-8")
    code = re.sub(r"//[^\n]*", " ",
                  re.sub(r"/\*.*?\*/", " ", header, flags=re.S))
    # No public ABI declaration lives in the internal header.
    assert "TF_EXPORT" not in code
    assert K3_EXPORT not in code
    # ...and the templated traversal and the role guard do.
    assert "argmax_contiguous" in code
    assert "require_index" in code


def test_no_public_capability_registry_or_version_moved_at_k3():
    import tensorforge.experimental as experimental
    from tensorforge.experimental import (native_checkpoint,
                                          native_data_loader,
                                          native_optimizer_state,
                                          native_sampler)

    assert cpp.SUPPORTED_DTYPES == ("float64", "float32")
    assert cpp.INDEX_DTYPES == ("int64",)
    assert cpp.SUPPORTED_DEVICES == ("cpu",)
    assert cpp.UNSUPPORTED == ("cuda", "amp")
    assert cpp.RAW_KERNEL_DTYPES == ("float64",)
    assert cpp.normalize_dtype(None) == "float64"
    with pytest.raises(ValueError):
        cpp.normalize_dtype("int64")
    assert len(experimental.__all__) == EXPERIMENTAL_EXPORTS
    assert len(set(experimental.__all__)) == EXPERIMENTAL_EXPORTS
    assert "argmax" not in experimental.__all__
    assert native_checkpoint._FORMAT_VERSION == 3
    assert native_checkpoint._SUPPORTED_FORMAT_VERSIONS == (1, 2, 3)
    assert native_optimizer_state.FORMAT_VERSION == 1
    assert native_data_loader._FORMAT_VERSION == 1
    assert native_sampler._FORMAT_VERSION == 1
    # Phase K, milestone K6 took the example inventory from 16 to 17
    # (examples/native_integer_indexing.py) and **K8** took the benchmark
    # inventory from 9 to 10 (benchmarks/benchmark_native_integer.py). The
    # numbers are updated rather than the assertions relaxed: these still
    # pin exact inventories, K3's own delta to both is still zero, and an
    # unrecorded addition still fails.
    assert len(list((REPO_ROOT / "examples").glob("*.py"))) == 17
    assert len(list((REPO_ROOT / "benchmarks").glob("*.py"))) == 10


def test_native_accuracy_still_takes_the_host_boundary():
    """K3 shipped an `argmax` and deliberately did not rewrite the metric.
    Read from the code, so the docstring that explains the choice cannot
    satisfy the check."""
    from tensorforge.experimental import native_metrics

    source = Path(native_metrics.__file__).read_text(encoding="utf-8")
    body = source.split("def native_accuracy(", 1)[1]
    assert "logits.to_numpy()" in body
    assert "np.argmax(values, axis=1)" in body
    # No native argmax call: the only ``.argmax(`` in the body is NumPy's.
    assert ".argmax(" not in body.replace("np.argmax(", "")
    assert cpp.NATIVE_METRICS == ("native_accuracy",)


# ===========================================================================
# 2. The exact value rule — §17.5, row by row, at each dtype separately
# ===========================================================================

VALUE_ROWS = (
    # (label, run, expected index)
    ("unique maximum", [1.0, 5.0, 3.0, 2.0], 1),
    ("unique maximum, all negative", [-9.0, -4.0, -100.0], 1),
    ("equal maxima at the front", [7.0, 7.0, 2.0], 0),
    ("equal maxima later", [1.0, 7.0, 7.0, 7.0], 1),
    ("every element equal", [4.0, 4.0, 4.0, 4.0], 0),
    ("signed zeros, -0.0 first", [-0.0, 0.0, -1.0], 0),
    ("signed zeros, +0.0 first", [-1.0, 0.0, -0.0], 1),
    ("signed zeros adjacent", [-1.0, -0.0, 0.0], 1),
    ("every element -inf", [-INF, -INF, -INF], 0),
    ("repeated +inf", [3.0, INF, 9.0, INF], 1),
    ("-inf, finite, +inf", [-INF, 0.0, INF], 2),
    ("one NaN in the middle", [1.0, NAN, 3.0], 1),
    ("one NaN at the end", [1.0, 2.0, NAN], 2),
    ("NaN after +inf", [INF, NAN], 1),
    ("NaN after -inf", [-INF, NAN], 1),
    ("NaN at index 0, then +inf", [NAN, INF, 1000.0], 0),
    ("NaN at index 0, then finite", [NAN, 1.0, 2.0, 3.0], 0),
    ("several NaNs", [1.0, NAN, NAN, 2.0], 1),
    ("several NaNs from index 0", [NAN, NAN, NAN], 0),
    ("NaN before and after a maximum", [0.0, NAN, 9.0, NAN], 1),
    ("length 1, finite", [42.0], 0),
    ("length 1, NaN", [NAN], 0),
    ("length 1, -inf", [-INF], 0),
)


@needs_native
@pytest.mark.parametrize("dtype", FLOATING_DTYPES)
@pytest.mark.parametrize("label,run,expected",
                         VALUE_ROWS, ids=[row[0] for row in VALUE_ROWS])
def test_the_value_rule_row_by_row(dtype, label, run, expected):
    """Every §17.5 row, at each dtype **separately**, as a full reduction
    and again as a one-run axis reduction — the two decompositions the one
    export covers."""
    tensor = _from_values(run, dtype)
    try:
        assert _argmax_indices(tensor) == ((), expected), label
        assert _argmax_indices(tensor, axis=0) == ((), expected), label
        assert _argmax_indices(tensor, axis=-1) == ((), expected), label
        assert _argmax_indices(tensor, axis=0, keepdims=True) == \
            ((1,), [expected]), label
    finally:
        tensor.close()


@needs_native
@pytest.mark.parametrize("dtype", FLOATING_DTYPES)
def test_a_nan_in_one_run_never_reaches_another(dtype):
    """The per-axis case answers each output position independently."""
    values = [[1.0, NAN, 3.0],      # NaN at 1
              [5.0, 4.0, 6.0],      # plain maximum at 2
              [NAN, NAN, 0.0],      # first NaN at 0
              [-INF, -INF, -INF]]   # everything ties, so 0
    tensor = _from_values(values, dtype)
    try:
        assert _argmax_indices(tensor, axis=1) == ((4,), [1, 2, 0, 0])
        # Down the columns: (1,5,NaN,-inf) -> 2; (NaN,4,NaN,-inf) -> 0;
        # (3,6,0,-inf) -> 1.
        assert _argmax_indices(tensor, axis=0) == ((3,), [2, 0, 1])
        # The flat reduction sees the first NaN in row-major order, at 1.
        assert _argmax_indices(tensor) == ((), 1)
    finally:
        tensor.close()


@needs_native
@pytest.mark.parametrize("dtype", FLOATING_DTYPES)
def test_the_full_reduction_index_is_the_flat_row_major_position(dtype):
    """`axis=None` returns the logical flat index — the same order
    ``to_numpy()`` produces — which is checked by using the returned index
    to address the materialized array."""
    values = np.arange(24.0).reshape(2, 3, 4)
    values[1, 2, 1] = 1000.0          # the unique maximum, at flat 21
    tensor = _from_values(values, dtype)
    try:
        shape, flat = _argmax_indices(tensor)
        assert shape == ()
        assert flat == 21
        assert tensor.to_numpy().reshape(-1)[flat] == \
            tensor.to_numpy().reshape(-1).max()
    finally:
        tensor.close()


# ===========================================================================
# 3. Shapes, dtype, ownership, and independence
# ===========================================================================

SHAPE_ROWS = (
    ((2, 3, 4), None, False, ()),
    ((2, 3, 4), None, True, (1, 1, 1)),
    ((2, 3, 4), 0, False, (3, 4)),
    ((2, 3, 4), 0, True, (1, 3, 4)),
    ((2, 3, 4), 1, False, (2, 4)),
    ((2, 3, 4), 1, True, (2, 1, 4)),
    ((2, 3, 4), 2, False, (2, 3)),
    ((2, 3, 4), -1, False, (2, 3)),
    ((2, 3, 4), -3, True, (1, 3, 4)),
    ((5,), 0, False, ()),
    ((5,), 0, True, (1,)),
    ((2, 3), None, True, (1, 1)),
)


@needs_native
@pytest.mark.parametrize("dtype", FLOATING_DTYPES)
@pytest.mark.parametrize("shape,axis,keepdims,expected", SHAPE_ROWS)
def test_the_output_shape_follows_the_reduction_authority(
        dtype, shape, axis, keepdims, expected):
    values = np.arange(float(np.prod(shape))).reshape(shape)
    tensor = _from_values(values, dtype)
    try:
        result = tensor.argmax(axis=axis, keepdims=keepdims)
        try:
            assert result.shape == expected
            # ...and it is exactly what the shared authority says.
            assert result.shape == cpp.reduce_shape(shape, axis, keepdims)
            assert result.dtype == INDEX_DTYPE
        finally:
            result.close()
    finally:
        tensor.close()


@needs_native
@pytest.mark.parametrize("dtype", FLOATING_DTYPES)
def test_a_rank_zero_input_reduces_to_itself(dtype):
    tensor = NativeTensor.zeros((), dtype=dtype)
    try:
        assert tensor.ndim == 0 and tensor.numel == 1
        assert _argmax_indices(tensor) == ((), 0)
        assert _argmax_indices(tensor, keepdims=True) == ((), 0)
        # Any explicit axis is out of range on a scalar.
        for axis in (0, -1, 1):
            with pytest.raises(ValueError):
                tensor.argmax(axis=axis)
    finally:
        tensor.close()


@needs_native
@pytest.mark.parametrize("dtype", FLOATING_DTYPES)
def test_the_result_owns_fresh_contiguous_storage_and_survives_its_input(
        dtype):
    tensor = _from_values([[1.0, 7.0, 3.0], [9.0, 2.0, 2.0]], dtype)
    result = tensor.argmax(axis=1)
    try:
        assert result.dtype == INDEX_DTYPE
        assert result.device == "cpu"
        assert result.owns_core and result.contiguous
        assert result._require_open().offset == 0
        assert result._require_open().storage is not \
            tensor._require_open().storage
        # Closing the input leaves the result intact and readable.
        tensor.close()
        assert result.tolist() == [1, 0]
        assert result.to_numpy().dtype == np.int64
    finally:
        result.close()
        if not tensor.closed:
            tensor.close()


@needs_native
def test_host_inspection_returns_built_in_python_integers():
    tensor = _from_values([[1.0, 7.0], [9.0, 2.0]], "float64")
    try:
        scalar = tensor.argmax()
        try:
            value = scalar.item()
            assert type(value) is int and value == 2
        finally:
            scalar.close()
        nested = tensor.argmax(axis=1, keepdims=True)
        try:
            listed = nested.tolist()
            assert listed == [[1], [0]]
            assert all(type(cell) is int for row in listed for cell in row)
            assert nested.to_numpy().dtype == np.int64
        finally:
            nested.close()
    finally:
        tensor.close()


# ===========================================================================
# 4. Non-contiguous input — Policy-B, and identical answers
# ===========================================================================

@needs_native
@pytest.mark.parametrize("dtype", FLOATING_DTYPES)
def test_non_contiguous_views_answer_exactly_as_their_materialization(dtype):
    """A transposed, narrowed, offset, or chained view must give the same
    indices as its own contiguous copy — the property that makes Policy-B
    legitimate rather than a compromise."""
    values = np.array([
        [1.0, 9.0, 3.0, NAN],
        [5.0, 5.0, -INF, 2.0],
        [INF, 0.0, -0.0, 8.0],
    ])
    base = _from_values(values, dtype)
    views = []
    try:
        transposed = base.T
        views.append(transposed)
        narrowed = base.narrow(1, 1, 3)
        views.append(narrowed)
        chained = narrowed.T
        views.append(chained)
        rows = base.narrow(0, 1, 2)
        views.append(rows)
        # The set really covers both Policy-B routes: at least one view is
        # genuinely strided, and at least one is contiguous but offset — so
        # a materialize-everything implementation and an
        # ignore-the-offset one would both fail below.
        assert any(not view.contiguous for view in views)
        assert any(view.contiguous and view._require_open().offset != 0
                   for view in views)
        for view in views:
            materialized = view.contiguous_copy()
            try:
                for axis in (None, 0, 1, -1):
                    for keepdims in (False, True):
                        assert _argmax_indices(
                            view, axis=axis, keepdims=keepdims) == \
                            _argmax_indices(materialized, axis=axis,
                                            keepdims=keepdims), (axis,
                                                                 keepdims)
            finally:
                materialized.close()
    finally:
        for view in views:
            view.close()
        base.close()


@needs_native
def test_a_narrowed_view_with_a_nonzero_offset_is_read_from_its_offset():
    """The offset really crosses the ABI: a narrow that starts past element
    0 must not answer as though it started at 0."""
    base = _from_values([9.0, 1.0, 2.0, 3.0, 8.0], "float64")
    view = base.narrow(0, 1, 4)
    try:
        assert view._require_open().offset == 1
        assert view.contiguous          # contiguous, but not at offset 0
        assert _argmax_indices(view) == ((), 3)     # 8.0, at view position 3
        assert _argmax_indices(base) == ((), 0)     # 9.0, at base position 0
    finally:
        view.close()
        base.close()


@needs_native
def test_the_policy_b_temporary_is_closed_and_the_input_is_untouched():
    """A non-contiguous input allocates exactly one private contiguous
    temporary, and live storage returns to baseline."""
    base = _from_values([[1.0, 4.0], [3.0, 2.0]], "float64")
    view = base.T
    before = view.to_numpy().tobytes()
    try:
        with live_storage_baseline():
            result = view.argmax(axis=1)
            result.close()
        assert view.to_numpy().tobytes() == before
        assert not view.contiguous
    finally:
        view.close()
        base.close()


# ===========================================================================
# 5. Validation and error precedence
# ===========================================================================

@needs_native
def test_a_closed_tensor_rejects_before_anything_else():
    tensor = _from_values([1.0, 2.0], "float64")
    tensor.close()
    for kwargs in ({}, {"axis": 0}, {"axis": "not an axis"},
                   {"keepdims": "not a bool"},
                   {"axis": 99, "keepdims": 3}):
        with pytest.raises(RuntimeError, match="closed"):
            tensor.argmax(**kwargs)


@needs_native
def test_an_int64_input_is_rejected_at_both_layers():
    with unchanged_world() as world:
        indices = NativeTensor.from_int64_array(
            np.array([1, 5, 2], dtype=np.int64))
        try:
            with pytest.raises(ValueError, match="floating"):
                indices.argmax()
            with pytest.raises(ValueError, match="floating"):
                indices._require_open().argmax()
            # ...and it beats every argument error.
            with pytest.raises(ValueError, match="floating"):
                indices.argmax(axis="not an axis")
            with pytest.raises(ValueError, match="floating"):
                indices.argmax(keepdims="not a bool")
            with pytest.raises(ValueError, match="floating"):
                indices.argmax(axis=99)
        finally:
            indices.close()
        assert world is not None


@needs_native
@pytest.mark.parametrize("axis", [1.0, "0", (0,), [0], None.__class__,
                                 object(), np.float64(0.0)])
def test_an_invalid_axis_type_raises_type_error(axis):
    tensor = _from_values([[1.0, 2.0]], "float64")
    try:
        with pytest.raises(TypeError, match="axis"):
            tensor.argmax(axis=axis)
    finally:
        tensor.close()


@needs_native
@pytest.mark.parametrize("axis", [True, False])
def test_a_bool_axis_is_rejected_even_though_bool_is_an_int(axis):
    tensor = _from_values([[1.0, 2.0]], "float64")
    try:
        with pytest.raises(TypeError, match="axis"):
            tensor.argmax(axis=axis)
    finally:
        tensor.close()


@needs_native
def test_a_numpy_integer_axis_is_accepted():
    """The existing axis authority takes ``numpy.integer``; ``argmax`` gets
    that for free by asking it rather than writing a second validator."""
    tensor = _from_values([[1.0, 7.0], [9.0, 2.0]], "float64")
    try:
        assert _argmax_indices(tensor, axis=np.int64(1)) == ((2,), [1, 0])
        assert _argmax_indices(tensor, axis=np.int32(-1)) == ((2,), [1, 0])
    finally:
        tensor.close()


@needs_native
@pytest.mark.parametrize("axis", [2, 3, 99, -3, -99])
def test_an_out_of_range_axis_raises_value_error(axis):
    tensor = _from_values([[1.0, 2.0]], "float64")
    try:
        with pytest.raises(ValueError, match="out of bounds"):
            tensor.argmax(axis=axis)
    finally:
        tensor.close()


@needs_native
@pytest.mark.parametrize("keepdims", [1, 0, None, "True", 1.0, (), [True]])
def test_a_non_bool_keepdims_raises_type_error(keepdims):
    tensor = _from_values([[1.0, 2.0]], "float64")
    try:
        with pytest.raises(TypeError, match="keepdims"):
            tensor.argmax(keepdims=keepdims)
        with pytest.raises(TypeError, match="keepdims"):
            tensor.argmax(axis=0, keepdims=keepdims)
    finally:
        tensor.close()


@needs_native
def test_the_error_precedence_is_the_contracted_one():
    """§17.6, driven by calls that are invalid in **two** ways at once. A
    guardrail that only ever passed one bad argument could not tell the
    ordering from the set of checks."""
    tensor = _from_values([[1.0, 2.0]], "float64")
    try:
        # 4 before 6: an invalid axis *type* beats an invalid keepdims.
        with pytest.raises(TypeError, match="axis"):
            tensor.argmax(axis="not an axis", keepdims="not a bool")
        # 5 before 6: an out-of-range axis beats an invalid keepdims, and
        # the axis error is a ValueError while the keepdims one is a
        # TypeError, so the two are distinguishable by kind as well.
        with pytest.raises(ValueError, match="out of bounds"):
            tensor.argmax(axis=99, keepdims="not a bool")
        # ...and a *valid* axis lets the keepdims error through, which is
        # what proves the two checks above were ordered rather than merely
        # both present.
        with pytest.raises(TypeError, match="keepdims"):
            tensor.argmax(axis=0, keepdims="not a bool")
    finally:
        tensor.close()
    # 1 before everything: a closed int64 tensor reports being closed, not
    # being an integer.
    indices = NativeTensor.from_int64_array(np.array([1], dtype=np.int64))
    indices.close()
    with pytest.raises(RuntimeError, match="closed"):
        indices.argmax(axis="not an axis")


@needs_native
def test_every_rejection_leaves_the_world_and_live_storage_untouched():
    for dtype in FLOATING_DTYPES:
        with unchanged_world(dtype) as world:
            with live_storage_baseline():
                for call in (
                    lambda: world.operand.argmax(axis="bad"),
                    lambda: world.operand.argmax(axis=True),
                    lambda: world.operand.argmax(axis=99),
                    lambda: world.operand.argmax(keepdims=1),
                    lambda: world.operand.argmax(axis=0, keepdims=None),
                    lambda: world.operand.argmax(axis=-5),
                ):
                    with pytest.raises((TypeError, ValueError)):
                        call()


# ===========================================================================
# 6. Autograd exclusion
# ===========================================================================

@needs_native
@pytest.mark.parametrize("dtype", FLOATING_DTYPES)
def test_the_result_is_a_plain_leaf_even_from_a_gradient_tracking_input(
        dtype):
    """The one place ``argmax`` differs from every other operation on a
    gradient-tracking tensor, and it is correct: the derivative of an index
    with respect to a value does not exist."""
    tensor = NativeTensor.from_array(
        np.array([[1.0, 7.0], [9.0, 2.0]]), dtype=dtype, requires_grad=True)
    try:
        assert tensor.requires_grad
        result = tensor.argmax(axis=1)
        try:
            assert result.requires_grad is False
            assert result.grad is None
            assert result.is_leaf is True
            assert result._parents == ()
            assert result._backward is None
            assert result._op == ""
            assert result._graph_resources == ()
            assert result._expected_versions == ()
            # backward() on the result is refused by the integer barrier
            # rather than by a missing graph.
            with pytest.raises((ValueError, RuntimeError)):
                result.backward()
        finally:
            result.close()
        # The input's own graph is untouched and still usable.
        assert tensor.requires_grad and tensor.grad is None
        loss = tensor.sum()
        try:
            loss.backward()
        finally:
            loss.close()
        assert tensor.grad is not None
        assert np.array_equal(tensor.grad.to_numpy(), np.ones((2, 2)))
    finally:
        tensor.close()


@needs_native
def test_argmax_never_calls_the_graph_constructor():
    """Structural, from the AST: ``NativeTensor.argmax`` reaches
    ``_from_core`` and never ``_from_op``. A future edit that made it build
    a graph would fail here as well as behaviourally."""
    from tensorforge.experimental import native_tensor

    tree = ast.parse(Path(native_tensor.__file__).read_text(encoding="utf-8"))
    cls = next(node for node in ast.walk(tree)
               if isinstance(node, ast.ClassDef) and node.name
               == "NativeTensor")
    method = next(child for child in cls.body
                  if isinstance(child, ast.FunctionDef)
                  and child.name == "argmax")
    called = {node.attr for node in ast.walk(method)
              if isinstance(node, ast.Attribute)}
    assert "_from_core" in called
    assert "_from_op" not in called
    # ...and the control: a method that *does* build a graph is found by the
    # same scan, so "not present" is a measurement.
    contiguous = next(child for child in cls.body
                      if isinstance(child, ast.FunctionDef)
                      and child.name == "contiguous_copy")
    assert "_from_op" in {node.attr for node in ast.walk(contiguous)
                          if isinstance(node, ast.Attribute)}


@needs_native
def test_a_parameters_version_and_gradient_are_untouched_by_argmax():
    parameter = NativeParameter(np.array([[1.0, 5.0], [3.0, 2.0]]))
    try:
        version = parameter.version
        result = parameter.argmax(axis=1)
        try:
            assert result.tolist() == [1, 0]
            assert result.requires_grad is False
        finally:
            result.close()
        assert parameter.version == version
        assert parameter.grad is None
        assert parameter.requires_grad is True
    finally:
        parameter.close()


# ===========================================================================
# 7. The C ABI, driven directly
# ===========================================================================

def _typed_core(shape, dtype, values=None):
    core = cpp.NativeTensorCore._typed(shape, dtype)
    if values is not None:
        core._storage.copy_from(
            np.ascontiguousarray(values, dtype=cpp._DTYPE_NUMPY[dtype]))
    return core


@needs_native
@pytest.mark.parametrize("dtype", FLOATING_DTYPES)
def test_the_abi_accepts_a_floating_source_with_an_int64_destination(dtype):
    """The claim no structural check can make, and exit-gate item 3a: a
    ``require_floating`` or a ``require_matching_dtype`` on the destination
    would reject **every** valid call, so the valid call is driven."""
    library = cpp._require_library()
    source = _typed_core((6,), dtype, [1.0, 7.0, 3.0, 9.0, 9.0, 2.0])
    destination = _typed_core((2,), INDEX_DTYPE)
    try:
        library.tf_core_argmax(
            source._storage._require_open(), 0,
            destination._storage._require_open(), 2, 3, 1)
        assert destination.to_numpy().tolist() == [1, 0]
        assert source.dtype == dtype and destination.dtype == INDEX_DTYPE
        assert source.dtype != destination.dtype
    finally:
        destination.close()
        source.close()


@needs_native
def test_the_abi_rejects_an_int64_source_independently_of_python():
    library = cpp._require_library()
    source = _typed_core((4,), INDEX_DTYPE, [1, 5, 2, 3])
    destination = _typed_core((1,), INDEX_DTYPE)
    before = destination.to_numpy().tobytes()
    try:
        with pytest.raises(ValueError, match="floating"):
            library.tf_core_argmax(
                source._storage._require_open(), 0,
                destination._storage._require_open(), 1, 4, 1)
        assert destination.to_numpy().tobytes() == before
    finally:
        destination.close()
        source.close()


@needs_native
@pytest.mark.parametrize("dtype", FLOATING_DTYPES)
def test_the_abi_rejects_a_floating_destination(dtype):
    library = cpp._require_library()
    source = _typed_core((4,), "float64", [1.0, 5.0, 2.0, 3.0])
    destination = _typed_core((1,), dtype, [-7.5])
    before = destination.to_numpy().tobytes()
    try:
        with pytest.raises(ValueError, match="int64"):
            library.tf_core_argmax(
                source._storage._require_open(), 0,
                destination._storage._require_open(), 1, 4, 1)
        assert destination.to_numpy().tobytes() == before
    finally:
        destination.close()
        source.close()


@needs_native
def test_the_abi_rejects_every_malformed_layout_without_writing():
    library = cpp._require_library()
    source = _typed_core((6,), "float64", [1.0, 7.0, 3.0, 9.0, 9.0, 2.0])
    destination = _typed_core((2,), INDEX_DTYPE, [-424242, 987654321])
    before = destination.to_numpy().tobytes()
    huge = (2 ** 63 - 1) // 2
    cases = {
        "zero outer": (0, 3, 1, 0),
        "zero axis_length": (2, 0, 1, 0),
        "zero inner": (2, 3, 0, 0),
        "negative outer": (-2, 3, 1, 0),
        "negative axis_length": (2, -3, 1, 0),
        "negative inner": (2, 3, -1, 0),
        "product overflow": (huge, 4, 1, 0),
        "negative offset": (2, 3, 1, -1),
        "source span too long": (2, 4, 1, 0),
        "offset pushes the span out": (2, 3, 1, 2),
        "destination too small": (3, 2, 1, 0),
        "destination too large": (1, 6, 1, 0),
    }
    try:
        for label, (outer, axis_length, inner, offset) in cases.items():
            with pytest.raises(ValueError):
                library.tf_core_argmax(
                    source._storage._require_open(), offset,
                    destination._storage._require_open(),
                    outer, axis_length, inner)
            assert destination.to_numpy().tobytes() == before, label
        # A null destination handle is refused too, and does not crash.
        with pytest.raises(ValueError):
            library.tf_core_argmax(
                source._storage._require_open(), 0, None, 2, 3, 1)
        with pytest.raises(ValueError):
            library.tf_core_argmax(
                None, 0, destination._storage._require_open(), 2, 3, 1)
        assert destination.to_numpy().tobytes() == before
        # The non-vacuity control: the same destination with valid
        # arguments succeeds and writes.
        library.tf_core_argmax(
            source._storage._require_open(), 0,
            destination._storage._require_open(), 2, 3, 1)
        assert destination.to_numpy().tolist() == [1, 0]
        assert destination.to_numpy().tobytes() != before
    finally:
        destination.close()
        source.close()


@needs_native
def test_the_abi_rejects_a_self_aliasing_handle_and_writes_nothing():
    """A genuine alias is unreachable through the role checks — one storage
    carries one dtype, and this call needs a floating source and an int64
    destination — so what is observable, and what is asserted, is that
    passing the same handle for both is **rejected** and leaves the storage
    byte-for-byte unchanged, at either dtype."""
    library = cpp._require_library()
    for dtype, values in (("float64", [1.0, 2.0]), (INDEX_DTYPE, [1, 2])):
        storage = _typed_core((2,), dtype, values)
        before = storage.to_numpy().tobytes()
        try:
            with pytest.raises(ValueError):
                library.tf_core_argmax(
                    storage._storage._require_open(), 0,
                    storage._storage._require_open(), 2, 1, 1)
            assert storage.to_numpy().tobytes() == before, dtype
        finally:
            storage.close()


@needs_native
def test_the_export_records_and_then_clears_the_thread_local_slot():
    """A guarded export records a rejection in the thread-local slot and
    clears it on the **next** entry, so a stale code is never misread as a
    later call's failure.

    Driven with the ``errcheck`` hook temporarily detached, because that
    hook is what normally consumes and clears the slot — leaving it
    attached would mean testing the wrapper rather than the export. It is
    restored in a ``finally``."""
    library = cpp._require_library()
    source = _typed_core((4,), "float64", [1.0, 4.0, 2.0, 3.0])
    destination = _typed_core((1,), INDEX_DTYPE)
    hook = library.tf_core_argmax.errcheck
    assert hook is not None, "the export is not hooked at all"
    try:
        # ``errcheck`` must stay callable, so it is replaced by an identity
        # rather than removed; that is what leaves the slot for this test to
        # read instead of having it consumed and cleared by the wrapper.
        library.tf_core_argmax.errcheck = \
            lambda result, function, arguments: result
        library.tf_clear_error()
        # A rejection: recorded, with a message naming the operation.
        library.tf_core_argmax(
            source._storage._require_open(), 0,
            destination._storage._require_open(), 0, 4, 1)
        assert library.tf_last_error_code() != cpp.TF_OK
        message = library.tf_last_error_message().decode()
        assert "argmax" in message
        # A valid call clears the slot on entry and writes its result.
        library.tf_core_argmax(
            source._storage._require_open(), 0,
            destination._storage._require_open(), 1, 4, 1)
        assert library.tf_last_error_code() == cpp.TF_OK
        assert destination.to_numpy().tolist() == [1]
    finally:
        library.tf_core_argmax.errcheck = hook
        library.tf_clear_error()
        destination.close()
        source.close()
    # ...and with the hook restored, a rejection is a Python exception again.
    source = _typed_core((4,), "float64", [1.0, 4.0, 2.0, 3.0])
    destination = _typed_core((1,), INDEX_DTYPE)
    try:
        with pytest.raises(ValueError, match="argmax"):
            library.tf_core_argmax(
                source._storage._require_open(), 0,
                destination._storage._require_open(), 0, 4, 1)
    finally:
        destination.close()
        source.close()


@needs_native
def test_a_source_offset_crosses_the_abi_exactly():
    library = cpp._require_library()
    source = _typed_core((6,), "float64", [9.0, 1.0, 2.0, 8.0, 3.0, 4.0])
    destination = _typed_core((1,), INDEX_DTYPE)
    try:
        library.tf_core_argmax(
            source._storage._require_open(), 1,
            destination._storage._require_open(), 1, 5, 1)
        # From element 1: 1, 2, 8, 3, 4 -> the maximum 8.0 at position 2.
        assert destination.to_numpy().tolist() == [2]
    finally:
        destination.close()
        source.close()


# ===========================================================================
# 8. Failure cleanup — four distinct injections
# ===========================================================================

@needs_native
def test_a_failed_policy_b_temporary_allocation_leaks_nothing():
    """Injection 1: the private contiguous copy a non-contiguous input
    needs. Nothing else has been allocated when it fires."""
    base = _from_values([[1.0, 4.0], [3.0, 2.0]], "float64")
    view = base.T
    fired = []
    original = cpp.NativeTensorCore.contiguous_copy

    def failing(self):
        fired.append(True)
        raise MemoryError("injected: Policy-B temporary")

    try:
        with unchanged_world() as world:
            with live_storage_baseline():
                cpp.NativeTensorCore.contiguous_copy = failing
                try:
                    with pytest.raises(MemoryError, match="injected"):
                        view.argmax(axis=1)
                finally:
                    cpp.NativeTensorCore.contiguous_copy = original
            assert world is not None
        assert fired, "the injection never fired"
        # The control: the same call succeeds once the injection is removed.
        result = view.argmax(axis=1)
        try:
            assert result.tolist() == [1, 0]
        finally:
            result.close()
    finally:
        cpp.NativeTensorCore.contiguous_copy = original
        view.close()
        base.close()


@needs_native
def test_a_failed_destination_allocation_closes_the_policy_b_temporary():
    """Injection 2: the ``int64`` destination. It fires **after** the
    Policy-B temporary exists, so this is where the temporary's cleanup is
    proved — the reason the two are distinct injections rather than one."""
    base = _from_values([[1.0, 4.0], [3.0, 2.0]], "float64")
    view = base.T
    allocated = []
    fired = []
    original_typed = cpp.NativeTensorCore._typed
    original_copy = cpp.NativeTensorCore.contiguous_copy

    def recording_copy(self):
        temporary = original_copy(self)
        allocated.append(temporary)
        return temporary

    def failing_typed(cls, shape, dtype, device="cpu", **kwargs):
        if dtype == INDEX_DTYPE:
            fired.append(True)
            raise MemoryError("injected: index destination")
        return original_typed.__func__(cls, shape, dtype, device=device,
                                       **kwargs)

    try:
        with unchanged_world() as world:
            with live_storage_baseline():
                cpp.NativeTensorCore.contiguous_copy = recording_copy
                cpp.NativeTensorCore._typed = classmethod(failing_typed)
                try:
                    with pytest.raises(MemoryError, match="injected"):
                        view.argmax(axis=1)
                finally:
                    cpp.NativeTensorCore._typed = original_typed
                    cpp.NativeTensorCore.contiguous_copy = original_copy
            assert world is not None
        assert fired, "the injection never fired"
        # The temporary is retained **strongly** by the test and is proved
        # closed while still referenced — no collection timing involved.
        assert len(allocated) == 1
        assert allocated[0]._closed is True
    finally:
        cpp.NativeTensorCore._typed = original_typed
        cpp.NativeTensorCore.contiguous_copy = original_copy
        view.close()
        base.close()


@needs_native
def test_a_failed_native_call_closes_the_destination_and_the_temporary():
    """Injection 3: the kernel call itself, which fires with both the
    temporary and the destination alive. ``BaseException`` is used here, so
    the cleanup is proved unconditional rather than ``except Exception``."""

    class Interrupt(BaseException):
        """Not an ``Exception``: an ``except Exception`` cleanup would miss
        it, which is exactly what this injection is for."""

    base = _from_values([[1.0, 4.0], [3.0, 2.0]], "float64")
    view = base.T
    allocated = []
    fired = []
    original_typed = cpp.NativeTensorCore._typed
    original_copy = cpp.NativeTensorCore.contiguous_copy
    library = cpp._require_library()
    original_kernel = library.tf_core_argmax

    def recording_copy(self):
        temporary = original_copy(self)
        allocated.append(("temporary", temporary))
        return temporary

    def recording_typed(cls, shape, dtype, device="cpu", **kwargs):
        core = original_typed.__func__(cls, shape, dtype, device=device,
                                       **kwargs)
        if dtype == INDEX_DTYPE:
            allocated.append(("destination", core))
        return core

    def failing_kernel(*args):
        fired.append(True)
        raise Interrupt("injected: native call")

    try:
        with unchanged_world() as world:
            with live_storage_baseline():
                cpp.NativeTensorCore.contiguous_copy = recording_copy
                cpp.NativeTensorCore._typed = classmethod(recording_typed)
                library.tf_core_argmax = failing_kernel
                try:
                    with pytest.raises(Interrupt):
                        view.argmax(axis=1)
                finally:
                    library.tf_core_argmax = original_kernel
                    cpp.NativeTensorCore._typed = original_typed
                    cpp.NativeTensorCore.contiguous_copy = original_copy
            assert world is not None
        assert fired, "the injection never fired"
        kinds = [kind for kind, _ in allocated]
        assert kinds == ["temporary", "destination"], kinds
        for kind, core in allocated:
            assert core._closed is True, kind
        # The control: the same call succeeds with the injection removed.
        result = view.argmax(axis=1)
        try:
            assert result.tolist() == [1, 0]
        finally:
            result.close()
    finally:
        library.tf_core_argmax = original_kernel
        cpp.NativeTensorCore._typed = original_typed
        cpp.NativeTensorCore.contiguous_copy = original_copy
        view.close()
        base.close()


@needs_native
def test_a_failed_wrapper_publication_closes_the_core_it_was_handed():
    """Injection 4: the ``NativeTensor`` publication, which is a different
    position again — the Core has already been published successfully and
    is the tensor layer's to release."""
    tensor = _from_values([[1.0, 4.0], [3.0, 2.0]], "float64")
    produced = []
    fired = []
    original_core_argmax = cpp.NativeTensorCore.argmax
    original_from_core = NativeTensor._from_core

    def recording_argmax(self, axis=None, keepdims=False):
        core = original_core_argmax(self, axis=axis, keepdims=keepdims)
        produced.append(core)
        return core

    def failing_from_core(cls, core, owns_core=True):
        fired.append(True)
        raise KeyboardInterrupt("injected: wrapper publication")

    try:
        with unchanged_world() as world:
            with live_storage_baseline():
                cpp.NativeTensorCore.argmax = recording_argmax
                NativeTensor._from_core = classmethod(failing_from_core)
                try:
                    with pytest.raises(KeyboardInterrupt, match="injected"):
                        tensor.argmax(axis=1)
                finally:
                    NativeTensor._from_core = original_from_core
                    cpp.NativeTensorCore.argmax = original_core_argmax
            assert world is not None
        assert fired, "the injection never fired"
        assert len(produced) == 1
        # Retained strongly here, and proved closed while still referenced.
        assert produced[0]._closed is True
        result = tensor.argmax(axis=1)
        try:
            assert result.tolist() == [1, 0]
        finally:
            result.close()
    finally:
        NativeTensor._from_core = original_from_core
        cpp.NativeTensorCore.argmax = original_core_argmax
        tensor.close()


@needs_native
def test_a_successful_argmax_returns_live_storage_to_baseline():
    """The control for all four injections above: the ordinary path
    allocates and releases exactly what it should."""
    for dtype in FLOATING_DTYPES:
        base = _from_values([[1.0, 4.0], [3.0, 2.0]], dtype)
        view = base.T
        with live_storage_baseline():
            for source in (base, view):
                for axis in (None, 0, 1):
                    result = source.argmax(axis=axis)
                    result.close()
        view.close()
        base.close()


# ===========================================================================
# 9. Absence — what K3 did not ship
# ===========================================================================

def test_no_max_or_second_output_was_smuggled_in():
    """§17.10 is permanent: a kernel that finds the position of a maximum
    necessarily knows the maximum, and Phase K does not expose it."""
    unit = (REPO_ROOT / "cpp" / "src" / "indexing.cpp").read_text(
        encoding="utf-8")
    code = re.sub(r"//[^\n]*", " ",
                  re.sub(r"/\*.*?\*/", " ", unit, flags=re.S))
    # ``argmax``'s signature still has exactly one destination handle and no
    # second output. The unit now carries two exports — K4 added
    # ``tf_core_index_select`` beside it, which is the phase's committed
    # maximum — so the count is asserted at two and the *signature* claim is
    # read from argmax's own declaration.
    assert code.count("TF_EXPORT") == 2
    signature = code.split("TF_EXPORT void tf_core_argmax(", 1)[1].split(")")[0]
    assert signature.count("void* dst_handle") == 1
    assert signature.count("void*") == 2      # one const source, one dest
    for banned in ("max_handle", "value_handle", "tf_core_max"):
        assert banned not in code, banned
    tensor_source = _code_only(
        (REPO_ROOT / "src" / "tensorforge" / "experimental"
         / "native_tensor.py").read_text(encoding="utf-8"))
    core_source = _code_only(
        (REPO_ROOT / "src" / "tensorforge" / "backends" / "cpp.py"
         ).read_text(encoding="utf-8"))
    for banned in ("argmin", "max_with_indices"):
        assert banned not in tensor_source, banned
        assert banned not in core_source, banned


def test_no_integer_arithmetic_casting_or_promotion_appeared():
    for owner in (NativeTensor, cpp.NativeTensorCore, cpp.NativeStorage):
        for absent in ("astype", "cast", "to", "type", "long", "int",
                       "float", "double", "promote", "as_type", "cpu",
                       "cuda", "is_integer", "is_floating"):
            assert not hasattr(owner, absent), (owner.__name__, absent)


@needs_native
def test_an_int64_result_is_still_refused_by_every_barrier():
    """The result of an `argmax` is an ordinary `int64` tensor, so every K1
    barrier applies to it unchanged — proved against a real one produced by
    the new operation rather than against a constructed stand-in."""
    tensor = _from_values([[1.0, 7.0], [9.0, 2.0]], "float64")
    indices = tensor.argmax(axis=1)
    try:
        assert indices.dtype == INDEX_DTYPE
        with pytest.raises(ValueError):
            NativeParameter(indices)
        module = NativeModule()
        for persistent in (True, False):
            with pytest.raises(ValueError):
                module.register_buffer("probe", indices,
                                       persistent=persistent)
        with pytest.raises((ValueError, RuntimeError)):
            indices.backward()
        for operation in ("relu", "sqrt", "reciprocal", "exp", "log", "sum",
                          "mean", "softmax", "log_softmax"):
            with pytest.raises(ValueError, match="floating"):
                getattr(indices, operation)()
        for other in (tensor,):
            with pytest.raises(ValueError, match="floating"):
                indices.add(other)
            with pytest.raises(ValueError, match="floating"):
                other.add(indices)
        # ...and an argmax of an argmax is refused for the same reason.
        with pytest.raises(ValueError, match="floating"):
            indices.argmax()
    finally:
        indices.close()
        tensor.close()


@needs_native
def test_the_int64_result_still_supports_exactly_what_k2_shipped():
    """The complementary half: views, the contiguous copy, host inspection,
    and ``close()`` all work on an `argmax` result, because it is the same
    integer tensor K2 shipped."""
    tensor = _from_values(np.arange(12.0).reshape(3, 4), "float64")
    indices = tensor.argmax(axis=1)
    try:
        assert indices.shape == (3,)
        assert indices.tolist() == [3, 3, 3]
        reshaped = indices.reshape((3, 1))
        try:
            assert reshaped.dtype == INDEX_DTYPE
            assert reshaped.tolist() == [[3], [3], [3]]
        finally:
            reshaped.close()
        transposed = indices.reshape((1, 3)).T
        try:
            assert transposed.shape == (3, 1)
            copied = transposed.contiguous_copy()
            try:
                assert copied.dtype == INDEX_DTYPE
                assert copied.tolist() == [[3], [3], [3]]
            finally:
                copied.close()
        finally:
            transposed.close()
        narrowed = indices.narrow(0, 1, 2)
        try:
            assert narrowed.tolist() == [3, 3]
        finally:
            narrowed.close()
        assert np.array_equal(indices.to_numpy(),
                              np.array([3, 3, 3], dtype=np.int64))
    finally:
        indices.close()
        tensor.close()


@needs_native
def test_argmax_never_touches_numpy_for_tensor_data():
    """The training path's rule holds here too: no tensor data round-trips
    through a host buffer. Read from the code, so the docstring explaining
    the boundary cannot satisfy it."""
    source = (REPO_ROOT / "src" / "tensorforge" / "backends" / "cpp.py"
              ).read_text(encoding="utf-8")
    tree = ast.parse(source)
    core = next(node for node in ast.walk(tree)
                if isinstance(node, ast.ClassDef)
                and node.name == "NativeTensorCore")
    method = next(child for child in core.body
                  if isinstance(child, ast.FunctionDef)
                  and child.name == "argmax")
    names = {node.id for node in ast.walk(method) if isinstance(node, ast.Name)}
    attributes = {node.attr for node in ast.walk(method)
                  if isinstance(node, ast.Attribute)}
    assert "np" not in names
    assert "to_numpy" not in attributes and "from_array" not in attributes
    # The control: a method that *does* touch NumPy is found by the same
    # scan, so "not present" is a measurement rather than an artifact.
    typed_from_array = next(child for child in core.body
                            if isinstance(child, ast.FunctionDef)
                            and child.name == "_typed_from_array")
    assert "np" in {node.id for node in ast.walk(typed_from_array)
                    if isinstance(node, ast.Name)}


@needs_native
def test_the_stable_line_is_untouched_and_gained_no_argmax():
    import tensorforge

    assert not hasattr(tensorforge.Tensor, "argmax")
    assert cpp.backend_info()["stable_framework_integration"] is False
    assert "argmax" not in tensorforge.__all__


# ===========================================================================
# 10. The live docstrings describe the post-K3 world
# ===========================================================================
#
# A production docstring that still says "there is no native argmax" is a
# defect of the same kind as a stale registry value: a reader who trusts it
# reaches a wrong conclusion about what the runtime does. These checks
# protect the **semantic claims** — what each docstring must no longer say,
# and what it must now say instead — and deliberately not the wording, the
# paragraph order, or the length, so ordinary prose editing does not break
# them.


def _docstring(obj):
    """One flattened, lower-cased docstring: whitespace collapsed and
    markup stripped, so a claim split across lines or wrapped in ``rst``
    emphasis still reads as one sentence."""
    return re.sub(r"\s+", " ", re.sub(r"[*`]", "", obj.__doc__ or "")).lower()


def _flat_cpp_comments(relative):
    """A C++ file's comment prose, flattened and lower-cased.

    The ``//`` leaders are removed **before** the whitespace collapse: a
    claim wrapped across two comment lines otherwise reads as
    ``"k3's // specific …"``, and a checker that missed it would pass on
    text that says the opposite of what it means."""
    text = (REPO_ROOT / relative).read_text(encoding="utf-8")
    stripped = re.sub(r"^\s*//+", " ", text, flags=re.M)
    return re.sub(r"\s+", " ", re.sub(r"[*`]", "", stripped)).lower()


def test_the_cpp_comment_flattener_joins_wrapped_lines():
    """Negative control for the flattener, on a temporary file-shaped
    string: the leader must be gone and the two halves must join."""
    import textwrap

    sample = textwrap.dedent("""\
        // It is **K3's
        //     specific traversal**: it searches a run.
        int x = 1;  // trailing
        """)
    stripped = re.sub(r"^\s*//+", " ", sample, flags=re.M)
    flat = re.sub(r"\s+", " ", re.sub(r"[*`]", "", stripped)).lower()
    # The two comment lines join into one readable claim...
    assert "k3's specific traversal" in flat
    # ...and the leaders that would have split it are gone. (A trailing
    # comment marker after code is deliberately left alone: this flattener
    # exists for the file-header comment blocks, not for code lines.)
    assert not re.search(r"^\s*//", flat)
    assert "k3's // specific" not in flat


# Each entry is (label, pattern). A match means the surface still carries a
# claim that K3 made false. Every one is driven against text it must catch
# and text it must not by the negative control below.
_STALE_ARGMAX_CLAIMS = (
    ("no native argmax exists",
     r"\bthere is (deliberately )?no native argmax\b"
     r"|\bno native argmax (exists|is (implemented|available|shipped))\b"),
    ("argmax is absent because nobody shipped it",
     r"\bargmax\b[^.]{0,60}\babsent\b[^.]{0,60}"
     r"\b(nobody|no one|has not been|hasn't been)\b"),
    ("argmax belongs to a later milestone",
     r"\bargmax\b[^.]{0,60}\b(belongs to|awaits|until) (a later|a future|k3)\b"),
    # The bare list form K2's ``native_tensor`` docstring used. Narrow on
    # purpose: it matches "no argmax" and not "no max", "no argmin", or
    # "no index_select", each of which is still an accurate thing to write.
    ("argmax listed among the absent operations",
     r"\bno argmax\b"),
    ("the caller list is closed",
     r"\band for nothing else\b|\bonly integer caller\b"
     r"|\bthe only caller is\b"),
)


def _stale_claims(text):
    """Every stale claim in one flattened body."""
    return [label for label, pattern in _STALE_ARGMAX_CLAIMS
            if re.search(pattern, text, re.I)]


def test_the_stale_claim_scanner_can_actually_fail():
    """The control every edit to ``_STALE_ARGMAX_CLAIMS`` requires, on
    temporary strings only — no repository file is read here.

    Each pair is the sentence a surface used to carry and the sentence that
    replaced it, so the scanner is proved able to tell them apart rather
    than merely to pass on today's text."""
    stale_and_corrected = (
        # native_metrics.py, as it read from K2 to K3, and as it reads now
        ("there is deliberately no native argmax",
         "a native argmax exists: phase k milestone k3 shipped it, and this "
         "metric still reports through the host boundary"),
        ("a native argmax is absent because nobody has shipped it",
         "native argmax exists; native_accuracy deliberately does not use it"),
        # native_tensor.py, as it read at K2, and as it reads now
        ("there is no integer arithmetic, no argmax, and no index selection",
         "argmax accepts a floating input and returns an int64 index tensor; "
         "there is still no max, argmin, or index_select"),
        ("the argmax operation belongs to a later milestone",
         "the argmax operation landed at k3"),
        # the two allocator docstrings, as they read at K2, and as they read
        # now
        ("its only integer caller is _from_int64_array",
         "its integer uses are the host ingress, the contiguous copy, and "
         "the k3 argmax destination"),
        ("for _from_int64_array and the integer arm of contiguous_copy, and "
         "for nothing else",
         "for the host ingress, the index arm of contiguous_copy, and the "
         "argmax destination"),
    )
    for stale, corrected in stale_and_corrected:
        assert _stale_claims(stale), f"the scanner missed: {stale!r}"
        assert _stale_claims(corrected) == [], (
            corrected, _stale_claims(corrected))
    # ...and it does not fire on an accurate sentence that merely names the
    # things it scans for, which is the false positive a blunter pattern
    # would produce on the very prose documenting the correction.
    for allowed in (
        "a native argmax exists since k3",
        "there is no native max beside the argmax",
        "argmin and index_select do not exist yet",
        "the k3 argmax destination is allocated through this helper",
        "no general integer allocator is exposed",
    ):
        assert _stale_claims(allowed) == [], (allowed, _stale_claims(allowed))


def test_the_metric_docstring_records_that_a_native_argmax_exists():
    """`native_metrics` must no longer claim the absence, and must record
    both halves of the truth: the operation exists, and this helper still
    does not use it."""
    from tensorforge.experimental import native_metrics

    text = _docstring(native_metrics)
    assert _stale_claims(text) == [], _stale_claims(text)
    # The operation exists, with the milestone and the three names.
    assert re.search(r"native argmax exists", text)
    assert "k3" in text
    for name in ("nativetensorcore.argmax", "nativetensor.argmax",
                 "tf_core_argmax"):
        assert name in text, name
    # ...and the metric still reports through the host boundary, on purpose.
    assert "deliberate" in text
    assert "to_numpy" in text and "np.argmax" in text
    # Substituting it would not change what the helper is.
    assert re.search(r"would not make (this|it)[^.]{0,80}"
                     r"(native runtime operation|runtime operation)", text)
    # The two tie/NaN conventions are not claimed to be equivalent.
    assert "first-nan" in text or "first nan" in text
    assert re.search(r"not[^.]{0,60}equivalent"
                     r"|not the same contract", text)


def test_the_tensor_module_docstring_records_the_k3_operation():
    """`native_tensor`'s module docstring must no longer say no argmax
    exists, and must state the capability *and* its boundary."""
    from tensorforge.experimental import native_tensor

    text = _docstring(native_tensor)
    assert _stale_claims(text) == [], _stale_claims(text)
    assert "k3" in text and "argmax" in text
    # What it accepts and what it returns.
    assert re.search(r"floating", text)
    assert re.search(r"int64", text)
    # int64 is still not a compute dtype, and integers still do nothing.
    assert re.search(r"not[^.]{0,60}compute dtype", text)
    assert re.search(r"no arithmetic|has no arithmetic", text)
    assert re.search(r"cannot[^.]{0,80}enter a reduction", text)
    # The operations that still do not exist.
    for absent in ("max", "argmin", "promotion", "casting"):
        assert absent in text, absent
    # ``index_select`` left that list at K4, which shipped it, so the
    # docstring must record it as a **capability** rather than as an
    # absence — the §37.2 move, asserted in both directions.
    assert "index_select" in text
    assert "k4" in text
    assert not re.search(r"no\s+index_select", text), text[:400]
    # ...and the exclusions an argmax result inherits.
    for role in ("autograd", "parameter", "buffer", "optimizer", "checkpoint"):
        assert role in text, role


@pytest.mark.parametrize("owner,name", [
    (cpp.NativeStorage, "NativeStorage"),
    (cpp.NativeTensorCore, "NativeTensorCore"),
])
def test_the_private_allocator_docstrings_list_the_k3_destination(owner, name):
    """Both private ``_typed`` helpers must name the K3 ``argmax``
    destination among their integer callers, and neither may still present
    its K2 caller list as exhaustive."""
    text = _docstring(owner._typed)
    assert _stale_claims(text) == [], (name, _stale_claims(text))
    # The K3 caller is named, and so are the two that preceded it.
    assert "argmax" in text, name
    assert "k3" in text, name
    assert re.search(r"ingress", text), name
    assert re.search(r"contiguous_copy|materializ", text), name
    # The list is presented as the current inventory, not a closed one.
    assert re.search(r"a later milestone may add a caller", text), name
    # ...and the facts that did not move with it.
    assert "private" in text, name
    assert re.search(r"no general integer allocator", text), name
    assert re.search(r"zero-initialized|zero_initialize", text), name
    assert re.search(r"nothing casts or promotes|no casting", text), name


def test_the_public_constructors_still_reject_int64_as_the_docstrings_say():
    """The docstrings above claim the public constructors are unmoved. That
    is a runtime claim, so it is driven rather than read."""
    for build in (
        lambda: cpp.NativeStorage(4, dtype="int64"),
        lambda: cpp.NativeTensorCore.zeros((2,), dtype="int64"),
        lambda: cpp.NativeTensorCore.full((2,), 1, dtype="int64"),
        lambda: cpp.NativeTensorCore.from_array([1, 2], dtype="int64"),
        lambda: NativeTensor.zeros((2,), dtype="int64"),
    ):
        with pytest.raises(ValueError):
            build()
    with pytest.raises(ValueError):
        cpp.normalize_dtype("int64")


def test_the_indexing_header_does_not_promise_a_shared_traversal():
    """The header must not claim both of its declarations are reused by the
    next milestone: the traversal is K3's own, and only the role guard is
    expected to carry forward."""
    flat = _flat_cpp_comments("cpp/include/tf_indexing_internal.h")
    assert not re.search(r"both\s+(narrow\s+and\s+)?reused", flat)
    assert not re.search(r"both reused by the milestone that follows", flat)
    # The traversal is described as K3's own, and the description follows the
    # name it qualifies. Located by position rather than by a bounded window,
    # because the intervening prose legitimately contains sentence stops.
    traversal = flat.index("argmax_contiguous")
    own = flat.index("k3's specific index-producing traversal")
    assert own > traversal
    # ...and the guard — not the traversal — is the piece named as expected
    # to carry to K4. Asserted as the claim itself rather than as a
    # proximity window, which would break on ordinary prose editing.
    assert "require_index" in flat
    assert re.search(r"expected to (stay|remain) useful to k4", flat)
    # The header is the shared *home*, which is a different claim.
    assert "architectural home" in flat
    assert re.search(r"owns its own traversal", flat)


def test_the_alias_comment_does_not_promise_a_shared_validator():
    """`argument_error` is file-local to K3's export. The comment beside the
    alias check must say so rather than describe itself as K4 groundwork —
    and now that K4 has landed, the claim is checked in either tense, so the
    record can say what actually happened without the guardrail forcing a
    stale future tense."""
    flat = _flat_cpp_comments("cpp/src/indexing.cpp")
    assert not re.search(r"a later milestone reusing this validator", flat)
    assert "defense in depth" in flat
    assert re.search(r"k4 (will validate|validates) its abi independently"
                     r"|validates? its abi independently", flat)
    # ...and it points at the contract that gives K4 its own order.
    assert "22.9" in flat
    # The structural half of the same claim, read from the code: K4 really
    # did define its own validator rather than reuse this one.
    code = re.sub(r"//[^\n]*", " ",
                  re.sub(r"/\*.*?\*/", " ",
                         (REPO_ROOT / "cpp" / "src" / "indexing.cpp"
                          ).read_text(encoding="utf-8"), flags=re.S))
    argmax_body = code.split("TF_EXPORT void tf_core_argmax(", 1)[1].split(
        "TF_EXPORT void tf_core_index_select(", 1)[0]
    assert "index_select_argument_error" not in argmax_body
    assert "index_select_argument_error" in code


def test_k3_added_no_benchmark_timing_or_performance_control():
    """K3 is a capability milestone, not a benchmark one, and it added no
    dispatch or timing surface to the production files it touched.

    Scanned over the **production** sources rather than over this module,
    which would trivially find its own banned list."""
    production = {
        "cpp/src/indexing.cpp",
        "cpp/include/tf_indexing_internal.h",
    }
    # Matched on **word boundaries**: a bare substring ban would fire on
    # "compute" for "omp" and on "clocking" for "clock", which is a scanner
    # that rejects the prose documenting the rule.
    banned = re.compile(
        r"\b(chrono|clock|getenv|omp|restrict|block_size|set_path|"
        r"which_kernel|rdtsc)\b|std::thread|#pragma\s+omp|fast-math|_mm_|"
        r"__builtin_prefetch")
    # ...and over **code only**: both files document that the predicate is
    # never a function of a clock or an environment variable, and a scan
    # that read that sentence would reject the very prose stating the rule.
    for relative in sorted(production):
        text = (REPO_ROOT / relative).read_text(encoding="utf-8")
        code = re.sub(r"//[^\n]*", " ",
                      re.sub(r"/\*.*?\*/", " ", text, flags=re.S))
        found = banned.search(code)
        assert found is None, (relative, found and found.group(0))
        # The comment stripper really removed something here, so "code
        # only" is a measurement rather than a no-op.
        assert len(code) < len(text), relative
    # The scanner's negative controls, on temporary strings.
    assert banned.search("#pragma omp parallel for") is not None
    assert banned.search("auto t = std::chrono::steady_clock::now();")
    assert banned.search("the noexcept compute kernel") is None
    assert banned.search("a clocking convention") is None
    # No result file exists. The example inventory went 16 -> 17 at **K6**
    # and the benchmark inventory 9 -> 10 at **K8**, each artifact named in
    # its own module; K3's own delta to both is still zero.
    assert len(list((REPO_ROOT / "benchmarks").glob("*.py"))) == 10
    assert len(list((REPO_ROOT / "examples").glob("*.py"))) == 17
    # ...and the control: a token that *is* in the unit is found, so the
    # scan is reading the files it claims to.
    unit = (REPO_ROOT / "cpp" / "src" / "indexing.cpp").read_text(
        encoding="utf-8")
    assert "argmax_contiguous" in unit
    # The module's exceptional constants are what they say they are.
    assert math.isnan(NAN) and math.isinf(INF) and INF > 0
