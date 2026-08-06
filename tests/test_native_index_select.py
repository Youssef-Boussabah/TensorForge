"""Phase K, milestone K4 — native ``index_select``.

K4 is the milestone at which Phase K ships its one index-**consuming**
operation and its second (and final) C ABI symbol. This module owns the
split ``docs/native_integer_tensors_design.md`` §30.1 assigns it:
everything about ``NativeTensor.index_select`` /
``NativeTensorCore.index_select`` and the ``tf_core_index_select`` export
behind them.

Six claims, and they only mean something together:

1. **The selection is exact, and it copies.** The output's *j*-th slice
   along the selected axis is the source's slice at ``indices[j]``, for
   every *j* — duplicates preserved, order preserved, nothing sorted,
   deduplicated, wrapped, or clamped. The result is a fresh owning
   contiguous tensor at offset 0 and is **never** a view.
2. **Floating values cross bit for bit.** Every comparison of a selected
   value here is a raw IEEE-754 bit-pattern comparison through a
   ``uint32``/``uint64`` view — never ``allclose``, ``approx``, or ``==``
   — because the operation copies object representations and must
   therefore preserve both signed zeros, both infinities, subnormals, and
   every NaN payload. ``==`` can see none of that.
3. **The index operand is a role, not a second input form.** It must be
   an ``int64`` ``NativeTensor`` of rank exactly 1. A NumPy array, a
   list, a tuple, a Python ``int``, and a floating tensor are all
   rejected; a caller with host indices goes through
   ``NativeTensor.from_int64_array``. Negative indices are **rejected**,
   never normalized.
4. **Nothing is allocated before the complete bounds scan succeeds.** The
   scan is complete rather than incremental, so a rejection writes
   nothing and allocates nothing — proved by watching the allocator, not
   by reading the code.
5. **It is forward only, and it says so.** A source with
   ``requires_grad=True`` is rejected with a message naming ``detach()``
   rather than silently detached, ``"index_select"`` never joins
   ``AUTOGRAD_OPS``, and the result is a plain leaf.
6. **The C ABI is a second authority.** ``tf_core_index_select`` is driven
   directly through ``ctypes``, with the Python layer bypassed entirely,
   to prove it rejects every role and layout error on its own — and that
   its own index scan is complete, rejecting a bad index that follows
   several valid ones without writing a single destination element.

Discipline this module inherits (integer design §29.6, §30.2):

* **Exact equality only** for integers, and **raw IEEE-754 bits** for
  floating values. No tolerance is used anywhere in this file.
* **Every rejection is followed by a complete before/after fingerprint of
  the observable world**, and the fingerprint has its own non-vacuity
  control proving each component can notice the change it exists for.
* **Every injected failure position is a distinct injection**, each with a
  control proving it can fire, and each followed by a live-storage
  baseline check. The tracker installs itself **outside** ``monkeypatch``,
  so a mid-test ``undo()`` cannot silently disarm it.
* **Abandonment is proved by explicit ``close()``.** No assertion here
  depends on garbage-collection timing; the injection tests retain their
  objects strongly and assert they were closed while still referenced.
* **Source scans read code, not prose** — docstrings and string literals
  are stripped through the AST first. Every scanner has a negative
  control.
* No test starts a thread, touches the network, needs a Git ancestor, or
  depends on a total suite count.
"""
import ast
import contextlib
import ctypes
import inspect
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
K4_EXPORT = "tf_core_index_select"
K4_EXPORT_COUNT = 56          # the Phase-K maximum (design §22.3, §33)
K4_CTEST_COUNT = 27
K4_CHECKED_KERNELS = 38
EXPERIMENTAL_EXPORTS = 25
EXAMPLE_COUNT = 16
BENCHMARK_COUNT = 9

# The raw-bit view each floating dtype is compared through. Never a
# tolerance, and each dtype is only ever compared against itself.
BIT_VIEW = {"float64": np.uint64, "float32": np.uint32}

# The operations K4 did **not** ship, at any layer. ``max`` is permanent
# (§17.10); the rest are no milestone's in this phase (§18.1, §35).
ABSENT_OPERATIONS = ("max", "amax", "max_with_indices", "argmin",
                     "gather", "scatter", "scatter_add", "embedding",
                     "take", "topk", "sort", "argsort", "nonzero", "where",
                     "bincount", "cumsum", "index_put", "index_add",
                     "masked_select", "index_select_backward")


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


@contextlib.contextmanager
def no_native_allocation():
    """Assert that the block allocates **no** native storage at all.

    Stronger than ``live_storage_baseline``, and it is the instrument
    §13.11 needs: "nothing is allocated before the bounds scan succeeds" is
    not the same claim as "whatever was allocated was released"."""
    allocated = []
    original_init = cpp.NativeStorage.__init__

    def counting_init(self, *args, **kwargs):
        allocated.append(True)
        original_init(self, *args, **kwargs)

    cpp.NativeStorage.__init__ = counting_init
    try:
        yield allocated
        assert not allocated, (
            f"{len(allocated)} native storages were allocated by a call that "
            f"must allocate nothing"
        )
    finally:
        cpp.NativeStorage.__init__ = original_init


# ---------------------------------------------------------------------------
# The observable-world fingerprint.
# ---------------------------------------------------------------------------

class World:
    """A snapshot of everything a rejected or failed ``index_select`` must
    leave alone. Proportional to this operation's boundary: **both**
    operands it was handed, a parameter and its graph state, the registries
    it must not move, and both global RNGs."""

    def __init__(self, source, indices, parameter, module, optimizer,
                 generator):
        self.source = source
        self.indices = indices
        self.parameter = parameter
        self.module = module
        self.optimizer = optimizer
        self.generator = generator

    @staticmethod
    def _tensor(tensor):
        # A closed tensor has no readable metadata, so the closed flag is
        # read first and the rest is skipped: a fingerprint that *raises*
        # after a close would be an instrument that cannot report the very
        # change it exists to notice.
        closed = tensor.closed
        if closed:
            return (id(tensor), True, None, None, None)
        return (
            id(tensor), False,
            tensor.to_numpy().tobytes(),
            (tensor.shape, tensor.strides, tensor.dtype, tensor.device,
             tensor.contiguous, tensor.requires_grad, tensor.is_leaf),
            None if tensor.grad is None else tensor.grad.to_numpy().tobytes(),
        )

    def fingerprint(self):
        parameter = self.parameter
        return (
            # both operands: identity, bits, layout, graph state
            self._tensor(self.source),
            self._tensor(self.indices),
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
            # The global NumPy RNG, which nothing here may reseed or
            # consume. Both the key's first word **and** the draw position
            # are read: a reseed moves the first, a draw moves the second,
            # and reading only one would leave the component unable to
            # notice half of what it exists for — and unable to notice a
            # reseed to the value some earlier test already seeded.
            np.random.get_state()[1][0], np.random.get_state()[2],
        )


def _build_world(dtype="float64"):
    source = NativeTensor.from_array(
        np.array([[1.0, 7.0, 3.0], [9.0, 9.0, 2.0]]), dtype=dtype)
    indices = NativeTensor.from_int64_array(np.array([1, 0, 1], dtype=np.int64))
    parameter = NativeParameter(np.array([[1.5, -2.5], [0.25, 4.0]]))
    module = NativeModule()
    module.weight = parameter
    generator = NativeGenerator(seed=11)
    module.register_generator("rng", generator)
    optimizer = NativeSGD([parameter], lr=0.1)
    return World(source, indices, parameter, module, optimizer, generator)


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
        world.indices.close()
        world.source.close()
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


def _bits(array):
    """One array's raw IEEE-754 bit patterns, as unsigned integers.

    The **only** way this module compares a floating value. ``==`` calls
    two NaNs unequal, calls ``+0.0`` and ``-0.0`` equal, and sees no
    difference between two NaN payloads, so it can prove none of what a
    bit-preserving copy promises (design §29.6)."""
    array = np.ascontiguousarray(array)
    return array.view(BIT_VIEW[str(array.dtype)])


def _same_bits(left, right):
    return (left.shape == right.shape
            and np.array_equal(_bits(left), _bits(right)))


def _floating(values, dtype):
    return NativeTensor.from_array(np.asarray(values, dtype=np.float64),
                                   dtype=dtype)


def _index(values):
    return NativeTensor.from_int64_array(np.asarray(values, dtype=np.int64))


def _numpy_selection(values, axis, indices):
    """The reference selection, computed on the host with plain NumPy
    fancy indexing — a *layout* reference, never a value one: the values it
    moves are the very bytes the native call must reproduce, and the
    comparison below is on bit patterns."""
    return np.take(values, np.asarray(indices, dtype=np.int64), axis=axis)


@contextlib.contextmanager
def _selected(source, axis, indices):
    """``source.index_select(axis, indices)``, closed on the way out."""
    result = source.index_select(axis, indices)
    try:
        yield result
    finally:
        result.close()


def _strided_index_view(base):
    """A genuinely **non-contiguous** rank-1 ``int64`` view over ``base``'s
    storage, taking every second element.

    Built through the public view constructor at the Core layer rather than
    through ``reshape``/``transpose``/``narrow``, because none of those can
    produce one: ``reshape`` refuses a non-contiguous input, ``transpose``
    is the identity at rank 1, and ``narrow`` preserves its parent's
    strides. That is a property of the shipped view set, not of this
    operation — and ``index_select`` must accept such a view anyway, which
    is exactly why it is constructed here rather than assumed unreachable.

    The view **borrows**: closing the returned tensor leaves ``base``
    open."""
    core = base._require_open()
    storage = core.storage
    length = core.numel // 2
    view = cpp.NativeTensorView(storage, (length,), strides=(2,), offset=0)
    strided_core = cpp.NativeTensorCore(storage, view, owns_storage=False)
    return NativeTensor._from_core(strided_core)


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
        world.indices.close()
        world.source.close()
        world.parameter.close()

    # 2. a gradient appearing on the parameter
    world = _build_world()
    before = world.fingerprint()
    try:
        grad = NativeTensor.from_array(np.ones((2, 2)))
        world.parameter._accumulate_grad(grad)
        assert world.fingerprint() != before
    finally:
        world.indices.close()
        world.source.close()
        world.parameter.close()

    # 3. the source closing...
    world = _build_world()
    before = world.fingerprint()
    try:
        world.source.close()
        assert world.fingerprint() != before
    finally:
        world.indices.close()
        world.parameter.close()

    # 4. ...and the **index operand** closing, which is a separate component
    world = _build_world()
    before = world.fingerprint()
    try:
        world.indices.close()
        assert world.fingerprint() != before
    finally:
        world.source.close()
        world.parameter.close()

    # 5. the generator's committed-call counter advancing
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
        world.indices.close()
        world.source.close()
        world.parameter.close()

    # 6. a registry moving (simulated on a copy, never on the real module)
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
        world.indices.close()
        world.source.close()
        world.parameter.close()

    # 7. the global NumPy RNG being consumed, and separately reseeded.
    # Consuming is the collision-free half: one draw advances the position
    # unconditionally, whereas a reseed to a value some earlier test already
    # used would leave the key's first word exactly where it was.
    world = _build_world()
    before = world.fingerprint()
    try:
        np.random.random()
        assert world.fingerprint() != before
        np.random.seed(20240630)
        assert world.fingerprint() != before
    finally:
        world.indices.close()
        world.source.close()
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


@needs_native
def test_the_no_allocation_probe_can_actually_fail():
    """Non-vacuity for the stronger instrument: an allocation that is
    immediately closed still trips it, which is what makes "nothing was
    allocated before the scan" a real claim rather than a restatement of
    the baseline check."""
    with pytest.raises(AssertionError, match="must allocate nothing"):
        with no_native_allocation():
            probe = NativeTensor.from_array(np.array([1.0]))
            probe.close()
    with no_native_allocation():
        pass


def test_the_bit_comparison_can_actually_fail():
    """Negative control for the only comparison this module uses: it must
    tell apart exactly the pairs ``==`` cannot."""
    for dtype in FLOATING_DTYPES:
        zero = np.array([0.0], dtype=dtype)
        negative_zero = np.array([-0.0], dtype=dtype)
        assert zero[0] == negative_zero[0]           # what == says
        assert not _same_bits(zero, negative_zero)   # what the bits say

        quiet = np.array([np.nan], dtype=dtype)
        assert not (quiet[0] == quiet[0])            # what == says
        assert _same_bits(quiet, quiet.copy())       # what the bits say

        # ...and two NaNs with different payloads are distinguished.
        raw = BIT_VIEW[dtype]
        one = np.array([np.nan], dtype=dtype)
        other = one.copy()
        other.view(raw)[0] = one.view(raw)[0] ^ raw(1)
        assert not _same_bits(one, other)
        # A genuine difference in an ordinary value is caught too.
        assert not _same_bits(np.array([1.0], dtype=dtype),
                              np.array([2.0], dtype=dtype))
        # ...and a shape difference is not silently broadcast away.
        assert not _same_bits(np.array([1.0], dtype=dtype),
                              np.array([1.0, 1.0], dtype=dtype))


def test_the_code_only_reader_ignores_prose():
    """Negative control for the absence scanner, on a temporary string."""
    names = _code_only('"""a docstring naming gather and scatter."""\n'
                       'def f(x):\n    return g(x, axis=0)\n')
    assert "gather" not in names and "scatter" not in names
    assert "axis" in names and "g" in names


def test_the_export_scanner_can_actually_fail():
    """Negative control, on a temporary string: an export really is found,
    so the inventory below is a measurement rather than an artifact."""
    found = re.findall(r"TF_EXPORT[^;{]*?\b(tf_[a-z0-9_]+)\s*\(",
                       "TF_EXPORT void tf_core_probe(const void* a) { x(a); }",
                       re.S)
    assert found == ["tf_core_probe"]
    assert re.findall(r"TF_EXPORT[^;{]*?\b(tf_[a-z0-9_]+)\s*\(",
                      "void tf_core_probe(void);", re.S) == []


@needs_native
def test_the_strided_index_helper_really_builds_a_non_contiguous_view():
    """Non-vacuity for every layout test below: the helper's precondition
    is asserted rather than assumed, and the view borrows."""
    base = _index([0, 9, 1, 9, 2, 9])
    strided = _strided_index_view(base)
    try:
        assert strided.ndim == 1
        assert strided.shape == (3,)
        assert strided.contiguous is False, "the helper produced a contiguous view"
        assert strided.dtype == INDEX_DTYPE
        assert strided.tolist() == [0, 1, 2]
    finally:
        strided.close()
    # The view borrowed: the base is still open and still complete.
    assert base.tolist() == [0, 9, 1, 9, 2, 9]
    base.close()


# ===========================================================================
# 1. The public API and the inventories
# ===========================================================================

def test_both_layers_gained_index_select_and_only_index_select():
    assert hasattr(NativeTensor, "index_select")
    assert hasattr(cpp.NativeTensorCore, "index_select")
    assert not hasattr(cpp.NativeStorage, "index_select")
    for owner in (NativeTensor, cpp.NativeTensorCore, cpp.NativeStorage):
        for absent in ABSENT_OPERATIONS:
            assert not hasattr(owner, absent), (owner.__name__, absent)


def test_the_signature_is_axis_then_indices_at_both_layers():
    """Two positional arguments, in that order, with no defaults and no
    keyword-only restriction — the repository's ordinary spelling."""
    for owner in (NativeTensor, cpp.NativeTensorCore):
        signature = inspect.signature(owner.index_select)
        assert list(signature.parameters) == ["self", "axis", "indices"]
        for name in ("axis", "indices"):
            parameter = signature.parameters[name]
            assert parameter.default is inspect.Parameter.empty, name
            assert parameter.kind is inspect.Parameter.POSITIONAL_OR_KEYWORD


def test_index_select_joined_exactly_one_inventory():
    assert cpp.TENSOR_CORE_OPS.count("index_select") == 1
    assert "index_select" not in cpp.AUTOGRAD_OPS
    assert "index_select" not in cpp.TENSOR_CORE_KERNELS
    assert "index_select" not in cpp.RAW_KERNELS
    assert "index_select" not in cpp.NATIVE_METRICS
    assert "index_select" not in cpp.NATIVE_MODULES
    assert "index_select" not in cpp.NATIVE_LOSSES
    assert "index_select" not in cpp.NATIVE_OPTIMIZERS
    assert "index_select" not in cpp.STATE_SUPPORT
    # The deliberately frozen historical registry is exactly what it was.
    assert cpp.TENSOR_CORE_KERNELS == (
        "relu", "add", "subtract", "multiply", "matmul")
    assert cpp.RAW_KERNEL_DTYPES == ("float64",)
    assert cpp.NATIVE_METRICS == ("native_accuracy",)
    assert cpp.backend_info()["tensor_core_ops"] == cpp.TENSOR_CORE_OPS
    assert cpp.backend_info()["autograd_ops"] == cpp.AUTOGRAD_OPS
    # K3's entry is untouched beside it.
    assert cpp.TENSOR_CORE_OPS.count("argmax") == 1
    assert "argmax" not in cpp.AUTOGRAD_OPS


def test_the_source_export_inventory_is_fifty_six():
    exports = _source_exports()
    assert len(exports) == K4_EXPORT_COUNT, sorted(exports)
    assert K4_EXPORT in exports
    assert "tf_core_argmax" in exports
    for absent in ("tf_core_index_select_backward", "tf_core_gather",
                   "tf_core_scatter", "tf_core_scatter_add",
                   "tf_core_embedding", "tf_core_max", "tf_core_argmin",
                   "tf_storage_dtype"):
        assert absent not in exports, absent


@needs_native
def test_the_built_library_exports_the_same_inventory():
    """Source and built library must agree, which is also the stale-artifact
    guard: a library built before K4 exports 55 and fails here rather than
    silently satisfying the tests that call the new symbol."""
    storage_tests = pytest.importorskip("test_native_storage_allocation")
    _, names = storage_tests.exported_names(cpp._LIBRARY_PATH)
    if names is None:
        pytest.skip("this image format is not parsed here")
    exported = sorted(name for name in names if name.startswith("tf_"))
    assert len(exported) == K4_EXPORT_COUNT, exported
    assert set(exported) == _source_exports()
    assert K4_EXPORT in exported


@needs_native
def test_the_export_is_declared_and_carries_the_error_hook():
    library = cpp._require_library()
    function = getattr(library, K4_EXPORT)
    assert function.argtypes == [
        ctypes.c_void_p, ctypes.c_int64,
        ctypes.c_void_p, ctypes.c_int64,
        ctypes.c_void_p,
        ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64,
    ]
    assert function.restype is None
    assert K4_EXPORT in cpp._CHECKED_KERNELS
    assert cpp._CHECKED_KERNELS.count(K4_EXPORT) == 1
    assert len(cpp._CHECKED_KERNELS) == K4_CHECKED_KERNELS
    assert function.errcheck is not None
    # No array position: three storage handles and six int64 scalars, so
    # neither array binding is involved.
    assert not [argtype for argtype in function.argtypes
                if argtype is cpp._LAYOUT_POINTER
                or getattr(argtype, "__name__", "").startswith("ndpointer")]


def test_the_ctest_inventory_moved_by_exactly_one():
    cmake = (REPO_ROOT / "cpp" / "CMakeLists.txt").read_text(encoding="utf-8")
    registered = re.findall(r"add_test\s*\(\s*NAME\s+(\w+)", cmake)
    assert len(registered) == K4_CTEST_COUNT, registered
    assert len(set(registered)) == len(registered)
    assert "index_select" in registered
    assert "argmax" in registered
    sources = {path.stem for path in
               (REPO_ROOT / "cpp" / "tests").glob("test_*.cpp")}
    assert sources == {f"test_{name}" for name in registered}
    assert (REPO_ROOT / "cpp" / "tests" / "test_index_select.cpp").is_file()


def test_the_indexing_unit_carries_both_exports_and_the_header_none():
    """K4 extends the K3 translation unit rather than adding a second one,
    and the internal header still declares no ABI."""
    unit = (REPO_ROOT / "cpp" / "src" / "indexing.cpp").read_text(
        encoding="utf-8")
    code = re.sub(r"//[^\n]*", " ",
                  re.sub(r"/\*.*?\*/", " ", unit, flags=re.S))
    assert code.count("TF_EXPORT") == 2
    assert "TF_EXPORT void tf_core_argmax(" in unit
    assert f"TF_EXPORT void {K4_EXPORT}(" in unit
    # No second indexing source unit appeared.
    present = sorted(path.name for path in
                     (REPO_ROOT / "cpp" / "src").glob("*.cpp"))
    assert present.count("indexing.cpp") == 1
    assert not [name for name in present
                if "index" in name and name != "indexing.cpp"]
    header = (REPO_ROOT / "cpp" / "include" / "tf_indexing_internal.h"
              ).read_text(encoding="utf-8")
    header_code = re.sub(r"//[^\n]*", " ",
                         re.sub(r"/\*.*?\*/", " ", header, flags=re.S))
    assert "TF_EXPORT" not in header_code
    assert K4_EXPORT not in header_code
    assert "tf_core_argmax" not in header_code
    # ...and both templated traversals live there, separately named.
    assert "index_select_contiguous" in header_code
    assert "argmax_contiguous" in header_code
    assert "require_index" in header_code


def test_no_public_capability_registry_or_version_moved_at_k4():
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
    assert set(cpp._DTYPE_CODES) == (set(cpp.SUPPORTED_DTYPES)
                                     | set(cpp.INDEX_DTYPES))
    assert len(experimental.__all__) == EXPERIMENTAL_EXPORTS
    assert len(set(experimental.__all__)) == EXPERIMENTAL_EXPORTS
    assert native_checkpoint._FORMAT_VERSION == 3
    assert tuple(sorted(native_checkpoint._SUPPORTED_FORMAT_VERSIONS)) == (1, 2, 3)
    assert native_optimizer_state.FORMAT_VERSION == 1
    assert native_data_loader._FORMAT_VERSION == 1
    assert native_sampler._FORMAT_VERSION == 1
    assert len(list((REPO_ROOT / "examples").glob("*.py"))) == EXAMPLE_COUNT
    assert len(list((REPO_ROOT / "benchmarks").glob("*.py"))) == BENCHMARK_COUNT


# ===========================================================================
# 2. Exact selection — the values, at each dtype separately
# ===========================================================================

@needs_native
@pytest.mark.parametrize("dtype", FLOATING_DTYPES)
@pytest.mark.parametrize("shape,axis,indices", [
    ((5,), 0, [0]),
    ((5,), 0, [4, 3, 2, 1, 0]),
    ((5,), 0, [2, 2, 2]),
    ((5,), -1, [1, 3]),
    ((3, 4), 0, [2, 0, 2]),
    ((3, 4), 1, [3, 3, 0, 1]),
    ((3, 4), -1, [0]),
    ((3, 4), -2, [1, 1]),
    ((2, 3, 4), 0, [1, 0]),
    ((2, 3, 4), 1, [2, 0, 2, 1]),
    ((2, 3, 4), 2, [3, 3]),
    ((2, 3, 4), -3, [0, 0, 1]),
    ((2, 3, 2, 3), 1, [2, 1, 0]),
    ((2, 3, 2, 3), 2, [1, 1, 0]),
    ((2, 3, 2, 3), 3, [2, 0]),
    ((2, 3, 2, 3), -4, [1, 0, 1]),
])
def test_the_selection_matches_the_host_reference_bit_for_bit(
        dtype, shape, axis, indices):
    """Every axis, every rank 1 through 4, negative axes, one index,
    several, duplicates, and reverse order — with the values compared as
    raw bit patterns rather than by any tolerance."""
    count = int(np.prod(shape))
    values = (np.arange(count, dtype=np.float64) * 0.5 - 3.25).reshape(shape)
    values = values.astype(dtype)
    source = NativeTensor.from_array(values, dtype=dtype)
    index_tensor = _index(indices)
    try:
        with _selected(source, axis, index_tensor) as result:
            expected = _numpy_selection(values, axis, indices)
            assert result.shape == expected.shape
            assert result.dtype == dtype
            assert _same_bits(result.to_numpy(), expected)
    finally:
        index_tensor.close()
        source.close()


@needs_native
@pytest.mark.parametrize("dtype", FLOATING_DTYPES)
def test_exceptional_values_are_copied_bit_for_bit(dtype):
    """Signed zeros, both infinities, subnormals, and distinct NaN payloads
    all survive the copy exactly — the claim the ``memcpy`` slice copy
    exists for, and one that no tolerance and no ``==`` could state."""
    raw = BIT_VIEW[dtype]
    values = np.array([0.0, -0.0, np.inf, -np.inf, 1.0, -1.0],
                      dtype=dtype)
    # A subnormal and two NaNs with deliberately different payloads, built
    # from bit patterns so nothing about them is inherited.
    values.view(raw)[4] = raw(3)                      # a subnormal
    values.view(raw)[5] = np.array([np.nan], dtype=dtype).view(raw)[0]
    other = np.array([np.nan], dtype=dtype).view(raw)[0] ^ raw(0x0BAD)
    values = np.concatenate([values, np.array([0.0], dtype=dtype)])
    values.view(raw)[6] = other

    source = NativeTensor.from_array(values, dtype=dtype)
    # Every position, twice, so a duplicate is proved to copy independently.
    order = [k for k in range(values.size) for _ in range(2)]
    index_tensor = _index(order)
    try:
        with _selected(source, 0, index_tensor) as result:
            assert _same_bits(result.to_numpy(), values[order])
            # ...and the *source* is untouched, bit for bit.
            assert _same_bits(source.to_numpy(), values)
    finally:
        index_tensor.close()
        source.close()


@needs_native
@pytest.mark.parametrize("dtype", FLOATING_DTYPES)
def test_duplicates_and_order_are_preserved_exactly(dtype):
    """The contract §13.5 states, driven literally: nothing is sorted,
    deduplicated, normalized, wrapped, or clamped."""
    values = np.array([[10.0, 11.0], [20.0, 21.0], [30.0, 31.0]], dtype=dtype)
    source = NativeTensor.from_array(values, dtype=dtype)
    index_tensor = _index([2, 0, 2, 1])
    try:
        with _selected(source, 0, index_tensor) as result:
            expected = np.array([[30.0, 31.0], [10.0, 11.0],
                                 [30.0, 31.0], [20.0, 21.0]], dtype=dtype)
            assert _same_bits(result.to_numpy(), expected)
            # The two copies of row 2 are independent storage, not a shared
            # slice: mutating one in the host copy cannot reach the other,
            # and the native result has one contiguous allocation.
            assert result.contiguous is True
            assert result.shape == (4, 2)
    finally:
        index_tensor.close()
        source.close()
    # ...and a sorted or deduplicated implementation would have produced a
    # different answer, which the reference makes explicit.
    assert not np.array_equal(
        _numpy_selection(values, 0, [2, 0, 2, 1]),
        _numpy_selection(values, 0, sorted(set([2, 0, 2, 1]))))


@needs_native
@pytest.mark.parametrize("dtype", FLOATING_DTYPES)
def test_an_argmax_result_is_a_valid_index_operand(dtype):
    """The two Phase-K operations compose directly, which is the reason
    ``index_select`` rather than ``gather`` was the primitive chosen
    (§18.10): ``argmax``'s output is exactly a rank-1 ``int64`` tensor of
    non-negative values, so it never meets the negative-index rule."""
    logits = np.array([[1.0, 7.0, 3.0], [9.0, 2.0, 2.0], [0.5, 0.5, 4.0]],
                      dtype=dtype)
    source = NativeTensor.from_array(logits, dtype=dtype)
    predicted = source.argmax(axis=1)
    try:
        assert predicted.dtype == INDEX_DTYPE
        assert predicted.ndim == 1
        assert predicted.tolist() == [1, 0, 2]
        with _selected(source, 1, predicted) as result:
            expected = _numpy_selection(logits, 1, [1, 0, 2])
            assert _same_bits(result.to_numpy(), expected)
    finally:
        predicted.close()
        source.close()


# ===========================================================================
# 3. Layout — non-contiguous and offset operands
# ===========================================================================

@needs_native
@pytest.mark.parametrize("dtype", FLOATING_DTYPES)
@pytest.mark.parametrize("axis,indices", [
    (0, [3, 0, 3]),
    (1, [1, 0]),
    (2, [2, 2, 0]),
    (-1, [0, 2]),
    (-3, [1, 3, 1]),
])
def test_a_transposed_source_selects_from_its_logical_layout(
        dtype, axis, indices):
    """The Policy-B arm for the *source*: a transposed view's answer is the
    selection from its **logical** materialization, at every axis."""
    values = (np.arange(24.0) * 0.25 - 1.0).reshape(2, 3, 4).astype(dtype)
    base = NativeTensor.from_array(values, dtype=dtype)
    view = base.transpose(2, 0, 1)       # shape (4, 2, 3), non-contiguous
    index_tensor = _index(indices)
    try:
        assert view.contiguous is False, "the test's precondition is gone"
        logical = np.transpose(values, (2, 0, 1))
        assert view.shape == logical.shape
        with _selected(view, axis, index_tensor) as result:
            assert _same_bits(result.to_numpy(),
                              _numpy_selection(logical, axis, indices))
    finally:
        index_tensor.close()
        view.close()
        base.close()


@needs_native
@pytest.mark.parametrize("dtype", FLOATING_DTYPES)
def test_a_narrowed_offset_source_selects_from_its_own_values(dtype):
    values = (np.arange(20.0) - 7.5).reshape(4, 5).astype(dtype)
    base = NativeTensor.from_array(values, dtype=dtype)
    view = base.narrow(0, 1, 2)          # contiguous, non-zero offset
    inner_indices = _index([4, 0, 4])
    outer_indices = _index([1, 1, 0])
    try:
        assert view._require_open().offset != 0, \
            "the test's precondition is gone"
        assert view.contiguous is True
        # The offset crosses the ABI directly (no Policy-B copy is made for
        # a contiguous operand), so it is exercised on the axis where
        # ``inner > 1`` **and** on the axis where ``outer > 1`` is trivial —
        # the two places the plane arithmetic could drop it.
        with _selected(view, 1, inner_indices) as result:
            assert _same_bits(result.to_numpy(),
                              _numpy_selection(values[1:3], 1, [4, 0, 4]))
        with _selected(view, 0, outer_indices) as result:
            assert _same_bits(result.to_numpy(),
                              _numpy_selection(values[1:3], 0, [1, 1, 0]))
    finally:
        outer_indices.close()
        inner_indices.close()
        view.close()
        base.close()


@needs_native
@pytest.mark.parametrize("dtype", FLOATING_DTYPES)
def test_a_chained_non_contiguous_source_selects_correctly(dtype):
    values = (np.arange(24.0) * -0.125).reshape(2, 3, 4).astype(dtype)
    base = NativeTensor.from_array(values, dtype=dtype)
    transposed = base.transpose(1, 2, 0)
    chained = transposed.narrow(1, 1, 2)
    index_tensor = _index([2, 0])
    try:
        assert chained.contiguous is False, "the test's precondition is gone"
        logical = np.transpose(values, (1, 2, 0))[:, 1:3, :]
        with _selected(chained, 0, index_tensor) as result:
            assert _same_bits(result.to_numpy(),
                              _numpy_selection(logical, 0, [2, 0]))
    finally:
        index_tensor.close()
        chained.close()
        transposed.close()
        base.close()


@needs_native
@pytest.mark.parametrize("dtype", FLOATING_DTYPES)
def test_an_offset_index_view_is_read_in_its_own_logical_order(dtype):
    values = (np.arange(12.0) + 0.5).reshape(3, 4).astype(dtype)
    source = NativeTensor.from_array(values, dtype=dtype)
    base = _index([9, 9, 2, 0, 2])
    narrowed = base.narrow(0, 2, 3)      # contiguous, non-zero offset
    try:
        assert narrowed._require_open().offset != 0, \
            "the test's precondition is gone"
        assert narrowed.tolist() == [2, 0, 2]
        with _selected(source, 0, narrowed) as result:
            assert _same_bits(result.to_numpy(),
                              _numpy_selection(values, 0, [2, 0, 2]))
    finally:
        narrowed.close()
        base.close()
        source.close()


@needs_native
@pytest.mark.parametrize("dtype", FLOATING_DTYPES)
def test_a_non_contiguous_index_view_is_read_in_its_own_logical_order(dtype):
    """The Policy-B arm for the *index* operand, which is a separate
    materialization from the source's and is proved separately."""
    values = (np.arange(12.0) - 5.0).reshape(3, 4).astype(dtype)
    source = NativeTensor.from_array(values, dtype=dtype)
    base = _index([3, 9, 0, 9, 3, 9])
    strided = _strided_index_view(base)
    try:
        assert strided.contiguous is False, "the test's precondition is gone"
        assert strided.tolist() == [3, 0, 3]
        with _selected(source, 1, strided) as result:
            assert _same_bits(result.to_numpy(),
                              _numpy_selection(values, 1, [3, 0, 3]))
    finally:
        strided.close()
        base.close()
        source.close()


@needs_native
def test_both_operands_non_contiguous_at_once():
    """Two Policy-B temporaries in one call, both closed."""
    values = (np.arange(12.0) * 1.5).reshape(3, 4)
    base = NativeTensor.from_array(values)
    view = base.T
    index_base = _index([2, 7, 0, 7])
    strided = _strided_index_view(index_base)
    try:
        assert view.contiguous is False and strided.contiguous is False
        with live_storage_baseline():
            with _selected(view, 1, strided) as result:
                assert _same_bits(result.to_numpy(),
                                  _numpy_selection(values.T, 1, [2, 0]))
    finally:
        strided.close()
        index_base.close()
        view.close()
        base.close()


@needs_native
def test_a_contiguous_operand_is_not_copied_and_a_strided_one_is():
    """Policy B engages exactly when it must: the contiguous path performs
    no ``contiguous_copy`` at all, and each strided operand causes exactly
    one."""
    values = np.arange(12.0).reshape(3, 4)
    base = NativeTensor.from_array(values)
    view = base.T
    index_base = _index([1, 5, 0, 5])
    strided = _strided_index_view(index_base)
    contiguous_index = _index([1, 0])
    calls = []
    original = cpp.NativeTensorCore.contiguous_copy

    def counting(self):
        calls.append(self.dtype)
        return original(self)

    cpp.NativeTensorCore.contiguous_copy = counting
    try:
        with _selected(base, 0, contiguous_index):
            pass
        assert calls == [], calls
        with _selected(view, 0, contiguous_index):
            pass
        assert calls == ["float64"], calls
        calls.clear()
        with _selected(base, 0, strided):
            pass
        assert calls == [INDEX_DTYPE], calls
        calls.clear()
        with _selected(view, 0, strided):
            pass
        assert calls == ["float64", INDEX_DTYPE], calls
    finally:
        cpp.NativeTensorCore.contiguous_copy = original
        contiguous_index.close()
        strided.close()
        index_base.close()
        view.close()
        base.close()


# ===========================================================================
# 4. Shape and ownership
# ===========================================================================

@needs_native
@pytest.mark.parametrize("dtype", FLOATING_DTYPES)
def test_the_result_is_fresh_owning_contiguous_storage(dtype):
    values = (np.arange(12.0) + 1.0).reshape(3, 4).astype(dtype)
    source = NativeTensor.from_array(values, dtype=dtype)
    index_tensor = _index([2, 0])
    result = source.index_select(0, index_tensor)
    try:
        assert result.shape == (2, 4)
        assert result.dtype == dtype
        assert result.device == "cpu"
        assert result.contiguous is True
        assert result.owns_core is True
        assert result._require_open().offset == 0
        # It shares storage with neither operand.
        assert result._require_open().storage is not source._require_open().storage
        assert result._require_open().storage is not \
            index_tensor._require_open().storage
        expected = _numpy_selection(values, 0, [2, 0])
        # ...and it survives both operands closing.
        index_tensor.close()
        source.close()
        assert _same_bits(result.to_numpy(), expected)
        assert result.shape == (2, 4)
    finally:
        result.close()
        if not index_tensor.closed:
            index_tensor.close()
        if not source.closed:
            source.close()


@needs_native
def test_the_output_shape_replaces_exactly_one_axis():
    values = np.arange(2 * 3 * 4 * 5, dtype=np.float64).reshape(2, 3, 4, 5)
    source = NativeTensor.from_array(values)
    index_tensor = _index([1, 1, 0, 2, 1, 0, 1])
    try:
        for axis, expected in (
            (1, (2, 7, 4, 5)),
            (-3, (2, 7, 4, 5)),
        ):
            with _selected(source, axis, index_tensor) as result:
                assert result.shape == expected
        # Only the selected axis changes size; every other extent is the
        # source's, in order.
        with _selected(source, 2, _index([0])) as result:
            assert result.shape == (2, 3, 1, 5)
    finally:
        index_tensor.close()
        source.close()


@needs_native
def test_a_successful_selection_returns_live_storage_to_baseline():
    """The control for all the injections below: the ordinary path
    allocates and releases exactly what it should, at every layout."""
    for dtype in FLOATING_DTYPES:
        base = NativeTensor.from_array(np.arange(12.0).reshape(3, 4),
                                       dtype=dtype)
        view = base.T
        index_base = _index([1, 8, 0, 8])
        strided = _strided_index_view(index_base)
        contiguous = _index([1, 0, 1])
        with live_storage_baseline():
            for source in (base, view):
                for indices in (contiguous, strided):
                    for axis in (0, 1, -1):
                        result = source.index_select(axis, indices)
                        result.close()
        contiguous.close()
        strided.close()
        index_base.close()
        view.close()
        base.close()


# ===========================================================================
# 5. Validation and error precedence
# ===========================================================================

@needs_native
@pytest.mark.parametrize("axis", ["0", 0.0, None, (0,), [0], True, False,
                                  np.float64(0.0), object()])
def test_a_non_integer_axis_is_a_type_error(axis):
    with unchanged_world() as world:
        with no_native_allocation():
            with pytest.raises(TypeError, match="axis"):
                world.source.index_select(axis, world.indices)


@needs_native
@pytest.mark.parametrize("axis", [0, -1, np.int64(0), np.int32(-1)])
def test_the_canonical_integer_axis_domain_is_inherited(axis):
    """K4 asks the same predicate the reductions ask, so a NumPy integer
    scalar is accepted exactly as it always was and a negative axis
    normalizes the same way."""
    source = _floating([[1.0, 2.0], [3.0, 4.0]], "float64")
    index_tensor = _index([1, 0])
    try:
        with _selected(source, axis, index_tensor) as result:
            assert result.shape == (2, 2)
    finally:
        index_tensor.close()
        source.close()
    # ...and the two axis helpers agree about the domain, so the split into
    # a type half and a range half cannot drift.
    assert cpp._is_axis_int(axis)
    assert cpp._require_axis_int(axis, "probe") is axis
    for rejected in (True, "0", 1.0, None):
        assert not cpp._is_axis_int(rejected)
        with pytest.raises(TypeError):
            cpp._require_axis_int(rejected, "probe")
        with pytest.raises(TypeError):
            cpp._normalize_axis_checked(rejected, (2,))


@needs_native
@pytest.mark.parametrize("indices", [
    np.array([0, 1], dtype=np.int64), [0, 1], (0, 1), 0, np.int64(0), None,
    "01",
])
def test_a_non_native_index_operand_is_a_type_error(indices):
    """There is exactly one index input form, and a host array is not it."""
    with unchanged_world() as world:
        with no_native_allocation():
            with pytest.raises(TypeError, match="NativeTensor"):
                world.source.index_select(0, indices)


@needs_native
def test_a_closed_source_or_index_rejects_before_anything_else():
    source = _floating([[1.0, 2.0], [3.0, 4.0]], "float64")
    indices = _index([1, 0])
    source.close()
    try:
        with no_native_allocation():
            with pytest.raises(RuntimeError, match="closed"):
                source.index_select(0, indices)
    finally:
        indices.close()

    source = _floating([[1.0, 2.0], [3.0, 4.0]], "float64")
    indices = _index([1, 0])
    indices.close()
    try:
        with no_native_allocation():
            with pytest.raises(RuntimeError, match="closed"):
                source.index_select(0, indices)
    finally:
        source.close()


@needs_native
def test_an_int64_source_is_rejected_as_a_role_error():
    """The K1 barrier, re-proved on the new operation's *source* operand:
    an integer tensor cannot be selected **from**, only selected **with**."""
    integers = _index([5, 6, 7])
    indices = _index([1, 0])
    try:
        with no_native_allocation():
            with pytest.raises(ValueError, match="floating"):
                integers.index_select(0, indices)
            with pytest.raises(ValueError, match="floating"):
                integers._require_open().index_select(
                    0, indices._require_open())
    finally:
        indices.close()
        integers.close()


@needs_native
@pytest.mark.parametrize("dtype", FLOATING_DTYPES)
def test_a_floating_index_operand_is_rejected_as_a_role_error(dtype):
    source = _floating([[1.0, 2.0], [3.0, 4.0]], "float64")
    floating_indices = _floating([1.0, 0.0], dtype)
    try:
        with no_native_allocation():
            with pytest.raises(ValueError, match="index dtype"):
                source.index_select(0, floating_indices)
    finally:
        floating_indices.close()
        source.close()


@needs_native
@pytest.mark.parametrize("axis", [2, -3, 99, -99])
def test_an_out_of_range_axis_is_a_value_error(axis):
    with unchanged_world() as world:
        with no_native_allocation():
            with pytest.raises(ValueError, match="out of bounds"):
                world.source.index_select(axis, world.indices)


@needs_native
def test_every_axis_on_a_rank_zero_source_is_out_of_range():
    scalar = NativeTensor.zeros(())
    indices = _index([0])
    try:
        assert scalar.ndim == 0 and scalar.shape == ()
        with no_native_allocation():
            for axis in (0, -1, 1):
                with pytest.raises(ValueError, match="out of bounds"):
                    scalar.index_select(axis, indices)
    finally:
        indices.close()
        scalar.close()


@needs_native
@pytest.mark.parametrize("shape", [(), (2, 2), (1, 3), (2, 1, 2)])
def test_an_index_tensor_that_is_not_rank_one_is_rejected(shape):
    source = _floating(np.arange(6.0).reshape(2, 3), "float64")
    flat = _index(np.arange(int(np.prod(shape)) or 1))
    indices = flat.reshape(shape) if shape else flat.reshape(())
    try:
        with no_native_allocation():
            with pytest.raises(ValueError, match="rank-1"):
                source.index_select(0, indices)
    finally:
        indices.close()
        flat.close()
        source.close()


@needs_native
@pytest.mark.parametrize("values,position,offender", [
    ([-1], 0, -1),
    ([0, -1], 1, -1),
    ([2], 0, 2),
    ([0, 1, 2], 2, 2),
    ([0, 1, 99], 2, 99),
    ([0, -(2 ** 63)], 1, -(2 ** 63)),
    ([0, 2 ** 63 - 1], 1, 2 ** 63 - 1),
])
def test_an_out_of_range_index_names_its_value_and_its_position(
        values, position, offender):
    """Negative indices reject rather than wrap (§14.2), the boundary value
    ``axis_length`` rejects, and the report identifies the offending value
    and its zero-based logical position."""
    source = _floating([[1.0, 2.0], [3.0, 4.0]], "float64")   # axis 0 has 2
    indices = _index(values)
    try:
        with no_native_allocation():
            with pytest.raises(ValueError) as caught:
                source.index_select(0, indices)
        message = str(caught.value)
        assert str(offender) in message, message
        assert f"position {position}" in message, message
    finally:
        indices.close()
        source.close()


@needs_native
def test_the_bounds_scan_is_complete_before_any_allocation():
    """§13.11 and §14.4 together: the scan reads **every** value and
    finishes before the destination exists. The offending index is last, so
    an implementation that checked as it copied would already have written
    — and would already have allocated."""
    values = np.arange(20.0).reshape(4, 5)
    source = NativeTensor.from_array(values)
    indices = _index([0, 1, 2, 3, 4, 5])      # only the last is out of range
    try:
        with unchanged_world():
            with no_native_allocation():
                with pytest.raises(ValueError, match="position 5"):
                    source.index_select(1, indices)
        # The control: the same call with the last index repaired succeeds,
        # so the rejection above was about that value and nothing else.
        repaired = _index([0, 1, 2, 3, 4, 4])
        try:
            with _selected(source, 1, repaired) as result:
                assert _same_bits(result.to_numpy(),
                                  _numpy_selection(values, 1,
                                                   [0, 1, 2, 3, 4, 4]))
        finally:
            repaired.close()
    finally:
        indices.close()
        source.close()


@contextlib.contextmanager
def lowered_int64_max(limit):
    """Temporarily lower the module's addressable-element ceiling.

    Installed **outside** ``monkeypatch``, like the allocator instruments
    above, and restored in a ``finally`` so a failing assertion inside the
    block cannot leave the ceiling lowered for the rest of the session. The
    limit is read out of ``cpp`` inside the block by the caller, so "the
    instrument really fired" is a measurement rather than an assumption."""
    original = cpp._INT64_MAX
    cpp._INT64_MAX = limit
    try:
        yield original
    finally:
        cpp._INT64_MAX = original


@needs_native
def test_the_lowered_ceiling_instrument_can_actually_fail():
    """Non-vacuity for the two tests below, in both directions.

    The instrument must (a) really replace the ceiling inside the block,
    (b) really restore it afterwards, and (c) restore it even when the block
    raises — otherwise a "rejected because the count was too large" result
    below would prove nothing about ``index_select``."""
    real = cpp._INT64_MAX
    with lowered_int64_max(7) as reported:
        assert reported == real
        assert cpp._INT64_MAX == 7
    assert cpp._INT64_MAX == real
    with pytest.raises(ZeroDivisionError):
        with lowered_int64_max(7):
            assert cpp._INT64_MAX == 7
            1 / 0
    assert cpp._INT64_MAX == real
    # ...and the real ceiling is the signed-int64 maximum, so the check the
    # tests below reach is unreachable in ordinary use for a reason — the
    # operands would have to be astronomically large — rather than by luck.
    assert real == 2 ** 63 - 1


@needs_native
@pytest.mark.parametrize("dtype", FLOATING_DTYPES)
def test_an_unrepresentable_output_count_is_rejected_before_any_allocation(
        dtype):
    """§18.6 step 11, on the **real public path** with real small operands.

    ``index_select`` is the one operation in the runtime whose output can be
    *larger* than its source — a repeated index costs a whole extra slice —
    so ``outer * index_count * inner`` is bounded by nothing the source's
    own representability already proved, and the count must be asked about
    explicitly before the destination is allocated.

    Lowering the ceiling is what makes that boundary reachable with a 2x3
    source and a 4-element index tensor instead of an allocation no machine
    could satisfy. Everything else about the call is ordinary and valid: the
    operands are genuine native tensors, the axis is in range, the index
    rank is 1, and every index value is inside ``[0, 3)``, so the *only*
    thing that can reject is the output-count check."""
    # Built at the tensor's own width, so every comparison below is a
    # same-dtype raw-bit comparison and never a cross-dtype one.
    values = np.arange(6.0).reshape(2, 3).astype(dtype)
    source = NativeTensor.from_array(values, dtype=dtype)
    indices = _index([1, 0, 2, 2])            # duplicates: 4 > the axis's 3
    # outer(2) * index_count(4) * inner(1); written out here independently of
    # the implementation so a silent change to the decomposition fails.
    output_count = 2 * 4 * 1
    try:
        # The control **first**: unlowered, this exact call succeeds — so the
        # rejection below is caused by the ceiling and by nothing else.
        with _selected(source, 1, indices) as result:
            assert result.shape == (2, 4)
            assert _same_bits(result.to_numpy(),
                              _numpy_selection(values, 1, [1, 0, 2, 2]))

        # The public path, with the ceiling one below the derived count.
        with unchanged_world():
            with lowered_int64_max(output_count - 1):
                with live_storage_baseline():
                    with no_native_allocation():
                        with pytest.raises(ValueError) as caught:
                            source.index_select(1, indices)
        message = str(caught.value)
        assert "index_select" in message, message
        assert str(output_count) in message, message
        assert "int64" in message, message
        assert "output element count" in message, message
        # A *caller* error, and not the internal consistency assertion — the
        # two are separate questions and separate types (§18.6 step 11).
        assert not isinstance(caught.value, RuntimeError)

        # The Core path takes it too: the check lives on the Core, and the
        # tensor method adds only the graph rule above it.
        core = source._require_open()
        index_core = indices._require_open()
        with lowered_int64_max(output_count - 1):
            with live_storage_baseline():
                with no_native_allocation():
                    with pytest.raises(ValueError, match="output element "
                                                         "count"):
                        core.index_select(1, index_core)

        # The comparison is exact rather than approximate: a ceiling equal to
        # the derived count is representable and must be **accepted**, and
        # that acceptance goes all the way through to the right answer.
        with lowered_int64_max(output_count):
            with _selected(source, 1, indices) as result:
                assert result.shape == (2, 4)
                assert _same_bits(result.to_numpy(),
                                  _numpy_selection(values, 1, [1, 0, 2, 2]))

        # Restored, and the ordinary call still works — the ceiling was
        # borrowed, not spent.
        assert cpp._INT64_MAX == 2 ** 63 - 1
        with live_storage_baseline():
            with _selected(source, 1, indices) as result:
                assert _same_bits(result.to_numpy(),
                                  _numpy_selection(values, 1, [1, 0, 2, 2]))
    finally:
        indices.close()
        source.close()


@needs_native
def test_an_invalid_index_value_beats_an_unrepresentable_output_count():
    """The complete bounds scan stays **before** the representability check
    (§18.6 steps 10 then 11).

    Both would reject this call, so only the ordering can decide which error
    arrives — and it must be the index scan's, because a check that ran
    first would report a shape problem for what is really a bad index, and
    because the scan is what §14.4 requires to complete before anything
    downstream of it runs."""
    source = _floating(np.arange(6.0).reshape(2, 3), "float64")
    bad = _index([1, 0, 2, 99])               # 99 is outside [0, 3)
    good = _index([1, 0, 2, 2])
    try:
        with unchanged_world():
            with lowered_int64_max(1):        # every output count exceeds 1
                with no_native_allocation():
                    with pytest.raises(ValueError) as caught:
                        source.index_select(1, bad)
        message = str(caught.value)
        assert "position 3" in message, message
        assert "99" in message, message
        assert "output element count" not in message, message

        # The two controls that make the ordering claim mean something:
        # with the *same* lowered ceiling and the index repaired, the count
        # check is what rejects...
        with lowered_int64_max(1):
            with no_native_allocation():
                with pytest.raises(ValueError, match="output element count"):
                    source.index_select(1, good)
        # ...and with the ceiling restored and the index still bad, the scan
        # is what rejects. Each check is reachable on its own, so neither
        # result above was an accident of the other being unreachable.
        with no_native_allocation():
            with pytest.raises(ValueError, match="position 3"):
                source.index_select(1, bad)
    finally:
        good.close()
        bad.close()
        source.close()


@needs_native
def test_the_error_precedence_is_the_contracted_one():
    """Calls that are invalid in **two** ways at once, asserting which
    error arrives — the architecture of §18.6's ordering rather than its
    wording."""
    closed_source = _floating([[1.0, 2.0]], "float64")
    closed_source.close()
    closed_index = _index([0])
    closed_index.close()
    integer_source = _index([1, 2, 3])
    floating_index = _floating([0.0, 1.0], "float64")
    grad_source = NativeTensor.from_array(np.arange(6.0).reshape(2, 3),
                                          requires_grad=True)
    good_source = _floating(np.arange(6.0).reshape(2, 3), "float64")
    good_index = _index([1, 0])
    # Doubly invalid on purpose: rank 2 **and** a value out of range for
    # axis 0's extent of 2, so only the ordering can decide which is
    # reported. ``reshape`` borrows, so the flat base is retained and closed.
    rank2_base = _index([0, 99, 0, 1])
    rank2_index = rank2_base.reshape((2, 2))
    bad_index = _index([0, 99])

    try:
        cases = (
            # bad axis type beats bad indices type
            ("axis type over index type", TypeError, "axis",
             lambda: good_source.index_select("0", [1, 2])),
            # bad indices type beats a closed source
            ("index type over closed source", TypeError, "NativeTensor",
             lambda: closed_source.index_select(0, [1, 2])),
            # closed source beats closed indices
            ("closed source over closed index", RuntimeError, "closed",
             lambda: closed_source.index_select(0, closed_index)),
            # closed indices beats dtype errors
            ("closed index over dtype", RuntimeError, "closed",
             lambda: integer_source.index_select(0, closed_index)),
            # an int64 source beats a floating index and the grad rule
            ("int64 source over floating index", ValueError, "floating",
             lambda: integer_source.index_select(0, floating_index)),
            # a floating index dtype beats the source-grad rejection
            ("floating index over requires_grad", ValueError, "index dtype",
             lambda: grad_source.index_select(0, floating_index)),
            # requires_grad beats an out-of-range axis
            ("requires_grad over axis range", ValueError, "detach",
             lambda: grad_source.index_select(9, good_index)),
            # an out-of-range axis beats a bad index rank
            ("axis range over index rank", ValueError, "out of bounds",
             lambda: good_source.index_select(9, rank2_index)),
            # a bad index rank beats bad index values
            ("index rank over index values", ValueError, "rank-1",
             lambda: good_source.index_select(0, rank2_index)),
            # a bad index value beats output allocation
            ("index values over allocation", ValueError, "position",
             lambda: good_source.index_select(0, bad_index)),
        )
        for label, kind, needle, call in cases:
            with no_native_allocation():
                with pytest.raises(kind) as caught:
                    call()
            assert needle in str(caught.value), (label, str(caught.value))
    finally:
        for tensor in (bad_index, rank2_index, rank2_base, good_index,
                       good_source, grad_source, floating_index,
                       integer_source):
            tensor.close()


@needs_native
def test_the_core_layer_keeps_the_same_order_without_the_graph_step():
    """The Core carries no graph metadata, so its order is the public one
    minus step 7 — and it requires a ``NativeTensorCore``, not a
    ``NativeTensor``."""
    source = _floating(np.arange(6.0).reshape(2, 3), "float64")
    indices = _index([1, 0])
    try:
        core = source._require_open()
        index_core = indices._require_open()
        with no_native_allocation():
            with pytest.raises(TypeError, match="axis"):
                core.index_select("0", index_core)
            with pytest.raises(TypeError, match="NativeTensorCore"):
                core.index_select(0, indices)          # the wrapper, not a core
            with pytest.raises(ValueError, match="out of bounds"):
                core.index_select(5, index_core)
        # ...and the ordinary Core call works and is the same answer.
        out_core = core.index_select(1, index_core)
        try:
            assert out_core.shape == (2, 2)
            assert _same_bits(out_core.to_numpy(),
                              _numpy_selection(source.to_numpy(), 1, [1, 0]))
        finally:
            out_core.close()
    finally:
        indices.close()
        source.close()


# ===========================================================================
# 6. Autograd exclusion
# ===========================================================================

@needs_native
@pytest.mark.parametrize("dtype", FLOATING_DTYPES)
def test_a_gradient_tracking_source_is_rejected_and_names_detach(dtype):
    """Forward only, and it says so (§18.9). A silent detach would be a
    gradient hole, so the call raises instead."""
    source = NativeTensor.from_array(np.arange(6.0).reshape(2, 3),
                                     dtype=dtype, requires_grad=True)
    indices = _index([2, 0])
    try:
        before = (source.requires_grad, source.grad, source.is_leaf,
                  source.to_numpy().tobytes())
        with no_native_allocation():
            with pytest.raises(ValueError) as caught:
                source.index_select(1, indices)
        assert "detach()" in str(caught.value)
        # The source's graph, gradient, and values are untouched.
        assert (source.requires_grad, source.grad, source.is_leaf,
                source.to_numpy().tobytes()) == before
        # ...and the index operand is untouched too.
        assert indices.requires_grad is False
        assert indices.tolist() == [2, 0]
    finally:
        indices.close()
        source.close()


@needs_native
@pytest.mark.parametrize("dtype", FLOATING_DTYPES)
def test_the_detached_route_works_and_returns_a_plain_leaf(dtype):
    source = NativeTensor.from_array(np.arange(6.0).reshape(2, 3),
                                     dtype=dtype, requires_grad=True)
    indices = _index([2, 0])
    detached = source.detach()
    try:
        result = detached.index_select(1, indices)
        try:
            assert result.requires_grad is False
            assert result.grad is None
            assert result.is_leaf is True
            assert result._parents == ()
            assert result._backward is None
            assert result._op == ""
            assert result._graph_resources == ()
            assert result._expected_versions == ()
            assert _same_bits(result.to_numpy(),
                              _numpy_selection(source.to_numpy(), 1, [2, 0]))
            # A backward on a non-grad leaf is refused by the ordinary rule,
            # not by anything index_select added.
            with pytest.raises((RuntimeError, ValueError)):
                result.backward()
        finally:
            result.close()
        # The gradient-tracking original is still intact and still refused.
        assert source.requires_grad is True
        with pytest.raises(ValueError, match="detach"):
            source.index_select(1, indices)
    finally:
        detached.close()
        indices.close()
        source.close()


@needs_native
def test_index_select_never_calls_the_graph_constructor():
    """Structural, not incidental: ``_from_op`` is never reached, at either
    layer, for either the source or the index operand."""
    source = NativeTensor.from_array(np.arange(6.0).reshape(2, 3))
    indices = _index([1, 0])
    seen = []
    original = NativeTensor._from_op

    def watching(cls, *args, **kwargs):
        seen.append(args[3] if len(args) > 3 else kwargs.get("op"))
        return original.__func__(cls, *args, **kwargs)

    NativeTensor._from_op = classmethod(watching)
    try:
        with _selected(source, 0, indices):
            pass
        assert seen == [], seen
        # The control: an operation that *does* build a graph is seen by the
        # same watcher, so "not called" is a measurement.
        tracked = NativeTensor.from_array(np.ones((2, 3)), requires_grad=True)
        try:
            produced = tracked.relu()
            produced.close()
        finally:
            tracked.close()
        assert seen == ["relu"], seen
    finally:
        NativeTensor._from_op = original
        indices.close()
        source.close()
    # ...and the source reads it, so the method really is the entry point.
    tensor_source = _code_only(
        (REPO_ROOT / "src" / "tensorforge" / "experimental"
         / "native_tensor.py").read_text(encoding="utf-8"))
    assert "_from_op" in tensor_source
    tree = ast.parse((REPO_ROOT / "src" / "tensorforge" / "experimental"
                      / "native_tensor.py").read_text(encoding="utf-8"))
    method = next(
        node for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "index_select")
    attributes = {node.attr for node in ast.walk(method)
                  if isinstance(node, ast.Attribute)}
    assert "_from_op" not in attributes
    assert "_from_core" in attributes


@needs_native
def test_a_parameters_version_and_gradient_are_untouched_by_a_selection():
    parameter = NativeParameter(np.arange(6.0).reshape(2, 3))
    indices = _index([1, 0])
    try:
        version = parameter.version
        detached = parameter.detach()
        try:
            with _selected(detached, 1, indices):
                pass
        finally:
            detached.close()
        assert parameter.version == version
        assert parameter.grad is None
        # A NativeParameter always requires grad, so the direct call is the
        # rejection, not a silent detach.
        with pytest.raises(ValueError, match="detach"):
            parameter.index_select(1, indices)
        assert parameter.version == version
    finally:
        indices.close()
        parameter.close()


def test_index_select_is_absent_from_the_autograd_inventory():
    assert "index_select" not in cpp.AUTOGRAD_OPS
    assert "index_select_backward" not in cpp.AUTOGRAD_OPS
    assert not [name for name in cpp.AUTOGRAD_OPS if "index" in name]
    assert not [name for name in cpp.AUTOGRAD_OPS if "gather" in name]


# ===========================================================================
# 7. The C ABI is a second authority
# ===========================================================================

def _typed_core(shape, dtype, values=None):
    core = cpp.NativeTensorCore._typed(shape, dtype)
    if values is not None:
        core._storage.copy_from(
            np.ascontiguousarray(values, dtype=cpp._DTYPE_NUMPY[dtype]))
    return core


@needs_native
@pytest.mark.parametrize("dtype", FLOATING_DTYPES)
def test_the_abi_accepts_the_three_roles_together(dtype):
    library = cpp._require_library()
    source = _typed_core((6,), dtype, [10.0, 11.0, 12.0, 13.0, 14.0, 15.0])
    indices = _typed_core((3,), INDEX_DTYPE, [2, 0, 2])
    destination = _typed_core((3,), dtype)
    try:
        library.tf_core_index_select(
            source._storage._require_open(), 0,
            indices._storage._require_open(), 0,
            destination._storage._require_open(),
            1, 6, 3, 1)
        assert _same_bits(destination.to_numpy(),
                          np.array([12.0, 10.0, 12.0], dtype=dtype))
        assert source.dtype == destination.dtype == dtype
        assert indices.dtype == INDEX_DTYPE
        assert indices.dtype != source.dtype
    finally:
        destination.close()
        indices.close()
        source.close()


@needs_native
def test_the_abi_rejects_every_role_error_independently_of_python():
    library = cpp._require_library()
    f64 = _typed_core((6,), "float64", [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    f32 = _typed_core((6,), "float32", [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    integers = _typed_core((6,), INDEX_DTYPE, [1, 2, 3, 4, 5, 0])
    indices = _typed_core((3,), INDEX_DTYPE, [2, 0, 2])
    f64_dst = _typed_core((3,), "float64", [-1.5, -2.5, -3.5])
    f32_dst = _typed_core((3,), "float32", [-1.5, -2.5, -3.5])
    i64_dst = _typed_core((3,), INDEX_DTYPE, [-7, -8, -9])
    try:
        guards = {
            "int64 source": (integers, indices, f64_dst, "floating"),
            "int64 destination": (f64, indices, i64_dst, "floating"),
            "dtype mismatch": (f64, indices, f32_dst, "same dtype"),
            "floating index": (f64, f32, f64_dst, "int64"),
        }
        for label, (source, index, destination, needle) in guards.items():
            before = destination.to_numpy().tobytes()
            with pytest.raises(ValueError, match=needle):
                library.tf_core_index_select(
                    source._storage._require_open(), 0,
                    index._storage._require_open(), 0,
                    destination._storage._require_open(),
                    1, 6, 3, 1)
            assert destination.to_numpy().tobytes() == before, label
        # The non-vacuity control: sound roles succeed through the same
        # destination, which every rejection above would also have produced
        # had it been about something else.
        library.tf_core_index_select(
            f64._storage._require_open(), 0,
            indices._storage._require_open(), 0,
            f64_dst._storage._require_open(), 1, 6, 3, 1)
        assert _same_bits(f64_dst.to_numpy(), np.array([3.0, 1.0, 3.0]))
        library.tf_core_index_select(
            f32._storage._require_open(), 0,
            indices._storage._require_open(), 0,
            f32_dst._storage._require_open(), 1, 6, 3, 1)
        assert _same_bits(f32_dst.to_numpy(),
                          np.array([3.0, 1.0, 3.0], dtype=np.float32))
    finally:
        for core in (i64_dst, f32_dst, f64_dst, indices, integers, f32, f64):
            core.close()


@needs_native
def test_the_abi_rejects_every_malformed_layout_without_writing():
    library = cpp._require_library()
    source = _typed_core((6,), "float64", [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    indices = _typed_core((3,), INDEX_DTYPE, [2, 0, 2])
    destination = _typed_core((3,), "float64", [-424242.5, 987654321.25, -7.5])
    before = destination.to_numpy().tobytes()
    huge = (2 ** 63 - 1) // 2
    cases = {
        # (outer, axis_length, index_count, inner, src_offset, idx_offset)
        "zero outer": (0, 6, 3, 1, 0, 0),
        "zero axis_length": (1, 0, 3, 1, 0, 0),
        "zero index_count": (1, 6, 0, 1, 0, 0),
        "zero inner": (1, 6, 3, 0, 0, 0),
        "negative outer": (-1, 6, 3, 1, 0, 0),
        "negative axis_length": (1, -6, 3, 1, 0, 0),
        "negative index_count": (1, 6, -3, 1, 0, 0),
        "negative inner": (1, 6, 3, -1, 0, 0),
        "source product overflow": (huge, 4, 3, 1, 0, 0),
        "destination product overflow": (1, 6, huge, 4, 0, 0),
        "negative source offset": (1, 6, 3, 1, -1, 0),
        "negative index offset": (1, 6, 3, 1, 0, -1),
        "source span too long": (1, 7, 3, 1, 0, 0),
        "source offset pushes the span out": (1, 6, 3, 1, 1, 0),
        "index span too long": (1, 6, 3, 1, 0, 1),
        "destination too small": (2, 3, 2, 1, 0, 0),
        "destination too large": (1, 6, 2, 1, 0, 0),
    }
    try:
        for label, (outer, axis_length, count, inner,
                    src_offset, idx_offset) in cases.items():
            with pytest.raises(ValueError):
                library.tf_core_index_select(
                    source._storage._require_open(), src_offset,
                    indices._storage._require_open(), idx_offset,
                    destination._storage._require_open(),
                    outer, axis_length, count, inner)
            assert destination.to_numpy().tobytes() == before, label
        # Null handles in each of the three positions.
        for null_position in range(3):
            handles = [source._storage._require_open(),
                       indices._storage._require_open(),
                       destination._storage._require_open()]
            handles[null_position] = None
            with pytest.raises(ValueError):
                library.tf_core_index_select(
                    handles[0], 0, handles[1], 0, handles[2], 1, 6, 3, 1)
            assert destination.to_numpy().tobytes() == before, null_position
        # The self-aliasing pair the roles cannot otherwise reach.
        with pytest.raises(ValueError):
            library.tf_core_index_select(
                destination._storage._require_open(), 0,
                indices._storage._require_open(), 0,
                destination._storage._require_open(), 1, 3, 3, 1)
        assert destination.to_numpy().tobytes() == before
        # The non-vacuity control.
        library.tf_core_index_select(
            source._storage._require_open(), 0,
            indices._storage._require_open(), 0,
            destination._storage._require_open(), 1, 6, 3, 1)
        assert _same_bits(destination.to_numpy(), np.array([3.0, 1.0, 3.0]))
        assert destination.to_numpy().tobytes() != before
    finally:
        destination.close()
        indices.close()
        source.close()


@needs_native
def test_the_abi_scans_every_index_before_writing_anything():
    """The C-side scan is a **second** authority: driven with the Python
    layer bypassed entirely, over a span whose bad value comes last."""
    library = cpp._require_library()
    source = _typed_core((4,), "float64", [10.0, 11.0, 12.0, 13.0])
    destination = _typed_core((4,), "float64", [-1.5, -2.5, -3.5, -4.5])
    before = destination.to_numpy().tobytes()
    cases = {
        "negative first": [-1, 0, 1, 2],
        "negative last": [0, 1, 2, -1],
        "equal to axis_length": [0, 1, 2, 4],
        "beyond axis_length": [0, 1, 2, 99],
        "int64 minimum": [0, -(2 ** 63), 2, 3],
        "int64 maximum": [0, 1, 2, 2 ** 63 - 1],
    }
    try:
        for label, values in cases.items():
            indices = _typed_core((4,), INDEX_DTYPE, values)
            try:
                with pytest.raises(ValueError) as caught:
                    library.tf_core_index_select(
                        source._storage._require_open(), 0,
                        indices._storage._require_open(), 0,
                        destination._storage._require_open(), 1, 4, 4, 1)
                message = str(caught.value)
                assert "index_select" in message, (label, message)
                assert "position" in message, (label, message)
                assert destination.to_numpy().tobytes() == before, label
            finally:
                indices.close()
        # The boundary that must be accepted, beside the ones rejected.
        indices = _typed_core((4,), INDEX_DTYPE, [3, 3, 0, 3])
        try:
            library.tf_core_index_select(
                source._storage._require_open(), 0,
                indices._storage._require_open(), 0,
                destination._storage._require_open(), 1, 4, 4, 1)
            assert _same_bits(destination.to_numpy(),
                              np.array([13.0, 13.0, 10.0, 13.0]))
        finally:
            indices.close()
    finally:
        destination.close()
        source.close()


@needs_native
def test_offsets_cross_the_abi_exactly():
    library = cpp._require_library()
    source = _typed_core((6,), "float64", [9.0, 1.0, 2.0, 8.0, 3.0, 4.0])
    indices = _typed_core((4,), INDEX_DTYPE, [0, 0, 3, 1])
    destination = _typed_core((2,), "float64")
    try:
        # Source read from element 1 (1, 2, 8, 3, 4); index read from
        # position 2 (3, 1) -> values 3.0 and 2.0.
        library.tf_core_index_select(
            source._storage._require_open(), 1,
            indices._storage._require_open(), 2,
            destination._storage._require_open(), 1, 5, 2, 1)
        assert _same_bits(destination.to_numpy(), np.array([3.0, 2.0]))
    finally:
        destination.close()
        indices.close()
        source.close()


@needs_native
def test_the_export_records_and_then_clears_the_thread_local_slot():
    """A guarded export records a rejection and clears it on the **next**
    entry, so a stale code is never misread as a later call's failure.

    Driven with the ``errcheck`` hook temporarily detached, because that
    hook is what normally consumes and clears the slot; it is restored in a
    ``finally``."""
    library = cpp._require_library()
    source = _typed_core((4,), "float64", [1.0, 4.0, 2.0, 3.0])
    indices = _typed_core((2,), INDEX_DTYPE, [1, 3])
    destination = _typed_core((2,), "float64")
    hook = library.tf_core_index_select.errcheck
    assert hook is not None, "the export is not hooked at all"
    try:
        library.tf_core_index_select.errcheck = \
            lambda result, function, arguments: result
        library.tf_clear_error()
        library.tf_core_index_select(
            source._storage._require_open(), 0,
            indices._storage._require_open(), 0,
            destination._storage._require_open(), 0, 4, 2, 1)
        assert library.tf_last_error_code() != cpp.TF_OK
        assert "index_select" in library.tf_last_error_message().decode()
        library.tf_core_index_select(
            source._storage._require_open(), 0,
            indices._storage._require_open(), 0,
            destination._storage._require_open(), 1, 4, 2, 1)
        assert library.tf_last_error_code() == cpp.TF_OK
        assert _same_bits(destination.to_numpy(), np.array([4.0, 3.0]))
    finally:
        library.tf_core_index_select.errcheck = hook
        library.tf_clear_error()
        destination.close()
        indices.close()
        source.close()
    # ...and with the hook restored, a rejection is a Python exception again.
    source = _typed_core((4,), "float64", [1.0, 4.0, 2.0, 3.0])
    indices = _typed_core((2,), INDEX_DTYPE, [1, 3])
    destination = _typed_core((2,), "float64")
    try:
        with pytest.raises(ValueError, match="index_select"):
            library.tf_core_index_select(
                source._storage._require_open(), 0,
                indices._storage._require_open(), 0,
                destination._storage._require_open(), 0, 4, 2, 1)
    finally:
        destination.close()
        indices.close()
        source.close()


# ===========================================================================
# 8. Failure cleanup — five distinct injections
# ===========================================================================

@needs_native
def test_a_failed_source_policy_b_temporary_leaks_nothing():
    """Injection 1: the source's private contiguous copy. Nothing else has
    been allocated when it fires."""
    base = _floating([[1.0, 4.0], [3.0, 2.0]], "float64")
    view = base.T
    indices = _index([1, 0])
    fired = []
    original = cpp.NativeTensorCore.contiguous_copy

    def failing(self):
        if cpp._is_floating_dtype(self.dtype):
            fired.append(True)
            raise MemoryError("injected: source Policy-B temporary")
        return original(self)

    try:
        with unchanged_world() as world:
            with live_storage_baseline():
                cpp.NativeTensorCore.contiguous_copy = failing
                try:
                    with pytest.raises(MemoryError, match="injected"):
                        view.index_select(1, indices)
                finally:
                    cpp.NativeTensorCore.contiguous_copy = original
            assert world is not None
        assert fired, "the injection never fired"
        with _selected(view, 1, indices) as result:
            assert result.shape == (2, 2)
    finally:
        cpp.NativeTensorCore.contiguous_copy = original
        indices.close()
        view.close()
        base.close()


@needs_native
def test_a_failed_index_policy_b_temporary_closes_the_source_temporary():
    """Injection 2: the **index** operand's contiguous copy — a different
    position from the source's, and the one that proves the source
    temporary is released when a later step fails."""
    base = _floating([[1.0, 4.0], [3.0, 2.0]], "float64")
    view = base.T
    index_base = _index([1, 7, 0, 7])
    strided = _strided_index_view(index_base)
    allocated = []
    fired = []
    original = cpp.NativeTensorCore.contiguous_copy

    def injecting(self):
        if cpp._is_index_dtype(self.dtype):
            fired.append(True)
            raise MemoryError("injected: index Policy-B temporary")
        temporary = original(self)
        allocated.append(temporary)
        return temporary

    try:
        with unchanged_world() as world:
            with live_storage_baseline():
                cpp.NativeTensorCore.contiguous_copy = injecting
                try:
                    with pytest.raises(MemoryError, match="injected"):
                        view.index_select(1, strided)
                finally:
                    cpp.NativeTensorCore.contiguous_copy = original
            assert world is not None
        assert fired, "the injection never fired"
        # The source temporary is retained strongly here and proved closed
        # while still referenced — no collection timing involved.
        assert len(allocated) == 1
        assert allocated[0]._closed is True
    finally:
        cpp.NativeTensorCore.contiguous_copy = original
        strided.close()
        index_base.close()
        view.close()
        base.close()


@needs_native
def test_a_failed_destination_allocation_closes_both_temporaries():
    """Injection 3: the floating destination, which fires **after** both
    Policy-B temporaries exist."""
    base = _floating([[1.0, 4.0], [3.0, 2.0]], "float64")
    view = base.T
    index_base = _index([1, 7, 0, 7])
    strided = _strided_index_view(index_base)
    allocated = []
    fired = []
    original_typed = cpp.NativeTensorCore._typed
    original_copy = cpp.NativeTensorCore.contiguous_copy
    seen_shapes = []

    def recording_copy(self):
        temporary = original_copy(self)
        allocated.append(temporary)
        return temporary

    def failing_typed(cls, shape, dtype, device="cpu", **kwargs):
        # Only the *destination* allocation fails: the Policy-B copies
        # reach ``_typed`` too, through ``contiguous_copy``, so the
        # injection is keyed on the shape the selection derives.
        if tuple(shape) == (2, 2) and not seen_shapes:
            seen_shapes.append(tuple(shape))
            fired.append(True)
            raise MemoryError("injected: destination allocation")
        return original_typed.__func__(cls, shape, dtype, device=device,
                                       **kwargs)

    try:
        with unchanged_world() as world:
            with live_storage_baseline():
                cpp.NativeTensorCore.contiguous_copy = recording_copy
                cpp.NativeTensorCore._typed = classmethod(failing_typed)
                try:
                    with pytest.raises(MemoryError, match="injected"):
                        view.index_select(1, strided)
                finally:
                    cpp.NativeTensorCore._typed = original_typed
                    cpp.NativeTensorCore.contiguous_copy = original_copy
            assert world is not None
        assert fired, "the injection never fired"
        assert len(allocated) == 2, [core.dtype for core in allocated]
        for temporary in allocated:
            assert temporary._closed is True
    finally:
        cpp.NativeTensorCore._typed = original_typed
        cpp.NativeTensorCore.contiguous_copy = original_copy
        strided.close()
        index_base.close()
        view.close()
        base.close()


@needs_native
def test_a_failed_native_call_closes_the_destination_and_the_temporaries():
    """Injection 4: the kernel call itself, which fires with the
    temporaries **and** the destination alive. ``BaseException`` is used
    here, so the cleanup is proved unconditional rather than
    ``except Exception``."""

    class Interrupt(BaseException):
        """Not an ``Exception``: an ``except Exception`` cleanup would miss
        it, which is exactly what this injection is for."""

    base = _floating([[1.0, 4.0], [3.0, 2.0]], "float64")
    view = base.T
    indices = _index([1, 0])
    allocated = []
    fired = []
    original_typed = cpp.NativeTensorCore._typed
    original_copy = cpp.NativeTensorCore.contiguous_copy
    library = cpp._require_library()
    original_kernel = library.tf_core_index_select

    def recording_copy(self):
        temporary = original_copy(self)
        allocated.append(("temporary", temporary))
        return temporary

    def recording_typed(cls, shape, dtype, device="cpu", **kwargs):
        core = original_typed.__func__(cls, shape, dtype, device=device,
                                       **kwargs)
        if tuple(shape) == (2, 2):
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
                library.tf_core_index_select = failing_kernel
                try:
                    with pytest.raises(Interrupt):
                        view.index_select(1, indices)
                finally:
                    library.tf_core_index_select = original_kernel
                    cpp.NativeTensorCore._typed = original_typed
                    cpp.NativeTensorCore.contiguous_copy = original_copy
            assert world is not None
        assert fired, "the injection never fired"
        kinds = [kind for kind, _ in allocated]
        assert kinds == ["temporary", "destination"], kinds
        for kind, core in allocated:
            assert core._closed is True, kind
        with _selected(view, 1, indices) as result:
            assert result.shape == (2, 2)
    finally:
        library.tf_core_index_select = original_kernel
        cpp.NativeTensorCore._typed = original_typed
        cpp.NativeTensorCore.contiguous_copy = original_copy
        indices.close()
        view.close()
        base.close()


@needs_native
def test_a_failed_wrapper_publication_closes_the_core_it_was_handed():
    """Injection 5: the ``NativeTensor`` publication, a different position
    again — the Core has already been published successfully and is the
    tensor layer's to release."""
    source = _floating([[1.0, 4.0], [3.0, 2.0]], "float64")
    indices = _index([1, 0])
    produced = []
    fired = []
    original_core = cpp.NativeTensorCore.index_select
    original_from_core = NativeTensor._from_core

    def recording_core(self, axis, index_core):
        core = original_core(self, axis, index_core)
        produced.append(core)
        return core

    def failing_from_core(cls, core, owns_core=True):
        fired.append(True)
        raise KeyboardInterrupt("injected: wrapper publication")

    try:
        with unchanged_world() as world:
            with live_storage_baseline():
                cpp.NativeTensorCore.index_select = recording_core
                NativeTensor._from_core = classmethod(failing_from_core)
                try:
                    with pytest.raises(KeyboardInterrupt, match="injected"):
                        source.index_select(1, indices)
                finally:
                    NativeTensor._from_core = original_from_core
                    cpp.NativeTensorCore.index_select = original_core
            assert world is not None
        assert fired, "the injection never fired"
        assert len(produced) == 1
        # Retained strongly here, and proved closed while still referenced.
        assert produced[0]._closed is True
        with _selected(source, 1, indices) as result:
            assert result.shape == (2, 2)
    finally:
        NativeTensor._from_core = original_from_core
        cpp.NativeTensorCore.index_select = original_core
        indices.close()
        source.close()


# ===========================================================================
# 9. Absence — what K4 did not ship
# ===========================================================================

def test_no_general_gather_scatter_embedding_or_subscript_appeared():
    """K4 ships one primitive, and §18.1 keeps five things apart. The four
    it is not are absent at every layer."""
    for owner in (NativeTensor, cpp.NativeTensorCore, cpp.NativeStorage):
        for absent in ("gather", "scatter", "scatter_add", "embedding",
                       "take", "index_put", "index_add", "masked_select",
                       "__getitem__", "__setitem__"):
            assert not hasattr(owner, absent), (owner.__name__, absent)
    tensor_code = _code_only(
        (REPO_ROOT / "src" / "tensorforge" / "experimental"
         / "native_tensor.py").read_text(encoding="utf-8"))
    core_code = _code_only(
        (REPO_ROOT / "src" / "tensorforge" / "backends" / "cpp.py"
         ).read_text(encoding="utf-8"))
    for banned in ("argmin", "max_with_indices", "scatter_add",
                   "index_select_backward"):
        assert banned not in tensor_code, banned
        assert banned not in core_code, banned


def test_no_backward_export_or_autograd_entry_was_added():
    exports = _source_exports()
    for absent in ("tf_core_index_select_backward", "tf_core_scatter_add",
                   "tf_core_gather", "tf_core_embedding"):
        assert absent not in exports, absent
        assert absent not in cpp._CHECKED_KERNELS, absent
    assert "index_select" not in cpp.AUTOGRAD_OPS


def test_no_integer_arithmetic_casting_or_promotion_appeared():
    for owner in (NativeTensor, cpp.NativeTensorCore, cpp.NativeStorage):
        for absent in ("astype", "cast", "to", "type", "long", "int",
                       "float", "double", "promote", "as_type", "cpu",
                       "cuda", "is_integer", "is_floating"):
            assert not hasattr(owner, absent), (owner.__name__, absent)


@needs_native
def test_an_index_operand_is_still_refused_by_every_k1_barrier():
    """The index tensor is an ordinary ``int64`` tensor throughout, so
    every K1 barrier applies to it unchanged before and after a selection —
    including the one that matters most here, that it never receives a
    gradient."""
    source = _floating([[1.0, 7.0], [9.0, 2.0]], "float64")
    indices = _index([1, 0, 1])
    try:
        with _selected(source, 1, indices):
            pass
        with pytest.raises(ValueError):
            NativeParameter(indices)
        module = NativeModule()
        for persistent in (True, False):
            with pytest.raises(ValueError):
                module.register_buffer("probe", indices,
                                       persistent=persistent)
        with pytest.raises((ValueError, RuntimeError)):
            indices.backward()
        with pytest.raises(RuntimeError, match="differentiable"):
            indices._accumulate_grad(source)
        for operation in ("relu", "sqrt", "exp", "sum", "mean"):
            with pytest.raises(ValueError, match="floating"):
                getattr(indices, operation)()
        # ...and an index_select **of** an index tensor is refused for the
        # same reason: an integer tensor can select with, never be selected
        # from.
        with pytest.raises(ValueError, match="floating"):
            indices.index_select(0, indices)
        # The index operand is unchanged by all of it.
        assert indices.tolist() == [1, 0, 1]
        assert indices.requires_grad is False
        assert indices.grad is None
    finally:
        indices.close()
        source.close()


@needs_native
def test_the_stable_line_is_untouched_and_gained_no_index_select():
    import tensorforge

    assert not hasattr(tensorforge.Tensor, "index_select")
    assert cpp.backend_info()["stable_framework_integration"] is False
    assert "index_select" not in tensorforge.__all__


def test_k4_added_no_benchmark_timing_or_performance_control():
    """K4 is a capability milestone, not a benchmark one, and it added no
    dispatch or timing surface to the production files it touched.

    Scanned over the **production** sources rather than over this module,
    which would trivially find its own banned list, and over **code only**,
    because both files document that no predicate is a function of a clock
    or an environment variable."""
    production = ("cpp/src/indexing.cpp", "cpp/include/tf_indexing_internal.h")
    banned = re.compile(
        r"\b(chrono|clock|getenv|omp|restrict|block_size|set_path|"
        r"which_kernel|rdtsc)\b|std::thread|#pragma\s+omp|fast-math|_mm_|"
        r"__builtin_prefetch")
    for relative in production:
        text = (REPO_ROOT / relative).read_text(encoding="utf-8")
        code = re.sub(r"//[^\n]*", " ",
                      re.sub(r"/\*.*?\*/", " ", text, flags=re.S))
        found = banned.search(code)
        assert found is None, (relative, found and found.group(0))
        assert len(code) < len(text), relative
    # The scanner's negative controls, on temporary strings.
    assert banned.search("#pragma omp parallel for") is not None
    assert banned.search("auto t = std::chrono::steady_clock::now();")
    assert banned.search("the noexcept compute kernel") is None
    # ...and the control: a token that *is* in the unit is found.
    unit = (REPO_ROOT / "cpp" / "src" / "indexing.cpp").read_text(
        encoding="utf-8")
    assert "index_select_contiguous" in unit
    # No example or benchmark inventory moved.
    assert len(list((REPO_ROOT / "examples").glob("*.py"))) == EXAMPLE_COUNT
    assert len(list((REPO_ROOT / "benchmarks").glob("*.py"))) == BENCHMARK_COUNT


def test_the_kernel_copies_bytes_and_reads_no_floating_value():
    """The bit-preservation mechanism is structural: the traversal moves
    object representations with ``memcpy`` and contains no floating
    arithmetic, no comparison, and no dtype branch inside its loop."""
    header = (REPO_ROOT / "cpp" / "include" / "tf_indexing_internal.h"
              ).read_text(encoding="utf-8")
    code = re.sub(r"//[^\n]*", " ",
                  re.sub(r"/\*.*?\*/", " ", header, flags=re.S))
    body = code.split("index_select_contiguous", 1)[1]
    assert "std::memcpy" in body
    assert "noexcept" in body
    for banned in ("isnan", "new ", "malloc", "std::vector", "static_cast<double>",
                   "static_cast<float>"):
        assert banned not in body, banned
    # The negative control for the split: the argmax traversal above it is
    # the one that compares, and it is a different routine.
    argmax_body = code.split("argmax_contiguous", 1)[1].split(
        "index_select_contiguous", 1)[0]
    assert "isnan" in argmax_body
    assert "memcpy" not in argmax_body


def test_the_two_exports_do_not_share_a_validator():
    """§22.10: shared arithmetic primitives, separate validation lists. A
    blanket validator would make two contracts one."""
    unit = (REPO_ROOT / "cpp" / "src" / "indexing.cpp").read_text(
        encoding="utf-8")
    code = re.sub(r"//[^\n]*", " ",
                  re.sub(r"/\*.*?\*/", " ", unit, flags=re.S))
    assert "index_select_argument_error" in code
    assert "argument_error(" in code
    # Two distinct validator definitions, and the index one is not called by
    # argmax's export.
    argmax_body = code.split("TF_EXPORT void tf_core_argmax(", 1)[1].split(
        "TF_EXPORT void tf_core_index_select(", 1)[0]
    assert "index_select_argument_error" not in argmax_body
    assert "reject_index_range" not in argmax_body
    select_body = code.split("TF_EXPORT void tf_core_index_select(", 1)[1]
    assert "index_select_argument_error" in select_body
    assert "reject_index_range" in select_body
    # ...and the role guards each export applies are the contracted ones.
    assert "require_matching_dtype" in select_body
    assert "require_matching_dtype" not in argmax_body
    assert "require_index" in argmax_body and "require_index" in select_body
