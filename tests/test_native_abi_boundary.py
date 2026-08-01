"""The H7 Python/C ABI boundary contract (Phase H, milestone H7).

H7 changed how this project's own layout metadata reaches the strided C
ABI. It changed **no** C++, no exported symbol, no kernel, no arithmetic,
no public API, and no capability. What it changed is which ctypes
argument type stands between a Python object and a native pointer, and it
did so for exactly one category of argument.

The architecture, and what this file proves about it:

1. **Two categories, two bindings.** Every array position in
   ``backends/cpp.py`` is either *data* — a float64 buffer a caller
   supplied or that native code writes into and hands back, plus the
   cross-entropy class labels — or *layout metadata* this module built for
   itself from a tuple it had already validated. Data positions keep the
   checked ``numpy.ctypeslib.ndpointer`` binding, which re-verifies the
   array, its exact dtype, and its contiguity at every call. Layout
   positions take ``ctypes.POINTER(ctypes.c_int64)``, because every
   property ``ndpointer`` would re-check is established once, at an
   immutable construction boundary, and cannot subsequently change.

2. **Exactly two producers.** ``NativeTensorView._native_layout_pointers``
   returns the per-view pair, derived from the H3 read-only NumPy arrays
   that remain the owning buffers; ``cpp._layout_vector`` builds a fresh
   ``c_int64`` vector for metadata that belongs to one *operation* rather
   than to any tensor. Nothing else may reach a trusted position.

3. **Length is rank, by construction.** This is the invariant
   ``ndpointer`` never checked and could not check: the ABI receives a
   pointer and an ``ndim``, and has no access to the Python object's
   length. Both producers make the two agree structurally — a view's
   arrays were built from its own ``shape``/``strides``, and a vector is
   exactly ``len(values)`` long — and this file asserts it directly.

4. **Owner lifetime is NumPy's guarantee, not a convention.**
   ``ndarray.ctypes.data_as`` stores the array on the pointer it returns,
   so a cached pointer cannot outlive its buffer.
   ``POINTER(...).from_address(...)`` would be cheaper and is deliberately
   not used, precisely because it produces a pointer with no owner.

5. **The binding changed no arithmetic, and that is proved against the
   other binding.** Section 9 drives every trusted export twice from the
   same production call site — once through the shipped
   ``POINTER(c_int64)`` binding and once through a test-local
   reconstruction of the checked ``ndpointer`` declaration H7 replaced —
   and compares raw IEEE-754 bits. NumPy appears there only as an
   *external mathematical oracle*, and only where it is one: exactly for
   the operations IEEE-754 defines uniquely. ``exp`` and ``log`` are
   compared against it within a measured one-step budget instead, because
   bit equality between two different libm implementations is a property
   of the platform rather than of this project. Section 9's own comment
   records the measurement that fixes the budget.

What H7 must not have done, also asserted here: weaken any public
rejection, change any error type or message, expose a raw pointer or a
native address, add a trusted-call flag or an environment fast path, make
a closed object usable, reconfigure a binding at call time, move the
exported symbol count, or change any capability, dtype, device, or
checkpoint value.

No test in this file asserts a duration. H7's measurements live in the
design document and in the benchmark harness, never in the suite.
"""

import ctypes
import gc
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from tensorforge.backends import cpp

REPO_ROOT = Path(__file__).resolve().parent.parent
BACKEND_FILE = REPO_ROOT / "src" / "tensorforge" / "backends" / "cpp.py"

needs_native = pytest.mark.skipif(
    not cpp.is_available(), reason="native backend not built"
)

Core = cpp.NativeTensorCore


@pytest.fixture
def live_storages(monkeypatch):
    """The ids of every open NativeStorage — a real live-allocation count,
    so an ownership test can prove the count returns exactly to its
    baseline instead of trusting collection."""
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


# ==========================================================================
# 1. The complete binding inventory
# ==========================================================================

# Every exported function this module configures, and which of its
# arguments are arrays. The two categories are spelled out per position:
# "checked" for an ndpointer binding, "trusted" for a layout pointer.
#
# This table is the milestone's inventory in executable form. A future
# change that adds an array argument, or that quietly moves one position
# from checked to trusted, fails here and has to say so.
EXPECTED_ARRAY_BINDINGS = {
    # -- public raw-buffer kernels: caller-supplied float64 -----------------
    "tf_elementwise_add": ("checked", "checked", "checked"),
    "tf_elementwise_subtract": ("checked", "checked", "checked"),
    "tf_elementwise_multiply": ("checked", "checked", "checked"),
    "tf_elementwise_divide": ("checked", "checked", "checked"),
    "tf_relu": ("checked", "checked"),
    "tf_matmul": ("checked", "checked", "checked"),
    "tf_matmul_tiled": ("checked", "checked", "checked"),
    # -- the explicit host conversion boundary -----------------------------
    #
    # ``tf_storage_copy_from``, ``tf_storage_copy_to``, and the destination
    # of ``tf_storage_materialize`` left this table at Phase I milestone I2
    # and are **not** a gap in it. Their C host positions became ``void*``
    # (the storage handle's dtype tag says how the bytes are read), and one
    # ctypes ``argtypes`` slot cannot describe two dtypes — so the binding
    # moved from the slot to the call site, where the dtype is known. It is
    # the same ``ndpointer`` check, run per call by ``cpp._host_pointer``
    # against the storage's own dtype; see
    # ``test_a_retyped_transfer_position_is_still_checked_per_dtype``.
    #
    # ``tf_storage_materialize`` keeps its trusted layout pair here, so the
    # export is still listed — with one fewer position.
    "tf_storage_materialize": ("trusted", "trusted"),
    # -- strided unary Core kernels ----------------------------------------
    "tf_core_relu": ("trusted", "trusted"),
    "tf_core_sqrt": ("trusted", "trusted"),
    "tf_core_reciprocal": ("trusted", "trusted"),
    "tf_core_exp": ("trusted", "trusted"),
    "tf_core_log": ("trusted", "trusted"),
    "tf_core_contiguous_copy": ("trusted", "trusted"),
    # -- strided binary Core kernels ---------------------------------------
    "tf_core_add": ("trusted", "trusted", "trusted"),
    "tf_core_subtract": ("trusted", "trusted", "trusted"),
    "tf_core_multiply": ("trusted", "trusted", "trusted"),
    "tf_core_relu_backward": ("trusted", "trusted", "trusted"),
    # -- reduction and its scatter dual ------------------------------------
    "tf_core_sum": ("trusted", "trusted", "trusted"),
    "tf_core_narrow_backward": ("trusted", "trusted", "trusted"),
    # -- cross-entropy class labels: int64, but deliberately checked -------
    "tf_core_cross_entropy_forward": ("checked",),
    "tf_core_cross_entropy_backward": ("checked",),
}


def _carrier_length(carrier):
    """How many int64 entries a trusted carrier describes.

    Both producers can answer this, which is more than the checked binding
    ever offered — an ``ndpointer`` argument's length was invisible to
    everything, which is precisely the gap H3's sanitizer work exposed. A
    ``c_int64`` vector carries its length in its own type; a cached pointer
    carries the owning NumPy array NumPy attached to it when ``data_as``
    built it, and that array's length is the view's rank."""
    owner = getattr(carrier, "_arr", None)
    if owner is not None:
        return len(owner)
    return len(carrier)


def _array_kinds(function):
    """The category of each array argument of a configured function."""
    kinds = []
    for argtype in function.argtypes or ():
        name = getattr(argtype, "__name__", "")
        if name.startswith("ndpointer"):
            kinds.append("checked")
        elif argtype is cpp._LAYOUT_POINTER:
            kinds.append("trusted")
    return tuple(kinds)


def _configured_names():
    source = BACKEND_FILE.read_text(encoding="utf-8")
    return sorted(set(re.findall(r"\btf_[a-z0-9_]+\b", source)))


@needs_native
def test_the_binding_inventory_is_exactly_the_declared_one():
    """Every configured export's array arguments, position by position."""
    library = cpp._require_library()
    seen = {}
    for name in _configured_names():
        function = getattr(library, name, None)
        if function is None or function.argtypes is None:
            continue
        kinds = _array_kinds(function)
        if kinds:
            seen[name] = kinds
    assert seen == EXPECTED_ARRAY_BINDINGS


@needs_native
def test_every_configured_export_declares_its_argument_types():
    """No function is reachable with ctypes' default (unchecked) argument
    handling — a missing ``argtypes`` would silently accept anything."""
    library = cpp._require_library()
    for name in _configured_names():
        function = getattr(library, name, None)
        if function is None:
            continue
        assert function.argtypes is not None, name


@needs_native
def test_the_two_bindings_are_the_only_array_argument_types():
    """Exactly two array bindings exist, and no *layout* or *kernel-data*
    position uses a bare ``c_void_p`` — which would express the ABI less
    precisely than a typed pointer and would accept an integer address.

    The three retyped transfer positions are the deliberate exception
    (Phase I, milestone I2): their C parameter genuinely **is** ``void*``,
    because the storage handle's dtype tag is what says how the bytes are
    read, so ``c_void_p`` is the precise declaration rather than a looser
    one. Their check did not disappear — it moved to ``_host_pointer``,
    which runs the per-dtype ``ndpointer`` binding at every call.
    """
    library = cpp._require_library()
    layout_positions = 0
    checked_positions = 0
    for name in _configured_names():
        function = getattr(library, name, None)
        if function is None or function.argtypes is None:
            continue
        for argtype in function.argtypes:
            if argtype is cpp._LAYOUT_POINTER:
                layout_positions += 1
            elif getattr(argtype, "__name__", "").startswith("ndpointer"):
                checked_positions += 1
    assert layout_positions == 32
    # 25 before I2; the three transfer host positions moved to the call
    # site, and nothing else changed.
    assert checked_positions == 22


@needs_native
def test_the_checked_bindings_are_unchanged_ndpointers():
    """The checked binding is exactly what it always was: an ndpointer of
    the exact dtype, requiring C-contiguity."""
    assert cpp._CHECKED_F64_ARRAY is np.ctypeslib.ndpointer(
        dtype=np.float64, flags="C_CONTIGUOUS")
    assert cpp._CHECKED_I64_ARRAY is np.ctypeslib.ndpointer(
        dtype=np.int64, flags="C_CONTIGUOUS")
    assert cpp._CHECKED_F64_ARRAY._dtype_ == np.dtype(np.float64)
    assert cpp._CHECKED_I64_ARRAY._dtype_ == np.dtype(np.int64)
    # Phase I, milestone I2: the float32 sibling, built the same way.
    assert cpp._CHECKED_F32_ARRAY is np.ctypeslib.ndpointer(
        dtype=np.float32, flags="C_CONTIGUOUS")
    assert cpp._CHECKED_F32_ARRAY._dtype_ == np.dtype(np.float32)
    assert cpp._CHECKED_HOST_ARRAYS == {
        "float64": cpp._CHECKED_F64_ARRAY,
        "float32": cpp._CHECKED_F32_ARRAY,
    }


@needs_native
def test_a_retyped_transfer_position_is_still_checked_per_dtype():
    """The three transfer boundaries declare ``c_void_p`` from I2, and the
    check that used to sit in that argtypes slot moved to ``_host_pointer``
    — where it can be chosen from the storage's dtype, which one slot could
    never express. It is the *same* check, so it still rejects everything
    it always rejected, and it now also rejects the *other* supported
    dtype."""
    library = cpp._require_library()
    for name in ("tf_storage_copy_from", "tf_storage_copy_to",
                 "tf_storage_materialize"):
        argtypes = getattr(library, name).argtypes
        assert argtypes[1] is ctypes.c_void_p, name

    good = np.zeros(4, dtype=np.float64)
    # An ndarray of exactly the right dtype, C-contiguous: accepted, and
    # the pointer keeps its owner alive.
    pointer = cpp._host_pointer(good, "float64")
    assert isinstance(pointer, ctypes.c_void_p)
    assert getattr(pointer, "_arr", None) is not None

    for wrong, why in (
        (np.zeros(4, dtype=np.float32), "the other supported dtype"),
        (np.zeros(4, dtype=np.int64), "an integer array"),
        (np.zeros(4, dtype=">f8"), "a byte-swapped array"),
        (np.zeros((4, 4))[::2], "a non-contiguous array"),
        ([0.0, 1.0], "a Python list"),
        (b"\x00" * 32, "bytes"),
    ):
        with pytest.raises(TypeError):
            cpp._host_pointer(wrong, "float64")
    # ...and symmetrically for float32 storage: a float64 host buffer is
    # rejected rather than narrowed.
    with pytest.raises(TypeError):
        cpp._host_pointer(np.zeros(4, dtype=np.float64), "float32")
    assert isinstance(
        cpp._host_pointer(np.zeros(4, dtype=np.float32), "float32"),
        ctypes.c_void_p)
    # An unknown dtype names the ones that exist rather than guessing a
    # width.
    with pytest.raises(ValueError):
        cpp._host_pointer(good, "float16")


def test_the_layout_binding_is_a_typed_int64_pointer():
    """Not ``c_void_p``, and not an untyped address."""
    assert cpp._LAYOUT_POINTER._type_ is ctypes.c_int64
    assert issubclass(cpp._LAYOUT_POINTER, ctypes._Pointer)


# ==========================================================================
# 2. What the trusted binding still rejects
# ==========================================================================

@needs_native
@pytest.mark.parametrize("wrong", [
    pytest.param(lambda: np.asarray((2, 3), dtype=np.int64), id="numpy_int64"),
    pytest.param(lambda: np.asarray((2, 3), dtype=np.int32), id="numpy_int32"),
    pytest.param(lambda: np.asarray((2.0, 3.0)), id="numpy_float64"),
    pytest.param(lambda: [2, 3], id="python_list"),
    pytest.param(lambda: (2, 3), id="python_tuple"),
    pytest.param(lambda: 12345, id="python_int"),
    pytest.param(lambda: b"\x02\x00\x00\x00\x00\x00\x00\x00", id="bytes"),
    pytest.param(lambda: "23", id="str"),
    pytest.param(lambda: (ctypes.c_int32 * 2)(2, 3), id="c_int32_vector"),
    pytest.param(lambda: (ctypes.c_double * 2)(2.0, 3.0), id="c_double_vector"),
    pytest.param(
        lambda: np.asarray((2, 3), dtype=np.int64).ctypes.data_as(
            ctypes.POINTER(ctypes.c_int32)),
        id="c_int32_pointer"),
    pytest.param(lambda: ctypes.c_void_p(1234), id="c_void_p"),
])
def test_a_trusted_position_rejects_everything_but_an_int64_carrier(wrong):
    """ctypes still type-checks every call. The trusted binding is not a
    hole: it accepts a ``c_int64`` pointer or a ``c_int64`` vector and
    rejects a NumPy array (of any dtype), a sequence, an integer, bytes, a
    differently typed pointer or vector, and an untyped ``c_void_p``.

    A NumPy array being rejected is a *deliberate* consequence: it makes
    it impossible to reach a trusted position by accident from code that
    still thinks in terms of the old checked binding."""
    library = cpp._require_library()
    source = Core.from_array(np.arange(6.0).reshape(2, 3))
    out = Core.zeros((3,))
    try:
        good = cpp._layout_vector((0, 1))
        with pytest.raises(ctypes.ArgumentError):
            library.tf_core_sum(
                source.storage._require_open(), out.storage._require_open(),
                wrong(), wrong(), good, 0, 2)
    finally:
        out.close()
        source.close()


@needs_native
def test_a_trusted_position_accepts_both_producers_and_nothing_else():
    """The two producers are interchangeable at the boundary and both
    really work — this is the positive half of the test above."""
    library = cpp._require_library()
    values = np.arange(6.0).reshape(2, 3)
    source = Core.from_array(values)
    try:
        for shape_arg, strides_arg in (
            (source._layout_pointers()[0], source._layout_pointers()[1]),
            (cpp._layout_vector((2, 3)), cpp._layout_vector((3, 1))),
        ):
            out = Core.zeros((3,))
            try:
                library.tf_core_sum(
                    source.storage._require_open(),
                    out.storage._require_open(),
                    shape_arg, strides_arg, cpp._layout_vector((0, 1)), 0, 2)
                assert np.array_equal(out.to_numpy(), values.sum(axis=0))
            finally:
                out.close()
    finally:
        source.close()


# ==========================================================================
# 3. The public raw-buffer contract is exactly what it was
# ==========================================================================

RAW_HELPERS = (
    ("elementwise_add", 2), ("elementwise_subtract", 2),
    ("elementwise_multiply", 2), ("elementwise_divide", 2),
    ("relu", 1), ("matmul", 2), ("matmul_tiled", 2),
)


def _raw_call(name, arity, first):
    helper = getattr(cpp, name)
    # Every raw helper normalizes with ascontiguousarray first, so the
    # reference shape is the normalized one, whatever was passed in.
    normalized = np.ascontiguousarray(first, dtype=np.float64)
    if name in ("matmul", "matmul_tiled"):
        return helper(first, np.ones((normalized.shape[-1], 2)))
    if arity == 1:
        return helper(first)
    return helper(first, np.ones_like(normalized))


@needs_native
@pytest.mark.parametrize("name,arity", RAW_HELPERS)
@pytest.mark.parametrize("build,label", [
    (lambda: np.ones((4, 4), dtype=np.float32), "float32"),
    (lambda: np.ones((4, 4), dtype=np.int64), "int64"),
    (lambda: np.ones((4, 4), dtype=np.int32), "int32"),
    (lambda: np.asfortranarray(np.ones((4, 4))), "fortran_order"),
    (lambda: np.ones((8, 8))[::2, ::2], "sliced_non_contiguous"),
    (lambda: np.ones((4, 4)).T, "transposed"),
    (lambda: np.ones((4, 4), dtype=">f8"), "big_endian"),
    (lambda: np.ones((4, 4), dtype="<f8"), "little_endian"),
    (lambda: [[1.0] * 4] * 4, "python_list"),
    (lambda: np.ones((4, 4)).astype(object), "object_dtype"),
])
def test_a_raw_public_helper_accepts_exactly_what_it_always_did(
    name, arity, build, label
):
    """The raw helpers normalize their inputs with
    ``numpy.ascontiguousarray(..., dtype=float64)`` before the checked
    binding ever sees them, so they accept every one of these and produce
    a correct float64 result. H7 changed nothing here — none of these
    functions has a trusted position — and this pins that."""
    values = build()
    result = _raw_call(name, arity, values)
    assert isinstance(result, np.ndarray)
    assert result.dtype == np.float64
    assert result.flags.c_contiguous
    assert np.isfinite(result).all()


@needs_native
@pytest.mark.parametrize("bad", [
    "not an array",
    [[1.0, 2.0], [3.0]],
    [object()],
    {"a": 1.0},
])
def test_a_raw_public_helper_still_rejects_what_it_always_rejected(bad):
    """The rejections are the same objects and the same exception types:
    whatever ``ascontiguousarray(..., dtype=float64)`` cannot convert is
    rejected there, before the checked binding is ever reached. H7 moved
    nothing here, and this pins the boundary where the rejection happens."""
    with pytest.raises((ValueError, TypeError)):
        cpp.relu(bad)


@needs_native
def test_a_raw_public_helper_still_rejects_mismatched_shapes():
    with pytest.raises(ValueError, match="identical"):
        cpp.elementwise_add(np.ones((2, 3)), np.ones((3, 2)))
    with pytest.raises(ValueError, match="inner dimensions"):
        cpp.matmul(np.ones((2, 3)), np.ones((4, 5)))
    with pytest.raises(ValueError, match="2-D"):
        cpp.matmul(np.ones(3), np.ones((3, 3)))


@needs_native
def test_a_read_only_input_is_still_accepted_by_a_raw_helper():
    """The kernels only read their inputs, so a read-only array is legal
    and always was."""
    values = np.ones((4, 4))
    values.flags.writeable = False
    assert np.array_equal(cpp.relu(values), values)


@needs_native
def test_the_checked_binding_still_rejects_a_wrong_dtype_at_the_abi():
    """Driven at the ABI, below the helpers' normalization: the checked
    positions really do reject, which is why they are still checked."""
    library = cpp._require_library()
    good = np.ones(4, dtype=np.float64)
    for bad in (np.ones(4, dtype=np.float32), np.ones(4, dtype=np.int64),
                np.ones(4, dtype=">f8"), np.ones(8)[::2], [1.0] * 4, None):
        with pytest.raises((ctypes.ArgumentError, TypeError)):
            library.tf_relu(bad, good, 4)


@needs_native
def test_the_cross_entropy_labels_stay_a_checked_position():
    """Deliberately not trusted: a label array's required length comes
    from the logits, not from the array, so a dtype and layout check at
    the boundary is still doing work the construction site cannot."""
    library = cpp._require_library()
    assert library.tf_core_cross_entropy_forward.argtypes[2] is (
        cpp._CHECKED_I64_ARRAY)
    assert library.tf_core_cross_entropy_backward.argtypes[2] is (
        cpp._CHECKED_I64_ARRAY)
    logits = Core.from_array(np.array([[2.0, 1.0, 0.1], [0.5, 2.5, 0.2]]))
    try:
        # The public path still validates strictly, unchanged.
        with pytest.raises(TypeError):
            logits.cross_entropy_forward(np.array([0, 1], dtype=np.float64))
        with pytest.raises(ValueError):
            logits.cross_entropy_forward([0, 1, 2])
        with pytest.raises(ValueError):
            logits.cross_entropy_forward([0, 7])
        result = logits.cross_entropy_forward([0, 1])
        try:
            assert result.targets.dtype == np.int64
            assert not result.targets.flags.writeable
        finally:
            result.close()
    finally:
        logits.close()


# ==========================================================================
# 4. Metadata length equals rank — the invariant ndpointer never checked
# ==========================================================================

SHAPES = [(), (5,), (3, 4), (2, 3, 4), (2, 3, 4, 5)]


def _core_of_shape(shape):
    """A core of exactly ``shape``. ``from_array`` cannot make a rank-0 one
    — ``numpy.ascontiguousarray`` has ``ndmin=1`` — so the scalar shape
    comes from ``full``, which is how the runtime builds one anyway."""
    if not shape:
        return Core.full((), 3.0)
    return Core.from_array(np.ones(shape))


@needs_native
@pytest.mark.parametrize("shape", SHAPES)
def test_a_views_layout_length_is_its_rank(shape):
    """Both cached arrays are exactly ``ndim`` long, for every rank the
    runtime can construct — rank 0 included, where both are empty and the
    kernels never dereference them."""
    core = _core_of_shape(shape)
    try:
        shape_array, strides_array = core._layout_arrays()
        assert len(shape_array) == core.ndim == len(shape)
        assert len(strides_array) == core.ndim
        assert shape_array.tolist() == list(core.shape)
        assert strides_array.tolist() == list(core.strides)
    finally:
        core.close()


@pytest.mark.parametrize("values", [(), (5,), (3, 4), (2, 3, 4), (2, 3, 4, 5)])
def test_an_operation_local_vector_length_is_its_input_length(values):
    """A vector carries its length in its own type, which is the property
    a raw pointer plus a separate ``ndim`` cannot have."""
    vector = cpp._layout_vector(values)
    assert len(vector) == len(values)
    assert list(vector) == list(values)
    assert vector._type_ is ctypes.c_int64


def test_the_layout_vector_producer_is_total():
    """It returns a usable carrier for every rank, and never ``None`` —
    which is the one value the trusted binding would accept and the C ABI
    could not survive. Rank 0 gives a valid empty vector."""
    for rank in range(0, 9):
        vector = cpp._layout_vector(tuple(range(1, rank + 1)))
        assert vector is not None
        assert len(vector) == rank


@needs_native
@pytest.mark.parametrize("shape", SHAPES)
def test_the_pointer_producer_is_total_and_never_none(shape):
    core = _core_of_shape(shape)
    try:
        pointers = core._layout_pointers()
        assert len(pointers) == 2
        for pointer in pointers:
            assert pointer is not None
            assert isinstance(pointer, cpp._LAYOUT_POINTER)
    finally:
        core.close()


@needs_native
@pytest.mark.parametrize("shape", SHAPES)
def test_the_pointers_address_the_arrays_element_for_element(shape):
    """The rank the kernel is told and the data the pointer addresses come
    from one object: reading ``ndim`` entries through the pointer yields
    exactly the view's own shape and strides, and the owning array NumPy
    attached to the pointer is exactly ``ndim`` long."""
    core = _core_of_shape(shape)
    try:
        shape_pointer, strides_pointer = core._layout_pointers()
        assert [shape_pointer[i] for i in range(core.ndim)] == list(core.shape)
        assert [strides_pointer[i]
                for i in range(core.ndim)] == list(core.strides)
        assert _carrier_length(shape_pointer) == core.ndim
        assert _carrier_length(strides_pointer) == core.ndim
    finally:
        core.close()


@needs_native
def test_every_production_kernel_call_agrees_on_rank(monkeypatch):
    """The structural version of the two tests above, over real calls: for
    every strided export invoked by a real workload, the ``ndim`` argument
    equals the length of every layout carrier passed with it.

    This is the invariant H3's sanitizer work showed the C ABI depends on
    and cannot check for itself — a two-element metadata array with
    ``ndim=3`` is an out-of-bounds read — so it is asserted here over the
    production call path rather than argued in prose."""
    library = cpp._require_library()
    # (function name, index of the ndim argument, indices of the carriers)
    strided = {
        "tf_core_relu": (5, (2, 3)), "tf_core_sqrt": (5, (2, 3)),
        "tf_core_reciprocal": (5, (2, 3)), "tf_core_exp": (5, (2, 3)),
        "tf_core_log": (5, (2, 3)), "tf_core_contiguous_copy": (5, (2, 3)),
        "tf_core_add": (8, (3, 4, 5)), "tf_core_subtract": (8, (3, 4, 5)),
        "tf_core_multiply": (8, (3, 4, 5)),
        "tf_core_relu_backward": (8, (3, 4, 5)),
        "tf_core_sum": (6, (2, 3, 4)),
        "tf_core_narrow_backward": (7, (2, 3, 4)),
        "tf_storage_materialize": (5, (2, 3)),
    }
    observed = set()

    for name, (ndim_index, carrier_indices) in strided.items():
        original = getattr(library, name)

        def checking(*args, _o=original, _n=name, _i=ndim_index,
                     _c=carrier_indices):
            rank = args[_i]
            for position in _c:
                length = _carrier_length(args[position])
                assert length == rank, (_n, position, length, rank)
            observed.add(_n)
            return _o(*args)

        monkeypatch.setattr(library, name, checking, raising=False)

    values = np.arange(24.0).reshape(2, 3, 4)
    base = Core.from_array(values)
    transposed = base.transpose(2, 1, 0)
    scalar = Core.full((), 2.0)
    try:
        for produced in (
            transposed.relu(), transposed.sqrt(), transposed.reciprocal(),
            transposed.exp(), transposed.log(), transposed.contiguous_copy(),
            base.add(scalar), base.subtract(scalar), base.multiply(scalar),
            base.sum(axis=1), base.sum(),
            transposed.relu_backward(transposed),
            base.narrow(0, 0, 1).contiguous_copy().narrow_backward(
                0, 1, (3, 3, 4)),
        ):
            produced.close()
        transposed.to_numpy()
    finally:
        scalar.close()
        transposed.close()
        base.close()

    assert observed == set(strided), sorted(set(strided) - observed)


# ==========================================================================
# 5. Pointer ownership and lifetime
# ==========================================================================

@needs_native
def test_a_cached_pointer_holds_its_owning_array_alive():
    """NumPy's own guarantee, relied on rather than assumed: ``data_as``
    stores the array on the pointer. Asserted with the cyclic collector
    disabled, so nothing here depends on a collection pass."""
    core = Core.from_array(np.arange(12.0).reshape(3, 4))
    gc.disable()
    try:
        shape_array, strides_array = core._layout_arrays()
        shape_pointer, strides_pointer = core._layout_pointers()
        assert shape_pointer._arr is shape_array
        assert strides_pointer._arr is strides_array
        # Drop every other reference we hold and read through the pointer.
        del shape_array, strides_array
        core.view._layout_cache = None
        assert [shape_pointer[i] for i in range(3 if False else 2)] == [3, 4]
        assert [strides_pointer[i] for i in range(2)] == [4, 1]
    finally:
        gc.enable()
        core.close()


@needs_native
def test_the_pointer_cache_introduces_no_reference_cycle():
    """A view holds its arrays and its pointers; each pointer holds its
    array. Nothing holds the view, so dropping one needs no collector."""
    gc.collect()
    core = Core.from_array(np.arange(12.0).reshape(3, 4))
    core._layout_pointers()
    view = core.view
    shape_array = core._layout_arrays()[0]
    referrers = [type(r).__name__ for r in gc.get_referrers(shape_array)]
    assert "NativeTensorView" in referrers or "tuple" in referrers
    core.close()
    del core, view, shape_array, referrers
    gc.collect()
    collected = gc.collect()
    assert collected == 0


@needs_native
def test_a_layout_pointer_carries_no_native_storage_and_keeps_none_alive(
    live_storages
):
    """The layout describes a tensor; it does not own one. Holding the
    pointers after closing the tensor leaks no native storage, and the
    pointers still address plain integers."""
    baseline = len(live_storages)
    core = Core.from_array(np.arange(12.0).reshape(3, 4))
    shape_pointer, strides_pointer = core._layout_pointers()
    core.close()
    assert len(live_storages) == baseline
    assert [shape_pointer[i] for i in range(2)] == [3, 4]
    assert [strides_pointer[i] for i in range(2)] == [4, 1]
    # And there is no route from a layout pointer back to native memory.
    for pointer in (shape_pointer, strides_pointer):
        assert not hasattr(pointer, "_storage")
        assert not isinstance(getattr(pointer, "_arr", None),
                              cpp.NativeStorage)


@needs_native
def test_an_operation_local_vector_lives_exactly_as_long_as_its_call(
    monkeypatch
):
    """The vector is a live local of the calling frame for the whole
    native call, and nothing retains it afterwards — no global cache, no
    id-keyed table, no attribute on any tensor."""
    library = cpp._require_library()
    original = library.tf_core_sum
    seen = []

    def capture(*args):
        seen.append(args[4])
        # Readable *during* the call, which is the property that matters.
        assert list(args[4]) == [0, 1]
        return original(*args)

    monkeypatch.setattr(library, "tf_core_sum", capture, raising=False)
    core = Core.from_array(np.arange(6.0).reshape(2, 3))
    try:
        core.sum(axis=0).close()
    finally:
        core.close()
    vector = seen[0]
    referrers = [r for r in gc.get_referrers(vector)
                 if not isinstance(r, (list, dict)) or r is not seen]
    assert not any(isinstance(r, cpp.NativeTensorView) for r in referrers)
    assert not any(isinstance(r, cpp.NativeTensorCore) for r in referrers)
    assert cpp.__dict__.get("_vector_cache") is None


def _backend_code_names():
    """Every attribute and plain name the backend module's *code* uses.

    Parsed rather than grepped, so a rule stated in a docstring — such as
    the recorded reason ``from_address`` is not used — cannot be mistaken
    for the thing it forbids."""
    import ast
    tree = ast.parse(BACKEND_FILE.read_text(encoding="utf-8"))
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute):
            names.add(node.attr)
        elif isinstance(node, ast.Name):
            names.add(node.id)
    return names


def test_the_module_holds_no_global_pointer_cache():
    """No pointer table keyed by object id, no interning, no thread-local
    state, and no owner-free pointer construction. The only pointers that
    persist are per-view attributes."""
    used = _backend_code_names()
    for banned in ("from_address", "from_buffer", "WeakValueDictionary",
                   "WeakKeyDictionary", "local", "lru_cache", "cache",
                   "byref", "addressof"):
        assert banned not in used, banned
    # ``id()`` is the classic way to key a pointer cache; the module must
    # not call it at all.
    assert "id" not in used
    module_level = [name for name in vars(cpp)
                    if not name.startswith("__")
                    and ("cache" in name.lower() or "pointer" in name.lower())]
    # ``_LAYOUT_POINTER`` is the trusted int64 binding; ``_host_pointer``
    # (Phase I, milestone I2) is a *function* that builds a fresh, owner-
    # attached ``c_void_p`` per call for the three retyped transfer
    # boundaries. Neither is a cache, and nothing else may join them.
    assert sorted(module_level) == ["_LAYOUT_POINTER", "_host_pointer"]
    assert callable(cpp._host_pointer)


# ==========================================================================
# 6. Immutability, views, and staleness
# ==========================================================================

@needs_native
def test_the_pointer_cache_is_built_once_and_reused():
    core = Core.from_array(np.arange(24.0).reshape(2, 3, 4))
    try:
        first = core._layout_pointers()
        assert core._layout_pointers() is first
        core.sum(axis=1).close()
        assert core._layout_pointers() is first
    finally:
        core.close()


@needs_native
def test_the_pointer_cache_is_lazy():
    """The contiguous fast path takes a flat count, so it never builds
    layout metadata of either kind."""
    core = Core.from_array(np.ones((8, 8)))
    other = Core.from_array(np.ones((8, 8)))
    try:
        assert core.view._layout_pointer_cache is None
        core.add(other).close()
        core.relu().close()
        assert core.view._layout_pointer_cache is None
        assert core.view._layout_cache is None
        core.to_numpy()
        assert core.view._layout_pointer_cache is not None
        assert core.view._layout_cache is not None
    finally:
        other.close()
        core.close()


@needs_native
def test_a_view_gets_its_own_pointers_not_its_parents():
    base = Core.from_array(np.arange(24.0).reshape(2, 3, 4))
    transposed = base.transpose(2, 0, 1)
    narrowed = base.narrow(2, 1, 2)
    try:
        pointers = [c._layout_pointers() for c in (base, transposed, narrowed)]
        addresses = [(ctypes.addressof(p[0].contents),
                      ctypes.addressof(p[1].contents)) for p in pointers]
        assert len(set(a for pair in addresses for a in pair)) == 6
        for core, pair in zip((base, transposed, narrowed), pointers):
            assert [pair[0][i] for i in range(core.ndim)] == list(core.shape)
            assert [pair[1][i] for i in range(core.ndim)] == list(core.strides)
    finally:
        narrowed.close()
        transposed.close()
        base.close()


@needs_native
def test_chained_views_each_carry_their_own_correct_metadata():
    values = np.arange(120.0).reshape(2, 3, 4, 5)
    base = Core.from_array(values)
    step1 = base.transpose(3, 1, 0, 2)
    step2 = step1.narrow(0, 1, 3)
    step3 = step2.transpose(1, 0, 3, 2)
    try:
        for core in (base, step1, step2, step3):
            shape_pointer, strides_pointer = core._layout_pointers()
            assert [shape_pointer[i]
                    for i in range(core.ndim)] == list(core.shape)
            assert [strides_pointer[i]
                    for i in range(core.ndim)] == list(core.strides)
        expected = values.transpose(3, 1, 0, 2)[1:4].transpose(1, 0, 3, 2)
        assert np.array_equal(step3.to_numpy(), expected)
    finally:
        for core in (step3, step2, step1, base):
            core.close()


@needs_native
def test_a_layout_can_never_go_stale_because_nothing_reassigns_it():
    """Staleness is impossible by construction, not prevented by
    invalidation: every layout-changing operation returns a *new* view, so
    a view's shape, strides, arrays, and pointers are each assigned once."""
    core = Core.from_array(np.arange(24.0).reshape(2, 3, 4))
    try:
        first = core._layout_pointers()
        for derived in (core.reshape((6, 4)), core.transpose(),
                        core.narrow(0, 0, 1)):
            try:
                assert derived.view is not core.view
                assert derived._layout_pointers() is not first
            finally:
                derived.close()
        assert core._layout_pointers() is first
        assert core.shape == (2, 3, 4) and core.strides == (12, 4, 1)
    finally:
        core.close()
    # And no invalidation API exists, because none is needed.
    for banned in ("invalidate", "clear_layout", "reset_layout",
                   "refresh_layout"):
        assert not hasattr(cpp.NativeTensorView, banned), banned


@needs_native
def test_closing_a_tensor_leaves_its_metadata_readable_and_its_calls_refused():
    """Metadata stays readable after close — the long-standing contract —
    while every data operation still raises."""
    core = Core.from_array(np.arange(12.0).reshape(3, 4))
    core._layout_pointers()
    core.close()
    assert core.shape == (3, 4) and core.strides == (4, 1) and core.ndim == 2
    assert core._layout_arrays()[0].tolist() == [3, 4]
    assert [core._layout_pointers()[0][i] for i in range(2)] == [3, 4]
    with pytest.raises(RuntimeError, match="closed"):
        core.to_numpy()
    with pytest.raises(RuntimeError, match="closed"):
        core.sum(axis=0)
    with pytest.raises(RuntimeError, match="closed"):
        core.contiguous_copy()


@needs_native
def test_a_view_over_closed_storage_refuses_the_kernel_not_the_pointer():
    """Closing the owner does not turn a metadata pointer into a way in:
    the storage open check still runs and raises before any native call."""
    owner = Core.from_array(np.arange(12.0).reshape(3, 4))
    view = owner.transpose()
    view._layout_pointers()  # metadata built while the storage was open
    owner.close()
    try:
        with pytest.raises(RuntimeError, match="closed"):
            view.to_numpy()
        with pytest.raises(RuntimeError, match="closed"):
            view.contiguous_copy()
        with pytest.raises(RuntimeError, match="closed"):
            view.sum(axis=0)
    finally:
        view.close()


# ==========================================================================
# 7. The error contract
# ==========================================================================

@needs_native
def test_the_errcheck_hook_is_attached_to_exactly_the_checked_kernels():
    library = cpp._require_library()
    for name in _configured_names():
        function = getattr(library, name, None)
        if function is None or function.argtypes is None:
            continue
        has_hook = getattr(function, "errcheck", None) is not None
        assert has_hook == (name in cpp._CHECKED_KERNELS), name


@needs_native
def test_a_native_failure_through_a_trusted_call_still_raises_correctly():
    """The kernels that self-validate still reject through the same hook,
    with the same exception type and message, when reached with the
    trusted binding."""
    library = cpp._require_library()
    source = Core.from_array(np.arange(4.0))
    destination = Core.from_array(np.full(4, 7777.5))
    try:
        src, dst = (source.storage._require_open(),
                    destination.storage._require_open())
        with pytest.raises(ValueError, match="span exceeds its storage"):
            library.tf_core_exp(src, dst, cpp._layout_vector([4]),
                                cpp._layout_vector([2]), 0, 1)
        with pytest.raises(ValueError, match="non-positive dimension"):
            library.tf_core_exp(src, dst, cpp._layout_vector([0]),
                                cpp._layout_vector([1]), 0, 1)
        with pytest.raises(ValueError, match="negative offset"):
            library.tf_core_exp(src, dst, cpp._layout_vector([4]),
                                cpp._layout_vector([1]), -1, 1)
        # Nothing was written by any rejected call.
        assert np.array_equal(destination.to_numpy(), np.full(4, 7777.5))
    finally:
        destination.close()
        source.close()


@needs_native
def test_alternating_success_and_failure_leaves_no_stale_error():
    """A failure does not contaminate the next call, and a success does
    not hide a later failure."""
    library = cpp._require_library()
    source = Core.from_array(np.arange(1.0, 5.0))
    destination = Core.zeros((4,))
    try:
        src, dst = (source.storage._require_open(),
                    destination.storage._require_open())
        good_shape, good_strides = (cpp._layout_vector([4]),
                                    cpp._layout_vector([1]))
        for _ in range(4):
            library.tf_core_exp(src, dst, good_shape, good_strides, 0, 1)
            assert library.tf_last_error_code() == cpp.TF_OK
            with pytest.raises(ValueError):
                library.tf_core_exp(src, dst, good_shape,
                                    cpp._layout_vector([2]), 0, 1)
            assert library.tf_last_error_code() == cpp.TF_OK
            library.tf_core_exp(src, dst, good_shape, good_strides, 0, 1)
            assert np.allclose(destination.to_numpy(), np.exp(np.arange(1.0, 5.0)))
    finally:
        destination.close()
        source.close()


@needs_native
def test_an_allocation_failure_in_a_trusted_operation_releases_everything(
    live_storages
):
    """The failure contract is unchanged: nothing half-built escapes and
    live storage returns exactly to baseline, with no collector pass."""
    if not cpp.fault_injection_available():
        pytest.skip("fault injection not compiled in")
    core = Core.from_array(np.arange(24.0).reshape(2, 3, 4))
    try:
        baseline = len(live_storages)
        before = core.to_numpy().copy()
        cpp._arm_alloc_failure(1)
        with pytest.raises(MemoryError):
            core.transpose().contiguous_copy()
        cpp._arm_alloc_failure(0)
        assert len(live_storages) == baseline
        assert np.array_equal(core.to_numpy(), before)
        # Still fully usable afterwards.
        core.transpose().contiguous_copy().close()
        core.sum(axis=1).close()
        assert len(live_storages) == baseline
    finally:
        core.close()


@needs_native
@pytest.mark.parametrize("error", [RuntimeError, MemoryError,
                                   KeyboardInterrupt])
def test_a_failure_building_the_pointers_releases_the_output(
    error, live_storages, monkeypatch
):
    """The pointer lookup sits inside the same cleanup boundary the layout
    arrays sat in before H7, so a failure there still releases the freshly
    allocated output."""
    core = Core.from_array(np.arange(24.0).reshape(2, 3, 4))
    try:
        baseline = len(live_storages)

        def exploding(*args, **kwargs):
            raise error("injected layout-pointer failure")

        monkeypatch.setattr(Core, "_layout_pointers", exploding)
        with pytest.raises(error):
            core.sum(axis=1)
        with pytest.raises(error):
            core.transpose().contiguous_copy()
        monkeypatch.undo()
        assert len(live_storages) == baseline
        core.sum(axis=1).close()
        assert len(live_storages) == baseline
    finally:
        core.close()


@needs_native
def test_a_failure_building_an_operation_local_vector_releases_the_output(
    live_storages, monkeypatch
):
    core = Core.from_array(np.arange(24.0).reshape(2, 3, 4))
    scalar = Core.full((), 2.0)
    try:
        baseline = len(live_storages)

        def exploding(*args, **kwargs):
            raise RuntimeError("injected layout-vector failure")

        monkeypatch.setattr(cpp, "_layout_vector", exploding)
        with pytest.raises(RuntimeError):
            core.sum(axis=1)
        with pytest.raises(RuntimeError):
            core.multiply(scalar)
        monkeypatch.undo()
        assert len(live_storages) == baseline
        core.multiply(scalar).close()
        assert len(live_storages) == baseline
    finally:
        scalar.close()
        core.close()


@needs_native
def test_repeated_calls_return_live_storage_exactly_to_baseline(live_storages):
    """Success and failure cycles, with no ``gc.collect()`` anywhere."""
    core = Core.from_array(np.arange(24.0).reshape(2, 3, 4))
    scalar = Core.full((), 2.0)
    try:
        baseline = len(live_storages)
        for _ in range(20):
            core.sum(axis=1).close()
            core.multiply(scalar).close()
            core.transpose().contiguous_copy().close()
            core.to_numpy()
            with pytest.raises(ValueError):
                core.sum(axis=9)
        assert len(live_storages) == baseline
    finally:
        scalar.close()
        core.close()


# ==========================================================================
# 8. Binding configuration is done once, at load
# ==========================================================================

@needs_native
def test_no_production_path_reconfigures_a_binding():
    """``argtypes``/``restype``/``errcheck`` are assigned exactly once,
    inside the two loader functions. Reconfiguring a shared function
    object per call would be a data race the project does not claim to be
    safe against, so it must not exist."""
    source = BACKEND_FILE.read_text(encoding="utf-8")
    for attribute in ("argtypes", "restype", "errcheck"):
        for line_number, line in enumerate(source.splitlines(), 1):
            if f".{attribute} =" not in line:
                continue
            # Every assignment must be inside _load_library or
            # _configure_error_contract, both of which run once per process.
            preceding = "\n".join(source.splitlines()[:line_number])
            owner = re.findall(r"^def (\w+)", preceding, re.M)[-1]
            assert owner in ("_load_library", "_configure_error_contract"), (
                attribute, line_number, owner)


@needs_native
def test_the_configured_bindings_are_stable_across_many_calls():
    """The function objects and their argument types are the same after a
    workload as before it — nothing swaps a binding in and out."""
    library = cpp._require_library()
    before = {name: (getattr(library, name).argtypes,
                     getattr(library, name).restype,
                     getattr(library, name).errcheck)
              for name in EXPECTED_ARRAY_BINDINGS}
    core = Core.from_array(np.arange(24.0).reshape(2, 3, 4))
    scalar = Core.full((), 2.0)
    try:
        for _ in range(5):
            core.sum(axis=1).close()
            core.multiply(scalar).close()
            core.transpose().exp().close()
    finally:
        scalar.close()
        core.close()
    for name, expected in before.items():
        function = getattr(library, name)
        assert (function.argtypes, function.restype,
                function.errcheck) == expected, name


@needs_native
def test_reentrant_calls_through_the_same_binding_are_correct():
    """A kernel call made from inside a wrapper of another kernel call
    uses the same configured function object and must not disturb it."""
    library = cpp._require_library()
    original = library.tf_core_sum
    inner_results = []
    depth = []

    def reentrant(*args):
        if not depth:                      # re-enter exactly once
            depth.append(1)
            nested = Core.from_array(np.arange(6.0).reshape(2, 3))
            try:
                inner = nested.sum(axis=0)
                try:
                    inner_results.append(inner.to_numpy().copy())
                finally:
                    inner.close()
            finally:
                nested.close()
        return original(*args)

    # Assigned and restored by *value*, never deleted: ctypes caches a
    # configured function object on the library, and deleting it would make
    # the next lookup rebuild an **unconfigured** one that accepts anything.
    library.tf_core_sum = reentrant
    try:
        core = Core.from_array(np.arange(12.0).reshape(3, 4))
        try:
            outer = core.sum(axis=0)
            try:
                assert np.array_equal(outer.to_numpy(),
                                      np.arange(12.0).reshape(3, 4).sum(axis=0))
            finally:
                outer.close()
        finally:
            core.close()
    finally:
        library.tf_core_sum = original
    assert inner_results
    assert library.tf_core_sum is original
    assert library.tf_core_sum.argtypes is not None
    assert np.array_equal(inner_results[0], np.array([3.0, 5.0, 7.0]))


# ==========================================================================
# 9. Numerical parity: one milestone, two contracts
# ==========================================================================
#
# H7's guarantee is that swapping the layout binding **changed no
# arithmetic**. NumPy is not the right witness for that claim, and using
# it as one was an overclaim this section now corrects.
#
# * For the operations IEEE-754 **specifies** — addition, subtraction,
#   multiplication, square root, division (so ``reciprocal``), a value
#   copy, and the kernel's own ``x > 0 ? x : 0`` — the correctly rounded
#   result is uniquely defined, every conforming implementation produces
#   it, and NumPy really is an exact oracle. Those keep raw-bit equality,
#   unchanged and unweakened.
#
# * ``exp`` and ``log`` are **not** specified that way. IEEE-754
#   recommends correct rounding for them and requires nothing, so every
#   platform ships a different near-correct implementation. TensorForge
#   calls ``std::exp``/``std::log`` (deliberately: H8 excluded both from
#   its templated traversal for this reason) while NumPy may use its own
#   SIMD kernel. Comparing those two bit-for-bit tests the platform, not
#   TensorForge.
#
# Measured, on identical inputs and identical C++ source:
#
#   | pair                                        | exp     | log    |
#   |---------------------------------------------|---------|--------|
#   | Windows UCRT vs Linux glibc 2.39, 5,000+ in. | 18 diff | 6 diff |
#   | worst distance in either                     | 1 ULP   | 1 ULP  |
#   | each implementation vs correctly rounded     | <=1 ULP | <=1 ULP|
#
# Neither library is universally the correctly rounded one (UCRT wins
# some cases, glibc others), and within one platform the two agree
# perfectly — which is why this only ever fails on a *different* machine
# from the one a change was written on. So the transcendental contract
# below is: special values exactly, ordinary finite values within
# ``TRANSCENDENTAL_ULP`` — plus, as the actual defect guard, an
# **absolute** accuracy check against a correctly rounded reference that
# does not involve NumPy at all.
#
# The H7 claim itself is proved where it belongs, in
# ``test_the_trusted_and_checked_bindings_agree_bit_for_bit``: the same
# kernel, the same call site, the same inputs, the two bindings.

# One representable step. This is the tightest bound the evidence
# supports and it is not a guess: two independent shipped libm
# implementations were measured against each other over 10,000+ inputs
# and never differed by more than one step, and each stayed within one
# step of the correctly rounded result. A larger bound would start to
# hide real error; a bound of zero is the platform-dependent claim that
# failed.
TRANSCENDENTAL_ULP = 1


def _same_bits(a, b):
    return np.array_equal(np.asarray(a, dtype=np.float64).view(np.int64),
                          np.asarray(b, dtype=np.float64).view(np.int64))


def _monotone(value):
    """A float64's position on one monotone integer line.

    IEEE-754 orders positive floats by their bit patterns already;
    negatives run the other way in sign-magnitude, so they are reflected.
    ``-0.0`` maps onto the same point as ``+0.0``, which is correct for a
    *numerical* distance: they are the same number. Sign-of-zero is a bit
    property and is asserted with ``_same_bits`` instead."""
    bits = int(np.float64(value).view(np.int64))
    return -(2 ** 63) - bits if bits < 0 else bits


def _ulp_distance(a, b):
    """How many representable float64 steps apart two values are.

    Neighbouring floats are 1 apart, a value is 0 from itself, and the
    count is exact across the denormal/normal boundary and across zero.
    NaN is unordered, so it has no distance to anything and is rejected
    rather than given a meaningless number — NaN is checked by position
    and quietness instead."""
    if np.isnan(a) or np.isnan(b):
        raise ValueError("ULP distance is undefined for NaN")
    return abs(_monotone(a) - _monotone(b))


def _is_quiet_nan(value):
    """A NaN whose most significant mantissa bit is set (IEEE-754 §6.2.1)."""
    bits = int(np.float64(value).view(np.int64)) & ((1 << 64) - 1)
    return np.isnan(value) and bool(bits & (1 << 51))


def _assert_within_ulp(produced, expected, limit, label):
    """The transcendental contract, elementwise. Returns the worst
    distance seen so a caller can report how much of the budget is real.

    Only *ordinary finite non-zero* results get the tolerance. A NaN must
    be in the same place on both sides, an infinity must match exactly
    (sign included), and a zero must match exactly — the distance cannot
    see a zero's sign, so it would silently accept ``-0.0`` for ``+0.0``
    and that is precisely the kind of thing this suite exists to catch."""
    produced = np.asarray(produced, dtype=np.float64)
    expected = np.asarray(expected, dtype=np.float64)
    assert produced.shape == expected.shape, label
    flat_produced, flat_expected = produced.ravel(), expected.ravel()
    assert np.array_equal(np.isnan(flat_produced), np.isnan(flat_expected)), (
        f"{label}: NaN positions differ")
    worst = 0
    for index in range(flat_produced.size):
        got, want = flat_produced[index], flat_expected[index]
        if np.isnan(got):
            assert _is_quiet_nan(got), f"{label}[{index}]: signalling NaN"
            continue
        if np.isinf(got) or np.isinf(want) or got == 0.0 or want == 0.0:
            assert _same_bits(got, want), (
                f"{label}[{index}]: {got!r} is not exactly {want!r}")
            continue
        distance = _ulp_distance(got, want)
        assert distance <= limit, (
            f"{label}[{index}]: {got!r} vs {want!r} is {distance} ULP apart, "
            f"over the {limit} ULP contract")
        worst = max(worst, distance)
    return worst


def _reproducer_values(shape):
    """The exact array the CI failure was reported on: seeded standard
    normals with the special-value matrix written over the front."""
    rng = np.random.default_rng(20260801)
    values = rng.standard_normal(shape)
    specials = np.array([0.0, -0.0, np.inf, -np.inf, np.nan, 1e308, -1e308,
                         5e-324, 2.2250738585072014e-308])
    count = min(values.size, specials.size)
    values.ravel()[:count] = specials[:count]
    return values


# -- the ULP helper's own contract -----------------------------------------

@pytest.mark.parametrize("a,b,expected", [
    pytest.param(1.5, 1.5, 0, id="equal_positive"),
    pytest.param(-1.5, -1.5, 0, id="equal_negative"),
    pytest.param(1e308, 1e308, 0, id="equal_huge"),
    pytest.param(1.0, float(np.nextafter(1.0, np.inf)), 1,
                 id="neighbouring_positive"),
    pytest.param(-1.0, float(np.nextafter(-1.0, -np.inf)), 1,
                 id="neighbouring_negative"),
    pytest.param(-1.0, float(np.nextafter(-1.0, np.inf)), 1,
                 id="neighbouring_negative_toward_zero"),
    pytest.param(0.0, -0.0, 0, id="signed_zeros_are_one_number"),
    pytest.param(0.0, 5e-324, 1, id="zero_to_smallest_denormal"),
    pytest.param(-0.0, -5e-324, 1, id="negative_zero_to_smallest_denormal"),
    pytest.param(5e-324, -5e-324, 2, id="across_zero"),
    pytest.param(2.225073858507201e-308, 2.2250738585072014e-308, 1,
                 id="largest_denormal_to_smallest_normal"),
    pytest.param(np.inf, np.inf, 0, id="equal_infinities"),
    pytest.param(-np.inf, -np.inf, 0, id="equal_negative_infinities"),
    pytest.param(np.inf, 1.7976931348623157e308, 1,
                 id="infinity_is_one_step_past_the_largest_finite"),
    pytest.param(np.inf, -np.inf, 2 ** 64 - 2 ** 53,
                 id="opposite_infinities"),
])
def test_the_ulp_helper_counts_representable_steps(a, b, expected):
    """The helper the transcendental contract is expressed in, pinned
    before anything is asserted with it. Symmetric, exact at the
    denormal boundary, and unsurprised by zeros or infinities."""
    assert _ulp_distance(a, b) == expected
    assert _ulp_distance(b, a) == expected


@pytest.mark.parametrize("a,b", [
    (np.nan, 1.0), (1.0, np.nan), (np.nan, np.nan), (np.nan, np.inf),
])
def test_the_ulp_helper_refuses_nan(a, b):
    """NaN is unordered: there is no honest number of steps to it, so the
    helper raises rather than returning one. NaN is checked by position
    and quietness, which is what the operations actually promise."""
    with pytest.raises(ValueError, match="undefined for NaN"):
        _ulp_distance(a, b)


def test_the_ulp_helper_rejects_a_signed_zero_swap_through_same_bits():
    """The distance treats ``+0.0`` and ``-0.0`` as one number, so the
    exactness of a zero is somebody else's job — and that job is done."""
    assert _ulp_distance(0.0, -0.0) == 0
    assert not _same_bits(0.0, -0.0)
    with pytest.raises(AssertionError):
        _assert_within_ulp(np.array([0.0]), np.array([-0.0]), 1, "zero")


def test_the_ulp_assertion_helper_enforces_the_stated_boundary():
    """One step passes at a one-step budget, two steps do not, and a NaN
    in the wrong place fails however small the distances are."""
    one = np.array([1.0])
    _assert_within_ulp(one, np.array([np.nextafter(1.0, np.inf)]), 1, "one")
    with pytest.raises(AssertionError, match="2 ULP apart"):
        _assert_within_ulp(
            one, np.array([np.nextafter(np.nextafter(1.0, np.inf), np.inf)]),
            1, "two")
    with pytest.raises(AssertionError, match="NaN positions differ"):
        _assert_within_ulp(np.array([np.nan]), one, 1, "nan")
    with pytest.raises(AssertionError):
        _assert_within_ulp(np.array([np.inf]), np.array([-np.inf]), 1, "inf")


# -- the exact contract, unweakened ----------------------------------------

# The operations IEEE-754 defines exactly, with the reference NumPy must
# agree with bit for bit on every platform. relu's reference is the
# kernel's own long-standing rule (`x > 0 ? x : 0`, so a NaN gives 0),
# not numpy.maximum, which propagates NaN. That difference predates
# Phase H entirely.
EXACT_UNARY = (
    ("sqrt", np.sqrt),
    ("reciprocal", lambda x: 1.0 / x),
    ("relu", lambda x: np.where(x > 0.0, x, 0.0)),
)


@needs_native
@pytest.mark.parametrize("shape", [(3,), (2, 3), (2, 3, 4), (2, 3, 4, 5)])
def test_every_exact_trusted_operation_matches_numpy_bit_for_bit(shape):
    """H7 changed no arithmetic. Every operation that crosses a trusted
    position **and** has a uniquely defined IEEE-754 result is compared
    against NumPy as raw bits, over contiguous and strided operands and
    the whole special-value matrix.

    This is the original assertion, kept exactly as strict as it was for
    every operation it is actually true of. ``exp`` and ``log`` moved to
    the transcendental contract below; nothing else did."""
    values = _reproducer_values(shape)
    base = Core.from_array(values)
    transposed = base.T
    scalar = Core.full((), 2.5)
    try:
        for core, host in ((base, values), (transposed, values.T)):
            for name, reference in EXACT_UNARY:
                produced = getattr(core, name)()
                try:
                    with np.errstate(all="ignore"):
                        expected = reference(host)
                    assert _same_bits(produced.to_numpy(), expected), name
                finally:
                    produced.close()
            copied = core.contiguous_copy()
            try:
                assert _same_bits(copied.to_numpy(), host)
            finally:
                copied.close()
            assert _same_bits(core.to_numpy(), host)
        with np.errstate(all="ignore"):
            for name, op in (("add", np.add), ("subtract", np.subtract),
                             ("multiply", np.multiply)):
                produced = getattr(base, name)(scalar)
                try:
                    assert _same_bits(produced.to_numpy(), op(values, 2.5)), name
                finally:
                    produced.close()
    finally:
        scalar.close()
        transposed.close()
        base.close()


# -- the transcendental contract -------------------------------------------

@needs_native
@pytest.mark.parametrize("shape", [(3,), (2, 3), (2, 3, 4), (2, 3, 4, 5)])
@pytest.mark.parametrize("name,reference", [
    pytest.param("exp", np.exp, id="exp"),
    pytest.param("log", np.log, id="log"),
])
def test_the_transcendental_operations_agree_with_numpy_within_one_ulp(
    shape, name, reference
):
    """``exp`` and ``log`` against NumPy as an external mathematical
    oracle, over exactly the arrays and layouts the exact contract above
    uses. Special values are still exact; ordinary finite results are
    allowed the one representable step two shipped libm implementations
    were measured to differ by."""
    values = _reproducer_values(shape)
    base = Core.from_array(values)
    transposed = base.T
    try:
        for core, host in ((base, values), (transposed, values.T)):
            produced = getattr(core, name)()
            try:
                with np.errstate(all="ignore"):
                    expected = reference(host)
                _assert_within_ulp(produced.to_numpy(), expected,
                                   TRANSCENDENTAL_ULP, f"{name}{shape}")
            finally:
                produced.close()
    finally:
        transposed.close()
        base.close()


@needs_native
def test_the_transcendental_special_values_are_exact_not_approximate():
    """Everything the tolerance must never cover. These are IEEE-754
    *requirements* on exp and log, not rounding choices, so they are
    asserted as raw bits — a platform that got one of these wrong would
    have a real defect, and the ULP budget must not absorb it."""
    exp_inputs = np.array([0.0, -0.0, np.inf, -np.inf, np.nan,
                           1000.0, -1000.0, 710.0, -746.0])
    core = Core.from_array(exp_inputs)
    try:
        produced = core.exp()
        try:
            out = produced.to_numpy()
        finally:
            produced.close()
    finally:
        core.close()
    assert _same_bits(out[0], 1.0)            # exp(+0) == 1 exactly
    assert _same_bits(out[1], 1.0)            # exp(-0) == 1 exactly
    assert _same_bits(out[2], np.inf)         # +inf -> +inf
    assert _same_bits(out[3], 0.0)            # -inf -> +0, sign included
    assert np.isnan(out[4]) and _is_quiet_nan(out[4])
    assert _same_bits(out[5], np.inf)         # overflow -> +inf
    assert _same_bits(out[6], 0.0)            # underflow -> +0
    assert _same_bits(out[7], np.inf)         # just past the overflow edge
    assert _same_bits(out[8], 0.0)            # past the underflow edge

    log_inputs = np.array([1.0, 0.0, -0.0, -1.0, -np.inf, np.inf, np.nan])
    core = Core.from_array(log_inputs)
    try:
        produced = core.log()
        try:
            out = produced.to_numpy()
        finally:
            produced.close()
    finally:
        core.close()
    assert _same_bits(out[0], 0.0)            # log(1) == +0 exactly
    assert _same_bits(out[1], -np.inf)        # log(+0) == -inf
    assert _same_bits(out[2], -np.inf)        # log(-0) == -inf
    assert np.isnan(out[3]) and _is_quiet_nan(out[3])   # domain -> NaN
    assert np.isnan(out[4]) and _is_quiet_nan(out[4])   # -inf is negative
    assert _same_bits(out[5], np.inf)         # +inf -> +inf
    assert np.isnan(out[6]) and _is_quiet_nan(out[6])   # NaN propagates


@needs_native
@pytest.mark.parametrize("name", ["exp", "log"])
def test_the_transcendental_results_are_correctly_rounded_or_next_to_it(name):
    """The defect guard, and the reason the ULP budget above cannot hide
    anything: an **absolute** accuracy check against the correctly
    rounded result, computed in 60 decimal digits. NumPy is not involved,
    so this holds identically on every platform and would catch a wrong
    kernel, a wrong dispatch, or a lost operand — none of which lands
    within a step of the true value.

    One step, not zero: neither shipped libm this project runs on is
    correctly rounded everywhere (measured: glibc 2.39 mis-rounds 5 of
    5,001 exp inputs, Windows UCRT 19 of the same 5,001)."""
    from decimal import Decimal, getcontext
    getcontext().prec = 60

    values = _reproducer_values((2, 3, 4, 5)).ravel()
    if name == "log":
        values = np.abs(values)
    core = Core.from_array(values)
    try:
        produced = getattr(core, name)()
        try:
            out = produced.to_numpy()
        finally:
            produced.close()
    finally:
        core.close()

    checked, worst = 0, 0
    for index, x in enumerate(values):
        if not np.isfinite(x) or not np.isfinite(out[index]):
            continue
        if x == 0.0 or out[index] == 0.0:
            continue
        exact = Decimal(float(x)).exp() if name == "exp" else Decimal(float(x)).ln()
        reference = float(exact)
        distance = _ulp_distance(out[index], reference)
        assert distance <= TRANSCENDENTAL_ULP, (
            f"{name}({x!r}) = {out[index]!r} is {distance} ULP from the "
            f"correctly rounded {reference!r}")
        worst = max(worst, distance)
        checked += 1
    assert checked > 50, f"only {checked} finite {name} results were checked"
    assert worst <= TRANSCENDENTAL_ULP


@needs_native
def test_the_linux_ci_reproducer_case(monkeypatch):
    """The exact case GitHub Actions failed on — seed 20260801, shape
    (2, 3, 4, 5), operation ``exp`` — asserted under the corrected
    contract, and pinned so the reproducer cannot drift out of the suite.

    The near-tie element is what makes this shape the one that failed:
    ``exp(0.3470383329193902)`` lands 0.4956 ULP from the double it
    correctly rounds to, i.e. 0.0044 ULP from an exact rounding tie, so
    two implementations that are each within a step of the true value can
    legitimately land on either side of it.

    It also asserts what was *not* involved: no convolution export is
    reached, so this was never an H9 failure."""
    library = cpp._require_library()
    conv_exports = ("tf_core_conv2d_forward", "tf_core_conv2d_input_backward",
                    "tf_core_conv2d_weight_backward")
    originals = {name: getattr(library, name) for name in conv_exports}

    def tripwire(name):
        def call(*args):
            raise AssertionError(f"{name} was reached by the exp reproducer")
        return call

    for name in conv_exports:
        monkeypatch.setattr(library, name, tripwire(name), raising=False)

    values = _reproducer_values((2, 3, 4, 5))
    assert values.shape == (2, 3, 4, 5) and values.size == 120
    # The near-tie element, pinned by value and by position.
    assert values[1, 0, 0, 2] == 0.3470383329193902

    base = Core.from_array(values)
    transposed = base.T
    try:
        for core, host in ((base, values), (transposed, values.T)):
            produced = core.exp()
            try:
                with np.errstate(all="ignore"):
                    expected = np.exp(host)
                _assert_within_ulp(produced.to_numpy(), expected,
                                   TRANSCENDENTAL_ULP, "exp reproducer")
            finally:
                produced.close()
        # The correctly rounded value for the near-tie element, and the
        # one neighbour a conforming implementation may return instead.
        produced = base.exp()
        try:
            got = produced.to_numpy()[1, 0, 0, 2]
        finally:
            produced.close()
        assert got in (1.4148709604654022, 1.4148709604654024)
        assert _ulp_distance(1.4148709604654022, 1.4148709604654024) == 1
    finally:
        transposed.close()
        base.close()

    monkeypatch.undo()
    for name in conv_exports:
        assert getattr(library, name) is originals[name]


# -- the H7 claim itself: two bindings, one kernel -------------------------

def _checked_binding(name):
    """A second, **test-local** binding of one export that uses the
    checked ``numpy.ctypeslib.ndpointer`` argument type H7 replaced for
    layout metadata — the pre-H7 declaration, rebuilt from the live one
    by swapping exactly that argument type so the two cannot drift apart.

    Built with ``ctypes.CFUNCTYPE`` against the already-loaded library, so
    it neither reloads anything nor touches the configured production
    function object. No exported symbol is added, no production selector
    exists, and nothing here is reachable from any public API."""
    library = cpp._require_library()
    original = getattr(library, name)
    argtypes = [cpp._CHECKED_I64_ARRAY if argtype is cpp._LAYOUT_POINTER
                else argtype for argtype in original.argtypes]
    assert cpp._LAYOUT_POINTER in original.argtypes, name
    assert cpp._CHECKED_I64_ARRAY in argtypes, name
    return ctypes.CFUNCTYPE(original.restype, *argtypes)((name, library))


# Every export with a trusted position, with the index of its ``ndim``
# argument and the indices of its layout carriers — the same table
# test_every_production_kernel_call_agrees_on_rank drives.
TRUSTED_EXPORTS = {
    "tf_core_relu": (5, (2, 3)), "tf_core_sqrt": (5, (2, 3)),
    "tf_core_reciprocal": (5, (2, 3)), "tf_core_exp": (5, (2, 3)),
    "tf_core_log": (5, (2, 3)), "tf_core_contiguous_copy": (5, (2, 3)),
    "tf_core_add": (8, (3, 4, 5)), "tf_core_subtract": (8, (3, 4, 5)),
    "tf_core_multiply": (8, (3, 4, 5)),
    "tf_core_relu_backward": (8, (3, 4, 5)),
    "tf_core_sum": (6, (2, 3, 4)),
    "tf_core_narrow_backward": (7, (2, 3, 4)),
    "tf_storage_materialize": (5, (2, 3)),
}


def _run_the_trusted_workload(core, scalar):
    """One call of every trusted export, from the production call sites,
    returning the results keyed by a label."""
    transposed = core.transpose(2, 1, 0)
    narrowed = core.narrow(0, 0, 1).contiguous_copy()
    results = {}
    try:
        for name in ("relu", "sqrt", "reciprocal", "exp", "log",
                     "contiguous_copy"):
            produced = getattr(transposed, name)()
            try:
                results[name] = produced.to_numpy().copy()
            finally:
                produced.close()
        for name in ("add", "subtract", "multiply"):
            produced = getattr(core, name)(scalar)
            try:
                results[name] = produced.to_numpy().copy()
            finally:
                produced.close()
        produced = transposed.relu_backward(transposed)
        try:
            results["relu_backward"] = produced.to_numpy().copy()
        finally:
            produced.close()
        for label, produced in (("sum_axis", core.sum(axis=1)),
                                ("sum_all", core.sum())):
            try:
                results[label] = produced.to_numpy().copy()
            finally:
                produced.close()
        produced = narrowed.narrow_backward(0, 1, (3, 3, 4))
        try:
            results["narrow_backward"] = produced.to_numpy().copy()
        finally:
            produced.close()
        results["materialize"] = transposed.to_numpy().copy()
    finally:
        narrowed.close()
        transposed.close()
    return results


@needs_native
def test_the_trusted_and_checked_bindings_agree_bit_for_bit(monkeypatch):
    """**This is H7's actual guarantee**, and the proof belongs here
    rather than in a comparison against a second math library.

    Every trusted export is driven twice from the *same* production call
    site, over the same inputs, into the same kernel — once through the
    shipped ``POINTER(c_int64)`` binding, and once through a test-local
    binding that converts each layout carrier back into the NumPy int64
    array the checked ``ndpointer`` declaration took before H7. Nothing
    about the C++ differs between the two runs; only the ctypes argument
    type does. Every result is compared as raw IEEE-754 bits.

    ``exp`` and ``log`` are in this comparison too, and they are **exact**
    here — which is the point. Their platform dependence is a property of
    the C library, not of the binding, so the moment both sides call the
    same ``std::exp`` the bits agree perfectly."""
    library = cpp._require_library()
    values = np.arange(-12.0, 12.0).reshape(2, 3, 4) / 3.0
    core = Core.from_array(values)
    scalar = Core.full((), 2.5)
    try:
        trusted = _run_the_trusted_workload(core, scalar)

        observed = set()
        for name, (ndim_index, carrier_indices) in TRUSTED_EXPORTS.items():
            checked = _checked_binding(name)

            def shim(*args, _checked=checked, _n=name, _i=ndim_index,
                     _c=carrier_indices):
                rank = args[_i]
                converted = list(args)
                for position in _c:
                    carrier = args[position]
                    converted[position] = np.asarray(
                        [carrier[index] for index in range(rank)],
                        dtype=np.int64)
                observed.add(_n)
                return _checked(*converted)

            monkeypatch.setattr(library, name, shim, raising=False)

        checked_results = _run_the_trusted_workload(core, scalar)
        monkeypatch.undo()
    finally:
        scalar.close()
        core.close()

    # Every trusted export really was driven through the checked binding.
    assert observed == set(TRUSTED_EXPORTS), sorted(
        set(TRUSTED_EXPORTS) - observed)
    assert set(checked_results) == set(trusted)
    for label, produced in trusted.items():
        assert _same_bits(produced, checked_results[label]), label
    # And no native error was raised or left behind by either binding.
    assert library.tf_last_error_code() == cpp.TF_OK


@needs_native
@pytest.mark.parametrize("name", ["exp", "log"])
def test_the_two_bindings_agree_on_the_reproducer_case(name, monkeypatch):
    """The same two-binding comparison, on the exact array CI failed on.

    The bits agree perfectly here while the NumPy comparison needs a one
    step budget — which is the whole diagnosis in one test: the variable
    is the math library, never the binding."""
    library = cpp._require_library()
    values = _reproducer_values((2, 3, 4, 5))
    if name == "log":
        values = np.abs(values)
    core = Core.from_array(values)
    transposed = core.T
    try:
        produced = getattr(transposed, name)()
        try:
            trusted = produced.to_numpy().copy()
        finally:
            produced.close()

        export = f"tf_core_{name}"
        checked = _checked_binding(export)

        def shim(*args):
            rank = args[5]
            converted = list(args)
            for position in (2, 3):
                converted[position] = np.asarray(
                    [args[position][index] for index in range(rank)],
                    dtype=np.int64)
            return checked(*converted)

        monkeypatch.setattr(library, export, shim, raising=False)
        produced = getattr(transposed, name)()
        try:
            through_checked = produced.to_numpy().copy()
        finally:
            produced.close()
        monkeypatch.undo()
    finally:
        transposed.close()
        core.close()
    assert _same_bits(trusted, through_checked)
    assert library.tf_last_error_code() == cpp.TF_OK


@needs_native
def test_reductions_preserve_the_h6_signed_zero_contract():
    """H6's contract, re-asserted through the trusted binding: a sum of
    negative zeros is ``+0.0`` on both traversals, matching NumPy."""
    for values in (np.full((3, 4), -0.0), np.zeros((3, 4)),
                   np.array([[-0.0, 0.0], [0.0, -0.0]])):
        core = Core.from_array(values)
        try:
            for axis in (None, 0, 1):
                for keepdims in (False, True):
                    produced = core.sum(axis=axis, keepdims=keepdims)
                    try:
                        assert _same_bits(produced.to_numpy(),
                                          values.sum(axis=axis,
                                                     keepdims=keepdims))
                    finally:
                        produced.close()
        finally:
            core.close()


@needs_native
def test_matmul_is_untouched_by_h7():
    """``tf_core_matmul`` takes no array at all — it never did — so H2's
    result is structurally unaffected. Asserted rather than assumed."""
    library = cpp._require_library()
    assert _array_kinds(library.tf_core_matmul) == ()
    rng = np.random.default_rng(4242)
    a, b = rng.standard_normal((32, 24)), rng.standard_normal((24, 16))
    core_a, core_b = Core.from_array(a), Core.from_array(b)
    try:
        produced = core_a.matmul(core_b)
        try:
            assert np.allclose(produced.to_numpy(), a @ b, atol=1e-12)
        finally:
            produced.close()
    finally:
        core_b.close()
        core_a.close()


@needs_native
def test_deterministic_training_and_exact_resume_are_unchanged(tmp_path):
    """The end-to-end guarantee: two runs are bit-identical, and an
    interrupted run resumes into a fresh model/optimizer pair that
    reproduces the remainder exactly."""
    from tensorforge.experimental import (
        NativeAdam, NativeBatchNorm1d, NativeLayerNorm, NativeLinear,
        NativeMSELoss, NativeReLU, NativeSequential, NativeTensor,
        load_native_checkpoint, save_native_checkpoint,
    )

    def build():
        model = NativeSequential(
            NativeLinear(4, 8, seed=0), NativeBatchNorm1d(8), NativeReLU(),
            NativeLayerNorm(8), NativeLinear(8, 2, seed=1))
        return model, NativeAdam(model.parameters(), lr=0.05)

    inputs = np.array([[(i + j) / 8.0 for j in range(4)] for i in range(12)])
    targets = np.array([[(i % 3) / 2.0, (i % 2) / 2.0] for i in range(12)])

    def run(steps, model=None, optimizer=None, start=0):
        if model is None:
            model, optimizer = build()
        loss_fn = NativeMSELoss()
        x = NativeTensor.from_array(inputs)
        y = NativeTensor.from_array(targets)
        losses = []
        for _ in range(start, steps):
            optimizer.zero_grad()
            loss = loss_fn(model(x), y)
            losses.append(float(loss.to_numpy()))
            loss.backward()
            optimizer.step()
            loss.close()
        return losses, model, optimizer

    first, model_a, opt_a = run(10)
    second, model_b, opt_b = run(10)
    assert first == second

    losses, model_c, opt_c = run(4)
    path = tmp_path / "resume.npz"
    save_native_checkpoint(path, model_c, optimizer=opt_c)
    model_d, opt_d = build()
    load_native_checkpoint(path, model_d, optimizer=opt_d)
    tail, _, _ = run(10, model_d, opt_d, start=4)
    assert losses + tail == first

    for parameter_a, parameter_d in zip(model_a.parameters(),
                                        model_d.parameters()):
        assert _same_bits(parameter_a.to_numpy(), parameter_d.to_numpy())


# ==========================================================================
# 10. Public surface, capabilities, and isolation
# ==========================================================================

def test_h7_added_no_public_api():
    """Both producers are private, and neither the pointer cache nor the
    binding categories are reachable through any public name."""
    public = [name for name in dir(cpp) if not name.startswith("_")]
    for banned in ("layout_vector", "layout_pointer", "layout_pointers",
                   "native_layout", "native_layout_pointers", "trusted_call",
                   "checked_call", "set_binding", "binding_mode",
                   "pointer_cache", "data_pointer", "address_of",
                   "raw_pointer", "storage_address"):
        assert banned not in public, banned
        assert not any(banned in name.lower() for name in public), banned
    for name in ("_layout_vector", "_LAYOUT_POINTER"):
        assert hasattr(cpp, name)
        assert name.startswith("_")


def test_no_trusted_flag_or_environment_fast_path_exists():
    source = BACKEND_FILE.read_text(encoding="utf-8")
    for banned in ("trusted=True", "trusted =", "unsafe", "os.environ",
                   "getenv", "TF_FAST", "TF_TRUSTED", "fast_mode",
                   "skip_validation", "no_validate"):
        assert banned not in source, banned
    assert "import os" not in source


@needs_native
def test_no_native_address_is_exposed_by_any_public_api():
    """A storage handle never leaves the module as a number, and no public
    accessor returns a pointer or an address."""
    core = Core.from_array(np.arange(6.0).reshape(2, 3))
    try:
        for holder in (core, core.storage, core.view):
            for name in dir(holder):
                if name.startswith("_"):
                    continue
                value = getattr(holder, name)
                if callable(value):
                    continue
                assert not isinstance(value, (ctypes.c_void_p, ctypes._Pointer))
        assert "handle" not in dir(core.storage)
        assert repr(core.storage) == "NativeStorage(size=6)"
    finally:
        core.close()


@needs_native
def test_the_exported_symbol_count_is_unchanged_apart_from_phase_i():
    """H7 added no C ABI symbol, and the only symbols added since are the
    two typed storage creators of Phase I milestone I1 — so the library
    exports 54: Phase H's 52 plus exactly those two."""
    source_exports = set()
    for path in sorted((REPO_ROOT / "cpp" / "src").glob("*.cpp")):
        source_exports.update(
            re.findall(r"TF_EXPORT[^;{]*?\b(tf_[a-z0-9_]+)\s*\(",
                       path.read_text(encoding="utf-8")))
    typed_creators = {"tf_storage_create_typed",
                      "tf_storage_create_uninitialized_typed"}
    assert typed_creators <= source_exports
    assert len(source_exports) == 54
    assert len(source_exports - typed_creators) == 52
    library = cpp._require_library()
    for name in source_exports:
        assert getattr(library, name, None) is not None, name
    for absent in ("tf_core_metadata_pointer", "tf_layout_pointer",
                   "tf_core_trusted_add", "tf_validate_metadata",
                   "tf_set_binding_mode", "tf_core_noop", "tf_boundary_probe"):
        assert absent not in source_exports, absent
        assert absent not in cpp._CHECKED_KERNELS, absent


def test_h7_changed_no_capability_dtype_device_or_checkpoint_value():
    assert cpp.UNSUPPORTED == ("float32", "cuda", "amp")
    assert cpp.SUPPORTED_DTYPES == ("float64",)
    assert cpp.SUPPORTED_DEVICES == ("cpu",)
    # 34 at Phase-H closure, plus the two Phase-I typed creators, which
    # report failure through the identical hook rather than inventing a
    # second convention. Nothing else joined: I1 deliberately left the
    # unguarded storage primitives (fill, scale, copy_from, copy_to)
    # hookless, so their per-call boundary cost is exactly what H7 left.
    assert len(cpp._CHECKED_KERNELS) == 36
    for name in ("tf_storage_create_typed",
                 "tf_storage_create_uninitialized_typed"):
        assert name in cpp._CHECKED_KERNELS, name
    for unhooked in ("tf_storage_fill", "tf_storage_scale",
                     "tf_storage_copy_from", "tf_storage_copy_to"):
        assert unhooked not in cpp._CHECKED_KERNELS, unhooked
    from tensorforge.experimental import native_checkpoint
    assert native_checkpoint._FORMAT_VERSION == 2
    assert tuple(sorted(native_checkpoint._SUPPORTED_FORMAT_VERSIONS)) == (1, 2)


def test_importing_stable_tensorforge_loads_no_native_binding():
    """The isolation contract: importing the stable framework must not
    import the backend package, the experimental package, or build a
    single layout pointer."""
    code = (
        "import sys, json;"
        "import tensorforge;"
        "leaked = [m for m in sys.modules"
        " if m.startswith('tensorforge.backends')"
        " or m.startswith('tensorforge.experimental')];"
        "print(json.dumps(leaked))"
    )
    result = subprocess.run([sys.executable, "-c", code], check=True,
                            capture_output=True, text=True, cwd=str(REPO_ROOT))
    assert __import__("json").loads(result.stdout.strip()) == []


def test_importing_the_backend_module_loads_no_library_and_no_pointer():
    """Importing the wrapper is still safe and still lazy: the two binding
    constants exist (they touch nothing), and the library does not load."""
    code = (
        "import tensorforge.backends.cpp as cpp;"
        "assert cpp._lib is None;"
        "assert cpp._LAYOUT_POINTER is not None;"
        "assert cpp._CHECKED_F64_ARRAY is not None;"
        "assert len(cpp._layout_vector((2, 3))) == 2;"
        "assert cpp._lib is None;"
        "print('ok')"
    )
    result = subprocess.run([sys.executable, "-c", code], check=True,
                            capture_output=True, text=True, cwd=str(REPO_ROOT))
    assert result.stdout.strip() == "ok"


@needs_native
def test_exactly_three_trusted_exports_reject_null_metadata_natively():
    """The precise version of a claim that is easy to round up, settled by
    running the ABI rather than by reading the C++.

    Thirteen exports have at least one trusted position. Only **three** of
    them — ``tf_core_exp``, ``tf_core_log``, and
    ``tf_core_contiguous_copy``, the self-validating Phase-E-era exports —
    reject a null layout pointer through the error contract. The other ten
    dereference it: the older ``tf_core_relu``/``sqrt``/``reciprocal`` sit
    beside the shared validator in the same translation unit but do not
    call it, and the binary family, ``tf_core_sum``,
    ``tf_core_narrow_backward``, and ``tf_storage_materialize`` have no
    such validator at all.

    That is exactly why "both producers are total" is the **load-bearing**
    half of H7's null argument rather than a belt-and-braces remark, and
    this test pins the real number instead of a comfortable one. The
    dereferencing exports are deliberately not called with a null here —
    doing so is an access violation, not a testable behavior.
    """
    library = cpp._require_library()
    trusted = [name for name, kinds in EXPECTED_ARRAY_BINDINGS.items()
               if "trusted" in kinds]
    assert len(trusted) == 13

    validating = ("tf_core_exp", "tf_core_log", "tf_core_contiguous_copy")
    assert all(name in trusted for name in validating)

    source = Core.from_array(np.arange(4.0))
    destination = Core.zeros((4,))
    try:
        src = source.storage._require_open()
        dst = destination.storage._require_open()
        good = cpp._layout_vector([4])
        unit = cpp._layout_vector([1])
        for name in validating:
            kernel = getattr(library, name)
            with pytest.raises(ValueError, match="null shape or stride"):
                kernel(src, dst, None, unit, 0, 1)
            with pytest.raises(ValueError, match="null shape or stride"):
                kernel(src, dst, good, None, 0, 1)
            # A valid call still works afterwards: no stale error survives.
            kernel(src, dst, good, unit, 0, 1)
            assert library.tf_last_error_code() == cpp.TF_OK
    finally:
        destination.close()
        source.close()

    # The one shared validator, and the fact that it is shared by exactly
    # the exports that opt into it.
    elementwise = (REPO_ROOT / "cpp" / "src"
                   / "elementwise.cpp").read_text(encoding="utf-8")
    assert elementwise.count("null shape or stride array") == 1


@needs_native
def test_no_production_path_can_pass_a_null_layout_pointer():
    """The structural half of the argument above: both producers are
    total, so the ten dereferencing exports are unreachable with a null.

    Asserted over every rank the runtime can construct and over a real
    workload, rather than by inspection."""
    for shape in SHAPES:
        core = _core_of_shape(shape)
        try:
            assert all(p is not None for p in core._layout_pointers())
            assert cpp._layout_vector(core.shape) is not None
            assert cpp._layout_vector(core.strides) is not None
        finally:
            core.close()

    library = cpp._require_library()
    strided = ("tf_core_relu", "tf_core_sqrt", "tf_core_reciprocal",
               "tf_core_exp", "tf_core_log", "tf_core_contiguous_copy",
               "tf_core_add", "tf_core_subtract", "tf_core_multiply",
               "tf_core_relu_backward", "tf_core_sum",
               "tf_core_narrow_backward", "tf_storage_materialize")
    seen = set()
    originals = {name: getattr(library, name) for name in strided}

    def watcher(name):
        original = originals[name]

        def call(*args):
            for argument, argtype in zip(args, original.argtypes):
                if argtype is cpp._LAYOUT_POINTER:
                    assert argument is not None, (name, "null layout pointer")
            seen.add(name)
            return original(*args)
        return call

    for name in strided:
        setattr(library, name, watcher(name))
    try:
        values = np.arange(24.0).reshape(2, 3, 4)
        base = Core.from_array(values)
        transposed = base.transpose(2, 1, 0)
        scalar = Core.full((), 2.0)
        try:
            for produced in (
                transposed.relu(), transposed.sqrt(), transposed.reciprocal(),
                transposed.exp(), transposed.log(),
                transposed.contiguous_copy(),
                base.add(scalar), base.subtract(scalar), base.multiply(scalar),
                base.sum(axis=1), transposed.relu_backward(transposed),
                base.narrow(0, 0, 1).contiguous_copy().narrow_backward(
                    0, 1, (3, 3, 4)),
            ):
                produced.close()
            transposed.to_numpy()
        finally:
            scalar.close()
            transposed.close()
            base.close()
    finally:
        for name, original in originals.items():
            setattr(library, name, original)
    assert seen == set(strided), sorted(set(strided) - seen)


@needs_native
def test_the_inventory_arithmetic_adds_up():
    """The documented category totals are arithmetic over the real
    bindings, so a future change cannot leave the design's table stale
    without failing here.

    54 exports = 22 with at least one array position + 30 that carry only
    storage handles and integers + 2 test-only hooks; and 54 array
    positions = 32 trusted + 22 checked.

    Phase I milestone I1 moved the handle-only column from 26 to 28: its
    two typed creators take an int64 element count and an int32 dtype code
    and no array at all, so neither the trusted nor the checked array tally
    moved.

    Phase I milestone I2 moved it again, 28 to 30, and the checked tally
    from 25 to 22 — **without adding or removing a single export**.
    ``tf_storage_copy_from`` and ``tf_storage_copy_to`` now carry no array
    position at all (a handle and a ``void*``), which is what puts them in
    the handle-only column, and ``tf_storage_materialize`` keeps its
    trusted layout pair but lost its checked destination. The three checks
    did not vanish: they run per call, per dtype, in ``_host_pointer``."""
    library = cpp._require_library()
    with_arrays, handle_only, test_only = 0, 0, 0
    trusted_positions, checked_positions = 0, 0
    for name in _configured_names():
        function = getattr(library, name, None)
        if function is None or function.argtypes is None:
            continue
        kinds = _array_kinds(function)
        trusted_positions += kinds.count("trusted")
        checked_positions += kinds.count("checked")
        if kinds:
            with_arrays += 1
        elif name in ("tf_test_arm_alloc_failure",
                      "tf_fault_injection_available"):
            test_only += 1
        else:
            handle_only += 1
    assert (with_arrays, handle_only, test_only) == (22, 30, 2)
    assert with_arrays + handle_only + test_only == 54
    assert (trusted_positions, checked_positions) == (32, 22)
    assert trusted_positions + checked_positions == 54
    # Thirteen of the 22 array-carrying exports have a trusted position —
    # unchanged by I2, which removed only checked positions.
    trusted_exports = sum(
        1 for name in _configured_names()
        if (f := getattr(library, name, None)) is not None
        and f.argtypes is not None and "trusted" in _array_kinds(f))
    assert trusted_exports == 13
