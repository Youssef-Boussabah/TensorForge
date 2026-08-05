"""Phase K, milestone K1 — the ``int64`` representation and **every**
reachability barrier.

K1 makes ``int64`` representable by the raw private C ABI and installs the
barriers that keep it from reaching anything else. The three facts it must
prove hold **together**, and this module proves them together because any
one of them alone is misleading:

1. **The raw private ABI can allocate and round-trip int64 exactly.**
   ``tf_storage_create_typed(size, 2)`` through ``ctypes``, values in and
   out bit for bit, including ``INT64_MIN``, ``INT64_MAX``, and values
   beyond 2^53 where a floating detour would start rounding.
2. **No supported TensorForge Python wrapper can be built over it.** Every
   public constructor rejects ``"int64"`` by name; every private ``_typed*``
   route rejects it because ``_DTYPE_CODES`` does not know the name at K1;
   and a ``NativeStorage`` / ``NativeTensorCore`` / ``NativeTensor``
   assembled around such a handle **without** its constructor is refused by
   the next layer up.
3. **The raw int64 storage cannot enter a floating C++ compute export.**
   Every audited export rejects it at the ABI, independently of Python, and
   writes nothing when it does.

Reaching fact 1 needs direct ``ctypes``: that is the *only* route to int64
at this milestone, and using it here is what makes facts 2 and 3 testable
against a genuine int64 handle rather than against a dtype string. The
scaffolding is confined to this module, is loudly marked test-only, and
introduces **no** production API — the milestone deliberately ships none.

Discipline this module inherits (integer design §30.2):

* **Exact equality only** for integers — Python ``int`` comparison and
  ``numpy.int64`` array equality, never a tolerance.
* **Every rejection is followed by a fingerprint of the observable world**,
  and the fingerprint itself has a non-vacuity control proving each
  component can notice the change it exists for.
* **Native handles are released even when an assertion fails** — every raw
  handle is owned by a context manager whose ``finally`` destroys it.
* **Every scanner has a negative control.**
* No test starts a thread, touches the network, or needs a Git ancestor.
"""
import ast
import contextlib
import ctypes
import io
import re
import tokenize
from pathlib import Path

import numpy as np
import pytest

from tensorforge.backends import cpp
from tensorforge.experimental import (
    NativeAdam,
    NativeLinear,
    NativeModule,
    NativeParameter,
    NativeSGD,
    NativeTensor,
)

REPO_ROOT = Path(__file__).resolve().parent.parent

needs_native = pytest.mark.skipif(
    not cpp.is_available(), reason="experimental C++ backend not built"
)

# The ABI dtype codes, written here independently of the module under test.
CODE_FLOAT64 = 0
CODE_FLOAT32 = 1
CODE_INT64 = 2

# Values that catch width, sign, and truncation errors. 2**53 + 1 is the
# smallest positive integer float64 cannot represent, so a value past it
# surviving intact is what separates an exact integer transfer from one that
# took a floating detour.
PROBE_VALUES = (
    0,
    1,
    -1,
    42,
    -42,
    2 ** 31 - 1,
    2 ** 31,
    -(2 ** 31),
    -(2 ** 31) - 1,
    2 ** 32,
    2 ** 53 + 1,
    -(2 ** 53) - 1,
    2 ** 63 - 1,          # INT64_MAX
    -(2 ** 63),           # INT64_MIN
)


# ---------------------------------------------------------------------------
# Test-only scaffolding.
#
# **None of this is a production API and none of it may become one.** It
# exists to reach the one representation K1 makes available — raw int64
# storage behind the private C ABI — so the barriers above it can be driven
# against a real object instead of a dtype string. Each helper assembles a
# wrapper **around** its constructor precisely because the constructor is
# the barrier under test; a helper that went *through* the constructor
# would be testing nothing.
# ---------------------------------------------------------------------------

@contextlib.contextmanager
def raw_int64_handle(size):
    """A raw ``int64`` storage handle from the private C ABI.

    The whole of K1's construction surface: one ``ctypes`` call. Destroyed
    in a ``finally`` so a failing assertion still releases it."""
    library = cpp._require_library()
    handle = library.tf_storage_create_typed(int(size), CODE_INT64)
    assert handle, "the private ABI could not allocate int64 storage"
    try:
        yield handle
    finally:
        library.tf_storage_destroy(handle)


@contextlib.contextmanager
def raw_int64_storage(size):
    """A ``NativeStorage``-shaped object over genuine int64 storage,
    assembled **without** running ``NativeStorage.__init__``.

    Test-only, and the bypass is the point: the public constructor rejects
    ``"int64"`` (which a test below proves separately), so this is how the
    *next* layer's barrier is driven with a real operand."""
    library = cpp._require_library()
    with raw_int64_handle(size) as handle:
        storage = cpp.NativeStorage.__new__(cpp.NativeStorage)
        storage._lib = library
        storage._handle = handle
        storage._size = int(size)
        storage._dtype = "int64"
        storage._device = "cpu"
        try:
            yield storage
        finally:
            # The handle belongs to ``raw_int64_handle``; detach the
            # wrapper so its ``__del__`` fallback cannot double-destroy.
            storage._handle = None


@contextlib.contextmanager
def raw_int64_core(shape):
    """A ``NativeTensorCore`` over genuine int64 storage, assembled without
    running ``NativeTensorCore.__init__`` — the barrier under test."""
    dims = tuple(int(d) for d in shape)
    size = 1
    for dim in dims:
        size *= dim
    with raw_int64_storage(size) as storage:
        view = cpp.NativeTensorView._from_validated(
            storage, dims, cpp.row_major_strides(dims), 0
        )
        core = cpp.NativeTensorCore.__new__(cpp.NativeTensorCore)
        core._storage = storage
        core._view = view
        core._owns_storage = False   # the context manager owns the handle
        core._closed = False
        yield core


@contextlib.contextmanager
def raw_int64_tensor(shape):
    """A ``NativeTensor`` over genuine int64 storage, assembled without
    running ``NativeTensor.__init__`` — the barrier under test."""
    with raw_int64_core(shape) as core:
        tensor = NativeTensor.__new__(NativeTensor)
        tensor._core = core
        # An **owning** tensor, because several barriers legitimately check
        # ownership before they reach the dtype (``register_buffer`` is the
        # one that matters), and a borrowing stand-in would be rejected for
        # the wrong reason. The native handle still belongs to
        # ``raw_int64_handle``; the ``finally`` below disarms the ``__del__``
        # fallback so nothing can double-release it.
        tensor._owns_core = True
        tensor._closed = False
        tensor._requires_grad = False
        tensor._grad = None
        tensor._parents = ()
        tensor._backward = None
        tensor._op = ""
        tensor._is_leaf = True
        tensor._graph_freed = False
        tensor._expected_versions = ()
        tensor._graph_resources = ()
        try:
            yield tensor
        finally:
            tensor._owns_core = False
            tensor._closed = True


def write_int64(storage, values):
    """Move an exact ``int64`` host array into raw int64 storage.

    The Python wrapper has no int64 host binding at K1 — that is part of
    what makes int64 unreachable from the package — so the pointer is taken
    directly, which is the raw ABI contract a foreign caller satisfies."""
    array = np.ascontiguousarray(values, dtype=np.int64)
    assert array.dtype == np.dtype(np.int64)
    storage._lib.tf_storage_copy_from(
        storage._handle, array.ctypes.data_as(ctypes.c_void_p)
    )


def read_int64(storage):
    """Read raw int64 storage back into a fresh exact ``int64`` array."""
    out = np.zeros(storage._size, dtype=np.int64)
    storage._lib.tf_storage_copy_to(
        storage._handle, out.ctypes.data_as(ctypes.c_void_p)
    )
    return out


# ---------------------------------------------------------------------------
# The observable-world fingerprint.
#
# Every rejection below is followed by a before/after comparison of this,
# and ``test_the_fingerprint_can_notice_each_change_it_exists_for`` proves
# each component can notice the change it is there for — a fingerprint that
# cannot fail would make every "unchanged" assertion vacuous.
# ---------------------------------------------------------------------------

class World:
    """A snapshot of everything a rejected call must leave alone."""

    def __init__(self, parameter, buffer_tensor, module, optimizer):
        self.parameter = parameter
        self.buffer_tensor = buffer_tensor
        self.module = module
        self.optimizer = optimizer

    def fingerprint(self):
        parameter = self.parameter
        return (
            # the parameter: identity, value bits, version, gradient
            id(parameter),
            parameter.to_numpy().tobytes(),
            parameter.version,
            parameter.requires_grad,
            None if parameter.grad is None else parameter.grad.to_numpy().tobytes(),
            # a registered buffer, by identity and by value
            id(self.buffer_tensor),
            self.buffer_tensor.to_numpy().tobytes(),
            tuple(sorted(name for name, _ in self.module.named_buffers())),
            tuple(sorted(name for name, _ in self.module.named_parameters())),
            # the live optimizer's charges, by identity and order
            tuple(id(p) for p in self.optimizer.parameters()),
            self.optimizer.lr,
            # the capability registries, which no rejection may move
            cpp.SUPPORTED_DTYPES,
            cpp.SUPPORTED_DEVICES,
            cpp.UNSUPPORTED,
            cpp.RAW_KERNEL_DTYPES,
            tuple(sorted(cpp._DTYPE_CODES.items())),
            tuple(sorted(cpp._DTYPE_NUMPY)),
            cpp.TENSOR_CORE_OPS,
            cpp.AUTOGRAD_OPS,
            # both global RNGs, which nothing here may consume
            np.random.get_state()[1][0],
        )


@contextlib.contextmanager
def unchanged_world():
    """Build the world, hand it over, and assert its fingerprint is
    byte-identical afterwards."""
    parameter = NativeParameter(np.array([[1.5, -2.5], [0.25, 4.0]]))
    buffer_tensor = NativeTensor.from_array(np.array([3.0, -1.0]))
    module = NativeModule()
    module.weight = parameter
    module.register_buffer("stat", buffer_tensor, persistent=True)
    optimizer = NativeSGD([parameter], lr=0.1)
    world = World(parameter, buffer_tensor, module, optimizer)
    before = world.fingerprint()
    try:
        yield world
        assert world.fingerprint() == before, "a rejection changed the world"
    finally:
        parameter.close()
        buffer_tensor.close()


# ===========================================================================
# 0. The scaffolding and the fingerprint can both actually fail
# ===========================================================================

@needs_native
def test_the_fingerprint_can_notice_each_change_it_exists_for():
    """Non-vacuity for every ``unchanged_world`` assertion below.

    Each component is perturbed on its own and the fingerprint must differ;
    a fingerprint that could not notice would make every "the world is
    unchanged" claim below meaningless."""
    def build():
        parameter = NativeParameter(np.array([[1.5, -2.5], [0.25, 4.0]]))
        buffer_tensor = NativeTensor.from_array(np.array([3.0, -1.0]))
        module = NativeModule()
        module.weight = parameter
        module.register_buffer("stat", buffer_tensor, persistent=True)
        optimizer = NativeSGD([parameter], lr=0.1)
        return World(parameter, buffer_tensor, module, optimizer), parameter, \
            buffer_tensor

    # 1. a changed parameter value (and, with it, its version)
    world, parameter, buffer_tensor = build()
    before = world.fingerprint()
    try:
        replacement = NativeTensor.from_array(np.zeros((2, 2)))
        try:
            parameter.copy_value_(replacement)
        finally:
            replacement.close()
        assert world.fingerprint() != before
    finally:
        parameter.close()
        buffer_tensor.close()

    # 2. the buffer registry losing its one entry
    world, parameter, buffer_tensor = build()
    before = world.fingerprint()
    try:
        world.module.register_buffer("stat", None)
        assert world.fingerprint() != before
    finally:
        parameter.close()
        buffer_tensor.close()

    # 3. a gradient appearing on the parameter
    world, parameter, buffer_tensor = build()
    before = world.fingerprint()
    try:
        grad = NativeTensor.from_array(np.ones((2, 2)))
        parameter._accumulate_grad(grad)
        assert world.fingerprint() != before
    finally:
        parameter.close()
        buffer_tensor.close()

    # 4. a registry moving (simulated on a copy, never on the real module)
    world, parameter, buffer_tensor = build()
    before = world.fingerprint()
    try:
        world.module.extra = NativeParameter(np.array([1.0]))
        try:
            assert world.fingerprint() != before
        finally:
            world.module.extra.close()
    finally:
        parameter.close()
        buffer_tensor.close()


@needs_native
def test_the_scaffolding_really_builds_int64_objects():
    """Non-vacuity for the scaffolding itself: the objects it assembles do
    carry the int64 tag and do sit over the private ABI's storage.

    Without this, every barrier test below could pass because the helper
    silently produced a float64 object nobody could reject."""
    with raw_int64_storage(4) as storage:
        assert storage.dtype == "int64"
        assert storage.size == 4
        assert isinstance(storage, cpp.NativeStorage)
    with raw_int64_core((2, 2)) as core:
        assert core.dtype == "int64"
        assert core.shape == (2, 2)
        assert isinstance(core, cpp.NativeTensorCore)
    with raw_int64_tensor((2, 2)) as tensor:
        assert tensor.dtype == "int64"
        assert isinstance(tensor, NativeTensor)
        assert tensor.requires_grad is False


# ===========================================================================
# 1. The raw private ABI represents int64 exactly
# ===========================================================================

@needs_native
def test_the_private_abi_allocates_and_tags_int64_storage():
    library = cpp._require_library()
    with raw_int64_handle(6) as handle:
        # The size is a **logical element count**, at every dtype.
        assert library.tf_storage_size(handle) == 6


@needs_native
def test_int64_storage_is_zero_initialized_exactly():
    with raw_int64_storage(5) as storage:
        assert read_int64(storage).tolist() == [0, 0, 0, 0, 0]


@needs_native
@pytest.mark.parametrize("value", PROBE_VALUES)
def test_every_probe_value_round_trips_exactly(value):
    """Exact integer equality, never a tolerance (§29.6)."""
    with raw_int64_storage(1) as storage:
        write_int64(storage, [value])
        out = read_int64(storage)
        assert out.dtype == np.dtype(np.int64)
        assert int(out[0]) == value


@needs_native
def test_the_whole_probe_sequence_round_trips_bit_for_bit():
    source = np.array(PROBE_VALUES, dtype=np.int64)
    with raw_int64_storage(source.size) as storage:
        write_int64(storage, source)
        out = read_int64(storage)
        # Raw object representation, not a value comparison: a transfer that
        # happened to round-trip through another type would still be caught.
        assert out.tobytes() == source.tobytes()
        assert np.array_equal(out, source)


@needs_native
def test_no_truncation_at_the_signed_extremes():
    """The two values a 32-bit or unsigned path would destroy."""
    with raw_int64_storage(2) as storage:
        write_int64(storage, [2 ** 63 - 1, -(2 ** 63)])
        out = read_int64(storage)
        assert int(out[0]) == 2 ** 63 - 1
        assert int(out[1]) == -(2 ** 63)
        # ...and the bytes really are two's complement, checked directly.
        assert out.tobytes() == (
            (2 ** 63 - 1).to_bytes(8, "little", signed=True)
            + (-(2 ** 63)).to_bytes(8, "little", signed=True)
        )


@needs_native
def test_no_float_reinterpretation_of_an_int64_buffer():
    """A value beyond 2^53 survives, which a float64 detour would round —
    and the stored bytes are the integer's, not a double's."""
    value = 2 ** 53 + 1
    with raw_int64_storage(1) as storage:
        write_int64(storage, [value])
        assert int(read_int64(storage)[0]) == value
        # The negative control that makes this mean something: the *same*
        # value through a float64 buffer does **not** survive.
        with cpp.NativeStorage(1, dtype="float64") as floating:
            floating.copy_from(np.array([float(value)]))
            assert int(floating.to_numpy()[0]) != value


@needs_native
def test_int64_materialization_follows_the_logical_order():
    values = np.array([-(2 ** 53) - 1, 2, -3, 4, 2 ** 53 + 1, 6],
                      dtype=np.int64)
    with raw_int64_storage(6) as storage:
        write_int64(storage, values)
        out = np.zeros(6, dtype=np.int64)
        shape = cpp._layout_vector([3, 2])
        strides = cpp._layout_vector([1, 3])
        storage._lib.tf_storage_materialize(
            storage._handle, out.ctypes.data_as(ctypes.c_void_p),
            shape, strides, 0, 2,
        )
        expected = values.reshape(2, 3).T.reshape(-1)
        assert np.array_equal(out, expected)
        assert out.tobytes() == np.ascontiguousarray(expected).tobytes()


@needs_native
def test_allocation_and_cleanup_return_live_storage_to_baseline():
    """Handles are released explicitly; nothing here depends on collection
    timing."""
    library = cpp._require_library()
    handles = []
    for _ in range(8):
        handle = library.tf_storage_create_typed(3, CODE_INT64)
        assert handle
        handles.append(handle)
    assert len(set(handles)) == len(handles), "the ABI reused a live handle"
    for handle in handles:
        library.tf_storage_destroy(handle)


@needs_native
def test_the_private_abi_still_rejects_a_genuinely_unknown_code():
    library = cpp._require_library()
    for code in (-1, 3, 4, 99, 2 ** 31 - 1):
        with pytest.raises(ValueError, match="dtype"):
            library.tf_storage_create_typed(4, code)


@needs_native
def test_float32_and_float64_creation_are_unchanged():
    """The negative control for the representation change: a third element
    type moved neither of the two already there."""
    library = cpp._require_library()
    for code, dtype in ((CODE_FLOAT64, "float64"), (CODE_FLOAT32, "float32")):
        handle = library.tf_storage_create_typed(4, code)
        assert handle
        try:
            assert library.tf_storage_size(handle) == 4
        finally:
            library.tf_storage_destroy(handle)
        with cpp.NativeStorage(4, dtype=dtype) as storage:
            assert storage.dtype == dtype
            assert storage.to_numpy().tolist() == [0.0, 0.0, 0.0, 0.0]


# ===========================================================================
# 2. Nothing in Python can name, allocate, or wrap int64
# ===========================================================================

def test_the_public_registries_have_not_moved():
    assert cpp.SUPPORTED_DTYPES == ("float64", "float32")
    assert cpp.SUPPORTED_DEVICES == ("cpu",)
    assert cpp.UNSUPPORTED == ("cuda", "amp")
    assert cpp.RAW_KERNEL_DTYPES == ("float64",)
    assert cpp.normalize_dtype(None) == "float64"
    assert cpp.backend_info()["dtype"] == "float64"


def test_normalize_dtype_still_rejects_int64():
    with pytest.raises(ValueError, match="int64"):
        cpp.normalize_dtype("int64")


def test_the_index_registry_does_not_exist_yet():
    """``INDEX_DTYPES`` is K2's, in the same commit as the public
    constructor. Its absence is the promise half of "prove first, then
    promise"."""
    for absent in ("INDEX_DTYPES", "COMPUTE_DTYPES", "INTEGER_DTYPES",
                   "TENSOR_DTYPES"):
        assert not hasattr(cpp, absent), absent
    assert "index_dtypes" not in cpp.backend_info()


def test_the_python_dtype_tables_do_not_know_int64():
    """The Phase-I no-drift invariant is still exactly true: the
    representation table and the public registry accept the same set, so
    nothing representable in Python is unpromised."""
    assert set(cpp._DTYPE_CODES) == set(cpp.SUPPORTED_DTYPES)
    assert "int64" not in cpp._DTYPE_CODES
    assert "int64" not in cpp._DTYPE_ITEM_SIZES
    assert "int64" not in cpp._DTYPE_NUMPY
    assert "int64" not in cpp._CHECKED_HOST_ARRAYS
    with pytest.raises(ValueError, match="int64"):
        cpp._normalize_internal_dtype("int64")


@needs_native
def test_every_public_constructor_rejects_int64_and_allocates_nothing():
    builders = {
        "NativeStorage": lambda: cpp.NativeStorage(4, dtype="int64"),
        "NativeStorage.from_array":
            lambda: cpp.NativeStorage.from_array([1, 2], dtype="int64"),
        "NativeTensorCore.from_array":
            lambda: cpp.NativeTensorCore.from_array([1, 2], dtype="int64"),
        "NativeTensorCore.zeros":
            lambda: cpp.NativeTensorCore.zeros((2,), dtype="int64"),
        "NativeTensorCore.full":
            lambda: cpp.NativeTensorCore.full((2,), 1, dtype="int64"),
        "NativeTensor.from_array":
            lambda: NativeTensor.from_array([1, 2], dtype="int64"),
        "NativeTensor.zeros":
            lambda: NativeTensor.zeros((2,), dtype="int64"),
        "NativeTensor.full":
            lambda: NativeTensor.full((2,), 1, dtype="int64"),
        "NativeParameter": lambda: NativeParameter([1.0], dtype="int64"),
        "NativeLinear": lambda: NativeLinear(2, 2, dtype="int64"),
    }
    with unchanged_world():
        for name, build in builders.items():
            with pytest.raises(ValueError, match="int64"):
                build()


@needs_native
def test_every_private_typed_constructor_rejects_int64_at_k1():
    """The private ``_typed*`` family is **not** a way around the public
    validator, and at K1 it cannot name int64 at all: ``_DTYPE_CODES`` has
    no entry, so the internal validator refuses the string before any
    allocation. Construction is K2's, and it is K2's atomically."""
    builders = {
        "NativeStorage._typed":
            lambda: cpp.NativeStorage._typed(4, "int64"),
        "NativeStorage._uninitialized":
            lambda: cpp.NativeStorage._uninitialized(4, dtype="int64"),
        "NativeStorage._typed_from_array":
            lambda: cpp.NativeStorage._typed_from_array([1, 2], "int64"),
        "NativeTensorCore._typed":
            lambda: cpp.NativeTensorCore._typed((2,), "int64"),
        "NativeTensorCore._uninitialized":
            lambda: cpp.NativeTensorCore._uninitialized((2,), dtype="int64"),
        "NativeTensorCore._typed_from_array":
            lambda: cpp.NativeTensorCore._typed_from_array([1, 2], "int64"),
        "NativeTensorCore._typed_full":
            lambda: cpp.NativeTensorCore._typed_full((2,), 1, "int64"),
        "NativeTensorCore.zeros(trusted)":
            lambda: cpp.NativeTensorCore.zeros((2,), dtype="int64",
                                               _trusted_dtype=True),
        "NativeTensor._typed_from_array":
            lambda: NativeTensor._typed_from_array([1, 2], "int64"),
        "NativeTensor._typed_zeros":
            lambda: NativeTensor._typed_zeros((2,), "int64"),
        "NativeTensor._typed_full":
            lambda: NativeTensor._typed_full((2,), 1, "int64"),
        "cpp._narrowed_to_dtype":
            lambda: cpp._narrowed_to_dtype(1.0, "int64"),
    }
    with unchanged_world():
        for name, build in builders.items():
            with pytest.raises(ValueError, match="int64"):
                build()


@needs_native
def test_no_public_int64_constructor_name_exists():
    """The K2 names, absent. Read from the AST rather than by ``hasattr``
    so a name defined but not yet reachable would still be caught."""
    for module, class_name in (
        ("src/tensorforge/experimental/native_tensor.py", "NativeTensor"),
        ("src/tensorforge/backends/cpp.py", "NativeTensorCore"),
        ("src/tensorforge/backends/cpp.py", "NativeStorage"),
    ):
        tree = ast.parse((REPO_ROOT / module).read_text(encoding="utf-8"))
        defined = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == class_name:
                defined = {child.name for child in node.body
                           if isinstance(child, ast.FunctionDef)}
        assert defined, (module, class_name)
        for absent in ("from_int64_array", "_from_int64_array", "item",
                       "tolist", "argmax", "index_select", "astype", "to",
                       "long", "int"):
            assert absent not in defined, (class_name, absent)


@needs_native
def test_a_raw_int64_handle_cannot_become_a_storage_core_or_tensor():
    """Fact 2, against the real handle. Each layer is driven separately,
    because each is a separate barrier with its own first authority."""
    with unchanged_world():
        # The storage layer: the public constructor never sees the handle,
        # because it allocates its own — and it refuses the dtype outright.
        with pytest.raises(ValueError, match="int64"):
            cpp.NativeStorage(4, dtype="int64")
        # The core layer, handed a genuine int64 storage.
        with raw_int64_storage(4) as storage:
            view = cpp.NativeTensorView._from_validated(
                storage, (4,), (1,), 0
            )
            with pytest.raises(ValueError, match="int64"):
                cpp.NativeTensorCore(storage, view)
        # The tensor layer, handed a genuine int64 core.
        with raw_int64_core((4,)) as core:
            with pytest.raises(ValueError, match="int64"):
                NativeTensor(core)
            with pytest.raises(ValueError, match="int64"):
                NativeTensor._from_core(core)


# ===========================================================================
# 3. Autograd refuses a non-differentiable dtype
# ===========================================================================

@needs_native
def test_from_op_refuses_to_build_a_graph_over_a_non_floating_core():
    """The structural backstop: no operation written after K1 can produce a
    differentiable integer result, because the single graph-construction
    entry refuses to build one — and it closes what it was handed."""
    parent = NativeTensor.from_array(np.array([1.0, 2.0]), requires_grad=True)
    try:
        with raw_int64_core((2,)) as core:
            with pytest.raises(ValueError, match="int64"):
                NativeTensor._from_op(core, (parent,), lambda g: None, "probe")
            # It closed the core it was given, so nothing leaked.
            assert core._closed
    finally:
        parent.close()


@needs_native
def test_from_op_also_releases_the_saved_state_it_was_handed():
    """A rejected graph leaks nothing at all — the saved resources go the
    same way as the core, exactly as they do on the no-grad path."""
    parent = NativeTensor.from_array(np.array([1.0]), requires_grad=True)
    saved = NativeTensor.from_array(np.array([5.0]))
    try:
        with raw_int64_core((1,)) as core:
            with pytest.raises(ValueError, match="int64"):
                NativeTensor._from_op(core, (parent,), lambda g: None,
                                      "probe", graph_resources=(saved,))
        assert saved.closed, "a rejected graph kept its saved state alive"
    finally:
        parent.close()


@needs_native
def test_backward_and_accumulate_grad_refuse_a_non_floating_tensor():
    with unchanged_world():
        with raw_int64_tensor((2,)) as tensor:
            with pytest.raises(RuntimeError, match="int64"):
                tensor.backward()
            contribution = NativeTensor.from_array(np.array([1.0, 1.0]))
            try:
                with pytest.raises(RuntimeError, match="int64"):
                    tensor._accumulate_grad(contribution)
            finally:
                contribution.close()
            assert tensor.grad is None


@needs_native
def test_a_floating_graph_still_works_exactly_as_it_did():
    """The negative control for the autograd barriers."""
    x = NativeTensor.from_array(np.array([2.0, 3.0]), requires_grad=True)
    try:
        y = x.multiply(x)
        try:
            loss = y.sum()
            try:
                loss.backward()
            finally:
                loss.close()
        finally:
            y.close()
        assert np.allclose(x.grad.to_numpy(), [4.0, 6.0], atol=1e-12)
    finally:
        x.close()


# ===========================================================================
# 4. Parameters, buffers, and optimizers refuse a non-floating tensor
# ===========================================================================

@needs_native
def test_no_integer_parameter_can_be_constructed():
    with unchanged_world():
        with raw_int64_tensor((2, 2)) as tensor:
            with pytest.raises(ValueError, match="int64"):
                NativeParameter(tensor)
            # ...and naming the dtype fails too, through the narrowed
            # module validator, before the source is even examined.
            with pytest.raises(ValueError, match="int64"):
                NativeParameter(tensor, dtype="int64")


@needs_native
@pytest.mark.parametrize("persistent", [True, False])
def test_no_integer_buffer_of_either_kind_can_be_registered(persistent):
    """Stricter than merely excluding integers from checkpoints: **neither**
    buffer kind is permitted at K1 (integer design §10.2)."""
    module = NativeModule()
    with raw_int64_tensor((2,)) as tensor:
        with pytest.raises(ValueError, match="int64"):
            module.register_buffer("stat", tensor, persistent=persistent)
    # The registration left no trace at all.
    assert list(module.named_buffers()) == []
    assert not hasattr(module, "stat")


@needs_native
def test_a_floating_buffer_of_either_kind_still_registers():
    """The negative control for the buffer barrier."""
    for persistent in (True, False):
        module = NativeModule()
        tensor = NativeTensor.from_array(np.array([1.0, 2.0]))
        try:
            module.register_buffer("stat", tensor, persistent=persistent)
            assert [name for name, _ in module.named_buffers()] == ["stat"]
            assert ("stat" in module.state_dict()) is persistent
        finally:
            tensor.close()


@needs_native
def test_no_optimizer_accepts_a_non_floating_parameter():
    """Both optimizers, both barriers — the type check that has always been
    there, and the direct per-parameter dtype check K1 added beside it."""
    with unchanged_world():
        with raw_int64_tensor((2,)) as tensor:
            # A plain tensor is not a NativeParameter: the type check
            # rejects first, which is the pre-existing transitive closure.
            with pytest.raises(TypeError, match="NativeParameter"):
                NativeSGD([tensor], lr=0.1)
            with pytest.raises(TypeError, match="NativeParameter"):
                NativeAdam([tensor])
            # And the direct check, driven against a NativeParameter-shaped
            # object carrying the int64 tag — the case that would survive if
            # the barrier were only transitive.
            fake = NativeParameter.__new__(NativeParameter)
            fake._core = tensor._core
            fake._owns_core = False
            fake._closed = False
            fake._requires_grad = True
            fake._grad = None
            fake._parents = ()
            fake._backward = None
            fake._op = ""
            fake._is_leaf = True
            fake._graph_freed = False
            fake._expected_versions = ()
            fake._graph_resources = ()
            fake._version = 0
            with pytest.raises(ValueError, match="int64"):
                NativeSGD([fake], lr=0.1)
            with pytest.raises(ValueError, match="int64"):
                NativeAdam([fake])


@needs_native
def test_both_optimizers_still_accept_a_floating_parameter():
    """The negative control for the optimizer barriers, and the proof that
    the rejection above is about the dtype rather than about the
    hand-assembled object: the same shape of object, floating, is
    accepted."""
    parameter = NativeParameter(np.array([1.0, 2.0]))
    try:
        sgd = NativeSGD([parameter], lr=0.1)
        assert sgd.parameters() == [parameter]
        adam = NativeAdam([parameter])
        try:
            assert adam.parameters() == [parameter]
        finally:
            adam.close()
    finally:
        parameter.close()


# ===========================================================================
# 5. Checkpoint and state entries refuse a non-floating dtype
# ===========================================================================

def test_no_archive_entry_may_declare_a_non_floating_dtype():
    from tensorforge.experimental import native_checkpoint

    for version in native_checkpoint._SUPPORTED_FORMAT_VERSIONS:
        with pytest.raises(ValueError, match="int64"):
            native_checkpoint._validated_entry_dtype(
                "int64", version, "manifest['model']['entries']['w']",
                "load_native_checkpoint",
            )
    # The negative control: the dtypes an archive *may* declare still pass,
    # and the version-1/2 float64-only rule is untouched.
    assert native_checkpoint._validated_entry_dtype(
        "float64", 3, "e", "w") == "float64"
    assert native_checkpoint._validated_entry_dtype(
        "float32", 3, "e", "w") == "float32"
    for version in native_checkpoint._FLOAT64_ONLY_VERSIONS:
        with pytest.raises(ValueError, match="float64 only"):
            native_checkpoint._validated_entry_dtype("float32", version, "e",
                                                     "w")


def test_the_checkpoint_and_state_versions_are_unmoved():
    from tensorforge.experimental import (native_checkpoint,
                                          native_data_loader,
                                          native_optimizer_state,
                                          native_sampler)

    assert native_checkpoint._FORMAT == "tensorforge.native_checkpoint"
    assert native_checkpoint._FORMAT_VERSION == 3
    assert native_checkpoint._SUPPORTED_FORMAT_VERSIONS == (1, 2, 3)
    assert 4 not in native_checkpoint._SUPPORTED_FORMAT_VERSIONS
    assert native_optimizer_state.FORMAT_VERSION == 1
    assert native_data_loader._FORMAT_VERSION == 1
    assert native_data_loader._SUPPORTED_FORMAT_VERSIONS == (1,)
    assert native_sampler._FORMAT_VERSION == 1
    assert native_sampler._SUPPORTED_FORMAT_VERSIONS == (1,)


# ===========================================================================
# 6. Every floating operation refuses a non-floating operand
# ===========================================================================

# One row per Core-level operation entry, as a callable taking the probed
# core. Table-driven so a new operation cannot quietly avoid the audit.
def _core_operations(core, floating):
    return {
        "relu": lambda: core.relu(),
        "sqrt": lambda: core.sqrt(),
        "reciprocal": lambda: core.reciprocal(),
        "exp": lambda: core.exp(),
        "log": lambda: core.log(),
        "softmax": lambda: core.softmax(axis=-1),
        "log_softmax": lambda: core.log_softmax(axis=-1),
        "sum": lambda: core.sum(),
        "mean": lambda: core.mean(),
        "add": lambda: core.add(floating),
        "subtract": lambda: core.subtract(floating),
        "multiply": lambda: core.multiply(floating),
        "matmul": lambda: core.matmul(floating),
        "relu_backward": lambda: core.relu_backward(floating),
        "narrow_backward": lambda: core.narrow_backward(0, 0, (4, 2)),
        "cross_entropy_forward":
            lambda: core.cross_entropy_forward(np.array([0, 1],
                                                        dtype=np.int64)),
        "cross_entropy_backward":
            lambda: core.cross_entropy_backward(
                np.array([0, 1], dtype=np.int64), floating),
        "maxpool2d_forward":
            lambda: core.reshape((1, 1, 2, 2)).maxpool2d_forward(
                kernel_size=1),
        "dropout_forward":
            lambda: core.dropout_forward(0.5, seed=1, call_index=1),
        "conv2d_forward":
            lambda: core.reshape((1, 1, 2, 2)).conv2d_forward(
                floating.reshape((1, 1, 2, 2)), None,
                stride=1, padding=0),
    }


@needs_native
def test_every_core_operation_refuses_a_non_floating_operand():
    with unchanged_world():
        floating = cpp.NativeTensorCore.from_array(
            np.ones((2, 2), dtype=np.float64))
        try:
            with raw_int64_core((2, 2)) as core:
                operations = _core_operations(core, floating)
                assert len(operations) == 20, "the operation audit shrank"
                for name, call in operations.items():
                    with pytest.raises(ValueError, match="int64"):
                        call()
        finally:
            floating.close()


@needs_native
def test_a_mixed_floating_and_integer_request_is_a_role_error():
    """An integer operand is refused as *floating-only*, never as two
    dtypes that disagree — the ordering the C ABI takes too."""
    with unchanged_world():
        floating = cpp.NativeTensorCore.from_array(
            np.ones((2, 2), dtype=np.float64))
        try:
            with raw_int64_core((2, 2)) as core:
                with pytest.raises(ValueError) as caught:
                    floating.add(core)
                assert "floating" in str(caught.value)
                assert "int64" in str(caught.value)
        finally:
            floating.close()


@needs_native
def test_the_core_operation_audit_covers_every_floating_entry():
    """Structural completeness, so a new floating operation cannot skip the
    barrier: every ``NativeTensorCore`` compute entry either calls the
    shared floating gate itself or reaches it through
    ``_require_matching_metadata``.

    Read from the AST rather than by substring, and driven with a negative
    control below."""
    tree = ast.parse((REPO_ROOT / "src" / "tensorforge" / "backends"
                      / "cpp.py").read_text(encoding="utf-8"))
    core = next(node for node in ast.walk(tree)
                if isinstance(node, ast.ClassDef)
                and node.name == "NativeTensorCore")
    gated = set()
    for method in core.body:
        if not isinstance(method, ast.FunctionDef):
            continue
        for node in ast.walk(method):
            if isinstance(node, ast.Attribute) and node.attr in (
                "_require_floating_operand", "_require_matching_metadata",
            ):
                gated.add(method.name)
    # Every entry that reaches a compute kernel, named explicitly.
    required = {
        "relu", "_unary_compute", "_axis_fused_forward",
        "cross_entropy_forward", "cross_entropy_backward", "relu_backward",
        "_binary_core_op", "matmul", "sum", "mean",
        "conv2d_forward", "conv2d_input_backward", "conv2d_weight_backward",
        "_maxpool2d_forward_with_winners", "maxpool2d_backward",
        "_dropout_forward_with_mask", "narrow_backward",
    }
    assert required <= gated, sorted(required - gated)


def test_the_operation_audit_scanner_can_actually_fail():
    """Negative control for the structural scan above, on a temporary
    string: a method that does not ask the gate is not reported as gated."""
    source = (
        "class NativeTensorCore:\n"
        "    def gated(self):\n"
        "        self._require_floating_operand('gated')\n"
        "    def ungated(self):\n"
        "        return 1\n"
    )
    tree = ast.parse(source)
    core = next(node for node in ast.walk(tree)
                if isinstance(node, ast.ClassDef))
    gated = set()
    for method in core.body:
        for node in ast.walk(method):
            if isinstance(node, ast.Attribute) and node.attr in (
                "_require_floating_operand", "_require_matching_metadata",
            ):
                gated.add(method.name)
    assert gated == {"gated"}


@needs_native
def test_every_floating_only_export_carries_the_c_side_guard():
    """The C ABI is a **second** authority, not a restatement of Python's.

    Structural completeness over the C++ sources: every handle-based export
    either calls ``tf::require_floating`` or is one of the four
    deliberately generalized transfer boundaries. Comments and string
    literals are stripped first, so prose about the rule cannot satisfy
    it."""
    generalized = {
        # Transfers, not arithmetic: same-type assignment at every dtype.
        "tf_storage_copy_from", "tf_storage_copy_to",
        "tf_storage_materialize", "tf_core_contiguous_copy",
        # Dtype-independent, or carrying no storage handle at all.
        "tf_storage_create", "tf_storage_create_uninitialized",
        "tf_storage_create_typed", "tf_storage_create_uninitialized_typed",
        "tf_storage_destroy", "tf_storage_size",
        "tf_elementwise_add", "tf_elementwise_subtract",
        "tf_elementwise_multiply", "tf_elementwise_divide", "tf_relu",
        "tf_matmul", "tf_matmul_tiled",
        "tf_last_error_code", "tf_last_error_message", "tf_clear_error",
        "tf_test_arm_alloc_failure", "tf_fault_injection_available",
    }
    exports, guarded = set(), set()
    for path in sorted((REPO_ROOT / "cpp" / "src").glob("*.cpp")):
        text = path.read_text(encoding="utf-8")
        code = re.sub(r"/\*.*?\*/", " ", text, flags=re.S)
        code = re.sub(r"//[^\n]*", " ", code)
        exports.update(re.findall(
            r"TF_EXPORT[^;{]*?\b(tf_[a-z0-9_]+)\s*\(", code, re.S))
        guarded.update(re.findall(
            r'tf::require_floating\(\s*"(tf_[a-z0-9_]+)"', code))
    assert len(exports) == 54, sorted(exports)
    missing = sorted(exports - guarded - generalized)
    assert missing == [], missing
    # ...and every guarded name really is an export, so a typo in a guard
    # would be caught rather than silently satisfying the rule.
    assert guarded <= exports, sorted(guarded - exports)
    assert len(guarded) == 32, sorted(guarded)


def test_the_export_guard_scanner_can_actually_fail():
    """Negative control, on temporary strings: an unguarded export is
    reported, and prose about the guard does not satisfy it."""
    unguarded = 'TF_EXPORT void tf_core_probe(const void* a) { use(a); }'
    prose = ('// tf::require_floating("tf_core_probe") protects this\n'
             'TF_EXPORT void tf_core_probe(const void* a) { use(a); }')
    guarded = ('TF_EXPORT void tf_core_probe(const void* a) {\n'
               '    if (!tf::require_floating("tf_core_probe", {a})) '
               'return;\n}')
    for source, expected in ((unguarded, set()), (prose, set()),
                             (guarded, {"tf_core_probe"})):
        code = re.sub(r"//[^\n]*", " ", source)
        found = set(re.findall(
            r'tf::require_floating\(\s*"(tf_[a-z0-9_]+)"', code))
        assert found == expected, source


@needs_native
def test_the_c_abi_rejects_int64_independently_of_python():
    """Fact 3, driven straight at the ABI with the Python layer bypassed,
    and proved to write nothing."""
    library = cpp._require_library()
    with raw_int64_storage(4) as source:
        with cpp.NativeStorage(4, dtype="float64") as destination:
            destination.fill(-7.5)
            before = destination.to_numpy().tobytes()
            shape = cpp._layout_vector([4])
            strides = cpp._layout_vector([1])
            with pytest.raises(ValueError, match="floating-only"):
                library.tf_core_relu(
                    source._handle, destination._handle, shape, strides, 0, 1
                )
            assert destination.to_numpy().tobytes() == before
            # The two scalar primitives, which have no Python errcheck hook:
            # they refuse and write nothing, and the guard they gained keeps
            # the recorded code from going stale.
            library.tf_storage_fill(source._handle, 3.0)
            assert library.tf_last_error_code() == 2   # TF_ERROR_INVALID
            assert read_int64(source).tolist() == [0, 0, 0, 0]
            library.tf_storage_scale(source._handle, 2.0)
            assert library.tf_last_error_code() == 2
            assert read_int64(source).tolist() == [0, 0, 0, 0]
            library.tf_clear_error()


@needs_native
def test_the_python_fill_barrier_rejects_before_the_native_call():
    with raw_int64_storage(4) as storage:
        with pytest.raises(ValueError, match="int64"):
            storage.fill(1.0)
        assert read_int64(storage).tolist() == [0, 0, 0, 0]
    # The negative control: a floating fill still works.
    with cpp.NativeStorage(3, dtype="float64") as floating:
        floating.fill(2.5)
        assert floating.to_numpy().tolist() == [2.5, 2.5, 2.5]


# ---------------------------------------------------------------------------
# The guarded-but-unhooked pair: fill and scale, pinned as one arrangement.
#
# ``tf_storage_fill`` and ``tf_storage_scale`` are the **only** two exceptions
# to "guarded implies hooked" (docs/native_abi_error_contract.md). The
# arrangement has four halves and is only sound while all four hold together,
# so one scanner proves them together rather than four proving them apart:
# the C-side guard exists, the names stay out of ``_CHECKED_KERNELS``, the
# Python wrapper above each validates the dtype **before** the native call,
# and no comment anywhere still files these two under "unguarded".
# ---------------------------------------------------------------------------

GUARDED_BUT_UNHOOKED = ("tf_storage_fill", "tf_storage_scale")

# A sentence that both uses the word and names one of the pair is classifying
# them; "unhooked" is a different word and a different, still-true claim.
_UNGUARDED_WORD = re.compile(r"\bunguarded\b", re.IGNORECASE)
_NAMES_THE_PAIR = re.compile(r"\b(?:tf_storage_)?(?:fill|scale)s?\b",
                             re.IGNORECASE)


def _stripped_code(source):
    """``source`` with comments and string literals replaced by blanks, so
    prose about a guard can never satisfy a scan for the guard."""
    code = re.sub(r"/\*.*?\*/", " ", source, flags=re.S)
    code = re.sub(r"//[^\n]*", " ", code)
    return re.sub(r'"(?:\\.|[^"\\])*"', '""', code)


def _export_body(source, name):
    """The brace-matched body of ``TF_EXPORT … name(…) { … }``, read from
    code alone."""
    code = _stripped_code(source)
    opening = re.search(
        r"TF_EXPORT[^;{]*?\b" + re.escape(name) + r"\s*\([^)]*\)\s*\{", code)
    assert opening is not None, name
    depth, start = 0, opening.end() - 1
    for index in range(start, len(code)):
        if code[index] == "{":
            depth += 1
        elif code[index] == "}":
            depth -= 1
            if depth == 0:
                return code[start:index + 1]
    raise AssertionError(f"unbalanced braces in {name}")


def _comment_sentences(source, suffix):
    """Every sentence of every comment in ``source``.

    Python is tokenized, so a ``#`` inside a string literal is not mistaken
    for a comment. Consecutive single-line comments are joined before the
    split, because a sentence wrapped over three lines is still one
    sentence and must be read as one."""
    fragments = []  # (line number, text)
    if suffix == ".py":
        for token in tokenize.generate_tokens(io.StringIO(source).readline):
            if token.type == tokenize.COMMENT:
                fragments.append((token.start[0], token.string.lstrip("#")))
    else:
        for match in re.finditer(r"/\*.*?\*/", source, re.S):
            fragments.append((source[:match.start()].count("\n") + 1,
                              match.group()))
        for number, line in enumerate(source.splitlines(), 1):
            if line.strip().startswith("//"):
                fragments.append((number, line.strip()[2:]))
    blocks, previous = [], None
    for number, text in sorted(fragments):
        if previous is not None and number == previous + 1:
            blocks[-1] += " " + text
        else:
            blocks.append(text)
        previous = number
    return [sentence for block in blocks
            for sentence in re.split(r"(?<=[.!?])\s+", block)]


def _misfiled_as_unguarded(source, suffix):
    """Comment sentences that classify fill or scale as unguarded."""
    return [sentence for sentence in _comment_sentences(source, suffix)
            if _UNGUARDED_WORD.search(sentence)
            and _NAMES_THE_PAIR.search(sentence)]


def _first_position(method, name):
    """Where ``name`` is first mentioned inside ``method``, as
    ``(line, column)`` — an attribute (``x.name``) or a plain call
    (``name(...)``), whichever appears earliest in the source."""
    positions = [
        (node.lineno, node.col_offset)
        for node in ast.walk(method)
        if (isinstance(node, ast.Attribute) and node.attr == name)
        or (isinstance(node, ast.Name) and node.id == name)
    ]
    return min(positions) if positions else None


def _method(tree, class_name, method_name):
    owner = next(node for node in ast.walk(tree)
                 if isinstance(node, ast.ClassDef) and node.name == class_name)
    return next(node for node in owner.body
                if isinstance(node, ast.FunctionDef)
                and node.name == method_name)


def test_fill_and_scale_are_guarded_unhooked_and_validated_in_python():
    """The four halves of the one documented exception, proved together.

    Any one of them alone is misleading: a guard with no Python check would
    leave a reachable rejection nobody raises, a Python check with no guard
    would leave the C ABI a restatement rather than a second authority, and
    adding either name to ``_CHECKED_KERNELS`` would cost every fill and
    every mean a second native call for no correctness."""
    storage_cpp = (REPO_ROOT / "cpp" / "src" / "storage.cpp").read_text(
        encoding="utf-8")
    backend = REPO_ROOT / "src" / "tensorforge" / "backends" / "cpp.py"
    backend_source = backend.read_text(encoding="utf-8")
    tree = ast.parse(backend_source)

    # 1. The C-side guard is really in both bodies, read from code alone.
    for name in GUARDED_BUT_UNHOOKED:
        body = _export_body(storage_cpp, name)
        assert "TF_GUARD_BEGIN" in body, name
        assert "TF_GUARD_END_VOID" in body, name
        assert "tf::require_floating" in body, name

    # 2. ...and both are intentionally absent from the errcheck registry,
    # which is otherwise exactly what it was (36 names, Phase H's count).
    for name in GUARDED_BUT_UNHOOKED:
        assert name not in cpp._CHECKED_KERNELS, name
    assert len(cpp._CHECKED_KERNELS) == 36
    # Non-vacuity: a genuinely hooked storage export is in the tuple, so the
    # assertion above is about these two rather than about an empty registry.
    assert "tf_storage_materialize" in cpp._CHECKED_KERNELS

    # 3. Every Python wrapper that reaches them validates the dtype first.
    for class_name, method_name, guard, native in (
        ("NativeStorage", "fill", "_require_floating_dtype",
         "tf_storage_fill"),
        ("NativeTensorCore", "mean", "_require_floating_operand",
         "tf_storage_scale"),
    ):
        method = _method(tree, class_name, method_name)
        checked = _first_position(method, guard)
        called = _first_position(method, native)
        assert checked is not None, (class_name, method_name, guard)
        assert called is not None, (class_name, method_name, native)
        assert checked < called, (class_name, method_name)
    # ...and those are the only two call sites in the whole backend module,
    # so no third path can reach either export unchecked: the only other
    # function that names them at all is ``_load_library``, which declares
    # their ``argtypes``/``restype``.
    for name, wrapper in zip(GUARDED_BUT_UNHOOKED, ("fill", "mean")):
        mentioning = {
            function.name for function in ast.walk(tree)
            if isinstance(function, ast.FunctionDef)
            and any(isinstance(node, ast.Attribute) and node.attr == name
                    for node in ast.walk(function))
        }
        assert mentioning == {"_load_library", wrapper}, (name, mentioning)

    # 4. No comment anywhere still files them under "unguarded".
    sources = [(backend_source, ".py")]
    for path in sorted((REPO_ROOT / "cpp" / "src").glob("*.cpp")):
        sources.append((path.read_text(encoding="utf-8"), ".cpp"))
    for path in sorted((REPO_ROOT / "cpp" / "include").glob("*.h")):
        sources.append((path.read_text(encoding="utf-8"), ".h"))
    misfiled = [sentence for source, suffix in sources
                for sentence in _misfiled_as_unguarded(source, suffix)]
    assert misfiled == [], misfiled


def test_the_guarded_but_unhooked_scanner_can_actually_fail():
    """Negative control for each of the four halves, on temporary sources.

    Zero findings mean something only when the detector is known to work."""
    # 1. A body without the guard is reported, and a comment about the guard
    # does not satisfy the scan.
    real = ("TF_EXPORT void tf_storage_fill(void* h, double v) {\n"
            "    TF_GUARD_BEGIN\n"
            "    if (!tf::require_floating(\"tf_storage_fill\", {h})) return;\n"
            "    TF_GUARD_END_VOID()\n"
            "}\n")
    bare = "TF_EXPORT void tf_storage_fill(void* h, double v) { use(h); }\n"
    prose = ("// TF_GUARD_BEGIN protects tf_storage_fill\n"
             "TF_EXPORT void tf_storage_fill(void* h, double v) { use(h); }\n")
    assert "TF_GUARD_BEGIN" in _export_body(real, "tf_storage_fill")
    assert "TF_GUARD_BEGIN" not in _export_body(bare, "tf_storage_fill")
    assert "TF_GUARD_BEGIN" not in _export_body(prose, "tf_storage_fill")

    # 2. The registry check notices a name that was added to it.
    assert "tf_storage_fill" not in cpp._CHECKED_KERNELS
    assert "tf_storage_fill" in cpp._CHECKED_KERNELS + ("tf_storage_fill",)

    # 3. The ordering check notices a wrapper that calls before it checks.
    ordered = ("class S:\n"
               "    def fill(self, value):\n"
               "        _require_floating_dtype(self._dtype, 'S.fill')\n"
               "        self._lib.tf_storage_fill(self._handle, value)\n")
    inverted = ("class S:\n"
                "    def fill(self, value):\n"
                "        self._lib.tf_storage_fill(self._handle, value)\n"
                "        _require_floating_dtype(self._dtype, 'S.fill')\n")
    for source, expected in ((ordered, True), (inverted, False)):
        method = _method(ast.parse(source), "S", "fill")
        checked = _first_position(method, "_require_floating_dtype")
        called = _first_position(method, "tf_storage_fill")
        assert (checked < called) is expected, source
    # ...and a wrapper missing the check entirely is not silently "in order".
    missing = ("class S:\n"
               "    def fill(self, value):\n"
               "        self._lib.tf_storage_fill(self._handle, value)\n")
    assert _first_position(_method(ast.parse(missing), "S", "fill"),
                           "_require_floating_dtype") is None

    # 4. The prose scan flags the pre-K1 wording, accepts the shipped
    # wording, and is not fooled by a `#` inside a string literal.
    stale = "# The unguarded storage kernels (destroy, size, fill, scale)\n"
    shipped = ("# The genuinely unguarded storage kernels (destroy, size)\n"
               "# never touch the slot. Fill and scale are guarded instead.\n")
    quoted = "MESSAGE = 'the unguarded fill and scale kernels'\n"
    assert len(_misfiled_as_unguarded(stale, ".py")) == 1
    assert _misfiled_as_unguarded(quoted, ".py") == []
    # The shipped wording keeps the two claims in two sentences, which is
    # exactly why sentence-level scanning is the right granularity.
    assert _misfiled_as_unguarded(shipped, ".py") == []
    stale_cpp = ("// The unguarded storage kernels (destroy, size, fill,\n"
                 "// scale) never clear the slot.\n")
    assert len(_misfiled_as_unguarded(stale_cpp, ".cpp")) == 1


# ===========================================================================
# 7. K2 and later remain unstarted
# ===========================================================================

def test_no_k3_or_k4_operation_name_exists_anywhere():
    exports = set()
    for path in sorted((REPO_ROOT / "cpp" / "src").glob("*.cpp")):
        exports.update(re.findall(
            r"TF_EXPORT[^;{]*?\b(tf_[a-z0-9_]+)\s*\(",
            path.read_text(encoding="utf-8"), re.S))
    for absent in ("tf_core_argmax", "tf_core_index_select"):
        assert absent not in exports, absent
    for inventory in (cpp.TENSOR_CORE_OPS, cpp.AUTOGRAD_OPS,
                      cpp.RAW_KERNELS, cpp.TENSOR_CORE_KERNELS):
        for banned in ("argmax", "index_select", "gather", "int64"):
            assert not [n for n in inventory if banned in n.lower()], banned


def test_the_experimental_export_list_is_still_twenty_five():
    import tensorforge.experimental as experimental

    assert len(experimental.__all__) == 25
    assert len(set(experimental.__all__)) == 25
    for name in experimental.__all__:
        assert hasattr(experimental, name), name
    for absent in ("NativeIntTensor", "NativeIndexTensor", "INDEX_DTYPES"):
        assert absent not in experimental.__all__, absent
