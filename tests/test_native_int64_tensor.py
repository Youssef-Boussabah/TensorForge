"""Phase K, milestone K2 — the public ``int64`` tensor: construction,
ownership, views, copies, and host inspection.

K2 is the milestone at which a native integer tensor becomes **publicly
constructible**, and it landed atomically. This module owns the split
``docs/native_integer_tensors_design.md`` §30.1 assigns it — construction,
ownership, views, copies, host exit — plus the one obligation the design
attaches to *this* milestone specifically: **every K1 barrier re-proved
against a real ``NativeTensor.from_int64_array`` result**, not against a
dtype string and not against a hand-assembled stand-in.

Four claims, and they only mean something together:

1. **One public door, and it converts nothing.**
   ``NativeTensor.from_int64_array`` is the only public API in the
   repository through which an ``int64`` buffer can come into existence;
   it takes exactly a ``numpy.ndarray`` of exactly native ``int64`` and
   rejects every other input rather than coercing it.
2. **The inherited machinery carries the new dtype unchanged.** Views
   borrow and cannot cast, copies own, ``close()`` is idempotent, and a
   view never frees its parent's storage — none of that is re-implemented
   for ``int64`` and all of it is driven here.
3. **Host inspection is exact.** ``to_numpy`` / ``item`` / ``tolist``
   return independent host values with every bit intact, including the
   signed extremes and values beyond float64's exact integer range.
4. **Nothing else moved.** The tensor is still refused by autograd, by
   parameters, by buffers at both persistence values, by both optimizers,
   by checkpoints, and by every floating operation; ``SUPPORTED_DTYPES``
   did not move; no generic constructor accepts ``int64``; and K3/K4
   remain absent.

Discipline this module inherits (integer design §29.6, §30.2):

* **Exact equality only** for integers — Python ``int`` comparison, raw
  ``tobytes()``, or ``numpy.int64`` array equality. **No tolerance is used
  for an integer anywhere.** Where a floating value is compared it is
  compared as raw IEEE-754 bits.
* **Every rejection is followed by a complete before/after fingerprint of
  the observable world**, and the fingerprint has its own non-vacuity
  control proving each component can notice the change it exists for.
* **Every injected failure position is a distinct injection**, each with a
  control proving it can fire, and each followed by a live-storage baseline
  check. The tracker installs itself **outside** ``monkeypatch``, so a
  mid-test ``undo()`` cannot silently disarm it.
* **Abandonment is proved by explicit ``close()``.** No assertion here
  depends on garbage-collection timing.
* **Source scans read code, not prose** — docstrings and string literals
  are stripped through the AST first, and keyword-argument names are read
  too. Every scanner has a negative control.
* No test starts a thread, touches the network, needs a Git ancestor, or
  depends on a total suite count.
"""
import ast
import contextlib
import re
from pathlib import Path

import numpy as np
import pytest

from tensorforge.backends import cpp
from tensorforge.experimental import (
    NativeAdam,
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

# Written here independently of the module under test, so a silent change
# fails rather than propagating.
INDEX_DTYPES = ("int64",)
FLOATING_DTYPES = ("float64", "float32")
ABI_CODE_INT64 = 2
INT64_MIN = -(2 ** 63)
INT64_MAX = 2 ** 63 - 1

# Values that catch width, sign, and truncation errors. 2**53 + 1 is the
# smallest positive integer float64 cannot represent, so a value past it
# surviving intact is what separates an exact integer path from one that
# took a floating detour.
PROBE_VALUES = (
    0, 1, -1, 42, -42,
    2 ** 31 - 1, 2 ** 31, -(2 ** 31), -(2 ** 31) - 1, 2 ** 32,
    2 ** 53, 2 ** 53 + 1, -(2 ** 53) - 1,
    INT64_MAX, INT64_MIN,
)


def probe_array(shape=None):
    """An exact ``int64`` array of the probe values, optionally reshaped."""
    values = np.array(PROBE_VALUES, dtype=np.int64)
    return values if shape is None else values[:int(np.prod(shape))].reshape(
        shape)


# ---------------------------------------------------------------------------
# Live-storage accounting.
#
# Installed **outside** ``monkeypatch`` on purpose (design §30.2): a
# mid-test ``undo()`` must not be able to disarm the tracker that proves a
# failure leaked nothing.
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
#
# Every rejection below is followed by a before/after comparison of this,
# and ``test_the_fingerprint_can_notice_each_change_it_exists_for`` proves
# each component can notice the change it is there for.
# ---------------------------------------------------------------------------

class World:
    """A snapshot of everything a rejected call must leave alone."""

    def __init__(self, parameter, buffer_tensor, module, optimizer,
                 generator):
        self.parameter = parameter
        self.buffer_tensor = buffer_tensor
        self.module = module
        self.optimizer = optimizer
        self.generator = generator

    def fingerprint(self):
        parameter = self.parameter
        return (
            # the parameter: identity, value bits, version, gradient
            id(parameter),
            parameter.to_numpy().tobytes(),
            parameter.version,
            parameter.requires_grad,
            None if parameter.grad is None
            else parameter.grad.to_numpy().tobytes(),
            # a registered buffer, by identity and by value
            id(self.buffer_tensor),
            self.buffer_tensor.to_numpy().tobytes(),
            tuple(sorted(name for name, _ in self.module.named_buffers())),
            tuple(sorted(name for name, _ in self.module.named_parameters())),
            # the live optimizer's charges, by identity and order
            tuple(id(p) for p in self.optimizer.parameters()),
            self.optimizer.lr,
            # a registered generator, whose call counter nothing here spends
            self.generator.state(),
            # every capability registry, which no rejection may move
            cpp.SUPPORTED_DTYPES,
            cpp.INDEX_DTYPES,
            cpp.SUPPORTED_DEVICES,
            cpp.UNSUPPORTED,
            cpp.RAW_KERNEL_DTYPES,
            tuple(sorted(cpp._DTYPE_CODES.items())),
            tuple(sorted(cpp._DTYPE_NUMPY)),
            tuple(sorted(cpp._CHECKED_HOST_ARRAYS)),
            cpp.TENSOR_CORE_OPS,
            cpp.AUTOGRAD_OPS,
            # the global RNG, which nothing here may consume
            np.random.get_state()[1][0],
        )


def _build_world():
    parameter = NativeParameter(np.array([[1.5, -2.5], [0.25, 4.0]]))
    buffer_tensor = NativeTensor.from_array(np.array([3.0, -1.0]))
    module = NativeModule()
    module.weight = parameter
    module.register_buffer("stat", buffer_tensor, persistent=True)
    generator = NativeGenerator(seed=7)
    module.register_generator("rng", generator)
    optimizer = NativeSGD([parameter], lr=0.1)
    return (World(parameter, buffer_tensor, module, optimizer, generator),
            parameter, buffer_tensor)


@contextlib.contextmanager
def unchanged_world():
    """Build the world, hand it over, and assert its fingerprint is
    byte-identical afterwards."""
    world, parameter, buffer_tensor = _build_world()
    before = world.fingerprint()
    try:
        yield world
        assert world.fingerprint() == before, "a rejection changed the world"
    finally:
        parameter.close()
        buffer_tensor.close()


# ===========================================================================
# 0. The instruments can fail
# ===========================================================================

@needs_native
def test_the_fingerprint_can_notice_each_change_it_exists_for():
    """Non-vacuity for every ``unchanged_world`` assertion below."""
    # 1. a changed parameter value (and, with it, its version)
    world, parameter, buffer_tensor = _build_world()
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
    world, parameter, buffer_tensor = _build_world()
    before = world.fingerprint()
    try:
        world.module.register_buffer("stat", None)
        assert world.fingerprint() != before
    finally:
        parameter.close()
        buffer_tensor.close()

    # 3. a gradient appearing on the parameter
    world, parameter, buffer_tensor = _build_world()
    before = world.fingerprint()
    try:
        grad = NativeTensor.from_array(np.ones((2, 2)))
        parameter._accumulate_grad(grad)
        assert world.fingerprint() != before
    finally:
        parameter.close()
        buffer_tensor.close()

    # 4. the parameter registry gaining an entry
    world, parameter, buffer_tensor = _build_world()
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

    # 5. the generator's committed-call counter advancing
    world, parameter, buffer_tensor = _build_world()
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
        parameter.close()
        buffer_tensor.close()

    # 6. a registry moving (simulated on a copy, never on the real module)
    world, parameter, buffer_tensor = _build_world()
    before = world.fingerprint()
    try:
        original = cpp.INDEX_DTYPES
        cpp.INDEX_DTYPES = ("int64", "int32")
        try:
            assert world.fingerprint() != before
        finally:
            cpp.INDEX_DTYPES = original
        assert world.fingerprint() == before
    finally:
        parameter.close()
        buffer_tensor.close()


@needs_native
def test_the_live_storage_tracker_can_actually_fail():
    """Non-vacuity for every baseline assertion below: an unclosed storage
    really is reported."""
    with pytest.raises(AssertionError, match="never closed"):
        with live_storage_baseline():
            leaked = NativeTensor.from_int64_array(
                np.array([1, 2], dtype=np.int64))
            assert leaked.dtype == "int64"      # deliberately not closed
    leaked.close()
    # ...and the balanced case passes, so the tracker is not simply broken.
    with live_storage_baseline():
        tensor = NativeTensor.from_int64_array(np.array([1], dtype=np.int64))
        tensor.close()


# ===========================================================================
# 1. Registry and dtype metadata
# ===========================================================================

def test_the_index_registry_is_exactly_int64():
    assert cpp.INDEX_DTYPES == INDEX_DTYPES
    assert isinstance(cpp.INDEX_DTYPES, tuple)
    assert len(cpp.INDEX_DTYPES) == 1


def test_the_compute_registry_did_not_move_and_never_gains_int64():
    assert cpp.SUPPORTED_DTYPES == FLOATING_DTYPES
    assert "int64" not in cpp.SUPPORTED_DTYPES
    assert not set(cpp.SUPPORTED_DTYPES) & set(cpp.INDEX_DTYPES)
    assert cpp.SUPPORTED_DEVICES == ("cpu",)
    assert cpp.UNSUPPORTED == ("cuda", "amp")
    assert cpp.RAW_KERNEL_DTYPES == ("float64",)
    assert cpp.normalize_dtype(None) == "float64"


def test_normalize_dtype_still_rejects_int64_permanently():
    """Taxonomy B's load-bearing property: the floating validator never
    accepts the index dtype, which is why no generic constructor changed."""
    with pytest.raises(ValueError, match="int64"):
        cpp.normalize_dtype("int64")
    # ...and the message names the registry it measured against, so a
    # caller is not told to look at a row that does not contain the answer.
    with pytest.raises(ValueError) as caught:
        cpp.normalize_dtype("int64")
    assert "float64" in str(caught.value) and "float32" in str(caught.value)


def test_the_dtype_tables_know_int64_exactly():
    assert cpp._DTYPE_CODES["int64"] == ABI_CODE_INT64
    assert cpp._DTYPE_ITEM_SIZES["int64"] == 8
    assert cpp._DTYPE_NUMPY["int64"] is np.int64
    # The Python side agrees with NumPy's own width, so neither can drift.
    assert np.dtype(np.int64).itemsize == cpp._DTYPE_ITEM_SIZES["int64"]
    # The two floating codes did not shift when the third arrived.
    assert cpp._DTYPE_CODES["float64"] == 0
    assert cpp._DTYPE_CODES["float32"] == 1


def test_the_generalized_no_drift_invariant_is_an_exact_equality():
    """The Phase-I guard, generalized rather than deleted (§5.1).

    Written as an equality on purpose: a subset check would be strictly
    weaker and would let a fourth representable dtype in unnoticed."""
    assert set(cpp._DTYPE_CODES) == (set(cpp.SUPPORTED_DTYPES)
                                     | set(cpp.INDEX_DTYPES))
    # ...and every table describes that same set, so none can gain an entry
    # the others do not know about.
    assert (set(cpp._DTYPE_CODES) == set(cpp._DTYPE_ITEM_SIZES)
            == set(cpp._DTYPE_NUMPY) == set(cpp._CHECKED_HOST_ARRAYS))


def test_the_host_binding_reuses_the_existing_int64_ndpointer():
    """One binding cannot diverge from itself. Building a second
    ``ndpointer`` with the same arguments would be two objects that happen
    to agree today."""
    assert cpp._CHECKED_HOST_ARRAYS["int64"] is cpp._CHECKED_I64_ARRAY
    assert cpp._CHECKED_I64_ARRAY._dtype_ == np.dtype(np.int64)


def test_backend_info_reports_the_index_row_beside_the_compute_row():
    info = cpp.backend_info()
    assert info["index_dtypes"] == INDEX_DTYPES
    assert info["supported_dtypes"] == FLOATING_DTYPES
    assert info["index_dtypes"] is cpp.INDEX_DTYPES     # not a copy/literal
    # The default is still the default: no omitted dtype selects int64.
    assert info["dtype"] == "float64"
    assert info["dtype"] not in info["index_dtypes"]
    # The union is stated in prose, never materialized as a fifth key.
    for absent in ("compute_dtypes", "tensor_dtypes", "all_dtypes",
                   "integer_dtypes", "dtypes"):
        assert absent not in info, absent
    # And the two rows the phase must never conflate stay separate.
    assert info["raw_kernel_dtypes"] == ("float64",)
    assert info["stable_framework_integration"] is False


def test_the_index_validator_is_private_narrow_and_has_no_default():
    assert cpp._normalize_index_dtype("int64") == "int64"
    for rejected in ("float64", "float32", "int32", "uint64", "Int64",
                     "INT64", " int64", "int64 ", "i8", "long", ""):
        with pytest.raises(ValueError):
            cpp._normalize_index_dtype(rejected)
    # ``None`` is a TypeError rather than a default, unlike every floating
    # validator: an index dtype has no fallback to offer.
    for bad_type in (None, 64, np.int64, np.dtype(np.int64), True, b"int64"):
        with pytest.raises(TypeError):
            cpp._normalize_index_dtype(bad_type)


def test_the_tensor_dtype_predicates_are_the_union_and_nothing_more():
    for dtype in FLOATING_DTYPES:
        assert cpp._is_floating_dtype(dtype)
        assert not cpp._is_index_dtype(dtype)
        assert cpp._is_tensor_dtype(dtype)
    assert cpp._is_index_dtype("int64")
    assert not cpp._is_floating_dtype("int64")
    assert cpp._is_tensor_dtype("int64")
    # Total, and never a validator: anything else is simply not a member.
    for outsider in ("float16", "bfloat16", "int32", "bool", None, 0, ""):
        assert not cpp._is_index_dtype(outsider)
        assert not cpp._is_tensor_dtype(outsider)


# ---------------------------------------------------------------------------
# The registry gate in front of the one public door.
#
# ``cpp._normalize_index_dtype`` is the **canonical registry gate** for the
# phase's one fixed-format construction door (design §5.2, §26.1 step 2a):
# a constructor that carries its dtype in its *name* still measures that
# name against the public ``INDEX_DTYPES`` registry, so the registry and
# the door cannot disagree.
#
# These read the production source through the AST rather than trusting a
# docstring, and the scanner has its own negative control below.
# ---------------------------------------------------------------------------

NATIVE_TENSOR_MODULE = "src/tensorforge/experimental/native_tensor.py"
CPP_MODULE = "src/tensorforge/backends/cpp.py"
INDEX_GATE = "_normalize_index_dtype"


def _module_text(relative):
    return (REPO_ROOT / relative).read_text(encoding="utf-8")


def _function_def(source, class_name, function_name):
    """One method's AST node, found by class and name."""
    tree = ast.parse(source)
    cls = next(node for node in ast.walk(tree)
               if isinstance(node, ast.ClassDef) and node.name == class_name)
    return next(child for child in cls.body
                if isinstance(child, ast.FunctionDef)
                and child.name == function_name)


def _attribute_call_lines(node, attribute):
    """The source lines of every ``<something>.<attribute>(...)`` call."""
    return sorted(call.lineno for call in ast.walk(node)
                  if isinstance(call, ast.Call)
                  and isinstance(call.func, ast.Attribute)
                  and call.func.attr == attribute)


def _raise_lines(node, exception):
    """The source lines of every ``raise <exception>(...)``."""
    return sorted(item.lineno for item in ast.walk(node)
                  if isinstance(item, ast.Raise)
                  and isinstance(item.exc, ast.Call)
                  and isinstance(item.exc.func, ast.Name)
                  and item.exc.func.id == exception)


def _functions_calling(source, name):
    """Every function in a module that calls ``name``, bare or attributed."""
    hits = set()
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for call in ast.walk(node):
            if not isinstance(call, ast.Call):
                continue
            func = call.func
            if ((isinstance(func, ast.Attribute) and func.attr == name)
                    or (isinstance(func, ast.Name) and func.id == name)):
                hits.add(node.name)
    return hits


def test_the_public_door_asks_the_index_registry_gate_with_int64():
    """Structural, not behavioural: the production door really calls
    ``cpp._normalize_index_dtype`` and really passes ``"int64"``."""
    door = _function_def(_module_text(NATIVE_TENSOR_MODULE), "NativeTensor",
                         "from_int64_array")
    calls = [call for call in ast.walk(door)
             if isinstance(call, ast.Call)
             and isinstance(call.func, ast.Attribute)
             and call.func.attr == INDEX_GATE]
    assert len(calls) == 1, "the door asks the gate exactly once"
    call, = calls
    # On the backend module itself, not on a local shadow that could drift.
    assert isinstance(call.func.value, ast.Name), ast.dump(call.func)
    assert call.func.value.id == "cpp"
    # One literal argument, and it is the one index dtype. A name or an
    # f-string here would mean the door could ask about something else.
    assert not call.keywords, "the gate takes no keyword arguments"
    assert len(call.args) == 1
    assert isinstance(call.args[0], ast.Constant)
    assert call.args[0].value == "int64"


def test_the_gate_runs_after_both_requires_grad_checks_and_before_ingress():
    """The §26.1 precedence, read out of the source in order: both
    ``requires_grad`` rejections, then the registry gate, then the private
    Core ingress — which is the first step that can allocate."""
    door = _function_def(_module_text(NATIVE_TENSOR_MODULE), "NativeTensor",
                         "from_int64_array")
    gate_line, = _attribute_call_lines(door, INDEX_GATE)
    type_error, = _raise_lines(door, "TypeError")     # requires_grad type
    value_error, = _raise_lines(door, "ValueError")   # requires_grad value
    ingress = _attribute_call_lines(door, "_from_int64_array")
    assert ingress, "the door must reach the private Core ingress"
    assert type_error < gate_line, (type_error, gate_line)
    assert value_error < gate_line, (value_error, gate_line)
    assert gate_line < min(ingress), (gate_line, ingress)
    # ...and nothing that inspects the input runs before the gate either:
    # the door hands ``values`` straight to the ingress, so the ingress
    # line above *is* the first inspection.
    assert _attribute_call_lines(door, "_from_core") > [gate_line]


def test_the_source_order_scanner_can_actually_fail():
    """Negative control for the two structural tests above, on a temporary
    string: the helpers really locate each landmark, and a door that asked
    the gate in the wrong place is reported as out of order rather than
    passing."""
    wrong = (
        "class NativeTensor:\n"
        "    @classmethod\n"
        "    def from_int64_array(cls, values, *, requires_grad=False):\n"
        "        cpp._normalize_index_dtype('int64')\n"
        "        if not isinstance(requires_grad, bool):\n"
        "            raise TypeError('x')\n"
        "        if requires_grad:\n"
        "            raise ValueError('y')\n"
        "        core = cpp.NativeTensorCore._from_int64_array(values)\n"
        "        return cls._from_core(core)\n"
    )
    door = _function_def(wrong, "NativeTensor", "from_int64_array")
    gate_line, = _attribute_call_lines(door, INDEX_GATE)
    type_error, = _raise_lines(door, "TypeError")
    value_error, = _raise_lines(door, "ValueError")
    ingress, = _attribute_call_lines(door, "_from_int64_array")
    # The landmarks are found — so "no offender" above is a measurement...
    assert gate_line and type_error and value_error and ingress
    # ...and this ordering is exactly what the real test forbids.
    assert not (type_error < gate_line and value_error < gate_line)
    assert gate_line < ingress

    # The other half of the control: a door that never asks the gate at
    # all is reported as absent rather than silently passing.
    missing = ("class NativeTensor:\n"
               "    def from_int64_array(cls, values):\n"
               "        return cpp.NativeTensorCore._from_int64_array(values)\n")
    absent = _function_def(missing, "NativeTensor", "from_int64_array")
    assert _attribute_call_lines(absent, INDEX_GATE) == []
    # ...and the literal-argument check really reads the argument.
    other = ("class NativeTensor:\n"
             "    def from_int64_array(cls, values):\n"
             "        return cpp._normalize_index_dtype('int32')\n")
    call, = [node for node in ast.walk(
        _function_def(other, "NativeTensor", "from_int64_array"))
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == INDEX_GATE]
    assert call.args[0].value == "int32" != "int64"


def test_no_floating_constructor_asks_the_index_gate():
    """The gate belongs to the one fixed-format door and to nothing else.

    A floating constructor that asked it would either be dead code or a
    second authority; either way the taxonomy would have two answers to
    "what dtype may this build?" instead of one."""
    tensor_callers = _functions_calling(_module_text(NATIVE_TENSOR_MODULE),
                                        INDEX_GATE)
    assert tensor_callers == {"from_int64_array"}, sorted(tensor_callers)
    # In the backend module the gate is *defined* and never called: no
    # storage or core constructor asks it, floating or otherwise.
    assert _functions_calling(_module_text(CPP_MODULE), INDEX_GATE) == set()
    # ...and the generic floating constructors are individually clean, at
    # both layers, so a rename could not hide one from the scan above.
    for class_name, module, names in (
        ("NativeTensor", NATIVE_TENSOR_MODULE,
         ("from_array", "zeros", "full", "_typed_from_array", "_typed_zeros")),
        ("NativeTensorCore", CPP_MODULE,
         ("from_array", "zeros", "full", "_typed", "_typed_from_array",
          "_typed_full", "_uninitialized")),
        ("NativeStorage", CPP_MODULE,
         ("__init__", "_typed", "_typed_from_array", "_uninitialized")),
    ):
        source = _module_text(module)
        for name in names:
            method = _function_def(source, class_name, name)
            assert _attribute_call_lines(method, INDEX_GATE) == [], (
                class_name, name)


@needs_native
def test_a_changed_index_registry_closes_the_door_before_any_allocation():
    """The gate is load-bearing, not decorative: with ``INDEX_DTYPES`` no
    longer listing ``"int64"``, construction is refused **before** a single
    native storage is allocated.

    The allocation counter is installed **outside** ``monkeypatch`` (§30.2)
    and the registry is restored in a ``finally``, so neither can leak."""
    allocations = []
    original_init = cpp.NativeStorage.__init__

    def counting_init(self, *args, **kwargs):
        allocations.append(kwargs.get("dtype"))
        original_init(self, *args, **kwargs)

    original_registry = cpp.INDEX_DTYPES
    cpp.NativeStorage.__init__ = counting_init
    try:
        cpp.INDEX_DTYPES = ()
        with live_storage_baseline():
            with pytest.raises(ValueError, match="index dtype"):
                NativeTensor.from_int64_array(probe_array())
        assert allocations == [], allocations
    finally:
        cpp.INDEX_DTYPES = original_registry
        cpp.NativeStorage.__init__ = original_init
    # The control that makes it non-vacuous: with the registry restored the
    # very next call succeeds and *does* allocate, so the rejection above
    # was the gate rather than a broken runtime.
    assert cpp.INDEX_DTYPES == INDEX_DTYPES
    tensor = NativeTensor.from_int64_array(probe_array())
    try:
        assert tensor.dtype == "int64"
        assert tensor.tolist() == list(PROBE_VALUES)
    finally:
        tensor.close()


# ===========================================================================
# 2. Exact construction through the one public door
# ===========================================================================

@needs_native
@pytest.mark.parametrize("value", PROBE_VALUES)
def test_every_probe_value_round_trips_exactly(value):
    """Exact integer equality, never a tolerance (§29.6)."""
    tensor = NativeTensor.from_int64_array(np.array([value], dtype=np.int64))
    try:
        assert tensor.dtype == "int64"
        out = tensor.to_numpy()
        assert out.dtype == np.dtype(np.int64)
        assert int(out[0]) == value
        assert tensor.item() == value
        assert tensor.tolist() == [value]
    finally:
        tensor.close()


@needs_native
def test_the_whole_probe_sequence_round_trips_bit_for_bit():
    source = probe_array()
    with live_storage_baseline():
        tensor = NativeTensor.from_int64_array(source)
        try:
            out = tensor.to_numpy()
            # Raw object representation, not a value comparison: a path that
            # happened to round-trip through another type would be caught.
            assert out.tobytes() == source.tobytes()
            assert np.array_equal(out, source)
        finally:
            tensor.close()


@needs_native
def test_the_signed_extremes_are_two_s_complement_on_the_way_out():
    tensor = NativeTensor.from_int64_array(
        np.array([INT64_MAX, INT64_MIN], dtype=np.int64))
    try:
        out = tensor.to_numpy()
        assert out.tobytes() == (
            INT64_MAX.to_bytes(8, "little", signed=True)
            + INT64_MIN.to_bytes(8, "little", signed=True)
        )
        assert tensor.tolist() == [INT64_MAX, INT64_MIN]
    finally:
        tensor.close()


@needs_native
def test_values_beyond_float64_precision_are_not_rounded():
    """The negative control that makes this mean something sits beside it:
    the *same* value through a float64 tensor does not survive."""
    value = 2 ** 53 + 1
    tensor = NativeTensor.from_int64_array(np.array([value], dtype=np.int64))
    try:
        assert tensor.item() == value
    finally:
        tensor.close()
    floating = NativeTensor.from_array(np.array([float(value)]))
    try:
        assert int(floating.item()) != value
    finally:
        floating.close()


@needs_native
@pytest.mark.parametrize("shape", [(), (1,), (5,), (3, 2), (2, 3, 2),
                                   (2, 1, 3, 2)])
def test_shapes_of_every_rank_are_preserved_exactly(shape):
    source = np.arange(int(np.prod(shape)), dtype=np.int64).reshape(shape)
    tensor = NativeTensor.from_int64_array(source)
    try:
        assert tensor.shape == shape
        assert tensor.ndim == len(shape)
        assert tensor.numel == int(np.prod(shape))
        assert tensor.contiguous is True
        assert tensor._core.offset == 0
        assert tensor.strides == cpp.row_major_strides(shape)
        out = tensor.to_numpy()
        assert out.shape == shape
        assert np.array_equal(out, source)
    finally:
        tensor.close()


@needs_native
def test_a_rank_zero_array_stays_rank_zero():
    """``np.ascontiguousarray`` promotes a 0-d array to ``(1,)``, which
    would be a silent rank change — the exact kind of quiet
    reinterpretation this boundary exists to refuse."""
    tensor = NativeTensor.from_int64_array(np.array(INT64_MIN, dtype=np.int64))
    try:
        assert tensor.shape == ()
        assert tensor.ndim == 0
        assert tensor.numel == 1
        assert tensor.to_numpy().shape == ()
        assert tensor.item() == INT64_MIN
        # NumPy's own rule: a rank-0 ``tolist`` returns the scalar itself.
        assert tensor.tolist() == INT64_MIN
        assert isinstance(tensor.tolist(), int)
    finally:
        tensor.close()


@needs_native
def test_a_non_contiguous_host_array_is_accepted_and_normalized():
    """Layout normalization is not conversion (§8.4): the values and their
    logical order are identical, and only where they live changes."""
    source = np.arange(6, dtype=np.int64).reshape(2, 3).T
    assert not source.flags["C_CONTIGUOUS"]
    tensor = NativeTensor.from_int64_array(source)
    try:
        assert tensor.dtype == "int64"
        assert tensor.shape == (3, 2)
        assert tensor.contiguous is True        # fresh contiguous storage
        assert np.array_equal(tensor.to_numpy(), source)
        assert tensor.tolist() == source.tolist()
    finally:
        tensor.close()


@needs_native
def test_the_result_never_aliases_the_host_array():
    source = np.array([[1, 2], [3, 4]], dtype=np.int64)
    tensor = NativeTensor.from_int64_array(source)
    try:
        source[0, 0] = 999
        source[1, 1] = -999
        assert tensor.tolist() == [[1, 2], [3, 4]]
        # ...and the exit boundary is independent in the other direction.
        out = tensor.to_numpy()
        out[0, 0] = -5
        assert tensor.tolist() == [[1, 2], [3, 4]]
    finally:
        tensor.close()


@needs_native
def test_two_calls_with_one_array_give_two_independent_tensors():
    source = np.array([7, 8, 9], dtype=np.int64)
    first = NativeTensor.from_int64_array(source)
    second = NativeTensor.from_int64_array(source)
    try:
        assert first.tolist() == second.tolist() == [7, 8, 9]
        assert first._core.storage is not second._core.storage
        first.close()
        # Closing one leaves the other completely usable.
        assert second.tolist() == [7, 8, 9]
    finally:
        first.close()
        second.close()


@needs_native
def test_the_result_is_an_owning_gradient_free_leaf():
    tensor = NativeTensor.from_int64_array(np.array([1, 2], dtype=np.int64))
    try:
        assert tensor.owns_core is True
        assert tensor.closed is False
        assert tensor.requires_grad is False
        assert tensor.grad is None
        assert tensor.is_leaf is True
        assert tensor.device == "cpu"
    finally:
        tensor.close()
    assert tensor.closed is True


@needs_native
def test_construction_and_close_return_live_storage_to_baseline():
    with live_storage_baseline():
        for _ in range(8):
            tensor = NativeTensor.from_int64_array(probe_array())
            tensor.close()
            tensor.close()                       # idempotent


@needs_native
def test_the_context_manager_works_exactly_as_it_does_for_a_float_tensor():
    with NativeTensor.from_int64_array(np.array([4, 5], dtype=np.int64)) as t:
        assert t.tolist() == [4, 5]
    assert t.closed is True


@needs_native
def test_repr_is_metadata_only_and_survives_close():
    tensor = NativeTensor.from_int64_array(np.array([[1, 2]], dtype=np.int64))
    text = repr(tensor)
    assert "shape=(1, 2)" in text
    assert "1" in text                       # the shape, not the values
    assert "requires_grad" not in text
    tensor.close()
    assert repr(tensor) == "NativeTensor(closed)"


# ===========================================================================
# 3. Rejections — the ingress converts nothing
# ===========================================================================

# Each row is (label, value, expected exception). Table-driven so a new
# input kind cannot quietly avoid the audit.
_REJECTED_INPUTS = (
    ("float64 array", np.array([1.0, 2.0]), TypeError),
    ("float64 integral values", np.array([1.0, 2.0, 3.0]), TypeError),
    ("float32 array", np.array([1.0, 2.0], dtype=np.float32), TypeError),
    ("int32 array", np.array([1, 2], dtype=np.int32), TypeError),
    ("int16 array", np.array([1, 2], dtype=np.int16), TypeError),
    ("int8 array", np.array([1, 2], dtype=np.int8), TypeError),
    ("uint8 array", np.array([1, 2], dtype=np.uint8), TypeError),
    ("uint64 array", np.array([1, 2], dtype=np.uint64), TypeError),
    ("bool array", np.array([True, False]), TypeError),
    ("object array", np.array([1, 2], dtype=object), TypeError),
    ("string array", np.array(["1", "2"]), TypeError),
    ("complex array", np.array([1 + 2j]), TypeError),
    ("big-endian int64", np.array([1, 2], dtype=">i8"), TypeError),
    ("python list", [1, 2], TypeError),
    ("python tuple", (1, 2), TypeError),
    ("python int", 5, TypeError),
    ("numpy scalar", np.int64(5), TypeError),
    ("None", None, TypeError),
    ("empty array", np.array([], dtype=np.int64), ValueError),
    ("empty 2-D array", np.zeros((0, 3), dtype=np.int64), ValueError),
)


@needs_native
@pytest.mark.parametrize("label,value,expected",
                         _REJECTED_INPUTS,
                         ids=[row[0] for row in _REJECTED_INPUTS])
def test_the_ingress_rejects_everything_that_is_not_exact_int64(
        label, value, expected):
    with live_storage_baseline():
        with pytest.raises(expected):
            NativeTensor.from_int64_array(value)


@needs_native
def test_every_rejection_leaves_the_observable_world_unchanged():
    with unchanged_world():
        with live_storage_baseline():
            for _, value, expected in _REJECTED_INPUTS:
                with pytest.raises(expected):
                    NativeTensor.from_int64_array(value)


@needs_native
def test_an_ndarray_subclass_is_rejected_rather_than_silently_flattened():
    """A subclass carries semantics a plain element copy discards, so the
    strictness starts at the type rather than at a convenience layer."""
    class Labelled(np.ndarray):
        pass

    subclass = np.array([1, 2], dtype=np.int64).view(Labelled)
    assert isinstance(subclass, np.ndarray)
    assert subclass.dtype == np.dtype(np.int64)     # only the type differs
    with pytest.raises(TypeError, match="ndarray"):
        NativeTensor.from_int64_array(subclass)
    # The negative control: the identical values as a plain ndarray pass.
    plain = np.asarray(subclass).copy()
    tensor = NativeTensor.from_int64_array(plain)
    try:
        assert tensor.tolist() == [1, 2]
    finally:
        tensor.close()


@needs_native
def test_a_byte_swapped_array_is_rejected_rather_than_swapped():
    """Host-native only (§29.4). The values are *representable*; the
    representation is not this host's, and it is refused rather than
    fixed."""
    swapped = np.array([1, 2, 3], dtype=">i8")
    assert swapped.tolist() == [1, 2, 3]         # values are fine
    with pytest.raises(TypeError):
        NativeTensor.from_int64_array(swapped)
    native = np.ascontiguousarray(swapped, dtype=np.int64)
    tensor = NativeTensor.from_int64_array(native)
    try:
        assert tensor.tolist() == [1, 2, 3]
    finally:
        tensor.close()


@needs_native
def test_requires_grad_is_validated_before_the_array_is_even_examined():
    """§26.2: the rejection order is requested route → ``requires_grad``
    type → value → input type, and every step precedes allocation."""
    good = np.array([1, 2], dtype=np.int64)
    with live_storage_baseline():
        with pytest.raises(ValueError, match="requires_grad"):
            NativeTensor.from_int64_array(good, requires_grad=True)
        for bad in (1, 0, "True", None, np.True_):
            with pytest.raises(TypeError):
                NativeTensor.from_int64_array(good, requires_grad=bad)
        # ...and the ordering is proved by a call that is invalid in *two*
        # ways at once: the requires_grad error arrives, not the dtype one.
        with pytest.raises(ValueError, match="requires_grad"):
            NativeTensor.from_int64_array(np.array([1.0]), requires_grad=True)
        with pytest.raises(TypeError, match="requires_grad"):
            NativeTensor.from_int64_array([1, 2], requires_grad=1)


@needs_native
def test_requires_grad_is_keyword_only():
    with pytest.raises(TypeError):
        NativeTensor.from_int64_array(np.array([1], dtype=np.int64), False)


@needs_native
def test_no_generic_constructor_accepts_int64():
    """§5.4 and §5.5, driven: the one door is a **new** name, never a
    widened old one, so nothing that existed before K2 changed."""
    builders = {
        "NativeStorage": lambda: cpp.NativeStorage(4, dtype="int64"),
        "NativeStorage.from_array":
            lambda: cpp.NativeStorage.from_array([1, 2], dtype="int64"),
        "NativeStorage._uninitialized":
            lambda: cpp.NativeStorage._uninitialized(4, dtype="int64"),
        "NativeStorage._typed_from_array":
            lambda: cpp.NativeStorage._typed_from_array([1, 2], "int64"),
        "NativeTensorCore.from_array":
            lambda: cpp.NativeTensorCore.from_array([1, 2], dtype="int64"),
        "NativeTensorCore.zeros":
            lambda: cpp.NativeTensorCore.zeros((2,), dtype="int64"),
        "NativeTensorCore.zeros(trusted)":
            lambda: cpp.NativeTensorCore.zeros((2,), dtype="int64",
                                               _trusted_dtype=True),
        "NativeTensorCore.full":
            lambda: cpp.NativeTensorCore.full((2,), 1, dtype="int64"),
        "NativeTensorCore._uninitialized":
            lambda: cpp.NativeTensorCore._uninitialized((2,), dtype="int64"),
        "NativeTensorCore._typed_from_array":
            lambda: cpp.NativeTensorCore._typed_from_array([1, 2], "int64"),
        "NativeTensorCore._typed_full":
            lambda: cpp.NativeTensorCore._typed_full((2,), 1, "int64"),
        "NativeTensor.from_array":
            lambda: NativeTensor.from_array([1, 2], dtype="int64"),
        "NativeTensor.zeros":
            lambda: NativeTensor.zeros((2,), dtype="int64"),
        "NativeTensor.full":
            lambda: NativeTensor.full((2,), 1, dtype="int64"),
        "NativeTensor._typed_from_array":
            lambda: NativeTensor._typed_from_array([1, 2], "int64"),
        "NativeTensor._typed_zeros":
            lambda: NativeTensor._typed_zeros((2,), "int64"),
        "NativeTensor._typed_full":
            lambda: NativeTensor._typed_full((2,), 1, "int64"),
    }
    with unchanged_world():
        with live_storage_baseline():
            for name, build in builders.items():
                with pytest.raises(ValueError, match="int64"):
                    build()
    # The negative control: the same constructors still build float tensors.
    for dtype in FLOATING_DTYPES:
        core = cpp.NativeTensorCore.zeros((2,), dtype=dtype)
        try:
            assert core.dtype == dtype
        finally:
            core.close()


@needs_native
def test_no_public_integer_constructor_exists_on_storage_or_core():
    """The single-door claim, literal rather than approximate."""
    for owner in (cpp.NativeStorage, cpp.NativeTensorCore):
        assert not hasattr(owner, "from_int64_array"), owner.__name__
        # ...while the private helper is there, and is private.
        assert hasattr(owner, "_from_int64_array"), owner.__name__
    assert hasattr(NativeTensor, "from_int64_array")
    # The two host-inspection methods live only on the tensor.
    for owner in (cpp.NativeStorage, cpp.NativeTensorCore):
        assert not hasattr(owner, "item"), owner.__name__
        assert not hasattr(owner, "tolist"), owner.__name__


@needs_native
def test_the_private_ingress_applies_the_same_contract_as_the_public_door():
    """It is not a supported way around the public validator: the exact
    same rejections apply one and two layers down."""
    with live_storage_baseline():
        for build in (cpp.NativeStorage._from_int64_array,
                      cpp.NativeTensorCore._from_int64_array):
            with pytest.raises(TypeError):
                build(np.array([1.0, 2.0]))
            with pytest.raises(TypeError):
                build([1, 2])
            with pytest.raises(ValueError):
                build(np.array([], dtype=np.int64))
        # The accepting control, at both layers.
        storage = cpp.NativeStorage._from_int64_array(
            np.array([9, -9], dtype=np.int64))
        try:
            assert storage.dtype == "int64" and storage.size == 2
            assert storage.to_numpy().tolist() == [9, -9]
        finally:
            storage.close()
        core = cpp.NativeTensorCore._from_int64_array(
            np.array([[1, 2], [3, 4]], dtype=np.int64))
        try:
            assert core.dtype == "int64" and core.shape == (2, 2)
        finally:
            core.close()


@needs_native
def test_copy_from_into_integer_storage_converts_nothing():
    """§8.3 applied at the *other* place an integer buffer can be written,
    so no route can truncate a float into integer storage."""
    storage = cpp.NativeStorage._typed(3, "int64")
    try:
        with pytest.raises(TypeError):
            storage.copy_from([1.5, 2.5, 3.5])
        with pytest.raises(TypeError):
            storage.copy_from(np.array([1.0, 2.0, 3.0]))
        with pytest.raises(TypeError):
            storage.copy_from(np.array([1, 2, 3], dtype=np.int32))
        assert storage.to_numpy().tolist() == [0, 0, 0]   # nothing written
        storage.copy_from(np.array([2 ** 62, -3, 7], dtype=np.int64))
        assert storage.to_numpy().tolist() == [2 ** 62, -3, 7]
    finally:
        storage.close()
    # The negative control: floating storage still converts, exactly as it
    # always has, so the split is by role rather than a blanket narrowing.
    floating = cpp.NativeStorage(3, dtype="float64")
    try:
        floating.copy_from([1, 2, 3])
        assert floating.to_numpy().tolist() == [1.0, 2.0, 3.0]
    finally:
        floating.close()


@needs_native
def test_integer_storage_still_refuses_fill_and_scale():
    """Both carry their scalar as a ``double``, which is inexact above
    2**53, so neither is an exact integer primitive and neither may become
    one (§22.5)."""
    storage = cpp.NativeStorage._typed(2, "int64")
    try:
        with pytest.raises(ValueError, match="int64"):
            storage.fill(1.0)
        assert storage.to_numpy().tolist() == [0, 0]
    finally:
        storage.close()


# ===========================================================================
# 4. Views, ownership, and copies
# ===========================================================================

def _view_cases(tensor):
    """Every view operation the design's §11.2 table assigns to K2."""
    return {
        "reshape": lambda: tensor.reshape((3, 2)),
        "transpose": lambda: tensor.transpose(),
        "transpose(explicit)": lambda: tensor.transpose(1, 0),
        "T": lambda: tensor.T,
        "narrow": lambda: tensor.narrow(0, 1, 1),
    }


@needs_native
def test_every_view_preserves_the_dtype_and_borrows_its_storage():
    source = np.arange(6, dtype=np.int64).reshape(2, 3)
    with live_storage_baseline():
        tensor = NativeTensor.from_int64_array(source)
        try:
            for name, build in _view_cases(tensor).items():
                view = build()
                try:
                    assert view.dtype == "int64", name
                    assert view.owns_core is False, name
                    assert view.requires_grad is False, name
                    assert view.is_leaf is True, name
                    # A view shares its parent's storage, which is exactly
                    # why it cannot cast.
                    assert view._core.storage is tensor._core.storage, name
                finally:
                    view.close()
                # Closing the view left the owner completely usable.
                assert tensor.tolist() == source.tolist(), name
        finally:
            tensor.close()


@needs_native
def test_view_metadata_is_exactly_the_floating_rules():
    source = np.arange(6, dtype=np.int64).reshape(2, 3)
    tensor = NativeTensor.from_int64_array(source)
    try:
        reshaped = tensor.reshape((3, 2))
        try:
            assert reshaped.shape == (3, 2)
            assert reshaped.strides == (2, 1)
            assert reshaped.contiguous is True
            assert np.array_equal(reshaped.to_numpy(), source.reshape(3, 2))
        finally:
            reshaped.close()

        transposed = tensor.T
        try:
            assert transposed.shape == (3, 2)
            assert transposed.strides == (1, 3)
            assert transposed.contiguous is False
            # Logical order, not storage order.
            assert np.array_equal(transposed.to_numpy(), source.T)
        finally:
            transposed.close()

        narrowed = tensor.narrow(1, 1, 2)
        try:
            assert narrowed.shape == (2, 2)
            assert narrowed._core.offset == 1
            assert np.array_equal(narrowed.to_numpy(), source[:, 1:3])
        finally:
            narrowed.close()
    finally:
        tensor.close()


@needs_native
def test_a_chained_view_keeps_the_whole_chain_reachable():
    source = np.arange(12, dtype=np.int64).reshape(3, 4)
    tensor = NativeTensor.from_int64_array(source)
    try:
        first = tensor.narrow(0, 1, 2)
        second = first.T
        third = second.narrow(0, 1, 2)
        try:
            expected = source[1:3].T[1:3]
            assert third.dtype == "int64"
            assert np.array_equal(third.to_numpy(), expected)
            assert third.tolist() == expected.tolist()
        finally:
            third.close()
            second.close()
            first.close()
    finally:
        tensor.close()


@needs_native
def test_closing_the_owner_leaves_a_live_view_rejecting_rather_than_reading():
    tensor = NativeTensor.from_int64_array(
        np.arange(6, dtype=np.int64).reshape(2, 3))
    view = tensor.T
    try:
        assert view.dtype == "int64"
        tensor.close()
        with pytest.raises(RuntimeError):
            view.to_numpy()
        with pytest.raises(RuntimeError):
            view.tolist()
        # Metadata that never touches the buffer is still readable.
        assert view.shape == (3, 2)
        assert view.closed is False
    finally:
        view.close()


@needs_native
def test_close_is_idempotent_at_every_layer():
    tensor = NativeTensor.from_int64_array(np.array([1, 2], dtype=np.int64))
    view = tensor.reshape((2, 1))
    view.close()
    view.close()
    tensor.close()
    tensor.close()
    assert tensor.closed and view.closed
    with pytest.raises(RuntimeError):
        tensor.to_numpy()


@needs_native
def test_operations_on_a_closed_integer_tensor_reject_clearly():
    tensor = NativeTensor.from_int64_array(np.array([1, 2], dtype=np.int64))
    tensor.close()
    for name, call in (
        ("to_numpy", tensor.to_numpy),
        ("item", tensor.item),
        ("tolist", tensor.tolist),
        ("reshape", lambda: tensor.reshape((2, 1))),
        ("transpose", tensor.transpose),
        ("narrow", lambda: tensor.narrow(0, 0, 1)),
        ("contiguous_copy", tensor.contiguous_copy),
        ("dtype", lambda: tensor.dtype),
        ("shape", lambda: tensor.shape),
    ):
        with pytest.raises(RuntimeError, match="closed"):
            call()


@needs_native
def test_contiguous_copy_owns_independent_storage_and_preserves_values():
    source = probe_array((5, 3))
    with live_storage_baseline():
        tensor = NativeTensor.from_int64_array(source)
        try:
            copy = tensor.contiguous_copy()
            try:
                assert copy.dtype == "int64"
                assert copy.owns_core is True
                assert copy.contiguous is True
                assert copy._core.storage is not tensor._core.storage
                assert copy.to_numpy().tobytes() == source.tobytes()
                # Independent lifetime: closing the source leaves it alive.
                tensor.close()
                assert np.array_equal(copy.to_numpy(), source)
            finally:
                copy.close()
        finally:
            tensor.close()


@needs_native
def test_a_non_contiguous_view_materializes_in_logical_order():
    """The claim that makes views usable: `contiguous_copy` reproduces the
    logical order exactly, so the copy and `to_numpy` agree."""
    source = np.arange(12, dtype=np.int64).reshape(3, 4)
    tensor = NativeTensor.from_int64_array(source)
    try:
        view = tensor.T
        try:
            assert view.contiguous is False
            copy = view.contiguous_copy()
            try:
                assert copy.contiguous is True
                assert copy.shape == (4, 3)
                expected = np.ascontiguousarray(source.T)
                assert copy.to_numpy().tobytes() == expected.tobytes()
                assert copy.tolist() == view.tolist()
            finally:
                copy.close()
        finally:
            view.close()
    finally:
        tensor.close()


@needs_native
def test_the_integer_contiguous_copy_destination_is_zero_initialized():
    """§27.3: no integer path uses the uninitialized allocator, so the H1
    audit table gains no row and needs no integer poison test.

    Proved structurally rather than by observing values — the kernel
    overwrites every element, so a value check could not tell the two
    allocators apart. The floating arm must still take the uninitialized
    path, or the H1 coverage proof would have quietly changed."""
    calls = []
    original = cpp.NativeTensorCore._uninitialized.__func__

    def recording(cls, shape, dtype="float64", device="cpu"):
        calls.append(dtype)
        return original(cls, shape, dtype=dtype, device=device)

    cpp.NativeTensorCore._uninitialized = classmethod(recording)
    try:
        integer = NativeTensor.from_int64_array(
            np.arange(4, dtype=np.int64).reshape(2, 2))
        try:
            copy = integer.contiguous_copy()
            copy.close()
        finally:
            integer.close()
        assert calls == [], f"an int64 copy took the uninitialized path: {calls}"

        floating = NativeTensor.from_array(np.zeros((2, 2)))
        try:
            copy = floating.contiguous_copy()
            copy.close()
        finally:
            floating.close()
        assert calls == ["float64"], calls
    finally:
        cpp.NativeTensorCore._uninitialized = classmethod(original)


@needs_native
def test_detach_returns_an_owning_integer_copy():
    """Inherited behavior, stated rather than assumed: ``detach`` is a
    ``contiguous_copy`` under the hood, and an integer tensor was never in
    a graph to begin with."""
    tensor = NativeTensor.from_int64_array(np.array([INT64_MAX, -1],
                                                    dtype=np.int64))
    try:
        detached = tensor.detach()
        try:
            assert detached.dtype == "int64"
            assert detached.owns_core is True
            assert detached.requires_grad is False
            assert detached.tolist() == [INT64_MAX, -1]
            assert detached._core.storage is not tensor._core.storage
        finally:
            detached.close()
    finally:
        tensor.close()


@needs_native
def test_a_view_over_a_probe_sequence_keeps_every_value_exact():
    """Views must not lose the values a floating detour would round."""
    source = probe_array((5, 3))
    tensor = NativeTensor.from_int64_array(source)
    try:
        view = tensor.T
        try:
            assert np.array_equal(view.to_numpy(), source.T)
            assert view.to_numpy().tobytes() == \
                np.ascontiguousarray(source.T).tobytes()
        finally:
            view.close()
    finally:
        tensor.close()


# ===========================================================================
# 5. Host inspection
# ===========================================================================

@needs_native
def test_to_numpy_returns_a_fresh_independent_exact_int64_array():
    source = probe_array((5, 3))
    tensor = NativeTensor.from_int64_array(source)
    try:
        first = tensor.to_numpy()
        second = tensor.to_numpy()
        assert first.dtype == np.dtype(np.int64)
        assert first.shape == source.shape
        assert first is not second
        assert first.tobytes() == second.tobytes() == source.tobytes()
        # The returned array owns its memory, so mutating it reaches nothing.
        first[0, 0] = 12345
        assert tensor.to_numpy()[0, 0] == source[0, 0]
        # ...and the tensor's lifetime does not govern the array's.
        held = tensor.to_numpy()
        tensor.close()
        assert held.tobytes() == source.tobytes()
    finally:
        tensor.close()


@needs_native
@pytest.mark.parametrize("shape", [(), (1,), (1, 1), (1, 1, 1)])
def test_item_accepts_one_element_at_any_rank(shape):
    tensor = NativeTensor.from_int64_array(
        np.full(shape, INT64_MIN, dtype=np.int64))
    try:
        value = tensor.item()
        assert value == INT64_MIN
        assert type(value) is int          # a built-in, never a NumPy scalar
        assert not isinstance(value, np.integer)
    finally:
        tensor.close()


@needs_native
def test_item_rejects_a_non_scalar_naming_the_actual_count():
    for shape in ((2,), (2, 3), (1, 4)):
        tensor = NativeTensor.from_int64_array(
            np.ones(shape, dtype=np.int64))
        try:
            with pytest.raises(ValueError) as caught:
                tensor.item()
            assert str(int(np.prod(shape))) in str(caught.value)
        finally:
            tensor.close()


@needs_native
def test_item_is_dtype_general_and_returns_the_right_builtin():
    for dtype in FLOATING_DTYPES:
        tensor = NativeTensor.from_array(np.array([1.5]), dtype=dtype)
        try:
            value = tensor.item()
            assert type(value) is float
            assert value == 1.5
        finally:
            tensor.close()
    integer = NativeTensor.from_int64_array(np.array([7], dtype=np.int64))
    try:
        assert type(integer.item()) is int
    finally:
        integer.close()


@needs_native
def test_tolist_preserves_nested_shape_and_returns_builtin_ints():
    source = np.array([[INT64_MAX, -1], [0, INT64_MIN]], dtype=np.int64)
    tensor = NativeTensor.from_int64_array(source)
    try:
        listed = tensor.tolist()
        assert listed == [[INT64_MAX, -1], [0, INT64_MIN]]
        assert type(listed) is list and type(listed[0]) is list
        for row in listed:
            for value in row:
                assert type(value) is int
                assert not isinstance(value, np.integer)
    finally:
        tensor.close()


@needs_native
def test_tolist_follows_logical_order_on_a_non_contiguous_view():
    source = np.arange(6, dtype=np.int64).reshape(2, 3)
    tensor = NativeTensor.from_int64_array(source)
    try:
        view = tensor.T
        try:
            assert view.tolist() == source.T.tolist()
        finally:
            view.close()
    finally:
        tensor.close()


@needs_native
def test_tolist_is_dtype_general_and_returns_builtin_floats():
    tensor = NativeTensor.from_array(np.array([[1.5, -2.5]]))
    try:
        listed = tensor.tolist()
        assert listed == [[1.5, -2.5]]
        assert all(type(v) is float for v in listed[0])
    finally:
        tensor.close()


@needs_native
def test_inspection_builds_no_graph_and_touches_no_state():
    """§16.2: a graph built before an inspection call is fully usable
    after it, and no version, gradient, or generator moved."""
    with unchanged_world() as world:
        x = NativeTensor.from_array(np.array([2.0, 3.0]), requires_grad=True)
        try:
            y = x.multiply(x)
            try:
                integer = NativeTensor.from_int64_array(
                    np.array([1, 2], dtype=np.int64))
                try:
                    integer.to_numpy()
                    integer.tolist()
                    integer.narrow(0, 0, 1).close()
                finally:
                    integer.close()
                loss = y.sum()
                try:
                    loss.backward()
                finally:
                    loss.close()
            finally:
                y.close()
            assert x.grad.to_numpy().tobytes() == \
                np.array([4.0, 6.0]).tobytes()
        finally:
            x.close()
        assert world.parameter.version == 0


# ===========================================================================
# 6. Every K1 barrier, re-proved against a real int64 tensor
# ===========================================================================

@contextlib.contextmanager
def integer_tensor(shape=(2, 2)):
    """A **real** public integer tensor — the whole point of this section.

    K1 drove its barriers from a hand-assembled object over a raw C ABI
    handle, because that was the only int64 there was. K2 must drive them
    from the object the public door actually returns."""
    values = np.arange(int(np.prod(shape)), dtype=np.int64).reshape(shape) + 1
    tensor = NativeTensor.from_int64_array(values)
    try:
        yield tensor
    finally:
        tensor.close()


@needs_native
def test_the_real_tensor_is_what_the_barriers_are_being_driven_with():
    """Non-vacuity for the whole section: the operand really is a public
    ``int64`` tensor, so a barrier that fired for another reason would be
    visible."""
    with integer_tensor((2, 3)) as tensor:
        assert isinstance(tensor, NativeTensor)
        assert tensor.dtype == "int64"
        assert tensor.shape == (2, 3)
        assert tensor.closed is False
        assert tensor.owns_core is True
        assert tensor.tolist() == [[1, 2, 3], [4, 5, 6]]


@needs_native
def test_autograd_refuses_a_real_integer_tensor():
    with unchanged_world():
        with integer_tensor() as tensor:
            # Construction with gradient tracking, before any allocation.
            with pytest.raises(ValueError, match="requires_grad"):
                NativeTensor.from_int64_array(
                    np.array([1], dtype=np.int64), requires_grad=True)
            # The graph-construction backstop.
            parent = NativeTensor.from_array(np.array([1.0]),
                                             requires_grad=True)
            try:
                core = cpp.NativeTensorCore._from_int64_array(
                    np.array([1], dtype=np.int64))
                with pytest.raises(ValueError, match="int64"):
                    NativeTensor._from_op(core, (parent,), lambda g: None,
                                          "probe")
                assert core._closed, "a rejected graph leaked its core"
            finally:
                parent.close()
            # backward and gradient accumulation.
            with pytest.raises(RuntimeError, match="int64"):
                tensor.backward()
            contribution = NativeTensor.from_array(np.ones((2, 2)))
            try:
                with pytest.raises(RuntimeError, match="int64"):
                    tensor._accumulate_grad(contribution)
            finally:
                contribution.close()
            assert tensor.grad is None
            assert tensor.requires_grad is False
            assert tensor.is_leaf is True


@needs_native
def test_from_op_also_releases_the_saved_state_it_was_handed():
    """A rejected graph leaks nothing at all — the saved resources go the
    same way as the core."""
    parent = NativeTensor.from_array(np.array([1.0]), requires_grad=True)
    saved = NativeTensor.from_array(np.array([5.0]))
    try:
        core = cpp.NativeTensorCore._from_int64_array(
            np.array([1], dtype=np.int64))
        with pytest.raises(ValueError, match="int64"):
            NativeTensor._from_op(core, (parent,), lambda g: None, "probe",
                                  graph_resources=(saved,))
        assert core._closed
        assert saved.closed, "a rejected graph kept its saved state alive"
    finally:
        parent.close()


@needs_native
def test_parameters_refuse_a_real_integer_tensor():
    with unchanged_world():
        with integer_tensor() as tensor:
            with pytest.raises(ValueError, match="int64"):
                NativeParameter(tensor)
            # ...and naming the dtype fails through the narrowed module
            # validator, before the source is even examined.
            with pytest.raises(ValueError, match="int64"):
                NativeParameter(tensor, dtype="int64")
            with pytest.raises(ValueError, match="int64"):
                NativeParameter([1.0], dtype="int64")


@needs_native
@pytest.mark.parametrize("persistent", [True, False])
def test_buffers_of_either_persistence_refuse_a_real_integer_tensor(
        persistent):
    module = NativeModule()
    with integer_tensor((2,)) as tensor:
        with pytest.raises(ValueError, match="int64"):
            module.register_buffer("stat", tensor, persistent=persistent)
    # The registration left no trace at all.
    assert list(module.named_buffers()) == []
    assert not hasattr(module, "stat")
    assert module.state_dict() == {}


@needs_native
def test_both_optimizers_refuse_a_real_integer_tensor():
    with unchanged_world():
        with integer_tensor((2,)) as tensor:
            # The type check rejects first — the pre-existing transitive
            # closure, which holds because an integer tensor can never be a
            # NativeParameter in the first place.
            with pytest.raises(TypeError, match="NativeParameter"):
                NativeSGD([tensor], lr=0.1)
            with pytest.raises(TypeError, match="NativeParameter"):
                NativeAdam([tensor])
            # And the direct per-parameter check, driven against a
            # NativeParameter-shaped object carrying the int64 tag — the
            # case that would survive if the barrier were only transitive.
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
def test_a_live_optimizer_is_untouched_by_a_rejected_integer_registration():
    parameter = NativeParameter(np.array([1.0, 2.0]))
    try:
        optimizer = NativeAdam([parameter])
        try:
            def snapshot():
                """A comparable fingerprint of the optimizer's state.

                ``state_dict()`` hands back **fresh** ``NativeTensor``
                moment snapshots at every call, so comparing the
                dictionaries directly would compare object identities and
                always differ. The moments are compared as raw bytes and
                closed explicitly."""
                state = optimizer.state_dict()
                moments = []
                for key in ("m", "v"):
                    for tensor in state.get(key, []):
                        moments.append(tensor.to_numpy().tobytes())
                        tensor.close()
                return (tuple(id(p) for p in optimizer.parameters()),
                        state["format_version"], state["optimizer"],
                        state["lr"], state["betas"], state["eps"],
                        state["parameters"], state["step_counts"],
                        tuple(moments))

            before = snapshot()
            with integer_tensor((2,)) as tensor:
                with pytest.raises(TypeError):
                    NativeAdam([parameter, tensor])
            assert snapshot() == before
            assert parameter.version == 0
            assert parameter.to_numpy().tolist() == [1.0, 2.0]
        finally:
            optimizer.close()
    finally:
        parameter.close()


def test_no_archive_entry_may_declare_int64_and_the_versions_are_unmoved():
    from tensorforge.experimental import (native_checkpoint,
                                          native_data_loader,
                                          native_optimizer_state,
                                          native_sampler)

    for version in native_checkpoint._SUPPORTED_FORMAT_VERSIONS:
        with pytest.raises(ValueError, match="int64"):
            native_checkpoint._validated_entry_dtype(
                "int64", version, "manifest['model']['entries']['w']",
                "load_native_checkpoint",
            )
    # The negative control: the dtypes an archive *may* declare still pass.
    assert native_checkpoint._validated_entry_dtype(
        "float64", 3, "e", "w") == "float64"
    assert native_checkpoint._validated_entry_dtype(
        "float32", 3, "e", "w") == "float32"
    # Every version row K2 must not move.
    assert native_checkpoint._FORMAT == "tensorforge.native_checkpoint"
    assert native_checkpoint._FORMAT_VERSION == 3
    assert native_checkpoint._SUPPORTED_FORMAT_VERSIONS == (1, 2, 3)
    assert 4 not in native_checkpoint._SUPPORTED_FORMAT_VERSIONS
    assert native_optimizer_state.FORMAT_VERSION == 1
    assert native_data_loader._FORMAT_VERSION == 1
    assert native_data_loader._SUPPORTED_FORMAT_VERSIONS == (1,)
    assert native_sampler._FORMAT_VERSION == 1
    assert native_sampler._SUPPORTED_FORMAT_VERSIONS == (1,)


@needs_native
def test_an_integer_tensor_has_no_route_into_module_state_or_a_checkpoint(
        tmp_path):
    """Three independent layers (§10.3), so a single point of failure
    cannot open the route: it cannot be a parameter, it cannot be a buffer,
    and an archive cannot declare its dtype."""
    from tensorforge.experimental import (load_native_checkpoint,
                                          save_native_checkpoint)

    module = NativeModule()
    module.weight = NativeParameter(np.array([[1.0, 2.0]]))
    try:
        with integer_tensor((1, 2)) as tensor:
            with pytest.raises(ValueError, match="int64"):
                module.indices = NativeParameter(tensor)
            with pytest.raises(ValueError, match="int64"):
                module.register_buffer("indices", tensor)
        # The state dictionary and the archive contain floating state only.
        state = module.state_dict()
        assert set(state) == {"weight"}
        path = tmp_path / "model.npz"
        save_native_checkpoint(str(path), module)
        loaded = NativeModule()
        loaded.weight = NativeParameter(np.zeros((1, 2)))
        try:
            load_native_checkpoint(str(path), loaded)
            assert loaded.weight.to_numpy().tolist() == [[1.0, 2.0]]
        finally:
            loaded.weight.close()
    finally:
        module.weight.close()


# One row per public floating operation family, as a callable taking the
# real integer tensor and a floating partner. Table-driven so a new
# operation cannot quietly avoid the audit.
def _floating_operations(tensor, floating, generator):
    return {
        # unary arithmetic / activations
        "relu": lambda: tensor.relu(),
        "sqrt": lambda: tensor.sqrt(),
        "reciprocal": lambda: tensor.reciprocal(),
        "exp": lambda: tensor.exp(),
        "log": lambda: tensor.log(),
        # binary arithmetic, integer on the left
        "add": lambda: tensor.add(floating),
        "subtract": lambda: tensor.subtract(floating),
        "multiply": lambda: tensor.multiply(floating),
        # ...and integer on the right, which is a different operand slot
        "add(mixed, float first)": lambda: floating.add(tensor),
        "subtract(mixed, float first)": lambda: floating.subtract(tensor),
        "multiply(mixed, float first)": lambda: floating.multiply(tensor),
        # matrix operations, both operand positions
        "matmul": lambda: tensor.matmul(floating),
        "matmul(mixed, float first)": lambda: floating.matmul(tensor),
        # reductions
        "sum": lambda: tensor.sum(),
        "sum(axis)": lambda: tensor.sum(axis=0),
        "mean": lambda: tensor.mean(),
        "mean(axis)": lambda: tensor.mean(axis=1, keepdims=True),
        # normalization / classification
        "softmax": lambda: tensor.softmax(),
        "log_softmax": lambda: tensor.log_softmax(axis=-1),
        "cross_entropy": lambda: tensor.cross_entropy(
            np.array([0, 1], dtype=np.int64)),
        # convolution and pooling
        "conv2d": lambda: tensor.reshape((1, 1, 2, 2)).conv2d(
            floating.reshape((1, 1, 2, 2))),
        "maxpool2d": lambda: tensor.reshape((1, 1, 2, 2)).maxpool2d(
            kernel_size=1),
        # the random path
        "dropout": lambda: tensor.dropout(0.5, generator=generator),
    }


@needs_native
def test_every_floating_operation_refuses_a_real_integer_operand():
    with unchanged_world():
        with live_storage_baseline():
            with integer_tensor((2, 2)) as tensor:
                floating = NativeTensor.from_array(np.ones((2, 2)))
                generator = NativeGenerator(seed=3)
                try:
                    operations = _floating_operations(tensor, floating,
                                                      generator)
                    assert len(operations) == 23, "the operation audit shrank"
                    for name, call in operations.items():
                        with pytest.raises(ValueError, match="int64"):
                            call()
                    # Nothing was consumed and nothing was written.
                    assert generator.state()["calls"] == 0
                    assert floating.to_numpy().tolist() == [[1.0, 1.0],
                                                            [1.0, 1.0]]
                    assert tensor.tolist() == [[1, 2], [3, 4]]
                finally:
                    floating.close()


@needs_native
def test_a_mixed_float_integer_request_is_reported_as_a_role_error():
    """An integer operand is refused as *floating-only*, never as two
    dtypes that disagree — the ordering the C ABI takes too (§12.4)."""
    with integer_tensor((2, 2)) as tensor:
        floating = NativeTensor.from_array(np.ones((2, 2)))
        try:
            for call in (lambda: floating.add(tensor),
                         lambda: tensor.add(floating)):
                with pytest.raises(ValueError) as caught:
                    call()
                message = str(caught.value)
                assert "floating" in message
                assert "int64" in message
        finally:
            floating.close()


@needs_native
def test_the_floating_negative_control_still_computes():
    """Every rejection above means something only if the same operations
    still work on floating operands."""
    a = NativeTensor.from_array(np.array([[1.0, 2.0], [3.0, 4.0]]))
    b = NativeTensor.from_array(np.ones((2, 2)))
    try:
        for build in (lambda: a.add(b), lambda: a.matmul(b), lambda: a.sum(),
                      lambda: a.relu(), lambda: a.mean(axis=0)):
            result = build()
            assert result.dtype == "float64"
            result.close()
    finally:
        a.close()
        b.close()


def _is_property(node):
    """True when an AST function definition is decorated ``@property``."""
    return any(isinstance(decorator, ast.Name) and decorator.id == "property"
               for decorator in node.decorator_list)


@needs_native
def test_the_operation_audit_covers_every_public_floating_tensor_entry():
    """Structural completeness, so a new operation cannot skip the audit:
    every ``NativeTensor`` compute method is either in the table above or
    is explicitly classified as dtype-general.

    Read from the AST rather than by substring, with a negative control
    below. Properties are excluded because they are metadata readers rather
    than operations — §11.1 lists every one of them as supported at every
    dtype — and ``_is_property`` is driven by the control too."""
    tree = ast.parse((REPO_ROOT / "src" / "tensorforge" / "experimental"
                      / "native_tensor.py").read_text(encoding="utf-8"))
    cls = next(node for node in ast.walk(tree)
               if isinstance(node, ast.ClassDef) and node.name
               == "NativeTensor")
    public = {child.name for child in cls.body
              if isinstance(child, ast.FunctionDef)
              and not child.name.startswith("_")
              and not _is_property(child)}
    # Names that legitimately work at every dtype a tensor may carry.
    dtype_general = {
        "from_array", "zeros", "full", "from_int64_array",
        "to_numpy", "item", "tolist",
        "reshape", "transpose", "narrow", "contiguous_copy", "detach",
        "zero_grad", "backward", "close",
    }
    audited = {"relu", "sqrt", "reciprocal", "exp", "log", "add", "subtract",
               "multiply", "matmul", "sum", "mean", "softmax", "log_softmax",
               "cross_entropy", "conv2d", "maxpool2d", "dropout"}
    unclassified = public - dtype_general - audited
    assert unclassified == set(), sorted(unclassified)
    # ...and every audited name really is a method, so a typo in the set
    # above would be caught rather than silently shrinking the audit.
    assert audited <= public, sorted(audited - public)


def test_the_operation_audit_scanner_can_actually_fail():
    """Negative control for the structural scan above, on a temporary
    string: an unclassified public method is reported, a private one is
    not, and a property is correctly excluded rather than reported as an
    unaudited operation."""
    source = ("class NativeTensor:\n"
              "    def relu(self):\n        pass\n"
              "    def brand_new_op(self):\n        pass\n"
              "    def _private(self):\n        pass\n"
              "    @property\n"
              "    def shape(self):\n        pass\n")
    tree = ast.parse(source)
    cls = next(node for node in ast.walk(tree)
               if isinstance(node, ast.ClassDef))
    public = {child.name for child in cls.body
              if isinstance(child, ast.FunctionDef)
              and not child.name.startswith("_")
              and not _is_property(child)}
    assert public == {"relu", "brand_new_op"}
    assert public - {"relu"} == {"brand_new_op"}
    # ...and the property filter really is doing work, rather than the set
    # happening to exclude it for another reason.
    with_properties = {child.name for child in cls.body
                       if isinstance(child, ast.FunctionDef)
                       and not child.name.startswith("_")}
    assert "shape" in with_properties


# ===========================================================================
# 7. Failure atomicity — distinct injections, each with a control
# ===========================================================================
#
# Three of the injections below share one instrument and differ in the
# *position* they fire from, which is the only thing that makes them three
# tests rather than one: publication after ``from_int64_array``'s ingress,
# publication after ``contiguous_copy``'s destination allocation, and
# publication after ``detach``'s copy. Each names its own position.
#
# The instrument retains the core it was handed in an **external list**
# before raising, and that retention is the whole point (design §30.2's
# rule that abandonment is proved by explicit ``close()``, never by
# collection timing). Without it, the injected ``__init__`` would drop the
# only reference to the core as it unwound, CPython's reference counting
# would run ``NativeTensorCore.__del__``, the storage would be released by
# the safety net, and an implementation with **no** explicit cleanup would
# pass the baseline check anyway. Holding the core alive removes that
# rescue: the storage is freed only if the caller freed it.
#
# The instrument also never closes the core itself, so what the assertions
# below observe is the caller's cleanup and nothing else.
# ---------------------------------------------------------------------------

@contextlib.contextmanager
def retained_publication_failure(exception):
    """Make ``NativeTensor.__init__`` retain its core and then raise.

    Yields the external list the cores land in. The constructor is restored
    in a ``finally`` so the injection can never outlive the block, and it is
    patched **outside** ``monkeypatch`` for the tracker's reason: a mid-test
    ``undo()`` must not be able to disarm it."""
    retained = []
    original = NativeTensor.__init__

    def failing(self, core, owns_core=True):
        # A strong reference the caller keeps, established *before* the
        # raise: from here on nothing but an explicit close can free this
        # core's storage.
        retained.append(core)
        raise exception

    NativeTensor.__init__ = failing
    try:
        yield retained
    finally:
        NativeTensor.__init__ = original


def _assert_core_was_explicitly_closed(retained):
    """Exactly one core was handed to the failing constructor, and it was
    closed — while this list still holds it, so ``__del__`` cannot be what
    closed it."""
    assert len(retained) == 1, retained
    core = retained[0]
    assert core._closed is True, "the core was never closed"
    assert core.storage._handle is None, "the core's storage was never released"


@needs_native
def test_the_retained_reference_instrument_can_actually_fail():
    """Non-vacuity for the three publication-failure tests below, in both
    directions.

    1. A core retained the same way but **not** closed is reported as open,
       so the assertion is a measurement rather than a formality.
    2. The retained reference really does suppress the ``__del__`` safety
       net — the core stays open, and readable, for as long as the list
       holds it. That is what makes an unprotected implementation fail
       instead of being rescued by CPython reference counting."""
    retained = [cpp.NativeTensorCore._from_int64_array(probe_array())]
    try:
        with pytest.raises(AssertionError, match="never closed"):
            _assert_core_was_explicitly_closed(retained)
        assert retained[0].storage._handle is not None
        assert retained[0].to_numpy().tolist() == list(PROBE_VALUES)
    finally:
        retained[0].close()
    # ...and it passes once the core really is closed.
    _assert_core_was_explicitly_closed(retained)


@needs_native
def test_a_native_allocation_failure_leaves_no_storage_behind():
    """Injection 1 of 3: the deterministic thread-local allocation arm, the
    only allocation-failure seam the runtime has. Disarmed in a ``finally``
    so it can never leak into another test."""
    with live_storage_baseline():
        cpp._arm_alloc_failure(1)
        try:
            with pytest.raises(MemoryError):
                NativeTensor.from_int64_array(probe_array())
        finally:
            cpp._arm_alloc_failure(0)
    # The control that makes it non-vacuous: the very next call succeeds,
    # so the failure was the injection rather than a broken runtime.
    tensor = NativeTensor.from_int64_array(probe_array())
    try:
        assert tensor.numel == len(PROBE_VALUES)
    finally:
        tensor.close()


@needs_native
def test_a_failing_host_to_native_transfer_closes_the_storage_it_allocated():
    """Injection 2 of 3: the transfer step, which is a **different**
    position from the allocation above and is asserted as one."""
    calls = []
    original = cpp.NativeStorage.copy_from

    def failing(self, values):
        calls.append(self.dtype)
        raise RuntimeError("injected transfer failure")

    with live_storage_baseline():
        cpp.NativeStorage.copy_from = failing
        try:
            with pytest.raises(RuntimeError, match="injected transfer"):
                NativeTensor.from_int64_array(probe_array())
        finally:
            cpp.NativeStorage.copy_from = original
    assert calls == ["int64"], calls        # the injection really fired


@needs_native
def test_a_base_exception_through_the_transfer_frees_exactly_the_same():
    """The cleanup is unconditional, so a ``KeyboardInterrupt`` between
    allocation and publication frees exactly what a ``ValueError`` would."""
    original = cpp.NativeStorage.copy_from

    def interrupting(self, values):
        raise KeyboardInterrupt("injected")

    with live_storage_baseline():
        cpp.NativeStorage.copy_from = interrupting
        try:
            with pytest.raises(KeyboardInterrupt):
                NativeTensor.from_int64_array(probe_array())
        finally:
            cpp.NativeStorage.copy_from = original


@needs_native
def test_a_failing_wrapper_construction_closes_the_core_it_was_handed():
    """Injection 3 of 3, **publication position 1 of 3**: the step after
    ``from_int64_array``'s ingress has produced storage and a core, but
    before the wrapper exists. Distinct from the two injections above —
    allocation and transfer — and labelled as such.

    A ``KeyboardInterrupt`` rather than a ``RuntimeError``, so what is
    proved is an *unconditional* cleanup: the constructor's protection is a
    bare ``except BaseException``, not an ``except Exception`` that a
    non-``Exception`` unwind would slip past."""
    with live_storage_baseline():
        with retained_publication_failure(
            KeyboardInterrupt("injected wrapper failure")
        ) as retained:
            with pytest.raises(KeyboardInterrupt, match="injected wrapper"):
                NativeTensor.from_int64_array(probe_array())
            # The injection really fired, at an int64 core, and that core was
            # closed by ``from_int64_array`` — not by ``__del__``, which the
            # list's strong reference has kept from running.
            assert [core.dtype for core in retained] == ["int64"], retained
            _assert_core_was_explicitly_closed(retained)
    # The control: real construction still works once the injection is gone.
    tensor = NativeTensor.from_int64_array(probe_array())
    try:
        assert tensor.dtype == "int64"
        assert tensor.tolist() == list(PROBE_VALUES)
    finally:
        tensor.close()


@needs_native
def test_a_failed_construction_leaves_the_observable_world_unchanged():
    with unchanged_world():
        original = cpp.NativeStorage.copy_from

        def failing(self, values):
            raise RuntimeError("injected")

        with live_storage_baseline():
            cpp.NativeStorage.copy_from = failing
            try:
                with pytest.raises(RuntimeError):
                    NativeTensor.from_int64_array(probe_array())
            finally:
                cpp.NativeStorage.copy_from = original


@needs_native
def test_a_failing_contiguous_copy_publication_closes_its_destination():
    """**Publication position 2 of 3**: ``contiguous_copy`` has already
    allocated its owning destination core, and the wrapper that would take
    ownership of it fails.

    The injection is at ``NativeTensor.__init__`` rather than at the
    allocator on purpose. An allocator that closed the core itself would
    prove only that the *injection* cleans up after itself; here the core
    is allocated for real, retained externally, and never touched by the
    instrument — so the close the assertion observes can only be
    ``contiguous_copy``'s own."""
    tensor = NativeTensor.from_int64_array(
        np.arange(6, dtype=np.int64).reshape(2, 3))
    try:
        with live_storage_baseline():
            with retained_publication_failure(
                RuntimeError("injected destination failure")
            ) as retained:
                with pytest.raises(RuntimeError,
                                   match="injected destination"):
                    tensor.contiguous_copy()
                assert [core.dtype for core in retained] == ["int64"], retained
                _assert_core_was_explicitly_closed(retained)
        # The control: the real copy still works afterwards, and the source
        # the failed copy read is untouched.
        copy = tensor.contiguous_copy()
        try:
            assert copy.tolist() == [[0, 1, 2], [3, 4, 5]]
        finally:
            copy.close()
        assert tensor.tolist() == [[0, 1, 2], [3, 4, 5]]
    finally:
        tensor.close()


@needs_native
def test_a_failing_detach_publication_closes_the_copy_it_allocated():
    """**Publication position 3 of 3**: ``detach`` has already allocated the
    owning copy that makes its result independent, and the wrapper that
    would take ownership of it fails.

    A separate position from ``contiguous_copy``'s, and asserted as one:
    the two methods reach the same Core copy through different call sites,
    and a fix applied to only one of them leaves the other leaking."""
    tensor = NativeTensor.from_int64_array(probe_array())
    try:
        with live_storage_baseline():
            with retained_publication_failure(
                RuntimeError("injected detach failure")
            ) as retained:
                with pytest.raises(RuntimeError, match="injected detach"):
                    tensor.detach()
                assert [core.dtype for core in retained] == ["int64"], retained
                _assert_core_was_explicitly_closed(retained)
        # The control: a real detach still works afterwards and still owns
        # storage independent of the source's.
        detached = tensor.detach()
        try:
            assert detached.dtype == "int64"
            assert detached.owns_core is True
            assert detached.requires_grad is False
            assert detached.tolist() == list(PROBE_VALUES)
            assert detached._core.storage is not tensor._core.storage
        finally:
            detached.close()
    finally:
        tensor.close()


# ===========================================================================
# 8. Lifecycle and absence
# ===========================================================================

def _source_exports():
    names = set()
    for path in sorted((REPO_ROOT / "cpp" / "src").glob("*.cpp")):
        names.update(re.findall(r"TF_EXPORT[^;{]*?\b(tf_[a-z0-9_]+)\s*\(",
                                path.read_text(encoding="utf-8"), re.S))
    return names


def test_the_c_abi_export_inventory_did_not_move_at_k2():
    exports = _source_exports()
    assert len(exports) == 54, sorted(exports)
    for absent in ("tf_core_argmax", "tf_core_index_select",
                   "tf_core_gather", "tf_storage_dtype"):
        assert absent not in exports, absent


def test_the_export_scanner_can_actually_fail():
    """Negative control, on a temporary string: an export really is found,
    so "54" is a measurement rather than an artifact."""
    source = 'TF_EXPORT void tf_core_probe(const void* a) { use(a); }'
    assert re.findall(r"TF_EXPORT[^;{]*?\b(tf_[a-z0-9_]+)\s*\(", source,
                      re.S) == ["tf_core_probe"]
    assert re.findall(r"TF_EXPORT[^;{]*?\b(tf_[a-z0-9_]+)\s*\(",
                      "void tf_core_probe(void);", re.S) == []


def test_the_ctest_example_and_benchmark_inventories_did_not_move_at_k2():
    cmake = (REPO_ROOT / "cpp" / "CMakeLists.txt").read_text(encoding="utf-8")
    assert len(re.findall(r"^\s*add_test\(", cmake, re.M)) == 25
    assert len(list((REPO_ROOT / "cpp" / "tests").glob("*.cpp"))) == 25
    assert len(list((REPO_ROOT / "examples").glob("*.py"))) == 16
    assert len(list((REPO_ROOT / "benchmarks").glob("*.py"))) == 9


def test_the_experimental_export_list_is_still_twenty_five():
    import tensorforge.experimental as experimental

    assert len(experimental.__all__) == 25
    assert len(set(experimental.__all__)) == 25
    for name in experimental.__all__:
        assert hasattr(experimental, name), name
    for absent in ("NativeIntTensor", "NativeIndexTensor", "INDEX_DTYPES",
                   "from_int64_array", "native_argmax"):
        assert absent not in experimental.__all__, absent


def test_no_k3_or_k4_operation_exists_on_any_surface():
    for owner in (NativeTensor, cpp.NativeTensorCore, cpp.NativeStorage):
        for absent in ("argmax", "argmin", "index_select", "gather",
                       "scatter", "take", "nonzero", "sort", "argsort",
                       "topk", "unique", "where", "bincount", "cumsum"):
            assert not hasattr(owner, absent), (owner.__name__, absent)
    for inventory in (cpp.TENSOR_CORE_OPS, cpp.AUTOGRAD_OPS,
                      cpp.RAW_KERNELS, cpp.TENSOR_CORE_KERNELS):
        for banned in ("argmax", "index_select", "gather", "int64",
                       "integer"):
            assert not [n for n in inventory if banned in n.lower()], banned


def test_no_casting_or_promotion_surface_appeared():
    for owner in (NativeTensor, cpp.NativeTensorCore, cpp.NativeStorage):
        for absent in ("astype", "cast", "to", "type", "long", "int",
                       "float", "double", "promote", "as_type", "cpu",
                       "cuda", "is_integer", "is_floating"):
            assert not hasattr(owner, absent), (owner.__name__, absent)


def test_no_integer_dtype_argument_appeared_on_any_state_owning_class():
    """A class owning no dtype-bearing state must not gain one, and one
    that does must still accept only the floating pair."""
    from tensorforge.experimental import _native_dtype

    assert _native_dtype.MODULE_DTYPES == FLOATING_DTYPES
    for dtype in FLOATING_DTYPES:
        assert _native_dtype.normalize_module_dtype(dtype) == dtype
    with pytest.raises(ValueError, match="int64"):
        _native_dtype.normalize_module_dtype("int64")
    assert _native_dtype.normalize_module_dtype(None) == "float64"


def test_the_stable_line_is_untouched_and_knows_nothing_about_int64():
    import subprocess
    import sys

    program = (
        "import sys, tensorforge\n"
        "assert 'tensorforge.backends.cpp' not in sys.modules\n"
        "assert 'tensorforge.experimental' not in sys.modules\n"
        "from tensorforge import Tensor, nn, optim, data\n"
        "assert not hasattr(Tensor, 'from_int64_array')\n"
        "print(Tensor.__name__)\n"
    )
    result = subprocess.run([sys.executable, "-c", program],
                            capture_output=True, text=True,
                            cwd=str(REPO_ROOT))
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "Tensor"


def test_the_provenance_scanner_is_still_active_over_the_new_module():
    """K2 added a file to the repository, so the external-reference sweep
    must still cover it. Delegated to the Phase-K module that owns the
    scanner rather than reimplemented, so there is one implementation."""
    contract = pytest.importorskip("test_native_phase_k")
    scanned = {path.relative_to(REPO_ROOT).as_posix()
               for path, _ in contract._repository_text_files()}
    assert "tests/test_native_int64_tensor.py" in scanned
    own_text = (REPO_ROOT / "tests"
                / "test_native_int64_tensor.py").read_text(encoding="utf-8")
    assert contract._provenance_hits(own_text) == []
    # Non-vacuity: the scanner really can fire, on a decoded control string
    # so this module contains none of the text it scans for.
    control = "".join(chr(ord(c) - 1)
                      for c in 'uijt!lfsofm!xbt!qpsufe!gspn!uif!vqtusfbn!'
                               'sfqptjupsz')
    assert contract._provenance_hits(control)


def test_no_status_surface_over_claims_about_the_integer_tensor():
    """Delegated to the Phase-K over-claim scanner, so K2's own new file
    and the surfaces it edited are held to the same rule as every other."""
    contract = pytest.importorskip("test_native_phase_k")
    for surface in contract.STATUS_SURFACES + (
        "src/tensorforge/experimental/__init__.py",
        "src/tensorforge/experimental/native_tensor.py",
        "src/tensorforge/backends/cpp.py",
    ):
        text = (REPO_ROOT / surface).read_text(encoding="utf-8")
        assert contract._overclaims(text) == [], (surface,
                                                  contract._overclaims(text)[:3])
