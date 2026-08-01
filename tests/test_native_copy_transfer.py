"""Phase H, milestone H5 — native copy and mutation-transfer semantics.

H5 replaced the native line's value-transfer primitive. ``_native_copy``
used to be ``zeros(shape) + core`` — two allocations, a full zero-fill
pass, and a full elementwise-addition pass — and is now the E3.1 native
identity gather, ``NativeTensorCore.contiguous_copy()``: one
uninitialized allocation (H1) and one pass. Underneath it,
``tf_core_contiguous_copy`` gained a second *traversal* (not a second
kernel and not a second export): a row-major source is swept with the
flat pointer loop every other unary op's contiguous path already uses,
and everything else keeps the generic odometer.

This suite proves the parts of that which are semantic rather than
numerical-performance:

1.  **The value contract.** A copy copies. The pre-H5 addition
    normalized ``-0.0`` to ``+0.0`` and quieted signaling NaNs; the
    gather preserves both, which is what makes ``copy_value_`` finally
    agree with ``NativeParameter(...)`` construction, ``detach()``, and
    the ``to_numpy()``/``from_array`` serialization boundary — three
    value-copy paths that always used the gather. The pre-H5
    composition is retained here, executed natively, as the reference
    that pins exactly which patterns moved and which did not.
2.  **Layout totality.** Contiguous, transposed, narrowed, chained,
    offset, reversed, rank-0 and one-element sources all copy, and the
    two traversals agree.
3.  **Aliasing and overlap.** Every source/destination relationship the
    runtime can construct, including self-copy and genuinely
    overlapping views.
4.  **Identity, storage, and version behavior** at every mutation and
    state-transfer callsite.
5.  **Failure atomicity** at each stage of each callsite.
6.  **H1 compatibility** — the gather's destination is uninitialized, so
    it must write every element, proved by poison injected purely by
    test infrastructure around the allocator.
7.  **The surface guardrails** — no new export, no copy selector, no
    profiling control, no public helper.

No timing is asserted anywhere in this file.
"""

import gc
import importlib
import math
import struct
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from tensorforge.backends import cpp
from tensorforge.experimental import (
    NativeAdam,
    NativeBatchNorm1d,
    NativeLinear,
    NativeModule,
    NativeParameter,
    NativeSGD,
    NativeTensor,
    load_native_checkpoint,
    save_native_checkpoint,
)
from tensorforge.experimental import native_tensor as native_tensor_module
from tensorforge.experimental.native_tensor import _native_copy

Core = cpp.NativeTensorCore
REPO_ROOT = Path(__file__).resolve().parents[1]

needs_native = pytest.mark.skipif(
    not cpp.is_available(),
    reason="experimental C++ backend not built; " + cpp.build_instructions(),
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _bits(values):
    """Raw IEEE-754 bit patterns. The whole subject of this file is which
    bits survive a copy, and a value comparison cannot see the answer:
    ``-0.0 == +0.0`` is True and ``nan == nan`` is False."""
    return np.ascontiguousarray(values, dtype=np.float64).view(np.uint64)


def _same_bits(left, right):
    left = np.ascontiguousarray(left, dtype=np.float64)
    right = np.ascontiguousarray(right, dtype=np.float64)
    return (left.shape == right.shape
            and np.array_equal(left.view(np.uint64), right.view(np.uint64)))


def _from_bit_patterns(patterns):
    return np.array(list(patterns), dtype=np.uint64).view(np.float64)


# The IEEE-754 sweep, mirrored bit for bit in cpp/tests/test_contiguous_copy.cpp
# so the two layers cannot drift. The three marked patterns are exactly the
# ones the pre-H5 arithmetic copy changed.
PATTERNS = {
    "+0.0": 0x0000000000000000,
    "-0.0": 0x8000000000000000,            # <- addition normalized this
    "1.0": 0x3FF0000000000000,
    "-1.0": 0xBFF0000000000000,
    "+inf": 0x7FF0000000000000,
    "-inf": 0xFFF0000000000000,
    "qNaN payload 1": 0x7FF8000000000001,
    "qNaN payload A": 0x7FF800000000000A,
    "qNaN payload DEADBEEF": 0x7FF8DEADBEEFCAFE,
    "-qNaN payload 1": 0xFFF8000000000001,
    "sNaN payload 1": 0x7FF0000000000001,  # <- addition quieted this
    "-sNaN payload 1": 0xFFF0000000000001,  # <- addition quieted this
    "smallest subnormal": 0x0000000000000001,
    "-smallest subnormal": 0x8000000000000001,
    "largest subnormal": 0x000FFFFFFFFFFFFF,
    "smallest normal": 0x0010000000000000,
    "largest finite": 0x7FEFFFFFFFFFFFFF,
    "-largest finite": 0xFFEFFFFFFFFFFFFF,
}
PATTERN_NAMES = list(PATTERNS)
SWEEP = _from_bit_patterns(PATTERNS.values())

# The patterns an arithmetic copy would have altered, and what it made them.
NORMALIZED_BY_ADDITION = {
    "-0.0": 0x0000000000000000,
    "sNaN payload 1": 0x7FF8000000000001,
    "-sNaN payload 1": 0xFFF8000000000001,
}


def _pre_h5_native_copy(core):
    """The **retained pre-H5 composition**, executed natively: allocate a
    zero-filled destination of the source's shape and add the source into
    it. This is the literal body ``_native_copy`` had before H5, kept
    here so every H5 claim is stated against the real previous behavior
    rather than against a description of it."""
    zeros = Core.zeros(core.shape, dtype=core.dtype, device=core.device)
    try:
        return zeros.add(core)
    finally:
        zeros.close()


def _close_module(module):
    """Release every native tensor a module owns. ``NativeModule`` has no
    ``close()`` — lifetime lives with the tensors — so the live-storage
    tests release them explicitly rather than relying on the collector."""
    for _name, parameter in module.named_parameters():
        parameter.close()
    for _name, buffer in module.named_buffers():
        buffer.close()


class _Boom(BaseException):
    """A non-``Exception`` BaseException, so cleanup paths are proved to
    run for something ``except Exception`` would never catch."""


FAILURE_CLASSES = (RuntimeError, MemoryError, KeyboardInterrupt, _Boom)


@pytest.fixture
def live_storages(monkeypatch):
    """The ids of every NativeStorage currently open — the project's
    deterministic native-allocation instrumentation. Exact, because it
    hooks ``close()`` rather than relying on garbage collection."""
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


# ===========================================================================
# 1. The value contract: a copy copies
# ===========================================================================


@needs_native
def test_the_copy_preserves_every_ieee754_bit_pattern():
    """The headline H5 property, over the whole sweep at once."""
    source = Core.from_array(SWEEP)
    try:
        copy = _native_copy(source)
        try:
            assert _same_bits(copy.to_numpy(), SWEEP)
        finally:
            copy.close()
    finally:
        source.close()


@needs_native
@pytest.mark.parametrize("name", PATTERN_NAMES)
def test_each_pattern_individually_and_against_the_pre_h5_composition(name):
    """Pattern by pattern, so a failure names the value. Each is checked
    twice: against the source (H5's contract) and against the retained
    pre-H5 composition (which of the two behaviors changed, and how)."""
    value = _from_bit_patterns([PATTERNS[name]])
    source = Core.from_array(value)
    try:
        new = _native_copy(source)
        old = _pre_h5_native_copy(source)
        try:
            assert _same_bits(new.to_numpy(), value), name
            expected_old = _from_bit_patterns(
                [NORMALIZED_BY_ADDITION.get(name, PATTERNS[name])]
            )
            assert _same_bits(old.to_numpy(), expected_old), name
            if name in NORMALIZED_BY_ADDITION:
                assert not _same_bits(new.to_numpy(), old.to_numpy()), name
            else:
                assert _same_bits(new.to_numpy(), old.to_numpy()), name
        finally:
            new.close()
            old.close()
    finally:
        source.close()


@needs_native
def test_exactly_three_of_the_eighteen_patterns_changed():
    """The scope of the semantic change, stated as a closed set rather
    than left implicit: negative zero and both signs of signaling NaN.
    Everything else — every NaN payload included — the addition already
    preserved, so H5 moved nothing there."""
    source = Core.from_array(SWEEP)
    try:
        new = _native_copy(source)
        old = _pre_h5_native_copy(source)
        try:
            differing = {
                PATTERN_NAMES[i]
                for i, (a, b) in enumerate(
                    zip(_bits(new.to_numpy()), _bits(old.to_numpy()))
                )
                if a != b
            }
        finally:
            new.close()
            old.close()
    finally:
        source.close()
    assert differing == set(NORMALIZED_BY_ADDITION)


@needs_native
def test_nan_payloads_and_signs_survive_including_manufactured_ones():
    """NaNs are copied, not computed: payload and sign both survive, and
    a NaN that arrived through arithmetic is copied like any other."""
    manufactured = np.array([math.inf - math.inf, 0.0 * math.inf,
                             math.inf / math.inf], dtype=np.float64)
    assert np.isnan(manufactured).all()
    payloads = _from_bit_patterns([
        0x7FF8000000000001, 0x7FF8000000000002, 0x7FF8FFFFFFFFFFFF,
        0xFFF80000DEADBEEF, 0x7FF0000000000001, 0xFFF0000000000001,
    ])
    values = np.concatenate([manufactured, payloads])
    source = Core.from_array(values)
    try:
        copy = _native_copy(source)
        try:
            assert _same_bits(copy.to_numpy(), values)
        finally:
            copy.close()
    finally:
        source.close()


@needs_native
def test_the_four_value_copy_paths_now_agree_with_each_other():
    """The coherence H5 restores. All four document 'an independent
    owning contiguous copy of the source's current value'; before H5
    ``copy_value_`` was the one that did not deliver it, because it alone
    went through the arithmetic composition."""
    source = NativeTensor.from_array(SWEEP)
    constructed = NativeParameter(source)
    detached = source.detach()
    mutated = NativeParameter(np.zeros_like(SWEEP))
    mutated.copy_value_(source)
    round_tripped = NativeTensor.from_array(source.to_numpy())
    try:
        for label, tensor in (("construction", constructed),
                              ("detach", detached),
                              ("copy_value_", mutated),
                              ("to_numpy/from_array", round_tripped)):
            assert _same_bits(tensor.to_numpy(), SWEEP), label
    finally:
        for tensor in (source, constructed, detached, mutated,
                       round_tripped):
            tensor.close()


# ===========================================================================
# 2. Layout totality — every source layout the runtime can build
# ===========================================================================


def _layout_cases(base):
    """Every layout family a ``_native_copy`` caller can hand over, built
    from one (6, 4) source. Each entry is (label, view, expected NumPy)."""
    reference = base.to_numpy()
    return [
        ("contiguous", base, reference),
        ("transposed", base.transpose(), reference.T),
        ("narrow axis 0 (stays contiguous)", base.narrow(0, 2, 3),
         reference[2:5]),
        ("narrow axis 1 (strided)", base.narrow(1, 1, 2),
         reference[:, 1:3]),
        ("chained narrow then transpose",
         base.narrow(0, 1, 4).transpose(), reference[1:5].T),
        ("chained transpose then narrow",
         base.transpose().narrow(0, 1, 2), reference.T[1:3]),
        ("reshaped", base.reshape((base.numel,)),
         reference.reshape(base.numel)),
    ]


@needs_native
def test_every_layout_copies_to_the_right_values():
    base = Core.from_array(np.arange(24, dtype=np.float64).reshape(6, 4))
    views = []
    try:
        for label, view, expected in _layout_cases(base):
            views.append(view)
            copy = _native_copy(view)
            try:
                assert copy.shape == expected.shape, label
                assert copy.contiguous, label
                assert _same_bits(copy.to_numpy(), expected), label
            finally:
                copy.close()
    finally:
        for view in views:
            if view is not base:
                view.close()
        base.close()


@needs_native
def test_every_layout_carries_the_special_patterns_through_unchanged():
    """The layout sweep and the value sweep crossed: a transposed or
    narrowed view of a buffer full of signed zeros and signaling NaNs
    must copy them exactly, which is what forces the odometer traversal
    to be as faithful as the flat one."""
    padded = np.zeros((len(SWEEP), 3), dtype=np.float64)
    padded[:, 1] = SWEEP
    base = Core.from_array(padded)
    views = []
    try:
        for label, view, expected in _layout_cases(base):
            views.append(view)
            copy = _native_copy(view)
            try:
                assert _same_bits(copy.to_numpy(), expected), label
            finally:
                copy.close()
    finally:
        for view in views:
            if view is not base:
                view.close()
        base.close()


@needs_native
@pytest.mark.parametrize("shape", [(), (1,), (1, 1), (1, 1, 1), (3,),
                                   (2, 3), (2, 3, 4), (2, 3, 4, 5)])
def test_scalars_one_element_and_higher_rank_all_copy(shape):
    numel = int(np.prod(shape)) if shape else 1
    values = np.arange(numel, dtype=np.float64).reshape(shape) + 0.25
    source = Core.from_array(values)
    try:
        copy = _native_copy(source)
        try:
            assert copy.shape == source.shape
            assert copy.ndim == source.ndim
            assert _same_bits(copy.to_numpy(), values)
        finally:
            copy.close()
    finally:
        source.close()


@needs_native
def test_a_rank_zero_special_value_copies():
    """A scalar view is the one layout with no shape or stride array at
    all, so it is worth pinning separately — and with a pattern that
    would move under arithmetic."""
    for pattern in (0x8000000000000000, 0x7FF0000000000001):
        value = _from_bit_patterns([pattern])
        flat = Core.from_array(value)
        scalar = flat.reshape(())
        try:
            copy = _native_copy(scalar)
            try:
                assert copy.shape == ()
                assert _bits(copy.to_numpy())[0] == pattern
            finally:
                copy.close()
        finally:
            scalar.close()
            flat.close()


@needs_native
def test_a_large_copy_is_exact():
    values = np.linspace(-1e18, 1e18, 1 << 16, dtype=np.float64)
    source = Core.from_array(values.reshape(256, 256))
    try:
        copy = _native_copy(source)
        try:
            assert _same_bits(copy.to_numpy(), values.reshape(256, 256))
        finally:
            copy.close()
    finally:
        source.close()


@needs_native
def test_an_offset_view_reads_from_the_right_place():
    """A narrow along axis 0 keeps row-major strides and only moves the
    offset, so it is the case where the flat traversal must honor the
    offset rather than start at element zero."""
    values = np.arange(40, dtype=np.float64).reshape(10, 4)
    base = Core.from_array(values)
    for start in range(10):
        view = base.narrow(0, start, 10 - start)
        assert view.contiguous, start
        assert view.offset == start * 4
        copy = _native_copy(view)
        try:
            assert _same_bits(copy.to_numpy(), values[start:])
        finally:
            copy.close()
            view.close()
    base.close()


# ===========================================================================
# 3. Aliasing and overlap
# ===========================================================================


@needs_native
def test_the_result_shares_no_storage_with_the_source():
    base = Core.from_array(np.arange(12, dtype=np.float64).reshape(3, 4))
    try:
        copy = _native_copy(base)
        try:
            assert copy.storage is not base.storage
            assert copy.offset == 0
            assert copy.contiguous
            # Mutating the copy's storage cannot reach the source.
            copy.storage.fill(-7.0)
            assert _same_bits(base.to_numpy(),
                              np.arange(12, dtype=np.float64).reshape(3, 4))
        finally:
            copy.close()
    finally:
        base.close()


@needs_native
def test_closing_the_source_leaves_the_copy_usable_and_vice_versa():
    base = Core.from_array(np.arange(6, dtype=np.float64))
    copy = _native_copy(base)
    base.close()
    assert _same_bits(copy.to_numpy(), np.arange(6, dtype=np.float64))
    copy.close()

    other = Core.from_array(np.arange(6, dtype=np.float64))
    second = _native_copy(other)
    second.close()
    assert _same_bits(other.to_numpy(), np.arange(6, dtype=np.float64))
    other.close()


@needs_native
def test_a_closed_source_is_rejected_and_nothing_is_allocated(live_storages):
    base = Core.from_array(np.arange(6, dtype=np.float64))
    base.close()
    gc.collect()
    baseline = len(live_storages)
    with pytest.raises(RuntimeError, match="closed"):
        _native_copy(base)
    gc.collect()
    assert len(live_storages) == baseline


@needs_native
def test_copy_value_from_the_parameter_itself_is_a_faithful_no_op():
    """Self-copy: the source *is* the destination. Staging reads the
    current core before anything is swapped, so this is well defined —
    and after H5 it is bit-preserving, which is what makes it a true
    no-op on values that an addition would have changed."""
    parameter = NativeParameter(SWEEP)
    try:
        before = parameter.to_numpy()
        parameter.copy_value_(parameter)
        assert _same_bits(parameter.to_numpy(), before)
        assert _same_bits(parameter.to_numpy(), SWEEP)
        assert parameter.version == 1
    finally:
        parameter.close()


@needs_native
def test_copy_value_from_a_view_of_the_parameters_own_storage():
    """Exact overlap through a *different object*: the source is a view
    over the destination's own storage. Staging materializes into
    independent storage before the swap, so the read finishes before the
    write begins and the transfer is well defined."""
    values = np.arange(12, dtype=np.float64).reshape(3, 4)
    parameter = NativeParameter(values)
    alias = NativeTensor._from_core(
        parameter._require_open().narrow(0, 0, 3), owns_core=False
    )
    try:
        parameter.copy_value_(alias)
        assert _same_bits(parameter.to_numpy(), values)
        assert parameter.version == 1
    finally:
        alias.close()
        parameter.close()


@needs_native
def test_copy_value_from_a_partially_overlapping_transposed_self_view():
    """Partial overlap, the hardest arrangement the runtime can express:
    a square parameter's own transpose. Every destination element is read
    from a *different* source element, so an in-place transfer would
    corrupt it — proving the staged copy really is staged."""
    values = np.arange(16, dtype=np.float64).reshape(4, 4)
    parameter = NativeParameter(values)
    transposed = NativeTensor._from_core(
        parameter._require_open().transpose(), owns_core=False
    )
    try:
        parameter.copy_value_(transposed)
        assert _same_bits(parameter.to_numpy(), values.T)
        assert parameter.version == 1
    finally:
        transposed.close()
        parameter.close()


@needs_native
def test_sibling_views_of_one_storage_copy_independently():
    base = Core.from_array(np.arange(24, dtype=np.float64).reshape(4, 6))
    left = base.narrow(1, 0, 3)
    right = base.narrow(1, 3, 3)
    try:
        left_copy = _native_copy(left)
        right_copy = _native_copy(right)
        try:
            reference = base.to_numpy()
            assert _same_bits(left_copy.to_numpy(), reference[:, 0:3])
            assert _same_bits(right_copy.to_numpy(), reference[:, 3:6])
            assert left_copy.storage is not right_copy.storage
            assert left_copy.storage is not base.storage
        finally:
            left_copy.close()
            right_copy.close()
    finally:
        left.close()
        right.close()
        base.close()


@needs_native
def test_repeated_copies_are_all_independent_and_all_exact():
    source = Core.from_array(SWEEP)
    copies = []
    try:
        for _ in range(8):
            copies.append(_native_copy(source))
        storages = {id(copy.storage) for copy in copies}
        assert len(storages) == len(copies)
        for copy in copies:
            assert _same_bits(copy.to_numpy(), SWEEP)
    finally:
        for copy in copies:
            copy.close()
        source.close()


@needs_native
def test_a_shared_parameter_updated_through_two_optimizers_stays_coherent():
    """Duplicate parameters and multiple optimizers: the value contract
    does not change because two owners are involved, and each commit is
    still exactly one version increment."""
    parameter = NativeParameter(np.ones((3, 3)))
    parameter._grad = NativeTensor.from_array(np.full((3, 3), 0.5))
    first = NativeSGD([parameter, parameter], lr=0.1)
    try:
        first.step()
        # The deduplicated optimizer sees one parameter, so one update.
        assert parameter.version == 1
        assert _same_bits(parameter.to_numpy(), np.full((3, 3), 0.95))
    finally:
        pass
    second = NativeSGD([parameter], lr=0.1)
    try:
        second.step()
        assert parameter.version == 2
    finally:
        pass
        parameter.close()


# ===========================================================================
# 4. Source immutability and destination ownership
# ===========================================================================


@needs_native
def test_the_source_is_never_mutated_on_any_layout():
    """Every layout family, value and metadata alike: gathering out of a
    view must leave the view — and the storage under it — exactly as it
    was. The sweep values make this sharp, since a path that wrote back
    a normalized zero would show up immediately."""
    base = Core.from_array(SWEEP.reshape(6, 3))
    reference = base.to_numpy()
    views = []
    try:
        for label, view, _expected in _layout_cases(base):
            views.append(view)
            before = view.to_numpy()
            before_offset = view.offset
            before_strides = view.strides
            before_shape = view.shape
            _native_copy(view).close()
            assert _same_bits(view.to_numpy(), before), label
            assert view.offset == before_offset, label
            assert view.strides == before_strides, label
            assert view.shape == before_shape, label
            # ...and the shared storage under every view is untouched.
            assert _same_bits(base.to_numpy(), reference), label
    finally:
        for view in views:
            if view is not base:
                view.close()
        base.close()


@needs_native
def test_the_result_owns_contiguous_storage_at_offset_zero():
    base = Core.from_array(np.arange(24, dtype=np.float64).reshape(6, 4))
    views = []
    try:
        for label, view, expected in _layout_cases(base):
            views.append(view)
            copy = _native_copy(view)
            try:
                assert copy.contiguous, label
                assert copy.offset == 0, label
                assert copy.storage.size == int(np.prod(expected.shape)), label
            finally:
                copy.close()
    finally:
        for view in views:
            if view is not base:
                view.close()
        base.close()


# ===========================================================================
# 5. H1 — the destination is uninitialized, so it must be written in full
# ===========================================================================


@pytest.fixture
def poisoned_allocator(monkeypatch):
    """Deterministic poison, injected **exclusively by test
    infrastructure, around the allocator** — exactly H1's technique. The
    real constructor allocates; this fills the returned storage through
    the ordinary ``tf_storage_fill`` primitive and hands that same
    storage to the real operation. No poison control exists in the
    shipped runtime.

    Yields a setter taking the fill value."""
    real = cpp.NativeStorage._uninitialized.__func__
    state = {"value": None}

    def poisoned(cls, size, dtype=None, device="cpu"):
        storage = real(cls, size, dtype=dtype, device=device)
        if state["value"] is not None:
            storage.fill(state["value"])
        return storage

    monkeypatch.setattr(cpp.NativeStorage, "_uninitialized",
                        classmethod(poisoned))
    return state


QUIET_NAN_POISON = float(_from_bit_patterns([0x7FF8BADC0FFEE000])[0])
FINITE_POISON = -1.2345678901234e300


@needs_native
@pytest.mark.parametrize("poison", [QUIET_NAN_POISON, FINITE_POISON])
def test_the_gather_writes_every_destination_element(poisoned_allocator,
                                                     poison):
    """If any destination element were left unwritten, the poison would
    survive into the result. Run over every layout family, because the
    two traversals cover different ground."""
    poisoned_allocator["value"] = poison
    base = Core.from_array(np.arange(24, dtype=np.float64).reshape(6, 4))
    views = []
    try:
        for label, view, expected in _layout_cases(base):
            views.append(view)
            copy = _native_copy(view)
            try:
                assert _same_bits(copy.to_numpy(), expected), label
            finally:
                copy.close()
    finally:
        for view in views:
            if view is not base:
                view.close()
        base.close()


@needs_native
@pytest.mark.parametrize("poison", [QUIET_NAN_POISON, FINITE_POISON])
def test_the_poison_detector_can_actually_fail(poisoned_allocator, poison):
    """The negative control: with the poison armed, a destination the
    runtime deliberately leaves partly unwritten really does show it. If
    this passed unconditionally, the test above would prove nothing.

    ``narrow_backward`` is the runtime's documented partial writer, and
    H1 explicitly *rejected* it — it keeps a zero-initialized
    destination. Reaching it through the uninitialized allocator instead
    is what makes the poison visible."""
    poisoned_allocator["value"] = poison
    values = np.arange(6, dtype=np.float64).reshape(2, 3)
    gradient = Core.from_array(values)
    try:
        out = Core._uninitialized((4, 3), dtype=gradient.dtype,
                                  device=gradient.device)
        try:
            # Only the narrowed region is written by design. The layout
            # metadata crosses through H7's trusted binding, exactly as the
            # production scatter does.
            shape_ptr, strides_ptr = gradient._layout_pointers()
            full = cpp.row_major_strides((4, 3))
            gradient._storage._lib.tf_core_narrow_backward(
                gradient._storage._require_open(),
                out._storage._require_open(),
                shape_ptr, strides_ptr,
                cpp._layout_vector(full),
                gradient.offset, 1 * full[0], gradient.ndim,
            )
            written = out.to_numpy()
        finally:
            out.close()
    finally:
        gradient.close()
    untouched = written[0]
    if math.isnan(poison):
        assert np.isnan(untouched).all()
    else:
        assert _same_bits(untouched, np.full(3, poison))


@needs_native
@pytest.mark.parametrize("poison", [QUIET_NAN_POISON, FINITE_POISON])
def test_poison_never_reaches_a_committed_parameter_or_state(
    poisoned_allocator, poison
):
    """The same proof at the callsites that matter: with every
    uninitialized allocation poisoned, a full optimizer step, a
    state_dict round trip, and a BatchNorm running-statistics update all
    produce exactly the values the un-poisoned runtime produces."""
    def _run():
        model = NativeLinear(4, 3, seed=7)
        parameters = list(model.parameters())
        for parameter in parameters:
            parameter._grad = NativeTensor.from_array(
                np.full(parameter.shape, 0.25)
            )
        optimizer = NativeAdam(parameters, lr=0.05)
        optimizer.step()
        snapshot = model.state_dict()
        values = {k: v.to_numpy() for k, v in snapshot.items()}
        model.load_state_dict(snapshot)
        after = {k: v.to_numpy() for k, v in model.state_dict().items()}
        for tensor in snapshot.values():
            tensor.close()
        norm = NativeBatchNorm1d(3)
        norm.train()
        out = norm(NativeTensor.from_array(
            np.arange(12, dtype=np.float64).reshape(4, 3)
        ))
        running = (norm.running_mean.to_numpy(), norm.running_var.to_numpy())
        result = (values, after, out.to_numpy(), running)
        out.close()
        optimizer.close()
        return result

    poisoned_allocator["value"] = None
    clean = _run()
    poisoned_allocator["value"] = poison
    poisoned = _run()

    for clean_map, poisoned_map in zip(clean[:2], poisoned[:2]):
        assert set(clean_map) == set(poisoned_map)
        for key in clean_map:
            assert _same_bits(clean_map[key], poisoned_map[key]), key
    assert _same_bits(clean[2], poisoned[2])
    for a, b in zip(clean[3], poisoned[3]):
        assert _same_bits(a, b)


# ===========================================================================
# 6. Parameter mutation: identity, storage, version
# ===========================================================================


@needs_native
def test_copy_value_replaces_storage_and_preserves_identity():
    """H5 changed *how* the staged value is produced, not what the
    mutation does with it: ``copy_value_`` still stages an independent
    core and **replaces** the parameter's storage, keeping the Python
    object, its gradient, its ``requires_grad``, and its registrations."""
    parameter = NativeParameter(np.ones((2, 3)))
    gradient = NativeTensor.from_array(np.full((2, 3), 0.5))
    parameter._grad = gradient
    source = NativeTensor.from_array(np.full((2, 3), 4.0))
    identity = id(parameter)
    old_storage = id(parameter._require_open().storage)
    try:
        result = parameter.copy_value_(source)
        assert result is parameter
        assert id(parameter) == identity
        assert parameter.grad is gradient
        assert _same_bits(parameter.grad.to_numpy(), np.full((2, 3), 0.5))
        assert parameter.requires_grad is True
        assert parameter.version == 1
        assert id(parameter._require_open().storage) != old_storage
        assert _same_bits(parameter.to_numpy(), np.full((2, 3), 4.0))
        # ...and the source is untouched by identity and by value.
        assert _same_bits(source.to_numpy(), np.full((2, 3), 4.0))
    finally:
        parameter.close()
        source.close()


@needs_native
def test_copy_value_increments_the_version_exactly_once_per_call():
    parameter = NativeParameter(np.zeros(4))
    source = NativeTensor.from_array(np.arange(4, dtype=np.float64))
    try:
        for expected in range(1, 6):
            parameter.copy_value_(source)
            assert parameter.version == expected
    finally:
        parameter.close()
        source.close()


@needs_native
@pytest.mark.parametrize("error", FAILURE_CLASSES)
def test_a_staging_failure_in_copy_value_changes_nothing(error,
                                                         live_storages):
    """Failure before the swap: value, core, storage identity, gradient,
    and version are all exactly as they were, and no storage leaks."""
    parameter = NativeParameter(np.arange(6, dtype=np.float64))
    parameter._grad = NativeTensor.from_array(np.ones(6))
    source = NativeTensor.from_array(np.full(6, 9.0))
    gc.collect()
    baseline = len(live_storages)
    before_value = parameter.to_numpy()
    before_core = parameter._require_open()
    before_storage = id(before_core.storage)
    before_grad = parameter.grad

    real = Core.contiguous_copy

    def failing(self):
        raise error("injected copy failure")

    cpp.NativeTensorCore.contiguous_copy = failing
    try:
        with pytest.raises(error, match="injected copy failure"):
            parameter.copy_value_(source)
    finally:
        cpp.NativeTensorCore.contiguous_copy = real

    assert parameter._require_open() is before_core
    assert id(parameter._require_open().storage) == before_storage
    assert parameter.version == 0
    assert parameter.grad is before_grad
    assert _same_bits(parameter.to_numpy(), before_value)
    assert _same_bits(source.to_numpy(), np.full(6, 9.0))
    gc.collect()
    assert len(live_storages) == baseline
    parameter.close()
    source.close()


@needs_native
@pytest.mark.parametrize("error", FAILURE_CLASSES)
def test_an_adoption_failure_releases_the_staged_copy(error, live_storages):
    """Failure *after* the staged copy exists but before the swap
    completes: the staged core must be released, not stranded."""
    parameter = NativeParameter(np.arange(6, dtype=np.float64))
    source = NativeTensor.from_array(np.full(6, 9.0))
    gc.collect()
    baseline = len(live_storages)
    before_value = parameter.to_numpy()

    real = NativeParameter._adopt_value_core

    def failing(self, new_core):
        raise error("injected adoption failure")

    NativeParameter._adopt_value_core = failing
    try:
        with pytest.raises(error, match="injected adoption failure"):
            parameter.copy_value_(source)
    finally:
        NativeParameter._adopt_value_core = real

    assert parameter.version == 0
    assert _same_bits(parameter.to_numpy(), before_value)
    gc.collect()
    assert len(live_storages) == baseline
    parameter.close()
    source.close()


@needs_native
def test_repeated_copy_cycles_return_live_storage_to_baseline(live_storages):
    parameter = NativeParameter(np.ones((16, 16)))
    source = NativeTensor.from_array(np.full((16, 16), 3.0))
    parameter.copy_value_(source)
    gc.collect()
    baseline = len(live_storages)
    for _ in range(25):
        parameter.copy_value_(source)
        _native_copy(source._require_open()).close()
    gc.collect()
    assert len(live_storages) == baseline
    parameter.close()
    source.close()


# ===========================================================================
# 7. Gradients — adoption is not accumulation
# ===========================================================================


@needs_native
def test_the_first_gradient_is_adopted_and_later_ones_are_added():
    """H5 touched the *materialization* of a gradient contribution, never
    the accumulation rule. The first contribution is adopted by identity;
    a second is summed with the native add kernel."""
    tensor = NativeTensor.from_array(np.zeros(3), requires_grad=True)
    try:
        first = NativeTensor.from_array(np.array([1.0, 2.0, 3.0]))
        tensor._accumulate_grad(first)
        assert tensor.grad is first
        second = NativeTensor.from_array(np.array([0.5, 0.5, 0.5]))
        tensor._accumulate_grad(second)
        assert tensor.grad is not first
        assert tensor.grad is not second
        assert _same_bits(tensor.grad.to_numpy(),
                          np.array([1.5, 2.5, 3.5]))
    finally:
        tensor.close()


@needs_native
def test_gradient_accumulation_still_sums_rather_than_assigns():
    """The property a copy-shaped optimization could plausibly break:
    two paths into one leaf must add, not overwrite."""
    leaf = NativeTensor.from_array(np.array([2.0, 3.0]), requires_grad=True)
    try:
        doubled = leaf.add(leaf)
        doubled.backward(NativeTensor.from_array(np.ones(2)))
        assert _same_bits(leaf.grad.to_numpy(), np.full(2, 2.0))
    finally:
        leaf.close()


@needs_native
def test_a_transposed_backward_materializes_an_owning_exact_gradient():
    """The reshape/transpose backward paths are ``_native_copy`` callers.
    The contribution must own its storage and carry the upstream's bits
    — including the ones an addition would have normalized."""
    padded = np.zeros((3, len(SWEEP)), dtype=np.float64)
    padded[0] = SWEEP
    leaf = NativeTensor.from_array(padded, requires_grad=True)
    try:
        transposed = leaf.transpose()
        upstream = NativeTensor.from_array(padded.T)
        transposed.backward(upstream)
        assert leaf.grad.owns_core
        assert leaf.grad.contiguous
        assert _same_bits(leaf.grad.to_numpy(), padded)
    finally:
        leaf.close()


@needs_native
def test_a_reshape_backward_materializes_an_owning_exact_gradient():
    values = np.concatenate([SWEEP, SWEEP])
    leaf = NativeTensor.from_array(values.reshape(4, 9), requires_grad=True)
    try:
        reshaped = leaf.reshape((6, 6))
        reshaped.backward(NativeTensor.from_array(values.reshape(6, 6)))
        assert leaf.grad.owns_core
        assert _same_bits(leaf.grad.to_numpy(), values.reshape(4, 9))
    finally:
        leaf.close()


@needs_native
def test_a_step_reads_gradients_and_never_writes_them():
    parameters = [NativeParameter(np.ones((2, 2))),
                  NativeParameter(np.full((3,), 2.0))]
    grads = []
    for parameter in parameters:
        grad = NativeTensor.from_array(np.full(parameter.shape, 0.25))
        parameter._grad = grad
        grads.append((grad, grad.to_numpy(), id(grad._require_open().storage)))
    optimizer = NativeAdam(parameters, lr=0.05)
    try:
        optimizer.step()
        for parameter, (grad, values, storage_id) in zip(parameters, grads):
            assert parameter.grad is grad
            assert _same_bits(grad.to_numpy(), values)
            assert id(grad._require_open().storage) == storage_id
    finally:
        optimizer.close()
        for parameter in parameters:
            parameter.close()


@needs_native
def test_zero_grad_still_drops_rather_than_closes_the_gradient():
    parameter = NativeParameter(np.ones(3))
    grad = NativeTensor.from_array(np.full(3, 0.5))
    parameter._grad = grad
    try:
        parameter.zero_grad()
        assert parameter.grad is None
        assert not grad.closed
        assert _same_bits(grad.to_numpy(), np.full(3, 0.5))
    finally:
        grad.close()
        parameter.close()


# ===========================================================================
# 8. Module and optimizer state transfer
# ===========================================================================


def _sweep_model():
    """A Linear whose parameters hold the whole IEEE-754 sweep, so every
    state-transfer path is exercised on the values that distinguish a
    copy from an addition."""
    model = NativeLinear(len(SWEEP), 2, seed=3)
    model.weight.copy_value_(NativeTensor.from_array(
        np.column_stack([SWEEP, SWEEP[::-1]])
    ))
    model.bias.copy_value_(NativeTensor.from_array(SWEEP[:2].copy()))
    return model


@needs_native
def test_a_state_dict_snapshot_is_bit_identical_to_the_parameter():
    model = _sweep_model()
    snapshot = model.state_dict()
    try:
        for name, value in snapshot.items():
            live = dict(model.named_parameters())[name]
            assert _same_bits(value.to_numpy(), live.to_numpy())
            assert value is not live
            assert value._require_open().storage is not \
                live._require_open().storage
    finally:
        for value in snapshot.values():
            value.close()
        _close_module(model)


@needs_native
def test_a_state_dict_round_trip_preserves_bits_identity_and_versions():
    model = _sweep_model()
    before = {n: p.to_numpy() for n, p in model.named_parameters()}
    identities = {n: id(p) for n, p in model.named_parameters()}
    versions = {n: p.version for n, p in model.named_parameters()}
    snapshot = model.state_dict()
    try:
        model.load_state_dict(snapshot)
        for name, parameter in model.named_parameters():
            assert _same_bits(parameter.to_numpy(), before[name]), name
            assert id(parameter) == identities[name], name
            assert parameter.version == versions[name] + 1, name
    finally:
        for value in snapshot.values():
            value.close()
        _close_module(model)


@needs_native
@pytest.mark.parametrize("error", FAILURE_CLASSES)
def test_a_module_load_failure_commits_nothing(error, live_storages):
    model = _sweep_model()
    snapshot = model.state_dict()
    replacement = {name: NativeTensor.from_array(np.full(value.shape, 7.0))
                   for name, value in snapshot.items()}
    before = {n: p.to_numpy() for n, p in model.named_parameters()}
    identities = {n: id(p) for n, p in model.named_parameters()}
    storages = {n: id(p._require_open().storage)
                for n, p in model.named_parameters()}
    versions = {n: p.version for n, p in model.named_parameters()}
    gc.collect()
    baseline = len(live_storages)

    calls = {"n": 0}
    real = native_tensor_module._native_copy

    def failing(core):
        calls["n"] += 1
        if calls["n"] == 2:
            raise error("injected staging failure")
        return real(core)

    module = importlib.import_module(
        "tensorforge.experimental.native_module"
    )
    original = module._native_copy
    module._native_copy = failing
    try:
        with pytest.raises(error, match="injected staging failure"):
            model.load_state_dict(replacement)
    finally:
        module._native_copy = original

    for name, parameter in model.named_parameters():
        assert _same_bits(parameter.to_numpy(), before[name]), name
        assert id(parameter) == identities[name], name
        assert id(parameter._require_open().storage) == storages[name], name
        assert parameter.version == versions[name], name
    for value in list(snapshot.values()) + list(replacement.values()):
        value.close()
    _close_module(model)
    gc.collect()
    assert len(live_storages) <= baseline


@needs_native
def test_optimizer_state_snapshots_are_bit_identical_and_independent():
    model = _sweep_model()
    parameters = list(model.parameters())
    for parameter in parameters:
        parameter._grad = NativeTensor.from_array(
            np.full(parameter.shape, 0.125)
        )
    optimizer = NativeAdam(parameters, lr=0.05)
    optimizer.step()
    state = optimizer.state_dict()
    try:
        for label in ("m", "v"):
            for index, snapshot in enumerate(state[label]):
                internal = getattr(optimizer, f"_{label}")[index]
                assert snapshot is not internal
                assert _same_bits(snapshot.to_numpy(), internal.to_numpy())
                assert snapshot._require_open().storage is not \
                    internal._require_open().storage
    finally:
        for label in ("m", "v"):
            for snapshot in state[label]:
                snapshot.close()
        optimizer.close()
        _close_module(model)


@needs_native
def test_an_optimizer_state_round_trip_is_exact_and_keeps_parameters():
    model = _sweep_model()
    parameters = list(model.parameters())
    for parameter in parameters:
        parameter._grad = NativeTensor.from_array(
            np.full(parameter.shape, 0.125)
        )
    optimizer = NativeAdam(parameters, lr=0.05)
    optimizer.step()
    state = optimizer.state_dict()
    parameter_versions = [p.version for p in parameters]
    try:
        before_m = [t.to_numpy() for t in optimizer._m]
        optimizer.load_state_dict(state)
        for original, current in zip(before_m, optimizer._m):
            assert _same_bits(current.to_numpy(), original)
        assert list(optimizer.step_counts) == list(state["step_counts"])
        # Loading optimizer state moves no parameter version.
        assert [p.version for p in parameters] == parameter_versions
    finally:
        for label in ("m", "v"):
            for snapshot in state[label]:
                snapshot.close()
        optimizer.close()
        _close_module(model)


@needs_native
@pytest.mark.parametrize("error", FAILURE_CLASSES)
def test_an_optimizer_load_failure_commits_nothing(error, live_storages):
    model = NativeLinear(4, 3, seed=11)
    parameters = list(model.parameters())
    for parameter in parameters:
        parameter._grad = NativeTensor.from_array(
            np.full(parameter.shape, 0.5)
        )
    optimizer = NativeAdam(parameters, lr=0.05)
    optimizer.step()
    state = optimizer.state_dict()
    optimizer.step()
    before_m = [t.to_numpy() for t in optimizer._m]
    before_ids = [id(t) for t in optimizer._m]
    before_steps = list(optimizer.step_counts)
    gc.collect()
    baseline = len(live_storages)

    adam = importlib.import_module("tensorforge.experimental.native_adam")
    calls = {"n": 0}
    real = adam._native_copy

    def failing(core):
        calls["n"] += 1
        if calls["n"] == 2:
            raise error("injected staging failure")
        return real(core)

    adam._native_copy = failing
    try:
        with pytest.raises(error, match="injected staging failure"):
            optimizer.load_state_dict(state)
    finally:
        adam._native_copy = real

    for original, current in zip(before_m, optimizer._m):
        assert _same_bits(current.to_numpy(), original)
    assert [id(t) for t in optimizer._m] == before_ids
    assert list(optimizer.step_counts) == before_steps
    for label in ("m", "v"):
        for snapshot in state[label]:
            snapshot.close()
    optimizer.close()
    _close_module(model)
    gc.collect()
    assert len(live_storages) <= baseline


# ===========================================================================
# 9. Persistent buffers and normalization running state
# ===========================================================================


@needs_native
def test_batchnorm_running_state_keeps_buffer_identity_across_updates():
    norm = NativeBatchNorm1d(3)
    norm.train()
    mean_id = id(norm.running_mean)
    var_id = id(norm.running_var)
    try:
        for step in range(4):
            data = np.arange(12, dtype=np.float64).reshape(4, 3) + step
            out = norm(NativeTensor.from_array(data))
            out.close()
            assert id(norm.running_mean) == mean_id
            assert id(norm.running_var) == var_id
    finally:
        _close_module(norm)


@needs_native
def test_batchnorm_running_state_round_trips_bit_exactly():
    norm = NativeBatchNorm1d(3)
    norm.train()
    out = norm(NativeTensor.from_array(
        np.arange(12, dtype=np.float64).reshape(4, 3)
    ))
    out.close()
    before = (norm.running_mean.to_numpy(), norm.running_var.to_numpy())
    snapshot = norm.state_dict()
    try:
        norm.load_state_dict(snapshot)
        assert _same_bits(norm.running_mean.to_numpy(), before[0])
        assert _same_bits(norm.running_var.to_numpy(), before[1])
    finally:
        for value in snapshot.values():
            value.close()
        _close_module(norm)


@needs_native
def test_a_persistent_buffer_load_preserves_identity_and_moves_no_version():
    norm = NativeBatchNorm1d(3)
    norm.train()
    out = norm(NativeTensor.from_array(
        np.arange(12, dtype=np.float64).reshape(4, 3)
    ))
    out.close()
    mean_id = id(norm.running_mean)
    gamma_version = norm.gamma.version
    snapshot = norm.state_dict()
    try:
        replacement = dict(snapshot)
        replacement["running_mean"] = NativeTensor.from_array(
            np.full(3, -0.0)
        )
        norm.load_state_dict(replacement)
        assert id(norm.running_mean) == mean_id
        # The gather preserves the sign of zero the caller supplied.
        assert _same_bits(norm.running_mean.to_numpy(), np.full(3, -0.0))
        assert np.signbit(norm.running_mean.to_numpy()).all()
        # A buffer carries no version; the affine parameters moved once
        # each, because the same load replaced them too.
        assert norm.gamma.version == gamma_version + 1
        replacement["running_mean"].close()
    finally:
        for value in snapshot.values():
            value.close()
        _close_module(norm)


# ===========================================================================
# 10. Checkpoints and exact resume
# ===========================================================================


@needs_native
def test_a_checkpoint_round_trip_preserves_parameter_bits(tmp_path):
    model = _sweep_model()
    parameters = list(model.parameters())
    for parameter in parameters:
        parameter._grad = NativeTensor.from_array(
            np.full(parameter.shape, 0.125)
        )
    optimizer = NativeAdam(parameters, lr=0.05)
    optimizer.step()
    before = {n: p.to_numpy() for n, p in model.named_parameters()}
    path = tmp_path / "sweep.npz"
    save_native_checkpoint(path, model, optimizer=optimizer)

    restored = _sweep_model()
    restored_parameters = list(restored.parameters())
    restored_optimizer = NativeAdam(restored_parameters, lr=0.05)
    try:
        load_native_checkpoint(path, restored, optimizer=restored_optimizer)
        for name, parameter in restored.named_parameters():
            assert _same_bits(parameter.to_numpy(), before[name]), name
    finally:
        restored_optimizer.close()
        _close_module(restored)
        optimizer.close()
        _close_module(model)


@needs_native
def test_the_checkpoint_format_and_supported_versions_did_not_move():
    checkpoint = importlib.import_module(
        "tensorforge.experimental.native_checkpoint"
    )
    assert checkpoint._FORMAT == "tensorforge.native_checkpoint"
    assert checkpoint._FORMAT_VERSION == 2
    assert set(checkpoint._SUPPORTED_FORMAT_VERSIONS) == {1, 2}


@needs_native
def test_an_interrupted_run_resumes_to_an_exact_next_step(tmp_path):
    """The end-to-end property H5 must not weaken: a run interrupted,
    checkpointed, and resumed into a *fresh* model and optimizer
    reproduces the uninterrupted run's parameters and losses exactly."""
    def _build():
        model = NativeLinear(4, 3, seed=5)
        parameters = list(model.parameters())
        return model, parameters, NativeAdam(parameters, lr=0.05)

    inputs = np.arange(24, dtype=np.float64).reshape(6, 4) / 8.0
    targets = np.arange(18, dtype=np.float64).reshape(6, 3) / 16.0

    def _step(model, optimizer):
        prediction = model(NativeTensor.from_array(inputs))
        difference = prediction.subtract(NativeTensor.from_array(targets))
        squared = difference.multiply(difference)
        loss = squared.mean()
        value = float(loss.to_numpy())
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        for tensor in (prediction, difference, squared, loss):
            tensor.close()
        return value

    model, parameters, optimizer = _build()
    reference = [_step(model, optimizer) for _ in range(8)]
    final = {n: p.to_numpy() for n, p in model.named_parameters()}
    optimizer.close()
    _close_module(model)

    model, parameters, optimizer = _build()
    for _ in range(4):
        _step(model, optimizer)
    path = tmp_path / "resume.npz"
    save_native_checkpoint(path, model, optimizer=optimizer)
    optimizer.close()
    _close_module(model)

    model, parameters, optimizer = _build()
    try:
        load_native_checkpoint(path, model, optimizer=optimizer)
        resumed = [_step(model, optimizer) for _ in range(4)]
        assert _same_bits(np.array(resumed), np.array(reference[4:]))
        for name, parameter in model.named_parameters():
            assert _same_bits(parameter.to_numpy(), final[name]), name
    finally:
        optimizer.close()
        _close_module(model)


# ===========================================================================
# 11. Live-storage accounting
# ===========================================================================


@needs_native
def test_repeated_state_transfer_cycles_return_storage_to_baseline(
    live_storages, tmp_path
):
    model = NativeLinear(6, 4, seed=17)
    parameters = list(model.parameters())
    for parameter in parameters:
        parameter._grad = NativeTensor.from_array(
            np.full(parameter.shape, 0.5)
        )
    optimizer = NativeAdam(parameters, lr=0.05)
    optimizer.step()
    path = tmp_path / "cycle.npz"
    save_native_checkpoint(path, model, optimizer=optimizer)
    load_native_checkpoint(path, model, optimizer=optimizer)
    gc.collect()
    baseline = len(live_storages)
    for _ in range(6):
        optimizer.step()
        snapshot = model.state_dict()
        model.load_state_dict(snapshot)
        for value in snapshot.values():
            value.close()
        state = optimizer.state_dict()
        optimizer.load_state_dict(state)
        for label in ("m", "v"):
            for tensor in state[label]:
                tensor.close()
        save_native_checkpoint(path, model, optimizer=optimizer)
        load_native_checkpoint(path, model, optimizer=optimizer)
    gc.collect()
    assert len(live_storages) == baseline
    optimizer.close()
    _close_module(model)


@needs_native
@pytest.mark.parametrize("error", FAILURE_CLASSES)
def test_failed_copy_cycles_also_return_storage_to_baseline(
    error, live_storages
):
    parameter = NativeParameter(np.ones((8, 8)))
    source = NativeTensor.from_array(np.full((8, 8), 2.0))
    parameter.copy_value_(source)
    gc.collect()
    baseline = len(live_storages)

    real = Core.contiguous_copy
    calls = {"n": 0}

    def flaky(self):
        calls["n"] += 1
        if calls["n"] % 2 == 0:
            raise error("injected copy failure")
        return real(self)

    cpp.NativeTensorCore.contiguous_copy = flaky
    try:
        for _ in range(10):
            try:
                parameter.copy_value_(source)
            except error:
                pass
    finally:
        cpp.NativeTensorCore.contiguous_copy = real
    gc.collect()
    assert len(live_storages) == baseline
    parameter.close()
    source.close()


# ===========================================================================
# 12. Surface guardrails
# ===========================================================================


@needs_native
def test_h5_added_no_exported_symbol():
    """H5 added no exported symbol: its traversal choice lives *inside* an
    existing export, so Phase H's surface is still exactly the 52 symbols
    H1 left. (The live library exports 54 — Phase I milestone I1 added the
    two typed storage creators — which is why the Phase-H claim is checked
    against the Phase-H subset.)"""
    storage_tests = importlib.import_module("test_native_storage_allocation") \
        if "test_native_storage_allocation" in sys.modules else None
    sys.path.insert(0, str(REPO_ROOT / "tests"))
    try:
        storage_tests = importlib.import_module(
            "test_native_storage_allocation"
        )
    finally:
        sys.path.pop(0)
    image, names = storage_tests.exported_names(cpp._LIBRARY_PATH)
    if names is None:
        pytest.skip("this image format is not parsed here")
    exported = sorted(name for name in names if name.startswith("tf_"))
    # H5's claim is about Phase H, so it is measured against Phase H's own
    # surface: the two extra symbols in the live library are Phase I's
    # typed storage creators, added at milestone I1.
    assert len(exported) == storage_tests.EXPECTED_TF_EXPORTS, exported
    assert (len(storage_tests.phase_h_export_names(exported))
            == storage_tests.PHASE_H_TF_EXPORTS)
    assert "tf_core_contiguous_copy" in exported
    # Nothing copy-, overlap-, or traversal-flavored was added.
    for banned in ("copy_mode", "set_copy", "overlap", "memcpy",
                   "traversal", "prefers", "select"):
        assert not [n for n in exported if banned in n.lower()], banned


@needs_native
def test_no_copy_selector_or_profiling_control_exists_anywhere():
    """H5 adds no public control of any kind: no selector, no counter, no
    environment variable, no validated constructor."""
    library = cpp._require_library()
    for name in ("tf_copy_set_mode", "tf_core_contiguous_copy_contiguous",
                 "tf_set_copy_path", "tf_copy_prefers_contiguous"):
        with pytest.raises(AttributeError):
            getattr(library, name)
    for module_name in ("tensorforge.backends.cpp",
                        "tensorforge.experimental.native_tensor",
                        "tensorforge.experimental.native_parameter"):
        module = importlib.import_module(module_name)
        for attribute in dir(module):
            lowered = attribute.lower()
            assert "copy_mode" not in lowered, (module_name, attribute)
            assert "copy_path" not in lowered, (module_name, attribute)
            assert not lowered.startswith("set_copy"), (module_name, attribute)


@needs_native
def test_the_copy_helper_stays_private():
    """``_native_copy`` is an internal seam, not an API."""
    import tensorforge.experimental as experimental

    assert "_native_copy" not in experimental.__all__
    assert not hasattr(experimental, "_native_copy")
    assert not hasattr(NativeTensor, "_native_copy")
    # And no public mutation surface was added beside copy_value_.
    public = {n for n in dir(NativeParameter) if not n.startswith("_")}
    mutators = {n for n in public if n.endswith("_")}
    assert mutators == {"copy_value_"}, mutators


def test_no_environment_variable_or_dispatch_hook_in_the_copy_sources():
    """Read from the sources themselves: the dispatch is a function of
    metadata, so nothing may consult the environment, the clock, or a
    CPU-feature probe."""
    sources = [
        REPO_ROOT / "cpp" / "src" / "elementwise.cpp",
        REPO_ROOT / "cpp" / "include" / "tf_copy_internal.h",
    ]
    for path in sources:
        text = path.read_text(encoding="utf-8")
        for banned in ("getenv", "std::chrono", "__cpuid", "cpuid",
                       "std::thread", "omp ", "#pragma omp"):
            assert banned not in text, (path.name, banned)


def test_the_stable_framework_still_imports_nothing_native():
    """Importing stable TensorForge must not pull in the native backend,
    the experimental modules, or the DLL — unchanged by H5."""
    # ``ctypes`` is deliberately *not* the probe: NumPy imports it
    # transitively, so its presence says nothing about TensorForge. The
    # decisive checks are that no native TensorForge module is imported
    # and that the DLL itself was never loaded.
    script = (
        "import sys\n"
        "import tensorforge\n"
        "bad = [m for m in sys.modules\n"
        "       if 'experimental' in m or m.endswith('backends.cpp')\n"
        "       or m.endswith('backends.native_backend')]\n"
        "assert not bad, bad\n"
        "loaded = [n for n in getattr(sys, 'modules', {})\n"
        "          if 'tensorforge_cpp' in n]\n"
        "assert not loaded, loaded\n"
        "assert hasattr(tensorforge, 'Adam')\n"
        "print('clean')\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True, text=True, cwd=str(REPO_ROOT),
    )
    assert result.returncode == 0, result.stderr
    assert "clean" in result.stdout


H5_SURFACES = ("README.md", "CLAUDE.md", "docs/roadmap.md",
               "docs/project_summary.md", "docs/architecture.md",
               "docs/backend_experiments.md",
               "docs/native_support_matrix.md",
               "docs/release_history.md",
               "docs/native_cpu_performance_design.md")


def _flat(path):
    """Markdown is hard-wrapped, so a sentence spans line breaks."""
    return " ".join((REPO_ROOT / path).read_text(encoding="utf-8").split())


def test_every_h5_surface_records_the_milestone_semantically():
    """Semantic documentation guardrails, not exact prose matching: each
    status surface must say H5 is complete, name what it actually did,
    and never claim an export or a control that does not exist."""
    import re

    for surface in H5_SURFACES:
        text = _flat(surface)
        lowered = text.lower()
        assert re.search(r"\bh5\b", lowered), surface
        # It names the subject...
        assert "copy" in lowered, surface
        # ...and never invents an ABI symbol or a wrong export count.
        for banned in ("tf_core_copy", "tf_storage_copy_value",
                       "tf_core_contiguous_copy_contiguous",
                       "53 exported", "53 `tf_*`"):
            assert banned not in lowered, (surface, banned)


def test_no_h5_surface_claims_an_added_export_or_a_public_control():
    """The two claims H5 must never let erode: it added no exported
    symbol, and it added no selector, mode, or profiling control."""
    import re

    claims = re.compile(
        r"\bH5\b[^.]{0,90}\b(added|adds|introduc\w+|export\w*)\b[^.]{0,60}"
        r"\b(symbol|export|selector|flag|mode|control|counter|"
        r"environment variable)\b",
        re.I)
    negations = re.compile(
        r"\b(no|not|never|none|without|neither|nor|nothing|unchanged"
        r"|deliberately|rather than|instead of)\b", re.I)
    for surface in H5_SURFACES:
        text = _flat(surface)
        offenders = [
            match.group(0) for match in claims.finditer(text)
            if not negations.search(
                text[max(0, match.start() - 120):match.end() + 60])
        ]
        assert offenders == [], (surface, offenders[:2])


def test_the_h5_surfaces_state_the_value_transfer_rule():
    """The one place exact prose matters: the contract H5 exists to fix.
    A surface that stops saying which patterns moved has stopped
    documenting the semantic change."""
    for surface in ("docs/native_cpu_performance_design.md",
                    "docs/backend_experiments.md",
                    "docs/project_summary.md", "README.md"):
        text = _flat(surface).lower()
        # It says the addition normalized negative zero...
        assert "-0.0" in text or "negative zero" in text, surface
        # ...and that a signaling NaN was quieted...
        assert "signaling nan" in text, surface
        # ...and it names the count, so "some patterns changed" is not
        # allowed to replace "exactly three".
        assert "exactly three" in text or "exactly **three**" in text, surface

    # CLAUDE.md is held to the *durable rule* rather than to H5's
    # historical count (H10; see ``_AGENT_INSTRUCTIONS`` in test_docs.py).
    # Which eighteen IEEE-754 patterns were swept and which three moved is
    # a milestone record, and the four surfaces above still carry it. What
    # the agent instructions must carry is the contract that survived: a
    # value transfer preserves its source's bits, negative zero and
    # signaling NaN included, and an *operation* does not.
    instructions = _flat("CLAUDE.md").lower()
    assert "-0.0" in instructions or "negative zero" in instructions
    assert "signaling nan" in instructions
    assert "value transfer" in instructions


def test_no_h5_surface_generalizes_h2s_nan_payload_carve_out_to_copies():
    """H2's payload carve-out is matmul-specific and must never be
    restated as if copies had one: a copy performs no arithmetic, so
    nothing selects a payload.

    A span carrying its own negation — "**no** NaN payload differed",
    "the payload cannot change" — is the honest form and passes; the
    thing being guarded against is a surface saying a *copy's* payloads
    may differ, which would be false."""
    import re

    claims = re.compile(
        r"(copy|copies|gather|value transfer)[^.]{0,80}"
        r"NaN payload[^.]{0,60}\b(may|might|can|differ\w*|outside|"
        r"not part of|unspecified)\b", re.I)
    negations = re.compile(
        r"\b(no|not|never|none|nothing|cannot|without|unchanged"
        r"|identical|preserv\w*)\b", re.I)
    for surface in H5_SURFACES:
        text = _flat(surface)
        offenders = [
            match.group(0) for match in claims.finditer(text)
            if not negations.search(
                text[max(0, match.start() - 60):match.end() + 40])
        ]
        assert offenders == [], (surface, offenders[:2])


@needs_native
def test_the_capability_boundary_did_not_move():
    assert cpp.UNSUPPORTED == ("float32", "cuda", "amp")
    assert cpp.SUPPORTED_DTYPES == ("float64",)
    assert cpp.SUPPORTED_DEVICES == ("cpu",)
