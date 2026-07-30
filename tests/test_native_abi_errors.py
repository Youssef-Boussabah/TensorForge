"""Native C ABI failure contract (repair milestone, Stages 2-3).

No C++ exception may cross ``extern "C"``: each fallible native function
clears a thread-local error slot on entry and, on any exception, records a
status code + message there and returns benignly; a ctypes errcheck hook
turns that into the right Python exception. These tests exercise the
failure paths deterministically through the test-only, inert-until-armed
allocation fault injection (docs/native_abi_error_contract.md).

Selector: python -m pytest -q -k native_abi
"""

import numpy as np
import pytest

from tensorforge.backends import cpp

needs_native = pytest.mark.skipif(
    not cpp.is_available(),
    reason="experimental C++ backend not built",
)
needs_fault_injection = pytest.mark.skipif(
    not (cpp.is_available() and cpp.fault_injection_available()),
    reason="fault injection not compiled into the backend",
)


@pytest.fixture(autouse=True)
def _disarm_after_each():
    """Every test leaves the injection hook disarmed and the error slot
    clear, so an armed countdown can never leak into another test."""
    yield
    if cpp.is_available():
        cpp._arm_alloc_failure(0)
        cpp._require_library().tf_clear_error()


@needs_native
def test_fault_injection_hook_present():
    assert cpp.fault_injection_available() is True


@needs_native
def test_no_error_after_normal_op():
    t = cpp.NativeTensorCore.from_array(np.array([1.0, -2.0, 3.0]))
    t.relu().close()
    assert cpp._require_library().tf_last_error_code() == cpp.TF_OK


@needs_fault_injection
def test_alloc_failure_during_storage_creation_raises_memoryerror():
    with pytest.raises(MemoryError):
        cpp._arm_alloc_failure(1)
        cpp.NativeStorage(16)


@needs_fault_injection
def test_alloc_failure_message_has_context():
    with pytest.raises(MemoryError) as info:
        cpp._arm_alloc_failure(1)
        cpp.NativeStorage(16)
    text = str(info.value)
    assert "native backend" in text
    assert "tf_storage_create" in text  # the failing operation is named


@needs_fault_injection
def test_alloc_failure_during_walker_counter():
    # The odometer kernel allocates a counter *after* the output storage.
    # Arm the 2nd allocation so the output succeeds and the counter alloc
    # fails.
    #
    # Phase H, milestone H8 gave the elementwise kernels a second traversal
    # that walks a collapsed operation-local plan instead, with no counter
    # and therefore no second allocation — so this test is anchored to a
    # layout the plan builder **rejects** and the odometer still owns: a
    # rank-5 fully reversed transpose, whose axes cannot be merged and whose
    # rank exceeds the plan's bound. The assertion is exactly what it was;
    # only the operand is chosen to still reach the counter.
    # ``test_the_planned_traversal_allocates_no_counter`` below is the other
    # half, proving the plan path really does skip the allocation.
    base = cpp.NativeTensorCore.zeros((2, 2, 2, 2, 2))
    strided = base.transpose((4, 3, 2, 1, 0))  # plan rejected -> odometer
    with pytest.raises(MemoryError):
        cpp._arm_alloc_failure(2)
        strided.relu()
    base.close()


@needs_fault_injection
def test_the_planned_traversal_allocates_no_counter():
    """The H8 counterpart: a strided view the plan *accepts* makes exactly
    one allocation (its output), so arming the second one leaves the call
    to succeed. Pre-H8 this raised MemoryError."""
    base = cpp.NativeTensorCore.from_array(np.arange(6.0).reshape(2, 3))
    strided = base.T  # non-contiguous, but a rank-2 plan the builder takes
    cpp._arm_alloc_failure(2)
    try:
        result = strided.relu()
    finally:
        cpp._arm_alloc_failure(0)  # disarm before anything else allocates
    try:
        assert np.array_equal(result.to_numpy(),
                              np.arange(6.0).reshape(2, 3).T)
    finally:
        result.close()
        base.close()


@needs_fault_injection
def test_operand_unchanged_and_recovers_after_failure():
    values = np.array([[1.0, -2.0], [3.0, 4.0]])
    t = cpp.NativeTensorCore.from_array(values)
    strided = t.T
    with pytest.raises(MemoryError):
        cpp._arm_alloc_failure(1)
        strided.relu()
    # The operand is untouched and a fresh op works (no stale error, no
    # partial mutation, no leaked native state visible to the caller).
    assert np.array_equal(t.to_numpy(), values)
    recovered = t.relu()
    assert np.array_equal(recovered.to_numpy(), np.maximum(values, 0.0))
    assert cpp._require_library().tf_last_error_code() == cpp.TF_OK
    recovered.close()
    t.close()


@needs_fault_injection
def test_no_contamination_across_calls():
    # A failed call must not leave a stale error that a later successful
    # call misreports. Each guarded native call clears the slot on entry.
    t = cpp.NativeTensorCore.from_array(np.array([1.0, 2.0, 3.0]))
    with pytest.raises(MemoryError):
        cpp._arm_alloc_failure(1)
        cpp.NativeTensorCore.zeros((4,))
    # Immediately do several successful ops; none should raise.
    for _ in range(3):
        out = t.relu()
        assert np.array_equal(out.to_numpy(), np.array([1.0, 2.0, 3.0]))
        out.close()
    t.close()


@needs_fault_injection
def test_disarm_and_nth_targeting():
    # nth=2 lets the first allocation through and fails the second.
    cpp._arm_alloc_failure(2)
    first = cpp.NativeStorage(4)  # allocation #1 succeeds
    first.close()
    with pytest.raises(MemoryError):
        cpp.NativeStorage(4)  # allocation #2 fails
    # Disarmed now; allocations succeed again.
    cpp.NativeStorage(4).close()


@needs_native
def test_invalid_argument_maps_to_valueerror():
    # Reach the INVALID status directly: the native storage constructor
    # rejects a non-positive size (Python validates earlier, so call the
    # raw kernel through the errcheck hook).
    library = cpp._require_library()
    with pytest.raises(ValueError):
        library.tf_storage_create(0)
    assert library.tf_last_error_code() == cpp.TF_OK  # errcheck cleared it


@needs_native
def test_error_accessors_roundtrip():
    library = cpp._require_library()
    library.tf_clear_error()
    assert library.tf_last_error_code() == cpp.TF_OK
    assert library.tf_last_error_message() == b""
