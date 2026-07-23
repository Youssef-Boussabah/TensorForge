"""MaxPool2d backward at the ``NativeTensorCore`` layer (Phase D,
milestone D9).

D9 adds the internal scatter-add kernel, the checked
``tf_core_maxpool2d_backward`` C ABI wrapper (which validates every winner
value), its ctypes/errcheck registration, and
``NativeTensorCore.maxpool2d_backward`` — the autograd-unaware Core method
the ``NativeTensor.maxpool2d`` node calls from its single input-gradient
callback. These tests cover scatter correctness, the output contract, the
Policy-B handling of non-contiguous operands, the validation surface
(including malformed winner rejection at the raw boundary), allocation
failure, and the continued privacy of the winner buffer.

Backend-dependent, so the module skips cleanly when the compiled backend
is not built. Cleanup is explicit via close().

Selector: python -m pytest -q -k native_maxpool2d_backward_core
"""

import numpy as np
import pytest

from tensorforge.backends import cpp

pytestmark = pytest.mark.skipif(
    not cpp.is_available(),
    reason="experimental C++ backend not built; " + cpp.build_instructions(),
)

needs_fault_injection = pytest.mark.skipif(
    not (cpp.is_available() and cpp.fault_injection_available()),
    reason="fault injection not compiled into the backend",
)


@pytest.fixture(autouse=True)
def _disarm_after_each():
    yield
    if cpp.is_available():
        cpp._arm_alloc_failure(0)
        cpp._require_library().tf_clear_error()


@pytest.fixture
def live_storages(monkeypatch):
    """The ids of every NativeStorage currently open — a real
    live-allocation count for the failure-atomicity tests."""
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


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

def _core(values):
    return cpp.NativeTensorCore.from_array(np.asarray(values, dtype=np.float64))


def _scatter(grad_output, winners, input_shape):
    """Run the Core backward over fresh cores; returns grad_input as NumPy
    and closes every native object it created."""
    gi = _core(grad_output)
    wi = _core(winners)
    try:
        grad = gi.maxpool2d_backward(wi, input_shape=input_shape)
        try:
            return grad.to_numpy()
        finally:
            grad.close()
    finally:
        gi.close()
        wi.close()


def _pooled(x, kernel_size, stride=None, padding=0):
    """The D8 forward's ``(output, winners)`` cores for ``x``."""
    xi = _core(x)
    try:
        return xi._maxpool2d_forward_with_winners(
            kernel_size=kernel_size, stride=stride, padding=padding
        )
    finally:
        xi.close()


# --------------------------------------------------------------------------
# Scatter correctness
# --------------------------------------------------------------------------

def test_hand_computed_scatter():
    grad = _scatter([[[[1.0, 2.0], [3.0, 4.0]]]],
                    [[[[5.0, 7.0], [13.0, 15.0]]]], (1, 1, 4, 4))
    assert grad.tolist() == [[[
        [0, 0, 0, 0],
        [0, 1, 0, 2],
        [0, 0, 0, 0],
        [0, 3, 0, 4],
    ]]]


def test_overlapping_windows_accumulate():
    # Four overlapping windows all selected the centre of a 3x3 plane.
    grad = _scatter([[[[1.0, 2.0], [3.0, 4.0]]]],
                    [[[[4.0, 4.0], [4.0, 4.0]]]], (1, 1, 3, 3))
    assert grad.tolist() == [[[[0, 0, 0], [0, 10.0, 0], [0, 0, 0]]]]


def test_multiple_channels_and_batch():
    grad = _scatter(
        [[[[1.0]], [[2.0]]], [[[3.0]], [[4.0]]]],
        [[[[3.0]], [[0.0]]], [[[1.0]], [[2.0]]]],
        (2, 2, 2, 2),
    )
    assert grad.tolist() == [
        [[[0, 0], [0, 1.0]], [[2.0, 0], [0, 0]]],
        [[[0, 3.0], [0, 0]], [[0, 0], [4.0, 0]]],
    ]


def test_padding_sentinel_drops_the_gradient():
    grad = _scatter([[[[1.0, 2.0], [3.0, 4.0]]]],
                    [[[[-1.0, 1.0], [-1.0, 3.0]]]], (1, 1, 2, 2))
    assert grad.tolist() == [[[[0.0, 2.0], [0.0, 4.0]]]]


def test_all_sentinel_winners_give_a_zero_gradient():
    grad = _scatter(np.ones((1, 1, 3, 3)), np.full((1, 1, 3, 3), -1.0),
                    (1, 1, 1, 1))
    assert grad.tolist() == [[[[0.0]]]]


def test_rectangular_shapes():
    grad = _scatter([[[[5.0, 6.0]]]], [[[[4.0, 5.0]]]], (1, 1, 2, 3))
    assert grad.tolist() == [[[[0, 0, 0], [0, 5.0, 6.0]]]]


def test_boundary_winner_offsets():
    grad = _scatter([[[[2.0, 3.0]]]], [[[[0.0, 11.0]]]], (1, 1, 3, 4))
    assert grad[0, 0, 0, 0] == 2.0
    assert grad[0, 0, 2, 3] == 3.0
    assert grad.sum() == 5.0


def test_forward_winners_drive_the_backward():
    # End-to-end at the Core layer: pool, then scatter a unit upstream
    # through the winners the forward saved.
    x = np.array([[[[1.0, 2.0, 3.0], [4.0, 9.0, 5.0], [6.0, 7.0, 8.0]]]])
    out, winners = _pooled(x, 2, stride=1)
    try:
        upstream = cpp.NativeTensorCore.full(out.shape, 1.0)
        try:
            grad = upstream.maxpool2d_backward(winners, input_shape=(1, 1, 3, 3))
        finally:
            upstream.close()
        # Every window selected the centre 9.0, so all four units land there.
        assert grad.to_numpy().tolist() == [[[[0, 0, 0], [0, 4.0, 0], [0, 0, 0]]]]
        grad.close()
    finally:
        out.close()
        winners.close()


# --------------------------------------------------------------------------
# Output contract
# --------------------------------------------------------------------------

def test_output_shape_layout_and_ownership():
    gi = _core(np.ones((2, 3, 2, 2)))
    wi = _core(np.zeros((2, 3, 2, 2)))
    grad = gi.maxpool2d_backward(wi, input_shape=(2, 3, 4, 4))
    assert grad.shape == (2, 3, 4, 4)
    assert grad.strides == cpp.row_major_strides(grad.shape)
    assert grad.offset == 0
    assert grad.contiguous
    assert grad._owns_storage is True
    assert grad.dtype == "float64" and grad.device == "cpu"
    assert grad.storage is not gi.storage and grad.storage is not wi.storage
    grad.close()
    gi.close()
    wi.close()


def test_operands_are_unchanged_and_outlived():
    upstream = np.array([[[[1.0, 2.0], [3.0, 4.0]]]])
    winners = np.array([[[[5.0, 7.0], [13.0, 15.0]]]])
    gi, wi = _core(upstream), _core(winners)
    grad = gi.maxpool2d_backward(wi, input_shape=(1, 1, 4, 4))
    assert np.array_equal(gi.to_numpy(), upstream)
    assert np.array_equal(wi.to_numpy(), winners)
    gi.close()
    wi.close()
    assert grad.to_numpy().sum() == 10.0  # independent of both operands
    grad.close()


# --------------------------------------------------------------------------
# Non-contiguous operands (Policy B)
# --------------------------------------------------------------------------

def test_non_contiguous_grad_output_matches_a_contiguous_copy():
    upstream = np.random.default_rng(1).standard_normal((2, 2, 3, 3))
    winners = np.random.default_rng(2).integers(0, 16, size=(2, 2, 3, 3))
    winners = winners.astype(np.float64)
    owner = _core(np.ascontiguousarray(upstream.transpose(0, 1, 3, 2)))
    view = owner.transpose(0, 1, 3, 2)
    assert not view.contiguous
    wi = _core(winners)
    grad = view.maxpool2d_backward(wi, input_shape=(2, 2, 4, 4))
    reference = _scatter(upstream, winners, (2, 2, 4, 4))
    assert np.allclose(grad.to_numpy(), reference, atol=1e-12)
    # The caller's view is untouched.
    assert np.allclose(view.to_numpy(), upstream, atol=1e-12)
    assert not view.contiguous
    grad.close()
    owner.close()
    wi.close()


def test_non_contiguous_winner_core_is_copied(monkeypatch):
    created = []
    original = cpp.NativeTensorCore.contiguous_copy

    def tracking_copy(self):
        result = original(self)
        created.append(result)
        return result

    monkeypatch.setattr(cpp.NativeTensorCore, "contiguous_copy", tracking_copy)
    winners = np.array([[[[0.0, 3.0], [5.0, 15.0]]]])
    owner = _core(np.ascontiguousarray(winners.transpose(0, 1, 3, 2)))
    view = owner.transpose(0, 1, 3, 2)
    gi = _core([[[[1.0, 2.0], [3.0, 4.0]]]])
    grad = gi.maxpool2d_backward(view, input_shape=(1, 1, 4, 4))
    assert len(created) == 1                      # only the winner copy
    assert all(copy._closed for copy in created)  # released after the call
    reference = _scatter([[[[1.0, 2.0], [3.0, 4.0]]]], winners, (1, 1, 4, 4))
    assert np.array_equal(grad.to_numpy(), reference)
    grad.close()
    owner.close()
    gi.close()


def test_contiguous_operands_are_not_copied(monkeypatch):
    calls = []
    original = cpp.NativeTensorCore.contiguous_copy

    def counting_copy(self):
        calls.append(self)
        return original(self)

    monkeypatch.setattr(cpp.NativeTensorCore, "contiguous_copy", counting_copy)
    gi, wi = _core(np.ones((1, 1, 2, 2))), _core(np.zeros((1, 1, 2, 2)))
    gi.maxpool2d_backward(wi, input_shape=(1, 1, 4, 4)).close()
    assert calls == []
    gi.close()
    wi.close()


# --------------------------------------------------------------------------
# Validation
# --------------------------------------------------------------------------

def test_closed_grad_output_rejected():
    gi, wi = _core(np.ones((1, 1, 2, 2))), _core(np.zeros((1, 1, 2, 2)))
    gi.close()
    with pytest.raises(RuntimeError):
        gi.maxpool2d_backward(wi, input_shape=(1, 1, 4, 4))
    wi.close()


def test_closed_winners_rejected():
    gi, wi = _core(np.ones((1, 1, 2, 2))), _core(np.zeros((1, 1, 2, 2)))
    wi.close()
    with pytest.raises(RuntimeError):
        gi.maxpool2d_backward(wi, input_shape=(1, 1, 4, 4))
    gi.close()


def test_non_core_winners_rejected():
    gi = _core(np.ones((1, 1, 2, 2)))
    with pytest.raises(TypeError):
        gi.maxpool2d_backward(np.zeros((1, 1, 2, 2)), input_shape=(1, 1, 4, 4))
    gi.close()


def test_grad_output_rank_not_four_rejected():
    gi, wi = _core(np.ones((2, 2))), _core(np.zeros((2, 2)))
    with pytest.raises(ValueError, match="4-D NCHW"):
        gi.maxpool2d_backward(wi, input_shape=(1, 1, 4, 4))
    gi.close()
    wi.close()


def test_winner_shape_mismatch_rejected():
    gi, wi = _core(np.ones((1, 1, 2, 2))), _core(np.zeros((1, 1, 2, 3)))
    with pytest.raises(ValueError, match="winner shape"):
        gi.maxpool2d_backward(wi, input_shape=(1, 1, 4, 4))
    gi.close()
    wi.close()


def test_batch_or_channel_mismatch_rejected():
    gi, wi = _core(np.ones((1, 1, 2, 2))), _core(np.zeros((1, 1, 2, 2)))
    with pytest.raises(ValueError, match="batch/channels"):
        gi.maxpool2d_backward(wi, input_shape=(2, 3, 4, 4))
    gi.close()
    wi.close()


@pytest.mark.parametrize(
    "input_shape", [(1, 1, 4), (1, 1, 4, 4, 4), (1, 1, 0, 4), (1, 1, -4, 4)]
)
def test_invalid_input_shape_rejected(input_shape):
    gi, wi = _core(np.ones((1, 1, 2, 2))), _core(np.zeros((1, 1, 2, 2)))
    with pytest.raises(ValueError):
        gi.maxpool2d_backward(wi, input_shape=input_shape)
    gi.close()
    wi.close()


def test_boolean_input_shape_member_rejected():
    gi, wi = _core(np.ones((1, 1, 2, 2))), _core(np.zeros((1, 1, 2, 2)))
    with pytest.raises(TypeError):
        gi.maxpool2d_backward(wi, input_shape=(1, 1, True, 4))
    gi.close()
    wi.close()


@pytest.mark.parametrize(
    "bad_winner",
    [0.5, np.nan, np.inf, -np.inf, -2.0, 16.0, 1e9],
    ids=["fractional", "nan", "posinf", "neginf", "below_sentinel",
         "one_past_end", "far_out_of_range"],
)
def test_malformed_winner_values_rejected(bad_winner):
    # The checked boundary rejects anything that is not -1 or an exact
    # in-range integer — it never rounds or truncates.
    winners = np.array([[[[0.0, 3.0], [5.0, 15.0]]]])
    winners[0, 0, 1, 0] = bad_winner
    gi, wi = _core(np.ones((1, 1, 2, 2))), _core(winners)
    with pytest.raises(ValueError, match="winner"):
        gi.maxpool2d_backward(wi, input_shape=(1, 1, 4, 4))
    # The error slot was consumed by errcheck, so the next call is clean.
    assert cpp._require_library().tf_last_error_code() == cpp.TF_OK
    gi.close()
    wi.close()


def test_valid_winner_domain_accepted():
    # The sentinel and both offset boundaries are legal.
    winners = np.array([[[[-1.0, 0.0], [15.0, 7.0]]]])
    grad = _scatter(np.ones((1, 1, 2, 2)), winners, (1, 1, 4, 4))
    assert grad.sum() == 3.0
    assert grad[0, 0, 0, 0] == 1.0
    assert grad[0, 0, 3, 3] == 1.0
    assert grad[0, 0, 1, 3] == 1.0


def test_plane_beyond_float64_exactness_rejected():
    storage = cpp.NativeStorage(4)
    view = cpp.NativeTensorView(storage, (1, 1, 2, 2))
    gi = cpp.NativeTensorCore(storage, view)
    wi = _core(np.zeros((1, 1, 2, 2)))
    with pytest.raises(ValueError, match="index"):
        gi.maxpool2d_backward(wi, input_shape=(1, 1, 2 ** 27, 2 ** 27))
    gi.close()
    wi.close()


# --------------------------------------------------------------------------
# Raw C ABI failure behavior
# --------------------------------------------------------------------------

# N, C, H, W, out_h, out_w for a 4x4 input pooled to 2x2.
_VALID_DIMS = (1, 1, 4, 4, 2, 2)


def _raw_call(lib, go, go_offset, wn, wn_offset, gi, dims=_VALID_DIMS):
    lib.tf_core_maxpool2d_backward(go, go_offset, wn, wn_offset, gi, *dims)


def test_raw_abi_null_handle_is_valueerror():
    lib = cpp._require_library()
    wi = _core(np.zeros((1, 1, 2, 2)))
    grad = cpp.NativeTensorCore.zeros((1, 1, 4, 4))
    with pytest.raises(ValueError):
        _raw_call(lib, None, 0, wi._storage._require_open(), 0,
                  grad._storage._require_open())
    assert lib.tf_last_error_code() == cpp.TF_OK
    grad.close()
    wi.close()


def test_raw_abi_negative_offset_is_valueerror():
    lib = cpp._require_library()
    gi = _core(np.ones((1, 1, 2, 2)))
    wi = _core(np.zeros((1, 1, 2, 2)))
    grad = cpp.NativeTensorCore.zeros((1, 1, 4, 4))
    with pytest.raises(ValueError):
        _raw_call(lib, gi._storage._require_open(), -1,
                  wi._storage._require_open(), 0,
                  grad._storage._require_open())
    grad.close()
    gi.close()
    wi.close()


@pytest.mark.parametrize("index", [0, 1, 2, 3, 4, 5])
def test_raw_abi_non_positive_dimension_is_valueerror(index):
    lib = cpp._require_library()
    gi = _core(np.ones((1, 1, 2, 2)))
    wi = _core(np.zeros((1, 1, 2, 2)))
    grad = cpp.NativeTensorCore.zeros((1, 1, 4, 4))
    dims = list(_VALID_DIMS)
    dims[index] = 0
    with pytest.raises(ValueError):
        _raw_call(lib, gi._storage._require_open(), 0,
                  wi._storage._require_open(), 0,
                  grad._storage._require_open(), tuple(dims))
    grad.close()
    gi.close()
    wi.close()


def test_raw_abi_undersized_grad_output_span_is_valueerror():
    lib = cpp._require_library()
    gi = _core(np.ones((1, 1, 1, 2)))  # only 2 doubles
    wi = _core(np.zeros((1, 1, 2, 2)))
    grad = cpp.NativeTensorCore.zeros((1, 1, 4, 4))
    with pytest.raises(ValueError):
        _raw_call(lib, gi._storage._require_open(), 0,
                  wi._storage._require_open(), 0,
                  grad._storage._require_open())
    grad.close()
    gi.close()
    wi.close()


def test_raw_abi_undersized_winner_span_is_valueerror():
    lib = cpp._require_library()
    gi = _core(np.ones((1, 1, 2, 2)))
    wi = _core(np.zeros((1, 1, 1, 2)))  # only 2 doubles
    grad = cpp.NativeTensorCore.zeros((1, 1, 4, 4))
    with pytest.raises(ValueError):
        _raw_call(lib, gi._storage._require_open(), 0,
                  wi._storage._require_open(), 0,
                  grad._storage._require_open())
    grad.close()
    gi.close()
    wi.close()


def test_raw_abi_undersized_grad_input_span_is_valueerror():
    lib = cpp._require_library()
    gi = _core(np.ones((1, 1, 2, 2)))
    wi = _core(np.zeros((1, 1, 2, 2)))
    small = cpp.NativeTensorCore.zeros((1, 1, 2, 2))  # needs 16 doubles
    with pytest.raises(ValueError):
        _raw_call(lib, gi._storage._require_open(), 0,
                  wi._storage._require_open(), 0,
                  small._storage._require_open())
    small.close()
    gi.close()
    wi.close()


def test_raw_abi_leaves_grad_input_untouched_on_invalid_winner():
    lib = cpp._require_library()
    gi = _core(np.ones((1, 1, 2, 2)))
    wi = _core([[[[0.0, 3.0], [0.25, 15.0]]]])  # fractional winner
    grad = cpp.NativeTensorCore.full((1, 1, 4, 4), 5.0)
    with pytest.raises(ValueError):
        _raw_call(lib, gi._storage._require_open(), 0,
                  wi._storage._require_open(), 0,
                  grad._storage._require_open())
    # Rejected before the kernel wrote anything.
    assert np.all(grad.to_numpy() == 5.0)
    grad.close()
    gi.close()
    wi.close()


def test_no_stale_error_after_a_raw_failure():
    lib = cpp._require_library()
    wi = _core(np.zeros((1, 1, 2, 2)))
    grad = cpp.NativeTensorCore.zeros((1, 1, 4, 4))
    with pytest.raises(ValueError):
        _raw_call(lib, None, 0, wi._storage._require_open(), 0,
                  grad._storage._require_open())
    grad.close()
    gi = _core(np.ones((1, 1, 2, 2)))
    good = gi.maxpool2d_backward(wi, input_shape=(1, 1, 4, 4))
    assert good.shape == (1, 1, 4, 4)
    assert lib.tf_last_error_code() == cpp.TF_OK
    good.close()
    gi.close()
    wi.close()


# --------------------------------------------------------------------------
# Failure atomicity
# --------------------------------------------------------------------------

@needs_fault_injection
def test_grad_input_allocation_failure_is_atomic(live_storages):
    upstream = np.ones((1, 1, 2, 2))
    winners = np.array([[[[5.0, 7.0], [13.0, 15.0]]]])
    gi, wi = _core(upstream), _core(winners)
    baseline = len(live_storages)
    cpp._arm_alloc_failure(1)  # contiguous operands: the output is alloc #1
    with pytest.raises(MemoryError):
        gi.maxpool2d_backward(wi, input_shape=(1, 1, 4, 4))
    assert len(live_storages) == baseline          # nothing leaked
    assert np.array_equal(gi.to_numpy(), upstream)  # operands untouched
    assert np.array_equal(wi.to_numpy(), winners)
    recovered = gi.maxpool2d_backward(wi, input_shape=(1, 1, 4, 4))
    assert recovered.to_numpy().sum() == 4.0
    assert cpp._require_library().tf_last_error_code() == cpp.TF_OK
    recovered.close()
    gi.close()
    wi.close()


def test_output_closed_when_the_native_call_fails(monkeypatch, live_storages):
    lib = cpp._require_library()
    original = lib.tf_core_maxpool2d_backward

    def boom(*args):
        raise RuntimeError("simulated native scatter failure")

    monkeypatch.setattr(lib, "tf_core_maxpool2d_backward", boom)
    gi, wi = _core(np.ones((1, 1, 2, 2))), _core(np.zeros((1, 1, 2, 2)))
    baseline = len(live_storages)
    with pytest.raises(RuntimeError, match="simulated"):
        gi.maxpool2d_backward(wi, input_shape=(1, 1, 4, 4))
    assert len(live_storages) == baseline  # the fresh output was closed
    monkeypatch.setattr(lib, "tf_core_maxpool2d_backward", original)
    grad = gi.maxpool2d_backward(wi, input_shape=(1, 1, 4, 4))
    grad.close()
    gi.close()
    wi.close()


def test_temporary_copy_released_when_allocation_fails(monkeypatch,
                                                      live_storages):
    created = []
    inside_copy = []
    original_copy = cpp.NativeTensorCore.contiguous_copy
    original_zeros = cpp.NativeTensorCore.zeros

    def tracking_copy(self):
        # As of E3.1 the Policy-B copy is a native storage-to-storage
        # gather that allocates its destination through zeros() too, so
        # the flag below distinguishes the copy's own allocation from the
        # grad_input allocation (which is what must fail here).
        inside_copy.append(True)
        try:
            result = original_copy(self)
        finally:
            inside_copy.pop()
        created.append(result)
        return result

    def boom(*args, **kwargs):
        if inside_copy:
            return original_zeros(*args, **kwargs)
        raise MemoryError("simulated grad_input allocation failure")

    monkeypatch.setattr(cpp.NativeTensorCore, "contiguous_copy", tracking_copy)
    monkeypatch.setattr(cpp.NativeTensorCore, "zeros", staticmethod(boom))
    upstream = np.random.default_rng(3).standard_normal((1, 1, 2, 2))
    owner = _core(np.ascontiguousarray(upstream.transpose(0, 1, 3, 2)))
    view = owner.transpose(0, 1, 3, 2)
    wi = _core(np.zeros((1, 1, 2, 2)))
    baseline = len(live_storages)
    with pytest.raises(MemoryError):
        view.maxpool2d_backward(wi, input_shape=(1, 1, 4, 4))
    assert created, "expected a Policy-B temporary"
    assert all(copy._closed for copy in created)
    assert len(live_storages) == baseline
    assert np.allclose(view.to_numpy(), upstream, atol=1e-12)
    owner.close()
    wi.close()


# --------------------------------------------------------------------------
# Capability separation and winner privacy
# --------------------------------------------------------------------------

def test_core_backward_advertised_in_tensor_core_ops():
    assert "maxpool2d_backward" in cpp.TENSOR_CORE_OPS
    assert hasattr(cpp.NativeTensorCore, "maxpool2d_backward")
    assert "maxpool2d_forward" in cpp.TENSOR_CORE_OPS


def test_raw_backward_symbol_registered_in_the_error_contract_only():
    assert "tf_core_maxpool2d_backward" in cpp._CHECKED_KERNELS
    lib = cpp._require_library()
    assert callable(getattr(lib, "tf_core_maxpool2d_backward"))
    assert "maxpool2d_backward" not in cpp.RAW_KERNELS
    assert "maxpool2d_backward" not in cpp.TENSOR_CORE_KERNELS


def test_backward_exposes_no_public_winner_surface():
    public_core = [name for name in dir(cpp.NativeTensorCore)
                   if not name.startswith("_")]
    assert not [name for name in public_core if "winner" in name.lower()]
    assert not [name for name in public_core if "indices" in name.lower()]
    # Still float64-only: the winner buffer introduced no index dtype.
    assert cpp.SUPPORTED_DTYPES == ("float64",)
