"""Forward-only native MaxPool2d at the ``NativeTensorCore`` layer (Phase
D, milestone D8).

D8 ships the internal CPU float64 window-maximum kernel, its
exception-safe C ABI wrapper (``tf_core_maxpool2d_forward``), the
ctypes/errcheck registration, and ``NativeTensorCore.maxpool2d_forward`` —
a forward-only, autograd-unaware Core wrapper that also produces the
**private saved-winner buffer** the D9 backward will consume. These tests
cover forward correctness and stable-framework parity, the winner
representation and its exactness bound, the output/winner ownership and
lifetime contract, the Policy-B handling of non-contiguous input, the full
validation surface, the raw-ABI failure behavior, allocation atomicity,
and the capability-registry separation (Core forward implemented; the
differentiable ``maxpool2d`` autograd op, MaxPool2d backward, and the
``NativeMaxPool2d`` module all stay unsupported).

Backend-dependent, so the module skips cleanly when the compiled backend
is not built. Cleanup is explicit via close(); nothing depends on
garbage-collection timing.

Selector: python -m pytest -q -k native_maxpool2d_core
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

NEG_INF = -np.inf


@pytest.fixture(autouse=True)
def _disarm_after_each():
    """Leave the injection hook disarmed and the error slot clear after
    every test, so an armed countdown never leaks into the next one."""
    yield
    if cpp.is_available():
        cpp._arm_alloc_failure(0)
        cpp._require_library().tf_clear_error()


@pytest.fixture
def live_storages(monkeypatch):
    """A set of the ids of every NativeStorage currently open.

    Wrapping the storage constructor and ``close()`` gives a real
    live-native-allocation count, so a failure test can prove the count
    returns to its baseline instead of trusting garbage collection."""
    open_ids = set()
    original_init = cpp.NativeStorage.__init__
    original_close = cpp.NativeStorage.close

    def tracked_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)  # raises => never recorded
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
    """A contiguous NativeTensorCore holding a copy of ``values``."""
    return cpp.NativeTensorCore.from_array(np.asarray(values, dtype=np.float64))


def _pool(x, kernel_size, stride=None, padding=0):
    """Run the Core forward over fresh storage and return
    ``(output, winners)`` as NumPy arrays, closing every native object."""
    xi = _core(x)
    try:
        out, winners = xi._maxpool2d_forward_with_winners(
            kernel_size=kernel_size, stride=stride, padding=padding
        )
        try:
            return out.to_numpy(), winners.to_numpy()
        finally:
            out.close()
            winners.close()
    finally:
        xi.close()


def _stable_reference(x, kernel_size, stride=None, padding=0):
    """The stable ``tensorforge.nn.MaxPool2d`` forward on the same data —
    the numerical reference the native line must match."""
    from tensorforge.nn import MaxPool2d
    from tensorforge.tensor import Tensor

    layer = MaxPool2d(kernel_size, stride=stride, padding=padding)
    return layer(Tensor(np.asarray(x, dtype=np.float64))).data


def _stable_winner_offsets(x, kernel_size, stride=None, padding=0):
    """The stable layer's window-local ``argmax`` winners re-expressed in
    the native representation: the flat offset into the ``(n, c)`` plane,
    or ``-1`` when the selected cell is padding."""
    x = np.asarray(x, dtype=np.float64)
    n, c, h, w = x.shape
    kh, kw = (kernel_size, kernel_size) if isinstance(kernel_size, int) else kernel_size
    if stride is None:
        sh, sw = kh, kw
    else:
        sh, sw = (stride, stride) if isinstance(stride, int) else stride
    ph, pw = (padding, padding) if isinstance(padding, int) else padding
    out_h = (h + 2 * ph - kh) // sh + 1
    out_w = (w + 2 * pw - kw) // sw + 1
    padded = np.pad(
        x, ((0, 0), (0, 0), (ph, ph), (pw, pw)), constant_values=-np.inf
    )
    winners = np.zeros((n, c, out_h, out_w))
    for bi in range(n):
        for ci in range(c):
            for i in range(out_h):
                for j in range(out_w):
                    window = padded[
                        bi, ci, i * sh:i * sh + kh, j * sw:j * sw + kw
                    ]
                    idx = int(window.reshape(-1).argmax())
                    ih = i * sh + idx // kw - ph
                    iw = j * sw + idx % kw - pw
                    inside = 0 <= ih < h and 0 <= iw < w
                    winners[bi, ci, i, j] = ih * w + iw if inside else -1
    return winners


def _noncontiguous_input(values):
    """``(owner, view)`` where ``view`` is a non-contiguous NCHW core with
    exactly ``values``. The owner holds the real storage."""
    array = np.asarray(values, dtype=np.float64)
    # Store the axis-swapped array, then transpose back: same values, a
    # genuinely non-contiguous view.
    owner = _core(np.ascontiguousarray(array.transpose(0, 1, 3, 2)))
    view = owner.transpose(0, 1, 3, 2)
    assert not view.contiguous
    return owner, view


# --------------------------------------------------------------------------
# Forward correctness
# --------------------------------------------------------------------------

def test_hand_computed_output_and_winners():
    x = np.arange(1, 17, dtype=float).reshape(1, 1, 4, 4)
    out, winners = _pool(x, 2)  # stride defaults to the kernel
    assert out.tolist() == [[[[6.0, 8.0], [14.0, 16.0]]]]
    assert winners.tolist() == [[[[5.0, 7.0], [13.0, 15.0]]]]


def test_batch_and_multiple_channels():
    x = np.array([
        [[[1, 2], [3, 4]], [[8, 7], [6, 5]]],
        [[[-1, -2], [-3, -4]], [[0.5, 9.5], [2.5, 1.5]]],
    ], dtype=float)
    out, winners = _pool(x, 2)
    assert out.tolist() == [[[[4.0]], [[8.0]]], [[[-1.0]], [[9.5]]]]
    # Winners are per-plane offsets, so the same offset repeats freely.
    assert winners.tolist() == [[[[3.0]], [[0.0]]], [[[0.0]], [[1.0]]]]


def test_rectangular_kernel():
    x = np.arange(1, 10, dtype=float).reshape(1, 1, 3, 3)
    out, winners = _pool(x, (1, 3), stride=1)
    assert out.tolist() == [[[[3.0], [6.0], [9.0]]]]
    assert winners.tolist() == [[[[2.0], [5.0], [8.0]]]]


def test_integer_and_tuple_arguments_agree():
    x = np.arange(1, 17, dtype=float).reshape(1, 1, 4, 4)
    int_out, int_win = _pool(x, 2, stride=2, padding=1)
    tuple_out, tuple_win = _pool(x, (2, 2), stride=(2, 2), padding=(1, 1))
    assert np.array_equal(int_out, tuple_out)
    assert np.array_equal(int_win, tuple_win)


def test_stride_none_defaults_to_kernel_size():
    x = np.arange(1, 17, dtype=float).reshape(1, 1, 4, 4)
    default_out, default_win = _pool(x, 2, stride=None)
    explicit_out, explicit_win = _pool(x, 2, stride=2)
    assert np.array_equal(default_out, explicit_out)
    assert np.array_equal(default_win, explicit_win)
    # ... and differs from an overlapping stride, so the default is real.
    overlapping_out, _ = _pool(x, 2, stride=1)
    assert overlapping_out.shape != default_out.shape


def test_padding_gives_one_window_per_input_cell():
    x = np.array([[[[1.0, 2.0], [3.0, 4.0]]]])
    out, winners = _pool(x, 2, stride=2, padding=1)
    # Each window holds exactly one real cell; padding (-inf) never wins.
    assert out.tolist() == [[[[1.0, 2.0], [3.0, 4.0]]]]
    assert winners.tolist() == [[[[0.0, 1.0], [2.0, 3.0]]]]


def test_combined_stride_and_padding():
    x = np.arange(1, 10, dtype=float).reshape(1, 1, 3, 3)
    out, winners = _pool(x, 2, stride=2, padding=1)
    assert out.tolist() == [[[[1.0, 3.0], [7.0, 9.0]]]]
    assert winners.tolist() == [[[[0.0, 2.0], [6.0, 8.0]]]]


def test_negative_and_fractional_values():
    x = np.array([[[[-4.0, -1.5], [-3.25, -2.0]]]])
    out, winners = _pool(x, 2)
    assert out.tolist() == [[[[-1.5]]]]  # not 0: no zero-seeded accumulator
    assert winners.tolist() == [[[[1.0]]]]


def test_tie_selects_the_first_row_major_position():
    assert _pool(np.full((1, 1, 2, 2), 5.0), 2)[1].tolist() == [[[[0.0]]]]
    x = np.array([[[[1.0, 5.0], [5.0, 2.0]]]])
    out, winners = _pool(x, 2)
    assert out.tolist() == [[[[5.0]]]]
    assert winners.tolist() == [[[[1.0]]]]  # the earlier of the two maxima


def test_all_negative_infinity_without_padding_picks_the_first_real_cell():
    x = np.full((1, 1, 2, 2), NEG_INF)
    out, winners = _pool(x, 2)
    assert out.tolist() == [[[[NEG_INF]]]]
    assert winners.tolist() == [[[[0.0]]]]  # a real cell, not the sentinel


def test_negative_infinity_versus_padding_tie():
    # Every value is -inf, so the first row-major position decides: three
    # windows begin on padding (-1), the last begins on the real (1, 1).
    x = np.full((1, 1, 2, 2), NEG_INF)
    out, winners = _pool(x, 2, stride=2, padding=1)
    assert out.tolist() == [[[[NEG_INF, NEG_INF], [NEG_INF, NEG_INF]]]]
    assert winners.tolist() == [[[[-1.0, -1.0], [-1.0, 3.0]]]]


def test_completely_padded_windows_are_allowed():
    x = np.array([[[[7.0]]]])
    out, winners = _pool(x, 1, stride=1, padding=1)  # 3x3 output
    assert out.tolist() == [[[
        [NEG_INF, NEG_INF, NEG_INF],
        [NEG_INF, 7.0, NEG_INF],
        [NEG_INF, NEG_INF, NEG_INF],
    ]]]
    assert winners.tolist() == [[[
        [-1.0, -1.0, -1.0],
        [-1.0, 0.0, -1.0],
        [-1.0, -1.0, -1.0],
    ]]]


def test_determinism():
    x = np.random.default_rng(3).standard_normal((2, 2, 5, 5))
    first_out, first_win = _pool(x, 3, stride=2, padding=1)
    second_out, second_win = _pool(x, 3, stride=2, padding=1)
    assert np.array_equal(first_out, second_out)  # bit-identical
    assert np.array_equal(first_win, second_win)


@pytest.mark.parametrize(
    "kernel_size, stride, padding",
    [
        (2, None, 0),
        (2, 1, 0),
        (3, 2, 1),
        ((3, 2), (2, 1), (1, 0)),
        ((2, 3), 1, (0, 1)),
    ],
)
def test_stable_framework_parity(kernel_size, stride, padding):
    x = np.round(np.random.default_rng(11).standard_normal((2, 3, 6, 5)) * 3, 3)
    out, winners = _pool(x, kernel_size, stride=stride, padding=padding)
    reference = _stable_reference(x, kernel_size, stride=stride, padding=padding)
    # Pooling selects a value verbatim — no summation — so parity is exact.
    assert np.array_equal(out, reference)
    assert np.array_equal(
        winners, _stable_winner_offsets(x, kernel_size, stride, padding)
    )


# --------------------------------------------------------------------------
# Output contract
# --------------------------------------------------------------------------

def test_output_shape_strides_offset_and_ownership():
    xi = _core(np.arange(48, dtype=float).reshape(2, 2, 4, 3))
    out, winners = xi._maxpool2d_forward_with_winners(kernel_size=2)
    assert out.shape == (2, 2, 2, 1)
    assert out.strides == cpp.row_major_strides(out.shape)  # canonical
    assert out.offset == 0
    assert out.contiguous
    assert out._owns_storage is True
    assert out.dtype == "float64" and out.device == "cpu"
    out.close()
    winners.close()
    xi.close()


def test_output_does_not_alias_the_input_and_survives_it():
    values = np.arange(1, 17, dtype=float).reshape(1, 1, 4, 4)
    xi = _core(values)
    out, winners = xi._maxpool2d_forward_with_winners(kernel_size=2)
    assert out.storage is not xi.storage  # fresh storage, no aliasing
    assert winners.storage is not xi.storage
    xi.close()  # the input's storage is gone...
    assert out.to_numpy().tolist() == [[[[6.0, 8.0], [14.0, 16.0]]]]
    assert winners.to_numpy().tolist() == [[[[5.0, 7.0], [13.0, 15.0]]]]
    out.close()
    winners.close()


def test_input_is_unchanged_by_the_forward():
    values = np.random.default_rng(4).standard_normal((1, 2, 4, 4))
    xi = _core(values)
    out, winners = xi._maxpool2d_forward_with_winners(kernel_size=2, padding=1)
    assert np.array_equal(xi.to_numpy(), values)
    assert xi.shape == (1, 2, 4, 4) and xi.contiguous
    out.close()
    winners.close()
    xi.close()


def test_public_core_method_returns_only_the_pooled_values():
    xi = _core(np.arange(1, 17, dtype=float).reshape(1, 1, 4, 4))
    out = xi.maxpool2d_forward(kernel_size=2)
    assert isinstance(out, cpp.NativeTensorCore)
    assert out.to_numpy().tolist() == [[[[6.0, 8.0], [14.0, 16.0]]]]
    out.close()
    xi.close()


# --------------------------------------------------------------------------
# Winner contract (private saved state, D9's input)
# --------------------------------------------------------------------------

def test_winner_shape_layout_and_ownership():
    xi = _core(np.arange(48, dtype=float).reshape(2, 2, 4, 3))
    out, winners = xi._maxpool2d_forward_with_winners(kernel_size=2)
    assert winners.shape == out.shape == (2, 2, 2, 1)
    assert winners.strides == cpp.row_major_strides(winners.shape)
    assert winners.offset == 0
    assert winners.contiguous
    assert winners._owns_storage is True
    out.close()
    winners.close()
    xi.close()


def test_winner_values_are_integral_float64_or_the_sentinel():
    x = np.random.default_rng(5).standard_normal((2, 2, 5, 5))
    _, winners = _pool(x, 3, stride=1, padding=1)
    assert winners.dtype == np.float64
    assert np.all(winners == np.floor(winners))       # exact integers
    assert np.all(winners >= -1)                      # -1 is the only negative
    finite = winners[winners >= 0]
    assert np.all(finite < 25)                        # inside the H*W plane


def test_winner_offsets_point_at_the_selected_input_element():
    x = np.random.default_rng(6).standard_normal((2, 3, 5, 4))
    out, winners = _pool(x, 2, stride=1)
    n, c, out_h, out_w = out.shape
    planes = x.reshape(n, c, -1)
    for bi in range(n):
        for ci in range(c):
            for i in range(out_h):
                for j in range(out_w):
                    index = int(winners[bi, ci, i, j])
                    assert index >= 0  # no padding here
                    assert planes[bi, ci, index] == out[bi, ci, i, j]


def test_winners_survive_the_input_and_close_exactly_once():
    xi = _core(np.arange(1, 17, dtype=float).reshape(1, 1, 4, 4))
    out, winners = xi._maxpool2d_forward_with_winners(kernel_size=2)
    xi.close()
    assert winners.to_numpy().tolist() == [[[[5.0, 7.0], [13.0, 15.0]]]]
    winners.close()
    assert winners._closed is True
    winners.close()  # idempotent
    with pytest.raises(RuntimeError):
        winners.to_numpy()
    out.close()


def test_winner_buffer_is_not_public_surface():
    from tensorforge.experimental import NativeTensor

    # No public winner/index API anywhere on the native tensor layers.
    public_core = [name for name in dir(cpp.NativeTensorCore)
                   if not name.startswith("_")]
    assert not [name for name in public_core if "winner" in name.lower()]
    assert not [name for name in public_core if "indices" in name.lower()]
    # The winner-carrying helper is deliberately private.
    assert "_maxpool2d_forward_with_winners" in dir(cpp.NativeTensorCore)
    assert "maxpool2d_forward" in dir(cpp.NativeTensorCore)
    # The differentiable op exists as of D9, but it still exposes no winner
    # surface: the buffer stays private graph state.
    assert hasattr(NativeTensor, "maxpool2d")
    public_tensor = [name for name in dir(NativeTensor)
                     if not name.startswith("_")]
    assert not [name for name in public_tensor if "winner" in name.lower()]
    assert not [name for name in public_tensor if "indices" in name.lower()]


def test_winner_buffer_claims_no_new_dtype_capability():
    info = cpp.backend_info()
    assert cpp.SUPPORTED_DTYPES == ("float64",)
    assert info["supported_dtypes"] == ("float64",)
    assert info["dtype"] == "float64"
    # No integer dtype/index capability is advertised anywhere.
    advertised = " ".join(
        str(info[key]) for key in
        ("raw_kernels", "tensor_core_ops", "autograd_ops", "native_modules")
    )
    for banned in ("int64", "int32", "winner", "indices"):
        assert banned not in advertised


# --------------------------------------------------------------------------
# Non-contiguous input (Policy B)
# --------------------------------------------------------------------------

def test_non_contiguous_input_matches_an_explicit_contiguous_copy():
    values = np.random.default_rng(7).standard_normal((2, 2, 5, 4))
    owner, view = _noncontiguous_input(values)
    assert np.allclose(view.to_numpy(), values, atol=1e-12)
    out, winners = view._maxpool2d_forward_with_winners(
        kernel_size=3, stride=2, padding=1
    )
    reference_out, reference_win = _pool(values, 3, stride=2, padding=1)
    assert np.array_equal(out.to_numpy(), reference_out)
    assert np.array_equal(winners.to_numpy(), reference_win)
    assert np.array_equal(
        out.to_numpy(), _stable_reference(values, 3, stride=2, padding=1)
    )
    # The caller's view and its storage are untouched and still usable.
    assert np.allclose(view.to_numpy(), values, atol=1e-12)
    assert not view.contiguous
    out.close()
    winners.close()
    owner.close()


def test_non_contiguous_temporary_is_released(monkeypatch):
    created = []
    original = cpp.NativeTensorCore.contiguous_copy

    def tracking_copy(self):
        result = original(self)
        created.append(result)
        return result

    monkeypatch.setattr(cpp.NativeTensorCore, "contiguous_copy", tracking_copy)
    values = np.random.default_rng(8).standard_normal((1, 2, 4, 4))
    owner, view = _noncontiguous_input(values)
    out, winners = view._maxpool2d_forward_with_winners(kernel_size=2)
    assert len(created) == 1                       # exactly one input copy
    assert all(copy._closed for copy in created)   # closed after the call
    # The results are independent of that temporary and stay valid.
    assert out.to_numpy().shape == (1, 2, 2, 2)
    assert winners.to_numpy().shape == (1, 2, 2, 2)
    out.close()
    winners.close()
    owner.close()


def test_contiguous_input_is_not_copied(monkeypatch):
    calls = []
    original = cpp.NativeTensorCore.contiguous_copy

    def counting_copy(self):
        calls.append(self)
        return original(self)

    monkeypatch.setattr(cpp.NativeTensorCore, "contiguous_copy", counting_copy)
    xi = _core(np.ones((1, 1, 4, 4)))
    xi.maxpool2d_forward(kernel_size=2).close()
    assert calls == []  # no unnecessary copy
    xi.close()


def test_offset_contiguous_view_is_passed_through(monkeypatch):
    calls = []
    original = cpp.NativeTensorCore.contiguous_copy

    def counting_copy(self):
        calls.append(self)
        return original(self)

    monkeypatch.setattr(cpp.NativeTensorCore, "contiguous_copy", counting_copy)
    values = np.arange(2 * 1 * 4 * 4, dtype=float).reshape(2, 1, 4, 4)
    owner = _core(values)
    tail = owner.narrow(0, 1, 1)  # contiguous, non-zero offset
    assert tail.contiguous and tail.offset == 16
    out, winners = tail._maxpool2d_forward_with_winners(kernel_size=2)
    assert calls == []  # already contiguous: only the offset is passed
    expected_out, expected_win = _pool(values[1:], 2)
    assert np.array_equal(out.to_numpy(), expected_out)
    assert np.array_equal(winners.to_numpy(), expected_win)
    out.close()
    winners.close()
    tail.close()
    owner.close()


# --------------------------------------------------------------------------
# Validation (all before any allocation)
# --------------------------------------------------------------------------

def test_closed_input_rejected():
    xi = _core(np.ones((1, 1, 4, 4)))
    xi.close()
    with pytest.raises(RuntimeError):
        xi.maxpool2d_forward(kernel_size=2)


@pytest.mark.parametrize("shape", [(4, 4), (1, 4, 4), (1, 1, 1, 4, 4)])
def test_rank_other_than_four_rejected(shape):
    xi = _core(np.ones(shape))
    with pytest.raises(ValueError, match="4-D NCHW"):
        xi.maxpool2d_forward(kernel_size=2)
    xi.close()


@pytest.mark.parametrize("kernel_size", [0, -1, (0, 2), (2, -3)])
def test_invalid_kernel_size_rejected(kernel_size):
    xi = _core(np.ones((1, 1, 4, 4)))
    with pytest.raises(ValueError):
        xi.maxpool2d_forward(kernel_size=kernel_size)
    xi.close()


@pytest.mark.parametrize("stride", [0, -2, (1, 0)])
def test_invalid_stride_rejected(stride):
    xi = _core(np.ones((1, 1, 4, 4)))
    with pytest.raises(ValueError):
        xi.maxpool2d_forward(kernel_size=2, stride=stride)
    xi.close()


@pytest.mark.parametrize("padding", [-1, (0, -1)])
def test_invalid_padding_rejected(padding):
    xi = _core(np.ones((1, 1, 4, 4)))
    with pytest.raises(ValueError):
        xi.maxpool2d_forward(kernel_size=2, padding=padding)
    xi.close()


@pytest.mark.parametrize(
    "kwargs",
    [
        {"kernel_size": True},
        {"kernel_size": 2, "stride": True},
        {"kernel_size": 2, "padding": False},
        {"kernel_size": (True, 2)},
    ],
)
def test_boolean_values_rejected(kwargs):
    xi = _core(np.ones((1, 1, 4, 4)))
    with pytest.raises(ValueError):
        xi.maxpool2d_forward(**kwargs)
    xi.close()


@pytest.mark.parametrize(
    "kernel_size", [(2,), (2, 2, 2), [1.5, 2], "2", None, (2, 1.5)]
)
def test_malformed_pairs_rejected(kernel_size):
    xi = _core(np.ones((1, 1, 4, 4)))
    with pytest.raises(ValueError):
        xi.maxpool2d_forward(kernel_size=kernel_size)
    xi.close()


def test_window_larger_than_padded_input_rejected():
    xi = _core(np.ones((1, 1, 3, 3)))
    with pytest.raises(ValueError, match="does not fit"):
        xi.maxpool2d_forward(kernel_size=5)
    xi.close()


def test_plane_beyond_float64_index_exactness_rejected():
    # A stride-0 view fabricates a logical plane of 2**54 elements over one
    # real element, so the exactness guard is provable without allocating
    # anything: it must fire before any output/winner storage or Policy-B
    # copy is attempted.
    storage = cpp.NativeStorage(4)
    view = cpp.NativeTensorView(
        storage, (1, 1, 2 ** 27, 2 ** 27), strides=(0, 0, 0, 0)
    )
    core = cpp.NativeTensorCore(storage, view)
    with pytest.raises(ValueError, match="index"):
        core.maxpool2d_forward(kernel_size=1)
    core.close()


def test_unsafe_allocation_product_rejected():
    # H*W stays exactly indexable, but N*C*out_h*out_w would not fit int64.
    storage = cpp.NativeStorage(4)
    view = cpp.NativeTensorView(
        storage, (2 ** 40, 2 ** 40, 2, 2), strides=(0, 0, 0, 0)
    )
    core = cpp.NativeTensorCore(storage, view)
    with pytest.raises(ValueError, match="int64"):
        core.maxpool2d_forward(kernel_size=1)
    core.close()


def test_non_float64_or_non_cpu_metadata_rejected(monkeypatch):
    # dtype/device are float64/cpu-only today, so a mismatch is not
    # constructible through the public API; force one to prove the guard
    # fires before any allocation.
    xi = _core(np.ones((1, 1, 4, 4)))
    monkeypatch.setattr(xi._storage, "_dtype", "float32")
    with pytest.raises(ValueError, match="float64"):
        xi.maxpool2d_forward(kernel_size=2)
    xi.close()


# --------------------------------------------------------------------------
# Raw C ABI failure behavior
#
# These call the exported symbol directly with explicit integer metadata.
# The ABI receives handles + the input offset + dimensions only — never
# stride arrays — so it validates that metadata and bounds-checks each
# contiguous span against its storage; it cannot and does not inspect
# logical contiguity (that is the Core wrapper's Policy-B responsibility).
# --------------------------------------------------------------------------

def _raw_call(lib, input_handle, input_offset, out_handle, win_handle, dims):
    lib.tf_core_maxpool2d_forward(
        input_handle, input_offset, out_handle, win_handle, *dims
    )


# N, C, H, W, kh, kw, sh, sw, ph, pw, out_h, out_w for a 4x4 / 2x2 pool.
_VALID_DIMS = (1, 1, 4, 4, 2, 2, 2, 2, 0, 0, 2, 2)


def test_raw_abi_null_handle_is_valueerror():
    lib = cpp._require_library()
    out = cpp.NativeTensorCore.zeros((1, 1, 2, 2))
    win = cpp.NativeTensorCore.zeros((1, 1, 2, 2))
    with pytest.raises(ValueError):
        _raw_call(lib, None, 0, out._storage._require_open(),
                  win._storage._require_open(), _VALID_DIMS)
    assert lib.tf_last_error_code() == cpp.TF_OK  # errcheck cleared it
    out.close()
    win.close()


def test_raw_abi_null_winner_handle_is_valueerror():
    lib = cpp._require_library()
    xi = _core(np.ones((1, 1, 4, 4)))
    out = cpp.NativeTensorCore.zeros((1, 1, 2, 2))
    with pytest.raises(ValueError):
        _raw_call(lib, xi._storage._require_open(), 0,
                  out._storage._require_open(), None, _VALID_DIMS)
    out.close()
    xi.close()


def test_raw_abi_negative_offset_is_valueerror():
    lib = cpp._require_library()
    xi = _core(np.ones((1, 1, 4, 4)))
    out = cpp.NativeTensorCore.zeros((1, 1, 2, 2))
    win = cpp.NativeTensorCore.zeros((1, 1, 2, 2))
    with pytest.raises(ValueError):
        _raw_call(lib, xi._storage._require_open(), -1,
                  out._storage._require_open(),
                  win._storage._require_open(), _VALID_DIMS)
    out.close()
    win.close()
    xi.close()


@pytest.mark.parametrize("index, value", [(0, 0), (1, 0), (4, 0), (6, -1)])
def test_raw_abi_invalid_dimension_is_valueerror(index, value):
    # batch, channels, kernel_height must be >= 1; stride_height too.
    lib = cpp._require_library()
    xi = _core(np.ones((1, 1, 4, 4)))
    out = cpp.NativeTensorCore.zeros((1, 1, 2, 2))
    win = cpp.NativeTensorCore.zeros((1, 1, 2, 2))
    dims = list(_VALID_DIMS)
    dims[index] = value
    with pytest.raises(ValueError):
        _raw_call(lib, xi._storage._require_open(), 0,
                  out._storage._require_open(),
                  win._storage._require_open(), tuple(dims))
    out.close()
    win.close()
    xi.close()


def test_raw_abi_negative_padding_is_valueerror():
    lib = cpp._require_library()
    xi = _core(np.ones((1, 1, 4, 4)))
    out = cpp.NativeTensorCore.zeros((1, 1, 2, 2))
    win = cpp.NativeTensorCore.zeros((1, 1, 2, 2))
    dims = list(_VALID_DIMS)
    dims[8] = -1  # pad_height
    with pytest.raises(ValueError):
        _raw_call(lib, xi._storage._require_open(), 0,
                  out._storage._require_open(),
                  win._storage._require_open(), tuple(dims))
    out.close()
    win.close()
    xi.close()


def test_raw_abi_output_shape_mismatch_is_valueerror():
    lib = cpp._require_library()
    xi = _core(np.ones((1, 1, 4, 4)))
    out = cpp.NativeTensorCore.zeros((1, 1, 3, 2))
    win = cpp.NativeTensorCore.zeros((1, 1, 3, 2))
    dims = list(_VALID_DIMS)
    dims[10] = 3  # claimed out_h, but the floor formula gives 2
    with pytest.raises(ValueError):
        _raw_call(lib, xi._storage._require_open(), 0,
                  out._storage._require_open(),
                  win._storage._require_open(), tuple(dims))
    out.close()
    win.close()
    xi.close()


def test_raw_abi_undersized_input_span_is_valueerror():
    lib = cpp._require_library()
    xi = _core(np.ones((1, 1, 4, 4)))  # storage holds 16 doubles
    out = cpp.NativeTensorCore.zeros((1, 1, 2, 3))
    win = cpp.NativeTensorCore.zeros((1, 1, 2, 3))
    # Claim W=6 (needs 24 > 16). out_w = (6-2)/2+1 = 3 agrees with the
    # formula, so only the storage-span guard can reject this.
    dims = (1, 1, 4, 6, 2, 2, 2, 2, 0, 0, 2, 3)
    with pytest.raises(ValueError):
        _raw_call(lib, xi._storage._require_open(), 0,
                  out._storage._require_open(),
                  win._storage._require_open(), dims)
    out.close()
    win.close()
    xi.close()


def test_raw_abi_undersized_output_span_is_valueerror():
    lib = cpp._require_library()
    xi = _core(np.ones((1, 1, 4, 4)))
    small = cpp.NativeTensorCore.zeros((1, 1, 1, 1))  # needs 4 doubles
    win = cpp.NativeTensorCore.zeros((1, 1, 2, 2))
    with pytest.raises(ValueError):
        _raw_call(lib, xi._storage._require_open(), 0,
                  small._storage._require_open(),
                  win._storage._require_open(), _VALID_DIMS)
    small.close()
    win.close()
    xi.close()


def test_raw_abi_undersized_winner_span_is_valueerror():
    lib = cpp._require_library()
    xi = _core(np.ones((1, 1, 4, 4)))
    out = cpp.NativeTensorCore.zeros((1, 1, 2, 2))
    small = cpp.NativeTensorCore.zeros((1, 1, 1, 1))
    with pytest.raises(ValueError):
        _raw_call(lib, xi._storage._require_open(), 0,
                  out._storage._require_open(),
                  small._storage._require_open(), _VALID_DIMS)
    out.close()
    small.close()
    xi.close()


def test_raw_abi_rejects_a_plane_beyond_float64_exactness():
    # The ABI re-proves the winner-exactness bound in its own fixed-width
    # arithmetic, so a direct caller cannot bypass the Python check.
    lib = cpp._require_library()
    xi = _core(np.ones((1, 1, 4, 4)))
    out = cpp.NativeTensorCore.zeros((1, 1, 1, 1))
    win = cpp.NativeTensorCore.zeros((1, 1, 1, 1))
    plane = 2 ** 27
    dims = (1, 1, plane, plane, plane, plane, 1, 1, 0, 0, 1, 1)
    with pytest.raises(ValueError):
        _raw_call(lib, xi._storage._require_open(), 0,
                  out._storage._require_open(),
                  win._storage._require_open(), dims)
    out.close()
    win.close()
    xi.close()


def test_no_stale_error_after_a_raw_failure():
    lib = cpp._require_library()
    out = cpp.NativeTensorCore.zeros((1, 1, 2, 2))
    win = cpp.NativeTensorCore.zeros((1, 1, 2, 2))
    with pytest.raises(ValueError):
        _raw_call(lib, None, 0, out._storage._require_open(),
                  win._storage._require_open(), _VALID_DIMS)
    out.close()
    win.close()
    # A later valid Core call must succeed — no contamination.
    xi = _core(np.arange(1, 17, dtype=float).reshape(1, 1, 4, 4))
    good = xi.maxpool2d_forward(kernel_size=2)
    assert good.to_numpy().tolist() == [[[[6.0, 8.0], [14.0, 16.0]]]]
    assert lib.tf_last_error_code() == cpp.TF_OK
    good.close()
    xi.close()


# --------------------------------------------------------------------------
# Failure atomicity
# --------------------------------------------------------------------------

@needs_fault_injection
def test_output_allocation_failure_is_atomic(live_storages):
    values = np.arange(1, 17, dtype=float).reshape(1, 1, 4, 4)
    xi = _core(values)
    baseline = len(live_storages)
    # A contiguous input needs no copy, so allocation #1 is the output.
    cpp._arm_alloc_failure(1)
    with pytest.raises(MemoryError):
        xi._maxpool2d_forward_with_winners(kernel_size=2)
    assert len(live_storages) == baseline          # nothing leaked
    assert np.array_equal(xi.to_numpy(), values)   # input untouched
    recovered = xi.maxpool2d_forward(kernel_size=2)  # and it still works
    assert recovered.to_numpy().tolist() == [[[[6.0, 8.0], [14.0, 16.0]]]]
    assert cpp._require_library().tf_last_error_code() == cpp.TF_OK
    recovered.close()
    xi.close()


@needs_fault_injection
def test_winner_allocation_failure_closes_the_output(live_storages):
    values = np.arange(1, 17, dtype=float).reshape(1, 1, 4, 4)
    xi = _core(values)
    baseline = len(live_storages)
    # Allocation #1 (the output) succeeds; #2 (the winner buffer) fails.
    cpp._arm_alloc_failure(2)
    with pytest.raises(MemoryError):
        xi._maxpool2d_forward_with_winners(kernel_size=2)
    # The already-allocated output was closed: the live count is back to
    # its baseline, so no partial result survived.
    assert len(live_storages) == baseline
    assert np.array_equal(xi.to_numpy(), values)
    recovered, recovered_winners = xi._maxpool2d_forward_with_winners(
        kernel_size=2
    )
    assert recovered.to_numpy().tolist() == [[[[6.0, 8.0], [14.0, 16.0]]]]
    assert cpp._require_library().tf_last_error_code() == cpp.TF_OK
    recovered.close()
    recovered_winners.close()
    xi.close()


def test_native_call_failure_closes_output_and_winners(monkeypatch,
                                                       live_storages):
    # Both allocations succeed and the native call then fails: every
    # allocated object must be closed and no partial result returned.
    lib = cpp._require_library()
    original = lib.tf_core_maxpool2d_forward

    def boom(*args):
        raise RuntimeError("simulated native pooling failure")

    monkeypatch.setattr(lib, "tf_core_maxpool2d_forward", boom)
    values = np.arange(1, 17, dtype=float).reshape(1, 1, 4, 4)
    xi = _core(values)
    baseline = len(live_storages)
    with pytest.raises(RuntimeError, match="simulated"):
        xi._maxpool2d_forward_with_winners(kernel_size=2)
    assert len(live_storages) == baseline
    assert np.array_equal(xi.to_numpy(), values)
    # Restoring the real symbol, a subsequent call succeeds.
    monkeypatch.setattr(lib, "tf_core_maxpool2d_forward", original)
    out = xi.maxpool2d_forward(kernel_size=2)
    assert out.to_numpy().tolist() == [[[[6.0, 8.0], [14.0, 16.0]]]]
    out.close()
    xi.close()


def test_temporary_copy_released_when_allocation_fails(monkeypatch,
                                                       live_storages):
    created = []
    original_copy = cpp.NativeTensorCore.contiguous_copy

    def tracking_copy(self):
        result = original_copy(self)
        created.append(result)
        return result

    def boom(*args, **kwargs):
        raise MemoryError("simulated output allocation failure")

    monkeypatch.setattr(cpp.NativeTensorCore, "contiguous_copy", tracking_copy)
    monkeypatch.setattr(cpp.NativeTensorCore, "zeros", staticmethod(boom))

    values = np.random.default_rng(9).standard_normal((1, 2, 4, 4))
    owner, view = _noncontiguous_input(values)
    baseline = len(live_storages)
    with pytest.raises(MemoryError):
        view._maxpool2d_forward_with_winners(kernel_size=2)
    assert created, "expected a Policy-B temporary to have been created"
    assert all(copy._closed for copy in created)
    assert len(live_storages) == baseline  # the copy's storage was freed too
    # The caller's view is still valid and unchanged.
    assert np.allclose(view.to_numpy(), values, atol=1e-12)
    owner.close()


# --------------------------------------------------------------------------
# Capability separation (registry honesty)
#
# TENSOR_CORE_OPS is the modern, accurate inventory of NativeTensorCore
# operations, so the D8 capability is advertised there as the
# layer-qualified `maxpool2d_forward`. `_CHECKED_KERNELS` is the ctypes
# error-hook registry (it also holds non-compute entries like
# tf_storage_create) — membership there is an error-contract property, NOT
# a capability advertisement. The differentiable op (D9) and the module
# (D10) remain unsupported.
# --------------------------------------------------------------------------

def test_core_forward_advertised_in_tensor_core_ops():
    assert "maxpool2d_forward" in cpp.TENSOR_CORE_OPS
    assert hasattr(cpp.NativeTensorCore, "maxpool2d_forward")
    assert "maxpool2d_forward" in cpp.backend_info()["tensor_core_ops"]


def test_raw_symbol_registered_in_the_error_contract_only():
    assert "tf_core_maxpool2d_forward" in cpp._CHECKED_KERNELS
    lib = cpp._require_library()
    assert callable(getattr(lib, "tf_core_maxpool2d_forward"))
    # Not advertised as a NumPy-buffer reference kernel or in the frozen
    # historical registry.
    assert "maxpool2d_forward" not in cpp.RAW_KERNELS
    assert "maxpool2d" not in cpp.RAW_KERNELS
    assert "maxpool2d_forward" not in cpp.TENSOR_CORE_KERNELS


def test_raw_kernels_and_frozen_registry_unchanged():
    assert cpp.RAW_KERNELS == (
        "elementwise_add", "elementwise_subtract", "elementwise_multiply",
        "elementwise_divide", "relu", "matmul", "matmul_tiled",
    )
    assert cpp.TENSOR_CORE_KERNELS == (
        "relu", "add", "subtract", "multiply", "matmul",
    )


def test_autograd_op_and_module_are_both_supported():
    from tensorforge.experimental import NativeTensor

    # The differentiable operation landed in D9 (this file's forward is its
    # first half) and the NativeMaxPool2d module in D10 — distinct layers,
    # both now present.
    assert "maxpool2d" in cpp.AUTOGRAD_OPS
    assert hasattr(NativeTensor, "maxpool2d")
    assert "maxpool2d" not in cpp.UNSUPPORTED
    assert "NativeMaxPool2d" in cpp.NATIVE_MODULES
    assert "NativeMaxPool2d" not in cpp.UNSUPPORTED
    import tensorforge.experimental as experimental

    assert "NativeMaxPool2d" in experimental.__all__


def test_forward_and_backward_are_separate_core_ops():
    # D8 shipped the forward Core op; D9 added the backward one beside it,
    # each backed by its own checked C ABI symbol.
    assert "maxpool2d_forward" in cpp.TENSOR_CORE_OPS
    assert "maxpool2d_backward" in cpp.TENSOR_CORE_OPS
    assert hasattr(cpp.NativeTensorCore, "maxpool2d_backward")
    assert "tf_core_maxpool2d_backward" in cpp._CHECKED_KERNELS
    lib = cpp._require_library()
    assert callable(getattr(lib, "tf_core_maxpool2d_backward"))


def test_conv2d_support_is_unaffected():
    assert "conv2d" in cpp.AUTOGRAD_OPS
    assert "conv2d_forward" in cpp.TENSOR_CORE_OPS
    assert "conv2d_input_backward" in cpp.TENSOR_CORE_OPS
    assert "conv2d_weight_backward" in cpp.TENSOR_CORE_OPS
    assert "NativeConv2d" in cpp.NATIVE_MODULES
    assert "NativeFlatten" in cpp.NATIVE_MODULES
