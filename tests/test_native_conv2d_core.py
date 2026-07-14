"""Forward-only native Conv2d at the ``NativeTensorCore`` layer (Phase D,
milestone D3).

D3 exposes the D2 internal cross-correlation kernel through the
exception-safe C ABI (``tf_core_conv2d_forward``), its ctypes/errcheck
registration, and ``NativeTensorCore.conv2d_forward`` — a forward-only,
autograd-unaware Core wrapper. These tests cover forward correctness and
stable-framework parity, the output ownership/layout contract, the
Policy-B copy-then-compute handling of non-contiguous operands, the full
validation surface, the C ABI failure/atomicity behavior, and the
capability-registry separation (Core forward implemented; the
differentiable ``conv2d`` autograd op and the ``NativeConv2d`` module stay
unsupported).

Backend-dependent, so the module skips cleanly when the compiled backend
is not built. Cleanup is explicit via close(); nothing depends on
garbage-collection timing.

Selector: python -m pytest -q -k native_conv2d_core
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
    """Leave the injection hook disarmed and the error slot clear after
    every test, so an armed countdown never leaks into the next one."""
    yield
    if cpp.is_available():
        cpp._arm_alloc_failure(0)
        cpp._require_library().tf_clear_error()


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

def _core(values):
    """A contiguous NativeTensorCore holding a copy of ``values``."""
    return cpp.NativeTensorCore.from_array(np.asarray(values, dtype=np.float64))


def _stable_reference(x, weight, bias, stride, padding):
    """The stable ``tensorforge.nn.Conv2d`` forward on the same data — the
    numerical reference the native line matches to tolerance."""
    from tensorforge.nn import Conv2d
    from tensorforge.tensor import Tensor

    out_channels, in_channels, kh, kw = weight.shape
    layer = Conv2d(
        in_channels, out_channels, (kh, kw),
        stride=stride, padding=padding, bias=bias is not None,
    )
    layer.weight.data = np.array(weight, dtype=np.float64)
    if bias is not None:
        layer.bias.data = np.array(bias, dtype=np.float64)
    return layer(Tensor(np.asarray(x, dtype=np.float64))).data


def _run(x, weight, bias=None, stride=1, padding=0):
    """conv2d_forward over fresh cores; returns the NumPy output. Closes
    the operand cores it created (not the returned array)."""
    xi = _core(x)
    wi = _core(weight)
    bi = _core(bias) if bias is not None else None
    try:
        out = xi.conv2d_forward(wi, bi, stride=stride, padding=padding)
        result = out.to_numpy()
        out.close()
        return result
    finally:
        xi.close()
        wi.close()
        if bi is not None:
            bi.close()


# --------------------------------------------------------------------------
# Forward correctness
# --------------------------------------------------------------------------

def test_single_channel_hand_computed():
    # 1x1x3x3 input, 2x2 identity-diagonal kernel, no bias, stride 1, pad 0.
    x = [[[[1, 2, 3], [4, 5, 6], [7, 8, 9]]]]
    w = [[[[1, 0], [0, 1]]]]
    out = _run(x, w)
    assert out.tolist() == [[[[6.0, 8.0], [12.0, 14.0]]]]


def test_bias_added_once():
    x = [[[[1, 2, 3], [4, 5, 6], [7, 8, 9]]]]
    w = [[[[1, 0], [0, 1]]]]
    out = _run(x, w, bias=[10.0])
    assert out.tolist() == [[[[16.0, 18.0], [22.0, 24.0]]]]


def test_no_bias_matches_zero_bias():
    x = [[[[1, 2, 3], [4, 5, 6], [7, 8, 9]]]]
    w = [[[[1, 0], [0, 1]]]]
    assert _run(x, w, bias=None).tolist() == _run(x, w, bias=[0.0]).tolist()


def test_multiple_input_channels_accumulate():
    # N=1, C=2, 2x2 each; 1x1 kernel per channel weight (1, 2): out = c0 + 2*c1.
    x = [[[[1, 2], [3, 4]], [[10, 20], [30, 40]]]]
    w = [[[[1]], [[2]]]]  # O=1, C=2, 1x1
    out = _run(x, w)
    assert out.tolist() == [[[[21.0, 42.0], [63.0, 84.0]]]]


def test_multiple_output_channels():
    x = [[[[1, 2], [3, 4]]]]  # N=1, C=1, 2x2
    w = [[[[2]]], [[[-1]]]]   # O=2, C=1, 1x1: ch0=2x, ch1=-x
    out = _run(x, w, bias=[0.5, -0.5])
    assert out.tolist() == [[
        [[2.5, 4.5], [6.5, 8.5]],
        [[-1.5, -2.5], [-3.5, -4.5]],
    ]]


def test_batch_greater_than_one():
    x = [[[[1, 2], [3, 4]]], [[[5, 6], [7, 8]]]]  # N=2
    w = [[[[2]]]]
    out = _run(x, w)
    assert out.tolist() == [
        [[[2.0, 4.0], [6.0, 8.0]]],
        [[[10.0, 12.0], [14.0, 16.0]]],
    ]


def test_symmetric_padding():
    x = [[[[1, 2, 3], [4, 5, 6], [7, 8, 9]]]]
    w = [[[[1, 1, 1], [1, 1, 1], [1, 1, 1]]]]  # 3x3 all-ones
    out = _run(x, w, padding=1)
    assert out.tolist() == [[[[12.0, 21.0, 16.0],
                              [27.0, 45.0, 33.0],
                              [24.0, 39.0, 28.0]]]]


def test_stride_greater_than_one():
    x = [[[list(range(1, 5)), list(range(5, 9)),
           list(range(9, 13)), list(range(13, 17))]]]  # 1x1x4x4
    w = [[[[1, 1], [1, 1]]]]
    out = _run(x, w, stride=2)
    assert out.tolist() == [[[[14.0, 22.0], [46.0, 54.0]]]]


def test_rectangular_input():
    x = [[[[1, 2, 3], [4, 5, 6]]]]  # 1x1x2x3
    w = [[[[1, 1], [1, 1]]]]
    out = _run(x, w)  # out_h=1, out_w=2
    assert out.tolist() == [[[[12.0, 16.0]]]]


def test_rectangular_kernel():
    x = [[[[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]]]]  # 1x1x3x4
    w = [[[[1, 1, 1, 1], [1, 1, 1, 1]]]]  # 2x4 kernel
    got = _run(x, w)
    ref = _stable_reference(x, np.array(w, dtype=float), None, 1, 0)
    assert np.allclose(got, ref, atol=1e-9)
    assert got.shape == (1, 1, 2, 1)


def test_tuple_stride():
    x = np.arange(1, 65, dtype=float).reshape(1, 1, 8, 8)
    w = np.ones((1, 1, 3, 3))
    got = _run(x, w, stride=(2, 1))
    ref = _stable_reference(x, w, None, (2, 1), 0)
    assert np.allclose(got, ref, atol=1e-9)


def test_tuple_padding():
    x = np.arange(1, 65, dtype=float).reshape(1, 1, 8, 8)
    w = np.ones((1, 1, 3, 3))
    got = _run(x, w, padding=(1, 0))
    ref = _stable_reference(x, w, None, 1, (1, 0))
    assert np.allclose(got, ref, atol=1e-9)


def test_combined_stride_and_padding():
    x = [[[[1, 2, 3], [4, 5, 6], [7, 8, 9]]]]
    w = [[[[1, 1], [1, 1]]]]
    out = _run(x, w, stride=2, padding=1)
    assert out.tolist() == [[[[1.0, 5.0], [11.0, 28.0]]]]


def test_negative_and_fractional_values():
    x = [[[[-1.5, 2.5], [0.5, -3.0]]]]
    w = [[[[2.0]]]]
    out = _run(x, w, bias=[-0.5])
    assert out.tolist() == [[[[-3.5, 4.5], [0.5, -6.5]]]]


def test_stable_framework_parity_full():
    rng = np.random.default_rng(7)
    x = rng.standard_normal((2, 3, 5, 4))
    w = rng.standard_normal((4, 3, 3, 2))
    b = rng.standard_normal(4)
    got = _run(x, w, bias=b, stride=(2, 1), padding=(1, 0))
    ref = _stable_reference(x, w, b, (2, 1), (1, 0))
    assert got.shape == ref.shape
    assert np.allclose(got, ref, atol=1e-9)


def test_deterministic_repeated_execution():
    rng = np.random.default_rng(3)
    x = rng.standard_normal((2, 2, 5, 5))
    w = rng.standard_normal((3, 2, 3, 3))
    b = rng.standard_normal(3)
    first = _run(x, w, bias=b, stride=1, padding=1)
    second = _run(x, w, bias=b, stride=1, padding=1)
    assert np.array_equal(first, second)  # bit-identical


# --------------------------------------------------------------------------
# Output contract
# --------------------------------------------------------------------------

def test_output_shape_and_canonical_strides():
    x = np.arange(2 * 3 * 7 * 6, dtype=float).reshape(2, 3, 7, 6)
    w = np.ones((4, 3, 3, 2))
    xi, wi = _core(x), _core(w)
    out = xi.conv2d_forward(wi, None, stride=(2, 1), padding=(1, 0))
    assert out.shape == (2, 4, 4, 5)
    assert out.strides == cpp.row_major_strides(out.shape)
    assert out.contiguous is True
    assert out.offset == 0
    out.close()
    xi.close()
    wi.close()


def test_output_ownership_and_metadata():
    xi, wi = _core(np.ones((1, 1, 3, 3))), _core(np.ones((1, 1, 2, 2)))
    out = xi.conv2d_forward(wi)
    assert out._owns_storage is True
    assert out.dtype == "float64"
    assert out.device == "cpu"
    # A fresh owning result, not an alias of any operand's storage.
    assert out._storage is not xi._storage
    assert out._storage is not wi._storage
    out.close()
    xi.close()
    wi.close()


def test_output_valid_after_inputs_closed():
    x = np.arange(1, 10, dtype=float).reshape(1, 1, 3, 3)
    w = np.array([[[[1.0, 0.0], [0.0, 1.0]]]])
    xi, wi, bi = _core(x), _core(w), _core([5.0])
    out = xi.conv2d_forward(wi, bi)
    expected = out.to_numpy().copy()
    # Closing every input must not disturb the owning output.
    xi.close()
    wi.close()
    bi.close()
    assert np.array_equal(out.to_numpy(), expected)
    out.close()


def test_inputs_unchanged_by_forward():
    x = np.arange(1, 10, dtype=float).reshape(1, 1, 3, 3)
    w = np.array([[[[1.0, 0.5], [0.25, 2.0]]]])
    b = np.array([1.5])
    xi, wi, bi = _core(x), _core(w), _core(b)
    xi.conv2d_forward(wi, bi, padding=1).close()
    assert np.array_equal(xi.to_numpy(), x)
    assert np.array_equal(wi.to_numpy(), w)
    assert np.array_equal(bi.to_numpy(), b)
    xi.close()
    wi.close()
    bi.close()


# --------------------------------------------------------------------------
# Non-contiguous inputs (Policy B: copy-then-compute)
# --------------------------------------------------------------------------

def _noncontiguous_like(array):
    """A non-contiguous core whose logical value equals ``array`` (NCHW),
    built by transposing the last two axes of a contiguous base."""
    array = np.asarray(array, dtype=np.float64)
    base = cpp.NativeTensorCore.from_array(
        np.ascontiguousarray(array.transpose(0, 1, 3, 2))
    )
    view = base.transpose(0, 1, 3, 2)
    assert not view.contiguous
    return base, view


def _noncontiguous_weight(array):
    """A non-contiguous OIHW weight view equal to ``array``."""
    array = np.asarray(array, dtype=np.float64)
    base = cpp.NativeTensorCore.from_array(
        np.ascontiguousarray(array.transpose(0, 1, 3, 2))
    )
    view = base.transpose(0, 1, 3, 2)
    assert not view.contiguous
    return base, view


def _noncontiguous_bias(values):
    """A rank-1 non-contiguous bias view (stride 2) equal to ``values``,
    built through the documented low-level view constructor."""
    values = np.asarray(values, dtype=np.float64)
    o = values.shape[0]
    raw = np.zeros(2 * o, dtype=np.float64)
    raw[0::2] = values
    storage = cpp.NativeStorage.from_array(raw)
    view = cpp.NativeTensorView(storage, (o,), strides=(2,), offset=0)
    core = cpp.NativeTensorCore(storage, view, owns_storage=True)
    assert not core.contiguous
    return core


def test_non_contiguous_input_parity():
    x = np.random.default_rng(1).standard_normal((1, 2, 4, 4))
    w = np.ones((3, 2, 2, 2))
    base, xv = _noncontiguous_like(x)
    wi = _core(w)
    out = xv.conv2d_forward(wi, stride=1, padding=1)
    ref = _stable_reference(x, w, None, 1, 1)
    assert np.allclose(out.to_numpy(), ref, atol=1e-9)
    # The caller's view/storage are untouched and still usable.
    assert np.allclose(xv.to_numpy(), x, atol=1e-12)
    out.close()
    base.close()
    wi.close()


def test_non_contiguous_weight_parity():
    x = np.random.default_rng(2).standard_normal((1, 2, 5, 5))
    w = np.random.default_rng(9).standard_normal((3, 2, 3, 3))
    xi = _core(x)
    base, wv = _noncontiguous_weight(w)
    out = xi.conv2d_forward(wv, padding=1)
    ref = _stable_reference(x, w, None, 1, 1)
    assert np.allclose(out.to_numpy(), ref, atol=1e-9)
    assert np.allclose(wv.to_numpy(), w, atol=1e-12)
    out.close()
    xi.close()
    base.close()


def test_non_contiguous_bias_parity():
    x = np.random.default_rng(4).standard_normal((1, 1, 4, 4))
    w = np.random.default_rng(5).standard_normal((2, 1, 2, 2))
    b = np.array([1.25, -3.5])
    xi, wi = _core(x), _core(w)
    bv = _noncontiguous_bias(b)
    out = xi.conv2d_forward(wi, bv)
    ref = _stable_reference(x, w, b, 1, 0)
    assert np.allclose(out.to_numpy(), ref, atol=1e-9)
    assert np.allclose(bv.to_numpy(), b, atol=1e-12)
    out.close()
    xi.close()
    wi.close()
    bv.close()


def test_multiple_non_contiguous_operands_together():
    x = np.random.default_rng(11).standard_normal((2, 2, 5, 4))
    w = np.random.default_rng(12).standard_normal((3, 2, 3, 2))
    b = np.array([0.5, -1.0, 2.0])
    base_x, xv = _noncontiguous_like(x)
    base_w, wv = _noncontiguous_weight(w)
    bv = _noncontiguous_bias(b)
    out = xv.conv2d_forward(wv, bv, stride=(2, 1), padding=(1, 0))
    ref = _stable_reference(x, w, b, (2, 1), (1, 0))
    assert np.allclose(out.to_numpy(), ref, atol=1e-9)
    # Parity with the explicitly contiguous copies of every operand.
    xi, wi, bi = _core(x), _core(w), _core(b)
    contig = xi.conv2d_forward(wi, bi, stride=(2, 1), padding=(1, 0))
    assert np.allclose(out.to_numpy(), contig.to_numpy(), atol=1e-12)
    for t in (out, contig, base_x, base_w, bv, xi, wi, bi):
        t.close()


def test_non_contiguous_copies_are_closed(monkeypatch):
    # Every Policy-B temporary copy must be closed after the call — no leak.
    created = []
    original = cpp.NativeTensorCore.contiguous_copy

    def tracking_copy(self):
        result = original(self)
        created.append(result)
        return result

    monkeypatch.setattr(cpp.NativeTensorCore, "contiguous_copy", tracking_copy)

    x = np.random.default_rng(6).standard_normal((1, 2, 4, 4))
    w = np.random.default_rng(7).standard_normal((2, 2, 2, 2))
    b = np.array([1.0, 2.0])
    base_x, xv = _noncontiguous_like(x)
    base_w, wv = _noncontiguous_weight(w)
    bv = _noncontiguous_bias(b)
    out = xv.conv2d_forward(wv, bv, padding=1)
    assert len(created) == 3  # input, weight, bias each copied once
    assert all(copy._closed for copy in created)  # all temporaries closed
    for t in (out, base_x, base_w, bv):
        t.close()


def test_contiguous_operands_are_not_copied(monkeypatch):
    # An already-contiguous operand must be passed through without a copy.
    calls = []
    original = cpp.NativeTensorCore.contiguous_copy

    def counting_copy(self):
        calls.append(self)
        return original(self)

    monkeypatch.setattr(cpp.NativeTensorCore, "contiguous_copy", counting_copy)
    xi, wi, bi = _core(np.ones((1, 1, 3, 3))), _core(np.ones((1, 1, 2, 2))), _core([1.0])
    xi.conv2d_forward(wi, bi).close()
    assert calls == []  # no unnecessary copies
    xi.close()
    wi.close()
    bi.close()


# --------------------------------------------------------------------------
# Validation
# --------------------------------------------------------------------------

def test_non_core_weight_rejected():
    xi = _core(np.ones((1, 1, 3, 3)))
    with pytest.raises(TypeError):
        xi.conv2d_forward(np.ones((1, 1, 2, 2)))
    xi.close()


def test_non_core_bias_rejected():
    xi, wi = _core(np.ones((1, 1, 3, 3))), _core(np.ones((1, 1, 2, 2)))
    with pytest.raises(TypeError):
        xi.conv2d_forward(wi, [1.0])
    xi.close()
    wi.close()


def test_closed_input_rejected():
    xi, wi = _core(np.ones((1, 1, 3, 3))), _core(np.ones((1, 1, 2, 2)))
    xi.close()
    with pytest.raises(RuntimeError):
        xi.conv2d_forward(wi)
    wi.close()


def test_closed_weight_rejected():
    xi, wi = _core(np.ones((1, 1, 3, 3))), _core(np.ones((1, 1, 2, 2)))
    wi.close()
    with pytest.raises(RuntimeError):
        xi.conv2d_forward(wi)
    xi.close()


def test_closed_bias_rejected():
    xi, wi, bi = _core(np.ones((1, 1, 3, 3))), _core(np.ones((1, 1, 2, 2))), _core([1.0])
    bi.close()
    with pytest.raises(RuntimeError):
        xi.conv2d_forward(wi, bi)
    xi.close()
    wi.close()


def test_input_rank_not_4_rejected():
    xi, wi = _core(np.ones((3, 3))), _core(np.ones((1, 1, 2, 2)))
    with pytest.raises(ValueError):
        xi.conv2d_forward(wi)
    xi.close()
    wi.close()


def test_weight_rank_not_4_rejected():
    xi, wi = _core(np.ones((1, 1, 3, 3))), _core(np.ones((1, 2, 2)))
    with pytest.raises(ValueError):
        xi.conv2d_forward(wi)
    xi.close()
    wi.close()


def test_bias_rank_not_1_rejected():
    xi, wi, bi = _core(np.ones((1, 1, 3, 3))), _core(np.ones((1, 1, 2, 2))), _core([[1.0]])
    with pytest.raises(ValueError):
        xi.conv2d_forward(wi, bi)
    xi.close()
    wi.close()
    bi.close()


def test_channel_mismatch_rejected():
    xi, wi = _core(np.ones((1, 2, 3, 3))), _core(np.ones((1, 3, 2, 2)))
    with pytest.raises(ValueError):
        xi.conv2d_forward(wi)
    xi.close()
    wi.close()


def test_bias_length_mismatch_rejected():
    xi, wi, bi = _core(np.ones((1, 1, 3, 3))), _core(np.ones((2, 1, 2, 2))), _core([1.0, 2.0, 3.0])
    with pytest.raises(ValueError):
        xi.conv2d_forward(wi, bi)
    xi.close()
    wi.close()
    bi.close()


def test_kernel_larger_than_input_rejected():
    xi, wi = _core(np.ones((1, 1, 3, 3))), _core(np.ones((1, 1, 5, 5)))
    with pytest.raises(ValueError):
        xi.conv2d_forward(wi)
    xi.close()
    wi.close()


@pytest.mark.parametrize("stride", [0, -1])
def test_zero_or_negative_stride_rejected(stride):
    xi, wi = _core(np.ones((1, 1, 3, 3))), _core(np.ones((1, 1, 2, 2)))
    with pytest.raises(ValueError):
        xi.conv2d_forward(wi, stride=stride)
    xi.close()
    wi.close()


def test_negative_padding_rejected():
    xi, wi = _core(np.ones((1, 1, 3, 3))), _core(np.ones((1, 1, 2, 2)))
    with pytest.raises(ValueError):
        xi.conv2d_forward(wi, padding=-1)
    xi.close()
    wi.close()


@pytest.mark.parametrize("kwargs", [{"stride": True}, {"padding": False}])
def test_boolean_stride_or_padding_rejected(kwargs):
    xi, wi = _core(np.ones((1, 1, 3, 3))), _core(np.ones((1, 1, 2, 2)))
    with pytest.raises(ValueError):
        xi.conv2d_forward(wi, **kwargs)
    xi.close()
    wi.close()


@pytest.mark.parametrize("bad", [(1,), (1, 2, 3)])
def test_invalid_tuple_length_rejected(bad):
    xi, wi = _core(np.ones((1, 1, 3, 3))), _core(np.ones((1, 1, 2, 2)))
    with pytest.raises(ValueError):
        xi.conv2d_forward(wi, stride=bad)
    xi.close()
    wi.close()


def test_invalid_tuple_member_type_rejected():
    xi, wi = _core(np.ones((1, 1, 3, 3))), _core(np.ones((1, 1, 2, 2)))
    with pytest.raises(ValueError):
        xi.conv2d_forward(wi, padding=(1, 1.5))
    xi.close()
    wi.close()


def test_metadata_mismatch_rejected(monkeypatch):
    # dtype/device are float64/cpu-only today, so a mismatch is not
    # constructible through the public API; force one to prove the guard
    # fires before any allocation.
    xi, wi = _core(np.ones((1, 1, 3, 3))), _core(np.ones((1, 1, 2, 2)))
    monkeypatch.setattr(wi._storage, "_dtype", "float32")
    with pytest.raises(ValueError):
        xi.conv2d_forward(wi)
    xi.close()
    wi.close()


# --------------------------------------------------------------------------
# Raw C ABI failure behavior
#
# These call the exported symbol directly with explicit integer metadata.
# The ABI receives handles + offsets + dimensions only — never stride
# arrays — so it validates that metadata and bounds-checks each contiguous
# span against its storage; it cannot and does not inspect logical
# contiguity (that is the Core wrapper's Policy-B responsibility). None of
# these tests assert the ABI recognizes a non-contiguous stride pattern,
# because no such pattern is ever supplied to it.
# --------------------------------------------------------------------------

def test_raw_abi_null_handle_is_valueerror():
    lib = cpp._require_library()
    wi = _core(np.ones((1, 1, 2, 2)))
    out = cpp.NativeTensorCore.zeros((1, 1, 2, 2))
    with pytest.raises(ValueError):
        lib.tf_core_conv2d_forward(
            None, 0,                                  # null input handle
            wi._storage._require_open(), 0,
            None, 0,
            out._storage._require_open(),
            1, 1, 3, 3, 1, 2, 2, 1, 1, 0, 0, 2, 2,
        )
    assert lib.tf_last_error_code() == cpp.TF_OK  # errcheck cleared it
    out.close()
    wi.close()


def test_raw_abi_output_dim_mismatch_is_valueerror():
    lib = cpp._require_library()
    xi = _core(np.ones((1, 1, 3, 3)))
    wi = _core(np.ones((1, 1, 2, 2)))
    out = cpp.NativeTensorCore.zeros((1, 1, 2, 2))
    with pytest.raises(ValueError):
        # Correct dims except a wrong claimed out_h (3, not the floor 2).
        lib.tf_core_conv2d_forward(
            xi._storage._require_open(), 0,
            wi._storage._require_open(), 0,
            None, 0,
            out._storage._require_open(),
            1, 1, 3, 3, 1, 2, 2, 1, 1, 0, 0, 3, 2,
        )
    out.close()
    xi.close()
    wi.close()


def test_raw_abi_input_span_exceeds_storage_is_valueerror():
    lib = cpp._require_library()
    xi = _core(np.ones((1, 1, 3, 3)))  # storage holds 9 doubles
    wi = _core(np.ones((1, 1, 2, 2)))
    out = cpp.NativeTensorCore.zeros((1, 1, 3, 3))
    with pytest.raises(ValueError):
        # Claim W=4 (needs 1*1*3*4 = 12 > 9). out_w = (4-2)/1+1 = 3 matches
        # the formula, so only the storage-span guard can reject this.
        lib.tf_core_conv2d_forward(
            xi._storage._require_open(), 0,
            wi._storage._require_open(), 0,
            None, 0,
            out._storage._require_open(),
            1, 1, 3, 4, 1, 2, 2, 1, 1, 0, 0, 2, 3,
        )
    out.close()
    xi.close()
    wi.close()


def test_raw_abi_invalid_dimension_is_valueerror():
    lib = cpp._require_library()
    xi = _core(np.ones((1, 1, 3, 3)))
    wi = _core(np.ones((1, 1, 2, 2)))
    out = cpp.NativeTensorCore.zeros((1, 1, 2, 2))
    with pytest.raises(ValueError):
        # batch = 0 is a non-positive extent -> TF_ERROR_INVALID.
        lib.tf_core_conv2d_forward(
            xi._storage._require_open(), 0,
            wi._storage._require_open(), 0,
            None, 0,
            out._storage._require_open(),
            0, 1, 3, 3, 1, 2, 2, 1, 1, 0, 0, 2, 2,
        )
    out.close()
    xi.close()
    wi.close()


def test_raw_abi_negative_offset_is_valueerror():
    lib = cpp._require_library()
    xi = _core(np.ones((1, 1, 3, 3)))
    wi = _core(np.ones((1, 1, 2, 2)))
    out = cpp.NativeTensorCore.zeros((1, 1, 2, 2))
    with pytest.raises(ValueError):
        # A negative input offset is rejected before any read.
        lib.tf_core_conv2d_forward(
            xi._storage._require_open(), -1,
            wi._storage._require_open(), 0,
            None, 0,
            out._storage._require_open(),
            1, 1, 3, 3, 1, 2, 2, 1, 1, 0, 0, 2, 2,
        )
    out.close()
    xi.close()
    wi.close()


def test_no_stale_error_after_raw_failure():
    lib = cpp._require_library()
    wi = _core(np.ones((1, 1, 2, 2)))
    out = cpp.NativeTensorCore.zeros((1, 1, 2, 2))
    with pytest.raises(ValueError):
        lib.tf_core_conv2d_forward(
            None, 0, wi._storage._require_open(), 0, None, 0,
            out._storage._require_open(),
            1, 1, 3, 3, 1, 2, 2, 1, 1, 0, 0, 2, 2,
        )
    out.close()
    # A later valid Core call must succeed — no contamination.
    xi = _core(np.ones((1, 1, 3, 3)))
    good = xi.conv2d_forward(wi)
    assert good.shape == (1, 1, 2, 2)
    good.close()
    xi.close()
    wi.close()


@needs_fault_injection
def test_allocation_failure_raises_memoryerror_and_is_atomic():
    x = np.arange(1, 10, dtype=float).reshape(1, 1, 3, 3)
    w = np.array([[[[1.0, 0.0], [0.0, 1.0]]]])
    xi, wi = _core(x), _core(w)
    # Contiguous operands -> no temp copy; the first (and only) allocation is
    # the output zeros, so nth=1 targets it deterministically.
    cpp._arm_alloc_failure(1)
    with pytest.raises(MemoryError):
        xi.conv2d_forward(wi)
    # Inputs untouched; a subsequent call succeeds (no leak, no stale error).
    assert np.array_equal(xi.to_numpy(), x)
    assert np.array_equal(wi.to_numpy(), w)
    recovered = xi.conv2d_forward(wi)
    assert recovered.shape == (1, 1, 2, 2)
    assert cpp._require_library().tf_last_error_code() == cpp.TF_OK
    recovered.close()
    xi.close()
    wi.close()


def test_temporary_copies_closed_when_output_alloc_fails(monkeypatch):
    # If output allocation fails after Policy-B copies were made, every
    # temporary must still be closed (failure atomicity, no leak).
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

    x = np.random.default_rng(8).standard_normal((1, 2, 4, 4))
    w = np.random.default_rng(9).standard_normal((2, 2, 2, 2))
    base_x, xv = _noncontiguous_like(x)
    base_w, wv = _noncontiguous_weight(w)
    with pytest.raises(MemoryError):
        xv.conv2d_forward(wv)
    assert created, "expected Policy-B temporaries to have been created"
    assert all(copy._closed for copy in created)  # all closed despite failure
    # Originals still valid and unchanged.
    assert np.allclose(xv.to_numpy(), x, atol=1e-12)
    base_x.close()
    base_w.close()


# --------------------------------------------------------------------------
# Capability separation (registry honesty)
#
# The modern, accurate capability inventory for native compute operations
# is TENSOR_CORE_OPS ("NativeTensorCore operations, each backed by a C ABI
# kernel"), so the forward Conv2d capability is advertised there as
# `conv2d_forward`. The raw C ABI symbol `tf_core_conv2d_forward` has no
# separate public inventory: RAW_KERNELS is the NumPy-buffer reference set
# and TENSOR_CORE_KERNELS is the intentionally frozen historical registry,
# so the forward kernel must NOT appear in either. `_CHECKED_KERNELS` is the
# ctypes error-hook registry (it also holds non-compute entries like
# tf_storage_create) — membership there is an error-contract property, NOT a
# capability advertisement.
# --------------------------------------------------------------------------

def test_core_forward_advertised_in_tensor_core_ops():
    # The Core forward operation (backed by the C ABI kernel) is advertised
    # in the modern op inventory and is a real NativeTensorCore method.
    assert "conv2d_forward" in cpp.TENSOR_CORE_OPS
    assert hasattr(cpp.NativeTensorCore, "conv2d_forward")


def test_forward_not_advertised_as_a_numpy_or_legacy_kernel():
    # It is neither a NumPy-buffer reference kernel nor a frozen legacy
    # tensor-core kernel, so it must stay out of both inventories.
    assert "conv2d_forward" not in cpp.RAW_KERNELS
    assert "conv2d" not in cpp.RAW_KERNELS
    assert "conv2d_forward" not in cpp.TENSOR_CORE_KERNELS
    assert "conv2d" not in cpp.TENSOR_CORE_KERNELS


def test_raw_symbol_registered_in_error_contract_only():
    # `_CHECKED_KERNELS` is the error-hook registry, not the capability
    # inventory: the forward symbol participates in the error contract and
    # is loadable via ctypes, which is a correctness property of the raw
    # kernel, distinct from how the capability is advertised (TENSOR_CORE_OPS
    # above).
    assert "tf_core_conv2d_forward" in cpp._CHECKED_KERNELS
    lib = cpp._require_library()
    assert callable(getattr(lib, "tf_core_conv2d_forward"))


def test_conv2d_autograd_and_module_remain_unsupported():
    # The differentiable op and the module are later milestones. "conv2d"
    # (the public/differentiable operation) stays unsupported and out of the
    # op/autograd inventories; only the layer-qualified "conv2d_forward"
    # Core op exists.
    assert "conv2d" not in cpp.AUTOGRAD_OPS
    assert "conv2d" not in cpp.TENSOR_CORE_OPS
    assert "conv2d" in cpp.UNSUPPORTED
    assert "NativeConv2d" not in cpp.NATIVE_MODULES
    import tensorforge.experimental as experimental

    assert "NativeConv2d" not in experimental.__all__


def test_conv2d_backward_and_maxpool2d_remain_unsupported():
    # No backward kernel and no pooling are exposed by D3, at any layer.
    assert "maxpool2d" in cpp.UNSUPPORTED
    assert "maxpool2d" not in cpp.TENSOR_CORE_OPS
    assert "maxpool2d" not in cpp.AUTOGRAD_OPS
    for name in cpp.TENSOR_CORE_OPS + cpp.AUTOGRAD_OPS:
        assert "backward" not in name  # no backward op advertised as a capability
    assert "NativeMaxPool2d" not in cpp.NATIVE_MODULES


def test_native_flatten_remains_supported():
    assert "NativeFlatten" in cpp.NATIVE_MODULES
    import tensorforge.experimental as experimental

    assert "NativeFlatten" in experimental.__all__
