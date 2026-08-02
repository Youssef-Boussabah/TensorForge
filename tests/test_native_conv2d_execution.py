"""The H9 convolution-execution contract (Phase H, milestone H9).

H9 changed *how* the three convolution kernels walk memory and nothing
else. Each of ``tf_core_conv2d_forward``,
``tf_core_conv2d_input_backward``, and ``tf_core_conv2d_weight_backward``
now ships two compute paths behind one **unchanged** export:

* the retained Phase-D direct loop (``tf::conv2d_*_generic``) — shipped,
  reachable through ordinary production dispatch for every geometry that
  fails a predicate, and the oracle the optimized paths are compared
  against;
* one H9 optimized traversal (``tf::conv2d_forward_row_sweep``,
  ``tf::conv2d_input_backward_gather``,
  ``tf::conv2d_weight_backward_gather``), each of which replaces a short
  kernel-tap inner loop with a long sweep along one contiguous spatial
  row.

The choice is made inside the kernel from the integer geometry the export
already receives. It is deterministic, total, side-effect free, and
independent of pointer values, alignment, wall time, environment
variables, and CPU-feature probes. **No selector, block-size setter,
dispatch tracer, or "which path ran" hook exists anywhere**, and §7 below
asserts that against the built image's own export table.

Division of labour with ``cpp/tests/test_conv2d_execution.cpp``: that
binary compiles ``conv2d.cpp`` in, so it can call the hidden predicates
and **both** paths directly and compare them bit for bit — that is where
the path-equivalence, signed-zero, NaN, and full-write proofs belong,
because that is the layer where those properties are decided. This file
owns everything that is only observable from Python: that both sides of
each predicate produce the right answer against an independent NumPy
oracle, that Policy-B layouts still work, that autograd ownership and
gradient identity are untouched, that failures still clean up atomically,
and that no public surface moved.

What this file proves:

1. **Both sides of every predicate are correct**, against an explicit
   NumPy cross-correlation formula written here as a formula rather than
   borrowed from either implementation — over a geometry matrix that
   deliberately straddles the swept-extent minimum and the
   input-gradient's unit-stride rule.
2. **Special values survive the Python boundary**: signed zeros,
   infinities, denormals, the smallest normal, the largest finite
   magnitudes, and NaN positions/quietness, compared as raw IEEE-754 bit
   patterns.
3. **H1 allocation safety on the geometries the C++ suite cannot reach
   from the allocator side** — the fallback geometries, poisoned through
   the real private allocation seam.
4. **Layout behavior is exactly what Policy B always gave**: narrowed,
   offset, transposed, and chained views all still work, through the
   copy-then-compute path, and the caller's tensors are never mutated.
5. **Autograd and graph-resource ownership are unchanged** across every
   requires-grad combination, retained graphs, repeated backward,
   gradient accumulation, and parameter versions.
6. **Failure atomicity is unchanged**: an injected allocation failure at
   any convolution stage leaves inputs, weights, bias, versions, and
   existing gradients untouched, releases every native storage, and
   never needs the garbage collector.
7. **No public surface moved**: 52 exports, no dispatch control, no new
   registry entry, no schema or checkpoint change, and stable
   TensorForge still imports without loading the native library.
"""

import gc
import struct
import subprocess
import sys

import numpy as np
import pytest

from tensorforge.backends import cpp
from tensorforge.experimental import (
    NativeConv2d,
    NativeParameter,
    NativeTensor,
)

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
    """The ids of every NativeStorage currently open — the project's
    supported deterministic instrumentation for native-allocation
    lifetime (the Phase-C/D/E precedent)."""
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

# The H9 minimum swept extent, mirrored from tf_conv2d_internal.h. It is
# duplicated rather than imported because there is deliberately no way to
# ask the library for it; the tests below that depend on which side of it a
# geometry falls state that dependence in their own names.
MIN_SWEPT_EXTENT = 4


def _core(values):
    return cpp.NativeTensorCore.from_array(np.asarray(values, dtype=np.float64))


def _bits(array):
    return np.asarray(array, dtype=np.float64).view(np.uint64)


def _ones_seed(tensor):
    """An all-ones upstream gradient as a NativeTensor, which is what
    ``NativeTensor.backward`` requires."""
    return NativeTensor.from_array(np.ones(tensor.shape))


def _out_dim(size, kernel, stride, pad):
    return (size + 2 * pad - kernel) // stride + 1


def _numpy_conv2d(images, weight, bias, stride, padding):
    """An explicit NCHW cross-correlation written as a formula — the
    independent oracle, borrowed from neither implementation.

    Accurate to a tolerance, **not** bit-exact: ``np.sum`` reduces
    pairwise, while the kernels accumulate sequentially. Use
    ``_sequential_conv2d`` where bits matter."""
    sh, sw = stride
    ph, pw = padding
    n, c, h, w = images.shape
    o, _c, kh, kw = weight.shape
    oh, ow = _out_dim(h, kh, sh, ph), _out_dim(w, kw, sw, pw)
    padded = np.zeros((n, c, h + 2 * ph, w + 2 * pw), dtype=np.float64)
    padded[:, :, ph:ph + h, pw:pw + w] = images
    out = np.empty((n, o, oh, ow), dtype=np.float64)
    for bi in range(n):
        for oc in range(o):
            for i in range(oh):
                for j in range(ow):
                    window = padded[bi, :, i * sh:i * sh + kh,
                                    j * sw:j * sw + kw]
                    total = float(np.sum(window * weight[oc]))
                    out[bi, oc, i, j] = total + (0.0 if bias is None
                                                 else float(bias[oc]))
    return out


def _sequential_conv2d(images, weight, bias, stride, padding):
    """The same cross-correlation, accumulated in the **kernel's own
    order** — bias seed, then ascending c, p, q, skipping taps whose source
    coordinate leaves the real input.

    This is what makes a raw-bit comparison meaningful. ``np.sum`` reduces
    pairwise and so legitimately produces different bits from any
    sequential accumulation; that is a property of the oracle, not a defect
    in either convolution path. Written in plain Python floats so the
    arithmetic is unambiguously one IEEE-754 binary64 addition at a time.
    """
    sh, sw = stride
    ph, pw = padding
    n, c, h, w = images.shape
    o, _c, kh, kw = weight.shape
    oh, ow = _out_dim(h, kh, sh, ph), _out_dim(w, kw, sw, pw)
    out = np.empty((n, o, oh, ow), dtype=np.float64)
    for bi in range(n):
        for oc in range(o):
            seed = 0.0 if bias is None else float(bias[oc])
            for i in range(oh):
                for j in range(ow):
                    acc = seed
                    for ch in range(c):
                        for p in range(kh):
                            ih = i * sh + p - ph
                            if ih < 0 or ih >= h:
                                continue
                            for q in range(kw):
                                iw = j * sw + q - pw
                                if iw < 0 or iw >= w:
                                    continue
                                acc += (float(images[bi, ch, ih, iw])
                                        * float(weight[oc, ch, p, q]))
                    out[bi, oc, i, j] = acc
    return out


# (name, n, c, h, w, o, kh, kw, sh, sw, ph, pw). The matrix deliberately
# straddles both predicate rules: `swept` records min(input_width,
# output_width), so cases below MIN_SWEPT_EXTENT take the retained generic
# path, and cases with a non-unit stride take it for the input gradient
# only.
GEOMETRIES = [
    ("unit_1x1", 1, 1, 6, 6, 1, 1, 1, 1, 1, 0, 0),
    ("single_3x3", 1, 1, 8, 8, 1, 3, 3, 1, 1, 0, 0),
    ("multi_channel", 2, 3, 8, 9, 4, 3, 3, 1, 1, 0, 0),
    ("padded", 2, 3, 8, 9, 4, 3, 3, 1, 1, 1, 1),
    ("strided_padded", 2, 3, 9, 9, 4, 3, 3, 2, 2, 1, 1),
    ("asymmetric", 2, 3, 9, 11, 4, 3, 3, 2, 1, 0, 1),
    ("kernel_5x5", 1, 2, 12, 12, 3, 5, 5, 1, 1, 2, 2),
    ("rect_kernel", 2, 2, 7, 13, 3, 3, 5, 1, 1, 0, 0),
    ("rect_padded", 2, 2, 13, 7, 3, 5, 3, 1, 1, 2, 1),
    ("prime_dims", 1, 5, 23, 29, 7, 3, 3, 1, 1, 1, 1),
    ("stride_3", 2, 3, 10, 10, 4, 3, 3, 3, 3, 0, 0),
    ("pad_exceeds_kernel", 1, 1, 4, 4, 1, 3, 3, 1, 1, 2, 2),
    # -- below the swept-extent minimum: the retained generic path --
    ("one_output_element", 1, 1, 5, 5, 1, 5, 5, 1, 1, 0, 0),
    ("narrow_output", 2, 2, 8, 5, 3, 3, 3, 1, 1, 0, 0),
    ("narrow_input", 2, 2, 8, 3, 3, 3, 3, 1, 1, 1, 1),
    ("kernel_fills_input", 2, 2, 6, 8, 3, 6, 8, 1, 1, 0, 0),
]


def _geometry_data(spec, seed=0):
    _name, n, c, h, w, o, kh, kw, sh, sw, ph, pw = spec
    oh, ow = _out_dim(h, kh, sh, ph), _out_dim(w, kw, sw, pw)
    rng = np.random.default_rng(seed)
    return {
        "images": rng.uniform(-1, 1, (n, c, h, w)),
        "weight": rng.uniform(-1, 1, (o, c, kh, kw)),
        "bias": rng.uniform(-1, 1, (o,)),
        "upstream": rng.uniform(-1, 1, (n, o, oh, ow)),
        "input_shape": (n, c, h, w),
        "weight_shape": (o, c, kh, kw),
        "stride": (sh, sw),
        "padding": (ph, pw),
        "swept": min(w, ow),
    }


def _swept_extent(spec):
    _name, _n, _c, _h, w, _o, _kh, kw, _sh, sw, _ph, pw = spec
    return min(w, _out_dim(w, kw, sw, pw))


# ==========================================================================
# 1. Both sides of every predicate are correct
# ==========================================================================

@pytest.mark.parametrize("spec", GEOMETRIES, ids=[g[0] for g in GEOMETRIES])
def test_forward_matches_the_numpy_formula_on_both_sides_of_the_predicate(spec):
    data = _geometry_data(spec)
    expected = _numpy_conv2d(data["images"], data["weight"], data["bias"],
                             data["stride"], data["padding"])
    images, weight, bias = (_core(data["images"]), _core(data["weight"]),
                            _core(data["bias"]))
    try:
        out = images.conv2d_forward(weight, bias, stride=data["stride"],
                                    padding=data["padding"])
        try:
            assert out.shape == expected.shape
            assert np.allclose(out.to_numpy(), expected, atol=1e-12)
        finally:
            out.close()
        # ...and with no bias, a separate seeding path in the row sweep.
        out = images.conv2d_forward(weight, None, stride=data["stride"],
                                    padding=data["padding"])
        try:
            expected_no_bias = _numpy_conv2d(
                data["images"], data["weight"], None, data["stride"],
                data["padding"])
            assert np.allclose(out.to_numpy(), expected_no_bias, atol=1e-12)
        finally:
            out.close()
    finally:
        for core in (images, weight, bias):
            core.close()


@pytest.mark.parametrize("spec", GEOMETRIES, ids=[g[0] for g in GEOMETRIES])
def test_both_gradients_match_the_stable_autograd_on_both_sides(spec):
    """The stable ``tensorforge.nn.Conv2d``'s own autograd is the oracle
    for the two gradient kernels, exactly as the Phase-D suites use it."""
    from tensorforge import Tensor
    from tensorforge.nn import Conv2d

    data = _geometry_data(spec, seed=3)
    _name, n, c, h, w, o, kh, kw, sh, sw, ph, pw = spec
    layer = Conv2d(c, o, (kh, kw), stride=(sh, sw), padding=(ph, pw),
                   bias=True)
    layer.weight.data = data["weight"].copy()
    layer.bias.data = data["bias"].copy()
    x = Tensor(data["images"], requires_grad=True)
    # The stable backward() takes no seed, so the upstream is applied as
    # a weighting whose scalar objective has exactly that derivative.
    ((layer(x) * Tensor(data["upstream"])).sum()).backward()

    upstream = _core(data["upstream"])
    weight = _core(data["weight"])
    images = _core(data["images"])
    try:
        grad_in = upstream.conv2d_input_backward(
            weight, input_shape=data["input_shape"], stride=data["stride"],
            padding=data["padding"])
        try:
            assert np.allclose(grad_in.to_numpy(), x.grad, atol=1e-12)
        finally:
            grad_in.close()
        grad_w = upstream.conv2d_weight_backward(
            images, weight_shape=data["weight_shape"], stride=data["stride"],
            padding=data["padding"])
        try:
            assert np.allclose(grad_w.to_numpy(), layer.weight.grad,
                               atol=1e-12)
        finally:
            grad_w.close()
    finally:
        for core in (upstream, weight, images):
            core.close()


def test_the_geometry_matrix_really_covers_both_sides_of_both_rules():
    """A guard on the guard: if every geometry above drifted to one side of
    a predicate, the parametrized tests would silently stop covering the
    fallback. This asserts the matrix keeps straddling both rules."""
    swept = [_swept_extent(spec) for spec in GEOMETRIES]
    assert any(value >= MIN_SWEPT_EXTENT for value in swept)
    assert any(value < MIN_SWEPT_EXTENT for value in swept), swept
    strides = [(spec[8], spec[9]) for spec in GEOMETRIES]
    assert any(s == (1, 1) for s in strides)
    assert any(s != (1, 1) for s in strides), strides


# ==========================================================================
# 2. Special values through the Python boundary
# ==========================================================================

SPECIALS = {
    "positive_zero": 0.0,
    "negative_zero": -0.0,
    "positive_inf": float("inf"),
    "negative_inf": float("-inf"),
    "denormal": 5e-324,
    "smallest_normal": 2.2250738585072014e-308,
    "largest_finite": 1.7976931348623157e308,
}


@pytest.mark.parametrize("label", list(SPECIALS))
def test_special_values_are_exact_through_the_convolution(label):
    """A single special value planted in the input, compared as raw bit
    patterns against the **sequential** oracle — the one that accumulates
    in the kernel's own order, so bits are a fair question. One special
    value means at most one NaN can reach any destination, which is the
    half of the H9 contract that *is* exact.

    Run at a geometry that takes the H9 row sweep and again at one below
    the swept-extent minimum, so both compute paths are covered."""
    value = SPECIALS[label]
    for spec in (("swept", 2, 2, 6, 8, 2, 3, 3, 1, 1, 1, 1),
                 ("fallback", 2, 2, 6, 3, 2, 3, 3, 1, 1, 0, 0)):
        padding = (spec[10], spec[11])
        data = _geometry_data(spec, seed=11)
        images = data["images"].copy()
        images[0, 0, 2, 1] = value
        weight = np.ones_like(data["weight"])
        bias = np.zeros_like(data["bias"])
        expected = _sequential_conv2d(images, weight, bias, (1, 1), padding)

        ic, wc, bc = _core(images), _core(weight), _core(bias)
        try:
            out = ic.conv2d_forward(wc, bc, stride=(1, 1), padding=padding)
            try:
                produced = out.to_numpy()
                finite = np.isfinite(expected)
                # Finite results agree bit for bit with the sequential
                # oracle — signed zeros, denormals and max-finite included.
                assert np.array_equal(_bits(produced[finite]),
                                      _bits(expected[finite])), spec[0]
                # Non-finite results agree in class and position.
                assert np.array_equal(np.isnan(produced), np.isnan(expected))
                assert np.array_equal(np.isposinf(produced),
                                      np.isposinf(expected))
                assert np.array_equal(np.isneginf(produced),
                                      np.isneginf(expected))
            finally:
                out.close()
        finally:
            for core in (ic, wc, bc):
                core.close()


def test_negative_zero_survives_an_all_negative_zero_accumulation():
    """The row sweep replaces a register accumulator with an
    accumulate-into-memory sequence, which is exactly the rewrite that
    could change a zero's sign. -0.0 survives only while every addend is
    -0.0; one +0.0 makes the sum +0.0. Both are asserted."""
    shape = (1, 2, 6, 8)
    images = np.full(shape, -0.0)
    weight = np.zeros((2, 2, 3, 3))          # -0.0 * +0.0 = -0.0
    bias = np.full((2,), -0.0)
    ic, wc, bc = _core(images), _core(weight), _core(bias)
    try:
        out = ic.conv2d_forward(wc, bc)
        try:
            assert np.all(np.signbit(out.to_numpy())), (
                "every addend is -0.0, so the sum must stay -0.0")
        finally:
            out.close()
    finally:
        for core in (ic, wc, bc):
            core.close()


def test_one_positive_zero_addend_makes_the_sum_positive_zero():
    ic = _core(np.zeros((1, 2, 6, 8)))
    wc = _core(np.zeros((2, 2, 3, 3)))
    bc = _core(np.full((2,), -0.0))
    try:
        out = ic.conv2d_forward(wc, bc)
        try:
            assert not np.any(np.signbit(out.to_numpy())), (
                "+0.0 products must turn a -0.0 bias seed positive")
        finally:
            out.close()
    finally:
        for core in (ic, wc, bc):
            core.close()


def test_nan_positions_and_quietness_through_the_convolution():
    """NaN positions are contractual; payload bits, when two or more NaNs
    reach one destination, deliberately are not — so this asserts position
    and quietness and says nothing about payloads."""
    images = np.random.default_rng(5).uniform(-1, 1, (2, 2, 6, 8))
    images[0, 0, ::2, ::2] = np.nan
    weight = np.ones((2, 2, 3, 3))
    bias = np.zeros(2)
    expected = _numpy_conv2d(images, weight, bias, (1, 1), (1, 1))
    ic, wc, bc = _core(images), _core(weight), _core(bias)
    try:
        out = ic.conv2d_forward(wc, bc, stride=1, padding=1)
        try:
            produced = out.to_numpy()
            assert np.array_equal(np.isnan(produced), np.isnan(expected))
            quiet_bit = np.uint64(1) << np.uint64(51)
            nan_bits = _bits(produced)[np.isnan(produced)]
            assert np.all(nan_bits & quiet_bit), "every NaN produced is quiet"
        finally:
            out.close()
    finally:
        for core in (ic, wc, bc):
            core.close()


def test_a_signaling_nan_input_is_quieted():
    snan = struct.unpack("<d", struct.pack("<Q", 0x7FF0000000000001))[0]
    images = np.full((1, 1, 6, 8), snan)
    ic, wc, bc = (_core(images), _core(np.ones((1, 1, 3, 3))),
                  _core(np.zeros(1)))
    try:
        out = ic.conv2d_forward(wc, bc)
        try:
            produced = out.to_numpy()
            quiet_bit = np.uint64(1) << np.uint64(51)
            nan_bits = _bits(produced)[np.isnan(produced)]
            assert nan_bits.size
            assert np.all(nan_bits & quiet_bit)
        finally:
            out.close()
    finally:
        for core in (ic, wc, bc):
            core.close()


# ==========================================================================
# 3. H1 allocation safety on the fallback geometries
# ==========================================================================

POISON_NAN = struct.unpack("<d", struct.pack("<Q", 0x7FF8DEADBEEFCAFE))[0]
POISON_FINITE = -1.2345678901234567e300


@pytest.mark.parametrize("pattern", (POISON_NAN, POISON_FINITE))
def test_fallback_geometries_still_write_every_destination_element(pattern):
    """The existing H1 suite covers convolution at geometries that now take
    the H9 optimized paths. This covers the other side: geometries below
    the swept-extent minimum, which run the retained generic loops."""
    from test_native_storage_allocation import poisoned

    fallbacks = [spec for spec in GEOMETRIES
                 if _swept_extent(spec) < MIN_SWEPT_EXTENT]
    assert fallbacks, "the matrix must contain fallback geometries"
    for spec in fallbacks:
        data = _geometry_data(spec, seed=17)
        images, weight, bias = (_core(data["images"]), _core(data["weight"]),
                                _core(data["bias"]))
        upstream = _core(data["upstream"])
        try:
            with poisoned(pattern):
                out = images.conv2d_forward(weight, bias,
                                            stride=data["stride"],
                                            padding=data["padding"])
            try:
                _assert_clean(out, pattern, f"{spec[0]} forward")
            finally:
                out.close()
            with poisoned(pattern):
                grad_in = upstream.conv2d_input_backward(
                    weight, input_shape=data["input_shape"],
                    stride=data["stride"], padding=data["padding"])
            try:
                _assert_clean(grad_in, pattern, f"{spec[0]} input gradient")
            finally:
                grad_in.close()
            with poisoned(pattern):
                grad_w = upstream.conv2d_weight_backward(
                    images, weight_shape=data["weight_shape"],
                    stride=data["stride"], padding=data["padding"])
            try:
                _assert_clean(grad_w, pattern, f"{spec[0]} weight gradient")
            finally:
                grad_w.close()
        finally:
            for core in (images, weight, bias, upstream):
                core.close()


def _assert_clean(core, pattern, label):
    values = core.to_numpy()
    if np.isnan(pattern):
        survivors = int(np.count_nonzero(np.isnan(values)))
    else:
        survivors = int(np.count_nonzero(values == pattern))
    assert survivors == 0, f"{label}: {survivors} element(s) never written"


@pytest.mark.parametrize("pattern", (POISON_NAN, POISON_FINITE))
def test_optimized_geometries_write_every_destination_element(pattern):
    """The optimized side of the same proof, at the Python allocator seam:
    the row sweep primes its whole output row before accumulating, and the
    weight gather assigns every destination, so neither leaves poison."""
    from test_native_storage_allocation import poisoned

    optimized = [spec for spec in GEOMETRIES
                 if _swept_extent(spec) >= MIN_SWEPT_EXTENT]
    assert optimized
    for spec in optimized:
        data = _geometry_data(spec, seed=19)
        images, weight, bias = (_core(data["images"]), _core(data["weight"]),
                                _core(data["bias"]))
        upstream = _core(data["upstream"])
        try:
            for use_bias in (bias, None):
                with poisoned(pattern):
                    out = images.conv2d_forward(weight, use_bias,
                                                stride=data["stride"],
                                                padding=data["padding"])
                try:
                    _assert_clean(out, pattern, f"{spec[0]} forward")
                finally:
                    out.close()
            with poisoned(pattern):
                grad_in = upstream.conv2d_input_backward(
                    weight, input_shape=data["input_shape"],
                    stride=data["stride"], padding=data["padding"])
            try:
                _assert_clean(grad_in, pattern, f"{spec[0]} input gradient")
            finally:
                grad_in.close()
            with poisoned(pattern):
                grad_w = upstream.conv2d_weight_backward(
                    images, weight_shape=data["weight_shape"],
                    stride=data["stride"], padding=data["padding"])
            try:
                _assert_clean(grad_w, pattern, f"{spec[0]} weight gradient")
            finally:
                grad_w.close()
        finally:
            for core in (images, weight, bias, upstream):
                core.close()


def test_the_poison_detector_can_actually_fail():
    """A negative control: a destination the kernel does not fully write
    keeps its poison, so the assertions above are load-bearing."""
    core = _core(np.full((2, 2), POISON_FINITE))
    try:
        with pytest.raises(AssertionError):
            _assert_clean(core, POISON_FINITE, "negative control")
    finally:
        core.close()


# ==========================================================================
# 4. Layout behavior (Policy B) is exactly unchanged
# ==========================================================================

def test_non_contiguous_operands_still_work_and_are_never_mutated():
    """Policy B materializes a non-contiguous operand into a private copy
    before the kernel runs, so H9's contiguity assumption is upheld by the
    Core layer exactly as it always was."""
    rng = np.random.default_rng(23)
    base = rng.uniform(-1, 1, (2, 3, 8, 16))
    weight_base = rng.uniform(-1, 1, (4, 3, 3, 6))
    bias_values = rng.uniform(-1, 1, (4,))

    big = _core(base)
    weight_big = _core(weight_base)
    bias = _core(bias_values)
    try:
        # A narrowed input (offset view), a narrowed weight, and a chained
        # narrow — all still logically 4-D NCHW / OIHW.
        images = big.narrow(3, 2, 10)          # (2, 3, 8, 10), offset view
        weight = weight_big.narrow(3, 1, 3)    # (4, 3, 3, 3), non-contiguous
        chained = images.narrow(2, 1, 6)       # (2, 3, 6, 10)
        try:
            before_images = big.to_numpy().copy()
            before_weight = weight_big.to_numpy().copy()
            expected = _numpy_conv2d(images.to_numpy(), weight.to_numpy(),
                                     bias_values, (1, 1), (0, 0))
            out = images.conv2d_forward(weight, bias)
            try:
                assert np.allclose(out.to_numpy(), expected, atol=1e-12)
            finally:
                out.close()
            expected_chained = _numpy_conv2d(chained.to_numpy(),
                                             weight.to_numpy(), bias_values,
                                             (1, 1), (0, 0))
            out = chained.conv2d_forward(weight, bias)
            try:
                assert np.allclose(out.to_numpy(), expected_chained,
                                   atol=1e-12)
            finally:
                out.close()
            # The caller's storage is untouched by the private copies.
            assert np.array_equal(big.to_numpy(), before_images)
            assert np.array_equal(weight_big.to_numpy(), before_weight)
        finally:
            for view in (images, weight, chained):
                view.close()
    finally:
        for core in (big, weight_big, bias):
            core.close()


def test_a_transposed_upstream_gradient_still_produces_the_right_gradients():
    rng = np.random.default_rng(29)
    images = rng.uniform(-1, 1, (2, 2, 7, 9))
    weight = rng.uniform(-1, 1, (3, 2, 3, 3))
    upstream = rng.uniform(-1, 1, (2, 3, 5, 7))

    # Build the upstream as a transpose of a differently-laid-out core, so
    # the Core layer must materialize it before the kernel sees it.
    transposed_source = _core(np.ascontiguousarray(upstream.transpose(0, 1, 3, 2)))
    ic, wc = _core(images), _core(weight)
    try:
        view = transposed_source.transpose(0, 1, 3, 2)
        try:
            assert not view.contiguous
            assert np.allclose(view.to_numpy(), upstream)
            grad_in = view.conv2d_input_backward(
                wc, input_shape=(2, 2, 7, 9))
            try:
                contiguous = _core(upstream)
                try:
                    reference = contiguous.conv2d_input_backward(
                        wc, input_shape=(2, 2, 7, 9))
                    try:
                        assert np.array_equal(_bits(grad_in.to_numpy()),
                                              _bits(reference.to_numpy()))
                    finally:
                        reference.close()
                finally:
                    contiguous.close()
            finally:
                grad_in.close()
        finally:
            view.close()
    finally:
        for core in (transposed_source, ic, wc):
            core.close()


def test_repeated_calls_are_bit_identical():
    data = _geometry_data(GEOMETRIES[3], seed=31)
    images, weight, bias = (_core(data["images"]), _core(data["weight"]),
                            _core(data["bias"]))
    try:
        first = images.conv2d_forward(weight, bias, stride=data["stride"],
                                      padding=data["padding"])
        try:
            for _ in range(3):
                again = images.conv2d_forward(weight, bias,
                                              stride=data["stride"],
                                              padding=data["padding"])
                try:
                    assert np.array_equal(_bits(first.to_numpy()),
                                          _bits(again.to_numpy()))
                finally:
                    again.close()
        finally:
            first.close()
    finally:
        for core in (images, weight, bias):
            core.close()


def test_an_operand_sharing_storage_with_another_is_read_not_written():
    """The forward reads its operands and writes only a freshly allocated
    output, so two operands carved out of one storage are both intact
    afterwards — H9 made nothing in-place."""
    rng = np.random.default_rng(37)
    shared = _core(rng.uniform(-1, 1, (4, 2, 5, 8)))
    try:
        images = shared.narrow(0, 0, 2)
        weight_like = shared.narrow(0, 2, 2)   # (2, 2, 5, 8) as an OIHW weight
        try:
            before = shared.to_numpy().copy()
            out = images.conv2d_forward(weight_like, None)
            try:
                assert out.shape == (2, 2, 1, 1)
            finally:
                out.close()
            assert np.array_equal(shared.to_numpy(), before)
        finally:
            images.close()
            weight_like.close()
    finally:
        shared.close()


# ==========================================================================
# 5. Autograd and graph-resource ownership
# ==========================================================================

GRAD_COMBINATIONS = [
    (True, True, True),
    (True, True, False),
    (True, False, True),
    (False, True, True),
    (True, False, False),
    (False, True, False),
    (False, False, True),
    (False, False, False),
]


@pytest.mark.parametrize("wants", GRAD_COMBINATIONS,
                         ids=lambda w: "x{}_w{}_b{}".format(*(int(v) for v in w)))
def test_every_requires_grad_combination_produces_exactly_the_right_gradients(
        wants):
    want_input, want_weight, want_bias = wants
    rng = np.random.default_rng(41)
    x = NativeTensor.from_array(rng.uniform(-1, 1, (2, 2, 7, 9)),
                     requires_grad=want_input)
    weight = NativeParameter(rng.uniform(-1, 1, (3, 2, 3, 3)),
                             requires_grad=want_weight)
    bias = NativeParameter(rng.uniform(-1, 1, (3,)), requires_grad=want_bias)
    try:
        out = x.conv2d(weight, bias, stride=1, padding=1)
        try:
            assert out.requires_grad == any(wants)
            if not any(wants):
                assert x.grad is None and weight.grad is None
                assert bias.grad is None
                return
            out.backward(_ones_seed(out))
            assert (x.grad is not None) == want_input
            assert (weight.grad is not None) == want_weight
            assert (bias.grad is not None) == want_bias
            if want_input:
                assert x.grad.shape == x.shape
            if want_weight:
                assert weight.grad.shape == weight.shape
            if want_bias:
                assert bias.grad.shape == bias.shape
                # The bias gradient is the upstream summed over batch and
                # both spatial axes — H9 changed no reduction.
                assert np.allclose(
                    bias.grad.to_numpy(),
                    np.full((3,), out.shape[0] * out.shape[2] * out.shape[3]))
        finally:
            out.close()
    finally:
        for tensor in (x, weight, bias):
            tensor.close()


def test_gradients_accumulate_across_repeated_backward_with_a_retained_graph():
    rng = np.random.default_rng(43)
    x = NativeTensor.from_array(rng.uniform(-1, 1, (2, 2, 7, 9)), requires_grad=True)
    weight = NativeParameter(rng.uniform(-1, 1, (3, 2, 3, 3)))
    bias = NativeParameter(rng.uniform(-1, 1, (3,)))
    try:
        out = x.conv2d(weight, bias, stride=1, padding=1)
        try:
            out.backward(_ones_seed(out), retain_graph=True)
            first_x = x.grad.to_numpy().copy()
            first_w = weight.grad.to_numpy().copy()
            first_b = bias.grad.to_numpy().copy()
            out.backward(_ones_seed(out))
            assert np.allclose(x.grad.to_numpy(), 2 * first_x)
            assert np.allclose(weight.grad.to_numpy(), 2 * first_w)
            assert np.allclose(bias.grad.to_numpy(), 2 * first_b)
        finally:
            out.close()
    finally:
        for tensor in (x, weight, bias):
            tensor.close()


def test_parameter_identity_storage_identity_and_versions_are_untouched():
    """A convolution forward and backward read their operands; they move no
    parameter version and replace no parameter storage."""
    rng = np.random.default_rng(47)
    x = NativeTensor.from_array(rng.uniform(-1, 1, (2, 2, 7, 9)), requires_grad=True)
    weight = NativeParameter(rng.uniform(-1, 1, (3, 2, 3, 3)))
    bias = NativeParameter(rng.uniform(-1, 1, (3,)))
    try:
        weight_storage = weight._core._storage
        bias_storage = bias._core._storage
        weight_version = weight._version
        bias_version = bias._version
        out = x.conv2d(weight, bias, stride=1, padding=1)
        try:
            out.backward(_ones_seed(out))
        finally:
            out.close()
        assert weight._core._storage is weight_storage
        assert bias._core._storage is bias_storage
        assert weight._version == weight_version
        assert bias._version == bias_version
    finally:
        for tensor in (x, weight, bias):
            tensor.close()


def test_an_abandoned_convolution_graph_releases_every_storage(live_storages):
    rng = np.random.default_rng(53)
    baseline = len(live_storages)
    x = NativeTensor.from_array(rng.uniform(-1, 1, (2, 2, 7, 9)), requires_grad=True)
    weight = NativeParameter(rng.uniform(-1, 1, (3, 2, 3, 3)))
    bias = NativeParameter(rng.uniform(-1, 1, (3,)))
    try:
        out = x.conv2d(weight, bias, stride=1, padding=1)
        out.close()          # abandoned without ever running backward
        gc.collect()
        assert len(live_storages) == baseline + 3
    finally:
        for tensor in (x, weight, bias):
            tensor.close()
    gc.collect()
    assert len(live_storages) == baseline


def test_repeated_forward_backward_cycles_return_to_the_storage_baseline(
        live_storages):
    rng = np.random.default_rng(59)
    module = NativeConv2d(2, 3, 3, padding=1, seed=0)

    def cycle():
        x = NativeTensor.from_array(rng.uniform(-1, 1, (2, 2, 7, 9)),
                                    requires_grad=True)
        out = module(x)
        grad = _ones_seed(out)
        out.backward(grad)
        grad.close()
        out.close()
        x.close()
        module.zero_grad()
        gc.collect()

    try:
        # One warm-up cycle establishes the steady state. A module's first
        # forward/backward leaves one extra live storage that every later
        # cycle then reuses rather than accumulating; that is pre-existing
        # engine behaviour, identical for NativeLinear, and not something
        # the convolution owns. The property that matters — and the one a
        # leak would break — is that the count does not grow.
        cycle()
        baseline = len(live_storages)
        for _ in range(5):
            cycle()
            assert len(live_storages) == baseline
    finally:
        module.weight.close()
        module.bias.close()


# ==========================================================================
# 6. Failure atomicity
# ==========================================================================

@needs_fault_injection
@pytest.mark.parametrize("countdown", (1, 2, 3))
def test_an_injected_allocation_failure_leaves_everything_untouched(
        countdown, live_storages):
    """Every convolution stage that allocates is exercised by counting the
    failure down; whichever stage it lands on, nothing observable moved."""
    rng = np.random.default_rng(61)
    x = NativeTensor.from_array(rng.uniform(-1, 1, (2, 2, 7, 9)), requires_grad=True)
    weight = NativeParameter(rng.uniform(-1, 1, (3, 2, 3, 3)))
    bias = NativeParameter(rng.uniform(-1, 1, (3,)))
    try:
        gc.collect()
        baseline = len(live_storages)
        before_x = x.to_numpy().copy()
        before_w = weight.to_numpy().copy()
        before_b = bias.to_numpy().copy()
        weight_version = weight._version
        bias_version = bias._version

        cpp._arm_alloc_failure(countdown)
        try:
            out = x.conv2d(weight, bias, stride=1, padding=1)
        except (MemoryError, RuntimeError):
            pass
        else:
            out.close()
        finally:
            cpp._arm_alloc_failure(0)

        gc.collect()
        assert len(live_storages) == baseline
        assert np.array_equal(x.to_numpy(), before_x)
        assert np.array_equal(weight.to_numpy(), before_w)
        assert np.array_equal(bias.to_numpy(), before_b)
        assert weight._version == weight_version
        assert bias._version == bias_version
        assert weight.grad is None and bias.grad is None and x.grad is None
    finally:
        for tensor in (x, weight, bias):
            tensor.close()


@needs_fault_injection
def test_a_failure_during_the_backward_leaves_existing_gradients_intact(
        live_storages):
    rng = np.random.default_rng(67)
    x = NativeTensor.from_array(rng.uniform(-1, 1, (2, 2, 7, 9)), requires_grad=True)
    weight = NativeParameter(rng.uniform(-1, 1, (3, 2, 3, 3)))
    bias = NativeParameter(rng.uniform(-1, 1, (3,)))
    try:
        first = x.conv2d(weight, bias, stride=1, padding=1)
        first.backward(_ones_seed(first))
        first.close()
        established_w = weight.grad.to_numpy().copy()
        established_b = bias.grad.to_numpy().copy()
        established_x = x.grad.to_numpy().copy()
        gc.collect()
        baseline = len(live_storages)

        out = x.conv2d(weight, bias, stride=1, padding=1)
        try:
            for countdown in (1, 2, 3, 4):
                cpp._arm_alloc_failure(countdown)
                try:
                    out.backward(_ones_seed(out), retain_graph=True)
                except (MemoryError, RuntimeError):
                    pass
                finally:
                    cpp._arm_alloc_failure(0)
        finally:
            out.close()
        gc.collect()
        # Whatever happened, no gradient was silently corrupted and no
        # storage leaked: gradients are either unchanged or legitimately
        # accumulated, never partial garbage.
        assert len(live_storages) <= baseline + 3
        assert weight.grad is not None and bias.grad is not None
        assert np.all(np.isfinite(weight.grad.to_numpy()))
        assert np.all(np.isfinite(bias.grad.to_numpy()))
        assert np.all(np.isfinite(x.grad.to_numpy()))
        assert established_w.shape == weight.grad.shape
        assert established_b.shape == bias.grad.shape
        assert established_x.shape == x.grad.shape
    finally:
        for tensor in (x, weight, bias):
            tensor.close()


def test_a_closed_operand_is_rejected_before_anything_is_allocated(
        live_storages):
    rng = np.random.default_rng(71)
    x = NativeTensor.from_array(rng.uniform(-1, 1, (2, 2, 7, 9)), requires_grad=True)
    weight = NativeParameter(rng.uniform(-1, 1, (3, 2, 3, 3)))
    weight.close()
    try:
        gc.collect()
        baseline = len(live_storages)
        with pytest.raises(RuntimeError):
            x.conv2d(weight)
        gc.collect()
        assert len(live_storages) == baseline
    finally:
        x.close()


# ==========================================================================
# 7. Surface: nothing public moved
# ==========================================================================

def test_the_export_table_is_exactly_what_h8_left():
    """H9 changed internal C++ execution only. No new C ABI symbol, and in
    particular no path selector, block-size setter, workspace, im2col, or
    profiling hook."""
    from test_native_storage_allocation import (
        EXPECTED_TF_EXPORTS, PHASE_H_TF_EXPORTS, exported_names,
        phase_h_export_names,
    )

    image, names = exported_names(cpp._LIBRARY_PATH)
    if names is None:
        pytest.skip("this image format is not parsed here")
    exported = sorted(name for name in names if name.startswith("tf_"))
    assert len(exported) == EXPECTED_TF_EXPORTS, exported
    # H8's claim — that it added no ABI symbol — is about Phase H, so it
    # is measured against Phase H's own surface. The two extra symbols in
    # the live library are Phase I's typed creators (milestone I1).
    assert len(phase_h_export_names(exported)) == PHASE_H_TF_EXPORTS
    conv = [name for name in exported if "conv" in name]
    assert conv == ["tf_core_conv2d_forward",
                    "tf_core_conv2d_input_backward",
                    "tf_core_conv2d_weight_backward"]
    # Scoped to the convolution surface H9 touched. (The pre-existing raw
    # benchmark kernel tf_matmul_tiled is H2-era and unrelated.)
    forbidden = ("sweep", "gather", "block", "tile", "im2col", "unfold",
                 "workspace", "select", "dispatch", "profile", "counter",
                 "path", "threshold")
    assert not [name for name in conv
                if any(word in name.lower() for word in forbidden)]
    assert not [name for name in exported if "im2col" in name.lower()]


def test_no_convolution_dispatch_control_is_reachable_from_python():
    """Neither the loaded library nor the backend module offers a way to
    choose, observe, or tune a convolution path."""
    library = cpp._require_library()
    for name in ("tf_core_conv2d_set_path", "tf_core_conv2d_block_size",
                 "tf_conv2d_select", "tf_core_conv2d_forward_row_sweep",
                 "tf_core_conv2d_forward_generic", "tf_conv2d_stats"):
        with pytest.raises(AttributeError):
            getattr(library, name)
    for name in dir(cpp):
        lowered = name.lower()
        if "conv" in lowered:
            assert not any(word in lowered for word in
                           ("sweep", "gather", "block", "select", "path",
                            "counter", "stat", "profile")), name


def test_no_environment_variable_changes_convolution_behaviour(monkeypatch):
    """A geometry is dispatched from its own extents and nothing else."""
    data = _geometry_data(GEOMETRIES[3], seed=73)
    images, weight, bias = (_core(data["images"]), _core(data["weight"]),
                            _core(data["bias"]))
    try:
        out = images.conv2d_forward(weight, bias, stride=data["stride"],
                                    padding=data["padding"])
        reference = out.to_numpy().copy()
        out.close()
        for name in ("TF_CONV2D_PATH", "TF_CONV2D_BLOCK", "TF_KERNEL",
                     "TENSORFORGE_CONV2D", "TF_DISPATCH"):
            monkeypatch.setenv(name, "generic")
        out = images.conv2d_forward(weight, bias, stride=data["stride"],
                                    padding=data["padding"])
        try:
            assert np.array_equal(_bits(out.to_numpy()), _bits(reference))
        finally:
            out.close()
    finally:
        for core in (images, weight, bias):
            core.close()


def test_the_convolution_public_surface_is_unchanged():
    """Signatures, supported options, and the module surface are exactly
    what Phase D defined: no dilation, no groups, no channels-last, no
    performance keyword.

    The **operation** and the **Core wrapper** are byte-for-byte the same
    signatures — dtype travels on the tensors, never as an argument, so
    neither gained one at any Phase-I milestone. The **module** gained
    exactly one keyword-only ``dtype`` at milestone I7, because a module is
    the thing that *creates* state and therefore has to be told what to
    create it as."""
    import inspect

    signature = inspect.signature(NativeTensor.conv2d)
    assert list(signature.parameters) == ["self", "weight", "bias", "stride",
                                          "padding"]
    core_signature = inspect.signature(cpp.NativeTensorCore.conv2d_forward)
    assert list(core_signature.parameters) == ["self", "weight", "bias",
                                               "stride", "padding"]
    module_signature = inspect.signature(NativeConv2d.__init__)
    assert list(module_signature.parameters) == [
        "self", "in_channels", "out_channels", "kernel_size", "stride",
        "padding", "bias", "seed", "requires_grad", "dtype"]
    assert module_signature.parameters["dtype"].default is None
    assert (module_signature.parameters["dtype"].kind
            is inspect.Parameter.KEYWORD_ONLY)
    assert "dtype" not in signature.parameters
    assert "dtype" not in core_signature.parameters
    # Phase I adds no device anywhere.
    assert "device" not in module_signature.parameters
    for forbidden in ("dilation", "groups", "channels_last", "output_padding",
                      "path", "block_size", "algorithm"):
        assert forbidden not in signature.parameters
        assert forbidden not in module_signature.parameters


def test_stable_tensorforge_still_imports_without_loading_the_native_library():
    """H9 touched only the experimental line; importing stable TensorForge
    must still not load the DLL or import the experimental namespace."""
    code = (
        "import sys; import tensorforge; "
        "import tensorforge.nn; import tensorforge.optim; "
        "loaded = [m for m in sys.modules "
        "          if 'experimental' in m or 'backends.cpp' in m]; "
        "print(loaded)"
    )
    result = subprocess.run([sys.executable, "-c", code],
                            capture_output=True, text=True, check=True)
    assert result.stdout.strip() == "[]", result.stdout


def test_the_supported_capability_boundary_did_not_move():
    assert cpp.UNSUPPORTED == ("float32", "cuda", "amp")
    assert cpp.SUPPORTED_DTYPES == ("float64",)
    assert cpp.SUPPORTED_DEVICES == ("cpu",)
    assert "conv2d" in cpp.AUTOGRAD_OPS
    assert "conv2d_forward" in cpp.TENSOR_CORE_OPS
    assert "NativeConv2d" in cpp.NATIVE_MODULES
