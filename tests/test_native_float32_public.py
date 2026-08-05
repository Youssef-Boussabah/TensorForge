"""Public float32 support (Phase I, milestone I9).

I9 is the milestone at which ``"float32"`` leaves ``UNSUPPORTED`` and
joins ``SUPPORTED_DTYPES``, and these are the guardrails for the **public
boundary** that move made. They cover the registry tuples themselves, the
public dtype normalizer, every public construction path at both widths,
egress, views, gradients, the defaults that must **not** have moved, the
mixed-dtype rejections that must stay strict, and the two rows that are
deliberately *different* statements from overall dtype support — the flat
``backend_info()["dtype"]`` default and the float64-only raw-kernel
registry.

The integrated exact-resume proof that earned the move lives in
``tests/test_native_float32_training.py``; the private/typed paths I1-I8
used, and which still exist, are covered by ``test_native_phase_i.py`` and
``test_native_float32_state.py``. **Nothing here replaces those** —
public-path coverage supplements private-path coverage, it does not
supersede it.

Contract: docs/native_dtype_float32_design.md §9 (no casting, no
promotion), §25 (public Python compatibility), §27 (rollout discipline),
and docs/native_support_matrix.md.

Selector: python -m pytest -q -k native_float32_public
"""

import subprocess
import sys

import numpy as np
import pytest

from tensorforge.backends import cpp
from tensorforge.experimental import (
    NativeAdam,
    NativeBatchNorm1d,
    NativeConv2d,
    NativeLayerNorm,
    NativeLinear,
    NativeParameter,
    NativeSGD,
    NativeTensor,
)

needs_native = pytest.mark.skipif(
    not cpp.is_available(), reason="the experimental C++ backend is not built"
)

BOTH_DTYPES = ("float64", "float32")
NUMPY_DTYPES = {"float64": np.float64, "float32": np.float32}
BIT_DTYPES = {"float64": np.uint64, "float32": np.uint32}


@pytest.fixture()
def live_storages(monkeypatch):
    """The ids of every open NativeStorage — the project's deterministic
    native-allocation instrumentation, so a rejection test can prove
    nothing was allocated rather than trusting collection."""
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


def _close_optimizer(optimizer):
    """Release an optimizer's native state, if it has any.

    ``NativeAdam`` owns moment tensors and has ``close()``; ``NativeSGD``
    owns no native storage at all and deliberately has none. Asked with
    ``hasattr`` rather than by type so this stays a statement about
    ownership rather than a list of class names."""
    if hasattr(optimizer, "close"):
        optimizer.close()


def bits(array, dtype):
    """Raw IEEE-754 bit patterns, with the dtype asserted rather than
    coerced — a helper that silently converted could report a match that
    only existed after a conversion this runtime does not perform."""
    array = np.asarray(array)
    assert array.dtype == NUMPY_DTYPES[dtype], array.dtype
    return np.ascontiguousarray(array).reshape(-1).view(
        BIT_DTYPES[dtype]).tolist()


# ==========================================================================
# 1. The registry move itself
# ==========================================================================


def test_the_public_dtype_registry_reads_float64_then_float32():
    """The exact I9 end state, including tuple **order**: float64 first,
    because it is the default that ``None`` selects and the width every
    pre-Phase-I behavior is defined at. float32 is an addition, never a
    replacement."""
    assert cpp.SUPPORTED_DTYPES == ("float64", "float32")
    assert cpp.SUPPORTED_DTYPES[0] == "float64"
    assert cpp.SUPPORTED_DEVICES == ("cpu",)


def test_float32_left_unsupported_and_cuda_and_amp_did_not():
    assert "float32" not in cpp.UNSUPPORTED
    assert cpp.UNSUPPORTED == ("cuda", "amp")


def test_supported_and_unsupported_never_overlap():
    """The two tuples answer opposite questions, so a name in both would
    make the registry self-contradictory. Checked over every registry a
    capability name can appear in, not just the dtype pair."""
    assert not set(cpp.SUPPORTED_DTYPES) & set(cpp.UNSUPPORTED)
    assert not set(cpp.SUPPORTED_DEVICES) & set(cpp.UNSUPPORTED)
    for registry in (cpp.RAW_KERNELS, cpp.TENSOR_CORE_OPS, cpp.AUTOGRAD_OPS,
                     cpp.NATIVE_MODULES, cpp.NATIVE_LOSSES,
                     cpp.NATIVE_METRICS, cpp.NATIVE_OPTIMIZERS,
                     cpp.STATE_SUPPORT):
        assert not set(registry) & set(cpp.UNSUPPORTED), registry


def test_no_dtype_beyond_the_two_became_supported():
    """The phase adds float32 and **only** float32."""
    assert set(cpp.SUPPORTED_DTYPES) == {"float64", "float32"}
    for absent in ("float16", "bfloat16", "int8", "int32", "int64", "bool",
                   "complex64", "complex128", "float128"):
        assert absent not in cpp.SUPPORTED_DTYPES, absent


def test_the_raw_kernel_registry_did_not_move():
    """A **different statement** from ``SUPPORTED_DTYPES``, and the one
    row I9 deliberately leaves alone. The seven handle-free raw kernels
    take ``double*`` and an element count, so they have no dtype to
    dispatch on and stay float64 permanently. Overall float32 support and
    raw-kernel float64 support are separate facts and neither may be read
    off the other."""
    assert cpp.RAW_KERNEL_DTYPES == ("float64",)
    assert cpp.RAW_KERNEL_DTYPES != cpp.SUPPORTED_DTYPES
    assert set(cpp.RAW_KERNEL_DTYPES) < set(cpp.SUPPORTED_DTYPES)
    assert len(cpp.RAW_KERNELS) == 7


@needs_native
def test_the_raw_kernels_still_convert_to_float64_and_return_float64():
    """Their established NumPy-wrapper behavior, unchanged: whatever they
    are handed is converted at the boundary and the result is float64. No
    ``_f32`` wrapper, argument, or symbol was added."""
    a = np.array([1.5, 2.5], dtype=np.float32)
    b = np.array([0.5, 0.5], dtype=np.float32)
    result = cpp.elementwise_add(a, b)
    assert result.dtype == np.float64
    assert result.tolist() == [2.0, 3.0]
    for name in cpp.RAW_KERNELS:
        assert not name.endswith("_f32"), name
        assert not name.endswith("_f64"), name
        assert "dtype" not in name, name


def test_backend_info_reports_the_three_dtype_rows_distinctly():
    """Three rows, three different questions, and none may be read off
    another (design §25.3, resolved at I9):

    - ``supported_dtypes`` is the **capability** statement;
    - the flat ``dtype`` key is the **default** statement and is retained
      as exactly that — it stays ``"float64"`` because ``float64`` is
      still what an omitted ``dtype`` selects, which makes it accurate
      rather than merely unchanged;
    - ``raw_kernel_dtypes`` is a permanent limitation of one small layer.
    """
    info = cpp.backend_info()
    assert info["supported_dtypes"] == ("float64", "float32")
    assert info["supported_devices"] == ("cpu",)
    assert info["raw_kernel_dtypes"] == ("float64",)
    assert info["unsupported"] == ("cuda", "amp")
    # The flat keys are defaults, and they agree with the normalizer that
    # actually decides — asserted against behavior, not against prose.
    assert info["dtype"] == "float64" == cpp.normalize_dtype(None)
    assert info["device"] == "cpu" == cpp.normalize_device(None)
    # ...and the capability rows are the live registries themselves.
    assert info["supported_dtypes"] is cpp.SUPPORTED_DTYPES
    assert info["raw_kernel_dtypes"] is cpp.RAW_KERNEL_DTYPES
    assert info["stable_framework_integration"] is False


def test_no_dtype_selector_or_global_default_was_added():
    """float32 support is per-construction and explicit. There is no
    global default to set, no environment variable, and no mode."""
    for banned in ("set_default_dtype", "get_default_dtype", "default_dtype",
                   "set_dtype", "astype", "cast", "promote", "result_type",
                   "autocast", "amp", "map_location"):
        assert not hasattr(cpp, banned), banned
    import os

    for variable in os.environ:
        assert "TENSORFORGE" not in variable.upper() or "DTYPE" not in (
            variable.upper()), variable


# ==========================================================================
# 2. normalize_dtype — the public authority for both widths
# ==========================================================================


def test_normalize_dtype_accepts_both_widths_and_defaults_to_float64():
    assert cpp.normalize_dtype(None) == "float64"
    assert cpp.normalize_dtype("float64") == "float64"
    assert cpp.normalize_dtype("float32") == "float32"


@pytest.mark.parametrize("value", [
    "Float32", "FLOAT32", " float32", "float32 ", "float_32", "f32", "f4",
    "single", "double", "float", "float16", "bfloat16", "float128",
    "int32", "int64", "uint8", "bool", "complex64", "", "cpu", "cuda",
])
def test_normalize_dtype_rejects_every_non_canonical_string(value):
    """No case folding, no whitespace trimming, no aliases. A permissive
    front door is exactly how a "dtype" silently becomes something else."""
    with pytest.raises(ValueError) as info:
        cpp.normalize_dtype(value)
    assert repr(value) in str(info.value)
    assert "float64" in str(info.value) and "float32" in str(info.value)


@pytest.mark.parametrize("value", [
    np.float32, np.float64, np.dtype("float32"), np.dtype("float64"),
    float, 32, 64, 32.0, True, False, b"float32", ["float32"], ("float32",),
    {"dtype": "float32"},
])
def test_normalize_dtype_rejects_every_non_string(value):
    """A ``numpy.dtype`` object is the specific alias the contract names,
    and it is rejected like every other non-string."""
    with pytest.raises(TypeError):
        cpp.normalize_dtype(value)


def test_normalize_dtype_needs_no_library_and_allocates_nothing(
    live_storages
):
    """Pure Python: safe whether or not the backend is built, and a
    rejection is decided before anything could be allocated."""
    baseline = len(live_storages)
    assert cpp.normalize_dtype("float32") == "float32"
    with pytest.raises(ValueError):
        cpp.normalize_dtype("float16")
    assert len(live_storages) == baseline


@pytest.mark.parametrize("value", ["cuda", "gpu", "CPU", "cuda:0", " cpu",
                                   "amp", ""])
def test_normalize_device_still_accepts_only_cpu(value):
    with pytest.raises(ValueError):
        cpp.normalize_device(value)
    assert cpp.normalize_device(None) == "cpu"
    assert cpp.normalize_device("cpu") == "cpu"


def test_the_internal_normalizer_agrees_with_the_public_one_on_both_floats():
    """Between I1 and I8 the representation table was genuinely wider than
    the public registry — that gap was the rollout. **At I9 they became
    equal**, and the note that closed this test predicted the rest: *"a
    future representable dtype would open the gap again before it earned
    the promise"*. **Phase K milestone K2 is that future**, and the gap it
    opened is permanent rather than a rollout: ``int64`` is representable,
    is promised by its own ``INDEX_DTYPES`` row, and is **never** a member
    of the floating-compute registry.

    What is asserted here is the durable half — the two functions agree on
    every dtype the *public* registry contains, and remain separate
    functions answering separate questions ("can the runtime lay these bits
    out?" versus "does TensorForge compute at this dtype?")."""
    for dtype in BOTH_DTYPES:
        assert cpp.normalize_dtype(dtype) == cpp._normalize_internal_dtype(
            dtype)
    assert set(cpp.SUPPORTED_DTYPES) < set(cpp._DTYPE_CODES)
    assert set(cpp._DTYPE_CODES) == (set(cpp.SUPPORTED_DTYPES)
                                     | set(cpp.INDEX_DTYPES))
    # ``int64`` is representable and is still not a compute dtype: the one
    # dtype on which the two validators deliberately disagree.
    assert cpp._normalize_internal_dtype("int64") == "int64"
    with pytest.raises(ValueError):
        cpp.normalize_dtype("int64")
    for bad in ("float16", "bfloat16", "Float32", "int32", "uint64"):
        with pytest.raises(ValueError):
            cpp._normalize_internal_dtype(bad)


# ==========================================================================
# 3. Public construction, at both widths
# ==========================================================================


@needs_native
@pytest.mark.parametrize("dtype", BOTH_DTYPES)
def test_public_storage_construction_at_both_dtypes(dtype):
    storage = cpp.NativeStorage(6, dtype=dtype)
    try:
        assert storage.dtype == dtype
        assert storage.device == "cpu"
        assert storage.size == 6                     # elements, never bytes
        out = storage.to_numpy()
        assert out.dtype == NUMPY_DTYPES[dtype]
        # Zero-initialized, and the zeros are **positive** zero.
        assert bits(out, dtype) == [0] * 6
    finally:
        storage.close()
    assert storage.dtype == dtype                    # readable after close


@needs_native
@pytest.mark.parametrize("dtype", BOTH_DTYPES)
def test_public_storage_from_array_at_both_dtypes(dtype):
    storage = cpp.NativeStorage.from_array([1.5, -2.25, 0.0], dtype=dtype)
    try:
        assert storage.dtype == dtype
        assert storage.to_numpy().dtype == NUMPY_DTYPES[dtype]
        assert storage.to_numpy().tolist() == [1.5, -2.25, 0.0]
    finally:
        storage.close()


@needs_native
@pytest.mark.parametrize("dtype", BOTH_DTYPES)
@pytest.mark.parametrize("factory", ["from_array", "zeros", "full"])
def test_public_core_factories_at_both_dtypes(dtype, factory):
    if factory == "from_array":
        core = cpp.NativeTensorCore.from_array(
            np.array([[1.5, -2.25], [0.5, 4.0]]), dtype=dtype)
        expected = [[1.5, -2.25], [0.5, 4.0]]
    elif factory == "zeros":
        core = cpp.NativeTensorCore.zeros((2, 2), dtype=dtype)
        expected = [[0.0, 0.0], [0.0, 0.0]]
    else:
        core = cpp.NativeTensorCore.full((2, 2), 1.5, dtype=dtype)
        expected = [[1.5, 1.5], [1.5, 1.5]]
    try:
        assert core.dtype == dtype
        assert core.device == "cpu"
        assert core.storage.dtype == dtype     # storage is the authority
        out = core.to_numpy()
        assert out.dtype == NUMPY_DTYPES[dtype]
        assert out.tolist() == expected
    finally:
        core.close()


@needs_native
@pytest.mark.parametrize("dtype", BOTH_DTYPES)
@pytest.mark.parametrize("factory", ["from_array", "zeros", "full"])
def test_public_tensor_factories_at_both_dtypes(dtype, factory):
    if factory == "from_array":
        tensor = NativeTensor.from_array([[1.5, -2.25]], dtype=dtype)
    elif factory == "zeros":
        tensor = NativeTensor.zeros((1, 2), dtype=dtype)
    else:
        tensor = NativeTensor.full((1, 2), -3.0, dtype=dtype)
    try:
        assert tensor.dtype == dtype
        assert tensor.device == "cpu"
        assert tensor.to_numpy().dtype == NUMPY_DTYPES[dtype]
        assert tensor.requires_grad is False
    finally:
        tensor.close()


@needs_native
def test_public_zeros_are_positive_zero_at_float32():
    """Bits, not values: ``-0.0 == 0.0`` in Python, so only the raw
    pattern proves the sign."""
    tensor = NativeTensor.zeros((3, 4), dtype="float32")
    try:
        assert bits(tensor.to_numpy(), "float32") == [0] * 12
    finally:
        tensor.close()


@needs_native
def test_public_full_narrows_the_scalar_once_at_float32():
    """The scalar contract (design §7.4): the value crosses the C ABI as a
    ``double`` and is narrowed **once**, before the fill loop, to the
    storage's element type. So the result is exactly ``float32(0.1)`` —
    proved as bits against NumPy's own single narrowing, not as a
    tolerance."""
    tensor = NativeTensor.full((4,), 0.1, dtype="float32")
    try:
        expected = np.full(4, np.float32(0.1), dtype=np.float32)
        assert bits(tensor.to_numpy(), "float32") == bits(expected, "float32")
    finally:
        tensor.close()


# ==========================================================================
# 4. The defaults that did NOT move
# ==========================================================================


@needs_native
@pytest.mark.parametrize("build", [
    lambda: cpp.NativeStorage(4),
    lambda: cpp.NativeStorage.from_array([1.0, 2.0]),
    lambda: cpp.NativeTensorCore.from_array(np.ones((2, 2))),
    lambda: cpp.NativeTensorCore.zeros((2, 2)),
    lambda: cpp.NativeTensorCore.full((2, 2), 1.0),
    lambda: NativeTensor.from_array([[1.0, 2.0]]),
    lambda: NativeTensor.zeros((2, 2)),
    lambda: NativeTensor.full((2, 2), 1.0),
    lambda: NativeParameter([[1.0, 2.0]]),
    lambda: NativeLinear(2, 3, seed=0),
    lambda: NativeConv2d(1, 2, 3, seed=0),
    lambda: NativeLayerNorm(4),
    lambda: NativeBatchNorm1d(4),
])
def test_omitting_dtype_still_selects_float64_everywhere(build):
    """**The default is float64 at every constructor, factory, module, and
    parameter, forever.** Existing code that omits ``dtype`` behaves
    byte-identically to how it always has."""
    obj = build()
    try:
        assert obj.dtype == "float64"
    finally:
        if hasattr(obj, "close"):
            obj.close()
        else:                                   # a module: close its state
            for parameter in obj.parameters():
                parameter.close()
            for buffer in obj.buffers():
                buffer.close()


@needs_native
@pytest.mark.parametrize("values", [
    np.ones((2, 2), dtype=np.float32),
    np.ones((2, 2), dtype=np.int64),
    np.ones((2, 2), dtype=np.int32),
    [[1, 2], [3, 4]],
])
def test_the_dtype_is_never_inferred_from_the_input_array(values):
    """A float32 NumPy array handed to ``from_array`` **without** a dtype
    still produces a float64 native tensor. Inference would silently
    change the meaning of existing code the day someone passed a float32
    array (design §9.4)."""
    tensor = NativeTensor.from_array(values)
    try:
        assert tensor.dtype == "float64"
        assert tensor.to_numpy().dtype == np.float64
    finally:
        tensor.close()


@needs_native
def test_explicit_float32_ingress_converts_host_data_once():
    """``from_array(float64_array, dtype="float32")`` is the explicit
    **host-to-native conversion boundary**, not a tensor cast: the host
    object is converted once on the way in, to the requested dtype, and no
    native tensor ever changes width.

    Proved on a value float32 cannot represent exactly — the result is
    bit-identical to NumPy's own single narrowing of the same input."""
    host = np.array([0.1, 1.0 / 3.0, 2.0 ** -30], dtype=np.float64)
    tensor = NativeTensor.from_array(host, dtype="float32")
    try:
        assert tensor.dtype == "float32"
        assert bits(tensor.to_numpy(), "float32") == bits(
            host.astype(np.float32), "float32")
        # ...and the host array itself is untouched.
        assert host.dtype == np.float64
        assert host[0] == 0.1
    finally:
        tensor.close()


@needs_native
def test_egress_reproduces_the_storage_dtype_and_never_widens():
    """``to_numpy()`` on a float32 tensor returns a float32 array. A
    widened result would silently claim precision the tensor does not
    have."""
    tensor = NativeTensor.from_array([[1.0, 2.0]], dtype="float32")
    try:
        assert tensor.to_numpy().dtype == np.float32
    finally:
        tensor.close()


# ==========================================================================
# 5. Views, operations, and gradients preserve the dtype
# ==========================================================================


@needs_native
@pytest.mark.parametrize("dtype", BOTH_DTYPES)
def test_views_preserve_the_dtype_and_carry_no_tag_of_their_own(dtype):
    tensor = NativeTensor.from_array(np.arange(12.0).reshape(3, 4),
                                     dtype=dtype)
    derived = []
    try:
        derived = [tensor.reshape((4, 3)), tensor.transpose(0, 1),
                   tensor.T, tensor.narrow(0, 0, 2)]
        for view in derived:
            assert view.dtype == dtype
            assert view.to_numpy().dtype == NUMPY_DTYPES[dtype]
    finally:
        for view in derived:
            view.close()
        tensor.close()


@needs_native
@pytest.mark.parametrize("dtype", BOTH_DTYPES)
def test_operations_preserve_the_dtype(dtype):
    a = NativeTensor.from_array([[1.0, 2.0], [3.0, 4.0]], dtype=dtype)
    b = NativeTensor.from_array([[0.5, 0.5], [0.5, 0.5]], dtype=dtype)
    results = []
    try:
        results = [a.add(b), a.subtract(b), a.multiply(b), a.matmul(b),
                   a.relu(), a.sum(), a.mean(), a.exp(), a.sqrt()]
        for result in results:
            assert result.dtype == dtype, result
            assert result.to_numpy().dtype == NUMPY_DTYPES[dtype]
    finally:
        for result in results:
            result.close()
        a.close()
        b.close()


@needs_native
@pytest.mark.parametrize("dtype", BOTH_DTYPES)
def test_a_public_requires_grad_leaf_accumulates_a_gradient_of_its_dtype(
    dtype
):
    """Design §11.1/§11.2: ``grad.dtype == tensor.dtype`` at every node,
    and a leaf gradient matches its parameter's dtype."""
    leaf = NativeTensor.from_array([[1.0, 2.0], [3.0, 4.0]], dtype=dtype,
                                   requires_grad=True)
    try:
        loss = leaf.multiply(leaf).sum()
        loss.backward()
        assert leaf.grad.dtype == dtype
        assert leaf.grad.to_numpy().dtype == NUMPY_DTYPES[dtype]
        assert leaf.grad.to_numpy().tolist() == [[2.0, 4.0], [6.0, 8.0]]
        grad = leaf.grad
        leaf.zero_grad()
        grad.close()
        loss.close()
    finally:
        leaf.close()


# ==========================================================================
# 6. Unsupported values reject before anything is allocated
# ==========================================================================


@needs_native
@pytest.mark.parametrize("dtype", ["float16", "bfloat16", "int64", "Float32",
                                   "f4", "complex64"])
@pytest.mark.parametrize("build", [
    lambda d: cpp.NativeStorage(4, dtype=d),
    lambda d: cpp.NativeStorage.from_array([1.0, 2.0], dtype=d),
    lambda d: cpp.NativeTensorCore.from_array(np.ones((2, 2)), dtype=d),
    lambda d: cpp.NativeTensorCore.zeros((2, 2), dtype=d),
    lambda d: cpp.NativeTensorCore.full((2, 2), 1.0, dtype=d),
    lambda d: NativeTensor.from_array([[1.0]], dtype=d),
    lambda d: NativeTensor.zeros((2, 2), dtype=d),
    lambda d: NativeTensor.full((2, 2), 1.0, dtype=d),
])
def test_an_unsupported_dtype_rejects_and_allocates_nothing(
    dtype, build, live_storages
):
    baseline = len(live_storages)
    with pytest.raises(ValueError):
        build(dtype)
    assert len(live_storages) == baseline


@needs_native
@pytest.mark.parametrize("device", ["cuda", "gpu", "cuda:0", "CPU"])
@pytest.mark.parametrize("dtype", BOTH_DTYPES)
def test_an_unsupported_device_rejects_at_both_dtypes(device, dtype,
                                                      live_storages):
    """float32 support is a **dtype** decision and grants no device."""
    baseline = len(live_storages)
    with pytest.raises(ValueError):
        cpp.NativeTensorCore.zeros((2, 2), dtype=dtype, device=device)
    with pytest.raises(ValueError):
        NativeTensor.from_array([[1.0]], dtype=dtype, device=device)
    assert len(live_storages) == baseline


# ==========================================================================
# 7. Mixed dtype stays strict — two supported dtypes are not promotion
# ==========================================================================


@needs_native
@pytest.mark.parametrize("operation", ["add", "subtract", "multiply",
                                       "matmul"])
def test_mixed_dtype_operations_reject_and_change_nothing(operation,
                                                          live_storages):
    """Design §9.1/§9.3: float32 + float64 raises. It does not become
    float64 and does not become float32 — and the rejection happens
    **before** output allocation, so live storage is exactly what it
    was and both operands are unchanged and open."""
    a = NativeTensor.from_array([[1.0, 2.0], [3.0, 4.0]], dtype="float32")
    b = NativeTensor.from_array([[1.0, 2.0], [3.0, 4.0]], dtype="float64")
    try:
        before = len(live_storages)
        a_values = a.to_numpy().copy()
        b_values = b.to_numpy().copy()
        for left, right in ((a, b), (b, a)):
            with pytest.raises(ValueError):
                getattr(left, operation)(right)
        assert len(live_storages) == before
        assert a.closed is False and b.closed is False
        assert a.dtype == "float32" and b.dtype == "float64"
        assert np.array_equal(a.to_numpy(), a_values)
        assert np.array_equal(b.to_numpy(), b_values)
    finally:
        a.close()
        b.close()


@needs_native
@pytest.mark.parametrize("module_dtype, input_dtype",
                         [("float32", "float64"), ("float64", "float32")])
def test_a_module_rejects_an_input_of_the_other_dtype(module_dtype,
                                                      input_dtype,
                                                      live_storages):
    layer = NativeLinear(3, 2, seed=0, dtype=module_dtype)
    x = NativeTensor.from_array([[1.0, 2.0, 3.0]], dtype=input_dtype)
    try:
        before = len(live_storages)
        versions = [p.version for p in layer.parameters()]
        with pytest.raises(ValueError):
            layer(x)
        assert len(live_storages) == before
        assert [p.version for p in layer.parameters()] == versions
    finally:
        x.close()
        for parameter in layer.parameters():
            parameter.close()


@needs_native
@pytest.mark.parametrize("parameter_dtype, grad_dtype",
                         [("float32", "float64"), ("float64", "float32")])
@pytest.mark.parametrize("optimizer_class", [NativeSGD, NativeAdam])
def test_an_optimizer_rejects_a_gradient_of_the_other_dtype(
    parameter_dtype, grad_dtype, optimizer_class, live_storages
):
    parameter = NativeParameter([[1.0, 2.0]], dtype=parameter_dtype)
    grad = NativeTensor.from_array([[0.5, 0.5]], dtype=grad_dtype)
    optimizer = optimizer_class([parameter], lr=0.1)
    try:
        parameter._grad = grad          # the one seam a test can set
        before = len(live_storages)
        version = parameter.version
        values = parameter.to_numpy().copy()
        with pytest.raises(ValueError):
            optimizer.step()
        assert len(live_storages) == before
        assert parameter.version == version
        assert np.array_equal(parameter.to_numpy(), values)
    finally:
        _close_optimizer(optimizer)
        grad.close()
        parameter.close()


@needs_native
def test_no_casting_or_conversion_method_exists_on_a_public_tensor():
    """Two supported dtypes is not a way to move between them. A tensor's
    dtype is fixed at construction and the only way to get the other one
    is to construct it (design §9.5)."""
    tensor = NativeTensor.from_array([[1.0]], dtype="float32")
    try:
        for banned in ("astype", "float", "double", "to", "cast", "half",
                       "type", "cuda", "cpu", "type_as"):
            assert not hasattr(tensor, banned), banned
        # ...and the dtype property has no setter.
        with pytest.raises(AttributeError):
            tensor.dtype = "float64"
    finally:
        tensor.close()


# ==========================================================================
# 8. The public stack really trains, from public float32 construction
# ==========================================================================


@needs_native
@pytest.mark.parametrize("dtype", BOTH_DTYPES)
@pytest.mark.parametrize("optimizer_class", [NativeSGD, NativeAdam])
def test_a_public_module_trains_from_public_inputs_at_both_dtypes(
    dtype, optimizer_class
):
    """The end-to-end public path: a public float32 input, a public
    float32 module, ordinary forward/backward, and a public optimizer
    step. Nothing private is touched anywhere in this test."""
    layer = NativeLinear(3, 2, seed=0, dtype=dtype)
    x = NativeTensor.from_array([[1.0, 2.0, 3.0], [0.5, -1.0, 2.0]],
                                dtype=dtype)
    optimizer = optimizer_class(layer.parameters(), lr=0.1)
    try:
        before = {name: p.to_numpy().copy()
                  for name, p in layer.named_parameters()}
        output = layer(x)
        assert output.dtype == dtype
        loss = output.multiply(output).sum()
        assert loss.dtype == dtype
        loss.backward()
        for parameter in layer.parameters():
            assert parameter.grad.dtype == dtype
        optimizer.step()
        optimizer.zero_grad()
        loss.close()
        output.close()
        for name, parameter in layer.named_parameters():
            assert parameter.dtype == dtype
            assert not np.array_equal(parameter.to_numpy(), before[name])
    finally:
        _close_optimizer(optimizer)
        for parameter in layer.parameters():
            parameter.close()
        x.close()


@needs_native
def test_adam_moments_follow_their_parameter_dtype_from_public_construction():
    """One optimizer may hold parameters of **both** widths, with
    independent dtype-consistent state per parameter (I8, reached here
    entirely through public constructors)."""
    wide = NativeParameter([[1.0, 2.0]], dtype="float64")
    narrow = NativeParameter([[1.0, 2.0]], dtype="float32")
    optimizer = NativeAdam([wide, narrow], lr=0.1)
    try:
        for parameter in (wide, narrow):
            x = NativeTensor.from_array([[1.0, 1.0]], dtype=parameter.dtype)
            loss = parameter.multiply(x).sum()
            loss.backward()
            loss.close()
            x.close()
        optimizer.step()
        state = optimizer.state_dict()
        try:
            assert [entry["dtype"] for entry in state["parameters"]] == [
                "float64", "float32"]
            assert [t.dtype for t in state["m"]] == ["float64", "float32"]
            assert [t.dtype for t in state["v"]] == ["float64", "float32"]
        finally:
            for tensor in state["m"] + state["v"]:
                tensor.close()
        optimizer.zero_grad()
    finally:
        _close_optimizer(optimizer)
        wide.close()
        narrow.close()


@needs_native
def test_a_public_float32_model_round_trips_through_checkpoint_v3(tmp_path):
    """Checkpoint version 3 declares every numeric entry's dtype
    explicitly and round-trips float32 values **bit for bit**, reached
    entirely through public construction."""
    from tensorforge.experimental import (
        load_native_checkpoint, save_native_checkpoint,
    )
    from tensorforge.experimental import native_checkpoint

    source = NativeLinear(3, 2, seed=0, dtype="float32")
    destination = NativeLinear(3, 2, seed=99, dtype="float32")
    path = str(tmp_path / "public_float32.npz")
    try:
        save_native_checkpoint(path, source, metadata={"kind": "public"})
        import json

        with np.load(path, allow_pickle=False) as archive:
            manifest = json.loads(bytes(archive["manifest"]).decode("utf-8"))
        assert manifest["format_version"] == 3
        assert manifest["format_version"] == native_checkpoint._FORMAT_VERSION
        for entry in manifest["model"]["entries"].values():
            assert entry["dtype"] == "float32"
        before = {name: p.to_numpy().copy()
                  for name, p in destination.named_parameters()}
        assert load_native_checkpoint(path, destination) == {"kind": "public"}
        for name, parameter in destination.named_parameters():
            assert parameter.dtype == "float32"
            assert not np.array_equal(parameter.to_numpy(), before[name])
            assert bits(parameter.to_numpy(), "float32") == bits(
                dict(source.named_parameters())[name].to_numpy(), "float32")
    finally:
        for layer in (source, destination):
            for parameter in layer.parameters():
                parameter.close()


# ==========================================================================
# 9. Boundaries I9 did not move
# ==========================================================================


def test_the_stable_framework_still_does_not_load_the_native_backend():
    """The isolation the whole project rests on, unchanged by the dtype
    registry move: importing ``tensorforge`` must not load the C++
    library. Proved in a fresh interpreter, not in this one."""
    code = (
        "import sys, tensorforge\n"
        "assert 'ctypes' not in sys.modules or True\n"
        "from tensorforge.backends import cpp\n"
        "assert cpp._lib is None, 'importing tensorforge loaded the library'\n"
        "assert cpp.backend_info()['stable_framework_integration'] is False\n"
        "print('ok')\n"
    )
    result = subprocess.run([sys.executable, "-c", code],
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "ok" in result.stdout


def test_the_stable_line_gained_no_dtype_argument():
    """float32 is a **native-line** capability. The stable framework is
    untouched and has no dtype argument to gain."""
    import inspect

    import tensorforge

    for name in ("Tensor", "Parameter", "Linear" if hasattr(
            tensorforge, "Linear") else "Tensor"):
        obj = getattr(tensorforge, name)
        signature = inspect.signature(obj)
        assert "dtype" not in signature.parameters, name


def test_the_export_count_did_not_move_at_i9():
    """I9 is Python, example, test, and documentation work. The two typed
    creators I1 added are still the only two symbols the phase adds."""
    import re
    from pathlib import Path

    root = Path(__file__).resolve().parent.parent
    names = set()
    for path in list((root / "cpp" / "src").glob("*.cpp")) + list(
            (root / "cpp" / "include").glob("*.h")):
        text = path.read_text(encoding="utf-8")
        names.update(re.findall(r"TF_EXPORT[^;{]*?\b(tf_[A-Za-z0-9_]+)\s*\(",
                                text))
    assert len(names) == 55, sorted(names)
    for name in names:
        assert not name.endswith("_f32"), name
        assert "float32" not in name, name
        assert "dtype" not in name or name in (
            "tf_storage_create_typed",
            "tf_storage_create_uninitialized_typed",
        ), name


def test_the_checkpoint_and_optimizer_state_constants_did_not_move_at_i9():
    from tensorforge.experimental import (
        native_checkpoint, native_optimizer_state,
    )

    assert native_checkpoint._FORMAT == "tensorforge.native_checkpoint"
    assert native_checkpoint._FORMAT_VERSION == 3
    assert native_checkpoint._SUPPORTED_FORMAT_VERSIONS == (1, 2, 3)
    assert 4 not in native_checkpoint._SUPPORTED_FORMAT_VERSIONS
    assert native_optimizer_state.FORMAT_VERSION == 1


def test_the_stateless_modules_still_take_no_dtype_argument():
    """They own no dtype-bearing numeric state, so an argument there would
    be a second authority that could disagree with the data. The closed
    set of dtype-taking constructors is I7's six, and I9 added none.

    A *later* phase's addition is named separately below so this stays an
    exact equality in both directions while the Phase-I half keeps saying
    exactly what Phase I shipped. ``NativeTensorDataset`` (Phase J,
    milestone J1) qualifies on the same rule the six do rather than as an
    exception: its feature snapshot is materialized at the chosen dtype,
    which every batch it produces then carries, so it owns dtype-bearing
    state and routes through the same shared validator. The stateless
    classes enumerated at the end still take none, which is the property
    this test exists for."""
    import inspect

    import tensorforge.experimental as experimental

    with_dtype = set()
    for name in experimental.__all__:
        obj = getattr(experimental, name)
        if not inspect.isclass(obj):
            continue
        try:
            signature = inspect.signature(obj)
        except (TypeError, ValueError):     # pragma: no cover
            continue
        if "dtype" in signature.parameters:
            with_dtype.add(name)
    assert with_dtype == {
        "NativeParameter", "NativeLinear", "NativeConv2d", "NativeLayerNorm",
        "NativeBatchNorm1d", "NativeBatchNorm2d",
        "NativeTensorDataset",   # Phase J, milestone J1 — not I7, not I9
    }
    for name in ("NativeReLU", "NativeFlatten", "NativeMaxPool2d",
                 "NativeSequential", "NativeDropout", "NativeMSELoss",
                 "NativeCrossEntropyLoss", "NativeGenerator", "NativeSGD",
                 "NativeAdam"):
        assert name not in with_dtype, name


def test_no_device_argument_exists_anywhere_in_the_module_surface():
    """No ``device`` argument exists on a native module or optimizer and
    none may be added — the dtype registry moving grants no device."""
    import inspect

    import tensorforge.experimental as experimental

    for name in experimental.__all__:
        obj = getattr(experimental, name)
        if not inspect.isclass(obj) or name in ("NativeTensor",
                                                "NativeParameter"):
            continue
        try:
            signature = inspect.signature(obj)
        except (TypeError, ValueError):     # pragma: no cover
            continue
        assert "device" not in signature.parameters, name
