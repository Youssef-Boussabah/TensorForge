"""Float32 optimizer state and native checkpoint version 3 (Phase I,
milestone I8 — see docs/native_dtype_float32_design.md §15, §16, §17).

I8 finishes the state-bearing half of the dtype stack. Everything before
it made *computation* dtype-general; this milestone makes the things that
**own state across a step or a file** dtype-general too:

- ``NativeSGD`` and ``NativeAdam`` execute at float32, with Adam's ``m``
  and ``v`` at the parameter's own width and one optimizer free to hold
  parameters of both widths at once;
- design §15.3's open question — whether H4's Python bias-correction
  reciprocal is still an exact substitution once the coefficient is
  narrowed — is **resolved by measurement** here rather than assumed;
- native checkpoint **format version 3** declares every numeric entry's
  dtype explicitly, so a float32 model round-trips bit for bit, while
  versions 1 and 2 stay float64-only formats permanently.

What I8 deliberately does **not** move is the public registry: float32 is
still absent from ``SUPPORTED_DTYPES`` and still listed in ``UNSUPPORTED``,
and no public constructor builds a float32 tensor. That is milestone I9.
So every float32 object here is built through the private typed entry
points (``NativeParameter(..., dtype="float32")`` from I7, and the
``_typed_*`` constructors), exactly as design §27.3 requires of I1-I8.

NumPy appears as the bit-level oracle and as the serialization boundary,
never as the computation: every optimizer update runs natively.

Selector: python -m pytest -q -k "native_float32_state"
"""

import ast
import gc
import json
import pathlib
import struct

import numpy as np
import pytest

from tensorforge.backends import cpp
from tensorforge.experimental import (
    NativeAdam,
    NativeBatchNorm1d,
    NativeLinear,
    NativeParameter,
    NativeSequential,
    NativeSGD,
    NativeTensor,
    load_native_checkpoint,
    native_checkpoint,
    native_optimizer_state,
    save_native_checkpoint,
)
from tensorforge.experimental import native_adam as native_adam_module

needs_native = pytest.mark.skipif(
    not cpp.is_available(),
    reason="experimental C++ backend not built",
)

Core = cpp.NativeTensorCore

BOTH_DTYPES = ("float64", "float32")
_NUMPY = {"float64": np.float64, "float32": np.float32}
_UNSIGNED = {"float64": np.uint64, "float32": np.uint32}

LR = 0.1
BETAS = (0.9, 0.999)
EPS = 1e-8


class _Boom(Exception):
    """An injected failure that is not a MemoryError, so it can never be
    confused with a real allocation problem."""


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _bits(array, dtype):
    """``array``'s raw IEEE-754 object representations, as unsigned ints."""
    assert array.dtype == np.dtype(dtype), (array.dtype, dtype)
    return np.ascontiguousarray(array).view(_UNSIGNED[dtype])


def _same_bits(got, expected, dtype):
    return np.array_equal(_bits(got, dtype), _bits(expected, dtype))


def _tensor(values, dtype):
    """A plain graph-free NativeTensor at ``dtype`` — through the private
    typed ingress, because no public constructor makes float32 before I9."""
    array = np.asarray(values, dtype=_NUMPY[dtype])
    return NativeTensor._from_core(Core._typed_from_array(array, dtype))


def _parameter(values, dtype, requires_grad=True):
    return NativeParameter(
        np.asarray(values, dtype=_NUMPY[dtype]),
        requires_grad, dtype=dtype,
    )


def _set_grad(parameter, grad_values):
    """Give ``parameter`` exactly ``grad_values`` as its gradient, through a
    real backward: d(sum(p * c))/dp = c. No gradient is ever fabricated."""
    parameter.zero_grad()
    out = parameter.multiply(_tensor(grad_values, parameter.dtype)).sum()
    try:
        out.backward()
    finally:
        out.close()


def _close_module(module):
    """Release every native object a module owns. Modules have no
    ``close()`` — lifetime stays with the tensors — so this is the shape
    the Phase-F and Phase-I suites already use."""
    for parameter in module.parameters():
        parameter.close()
    for buffer in module.buffers():
        buffer.close()


def _close_state(state):
    for label in ("m", "v"):
        for snapshot in state.get(label, ()):
            snapshot.close()


@pytest.fixture
def live_storages(monkeypatch):
    """The ids of every NativeStorage currently open — the project's
    supported deterministic allocation-lifetime instrumentation."""
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


# ---------------------------------------------------------------------------
# 1. NativeSGD at both widths
# ---------------------------------------------------------------------------


@needs_native
@pytest.mark.parametrize("dtype", BOTH_DTYPES)
def test_sgd_updates_a_parameter_at_its_own_width(dtype):
    """``value - lr * grad``, computed and stored at the parameter's dtype.

    Checked against a NumPy oracle **at the same width** and compared as
    raw bits, so a float32 result computed in float64 and narrowed at the
    end would not pass by accident."""
    values = np.array([1.0, -2.0, 3.5, 0.25], dtype=_NUMPY[dtype])
    grads = np.array([0.5, 0.25, -0.125, 2.0], dtype=_NUMPY[dtype])
    parameter = _parameter(values, dtype)
    try:
        _set_grad(parameter, grads)
        assert parameter.grad.dtype == dtype
        optimizer = NativeSGD([parameter], lr=LR)
        optimizer.step()
        expected = values - _NUMPY[dtype](LR) * grads
        got = parameter.to_numpy()
        assert got.dtype == np.dtype(dtype)
        assert _same_bits(got, expected, dtype)
        assert parameter.version == 1
    finally:
        parameter.close()


@needs_native
def test_sgd_handles_a_mixed_dtype_collection_independently():
    """One optimizer, both widths, each parameter updated at its own — and
    the two never meet, because SGD holds no shared tensor state."""
    a = _parameter([1.0, 2.0], "float32")
    b = _parameter([3.0, 4.0], "float64")
    c = _parameter([5.0, 6.0], "float32")
    try:
        for parameter in (a, b, c):
            _set_grad(parameter, [1.0, 1.0])
        optimizer = NativeSGD([a, b, c], lr=LR)
        optimizer.step()
        for parameter, start in ((a, 1.0), (b, 3.0), (c, 5.0)):
            expected = np.array(
                [start - LR, start + 1.0 - LR], dtype=_NUMPY[parameter.dtype]
            )
            assert _same_bits(parameter.to_numpy(), expected, parameter.dtype)
            assert parameter.version == 1
        metadata = optimizer.state_dict()["parameters"]
        assert [entry["dtype"] for entry in metadata] == [
            "float32", "float64", "float32"
        ]
    finally:
        for parameter in (a, b, c):
            parameter.close()


@needs_native
def test_sgd_builds_one_scalar_per_active_dtype_not_per_parameter():
    """H4's once-per-step scalar survives at two widths: the cache key is
    ``(dtype, device)``, so N parameters of one width still build **one**
    scalar and a mixed collection builds exactly two."""
    def scalars_built(parameters):
        optimizer = NativeSGD(parameters, lr=LR)
        for parameter in parameters:
            _set_grad(parameter, [1.0])
        calls = {"n": 0}
        real = Core._typed_full

        def counting(*args, **kwargs):
            calls["n"] += 1
            return real(*args, **kwargs)

        patcher = pytest.MonkeyPatch()
        patcher.setattr(Core, "_typed_full", counting)
        try:
            optimizer.step()
        finally:
            patcher.undo()
        return calls["n"]

    for dtype in BOTH_DTYPES:
        for count in (1, 2, 5):
            parameters = [_parameter([1.0], dtype) for _ in range(count)]
            try:
                assert scalars_built(parameters) == 1, (dtype, count)
            finally:
                for parameter in parameters:
                    parameter.close()

    mixed = [_parameter([1.0], "float32"), _parameter([1.0], "float64"),
             _parameter([1.0], "float32"), _parameter([1.0], "float64")]
    try:
        assert scalars_built(mixed) == 2
    finally:
        for parameter in mixed:
            parameter.close()


@needs_native
@pytest.mark.parametrize("dtype", BOTH_DTYPES)
def test_sgd_rejects_a_mismatched_gradient_in_both_directions(dtype):
    """A float32 parameter demands a float32 gradient and a float64 one a
    float64 gradient. The rejection is atomic: nothing staged, no version."""
    other = "float64" if dtype == "float32" else "float32"
    parameter = _parameter([1.0, 2.0], dtype)
    foreign = _tensor([0.5, 0.5], other)
    try:
        _set_grad(parameter, [1.0, 1.0])
        # Swap in a gradient of the wrong width through the same private
        # attribute the engine's own backward writes.
        parameter._grad = foreign
        before = parameter.to_numpy().copy()
        optimizer = NativeSGD([parameter], lr=LR)
        with pytest.raises(ValueError, match="dtype"):
            optimizer.step()
        assert _same_bits(parameter.to_numpy(), before, dtype)
        assert parameter.version == 0
        assert parameter.grad is foreign and not foreign.closed
    finally:
        foreign.close()
        parameter.close()


@needs_native
@pytest.mark.parametrize("dtype", BOTH_DTYPES)
def test_sgd_skips_frozen_and_gradientless_parameters_at_both_widths(dtype):
    # Built trainable so a real backward can leave a **stale** gradient on
    # it, then frozen: a frozen parameter with a gradient must never update.
    frozen = _parameter([1.0], dtype)
    gradientless = _parameter([2.0], dtype)
    active = _parameter([3.0], dtype)
    try:
        _set_grad(active, [1.0])
        _set_grad(frozen, [1.0])
        frozen._requires_grad = False
        optimizer = NativeSGD([frozen, gradientless, active], lr=LR)
        optimizer.step()
        assert frozen.version == 0 and gradientless.version == 0
        assert active.version == 1
        assert _same_bits(active.to_numpy(),
                          np.array([3.0 - LR], dtype=_NUMPY[dtype]), dtype)
    finally:
        for parameter in (frozen, gradientless, active):
            parameter.close()


@needs_native
@pytest.mark.parametrize("dtype", BOTH_DTYPES)
def test_sgd_a_present_zero_gradient_still_commits_at_both_widths(dtype):
    """A numerically unchanged update is still an update: the owned value
    was replaced, so the version moves."""
    parameter = _parameter([1.0, 2.0], dtype)
    try:
        _set_grad(parameter, [0.0, 0.0])
        before = parameter.to_numpy().copy()
        NativeSGD([parameter], lr=LR).step()
        assert _same_bits(parameter.to_numpy(), before, dtype)
        assert parameter.version == 1
    finally:
        parameter.close()


@needs_native
@pytest.mark.parametrize("dtype", BOTH_DTYPES)
@pytest.mark.parametrize("index", [1, 2, 3])
@pytest.mark.parametrize("error", [MemoryError, _Boom])
def test_sgd_a_staged_failure_leaves_every_width_untouched(
    dtype, index, error, live_storages
):
    """A failure at the first, middle, or last staged update — and at the
    scalar allocation itself — changes no value, version, or gradient, and
    strands no storage."""
    parameters = [_parameter([float(i) + 1.0], dtype) for i in range(3)]
    try:
        for parameter in parameters:
            _set_grad(parameter, [0.5])
        optimizer = NativeSGD(parameters, lr=LR)
        before = [p.to_numpy().copy() for p in parameters]
        grads = [p.grad for p in parameters]
        baseline = len(live_storages)

        calls = {"n": 0}
        real = Core.subtract

        def failing(*args, **kwargs):
            calls["n"] += 1
            if calls["n"] == index:
                raise error("injected sgd failure")
            return real(*args, **kwargs)

        patcher = pytest.MonkeyPatch()
        patcher.setattr(Core, "subtract", failing)
        try:
            with pytest.raises(error, match="injected sgd failure"):
                optimizer.step()
        finally:
            patcher.undo()

        for parameter, values, grad in zip(parameters, before, grads):
            assert _same_bits(parameter.to_numpy(), values, dtype)
            assert parameter.version == 0
            assert parameter.grad is grad and not grad.closed
        assert len(live_storages) <= baseline
        # ...and the same optimizer completes the step it was denied.
        optimizer.step()
        assert all(p.version == 1 for p in parameters)
    finally:
        for parameter in parameters:
            parameter.close()


@needs_native
def test_sgd_a_scalar_allocation_failure_at_float32_changes_nothing():
    parameter = _parameter([1.0], "float32")
    try:
        _set_grad(parameter, [1.0])
        optimizer = NativeSGD([parameter], lr=LR)
        real = Core._typed_full

        def failing(*args, **kwargs):
            raise MemoryError("injected scalar failure")

        patcher = pytest.MonkeyPatch()
        patcher.setattr(Core, "_typed_full", failing)
        try:
            with pytest.raises(MemoryError, match="injected scalar failure"):
                optimizer.step()
        finally:
            patcher.undo()
        assert parameter.version == 0
        assert _same_bits(parameter.to_numpy(),
                          np.array([1.0], dtype=np.float32), "float32")
        assert real is not None
    finally:
        parameter.close()


@needs_native
@pytest.mark.parametrize("dtype", BOTH_DTYPES)
def test_sgd_state_dict_round_trips_at_both_widths(dtype):
    parameter = _parameter([1.0, 2.0], dtype)
    try:
        optimizer = NativeSGD([parameter], lr=LR)
        state = optimizer.state_dict()
        assert state["format_version"] == native_optimizer_state.FORMAT_VERSION
        assert state["parameters"][0]["dtype"] == dtype
        other = NativeSGD([parameter], lr=0.9)
        other.load_state_dict(state)
        assert other.lr == LR
        assert parameter.version == 0        # loading touches no parameter
    finally:
        parameter.close()


@needs_native
def test_sgd_shared_parameters_deduplicate_at_float32():
    parameter = _parameter([1.0, 2.0], "float32")
    try:
        _set_grad(parameter, [1.0, 1.0])
        optimizer = NativeSGD([parameter, parameter, parameter], lr=LR)
        assert len(optimizer.parameters()) == 1
        optimizer.step()
        assert parameter.version == 1        # one update, one increment
        expected = np.array([1.0 - LR, 2.0 - LR], dtype=np.float32)
        assert _same_bits(parameter.to_numpy(), expected, "float32")
    finally:
        parameter.close()


@needs_native
@pytest.mark.parametrize("dtype", BOTH_DTYPES)
def test_sgd_repeated_steps_stay_at_the_parameter_width(dtype):
    parameter = _parameter([1.0], dtype)
    try:
        optimizer = NativeSGD([parameter], lr=LR)
        value = _NUMPY[dtype](1.0)
        for _ in range(5):
            _set_grad(parameter, [1.0])
            optimizer.step()
            value = _NUMPY[dtype](value - _NUMPY[dtype](LR) * _NUMPY[dtype](1.0))
            assert parameter.to_numpy().dtype == np.dtype(dtype)
            assert _same_bits(parameter.to_numpy(),
                              np.array([value], dtype=_NUMPY[dtype]), dtype)
        assert parameter.version == 5
    finally:
        parameter.close()


# ---------------------------------------------------------------------------
# 2. NativeAdam state construction
# ---------------------------------------------------------------------------


@needs_native
@pytest.mark.parametrize("dtype", BOTH_DTYPES)
def test_adam_moments_match_their_parameter_exactly(dtype):
    """``m`` and ``v`` carry the parameter's dtype, shape and device, start
    at **positive** zero, and the counters stay Python ints at 0."""
    parameter = _parameter(np.ones((2, 3)), dtype)
    try:
        optimizer = NativeAdam([parameter], lr=LR)
        try:
            for buffer in optimizer._m + optimizer._v:
                assert buffer.dtype == dtype
                assert buffer.shape == (2, 3)
                assert buffer.device == parameter.device
                assert not isinstance(buffer, NativeParameter)
                assert buffer.requires_grad is False
                zeros = np.zeros((2, 3), dtype=_NUMPY[dtype])
                # +0.0, proved as bits: -0.0 would compare equal by value.
                assert _same_bits(buffer.to_numpy(), zeros, dtype)
            assert optimizer.step_counts == (0,)
            assert all(type(count) is int for count in optimizer.step_counts)
        finally:
            optimizer.close()
    finally:
        parameter.close()


@needs_native
def test_adam_mixed_dtype_state_is_per_parameter_and_consistent():
    a = _parameter([1.0, 2.0], "float32")
    b = _parameter(np.ones((2, 2)), "float64")
    try:
        optimizer = NativeAdam([a, b], lr=LR)
        try:
            assert [t.dtype for t in optimizer._m] == ["float32", "float64"]
            assert [t.dtype for t in optimizer._v] == ["float32", "float64"]
            assert [t.shape for t in optimizer._m] == [(2,), (2, 2)]
        finally:
            optimizer.close()
    finally:
        a.close()
        b.close()


@needs_native
def test_adam_shared_parameters_get_one_moment_pair_at_float32():
    parameter = _parameter([1.0, 2.0], "float32")
    try:
        optimizer = NativeAdam([parameter, parameter], lr=LR)
        try:
            assert len(optimizer.parameters()) == 1
            assert len(optimizer._m) == 1 and len(optimizer._v) == 1
            assert optimizer.step_counts == (0,)
        finally:
            optimizer.close()
    finally:
        parameter.close()


@needs_native
@pytest.mark.parametrize("dtype", BOTH_DTYPES)
@pytest.mark.parametrize("position", [0, 1, 2, 3])
def test_adam_a_moment_allocation_failure_releases_everything(
    dtype, position, live_storages
):
    """A failure at **every** moment position, at both widths: every buffer
    already built is closed, and the caller's parameters are untouched."""
    first = _parameter([1.0, 2.0], dtype)
    second = _parameter([3.0, 4.0], dtype)
    try:
        _set_grad(first, [1.0, 1.0])
        baseline = len(live_storages)
        created = []
        real = NativeTensor._typed_zeros

        def tracking(*args, **kwargs):
            if len(created) == position:
                raise MemoryError("injected moment failure")
            tensor = real(*args, **kwargs)
            created.append(tensor)
            return tensor

        patcher = pytest.MonkeyPatch()
        patcher.setattr(NativeTensor, "_typed_zeros", staticmethod(tracking))
        try:
            with pytest.raises(MemoryError, match="injected moment failure"):
                NativeAdam([first, second], lr=LR)
        finally:
            patcher.undo()

        assert len(created) == position
        assert all(buffer.closed for buffer in created)
        assert len(live_storages) <= baseline
        for parameter in (first, second):
            assert not parameter.closed and parameter.version == 0
        assert first.grad is not None and not first.grad.closed
    finally:
        first.close()
        second.close()


# ---------------------------------------------------------------------------
# 3. NativeAdam step arithmetic
# ---------------------------------------------------------------------------


def _adam_oracle(value, grad, m, v, t, dtype, lr=LR, betas=BETAS, eps=EPS):
    """The Adam composition, operation for operation, **entirely at
    ``dtype``** — so a hidden float64 intermediate in the runtime would
    show up as a bit difference rather than being absorbed."""
    cast = _NUMPY[dtype]
    beta1, beta2 = betas
    b1, b1c = cast(beta1), cast(1.0 - beta1)
    b2, b2c = cast(beta2), cast(1.0 - beta2)
    m_new = cast(m * b1 + grad * b1c)
    v_new = cast(v * b2 + (grad * grad) * b2c)
    # The I8 coefficient rule: narrow the denominator, then reciprocate.
    c1 = cast(1.0 / cast(1.0 - beta1 ** t))
    c2 = cast(1.0 / cast(1.0 - beta2 ** t))
    m_hat = cast(m_new * c1)
    v_hat = cast(v_new * c2)
    denominator = cast(np.sqrt(v_hat) + cast(eps))
    update = cast(cast(m_hat * cast(lr)) * cast(cast(1.0) / denominator))
    return cast(value - update), m_new, v_new


@needs_native
@pytest.mark.parametrize("dtype", BOTH_DTYPES)
def test_adam_step_computes_entirely_at_the_parameter_width(dtype):
    """Five steps against the same-width oracle, compared as raw bits."""
    cast = _NUMPY[dtype]
    values = np.array([1.0, -2.0, 0.5], dtype=cast)
    grads = np.array([0.25, 0.75, -1.5], dtype=cast)
    parameter = _parameter(values, dtype)
    try:
        optimizer = NativeAdam([parameter], lr=LR, betas=BETAS, eps=EPS)
        try:
            value = values.copy()
            m = np.zeros_like(values)
            v = np.zeros_like(values)
            for t in range(1, 6):
                _set_grad(parameter, grads)
                optimizer.step()
                value, m, v = _adam_oracle(value, grads, m, v, t, dtype)
                assert _same_bits(parameter.to_numpy(), value, dtype)
                assert _same_bits(optimizer._m[0].to_numpy(), m, dtype)
                assert _same_bits(optimizer._v[0].to_numpy(), v, dtype)
                assert optimizer._m[0].dtype == dtype
                assert optimizer._v[0].dtype == dtype
            assert optimizer.step_counts == (5,)
            assert parameter.version == 5
        finally:
            optimizer.close()
    finally:
        parameter.close()


@needs_native
def test_adam_a_mixed_dtype_step_keeps_each_entry_at_its_own_width():
    a = _parameter([1.0, 2.0], "float32")
    b = _parameter([1.0, 2.0], "float64")
    try:
        for parameter in (a, b):
            _set_grad(parameter, [0.25, 0.75])
        optimizer = NativeAdam([a, b], lr=LR, betas=BETAS, eps=EPS)
        try:
            optimizer.step()
            for index, parameter in enumerate((a, b)):
                dtype = parameter.dtype
                cast = _NUMPY[dtype]
                expected, m, v = _adam_oracle(
                    np.array([1.0, 2.0], dtype=cast),
                    np.array([0.25, 0.75], dtype=cast),
                    np.zeros(2, dtype=cast), np.zeros(2, dtype=cast),
                    1, dtype,
                )
                assert _same_bits(parameter.to_numpy(), expected, dtype)
                assert _same_bits(optimizer._m[index].to_numpy(), m, dtype)
                assert _same_bits(optimizer._v[index].to_numpy(), v, dtype)
            assert optimizer.step_counts == (1, 1)
        finally:
            optimizer.close()
    finally:
        a.close()
        b.close()


@needs_native
@pytest.mark.parametrize("dtype", BOTH_DTYPES)
def test_adam_zero_gradient_ages_state_and_skipped_parameters_do_not(dtype):
    zero_grad = _parameter([1.0], dtype)
    frozen = _parameter([1.0], dtype, requires_grad=False)
    gradientless = _parameter([1.0], dtype)
    try:
        _set_grad(zero_grad, [0.0])
        optimizer = NativeAdam([zero_grad, frozen, gradientless], lr=LR)
        try:
            optimizer.step()
            # An active zero gradient is active: state, counter and version
            # all advance even though the moments stay zero.
            assert optimizer.step_counts == (1, 0, 0)
            assert zero_grad.version == 1
            assert frozen.version == 0 and gradientless.version == 0
            zeros = np.zeros(1, dtype=_NUMPY[dtype])
            assert _same_bits(optimizer._m[0].to_numpy(), zeros, dtype)
            assert _same_bits(optimizer._v[0].to_numpy(), zeros, dtype)
        finally:
            optimizer.close()
    finally:
        for parameter in (zero_grad, frozen, gradientless):
            parameter.close()


@needs_native
@pytest.mark.parametrize("dtype", BOTH_DTYPES)
def test_adam_rejects_a_mismatched_gradient_in_both_directions(dtype):
    other = "float64" if dtype == "float32" else "float32"
    parameter = _parameter([1.0, 2.0], dtype)
    foreign = _tensor([0.5, 0.5], other)
    try:
        _set_grad(parameter, [1.0, 1.0])
        parameter._grad = foreign
        before = parameter.to_numpy().copy()
        optimizer = NativeAdam([parameter], lr=LR)
        try:
            with pytest.raises(ValueError, match="dtype"):
                optimizer.step()
            assert _same_bits(parameter.to_numpy(), before, dtype)
            assert parameter.version == 0
            assert optimizer.step_counts == (0,)
            zeros = np.zeros(2, dtype=_NUMPY[dtype])
            assert _same_bits(optimizer._m[0].to_numpy(), zeros, dtype)
        finally:
            optimizer.close()
    finally:
        foreign.close()
        parameter.close()


# ---------------------------------------------------------------------------
# 4. design §15.3 — the bias-correction coefficient, resolved on evidence
# ---------------------------------------------------------------------------
#
# H4 replaced a native `full(1 - beta**t).reciprocal()` composition with a
# Python `1.0 / (1 - beta**t)`. At binary64 that is provably the same bits.
# At binary32 it is **not**, because the kernel divides by the *narrowed*
# denominator while Python divides by the un-narrowed one — two different
# real inputs, so two legitimately different correctly-rounded results.
#
# I8 therefore does what §15.3 pre-committed to for this outcome: it
# computes the coefficient the way the kernel does. The tests below prove
# (a) the shipped value equals real native execution of the retained
# composition, at both widths; (b) the question was not vacuous — the
# pre-I8 spelling really does differ at float32; and (c) the invariant
# coefficients follow the narrow-once rule instead.


def _retained_native_correction(beta, t, dtype):
    """The pre-H4 composition, executed **natively**: materialize
    ``1 - beta ** t`` at ``dtype`` (the fill narrows once) and apply the
    native ``reciprocal`` kernel. Not an algebraic re-derivation — this is
    the same code path the optimizer used before H4."""
    scalar = Core._typed_full((), 1.0 - beta ** t, dtype)
    try:
        inverse = scalar.reciprocal()
        try:
            return float(inverse.to_numpy())
        finally:
            inverse.close()
    finally:
        scalar.close()


def _shipped_correction(beta, t, dtype):
    """What ``_StepConstants.corrections`` materializes today."""
    holder = native_adam_module._StepConstants(LR, (beta, beta), EPS)
    try:
        first, _ = holder.corrections(dtype, "cpu", t)
        return float(first.to_numpy())
    finally:
        holder.close()


_CORRECTION_CASES = [
    (0.9, t) for t in (1, 2, 3, 5, 7, 8, 12, 26, 50, 200, 2000)
] + [
    (0.999, t) for t in (1, 5, 17, 100, 1000, 10000)
] + [
    (0.0, 1), (0.0, 5),                       # beta at the low bound
    (1e-8, 1), (1e-4, 3), (0.01, 3),          # beta near 0
    (0.9999, 3), (0.99999, 7), (0.999999, 1),  # beta near 1
    (1 - 2 ** -30, 1), (1 - 2 ** -52, 1),     # the extreme admissible betas
]


@needs_native
@pytest.mark.parametrize("dtype", BOTH_DTYPES)
@pytest.mark.parametrize("beta,t", _CORRECTION_CASES)
def test_the_shipped_correction_equals_the_retained_native_composition(
    dtype, beta, t
):
    """§15.3 resolved: the coefficient the optimizer materializes is
    bit-identical to real native execution of the pre-H4 composition, at
    **both** widths."""
    shipped = _shipped_correction(beta, t, dtype)
    retained = _retained_native_correction(beta, t, dtype)
    assert struct.pack("d", shipped) == struct.pack("d", retained), (
        f"beta={beta!r} t={t} dtype={dtype}: shipped {shipped!r} != "
        f"retained {retained!r}"
    )


@needs_native
def test_the_correction_question_was_not_vacuous_at_float32():
    """The non-vacuity witness §15.3 demands.

    If the two spellings could never differ, the test above would prove
    nothing. They differ for a large fraction of ordinary inputs — the
    **default betas included** — so the equality above is a real
    constraint. ``beta1 = 0.9, t = 5`` is the smallest default-beta case."""
    beta, t = 0.9, 5
    # The pre-I8 spelling: divide the un-narrowed denominator, narrow last.
    pre_i8 = np.float32(1.0 / (1.0 - beta ** t))
    shipped = np.float32(_shipped_correction(beta, t, "float32"))
    assert pre_i8.view(np.uint32) != shipped.view(np.uint32), (
        "the witness has gone vacuous: the two spellings now agree, so "
        "the equality test above no longer constrains anything"
    )
    assert int(pre_i8.view(np.uint32)) == 0x401C48CA
    assert int(shipped.view(np.uint32)) == 0x401C48CB

    # ...and at float64 the narrowing is the identity, so H4's original
    # exact-substitution proof still holds untouched.
    assert (np.float64(1.0 / (1.0 - beta ** t))
            == _shipped_correction(beta, t, "float64"))

    # A broader sweep, so "they differ" is not resting on one lucky pair.
    differing = 0
    for case_beta, case_t in _CORRECTION_CASES:
        naive = np.float32(1.0 / (1.0 - case_beta ** case_t))
        real = np.float32(_shipped_correction(case_beta, case_t, "float32"))
        if naive.view(np.uint32) != real.view(np.uint32):
            differing += 1
    assert differing >= 3, differing


@needs_native
@pytest.mark.parametrize("dtype", BOTH_DTYPES)
def test_the_invariant_coefficients_are_narrowed_once_from_binary64(dtype):
    """``beta1``, ``1 - beta1``, ``beta2``, ``1 - beta2``, ``eps`` and
    ``lr`` are hyperparameters, not accumulators: each is computed in
    Python binary64 and narrowed **once** when materialized (design §7.4).

    Proved as bits against the single narrowing, and — the half that
    matters — proved *not* to be a float32-throughout computation, which
    for ``1 - beta`` is a visibly different value."""
    beta1, beta2 = 0.9, 0.999
    lr, eps = 0.001, 1e-8
    holder = native_adam_module._StepConstants(lr, (beta1, beta2), eps)
    try:
        cores = holder.invariants(dtype, "cpu")
        expected = (beta1, 1.0 - beta1, beta2, 1.0 - beta2, eps, lr)
        for core, value in zip(cores, expected):
            got = core.to_numpy()
            assert got.dtype == np.dtype(dtype)
            once = np.array([_NUMPY[dtype](value)], dtype=_NUMPY[dtype])
            assert _same_bits(got.reshape(1), once, dtype)
    finally:
        holder.close()

    if dtype == "float32":
        # Non-vacuity: computing 1 - beta2 at float32 throughout gives a
        # different number from narrowing the binary64 result once, so the
        # equality above genuinely pins the narrow-once rule.
        narrow_once = np.float32(1.0 - beta2)
        all_float32 = np.float32(np.float32(1.0) - np.float32(beta2))
        assert narrow_once.view(np.uint32) != all_float32.view(np.uint32)


@needs_native
def test_the_narrowing_helper_agrees_with_the_native_fill():
    """``cpp._narrowed_to_dtype`` is the Python mirror of the one narrowing
    ``tf_storage_fill`` performs. It is only trustworthy if it agrees with
    the real thing, so it is checked against it rather than reasoned about."""
    values = [0.1, 1e-8, 1.0 - 0.9 ** 5, 3.3333333333, 1e-30, 12345.678,
              float(np.nextafter(np.float32(1.0), np.float32(2.0))), 0.0, -0.0]
    for dtype in BOTH_DTYPES:
        for value in values:
            core = Core._typed_full((), value, dtype)
            try:
                filled = core.to_numpy()
            finally:
                core.close()
            mirrored = np.array(cpp._narrowed_to_dtype(value, dtype),
                                dtype=_NUMPY[dtype])
            assert _same_bits(filled.reshape(1), mirrored.reshape(1), dtype), (
                dtype, value
            )
    # At float64 it is the identity, stated rather than inferred.
    for value in values:
        assert cpp._narrowed_to_dtype(value, "float64") == value


@needs_native
def test_the_correction_cache_keys_on_dtype_and_step_count():
    """H4's caching survives I8 unchanged: invariants per ``(dtype,
    device)``, corrections per ``(dtype, device, t)``, and the two widths
    never share an entry."""
    holder = native_adam_module._StepConstants(LR, BETAS, EPS)
    try:
        f64 = holder.invariants("float64", "cpu")
        f32 = holder.invariants("float32", "cpu")
        assert holder.invariants("float64", "cpu") is f64   # cached
        assert f32 is not f64
        assert all(core.dtype == "float64" for core in f64)
        assert all(core.dtype == "float32" for core in f32)
        first = holder.corrections("float32", "cpu", 1)
        assert holder.corrections("float32", "cpu", 1) is first
        assert holder.corrections("float32", "cpu", 2) is not first
        assert holder.corrections("float64", "cpu", 1) is not first
    finally:
        holder.close()


@needs_native
def test_adam_scalar_allocations_do_not_scale_with_the_parameter_count():
    """The H4 invariant, at two widths: the scalar count is the same for
    one parameter and for many sharing a step counter, and a mixed
    collection builds one set per active dtype rather than one per
    parameter."""
    def scalars_built(parameters):
        optimizer = NativeAdam(parameters, lr=LR)
        for parameter in parameters:
            _set_grad(parameter, [0.5])
        optimizer.step()                     # settle; all share t
        for parameter in parameters:
            _set_grad(parameter, [0.5])
        calls = {"n": 0}
        real = Core._typed_full

        def counting(*args, **kwargs):
            calls["n"] += 1
            return real(*args, **kwargs)

        patcher = pytest.MonkeyPatch()
        patcher.setattr(Core, "_typed_full", counting)
        try:
            optimizer.step()
        finally:
            patcher.undo()
            optimizer.close()
        return calls["n"]

    for dtype in BOTH_DTYPES:
        counts = {}
        for count in (1, 2, 5):
            parameters = [_parameter([1.0], dtype) for _ in range(count)]
            try:
                counts[count] = scalars_built(parameters)
            finally:
                for parameter in parameters:
                    parameter.close()
        assert len(set(counts.values())) == 1, (dtype, counts)
        assert counts[1] == 8, counts     # six invariants + two corrections

    mixed = [_parameter([1.0], "float32"), _parameter([1.0], "float64"),
             _parameter([1.0], "float32")]
    try:
        assert scalars_built(mixed) == 16   # one set per active dtype
    finally:
        for parameter in mixed:
            parameter.close()


@needs_native
def test_no_scalar_survives_a_float32_step(live_storages):
    """The holder is never stored on the optimizer, so a float32 step
    leaves exactly the optimizer's own moments behind."""
    parameter = _parameter([1.0, 2.0], "float32")
    try:
        optimizer = NativeAdam([parameter], lr=LR)
        try:
            _set_grad(parameter, [0.5, 0.5])
            optimizer.step()
            gc.collect()
            settled = len(live_storages)
            for _ in range(4):
                _set_grad(parameter, [0.5, 0.5])
                optimizer.step()
                gc.collect()
                assert len(live_storages) == settled
            assert not hasattr(optimizer, "_constants")
        finally:
            optimizer.close()
    finally:
        parameter.close()


# ---------------------------------------------------------------------------
# 5. Adam in-memory optimizer state at both widths
# ---------------------------------------------------------------------------


@needs_native
@pytest.mark.parametrize("dtype", BOTH_DTYPES)
def test_adam_state_dict_snapshots_carry_the_parameter_width(dtype):
    parameter = _parameter([1.0, 2.0], dtype)
    try:
        optimizer = NativeAdam([parameter], lr=LR)
        try:
            _set_grad(parameter, [0.5, 0.5])
            optimizer.step()
            state = optimizer.state_dict()
            try:
                assert state["format_version"] == 1
                assert state["parameters"][0]["dtype"] == dtype
                assert state["m"][0].dtype == dtype
                assert state["v"][0].dtype == dtype
                assert state["step_counts"] == (1,)
                # Independent storage: closing a snapshot cannot reach the
                # optimizer, and the values are bit-equal, not merely close.
                assert state["m"][0] is not optimizer._m[0]
                assert _same_bits(state["m"][0].to_numpy(),
                                  optimizer._m[0].to_numpy(), dtype)
            finally:
                _close_state(state)
            assert not optimizer._m[0].closed
        finally:
            optimizer.close()
    finally:
        parameter.close()


@needs_native
@pytest.mark.parametrize("dtype", BOTH_DTYPES)
def test_adam_state_round_trips_bit_for_bit_at_both_widths(dtype):
    source_parameter = _parameter([1.0, -2.0, 0.5], dtype)
    target_parameter = _parameter([1.0, -2.0, 0.5], dtype)
    try:
        source = NativeAdam([source_parameter], lr=LR, betas=BETAS, eps=EPS)
        target = NativeAdam([target_parameter], lr=0.9,
                            betas=(0.5, 0.5), eps=0.5)
        try:
            for _ in range(3):
                _set_grad(source_parameter, [0.25, 0.75, -1.5])
                source.step()
            state = source.state_dict()
            try:
                version_before = target_parameter.version
                target.load_state_dict(state)
                assert target.lr == LR and target.betas == BETAS
                assert target.eps == EPS
                assert target.step_counts == source.step_counts
                for label in ("m", "v"):
                    for got, expected in zip(getattr(target, f"_{label}"),
                                             getattr(source, f"_{label}")):
                        assert got.dtype == dtype
                        assert _same_bits(got.to_numpy(),
                                          expected.to_numpy(), dtype)
                # Loading optimizer state moves no parameter version.
                assert target_parameter.version == version_before
                # The caller's snapshots are neither consumed nor aliased.
                assert all(not snapshot.closed
                           for snapshot in state["m"] + state["v"])
            finally:
                _close_state(state)
        finally:
            source.close()
            target.close()
    finally:
        source_parameter.close()
        target_parameter.close()


@needs_native
def test_adam_mixed_dtype_state_round_trips():
    a, b = _parameter([1.0, 2.0], "float32"), _parameter([3.0], "float64")
    c, d = _parameter([1.0, 2.0], "float32"), _parameter([3.0], "float64")
    try:
        source = NativeAdam([a, b], lr=LR)
        target = NativeAdam([c, d], lr=0.9)
        try:
            for parameter in (a, b):
                _set_grad(parameter, [0.5] * parameter.shape[0])
            source.step()
            state = source.state_dict()
            try:
                target.load_state_dict(state)
                assert [t.dtype for t in target._m] == ["float32", "float64"]
                for label in ("m", "v"):
                    for got, expected, dtype in zip(
                        getattr(target, f"_{label}"),
                        getattr(source, f"_{label}"),
                        ("float32", "float64"),
                    ):
                        assert _same_bits(got.to_numpy(),
                                          expected.to_numpy(), dtype)
            finally:
                _close_state(state)
        finally:
            source.close()
            target.close()
    finally:
        for parameter in (a, b, c, d):
            parameter.close()


@needs_native
def test_adam_load_rejects_a_moment_of_the_wrong_width_atomically():
    """One mismatched entry rolls back the **whole** optimizer load: the
    hyperparameters, counters and both moment collections are all still
    what they were."""
    parameter = _parameter([1.0, 2.0], "float32")
    try:
        optimizer = NativeAdam([parameter], lr=LR, betas=BETAS, eps=EPS)
        try:
            _set_grad(parameter, [0.5, 0.5])
            optimizer.step()
            state = optimizer.state_dict()
            wrong = _tensor([0.0, 0.0], "float64")
            try:
                before_m = optimizer._m[0].to_numpy().copy()
                before_steps = optimizer.step_counts
                broken = dict(state)
                broken["m"] = [wrong]
                broken["lr"] = 0.5
                with pytest.raises(ValueError, match="float64|dtype"):
                    optimizer.load_state_dict(broken)
                assert optimizer.lr == LR            # nothing committed
                assert optimizer.step_counts == before_steps
                assert optimizer._m[0].dtype == "float32"
                assert _same_bits(optimizer._m[0].to_numpy(), before_m,
                                  "float32")
                assert not wrong.closed              # caller state untouched
            finally:
                wrong.close()
                _close_state(state)
        finally:
            optimizer.close()
    finally:
        parameter.close()


# ---------------------------------------------------------------------------
# 6. Checkpoint format version 3
# ---------------------------------------------------------------------------


def _manifest_of(path):
    with np.load(path, allow_pickle=False) as archive:
        return json.loads(archive["manifest"].tobytes().decode("utf-8"))


def _arrays_of(path):
    with np.load(path, allow_pickle=False) as archive:
        return {name: archive[name] for name in archive.files}


def _rewrite(source, target, mutate):
    arrays = _arrays_of(source)
    manifest = json.loads(arrays.pop("manifest").tobytes().decode("utf-8"))
    result = mutate(manifest)
    manifest = result if result is not None else manifest
    arrays["manifest"] = np.frombuffer(
        json.dumps(manifest).encode("utf-8"), dtype=np.uint8
    )
    with open(target, "wb") as handle:
        np.savez(handle, **arrays)
    return target


def _downgrade_moments(manifest):
    section = manifest.get("optimizer")
    if isinstance(section, dict) and section.get("type") == "NativeAdam":
        for label in ("m", "v"):
            section[label] = [
                entry["array"] if isinstance(entry, dict) else entry
                for entry in section[label]
            ]
    return manifest


@needs_native
def test_the_checkpoint_constants_are_version_three():
    assert native_checkpoint._FORMAT == "tensorforge.native_checkpoint"
    assert native_checkpoint._FORMAT_VERSION == 3
    assert native_checkpoint._SUPPORTED_FORMAT_VERSIONS == (1, 2, 3)
    assert native_checkpoint._FLOAT64_ONLY_VERSIONS == (1, 2)
    # The in-memory optimizer schema is a different thing and did not move.
    assert native_optimizer_state.FORMAT_VERSION == 1


@needs_native
@pytest.mark.parametrize("dtype", BOTH_DTYPES)
def test_every_new_save_writes_version_three(tmp_path, dtype):
    """The version describes the **schema**, not the content: a float64
    model writes 3 exactly as a float32 one does."""
    module = NativeLinear(3, 2, seed=1, dtype=dtype)
    try:
        path = tmp_path / f"{dtype}.npz"
        save_native_checkpoint(path, module)
        manifest = _manifest_of(path)
        assert manifest["format_version"] == 3
        for key in ("weight", "bias"):
            assert manifest["model"]["entries"][key]["dtype"] == dtype
        arrays = _arrays_of(path)
        for name, array in arrays.items():
            if name != "manifest":
                assert array.dtype == np.dtype(dtype), name
    finally:
        _close_module(module)


@needs_native
@pytest.mark.parametrize("dtype", BOTH_DTYPES)
def test_a_model_and_optimizer_round_trip_bitwise_at_both_widths(tmp_path,
                                                                 dtype):
    """Model values, persistent BatchNorm buffers, and Adam moments all
    survive a save/load bit for bit, at both widths."""
    def build(seed):
        return NativeSequential(
            NativeLinear(4, 3, seed=seed, dtype=dtype),
            NativeBatchNorm1d(3, dtype=dtype),
        )

    source, target = build(1), build(7)
    try:
        source_optimizer = NativeAdam(source.parameters(), lr=LR)
        target_optimizer = NativeAdam(target.parameters(), lr=0.9)
        try:
            for _ in range(3):
                out = source(_tensor(np.arange(8).reshape(2, 4) * 0.25, dtype))
                loss = out.sum()
                try:
                    loss.backward()
                finally:
                    loss.close()
                    out.close()
                source_optimizer.step()
                source_optimizer.zero_grad()

            path = tmp_path / "state.npz"
            save_native_checkpoint(path, source, optimizer=source_optimizer,
                                   metadata={"epoch": 3})
            manifest = _manifest_of(path)
            assert manifest["format_version"] == 3
            for entry in manifest["optimizer"]["m"]:
                assert entry["dtype"] == dtype
                assert set(entry) == {"array", "shape", "dtype", "device"}

            metadata = load_native_checkpoint(path, target,
                                              optimizer=target_optimizer)
            assert metadata == {"epoch": 3}

            source_state = dict(source.state_dict())
            target_state = dict(target.state_dict())
            try:
                assert set(source_state) == set(target_state)
                for key, expected in source_state.items():
                    got = target_state[key]
                    assert got.dtype == dtype, key
                    assert _same_bits(got.to_numpy(), expected.to_numpy(),
                                      dtype), key
            finally:
                for snapshot in source_state.values():
                    snapshot.close()
                for snapshot in target_state.values():
                    snapshot.close()

            assert target_optimizer.lr == LR
            assert target_optimizer.step_counts == source_optimizer.step_counts
            for label in ("m", "v"):
                for got, expected in zip(
                    getattr(target_optimizer, f"_{label}"),
                    getattr(source_optimizer, f"_{label}"),
                ):
                    assert got.dtype == dtype
                    assert _same_bits(got.to_numpy(), expected.to_numpy(),
                                      dtype)
        finally:
            source_optimizer.close()
            target_optimizer.close()
    finally:
        _close_module(source)
        _close_module(target)


@needs_native
def test_a_float32_checkpoint_preserves_every_ieee754_bit(tmp_path):
    """Signed zeros, both infinities, subnormals, and NaN **payloads** all
    survive — 23 mantissa bits at binary32. A transfer performs no
    arithmetic, so payload preservation is part of the contract (§17.3)."""
    patterns = np.array([
        0x00000000,  # +0.0
        0x80000000,  # -0.0
        0x7F800000,  # +inf
        0xFF800000,  # -inf
        0x7FC00001,  # quiet NaN, payload 1
        0x7F800001,  # signalling NaN, payload 1
        0xFFC12345,  # negative quiet NaN, a distinctive payload
        0x00000001,  # smallest positive subnormal
        0x007FFFFF,  # largest subnormal
        0x00800000,  # smallest normal
        0x7F7FFFFF,  # FLT_MAX
        0x3F800001,  # 1.0 + 1 ULP
    ], dtype=np.uint32)
    values = patterns.view(np.float32)

    # NativeLinear(in, out) holds weight (in, out) — six elements here, so
    # the sweep travels the ordinary model-state path rather than as a
    # special case.
    sweep = values[:6].reshape(3, 2)
    source_model = NativeLinear(3, 2, seed=1, dtype="float32")
    try:
        source_model.weight.copy_value_(_tensor(sweep, "float32"))
        # The parameter really holds the exotic patterns before we start.
        assert np.array_equal(
            source_model.weight.to_numpy().view(np.uint32),
            sweep.view(np.uint32),
        )
        target_model = NativeLinear(3, 2, seed=2, dtype="float32")
        try:
            path = tmp_path / "bits.npz"
            save_native_checkpoint(path, source_model)
            load_native_checkpoint(path, target_model)
            assert np.array_equal(
                target_model.weight.to_numpy().view(np.uint32),
                sweep.view(np.uint32),
            )
        finally:
            _close_module(target_model)
    finally:
        _close_module(source_model)


@needs_native
@pytest.mark.parametrize("version", [1, 2])
def test_versions_one_and_two_stay_float64_only(tmp_path, version):
    """§16.5: the older formats are float64 **by definition**. A declared
    float32 entry is rejected naming the version, and a float32 payload is
    never *guessed* to be float32 — the declaration is what is read."""
    module = NativeLinear(3, 2, seed=1, dtype="float64")
    try:
        path = tmp_path / "v3.npz"
        save_native_checkpoint(path, module)

        def downgrade(manifest):
            manifest["format_version"] = version
            if version == 1:
                manifest.pop("generators", None)
            return _downgrade_moments(manifest)

        legacy = _rewrite(path, tmp_path / f"v{version}.npz", downgrade)
        # A genuine float64 legacy archive still loads, unchanged.
        target = NativeLinear(3, 2, seed=9, dtype="float64")
        try:
            load_native_checkpoint(legacy, target)
            assert _same_bits(target.weight.to_numpy(),
                              module.weight.to_numpy(), "float64")
        finally:
            _close_module(target)

        # ...but the same archive declaring float32 is refused, and the
        # message says which version and why.
        def claim_float32(manifest):
            manifest["format_version"] = version
            if version == 1:
                manifest.pop("generators", None)
            manifest["model"]["entries"]["weight"]["dtype"] = "float32"
            return _downgrade_moments(manifest)

        forged = _rewrite(path, tmp_path / f"forged{version}.npz",
                          claim_float32)
        target = NativeLinear(3, 2, seed=9, dtype="float64")
        try:
            with pytest.raises(ValueError) as error:
                load_native_checkpoint(forged, target)
            message = str(error.value)
            assert "float32" in message
            assert f"version {version}" in message
            assert target.weight.version == 0
        finally:
            _close_module(target)
    finally:
        _close_module(module)


@needs_native
def test_a_float32_model_cannot_load_a_legacy_archive(tmp_path):
    """A v1/v2 archive can only load into a float64 model — correct
    behavior, not a limitation: the format has no way to say otherwise."""
    module = NativeLinear(3, 2, seed=1, dtype="float64")
    try:
        path = tmp_path / "v3.npz"
        save_native_checkpoint(path, module)
        legacy = _rewrite(path, tmp_path / "v2.npz",
                          lambda m: (m.__setitem__("format_version", 2)
                                     or _downgrade_moments(m)))
        target = NativeLinear(3, 2, seed=5, dtype="float32")
        try:
            before = target.weight.to_numpy().copy()
            with pytest.raises(ValueError, match="dtype"):
                load_native_checkpoint(legacy, target)
            assert _same_bits(target.weight.to_numpy(), before, "float32")
            assert target.weight.version == 0
        finally:
            _close_module(target)
    finally:
        _close_module(module)


@needs_native
@pytest.mark.parametrize("case", [
    "declared-float32-payload-float64",
    "declared-float64-payload-float32",
    "unknown-dtype-string",
    "dtype-not-a-string",
    "moment-dtype-mismatch",
    "moment-entry-is-a-bare-name",
    "moment-shape-mismatch",
    "foreign-byte-order",
])
def test_the_corruption_matrix_covers_the_dtype_cases(tmp_path, case):
    """Every dtype malformation is a ``ValueError`` raised **before** any
    live state is touched, in both directions and at every entry kind."""
    module = NativeLinear(3, 2, seed=1, dtype="float32")
    try:
        optimizer = NativeAdam(module.parameters(), lr=LR)
        try:
            out = module(_tensor(np.ones((2, 3)), "float32")).sum()
            try:
                out.backward()
            finally:
                out.close()
            optimizer.step()
            optimizer.zero_grad()
            path = tmp_path / "good.npz"
            save_native_checkpoint(path, module, optimizer=optimizer)

            def mutate_manifest(manifest):
                entries = manifest["model"]["entries"]
                section = manifest["optimizer"]
                if case == "declared-float32-payload-float64":
                    pass                      # handled in mutate_arrays
                elif case == "declared-float64-payload-float32":
                    entries["weight"]["dtype"] = "float64"
                elif case == "unknown-dtype-string":
                    entries["weight"]["dtype"] = "float16"
                elif case == "dtype-not-a-string":
                    entries["weight"]["dtype"] = 1
                elif case == "moment-dtype-mismatch":
                    section["m"][0]["dtype"] = "float64"
                elif case == "moment-entry-is-a-bare-name":
                    section["m"] = [e["array"] for e in section["m"]]
                elif case == "moment-shape-mismatch":
                    section["m"][0]["shape"] = [99]
                return manifest

            target = tmp_path / f"corrupt-{case}.npz"
            arrays = _arrays_of(path)
            manifest = json.loads(
                arrays.pop("manifest").tobytes().decode("utf-8")
            )
            manifest = mutate_manifest(manifest)
            if case == "declared-float32-payload-float64":
                name = manifest["model"]["entries"]["weight"]["array"]
                arrays[name] = arrays[name].astype(np.float64)
            elif case == "foreign-byte-order":
                name = manifest["model"]["entries"]["weight"]["array"]
                arrays[name] = arrays[name].astype(
                    arrays[name].dtype.newbyteorder(">")
                )
            arrays["manifest"] = np.frombuffer(
                json.dumps(manifest).encode("utf-8"), dtype=np.uint8
            )
            with open(target, "wb") as handle:
                np.savez(handle, **arrays)

            restored = NativeLinear(3, 2, seed=4, dtype="float32")
            restore_optimizer = NativeAdam(restored.parameters(), lr=0.9)
            try:
                before = {k: p.to_numpy().copy()
                          for k, p in restored.named_parameters()}
                versions = {k: p.version
                            for k, p in restored.named_parameters()}
                moments = [t.to_numpy().copy() for t in restore_optimizer._m]
                with pytest.raises((ValueError, TypeError)):
                    load_native_checkpoint(target, restored,
                                           optimizer=restore_optimizer)
                for key, parameter in restored.named_parameters():
                    assert _same_bits(parameter.to_numpy(), before[key],
                                      "float32"), key
                    assert parameter.version == versions[key], key
                for got, expected in zip(restore_optimizer._m, moments):
                    assert _same_bits(got.to_numpy(), expected, "float32")
                assert restore_optimizer.lr == 0.9
            finally:
                restore_optimizer.close()
                _close_module(restored)
        finally:
            optimizer.close()
    finally:
        _close_module(module)


@needs_native
def test_a_v3_load_preserves_identity_aliasing_and_versions(tmp_path):
    """Restoration stays **in place**: every parameter object survives, a
    shared parameter deduplicates to one slot and one version increment,
    and the optimizer's moment identity follows its existing contract."""
    from tensorforge.experimental import NativeModule

    class Shared(NativeModule):
        """One Linear plus the **same** parameter object registered under
        two names — the smallest model whose aliasing a load could lose."""

        def __init__(self):
            super().__init__()
            self.inner = NativeLinear(2, 2, seed=1, dtype="float32")
            extra = _parameter(np.ones((2, 2)), "float32")
            self.extra = extra
            self.also = extra

    model = Shared()
    try:
        optimizer = NativeAdam(model.parameters(), lr=LR)
        try:
            identities = {k: id(p) for k, p in model.named_parameters()}
            path = tmp_path / "alias.npz"
            save_native_checkpoint(path, model, optimizer=optimizer)
            versions = {k: p.version for k, p in model.named_parameters()}
            load_native_checkpoint(path, model, optimizer=optimizer)
            for key, parameter in model.named_parameters():
                assert id(parameter) == identities[key], key
                # A full load replaces parameters, so each unique parameter
                # takes exactly one increment.
                assert parameter.version == versions[key] + 1, key
            unique = {id(p) for p in model.parameters()}
            assert len(unique) == len(model.parameters())
        finally:
            optimizer.close()
    finally:
        _close_module(model)


@needs_native
def test_a_v3_staging_failure_commits_nothing(tmp_path, live_storages):
    """The staging phase is where every allocation happens, so a failure
    there leaves the model, the optimizer and live storage exactly as they
    were — at float32 as at float64."""
    model = NativeLinear(3, 2, seed=1, dtype="float32")
    try:
        optimizer = NativeAdam(model.parameters(), lr=LR)
        try:
            path = tmp_path / "stage.npz"
            save_native_checkpoint(path, model, optimizer=optimizer)
            before = {k: p.to_numpy().copy()
                      for k, p in model.named_parameters()}
            versions = {k: p.version for k, p in model.named_parameters()}
            baseline = len(live_storages)

            calls = {"n": 0}
            real = NativeTensor._typed_from_array

            def failing(*args, **kwargs):
                calls["n"] += 1
                if calls["n"] == 2:
                    raise _Boom("injected staging failure")
                return real(*args, **kwargs)

            patcher = pytest.MonkeyPatch()
            patcher.setattr(NativeTensor, "_typed_from_array",
                            staticmethod(failing))
            try:
                with pytest.raises(_Boom, match="injected staging failure"):
                    load_native_checkpoint(path, model, optimizer=optimizer)
            finally:
                patcher.undo()

            for key, parameter in model.named_parameters():
                assert _same_bits(parameter.to_numpy(), before[key], "float32")
                assert parameter.version == versions[key]
            assert len(live_storages) <= baseline
            # ...and the same archive loads cleanly afterwards.
            load_native_checkpoint(path, model, optimizer=optimizer)
        finally:
            optimizer.close()
    finally:
        _close_module(model)


@needs_native
def test_a_v3_checkpoint_carries_no_cast_and_no_map_location():
    """The surfaces that would make a dtype mismatch survivable do not
    exist, and I8 did not add them."""
    import inspect

    assert list(inspect.signature(save_native_checkpoint).parameters) == [
        "path", "model", "optimizer", "metadata",
    ]
    assert list(inspect.signature(load_native_checkpoint).parameters) == [
        "path", "model", "optimizer",
    ]
    # Read the **code**, not the prose: this module's docstring correctly
    # says map_location does not exist, and a naive substring search would
    # trip over the very sentence that promises it.
    source = pathlib.Path(native_checkpoint.__file__).read_text(
        encoding="utf-8"
    )
    tree = ast.parse(source)
    identifiers = {
        node.id for node in ast.walk(tree) if isinstance(node, ast.Name)
    } | {
        node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)
    } | {
        keyword.arg for node in ast.walk(tree)
        if isinstance(node, ast.Call) for keyword in node.keywords
    } | {
        argument.arg for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        for argument in node.args.args + node.args.kwonlyargs
    }
    for banned in ("map_location", "astype", "cast", "device_map"):
        assert banned not in identifiers, banned
    # allow_pickle is present, and must be present as False everywhere.
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            for keyword in node.keywords:
                if keyword.arg == "allow_pickle":
                    assert isinstance(keyword.value, ast.Constant)
                    assert keyword.value.value is False


@needs_native
def test_i8_added_no_export_and_i9_is_what_moved_the_public_registry():
    """The exit-gate half I8 must **not** have moved, restated where I9
    moved it to.

    Through I8 this asserted that the registry was untouched and that no
    public constructor built a float32 tensor — the whole point of the
    rollout being that float32 optimizer state and checkpoint v3 could
    exist without a promise attached. **I9 attached the promise.** So what
    stays assertable here is the attribution: the registry is at I9's
    values, the raw-kernel row did not move with it, and the two things I8
    itself owned — the export count and the optimizer signatures — are
    exactly as I8 left them."""
    assert cpp.SUPPORTED_DTYPES == ("float64", "float32")
    assert cpp.SUPPORTED_DEVICES == ("cpu",)
    assert cpp.UNSUPPORTED == ("cuda", "amp")
    # The raw-kernel registry is a different statement and did not move
    # with the public one, at I8 or at I9.
    assert cpp.RAW_KERNEL_DTYPES == ("float64",)
    assert cpp.normalize_dtype("float32") == "float32"
    assert cpp.normalize_dtype(None) == "float64"
    # Every public constructor builds a float32 tensor since I9 — and none
    # of them infers the dtype, which is the rule I8's staging path relied
    # on and which did not change when the registry moved.
    for factory, args in ((NativeTensor.zeros, ((2,),)),
                          (NativeTensor.full, ((2,), 1.0)),
                          (NativeTensor.from_array, (np.ones(2),))):
        tensor = factory(*args, dtype="float32")
        try:
            assert tensor.dtype == "float32"
        finally:
            tensor.close()
        inferred = factory(*args)
        try:
            assert inferred.dtype == "float64"
        finally:
            inferred.close()
    # Neither optimizer gained a dtype or device argument.
    import inspect

    assert list(inspect.signature(NativeSGD.__init__).parameters) == [
        "self", "parameters", "lr",
    ]
    assert list(inspect.signature(NativeAdam.__init__).parameters) == [
        "self", "parameters", "lr", "betas", "eps",
    ]
