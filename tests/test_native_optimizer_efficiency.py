"""Tests for Phase H, milestone H4 — native optimizer step efficiency.

H4 changed *how* ``NativeAdam.step()`` and ``NativeSGD.step()`` build
their operands and release their temporaries. It changed no arithmetic,
no public API, no C++, no C ABI symbol, no registry value, and no
checkpoint format. Three things moved:

1. **The step's scalar coefficients are built once per step, not once per
   parameter.** ``beta1``, ``1 - beta1``, ``beta2``, ``1 - beta2``,
   ``eps``, and ``lr`` are identical for every parameter in a step, and
   the two bias-correction reciprocals are identical for every parameter
   sharing a step counter. A private per-step ``_StepConstants`` holder
   builds each on first use and releases every one before the commit
   begins. NativeSGD does the same for its single ``lr`` scalar.
2. **The bias-correction reciprocal is evaluated in Python.** The native
   ``reciprocal`` kernel *is* ``1.0 / x`` on the same IEEE-754 binary64
   value, and IEEE-754 division is correctly rounded, so ``1.0 / (1 -
   beta ** t)`` in Python and ``full(1 - beta ** t).reciprocal()``
   natively are the same bits — an exact substitution, not a
   reassociation.
3. **Temporaries are released at their last use** instead of at the end
   of the staged expression, which cuts the live transient buffers during
   staging from roughly seventeen parameter-sized ones to at most four.

Everything else is the pre-H4 contract, and this module's job is to hold
it: the arithmetic bit for bit, the two-phase stage/commit transaction,
mutation atomicity under injected failure at every position, parameter
and state identity, version counting, gradient non-mutation, first-step
versus steady-state behavior, duplicate and aliased parameters,
hyperparameters observed per step, state_dict and checkpoint resume, and
the absence of any new public surface.

The pre-H4 composition is **retained in this file** as the numerical
reference (``_pre_h4_adam_stage`` / ``_pre_h4_sgd_stage``). It is a
literal transcription of the shipped pre-H4 ``_stage_entry`` body, so
every equality below is against real native execution of the old
composition rather than against a NumPy re-derivation.

No test here asserts a timing, a speed, or a wall-clock threshold.

Selector: python -m pytest -q -k "optimizer_efficiency"
"""

import gc
import inspect
import math

import numpy as np
import pytest

import tensorforge
from tensorforge.backends import cpp
from tensorforge.experimental import (
    NativeAdam,
    NativeLinear,
    NativeParameter,
    NativeReLU,
    NativeSequential,
    NativeSGD,
    NativeTensor,
    load_native_checkpoint,
    save_native_checkpoint,
)
from tensorforge.experimental import native_adam as native_adam_module
from tensorforge.experimental import native_sgd as native_sgd_module

needs_native = pytest.mark.skipif(
    not cpp.is_available(),
    reason="experimental C++ backend not built",
)

Core = cpp.NativeTensorCore

LR = 0.1
BETAS = (0.9, 0.999)
EPS = 1e-8


# ======================================================================
# Fixtures and helpers
# ======================================================================


@pytest.fixture
def live_storages(monkeypatch):
    """The ids of every NativeStorage currently open — the project's
    supported deterministic instrumentation for native-allocation
    lifetime (the Phase-C/D/E/F precedent)."""
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


def _injected(target, name, replacement):
    """Patch one seam in its own monkeypatch context, so undoing it never
    also removes the ``live_storages`` tracking hooks."""
    patcher = pytest.MonkeyPatch()
    patcher.setattr(target, name, replacement)
    return patcher


def _parameter(values, grad_values=None, requires_grad=True):
    """A NativeParameter whose grad is exactly ``grad_values``:
    d(sum(p * c))/dp = c, so one backward through multiply sets it.

    A frozen parameter accumulates no gradient, so ``requires_grad=False``
    is applied *after* the gradient is set — which is exactly the "frozen
    parameter carrying a stale gradient" case the optimizer contract says
    must be skipped."""
    parameter = NativeParameter(np.asarray(values, dtype=np.float64))
    if grad_values is not None:
        source = NativeTensor.from_array(
            np.asarray(grad_values, dtype=np.float64)
        )
        try:
            parameter.multiply(source).sum().backward()
        finally:
            source.close()
    if not requires_grad:
        parameter._requires_grad = False
    return parameter


def _mixed_parameters(seed=4, shapes=((3, 2), (2,), (2, 4), (4,))):
    rng = np.random.default_rng(seed)
    return [_parameter(rng.standard_normal(shape),
                       rng.standard_normal(shape) * 0.25)
            for shape in shapes]


def _settle_wrapper_cycles():
    """Drive the graph wrappers a backward leaves behind to their
    deterministic collection point.

    They participate in reference cycles — a property of the Python-managed
    native autograd engine since Phase B, unrelated to the optimizer — so a
    live-storage baseline taken while they are still pending would drift
    downwards later and read as a *negative* leak. This runs once, at
    setup, before a baseline is taken; the accumulation assertions
    afterwards use no collection at all."""
    gc.collect()


def _bits(values):
    return np.asarray(values, dtype=np.float64).view(np.uint64)


class _Boom(Exception):
    """A failure class that is not a MemoryError, so an injected failure
    can never be confused with a real allocation problem."""


# -- the retained pre-H4 composition -----------------------------------


def _pre_h4_adam_stage(parameter, grad, m, v, beta1, beta2, lr, eps, t):
    """A literal transcription of the pre-H4 ``_stage_entry`` body,
    executed natively. Returns fresh owning ``(m_new, v_new, p_new)``
    cores the caller closes.

    Phase I, milestone I8 spells the scalar constructor ``_typed_full``
    rather than ``full`` so this one reference runs at **both** widths and
    cannot drift into two. At float64 they are the same call — ``full``
    is ``_typed_full`` behind the public dtype gate — so this is still the
    literal pre-H4 body; at float32 it is the only spelling there is,
    because no public constructor builds a float32 tensor before I9."""
    dtype, device = parameter.dtype, parameter.device

    def scalar(value):
        return Core._typed_full((), value, dtype, device=device)

    transients = []
    try:
        beta1_scalar = scalar(beta1)
        transients.append(beta1_scalar)
        decayed_m = m.multiply(beta1_scalar)
        transients.append(decayed_m)
        beta1_complement = scalar(1.0 - beta1)
        transients.append(beta1_complement)
        fresh_m = grad.multiply(beta1_complement)
        transients.append(fresh_m)
        m_new = decayed_m.add(fresh_m)

        beta2_scalar = scalar(beta2)
        transients.append(beta2_scalar)
        decayed_v = v.multiply(beta2_scalar)
        transients.append(decayed_v)
        grad_squared = grad.multiply(grad)
        transients.append(grad_squared)
        beta2_complement = scalar(1.0 - beta2)
        transients.append(beta2_complement)
        fresh_v = grad_squared.multiply(beta2_complement)
        transients.append(fresh_v)
        v_new = decayed_v.add(fresh_v)

        correction1 = scalar(1.0 - beta1 ** t)
        transients.append(correction1)
        inverse_correction1 = correction1.reciprocal()
        transients.append(inverse_correction1)
        m_hat = m_new.multiply(inverse_correction1)
        transients.append(m_hat)
        correction2 = scalar(1.0 - beta2 ** t)
        transients.append(correction2)
        inverse_correction2 = correction2.reciprocal()
        transients.append(inverse_correction2)
        v_hat = v_new.multiply(inverse_correction2)
        transients.append(v_hat)

        root = v_hat.sqrt()
        transients.append(root)
        eps_scalar = scalar(eps)
        transients.append(eps_scalar)
        denominator = root.add(eps_scalar)
        transients.append(denominator)
        inverse_denominator = denominator.reciprocal()
        transients.append(inverse_denominator)
        lr_scalar = scalar(lr)
        transients.append(lr_scalar)
        scaled_m_hat = m_hat.multiply(lr_scalar)
        transients.append(scaled_m_hat)
        update = scaled_m_hat.multiply(inverse_denominator)
        transients.append(update)
        parameter_new = parameter.subtract(update)
    finally:
        for core in transients:
            core.close()
    return m_new, v_new, parameter_new


def _pre_h4_sgd_stage(parameter, grad, lr):
    """The pre-H4 NativeSGD staging body, executed natively (dtype-general
    since Phase I milestone I8 — see ``_pre_h4_adam_stage``)."""
    scale = Core._typed_full((), lr, grad.dtype, device=grad.device)
    try:
        scaled = grad.multiply(scale)
    finally:
        scale.close()
    try:
        return parameter.subtract(scaled)
    finally:
        scaled.close()


def _committed(core):
    """The value ``copy_value_`` would install for ``core``: the commit
    path materializes through ``_native_copy``, so the reference must
    use the same materialization.

    Phase H, milestone H5 changed what that is — from ``zeros + add`` to
    the native identity gather — so this reference moved with it. The
    two agree on every value these optimizer tests produce; they differ
    only on ``-0.0`` (the addition normalized it to ``+0.0``, the gather
    preserves it) and on a signaling NaN (the addition quieted it).
    Neither is reachable from Adam or SGD arithmetic, which is why every
    pre-H4 bit-identity comparison built on this helper still holds
    exactly. The dedicated proof of the difference itself lives in
    tests/test_native_copy_transfer.py."""
    return core.contiguous_copy()


def _pre_h4_adam_run(values, grads, steps, lr, betas, eps):
    """Run the pre-H4 composition for ``steps`` steps over one parameter
    and return ``(value, m, v)`` as NumPy arrays."""
    value = np.array(values, dtype=np.float64)
    grad = np.array(grads, dtype=np.float64)
    m = np.zeros_like(value)
    v = np.zeros_like(value)
    for t in range(1, steps + 1):
        cores = [Core.from_array(value), Core.from_array(grad),
                 Core.from_array(m), Core.from_array(v)]
        m_new, v_new, p_new = _pre_h4_adam_stage(
            cores[0], cores[1], cores[2], cores[3],
            betas[0], betas[1], lr, eps, t,
        )
        committed = _committed(p_new)
        m = m_new.to_numpy().copy()
        v = v_new.to_numpy().copy()
        value = committed.to_numpy().copy()
        for core in (*cores, m_new, v_new, p_new, committed):
            core.close()
    return value, m, v


# ======================================================================
# 1. Numerical order — the arithmetic did not move
# ======================================================================


@needs_native
def test_the_native_reciprocal_kernel_is_exactly_python_division():
    """The substitution H4 makes is exact, not approximate: the kernel is
    ``1.0 / x`` in C++ ``double``, IEEE-754 division is correctly
    rounded, so there is one possible result and both spellings give it.
    Asserted on raw bit patterns, never a tolerance."""
    rng = np.random.default_rng(20260301)
    sweep = list(rng.standard_normal(4000)
                 * np.float_power(10.0, rng.integers(-300, 300, 4000)))
    sweep += [1.0, -1.0, 0.5, -0.5, 3.0, 0.1, 0.9, 0.999, 1e16,
              1e-308, 5e-324, -5e-324, 1.7976931348623157e308,
              -1.7976931348623157e308, math.inf, -math.inf, 0.0, -0.0]
    # ...and the coefficients Adam actually forms.
    for beta in (0.0, 0.5, 0.9, 0.999, 0.99999, 1 - 2 ** -52):
        for t in (1, 2, 3, 10, 1000, 100000):
            sweep.append(1.0 - beta ** t)
    values = np.array(sweep, dtype=np.float64)

    core = Core.from_array(values)
    try:
        result = core.reciprocal()
        try:
            native = result.to_numpy()
        finally:
            result.close()
    finally:
        core.close()
    with np.errstate(divide="ignore", over="ignore"):
        in_python = np.float64(1.0) / values
    assert np.array_equal(_bits(native), _bits(in_python))


@needs_native
@pytest.mark.parametrize("shape", [(), (1,), (2, 2), (3, 1, 4)])
@pytest.mark.parametrize("steps", [1, 2, 5])
@pytest.mark.parametrize("hyper", [
    (0.001, (0.9, 0.999), 1e-8),
    (0.05, (0.5, 0.9), 1e-3),
    (1e-7, (0.0, 0.0), 1e-15),
    (3.5, (0.99999, 0.9999999), 1e-30),
    (1e10, (0.123456789, 0.987654321), 2.5),
])
def test_adam_is_bit_identical_to_the_pre_h4_composition(shape, steps, hyper):
    """Every value the shipped optimizer produces — the parameter, both
    moments — matches the retained pre-H4 native composition bit for bit,
    across shapes, step counts, and default and extreme
    hyperparameters."""
    lr, betas, eps = hyper
    rng = np.random.default_rng(20260302)
    values = rng.standard_normal(shape) if shape else np.array(0.75)
    grads = (rng.standard_normal(shape) * 0.3 if shape else np.array(0.125))

    parameter = _parameter(values, grads)
    optimizer = NativeAdam([parameter], lr=lr, betas=betas, eps=eps)
    for _ in range(steps):
        optimizer.step()

    want_value, want_m, want_v = _pre_h4_adam_run(
        values, grads, steps, lr, betas, eps
    )
    assert np.array_equal(_bits(parameter.to_numpy()), _bits(want_value))
    assert np.array_equal(_bits(optimizer._m[0].to_numpy()), _bits(want_m))
    assert np.array_equal(_bits(optimizer._v[0].to_numpy()), _bits(want_v))
    optimizer.close()
    parameter.close()


@needs_native
@pytest.mark.parametrize("lr", [0.001, 3.5, 1e-9, 1e12])
def test_sgd_is_bit_identical_to_the_pre_h4_composition(lr):
    parameters = _mixed_parameters(seed=17)
    expected = []
    for parameter in parameters:
        value = Core.from_array(parameter.to_numpy())
        grad = Core.from_array(parameter.grad.to_numpy())
        try:
            staged = _pre_h4_sgd_stage(value, grad, lr)
            try:
                committed = _committed(staged)
                try:
                    expected.append(committed.to_numpy().copy())
                finally:
                    committed.close()
            finally:
                staged.close()
        finally:
            value.close()
            grad.close()

    optimizer = NativeSGD(parameters, lr=lr)
    optimizer.step()
    for parameter, want in zip(parameters, expected):
        assert np.array_equal(_bits(parameter.to_numpy()), _bits(want))
    for parameter in parameters:
        parameter.close()


@needs_native
def test_a_whole_training_run_is_bit_identical_to_the_pre_h4_composition():
    """The end-to-end guarantee the exact-resume proofs rest on: a
    multi-step run over several parameters of different shapes lands on
    exactly the bits the old composition lands on."""
    parameters = _mixed_parameters(seed=99)
    grads = [p.grad.to_numpy().copy() for p in parameters]
    values = [p.to_numpy().copy() for p in parameters]

    optimizer = NativeAdam(parameters, lr=LR, betas=BETAS, eps=EPS)
    for _ in range(6):
        optimizer.step()

    for index, parameter in enumerate(parameters):
        want_value, want_m, want_v = _pre_h4_adam_run(
            values[index], grads[index], 6, LR, BETAS, EPS
        )
        assert np.array_equal(_bits(parameter.to_numpy()), _bits(want_value))
        assert np.array_equal(_bits(optimizer._m[index].to_numpy()),
                              _bits(want_m))
        assert np.array_equal(_bits(optimizer._v[index].to_numpy()),
                              _bits(want_v))
    optimizer.close()
    for parameter in parameters:
        parameter.close()


@needs_native
def test_the_staged_expression_issues_exactly_the_pre_h4_operations():
    """The *sequence* of native compute operations per staged entry is
    unchanged: nine multiplies, two adds, one subtract, one sqrt, and one
    reciprocal, in that interleaving. H4 removed the two one-element
    reciprocals and nothing else.

    Phase H, milestone H5 removed one more entry from this list, and it
    is deliberately *not* an arithmetic one: the trailing ``add`` used to
    be ``_native_copy``'s ``zeros + add`` inside ``copy_value_`` — a
    value copy spelled as arithmetic. It is now the native identity
    gather, so the commit issues no compute operation at all. The
    arithmetic that computes the update is byte-for-byte the same
    sequence it has been since before H4, which is the invariant this
    test exists to pin."""
    recorded = []
    copies = {"n": 0}
    real_binary = Core._binary_core_op
    real_unary = Core._unary_compute
    real_copy = Core.contiguous_copy

    def binary(self, other, kernel_name, op_name):
        recorded.append(op_name)
        return real_binary(self, other, kernel_name, op_name)

    def unary(self, odometer_name, contiguous_name):
        recorded.append(odometer_name.replace("tf_core_", ""))
        return real_unary(self, odometer_name, contiguous_name)

    def copy(self):
        copies["n"] += 1
        return real_copy(self)

    parameter = _parameter([[1.0, -2.0], [0.5, 3.0]],
                           [[0.25, 0.5], [-0.75, 1.0]])
    optimizer = NativeAdam([parameter], lr=LR)
    patchers = [_injected(Core, "_binary_core_op", binary),
                _injected(Core, "_unary_compute", unary),
                _injected(Core, "contiguous_copy", copy)]
    try:
        optimizer.step()
    finally:
        for patcher in patchers:
            patcher.undo()

    assert recorded == [
        "multiply", "multiply", "add",              # m_new
        "multiply", "multiply", "multiply", "add",  # v_new
        "multiply", "multiply",                     # m_hat, v_hat
        "sqrt", "add", "reciprocal",                # denominator
        "multiply", "multiply",                     # scaled, update
        "subtract",                                 # parameter_new
    ], recorded
    # ...and the commit is exactly one value copy per updated parameter,
    # issuing no arithmetic kernel of any kind (H5).
    assert copies["n"] == 1, copies
    optimizer.close()
    parameter.close()


# ======================================================================
# 2. The per-step scalar holder
# ======================================================================


@needs_native
def test_scalar_allocations_no_longer_scale_with_the_parameter_count():
    """The semantic H4 invariant, stated as a *shape* rather than as a
    brittle absolute: the number of one-element scalar cores a step
    builds is the same for one parameter and for many, as long as they
    share a step counter."""
    counts = {}
    for parameter_count in (1, 2, 5, 9):
        parameters = [_parameter([float(i) + 1.0], [0.5])
                      for i in range(parameter_count)]
        optimizer = NativeAdam(parameters, lr=LR)
        optimizer.step()                     # settle; all share t
        calls = {"n": 0}
        real_full = Core._typed_full

        def counting_full(*args, **kwargs):
            calls["n"] += 1
            return real_full(*args, **kwargs)

        patcher = _injected(Core, "_typed_full", counting_full)
        try:
            optimizer.step()
        finally:
            patcher.undo()
        counts[parameter_count] = calls["n"]
        optimizer.close()
        for parameter in parameters:
            parameter.close()

    assert len(set(counts.values())) == 1, counts
    # ...and there really are scalars: the count is not trivially zero.
    assert counts[1] > 0


@needs_native
def test_sgd_scalar_allocations_no_longer_scale_with_the_parameter_count():
    counts = {}
    for parameter_count in (1, 2, 6):
        parameters = [_parameter([float(i) + 1.0], [0.5])
                      for i in range(parameter_count)]
        optimizer = NativeSGD(parameters, lr=LR)
        calls = {"n": 0}
        real_full = Core._typed_full

        def counting_full(*args, **kwargs):
            calls["n"] += 1
            return real_full(*args, **kwargs)

        patcher = _injected(Core, "_typed_full", counting_full)
        try:
            optimizer.step()
        finally:
            patcher.undo()
        counts[parameter_count] = calls["n"]
        for parameter in parameters:
            parameter.close()
    assert counts == {1: 1, 2: 1, 6: 1}, counts


@needs_native
def test_parameters_at_different_step_counters_get_their_own_corrections():
    """The bias-correction pair is cached per step counter, not per step:
    a parameter that skipped earlier steps takes its own ``t``, and the
    values it lands on are the ones its own ``t`` implies."""
    active = _parameter([2.0], [0.5])
    latecomer = _parameter([2.0], None)      # no gradient yet
    optimizer = NativeAdam([active, latecomer], lr=LR, betas=BETAS, eps=EPS)
    for _ in range(3):
        optimizer.step()
    assert optimizer.step_counts == (3, 0)

    source = NativeTensor.from_array(np.array([0.5]))
    try:
        latecomer.multiply(source).sum().backward()
    finally:
        source.close()
    optimizer.step()                     # active at t=4, latecomer at t=1
    assert optimizer.step_counts == (4, 1)

    want_active, _, _ = _pre_h4_adam_run([2.0], [0.5], 4, LR, BETAS, EPS)
    want_late, _, _ = _pre_h4_adam_run([2.0], [0.5], 1, LR, BETAS, EPS)
    assert np.array_equal(_bits(active.to_numpy()), _bits(want_active))
    assert np.array_equal(_bits(latecomer.to_numpy()), _bits(want_late))
    optimizer.close()
    active.close()
    latecomer.close()


@needs_native
def test_a_step_with_no_active_parameter_allocates_nothing(live_storages):
    """The holder is lazy, so a step that stages nothing builds no scalar
    and allocates no native storage at all."""
    frozen = _parameter([1.0, 2.0], [0.5, 0.5], requires_grad=False)
    gradientless = _parameter([3.0, 4.0], None)
    optimizer = NativeAdam([frozen, gradientless], lr=LR)
    baseline = set(live_storages)
    calls = {"n": 0}
    real_full = Core._typed_full

    def counting_full(*args, **kwargs):
        calls["n"] += 1
        return real_full(*args, **kwargs)

    patcher = _injected(Core, "_typed_full", counting_full)
    try:
        optimizer.step()
    finally:
        patcher.undo()
    assert calls["n"] == 0
    assert set(live_storages) == baseline
    assert optimizer.step_counts == (0, 0)
    optimizer.close()
    frozen.close()
    gradientless.close()


@needs_native
def test_the_constants_holder_never_survives_a_step(live_storages):
    """No scalar outlives the step that built it: the optimizer keeps no
    scratch tensor, so live storage after a step is exactly the
    optimizer's own moments plus the caller's objects."""
    parameters = _mixed_parameters(seed=5)
    optimizer = NativeAdam(parameters, lr=LR)
    optimizer.step()
    _settle_wrapper_cycles()
    settled = set(live_storages)
    for _ in range(4):
        optimizer.step()
        assert len(live_storages) == len(settled)
    # The optimizer exposes no attribute holding native scratch.
    assert not hasattr(optimizer, "_constants")
    for name in NativeAdam.__slots__:
        assert "constant" not in name and "scratch" not in name, name
    optimizer.close()
    for parameter in parameters:
        parameter.close()


@needs_native
def test_the_constants_holder_releases_everything_it_built():
    holder = native_adam_module._StepConstants(LR, BETAS, EPS)
    built = list(holder.invariants("float64", "cpu"))
    built += list(holder.corrections("float64", "cpu", 3))
    built += list(holder.corrections("float64", "cpu", 7))
    assert all(not core._closed for core in built)
    # Repeated requests return the same cores, never new ones.
    assert holder.invariants("float64", "cpu") == tuple(built[:6])
    assert holder.corrections("float64", "cpu", 3) == tuple(built[6:8])
    holder.close()
    assert all(core._closed for core in built)
    holder.close()                     # idempotent


@needs_native
def test_a_failed_constant_build_closes_what_it_built(live_storages):
    """A failure part-way through building a scalar set releases the
    scalars that call already created and caches nothing."""
    holder = native_adam_module._StepConstants(LR, BETAS, EPS)
    baseline = set(live_storages)
    calls = {"n": 0}
    real_full = Core._typed_full

    def flaky_full(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 4:
            raise _Boom("forced constant failure")
        return real_full(*args, **kwargs)

    patcher = _injected(Core, "_typed_full", flaky_full)
    try:
        with pytest.raises(_Boom, match="forced constant failure"):
            holder.invariants("float64", "cpu")
    finally:
        patcher.undo()
    assert set(live_storages) == baseline
    holder.close()
    assert set(live_storages) == baseline


# ======================================================================
# 3. Temporary lifetime — released at last use, bounded peak
# ======================================================================


def _peak_live_during(callable_):
    """Allocation statistics for storages created while ``callable_``
    runs: the peak simultaneously live count, the peak simultaneously
    live bytes, the total count, and the total bytes."""
    live = {}
    stats = {"peak_count": 0, "peak_bytes": 0, "total": 0, "bytes": 0}
    real_init = cpp.NativeStorage.__init__
    real_close = cpp.NativeStorage.close

    def tracked_init(self, size, *args, **kwargs):
        real_init(self, size, *args, **kwargs)
        live[id(self)] = int(size) * 8
        stats["total"] += 1
        stats["bytes"] += int(size) * 8
        stats["peak_count"] = max(stats["peak_count"], len(live))
        stats["peak_bytes"] = max(stats["peak_bytes"], sum(live.values()))

    def tracked_close(self):
        live.pop(id(self), None)
        return real_close(self)

    patchers = [_injected(cpp.NativeStorage, "__init__", tracked_init),
                _injected(cpp.NativeStorage, "close", tracked_close)]
    try:
        callable_()
    finally:
        for patcher in patchers:
            patcher.undo()
    return stats


@needs_native
def test_staging_releases_temporaries_before_the_expression_completes():
    """The semantic statement of H4's release discipline.

    The staged Adam expression plus its commit allocates sixteen
    parameter-sized buffers. If every one were held until the expression
    completed — the pre-H4 discipline — the simultaneously live bytes
    would have to reach that whole total. Because each is released at its
    last use, the peak stays within a small fixed multiple of one
    parameter, independent of the expression's length. Stated in *bytes*,
    because the one-element scalar coefficients are counted the same as a
    parameter-sized buffer by a raw count and are irrelevant to the
    memory this measures.

    It was seventeen until Phase H, milestone H5: the commit's value copy
    used to allocate a zero-filled parameter-sized buffer and then add
    the staged result into it, and now allocates one uninitialized buffer
    and gathers into it. That is one whole parameter of allocation and
    one whole zero-fill pass removed from every committed update, and it
    is why this bound moved *down*."""
    shape = (32, 32)
    parameter_bytes = int(np.prod(shape)) * 8
    parameter = _parameter(np.ones(shape), np.full(shape, 0.25))
    optimizer = NativeAdam([parameter], lr=LR)
    optimizer.step()
    stats = _peak_live_during(optimizer.step)

    # The step really does allocate the whole expression...
    assert stats["bytes"] >= 16 * parameter_bytes, stats
    # ...and no longer allocates the seventeenth, pre-H5 buffer.
    assert stats["bytes"] < 17 * parameter_bytes, stats
    # ...and never holds more than a small working set of it at once.
    assert stats["peak_bytes"] <= 8 * parameter_bytes, stats
    optimizer.close()
    parameter.close()


@needs_native
def test_the_staging_peak_is_independent_of_the_expression_length():
    """The stronger form: the peak scales with the *parameter*, not with
    the number of operations, so it stays a fixed multiple across
    sizes."""
    ratios = []
    for shape in ((8, 8), (32, 32), (64, 64)):
        parameter_bytes = int(np.prod(shape)) * 8
        parameter = _parameter(np.ones(shape), np.full(shape, 0.25))
        optimizer = NativeAdam([parameter], lr=LR)
        optimizer.step()
        stats = _peak_live_during(optimizer.step)
        ratios.append(stats["peak_bytes"] / parameter_bytes)
        optimizer.close()
        parameter.close()
    assert max(ratios) - min(ratios) < 0.5, ratios
    assert max(ratios) <= 8.0, ratios


@needs_native
def test_the_staging_peak_does_not_grow_with_the_parameter_count_per_entry():
    """Adding parameters adds the *staged results* that must survive
    until the commit, but not extra simultaneously live temporaries per
    entry: the per-entry marginal peak is bounded and small."""
    peaks = {}
    for count in (1, 2, 4, 8):
        parameters = [_parameter(np.ones((4, 4)), np.full((4, 4), 0.25))
                      for _ in range(count)]
        optimizer = NativeAdam(parameters, lr=LR)
        optimizer.step()
        peaks[count] = _peak_live_during(optimizer.step)["peak_count"]
        optimizer.close()
        for parameter in parameters:
            parameter.close()
    # Each extra parameter contributes its three staged results plus its
    # two replaced moments; the per-entry temporary working set does not
    # grow. Five per parameter is that accounting, and the marginal cost
    # must not exceed it.
    marginal = (peaks[8] - peaks[1]) / 7
    assert marginal <= 5.0, peaks


@needs_native
def test_repeated_steps_and_lifecycles_return_live_storage_to_baseline(
    live_storages
):
    """No accumulation across steps, and none across whole create / step
    / close cycles — checked immediately, with no gc.collect()."""
    parameters = _mixed_parameters(seed=8)
    # One complete cycle first, so the baseline is measured after every
    # construction-time graph wrapper has reached its release point; the
    # assertions below are then about accumulation across cycles,
    # immediately and with no collection of any kind.
    warmup = NativeAdam(parameters, lr=LR)
    warmup.step()
    warmup.close()
    _settle_wrapper_cycles()
    baseline = len(live_storages)
    for _ in range(6):
        optimizer = NativeAdam(parameters, lr=LR)
        for _ in range(3):
            optimizer.step()
        optimizer.close()
        assert len(live_storages) == baseline
    for _ in range(6):
        optimizer = NativeSGD(parameters, lr=LR)
        optimizer.step()
        assert len(live_storages) == baseline
    for parameter in parameters:
        parameter.close()


# ======================================================================
# 4. Stage/commit contract
# ======================================================================


def _fingerprint(optimizer):
    """Everything a step is allowed to move, plus everything it is
    not — by value *and* by object identity."""
    return {
        "values": [p.to_numpy().copy() for p in optimizer.parameters()],
        "parameter_ids": [id(p) for p in optimizer.parameters()],
        "storage_ids": [id(p._core._storage) for p in optimizer.parameters()],
        "versions": [p.version for p in optimizer.parameters()],
        "grad_ids": [id(p.grad) for p in optimizer.parameters()],
        "grads": [None if p.grad is None else p.grad.to_numpy().copy()
                  for p in optimizer.parameters()],
        "moment_ids": ([id(b) for b in getattr(optimizer, "_m", [])]
                       + [id(b) for b in getattr(optimizer, "_v", [])]),
        "moments": [b.to_numpy().copy()
                    for b in (list(getattr(optimizer, "_m", []))
                              + list(getattr(optimizer, "_v", [])))],
        "moment_storage_ids": [id(b._core._storage)
                               for b in (list(getattr(optimizer, "_m", []))
                                         + list(getattr(optimizer, "_v", [])))],
        "steps": getattr(optimizer, "step_counts", ()),
        "hyper": (optimizer.lr, getattr(optimizer, "betas", None),
                  getattr(optimizer, "eps", None)),
    }


def _assert_identical(before, after, where=""):
    for key in before:
        if key in ("values", "grads", "moments"):
            assert len(before[key]) == len(after[key]), (where, key)
            for i, (a, b) in enumerate(zip(before[key], after[key])):
                if a is None or b is None:
                    assert a is b, (where, key, i)
                else:
                    assert np.array_equal(_bits(a), _bits(b)), (where, key, i)
        else:
            assert before[key] == after[key], (where, key)


@needs_native
def test_stage_entry_alone_mutates_nothing():
    """The stage phase computes candidate values and touches no
    parameter, moment, counter, version, or gradient — checked by calling
    the staging seam directly, with no commit behind it."""
    parameters = _mixed_parameters(seed=12)
    optimizer = NativeAdam(parameters, lr=LR)
    optimizer.step()
    before = _fingerprint(optimizer)

    staged = []
    for index, parameter in enumerate(parameters):
        staged.append(optimizer._stage_entry(index, parameter, parameter.grad))
    _assert_identical(before, _fingerprint(optimizer), "after staging")

    # The staged values are exactly what the next commit would install.
    for index, entry in enumerate(staged):
        assert entry[0] == index
        assert entry[1] is parameters[index]
        assert entry[5] == optimizer.step_counts[index] + 1
        for tensor in entry[2:5]:
            assert isinstance(tensor, NativeTensor)
            assert not isinstance(tensor, NativeParameter)
            assert not tensor.closed
            assert tensor.shape == parameters[index].shape
    for entry in staged:
        for tensor in entry[2:5]:
            tensor.close()
    optimizer.close()
    for parameter in parameters:
        parameter.close()


@needs_native
def test_stage_entry_is_correct_without_a_shared_constants_holder():
    """The staging seam stays standalone-correct: called with three
    arguments it builds and releases its own scalars, and produces the
    same bits the shared holder produces."""
    parameter = _parameter([[1.5, -0.5]], [[0.25, 0.75]])
    optimizer = NativeAdam([parameter], lr=LR, betas=BETAS, eps=EPS)
    optimizer.step()

    standalone = optimizer._stage_entry(0, parameter, parameter.grad)
    holder = native_adam_module._StepConstants(LR, BETAS, EPS)
    try:
        shared = optimizer._stage_entry(0, parameter, parameter.grad, holder)
    finally:
        holder.close()
    for a, b in zip(standalone[2:5], shared[2:5]):
        assert np.array_equal(_bits(a.to_numpy()), _bits(b.to_numpy()))
        a.close()
        b.close()
    optimizer.close()
    parameter.close()


@needs_native
def test_a_successful_step_moves_exactly_the_expected_state():
    parameters = _mixed_parameters(seed=13)
    optimizer = NativeAdam(parameters, lr=LR)
    before = _fingerprint(optimizer)
    optimizer.step()
    after = _fingerprint(optimizer)

    assert after["parameter_ids"] == before["parameter_ids"]
    assert after["grad_ids"] == before["grad_ids"]
    for a, b in zip(after["grads"], before["grads"]):
        assert np.array_equal(_bits(a), _bits(b))
    assert after["versions"] == [v + 1 for v in before["versions"]]
    assert list(after["steps"]) == [s + 1 for s in before["steps"]]
    assert after["hyper"] == before["hyper"]
    # Documented pre-H4 behavior, unchanged by H4: the commit installs
    # fresh storage for the parameter and *replaces* the moment objects.
    assert all(a != b for a, b in zip(after["storage_ids"],
                                      before["storage_ids"]))
    assert all(a != b for a, b in zip(after["moment_ids"],
                                      before["moment_ids"]))
    optimizer.close()
    for parameter in parameters:
        parameter.close()


# ======================================================================
# 5. Failure atomicity
# ======================================================================


_FAILURE_CLASSES = (RuntimeError, MemoryError, KeyboardInterrupt, _Boom)


def _step_and_assert_untouched(optimizer, seam_owner, seam, index,
                               error, live_storages):
    """Inject ``error`` at the ``index``-th call of ``seam`` during a
    step and assert the whole world is bit-identical afterwards."""
    before = _fingerprint(optimizer)
    baseline = len(live_storages)
    real = getattr(seam_owner, seam)
    calls = {"n": 0}

    def failing(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] == index:
            raise error("injected optimizer failure")
        return real(*args, **kwargs)

    patcher = _injected(seam_owner, seam, failing)
    try:
        with pytest.raises(error, match="injected optimizer failure"):
            optimizer.step()
    finally:
        patcher.undo()
    _assert_identical(before, _fingerprint(optimizer),
                      f"{seam}#{index}/{error.__name__}")
    assert len(live_storages) <= baseline, (seam, index, error)
    return calls["n"]


@needs_native
@pytest.mark.parametrize("error", _FAILURE_CLASSES)
@pytest.mark.parametrize("seam,index", [
    ("_typed_full", 1),    # the very first shared scalar
    ("_typed_full", 4),    # part-way through the invariant set
    ("_typed_full", 7),    # the first bias-correction reciprocal
    ("multiply", 1),      # the first entry's first arithmetic
    ("multiply", 5),      # mid-expression, first entry
    ("multiply", 9),      # the first entry's last multiply
    ("multiply", 10),     # the second entry, first fully staged
    ("multiply", 19),     # the third entry
    ("add", 1),           # m_new
    ("add", 2),           # v_new
    ("add", 3),           # the denominator
    ("subtract", 1),      # parameter_new
    ("subtract", 2),      # a later entry's parameter_new
    ("sqrt", 1),
    ("reciprocal", 1),
    # The first parameter's commit copy. Pre-H5 this seam was
    # ``("zeros", 1)``, because ``_native_copy`` began by allocating a
    # zero-filled destination; H5 replaced that composition with the
    # identity gather, so the same instant in the same transaction is now
    # reached through ``contiguous_copy``. Nothing else in a step calls
    # it, so index 1 still names the first entry's commit exactly.
    ("contiguous_copy", 1),
])
def test_every_injected_staging_failure_leaves_the_world_untouched(
    seam, index, error, live_storages
):
    parameters = _mixed_parameters(seed=21)
    optimizer = NativeAdam(parameters, lr=LR, betas=BETAS, eps=EPS)
    optimizer.step()
    fired = _step_and_assert_untouched(
        optimizer, Core, seam, index, error, live_storages
    )
    assert fired >= index, (seam, index, fired)
    # ...and the same optimizer completes the step it was denied.
    optimizer.step()
    assert list(optimizer.step_counts) == [2] * len(parameters)
    optimizer.close()
    for parameter in parameters:
        parameter.close()


@needs_native
@pytest.mark.parametrize("error", _FAILURE_CLASSES)
@pytest.mark.parametrize("entry", [1, 2, 4])
def test_a_failure_at_any_staged_entry_commits_nothing(
    entry, error, live_storages
):
    """The two-phase guarantee, at each entry boundary: a failure while
    staging entry *n* leaves entries 0..n-1 — already fully staged —
    completely uncommitted."""
    parameters = _mixed_parameters(seed=22)
    optimizer = NativeAdam(parameters, lr=LR)
    optimizer.step()
    before = _fingerprint(optimizer)
    baseline = len(live_storages)

    real_stage = NativeAdam._stage_entry
    calls = {"n": 0}

    def failing_stage(self, index, parameter, grad, *rest):
        calls["n"] += 1
        if calls["n"] == entry:
            raise error("injected optimizer failure")
        return real_stage(self, index, parameter, grad, *rest)

    patcher = _injected(NativeAdam, "_stage_entry", failing_stage)
    try:
        with pytest.raises(error, match="injected optimizer failure"):
            optimizer.step()
    finally:
        patcher.undo()
    _assert_identical(before, _fingerprint(optimizer), f"entry {entry}")
    assert len(live_storages) <= baseline
    optimizer.close()
    for parameter in parameters:
        parameter.close()


@needs_native
@pytest.mark.parametrize("error", _FAILURE_CLASSES)
@pytest.mark.parametrize("index", [1, 2, 3])
def test_a_first_step_failure_leaves_uninitialised_state_untouched(
    index, error, live_storages
):
    """First-step behavior gets its own coverage: the moments are still
    all zeros and the counters all zero, and a failed first step must
    leave exactly that."""
    parameters = _mixed_parameters(seed=23)
    optimizer = NativeAdam(parameters, lr=LR)
    _step_and_assert_untouched(
        optimizer, Core, "multiply", index, error, live_storages
    )
    assert optimizer.step_counts == (0,) * len(parameters)
    for buffer in optimizer._m + optimizer._v:
        assert np.array_equal(buffer.to_numpy(),
                              np.zeros(buffer.shape))
    optimizer.step()
    assert optimizer.step_counts == (1,) * len(parameters)
    optimizer.close()
    for parameter in parameters:
        parameter.close()


@needs_native
@pytest.mark.parametrize("error", _FAILURE_CLASSES)
def test_a_wrapper_construction_failure_releases_every_staged_core(
    error, live_storages
):
    """The Python wrapper the staging seam returns is the last thing it
    builds; a failure there must not strand the native cores behind it."""
    parameters = _mixed_parameters(seed=24)
    optimizer = NativeAdam(parameters, lr=LR)
    optimizer.step()
    _step_and_assert_untouched(
        optimizer, NativeTensor, "_from_core", 2, error, live_storages
    )
    optimizer.close()
    for parameter in parameters:
        parameter.close()


@needs_native
@pytest.mark.parametrize("error", _FAILURE_CLASSES)
@pytest.mark.parametrize("seam,index", [
    ("_typed_full", 1), ("multiply", 1), ("multiply", 2), ("subtract", 1),
    ("subtract", 3),
    # The first parameter's commit copy — pre-H5 ``("zeros", 1)``; see
    # the Adam parametrization above for why it moved.
    ("contiguous_copy", 1),
])
def test_every_injected_sgd_failure_leaves_the_world_untouched(
    seam, index, error, live_storages
):
    parameters = _mixed_parameters(seed=25)
    optimizer = NativeSGD(parameters, lr=LR)
    _step_and_assert_untouched(
        optimizer, Core, seam, index, error, live_storages
    )
    optimizer.step()
    assert all(p.version == 1 for p in parameters)
    for parameter in parameters:
        parameter.close()


@needs_native
def test_a_commit_failure_is_reported_and_leaves_no_staged_core_behind(
    live_storages
):
    """Commit is not *claimed* to be infallible, it is *tested*: injecting
    a failure into ``copy_value_`` proves the staged temporaries are still
    released and the exception reaches the caller. Entries committed
    before the failure legitimately stand — the documented per-entry
    boundary, unchanged by H4 — and are asserted to be exactly one."""
    parameters = _mixed_parameters(seed=26)
    optimizer = NativeAdam(parameters, lr=LR)
    optimizer.step()
    before = _fingerprint(optimizer)
    baseline = len(live_storages)

    real_copy = NativeParameter.copy_value_
    calls = {"n": 0}

    def failing_copy(self, source):
        calls["n"] += 1
        if calls["n"] == 2:
            raise _Boom("injected commit failure")
        return real_copy(self, source)

    patcher = _injected(NativeParameter, "copy_value_", failing_copy)
    try:
        with pytest.raises(_Boom, match="injected commit failure"):
            optimizer.step()
    finally:
        patcher.undo()

    # Exactly the first entry committed; every later one is untouched.
    after = _fingerprint(optimizer)
    assert after["versions"][0] == before["versions"][0] + 1
    assert after["versions"][1:] == before["versions"][1:]
    assert list(after["steps"])[0] == list(before["steps"])[0] + 1
    assert list(after["steps"])[1:] == list(before["steps"])[1:]
    # No staged core leaked, in either the committed or the abandoned half.
    assert len(live_storages) <= baseline
    optimizer.close()
    for parameter in parameters:
        parameter.close()


# ======================================================================
# 6. Gradients, identity, aliasing, and hyperparameters
# ======================================================================


@needs_native
@pytest.mark.parametrize("factory,kwargs", [
    (NativeAdam, {"lr": LR}),
    (NativeAdam, {"lr": LR, "betas": (0.5, 0.9), "eps": 1e-4}),
    (NativeSGD, {"lr": LR}),
])
def test_a_step_never_touches_a_gradient(factory, kwargs):
    parameters = _mixed_parameters(seed=31)
    optimizer = factory(parameters, **kwargs)
    identities = [id(p.grad) for p in parameters]
    values = [p.grad.to_numpy().copy() for p in parameters]
    storages = [id(p.grad._core._storage) for p in parameters]
    for _ in range(3):
        optimizer.step()
        assert [id(p.grad) for p in parameters] == identities
        assert [id(p.grad._core._storage) for p in parameters] == storages
        for parameter, want in zip(parameters, values):
            assert np.array_equal(_bits(parameter.grad.to_numpy()),
                                  _bits(want))
            assert not parameter.grad.closed
    if hasattr(optimizer, "close"):
        optimizer.close()
    for parameter in parameters:
        parameter.close()


@needs_native
@pytest.mark.parametrize("factory", [NativeAdam, NativeSGD])
def test_a_duplicate_parameter_reference_updates_exactly_once(factory):
    parameter = _parameter([2.0, 4.0], [0.5, 1.0])
    optimizer = factory([parameter, parameter, parameter], lr=LR)
    assert len(optimizer.parameters()) == 1
    optimizer.step()
    assert parameter.version == 1
    if factory is NativeAdam:
        assert optimizer.step_counts == (1,)
    if hasattr(optimizer, "close"):
        optimizer.close()
    parameter.close()


@needs_native
def test_distinct_parameters_over_equal_values_never_merge():
    """Deduplication is by identity, never by value: two equal-valued
    parameters keep separate state and both update."""
    first = _parameter([2.0, 4.0], [0.5, 1.0])
    second = _parameter([2.0, 4.0], [0.5, 1.0])
    optimizer = NativeAdam([first, second], lr=LR)
    assert len(optimizer.parameters()) == 2
    optimizer.step()
    assert first.version == 1 and second.version == 1
    assert optimizer._m[0] is not optimizer._m[1]
    assert np.array_equal(first.to_numpy(), second.to_numpy())
    optimizer.close()
    first.close()
    second.close()


@needs_native
def test_two_optimizers_over_the_same_parameter_each_commit_once():
    parameter = _parameter([2.0, 4.0], [0.5, 1.0])
    first = NativeAdam([parameter], lr=LR)
    second = NativeAdam([parameter], lr=LR)
    first.step()
    assert parameter.version == 1
    second.step()
    assert parameter.version == 2
    assert first.step_counts == (1,) and second.step_counts == (1,)
    first.close()
    second.close()
    parameter.close()


@needs_native
def test_a_gradient_sharing_storage_with_its_parameter_is_not_mutated():
    """A gradient built from the parameter's own values is still only
    read: the staged update never writes through either operand."""
    parameter = NativeParameter([3.0, -1.0])
    source = NativeTensor.from_array(np.array([2.0, 2.0]))
    try:
        parameter.multiply(source).sum().backward()
    finally:
        source.close()
    grad_before = parameter.grad.to_numpy().copy()
    optimizer = NativeAdam([parameter], lr=LR)
    optimizer.step()
    assert np.array_equal(_bits(parameter.grad.to_numpy()), _bits(grad_before))
    optimizer.close()
    parameter.close()


@needs_native
def test_a_parameter_that_loses_its_gradient_stops_ageing():
    parameter = _parameter([2.0], [0.5])
    other = _parameter([3.0], [0.25])
    optimizer = NativeAdam([parameter, other], lr=LR)
    optimizer.step()
    assert optimizer.step_counts == (1, 1)
    parameter.zero_grad()
    frozen_value = parameter.to_numpy().copy()
    frozen_m = optimizer._m[0].to_numpy().copy()
    optimizer.step()
    assert optimizer.step_counts == (1, 2)
    assert np.array_equal(parameter.to_numpy(), frozen_value)
    assert np.array_equal(optimizer._m[0].to_numpy(), frozen_m)
    assert parameter.version == 1
    optimizer.close()
    parameter.close()
    other.close()


@needs_native
def test_each_step_observes_the_hyperparameters_it_begins_with():
    """The per-step holder captures the hyperparameters at the start of
    the step, so a value replaced between steps is picked up by the next
    one exactly as before H4."""
    parameter = _parameter([2.0], [0.5])
    optimizer = NativeAdam([parameter], lr=LR, betas=BETAS, eps=EPS)
    optimizer.step()
    state = optimizer.state_dict()
    state["lr"] = 0.5
    state["betas"] = (0.5, 0.9)
    state["eps"] = 1e-4
    optimizer.load_state_dict(state)
    for snapshot in state["m"] + state["v"]:
        snapshot.close()
    assert (optimizer.lr, optimizer.betas, optimizer.eps) == \
        (0.5, (0.5, 0.9), 1e-4)

    before = parameter.to_numpy().copy()
    m_before = optimizer._m[0].to_numpy().copy()
    v_before = optimizer._v[0].to_numpy().copy()
    optimizer.step()

    reference_p = Core.from_array(before)
    reference_g = Core.from_array(parameter.grad.to_numpy())
    reference_m = Core.from_array(m_before)
    reference_v = Core.from_array(v_before)
    m_new, v_new, p_new = _pre_h4_adam_stage(
        reference_p, reference_g, reference_m, reference_v,
        0.5, 0.9, 0.5, 1e-4, 2,
    )
    committed = _committed(p_new)
    assert np.array_equal(_bits(parameter.to_numpy()),
                          _bits(committed.to_numpy()))
    for core in (reference_p, reference_g, reference_m, reference_v,
                 m_new, v_new, p_new, committed):
        core.close()
    optimizer.close()
    parameter.close()


@needs_native
def test_closed_and_malformed_inputs_still_raise_exactly_as_before():
    """Validation neither moved behind a mutation nor changed class."""
    parameter = _parameter([1.0, 2.0], [0.5, 0.5])
    optimizer = NativeAdam([parameter], lr=LR)

    optimizer._m[0].close()
    with pytest.raises(RuntimeError, match=r"m state for parameters\[0\]"):
        optimizer.step()
    assert parameter.version == 0
    optimizer.close()

    closed_grad = _parameter([1.0, 2.0], [0.5, 0.5])
    second = NativeAdam([closed_grad], lr=LR)
    closed_grad.grad.close()
    with pytest.raises(RuntimeError, match=r"parameters\[0\].grad has been"):
        second.step()
    assert closed_grad.version == 0
    second.close()

    shaped = NativeParameter([1.0, 2.0])
    source = NativeTensor.from_array(np.array([[1.0, 2.0], [3.0, 4.0]]))
    third = NativeAdam([shaped], lr=LR)
    try:
        shaped._grad = source           # a deliberately malformed gradient
        with pytest.raises(ValueError, match="grad shape"):
            third.step()
        assert shaped.version == 0
    finally:
        shaped._grad = None
        source.close()
        third.close()
        shaped.close()
    parameter.close()
    closed_grad.close()


# ======================================================================
# 7. State, checkpoints, and exact resume
# ======================================================================


@needs_native
def test_state_dict_round_trip_reproduces_the_next_update_exactly():
    parameters = _mixed_parameters(seed=41)
    optimizer = NativeAdam(parameters, lr=LR, betas=(0.5, 0.9), eps=1e-6)
    for _ in range(3):
        optimizer.step()
    state = optimizer.state_dict()

    # A state_dict carries optimizer state, never parameter values, so the
    # fresh parameters are brought to the trained values explicitly — that
    # is the whole restoration boundary this test isolates.
    fresh_parameters = _mixed_parameters(seed=41)
    for original, restored in zip(parameters, fresh_parameters):
        source = NativeTensor.from_array(original.to_numpy())
        try:
            restored.copy_value_(source)
        finally:
            source.close()
    fresh = NativeAdam(fresh_parameters, lr=0.9, betas=(0.1, 0.2), eps=1.0)
    fresh.load_state_dict(state)
    assert fresh.step_counts == optimizer.step_counts
    assert (fresh.lr, fresh.betas, fresh.eps) == (LR, (0.5, 0.9), 1e-6)

    optimizer.step()
    fresh.step()
    for original, restored in zip(parameters, fresh_parameters):
        assert np.array_equal(_bits(original.to_numpy()),
                              _bits(restored.to_numpy()))
    for index in range(len(parameters)):
        assert np.array_equal(_bits(optimizer._m[index].to_numpy()),
                              _bits(fresh._m[index].to_numpy()))
        assert np.array_equal(_bits(optimizer._v[index].to_numpy()),
                              _bits(fresh._v[index].to_numpy()))
    for snapshot in state["m"] + state["v"]:
        snapshot.close()
    optimizer.close()
    fresh.close()
    for parameter in parameters + fresh_parameters:
        parameter.close()


def _tiny_model(seed_a=0, seed_b=1):
    return NativeSequential(
        NativeLinear(3, 5, seed=seed_a),
        NativeReLU(),
        NativeLinear(5, 2, seed=seed_b),
    )


def _train(model, optimizer, x, y, steps):
    from tensorforge.experimental import NativeMSELoss
    losses = []
    for _ in range(steps):
        loss = NativeMSELoss()(model(x), y)
        losses.append(float(loss.to_numpy()))
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        loss.close()
    return losses


@needs_native
def test_a_checkpoint_resume_is_still_exact(tmp_path):
    rng = np.random.default_rng(20260304)
    x_values = rng.standard_normal((6, 3))
    y_values = rng.standard_normal((6, 2))

    model = _tiny_model()
    optimizer = NativeAdam(model.parameters(), lr=0.05)
    x = NativeTensor.from_array(x_values)
    y = NativeTensor.from_array(y_values)
    uninterrupted = _train(model, optimizer, x, y, 8)
    reference_parameters = [p.to_numpy().copy() for p in model.parameters()]

    interrupted_model = _tiny_model()
    interrupted_optimizer = NativeAdam(interrupted_model.parameters(), lr=0.05)
    _train(interrupted_model, interrupted_optimizer, x, y, 4)
    path = tmp_path / "h4_resume.npz"
    save_native_checkpoint(path, interrupted_model,
                           optimizer=interrupted_optimizer)
    interrupted_optimizer.close()
    for parameter in interrupted_model.parameters():
        parameter.close()

    resumed_model = _tiny_model(seed_a=7, seed_b=8)   # deliberately different
    resumed_optimizer = NativeAdam(resumed_model.parameters(), lr=0.05)
    load_native_checkpoint(path, resumed_model, optimizer=resumed_optimizer)
    suffix = _train(resumed_model, resumed_optimizer, x, y, 4)

    assert suffix == uninterrupted[4:]
    for restored, want in zip(resumed_model.parameters(),
                              reference_parameters):
        assert np.array_equal(_bits(restored.to_numpy()), _bits(want))
    assert resumed_optimizer.step_counts == optimizer.step_counts

    optimizer.close()
    resumed_optimizer.close()
    for parameter in list(model.parameters()) + list(
        resumed_model.parameters()
    ):
        parameter.close()
    x.close()
    y.close()


@needs_native
def test_a_failed_state_load_leaves_the_optimizer_exactly_as_it_was():
    parameters = _mixed_parameters(seed=42)
    optimizer = NativeAdam(parameters, lr=LR)
    optimizer.step()
    before = _fingerprint(optimizer)
    state = optimizer.state_dict()
    state["step_counts"] = (0,) * (len(parameters) - 1)   # wrong length
    with pytest.raises(ValueError, match="step_counts"):
        optimizer.load_state_dict(state)
    _assert_identical(before, _fingerprint(optimizer), "failed load")
    state["step_counts"] = optimizer.step_counts
    optimizer.load_state_dict(state)
    for snapshot in state["m"] + state["v"]:
        snapshot.close()
    optimizer.close()
    for parameter in parameters:
        parameter.close()


# ======================================================================
# 8. Public-surface and scope guardrails
# ======================================================================


@needs_native
def test_h4_added_no_public_optimizer_surface():
    """No cache control, statistic, reset, profiling counter, dispatch
    selector, or failure toggle appeared on either optimizer."""
    banned = ("profile", "counter", "stats", "statistic", "cache",
              "scratch", "pool", "arena", "reset_", "instrument",
              "allocation", "telemetry", "toggle")
    for optimizer_type in (NativeAdam, NativeSGD):
        public = [name for name in dir(optimizer_type)
                  if not name.startswith("_")]
        for name in public:
            lowered = name.lower()
            assert not any(word in lowered for word in banned), (
                optimizer_type.__name__, name
            )
    assert sorted(name for name in dir(NativeAdam)
                  if not name.startswith("_")) == [
        "betas", "close", "closed", "eps", "load_state_dict", "lr",
        "parameters", "state_dict", "step", "step_counts", "zero_grad",
    ]
    assert sorted(name for name in dir(NativeSGD)
                  if not name.startswith("_")) == [
        "load_state_dict", "lr", "parameters", "state_dict", "step",
        "zero_grad",
    ]


@needs_native
def test_the_constants_holder_is_private_and_unexported():
    import tensorforge.experimental as experimental
    assert not hasattr(experimental, "_StepConstants")
    assert "_StepConstants" not in getattr(experimental, "__all__", ())
    assert native_adam_module._StepConstants.__name__.startswith("_")
    # It is not reachable from the C++ backend surface either.
    assert not hasattr(cpp, "_StepConstants")


@needs_native
def test_h4_kept_the_public_constructor_signatures():
    adam = inspect.signature(NativeAdam.__init__)
    assert list(adam.parameters) == ["self", "parameters", "lr", "betas",
                                     "eps"]
    assert adam.parameters["lr"].default == 0.001
    assert adam.parameters["betas"].default == (0.9, 0.999)
    assert adam.parameters["eps"].default == 1e-8
    sgd = inspect.signature(NativeSGD.__init__)
    assert list(sgd.parameters) == ["self", "parameters", "lr"]
    for optimizer_type in (NativeAdam, NativeSGD):
        assert list(inspect.signature(optimizer_type.step).parameters) == \
            ["self"]


@needs_native
def test_h4_moved_no_capability_registry_dtype_device_or_format():
    assert cpp.UNSUPPORTED == ("cuda", "amp")
    assert cpp.SUPPORTED_DTYPES == ("float64", "float32")
    assert cpp.SUPPORTED_DEVICES == ("cpu",)
    assert cpp.NATIVE_OPTIMIZERS == ("NativeSGD", "NativeAdam")
    from tensorforge.experimental import native_checkpoint
    assert native_checkpoint._FORMAT_VERSION == 3
    assert native_checkpoint._SUPPORTED_FORMAT_VERSIONS == (1, 2, 3)
    from tensorforge.experimental import native_optimizer_state
    assert native_optimizer_state.FORMAT_VERSION == 1


@needs_native
def test_h4_added_no_c_abi_symbol():
    """The optimizer is Python composition over the existing Core: no
    fused Adam kernel, no fused SGD kernel, no scalar optimizer kernel,
    and no new export."""
    from pathlib import Path
    import re

    sources = Path(__file__).resolve().parent.parent / "cpp" / "src"
    exported = set()
    for source in sources.glob("*.cpp"):
        text = source.read_text(encoding="utf-8")
        for match in re.finditer(r"TF_EXPORT[^;{]*?\b(tf_[a-z0-9_]+)\s*\(",
                                 text, flags=re.S):
            exported.add(match.group(1))
    # H4's claim is about Phase H, so it is measured against Phase H's own
    # surface of 52. The three extra symbols in the source belong to later
    # phases: Phase I's two typed storage creators at milestone I1, and
    # Phase K's argmax forward at milestone K3 and its index_select forward at
    # milestone K4.
    phase_i_creators = {"tf_storage_create_typed",
                        "tf_storage_create_uninitialized_typed"}
    phase_k_exports = {"tf_core_argmax", "tf_core_index_select"}
    assert len(exported) == 56, sorted(exported)
    assert len(exported - phase_i_creators - phase_k_exports) == 52, \
        sorted(exported)
    for banned in ("tf_core_adam", "tf_core_sgd", "tf_core_optimizer",
                   "tf_core_scale", "tf_core_axpy", "tf_core_fused"):
        assert not any(name.startswith(banned) for name in exported), banned


@needs_native
def test_the_optimizer_sources_declare_no_dispatch_or_profiling_control():
    from pathlib import Path

    root = Path(__file__).resolve().parent.parent / "src" / "tensorforge"
    for name in ("native_adam.py", "native_sgd.py"):
        text = (root / "experimental" / name).read_text(encoding="utf-8")
        for banned in ("os.environ", "getenv", "threading.local",
                       "functools.lru_cache", "global ", "time.perf_counter"):
            assert banned not in text, (name, banned)


def test_importing_stable_tensorforge_imports_no_native_optimizer():
    """Stable-import isolation is unchanged: nothing H4 touched is
    reachable from the pure-Python line."""
    import subprocess
    import sys

    code = (
        "import sys, tensorforge\n"
        "leaked = [m for m in sys.modules if 'experimental' in m "
        "or m.endswith('backends.cpp')]\n"
        "print(leaked)\n"
        "assert not leaked, leaked\n"
        "assert hasattr(tensorforge, 'Adam')\n"
    )
    result = subprocess.run([sys.executable, "-c", code],
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr


@needs_native
def test_every_h4_surface_records_the_milestone_semantically():
    """Semantic documentation guardrails, not exact prose matching: each
    status surface must state that H4 is complete, must name what it
    actually did, and must not claim a fused optimizer kernel or a new
    export."""
    from pathlib import Path
    import re

    root = Path(__file__).resolve().parent.parent
    surfaces = ("README.md", "CLAUDE.md", "docs/roadmap.md",
                "docs/project_summary.md", "docs/architecture.md",
                "docs/backend_experiments.md",
                "docs/native_support_matrix.md",
                "docs/release_history.md",
                "docs/native_cpu_performance_design.md")
    for surface in surfaces:
        text = (root / surface).read_text(encoding="utf-8")
        lowered = text.lower()
        assert re.search(r"\bh4\b", lowered), surface
        # It says what H4 changed...
        assert "optimizer" in lowered, surface
        # ...and never names an optimizer ABI symbol that does not exist,
        # nor claims an export count H4 did not produce.
        for banned in ("tf_core_adam", "tf_core_sgd", "tf_core_optimizer",
                       "53 exported", "53 `tf_*`"):
            assert banned not in lowered, (surface, banned)
        # A "fused optimizer kernel" may only ever appear as something
        # the project deliberately did *not* ship.
        negation = re.compile(
            r"\b(not|never|no|without|deliberately|rather than|instead of"
            r"|belongs? to|later|future)\b", re.I,
        )
        for match in re.finditer(
            r"fused (?:adam|sgd|optimizer)[a-z0-9 ]{0,12}kernel", lowered
        ):
            window = text[max(0, match.start() - 120):match.end() + 60]
            assert negation.search(window), (surface, match.group(0))


@needs_native
def test_the_h4_surfaces_state_the_transactional_invariants_verbatim():
    """The critical transactional invariants are the one place exact
    prose matters: a surface that stops saying them has stopped
    documenting the contract."""
    from pathlib import Path

    root = Path(__file__).resolve().parent.parent
    def flat(path):
        """Markdown is hard-wrapped, so a sentence spans line breaks;
        collapse whitespace before matching prose."""
        return " ".join(
            (root / path).read_text(encoding="utf-8").split()
        )

    for surface in ("docs/native_cpu_performance_design.md",
                    "docs/backend_experiments.md",
                    "docs/project_summary.md",
                    "docs/native_support_matrix.md",
                    "docs/release_history.md"):
        text = flat(surface)
        assert "copy_value_" in text, surface
        assert "bit-identical" in text.lower(), surface
    design = flat("docs/native_cpu_performance_design.md")
    assert "16.4" in design
    for claim in (
        "`copy_value_` and exactly one version increment per updated",
        "exact substitution, not a reassociation",
        "released at their last use",
        "Optimizations measured and rejected",
        "every staged computation completes before the first",
    ):
        assert claim in design, claim


@needs_native
def test_the_stable_and_native_optimizers_still_refuse_each_others_objects():
    native = _parameter([1.0, 2.0], [0.5, 0.5])
    stable = tensorforge.Parameter(np.array([1.0, 2.0]))
    with pytest.raises(TypeError):
        NativeAdam([stable], lr=LR)
    with pytest.raises(TypeError):
        NativeSGD([stable], lr=LR)
    with pytest.raises((TypeError, AttributeError)):
        tensorforge.Adam([native], lr=LR).step()
    native.close()
