"""Differentiable native Dropout — ``NativeTensor.dropout`` (Phase G,
milestone G3).

G3 connects two things that already existed and could not previously reach
each other: the G1 ``NativeGenerator`` call transaction (reserve → commit /
abandon, lock-protected and token-validated) and the G2 **stateless**
Dropout-forward Core (a deterministic function of ``(seed, call_index,
element, p)`` that touches no generator at all). The operation owns the
transaction, adopts the Core's private multiplier mask as **graph-owned**
saved state, and differentiates through it with the existing native
``multiply`` — there is no Dropout backward kernel.

These tests cover the public surface and its explicit-generator rule, the
``p == 0`` identity bypass, forward determinism tied to G2's **committed**
known-answer vectors, the call-consumption contract at every boundary
(exactly one per success, none per failure, none in backward, none at
``p == 0``), no-grad behavior, saved-mask ownership and graph lifetime,
the backward read contract and the deliberate absence of version tracking,
independence from later input and generator mutation, concurrency and
reentrancy, deterministic failure injection at every position between
reservation and commit, and the capability boundary G3 does **not** move.

Backend-dependent, so the module skips cleanly when the compiled backend
is not built. Cleanup is explicit via close().

Selector: python -m pytest -q -k native_dropout_autograd
"""

import gc
import threading

import numpy as np
import pytest

from tensorforge.backends import cpp
from tensorforge.experimental import (
    NativeGenerator, NativeParameter, NativeTensor,
)
from tensorforge.experimental import native_generator as native_generator_module
from tensorforge.experimental import native_tensor as native_tensor_module

pytestmark = pytest.mark.skipif(
    not cpp.is_available(),
    reason="experimental C++ backend not built; " + cpp.build_instructions(),
)

needs_fault_injection = pytest.mark.skipif(
    not (cpp.is_available() and cpp.fault_injection_available()),
    reason="fault injection not compiled into the backend",
)

# A bounded join everywhere a thread is used: a regression must fail this
# suite, never hang the session.
JOIN_TIMEOUT = 10.0


@pytest.fixture(autouse=True)
def _disarm_after_each():
    yield
    if cpp.is_available():
        cpp._arm_alloc_failure(0)
        cpp._require_library().tf_clear_error()


@pytest.fixture
def live_storages(monkeypatch):
    """The ids of every NativeStorage currently open — a real
    live-native-allocation count, so an ownership test can prove the count
    returns exactly to its baseline instead of trusting collection."""
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
# One committed G2 known-answer vector, restated here on purpose.
#
# It is `DROPOUT_VECTORS["mixed_seed_call0"]` from tests/test_native_dropout
# _core.py and cpp/tests/test_dropout_forward.cpp — the *same* constants,
# asserted from C++, from the Core, and now from the differentiable
# operation. Restating rather than importing is deliberate: a known-answer
# vector that one suite can redefine for another is not a known answer.
# --------------------------------------------------------------------------

VECTOR_SEED = 0x0123456789ABCDEF
VECTOR_CALL_INDEX = 0
VECTOR_P = 0.25
VECTOR_KEEP = "011011111010"          # 12 logical elements, row-major
VECTOR_SCALE = 1.0 / (1.0 - VECTOR_P)


def committed_mask(shape=(12,)):
    """The multiplier mask the committed vector pins, as a NumPy array."""
    values = [VECTOR_SCALE if keep == "1" else 0.0 for keep in VECTOR_KEEP]
    return np.array(values, dtype=np.float64).reshape(shape)


# -- helpers ---------------------------------------------------------------

def core_reference(values, p, seed, call_index):
    """The G2 Core's ``(output, mask)`` for these arguments, as NumPy
    arrays, with every native object closed again.

    This is the operation's oracle: G3 adds a transaction and a graph, and
    must change the numbers by exactly nothing."""
    array = np.asarray(values, dtype=np.float64)
    source = cpp.NativeTensorCore.from_array(array)
    try:
        out, mask = source._dropout_forward_with_mask(
            p, seed=seed, call_index=call_index
        )
        try:
            return out.to_numpy().copy(), mask.to_numpy().copy()
        finally:
            out.close()
            mask.close()
    finally:
        source.close()


def saved_masks(result):
    """The private mask cores this graph node owns (white-box: the
    lifetime contract is exactly what these tests must pin down)."""
    return result._graph_resources


def saved_mask_array(result):
    resources = saved_masks(result)
    assert len(resources) == 1, "expected exactly one graph-owned mask"
    return resources[0].to_numpy().copy()


def ones_like(tensor):
    return NativeTensor.from_array(np.ones(tensor.shape, dtype=np.float64))


# ==========================================================================
# 1. The public surface
# ==========================================================================

def test_signature_is_p_positional_and_generator_keyword_only():
    import inspect

    signature = inspect.signature(NativeTensor.dropout)
    assert list(signature.parameters) == ["self", "p", "generator"]
    generator = signature.parameters["generator"]
    assert generator.kind is inspect.Parameter.KEYWORD_ONLY
    assert generator.default is inspect.Parameter.empty, (
        "the generator must have no default: there is no implicit stream"
    )
    assert signature.parameters["p"].default is inspect.Parameter.empty


def test_generator_is_required():
    x = NativeTensor.from_array(np.arange(4.0))
    with pytest.raises(TypeError):
        x.dropout(0.5)
    x.close()


def test_generator_cannot_be_passed_positionally():
    x = NativeTensor.from_array(np.arange(4.0))
    generator = NativeGenerator(1)
    with pytest.raises(TypeError):
        x.dropout(0.5, generator)
    assert generator.calls == 0
    x.close()


@pytest.mark.parametrize(
    "bad", [None, 0, 1.5, "generator", object(), np.random.default_rng(0)],
)
def test_a_non_generator_is_rejected_before_anything_happens(bad, live_storages):
    x = NativeTensor.from_array(np.arange(4.0), requires_grad=True)
    baseline = len(live_storages)
    with pytest.raises(TypeError, match="NativeGenerator"):
        x.dropout(0.5, generator=bad)
    assert len(live_storages) == baseline
    x.close()


def test_the_stable_numpy_generator_is_not_accepted():
    """No silent bridge to NumPy's RNG: the native line's randomness is
    explicit native state or nothing."""
    x = NativeTensor.from_array(np.arange(4.0))
    with pytest.raises(TypeError):
        x.dropout(0.5, generator=np.random.default_rng(7))
    x.close()


def test_a_closed_receiver_fails_before_the_reservation():
    x = NativeTensor.from_array(np.arange(4.0))
    x.close()
    generator = NativeGenerator(3)
    with pytest.raises(RuntimeError):
        x.dropout(0.5, generator=generator)
    assert generator.calls == 0
    assert generator._has_active_reservation() is False


@pytest.mark.parametrize("p", [1.0, 1, 1.5, -0.001, float("nan"),
                               float("inf"), float("-inf")])
def test_out_of_contract_probabilities_are_rejected_without_a_call(p):
    x = NativeTensor.from_array(np.arange(4.0), requires_grad=True)
    generator = NativeGenerator(5)
    with pytest.raises(ValueError):
        x.dropout(p, generator=generator)
    assert generator.calls == 0
    assert generator._has_active_reservation() is False
    x.close()


@pytest.mark.parametrize("p", [True, False, np.bool_(True)])
def test_a_bool_probability_is_a_type_error_not_an_integer(p):
    """``True`` is not ``1`` and ``False`` is not ``0``: a bool never
    reaches the identity path or the kernel."""
    x = NativeTensor.from_array(np.arange(4.0), requires_grad=True)
    generator = NativeGenerator(5)
    with pytest.raises(TypeError, match="bool"):
        x.dropout(p, generator=generator)
    assert generator.calls == 0
    x.close()


@pytest.mark.parametrize("p", [None, "0.5", [0.5], complex(0.5, 0)])
def test_a_non_real_probability_is_rejected(p):
    x = NativeTensor.from_array(np.arange(4.0))
    generator = NativeGenerator(5)
    with pytest.raises(TypeError):
        x.dropout(p, generator=generator)
    assert generator.calls == 0
    x.close()


@pytest.mark.parametrize("p", [0.5, np.float64(0.5), np.float32(0.5)])
def test_real_scalar_probabilities_are_accepted_like_the_core(p):
    """The operation reuses the Core's normalizer rather than defining a
    second rule, so the accepted set is identical by construction."""
    values = np.arange(1.0, 13.0)
    x = NativeTensor.from_array(values, requires_grad=True)
    generator = NativeGenerator(11)
    y = x.dropout(p, generator=generator)
    expected, _ = core_reference(values, float(p), 11, 0)
    assert np.array_equal(y.to_numpy(), expected)
    assert generator.calls == 1
    y.close()
    x.close()


def test_an_exhausted_generator_is_refused_before_any_allocation(live_storages):
    generator = NativeGenerator(2)
    generator.load_state({
        "algorithm": generator.algorithm,
        "algorithm_version": generator.algorithm_version,
        "seed": 2,
        "calls": 2 ** 64 - 1,
    })
    x = NativeTensor.from_array(np.arange(4.0), requires_grad=True)
    baseline = len(live_storages)
    with pytest.raises(RuntimeError, match="exhausted"):
        x.dropout(0.5, generator=generator)
    assert len(live_storages) == baseline
    assert generator.calls == 2 ** 64 - 1
    x.close()


def test_the_last_representable_call_index_still_works():
    """``calls`` is a count, so ``2**64 - 2`` is the last usable index and
    ``2**64 - 1`` is a reachable count (design §4.6)."""
    last_index = 2 ** 64 - 2
    generator = NativeGenerator(0)
    generator.load_state({
        "algorithm": generator.algorithm,
        "algorithm_version": generator.algorithm_version,
        "seed": 0,
        "calls": last_index,
    })
    values = np.arange(1.0, 13.0)
    x = NativeTensor.from_array(values, requires_grad=True)
    y = x.dropout(0.75, generator=generator)
    expected, _ = core_reference(values, 0.75, 0, last_index)
    assert np.array_equal(y.to_numpy(), expected)
    assert generator.calls == 2 ** 64 - 1
    y.close()
    x.close()


def test_no_public_surface_exposes_the_mask_or_the_token():
    values = np.arange(1.0, 13.0)
    x = NativeTensor.from_array(values, requires_grad=True)
    generator = NativeGenerator(1)
    y = x.dropout(0.5, generator=generator)
    for leaked in ("mask", "multiplier", "token", "reservation", "generator",
                   "seed", "call_index"):
        assert not hasattr(y, leaked), leaked
    # The mask is reachable only through the private graph-resource slot,
    # and it is a Core object, never a NativeTensor.
    assert not isinstance(saved_masks(y)[0], NativeTensor)
    assert isinstance(saved_masks(y)[0], cpp.NativeTensorCore)
    y.close()
    x.close()


# ==========================================================================
# 2. p == 0: identity, and nothing else
# ==========================================================================

@pytest.mark.parametrize("zero", [0.0, 0, np.float64(0.0), -0.0])
def test_p_zero_returns_the_input_object_itself(zero):
    values = np.arange(1.0, 13.0)
    x = NativeTensor.from_array(values, requires_grad=True)
    generator = NativeGenerator(9)
    result = x.dropout(zero, generator=generator)
    assert result is x
    assert generator.calls == 0
    assert generator._has_active_reservation() is False
    assert x.requires_grad is True
    assert x.is_leaf is True
    assert x._graph_resources == ()
    assert np.array_equal(x.to_numpy(), values)
    x.close()


def test_p_zero_allocates_nothing_and_calls_no_kernel(monkeypatch,
                                                      live_storages):
    """Proved with a seam, not by inference: the Core forward is replaced
    by a tripwire that fails the test if it is ever reached."""
    calls = []

    def tripwire(self, *args, **kwargs):
        calls.append(1)
        raise AssertionError("p == 0 must not reach the Dropout Core")

    monkeypatch.setattr(
        cpp.NativeTensorCore, "_dropout_forward_with_mask", tripwire
    )
    monkeypatch.setattr(
        cpp.NativeTensorCore, "dropout_forward", tripwire
    )
    # And no reservation may be minted either.
    reservations = []
    original_reserve = NativeGenerator._reserve_call

    def counting_reserve(self):
        reservations.append(1)
        return original_reserve(self)

    monkeypatch.setattr(NativeGenerator, "_reserve_call", counting_reserve)

    x = NativeTensor.from_array(np.arange(1.0, 13.0), requires_grad=True)
    generator = NativeGenerator(9)
    baseline = len(live_storages)
    result = x.dropout(0.0, generator=generator)
    assert result is x
    assert calls == []
    assert reservations == []
    assert len(live_storages) == baseline
    assert generator.calls == 0
    x.close()


def test_p_zero_still_validates_the_receiver_generator_and_probability():
    """Identity is the *result*, not a bypass of the contract."""
    closed = NativeTensor.from_array(np.arange(4.0))
    closed.close()
    with pytest.raises(RuntimeError):
        closed.dropout(0.0, generator=NativeGenerator(1))

    x = NativeTensor.from_array(np.arange(4.0))
    with pytest.raises(TypeError, match="NativeGenerator"):
        x.dropout(0.0, generator=None)
    with pytest.raises(TypeError, match="bool"):
        x.dropout(False, generator=NativeGenerator(1))
    x.close()


def test_p_zero_does_not_disturb_a_generator_mid_stream():
    values = np.arange(1.0, 13.0)
    x = NativeTensor.from_array(values, requires_grad=True)
    generator = NativeGenerator(21)
    first = x.dropout(0.5, generator=generator)
    assert generator.calls == 1
    identity = x.dropout(0.0, generator=generator)
    assert identity is x
    assert generator.calls == 1
    second = x.dropout(0.5, generator=generator)
    # The interposed identity consumed nothing, so this is call index 1.
    expected, _ = core_reference(values, 0.5, 21, 1)
    assert np.array_equal(second.to_numpy(), expected)
    assert generator.calls == 2
    for t in (first, second, x):
        t.close()


# ==========================================================================
# 3. Determinism, committed vectors, and call indices
# ==========================================================================

def test_forward_reproduces_the_committed_g2_vector():
    """The tie between G3 and G2's known-answer behavior: the same seed,
    call index, and probability the C++ CTest and the Core suite commit."""
    values = np.arange(1.0, 13.0)
    x = NativeTensor.from_array(values, requires_grad=True)
    generator = NativeGenerator(VECTOR_SEED)
    assert generator.calls == VECTOR_CALL_INDEX

    y = x.dropout(VECTOR_P, generator=generator)
    mask = committed_mask()
    assert np.array_equal(saved_mask_array(y), mask)
    assert np.array_equal(y.to_numpy(), values * mask)

    y.backward(gradient=ones_like(y))
    assert np.array_equal(x.grad.to_numpy(), mask)
    assert generator.calls == 1
    y.close()
    x.close()


def test_forward_and_gradient_equal_the_core_reference_exactly():
    values = np.arange(1.0, 13.0).reshape(3, 4)
    reference_out, reference_mask = core_reference(values, 0.4, 99, 0)

    x = NativeTensor.from_array(values, requires_grad=True)
    generator = NativeGenerator(99)
    y = x.dropout(0.4, generator=generator)
    assert np.array_equal(y.to_numpy(), reference_out)
    assert np.array_equal(saved_mask_array(y), reference_mask)
    y.backward(gradient=ones_like(y))
    assert np.array_equal(x.grad.to_numpy(), reference_mask)
    assert generator.calls == 1
    y.close()
    x.close()


def test_gradient_is_upstream_times_the_mask_for_a_nontrivial_upstream():
    values = np.arange(1.0, 13.0).reshape(3, 4)
    _, reference_mask = core_reference(values, 0.4, 99, 0)
    upstream = np.arange(12.0, 0.0, -1.0).reshape(3, 4)

    x = NativeTensor.from_array(values, requires_grad=True)
    generator = NativeGenerator(99)
    y = x.dropout(0.4, generator=generator)
    g = NativeTensor.from_array(upstream)
    y.backward(gradient=g)
    assert np.allclose(x.grad.to_numpy(), upstream * reference_mask,
                       atol=1e-12)
    for t in (y, g, x):
        t.close()


def test_gradient_matches_finite_differences_on_the_fixed_mask():
    """Dropout's randomness is fixed once the mask is drawn, so the map
    from input to output is linear and finite differences are exact to
    tolerance for the forward that actually ran."""
    values = np.array([1.0, -2.0, 3.0, -4.0, 5.0, 6.0])
    _, mask = core_reference(values, 0.5, 4242, 0)
    upstream = np.array([0.5, 1.5, -2.0, 3.0, 0.25, -1.0])
    eps = 1e-6

    def loss(sample):
        # The same fixed mask, applied by hand: this is the function whose
        # gradient the graph claims to compute.
        return float(np.sum(sample * mask * upstream))

    numerical = np.empty_like(values)
    for i in range(values.size):
        up = values.copy()
        down = values.copy()
        up[i] += eps
        down[i] -= eps
        numerical[i] = (loss(up) - loss(down)) / (2 * eps)

    x = NativeTensor.from_array(values, requires_grad=True)
    generator = NativeGenerator(4242)
    y = x.dropout(0.5, generator=generator)
    g = NativeTensor.from_array(upstream)
    y.backward(gradient=g)
    assert np.allclose(x.grad.to_numpy(), numerical, atol=1e-6)
    for t in (y, g, x):
        t.close()


def test_two_successive_calls_use_consecutive_indices():
    values = np.arange(1.0, 13.0)
    x = NativeTensor.from_array(values, requires_grad=True)
    generator = NativeGenerator(31)
    first = x.dropout(0.5, generator=generator)
    second = x.dropout(0.5, generator=generator)
    assert generator.calls == 2
    expected_first, _ = core_reference(values, 0.5, 31, 0)
    expected_second, _ = core_reference(values, 0.5, 31, 1)
    assert np.array_equal(first.to_numpy(), expected_first)
    assert np.array_equal(second.to_numpy(), expected_second)
    for t in (first, second, x):
        t.close()


def test_independent_generators_with_equal_state_agree():
    values = np.arange(1.0, 13.0)
    a = NativeTensor.from_array(values, requires_grad=True)
    b = NativeTensor.from_array(values, requires_grad=True)
    ga, gb = NativeGenerator(77), NativeGenerator(77)
    assert ga is not gb
    ya = a.dropout(0.3, generator=ga)
    yb = b.dropout(0.3, generator=gb)
    assert np.array_equal(ya.to_numpy(), yb.to_numpy())
    assert ga.calls == gb.calls == 1
    for t in (ya, yb, a, b):
        t.close()


def test_one_shared_generator_serves_two_tensors_as_one_ordered_stream():
    values = np.arange(1.0, 13.0)
    a = NativeTensor.from_array(values, requires_grad=True)
    b = NativeTensor.from_array(values * 2.0, requires_grad=True)
    shared = NativeGenerator(1001)
    ya = a.dropout(0.5, generator=shared)
    yb = b.dropout(0.5, generator=shared)
    assert shared.calls == 2
    _, mask0 = core_reference(values, 0.5, 1001, 0)
    _, mask1 = core_reference(values, 0.5, 1001, 1)
    assert np.array_equal(saved_mask_array(ya), mask0)
    assert np.array_equal(saved_mask_array(yb), mask1)
    for t in (ya, yb, a, b):
        t.close()


def test_a_failed_call_leaves_its_index_for_the_next_success(monkeypatch):
    """Cancellation does not skip a committed known-answer vector: the
    forward after a failure reproduces exactly what the failed one would
    have."""
    values = np.arange(1.0, 13.0)
    x = NativeTensor.from_array(values, requires_grad=True)
    generator = NativeGenerator(VECTOR_SEED)

    def boom(result):
        raise RuntimeError("injected delivery failure")

    monkeypatch.setattr(
        native_tensor_module, "_deliver_dropout_result", boom
    )
    with pytest.raises(RuntimeError, match="injected delivery failure"):
        x.dropout(VECTOR_P, generator=generator)
    assert generator.calls == 0
    monkeypatch.undo()

    y = x.dropout(VECTOR_P, generator=generator)
    assert np.array_equal(saved_mask_array(y), committed_mask())
    assert generator.calls == 1
    y.close()
    x.close()


def test_direct_core_calls_never_touch_generator_state():
    """The G2 separation, restated from above: a Core call is not a
    generator call and cannot be mistaken for one."""
    generator = NativeGenerator(5150)
    before = generator.state()
    source = cpp.NativeTensorCore.from_array(np.arange(1.0, 13.0))
    for index in range(3):
        out = source.dropout_forward(0.5, seed=5150, call_index=index)
        out.close()
    assert generator.state() == before
    assert generator._has_active_reservation() is False
    source.close()


def test_the_operation_never_reads_the_committed_counter_as_an_index(
    monkeypatch,
):
    """A subtle bug this pins shut: reading ``generator.calls`` *after*
    reserving would still give the right index today (commit happens
    later), but it is the committed count, not the reservation's. The
    operation must take the index from the token, so a generator whose
    ``calls`` reports something else still produces the reserved stream."""
    values = np.arange(1.0, 13.0)
    x = NativeTensor.from_array(values, requires_grad=True)
    generator = NativeGenerator(64)

    # Make `.calls` lie by a wide margin while leaving the real counter,
    # the reservation protocol, and the seed intact.
    original_calls = NativeGenerator.calls

    monkeypatch.setattr(
        NativeGenerator, "calls", property(lambda self: 4242),
    )
    y = x.dropout(0.5, generator=generator)
    monkeypatch.undo()

    expected, _ = core_reference(values, 0.5, 64, 0)
    assert np.array_equal(y.to_numpy(), expected), (
        "the operation used a counter read instead of the reserved index"
    )
    assert original_calls is NativeGenerator.calls
    assert generator.calls == 1
    y.close()
    x.close()


# ==========================================================================
# 4. Forward output behavior
# ==========================================================================

@pytest.mark.parametrize(
    "shape", [(1,), (5,), (3, 4), (2, 3, 4), (2, 2, 2, 2), (2, 1, 2, 1, 3)],
)
def test_shapes_are_preserved_across_ranks(shape):
    values = np.arange(1.0, 1.0 + int(np.prod(shape))).reshape(shape)
    x = NativeTensor.from_array(values, requires_grad=True)
    generator = NativeGenerator(8)
    y = x.dropout(0.5, generator=generator)
    assert y.shape == shape
    expected, _ = core_reference(values, 0.5, 8, 0)
    assert np.array_equal(y.to_numpy(), expected)
    assert generator.calls == 1
    y.close()
    x.close()


def test_a_scalar_tensor_gets_one_draw_and_one_call():
    x = NativeTensor.full((), 3.0, requires_grad=True)
    generator = NativeGenerator(12)
    y = x.dropout(0.5, generator=generator)
    assert y.shape == ()
    assert y.numel == 1
    # One element, so the reference is element 0 of that stream. (The
    # Core helper cannot express rank 0 through ``from_array``, which
    # promotes a 0-d array to shape ``(1,)`` — the values are the point.)
    expected, mask = core_reference([3.0], 0.5, 12, 0)
    assert float(y.to_numpy()) == float(expected[0])
    y.backward()
    assert float(x.grad.to_numpy()) == float(mask[0])
    assert generator.calls == 1
    y.close()
    x.close()


def test_the_output_is_fresh_owning_and_does_not_alias_the_input():
    values = np.arange(1.0, 13.0)
    x = NativeTensor.from_array(values, requires_grad=True)
    generator = NativeGenerator(13)
    y = x.dropout(0.5, generator=generator)
    assert y is not x
    assert y.owns_core is True
    assert y._core._storage is not x._core._storage
    assert saved_masks(y)[0]._storage is not x._core._storage
    assert saved_masks(y)[0]._storage is not y._core._storage
    assert y.contiguous is True
    # The input is untouched in value and in metadata.
    assert np.array_equal(x.to_numpy(), values)
    assert x.shape == (12,) and x.closed is False and x.requires_grad is True
    y.close()
    x.close()


@pytest.mark.parametrize("build", ["transpose", "narrow", "offset"])
def test_noncontiguous_views_get_the_same_logical_mask(build):
    """Policy B materializes first, so the mask is a function of the
    *logical* row-major index — never of the physical strides."""
    base = np.arange(1.0, 13.0).reshape(3, 4)
    x = NativeTensor.from_array(base, requires_grad=True)
    if build == "transpose":
        view = x.T                      # (4, 3), non-contiguous
    elif build == "narrow":
        view = x.narrow(1, 1, 2)        # (3, 2), non-contiguous
    else:
        view = x.narrow(0, 1, 2)        # (2, 4), contiguous, nonzero offset
    generator = NativeGenerator(55)
    y = view.dropout(0.5, generator=generator)

    expected_out, expected_mask = core_reference(
        view.to_numpy(), 0.5, 55, 0
    )
    assert np.array_equal(y.to_numpy(), expected_out)
    assert np.array_equal(saved_mask_array(y), expected_mask)
    # The mask is the one a contiguous tensor of the same logical shape
    # would have received.
    _, contiguous_mask = core_reference(
        np.ascontiguousarray(view.to_numpy()), 0.5, 55, 0
    )
    assert np.array_equal(saved_mask_array(y), contiguous_mask)
    assert generator.calls == 1

    y.backward(gradient=ones_like(y))
    assert x.grad is not None
    for t in (y, view, x):
        t.close()


def test_requires_grad_follows_the_input():
    values = np.arange(1.0, 13.0)
    generator = NativeGenerator(17)

    tracked = NativeTensor.from_array(values, requires_grad=True)
    y = tracked.dropout(0.5, generator=generator)
    assert y.requires_grad is True and y.is_leaf is False and y._op == "dropout"
    assert y._parents == (tracked,)

    plain = NativeTensor.from_array(values)
    z = plain.dropout(0.5, generator=generator)
    assert z.requires_grad is False and z.is_leaf is True
    assert z._parents == () and z._backward is None

    for t in (y, z, tracked, plain):
        t.close()


def test_a_parameter_input_produces_a_gradient_on_the_parameter():
    values = np.arange(1.0, 13.0).reshape(3, 4)
    parameter = NativeParameter(values)
    generator = NativeGenerator(23)
    y = parameter.dropout(0.5, generator=generator)
    _, mask = core_reference(values, 0.5, 23, 0)
    y.backward(gradient=ones_like(y))
    assert np.array_equal(parameter.grad.to_numpy(), mask)
    y.close()
    parameter.close()


def test_dropout_composes_inside_a_larger_graph():
    values = np.arange(1.0, 13.0).reshape(3, 4)
    weight = NativeParameter(np.full((4, 1), 0.5))
    x = NativeParameter(values)
    generator = NativeGenerator(29)
    dropped = x.dropout(0.5, generator=generator)
    loss = dropped.matmul(weight).sum()
    loss.backward()
    _, mask = core_reference(values, 0.5, 29, 0)
    assert np.allclose(x.grad.to_numpy(), mask * 0.5, atol=1e-12)
    assert generator.calls == 1
    for t in (loss, dropped, x, weight):
        t.close()


# ==========================================================================
# 5. No-grad behavior
# ==========================================================================

def test_a_no_grad_forward_consumes_a_call_and_keeps_no_graph(live_storages):
    values = np.arange(1.0, 13.0)
    x = NativeTensor.from_array(values)          # requires_grad=False
    baseline = len(live_storages)
    generator = NativeGenerator(37)
    y = x.dropout(0.5, generator=generator)

    assert generator.calls == 1, "a draw happened, so a call was consumed"
    assert y.requires_grad is False
    assert y._graph_resources == (), "the mask must not survive a no-grad forward"
    assert y._backward is None and y._parents == ()
    expected, _ = core_reference(values, 0.5, 37, 0)
    assert np.array_equal(y.to_numpy(), expected)
    # Only the output remains allocated: the mask was closed immediately.
    assert len(live_storages) == baseline + 1
    y.close()
    assert len(live_storages) == baseline
    x.close()


def test_the_no_grad_mask_is_closed_before_the_call_returns(monkeypatch):
    """Not "eventually collected" — closed by ``_from_op`` on the way out,
    with the object captured so its state can be read directly."""
    produced = []
    original = cpp.NativeTensorCore._dropout_forward_with_mask

    def capturing(self, p, *, seed, call_index):
        out, mask = original(self, p, seed=seed, call_index=call_index)
        produced.append((out, mask))
        return out, mask

    monkeypatch.setattr(
        cpp.NativeTensorCore, "_dropout_forward_with_mask", capturing
    )
    x = NativeTensor.from_array(np.arange(1.0, 13.0))
    generator = NativeGenerator(41)
    y = x.dropout(0.5, generator=generator)
    out_core, mask = produced[0]
    assert mask._closed is True
    assert out_core._closed is False
    assert generator.calls == 1
    y.close()
    x.close()


def test_a_detached_input_takes_the_no_grad_path_and_still_consumes_a_call():
    """The framework has no ``no_grad()`` context manager (the native line
    opts *in* to a graph); ``detach()`` is the equivalent, and it must not
    change the transaction."""
    values = np.arange(1.0, 13.0)
    x = NativeTensor.from_array(values, requires_grad=True)
    detached = x.detach()
    assert detached.requires_grad is False
    generator = NativeGenerator(43)
    y = detached.dropout(0.5, generator=generator)
    assert generator.calls == 1
    assert y.requires_grad is False
    assert y._graph_resources == ()
    expected, _ = core_reference(values, 0.5, 43, 0)
    assert np.array_equal(y.to_numpy(), expected)
    for t in (y, detached, x):
        t.close()


def test_a_no_grad_forward_advances_the_stream_for_the_next_grad_forward():
    values = np.arange(1.0, 13.0)
    plain = NativeTensor.from_array(values)
    tracked = NativeTensor.from_array(values, requires_grad=True)
    generator = NativeGenerator(47)
    first = plain.dropout(0.5, generator=generator)
    second = tracked.dropout(0.5, generator=generator)
    _, mask1 = core_reference(values, 0.5, 47, 1)
    assert np.array_equal(saved_mask_array(second), mask1)
    assert generator.calls == 2
    for t in (first, second, plain, tracked):
        t.close()


# ==========================================================================
# 6. Saved-mask ownership and graph lifetime
# ==========================================================================

def test_the_mask_is_graph_owned_and_released_by_a_one_shot_backward():
    x = NativeParameter(np.arange(1.0, 13.0).reshape(3, 4))
    generator = NativeGenerator(51)
    y = x.dropout(0.5, generator=generator)
    mask = saved_masks(y)[0]
    assert mask._closed is False
    y.backward(gradient=ones_like(y))
    assert mask._closed is True
    assert y._graph_resources == ()
    assert y._graph_freed is True
    y.close()
    x.close()


def test_retain_graph_keeps_the_mask_for_another_backward():
    x = NativeParameter(np.arange(1.0, 13.0).reshape(3, 4))
    generator = NativeGenerator(53)
    y = x.dropout(0.5, generator=generator)
    mask = saved_masks(y)[0]
    g = ones_like(y)
    y.backward(gradient=g, retain_graph=True)
    assert mask._closed is False
    once = x.grad.to_numpy().copy()
    y.backward(gradient=g)                    # second pass, one-shot
    assert np.allclose(x.grad.to_numpy(), 2 * once, atol=1e-12)
    assert mask._closed is True
    assert y._graph_resources == ()
    assert generator.calls == 1, "backward consumes no call, ever"
    for t in (y, g, x):
        t.close()


def test_repeated_backward_after_release_raises_without_double_closing():
    x = NativeParameter(np.arange(1.0, 13.0).reshape(3, 4))
    generator = NativeGenerator(59)
    y = x.dropout(0.5, generator=generator)
    mask = saved_masks(y)[0]
    g = ones_like(y)
    y.backward(gradient=g)
    assert mask._closed is True
    with pytest.raises(RuntimeError, match="freed autograd graph"):
        y.backward(gradient=g)
    assert mask._closed is True               # still closed exactly once
    assert generator.calls == 1
    for t in (y, g, x):
        t.close()


def test_closing_an_abandoned_graph_releases_the_mask():
    x = NativeParameter(np.arange(1.0, 13.0).reshape(3, 4))
    generator = NativeGenerator(61)
    y = x.dropout(0.5, generator=generator)
    mask = saved_masks(y)[0]
    assert mask._closed is False
    y.close()                                  # abandoned without backward
    assert mask._closed is True
    y.close()                                  # idempotent, no double close
    assert mask._closed is True
    assert generator.calls == 1
    x.close()


def test_dropping_an_abandoned_graph_does_not_leak_the_mask():
    """The ``__del__`` refcount/GC *fallback* — not a deterministic release
    point. The two deterministic ones are covered above; this only proves
    the safety net also frees the mask."""
    x = NativeParameter(np.arange(1.0, 13.0).reshape(3, 4))
    generator = NativeGenerator(67)
    holder = []

    def build():
        y = x.dropout(0.5, generator=generator)
        holder.append(saved_masks(y)[0])
        # y leaves scope without close() or backward()

    build()
    gc.collect()
    assert holder[0]._closed is True
    x.close()


def test_live_storage_returns_to_baseline_across_the_whole_lifecycle(
    live_storages,
):
    x = NativeParameter(np.arange(1.0, 13.0).reshape(3, 4))
    generator = NativeGenerator(71)
    baseline = len(live_storages)

    y = x.dropout(0.5, generator=generator)
    assert len(live_storages) == baseline + 2      # output + saved mask
    g = ones_like(y)
    assert len(live_storages) == baseline + 3      # ...plus the upstream
    y.backward(gradient=g)                         # releases the mask
    # The mask went, the leaf gradient arrived: output + upstream + grad.
    assert len(live_storages) == baseline + 3
    y.close()
    g.close()
    x.zero_grad()
    gc.collect()
    assert len(live_storages) == baseline
    x.close()


def test_a_failed_retryable_backward_keeps_the_mask_alive():
    """A stale-parameter error elsewhere in the graph must leave the saved
    mask intact, so the pass is genuinely retryable."""
    x = NativeParameter(np.arange(1.0, 13.0).reshape(3, 4))
    scale = NativeParameter(np.full((3, 4), 2.0))
    generator = NativeGenerator(73)
    dropped = x.dropout(0.5, generator=generator)
    # `multiply` DOES record a version for its direct parameter operands.
    product = dropped.multiply(scale)
    mask = saved_masks(dropped)[0]

    replacement = NativeTensor.from_array(np.full((3, 4), 3.0))
    scale.copy_value_(replacement)             # stales the multiply edge

    with pytest.raises(RuntimeError, match="stale parameter value"):
        product.backward(gradient=ones_like(product))
    assert mask._closed is False, "a retryable failure must not free state"
    assert dropped._graph_resources == (mask,)
    assert generator.calls == 1
    for t in (product, dropped, replacement, scale, x):
        t.close()
    assert mask._closed is True


def test_two_dropouts_in_one_graph_own_two_masks_released_together():
    x = NativeParameter(np.arange(1.0, 13.0).reshape(3, 4))
    generator = NativeGenerator(79)
    first = x.dropout(0.5, generator=generator)
    second = first.dropout(0.25, generator=generator)
    masks = (saved_masks(first)[0], saved_masks(second)[0])
    assert masks[0] is not masks[1]
    assert generator.calls == 2
    second.backward(gradient=ones_like(second))
    assert all(mask._closed for mask in masks)
    # The chained gradient is the product of the two masks.
    _, mask0 = core_reference(x.to_numpy(), 0.5, 79, 0)
    _, mask1 = core_reference(first.to_numpy(), 0.25, 79, 1)
    assert np.allclose(x.grad.to_numpy(), mask0 * mask1, atol=1e-12)
    for t in (second, first, x):
        t.close()


def test_a_dropout_mask_coexists_with_a_maxpool_winner_buffer():
    """The saved-state family shares one mechanism, so two members in one
    graph must release exactly once each."""
    x = NativeParameter(np.arange(16.0).reshape(1, 1, 4, 4))
    generator = NativeGenerator(83)
    pooled = x.maxpool2d(kernel_size=2)
    dropped = pooled.dropout(0.5, generator=generator)
    winners = saved_masks(pooled)[0]
    mask = saved_masks(dropped)[0]
    dropped.backward(gradient=ones_like(dropped))
    assert winners._closed is True and mask._closed is True
    assert pooled._graph_resources == () and dropped._graph_resources == ()
    assert generator.calls == 1
    for t in (dropped, pooled, x):
        t.close()


# ==========================================================================
# 7. Backward read contract and version behavior
# ==========================================================================

def test_no_expected_version_is_recorded():
    x = NativeParameter(np.arange(1.0, 13.0).reshape(3, 4))
    generator = NativeGenerator(89)
    y = x.dropout(0.5, generator=generator)
    assert y._expected_versions == (), (
        "dropout backward reads only the saved mask, so it must record no "
        "parameter version (the maxpool2d/cross_entropy archetype)"
    )
    y.close()
    x.close()


def test_input_mutation_after_forward_neither_raises_nor_changes_the_gradient():
    values = np.arange(1.0, 13.0).reshape(3, 4)
    x = NativeParameter(values)
    generator = NativeGenerator(97)
    y = x.dropout(0.5, generator=generator)
    _, mask = core_reference(values, 0.5, 97, 0)

    replacement = NativeTensor.from_array(np.zeros((3, 4)))
    x.copy_value_(replacement)                 # bumps the value version
    assert x._version > 0

    y.backward(gradient=ones_like(y))          # must not raise stale-graph
    assert np.array_equal(x.grad.to_numpy(), mask)
    assert generator.calls == 1
    for t in (y, replacement, x):
        t.close()


def test_the_gradient_is_identical_to_a_clean_control_after_input_mutation():
    values = np.arange(1.0, 13.0).reshape(3, 4)
    control_x = NativeParameter(values)
    control_y = control_x.dropout(0.5, generator=NativeGenerator(101))
    control_y.backward(gradient=ones_like(control_y))
    control = control_x.grad.to_numpy().copy()

    x = NativeParameter(values)
    y = x.dropout(0.5, generator=NativeGenerator(101))
    replacement = NativeTensor.from_array(np.full((3, 4), -7.0))
    x.copy_value_(replacement)
    y.backward(gradient=ones_like(y))
    assert np.array_equal(x.grad.to_numpy(), control)
    for t in (control_y, control_x, y, replacement, x):
        t.close()


def test_backward_never_calls_the_random_core_or_touches_the_generator(
    monkeypatch,
):
    values = np.arange(1.0, 13.0).reshape(3, 4)
    x = NativeParameter(values)
    generator = NativeGenerator(103)
    y = x.dropout(0.5, generator=generator)
    before = generator.state()

    # Every generator entry point becomes a tripwire for the backward pass.
    for name in ("_reserve_call", "_commit_call", "_abandon_call",
                 "load_state", "reseed", "reset"):
        monkeypatch.setattr(
            NativeGenerator, name,
            lambda *args, _n=name, **kwargs: (_ for _ in ()).throw(
                AssertionError(f"backward reached NativeGenerator.{_n}")
            ),
        )
    for name in ("_dropout_forward_with_mask", "dropout_forward"):
        monkeypatch.setattr(
            cpp.NativeTensorCore, name,
            lambda *args, _n=name, **kwargs: (_ for _ in ()).throw(
                AssertionError(f"backward reached the random Core ({_n})")
            ),
        )

    y.backward(gradient=ones_like(y))
    monkeypatch.undo()
    assert generator.state() == before
    _, mask = core_reference(values, 0.5, 103, 0)
    assert np.array_equal(x.grad.to_numpy(), mask)
    for t in (y, x):
        t.close()


def test_backward_does_not_mutate_the_saved_mask():
    x = NativeParameter(np.arange(1.0, 13.0).reshape(3, 4))
    generator = NativeGenerator(107)
    y = x.dropout(0.5, generator=generator)
    mask = saved_masks(y)[0]
    before = mask.to_numpy().copy()
    g = NativeTensor.from_array(np.full((3, 4), 5.0))
    y.backward(gradient=g, retain_graph=True)
    assert np.array_equal(mask.to_numpy(), before)
    y.backward(gradient=g)                     # a second identical pass
    for t in (y, g, x):
        t.close()


def test_higher_order_differentiation_is_not_offered():
    """Matched to the native line's policy, not invented for Dropout: the
    backward computes at the graph-unaware Core level, so the gradient it
    produces carries no graph of its own."""
    x = NativeParameter(np.arange(1.0, 13.0).reshape(3, 4))
    generator = NativeGenerator(109)
    y = x.dropout(0.5, generator=generator)
    y.backward(gradient=ones_like(y))
    assert x.grad.requires_grad is False
    assert x.grad.is_leaf is True
    assert x.grad._parents == () and x.grad._backward is None
    y.close()
    x.close()


# ==========================================================================
# 8. Generator changes after the forward
# ==========================================================================

@pytest.mark.parametrize("mutate", ["reseed", "reset", "load_state",
                                    "load_generator_state_dict"])
def test_later_generator_changes_cannot_reach_an_existing_graph(mutate):
    values = np.arange(1.0, 13.0).reshape(3, 4)
    _, expected_mask = core_reference(values, 0.5, 113, 0)

    x = NativeParameter(values)
    generator = NativeGenerator(113)
    y = x.dropout(0.5, generator=generator)
    saved = saved_mask_array(y)
    forward = y.to_numpy().copy()
    mask_core = saved_masks(y)[0]

    if mutate == "reseed":
        generator.reseed(999_999)
    elif mutate == "reset":
        generator.reset()
    elif mutate == "load_state":
        generator.load_state({
            "algorithm": generator.algorithm,
            "algorithm_version": generator.algorithm_version,
            "seed": 4, "calls": 900,
        })
    else:
        from tensorforge.experimental import NativeModule

        module = NativeModule()
        module.g = generator
        module.load_generator_state_dict({
            "g": {
                "algorithm": generator.algorithm,
                "algorithm_version": generator.algorithm_version,
                "seed": 12345, "calls": 77,
            }
        })

    calls_after_mutation = generator.calls
    # Nothing about the graph moved.
    assert np.array_equal(y.to_numpy(), forward)
    assert np.array_equal(saved_mask_array(y), saved)
    assert np.array_equal(saved, expected_mask)
    assert mask_core._closed is False

    y.backward(gradient=ones_like(y))
    assert np.array_equal(x.grad.to_numpy(), expected_mask)
    assert generator.calls == calls_after_mutation, (
        "backward must not advance a generator, changed or not"
    )
    for t in (y, x):
        t.close()


def test_destroying_every_reference_to_the_generator_leaves_the_graph_valid():
    values = np.arange(1.0, 13.0).reshape(3, 4)
    _, expected_mask = core_reference(values, 0.5, 127, 0)
    x = NativeParameter(values)
    generator = NativeGenerator(127)
    y = x.dropout(0.5, generator=generator)
    del generator
    gc.collect()
    y.backward(gradient=ones_like(y))
    assert np.array_equal(x.grad.to_numpy(), expected_mask)
    y.close()
    x.close()


def test_a_full_checkpoint_load_stales_the_graph_through_parameters_only(
    tmp_path,
):
    """The distinction §8.2 draws: a generator change is invisible to an
    existing graph, while a *parameter* change is caught by the unchanged
    v3.7 rule — through some other node, never through Dropout's."""
    from tensorforge.experimental import (
        NativeLinear, NativeSGD, load_native_checkpoint,
        save_native_checkpoint,
    )

    model = NativeLinear(4, 2)
    optimizer = NativeSGD(model.parameters(), lr=0.1)
    path = tmp_path / "ckpt.npz"
    save_native_checkpoint(path, model, optimizer)

    x = NativeTensor.from_array(np.arange(1.0, 13.0).reshape(3, 4),
                                requires_grad=True)
    generator = NativeGenerator(131)
    dropped = x.dropout(0.5, generator=generator)
    out = model(dropped)                        # a matmul against `weight`

    load_native_checkpoint(path, model, optimizer)   # bumps parameter versions
    with pytest.raises(RuntimeError, match="stale parameter value"):
        out.backward(gradient=ones_like(out))
    # ...and the Dropout node's own state is untouched and still valid.
    assert saved_masks(dropped)[0]._closed is False
    assert dropped._expected_versions == ()
    dropped.backward(gradient=ones_like(dropped))
    assert x.grad is not None
    for t in (out, dropped, x):
        t.close()
    for _, parameter in model.named_parameters():
        parameter.close()


# ==========================================================================
# 9. Concurrency and reentrancy
# ==========================================================================

def _blocking_core(entered, release):
    """A Core forward that parks inside the operation, between the
    published reservation and the commit."""
    original = cpp.NativeTensorCore._dropout_forward_with_mask

    def blocking(self, p, *, seed, call_index):
        entered.set()
        assert release.wait(JOIN_TIMEOUT), "the gate was never released"
        return original(self, p, seed=seed, call_index=call_index)

    return blocking


def test_a_second_dropout_on_a_busy_generator_fails_and_consumes_nothing(
    monkeypatch,
):
    entered = threading.Event()
    release = threading.Event()
    monkeypatch.setattr(
        cpp.NativeTensorCore, "_dropout_forward_with_mask",
        _blocking_core(entered, release),
    )
    values = np.arange(1.0, 13.0)
    a = NativeTensor.from_array(values, requires_grad=True)
    b = NativeTensor.from_array(values, requires_grad=True)
    generator = NativeGenerator(137)
    outcome = {}

    def first():
        try:
            outcome["first"] = a.dropout(0.5, generator=generator)
        except BaseException as error:          # pragma: no cover - failure
            outcome["first_error"] = error

    thread = threading.Thread(target=first, daemon=True)
    thread.start()
    assert entered.wait(JOIN_TIMEOUT), "the first forward never started"

    # The first operation holds a published reservation right now.
    assert generator._has_active_reservation() is True
    with pytest.raises(RuntimeError, match="outstanding call reservation"):
        b.dropout(0.5, generator=generator)
    assert generator.calls == 0, "the refused caller consumed nothing"

    # Replacement is refused for the same reason, and changes nothing.
    with pytest.raises(RuntimeError):
        generator.reseed(5)
    with pytest.raises(RuntimeError):
        generator.reset()
    assert generator.seed == 137 and generator.calls == 0

    release.set()
    thread.join(JOIN_TIMEOUT)
    assert not thread.is_alive()
    assert "first_error" not in outcome, outcome.get("first_error")
    assert generator.calls == 1
    monkeypatch.undo()

    # The refused caller can now proceed, and takes the *next* index.
    second = b.dropout(0.5, generator=generator)
    expected, _ = core_reference(values, 0.5, 137, 1)
    assert np.array_equal(second.to_numpy(), expected)
    assert generator.calls == 2
    for t in (outcome["first"], second, a, b):
        t.close()


def test_a_failed_first_operation_frees_the_index_for_the_waiting_caller(
    monkeypatch,
):
    entered = threading.Event()
    release = threading.Event()
    original = cpp.NativeTensorCore._dropout_forward_with_mask

    def failing(self, p, *, seed, call_index):
        entered.set()
        assert release.wait(JOIN_TIMEOUT)
        raise RuntimeError("injected Core failure")

    monkeypatch.setattr(
        cpp.NativeTensorCore, "_dropout_forward_with_mask", failing
    )
    values = np.arange(1.0, 13.0)
    a = NativeTensor.from_array(values, requires_grad=True)
    b = NativeTensor.from_array(values, requires_grad=True)
    generator = NativeGenerator(139)
    errors = {}

    def first():
        try:
            a.dropout(0.5, generator=generator)
        except RuntimeError as error:
            errors["first"] = error

    thread = threading.Thread(target=first, daemon=True)
    thread.start()
    assert entered.wait(JOIN_TIMEOUT)
    with pytest.raises(RuntimeError, match="outstanding call reservation"):
        b.dropout(0.5, generator=generator)
    release.set()
    thread.join(JOIN_TIMEOUT)
    assert not thread.is_alive()
    assert isinstance(errors.get("first"), RuntimeError)
    assert generator.calls == 0
    assert generator._has_active_reservation() is False

    monkeypatch.setattr(
        cpp.NativeTensorCore, "_dropout_forward_with_mask", original
    )
    # The failed operation's index is reused, not skipped.
    y = b.dropout(0.5, generator=generator)
    expected, _ = core_reference(values, 0.5, 139, 0)
    assert np.array_equal(y.to_numpy(), expected)
    assert generator.calls == 1
    for t in (y, a, b):
        t.close()


def test_unrelated_generators_run_independently_and_concurrently(monkeypatch):
    entered = threading.Event()
    release = threading.Event()
    monkeypatch.setattr(
        cpp.NativeTensorCore, "_dropout_forward_with_mask",
        _blocking_core(entered, release),
    )
    values = np.arange(1.0, 13.0)
    a = NativeTensor.from_array(values, requires_grad=True)
    b = NativeTensor.from_array(values, requires_grad=True)
    busy, idle = NativeGenerator(149), NativeGenerator(151)
    results = {}

    def parked():
        results["a"] = a.dropout(0.5, generator=busy)

    thread = threading.Thread(target=parked, daemon=True)
    thread.start()
    assert entered.wait(JOIN_TIMEOUT)
    # A different generator is not blocked by the first one's reservation
    # — but this Core is gated, so release first and then use it.
    assert idle._has_active_reservation() is False
    assert idle.calls == 0
    release.set()
    thread.join(JOIN_TIMEOUT)
    assert not thread.is_alive()
    monkeypatch.undo()

    results["b"] = b.dropout(0.5, generator=idle)
    assert busy.calls == 1 and idle.calls == 1
    expected, _ = core_reference(values, 0.5, 151, 0)
    assert np.array_equal(results["b"].to_numpy(), expected)
    for t in (results["a"], results["b"], a, b):
        t.close()


def test_a_reentrant_dropout_from_inside_the_core_is_refused(monkeypatch):
    """The same-thread case: a callback re-entering the operation on the
    same generator fails deterministically instead of duplicating an
    index."""
    values = np.arange(1.0, 13.0)
    x = NativeTensor.from_array(values, requires_grad=True)
    other = NativeTensor.from_array(values, requires_grad=True)
    generator = NativeGenerator(157)
    original = cpp.NativeTensorCore._dropout_forward_with_mask
    seen = {}

    def reentrant(self, p, *, seed, call_index):
        if "tried" not in seen:
            seen["tried"] = True
            try:
                other.dropout(0.5, generator=generator)
            except RuntimeError as error:
                seen["error"] = error
        return original(self, p, seed=seed, call_index=call_index)

    monkeypatch.setattr(
        cpp.NativeTensorCore, "_dropout_forward_with_mask", reentrant
    )
    y = x.dropout(0.5, generator=generator)
    monkeypatch.undo()
    assert isinstance(seen.get("error"), RuntimeError)
    assert "outstanding call reservation" in str(seen["error"])
    assert generator.calls == 1, "the reentrant attempt consumed nothing"
    expected, _ = core_reference(values, 0.5, 157, 0)
    assert np.array_equal(y.to_numpy(), expected)
    for t in (y, x, other):
        t.close()


def test_many_threads_on_one_generator_never_share_a_call_index():
    """A stress loop with no sleeps: whatever mixture of successes and
    deterministic refusals occurs, ``calls`` equals the number of
    successes and every successful mask is the one its index pins."""
    threads_count = 8
    values = np.arange(1.0, 13.0)
    generator = NativeGenerator(163)
    start = threading.Barrier(threads_count)
    lock = threading.Lock()
    successes = []
    refusals = []

    def worker():
        tensor = NativeTensor.from_array(values, requires_grad=True)
        try:
            start.wait(JOIN_TIMEOUT)
            try:
                result = tensor.dropout(0.5, generator=generator)
            except RuntimeError as error:
                with lock:
                    refusals.append(str(error))
                return
            try:
                with lock:
                    successes.append(result.to_numpy().copy())
            finally:
                result.close()
        finally:
            tensor.close()

    threads = [threading.Thread(target=worker, daemon=True)
               for _ in range(threads_count)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(JOIN_TIMEOUT)
        assert not thread.is_alive()

    assert len(successes) + len(refusals) == threads_count
    assert generator.calls == len(successes)
    assert all("reservation" in message for message in refusals), refusals
    # Every success is one of the first `len(successes)` indices, each used
    # exactly once — no index was handed out twice.
    expected = [core_reference(values, 0.5, 163, index)[0]
                for index in range(len(successes))]
    matched = set()
    for produced in successes:
        found = [i for i, candidate in enumerate(expected)
                 if np.array_equal(produced, candidate)]
        assert len(found) == 1, "a result matched no single call index"
        assert found[0] not in matched, "two forwards shared a call index"
        matched.add(found[0])
    assert len(matched) == len(successes)


# ==========================================================================
# 10. Failure injection: every position between reservation and commit
# ==========================================================================

def _assert_clean(generator, live_storages, baseline, x, values):
    """The five things every failed forward must be true of."""
    assert generator.calls == 0, "a failed forward consumed a call"
    assert generator._has_active_reservation() is False, (
        "a reservation or construction claim was stranded"
    )
    assert len(live_storages) == baseline, "native storage leaked"
    assert x.closed is False and np.array_equal(x.to_numpy(), values)
    assert x.grad is None


def _next_call_succeeds(x, generator, values, seed):
    """...and the very next forward reuses the unconsumed index 0."""
    y = x.dropout(0.5, generator=generator)
    expected, _ = core_reference(values, 0.5, seed, 0)
    assert np.array_equal(y.to_numpy(), expected)
    assert generator.calls == 1
    y.close()


def test_failure_immediately_after_the_reservation(monkeypatch, live_storages):
    values = np.arange(1.0, 13.0)
    x = NativeTensor.from_array(values, requires_grad=True)
    generator = NativeGenerator(167)
    baseline = len(live_storages)

    def exploding_seed(self):
        raise RuntimeError("injected post-reservation failure")

    monkeypatch.setattr(NativeGenerator, "seed", property(exploding_seed))
    with pytest.raises(RuntimeError, match="post-reservation"):
        x.dropout(0.5, generator=generator)
    monkeypatch.undo()
    _assert_clean(generator, live_storages, baseline, x, values)
    _next_call_succeeds(x, generator, values, 167)
    x.close()


def test_failure_before_the_core_invocation(monkeypatch, live_storages):
    values = np.arange(1.0, 13.0)
    x = NativeTensor.from_array(values, requires_grad=True)
    generator = NativeGenerator(173)
    baseline = len(live_storages)

    def refusing(self, p, *, seed, call_index):
        raise ValueError("injected Core validation failure")

    monkeypatch.setattr(
        cpp.NativeTensorCore, "_dropout_forward_with_mask", refusing
    )
    with pytest.raises(ValueError, match="Core validation"):
        x.dropout(0.5, generator=generator)
    monkeypatch.undo()
    _assert_clean(generator, live_storages, baseline, x, values)
    _next_call_succeeds(x, generator, values, 173)
    x.close()


@needs_fault_injection
@pytest.mark.parametrize("nth, what", [(1, "output"), (2, "mask")])
def test_native_allocation_failure(nth, what, live_storages):
    values = np.arange(1.0, 13.0)
    x = NativeTensor.from_array(values, requires_grad=True)
    generator = NativeGenerator(179)
    baseline = len(live_storages)
    with pytest.raises(MemoryError):
        cpp._arm_alloc_failure(nth)
        x.dropout(0.5, generator=generator)
    cpp._arm_alloc_failure(0)
    _assert_clean(generator, live_storages, baseline, x, values)
    _next_call_succeeds(x, generator, values, 179)
    x.close()


@needs_fault_injection
def test_repeated_allocation_failures_do_not_accumulate(live_storages):
    values = np.arange(1.0, 13.0)
    x = NativeTensor.from_array(values, requires_grad=True)
    generator = NativeGenerator(181)
    baseline = len(live_storages)
    for nth in (1, 2) * 5:
        with pytest.raises(MemoryError):
            cpp._arm_alloc_failure(nth)
            x.dropout(0.5, generator=generator)
        cpp._arm_alloc_failure(0)
        assert len(live_storages) == baseline
        assert generator.calls == 0
        assert generator._has_active_reservation() is False
    _next_call_succeeds(x, generator, values, 181)
    x.close()


def test_failure_after_both_core_results_exist(monkeypatch, live_storages):
    """The one ownership path where nothing has adopted either object:
    ``_from_op`` never ran, so cleanup must be explicit rather than left
    to ``__del__``."""
    values = np.arange(1.0, 13.0)
    x = NativeTensor.from_array(values, requires_grad=True)
    generator = NativeGenerator(191)
    baseline = len(live_storages)

    produced = []
    original = cpp.NativeTensorCore._dropout_forward_with_mask

    def capturing(self, p, *, seed, call_index):
        out, mask = original(self, p, seed=seed, call_index=call_index)
        produced.append((out, mask))
        return out, mask

    monkeypatch.setattr(
        cpp.NativeTensorCore, "_dropout_forward_with_mask", capturing
    )

    def exploding_from_op(cls, *args, **kwargs):
        raise RuntimeError("injected graph-node construction failure")

    monkeypatch.setattr(
        NativeTensor, "_from_op", classmethod(exploding_from_op)
    )
    with pytest.raises(RuntimeError, match="graph-node construction"):
        x.dropout(0.5, generator=generator)
    monkeypatch.undo()

    assert len(produced) == 1
    out_core, mask = produced[0]
    assert out_core._closed is True, "the output leaked on a failed graph"
    assert mask._closed is True, "the mask leaked on a failed graph"
    _assert_clean(generator, live_storages, baseline, x, values)
    # Closed exactly once each: a second close is a no-op.
    out_core.close()
    mask.close()
    assert len(live_storages) == baseline
    _next_call_succeeds(x, generator, values, 191)
    x.close()


def test_failure_while_wrapping_the_output(monkeypatch, live_storages):
    values = np.arange(1.0, 13.0)
    x = NativeTensor.from_array(values, requires_grad=True)
    generator = NativeGenerator(193)
    baseline = len(live_storages)

    def exploding_from_core(cls, core, owns_core=True):
        raise RuntimeError("injected wrapper construction failure")

    monkeypatch.setattr(
        NativeTensor, "_from_core", classmethod(exploding_from_core)
    )
    with pytest.raises(RuntimeError, match="wrapper construction"):
        x.dropout(0.5, generator=generator)
    monkeypatch.undo()
    _assert_clean(generator, live_storages, baseline, x, values)
    _next_call_succeeds(x, generator, values, 193)
    x.close()


def test_failure_while_constructing_the_backward_closure(monkeypatch,
                                                         live_storages):
    values = np.arange(1.0, 13.0)
    x = NativeTensor.from_array(values, requires_grad=True)
    generator = NativeGenerator(197)
    baseline = len(live_storages)

    def exploding(input_tensor, mask):
        raise RuntimeError("injected backward-closure failure")

    monkeypatch.setattr(native_tensor_module, "_dropout_backward", exploding)
    with pytest.raises(RuntimeError, match="backward-closure"):
        x.dropout(0.5, generator=generator)
    monkeypatch.undo()
    _assert_clean(generator, live_storages, baseline, x, values)
    _next_call_succeeds(x, generator, values, 197)
    x.close()


def test_failure_while_attaching_the_graph_resources(monkeypatch,
                                                     live_storages):
    """Injected at the real assignment site: ``_from_op`` adopts the saved
    state by materializing ``tuple(graph_resources)`` *after* the node
    exists and its edges are wired, so an iteration failure lands exactly
    where attachment does."""
    values = np.arange(1.0, 13.0)
    x = NativeTensor.from_array(values, requires_grad=True)
    generator = NativeGenerator(199)
    baseline = len(live_storages)

    class Exploding:
        def __iter__(self):
            raise RuntimeError("injected graph-resource attachment failure")

    original = NativeTensor._from_op.__func__

    def wrapper(cls, core, parents, backward, op, owns_core=True,
                expected_versions=(), graph_resources=()):
        return original(cls, core, parents, backward, op, owns_core,
                        expected_versions, Exploding())

    monkeypatch.setattr(NativeTensor, "_from_op", classmethod(wrapper))
    with pytest.raises(RuntimeError, match="graph-resource attachment"):
        x.dropout(0.5, generator=generator)
    monkeypatch.undo()
    gc.collect()          # the orphaned node, if any, holds nothing extra
    _assert_clean(generator, live_storages, baseline, x, values)
    _next_call_succeeds(x, generator, values, 199)
    x.close()


def test_failure_during_the_no_grad_mask_cleanup(monkeypatch, live_storages):
    """``_from_op`` closes the mask itself when nothing requires grad. If
    that close fails, the operation still cancels and still releases
    everything it can."""
    values = np.arange(1.0, 13.0)
    x = NativeTensor.from_array(values)          # no grad: the mask is closed
    generator = NativeGenerator(211)
    baseline = len(live_storages)

    original_close = cpp.NativeTensorCore.close
    state = {"armed": True}

    def failing_close(self):
        if state["armed"]:
            state["armed"] = False
            raise RuntimeError("injected mask-cleanup failure")
        original_close(self)

    monkeypatch.setattr(cpp.NativeTensorCore, "close", failing_close)
    with pytest.raises(RuntimeError, match="mask-cleanup"):
        x.dropout(0.5, generator=generator)
    monkeypatch.undo()
    gc.collect()
    assert generator.calls == 0
    assert generator._has_active_reservation() is False
    assert len(live_storages) == baseline
    _next_call_succeeds(x, generator, values, 211)
    x.close()


def test_failure_delivering_the_result_before_the_commit(monkeypatch,
                                                         live_storages):
    """The last window before the transaction boundary: the output exists
    and the graph owns the mask, and the call must still not be consumed."""
    values = np.arange(1.0, 13.0)
    x = NativeTensor.from_array(values, requires_grad=True)
    generator = NativeGenerator(223)
    baseline = len(live_storages)

    delivered = []

    def boom(result):
        delivered.append(result)
        raise RuntimeError("injected result-delivery failure")

    monkeypatch.setattr(
        native_tensor_module, "_deliver_dropout_result", boom
    )
    with pytest.raises(RuntimeError, match="result-delivery"):
        x.dropout(0.5, generator=generator)
    monkeypatch.undo()

    # The result really was fully built, and was then released whole.
    assert len(delivered) == 1
    result = delivered[0]
    assert result.closed is True
    assert result._graph_resources == ()
    _assert_clean(generator, live_storages, baseline, x, values)
    _next_call_succeeds(x, generator, values, 223)
    x.close()


@pytest.mark.parametrize(
    "exception", [KeyboardInterrupt, MemoryError, RuntimeError],
)
def test_base_exceptions_before_the_commit_are_handled_identically(
    exception, monkeypatch, live_storages,
):
    values = np.arange(1.0, 13.0)
    x = NativeTensor.from_array(values, requires_grad=True)
    generator = NativeGenerator(227)
    baseline = len(live_storages)

    def boom(result):
        raise exception("injected")

    monkeypatch.setattr(
        native_tensor_module, "_deliver_dropout_result", boom
    )
    with pytest.raises(exception):
        x.dropout(0.5, generator=generator)
    monkeypatch.undo()
    _assert_clean(generator, live_storages, baseline, x, values)
    _next_call_succeeds(x, generator, values, 227)
    x.close()


def test_repeated_failures_never_strand_the_generator(monkeypatch,
                                                      live_storages):
    values = np.arange(1.0, 13.0)
    x = NativeTensor.from_array(values, requires_grad=True)
    generator = NativeGenerator(229)
    baseline = len(live_storages)

    def boom(result):
        raise RuntimeError("injected")

    monkeypatch.setattr(
        native_tensor_module, "_deliver_dropout_result", boom
    )
    for _ in range(20):
        with pytest.raises(RuntimeError):
            x.dropout(0.5, generator=generator)
        assert generator.calls == 0
        assert generator._has_active_reservation() is False
        assert len(live_storages) == baseline
    monkeypatch.undo()
    _next_call_succeeds(x, generator, values, 229)
    x.close()


# ==========================================================================
# 10b. The commit boundary itself
#
# `_commit_call` is the transaction boundary, and the two sides of it need
# *different* cleanup. Before it, no call is consumed and the reservation
# must be abandoned. After it, the index is irreversibly spent, the
# reservation slot is already clear, and abandoning the committed token
# would raise "already committed" — masking the failure the caller
# actually needs to see.
#
# Both sides are injected deterministically by wrapping `_commit_call`,
# never by real signal timing: the pre-commit wrapper raises *instead of*
# committing, and the post-commit wrapper calls the **real** commit first
# and then raises. The post-commit wrapper is what makes the "do not rely
# on a local boolean" rule testable — the commit genuinely succeeds and
# the statement after it genuinely never runs.
# ==========================================================================

class InjectedAsyncFailure(BaseException):
    """A ``BaseException`` that is deliberately **not** an ``Exception``.

    An asynchronous interruption in the commit-to-return window is a
    `BaseException`, so a cleanup path written as ``except Exception``
    would silently skip it. Using a type that no ``except Exception``
    can catch makes that mistake fail this suite rather than pass it."""


def _context_chain(error):
    """``error`` and every exception chained behind it, cycle-safe."""
    chain = []
    seen = set()
    current = error
    while current is not None and id(current) not in seen:
        chain.append(current)
        seen.add(id(current))
        current = current.__context__
    return chain


def _assert_no_reservation_or_claim(generator):
    """Neither a published reservation nor a construction claim survives.

    Checked against the two private slots directly rather than only
    through ``_has_active_reservation()``, because the requirement
    distinguishes them and a single boolean cannot."""
    empty = native_generator_module._NO_RESERVATION
    assert generator._active_serial == empty, "a reservation was stranded"
    assert generator._claim_serial == empty, "a construction claim was stranded"
    assert generator._has_active_reservation() is False


def _commit_then_raise(exception, message):
    """A ``_commit_call`` wrapper that really commits, then raises."""
    real_commit = NativeGenerator._commit_call

    def wrapper(self, token):
        real_commit(self, token)          # the call is now consumed
        raise exception(message)

    return wrapper


@pytest.mark.parametrize(
    "exception",
    [KeyboardInterrupt, MemoryError, InjectedAsyncFailure],
    ids=["KeyboardInterrupt", "MemoryError", "custom-BaseException"],
)
def test_a_failure_after_a_successful_commit_keeps_the_call_consumed(
    exception, monkeypatch, live_storages,
):
    """The commit-to-return window. The call is **irreversibly** consumed,
    so the cleanup must not pretend otherwise — but the result never
    reached the caller, so it must still be released whole."""
    values = np.arange(1.0, 13.0)
    x = NativeTensor.from_array(values, requires_grad=True)
    generator = NativeGenerator(239)
    baseline = len(live_storages)

    # Capture the fully built result through the existing delivery seam,
    # so its release can be inspected directly rather than inferred.
    delivered = []
    original_deliver = native_tensor_module._deliver_dropout_result

    def capturing_deliver(result):
        delivered.append(result)
        return original_deliver(result)

    monkeypatch.setattr(
        native_tensor_module, "_deliver_dropout_result", capturing_deliver
    )
    # Tripwire: a confirmed commit must never be followed by an abandon.
    abandoned = []
    original_abandon = NativeGenerator._abandon_call

    def recording_abandon(self, token):
        abandoned.append(token)
        return original_abandon(self, token)

    monkeypatch.setattr(NativeGenerator, "_abandon_call", recording_abandon)
    monkeypatch.setattr(
        NativeGenerator, "_commit_call",
        _commit_then_raise(exception, "injected post-commit failure"),
    )

    with pytest.raises(exception) as info:
        x.dropout(0.5, generator=generator)
    monkeypatch.undo()

    # The exact injected exception propagates — not an "already
    # committed" or stale-token cleanup error standing in for it.
    assert type(info.value) is exception
    assert "injected post-commit failure" in str(info.value)
    for chained in _context_chain(info.value):
        text = str(chained).lower()
        assert "already" not in text, chained
        assert "stale" not in text, chained
        assert "no outstanding reservation" not in text, chained

    # The call is consumed exactly once, and nothing is left in flight.
    assert generator.calls == 1
    _assert_no_reservation_or_claim(generator)
    # ...and the committed token was never handed to `_abandon_call`.
    assert abandoned == [], "a committed reservation was abandoned"

    # The result was fully built, never returned, and released whole.
    assert len(delivered) == 1
    result = delivered[0]
    assert result.closed is True
    assert result._graph_resources == (), "the graph-owned mask survived"
    # No result escaped, so backward on it is impossible.
    with pytest.raises(RuntimeError):
        result.backward(gradient=NativeTensor.from_array(np.ones(12)))

    # Native storage is back to baseline and the input is untouched.
    assert len(live_storages) == baseline
    assert x.closed is False and np.array_equal(x.to_numpy(), values)
    assert x.grad is None

    # The next forward takes the **next** index, not the spent one.
    y = x.dropout(0.5, generator=generator)
    expected_next, _ = core_reference(values, 0.5, 239, 1)
    assert np.array_equal(y.to_numpy(), expected_next)
    spent, _ = core_reference(values, 0.5, 239, 0)
    assert not np.array_equal(y.to_numpy(), spent), (
        "the consumed call index was handed out a second time"
    )
    assert generator.calls == 2
    y.close()
    x.close()


@pytest.mark.parametrize(
    "exception",
    [KeyboardInterrupt, MemoryError, InjectedAsyncFailure],
    ids=["KeyboardInterrupt", "MemoryError", "custom-BaseException"],
)
def test_a_commit_that_fails_before_committing_abandons_the_reservation(
    exception, monkeypatch, live_storages,
):
    """The paired case, one statement earlier: `_commit_call` raises
    *instead of* committing. Nothing is consumed, so the reservation is
    abandoned and the same index stays retryable.

    This is the case a local "committed" flag set before the commit gets
    wrong, and the case a flag set *after* it gets right — which is
    exactly why the outcome comes from the token instead."""
    values = np.arange(1.0, 13.0)
    x = NativeTensor.from_array(values, requires_grad=True)
    generator = NativeGenerator(241)
    baseline = len(live_storages)

    def refusing_commit(self, token):
        raise exception("injected pre-commit failure")

    monkeypatch.setattr(NativeGenerator, "_commit_call", refusing_commit)
    with pytest.raises(exception) as info:
        x.dropout(0.5, generator=generator)
    monkeypatch.undo()

    assert "injected pre-commit failure" in str(info.value)
    assert generator.calls == 0, "a failed commit consumed a call"
    _assert_no_reservation_or_claim(generator)
    assert len(live_storages) == baseline
    assert x.closed is False and np.array_equal(x.to_numpy(), values)
    assert x.grad is None

    # The same index is retryable and reproduces what the failed forward
    # would have produced.
    y = x.dropout(0.5, generator=generator)
    expected, _ = core_reference(values, 0.5, 241, 0)
    assert np.array_equal(y.to_numpy(), expected)
    assert generator.calls == 1
    y.close()
    x.close()


def test_repeated_post_commit_failures_advance_once_each_and_leak_nothing(
    monkeypatch, live_storages,
):
    """Five consecutive interruptions in the commit-to-return window: each
    spends exactly one index, none is reused, and storage never grows."""
    values = np.arange(1.0, 13.0)
    x = NativeTensor.from_array(values, requires_grad=True)
    generator = NativeGenerator(251)
    baseline = len(live_storages)

    monkeypatch.setattr(
        NativeGenerator, "_commit_call",
        _commit_then_raise(KeyboardInterrupt, "injected post-commit"),
    )
    for expected_calls in range(1, 6):
        with pytest.raises(KeyboardInterrupt):
            x.dropout(0.5, generator=generator)
        assert generator.calls == expected_calls
        _assert_no_reservation_or_claim(generator)
        assert len(live_storages) == baseline
    monkeypatch.undo()

    # Five indices were spent, so the next forward is index 5.
    y = x.dropout(0.5, generator=generator)
    expected, _ = core_reference(values, 0.5, 251, 5)
    assert np.array_equal(y.to_numpy(), expected)
    assert generator.calls == 6
    y.close()
    x.close()


def test_the_committed_outcome_query_is_private_read_only_and_exact():
    """The narrow generator query the cleanup depends on, checked on its
    own terms: it is private, it changes nothing, it answers only about
    *this* generator's token, and a non-token is a caller bug."""
    generator = NativeGenerator(257)
    other = NativeGenerator(257)

    # Private by name, and with no public spelling beside it: every
    # generator attribute that is not dunder is either one of the four
    # documented state fields or an underscore-private.
    assert not hasattr(NativeGenerator, "call_committed")
    for public in ("committed", "outcome", "token", "reservation"):
        assert not hasattr(generator, public), public
    documented = {"algorithm", "algorithm_version", "seed", "calls",
                  "state", "load_state", "reseed", "reset"}
    leaked = [name for name in dir(NativeGenerator)
              if not name.startswith("_") and name not in documented]
    assert leaked == [], leaked

    token = generator._reserve_call()
    before = generator.state()
    assert generator._call_committed(token) is False
    # Inspecting changed nothing: not the state, not the live reservation.
    assert generator.state() == before
    assert generator._has_active_reservation() is True

    generator._commit_call(token)
    assert generator._call_committed(token) is True
    assert generator.calls == 1
    _assert_no_reservation_or_claim(generator)
    # ...and asking a *different* generator about it is False, not a
    # raise: the question is "did I commit this", not "is this valid".
    assert other._call_committed(token) is False
    assert other.calls == 0

    # An abandoned token is not a committed one.
    second = generator._reserve_call()
    generator._abandon_call(second)
    assert generator._call_committed(second) is False
    assert generator.calls == 1

    # A non-token is a caller bug.
    for not_a_token in (None, 0, "token", object()):
        with pytest.raises(TypeError):
            generator._call_committed(not_a_token)


def test_a_failing_cancellation_does_not_replace_the_original_failure(
    monkeypatch, live_storages,
):
    """The cleanup-failure convention: `_abandon_call` is non-failing for
    a live matching token, so if it *does* fail something is already
    wrong — and the caller still needs the operation's error, not the
    cleanup's. The original stays primary and the cleanup failure is
    chained onto it rather than swallowed."""
    values = np.arange(1.0, 13.0)
    x = NativeTensor.from_array(values, requires_grad=True)
    generator = NativeGenerator(263)
    baseline = len(live_storages)

    def boom(result):
        raise RuntimeError("injected result-delivery failure")

    def failing_abandon(self, token):
        raise ValueError("injected cancellation failure")

    monkeypatch.setattr(
        native_tensor_module, "_deliver_dropout_result", boom
    )
    monkeypatch.setattr(NativeGenerator, "_abandon_call", failing_abandon)
    with pytest.raises(RuntimeError) as info:
        x.dropout(0.5, generator=generator)
    monkeypatch.undo()

    # The operation's failure is what propagates...
    assert "injected result-delivery failure" in str(info.value)
    # ...and the cleanup failure is still reachable, not swallowed.
    chained = [error for error in _context_chain(info.value)
               if "injected cancellation failure" in str(error)]
    assert len(chained) == 1, "the cleanup failure was lost"

    # Native storage came back regardless: the result is released before
    # the reservation is settled, precisely so this ordering holds.
    assert len(live_storages) == baseline
    assert generator.calls == 0
    assert x.closed is False and np.array_equal(x.to_numpy(), values)
    x.close()


# ==========================================================================
# 11. The capability boundary G3 does not move
# ==========================================================================

def test_the_operation_exists_without_a_module_or_a_new_kernel():
    import tensorforge.experimental as experimental

    assert "dropout" in cpp.AUTOGRAD_OPS
    assert hasattr(NativeTensor, "dropout")
    # No module, no functional helper, no global stream.
    assert not hasattr(experimental, "NativeDropout")
    assert "NativeDropout" not in cpp.NATIVE_MODULES
    assert not hasattr(experimental, "dropout")
    for name in ("default_generator", "manual_seed", "seed_all",
                 "global_generator"):
        assert not hasattr(experimental, name), name
    # No backward kernel: the gradient is the existing multiply.
    assert "dropout_backward" not in cpp.TENSOR_CORE_OPS
    assert "dropout_backward" not in cpp.AUTOGRAD_OPS
    assert not hasattr(cpp.NativeTensorCore, "dropout_backward")
    assert "tf_core_dropout_backward" not in cpp._CHECKED_KERNELS
    dropout_symbols = [name for name in cpp._CHECKED_KERNELS
                       if "dropout" in name or "random" in name]
    assert dropout_symbols == ["tf_core_dropout_forward"]


def test_dropout_stays_unsupported_and_the_checkpoint_stays_version_one():
    from tensorforge.experimental import native_checkpoint

    assert cpp.UNSUPPORTED == ("dropout", "float32", "cuda", "amp")
    assert cpp.SUPPORTED_DTYPES == ("float64",)
    assert cpp.SUPPORTED_DEVICES == ("cpu",)
    assert native_checkpoint._FORMAT_VERSION == 1


def test_generator_state_is_not_in_state_dict_or_a_checkpoint(tmp_path):
    from tensorforge.experimental import (
        NativeLinear, NativeModule, NativeSGD, save_native_checkpoint,
    )
    import json

    class Model(NativeModule):
        def __init__(self):
            super().__init__()
            self.linear = NativeLinear(4, 2)
            self.g = NativeGenerator(5)

        def forward(self, x):
            return self.linear(x).dropout(0.5, generator=self.g)

    model = Model()
    x = NativeTensor.from_array(np.arange(1.0, 13.0).reshape(3, 4))
    y = model(x)
    assert model.g.calls == 1

    # state_dict() is still tensor-only.
    for value in model.state_dict().values():
        assert isinstance(value, NativeTensor)
    assert set(model.generator_state_dict()) == {"g"}

    optimizer = NativeSGD(model.parameters(), lr=0.1)
    path = tmp_path / "ckpt.npz"
    save_native_checkpoint(path, model, optimizer)
    with np.load(path, allow_pickle=False) as archive:
        manifest = json.loads(archive["manifest"].tobytes().decode("utf-8"))
    assert manifest["format_version"] == 1
    assert "generators" not in manifest, (
        "generator state must not be checkpointed until G5"
    )
    for t in (y, x):
        t.close()
    for _, parameter in model.named_parameters():
        parameter.close()
