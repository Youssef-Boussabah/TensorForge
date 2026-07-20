"""Differentiable native MaxPool2d — NativeTensor.maxpool2d autograd
integration (Phase D, milestone D9).

D9 completes the pooling operation: ``NativeTensor.maxpool2d(*,
kernel_size, stride=None, padding=0)`` builds one Python-managed graph
node whose single backward callback scatters the upstream gradient through
the **private winner buffer its own forward saved** — never rereading the
input value and never recomputing a window maximum. These tests cover
forward/gradient parity with the stable framework, the routing rules
(ties, padding sentinels, -inf, overlapping windows), argument handling,
explicit-gradient validation, graph lifetime and the saved-winner
ownership contract (retain/free/failure/abandon), the deliberate absence
of version tracking, failure rollback, and the capability split between
this operation and the ``NativeMaxPool2d`` module built on it (D10).

Backend-dependent, so the module skips cleanly when the compiled backend
is not built. Cleanup is explicit via close().

Selector: python -m pytest -q -k native_maxpool2d_autograd
"""

import gc

import numpy as np
import pytest

from tensorforge.backends import cpp
from tensorforge.experimental import NativeParameter, NativeTensor

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
# Helpers
# --------------------------------------------------------------------------

def _stable(x, g, kernel_size, stride=None, padding=0):
    """Stable ``tensorforge.nn.MaxPool2d`` forward + input gradient for a
    fixed upstream ``g`` (routed through a scalar loss, since the stable
    engine seeds ones)."""
    from tensorforge.nn import MaxPool2d
    from tensorforge.tensor import Tensor

    xt = Tensor(np.array(x, float), requires_grad=True)
    out = MaxPool2d(kernel_size, stride=stride, padding=padding)(xt)
    (out * Tensor(np.array(g, float))).sum().backward()
    return out.data, xt.grad


def _saved_winners(output):
    """The private winner cores this graph node owns (white-box: the
    lifetime contract is exactly what these tests must pin down)."""
    return output._graph_resources


def _winners_open(output):
    resources = _saved_winners(output)
    assert resources, "expected the node to own a saved winner buffer"
    return all(not core._closed for core in resources)


# --------------------------------------------------------------------------
# Forward and gradient parity with the stable framework
# --------------------------------------------------------------------------

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
def test_forward_and_gradient_parity(kernel_size, stride, padding):
    x = np.round(np.random.default_rng(1).standard_normal((2, 3, 6, 5)) * 3, 3)
    xi = NativeParameter(x)
    y = xi.maxpool2d(kernel_size=kernel_size, stride=stride, padding=padding)
    g = np.random.default_rng(2).standard_normal(y.shape)
    out, grad = _stable(x, g, kernel_size, stride=stride, padding=padding)
    # Pooling selects values verbatim, so the forward is exact.
    assert np.array_equal(y.to_numpy(), out)
    y.backward(gradient=NativeTensor.from_array(g))
    assert np.allclose(xi.grad.to_numpy(), grad, atol=1e-12)
    y.close()
    xi.close()


def test_hand_computed_gradient_routing():
    x = np.arange(1, 17, dtype=float).reshape(1, 1, 4, 4)
    xi = NativeParameter(x)
    y = xi.maxpool2d(kernel_size=2)
    y.backward(gradient=NativeTensor.from_array(
        np.array([[[[1.0, 2.0], [3.0, 4.0]]]])
    ))
    assert xi.grad.to_numpy().tolist() == [[[
        [0, 0, 0, 0],
        [0, 1, 0, 2],
        [0, 0, 0, 0],
        [0, 3, 0, 4],
    ]]]
    y.close()
    xi.close()


def test_tie_routes_the_whole_window_to_the_first_winner():
    # Two equal maxima: only the first in row-major order receives the
    # gradient, matching the stable argmax.
    x = np.array([[[[1.0, 5.0], [5.0, 2.0]]]])
    xi = NativeParameter(x)
    y = xi.maxpool2d(kernel_size=2)
    y.backward(gradient=NativeTensor.from_array(np.array([[[[7.0]]]])))
    assert xi.grad.to_numpy().tolist() == [[[[0.0, 7.0], [0.0, 0.0]]]]
    _, stable_grad = _stable(x, np.array([[[[7.0]]]]), 2)
    assert np.array_equal(xi.grad.to_numpy(), stable_grad)
    y.close()
    xi.close()


def test_padding_winner_drops_gradient_and_matches_stable():
    x = np.array([[[[1.0, 2.0], [3.0, 4.0]]]])
    xi = NativeParameter(x)
    y = xi.maxpool2d(kernel_size=2, stride=2, padding=1)
    g = np.array([[[[1.0, 2.0], [3.0, 4.0]]]])
    y.backward(gradient=NativeTensor.from_array(g))
    # Each padded window holds exactly one real cell, which wins.
    assert xi.grad.to_numpy().tolist() == [[[[1.0, 2.0], [3.0, 4.0]]]]
    _, stable_grad = _stable(x, g, 2, stride=2, padding=1)
    assert np.array_equal(xi.grad.to_numpy(), stable_grad)
    y.close()
    xi.close()


def test_completely_padded_windows_drop_their_gradient():
    # 1x1 input, 3x3 output: eight windows are entirely padding (winner
    # -1) and contribute nothing; only the centre routes gradient.
    xi = NativeParameter(np.array([[[[4.0]]]]))
    y = xi.maxpool2d(kernel_size=1, stride=1, padding=1)
    assert y.shape == (1, 1, 3, 3)
    assert np.isneginf(y.to_numpy()[0, 0, 0, 0])
    y.backward(gradient=NativeTensor.from_array(np.ones((1, 1, 3, 3))))
    assert xi.grad.to_numpy().tolist() == [[[[1.0]]]]
    y.close()
    xi.close()


def test_all_negative_infinity_routes_to_the_first_real_cell():
    xi = NativeParameter(np.full((1, 1, 2, 2), NEG_INF))
    y = xi.maxpool2d(kernel_size=2)
    y.backward(gradient=NativeTensor.from_array(np.array([[[[3.0]]]])))
    # Without padding the first row-major real cell wins the -inf tie.
    assert xi.grad.to_numpy().tolist() == [[[[3.0, 0.0], [0.0, 0.0]]]]
    y.close()
    xi.close()


def test_overlapping_windows_accumulate():
    # Every 2x2 window of this 3x3 input selects the centre maximum.
    x = np.array([[[[1.0, 2.0, 3.0], [4.0, 9.0, 5.0], [6.0, 7.0, 8.0]]]])
    xi = NativeParameter(x)
    y = xi.maxpool2d(kernel_size=2, stride=1)
    y.backward(gradient=NativeTensor.from_array(np.ones((1, 1, 2, 2))))
    assert xi.grad.to_numpy().tolist() == [[[[0, 0, 0], [0, 4.0, 0], [0, 0, 0]]]]
    y.close()
    xi.close()


def test_integer_and_tuple_arguments_agree():
    x = np.random.default_rng(3).standard_normal((1, 2, 5, 5))
    xi = NativeParameter(x)
    a = xi.maxpool2d(kernel_size=2, stride=2, padding=1)
    b = xi.maxpool2d(kernel_size=(2, 2), stride=(2, 2), padding=(1, 1))
    assert np.array_equal(a.to_numpy(), b.to_numpy())
    for t in (a, b, xi):
        t.close()


def test_stride_none_defaults_to_kernel_size():
    x = np.random.default_rng(4).standard_normal((1, 1, 4, 4))
    xi = NativeParameter(x)
    default = xi.maxpool2d(kernel_size=2)
    explicit = xi.maxpool2d(kernel_size=2, stride=2)
    assert np.array_equal(default.to_numpy(), explicit.to_numpy())
    overlapping = xi.maxpool2d(kernel_size=2, stride=1)
    assert overlapping.shape != default.shape
    for t in (default, explicit, overlapping, xi):
        t.close()


def test_non_contiguous_input_matches_its_contiguous_copy():
    values = np.random.default_rng(5).standard_normal((2, 2, 4, 5))
    owner = NativeParameter(np.ascontiguousarray(values.transpose(0, 1, 3, 2)))
    view = owner.transpose(0, 1, 3, 2)
    assert not view.contiguous
    y = view.maxpool2d(kernel_size=2)
    reference = NativeParameter(values).maxpool2d(kernel_size=2)
    assert np.allclose(y.to_numpy(), reference.to_numpy(), atol=1e-12)
    _, stable_grad = _stable(values, np.ones(y.shape), 2)
    y.backward(gradient=NativeTensor.from_array(np.ones(y.shape)))
    assert np.allclose(owner.grad.to_numpy().transpose(0, 1, 3, 2),
                       stable_grad, atol=1e-12)
    for t in (y, reference, view, owner):
        t.close()


def test_non_contiguous_explicit_upstream_gradient():
    x = np.random.default_rng(6).standard_normal((1, 2, 4, 4))
    xi = NativeParameter(x)
    y = xi.maxpool2d(kernel_size=2)
    g = np.random.default_rng(7).standard_normal(y.shape)
    strided_owner = NativeTensor.from_array(
        np.ascontiguousarray(g.transpose(0, 1, 3, 2))
    )
    strided = strided_owner.transpose(0, 1, 3, 2)
    assert not strided.contiguous
    y.backward(gradient=strided)
    _, stable_grad = _stable(x, g, 2)
    assert np.allclose(xi.grad.to_numpy(), stable_grad, atol=1e-12)
    for t in (y, strided, strided_owner, xi):
        t.close()


def test_scalar_loss_through_existing_reductions():
    x = np.random.default_rng(8).standard_normal((2, 2, 4, 4))
    xi = NativeParameter(x)
    loss = xi.maxpool2d(kernel_size=2).sum()
    assert loss.numel == 1
    loss.backward()  # no explicit gradient needed for a scalar
    # d(sum of maxima)/dx is 1 at each winner.
    _, stable_grad = _stable(x, np.ones((2, 2, 2, 2)), 2)
    assert np.allclose(xi.grad.to_numpy(), stable_grad, atol=1e-12)
    loss.close()
    xi.close()


def test_deterministic_repeated_forward_and_backward():
    x = np.random.default_rng(9).standard_normal((1, 2, 5, 5))
    g = np.random.default_rng(10).standard_normal((1, 2, 2, 2))
    grads = []
    for _ in range(2):
        xi = NativeParameter(x)
        y = xi.maxpool2d(kernel_size=2, stride=2, padding=0)
        y.backward(gradient=NativeTensor.from_array(g))
        grads.append(xi.grad.to_numpy())
        y.close()
        xi.close()
    assert np.array_equal(grads[0], grads[1])


# --------------------------------------------------------------------------
# Graph construction, ownership, and the no-grad path
# --------------------------------------------------------------------------

def test_graph_node_shape_parents_and_ownership():
    xi = NativeParameter(np.random.default_rng(11).standard_normal((2, 3, 4, 4)))
    y = xi.maxpool2d(kernel_size=2)
    assert y.shape == (2, 3, 2, 2)
    assert y.requires_grad is True
    assert y.is_leaf is False
    assert y._op == "maxpool2d"
    assert y._parents == (xi,)          # exactly one parent, the input
    assert y.owns_core is True
    assert y.contiguous
    y.close()
    xi.close()


def test_no_grad_input_builds_no_graph_and_closes_the_winners():
    xi = NativeTensor.from_array(np.arange(16, dtype=float).reshape(1, 1, 4, 4))
    assert xi.requires_grad is False
    y = xi.maxpool2d(kernel_size=2)
    assert y.requires_grad is False
    assert y.is_leaf is True
    assert y._parents == ()
    assert y._backward is None
    # No backward will ever run, so the private winner buffer was released
    # immediately rather than left for garbage collection.
    assert y._graph_resources == ()
    assert y.to_numpy().tolist() == [[[[5.0, 7.0], [13.0, 15.0]]]]
    with pytest.raises(RuntimeError):
        y.backward()
    y.close()
    xi.close()


def test_output_survives_the_input_being_closed():
    xi = NativeTensor.from_array(np.arange(16, dtype=float).reshape(1, 1, 4, 4))
    y = xi.maxpool2d(kernel_size=2)
    xi.close()
    assert y.to_numpy().tolist() == [[[[5.0, 7.0], [13.0, 15.0]]]]
    y.close()


def test_shared_input_across_two_pooling_branches_accumulates():
    x = np.random.default_rng(12).standard_normal((1, 1, 4, 4))
    xi = NativeParameter(x)
    a = xi.maxpool2d(kernel_size=2)          # non-overlapping
    b = xi.maxpool2d(kernel_size=2, stride=1)  # overlapping
    total = a.sum().add(b.sum())
    total.backward()
    _, grad_a = _stable(x, np.ones(a.shape), 2)
    _, grad_b = _stable(x, np.ones(b.shape), 2, stride=1)
    assert np.allclose(xi.grad.to_numpy(), grad_a + grad_b, atol=1e-12)
    for t in (total, a, b, xi):
        t.close()


def test_gradients_accumulate_and_zero_grad_clears():
    x = np.random.default_rng(13).standard_normal((1, 1, 4, 4))
    xi = NativeParameter(x)
    g = np.ones((1, 1, 2, 2))
    first = xi.maxpool2d(kernel_size=2)
    first.backward(gradient=NativeTensor.from_array(g))
    once = xi.grad.to_numpy().copy()
    second = xi.maxpool2d(kernel_size=2)
    second.backward(gradient=NativeTensor.from_array(g))
    assert np.allclose(xi.grad.to_numpy(), 2 * once, atol=1e-12)
    xi.zero_grad()
    assert xi.grad is None
    for t in (first, second, xi):
        t.close()


# --------------------------------------------------------------------------
# Saved-winner lifetime
# --------------------------------------------------------------------------

def test_winners_are_saved_at_forward_and_freed_by_a_one_shot_backward():
    xi = NativeParameter(np.random.default_rng(14).standard_normal((1, 1, 4, 4)))
    y = xi.maxpool2d(kernel_size=2)
    assert _winners_open(y)  # alive through forward return
    y.backward(gradient=NativeTensor.from_array(np.ones((1, 1, 2, 2))))
    # The default one-shot backward released the history, and the winner
    # buffer with it — deterministically, not via garbage collection.
    assert y._graph_freed is True
    assert y._graph_resources == ()
    y.close()
    xi.close()


def test_retain_graph_keeps_the_winners_for_another_backward():
    x = np.random.default_rng(15).standard_normal((1, 1, 4, 4))
    xi = NativeParameter(x)
    y = xi.maxpool2d(kernel_size=2)
    g = NativeTensor.from_array(np.ones((1, 1, 2, 2)))
    y.backward(gradient=g, retain_graph=True)
    assert _winners_open(y)      # still alive for the next pass
    assert y._graph_freed is False
    once = xi.grad.to_numpy().copy()
    y.backward(gradient=g)       # second pass, this one one-shot
    assert np.allclose(xi.grad.to_numpy(), 2 * once, atol=1e-12)
    assert y._graph_resources == ()  # released with the history
    for t in (y, g, xi):
        t.close()


def test_repeated_backward_after_free_raises_without_double_closing():
    xi = NativeParameter(np.random.default_rng(16).standard_normal((1, 1, 4, 4)))
    y = xi.maxpool2d(kernel_size=2)
    saved = _saved_winners(y)[0]
    g = NativeTensor.from_array(np.ones((1, 1, 2, 2)))
    y.backward(gradient=g)
    assert saved._closed is True
    with pytest.raises(RuntimeError, match="freed autograd graph"):
        y.backward(gradient=g)
    assert saved._closed is True  # still closed exactly once
    for t in (y, g, xi):
        t.close()


def test_closing_an_unused_graph_releases_the_winners():
    xi = NativeParameter(np.random.default_rng(17).standard_normal((1, 1, 4, 4)))
    y = xi.maxpool2d(kernel_size=2)
    saved = _saved_winners(y)[0]
    assert not saved._closed
    y.close()  # abandoned without ever running backward
    assert saved._closed is True
    y.close()  # idempotent, no double close
    assert saved._closed is True
    xi.close()


def test_dropping_an_unused_graph_does_not_leak_the_winners():
    # The __del__ refcount/GC *fallback* — not a deterministic release
    # point. The deterministic ones (a one-shot backward's history release
    # and an explicit close()) are covered by the two tests above; this one
    # only proves the safety net also frees the buffer.
    xi = NativeParameter(np.random.default_rng(18).standard_normal((1, 1, 4, 4)))
    holder = []

    def build():
        y = xi.maxpool2d(kernel_size=2)
        holder.append(_saved_winners(y)[0])
        # y goes out of scope here without close() or backward()

    build()
    gc.collect()
    assert holder[0]._closed is True
    xi.close()


def test_graph_construction_failure_releases_the_output_and_winners(
    monkeypatch, live_storages
):
    """Forward succeeds, graph construction then fails: neither the pooled
    output nor the private winner buffer may survive.

    This is the one ownership path where nothing has adopted either object
    yet — ``_from_op`` never ran to completion, so no node owns the output
    and no history owns the winners. Cleanup must therefore be explicit and
    deterministic here; ``__del__`` is only the fallback for graph objects
    that are dropped, and this test must not depend on it."""
    values = np.arange(16, dtype=float).reshape(1, 1, 4, 4)
    xi = NativeTensor.from_array(values, requires_grad=True)
    baseline = len(live_storages)

    # Capture the two cores the (successful) forward produces, so their
    # closed state can be inspected directly after the failure.
    produced = []
    original_forward = cpp.NativeTensorCore._maxpool2d_forward_with_winners

    def capturing_forward(self, **kwargs):
        out_core, winners = original_forward(self, **kwargs)
        produced.append((out_core, winners))
        return out_core, winners

    monkeypatch.setattr(
        cpp.NativeTensorCore, "_maxpool2d_forward_with_winners",
        capturing_forward,
    )

    # Force graph construction itself to fail, after the forward completed.
    def exploding_from_op(cls, *args, **kwargs):
        raise RuntimeError("simulated graph-construction failure")

    monkeypatch.setattr(
        NativeTensor, "_from_op", classmethod(exploding_from_op)
    )

    with pytest.raises(RuntimeError, match="simulated graph-construction"):
        xi.maxpool2d(kernel_size=2)

    # The forward really did run and allocate both cores.
    assert len(produced) == 1
    out_core, winners = produced[0]
    # Both were released before the exception propagated.
    assert out_core._closed is True, "pooled output leaked on a failed graph"
    assert winners._closed is True, "winner buffer leaked on a failed graph"
    # Closed exactly once each: a second close is a no-op, not an error, and
    # the live count is already back at baseline before it.
    assert len(live_storages) == baseline
    out_core.close()
    winners.close()
    assert len(live_storages) == baseline

    # No partially constructed tensor escaped: nothing was returned, and the
    # only NativeTensor still alive is the input, open and unchanged.
    assert xi.closed is False
    assert np.array_equal(xi.to_numpy(), values)
    assert xi.requires_grad is True
    assert xi.grad is None

    # Restore normal behavior: a later forward/backward works end to end.
    monkeypatch.undo()
    y = xi.maxpool2d(kernel_size=2)
    assert y.to_numpy().tolist() == [[[[5.0, 7.0], [13.0, 15.0]]]]
    y.backward(gradient=NativeTensor.from_array(np.ones((1, 1, 2, 2))))
    assert xi.grad.to_numpy().tolist() == [[[
        [0, 0, 0, 0],
        [0, 1, 0, 1],
        [0, 0, 0, 0],
        [0, 1, 0, 1],
    ]]]
    assert y._graph_resources == ()  # released with the history, as usual
    y.close()
    xi.close()


def test_backward_never_rereads_the_input_or_recomputes_winners():
    # Closing the input after forward must not break backward: the saved
    # winners carry everything the gradient needs.
    x = np.arange(16, dtype=float).reshape(1, 1, 4, 4)
    xi = NativeParameter(x)
    y = xi.maxpool2d(kernel_size=2)
    grad_target = NativeParameter(x)  # a second parameter to receive grad
    y.backward(gradient=NativeTensor.from_array(np.ones((1, 1, 2, 2))))
    assert xi.grad.to_numpy().sum() == 4.0
    for t in (y, xi, grad_target):
        t.close()


# --------------------------------------------------------------------------
# Version tracking: deliberately none (contrast with conv2d)
# --------------------------------------------------------------------------

def test_no_expected_version_is_recorded():
    xi = NativeParameter(np.random.default_rng(19).standard_normal((1, 1, 4, 4)))
    y = xi.maxpool2d(kernel_size=2)
    # conv2d records a version where a callback rereads an operand's value;
    # maxpool2d backward reads only saved winners, so it records nothing.
    assert y._expected_versions == ()
    y.close()
    xi.close()


def test_input_mutation_after_forward_does_not_raise_or_change_routing():
    x = np.arange(16, dtype=float).reshape(1, 1, 4, 4)
    xi = NativeParameter(x)
    y = xi.maxpool2d(kernel_size=2)
    # Mutate the parameter's value through the sanctioned path, which bumps
    # its version — a conv2d graph would raise stale-graph here.
    replacement = NativeTensor.from_array(np.zeros((1, 1, 4, 4)))
    xi.copy_value_(replacement)
    assert xi._version > 0
    y.backward(gradient=NativeTensor.from_array(np.ones((1, 1, 2, 2))))
    # Routing still follows the winners recorded at forward time.
    assert xi.grad.to_numpy().tolist() == [[[
        [0, 0, 0, 0],
        [0, 1, 0, 1],
        [0, 0, 0, 0],
        [0, 1, 0, 1],
    ]]]
    for t in (y, replacement, xi):
        t.close()


# --------------------------------------------------------------------------
# Explicit-gradient validation
# --------------------------------------------------------------------------

def test_non_scalar_backward_without_a_gradient_raises_before_mutation():
    xi = NativeParameter(np.random.default_rng(20).standard_normal((1, 1, 4, 4)))
    y = xi.maxpool2d(kernel_size=2)
    with pytest.raises(ValueError, match="requires an explicit gradient"):
        y.backward()
    assert xi.grad is None          # nothing was committed
    assert y._graph_freed is False  # and nothing was freed
    assert _winners_open(y)
    y.close()
    xi.close()


@pytest.mark.parametrize("shape", [(1, 1, 3, 3), (1, 1, 2), (2, 1, 2, 2)])
def test_wrong_shaped_gradient_raises_before_mutation(shape):
    x = np.random.default_rng(21).standard_normal((1, 1, 4, 4))
    xi = NativeParameter(x)
    y = xi.maxpool2d(kernel_size=2)
    y.backward(gradient=NativeTensor.from_array(np.ones((1, 1, 2, 2))),
               retain_graph=True)
    before = xi.grad.to_numpy().copy()
    bad = NativeTensor.from_array(np.ones(shape))
    with pytest.raises(ValueError):
        y.backward(gradient=bad)
    assert np.array_equal(xi.grad.to_numpy(), before)  # unchanged
    assert _winners_open(y)
    for t in (y, bad, xi):
        t.close()


def test_non_native_gradient_rejected():
    xi = NativeParameter(np.random.default_rng(22).standard_normal((1, 1, 4, 4)))
    y = xi.maxpool2d(kernel_size=2)
    with pytest.raises(TypeError):
        y.backward(gradient=np.ones((1, 1, 2, 2)))
    assert xi.grad is None
    y.close()
    xi.close()


# --------------------------------------------------------------------------
# Forward validation (delegated to the D8 Core path)
# --------------------------------------------------------------------------

def test_closed_input_rejected():
    xi = NativeParameter(np.ones((1, 1, 4, 4)))
    xi.close()
    with pytest.raises(RuntimeError):
        xi.maxpool2d(kernel_size=2)


@pytest.mark.parametrize("shape", [(4, 4), (1, 4, 4), (1, 1, 1, 4, 4)])
def test_rank_other_than_four_rejected(shape):
    xi = NativeParameter(np.ones(shape))
    with pytest.raises(ValueError, match="4-D NCHW"):
        xi.maxpool2d(kernel_size=2)
    xi.close()


@pytest.mark.parametrize(
    "kwargs",
    [
        {"kernel_size": 0},
        {"kernel_size": -1},
        {"kernel_size": True},
        {"kernel_size": (2,)},
        {"kernel_size": (2, 1.5)},
        {"kernel_size": 2, "stride": 0},
        {"kernel_size": 2, "stride": True},
        {"kernel_size": 2, "padding": -1},
        {"kernel_size": 2, "padding": False},
        {"kernel_size": 9},  # window larger than the padded input
    ],
)
def test_invalid_arguments_rejected(kwargs):
    xi = NativeParameter(np.ones((1, 1, 4, 4)))
    with pytest.raises(ValueError):
        xi.maxpool2d(**kwargs)
    xi.close()


# --------------------------------------------------------------------------
# Failure rollback and retry
# --------------------------------------------------------------------------

@needs_fault_injection
def test_backward_allocation_failure_rolls_back_and_retries():
    x = np.random.default_rng(23).standard_normal((1, 1, 4, 4))
    xi = NativeParameter(x)
    g = NativeTensor.from_array(np.ones((1, 1, 2, 2)))
    y = xi.maxpool2d(kernel_size=2)
    # Seed a pre-existing gradient so rollback has something to restore.
    y.backward(gradient=g, retain_graph=True)
    before = xi.grad.to_numpy().copy()
    saved = _saved_winners(y)[0]

    # The explicit gradient allocates nothing, and the input is contiguous,
    # so the very first allocation of the pass is grad_input's zeros.
    cpp._arm_alloc_failure(1)
    with pytest.raises(MemoryError):
        y.backward(gradient=g, retain_graph=True)
    # Nothing partially committed, nothing freed, winners still alive.
    assert np.array_equal(xi.grad.to_numpy(), before)
    assert y._graph_freed is False
    assert not saved._closed
    assert cpp._require_library().tf_last_error_code() == cpp.TF_OK

    # The graph is still usable: a retry succeeds and accumulates.
    y.backward(gradient=g)
    assert np.allclose(xi.grad.to_numpy(), 2 * before, atol=1e-12)
    assert saved._closed is True  # released with the history this time
    for t in (y, g, xi):
        t.close()


def test_malformed_winner_failure_leaves_the_graph_retryable(monkeypatch):
    # Corrupt the saved winners through the private core, then prove the
    # checked boundary rejects the backward without committing anything.
    x = np.arange(16, dtype=float).reshape(1, 1, 4, 4)
    xi = NativeParameter(x)
    g = NativeTensor.from_array(np.ones((1, 1, 2, 2)))
    y = xi.maxpool2d(kernel_size=2)
    y.backward(gradient=g, retain_graph=True)
    before = xi.grad.to_numpy().copy()
    winners = _saved_winners(y)[0]
    good = winners.to_numpy()
    winners._storage.copy_from(np.array([0.5, 7.0, 13.0, 15.0]))
    with pytest.raises(ValueError, match="winner"):
        y.backward(gradient=g, retain_graph=True)
    assert np.array_equal(xi.grad.to_numpy(), before)  # rolled back
    assert not winners._closed                          # still retryable
    # Restore the real winners: the same graph then works again.
    winners._storage.copy_from(good.reshape(-1))
    y.backward(gradient=g)
    assert np.allclose(xi.grad.to_numpy(), 2 * before, atol=1e-12)
    for t in (y, g, xi):
        t.close()


# --------------------------------------------------------------------------
# Capability separation (operation implemented, module not)
# --------------------------------------------------------------------------

def test_operation_is_advertised_as_an_autograd_op():
    assert "maxpool2d" in cpp.AUTOGRAD_OPS
    assert hasattr(NativeTensor, "maxpool2d")
    assert "maxpool2d" not in cpp.UNSUPPORTED
    assert "maxpool2d_forward" in cpp.TENSOR_CORE_OPS
    assert "maxpool2d_backward" in cpp.TENSOR_CORE_OPS


def test_module_wraps_this_operation_without_extending_it():
    # D10 added the NativeMaxPool2d module on top of this operation. The
    # module is a module only: it is never advertised as an autograd op or
    # a Core/raw kernel, and it adds no capability this file does not
    # already cover.
    import tensorforge.experimental as experimental

    assert "NativeMaxPool2d" in cpp.NATIVE_MODULES
    assert "NativeMaxPool2d" in experimental.__all__
    assert "NativeMaxPool2d" not in cpp.UNSUPPORTED
    assert "NativeMaxPool2d" not in cpp.AUTOGRAD_OPS
    assert "NativeMaxPool2d" not in cpp.TENSOR_CORE_OPS
    assert "NativeMaxPool2d" not in cpp.RAW_KERNELS


def test_winner_buffer_never_appears_publicly():
    xi = NativeParameter(np.ones((1, 1, 4, 4)))
    y = xi.maxpool2d(kernel_size=2)
    public = [name for name in dir(y) if not name.startswith("_")]
    assert not [name for name in public if "winner" in name.lower()]
    assert not [name for name in public if "indices" in name.lower()]
    # Pooling adds no parameters or buffers to the native stack.
    assert not hasattr(y, "parameters")
    assert cpp.SUPPORTED_DTYPES == ("float64",)
    y.close()
    xi.close()


def test_conv2d_and_flatten_support_unaffected():
    assert "conv2d" in cpp.AUTOGRAD_OPS
    assert "NativeConv2d" in cpp.NATIVE_MODULES
    assert "NativeFlatten" in cpp.NATIVE_MODULES
