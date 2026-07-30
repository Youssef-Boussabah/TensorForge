"""Phase H, milestone H8 — the ``NativeBatchNorm2d`` affine-ownership
correction.

The defect
----------

``_NativeBatchNorm._affine``'s NCHW path applies the rank-1 ``gamma`` and
``beta`` by moving the *activation* channels-last rather than reshaping
the parameters (F4's deliberate choice, so both stay direct versioned
``multiply`` operands and the stale-parameter guard keeps firing). That
transpose is **metadata only**: it returns a view that borrows the
normalized activation's storage and owns nothing.

Whether the graph keeps that storage alive depended on where the
gradients entered:

* the input requires gradients — the transpose is a graph node whose
  *parent* is the activation, so the graph holds the owner; or
* nothing requires gradients — no graph is built and nothing rereads it.

But ``gamma``/``beta`` can introduce gradients the activation does not
carry. Then the transpose is a plain **borrowing leaf**: the graph
reaches the view (as ``multiply``'s parent, and ``multiply``'s backward
rereads that operand's forward value to build ``gamma``'s gradient) and
never reaches its owner, which was an ordinary local temporary released
when the forward returned. Backward then read freed storage and raised
``RuntimeError: this NativeStorage has been closed`` — reproduced here
for both modes, and equally present in the pre-H8 composition, which this
suite also pins.

The correction
--------------

The borrowed operand's owner is genuinely graph state, so the **graph
owns it**: ``_retain_channels_last_source`` adopts exactly that one
tensor as a ``graph_resources`` entry on the output node — the same D9
contract that already carries the evaluation snapshots, MaxPool2d's
winners, and Phase E's saved probabilities. It is adopted **only** in the
configuration that needs it, so the input-requires-grad path, the
``beta``-only path, and the no-graph path keep their exact previous
topology and retain nothing extra.

``NativeBatchNorm1d`` never had the defect and needs no correction: with
the channel axis already trailing, ``gamma`` multiplies the activation
**directly**, so the graph holds the owner as a parent in every
configuration. It is covered here as the comparison arm, unrefactored.

"Affine disabled" note
----------------------

``_NativeBatchNorm`` deliberately has no ``affine=False`` mode —
``gamma``/``beta`` always exist. The corresponding real configuration is
a **frozen** affine (both parameters ``requires_grad=False``), which is
what the affine-disabled tests below exercise, in both input-gradient
modes.

Nothing here is a new capability: no operation, kernel, C ABI symbol,
ctypes declaration, ``NativeTensorCore`` method, module, export, registry
value, checkpoint field, or ``state_dict`` field is added or moved.
"""

import gc

import numpy as np
import pytest

from tensorforge.backends import cpp
from tensorforge.experimental import (
    NativeAdam, NativeBatchNorm1d, NativeBatchNorm2d, NativeParameter,
    NativeTensor, load_native_checkpoint, save_native_checkpoint,
)
from tensorforge.experimental import native_batchnorm

needs_native = pytest.mark.skipif(
    not cpp.is_available(), reason="the experimental C++ backend is not built"
)


# ==========================================================================
# Fixtures, oracles, helpers
# ==========================================================================

EPS = 1e-5
MOMENTUM = 0.25
CHANNELS = 4
SHAPE_2D = (3, CHANNELS, 2, 5)
SHAPE_1D = (6, CHANNELS)
AXES = (0, 2, 3)

GAMMA = np.array([1.25, -0.5, 2.0, 0.75])
BETA = np.array([0.5, 1.5, -1.0, 0.25])
RUNNING_MEAN = np.array([0.1, -0.2, 0.3, 0.05])
RUNNING_VAR = np.array([1.4, 0.7, 2.2, 0.9])


def _values(shape, offset=0):
    """Deterministic, exactly representable float64 data — every value a
    quarter or a third of an integer, so no test result depends on a
    seed."""
    count = int(np.prod(shape))
    raw = (np.arange(count, dtype=np.float64) + offset) % 13
    return (raw / 4.0 - 1.5).reshape(shape)


def _upstream(shape, offset=0):
    count = int(np.prod(shape))
    raw = ((np.arange(count, dtype=np.float64) + offset) * 7) % 11
    return (raw / 4.0 - 1.25).reshape(shape)


@pytest.fixture
def live_storages(monkeypatch):
    """The ids of every NativeStorage currently open — the project's
    deterministic native-allocation instrumentation. The count is exact
    (it hooks ``close()``)."""
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


def _collect():
    """Force the composed autograd graph's intermediate wrappers — which
    participate in reference cycles, a property of the Python-managed
    native autograd engine since Phase B, not of BatchNorm and not of this
    correction — to their collection point.

    Deliberately never used to establish that an **adopted** resource was
    released: every such assertion below is made on the resource object
    itself, before any collection, because the whole point of the fix is
    that its lifetime is explicit."""
    gc.collect()


def _channels(values):
    return np.asarray(values, dtype=np.float64).reshape(1, -1, 1, 1)


def _configured(cls, training, *, running_mean=None, running_var=None):
    """A module with known affine values and known running statistics, in
    the requested mode."""
    module = cls(CHANNELS, eps=EPS, momentum=MOMENTUM)
    module.load_state_dict({
        "gamma": NativeTensor.from_array(GAMMA),
        "beta": NativeTensor.from_array(BETA),
        "running_mean": NativeTensor.from_array(
            RUNNING_MEAN if running_mean is None else running_mean),
        "running_var": NativeTensor.from_array(
            RUNNING_VAR if running_var is None else running_var),
    })
    module.train() if training else module.eval()
    return module


def _close_module(module):
    for parameter in module.parameters():
        parameter.close()
    for buffer in module.buffers():
        buffer.close()


def _freeze(module, name):
    """Replace one affine parameter with a frozen copy of itself, through
    the module's ordinary assignment registration. The evicted parameter
    is dropped by the registry, never closed, so this closes it — keeping
    the live-storage accounting exact."""
    old = getattr(module, name)
    values = old.to_numpy().copy()
    setattr(module, name, NativeParameter(values, requires_grad=False))
    old.close()


def train_reference_grads(x, upstream, gamma, eps=EPS):
    """The explicit NumPy NCHW population-BatchNorm backward: gradients of
    ``sum(upstream * output)`` with respect to ``gamma``, ``beta``, and the
    input. Reduces over N, H, and W and leaves the channel axis alone; no
    Bessel correction, epsilon inside the root."""
    mean = x.mean(axis=AXES, keepdims=True)
    var = ((x - mean) ** 2).mean(axis=AXES, keepdims=True)
    inverse_std = 1.0 / np.sqrt(var + eps)
    normalized = (x - mean) * inverse_std
    d_gamma = (upstream * normalized).sum(axis=AXES)
    d_beta = upstream.sum(axis=AXES)
    count = x.shape[0] * x.shape[2] * x.shape[3]
    d_normalized = upstream * _channels(gamma)
    d_x = inverse_std * (
        d_normalized
        - d_normalized.sum(axis=AXES, keepdims=True) / count
        - normalized * (d_normalized * normalized).sum(axis=AXES,
                                                       keepdims=True) / count
    )
    return d_gamma, d_beta, d_x


def eval_reference_grads(x, upstream, gamma, running_mean, running_var,
                         eps=EPS):
    """The evaluation-mode counterpart: the statistics are constants, so
    the input gradient is a pure per-channel scale."""
    inverse_std = 1.0 / np.sqrt(_channels(running_var) + eps)
    normalized = (x - _channels(running_mean)) * inverse_std
    return ((upstream * normalized).sum(axis=AXES),
            upstream.sum(axis=AXES),
            upstream * _channels(gamma) * inverse_std)


def train_reference_grads_1d(x, upstream, gamma, eps=EPS):
    """The ``(N, C)`` twin, reducing over the batch only."""
    mean = x.mean(axis=0, keepdims=True)
    var = ((x - mean) ** 2).mean(axis=0, keepdims=True)
    normalized = (x - mean) / np.sqrt(var + eps)
    return (upstream * normalized).sum(axis=0), upstream.sum(axis=0)


def _walk_resources(root):
    """Every native resource the graph reachable from ``root`` owns."""
    resources, seen, stack = [], set(), [root]
    while stack:
        node = stack.pop()
        if id(node) in seen:
            continue
        seen.add(id(node))
        resources.extend(node._graph_resources)
        stack.extend(node._parents)
    return resources


def _affine_sources(root, shape):
    """The adopted resources shaped like the *activation* — that is, the
    channels-last affine sources, as opposed to the ``(1, C, ...)``
    evaluation snapshots."""
    return [r for r in _walk_resources(root) if tuple(r.shape) == tuple(shape)]


# ==========================================================================
# The defect itself — the two configurations that used to raise
# ==========================================================================

@needs_native
def test_training_affine_backward_with_a_non_grad_input_is_exact():
    """1. Training, affine enabled, the input does **not** require grad.

    Backward must succeed, both affine gradients must be exact, and no
    input gradient may be created. Verified against the explicit NumPy
    formula *and* — bit for bit — against the input-requires-grad path,
    which is the pre-existing correct path for the same arithmetic."""
    x_values = _values(SHAPE_2D)
    up = _upstream(SHAPE_2D)

    module = _configured(NativeBatchNorm2d, True)
    x = NativeTensor.from_array(x_values, requires_grad=False)
    out = module(x)
    out.backward(NativeTensor.from_array(up))

    expected_gamma, expected_beta, _ = train_reference_grads(
        x_values, up, GAMMA)
    assert np.allclose(module.gamma.grad.to_numpy(), expected_gamma,
                       atol=1e-9)
    assert np.allclose(module.beta.grad.to_numpy(), expected_beta, atol=1e-9)
    # No input gradient is created at all — not a zero-filled one.
    assert x.grad is None

    # The same forward with a gradient-tracking input runs the identical
    # affine arithmetic, so the affine gradients must match *exactly*.
    reference = _configured(NativeBatchNorm2d, True)
    reference_x = NativeTensor.from_array(x_values, requires_grad=True)
    reference_out = reference(reference_x)
    reference_out.backward(NativeTensor.from_array(up))
    assert np.array_equal(module.gamma.grad.to_numpy(),
                          reference.gamma.grad.to_numpy())
    assert np.array_equal(module.beta.grad.to_numpy(),
                          reference.beta.grad.to_numpy())
    assert np.array_equal(out.to_numpy(), reference_out.to_numpy())


@needs_native
def test_evaluation_affine_backward_with_a_non_grad_input_is_exact():
    """2. Evaluation, affine enabled, the input does **not** require
    grad — the same contract over the snapshot path."""
    x_values = _values(SHAPE_2D, offset=3)
    up = _upstream(SHAPE_2D, offset=1)

    module = _configured(NativeBatchNorm2d, False)
    x = NativeTensor.from_array(x_values, requires_grad=False)
    out = module(x)
    out.backward(NativeTensor.from_array(up))

    expected_gamma, expected_beta, _ = eval_reference_grads(
        x_values, up, GAMMA, RUNNING_MEAN, RUNNING_VAR)
    assert np.allclose(module.gamma.grad.to_numpy(), expected_gamma,
                       atol=1e-9)
    assert np.allclose(module.beta.grad.to_numpy(), expected_beta, atol=1e-9)
    assert x.grad is None

    reference = _configured(NativeBatchNorm2d, False)
    reference_x = NativeTensor.from_array(x_values, requires_grad=True)
    reference_out = reference(reference_x)
    reference_out.backward(NativeTensor.from_array(up))
    assert np.array_equal(module.gamma.grad.to_numpy(),
                          reference.gamma.grad.to_numpy())
    assert np.array_equal(module.beta.grad.to_numpy(),
                          reference.beta.grad.to_numpy())
    assert np.array_equal(out.to_numpy(), reference_out.to_numpy())
    # Evaluation still updates nothing.
    assert np.array_equal(module.running_mean.to_numpy(), RUNNING_MEAN)
    assert np.array_equal(module.running_var.to_numpy(), RUNNING_VAR)


@needs_native
@pytest.mark.parametrize("training", [True, False])
def test_an_input_requiring_grad_keeps_every_gradient_exact(training):
    """3. The path that always worked, re-pinned in full: input, weight,
    and bias gradients all exact, in both modes."""
    x_values = _values(SHAPE_2D, offset=5)
    up = _upstream(SHAPE_2D, offset=2)

    module = _configured(NativeBatchNorm2d, training)
    x = NativeTensor.from_array(x_values, requires_grad=True)
    out = module(x)
    out.backward(NativeTensor.from_array(up))

    if training:
        e_gamma, e_beta, e_x = train_reference_grads(x_values, up, GAMMA)
    else:
        e_gamma, e_beta, e_x = eval_reference_grads(
            x_values, up, GAMMA, RUNNING_MEAN, RUNNING_VAR)
    assert np.allclose(module.gamma.grad.to_numpy(), e_gamma, atol=1e-9)
    assert np.allclose(module.beta.grad.to_numpy(), e_beta, atol=1e-9)
    assert np.allclose(x.grad.to_numpy(), e_x, atol=1e-9)


@needs_native
@pytest.mark.parametrize("training", [True, False])
def test_only_the_weight_requiring_grad(training):
    """4. ``gamma`` trainable, ``beta`` frozen, the input not tracking.

    This is the narrowest configuration that still needs the borrowed
    operand's value: ``multiply``'s backward rereads the channels-last
    view to build ``gamma``'s gradient."""
    x_values = _values(SHAPE_2D, offset=7)
    up = _upstream(SHAPE_2D, offset=4)

    module = _configured(NativeBatchNorm2d, training)
    _freeze(module, "beta")
    x = NativeTensor.from_array(x_values, requires_grad=False)
    out = module(x)
    # Read the inventory before backward: a one-shot pass releases it.
    assert len(_affine_sources(out, SHAPE_2D)) == 1
    out.backward(NativeTensor.from_array(up))

    if training:
        e_gamma, _e_beta, _ = train_reference_grads(x_values, up, GAMMA)
    else:
        e_gamma, _e_beta, _ = eval_reference_grads(
            x_values, up, GAMMA, RUNNING_MEAN, RUNNING_VAR)
    assert np.allclose(module.gamma.grad.to_numpy(), e_gamma, atol=1e-9)
    assert module.beta.grad is None
    assert x.grad is None


@needs_native
@pytest.mark.parametrize("training", [True, False])
def test_only_the_bias_requiring_grad(training):
    """5. ``beta`` trainable, ``gamma`` frozen, the input not tracking.

    ``add``'s backward reads no operand value, so nothing ever rereads the
    borrowed view — and ``multiply``'s result is then a plain **owning**
    leaf that the graph holds as a parent. Nothing is adopted, and that
    minimality is asserted, not assumed."""
    x_values = _values(SHAPE_2D, offset=9)
    up = _upstream(SHAPE_2D, offset=6)

    module = _configured(NativeBatchNorm2d, training)
    _freeze(module, "gamma")
    x = NativeTensor.from_array(x_values, requires_grad=False)
    out = module(x)
    assert len(_affine_sources(out, SHAPE_2D)) == 0
    out.backward(NativeTensor.from_array(up))

    if training:
        _e_gamma, e_beta, _ = train_reference_grads(x_values, up, GAMMA)
    else:
        _e_gamma, e_beta, _ = eval_reference_grads(
            x_values, up, GAMMA, RUNNING_MEAN, RUNNING_VAR)
    assert np.allclose(module.beta.grad.to_numpy(), e_beta, atol=1e-9)
    assert module.gamma.grad is None
    assert x.grad is None


@needs_native
@pytest.mark.parametrize("training", [True, False])
def test_no_gradient_source_at_all_preserves_the_no_graph_behavior(training):
    """6. Neither affine parameter requires grad and neither does the
    input: the forward must still produce a plain owning no-grad leaf with
    no graph, no adopted resource, and — in training mode — a running
    update that still happened."""
    x_values = _values(SHAPE_2D, offset=11)
    module = _configured(NativeBatchNorm2d, training)
    _freeze(module, "gamma")
    _freeze(module, "beta")
    x = NativeTensor.from_array(x_values, requires_grad=False)
    out = module(x)

    assert out.requires_grad is False
    assert out.is_leaf is True
    assert out._graph_resources == ()
    assert out.owns_core and out.contiguous
    with pytest.raises(RuntimeError, match="does not require grad"):
        out.backward(NativeTensor.from_array(_upstream(SHAPE_2D)))

    if training:
        mean = x_values.mean(axis=AXES)
        variance = ((x_values - _channels(mean)) ** 2).mean(axis=AXES)
        assert np.allclose(
            module.running_mean.to_numpy(),
            (1 - MOMENTUM) * RUNNING_MEAN + MOMENTUM * mean, atol=1e-12)
        assert np.allclose(
            module.running_var.to_numpy(),
            (1 - MOMENTUM) * RUNNING_VAR + MOMENTUM * variance, atol=1e-12)
    else:
        assert np.array_equal(module.running_mean.to_numpy(), RUNNING_MEAN)
        assert np.array_equal(module.running_var.to_numpy(), RUNNING_VAR)


@needs_native
@pytest.mark.parametrize("training", [True, False])
@pytest.mark.parametrize("input_requires_grad", [True, False])
def test_a_frozen_affine_preserves_the_existing_graph_behavior(
        training, input_requires_grad):
    """7. "Affine disabled" — both affine parameters frozen — in both
    input-gradient modes.

    With the input tracking, the graph is exactly what it was: the input
    gradient is exact and, because the transpose is itself a graph node
    holding the activation as its parent, nothing extra is owned. With the
    input not tracking, there is no graph at all."""
    x_values = _values(SHAPE_2D, offset=13)
    up = _upstream(SHAPE_2D, offset=8)
    module = _configured(NativeBatchNorm2d, training)
    _freeze(module, "gamma")
    _freeze(module, "beta")
    x = NativeTensor.from_array(x_values, requires_grad=input_requires_grad)
    out = module(x)

    assert len(_affine_sources(out, SHAPE_2D)) == 0
    if not input_requires_grad:
        assert out.requires_grad is False and out.is_leaf is True
        return

    out.backward(NativeTensor.from_array(up))
    if training:
        _g, _b, e_x = train_reference_grads(x_values, up, GAMMA)
    else:
        _g, _b, e_x = eval_reference_grads(
            x_values, up, GAMMA, RUNNING_MEAN, RUNNING_VAR)
    assert np.allclose(x.grad.to_numpy(), e_x, atol=1e-9)
    assert module.gamma.grad is None and module.beta.grad is None


# ==========================================================================
# Ownership topology — what is adopted, and only what is needed
# ==========================================================================

@needs_native
@pytest.mark.parametrize("training", [True, False])
def test_the_affine_source_is_adopted_only_where_it_is_needed(training):
    """The whole configuration table at once, so a future change that
    starts retaining more (or less) fails here. The graph owns the
    activation exactly when the transposed operand is a borrowing leaf
    that a backward will reread."""
    x_values = _values(SHAPE_2D)
    cases = {
        # (input requires grad, gamma frozen, beta frozen): adopted sources
        (True, False, False): 0,    # transpose is a graph node: already owned
        (True, True, True): 0,      # ditto, frozen affine
        (False, False, False): 1,   # the defect's configuration
        (False, False, True): 1,    # gamma alone still rereads the view
        (False, True, False): 0,    # beta alone reads no operand value
        (False, True, True): 0,     # no graph at all
    }
    for (input_grad, freeze_gamma, freeze_beta), expected in cases.items():
        module = _configured(NativeBatchNorm2d, training)
        if freeze_gamma:
            _freeze(module, "gamma")
        if freeze_beta:
            _freeze(module, "beta")
        x = NativeTensor.from_array(x_values, requires_grad=input_grad)
        out = module(x)
        assert len(_affine_sources(out, SHAPE_2D)) == expected, (
            input_grad, freeze_gamma, freeze_beta)
        out.close()


@needs_native
@pytest.mark.parametrize("training", [True, False])
def test_the_adopted_source_is_owning_independent_and_not_the_input(
        training):
    """The adopted resource is the *normalized activation*, not the
    caller's input and not a borrowing view: it owns its own storage,
    shares none with the input or with either running buffer, and carries
    no graph of its own."""
    x_values = _values(SHAPE_2D, offset=2)
    module = _configured(NativeBatchNorm2d, training)
    x = NativeTensor.from_array(x_values, requires_grad=False)
    out = module(x)

    sources = _affine_sources(out, SHAPE_2D)
    assert len(sources) == 1
    source = sources[0]
    assert source.owns_core and source.contiguous
    assert source.requires_grad is False and source.is_leaf is True
    forbidden = {id(x._core.storage),
                 id(module.running_mean._core.storage),
                 id(module.running_var._core.storage),
                 id(module.gamma._core.storage),
                 id(module.beta._core.storage),
                 id(out._core.storage)}
    assert id(source._core.storage) not in forbidden
    # And it really is the normalized activation the affine consumed.
    if training:
        mean = x_values.mean(axis=AXES, keepdims=True)
        variance = ((x_values - mean) ** 2).mean(axis=AXES, keepdims=True)
    else:
        mean = _channels(RUNNING_MEAN)
        variance = _channels(RUNNING_VAR)
    assert np.allclose(source.to_numpy(),
                       (x_values - mean) / np.sqrt(variance + EPS), atol=1e-9)


@needs_native
def test_no_registered_buffer_or_parameter_is_ever_adopted():
    """The correction must not have widened what the graph owns: no
    adopted resource may be a registered buffer, a parameter, or a view
    onto their storage. Checked in both modes, by object identity **and**
    by storage identity."""
    for training in (True, False):
        module = _configured(NativeBatchNorm2d, training)
        x = NativeTensor.from_array(_values(SHAPE_2D), requires_grad=False)
        out = module(x)
        owned = {id(module.gamma), id(module.beta),
                 id(module.running_mean), id(module.running_var)}
        owned_storage = {id(module.gamma._core.storage),
                         id(module.beta._core.storage),
                         id(module.running_mean._core.storage),
                         id(module.running_var._core.storage)}
        for resource in _walk_resources(out):
            assert id(resource) not in owned
            assert id(resource._core.storage) not in owned_storage
            assert not isinstance(resource, NativeParameter)
        out.close()
        x.close()
        _close_module(module)


# ==========================================================================
# Saved-resource lifetime and cleanup
# ==========================================================================

@needs_native
@pytest.mark.parametrize("training", [True, False])
def test_the_source_lives_until_backward_finishes_and_closes_exactly_once(
        training):
    """11. The lifetime contract, asserted on the resource itself with no
    collection anywhere: open through the whole backward, closed **once**
    at the graph-release point, never used afterwards, never closed
    twice."""
    module = _configured(NativeBatchNorm2d, training)
    x = NativeTensor.from_array(_values(SHAPE_2D), requires_grad=False)
    out = module(x)
    source = _affine_sources(out, SHAPE_2D)[0]

    closes = []
    storage = source._core.storage
    original_close = type(storage).close

    def counting_close(self):
        if self is storage:
            closes.append(1)
        original_close(self)

    type(storage).close = counting_close
    try:
        assert not source.closed
        # The backward that actually rereads it must find it open.
        out.backward(NativeTensor.from_array(_upstream(SHAPE_2D)))
        assert source.closed
        assert sum(closes) == 1
        # A later close of the output is inert — the resource tuple was
        # cleared at release, so nothing can be closed a second time.
        out.close()
        assert sum(closes) == 1
        assert out._graph_resources == ()
        with pytest.raises(RuntimeError, match="has been closed"):
            source.to_numpy()
    finally:
        type(storage).close = original_close


@needs_native
@pytest.mark.parametrize("training", [True, False])
def test_retain_graph_keeps_the_source_for_a_second_backward(training):
    """A retained graph may run backward again, so its saved resources
    must survive — and the second pass must produce the same gradient
    contribution the first did."""
    x_values = _values(SHAPE_2D, offset=4)
    up = _upstream(SHAPE_2D, offset=3)
    module = _configured(NativeBatchNorm2d, training)
    x = NativeTensor.from_array(x_values, requires_grad=False)
    out = module(x)
    source = _affine_sources(out, SHAPE_2D)[0]

    out.backward(NativeTensor.from_array(up), retain_graph=True)
    first = module.gamma.grad.to_numpy().copy()
    assert not source.closed
    out.backward(NativeTensor.from_array(up), retain_graph=True)
    assert not source.closed
    # Gradients accumulate across retained passes.
    assert np.allclose(module.gamma.grad.to_numpy(), 2.0 * first, atol=1e-9)
    out.backward(NativeTensor.from_array(up))
    assert source.closed


@needs_native
@pytest.mark.parametrize("training", [True, False])
def test_an_abandoned_graph_releases_the_source_on_close(training):
    """No backward ever runs: an explicit ``close()`` on the output must
    still release the adopted source deterministically — not leave it to
    the collector."""
    module = _configured(NativeBatchNorm2d, training)
    x = NativeTensor.from_array(_values(SHAPE_2D, offset=6),
                                requires_grad=False)
    out = module(x)
    source = _affine_sources(out, SHAPE_2D)[0]
    assert not source.closed
    out.close()
    assert source.closed          # deterministic, before any collection
    out.close()                   # idempotent
    assert source.closed


@needs_native
@pytest.mark.parametrize("training", [True, False])
def test_a_failed_retryable_backward_keeps_the_source_alive(training):
    """A backward that raises leaves the graph intact by contract, so the
    saved source must still be open and the retry must succeed."""
    module = _configured(NativeBatchNorm2d, training)
    x = NativeTensor.from_array(_values(SHAPE_2D, offset=8),
                               requires_grad=False)
    out = module(x)
    source = _affine_sources(out, SHAPE_2D)[0]
    # A gradient of the wrong shape is rejected before anything is touched.
    with pytest.raises(ValueError):
        out.backward(NativeTensor.from_array(np.ones((2, 2))))
    assert not source.closed
    assert module.gamma.grad is None
    out.backward(NativeTensor.from_array(_upstream(SHAPE_2D, offset=8)))
    assert module.gamma.grad is not None
    assert source.closed


# ==========================================================================
# Failure atomicity around the publication itself
# ==========================================================================

@needs_native
@pytest.mark.parametrize("training", [True, False])
def test_a_failure_before_publication_leaks_nothing(
        training, monkeypatch, live_storages):
    """10a. The adoption step itself fails **before** anything is
    published: the forward must raise, adopt nothing, close every
    temporary it created, leave the running statistics exactly as they
    were, and return live storage to baseline."""
    module = _configured(NativeBatchNorm2d, training)
    x = NativeTensor.from_array(_values(SHAPE_2D), requires_grad=False)
    module(x).close()                     # warm up, then take the baseline
    _collect()
    baseline = set(live_storages)
    before_mean = module.running_mean.to_numpy().copy()
    before_var = module.running_var.to_numpy().copy()
    mean_id = id(module.running_mean)
    var_id = id(module.running_var)

    def exploding(*args, **kwargs):
        raise RuntimeError("injected publication failure")

    monkeypatch.setattr(native_batchnorm, "_retain_channels_last_source",
                        exploding)
    with pytest.raises(RuntimeError, match="injected publication failure"):
        module(x)
    monkeypatch.undo()

    _collect()
    assert live_storages == baseline
    # The publication sits inside step 1, before the F1 transaction, so a
    # training forward must not have advanced either buffer.
    assert np.array_equal(module.running_mean.to_numpy(), before_mean)
    assert np.array_equal(module.running_var.to_numpy(), before_var)
    assert id(module.running_mean) == mean_id
    assert id(module.running_var) == var_id
    # And the module still works afterwards.
    survivor = module(x)
    assert survivor.shape == SHAPE_2D
    survivor.close()


@needs_native
@pytest.mark.parametrize("training", [True, False])
def test_a_failure_after_publication_still_leaks_nothing(
        training, monkeypatch, live_storages):
    """10b. The harder half: the resource **is** published and the forward
    then fails. The output is closed by the forward's own cleanup, which
    releases the adopted resource exactly once — nothing is left owned by
    a graph nobody holds."""
    module = _configured(NativeBatchNorm2d, training)
    x = NativeTensor.from_array(_values(SHAPE_2D), requires_grad=False)
    module(x).close()
    _collect()
    baseline = set(live_storages)
    before_mean = module.running_mean.to_numpy().copy()
    before_var = module.running_var.to_numpy().copy()
    real_adopt = native_batchnorm._adopt_graph_resources
    published = []

    def adopt_then_fail(node, resources):
        adopted = real_adopt(node, resources)
        published.append((node, tuple(resources), adopted))
        raise RuntimeError("injected post-publication failure")

    monkeypatch.setattr(native_batchnorm, "_adopt_graph_resources",
                        adopt_then_fail)
    with pytest.raises(RuntimeError, match="injected post-publication"):
        module(x)
    monkeypatch.undo()

    node, resources, adopted = published[0]
    assert adopted is True                      # it really was published
    assert all(resource.closed for resource in resources)
    assert node.closed
    _collect()
    assert live_storages == baseline
    assert np.array_equal(module.running_mean.to_numpy(), before_mean)
    assert np.array_equal(module.running_var.to_numpy(), before_var)


# ==========================================================================
# Lifecycle: repeated cycles, repeated mode transitions
# ==========================================================================

@needs_native
def test_repeated_train_eval_transitions_stay_correct_and_bounded(
        live_storages):
    """8. Alternating modes across many forwards: every pass computes the
    exact affine gradients for its own mode, evaluation never advances a
    buffer, and the adopted source count is exactly one every time."""
    module = _configured(NativeBatchNorm2d, True)
    x_values = _values(SHAPE_2D, offset=1)
    up = _upstream(SHAPE_2D, offset=5)
    x = NativeTensor.from_array(x_values, requires_grad=False)
    module(x).close()
    _collect()
    baseline = set(live_storages)

    for step in range(12):
        training = step % 2 == 0
        module.train() if training else module.eval()
        running_mean = module.running_mean.to_numpy().copy()
        running_var = module.running_var.to_numpy().copy()
        out = module(x)
        assert len(_affine_sources(out, SHAPE_2D)) == 1
        out.backward(NativeTensor.from_array(up))
        if training:
            expected_gamma, expected_beta, _ = train_reference_grads(
                x_values, up, GAMMA)
        else:
            expected_gamma, expected_beta, _ = eval_reference_grads(
                x_values, up, GAMMA, running_mean, running_var)
            assert np.array_equal(module.running_mean.to_numpy(),
                                  running_mean)
            assert np.array_equal(module.running_var.to_numpy(), running_var)
        assert np.allclose(module.gamma.grad.to_numpy(), expected_gamma,
                           atol=1e-9)
        assert np.allclose(module.beta.grad.to_numpy(), expected_beta,
                           atol=1e-9)
        module.gamma.zero_grad()
        module.beta.zero_grad()
        out.close()

    _collect()
    # A training step legitimately replaces both running-buffer storages
    # through the F1 transaction, so across mixed modes the *count* is the
    # invariant rather than the identity set.
    assert len(live_storages) == len(baseline)


@needs_native
@pytest.mark.parametrize("training", [True, False])
def test_repeated_cycles_with_explicit_cleanup_return_to_baseline(
        training, live_storages):
    """9. Thirty forward/backward cycles with explicit cleanup return live
    storage exactly to its measured baseline.

    The adopted source's own release is asserted **without** any
    collection — each cycle checks the resource object it just created.
    The single ``_collect()`` at the end covers the autograd engine's
    pre-existing ``sqrt``/``reciprocal`` reference cycles, which predate
    this correction and are not what it fixes."""
    module = _configured(NativeBatchNorm2d, training)
    x = NativeTensor.from_array(_values(SHAPE_2D, offset=2),
                               requires_grad=False)
    up = NativeTensor.from_array(_upstream(SHAPE_2D, offset=2))
    module(x).close()
    _collect()
    baseline = set(live_storages)

    for _ in range(30):
        out = module(x)
        source = _affine_sources(out, SHAPE_2D)[0]
        out.backward(up)
        assert source.closed          # deterministic, no collection needed
        module.gamma.zero_grad()
        module.beta.zero_grad()
        out.close()

    _collect()
    if training:
        # Both running-buffer storages are replaced every step by the F1
        # transaction, so the count is the invariant here.
        assert len(live_storages) == len(baseline)
    else:
        assert live_storages == baseline
    x.close()
    up.close()
    _close_module(module)
    _collect()


# ==========================================================================
# Identity, state, and persistence are untouched
# ==========================================================================

@needs_native
@pytest.mark.parametrize("training", [True, False])
def test_identities_versions_and_buffers_survive_the_backward(training):
    """The correction touches ownership only. Parameter and buffer object
    identities, parameter version counters, the running-statistic values,
    and the ``state_dict`` order are all exactly what they were — and a
    backward mutates no persistent buffer."""
    module = _configured(NativeBatchNorm2d, training)
    identities = (id(module.gamma), id(module.beta),
                  id(module.running_mean), id(module.running_var))
    affine_storages = (id(module.gamma._core.storage),
                       id(module.beta._core.storage))
    versions = (module.gamma.version, module.beta.version)
    assert list(module.state_dict()) == [
        "gamma", "beta", "running_mean", "running_var"]

    x = NativeTensor.from_array(_values(SHAPE_2D), requires_grad=False)
    out = module(x)
    after_forward_mean = module.running_mean.to_numpy().copy()
    after_forward_var = module.running_var.to_numpy().copy()
    # A training forward legitimately replaces each running buffer's core
    # through the F1 transaction (identity preserved, storage swapped), so
    # the buffer *storages* are sampled after the forward and must then
    # survive the backward untouched.
    buffer_storages = (id(module.running_mean._core.storage),
                       id(module.running_var._core.storage))
    out.backward(NativeTensor.from_array(_upstream(SHAPE_2D)))

    assert (id(module.gamma), id(module.beta), id(module.running_mean),
            id(module.running_var)) == identities
    assert (id(module.gamma._core.storage),
            id(module.beta._core.storage)) == affine_storages
    assert (id(module.running_mean._core.storage),
            id(module.running_var._core.storage)) == buffer_storages
    assert (module.gamma.version, module.beta.version) == versions
    # Backward never touches a persistent buffer.
    assert np.array_equal(module.running_mean.to_numpy(), after_forward_mean)
    assert np.array_equal(module.running_var.to_numpy(), after_forward_var)
    assert list(module.state_dict()) == [
        "gamma", "beta", "running_mean", "running_var"]


@needs_native
def test_a_non_grad_input_training_run_still_resumes_exactly(tmp_path):
    """Checkpoint exact resume over the corrected path: a model trained
    with a **non-gradient** input (only the affine parameters learn) is
    interrupted, reloaded into a fresh model and optimizer, and reproduces
    the remaining loss suffix, every parameter, and both running buffers by
    exact equality."""
    def build():
        module = _configured(NativeBatchNorm2d, True)
        return module, NativeAdam(module.parameters(), lr=0.05)

    def step(module, optimizer, index):
        x = NativeTensor.from_array(_values(SHAPE_2D, offset=index),
                                    requires_grad=False)
        out = module(x)
        # A scalar objective built from existing native operations.
        loss = out.multiply(out).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        value = float(loss.to_numpy())
        loss.close()
        out.close()
        x.close()
        return value

    uninterrupted, optimizer = build()
    reference_losses = [step(uninterrupted, optimizer, i) for i in range(8)]

    interrupted, interrupted_optimizer = build()
    for index in range(4):
        step(interrupted, interrupted_optimizer, index)
    path = tmp_path / "affine_ownership.tfckpt"
    save_native_checkpoint(path, interrupted, interrupted_optimizer,
                           metadata={"training_step": 4})

    resumed, resumed_optimizer = build()
    metadata = load_native_checkpoint(path, resumed, resumed_optimizer)
    assert metadata["training_step"] == 4
    resumed_losses = [step(resumed, resumed_optimizer, i) for i in range(4, 8)]

    assert resumed_losses == reference_losses[4:]
    assert np.array_equal(resumed.gamma.to_numpy(),
                          uninterrupted.gamma.to_numpy())
    assert np.array_equal(resumed.beta.to_numpy(),
                          uninterrupted.beta.to_numpy())
    assert np.array_equal(resumed.running_mean.to_numpy(),
                          uninterrupted.running_mean.to_numpy())
    assert np.array_equal(resumed.running_var.to_numpy(),
                          uninterrupted.running_var.to_numpy())


# ==========================================================================
# NativeBatchNorm1d — the comparison arm, unrefactored
# ==========================================================================

@needs_native
@pytest.mark.parametrize("training", [True, False])
def test_batchnorm1d_never_borrows_and_adopts_no_affine_source(training):
    """``NativeBatchNorm1d`` applies its rank-1 affine parameters directly
    to the trailing channel axis, so ``multiply``'s parent **is** the
    owning activation and no channels-last view exists. It therefore
    adopts nothing beyond the evaluation snapshots — asserted, so a future
    change that gives the 1-D shape a transposed affine cannot silently
    reintroduce the 2-D defect."""
    module = _configured(NativeBatchNorm1d, training)
    x = NativeTensor.from_array(_values(SHAPE_1D), requires_grad=False)
    out = module(x)
    resources = _walk_resources(out)
    assert len(_affine_sources(out, SHAPE_1D)) == 0
    # Training adopts nothing at all; evaluation adopts exactly the two
    # (1, C) running-statistic snapshots.
    expected_shape = (1, CHANNELS)
    assert len(resources) == (0 if training else 2)
    assert all(tuple(r.shape) == expected_shape for r in resources)
    out.close()


@needs_native
@pytest.mark.parametrize("training", [True, False])
def test_batchnorm1d_affine_gradients_with_a_non_grad_input_are_exact(
        training):
    """The regression test the 2-D shape needed, run against the shape
    that never had the defect — so the two arms are compared under
    identical conditions rather than assumed equivalent."""
    x_values = _values(SHAPE_1D)
    up = _upstream(SHAPE_1D)
    module = _configured(NativeBatchNorm1d, training)
    x = NativeTensor.from_array(x_values, requires_grad=False)
    out = module(x)
    out.backward(NativeTensor.from_array(up))

    if training:
        expected_gamma, expected_beta = train_reference_grads_1d(
            x_values, up, GAMMA)
    else:
        normalized = ((x_values - RUNNING_MEAN)
                      / np.sqrt(RUNNING_VAR + EPS))
        expected_gamma = (up * normalized).sum(axis=0)
        expected_beta = up.sum(axis=0)
    assert np.allclose(module.gamma.grad.to_numpy(), expected_gamma,
                       atol=1e-9)
    assert np.allclose(module.beta.grad.to_numpy(), expected_beta, atol=1e-9)
    assert x.grad is None


@needs_native
@pytest.mark.parametrize("training", [True, False])
def test_batchnorm1d_repeated_cycles_return_to_baseline(training,
                                                        live_storages):
    """The 1-D lifecycle arm, so the comparison covers storage accounting
    as well as gradients."""
    module = _configured(NativeBatchNorm1d, training)
    x = NativeTensor.from_array(_values(SHAPE_1D, offset=3),
                               requires_grad=False)
    up = NativeTensor.from_array(_upstream(SHAPE_1D, offset=3))
    module(x).close()
    _collect()
    baseline = set(live_storages)
    for _ in range(30):
        out = module(x)
        out.backward(up)
        module.gamma.zero_grad()
        module.beta.zero_grad()
        out.close()
    _collect()
    if training:
        assert len(live_storages) == len(baseline)
    else:
        assert live_storages == baseline
    x.close()
    up.close()
    _close_module(module)
    _collect()


# ==========================================================================
# Guardrails — the correction adds no surface
# ==========================================================================

@needs_native
def test_the_correction_adds_no_public_surface():
    """One private module-level helper and nothing else: no new export, no
    new module attribute a caller could reach, and no new configuration
    knob on either public class."""
    import tensorforge.experimental as experimental

    assert not hasattr(experimental, "_retain_channels_last_source")
    assert "_retain_channels_last_source" not in experimental.__all__
    assert native_batchnorm._retain_channels_last_source.__name__ == (
        "_retain_channels_last_source")
    for cls in (NativeBatchNorm1d, NativeBatchNorm2d):
        module = cls(CHANNELS)
        for forbidden in ("graph_resources", "saved_source", "affine_source",
                          "retain_source", "_saved", "_source_cache"):
            assert not hasattr(module, forbidden), forbidden
        _close_module(module)


@needs_native
def test_the_shared_implementation_still_has_one_affine_method():
    """The two public classes still inherit every method by function
    identity — the correction added shape-specific behavior to the shared
    ``_affine``, not a second implementation."""
    assert (NativeBatchNorm1d._affine.__func__ is
            NativeBatchNorm2d._affine.__func__
            if hasattr(NativeBatchNorm1d._affine, "__func__")
            else NativeBatchNorm1d._affine is NativeBatchNorm2d._affine)
    assert NativeBatchNorm1d._CHANNELS_LAST is None
    assert NativeBatchNorm2d._CHANNELS_LAST == (0, 2, 3, 1)
