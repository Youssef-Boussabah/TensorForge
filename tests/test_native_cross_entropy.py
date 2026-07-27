"""Differentiable native cross-entropy — ``NativeTensor.cross_entropy``
(Phase E, milestone E6).

E6 is the **autograd integration** of the completed E5 Core contract, and
nothing else: `NativeTensor.cross_entropy(targets, reduction="mean")`
calls `NativeTensorCore.cross_entropy_forward` exactly once, wraps the
scalar loss core in one graph node, adopts the private saved
probabilities as that node's **graph-owned resource** (the D9
`graph_resources` contract MaxPool2d's winner buffer established), and
captures the independently copied `int64` targets and the normalized
reduction in its backward closure. Backward runs
`NativeTensorCore.cross_entropy_backward` over that saved state and the
native scalar upstream, then accumulates into the logits parent.

No new numerical capability appears here: the kernels, the C ABI, the
target contract, the reduction contract, and the stability transform are
all E5's, unchanged, and their contract is pinned separately in
tests/test_native_cross_entropy_core.py.

The load-bearing invariants these tests pin down:

* **backward never rereads the logits** — the Core backward's signature
  cannot even see them — so the graph records **no expected parameter
  version** and a logits `NativeParameter` mutated after forward still
  differentiates, against the probabilities the forward actually saved;
* the saved probabilities are released **exactly once**, at the
  deterministic graph-release points, and survive `retain_graph=True`,
  a failed retryable backward, and nothing else;
* caller target mutation after forward cannot reach backward;
* a failure anywhere — E5 forward, graph construction, backward — commits
  no gradient, leaks no core, and leaves the graph honestly retryable;
* no tensor data crosses NumPy on either path.

Backend-dependent, so the module skips cleanly when the compiled backend
is not built. Cleanup is explicit via close().

Selector: python -m pytest -q -k "native_cross_entropy and not core"
"""

import gc

import numpy as np
import pytest

import tensorforge
from tensorforge.backends import cpp
from tensorforge.experimental import NativeParameter, NativeTensor

needs_native = pytest.mark.skipif(
    not cpp.is_available(),
    reason="experimental C++ backend not built; " + cpp.build_instructions(),
)
needs_fault_injection = pytest.mark.skipif(
    not (cpp.is_available() and cpp.fault_injection_available()),
    reason="fault injection not compiled into the backend",
)

LOGITS = np.array([[1.0, 2.0, 0.5], [-1.0, 0.25, 3.0]])
TARGETS = [1, 2]
REDUCTIONS = ("mean", "sum")


@pytest.fixture(autouse=True)
def _disarm_after_each():
    """No test may leave the allocation injector armed or the native error
    slot dirty (the test_native_abi_errors.py convention)."""
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


# ----------------------------------------------------------------------
# References — the same stable algorithm the kernel implements, in NumPy,
# used only as an external oracle (never inside an armed tripwire).
# ----------------------------------------------------------------------


def loss_reference(logits, targets, reduction):
    logits = np.asarray(logits, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.int64)
    batch_size = logits.shape[0]
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    log_denom = np.log(np.sum(np.exp(shifted), axis=1))
    per_example = log_denom - shifted[np.arange(batch_size), targets]
    total = float(per_example.sum())
    return total / batch_size if reduction == "mean" else total


def probabilities_reference(logits):
    logits = np.asarray(logits, dtype=np.float64)
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exponentials = np.exp(shifted)
    return exponentials / np.sum(exponentials, axis=1, keepdims=True)


def grad_reference(logits, targets, reduction, upstream=1.0):
    """upstream * (p - onehot) / N, in exactly that order."""
    base = probabilities_reference(logits)
    targets = np.asarray(targets, dtype=np.int64)
    batch_size = base.shape[0]
    base[np.arange(batch_size), targets] -= 1.0
    if reduction == "mean":
        base /= batch_size
    return upstream * base


def core_backward_reference(logits, targets, reduction, upstream=1.0):
    """The same gradient computed through the **E5 Core** layer, so the
    public operation is checked against the contract it is built on and
    not only against NumPy."""
    core = cpp.NativeTensorCore.from_array(np.ascontiguousarray(logits))
    result = core.cross_entropy_forward(targets, reduction)
    seed = cpp.NativeTensorCore.full((), upstream)
    gradient = result.probabilities.cross_entropy_backward(
        result.targets, seed, reduction
    )
    values = gradient.to_numpy().copy()
    loss = float(result.loss.to_numpy())
    gradient.close()
    seed.close()
    result.close()
    core.close()
    return loss, values


def saved_probabilities(output):
    """The private probability cores this graph node owns (white-box: the
    lifetime contract is exactly what these tests must pin down)."""
    return output._graph_resources


def one_saved(output):
    resources = saved_probabilities(output)
    assert len(resources) == 1, (
        f"expected exactly one saved-probability resource, got {resources!r}"
    )
    return resources[0]


def scalar_seed(value=1.0):
    """An explicit rank-0 native upstream — the shape a scalar loss's
    gradient really has under this engine."""
    return NativeTensor.full((), value)


# ======================================================================
# Public forward
# ======================================================================


@needs_native
@pytest.mark.parametrize("reduction", REDUCTIONS)
def test_native_cross_entropy_forward_matches_the_core_contract(reduction):
    expected, _ = core_backward_reference(LOGITS, TARGETS, reduction)
    x = NativeTensor.from_array(LOGITS, requires_grad=True)
    loss = x.cross_entropy(TARGETS, reduction)
    assert isinstance(loss, NativeTensor)
    assert type(loss) is NativeTensor
    assert np.isclose(float(loss.to_numpy()), expected, atol=1e-15)
    assert np.isclose(float(loss.to_numpy()),
                      loss_reference(LOGITS, TARGETS, reduction), atol=1e-14)
    loss.close()
    x.close()


@needs_native
def test_native_cross_entropy_output_is_a_scalar_native_tensor():
    x = NativeTensor.from_array(LOGITS, requires_grad=True)
    loss = x.cross_entropy(TARGETS)
    # The repository's scalar convention: shape (), one element, owning,
    # contiguous, float64/cpu — exactly what backward's default seed wants.
    assert loss.shape == ()
    assert loss.numel == 1
    assert loss.ndim == 0
    assert loss.contiguous is True
    assert loss.owns_core is True
    assert loss.dtype == "float64" and loss.device == "cpu"
    loss.close()
    x.close()


@needs_native
def test_native_cross_entropy_mean_is_sum_over_the_batch():
    x = NativeTensor.from_array(LOGITS, requires_grad=True)
    mean = x.cross_entropy(TARGETS, "mean")
    total = x.cross_entropy(TARGETS, "sum")
    assert np.isclose(float(mean.to_numpy()) * LOGITS.shape[0],
                      float(total.to_numpy()), atol=1e-14)
    for t in (mean, total, x):
        t.close()


@needs_native
def test_native_cross_entropy_batch_size_one():
    logits = np.array([[2.0, -1.0]])
    x = NativeTensor.from_array(logits, requires_grad=True)
    loss = x.cross_entropy([0], "mean")
    total = x.cross_entropy([0], "sum")
    # A single example makes "mean" and "sum" identical.
    assert np.isclose(float(loss.to_numpy()), float(total.to_numpy()),
                      atol=1e-15)
    assert np.isclose(float(loss.to_numpy()),
                      loss_reference(logits, [0], "mean"), atol=1e-15)
    for t in (loss, total, x):
        t.close()


@needs_native
def test_native_cross_entropy_many_classes():
    rng = np.random.default_rng(6)
    logits = rng.standard_normal((5, 9))
    targets = [0, 8, 4, 4, 1]
    x = NativeTensor.from_array(logits, requires_grad=True)
    loss = x.cross_entropy(targets, "sum")
    assert np.isclose(float(loss.to_numpy()),
                      loss_reference(logits, targets, "sum"), atol=1e-13)
    loss.close()
    x.close()


@needs_native
@pytest.mark.parametrize("offset", [700.0, -700.0, 1e5, -1e10])
def test_native_cross_entropy_large_finite_offsets_stay_finite(offset):
    """The E5 stability transform is inherited whole: a huge common offset
    shifts nothing, and the naive -log(p[target]) form's overflow never
    appears."""
    logits = LOGITS + offset
    x = NativeTensor.from_array(logits, requires_grad=True)
    loss = x.cross_entropy(TARGETS, "mean")
    value = float(loss.to_numpy())
    assert np.isfinite(value)
    assert np.isclose(value, loss_reference(LOGITS, TARGETS, "mean"),
                      atol=1e-9)
    loss.close()
    x.close()


@needs_native
def test_native_cross_entropy_requires_grad_follows_the_logits():
    tracked = NativeTensor.from_array(LOGITS, requires_grad=True)
    plain = NativeTensor.from_array(LOGITS)
    assert tracked.cross_entropy(TARGETS).requires_grad is True
    assert plain.cross_entropy(TARGETS).requires_grad is False
    tracked.close()
    plain.close()


@needs_native
def test_native_cross_entropy_graph_node_shape_parents_and_ownership():
    x = NativeTensor.from_array(LOGITS, requires_grad=True)
    loss = x.cross_entropy(TARGETS)
    assert loss.is_leaf is False
    assert loss._op == "cross_entropy"
    assert loss._parents == (x,)          # exactly one parent: the logits
    assert loss._backward is not None
    assert loss.owns_core is True
    loss.close()
    x.close()


@needs_native
def test_native_cross_entropy_leaves_the_logits_unchanged():
    values = LOGITS.copy()
    x = NativeTensor.from_array(values, requires_grad=True)
    loss = x.cross_entropy(TARGETS, "sum")
    loss.backward()
    assert np.array_equal(x.to_numpy(), values)
    assert x.closed is False
    loss.close()
    x.close()


@needs_native
def test_native_cross_entropy_returns_only_the_loss():
    """No public saved probabilities, no public target copy, no secondary
    result object: the method returns one scalar NativeTensor."""
    x = NativeTensor.from_array(LOGITS, requires_grad=True)
    loss = x.cross_entropy(TARGETS)
    assert isinstance(loss, NativeTensor)
    assert not isinstance(loss, tuple)
    for absent in ("probabilities", "targets", "saved_probabilities",
                   "reduction", "loss"):
        assert not hasattr(loss, absent), absent
    # The saved state is reachable only through the private graph slot.
    saved = one_saved(loss)
    assert type(saved) is cpp.NativeTensorCore
    assert not isinstance(saved, NativeTensor)
    loss.close()
    x.close()


# ======================================================================
# Forward argument validation — E5's rules, propagated unchanged
# ======================================================================


@needs_native
@pytest.mark.parametrize("targets, error", [
    ([1.0, 2.0], TypeError),                       # floats, even integral
    ([True, False], TypeError),                    # bool
    (np.array([1, 2], dtype=bool), TypeError),     # NumPy bool
    (np.array([1.0, 2.0]), TypeError),             # float array
    ("12", TypeError),                             # a string is not labels
    (1, TypeError),                                # a scalar is not labels
    (np.array([[1], [2]]), ValueError),            # rank 2
    ([1], ValueError),                             # wrong count
    ([1, 3], ValueError),                          # >= num_classes
    ([1, -1], ValueError),                         # negative
])
def test_native_cross_entropy_invalid_targets_propagate(targets, error):
    x = NativeTensor.from_array(LOGITS, requires_grad=True)
    with pytest.raises(error):
        x.cross_entropy(targets)
    assert np.array_equal(x.to_numpy(), LOGITS)   # nothing happened
    assert x.grad is None
    x.close()


@needs_native
@pytest.mark.parametrize("reduction, error", [
    ("none", ValueError), ("Mean", ValueError), ("", ValueError),
    (None, TypeError), (0, TypeError), (True, TypeError),
])
def test_native_cross_entropy_invalid_reduction_propagates(reduction, error):
    x = NativeTensor.from_array(LOGITS, requires_grad=True)
    with pytest.raises(error):
        x.cross_entropy(TARGETS, reduction)
    x.close()


@needs_native
@pytest.mark.parametrize("shape", [(3,), (2, 2, 2), ()])
def test_native_cross_entropy_rank_other_than_two_rejected(shape):
    x = NativeTensor.zeros(shape, requires_grad=True)
    with pytest.raises(ValueError, match="2-D|batch_size"):
        x.cross_entropy([0] * (shape[0] if shape else 1))
    x.close()


@needs_native
def test_native_cross_entropy_closed_logits_rejected():
    x = NativeTensor.from_array(LOGITS, requires_grad=True)
    x.close()
    with pytest.raises(RuntimeError, match="closed"):
        x.cross_entropy(TARGETS)


@needs_native
def test_native_cross_entropy_has_no_axis_argument():
    """The class axis is fixed at dimension 1 by the E5 contract."""
    x = NativeTensor.from_array(LOGITS, requires_grad=True)
    with pytest.raises(TypeError):
        x.cross_entropy(TARGETS, axis=1)
    with pytest.raises(TypeError):
        x.cross_entropy(TARGETS, "mean", 1)
    x.close()


@needs_native
def test_native_cross_entropy_rejects_a_native_tensor_target():
    """Targets are integer host labels, never a native tensor: the runtime
    has no integer dtype and E6 added no overload."""
    x = NativeTensor.from_array(LOGITS, requires_grad=True)
    labels = NativeTensor.from_array(np.array([1.0, 2.0]))
    with pytest.raises((TypeError, ValueError)):
        x.cross_entropy(labels)
    labels.close()
    x.close()


# ======================================================================
# Backward correctness
# ======================================================================


@needs_native
@pytest.mark.parametrize("reduction", REDUCTIONS)
def test_native_cross_entropy_backward_matches_core_and_numpy(reduction):
    _, core_grad = core_backward_reference(LOGITS, TARGETS, reduction)
    x = NativeTensor.from_array(LOGITS, requires_grad=True)
    loss = x.cross_entropy(TARGETS, reduction)
    loss.backward()
    assert np.allclose(x.grad.to_numpy(), core_grad, atol=1e-15)
    assert np.allclose(x.grad.to_numpy(),
                       grad_reference(LOGITS, TARGETS, reduction), atol=1e-15)
    # The gradient is a native tensor with the logits' metadata.
    assert isinstance(x.grad, NativeTensor)
    assert x.grad.shape == LOGITS.shape
    assert x.grad.dtype == "float64" and x.grad.device == "cpu"
    loss.close()
    x.close()


@needs_native
@pytest.mark.parametrize("reduction", REDUCTIONS)
def test_native_cross_entropy_gradient_rows_sum_to_zero(reduction):
    rng = np.random.default_rng(11)
    logits = rng.standard_normal((4, 6))
    targets = [2, 5, 0, 3]
    x = NativeTensor.from_array(logits, requires_grad=True)
    loss = x.cross_entropy(targets, reduction)
    loss.backward()
    assert np.allclose(x.grad.to_numpy().sum(axis=1), 0.0, atol=1e-14)
    loss.close()
    x.close()


@needs_native
def test_native_cross_entropy_mean_gradient_is_sum_gradient_over_batch():
    x = NativeTensor.from_array(LOGITS, requires_grad=True)
    y = NativeTensor.from_array(LOGITS, requires_grad=True)
    x.cross_entropy(TARGETS, "mean").backward()
    y.cross_entropy(TARGETS, "sum").backward()
    assert np.allclose(x.grad.to_numpy(),
                       y.grad.to_numpy() / LOGITS.shape[0], atol=1e-15)
    x.close()
    y.close()


@needs_native
def test_native_cross_entropy_two_classes_hand_checked():
    """A two-class row where the gradient is exactly (p - onehot)."""
    logits = np.array([[0.0, 0.0]])
    x = NativeTensor.from_array(logits, requires_grad=True)
    loss = x.cross_entropy([0], "sum")
    assert np.isclose(float(loss.to_numpy()), np.log(2.0), atol=1e-15)
    loss.backward()
    assert np.allclose(x.grad.to_numpy(), [[-0.5, 0.5]], atol=1e-15)
    loss.close()
    x.close()


@needs_native
@pytest.mark.parametrize("reduction", REDUCTIONS)
@pytest.mark.parametrize("upstream", [1.0, 2.5, -1.5, 0.0])
def test_native_cross_entropy_explicit_upstream_scales_the_gradient(
    reduction, upstream
):
    x = NativeTensor.from_array(LOGITS, requires_grad=True)
    loss = x.cross_entropy(TARGETS, reduction)
    seed = scalar_seed(upstream)
    loss.backward(gradient=seed)
    assert np.allclose(x.grad.to_numpy(),
                       grad_reference(LOGITS, TARGETS, reduction, upstream),
                       atol=1e-15)
    for t in (loss, seed, x):
        t.close()


@needs_native
def test_native_cross_entropy_zero_upstream_gives_exactly_zero():
    x = NativeTensor.from_array(LOGITS, requires_grad=True)
    loss = x.cross_entropy(TARGETS)
    seed = scalar_seed(0.0)
    loss.backward(gradient=seed)
    assert np.array_equal(x.grad.to_numpy(), np.zeros_like(LOGITS))
    for t in (loss, seed, x):
        t.close()


@needs_native
def test_native_cross_entropy_negative_upstream_flips_the_sign():
    x = NativeTensor.from_array(LOGITS, requires_grad=True)
    y = NativeTensor.from_array(LOGITS, requires_grad=True)
    positive = scalar_seed(1.0)
    negative = scalar_seed(-1.0)
    x.cross_entropy(TARGETS).backward(gradient=positive)
    y.cross_entropy(TARGETS).backward(gradient=negative)
    assert np.allclose(x.grad.to_numpy(), -y.grad.to_numpy(), atol=1e-15)
    for t in (positive, negative, x, y):
        t.close()


@needs_native
@pytest.mark.parametrize("reduction", REDUCTIONS)
def test_native_cross_entropy_finite_differences(reduction):
    """Central differences over moderate finite logits — no exceptional
    values, so the comparison measures the gradient rule and not IEEE
    edge behavior."""
    rng = np.random.default_rng(21)
    logits = rng.standard_normal((3, 4)) * 0.8
    targets = [1, 3, 0]
    x = NativeTensor.from_array(logits, requires_grad=True)
    loss = x.cross_entropy(targets, reduction)
    loss.backward()
    analytic = x.grad.to_numpy()

    step = 1e-6
    numeric = np.zeros_like(logits)
    for row in range(logits.shape[0]):
        for column in range(logits.shape[1]):
            plus = logits.copy()
            plus[row, column] += step
            minus = logits.copy()
            minus[row, column] -= step
            numeric[row, column] = (
                loss_reference(plus, targets, reduction)
                - loss_reference(minus, targets, reduction)
            ) / (2 * step)
    assert np.allclose(analytic, numeric, atol=1e-7)
    loss.close()
    x.close()


@needs_native
def test_native_cross_entropy_chained_graph_differentiates():
    """cross_entropy composed with another differentiable op: the scalar
    loss keeps flowing through the ordinary engine."""
    x = NativeTensor.from_array(LOGITS, requires_grad=True)
    scale = NativeTensor.from_array(np.array(3.0).reshape(1, 1))
    scaled = x.multiply(scale)
    loss = scaled.cross_entropy(TARGETS, "sum")
    loss.backward()
    expected = 3.0 * grad_reference(3.0 * LOGITS, TARGETS, "sum")
    assert np.allclose(x.grad.to_numpy(), expected, atol=1e-13)
    for t in (loss, scaled, scale, x):
        t.close()


@needs_native
def test_native_cross_entropy_shared_subgraph_accumulates():
    """Two cross-entropy branches over the same logits: both contributions
    land on the one parent."""
    x = NativeTensor.from_array(LOGITS, requires_grad=True)
    first = x.cross_entropy(TARGETS, "mean")
    second = x.cross_entropy([0, 0], "mean")
    total = first.add(second)
    total.backward()
    expected = (grad_reference(LOGITS, TARGETS, "mean")
                + grad_reference(LOGITS, [0, 0], "mean"))
    assert np.allclose(x.grad.to_numpy(), expected, atol=1e-14)
    for t in (total, first, second, x):
        t.close()


@needs_native
def test_native_cross_entropy_repeated_backward_accumulates_and_zero_grad_clears():
    x = NativeTensor.from_array(LOGITS, requires_grad=True)
    once = grad_reference(LOGITS, TARGETS, "mean")
    x.cross_entropy(TARGETS).backward()
    assert np.allclose(x.grad.to_numpy(), once, atol=1e-15)
    # A *fresh* forward is a fresh graph; leaf gradients keep summing.
    x.cross_entropy(TARGETS).backward()
    assert np.allclose(x.grad.to_numpy(), 2 * once, atol=1e-14)
    x.zero_grad()
    assert x.grad is None
    x.close()


# ======================================================================
# Scalar upstream handling
# ======================================================================


@needs_native
def test_native_cross_entropy_implicit_seed_equals_an_explicit_one():
    x = NativeTensor.from_array(LOGITS, requires_grad=True)
    y = NativeTensor.from_array(LOGITS, requires_grad=True)
    x.cross_entropy(TARGETS).backward()                    # implicit
    seed = scalar_seed(1.0)
    y.cross_entropy(TARGETS).backward(gradient=seed)       # explicit rank-0
    assert np.array_equal(x.grad.to_numpy(), y.grad.to_numpy())
    for t in (seed, x, y):
        t.close()


@needs_native
@pytest.mark.parametrize("shape", [(1,), (1, 1), (2, 3)])
def test_native_cross_entropy_non_scalar_upstream_rejected(shape):
    """The engine's general seed contract: an explicit gradient must match
    the output's shape exactly, which for this loss is ``()``. A
    one-element ``(1,)`` is still the wrong shape and is refused before
    anything is committed."""
    x = NativeTensor.from_array(LOGITS, requires_grad=True)
    loss = x.cross_entropy(TARGETS)
    bad = NativeTensor.zeros(shape)
    with pytest.raises(ValueError, match="shape"):
        loss.backward(gradient=bad)
    assert x.grad is None
    assert loss._graph_freed is False
    assert not one_saved(loss)._closed
    for t in (bad, loss, x):
        t.close()


@needs_native
def test_native_cross_entropy_wrong_type_upstream_rejected():
    x = NativeTensor.from_array(LOGITS, requires_grad=True)
    loss = x.cross_entropy(TARGETS)
    for bad in (1.0, np.array(1.0), [1.0], tensorforge.Tensor(1.0)):
        with pytest.raises(TypeError):
            loss.backward(gradient=bad)
    assert x.grad is None
    loss.close()
    x.close()


@needs_native
def test_native_cross_entropy_closed_upstream_rejected():
    x = NativeTensor.from_array(LOGITS, requires_grad=True)
    loss = x.cross_entropy(TARGETS)
    seed = scalar_seed(1.0)
    seed.close()
    with pytest.raises(RuntimeError, match="closed"):
        loss.backward(gradient=seed)
    assert x.grad is None
    assert not one_saved(loss)._closed     # nothing released by the failure
    loss.close()
    x.close()


# ======================================================================
# Non-contiguous logits (E5 Policy B, inherited)
# ======================================================================


@needs_native
def test_native_cross_entropy_transposed_logits():
    base = NativeTensor.from_array(LOGITS.T, requires_grad=True)
    strided = base.T
    assert strided.contiguous is False
    loss = strided.cross_entropy(TARGETS, "sum")
    assert np.isclose(float(loss.to_numpy()),
                      loss_reference(LOGITS, TARGETS, "sum"), atol=1e-14)
    loss.backward()
    # The gradient lands on the *base* leaf, transposed back by the
    # existing transpose backward.
    assert np.allclose(base.grad.to_numpy(),
                       grad_reference(LOGITS, TARGETS, "sum").T, atol=1e-14)
    # The view's own data was never mutated.
    assert np.array_equal(base.to_numpy(), LOGITS.T)
    for t in (loss, strided, base):
        t.close()


@needs_native
def test_native_cross_entropy_narrowed_nonzero_offset_logits():
    values = np.arange(12, dtype=float).reshape(4, 3) / 10.0
    base = NativeTensor.from_array(values, requires_grad=True)
    window = base.narrow(0, 1, 2)          # rows 1..2, nonzero offset
    assert window.contiguous is True
    loss = window.cross_entropy(TARGETS, "mean")
    assert np.isclose(float(loss.to_numpy()),
                      loss_reference(values[1:3], TARGETS, "mean"), atol=1e-14)
    loss.backward()
    expected = np.zeros_like(values)
    expected[1:3] = grad_reference(values[1:3], TARGETS, "mean")
    assert np.allclose(base.grad.to_numpy(), expected, atol=1e-15)
    for t in (loss, window, base):
        t.close()


@needs_native
def test_native_cross_entropy_transpose_of_a_narrow():
    values = np.arange(12, dtype=float).reshape(3, 4) / 7.0
    base = NativeTensor.from_array(values, requires_grad=True)
    window = base.narrow(1, 1, 2)          # (3, 2), strided
    strided = window.T                     # (2, 3), strided and offset
    assert strided.contiguous is False
    logits = values[:, 1:3].T
    loss = strided.cross_entropy([0, 2], "sum")
    assert np.isclose(float(loss.to_numpy()),
                      loss_reference(logits, [0, 2], "sum"), atol=1e-14)
    loss.backward()
    expected = np.zeros_like(values)
    expected[:, 1:3] = grad_reference(logits, [0, 2], "sum").T
    assert np.allclose(base.grad.to_numpy(), expected, atol=1e-14)
    for t in (loss, strided, window, base):
        t.close()


@needs_native
def test_native_cross_entropy_saved_probabilities_are_contiguous_and_owning():
    """Policy B materializes the logits for the kernel, but the saved
    probabilities are always fresh contiguous owning storage aliasing
    nothing — including on the strided path."""
    base = NativeTensor.from_array(LOGITS.T, requires_grad=True)
    strided = base.T
    loss = strided.cross_entropy(TARGETS)
    saved = one_saved(loss)
    assert saved.shape == LOGITS.shape
    assert saved.contiguous is True
    assert saved.offset == 0
    assert np.allclose(saved.to_numpy(), probabilities_reference(LOGITS),
                       atol=1e-15)
    assert saved._storage is not base._core._storage
    assert saved._storage is not loss._core._storage
    for t in (loss, strided, base):
        t.close()


@needs_native
def test_native_cross_entropy_policy_b_temporary_is_released(live_storages):
    base = NativeTensor.from_array(LOGITS.T, requires_grad=True)
    baseline = len(live_storages)
    loss = base.T.cross_entropy(TARGETS)
    # Exactly two new allocations survive the call: the scalar loss and the
    # saved probabilities. The Policy-B contiguous copy was closed inside
    # the Core forward's finally.
    assert len(live_storages) == baseline + 2
    loss.backward()
    base.close()
    loss.close()


# ======================================================================
# Saved-probability graph resource: lifetime
# ======================================================================


@needs_native
def test_native_cross_entropy_owns_exactly_one_saved_probability_resource():
    x = NativeTensor.from_array(LOGITS, requires_grad=True)
    loss = x.cross_entropy(TARGETS)
    saved = one_saved(loss)
    assert type(saved) is cpp.NativeTensorCore
    assert saved._closed is False
    assert saved.shape == LOGITS.shape
    assert np.allclose(saved.to_numpy(), probabilities_reference(LOGITS),
                       atol=1e-15)
    loss.close()
    x.close()


@needs_native
def test_native_cross_entropy_no_grad_forward_closes_the_probabilities():
    """Directly instrumented, not inferred from an empty tuple: the
    probability core the E5 forward really produced must be closed exactly
    once by the time the no-grad forward returns."""
    produced = []
    original = cpp.NativeTensorCore.cross_entropy_forward

    def capturing(self, targets, reduction="mean"):
        result = original(self, targets, reduction)
        produced.append(result.probabilities)
        return result

    cpp.NativeTensorCore.cross_entropy_forward = capturing
    try:
        x = NativeTensor.from_array(LOGITS)          # requires_grad False
        loss = x.cross_entropy(TARGETS)
    finally:
        cpp.NativeTensorCore.cross_entropy_forward = original

    assert len(produced) == 1
    saved = produced[0]
    assert saved._closed is True, "no-grad forward leaked the probabilities"
    # ...and no graph survived it either.
    assert loss.requires_grad is False
    assert loss.is_leaf is True
    assert loss._parents == ()
    assert loss._backward is None
    assert loss._graph_resources == ()
    assert loss._expected_versions == ()
    # The scalar loss itself is untouched, valid, and closeable normally.
    assert loss.closed is False
    assert np.isclose(float(loss.to_numpy()),
                      loss_reference(LOGITS, TARGETS, "mean"), atol=1e-14)
    loss.close()
    assert loss.closed is True
    loss.close()                                     # idempotent
    x.close()


@needs_native
def test_native_cross_entropy_one_shot_backward_releases_the_probabilities():
    x = NativeParameter(LOGITS)
    loss = x.cross_entropy(TARGETS)
    saved = one_saved(loss)
    assert saved._closed is False                    # alive through forward
    loss.backward()
    assert loss._graph_freed is True
    assert loss._graph_resources == ()
    assert saved._closed is True                     # released with history
    loss.close()
    x.close()


@needs_native
def test_native_cross_entropy_retain_graph_keeps_the_probabilities():
    x = NativeParameter(LOGITS)
    loss = x.cross_entropy(TARGETS)
    saved = one_saved(loss)
    seed = scalar_seed(1.0)
    loss.backward(gradient=seed, retain_graph=True)
    assert saved._closed is False                    # still alive
    assert loss._graph_freed is False
    once = x.grad.to_numpy().copy()
    loss.backward(gradient=seed)                     # final, one-shot pass
    assert np.allclose(x.grad.to_numpy(), 2 * once, atol=1e-15)
    assert saved._closed is True                     # released now
    assert loss._graph_resources == ()
    for t in (loss, seed, x):
        t.close()


@needs_native
def test_native_cross_entropy_abandoned_graph_close_releases_the_probabilities():
    x = NativeTensor.from_array(LOGITS, requires_grad=True)
    loss = x.cross_entropy(TARGETS)
    saved = one_saved(loss)
    assert saved._closed is False
    loss.close()                                     # never ran backward
    assert saved._closed is True
    loss.close()                                     # idempotent, no double
    assert saved._closed is True
    x.close()


@needs_native
def test_native_cross_entropy_dropped_graph_does_not_leak_the_probabilities():
    """The ``__del__`` refcount/GC *fallback* — not a deterministic release
    point (those are the one-shot backward and close(), covered above);
    this only proves the safety net also frees the buffer."""
    x = NativeTensor.from_array(LOGITS, requires_grad=True)
    holder = []

    def build():
        loss = x.cross_entropy(TARGETS)
        holder.append(one_saved(loss))
        # loss goes out of scope without close() or backward()

    build()
    gc.collect()
    assert holder[0]._closed is True
    x.close()


@needs_native
def test_native_cross_entropy_repeated_backward_after_free_does_not_double_close():
    x = NativeParameter(LOGITS)
    loss = x.cross_entropy(TARGETS)
    saved = one_saved(loss)
    loss.backward()
    assert saved._closed is True
    with pytest.raises(RuntimeError, match="freed autograd graph"):
        loss.backward()
    assert saved._closed is True                     # closed exactly once
    loss.close()
    assert saved._closed is True
    x.close()


@needs_native
def test_native_cross_entropy_probabilities_never_become_public_state(
    live_storages
):
    """The saved core is private graph state: no public attribute, no
    module parameter or buffer, no state dict, no checkpoint — and it is
    accounted for exactly, not leaked."""
    from tensorforge.experimental import NativeLinear

    baseline = len(live_storages)
    x = NativeTensor.from_array(LOGITS, requires_grad=True)
    loss = x.cross_entropy(TARGETS)
    saved = one_saved(loss)
    assert saved not in [getattr(loss, name, None) for name in dir(loss)]
    model = NativeLinear(3, 2, seed=0)
    state = model.state_dict()
    for key, value in state.items():
        assert "probabilit" not in key and "cross_entropy" not in key
        assert value._core is not saved
        value.close()
    for parameter in model.parameters():
        assert parameter._core is not saved
        parameter.close()
    loss.backward()
    assert saved._closed is True
    gradient = x.grad
    for t in (loss, x, gradient):
        t.close()
    # Every native allocation this test made is accounted for: the saved
    # probabilities were released with the graph, not leaked into anything
    # public.
    assert len(live_storages) == baseline


# ======================================================================
# Target-copy lifetime and mutation immunity
# ======================================================================


def _closure_targets(output):
    """The int64 target copy captured by this node's backward closure."""
    cells = output._backward.__closure__
    names = output._backward.__code__.co_freevars
    return dict(zip(names, (cell.cell_contents for cell in cells)))[
        "saved_targets"
    ]


@needs_native
def test_native_cross_entropy_captured_targets_are_owned_int64_metadata():
    caller = np.array(TARGETS, dtype=np.int64)
    x = NativeTensor.from_array(LOGITS, requires_grad=True)
    loss = x.cross_entropy(caller)
    captured = _closure_targets(loss)
    assert isinstance(captured, np.ndarray)
    assert captured.dtype == np.int64
    assert captured.ndim == 1 and captured.size == LOGITS.shape[0]
    assert captured.flags["C_CONTIGUOUS"]
    assert captured.flags["OWNDATA"]
    assert captured.flags["WRITEABLE"] is False       # E5 marks it read-only
    # A copy was taken even though the caller already passed contiguous
    # int64 data: no view into caller memory is retained.
    assert not np.shares_memory(captured, caller)
    loss.close()
    x.close()


@needs_native
@pytest.mark.parametrize("as_array", [False, True])
def test_native_cross_entropy_caller_target_mutation_cannot_affect_backward(
    as_array
):
    caller = np.array(TARGETS, dtype=np.int64) if as_array else list(TARGETS)
    x = NativeTensor.from_array(LOGITS, requires_grad=True)
    loss = x.cross_entropy(caller, "mean")
    caller[0] = 0                                     # mutate after forward
    caller[1] = 0
    loss.backward()
    original = grad_reference(LOGITS, TARGETS, "mean")
    mutated = grad_reference(LOGITS, [0, 0], "mean")
    assert not np.allclose(original, mutated)         # a real difference
    assert np.allclose(x.grad.to_numpy(), original, atol=1e-15)
    loss.close()
    x.close()


@needs_native
def test_native_cross_entropy_targets_survive_a_retained_graph_and_are_released():
    x = NativeParameter(LOGITS)
    loss = x.cross_entropy(TARGETS)
    seed = scalar_seed(1.0)
    loss.backward(gradient=seed, retain_graph=True)
    # Still captured for another pass.
    assert _closure_targets(loss).tolist() == TARGETS
    loss.backward(gradient=seed)
    # The history is gone, and the closure holding the targets with it, so
    # the metadata is collectible. (It is never a graph_resource: that
    # collection is only for closeable *native* objects.)
    assert loss._backward is None
    assert loss._graph_resources == ()
    for t in (loss, seed, x):
        t.close()


@needs_native
def test_native_cross_entropy_targets_are_never_a_native_resource_or_state():
    x = NativeTensor.from_array(LOGITS, requires_grad=True)
    loss = x.cross_entropy(TARGETS)
    for resource in loss._graph_resources:
        assert not isinstance(resource, np.ndarray)
        assert type(resource) is cpp.NativeTensorCore
    assert not hasattr(loss, "targets")
    loss.close()
    x.close()


# ======================================================================
# Versioning contract: backward never rereads the logits
# ======================================================================


@needs_native
def test_native_cross_entropy_records_no_expected_versions():
    x = NativeParameter(LOGITS)
    loss = x.cross_entropy(TARGETS)
    assert loss._expected_versions == ()
    # ...unlike `log`, whose backward rereads the live input and therefore
    # does record one for a direct NativeParameter parent.
    w = NativeParameter(np.abs(LOGITS) + 1.0)
    logged = w.log()
    assert logged._expected_versions != ()
    for t in (loss, logged, w, x):
        t.close()


@needs_native
def test_native_cross_entropy_direct_parameter_mutation_after_forward():
    """The maxpool2d archetype: the gradient belongs to the forward that
    ran. Mutating the logits parameter afterwards must neither raise nor
    change the gradient, because backward reads the saved probabilities
    and not the parameter's current value."""
    original = LOGITS.copy()
    mutated_values = np.array([[-3.0, 0.5, 4.0], [2.0, -2.0, 0.25]])
    assert not np.allclose(original, mutated_values)

    x = NativeParameter(original)
    loss = x.cross_entropy(TARGETS, "mean")
    _, expected = core_backward_reference(original, TARGETS, "mean")

    replacement = NativeTensor.from_array(mutated_values)
    x.copy_value_(replacement)                        # sanctioned mutation
    assert x._version == 1

    loss.backward()                                   # succeeds: no version
    assert np.allclose(x.grad.to_numpy(), expected, atol=1e-15)
    # The gradient is the *original* forward's, and it is genuinely
    # different from the mutated logits' gradient.
    fresh_expected = grad_reference(mutated_values, TARGETS, "mean")
    assert not np.allclose(expected, fresh_expected, atol=1e-3)
    assert not np.allclose(x.grad.to_numpy(), fresh_expected, atol=1e-3)

    # A fresh forward after the mutation uses the new values.
    x.zero_grad()
    again = x.cross_entropy(TARGETS, "mean")
    assert np.isclose(float(again.to_numpy()),
                      loss_reference(mutated_values, TARGETS, "mean"),
                      atol=1e-14)
    again.backward()
    assert np.allclose(x.grad.to_numpy(), fresh_expected, atol=1e-15)
    for t in (loss, again, replacement, x):
        t.close()


@needs_native
def test_native_cross_entropy_retained_graph_survives_parameter_mutation():
    """Deliberately different from `log`'s live-input behavior: a retained
    cross-entropy graph keeps differentiating against the probabilities
    its own forward saved, even after the parameter changes."""
    original = LOGITS.copy()
    x = NativeParameter(original)
    loss = x.cross_entropy(TARGETS, "mean")
    seed = scalar_seed(1.0)
    loss.backward(gradient=seed, retain_graph=True)
    once = x.grad.to_numpy().copy()
    assert np.allclose(once, grad_reference(original, TARGETS, "mean"),
                       atol=1e-15)

    replacement = NativeTensor.from_array(
        np.array([[-3.0, 0.5, 4.0], [2.0, -2.0, 0.25]])
    )
    x.copy_value_(replacement)

    loss.backward(gradient=seed, retain_graph=True)   # second pass, retained
    assert np.allclose(x.grad.to_numpy(), 2 * once, atol=1e-15)
    assert not one_saved(loss)._closed                # still retained
    loss.backward(gradient=seed)                      # final, releases it
    assert np.allclose(x.grad.to_numpy(), 3 * once, atol=1e-15)
    assert loss._graph_resources == ()
    for t in (loss, seed, replacement, x):
        t.close()


# ======================================================================
# Parent lifetime, closed saved state, and mixed-graph rollback
# ======================================================================


@needs_native
def test_native_cross_entropy_closed_parent_fails_at_accumulation(
    live_storages
):
    """Backward does not reread the logits — but ``_accumulate_grad``
    still requires the parent tensor to be **open**, because that is where
    the gradient is stored. The failure is therefore about the parent's
    lifetime, not about reading its value: the Core backward has already
    produced a correct gradient by the time it raises. No version snapshot
    would help, and none is recorded."""
    x = NativeTensor.from_array(LOGITS, requires_grad=True)
    loss = x.cross_entropy(TARGETS)
    saved = one_saved(loss)
    baseline = len(live_storages)
    x.close()                                         # after forward

    with pytest.raises(RuntimeError, match="closed"):
        loss.backward()

    # Nothing committed, nothing freed, the saved probabilities intact:
    # this failure is retryable in the engine's sense.
    assert x._grad is None
    assert loss._graph_freed is False
    assert saved._closed is False
    # The gradient the Core backward produced was released, not leaked:
    # only the seed's storage (still referenced by the failed pass) may
    # differ, so compare against the count with the seed subtracted.
    assert len(live_storages) <= baseline + 1
    loss.close()
    assert saved._closed is True


@needs_native
def test_native_cross_entropy_unadopted_contribution_is_closed_explicitly(
    monkeypatch
):
    """``_accumulate_grad`` adopts a contribution only on the assignment
    that ends it, so a gradient produced for a closed parent is never
    adopted. This test holds a **strong reference** to every NativeTensor
    the backward builds, which disables the ``__del__`` safety net — so if
    the contribution is closed, it was closed deliberately."""
    produced = []
    original = NativeTensor._from_core.__func__

    def recording(cls, core, owns_core=True):
        tensor = original(cls, core, owns_core=owns_core)
        produced.append(tensor)          # strong ref: __del__ cannot fire
        return tensor

    x = NativeTensor.from_array(LOGITS, requires_grad=True)
    loss = x.cross_entropy(TARGETS)
    monkeypatch.setattr(NativeTensor, "_from_core", classmethod(recording))
    x.close()                            # the parent, after forward

    with pytest.raises(RuntimeError, match="closed"):
        loss.backward()
    monkeypatch.undo()

    # The seed came first, then the gradient contribution the Core backward
    # really produced. The seed is still the caller's; the contribution was
    # released rather than leaked.
    assert len(produced) == 2, produced
    assert produced[0].closed is False, "the seed was closed by mistake"
    assert produced[-1].closed is True, "the unadopted contribution leaked"
    produced[0].close()
    loss.close()


@needs_native
def test_native_cross_entropy_manually_closed_saved_resource_fails_atomically():
    """An internal-invariant test: corrupt the private saved state and the
    backward must fail clearly, commit nothing, keep the graph honestly
    un-freed, fail identically on retry, and not double-close at cleanup."""
    x = NativeParameter(LOGITS)
    loss = x.cross_entropy(TARGETS)
    seed = scalar_seed(1.0)
    loss.backward(gradient=seed, retain_graph=True)   # a real gradient first
    before = x.grad.to_numpy().copy()

    saved = one_saved(loss)
    saved.close()                                     # corrupt the state

    with pytest.raises(RuntimeError, match="closed"):
        loss.backward(gradient=seed, retain_graph=True)
    assert np.array_equal(x.grad.to_numpy(), before)  # rolled back
    assert loss._graph_freed is False                 # not falsely freed
    assert loss._graph_resources == (saved,)          # still owned

    # Deterministic: the retry fails identically while the state is closed.
    with pytest.raises(RuntimeError, match="closed"):
        loss.backward(gradient=seed, retain_graph=True)
    assert np.array_equal(x.grad.to_numpy(), before)

    # Final cleanup releases the (already closed) resource exactly once
    # more, harmlessly — close() is idempotent at the core level.
    loss.close()
    assert saved._closed is True
    for t in (seed, x):
        t.close()


@needs_native
def test_native_cross_entropy_mixed_stale_graph_commits_nothing():
    """A graph with a healthy cross-entropy branch and a stale
    live-value-reading branch (`log`): preflight raises before any
    callback, so cross-entropy's callback never commits and the saved
    probabilities stay owned and retryable."""
    logits = LOGITS.copy()
    weights = np.array([[1.5, 2.5], [3.5, 4.5]])
    x = NativeParameter(logits)
    w = NativeParameter(weights)

    ce = x.cross_entropy(TARGETS, "mean")
    stale = w.log().sum()                             # records w's version
    total = ce.add(stale)
    seed = scalar_seed(1.0)
    total.backward(gradient=seed, retain_graph=True)  # a healthy pass first
    before_x = x.grad.to_numpy().copy()
    before_w = w.grad.to_numpy().copy()
    saved = one_saved(ce)

    replacement = NativeTensor.from_array(weights * 2.0)
    w.copy_value_(replacement)                        # makes `log` stale

    with pytest.raises(RuntimeError, match="stale parameter value"):
        total.backward(gradient=seed, retain_graph=True)

    # Nothing committed anywhere, nothing freed, nothing released.
    assert np.array_equal(x.grad.to_numpy(), before_x)
    assert np.array_equal(w.grad.to_numpy(), before_w)
    assert total._graph_freed is False and ce._graph_freed is False
    assert saved._closed is False
    assert ce._graph_resources == (saved,)            # still owned, retryable

    # Deterministic: the same stale error repeats, still committing nothing.
    with pytest.raises(RuntimeError, match="stale parameter value"):
        total.backward(gradient=seed, retain_graph=True)
    assert np.array_equal(x.grad.to_numpy(), before_x)
    assert saved._closed is False
    for t in (total, ce, stale, seed, replacement, x, w):
        t.close()


@needs_native
def test_native_cross_entropy_preexisting_gradients_survive_a_failed_pass():
    """Gradients that existed *before* a failing pass — including on
    tensors the failing branch never reaches — are restored exactly."""
    x = NativeParameter(LOGITS)
    other = NativeParameter(np.array([[2.0, 1.0], [0.5, 3.0]]))
    # Seed a gradient on `other` through an entirely separate graph.
    other.multiply(other).sum().backward()
    other_before = other.grad.to_numpy().copy()

    loss = x.cross_entropy(TARGETS)
    seed = scalar_seed(1.0)
    loss.backward(gradient=seed, retain_graph=True)
    x_before = x.grad.to_numpy().copy()

    # Now break the pass by closing the saved probabilities.
    one_saved(loss).close()
    with pytest.raises(RuntimeError):
        loss.backward(gradient=seed, retain_graph=True)

    assert np.array_equal(x.grad.to_numpy(), x_before)
    assert np.array_equal(other.grad.to_numpy(), other_before)
    for t in (loss, seed, x, other):
        t.close()


@needs_native
@needs_fault_injection
def test_native_cross_entropy_healthy_sibling_branch_commits_nothing():
    """Two branches over the same parameter; the second callback fails.
    The engine's snapshot rollback must undo the first branch's committed
    contribution too — no branch commits independently."""
    x = NativeParameter(LOGITS)
    first = x.cross_entropy(TARGETS, "mean")
    second = x.cross_entropy([0, 0], "sum")
    total = first.add(second)
    seed = scalar_seed(1.0)
    total.backward(gradient=seed, retain_graph=True)
    before = x.grad.to_numpy().copy()

    # Some allocation in the pass fails; whichever it is, the whole pass
    # must roll back to the pre-existing gradient — including the branch
    # that had already committed before the failure.
    failures = 0
    for nth in range(1, 7):
        # Restore a deterministic pre-existing gradient before arming, so
        # each iteration starts from exactly the same committed state.
        x.zero_grad()
        restored = NativeTensor.from_array(before)
        x._grad = restored
        cpp._arm_alloc_failure(nth)
        try:
            total.backward(gradient=seed, retain_graph=True)
            failed = False
        except MemoryError:
            failed = True
        finally:
            # Disarm before any assertion: reading a gradient allocates.
            cpp._arm_alloc_failure(0)
            cpp._require_library().tf_clear_error()
        if failed:
            failures += 1
            assert np.array_equal(x.grad.to_numpy(), before), nth
            assert x.grad is restored, nth      # the exact prior reference
            assert total._graph_freed is False, nth
            assert not one_saved(first)._closed, nth
            assert not one_saved(second)._closed, nth
        else:
            # A pass that survived the injection committed both branches
            # in full, never a partial contribution.
            assert np.allclose(x.grad.to_numpy(), 2 * before, atol=1e-14), nth
    assert failures >= 1, "no injected failure was reachable in this pass"

    # The graph is still usable end to end, from a clean gradient.
    x.zero_grad()
    total.backward(gradient=seed)
    assert np.allclose(x.grad.to_numpy(), before, atol=1e-14)
    for t in (total, first, second, seed, x):
        t.close()


# ======================================================================
# Failure atomicity
# ======================================================================


@needs_native
@needs_fault_injection
def test_native_cross_entropy_forward_allocation_failure_escapes_nothing(
    live_storages
):
    """An E5 forward that fails mid-allocation must leave no output, no
    graph node, and no live storage behind."""
    x = NativeTensor.from_array(LOGITS, requires_grad=True)
    baseline = len(live_storages)
    reached = 0
    for nth in range(1, 4):
        cpp._arm_alloc_failure(nth)
        try:
            x.cross_entropy(TARGETS)
        except MemoryError:
            reached += 1
            assert len(live_storages) == baseline, nth
            assert x.closed is False
            assert np.array_equal(x.to_numpy(), LOGITS)
            assert x.grad is None
        finally:
            cpp._arm_alloc_failure(0)
            cpp._require_library().tf_clear_error()
    assert reached >= 2, "expected at least two reachable forward allocations"

    # Disarmed, the same call works.
    loss = x.cross_entropy(TARGETS)
    loss.backward()
    assert np.allclose(x.grad.to_numpy(),
                       grad_reference(LOGITS, TARGETS, "mean"), atol=1e-15)
    loss.close()
    x.close()
    assert len(live_storages) == baseline


@needs_native
def test_native_cross_entropy_validation_failure_builds_no_graph(live_storages):
    x = NativeTensor.from_array(LOGITS, requires_grad=True)
    baseline = len(live_storages)
    for bad_targets, bad_reduction in (([1, 3], "mean"), (TARGETS, "none")):
        with pytest.raises(ValueError):
            x.cross_entropy(bad_targets, bad_reduction)
        assert len(live_storages) == baseline       # nothing was allocated
    x.close()


@needs_native
def test_native_cross_entropy_graph_construction_failure_closes_both_outputs(
    monkeypatch, live_storages
):
    """The E5 forward succeeds (scalar loss + probabilities + target copy),
    then graph construction fails. Nothing has adopted either core yet, so
    cleanup must be explicit and deterministic here — ``__del__`` is only
    the fallback and this test must not depend on it."""
    x = NativeTensor.from_array(LOGITS, requires_grad=True)
    baseline = len(live_storages)

    produced = []
    original_forward = cpp.NativeTensorCore.cross_entropy_forward

    def capturing_forward(self, targets, reduction="mean"):
        result = original_forward(self, targets, reduction)
        produced.append(result)
        return result

    monkeypatch.setattr(cpp.NativeTensorCore, "cross_entropy_forward",
                        capturing_forward)

    def exploding_from_op(cls, *args, **kwargs):
        raise RuntimeError("simulated graph-construction failure")

    monkeypatch.setattr(NativeTensor, "_from_op",
                        classmethod(exploding_from_op))

    with pytest.raises(RuntimeError, match="simulated graph-construction"):
        x.cross_entropy(TARGETS)

    # The forward really ran and really allocated both cores.
    assert len(produced) == 1
    result = produced[0]
    assert result.loss._closed is True, "the scalar loss leaked"
    assert result.probabilities._closed is True, "the probabilities leaked"
    # Closed exactly once each: the live count is already back at baseline
    # before the redundant close below, which is a harmless no-op.
    assert len(live_storages) == baseline
    result.close()
    assert len(live_storages) == baseline
    # The original exception, not a cleanup error, reached the caller — and
    # no partial tensor escaped.
    assert x.closed is False
    assert np.array_equal(x.to_numpy(), LOGITS)
    assert x.grad is None

    monkeypatch.undo()
    loss = x.cross_entropy(TARGETS)
    loss.backward()
    assert np.allclose(x.grad.to_numpy(),
                       grad_reference(LOGITS, TARGETS, "mean"), atol=1e-15)
    loss.close()
    x.close()


@needs_native
@needs_fault_injection
def test_native_cross_entropy_backward_allocation_failure_rolls_back_and_retries(
    live_storages
):
    x = NativeParameter(LOGITS)
    seed = scalar_seed(1.0)
    loss = x.cross_entropy(TARGETS)
    loss.backward(gradient=seed, retain_graph=True)
    before = x.grad.to_numpy().copy()
    saved = one_saved(loss)
    baseline = len(live_storages)

    # With an explicit seed and contiguous saved probabilities, the first
    # allocation of the pass is the gradient output itself.
    cpp._arm_alloc_failure(1)
    with pytest.raises(MemoryError):
        loss.backward(gradient=seed, retain_graph=True)
    assert np.array_equal(x.grad.to_numpy(), before)   # rolled back
    assert loss._graph_freed is False                  # not falsely freed
    assert not saved._closed                           # retained for retry
    assert len(live_storages) == baseline              # no gradient leaked
    assert cpp._require_library().tf_last_error_code() == cpp.TF_OK

    # Retry after disarming: the same graph works and accumulates.
    loss.backward(gradient=seed)
    assert np.allclose(x.grad.to_numpy(), 2 * before, atol=1e-15)
    assert saved._closed is True
    for t in (loss, seed, x):
        t.close()


@needs_native
@needs_fault_injection
def test_native_cross_entropy_backward_accumulation_failure_leaks_nothing(
    live_storages
):
    """A *second* contribution goes through the native ``add``; failing
    that allocation must leave the existing gradient untouched and release
    the contribution rather than leaking it."""
    x = NativeParameter(LOGITS)
    seed = scalar_seed(1.0)
    loss = x.cross_entropy(TARGETS)
    loss.backward(gradient=seed, retain_graph=True)     # x.grad now exists
    before = x.grad.to_numpy().copy()
    baseline = len(live_storages)

    # Allocation 1 is the gradient core, allocation 2 the accumulating add.
    cpp._arm_alloc_failure(2)
    with pytest.raises(MemoryError):
        loss.backward(gradient=seed, retain_graph=True)
    assert np.array_equal(x.grad.to_numpy(), before)
    assert not one_saved(loss)._closed
    assert len(live_storages) == baseline, "the contribution leaked"

    cpp._arm_alloc_failure(0)
    cpp._require_library().tf_clear_error()
    loss.backward(gradient=seed)
    assert np.allclose(x.grad.to_numpy(), 2 * before, atol=1e-15)
    for t in (loss, seed, x):
        t.close()


@needs_native
def test_native_cross_entropy_backward_core_failure_leaves_the_graph_retryable(
    monkeypatch
):
    """A native Core backward failure (here forced through the checked Core
    method) commits nothing and keeps the saved probabilities."""
    x = NativeParameter(LOGITS)
    seed = scalar_seed(1.0)
    loss = x.cross_entropy(TARGETS)
    loss.backward(gradient=seed, retain_graph=True)
    before = x.grad.to_numpy().copy()
    saved = one_saved(loss)

    original = cpp.NativeTensorCore.cross_entropy_backward

    def exploding(self, targets, upstream, reduction="mean"):
        raise ValueError("simulated native backward failure")

    monkeypatch.setattr(cpp.NativeTensorCore, "cross_entropy_backward",
                        exploding)
    with pytest.raises(ValueError, match="simulated native backward"):
        loss.backward(gradient=seed, retain_graph=True)
    assert np.array_equal(x.grad.to_numpy(), before)
    assert loss._graph_freed is False
    assert not saved._closed
    monkeypatch.setattr(cpp.NativeTensorCore, "cross_entropy_backward",
                        original)

    loss.backward(gradient=seed)
    assert np.allclose(x.grad.to_numpy(), 2 * before, atol=1e-15)
    assert saved._closed is True
    for t in (loss, seed, x):
        t.close()


# ======================================================================
# NumPy tripwires
# ======================================================================


_NUMERICAL_NUMPY = (
    "max", "amax", "argmax", "exp", "log", "logaddexp", "sum", "divide",
    "true_divide", "add", "subtract", "multiply", "matmul", "mean",
    "negative", "power", "copyto", "take", "take_along_axis", "put",
    "put_along_axis", "where", "choose",
)
# Every route by which tensor *data* could enter or leave a NumPy host
# buffer. (np.array/np.asarray are deliberately absent: the int64 target
# copy legitimately uses them, and the instrumented test below proves that
# is all they ever receive.)
_DATA_NUMPY = ("empty", "frombuffer")


def _data_conversion_tripwire(monkeypatch, extra=()):
    """Arm every numerical NumPy routine and every tensor-data conversion
    route, including the Core-level ``to_numpy``/``from_array``."""
    def _tripwire(*args, **kwargs):
        raise AssertionError("tensor data was converted through NumPy")

    for name in _NUMERICAL_NUMPY + _DATA_NUMPY + tuple(extra):
        monkeypatch.setattr(np, name, _tripwire)
    monkeypatch.setattr(cpp.NativeTensorCore, "to_numpy", _tripwire)
    monkeypatch.setattr(cpp.NativeTensorCore, "from_array",
                        staticmethod(_tripwire))
    monkeypatch.setattr(cpp.NativeTensorView, "to_numpy", _tripwire)
    monkeypatch.setattr(cpp.NativeTensorView, "contiguous_copy", _tripwire)
    monkeypatch.setattr(cpp.NativeStorage, "from_array", staticmethod(_tripwire))
    monkeypatch.setattr(cpp.NativeStorage, "to_numpy", _tripwire)
    monkeypatch.setattr(NativeTensor, "to_numpy", _tripwire)


@needs_native
def test_native_cross_entropy_public_path_uses_no_numpy_compute(monkeypatch):
    """The whole public operation — forward, seed, backward, accumulation
    — under the strict tripwire. ``np.array`` stays available only because
    the int64 target copy is built with it (pinned below)."""
    x = NativeTensor.from_array(LOGITS, requires_grad=True)

    _data_conversion_tripwire(monkeypatch)
    loss = x.cross_entropy(TARGETS, "mean")
    loss.backward()
    monkeypatch.undo()

    assert np.isclose(float(loss.to_numpy()),
                      loss_reference(LOGITS, TARGETS, "mean"), atol=1e-14)
    assert np.allclose(x.grad.to_numpy(),
                       grad_reference(LOGITS, TARGETS, "mean"), atol=1e-15)
    loss.close()
    x.close()


@needs_native
def test_native_cross_entropy_policy_b_public_path_uses_no_numpy_compute(
    monkeypatch
):
    """The same strict tripwire on the strided path: the Policy-B copy is a
    native storage-to-storage gather (E3.1), so tensor data never leaves
    native memory even for a transposed view."""
    base = NativeTensor.from_array(LOGITS.T, requires_grad=True)

    _data_conversion_tripwire(monkeypatch)
    loss = base.T.cross_entropy(TARGETS, "sum")
    loss.backward()
    monkeypatch.undo()

    assert np.isclose(float(loss.to_numpy()),
                      loss_reference(LOGITS, TARGETS, "sum"), atol=1e-14)
    assert np.allclose(base.grad.to_numpy(),
                       grad_reference(LOGITS, TARGETS, "sum").T, atol=1e-14)
    loss.close()
    base.close()


@needs_native
def test_native_cross_entropy_backward_blocks_every_numpy_constructor(
    monkeypatch
):
    """With the target copy already prepared by the forward, the backward
    needs **no** NumPy array construction at all — so every constructor can
    be blocked outright for the backward pass."""
    x = NativeTensor.from_array(LOGITS, requires_grad=True)
    loss = x.cross_entropy(TARGETS, "mean")
    seed = scalar_seed(1.0)

    _data_conversion_tripwire(monkeypatch, extra=("array", "asarray", "zeros",
                                                  "copy", "ascontiguousarray",
                                                  "full"))
    loss.backward(gradient=seed)
    monkeypatch.undo()

    assert np.allclose(x.grad.to_numpy(),
                       grad_reference(LOGITS, TARGETS, "mean"), atol=1e-15)
    for t in (loss, seed, x):
        t.close()


@needs_native
def test_native_cross_entropy_numpy_construction_is_targets_and_metadata_only(
    monkeypatch
):
    """The deliberate boundary: ``np.array``/``np.asarray`` *are* used, to
    build the owned int64 target copy and to marshal shape/stride arrays
    for ctypes. Every value handed to a NumPy constructor across a full
    forward **and** backward (Policy-B included) must be a small tuple or
    list of Python ints — never a logit, a probability, an upstream value,
    or a gradient."""
    seen = []
    original_array = np.array
    original_asarray = np.asarray

    def _record(function):
        def _recording(values, *args, **kwargs):
            seen.append(values)
            return function(values, *args, **kwargs)
        return _recording

    monkeypatch.setattr(np, "array", _record(original_array))
    monkeypatch.setattr(np, "asarray", _record(original_asarray))
    base = NativeTensor.from_array(LOGITS.T, requires_grad=True)
    view = base.T                        # strided: Policy B is exercised
    seen.clear()                         # construction noise is not the probe
    loss = view.cross_entropy(TARGETS, "sum")
    loss.backward()
    monkeypatch.undo()

    assert seen, "no NumPy construction happened at all — check the probe"
    for values in seen:
        assert isinstance(values, (tuple, list)), values
        for item in values:
            assert isinstance(item, int) and not isinstance(item, bool), item
            assert abs(item) < 10_000, item      # metadata, never tensor data
    for t in (loss, view, base):
        t.close()


# ======================================================================
# Registry, checkpoint schema, and scope guardrails
# ======================================================================


def test_native_cross_entropy_registry_placement():
    """E6 shipped the differentiable operation, so the bare name joins
    AUTOGRAD_OPS while the layer-qualified Core wrappers stay exactly where
    E5 put them. Runs without the compiled backend: pure inventory facts."""
    assert "cross_entropy" in cpp.AUTOGRAD_OPS
    assert "cross_entropy" not in cpp.TENSOR_CORE_OPS
    assert "cross_entropy" not in cpp.UNSUPPORTED
    assert "cross_entropy" not in cpp.RAW_KERNELS
    assert "cross_entropy" not in cpp.TENSOR_CORE_KERNELS
    assert "cross_entropy" not in cpp.NATIVE_MODULES
    assert "cross_entropy" not in cpp.NATIVE_LOSSES
    for core_op in ("cross_entropy_forward", "cross_entropy_backward"):
        assert core_op in cpp.TENSOR_CORE_OPS, core_op
        assert core_op not in cpp.AUTOGRAD_OPS, core_op
        assert hasattr(cpp.NativeTensorCore, core_op), core_op
    # E7 shipped the module and the metric into their own inventories.
    assert "NativeCrossEntropyLoss" in cpp.NATIVE_LOSSES
    assert "native_accuracy" in cpp.NATIVE_METRICS
    for shipped in ("NativeCrossEntropyLoss", "native_accuracy"):
        assert shipped not in cpp.UNSUPPORTED, shipped
        assert shipped not in cpp.AUTOGRAD_OPS, shipped
        assert shipped not in cpp.TENSOR_CORE_OPS, shipped
        assert shipped not in cpp.NATIVE_MODULES, shipped
    # backend_info stays internally consistent.
    info = cpp.backend_info()
    assert "cross_entropy" in info["autograd_ops"]
    assert "cross_entropy" not in info["tensor_core_ops"]
    assert info["native_metrics"] == ("native_accuracy",)
    implemented = (set(info["tensor_core_ops"]) | set(info["autograd_ops"])
                   | set(info["raw_kernels"]))
    for name in info["unsupported"]:
        # "dropout" is the single deliberate overlap Phase G locks
        # (design §19): milestone G3 shipped the differentiable operation
        # while the *capability* stays unsupported until the G10 closure.
        # It is not a cross-entropy concern; the rule still binds every
        # other unsupported name.
        if name == "dropout":
            continue
        assert name not in implemented, name


def test_native_cross_entropy_is_a_native_tensor_operation_only():
    """The operation exists on NativeTensor and nowhere else: no Core
    ``cross_entropy``, no module, no metric, no stable-framework change."""
    import tensorforge.experimental as experimental

    assert hasattr(NativeTensor, "cross_entropy")
    assert callable(NativeTensor.cross_entropy)
    assert not hasattr(cpp.NativeTensorCore, "cross_entropy")
    assert not hasattr(NativeTensor, "nll_loss")
    # E7's public surface is built *on* this operation and lives beside
    # it, never on NativeTensor or NativeTensorCore.
    assert hasattr(experimental, "NativeCrossEntropyLoss")
    assert hasattr(experimental, "native_accuracy")
    for absent in ("cross_entropy_loss", "accuracy", "native_accuracy"):
        assert not hasattr(NativeTensor, absent), absent
        assert not hasattr(cpp.NativeTensorCore, absent), absent
    assert not hasattr(experimental, "NativeNLLLoss")


@needs_native
def test_native_cross_entropy_scope_boundaries_hold():
    """E6 is autograd integration only: no new numerical surface, no
    reduction="none", no public probability/gather/division helpers, and
    the stable framework is untouched."""
    x = NativeTensor.from_array(LOGITS, requires_grad=True)
    core = cpp.NativeTensorCore.from_array(LOGITS)
    for absent in ("max", "argmax", "amax", "divide", "gather", "scatter",
                   "sigmoid", "tanh", "one_hot", "binary_cross_entropy"):
        assert not hasattr(x, absent), absent
        assert not hasattr(core, absent), absent
    assert not hasattr(x, "__truediv__")
    with pytest.raises(ValueError):
        x.cross_entropy(TARGETS, "none")
    # No integer tensors, no new dtype/device.
    assert cpp.SUPPORTED_DTYPES == ("float64",)
    assert cpp.SUPPORTED_DEVICES == ("cpu",)
    # The stable framework keeps its own cross-entropy, entirely separately.
    logits = tensorforge.Tensor(LOGITS, requires_grad=True)
    loss = tensorforge.nn.cross_entropy(logits, [1, 2])
    loss.backward()
    assert type(loss) is tensorforge.Tensor
    assert logits.grad is not None
    # No implicit conversion in either direction.
    with pytest.raises((TypeError, AttributeError, ValueError)):
        x.cross_entropy(tensorforge.Tensor(np.array([1, 2])))
    core.close()
    x.close()


@needs_native
def test_native_cross_entropy_matches_the_stable_framework():
    """Parity with `tensorforge.nn.cross_entropy` — two independent
    engines, the same mathematics."""
    rng = np.random.default_rng(31)
    logits = rng.standard_normal((4, 5))
    targets = [0, 4, 2, 1]

    stable_logits = tensorforge.Tensor(logits, requires_grad=True)
    stable_loss = tensorforge.nn.cross_entropy(stable_logits, targets)
    stable_loss.backward()

    x = NativeTensor.from_array(logits, requires_grad=True)
    loss = x.cross_entropy(targets, "mean")
    loss.backward()

    assert np.isclose(float(loss.to_numpy()), float(stable_loss.data),
                      atol=1e-12)
    assert np.allclose(x.grad.to_numpy(), stable_logits.grad, atol=1e-12)
    loss.close()
    x.close()


def test_native_cross_entropy_checkpoint_schema_is_untouched():
    """E6 adds only ephemeral graph state: the native checkpoint format is
    still version 1, and neither the saved probabilities nor the target
    copy can reach a state dict."""
    from tensorforge.experimental import NativeLinear, native_checkpoint

    assert native_checkpoint._FORMAT_VERSION == 2
    if not cpp.is_available():
        return
    model = NativeLinear(3, 2, seed=0)
    state = model.state_dict()
    for key in state:
        assert "cross_entropy" not in key
        assert "probabilit" not in key
        assert "target" not in key
    assert set(state) == {"weight", "bias"}
    for parameter in model.parameters():
        parameter.close()
