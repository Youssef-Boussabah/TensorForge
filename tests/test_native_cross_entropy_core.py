"""Tests for the fused native cross-entropy Core contract — Phase E, E5.

The classification stack's loss, at the **graph-unaware Core layer only**
(docs/native_classification_design.md §4.5, §6, §9.2). One fused
operation through the native stack up to — and deliberately stopping at —
`NativeTensorCore`: the internal `tf::cross_entropy_forward_contiguous` /
`tf::cross_entropy_backward_contiguous` kernels in the Phase-E
classification unit → the guarded, self-validating, **contiguous-only**
C ABI (`tf_core_cross_entropy_forward` / `tf_core_cross_entropy_backward`)
→ ctypes → `NativeTensorCore.cross_entropy_forward` with Policy-B
copy-then-compute and `NativeTensorCore.cross_entropy_backward`.

**This file stays Core-only.** The `NativeTensor.cross_entropy`
autograd node E6 built on top of these helpers — its graph-owned saved
probabilities, its closure-captured targets, and its lifetime and
rollback rules — is pinned separately in
tests/test_native_cross_entropy.py; nothing here builds a graph. The loss
module and the metric are E7, and tests below still pin their absence so
that milestone boundary cannot be crossed by accident.

Forward is a **fused maximum shift + log-sum-exp** over rank-2
`(batch_size, num_classes)` logits with the class axis fixed at axis 1:
per row `log(sum(exp(x - max(x)))) - (x[target] - max(x))`, computed
inside the kernel in float64, producing the scalar loss **and** the
private saved probabilities in one pass. It is never
`-log(probabilities[target])` and never a public softmax/log-softmax
followed by an index. Backward reads **only** those saved probabilities,
the copied `int64` targets, the reduction, and a native one-element
upstream:

    grad[n, c] = upstream * (p[n, c] - [c == target_n]) / N

with the `/ N` only for `"mean"`. **The logits are never reread** — the
backward ABI cannot even see them.

Targets are strict: an independently owned, contiguous, read-only
`int64` copy is taken before anything is allocated, `bool` and
floating-point labels (including `1.0`) are rejected outright, and
mutating the caller's object afterwards cannot reach the kernel.

NumPy appears as an external oracle and, legitimately, to build the
`int64` target metadata; tripwire tests below prove no *tensor* data and
no numerical fallback ever go through it.

Selector: python -m pytest -q -k "native_cross_entropy"
"""

from pathlib import Path

import numpy as np
import pytest

import tensorforge
from tensorforge.backends import cpp
from tensorforge.experimental import NativeTensor

needs_native = pytest.mark.skipif(
    not cpp.is_available(),
    reason="experimental C++ backend not built",
)
needs_fault_injection = pytest.mark.skipif(
    not (cpp.is_available() and cpp.fault_injection_available()),
    reason="fault injection not compiled into the backend",
)

REPO_ROOT = Path(__file__).resolve().parent.parent

LOGITS = np.array([[1.0, 2.0, 0.5], [-1.0, 0.25, 3.0]])
TARGETS = [1, 2]
REDUCTIONS = ("mean", "sum")


@pytest.fixture(autouse=True)
def _disarm_after_each():
    """No test may leave the allocation injector armed or the native
    error slot dirty (the test_native_abi_errors.py convention)."""
    yield
    if cpp.is_available():
        cpp._arm_alloc_failure(0)
        cpp._require_library().tf_clear_error()


# ----------------------------------------------------------------------
# References — the SAME stable algorithm the kernel implements, in NumPy,
# used only as an external oracle (never inside an armed tripwire).
# ----------------------------------------------------------------------


def loss_reference(logits, targets, reduction):
    logits = np.asarray(logits, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.int64)
    batch_size = logits.shape[0]
    with np.errstate(invalid="ignore", over="ignore", under="ignore"):
        shifted = logits - np.max(logits, axis=1, keepdims=True)
        log_denom = np.log(np.sum(np.exp(shifted), axis=1))
        per_example = log_denom - shifted[np.arange(batch_size), targets]
    total = float(per_example.sum())
    return total / batch_size if reduction == "mean" else total


def probabilities_reference(logits):
    logits = np.asarray(logits, dtype=np.float64)
    with np.errstate(invalid="ignore", over="ignore", under="ignore"):
        shifted = logits - np.max(logits, axis=1, keepdims=True)
        exponentials = np.exp(shifted)
        return exponentials / np.sum(exponentials, axis=1, keepdims=True)


def grad_reference(probabilities, targets, reduction, upstream):
    """upstream * (p - onehot) / N, in exactly that order."""
    probabilities = np.array(probabilities, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.int64)
    batch_size = probabilities.shape[0]
    base = probabilities.copy()
    base[np.arange(batch_size), targets] -= 1.0
    if reduction == "mean":
        base /= batch_size
    return upstream * base


def scalar_core(value=1.0):
    """A native **rank-0** one-element upstream gradient — the shape the
    autograd engine's default seed for a scalar loss really has."""
    return cpp.NativeTensorCore.full((), value)


def run_forward(logits, targets, reduction="mean"):
    core = cpp.NativeTensorCore.from_array(logits)
    return core, core.cross_entropy_forward(targets, reduction)


# ======================================================================
# Kernel symbols, source unit, and the Core-only capability boundary
# ======================================================================


@needs_native
def test_native_cross_entropy_kernel_symbols_are_bound():
    library = cpp._require_library()
    assert library.tf_core_cross_entropy_forward is not None
    assert library.tf_core_cross_entropy_backward is not None
    for export in ("tf_core_cross_entropy_forward",
                   "tf_core_cross_entropy_backward"):
        assert export in cpp._CHECKED_KERNELS, export
        # An ABI symbol is not a capability name.
        assert export not in cpp.RAW_KERNELS
        assert export not in cpp.TENSOR_CORE_OPS
    # E5 added no raw NumPy-buffer kernel and left the frozen registry.
    assert cpp.TENSOR_CORE_KERNELS == ("relu", "add", "subtract",
                                       "multiply", "matmul")
    assert "cross_entropy" not in cpp.RAW_KERNELS
    assert not hasattr(cpp, "cross_entropy")


def test_native_cross_entropy_lives_in_the_classification_source_unit():
    """The E5 kernels and exports belong to the Phase-E classification
    unit locked by E0 §9.1 — not to the elementwise one."""
    classification_path = REPO_ROOT / "cpp" / "src" / "classification.cpp"
    assert classification_path.is_file()
    assert (REPO_ROOT / "cpp" / "tests" / "test_cross_entropy.cpp").is_file()
    classification = classification_path.read_text(encoding="utf-8")
    elementwise = (REPO_ROOT / "cpp" / "src" / "elementwise.cpp").read_text(
        encoding="utf-8"
    )
    header = (REPO_ROOT / "cpp" / "include" / "tf_classification_internal.h"
              ).read_text(encoding="utf-8")
    for symbol in ("tf_core_cross_entropy_forward",
                   "tf_core_cross_entropy_backward",
                   "cross_entropy_forward_contiguous",
                   "cross_entropy_backward_contiguous"):
        assert symbol in classification, symbol
        assert f"{symbol}(" not in elementwise, symbol
    for internal in ("cross_entropy_forward_contiguous",
                     "cross_entropy_backward_contiguous"):
        assert internal in header, internal
    # The CTest is registered as its own target.
    cmake = (REPO_ROOT / "cpp" / "CMakeLists.txt").read_text(encoding="utf-8")
    assert "test_cross_entropy" in cmake
    assert "add_test(NAME cross_entropy" in cmake


def test_native_cross_entropy_forward_is_fused_in_cpp():
    """The structural half of the fused contract: the kernel computes the
    maximum, the exponentials, the log-sum-exp and the loss itself, and
    never routes through the softmax or log-softmax kernels."""
    # Phase I milestone I6 made the four classification kernels templates
    # over the element type, so their definitions moved from
    # cpp/src/classification.cpp into the internal header beside it — a
    # template has to be visible where it is instantiated. The kernels are
    # the same kernels and this contract is unchanged; only where the source
    # lives moved.
    classification = (
        REPO_ROOT / "cpp" / "include" / "tf_classification_internal.h"
    ).read_text(encoding="utf-8")
    body = classification.split(
        "inline void cross_entropy_forward_contiguous(", 1
    )[1].split("\n}\n", 1)[0]
    assert "std::exp(" in body and "std::log(" in body
    for forbidden in ("softmax_forward_contiguous(",
                      "log_softmax_forward_contiguous("):
        assert forbidden not in body, forbidden
    # The backward reads probabilities and never a logit: the only
    # "logits" identifier in its body is the *destination* grad_logits,
    # and it recomputes no exponential or logarithm.
    backward = classification.split(
        "inline void cross_entropy_backward_contiguous(", 1
    )[1].split("\n}\n", 1)[0]
    assert "probabilities[" in backward
    assert backward.replace("grad_logits", "") .count("logits") == 0
    assert "std::exp(" not in backward and "std::log(" not in backward


@needs_native
def test_native_cross_entropy_core_calls_only_its_own_kernels(monkeypatch):
    """Dynamic proof: the Core forward/backward must not route through
    the softmax, log-softmax, exp, or log operations."""
    library = cpp._require_library()
    calls = []

    def _forbidden(name):
        def _tripwire(*args, **kwargs):
            calls.append(name)
            raise AssertionError(f"cross_entropy routed through {name}")
        return _tripwire

    for export in ("tf_core_softmax_forward", "tf_core_log_softmax_forward",
                   "tf_core_exp_contiguous", "tf_core_log_contiguous"):
        monkeypatch.setattr(library, export, _forbidden(export))
    for op in ("exp", "log", "softmax", "log_softmax"):
        monkeypatch.setattr(cpp.NativeTensorCore, op, _forbidden(op))
    core, result = run_forward(LOGITS, TARGETS, "sum")
    gradient = result.probabilities.cross_entropy_backward(
        result.targets, scalar_core(1.0), result.reduction
    )
    monkeypatch.undo()
    assert calls == []
    assert np.isclose(float(result.loss.to_numpy()),
                      loss_reference(LOGITS, TARGETS, "sum"), atol=1e-14)
    assert gradient.shape == LOGITS.shape
    result.close()
    gradient.close()
    core.close()


def test_native_cross_entropy_registry_placement():
    """E5 shipped the **Core layer only**, so the layer-qualified Core
    wrappers are advertised and the differentiable operation is not.
    Runs without the compiled backend: these are pure inventory facts."""
    for core_op in ("cross_entropy_forward", "cross_entropy_backward"):
        assert core_op in cpp.TENSOR_CORE_OPS, core_op
        assert core_op not in cpp.AUTOGRAD_OPS, core_op
        assert core_op not in cpp.UNSUPPORTED, core_op
        assert core_op not in cpp.NATIVE_MODULES, core_op
        assert core_op not in cpp.NATIVE_LOSSES, core_op
        assert core_op not in cpp.RAW_KERNELS, core_op
        assert core_op not in cpp.TENSOR_CORE_KERNELS, core_op
    # The differentiable operation shipped separately at E6 and lives
    # under the bare name in AUTOGRAD_OPS — never in this Core inventory.
    assert "cross_entropy" in cpp.AUTOGRAD_OPS
    assert "cross_entropy" not in cpp.TENSOR_CORE_OPS
    # E7's module and metric shipped into their own layer inventories,
    # and neither is a Core capability.
    assert "NativeCrossEntropyLoss" in cpp.NATIVE_LOSSES
    assert "native_accuracy" in cpp.NATIVE_METRICS
    for shipped in ("NativeCrossEntropyLoss", "native_accuracy"):
        assert shipped not in cpp.UNSUPPORTED, shipped
        assert shipped not in cpp.TENSOR_CORE_OPS, shipped
        assert shipped not in cpp.AUTOGRAD_OPS, shipped
        assert shipped not in cpp.NATIVE_MODULES, shipped
    # E1-E4 stay implemented alongside it.
    for shipped in ("exp", "log", "softmax", "log_softmax"):
        assert shipped in cpp.TENSOR_CORE_OPS and shipped in cpp.AUTOGRAD_OPS
    info = cpp.backend_info()
    assert "cross_entropy_forward" in info["tensor_core_ops"]
    assert "cross_entropy_backward" in info["tensor_core_ops"]
    assert "cross_entropy" in info["autograd_ops"]
    assert "cross_entropy" not in info["tensor_core_ops"]
    assert info["native_metrics"] == ("native_accuracy",)
    # backend_info stays internally consistent: nothing advertised as
    # unsupported may appear in an implemented inventory.
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


@needs_native
def test_native_cross_entropy_core_layer_stays_graph_unaware():
    """E5's hard boundary, still binding after E6 built on it: the Core
    layer itself knows nothing about graphs. The differentiable operation
    that E6 added lives one layer up, on NativeTensor, and its contract is
    tests/test_native_cross_entropy.py."""
    import tensorforge.experimental as experimental

    x = NativeTensor.from_array(LOGITS)
    core = cpp.NativeTensorCore.from_array(LOGITS)
    # The E6 operation is a NativeTensor method under the bare name only —
    # the Core wrappers never leaked upward, and no NLL variant appeared.
    assert hasattr(x, "cross_entropy")
    for absent in ("cross_entropy_forward", "cross_entropy_backward",
                   "nll_loss"):
        assert not hasattr(x, absent), absent
    # ...and no bare Core "cross_entropy" either: the Core surface is
    # layer-qualified on purpose.
    assert not hasattr(core, "cross_entropy")
    assert hasattr(core, "cross_entropy_forward")
    assert hasattr(core, "cross_entropy_backward")
    # E7's module and metric exist, but neither is a Core capability:
    # they are Python surfaces built on this layer, not methods of it.
    assert hasattr(experimental, "NativeCrossEntropyLoss")
    assert hasattr(experimental, "native_accuracy")
    assert not hasattr(experimental, "NativeNLLLoss")
    for absent in ("native_accuracy", "accuracy"):
        assert not hasattr(core, absent), absent
    # No graph-resource or expected-version machinery was wired up for
    # cross-entropy: the Core forward builds no node at all.
    result = core.cross_entropy_forward(TARGETS, "mean")
    assert not hasattr(result, "_graph_resources")
    assert not hasattr(result, "_expected_versions")
    assert type(result.loss) is cpp.NativeTensorCore
    assert type(result.probabilities) is cpp.NativeTensorCore
    result.close()
    core.close()


# ======================================================================
# Forward numerical correctness and stability
# ======================================================================


@needs_native
@pytest.mark.parametrize("reduction", REDUCTIONS)
def test_native_cross_entropy_forward_small_two_class_batch(reduction):
    logits = np.array([[2.0, 0.5], [-1.0, 1.5]])
    targets = [0, 1]
    core, result = run_forward(logits, targets, reduction)
    assert np.isclose(float(result.loss.to_numpy()),
                      loss_reference(logits, targets, reduction), atol=1e-15)
    result.close()
    core.close()


@needs_native
@pytest.mark.parametrize("reduction", REDUCTIONS)
def test_native_cross_entropy_forward_multi_class_batch(reduction):
    logits = np.array([
        [1.0, 2.0, 0.5, -1.0],
        [-2.0, 0.25, 3.0, 1.5],
        [0.0, 0.0, 0.0, 4.0],
    ])
    targets = [1, 2, 3]
    core, result = run_forward(logits, targets, reduction)
    assert np.isclose(float(result.loss.to_numpy()),
                      loss_reference(logits, targets, reduction), atol=1e-14)
    assert np.allclose(result.probabilities.to_numpy(),
                       probabilities_reference(logits), atol=1e-15)
    result.close()
    core.close()


@needs_native
def test_native_cross_entropy_forward_batch_size_one():
    logits = np.array([[1.0, 2.0, 0.5]])
    core, mean = run_forward(logits, [2], "mean")
    _, summed = run_forward(logits, [2], "sum")
    # With one example the two reductions coincide exactly.
    assert float(mean.loss.to_numpy()) == float(summed.loss.to_numpy())
    assert np.isclose(float(mean.loss.to_numpy()),
                      loss_reference(logits, [2], "sum"), atol=1e-15)
    mean.close()
    summed.close()
    core.close()


@needs_native
def test_native_cross_entropy_forward_equal_logits_is_log_num_classes():
    logits = np.full((3, 4), 3.25)
    core, result = run_forward(logits, [0, 3, 1], "mean")
    assert np.isclose(float(result.loss.to_numpy()), np.log(4.0), atol=1e-15)
    assert np.allclose(result.probabilities.to_numpy(), 0.25, atol=1e-15)
    result.close()
    core.close()


@needs_native
def test_native_cross_entropy_forward_single_class_is_zero():
    """With one class the target always wins: the loss is exactly 0."""
    core, result = run_forward(np.array([[5.0], [-3.0], [100.0]]), [0, 0, 0],
                               "sum")
    assert float(result.loss.to_numpy()) == 0.0
    assert np.array_equal(result.probabilities.to_numpy(), np.ones((3, 1)))
    result.close()
    core.close()


@needs_native
@pytest.mark.parametrize("reduction", REDUCTIONS)
def test_native_cross_entropy_forward_random_logits(reduction):
    rng = np.random.default_rng(20260722)
    logits = rng.normal(size=(6, 5)) * 2.5
    targets = rng.integers(0, 5, size=6).tolist()
    core, result = run_forward(logits, targets, reduction)
    assert np.isclose(float(result.loss.to_numpy()),
                      loss_reference(logits, targets, reduction), atol=1e-13)
    assert np.allclose(result.probabilities.to_numpy(),
                       probabilities_reference(logits), atol=1e-15)
    result.close()
    core.close()


@needs_native
@pytest.mark.parametrize("offset", [700.0, -700.0, 1e5, -1e5, 1e10, -1e10])
def test_native_cross_entropy_forward_is_stable_under_large_offsets(offset):
    """A large common offset must not overflow and must not change the
    loss — the whole point of the maximum shift. A naive
    log(sum(exp(x))) gives inf at +700 and -inf at -700."""
    base = np.array([[0.0, 1.0, 2.0, 3.0], [-1.0, 0.5, 4.0, 2.5]])
    targets = [2, 3]
    core, want = run_forward(base, targets, "mean")
    shifted_core, got = run_forward(base + offset, targets, "mean")
    assert np.isfinite(float(got.loss.to_numpy()))
    assert np.isclose(float(got.loss.to_numpy()),
                      float(want.loss.to_numpy()), atol=1e-6)
    probabilities = got.probabilities.to_numpy()
    assert np.all(np.isfinite(probabilities))
    assert np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-12)
    want.close()
    got.close()
    core.close()
    shifted_core.close()


@needs_native
def test_native_cross_entropy_forward_per_row_shift_invariance():
    """Adding a per-row constant leaves that row's loss unchanged."""
    logits = np.array([[0.5, -1.5, 2.25], [1.0, 3.0, -0.25]])
    targets = [2, 0]
    shifts = np.array([[37.5], [-12.25]])
    core, want = run_forward(logits, targets, "sum")
    shifted_core, got = run_forward(logits + shifts, targets, "sum")
    assert np.isclose(float(got.loss.to_numpy()),
                      float(want.loss.to_numpy()), atol=1e-13)
    assert np.allclose(got.probabilities.to_numpy(),
                       want.probabilities.to_numpy(), atol=1e-14)
    want.close()
    got.close()
    core.close()
    shifted_core.close()


@needs_native
def test_native_cross_entropy_forward_avoids_the_naive_log_probability():
    """The numerical half of the fused contract: for a target 800 below
    the row maximum the probability underflows to exactly 0, so
    ``-log(probability[target])`` would report an infinite loss while the
    log-sum-exp form stays finite and accurate."""
    core, result = run_forward(np.array([[0.0, -800.0]]), [1], "sum")
    loss = float(result.loss.to_numpy())
    assert np.isfinite(loss)
    assert np.isclose(loss, 800.0, atol=1e-9)
    probability = result.probabilities.to_numpy()[0, 1]
    assert probability == 0.0
    with np.errstate(divide="ignore"):
        assert np.isinf(-np.log(probability))  # the naive form fails here
    result.close()
    core.close()


@needs_native
def test_native_cross_entropy_probabilities_match_native_softmax():
    core = cpp.NativeTensorCore.from_array(LOGITS)
    result = core.cross_entropy_forward(TARGETS, "mean")
    softmax = core.softmax(-1)
    assert np.allclose(result.probabilities.to_numpy(), softmax.to_numpy(),
                       atol=1e-15)
    rows = result.probabilities.to_numpy().sum(axis=1)
    assert np.allclose(rows, 1.0, atol=1e-15)
    softmax.close()
    result.close()
    core.close()


@needs_native
def test_native_cross_entropy_mean_times_batch_equals_sum():
    logits = np.array([[1.0, 2.0, 0.5], [-1.0, 0.25, 3.0], [2.0, 0.0, 1.0]])
    targets = [1, 2, 0]
    core = cpp.NativeTensorCore.from_array(logits)
    mean = core.cross_entropy_forward(targets, "mean")
    summed = core.cross_entropy_forward(targets, "sum")
    assert np.isclose(float(mean.loss.to_numpy()) * 3,
                      float(summed.loss.to_numpy()), atol=1e-14)
    # The probabilities do not depend on the reduction.
    assert np.array_equal(mean.probabilities.to_numpy(),
                          summed.probabilities.to_numpy())
    assert mean.reduction == "mean" and summed.reduction == "sum"
    mean.close()
    summed.close()
    core.close()


@needs_native
def test_native_cross_entropy_forward_is_deterministic_and_pure():
    core = cpp.NativeTensorCore.from_array(LOGITS)
    first = core.cross_entropy_forward(TARGETS, "mean")
    second = core.cross_entropy_forward(TARGETS, "mean")
    assert float(first.loss.to_numpy()) == float(second.loss.to_numpy())
    assert np.array_equal(first.probabilities.to_numpy(),
                          second.probabilities.to_numpy())
    # The logits are untouched, and the two results are independent.
    assert np.array_equal(core.to_numpy(), LOGITS)
    assert first.probabilities.storage is not second.probabilities.storage
    first.close()
    second.close()
    core.close()


@needs_native
def test_native_cross_entropy_outputs_are_fresh_owning_and_scalar():
    core = cpp.NativeTensorCore.from_array(LOGITS)
    result = core.cross_entropy_forward(TARGETS, "mean")
    # The loss is a rank-0 scalar, the repository's scalar convention.
    assert result.loss.shape == ()
    assert result.loss.ndim == 0 and result.loss.numel == 1
    assert result.loss.contiguous and result.loss.offset == 0
    assert result.loss.dtype == "float64" and result.loss.device == "cpu"
    assert result.loss._owns_storage
    # The saved probabilities have the logits' shape and own fresh,
    # contiguous, offset-0 storage that aliases nothing.
    probabilities = result.probabilities
    assert probabilities.shape == LOGITS.shape
    assert probabilities.contiguous and probabilities.offset == 0
    assert probabilities._owns_storage
    assert probabilities.storage is not core.storage
    assert probabilities.storage is not result.loss.storage
    assert np.array_equal(core.to_numpy(), LOGITS)
    result.close()
    core.close()


@needs_native
def test_native_cross_entropy_result_close_is_idempotent():
    core = cpp.NativeTensorCore.from_array(LOGITS)
    result = core.cross_entropy_forward(TARGETS, "mean")
    result.close()
    assert result.loss._closed and result.probabilities._closed
    result.close()          # idempotent
    result.probabilities.close()
    result.loss.close()
    # The targets survive the close: they are ordinary host metadata.
    assert result.targets.tolist() == TARGETS
    assert np.array_equal(core.to_numpy(), LOGITS)
    core.close()


# ======================================================================
# Backward numerical correctness
# ======================================================================


@needs_native
@pytest.mark.parametrize("reduction", REDUCTIONS)
@pytest.mark.parametrize("upstream", [1.0, 2.5, -1.5, 0.0])
def test_native_cross_entropy_backward_matches_the_formula(reduction, upstream):
    core = cpp.NativeTensorCore.from_array(LOGITS)
    result = core.cross_entropy_forward(TARGETS, reduction)
    seed = scalar_core(upstream)
    gradient = result.probabilities.cross_entropy_backward(
        result.targets, seed, result.reduction
    )
    want = grad_reference(result.probabilities.to_numpy(), TARGETS, reduction,
                          upstream)
    assert np.allclose(gradient.to_numpy(), want, atol=1e-15)
    if upstream == 0.0:
        assert np.array_equal(gradient.to_numpy(), np.zeros(LOGITS.shape))
    gradient.close()
    seed.close()
    result.close()
    core.close()


@needs_native
def test_native_cross_entropy_backward_rows_sum_to_zero():
    core = cpp.NativeTensorCore.from_array(LOGITS)
    for reduction in REDUCTIONS:
        result = core.cross_entropy_forward(TARGETS, reduction)
        seed = scalar_core(1.5)
        gradient = result.probabilities.cross_entropy_backward(
            result.targets, seed, reduction
        )
        assert np.allclose(gradient.to_numpy().sum(axis=1), 0.0, atol=1e-15)
        gradient.close()
        seed.close()
        result.close()
    core.close()


@needs_native
def test_native_cross_entropy_backward_accepts_one_element_upstreams():
    """The upstream convention: any open core holding exactly one logical
    element, whatever its shape or offset — the value is read straight
    from its storage."""
    core = cpp.NativeTensorCore.from_array(LOGITS)
    result = core.cross_entropy_forward(TARGETS, "sum")
    want = grad_reference(result.probabilities.to_numpy(), TARGETS, "sum", 3.0)

    seeds = [
        cpp.NativeTensorCore.full((), 3.0),                       # rank 0
        cpp.NativeTensorCore.from_array(np.array([3.0])),         # (1,)
        cpp.NativeTensorCore.from_array(np.array([[3.0]])),       # (1, 1)
    ]
    for seed in seeds:
        assert seed.numel == 1
        gradient = result.probabilities.cross_entropy_backward(
            result.targets, seed, "sum"
        )
        assert np.allclose(gradient.to_numpy(), want, atol=1e-15), seed.shape
        gradient.close()
        seed.close()
    # ...including a one-element view at a nonzero offset, which proves
    # the offset really is honored rather than assumed to be 0.
    base = cpp.NativeTensorCore.from_array(np.array([99.0, 3.0]))
    narrowed = base.narrow(0, 1, 1)
    assert narrowed.offset == 1 and narrowed.numel == 1
    gradient = result.probabilities.cross_entropy_backward(
        result.targets, narrowed, "sum"
    )
    assert np.allclose(gradient.to_numpy(), want, atol=1e-15)
    gradient.close()
    base.close()
    result.close()
    core.close()


@needs_native
@pytest.mark.parametrize("reduction", REDUCTIONS)
def test_native_cross_entropy_backward_finite_differences(reduction):
    """Central differences of the forward loss agree with the backward
    gradient — the strongest cross-check that the two kernels describe
    the same function."""
    logits = np.array([[0.5, -1.0, 2.0], [1.25, 3.0, 0.0]])
    targets = [2, 0]
    core = cpp.NativeTensorCore.from_array(logits)
    result = core.cross_entropy_forward(targets, reduction)
    seed = scalar_core(1.0)
    gradient = result.probabilities.cross_entropy_backward(
        result.targets, seed, reduction
    ).to_numpy()

    eps = 1e-6
    numeric = np.zeros_like(logits)
    for index in np.ndindex(logits.shape):
        perturbed = logits.copy()
        perturbed[index] += eps
        plus_core = cpp.NativeTensorCore.from_array(perturbed)
        plus = plus_core.cross_entropy_forward(targets, reduction)
        perturbed[index] -= 2 * eps
        minus_core = cpp.NativeTensorCore.from_array(perturbed)
        minus = minus_core.cross_entropy_forward(targets, reduction)
        numeric[index] = (float(plus.loss.to_numpy())
                          - float(minus.loss.to_numpy())) / (2 * eps)
        plus.close()
        minus.close()
        plus_core.close()
        minus_core.close()
    assert np.allclose(gradient, numeric, atol=1e-7)
    seed.close()
    result.close()
    core.close()


@needs_native
def test_native_cross_entropy_backward_is_deterministic_and_independent():
    core = cpp.NativeTensorCore.from_array(LOGITS)
    result = core.cross_entropy_forward(TARGETS, "mean")
    original = result.probabilities.to_numpy()
    seed = scalar_core(2.0)
    first = result.probabilities.cross_entropy_backward(result.targets, seed,
                                                        "mean")
    second = result.probabilities.cross_entropy_backward(result.targets, seed,
                                                         "mean")
    assert np.array_equal(first.to_numpy(), second.to_numpy())
    assert first.storage is not second.storage
    # Repeated calls mutate neither the probabilities nor the upstream.
    assert np.array_equal(result.probabilities.to_numpy(), original)
    assert seed.to_numpy().item() == 2.0
    for gradient in (first, second):
        assert gradient.shape == LOGITS.shape
        assert gradient.contiguous and gradient.offset == 0
        assert gradient._owns_storage
        assert gradient.storage is not result.probabilities.storage
    first.close()
    second.close()
    seed.close()
    result.close()
    core.close()


@needs_native
def test_native_cross_entropy_backward_never_sees_the_logits():
    """The structural guarantee: the backward is called on the saved
    probabilities and has no logits argument at all, so closing the
    logits after the forward changes nothing."""
    core = cpp.NativeTensorCore.from_array(LOGITS)
    result = core.cross_entropy_forward(TARGETS, "mean")
    want = grad_reference(result.probabilities.to_numpy(), TARGETS, "mean", 1.0)
    core.close()                    # the logits are gone entirely
    seed = scalar_core(1.0)
    gradient = result.probabilities.cross_entropy_backward(result.targets, seed,
                                                           "mean")
    assert np.allclose(gradient.to_numpy(), want, atol=1e-15)
    gradient.close()
    seed.close()
    result.close()


@needs_native
def test_native_cross_entropy_backward_after_a_non_contiguous_forward():
    base = cpp.NativeTensorCore.from_array(LOGITS.T)
    strided = base.T
    assert not strided.contiguous
    result = strided.cross_entropy_forward(TARGETS, "mean")
    assert np.isclose(float(result.loss.to_numpy()),
                      loss_reference(LOGITS, TARGETS, "mean"), atol=1e-14)
    seed = scalar_core(1.0)
    gradient = result.probabilities.cross_entropy_backward(result.targets, seed,
                                                           "mean")
    assert np.allclose(gradient.to_numpy(),
                       grad_reference(probabilities_reference(LOGITS), TARGETS,
                                      "mean", 1.0), atol=1e-15)
    assert np.array_equal(base.to_numpy(), LOGITS.T)   # caller untouched
    gradient.close()
    seed.close()
    result.close()
    base.close()


@needs_native
def test_native_cross_entropy_backward_rejects_invalid_arguments():
    core = cpp.NativeTensorCore.from_array(LOGITS)
    result = core.cross_entropy_forward(TARGETS, "mean")
    probabilities, targets = result.probabilities, result.targets
    seed = scalar_core(1.0)

    # A non-Core upstream, and a closed one.
    with pytest.raises(TypeError, match="NativeTensorCore upstream"):
        probabilities.cross_entropy_backward(targets, 1.0, "mean")
    with pytest.raises(TypeError, match="NativeTensorCore upstream"):
        probabilities.cross_entropy_backward(targets, np.array(1.0), "mean")
    closed = scalar_core(1.0)
    closed.close()
    with pytest.raises(RuntimeError, match="closed"):
        probabilities.cross_entropy_backward(targets, closed, "mean")
    # A multi-element upstream: the loss is a scalar.
    wide = cpp.NativeTensorCore.from_array(np.array([1.0, 2.0]))
    with pytest.raises(ValueError, match="one-element upstream"):
        probabilities.cross_entropy_backward(targets, wide, "mean")
    # A closed probability core.
    spent = core.cross_entropy_forward(TARGETS, "mean")
    spent_probabilities = spent.probabilities
    spent.close()
    with pytest.raises(RuntimeError, match="closed"):
        spent_probabilities.cross_entropy_backward(targets, seed, "mean")
    # Rank and contiguity of the probabilities.
    flat = cpp.NativeTensorCore.from_array(np.array([0.25, 0.75]))
    with pytest.raises(ValueError, match="2-D"):
        flat.cross_entropy_backward(np.array([0], dtype=np.int64), seed, "mean")
    strided = cpp.NativeTensorCore.from_array(LOGITS).T
    with pytest.raises(ValueError, match="contiguous saved probabilities"):
        strided.cross_entropy_backward(np.array([0, 1, 0], dtype=np.int64),
                                       seed, "mean")
    # The trusted-copy contract for the targets.
    with pytest.raises(TypeError, match="int64 target copy"):
        probabilities.cross_entropy_backward([1, 2], seed, "mean")
    with pytest.raises(TypeError, match="int64 target copy"):
        probabilities.cross_entropy_backward(
            np.array([1, 2], dtype=np.int32), seed, "mean")
    with pytest.raises(ValueError, match="one-dimensional target copy"):
        probabilities.cross_entropy_backward(
            np.array([[1, 2]], dtype=np.int64), seed, "mean")
    with pytest.raises(ValueError, match="exactly 2 targets"):
        probabilities.cross_entropy_backward(
            np.array([1], dtype=np.int64), seed, "mean")
    with pytest.raises(ValueError, match="valid class range"):
        probabilities.cross_entropy_backward(
            np.array([1, 3], dtype=np.int64), seed, "mean")
    seed.close()
    wide.close()
    flat.close()
    result.close()
    core.close()


# ======================================================================
# Strict targets (design §6)
# ======================================================================


@needs_native
@pytest.mark.parametrize("targets", [
    [1, 2],
    (1, 2),
    np.array([1, 2], dtype=np.int64),
    np.array([1, 2], dtype=np.int32),
    np.array([1, 2], dtype=np.int8),
    np.array([1, 2], dtype=np.uint8),
    np.array([1, 2], dtype=np.uint32),
    np.array([1, 2], dtype=np.uint64),
    [np.int64(1), np.int32(2)],
])
def test_native_cross_entropy_accepts_integer_target_forms(targets):
    core = cpp.NativeTensorCore.from_array(LOGITS)
    result = core.cross_entropy_forward(targets, "sum")
    assert np.isclose(float(result.loss.to_numpy()),
                      loss_reference(LOGITS, [1, 2], "sum"), atol=1e-15)
    assert result.targets.dtype == np.int64
    assert result.targets.tolist() == [1, 2]
    result.close()
    core.close()


@needs_native
def test_native_cross_entropy_accepts_a_non_contiguous_integer_view():
    """A strided 1-D integer view is copied into contiguous owned
    storage, not borrowed."""
    source = np.array([1, 99, 2, 99], dtype=np.int64)
    view = source[::2]
    assert not view.flags["C_CONTIGUOUS"]
    core = cpp.NativeTensorCore.from_array(LOGITS)
    result = core.cross_entropy_forward(view, "sum")
    assert result.targets.tolist() == [1, 2]
    assert result.targets.flags["C_CONTIGUOUS"]
    assert result.targets.flags["OWNDATA"]
    assert not np.shares_memory(result.targets, source)
    assert np.isclose(float(result.loss.to_numpy()),
                      loss_reference(LOGITS, [1, 2], "sum"), atol=1e-15)
    result.close()
    core.close()


@needs_native
@pytest.mark.parametrize("targets", [
    True,                                   # a Python bool scalar
    np.bool_(True),                         # a NumPy bool scalar
    [True, False],                          # a list containing bools
    [1, True],                              # ...even one
    np.array([True, False]),                # a NumPy bool array
    1.5,                                    # a float scalar
    [1.0, 2.0],                             # a float list
    [1, 2.0],                               # a mixed int/float list
    np.array([1.0, 2.0]),                   # a float array
    np.array([1.0, 2.0], dtype=np.float32),
    [1.0, 1.0],                             # integral-looking floats
    np.array([1, 2], dtype=complex),
    complex(1, 0),
    "12",                                   # a string
    b"\x01\x02",                            # bytes
    bytearray(b"\x01\x02"),
    np.array([1, 2.0], dtype=object),       # an object array with a float
    [[1], [2]],                             # a nested list
    [[1, 2]],
    np.array([[1, 2]]),                     # a rank-2 array
    np.array(1),                            # a rank-0 array
    1,                                      # a bare scalar int
    np.int64(1),
    [1, [2]],                               # a ragged sequence
    None,
])
def test_native_cross_entropy_rejects_invalid_target_types(targets):
    core = cpp.NativeTensorCore.from_array(LOGITS)
    with pytest.raises((TypeError, ValueError)):
        core.cross_entropy_forward(targets, "mean")
    core.close()


@needs_native
@pytest.mark.parametrize("targets,message", [
    ([-1, 1], "valid class range"),
    ([0, -5], "valid class range"),
    ([3, 1], "valid class range"),          # == num_classes
    ([1, 4], "valid class range"),          # > num_classes
    ([2 ** 63, 1], "int64 range"),
    ([-(2 ** 63) - 1, 1], "int64 range"),
    (np.array([2 ** 63, 1], dtype=np.uint64), "int64 range"),
    ([1], "exactly 2 targets"),
    ([1, 2, 0], "exactly 2 targets"),
    ([], "exactly 2 targets"),
])
def test_native_cross_entropy_rejects_invalid_target_values(targets, message):
    core = cpp.NativeTensorCore.from_array(LOGITS)
    with pytest.raises(ValueError, match=message):
        core.cross_entropy_forward(targets, "mean")
    core.close()


@needs_fault_injection
def test_native_cross_entropy_target_validation_precedes_allocation():
    """Every invalid target must be rejected before a single native
    allocation — proved by making allocation itself fail."""
    core = cpp.NativeTensorCore.from_array(LOGITS)
    for targets in ([1, 2.0], [1, True], [-1, 1], [1, 3], [1], "12", None,
                    np.array([[1, 2]])):
        cpp._arm_alloc_failure(1)
        try:
            with pytest.raises((TypeError, ValueError)):
                core.cross_entropy_forward(targets, "mean")
        finally:
            cpp._arm_alloc_failure(0)
    # No stale native error was left behind, and the operation still works.
    assert cpp._require_library().tf_last_error_code() == cpp.TF_OK
    result = core.cross_entropy_forward(TARGETS, "mean")
    result.close()
    core.close()


# ======================================================================
# Target-copy ownership and caller-mutation immunity
# ======================================================================


@needs_native
def test_native_cross_entropy_target_copy_is_owned_and_read_only():
    core = cpp.NativeTensorCore.from_array(LOGITS)
    source = np.array([1, 2], dtype=np.int64)
    assert source.flags["C_CONTIGUOUS"]      # already the ideal input
    result = core.cross_entropy_forward(source, "mean")
    copy = result.targets
    assert copy.dtype == np.int64
    assert copy.ndim == 1 and copy.size == 2
    assert copy.flags["C_CONTIGUOUS"] and copy.flags["OWNDATA"]
    assert not copy.flags["WRITEABLE"]       # the saved copy cannot be edited
    # An independent copy is taken even from a perfect input.
    assert copy is not source
    assert not np.shares_memory(copy, source)
    with pytest.raises(ValueError):
        copy[0] = 0
    result.close()
    core.close()


@needs_native
@pytest.mark.parametrize("as_array", [False, True])
def test_native_cross_entropy_caller_target_mutation_cannot_reach_backward(
        as_array):
    """Run the forward, mutate the caller's targets, then run the
    backward: the gradient must still be based on the original labels."""
    caller_targets = (np.array([1, 2], dtype=np.int64) if as_array else [1, 2])
    core = cpp.NativeTensorCore.from_array(LOGITS)
    result = core.cross_entropy_forward(caller_targets, "mean")
    saved = result.targets
    want = grad_reference(result.probabilities.to_numpy(), [1, 2], "mean", 1.0)

    # Mutate the caller's object in place, thoroughly.
    if as_array:
        caller_targets[0] = 0
        caller_targets[1] = 0
    else:
        caller_targets[0] = 0
        caller_targets[1] = 0
        caller_targets.append(99)
    assert saved.tolist() == [1, 2], "the saved copy tracked caller memory"

    seed = scalar_core(1.0)
    gradient = result.probabilities.cross_entropy_backward(saved, seed, "mean")
    assert np.allclose(gradient.to_numpy(), want, atol=1e-15)
    # A gradient built from the mutated labels would differ.
    mutated = grad_reference(result.probabilities.to_numpy(), [0, 0], "mean",
                             1.0)
    assert not np.allclose(gradient.to_numpy(), mutated)
    if as_array:
        assert not np.shares_memory(saved, caller_targets)
    gradient.close()
    seed.close()
    result.close()
    core.close()


# ======================================================================
# Reduction contract
# ======================================================================


def test_native_cross_entropy_reduction_codes_are_the_locked_mapping():
    """The ABI carries a small integer, never a string, and the mapping
    matches the C++ constants (design §9.2)."""
    assert cpp._REDUCTION_CODES == {"mean": 0, "sum": 1}
    header = (REPO_ROOT / "cpp" / "include" / "tf_classification_internal.h"
              ).read_text(encoding="utf-8")
    assert "kCrossEntropyReductionMean = 0" in header
    assert "kCrossEntropyReductionSum = 1" in header
    assert cpp._normalize_reduction("mean", "op") == ("mean", 0)
    assert cpp._normalize_reduction("sum", "op") == ("sum", 1)


@needs_native
@pytest.mark.parametrize("reduction", [
    "none", "Mean", "SUM", "Sum", " mean", "mean ", "", "average", "batchmean",
])
def test_native_cross_entropy_rejects_unknown_reduction_strings(reduction):
    core = cpp.NativeTensorCore.from_array(LOGITS)
    with pytest.raises(ValueError, match="must be one of"):
        core.cross_entropy_forward(TARGETS, reduction)
    with pytest.raises(ValueError, match="must be one of"):
        cpp.NativeTensorCore.from_array(
            probabilities_reference(LOGITS)
        ).cross_entropy_backward(np.array(TARGETS, dtype=np.int64),
                                 scalar_core(1.0), reduction)
    core.close()


@needs_native
@pytest.mark.parametrize("reduction", [
    None, True, False, 0, 1, 1.0, ["mean"], ("mean",), {"mean"}, b"mean",
])
def test_native_cross_entropy_rejects_non_string_reductions(reduction):
    core = cpp.NativeTensorCore.from_array(LOGITS)
    with pytest.raises(TypeError, match="must be a str"):
        core.cross_entropy_forward(TARGETS, reduction)
    core.close()


@needs_fault_injection
def test_native_cross_entropy_reduction_validation_precedes_allocation():
    """An invalid reduction must be rejected before any allocation *and*
    before the targets are even inspected."""
    core = cpp.NativeTensorCore.from_array(LOGITS)
    for reduction in ("none", None, True, 1.0, ""):
        cpp._arm_alloc_failure(1)
        try:
            with pytest.raises((TypeError, ValueError)):
                # Targets that would themselves be rejected: the reduction
                # error must win, proving it is checked first.
                core.cross_entropy_forward([1.0, 2.0], reduction)
        finally:
            cpp._arm_alloc_failure(0)
    assert cpp._require_library().tf_last_error_code() == cpp.TF_OK
    core.close()


@needs_native
def test_native_cross_entropy_only_valid_codes_reach_the_abi(monkeypatch):
    """Whatever the caller passes, the ABI only ever sees 0 or 1."""
    library = cpp._require_library()
    seen = []
    original = library.tf_core_cross_entropy_forward

    def _recording(*args):
        seen.append(args[-1])
        return original(*args)

    monkeypatch.setattr(library, "tf_core_cross_entropy_forward", _recording)
    core = cpp.NativeTensorCore.from_array(LOGITS)
    for reduction in REDUCTIONS:
        core.cross_entropy_forward(TARGETS, reduction).close()
    for bad in ("none", "Mean", None, True, 1, 1.0, [], ""):
        with pytest.raises((TypeError, ValueError)):
            core.cross_entropy_forward(TARGETS, bad)
    monkeypatch.undo()
    assert seen == [0, 1]
    core.close()


# ======================================================================
# Shape, views, Policy B, and ownership
# ======================================================================


@needs_native
@pytest.mark.parametrize("values", [
    np.array([1.0, 2.0, 3.0]),                    # rank 1
    np.arange(8.0).reshape(2, 2, 2),              # rank 3
    np.array(1.0),                                # rank 0
])
def test_native_cross_entropy_rejects_non_rank_two_logits(values):
    core = cpp.NativeTensorCore.from_array(values)
    with pytest.raises(ValueError, match="2-D"):
        core.cross_entropy_forward([0], "mean")
    core.close()


@needs_native
def test_native_cross_entropy_rejects_closed_logits():
    core = cpp.NativeTensorCore.from_array(LOGITS)
    core.close()
    with pytest.raises(RuntimeError, match="closed"):
        core.cross_entropy_forward(TARGETS, "mean")


@needs_native
def test_native_cross_entropy_handles_non_contiguous_logits():
    # A transposed rank-2 view.
    base = cpp.NativeTensorCore.from_array(LOGITS.T)
    transposed = base.T
    assert not transposed.contiguous
    result = transposed.cross_entropy_forward(TARGETS, "sum")
    assert np.isclose(float(result.loss.to_numpy()),
                      loss_reference(LOGITS, TARGETS, "sum"), atol=1e-14)
    assert np.allclose(result.probabilities.to_numpy(),
                       probabilities_reference(LOGITS), atol=1e-15)
    result.close()

    # A narrowed view with a nonzero offset (still contiguous: the direct
    # path, but with an offset the ABI must honor).
    wide = cpp.NativeTensorCore.from_array(
        np.array([[9.0, 1.0, 2.0, 0.5], [9.0, -1.0, 0.25, 3.0]])
    )
    narrowed = wide.narrow(1, 1, 3)
    assert narrowed.offset != 0
    narrowed_result = narrowed.cross_entropy_forward(TARGETS, "sum")
    assert np.isclose(float(narrowed_result.loss.to_numpy()),
                      loss_reference(LOGITS, TARGETS, "sum"), atol=1e-14)
    narrowed_result.close()

    # ...and the transpose of a narrow: non-contiguous *and* offset.
    combined = wide.narrow(1, 1, 3).T
    assert not combined.contiguous and combined.offset != 0
    combined_result = combined.cross_entropy_forward([0, 1, 1], "sum")
    assert np.isclose(float(combined_result.loss.to_numpy()),
                      loss_reference(LOGITS.T, [0, 1, 1], "sum"), atol=1e-14)
    combined_result.close()

    # Every caller view is untouched and still usable.
    assert np.array_equal(base.to_numpy(), LOGITS.T)
    assert np.array_equal(wide.to_numpy(),
                          np.array([[9.0, 1.0, 2.0, 0.5],
                                    [9.0, -1.0, 0.25, 3.0]]))
    base.close()
    wide.close()


@needs_native
def test_native_cross_entropy_policy_b_temporary_is_closed_on_success(
        monkeypatch):
    made = []
    original = cpp.NativeTensorCore.contiguous_copy

    def _recording_copy(self):
        temp = original(self)
        made.append(temp)
        return temp

    monkeypatch.setattr(cpp.NativeTensorCore, "contiguous_copy",
                        _recording_copy)
    base = cpp.NativeTensorCore.from_array(LOGITS.T)
    result = base.T.cross_entropy_forward(TARGETS, "mean")
    assert len(made) == 1, "a non-contiguous input must be copied once"
    assert made[0]._closed, "the Policy-B temporary was not closed"
    assert not result.loss._closed and not result.probabilities._closed
    # A contiguous input makes no copy at all.
    made.clear()
    contiguous = cpp.NativeTensorCore.from_array(LOGITS)
    contiguous.cross_entropy_forward(TARGETS, "mean").close()
    assert made == []
    result.close()
    base.close()
    contiguous.close()


@needs_fault_injection
def test_native_cross_entropy_policy_b_temporary_is_closed_on_failure(
        monkeypatch):
    """...and also when a later allocation fails after the copy."""
    made = []
    original = cpp.NativeTensorCore.contiguous_copy

    def _recording_copy(self):
        temp = original(self)
        made.append(temp)
        return temp

    monkeypatch.setattr(cpp.NativeTensorCore, "contiguous_copy",
                        _recording_copy)
    base = cpp.NativeTensorCore.from_array(LOGITS.T)
    strided = base.T
    # The injector counts allocations and the copy itself consumes more
    # than one, so sweep rather than assume a stage number.
    post_copy_failures = 0
    for nth in range(1, 7):
        made.clear()
        try:
            cpp._arm_alloc_failure(nth)
            result = strided.cross_entropy_forward(TARGETS, "mean")
        except MemoryError:
            cpp._arm_alloc_failure(0)
            if made:
                assert made[0]._closed, (
                    f"the Policy-B temporary leaked when allocation {nth} "
                    f"failed"
                )
                post_copy_failures += 1
        else:
            cpp._arm_alloc_failure(0)
            assert made and made[0]._closed
            result.close()
    assert post_copy_failures >= 1, (
        "the sweep never reached a failure after the contiguous copy"
    )
    monkeypatch.undo()
    assert np.array_equal(strided.to_numpy(), LOGITS)
    recovered = strided.cross_entropy_forward(TARGETS, "mean")
    assert np.isclose(float(recovered.loss.to_numpy()),
                      loss_reference(LOGITS, TARGETS, "mean"), atol=1e-14)
    recovered.close()
    base.close()


@needs_native
def test_native_cross_entropy_outputs_close_if_the_kernel_fails(monkeypatch):
    """A native call that raises after both outputs were allocated must
    discard both rather than returning a half-built result."""
    library = cpp._require_library()
    allocated = []
    original_zeros = cpp.NativeTensorCore.zeros

    def _recording_zeros(shape, **kwargs):
        core = original_zeros(shape, **kwargs)
        allocated.append(core)
        return core

    def _failing_kernel(*args, **kwargs):
        raise RuntimeError("simulated native cross_entropy failure")

    monkeypatch.setattr(cpp.NativeTensorCore, "zeros",
                        staticmethod(_recording_zeros))
    # H1: the enabled output-allocation sites construct through
    # _uninitialized, so the same probe must watch both
    # constructors for this test to still observe the real path.
    monkeypatch.setattr(cpp.NativeTensorCore, "_uninitialized",
                        staticmethod(_recording_zeros))
    monkeypatch.setattr(library, "tf_core_cross_entropy_forward",
                        _failing_kernel)
    core = cpp.NativeTensorCore.from_array(LOGITS)
    with pytest.raises(RuntimeError, match="simulated"):
        core.cross_entropy_forward(TARGETS, "mean")
    monkeypatch.undo()
    assert len(allocated) == 2, "the forward allocates the loss and the probabilities"
    assert all(out._closed for out in allocated), (
        "a freshly allocated output leaked when the kernel failed"
    )
    # The same for the backward's single output.
    allocated.clear()
    result = core.cross_entropy_forward(TARGETS, "mean")
    seed = scalar_core(1.0)          # allocated before the recorder is armed
    monkeypatch.setattr(cpp.NativeTensorCore, "zeros",
                        staticmethod(_recording_zeros))
    # H1: the enabled output-allocation sites construct through
    # _uninitialized, so the same probe must watch both
    # constructors for this test to still observe the real path.
    monkeypatch.setattr(cpp.NativeTensorCore, "_uninitialized",
                        staticmethod(_recording_zeros))
    monkeypatch.setattr(library, "tf_core_cross_entropy_backward",
                        _failing_kernel)
    with pytest.raises(RuntimeError, match="simulated"):
        result.probabilities.cross_entropy_backward(result.targets, seed, "mean")
    monkeypatch.undo()
    assert len(allocated) == 1 and allocated[0]._closed
    # The inputs are untouched and everything still works.
    assert np.array_equal(core.to_numpy(), LOGITS)
    seed.close()
    result.close()
    core.close()


@needs_native
def test_native_cross_entropy_second_allocation_failure_closes_the_first(
        monkeypatch):
    """If the probability allocation fails after the loss succeeded, the
    loss must be closed and no result object may escape."""
    allocated = []
    original_zeros = cpp.NativeTensorCore.zeros
    calls = {"n": 0}

    def _failing_second_zeros(shape, **kwargs):
        calls["n"] += 1
        if calls["n"] == 2:
            raise MemoryError("simulated second-allocation failure")
        core = original_zeros(shape, **kwargs)
        allocated.append(core)
        return core

    monkeypatch.setattr(cpp.NativeTensorCore, "zeros",
                        staticmethod(_failing_second_zeros))
    # H1: the enabled output-allocation sites construct through
    # _uninitialized, so the same probe must watch both
    # constructors for this test to still observe the real path.
    monkeypatch.setattr(cpp.NativeTensorCore, "_uninitialized",
                        staticmethod(_failing_second_zeros))
    core = cpp.NativeTensorCore.from_array(LOGITS)
    with pytest.raises(MemoryError, match="simulated"):
        core.cross_entropy_forward(TARGETS, "mean")
    monkeypatch.undo()
    assert len(allocated) == 1, "the loss was allocated first"
    assert allocated[0]._closed, "the first output leaked"
    assert np.array_equal(core.to_numpy(), LOGITS)
    core.close()


# ======================================================================
# Allocation failure
# ======================================================================


@needs_fault_injection
def test_native_cross_entropy_allocation_failure_is_atomic():
    """Sweep the injector across a contiguous forward and a backward.
    Every reachable stage must raise MemoryError, mutate nothing, return
    no partial result, and recover once disarmed. (The injector counts
    allocations rather than naming stages, so the sweep reports what is
    reachable instead of assuming a fixed mapping.)"""
    core = cpp.NativeTensorCore.from_array(LOGITS)
    forward_failures = 0
    for nth in range(1, 5):
        try:
            cpp._arm_alloc_failure(nth)
            result = core.cross_entropy_forward(TARGETS, "mean")
        except MemoryError:
            forward_failures += 1
            cpp._arm_alloc_failure(0)
            assert np.array_equal(core.to_numpy(), LOGITS), nth
        else:
            cpp._arm_alloc_failure(0)
            assert np.isclose(float(result.loss.to_numpy()),
                              loss_reference(LOGITS, TARGETS, "mean"),
                              atol=1e-14)
            result.close()
    assert forward_failures >= 2, (
        "the forward's two output allocations must both be reachable"
    )

    # The backward's single gradient allocation.
    result = core.cross_entropy_forward(TARGETS, "mean")
    seed = scalar_core(1.0)
    original = result.probabilities.to_numpy()
    with pytest.raises(MemoryError):
        cpp._arm_alloc_failure(1)
        result.probabilities.cross_entropy_backward(result.targets, seed, "mean")
    cpp._arm_alloc_failure(0)
    assert np.array_equal(result.probabilities.to_numpy(), original)
    assert seed.to_numpy().item() == 1.0
    assert result.targets.tolist() == TARGETS
    # Retrying after disarming succeeds, and the error state recovered.
    gradient = result.probabilities.cross_entropy_backward(result.targets, seed,
                                                           "mean")
    assert np.allclose(gradient.to_numpy(),
                       grad_reference(original, TARGETS, "mean", 1.0),
                       atol=1e-15)
    assert cpp._require_library().tf_last_error_code() == cpp.TF_OK
    gradient.close()
    seed.close()
    result.close()
    core.close()


# ======================================================================
# Raw ABI misuse (direct ctypes calls, bypassing the Core wrapper)
# ======================================================================


@needs_native
def test_native_cross_entropy_forward_abi_rejects_invalid_calls():
    """The export validates its own arguments, so a malformed direct
    ctypes call raises ValueError and leaves BOTH destinations
    unchanged."""
    library = cpp._require_library()
    logits = cpp.NativeTensorCore.from_array(LOGITS)
    loss = cpp.NativeTensorCore.from_array(np.array(7777.5))
    probabilities = cpp.NativeTensorCore.from_array(np.full((2, 3), 7777.5))
    targets = np.array([1, 2], dtype=np.int64)
    logits_handle = logits.storage._require_open()
    loss_handle = loss.storage._require_open()
    probability_handle = probabilities.storage._require_open()

    def call(offset=0, target_array=targets, count=2, loss_h=loss_handle,
             probability_h=probability_handle, batch=2, classes=3, code=0):
        library.tf_core_cross_entropy_forward(
            logits_handle, offset, target_array, count, loss_h, probability_h,
            batch, classes, code)

    for kwargs, message in (
        ({"code": 2}, "unknown reduction code"),
        ({"code": -1}, "unknown reduction code"),
        ({"count": 1}, "target count does not equal batch_size"),
        ({"count": 3, "target_array": np.array([1, 2, 0], dtype=np.int64)},
         "target count does not equal batch_size"),
        ({"offset": -1}, "negative source offset"),
        ({"batch": 0}, "must each be >= 1"),
        ({"classes": 0}, "must each be >= 1"),
        ({"batch": -2}, "must each be >= 1"),
        ({"classes": -3}, "must each be >= 1"),
        ({"batch": 4, "count": 4,
          "target_array": np.array([0, 1, 2, 0], dtype=np.int64)},
         "source span exceeds its storage"),
        ({"offset": 3}, "source span exceeds its storage"),
        ({"target_array": np.array([1, 3], dtype=np.int64)},
         "outside the class range"),
        ({"target_array": np.array([-1, 2], dtype=np.int64)},
         "outside the class range"),
    ):
        with pytest.raises(ValueError, match=message):
            call(**kwargs)

    # Null handles.
    with pytest.raises(ValueError, match="null storage handle"):
        call(loss_h=None)
    with pytest.raises(ValueError, match="null storage handle"):
        call(probability_h=None)
    # An under-capacity probability destination. (An under-capacity
    # *loss* destination is unreachable from Python — native storage is
    # always at least one element — so that rejection is driven from
    # cpp/tests/test_cross_entropy.cpp instead.)
    small_probabilities = cpp.NativeTensorCore.from_array(np.full(2, 7777.5))
    with pytest.raises(ValueError, match="probability storage smaller"):
        call(probability_h=small_probabilities.storage._require_open())
    # A null target pointer is refused by the ctypes binding itself: the
    # argument type is a contiguous int64 array, which is the honest
    # declaration for host metadata. (The C++ null check behind it is
    # driven from the CTest.)
    with pytest.raises(Exception):
        call(target_array=None)
    # Overflow in the product and in offset + numel.
    huge = 2 ** 62 + 4
    with pytest.raises(ValueError, match="overflows int64"):
        call(batch=huge, classes=huge, count=huge)
    with pytest.raises(ValueError, match="source span exceeds its storage"):
        call(offset=2 ** 63 - 1)
    # Aliasing destinations.
    with pytest.raises(ValueError, match="aliases another operand"):
        call(probability_h=logits_handle)
    # Every message is attributed to the forward.
    with pytest.raises(ValueError, match="cross_entropy_forward"):
        call(code=9)

    # Nothing was written by any rejected call.
    assert loss.to_numpy().item() == 7777.5
    assert np.array_equal(probabilities.to_numpy(), np.full((2, 3), 7777.5))
    assert np.array_equal(small_probabilities.to_numpy(), np.full(2, 7777.5))
    assert np.array_equal(logits.to_numpy(), LOGITS)
    # A following valid call succeeds and clears the error.
    call()
    assert np.isclose(loss.to_numpy().item(),
                      loss_reference(LOGITS, [1, 2], "mean"), atol=1e-15)
    assert np.allclose(probabilities.to_numpy(),
                       probabilities_reference(LOGITS), atol=1e-15)
    assert library.tf_last_error_code() == cpp.TF_OK
    for core in (logits, loss, probabilities, small_probabilities):
        core.close()


@needs_native
def test_native_cross_entropy_backward_abi_rejects_invalid_calls():
    library = cpp._require_library()
    probabilities = cpp.NativeTensorCore.from_array(
        probabilities_reference(LOGITS)
    )
    upstream = cpp.NativeTensorCore.from_array(np.array(1.0))
    gradient = cpp.NativeTensorCore.from_array(np.full((2, 3), 7777.5))
    targets = np.array([1, 2], dtype=np.int64)
    probability_handle = probabilities.storage._require_open()
    upstream_handle = upstream.storage._require_open()
    gradient_handle = gradient.storage._require_open()

    def call(offset=0, target_array=targets, count=2,
             upstream_h=upstream_handle, upstream_offset=0,
             gradient_h=gradient_handle, batch=2, classes=3, code=0):
        library.tf_core_cross_entropy_backward(
            probability_handle, offset, target_array, count, upstream_h,
            upstream_offset, gradient_h, batch, classes, code)

    for kwargs, message in (
        ({"code": 5}, "unknown reduction code"),
        ({"count": 3, "target_array": np.array([1, 2, 0], dtype=np.int64)},
         "target count does not equal batch_size"),
        ({"offset": -1}, "negative source offset"),
        ({"upstream_offset": -1}, "negative upstream offset"),
        ({"upstream_offset": 1}, "upstream span exceeds its storage"),
        ({"batch": 0}, "must each be >= 1"),
        ({"classes": -1}, "must each be >= 1"),
        ({"offset": 4}, "source span exceeds its storage"),
        ({"target_array": np.array([1, 9], dtype=np.int64)},
         "outside the class range"),
        ({"target_array": np.array([1, -2], dtype=np.int64)},
         "outside the class range"),
    ):
        with pytest.raises(ValueError, match=message):
            call(**kwargs)
    with pytest.raises(ValueError, match="null storage handle"):
        call(upstream_h=None)
    with pytest.raises(ValueError, match="null storage handle"):
        call(gradient_h=None)
    small = cpp.NativeTensorCore.from_array(np.full(2, 7777.5))
    with pytest.raises(ValueError, match="gradient storage smaller"):
        call(gradient_h=small.storage._require_open())
    huge = 2 ** 62 + 4
    with pytest.raises(ValueError, match="overflows int64"):
        call(batch=huge, classes=huge, count=huge)
    with pytest.raises(ValueError, match="aliases another operand"):
        call(gradient_h=probability_handle)
    with pytest.raises(ValueError, match="cross_entropy_backward"):
        call(code=9)

    # Nothing was written by any rejected call.
    assert np.array_equal(gradient.to_numpy(), np.full((2, 3), 7777.5))
    assert np.array_equal(small.to_numpy(), np.full(2, 7777.5))
    assert np.allclose(probabilities.to_numpy(),
                       probabilities_reference(LOGITS), atol=1e-15)
    assert upstream.to_numpy().item() == 1.0
    # A following valid call succeeds and clears the error.
    call()
    assert np.allclose(gradient.to_numpy(),
                       grad_reference(probabilities_reference(LOGITS), [1, 2],
                                      "mean", 1.0), atol=1e-15)
    assert library.tf_last_error_code() == cpp.TF_OK
    for core in (probabilities, upstream, gradient, small):
        core.close()


# The unavailable-backend contract for `cross_entropy_forward` —
# ImportError with build instructions, never a silent NumPy fallback — is
# proved in tests/test_native_backend_unavailable.py (the "cross_entropy"
# case). It used to live here and skip whenever the backend was built,
# which is every machine that can run this file at all; it now runs
# unconditionally, in a fresh child process that repoints only its *own*
# library path.


# ======================================================================
# Exceptional IEEE values
# ======================================================================


@needs_native
def test_native_cross_entropy_exceptional_values_follow_the_algorithm():
    """NaN and infinities are plain IEEE values produced by the same
    maximum-shift order the reference uses — not special cases."""
    nan_row = np.array([[1.0, np.nan, 2.0], [1.0, 2.0, 3.0]])
    core, result = run_forward(nan_row, [0, 2], "sum")
    probabilities = result.probabilities.to_numpy()
    assert np.isnan(probabilities[0]).all()
    assert np.isnan(float(result.loss.to_numpy()))
    # The clean row is unaffected.
    assert np.allclose(probabilities[1],
                       probabilities_reference(nan_row[1:]), atol=1e-15,
                       equal_nan=True)
    result.close()
    core.close()

    # +inf poisons its row through inf - inf.
    core, result = run_forward(np.array([[np.inf, 1.0, 2.0]]), [0], "sum")
    assert np.isnan(result.probabilities.to_numpy()).all()
    assert np.isnan(float(result.loss.to_numpy()))
    result.close()
    core.close()

    # -inf beside finite values simply takes zero probability; the finite
    # part still normalizes.
    core, result = run_forward(np.array([[-np.inf, 1.0, 2.0]]), [2], "sum")
    probabilities = result.probabilities.to_numpy()
    assert probabilities[0, 0] == 0.0
    assert np.isclose(probabilities[0, 1:].sum(), 1.0, atol=1e-15)
    assert np.isfinite(float(result.loss.to_numpy()))
    result.close()
    core.close()

    # ...but a -inf *target* logit gives an infinite loss.
    core, result = run_forward(np.array([[-np.inf, 1.0, 2.0]]), [0], "sum")
    assert float(result.loss.to_numpy()) == np.inf
    result.close()
    core.close()

    # An all -inf row is NaN (-inf - (-inf)).
    core, result = run_forward(np.array([[-np.inf, -np.inf]]), [0], "sum")
    assert np.isnan(result.probabilities.to_numpy()).all()
    assert np.isnan(float(result.loss.to_numpy()))
    result.close()
    core.close()


@needs_native
def test_native_cross_entropy_exceptional_values_are_not_abi_errors():
    """A structurally valid call producing NaN/inf leaves the native error
    state clear — those are values, not failures."""
    library = cpp._require_library()
    library.tf_clear_error()
    core, result = run_forward(np.array([[np.nan, np.inf, -np.inf]]), [1],
                               "mean")
    assert library.tf_last_error_code() == cpp.TF_OK
    seed = scalar_core(1.0)
    gradient = result.probabilities.cross_entropy_backward(result.targets, seed,
                                                           "mean")
    assert library.tf_last_error_code() == cpp.TF_OK
    assert gradient.shape == (1, 3)
    gradient.close()
    seed.close()
    result.close()
    core.close()


# ======================================================================
# NumPy tripwires
# ======================================================================


# Everything NumPy could plausibly be used to *compute* a cross-entropy
# with.
_NUMERICAL_NUMPY = (
    "max", "amax", "argmax", "exp", "log", "logaddexp", "sum", "divide",
    "true_divide", "add", "subtract", "multiply", "matmul", "mean",
    "negative", "power", "copyto", "take", "take_along_axis", "put",
    "put_along_axis", "where", "choose",
)
# Every route by which tensor *data* could enter or leave a NumPy host
# buffer. (np.array/np.asarray are deliberately absent here: the target
# copy legitimately builds int64 host metadata with them, and the
# instrumented test below proves that is all they ever receive.)
_DATA_NUMPY = ("empty", "frombuffer")


def _numpy_tripwire(monkeypatch, names):
    def _tripwire(*args, **kwargs):
        raise AssertionError("NumPy compute reached the native path")

    for name in names:
        monkeypatch.setattr(np, name, _tripwire)


def _data_conversion_tripwire(monkeypatch, extra=()):
    """Arm every tensor-data conversion route, including the Core-level
    ``to_numpy``/``from_array`` methods themselves."""
    def _tripwire(*args, **kwargs):
        raise AssertionError("tensor data was converted through NumPy")

    _numpy_tripwire(monkeypatch, _NUMERICAL_NUMPY + _DATA_NUMPY + tuple(extra))
    monkeypatch.setattr(cpp.NativeTensorCore, "to_numpy", _tripwire)
    monkeypatch.setattr(cpp.NativeTensorCore, "from_array",
                        staticmethod(_tripwire))
    monkeypatch.setattr(cpp.NativeTensorView, "to_numpy", _tripwire)
    monkeypatch.setattr(cpp.NativeTensorView, "contiguous_copy", _tripwire)
    monkeypatch.setattr(cpp.NativeStorage, "from_array", staticmethod(_tripwire))
    monkeypatch.setattr(cpp.NativeStorage, "to_numpy", _tripwire)


@needs_native
def test_native_cross_entropy_core_paths_use_no_numpy_compute(monkeypatch):
    """Strict tripwire over the contiguous forward and the backward: no
    NumPy arithmetic, no numerical indexing fallback, and no tensor-data
    conversion. (``np.array`` stays available only because the int64
    target copy is built with it — pinned by the next test.)"""
    core = cpp.NativeTensorCore.from_array(LOGITS)
    seed = scalar_core(1.5)

    _data_conversion_tripwire(monkeypatch)
    result = core.cross_entropy_forward(TARGETS, "mean")
    gradient = result.probabilities.cross_entropy_backward(
        result.targets, seed, result.reduction
    )
    monkeypatch.undo()

    assert np.isclose(float(result.loss.to_numpy()),
                      loss_reference(LOGITS, TARGETS, "mean"), atol=1e-14)
    assert np.allclose(gradient.to_numpy(),
                       grad_reference(probabilities_reference(LOGITS), TARGETS,
                                      "mean", 1.5), atol=1e-15)
    gradient.close()
    seed.close()
    result.close()
    core.close()


@needs_native
def test_native_cross_entropy_policy_b_path_uses_no_numpy_compute(monkeypatch):
    """The **same strict** tripwire for the non-contiguous path: the
    Policy-B copy is a native storage-to-storage gather (E3.1), so a
    strided cross-entropy keeps tensor data in native memory end to
    end."""
    base = cpp.NativeTensorCore.from_array(LOGITS.T)
    strided = base.T
    assert not strided.contiguous
    _data_conversion_tripwire(monkeypatch)
    result = strided.cross_entropy_forward(TARGETS, "sum")
    monkeypatch.undo()
    assert np.isclose(float(result.loss.to_numpy()),
                      loss_reference(LOGITS, TARGETS, "sum"), atol=1e-14)
    result.close()
    base.close()


@needs_native
def test_native_cross_entropy_backward_blocks_every_numpy_constructor(
        monkeypatch):
    """The stricter second tripwire: with the target copy already
    prepared, the backward needs **no** NumPy array construction at all,
    so ``np.array``/``np.asarray`` can be blocked outright."""
    core = cpp.NativeTensorCore.from_array(LOGITS)
    result = core.cross_entropy_forward(TARGETS, "mean")
    seed = scalar_core(1.0)
    saved_targets = result.targets

    _data_conversion_tripwire(monkeypatch, extra=("array", "asarray", "zeros",
                                                  "copy", "ascontiguousarray"))
    gradient = result.probabilities.cross_entropy_backward(
        saved_targets, seed, "mean"
    )
    monkeypatch.undo()
    assert np.allclose(gradient.to_numpy(),
                       grad_reference(probabilities_reference(LOGITS), TARGETS,
                                      "mean", 1.0), atol=1e-15)
    gradient.close()
    seed.close()
    result.close()
    core.close()


@needs_native
def test_native_cross_entropy_numpy_construction_is_targets_and_metadata_only(
        monkeypatch):
    """The deliberate boundary: ``np.array``/``np.asarray`` *are* used —
    to build the owned int64 target copy and to marshal shape/stride
    arrays for ctypes. This test pins that distinction so the tripwires
    above cannot be quietly widened into a false claim: every value handed
    to a NumPy constructor on the cross-entropy path is a small tuple or
    list of Python ints, never a float or a tensor value."""
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
    base = cpp.NativeTensorCore.from_array(LOGITS.T)
    result = base.T.cross_entropy_forward(TARGETS, "sum")   # Policy B too
    monkeypatch.undo()

    assert seen, "the cross_entropy path builds int64 metadata through NumPy"
    for value in seen:
        assert isinstance(value, (tuple, list)), value
        assert all(isinstance(item, int) and not isinstance(item, bool)
                   for item in value), value
        # Target labels are batch-sized; shape/stride tuples are
        # rank-sized. Either way this is integer metadata, never the
        # float64 tensor data.
        assert len(value) <= max(LOGITS.shape), value
    result.close()
    base.close()


# ======================================================================
# Scope guardrails
# ======================================================================


@needs_native
def test_native_cross_entropy_scope_boundaries_hold():
    """E5 is the cross-entropy **Core contract** only: no later Phase-E
    surface, no new public operation, and the stable framework is
    untouched."""
    import tensorforge.experimental as experimental

    x = NativeTensor.from_array(LOGITS)
    core = cpp.NativeTensorCore.from_array(LOGITS)
    # (``argmax`` left this list at Phase K milestone K3, which shipped it
    # as a general index-producing reduction. ``max`` and ``amax`` stay
    # banned and always will: a kernel that finds the position of a
    # maximum necessarily knows the maximum, and Phase K deliberately does
    # not expose it — design §17.10.)
    for absent in ("max", "amax", "divide", "gather",
                   "scatter", "sigmoid", "tanh", "nll_loss", "one_hot",
                   "binary_cross_entropy"):
        assert not hasattr(x, absent), absent
        assert not hasattr(core, absent), absent
    # The bare name never joined the Core surface: E6 added it on
    # NativeTensor, over these layer-qualified wrappers.
    assert not hasattr(core, "cross_entropy")
    assert not hasattr(x, "__truediv__")
    # (`NativeCrossEntropyLoss` and `native_accuracy` left this list at
    # E7, which shipped both as Python surfaces over this Core contract.)
    for absent in ("NativeNLLLoss", "NativeSoftmax", "NativeLogSoftmax"):
        assert not hasattr(experimental, absent), absent
    # No integer tensors, no new dtype/device.
    assert cpp.SUPPORTED_DTYPES == ("float64", "float32")
    assert cpp.SUPPORTED_DEVICES == ("cpu",)
    with pytest.raises(ValueError):
        cpp.NativeTensorCore.zeros((2, 2), dtype="int64")
    # The stable framework keeps its own cross-entropy, entirely
    # separately, and gained nothing from this milestone.
    logits = tensorforge.Tensor(LOGITS, requires_grad=True)
    loss = tensorforge.nn.cross_entropy(logits, [1, 2])
    loss.backward()
    assert type(loss) is tensorforge.Tensor
    assert logits.grad is not None
    # No implicit dispatch in either direction.
    with pytest.raises((TypeError, AttributeError)):
        core.cross_entropy_forward(tensorforge.Tensor(np.array([1, 2])), "mean")
    core.close()


@needs_native
def test_native_cross_entropy_checkpoint_schema_is_untouched():
    """E5 adds no persistent state: the native checkpoint format version
    is still 1, and the saved probabilities and targets are graph data
    that never reach a state dict."""
    from tensorforge.experimental import (NativeLinear, native_checkpoint)

    assert native_checkpoint._FORMAT_VERSION == 3
    model = NativeLinear(3, 2, seed=0)
    state = model.state_dict()
    for key in state:
        assert "cross_entropy" not in key and "probabilit" not in key
        assert "target" not in key
