"""Phase-E completion guardrails: the invariants that span the whole
native classification stack at once (Advanced C++ Phase E, milestone
E10).

The per-milestone suites already cover `exp`/`log` (E1–E2), `softmax` and
`log_softmax` (E3–E4), the fused cross-entropy Core contract (E5), the
differentiable operation over it (E6), `NativeCrossEntropyLoss` and
`native_accuracy` (E7), the deterministic training + exact-resume proof
(E8), and the characterization benchmark (E9) in depth. This file
deliberately tests only what those files cannot: the **interactions** —
one graph carrying stable math, a convolutional stack, and the fused loss
together; the phase's differing versioning contracts meeting in one
backward; the saved-probability lifetime under a live graph and a
reporting metric; Policy-B strided inputs reaching the fused kernels
through a real model; state/checkpoint integration for the finished
classifier; cross-layer failure atomicity and error-state recovery; and
the final capability boundary of the closed phase.

Nothing here adds numerical behavior, and nothing here depends on one
implementation being faster than another. Every assertion is a property
the architecture promises, not an implementation detail.

Selector: python -m pytest -q -k native_phase_e
"""

import gc
import json
import math
from pathlib import Path

import numpy as np
import pytest

import tensorforge
from tensorforge.backends import cpp
from tensorforge.experimental import (
    NativeAdam,
    NativeConv2d,
    NativeCrossEntropyLoss,
    NativeFlatten,
    NativeLinear,
    NativeMaxPool2d,
    NativeModule,
    NativeMSELoss,
    NativeParameter,
    NativeReLU,
    NativeSequential,
    NativeSGD,
    NativeTensor,
    load_native_checkpoint,
    native_accuracy,
    save_native_checkpoint,
)
from tensorforge.experimental import native_checkpoint

pytestmark = pytest.mark.skipif(
    not cpp.is_available(),
    reason="experimental C++ backend not built; " + cpp.build_instructions(),
)

needs_fault_injection = pytest.mark.skipif(
    not (cpp.is_available() and cpp.fault_injection_available()),
    reason="fault injection not compiled into the backend",
)

REPO_ROOT = Path(__file__).resolve().parent.parent

BATCH, IN_CHANNELS, HEIGHT, WIDTH = 6, 1, 6, 6
CONV_CHANNELS = 3
FEATURES = CONV_CHANNELS * 2 * 2
NUM_CLASSES = 3


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
    lifetime (the Phase-C/D precedent)."""
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
# Shared fixtures for the integrated classification stack
# --------------------------------------------------------------------------

def _classifier(seed=0):
    """The canonical Phase-E model: the whole Phase-D layer set feeding a
    linear head whose **raw logits** go to the classification loss."""
    return NativeSequential(
        NativeConv2d(IN_CHANNELS, CONV_CHANNELS, 3, seed=seed),
        NativeReLU(),
        NativeMaxPool2d(2),
        NativeFlatten(),
        NativeLinear(FEATURES, NUM_CLASSES, seed=seed + 1),
    )


def _images(seed=11, requires_grad=False):
    values = np.round(
        np.random.default_rng(seed).standard_normal(
            (BATCH, IN_CHANNELS, HEIGHT, WIDTH)
        ),
        3,
    )
    return NativeTensor.from_array(values, requires_grad=requires_grad), values


def _labels():
    """Strict host integer targets — the native runtime has no integer
    dtype, and that is exactly why they stay on the host."""
    return [0, 1, 2, 0, 1, 2]


def _close_all(*objects):
    """Release anything a test owns. ``NativeSGD`` deliberately has no
    ``close()`` — it holds no optimizer state — so the closer is called
    only where one exists."""
    for item in objects:
        if item is None:
            continue
        if isinstance(item, NativeModule):
            for parameter in item.parameters():
                parameter.close()
        elif hasattr(item, "close"):
            item.close()


def _step(model, loss_fn, optimizer, x, targets):
    logits = model(x)
    loss = loss_fn(logits, targets)
    try:
        value = float(loss.to_numpy())
        loss.backward()
        optimizer.step()
    finally:
        loss.close()
        logits.close()
    optimizer.zero_grad()
    return value


# --------------------------------------------------------------------------
# 1. The public surface and the honest scope
# --------------------------------------------------------------------------

def test_every_phase_e_export_imports_and_is_callable():
    import tensorforge.experimental as experimental

    for name in ("NativeCrossEntropyLoss", "native_accuracy"):
        assert name in experimental.__all__, name
        assert callable(getattr(experimental, name)), name
    for op in ("exp", "log", "softmax", "log_softmax", "cross_entropy"):
        assert callable(getattr(NativeTensor, op)), op
    for core_op in ("exp", "log", "softmax", "log_softmax",
                    "cross_entropy_forward", "cross_entropy_backward"):
        assert callable(getattr(cpp.NativeTensorCore, core_op)), core_op


def test_backend_scope_is_reported_honestly():
    info = cpp.backend_info()
    assert info["available"] is True
    assert info["dtype"] == "float64" and info["device"] == "cpu"
    assert info["supported_dtypes"] == ("float64",)
    assert info["supported_devices"] == ("cpu",)
    assert info["native_autograd"] is True
    assert info["stable_framework_integration"] is False
    # The metric inventory is reported, and it holds exactly the one
    # reporting helper — never a kernel or an operation.
    assert info["native_metrics"] == cpp.NATIVE_METRICS == ("native_accuracy",)


def test_capability_inventories_are_internally_consistent():
    import tensorforge.experimental as experimental

    for op in cpp.TENSOR_CORE_OPS:
        assert hasattr(cpp.NativeTensorCore, op), op
    for op in cpp.AUTOGRAD_OPS:
        assert hasattr(NativeTensor, op), op
    for name in cpp.NATIVE_MODULES:
        if name != "NativeModule":
            assert name in experimental.__all__, name
    for name in cpp.NATIVE_LOSSES + cpp.NATIVE_METRICS + cpp.NATIVE_OPTIMIZERS:
        assert hasattr(experimental, name), name
    for name in cpp.RAW_KERNELS:
        assert callable(getattr(cpp, name)), name
    assert hasattr(NativeModule, "state_dict")
    assert hasattr(NativeModule, "load_state_dict")
    # Every advertised state capability maps to something real. Most
    # name a callable directly; the three *capability* names are resolved
    # explicitly rather than by relaxing the check —
    # "persistent_buffers" (Phase F milestone F1, reconciling a
    # capability that already existed) names the register_buffer /
    # buffers / named_buffers API, "generator_state" (Phase G milestone
    # G1) names the generator registration and in-memory state pair, and
    # "checkpoint_generator_state" (Phase G milestone G5) names the file
    # half — the existing save/load pair, which really does persist and
    # restore generator state through the version-2 manifest.
    _STATE_CAPABILITY_API = {
        "persistent_buffers": ("register_buffer", "buffers", "named_buffers"),
        "generator_state": (
            "register_generator", "generators", "named_generators",
            "generator_state_dict", "load_generator_state_dict",
        ),
        "checkpoint_generator_state": (
            "save_native_checkpoint", "load_native_checkpoint",
        ),
    }
    for name in cpp.STATE_SUPPORT:
        for attribute in _STATE_CAPABILITY_API.get(name, (name,)):
            assert (hasattr(experimental, attribute)
                    or hasattr(NativeModule, attribute)), (name, attribute)
    # Implemented and unsupported names stay disjoint everywhere, with
    # the one deliberate exception Phase G locks (design §19): milestone
    # G3 shipped the differentiable "dropout" operation while the
    # *capability* stays unsupported until the G10 closure. No Phase-E
    # name is involved, which is exactly what this asserts.
    implemented = (set(cpp.TENSOR_CORE_OPS) | set(cpp.AUTOGRAD_OPS)
                   | set(cpp.RAW_KERNELS) | set(cpp.NATIVE_MODULES)
                   | set(cpp.NATIVE_LOSSES) | set(cpp.NATIVE_METRICS)
                   | set(cpp.NATIVE_OPTIMIZERS))
    assert implemented & set(cpp.UNSUPPORTED) == {"dropout"}


def test_the_phase_e_capability_set_is_exactly_what_shipped():
    for core_op in ("exp", "log", "softmax", "log_softmax",
                    "cross_entropy_forward", "cross_entropy_backward"):
        assert core_op in cpp.TENSOR_CORE_OPS, core_op
    for op in ("exp", "log", "softmax", "log_softmax", "cross_entropy"):
        assert op in cpp.AUTOGRAD_OPS, op
    # The Core/autograd split stays layer-qualified in both directions.
    assert "cross_entropy" not in cpp.TENSOR_CORE_OPS
    assert not hasattr(cpp.NativeTensorCore, "cross_entropy")
    for core_only in ("cross_entropy_forward", "cross_entropy_backward"):
        assert core_only not in cpp.AUTOGRAD_OPS, core_only
    assert cpp.NATIVE_LOSSES == ("NativeMSELoss", "NativeCrossEntropyLoss")
    assert cpp.NATIVE_METRICS == ("native_accuracy",)
    # No classification kernel became a raw NumPy-buffer kernel, and the
    # frozen historical registry never grew.
    assert cpp.TENSOR_CORE_KERNELS == ("relu", "add", "subtract",
                                       "multiply", "matmul")
    for name in cpp.RAW_KERNELS:
        for phase_e in ("softmax", "cross_entropy", "exp", "log"):
            assert phase_e not in name, (name, phase_e)


def test_unsupported_stays_honest_after_closure():
    # Nothing Phase E shipped may still be called unsupported...
    for shipped in ("exp", "log", "softmax", "log_softmax", "cross_entropy",
                    "cross_entropy_forward", "cross_entropy_backward",
                    "NativeCrossEntropyLoss", "native_accuracy"):
        assert shipped not in cpp.UNSUPPORTED, shipped
    # ...and nothing still unimplemented may quietly disappear because the
    # phase closed. These are the boundaries Phase E deliberately kept.
    # ("layernorm" left UNSUPPORTED in Phase F milestone F2 and
    # "batchnorm" in F4, once both BatchNorm shapes shipped as composed
    # modules. Neither was ever a Phase-E boundary.)
    for absent in ("float32", "cuda", "amp"):
        assert absent in cpp.UNSUPPORTED, absent
        assert absent not in cpp.AUTOGRAD_OPS
        assert absent not in cpp.TENSOR_CORE_OPS
        assert absent not in cpp.NATIVE_MODULES
    # "dropout" is still an unsupported *capability* — a boundary Phase E
    # kept and Phase G has not yet moved — even though Phase G milestones
    # G2 and G3 shipped a Core wrapper and a differentiable operation
    # underneath it. It leaves UNSUPPORTED only at the G10 closure.
    assert "dropout" in cpp.UNSUPPORTED
    assert "dropout" not in cpp.NATIVE_MODULES
    import tensorforge.experimental as experimental

    # Classification extensions that were explicitly out of scope never
    # appeared as exports or operations.
    # (Both BatchNorm shapes are deliberately absent from this list:
    # Phase F milestones F3 and F4 shipped them, which is unrelated to
    # Phase E's scope.)
    # ("NativeDropout" is deliberately absent from this list: Phase G
    # milestone G4 shipped it, which is as unrelated to Phase E's scope as
    # the two BatchNorm shapes above.)
    for never in ("NativeNLLLoss", "NativeBCELoss", "NativeSoftmax",
                  "NativeLogSoftmax", "native_top_k_accuracy",
                  "native_confusion_matrix", "NativeDataLoader",
                  "NativeBatchNorm3d"):
        assert not hasattr(experimental, never), never
    for never in ("argmax", "nll_loss", "one_hot", "randn", "rand"):
        assert not hasattr(NativeTensor, never), never
        assert not hasattr(cpp.NativeTensorCore, never), never
    for absent in ("tf_core_accuracy", "tf_core_argmax", "tf_core_nll_loss",
                   "tf_core_softmax_backward", "tf_core_log_softmax_backward"):
        assert absent not in cpp._CHECKED_KERNELS, absent
    # No integer dtype and no second device appeared for targets.
    assert cpp.SUPPORTED_DTYPES == ("float64",)
    assert cpp.SUPPORTED_DEVICES == ("cpu",)


def test_no_out_of_scope_loss_option_exists():
    for kwargs in ({"weight": None}, {"ignore_index": 0},
                   {"label_smoothing": 0.1}):
        with pytest.raises(TypeError):
            NativeCrossEntropyLoss(**kwargs)
    with pytest.raises(ValueError):
        NativeCrossEntropyLoss("none")          # reduction="none" stays out
    x = NativeTensor.from_array(np.zeros((2, 3)))
    with pytest.raises(ValueError):
        x.cross_entropy([0, 1], reduction="none")
    x.close()


# --------------------------------------------------------------------------
# 2. Stable and native lines stay separate
# --------------------------------------------------------------------------

def test_stable_and_native_lines_stay_separate():
    import tensorforge.experimental as experimental

    for name in experimental.__all__:
        assert not hasattr(tensorforge, name), name
        assert not hasattr(tensorforge.nn, name), name
    # The stable cross-entropy is a separate function that never gained a
    # native path, and the native loss never gained a stable one.
    assert callable(tensorforge.nn.cross_entropy)
    assert not hasattr(tensorforge.nn.cross_entropy, "native")
    assert not hasattr(NativeCrossEntropyLoss, "to_stable")
    assert not hasattr(NativeTensor, "to_tensor")
    assert not hasattr(tensorforge.Tensor, "to_native")


def test_no_implicit_dispatch_or_conversion_between_the_lines():
    stable = tensorforge.Tensor(np.zeros((2, NUM_CLASSES)))
    native = NativeTensor.from_array(np.zeros((2, NUM_CLASSES)))
    targets = [0, 1]
    # A stable Tensor is rejected by the native loss and metric...
    with pytest.raises(TypeError):
        NativeCrossEntropyLoss()(stable, targets)
    with pytest.raises(TypeError):
        native_accuracy(stable, targets)
    # ...and a NativeTensor is never silently accepted by the stable loss.
    with pytest.raises(Exception):
        tensorforge.nn.cross_entropy(native, targets)
    # Mixing the two lines in one expression never converts implicitly.
    with pytest.raises(TypeError):
        native.multiply(stable)
    native.close()


def test_stable_framework_behavior_is_unchanged():
    """A representative stable classification path still behaves exactly
    as the stable line always has — Phase E changed nothing about it."""
    logits = tensorforge.Tensor(
        np.array([[2.0, 1.0, 0.1], [0.5, 2.5, 0.2]]), requires_grad=True
    )
    loss = tensorforge.nn.cross_entropy(logits, [0, 1])
    loss.backward()
    shifted = logits.data - logits.data.max(axis=1, keepdims=True)
    log_probs = shifted - np.log(np.exp(shifted).sum(axis=1, keepdims=True))
    expected = float(-log_probs[[0, 1], [0, 1]].mean())
    assert float(loss.data) == pytest.approx(expected, rel=1e-12)
    probabilities = np.exp(log_probs)
    probabilities[[0, 1], [0, 1]] -= 1.0
    assert np.allclose(logits.grad, probabilities / 2.0, atol=1e-15)
    assert float(tensorforge.nn.accuracy(logits, [0, 1])) == 1.0


# --------------------------------------------------------------------------
# 3. The complete classification mathematical path
# --------------------------------------------------------------------------

def test_one_graph_carries_the_whole_classification_path():
    """conv -> relu -> pool -> flatten -> linear -> fused cross-entropy,
    one backward, one optimizer step: every trainable parameter receives a
    finite gradient and every value moves."""
    model = _classifier()
    loss_fn = NativeCrossEntropyLoss()
    optimizer = NativeAdam(model.parameters(), lr=0.05)
    x, _ = _images()
    targets = _labels()
    before = {n: p.to_numpy().copy() for n, p in model.named_parameters()}

    logits = model(x)
    assert logits.shape == (BATCH, NUM_CLASSES)
    loss = loss_fn(logits, targets)
    assert loss.shape == () and loss.numel == 1
    assert math.isfinite(float(loss.to_numpy()))
    loss.backward()
    for name, parameter in model.named_parameters():
        assert parameter.grad is not None, name
        assert parameter.grad.shape == parameter.shape, name
        values = parameter.grad.to_numpy()
        assert np.isfinite(values).all(), name
        assert (values != 0.0).any(), name
    optimizer.step()
    optimizer.zero_grad()
    for name, parameter in model.named_parameters():
        assert not np.array_equal(parameter.to_numpy(), before[name]), name
    _close_all(loss, logits, x, model, optimizer)


def test_stable_math_remains_usable_around_a_classification_workload():
    """`exp`, `log`, `softmax`, and `log_softmax` still compose with the
    classification path — including in the same graph as the fused loss,
    which is the interaction no single-milestone suite covers."""
    x, values = _images(requires_grad=False)
    logits_source = NativeTensor.from_array(
        np.round(np.random.default_rng(3).standard_normal((BATCH, NUM_CLASSES)), 3),
        requires_grad=True,
    )
    reference = logits_source.to_numpy()

    probabilities = logits_source.softmax(axis=-1)
    log_probabilities = logits_source.log_softmax(axis=-1)
    exponentials = logits_source.exp()
    logarithms = exponentials.log()

    shifted = reference - reference.max(axis=-1, keepdims=True)
    expected_p = np.exp(shifted) / np.exp(shifted).sum(axis=-1, keepdims=True)
    assert np.allclose(probabilities.to_numpy(), expected_p, atol=1e-14)
    assert np.allclose(log_probabilities.to_numpy(), np.log(expected_p),
                       atol=1e-13)
    assert np.allclose(logarithms.to_numpy(), reference, atol=1e-13)

    # The fused loss agrees with the log-softmax path it never composes.
    loss = logits_source.cross_entropy(_labels(), reduction="mean")
    manual = float(-np.log(expected_p)[np.arange(BATCH), _labels()].mean())
    assert float(loss.to_numpy()) == pytest.approx(manual, rel=1e-12)
    loss.backward()
    assert logits_source.grad is not None
    assert np.isfinite(logits_source.grad.to_numpy()).all()
    _close_all(loss, logarithms, exponentials, log_probabilities,
               probabilities, logits_source, x)


def test_cross_entropy_takes_raw_logits_and_strict_host_integer_targets():
    model = _classifier()
    loss_fn = NativeCrossEntropyLoss()
    x, _ = _images()
    logits = model(x)
    values = logits.to_numpy()
    # Raw logits: not normalized, and no probability transform ran.
    assert not np.allclose(values.sum(axis=1), 1.0)
    loss = loss_fn(logits, _labels())
    assert math.isfinite(float(loss.to_numpy()))
    # The strict target contract holds through the module.
    for bad in ([0.0] * BATCH, [True] * BATCH, [0] * (BATCH - 1),
                [0] * (BATCH - 1) + [NUM_CLASSES], "012345"):
        with pytest.raises((TypeError, ValueError, IndexError)):
            loss_fn(logits, bad)
    # A rejected call changed nothing.
    again = loss_fn(logits, _labels())
    assert float(again.to_numpy()) == float(loss.to_numpy())
    _close_all(again, loss, logits, x, model)


# --------------------------------------------------------------------------
# 4. Autograd, saved probabilities, and the versioning contracts
# --------------------------------------------------------------------------

def test_saved_probabilities_and_winners_are_released_by_one_backward():
    model = _classifier()
    loss_fn = NativeCrossEntropyLoss()
    x, _ = _images()
    conv = model[0](x)
    relu = model[1](conv)
    pooled = model[2](relu)
    winners = pooled._graph_resources
    assert winners, "pooling should own a saved winner buffer"
    flat = model[3](pooled)
    logits = model[4](flat)
    loss = loss_fn(logits, _labels())
    probabilities = loss._graph_resources
    assert probabilities, "cross-entropy should own its saved probabilities"
    assert all(not core._closed for core in probabilities + winners)

    loss.backward()

    assert all(core._closed for core in probabilities), "saved probabilities"
    assert all(core._closed for core in winners), "saved winners"
    assert loss._graph_resources == () and pooled._graph_resources == ()
    assert loss._graph_freed is True
    with pytest.raises(RuntimeError, match="freed autograd graph"):
        loss.backward()
    _close_all(loss, logits, flat, pooled, relu, conv, x, model)


def test_cross_entropy_records_no_expected_version_and_survives_mutation():
    """The phase's central versioning asymmetry, checked end to end: the
    fused loss reads only its saved probabilities, so a post-forward
    parameter mutation cannot invalidate its backward — while `log`, which
    rereads its live input, still raises."""
    parameter = NativeParameter(np.array([[1.0, 2.0, 0.5], [0.2, 0.1, 3.0]]))
    loss = parameter.cross_entropy([1, 2], reduction="mean")
    assert loss._expected_versions == ()
    snapshot = parameter.to_numpy().copy()
    parameter.copy_value_(NativeTensor.from_array(snapshot + 5.0))
    loss.backward()                      # still correct for the forward that ran
    shifted = snapshot - snapshot.max(axis=1, keepdims=True)
    probabilities = np.exp(shifted) / np.exp(shifted).sum(axis=1, keepdims=True)
    probabilities[[0, 1], [1, 2]] -= 1.0
    assert np.allclose(parameter.grad.to_numpy(), probabilities / 2.0,
                       atol=1e-15)
    loss.close()
    parameter.zero_grad()

    logarithmic = NativeParameter(np.array([[2.0, 3.0]]))
    out = logarithmic.log().sum()
    logarithmic.copy_value_(NativeTensor.from_array(np.array([[5.0, 7.0]])))
    with pytest.raises(RuntimeError, match="stale parameter value"):
        out.backward()
    _close_all(out, logarithmic, parameter)


def test_mixed_graph_versioning_is_per_operation():
    """One graph holding a version-checked `log` edge and a saved-state
    cross-entropy edge keeps each contract independently."""
    logits = NativeParameter(np.array([[1.0, 0.5, 0.25], [0.75, 2.0, 0.5]]))
    scale = NativeParameter(np.array([[2.0, 4.0]]))
    fused = logits.cross_entropy([0, 1], reduction="sum")
    logarithmic = scale.log().sum()
    total = fused.add(logarithmic)
    # Mutating the *log* parent invalidates the graph; the cross-entropy
    # edge would have been fine on its own.
    scale.copy_value_(NativeTensor.from_array(np.array([[8.0, 16.0]])))
    with pytest.raises(RuntimeError, match="stale parameter value"):
        total.backward()
    assert logits.grad is None and scale.grad is None   # nothing committed
    _close_all(total, logarithmic, fused, scale, logits)


def test_retain_graph_holds_the_saved_probabilities_until_release():
    logits = NativeTensor.from_array(
        np.array([[1.0, 2.0, 0.5], [0.5, 0.25, 1.5]]), requires_grad=True
    )
    loss = logits.cross_entropy([1, 0], reduction="mean")
    saved = loss._graph_resources
    assert saved
    loss.backward(retain_graph=True)
    assert all(not core._closed for core in saved)
    first = logits.grad.to_numpy().copy()
    logits.zero_grad()
    loss.backward()                       # the releasing pass
    assert all(core._closed for core in saved)
    assert np.array_equal(logits.grad.to_numpy(), first)
    _close_all(loss, logits)


# --------------------------------------------------------------------------
# 5. Policy-B (non-contiguous) inputs through the classification stack
# --------------------------------------------------------------------------

def test_policy_b_inputs_reach_every_fused_classification_kernel():
    """A transposed (strided) view is copied natively before the fused
    kernels run, so softmax, log-softmax, and cross-entropy all produce
    the contiguous answer without mutating the caller's view."""
    base_values = np.round(
        np.random.default_rng(5).standard_normal((NUM_CLASSES, BATCH)), 3
    )
    base = NativeTensor.from_array(base_values, requires_grad=True)
    view = base.T
    expected = base_values.T
    assert view.shape == (BATCH, NUM_CLASSES)

    probabilities = view.softmax(axis=-1)
    log_probabilities = view.log_softmax(axis=-1)
    shifted = expected - expected.max(axis=1, keepdims=True)
    reference = np.exp(shifted) / np.exp(shifted).sum(axis=1, keepdims=True)
    assert np.allclose(probabilities.to_numpy(), reference, atol=1e-14)
    assert np.allclose(log_probabilities.to_numpy(), np.log(reference),
                       atol=1e-13)

    loss = view.cross_entropy(_labels(), reduction="mean")
    manual = float(-np.log(reference)[np.arange(BATCH), _labels()].mean())
    assert float(loss.to_numpy()) == pytest.approx(manual, rel=1e-12)
    loss.backward()
    # The gradient lands on the base in the base's own layout.
    assert base.grad.shape == base.shape
    reference_grad = reference.copy()
    reference_grad[np.arange(BATCH), _labels()] -= 1.0
    assert np.allclose(base.grad.to_numpy(), (reference_grad / BATCH).T,
                       atol=1e-14)
    # The caller's data was never mutated by the Policy-B copy.
    assert np.array_equal(base.to_numpy(), base_values)
    _close_all(loss, log_probabilities, probabilities, view, base)


def test_policy_b_metric_and_loss_agree_with_the_contiguous_form():
    values = np.round(np.random.default_rng(9).standard_normal(
        (NUM_CLASSES, BATCH)), 3)
    # The base must outlive the view: `.T` borrows its owner's storage,
    # which is the documented ownership rule, not an incidental detail.
    base = NativeTensor.from_array(values)
    strided = base.T
    contiguous = NativeTensor.from_array(values.T)
    targets = _labels()
    assert native_accuracy(strided, targets) == native_accuracy(contiguous,
                                                                targets)
    a = strided.cross_entropy(targets, reduction="sum")
    b = contiguous.cross_entropy(targets, reduction="sum")
    assert float(a.to_numpy()) == pytest.approx(float(b.to_numpy()), rel=1e-14)
    _close_all(a, b, strided, contiguous, base)


# --------------------------------------------------------------------------
# 6. The loss module and the reporting metric
# --------------------------------------------------------------------------

def test_loss_module_is_stateless_inside_a_real_model():
    loss_fn = NativeCrossEntropyLoss()
    assert list(loss_fn.parameters()) == []
    assert list(loss_fn.buffers()) == []
    assert loss_fn.state_dict() == {}
    # Even inside a container, it contributes no state and no keys.
    model = _classifier()
    x, _ = _images()
    before = sorted(model.state_dict())
    for _ in range(3):
        out = loss_fn(model(x), _labels())
        out.close()
    assert loss_fn.state_dict() == {}
    assert sorted(model.state_dict()) == before
    # train()/eval() propagate but never change the numbers.
    logits = model(x)
    loss_fn.train()
    trained = float(loss_fn(logits, _labels()).to_numpy())
    loss_fn.eval()
    evaluated = float(loss_fn(logits, _labels()).to_numpy())
    assert trained == evaluated
    _close_all(logits, x, model)


def test_reporting_metric_is_a_float_and_leaves_the_graph_intact():
    model = _classifier()
    loss_fn = NativeCrossEntropyLoss()
    x, _ = _images()
    targets = _labels()
    logits = model(x)
    loss = loss_fn(logits, targets)

    accuracy = native_accuracy(logits, targets)
    assert type(accuracy) is float and 0.0 <= accuracy <= 1.0
    # The metric touched no gradient, version, or graph state...
    assert logits.grad is None
    assert loss._graph_resources != ()
    versions = [p.version for p in model.parameters()]

    # ...and the graph it observed still differentiates correctly.
    loss.backward()
    assert all(p.grad is not None for p in model.parameters())
    assert [p.version for p in model.parameters()] == versions
    # Calling it after backward still works and still changes nothing.
    after = native_accuracy(logits, targets)
    assert after == accuracy
    _close_all(loss, logits, x, model)


# --------------------------------------------------------------------------
# 7. Deterministic training, checkpoints, and exact resume
# --------------------------------------------------------------------------

def test_the_shipped_classification_proof_still_learns_and_resumes_exactly():
    """The E8 artifact is re-run here as a Phase-level guarantee, not
    re-tested in detail: it must still learn and still resume exactly."""
    from examples.native_classification_training import (
        run_resume_proof, run_training,
    )

    run = run_training()
    assert all(math.isfinite(value) for value in run["loss_history"])
    assert run["final_loss"] < run["initial_loss"] / 10.0
    assert run["final_accuracy"] >= 0.90
    assert run["final_accuracy"] > run["initial_accuracy"]

    proof = run_resume_proof()
    for key in ("identical_start", "suffix_matches", "losses_match",
                "parameters_match", "optimizer_state_matches", "logits_match",
                "predictions_match", "accuracies_match"):
        assert proof[key] is True, key


def test_a_classification_checkpoint_round_trip_is_exact_for_both_optimizers(
    tmp_path
):
    """Both native optimizers carry a classification model through save,
    load into a fresh pair, and one further identical step."""
    for factory, name in ((lambda m: NativeAdam(m.parameters(), lr=0.05),
                           "NativeAdam"),
                          (lambda m: NativeSGD(m.parameters(), lr=0.05),
                           "NativeSGD")):
        model = _classifier()
        optimizer = factory(model)
        loss_fn = NativeCrossEntropyLoss()
        x, _ = _images()
        targets = _labels()
        _step(model, loss_fn, optimizer, x, targets)
        path = str(tmp_path / f"phase_e_{name}.npz")
        save_native_checkpoint(path, model, optimizer=optimizer,
                               metadata={"optimizer": name})

        fresh = _classifier()
        fresh_optimizer = factory(fresh)
        metadata = load_native_checkpoint(path, fresh, optimizer=fresh_optimizer)
        assert metadata == {"optimizer": name}
        for (label, a), (_, b) in zip(model.named_parameters(),
                                      fresh.named_parameters()):
            assert np.array_equal(a.to_numpy(), b.to_numpy()), (name, label)
        original_next = _step(model, loss_fn, optimizer, x, targets)
        resumed_next = _step(fresh, loss_fn, fresh_optimizer, x, targets)
        assert original_next == resumed_next, name
        for (label, a), (_, b) in zip(model.named_parameters(),
                                      fresh.named_parameters()):
            assert np.array_equal(a.to_numpy(), b.to_numpy()), (name, label)
        _close_all(x, model, optimizer, fresh, fresh_optimizer)


def test_the_checkpoint_format_is_version_1_and_holds_no_classification_state(
    tmp_path
):
    assert native_checkpoint._FORMAT_VERSION == 2
    model = _classifier()
    optimizer = NativeAdam(model.parameters(), lr=0.05)
    loss_fn = NativeCrossEntropyLoss()
    x, _ = _images()
    _step(model, loss_fn, optimizer, x, _labels())
    path = str(tmp_path / "phase_e.npz")
    save_native_checkpoint(path, model, optimizer=optimizer)

    with np.load(path, allow_pickle=False) as archive:
        names = list(archive.files)
        manifest = archive["manifest"].tobytes().decode("utf-8")
    blob = (" ".join(names) + " " + manifest).lower()
    # No graph resource, no saved probabilities, no target metadata, no
    # loss or metric state — the classification stack persists nothing.
    for banned in ("probabilit", "target", "label", "winner", "graph",
                   "grad", "accuracy", "metric", "crossentropy",
                   "cross_entropy", "softmax", "logit"):
        assert banned not in blob, banned
    assert '"format": "tensorforge.native_checkpoint"' in manifest
    assert '"format_version": 2' in manifest
    _close_all(x, model, optimizer)


# --------------------------------------------------------------------------
# 8. Lifetime, failure atomicity, and error-state recovery
# --------------------------------------------------------------------------

def test_repeated_classification_steps_do_not_accumulate_storage(live_storages):
    model = _classifier()
    optimizer = NativeAdam(model.parameters(), lr=0.05)
    loss_fn = NativeCrossEntropyLoss()
    x, _ = _images()
    targets = _labels()
    for _ in range(3):
        _step(model, loss_fn, optimizer, x, targets)
    gc.collect()
    baseline = len(live_storages)
    assert baseline > 0                     # persistent state is honestly live
    def report():
        """A reporting pass in its own scope: the forward's intermediates
        are graph parents of the logits, so they are released when the
        last reference to the chain goes out of scope here — not held
        alive by a lingering local in the caller."""
        logits = model(x)
        try:
            return native_accuracy(logits, targets)
        finally:
            logits.close()

    for _ in range(5):
        _step(model, loss_fn, optimizer, x, targets)
        # Reporting inside the loop must not accumulate anything either.
        assert 0.0 <= report() <= 1.0
        gc.collect()
        assert len(live_storages) == baseline
    _close_all(x, model, optimizer)


def test_explicit_cleanup_is_idempotent_across_the_stack():
    model = _classifier()
    optimizer = NativeAdam(model.parameters(), lr=0.05)
    loss_fn = NativeCrossEntropyLoss()
    x, _ = _images()
    logits = model(x)
    loss = loss_fn(logits, _labels())
    loss.backward()
    for _ in range(3):
        loss.close()
        logits.close()
        x.close()
        optimizer.close()
        for parameter in model.parameters():
            parameter.close()
    assert loss.closed and logits.closed and x.closed


@needs_fault_injection
def test_failure_in_a_mixed_classification_graph_is_atomic(live_storages):
    model = _classifier()
    optimizer = NativeAdam(model.parameters(), lr=0.05)
    loss_fn = NativeCrossEntropyLoss()
    x, _ = _images()
    targets = _labels()
    _step(model, loss_fn, optimizer, x, targets)       # settle Adam's state
    before = {n: p.to_numpy().copy() for n, p in model.named_parameters()}
    versions = [p.version for p in model.parameters()]
    gc.collect()
    baseline = len(live_storages)

    cpp._arm_alloc_failure(1)
    with pytest.raises(MemoryError):
        _step(model, loss_fn, optimizer, x, targets)
    cpp._arm_alloc_failure(0)

    # Nothing was committed and nothing leaked.
    for name, parameter in model.named_parameters():
        assert np.array_equal(parameter.to_numpy(), before[name]), name
    assert [p.version for p in model.parameters()] == versions
    optimizer.zero_grad()
    gc.collect()
    assert len(live_storages) <= baseline
    # The native error state is clear after the handled failure...
    assert cpp._require_library().tf_last_error_code() == cpp.TF_OK
    # ...and the stack works normally again.
    value = _step(model, loss_fn, optimizer, x, targets)
    assert math.isfinite(value)
    assert cpp._require_library().tf_last_error_code() == cpp.TF_OK
    _close_all(x, model, optimizer)


def test_a_rejected_classification_call_leaves_the_error_state_clean():
    """Validation failures are Python-level and must not leave a stale
    native error code behind for the next call to trip over."""
    logits = NativeTensor.from_array(np.zeros((2, NUM_CLASSES)))
    for bad in ([0.5, 1.0], [0, NUM_CLASSES], [0]):
        with pytest.raises((TypeError, ValueError, IndexError)):
            logits.cross_entropy(bad, reduction="mean")
    assert cpp._require_library().tf_last_error_code() == cpp.TF_OK
    loss = logits.cross_entropy([0, 1], reduction="mean")
    assert math.isfinite(float(loss.to_numpy()))
    assert cpp._require_library().tf_last_error_code() == cpp.TF_OK
    _close_all(loss, logits)


# --------------------------------------------------------------------------
# 9. The NumPy boundary of the closed phase
# --------------------------------------------------------------------------

_NUMERICAL_NUMPY = (
    "max", "amax", "argmax", "exp", "log", "logaddexp", "sum", "divide",
    "true_divide", "add", "subtract", "multiply", "matmul", "mean",
    "negative", "power", "copyto", "sqrt", "reciprocal", "take",
    "take_along_axis", "put", "put_along_axis", "where", "choose", "maximum",
)


def test_no_phase_e_workload_needs_a_numpy_fallback(monkeypatch):
    """One complete classification step plus every Phase-E forward runs
    with NumPy's numerical routines and the tensor-data conversion
    boundary armed. `native_accuracy` is deliberately outside — it
    converts on purpose — and the reference reads happen afterwards."""
    model = _classifier()
    optimizer = NativeAdam(model.parameters(), lr=0.05)
    loss_fn = NativeCrossEntropyLoss()
    x, _ = _images()
    targets = _labels()
    _step(model, loss_fn, optimizer, x, targets)     # allocate Adam's state

    def tripwire(*args, **kwargs):
        raise AssertionError("a Phase-E native path reached NumPy")

    for name in _NUMERICAL_NUMPY:
        monkeypatch.setattr(np, name, tripwire)
    monkeypatch.setattr(cpp.NativeTensorCore, "to_numpy", tripwire)
    monkeypatch.setattr(cpp.NativeTensorView, "to_numpy", tripwire)
    monkeypatch.setattr(cpp.NativeStorage, "to_numpy", tripwire)
    monkeypatch.setattr(NativeTensor, "to_numpy", tripwire)

    logits = model(x)
    loss = loss_fn(logits, targets)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    transforms = [logits.softmax(axis=-1), logits.log_softmax(axis=-1),
                  logits.exp()]
    transforms.append(transforms[-1].log())
    with pytest.raises(AssertionError, match="reached NumPy"):
        loss.to_numpy()
    monkeypatch.undo()

    assert math.isfinite(float(loss.to_numpy()))
    for tensor in transforms:
        assert np.isfinite(tensor.to_numpy()).all()
    _close_all(*transforms, loss, logits, x, model, optimizer)


# --------------------------------------------------------------------------
# 10. The shipped artifacts of the closed phase
# --------------------------------------------------------------------------

def test_phase_e_artifacts_are_present():
    assert (REPO_ROOT / "cpp" / "src" / "classification.cpp").is_file()
    assert (REPO_ROOT / "examples"
            / "native_classification_training.py").is_file()
    assert (REPO_ROOT / "benchmarks"
            / "benchmark_native_classification.py").is_file()
    assert (REPO_ROOT / "docs" / "native_classification_design.md").is_file()
    for name in ("test_exp", "test_log", "test_softmax", "test_log_softmax",
                 "test_cross_entropy"):
        assert (REPO_ROOT / "cpp" / "tests" / f"{name}.cpp").is_file(), name
    for name in ("test_native_exp", "test_native_log", "test_native_softmax",
                 "test_native_log_softmax", "test_native_cross_entropy_core",
                 "test_native_cross_entropy", "test_native_cross_entropy_loss",
                 "test_native_metrics", "test_native_classification_training",
                 "test_native_classification_benchmark"):
        assert (REPO_ROOT / "tests" / f"{name}.py").is_file(), name


def test_the_benchmark_smoke_and_json_paths_remain_valid():
    """The E9 harness stays runnable at closure: correctness gates pass,
    JSON parses, and no timing is asserted here either."""
    from benchmarks import benchmark_native_classification as harness

    payload = harness.run_benchmark(warmup=1, repetitions=2, smoke=True)
    assert payload["benchmark"] == "tensorforge.native_classification"
    assert len(payload["cases"]) == 7
    for record in payload["cases"]:
        assert record["correctness"]["status"] == "passed", record["case"]
        # Correctness metrics are separate from timing statistics.
        assert "max_abs_error" in record["correctness"]
        assert "median_s" not in record["correctness"]
        assert record["reference_type"] in ("stable_tensorforge", "numpy",
                                            "native_only")
        for value in record["native"]["samples_s"]:
            assert math.isfinite(value) and value >= 0.0
    assert json.loads(json.dumps(payload)) == payload
    report = harness.format_report(payload)
    assert "local characterization only" in report.lower()


def test_the_benchmark_carries_no_fixed_performance_threshold():
    from benchmarks import benchmark_native_classification as harness

    for name in dir(harness):
        if name.startswith("_"):
            continue
        value = getattr(harness, name)
        if isinstance(value, float):
            assert name in ("FORWARD_ATOL", "LOSS_ATOL", "GRADIENT_ATOL",
                            "PARAMETER_ATOL", "DEFAULT_LR"), name
    source = (REPO_ROOT / "benchmarks"
              / "benchmark_native_classification.py").read_text(encoding="utf-8")
    for banned in ("max_seconds", "min_speedup", "timing_threshold",
                   "performance_gate", "assert_faster"):
        assert banned not in source.lower(), banned


def test_phase_e_added_no_stable_framework_or_dispatch_surface():
    """The closure boundary: the stable line gained nothing, and no
    dispatch layer appeared between the two."""
    assert cpp.backend_info()["stable_framework_integration"] is False
    stable_names = set(dir(tensorforge)) | set(dir(tensorforge.nn))
    for native in ("NativeCrossEntropyLoss", "native_accuracy",
                   "NativeTensor", "NativeAdam"):
        assert native not in stable_names, native
    for dispatcher in ("set_backend", "use_native", "backend", "dispatch"):
        assert not hasattr(tensorforge, dispatcher), dispatcher
    # NativeMSELoss and NativeCrossEntropyLoss stay independent modules,
    # neither derived from nor registered into the stable loss functions.
    assert not issubclass(NativeCrossEntropyLoss, type(tensorforge.nn.Linear))
    assert issubclass(NativeCrossEntropyLoss, NativeModule)
    assert issubclass(NativeMSELoss, NativeModule)
