"""Phase-F integration guardrails: the invariants that span the whole
native normalization stack at once (Advanced C++ Phase F, milestone F8).

The per-milestone suites already cover the atomic state transaction (F1),
`NativeLayerNorm` (F2), `NativeBatchNorm1d` (F3), `NativeBatchNorm2d`
(F4), the state/checkpoint/ownership/graph-safety hardening (F5), the
deterministic normalized training and exact resume (F6), and the
correctness-gated characterization benchmark (F7) in depth. This file
deliberately tests only what those files cannot: the **interactions** —

- one graph carrying convolution, both normalization families, pooling,
  flatten, linear layers, and the fused classification loss together,
  trained by `NativeAdam` and resumed exactly from one checkpoint;
- three independent saved-resource families (BatchNorm eval snapshots,
  MaxPool2d winners, and cross-entropy probabilities) coexisting in one
  eval graph and releasing exactly once;
- the buffer-mutation rule and the parameter-version rule meeting in the
  same graph, so each is attributed to the right cause;
- the Phase-E versioning archetypes (saved-output `exp`, live-reread
  `log`, saved-probability cross-entropy) meeting BatchNorm snapshots;
- shared and frozen parameters through a normalized model;
- a non-contiguous NCHW input through the entire stack;
- failure atomicity at each **real** boundary, without pretending a whole
  training step is globally transactional;
- and the capability, export, inventory, and artifact boundary of the
  phase as it stands.

Nothing here adds numerical behavior, registers anything, or depends on
one implementation being faster than another. Every assertion is a
property the architecture promises.

Selector: python -m pytest -q -k native_phase_f
"""

import gc
import math
import os
from pathlib import Path

import numpy as np
import pytest

import tensorforge
from tensorforge.backends import cpp
from tensorforge.experimental import (
    NativeAdam,
    NativeBatchNorm1d,
    NativeBatchNorm2d,
    NativeConv2d,
    NativeCrossEntropyLoss,
    NativeFlatten,
    NativeLayerNorm,
    NativeLinear,
    NativeMaxPool2d,
    NativeModule,
    NativeParameter,
    NativeReLU,
    NativeTensor,
    load_native_checkpoint,
    native_accuracy,
    save_native_checkpoint,
)
from tensorforge.experimental import _native_state, native_checkpoint

pytestmark = pytest.mark.skipif(
    not cpp.is_available(),
    reason="experimental C++ backend not built; " + cpp.build_instructions(),
)

REPO_ROOT = Path(__file__).resolve().parent.parent

# The integrated architecture's fixed shape configuration. A 6x6 image
# through conv(3) is 4x4; through pool(2) it is 2x2; four channels give
# 4 * 2 * 2 = 16 flattened features.
IN_CHANNELS = 1
CONV_CHANNELS = 4
KERNEL_SIZE = 3
FLAT_FEATURES = CONV_CHANNELS * 2 * 2
HIDDEN_FEATURES = 8
NUM_CLASSES = 3
MOMENTUM = 0.1

CONV_SEED, HIDDEN_SEED, HEAD_SEED = 0, 1, 2

# The deterministic integrated schedule. Fixed, never sampled.
TOTAL_STEPS = 12
SPLIT_STEP = 5
LR = 0.05

PARAMETER_NAMES = (
    "conv.weight", "conv.bias",
    "batch_norm2d.gamma", "batch_norm2d.beta",
    "hidden.weight", "hidden.bias",
    "batch_norm1d.gamma", "batch_norm1d.beta",
    "layer_norm.weight", "layer_norm.bias",
    "head.weight", "head.bias",
)
BUFFER_NAMES = (
    "batch_norm2d.running_mean", "batch_norm2d.running_var",
    "batch_norm1d.running_mean", "batch_norm1d.running_var",
)


class _Boom(RuntimeError):
    """A distinctive injected failure, so a test can never mistake an
    unrelated error for the one it injected."""


def _injected(target, name, replacement):
    """Patch one seam in its **own** monkeypatch context.

    Deliberately not the shared ``monkeypatch`` fixture: undoing that one
    would also restore the ``live_storages`` fixture's tracking hooks, so
    every release after the undo would go unobserved and the baseline
    check would report a leak that does not exist."""
    patcher = pytest.MonkeyPatch()
    patcher.setattr(target, name, replacement)
    return patcher


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


def _collect():
    """Force the composed normalization graph's intermediate wrappers —
    which participate in reference cycles, a property of the
    Python-managed native autograd engine since Phase B — to their
    deterministic collection point. Used only at established Python
    wrapper-cycle boundaries; the failure paths that promise *immediate*
    deterministic cleanup are checked without it."""
    gc.collect()


# ==========================================================================
# The canonical integrated model and its fixed dataset
# ==========================================================================

class NativePhaseFClassifier(NativeModule):
    """**Test-only.** The whole native line in one model: the Phase-D
    convolutional stack, **both** normalization families, LayerNorm, and
    a linear head whose **raw logits** go to the fused Phase-E loss.

    Named children give readable canonical state keys. There is
    deliberately no softmax or log-softmax module: the loss consumes raw
    logits. Not a production class, not exported, not a public API."""

    def __init__(self, conv_seed=CONV_SEED, hidden_seed=HIDDEN_SEED,
                 head_seed=HEAD_SEED, momentum=MOMENTUM):
        super().__init__()
        self.conv = NativeConv2d(IN_CHANNELS, CONV_CHANNELS, KERNEL_SIZE,
                                 seed=conv_seed)
        self.batch_norm2d = NativeBatchNorm2d(CONV_CHANNELS, momentum=momentum)
        self.relu2d = NativeReLU()
        self.pool = NativeMaxPool2d(2)
        self.flatten = NativeFlatten()
        self.hidden = NativeLinear(FLAT_FEATURES, HIDDEN_FEATURES,
                                   seed=hidden_seed)
        self.batch_norm1d = NativeBatchNorm1d(HIDDEN_FEATURES,
                                              momentum=momentum)
        self.relu1d = NativeReLU()
        self.layer_norm = NativeLayerNorm(HIDDEN_FEATURES)
        self.head = NativeLinear(HIDDEN_FEATURES, NUM_CLASSES, seed=head_seed)

    def forward(self, images):
        hidden = self.conv(images)
        hidden = self.batch_norm2d(hidden)
        hidden = self.relu2d(hidden)
        hidden = self.pool(hidden)
        hidden = self.flatten(hidden)
        hidden = self.hidden(hidden)
        hidden = self.batch_norm1d(hidden)
        hidden = self.relu1d(hidden)
        hidden = self.layer_norm(hidden)
        return self.head(hidden)


def _dataset():
    """The E8 fixed twelve-image, three-class task: nested float literals
    and strict host integer targets. Nothing is generated, augmented,
    shuffled, downloaded, or randomly sampled."""
    from examples.native_classification_training import build_dataset

    return build_dataset()


def _inputs():
    images, targets = _dataset()
    return NativeTensor.from_array(images), targets


def _close_module(module):
    """There is no ``NativeModule.close()``, so a stateful module's owner
    releases **both** traversals explicitly (design §9). Both are
    identity-deduplicated, so a shared object closes exactly once."""
    for parameter in module.parameters():
        parameter.close()
    for buffer in module.buffers():
        buffer.close()


def _close_run(model=None, optimizer=None, *tensors):
    if optimizer is not None:
        optimizer.close()
    if model is not None:
        _close_module(model)
    for tensor in tensors:
        if tensor is not None:
            tensor.close()


def _train_step(model, loss_fn, optimizer, x, targets):
    """One complete integrated iteration, returning the pre-update loss."""
    model.train()
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


def _evaluate(model, x, targets):
    """A no-update reporting pass in **evaluation mode**, restoring the
    caller's previous mode. Returns plain Python values only."""
    was_training = model.training
    model.eval()
    logits = model(x)
    try:
        values = logits.to_numpy()
        return {
            "logits": values.tolist(),
            "predictions": values.argmax(axis=1).tolist(),
            "accuracy": native_accuracy(logits, targets),
        }
    finally:
        logits.close()
        model.train(was_training)


def _values(model):
    """Every parameter and buffer as plain nested lists, by canonical
    name — read through the public conversion boundary."""
    return {
        "parameters": {name: parameter.to_numpy().tolist()
                       for name, parameter in model.named_parameters()},
        "buffers": {name: buffer.to_numpy().tolist()
                    for name, buffer in model.named_buffers()},
    }


def _optimizer_values(optimizer):
    """The NativeAdam state as plain Python values, closing every
    caller-owned ``m``/``v`` snapshot the state dictionary hands back."""
    state = optimizer.state_dict()
    try:
        return {
            "format_version": state["format_version"],
            "optimizer": state["optimizer"],
            "lr": state["lr"],
            "betas": list(state["betas"]),
            "eps": state["eps"],
            "step_counts": list(state["step_counts"]),
            "m": [tensor.to_numpy().tolist() for tensor in state["m"]],
            "v": [tensor.to_numpy().tolist() for tensor in state["v"]],
        }
    finally:
        for key in ("m", "v"):
            for tensor in state[key]:
                tensor.close()


def _state_keys(model):
    state = model.state_dict()
    try:
        return list(state)
    finally:
        for snapshot in state.values():
            snapshot.close()


# -- graph inspection -------------------------------------------------------

def _walk_graph(root):
    """Every autograd node reachable from ``root`` plus every native
    object a node's history owns. The three saved-resource families use
    different object types — BatchNorm snapshots are ``NativeTensor``s,
    MaxPool2d winners and cross-entropy probabilities are cores — so the
    walk returns them together and the callers classify."""
    nodes, resources, seen = [], [], set()
    stack = [root]
    while stack:
        node = stack.pop()
        if id(node) in seen:
            continue
        seen.add(id(node))
        nodes.append(node)
        resources.extend(node._graph_resources)
        stack.extend(node._parents)
    return nodes, resources


def _saved_resources(root):
    """Only the saved resources a graph owns.

    Deliberately *not* the node list: retaining the nodes would keep an
    abandoned graph reachable, so the very cleanup these tests check
    could never happen. Callers that need the nodes ask for them
    explicitly."""
    return _walk_graph(root)[1]


def _graph_object_ids(root):
    nodes, resources = _walk_graph(root)
    return {id(item) for item in nodes} | {id(item) for item in resources}


def _graph_storage_ids(root):
    """Every native **storage** the graph can reach — stronger than the
    object walk, because a borrowing view of a buffer is a different
    object over the same bytes."""
    ids = set()
    nodes, resources = _walk_graph(root)
    for item in nodes + resources:
        if isinstance(item, NativeTensor) and not item.closed:
            ids.add(id(item._core.storage))
    return ids


def _resource_closed(resource):
    """``NativeTensor`` wrappers and raw cores both carry ``_closed``."""
    return bool(resource._closed)


def _load_buffers(module, **arrays):
    """Install running-statistic values through the public atomic loader,
    which preserves every buffer identity."""
    tensors = {name: NativeTensor.from_array(np.asarray(value,
                                                        dtype=np.float64))
               for name, value in arrays.items()}
    try:
        module.load_state_dict(tensors, strict=False)
    finally:
        for tensor in tensors.values():
            tensor.close()


def _nontrivial_running_state(model):
    """Give both BatchNorm modules running statistics that are neither
    the fresh zeros/ones nor the batch's own, so an eval forward really
    depends on them."""
    _load_buffers(
        model.batch_norm2d,
        running_mean=[0.3, -0.2, 0.5, 0.1],
        running_var=[2.0, 0.5, 1.25, 0.75],
    )
    _load_buffers(
        model.batch_norm1d,
        running_mean=[0.4, -0.3, 0.2, 0.6, -0.1, 0.25, -0.45, 0.15],
        running_var=[1.5, 0.6, 2.25, 0.8, 1.1, 0.45, 1.75, 0.9],
    )


class _RunningStatHolder(NativeModule):
    """**Test-only** parameter-free module registering *existing* running
    buffers as persistent aliases, so ``load_native_checkpoint()`` can be
    driven over exactly those objects without touching any affine
    parameter. Not a production helper and not exported."""

    def __init__(self, **buffers):
        super().__init__()
        for name, tensor in buffers.items():
            self.register_buffer(name, tensor, persistent=True)


# ==========================================================================
# 1. One graph carries the whole integrated path
# ==========================================================================

def test_one_graph_carries_conv_both_normalizations_pooling_and_the_loss():
    """The interaction no single-module suite covers: convolution, NCHW
    batch normalization, ReLU, pooling with its saved winners, flatten,
    linear, 2-D batch normalization, LayerNorm, the head, and the fused
    loss with its saved probabilities — one forward, one backward, one
    NativeAdam step."""
    model = NativePhaseFClassifier()
    loss_fn = NativeCrossEntropyLoss()
    optimizer = NativeAdam(model.parameters(), lr=LR)
    x, targets = _inputs()

    assert tuple(name for name, _ in model.named_parameters()) == PARAMETER_NAMES
    assert tuple(name for name, _ in model.named_buffers()) == BUFFER_NAMES
    before = {name: parameter.to_numpy().copy()
              for name, parameter in model.named_parameters()}
    versions = {name: parameter.version
                for name, parameter in model.named_parameters()}
    buffers_before = {name: buffer.to_numpy().copy()
                      for name, buffer in model.named_buffers()}
    parameter_ids = [id(p) for p in model.parameters()]
    buffer_ids = [id(b) for b in model.buffers()]

    model.train()
    logits = model(x)
    assert logits.shape == (len(targets), NUM_CLASSES)
    # Raw logits: no probability transform ran anywhere in the model.
    assert not np.allclose(logits.to_numpy().sum(axis=1), 1.0)
    loss = loss_fn(logits, targets)
    assert loss.shape == () and loss.numel == 1
    assert math.isfinite(float(loss.to_numpy()))

    # The three saved-resource families are all present in one graph.
    _nodes, resources = _walk_graph(loss)
    assert resources, "the integrated graph owns no saved resources"
    assert loss._graph_resources, "cross-entropy owns no saved probabilities"
    assert all(not _resource_closed(r) for r in resources)

    loss.backward()

    for name, parameter in model.named_parameters():
        assert parameter.grad is not None, name
        assert parameter.grad.shape == parameter.shape, name
        gradient = parameter.grad.to_numpy()
        assert np.isfinite(gradient).all(), name
    for name, buffer in model.named_buffers():
        assert buffer.grad is None, name
    # Both BatchNorm pairs advanced together during the training forward.
    for name, buffer in model.named_buffers():
        assert not np.array_equal(buffer.to_numpy(), buffers_before[name]), name
    # The completed graph and every saved resource were released once.
    assert loss._graph_freed is True
    assert all(_resource_closed(r) for r in resources)
    assert loss._graph_resources == ()

    optimizer.step()
    for name, parameter in model.named_parameters():
        assert not np.array_equal(parameter.to_numpy(), before[name]), name
        assert parameter.version == versions[name] + 1, name
    assert list(optimizer.step_counts) == [1] * len(parameter_ids)
    optimizer.zero_grad()
    assert all(p.grad is None for p in model.parameters())
    # Identities never moved.
    assert [id(p) for p in model.parameters()] == parameter_ids
    assert [id(b) for b in model.buffers()] == buffer_ids
    _close_run(model, optimizer, loss, logits, x)


def test_buffers_are_never_optimized_and_never_versioned():
    """The optimizer sees parameters only; the running buffers are model
    state that no optimizer, gradient, or version ever touches."""
    model = NativePhaseFClassifier()
    optimizer = NativeAdam(model.parameters(), lr=LR)
    buffer_ids = {id(b) for b in model.buffers()}
    assert buffer_ids and not (buffer_ids & {id(p)
                                             for p in optimizer.parameters()})
    assert len(optimizer.parameters()) == len(PARAMETER_NAMES)
    for buffer in model.buffers():
        assert buffer.requires_grad is False
        assert not isinstance(buffer, NativeParameter)
        assert not hasattr(buffer, "version")
    _close_run(model, optimizer)


# ==========================================================================
# 2. Deterministic integrated training and exact checkpoint resume
# ==========================================================================

def _run_schedule(steps, model=None, optimizer=None, x=None, targets=None):
    """Run ``steps`` integrated training iterations, returning the loss
    list. Ownership stays with the caller."""
    loss_fn = NativeCrossEntropyLoss()
    return [_train_step(model, loss_fn, optimizer, x, targets)
            for _ in range(steps)]


def _uninterrupted_run(total_steps=TOTAL_STEPS):
    model = NativePhaseFClassifier()
    optimizer = NativeAdam(model.parameters(), lr=LR)
    x, targets = _inputs()
    try:
        initial = _evaluate(model, x, targets)
        losses = _run_schedule(total_steps, model, optimizer, x, targets)
        loss_fn = NativeCrossEntropyLoss()
        model.train()
        final_logits_tensor = model(x)
        try:
            final_train_logits = final_logits_tensor.to_numpy().tolist()
        finally:
            final_logits_tensor.close()
        del loss_fn
        return {
            "losses": losses,
            "initial_eval": initial,
            "final_eval": _evaluate(model, x, targets),
            "final_train_logits": final_train_logits,
            "values": _values(model),
            "optimizer": _optimizer_values(optimizer),
            "parameter_order": [n for n, _ in model.named_parameters()],
            "buffer_order": [n for n, _ in model.named_buffers()],
            "state_keys": _state_keys(model),
        }
    finally:
        _close_run(model, optimizer, x)


def test_the_integrated_classifier_trains_deterministically():
    """Two independently constructed runs of the same fixed schedule are
    bit-identical, and the run makes real progress on the fixed task."""
    first = _uninterrupted_run()
    second = _uninterrupted_run()
    assert first["losses"] == second["losses"]
    assert first["values"] == second["values"]
    assert first["optimizer"] == second["optimizer"]
    assert first["final_eval"] == second["final_eval"]
    assert first["final_train_logits"] == second["final_train_logits"]

    losses = first["losses"]
    assert len(losses) == TOTAL_STEPS
    assert all(math.isfinite(value) for value in losses)
    # Broad, observed guardrails — never strict monotonicity, and never a
    # generalization or performance claim.
    assert losses[-1] < losses[0]
    assert min(losses) < losses[0] * 0.75
    assert first["final_eval"]["accuracy"] >= first["initial_eval"]["accuracy"]
    # Both BatchNorm pairs left their fresh zeros/ones behind.
    buffers = first["values"]["buffers"]
    assert buffers["batch_norm2d.running_mean"] != [0.0] * CONV_CHANNELS
    assert buffers["batch_norm2d.running_var"] != [1.0] * CONV_CHANNELS
    assert buffers["batch_norm1d.running_mean"] != [0.0] * HIDDEN_FEATURES
    assert buffers["batch_norm1d.running_var"] != [1.0] * HIDDEN_FEATURES


def test_the_integrated_stack_resumes_exactly_from_one_checkpoint(tmp_path):
    """The whole point of F8's training section: a model combining
    convolution, pooling, **both** BatchNorm shapes, LayerNorm, the fused
    loss, and NativeAdam is interrupted, checkpointed, reloaded into a
    completely fresh pair, and continued — reproducing everything
    exactly, including all four running-statistic buffers and the
    evaluation-mode output."""
    reference = _uninterrupted_run()

    model = NativePhaseFClassifier()
    optimizer = NativeAdam(model.parameters(), lr=LR)
    x, targets = _inputs()
    prefix = _run_schedule(SPLIT_STEP, model, optimizer, x, targets)
    path = os.path.join(str(tmp_path), "phase_f.npz")
    save_native_checkpoint(path, model, optimizer=optimizer,
                           metadata={"steps_completed": SPLIT_STEP, "lr": LR})
    _close_run(model, optimizer)

    fresh = NativePhaseFClassifier()
    fresh_optimizer = NativeAdam(fresh.parameters(), lr=LR)
    parameter_ids = [id(p) for p in fresh.parameters()]
    buffer_ids = [id(b) for b in fresh.buffers()]
    # Deliberately in eval mode before the load, to prove the training
    # flag is runtime state and is never serialized.
    fresh.eval()
    metadata = load_native_checkpoint(path, fresh, optimizer=fresh_optimizer)
    assert metadata == {"steps_completed": SPLIT_STEP, "lr": LR}
    assert fresh.training is False
    assert [id(p) for p in fresh.parameters()] == parameter_ids
    assert [id(b) for b in fresh.buffers()] == buffer_ids
    fresh.train()

    suffix = _run_schedule(TOTAL_STEPS - SPLIT_STEP, fresh, fresh_optimizer,
                           x, targets)
    fresh.train()
    resumed_logits_tensor = fresh(x)
    try:
        final_train_logits = resumed_logits_tensor.to_numpy().tolist()
    finally:
        resumed_logits_tensor.close()
    final_eval = _evaluate(fresh, x, targets)
    values = _values(fresh)
    optimizer_state = _optimizer_values(fresh_optimizer)
    parameter_order = [n for n, _ in fresh.named_parameters()]
    buffer_order = [n for n, _ in fresh.named_buffers()]
    state_keys = _state_keys(fresh)
    _close_run(fresh, fresh_optimizer, x)

    # Exact equality everywhere — never a tolerance.
    assert prefix == reference["losses"][:SPLIT_STEP]
    assert suffix == reference["losses"][SPLIT_STEP:]
    assert prefix + suffix == reference["losses"]
    assert values["parameters"] == reference["values"]["parameters"]
    assert optimizer_state == reference["optimizer"]
    for name in BUFFER_NAMES:
        assert values["buffers"][name] == reference["values"]["buffers"][name], name
    assert final_train_logits == reference["final_train_logits"]
    assert final_eval["logits"] == reference["final_eval"]["logits"]
    assert final_eval["predictions"] == reference["final_eval"]["predictions"]
    assert final_eval["accuracy"] == reference["final_eval"]["accuracy"]
    assert parameter_order == reference["parameter_order"] == list(PARAMETER_NAMES)
    assert buffer_order == reference["buffer_order"] == list(BUFFER_NAMES)
    assert state_keys == reference["state_keys"]
    assert state_keys == list(PARAMETER_NAMES) + list(BUFFER_NAMES)


def test_the_integrated_checkpoint_holds_only_model_state(tmp_path):
    """Four BatchNorm running buffers ride the manifest as ordinary
    canonical model-state entries — no normalization section of their own,
    and no graph, gradient, winner, probability, or eval snapshot is
    serialized.

    The format version is the *format's*, not Phase F's: G5 moved it to 2
    and added the generator section, which a Phase-F model leaves null.
    Phase F itself still contributes no manifest field."""
    assert native_checkpoint._FORMAT == "tensorforge.native_checkpoint"
    assert native_checkpoint._FORMAT_VERSION == 3
    model = NativePhaseFClassifier()
    optimizer = NativeAdam(model.parameters(), lr=LR)
    x, targets = _inputs()
    _train_step(model, NativeCrossEntropyLoss(), optimizer, x, targets)
    path = os.path.join(str(tmp_path), "schema.npz")
    save_native_checkpoint(path, model, optimizer=optimizer)

    with np.load(path, allow_pickle=False) as archive:
        names = list(archive.files)
        manifest = archive["manifest"].tobytes().decode("utf-8")
    assert '"format": "tensorforge.native_checkpoint"' in manifest
    assert '"format_version": 3' in manifest
    # The archived model section is exactly the canonical state keys —
    # every parameter and the four running buffers as ordinary entries,
    # with no normalization configuration (eps, momentum, num_features,
    # normalized_shape) and no extra section.
    import json

    parsed = json.loads(manifest)
    assert list(parsed["model"]["keys"]) == (list(PARAMETER_NAMES)
                                            + list(BUFFER_NAMES))
    assert set(parsed) == {"format", "format_version", "model", "optimizer",
                           "generators", "metadata"}
    # The Phase-F classifier registers no generator, so G5's section is an
    # explicit null: the format version moved, Phase F's state did not.
    assert parsed["generators"] is None
    blob = (" ".join(names) + " " + manifest).lower()
    for banned in ("winner", "probabilit", "snapshot", "graph", "grad",
                   "training", "logit", "target", "label", "rng", "seed",
                   "scheduler", "num_features", "normalized_shape",
                   "batchnorm3d"):
        assert banned not in blob, banned
    # The BatchNorm momentum is constructor configuration, never state.
    assert "momentum" not in json.dumps(parsed["model"]).lower()
    _close_run(model, optimizer, x)


# ==========================================================================
# 3. The eval graph: three saved-resource families at once
# ==========================================================================

def test_eval_graph_combines_snapshots_winners_and_probabilities_safely():
    """The system-level interaction: in one eval graph the BatchNorm
    snapshots, the MaxPool2d winners, and the cross-entropy probabilities
    coexist, none of them is a registered buffer, and one backward
    releases all three families exactly once."""
    model = NativePhaseFClassifier()
    _nontrivial_running_state(model)
    model.eval()
    x, targets = _inputs()
    buffers_before = {name: buffer.to_numpy().copy()
                      for name, buffer in model.named_buffers()}
    buffer_object_ids = {id(b) for b in model.buffers()}
    buffer_storage_ids = {id(b._core.storage) for b in model.buffers()}

    # Build the stack explicitly so each stage's own resources are visible.
    conv = model.conv(x)
    normalized2d = model.batch_norm2d(conv)
    activated2d = model.relu2d(normalized2d)
    pooled = model.pool(activated2d)
    winners = pooled._graph_resources
    assert winners, "pooling owns no saved winner buffer"
    flat = model.flatten(pooled)
    hidden = model.hidden(flat)
    normalized1d = model.batch_norm1d(hidden)
    activated1d = model.relu1d(normalized1d)
    layer_normed = model.layer_norm(activated1d)
    logits = model.head(layer_normed)
    loss = NativeCrossEntropyLoss()(logits, targets)
    probabilities = loss._graph_resources
    assert probabilities, "cross-entropy owns no saved probabilities"

    # No registered running buffer — object or storage — is in the graph.
    reachable_objects = _graph_object_ids(loss)
    reachable_storages = _graph_storage_ids(loss)
    assert not (buffer_object_ids & reachable_objects)
    assert not (buffer_storage_ids & reachable_storages)
    # ...while gamma/beta legitimately are direct operands.
    for module in (model.batch_norm2d, model.batch_norm1d):
        assert id(module.gamma) in reachable_objects
        assert id(module.beta) in reachable_objects

    # The snapshots the two BatchNorm modules adopted carry each shape's
    # own broadcast layout.
    snapshots_2d = normalized2d._graph_resources
    snapshots_1d = normalized1d._graph_resources
    assert len(snapshots_2d) == 2 and len(snapshots_1d) == 2
    for snapshot in snapshots_2d:
        assert snapshot.shape == (1, CONV_CHANNELS, 1, 1)
        assert snapshot.owns_core and snapshot.contiguous
        assert snapshot.requires_grad is False
    for snapshot in snapshots_1d:
        assert snapshot.shape == (1, HIDDEN_FEATURES)
        assert snapshot.owns_core and snapshot.contiguous
        assert snapshot.requires_grad is False

    _nodes, resources = _walk_graph(loss)
    assert len(resources) >= len(snapshots_2d) + len(snapshots_1d) + \
        len(winners) + len(probabilities)
    assert all(not _resource_closed(r) for r in resources)

    loss.backward()

    # Every family released exactly once, and a second release is a no-op.
    assert all(_resource_closed(r) for r in resources)
    assert loss._graph_resources == () and pooled._graph_resources == ()
    assert normalized2d._graph_resources == ()
    assert normalized1d._graph_resources == ()
    normalized2d._release_graph_resources()
    assert normalized2d._graph_resources == ()
    # The registered buffers are untouched and still open.
    for name, buffer in model.named_buffers():
        assert buffer.closed is False, name
        assert np.array_equal(buffer.to_numpy(), buffers_before[name]), name
    _close_run(model, None, loss, logits, layer_normed, activated1d,
               normalized1d, hidden, flat, pooled, activated2d, normalized2d,
               conv, x)


def test_closing_an_abandoned_eval_graph_releases_only_its_snapshots(
    live_storages
):
    """An eval graph that is never differentiated still frees its
    snapshots deterministically on ``close()``, without touching the
    registered state."""
    model = NativePhaseFClassifier()
    _nontrivial_running_state(model)
    model.eval()
    x, targets = _inputs()
    buffers_before = {name: buffer.to_numpy().copy()
                      for name, buffer in model.named_buffers()}
    _collect()
    baseline = len(live_storages)

    logits = model(x)
    loss = NativeCrossEntropyLoss()(logits, targets)
    resources = _saved_resources(loss)
    own = loss._graph_resources
    assert resources and all(not _resource_closed(r) for r in resources)

    # The closed node's *own* saved state goes immediately, without gc.
    loss.close()
    assert all(_resource_closed(r) for r in own)
    logits.close()
    # The rest belongs to intermediate nodes the caller never held, so it
    # is released when the abandoned chain is — a wrapper-cycle boundary.
    del loss, logits, own
    _collect()
    assert all(_resource_closed(r) for r in resources)
    for name, buffer in model.named_buffers():
        assert buffer.closed is False, name
        assert np.array_equal(buffer.to_numpy(), buffers_before[name]), name
    del resources
    _collect()
    assert len(live_storages) == baseline
    _close_run(model, None, x)


def test_eval_forward_mutates_no_running_state_in_either_shape():
    model = NativePhaseFClassifier()
    _nontrivial_running_state(model)
    model.eval()
    x, targets = _inputs()
    before = {name: buffer.to_numpy().copy()
              for name, buffer in model.named_buffers()}
    versions = {name: p.version for name, p in model.named_parameters()}
    for _ in range(3):
        result = _evaluate(model, x, targets)
        assert 0.0 <= result["accuracy"] <= 1.0
    for name, buffer in model.named_buffers():
        assert np.array_equal(buffer.to_numpy(), before[name]), name
    assert {name: p.version
            for name, p in model.named_parameters()} == versions
    # ...and a training forward on the same model does advance all four.
    model.train()
    out = model(x)
    out.close()
    for name, buffer in model.named_buffers():
        assert not np.array_equal(buffer.to_numpy(), before[name]), name
    _close_run(model, None, x)


# ==========================================================================
# 4. Buffer mutation versus parameter mutation, in one integrated graph
# ==========================================================================

def _eval_graph_with_control(model, x, targets):
    """Build an eval graph over the integrated model and, separately,
    compute the gradients a *clean* backward of the same forward gives —
    the control every buffer-mutation test compares against."""
    control_model = NativePhaseFClassifier()
    state = model.state_dict()
    try:
        control_model.load_state_dict(state)
    finally:
        for snapshot in state.values():
            snapshot.close()
    control_model.eval()
    control_logits = control_model(x)
    control_loss = NativeCrossEntropyLoss()(control_logits, targets)
    control_loss.backward()
    control = {name: parameter.grad.to_numpy().copy()
               for name, parameter in control_model.named_parameters()}
    _close_run(control_model, None, control_loss, control_logits)

    model.eval()
    logits = model(x)
    loss = NativeCrossEntropyLoss()(logits, targets)
    return logits, loss, control


def test_buffer_only_checkpoint_load_leaves_an_earlier_eval_graph_valid(
    tmp_path
):
    """§7 proved across the *whole* integrated model and over the real
    archive path: replacing all four running buffers through a
    parameter-free checkpoint holder that aliases the model's exact
    registered objects changes no earlier eval graph's gradient, moves no
    parameter version, and preserves every identity."""
    model = NativePhaseFClassifier()
    _nontrivial_running_state(model)
    x, targets = _inputs()
    logits, loss, control = _eval_graph_with_control(model, x, targets)

    graph_storages = _graph_storage_ids(loss)
    buffer_objects = {name: buffer
                      for name, buffer in model.named_buffers()}
    old_cores = {name: buffer._core for name, buffer in buffer_objects.items()}
    assert not ({id(core.storage) for core in old_cores.values()}
                & graph_storages)
    versions = {name: p.version for name, p in model.named_parameters()}

    # A donor archive holding four completely different running states.
    donor_tensors = {
        "bn2d_mean": NativeTensor.from_array([7.0] * CONV_CHANNELS),
        "bn2d_var": NativeTensor.from_array([25.0] * CONV_CHANNELS),
        "bn1d_mean": NativeTensor.from_array([9.0] * HIDDEN_FEATURES),
        "bn1d_var": NativeTensor.from_array([36.0] * HIDDEN_FEATURES),
    }
    donor = _RunningStatHolder(**donor_tensors)
    assert donor.parameters() == []
    path = os.path.join(str(tmp_path), "running_only.npz")
    save_native_checkpoint(path, donor, metadata={"kind": "running-stats"})
    for tensor in donor_tensors.values():
        tensor.close()

    # The holder aliases the integrated model's exact buffer objects, so
    # the real archive path drives them without touching gamma/beta.
    holder = _RunningStatHolder(
        bn2d_mean=model.batch_norm2d.running_mean,
        bn2d_var=model.batch_norm2d.running_var,
        bn1d_mean=model.batch_norm1d.running_mean,
        bn1d_var=model.batch_norm1d.running_var,
    )
    assert holder.parameters() == []
    metadata = load_native_checkpoint(path, holder)
    assert metadata == {"kind": "running-stats"}

    # Identities survived, cores were replaced and closed, no version moved.
    for name, buffer in model.named_buffers():
        assert buffer is buffer_objects[name], name
        assert buffer._core is not old_cores[name], name
        assert old_cores[name]._closed is True, name
    assert np.allclose(model.batch_norm2d.running_mean.to_numpy(), 7.0)
    assert np.allclose(model.batch_norm1d.running_var.to_numpy(), 36.0)
    assert {name: p.version for name, p in model.named_parameters()} == versions

    # The old graph is still valid and still gives the forward's answer.
    loss.backward()
    for name, parameter in model.named_parameters():
        assert np.allclose(parameter.grad.to_numpy(), control[name],
                           atol=1e-12), name
    for parameter in model.parameters():
        parameter.grad.close()
    _close_run(model, None, loss, logits, x)
    del holder, donor


def test_a_later_training_step_cannot_change_an_earlier_eval_backward():
    """The other real buffer-mutation path: a full training forward
    advances all four running buffers, and the earlier eval graph is
    still correct for the forward it recorded."""
    model = NativePhaseFClassifier()
    _nontrivial_running_state(model)
    x, targets = _inputs()
    logits, loss, control = _eval_graph_with_control(model, x, targets)
    before = {name: buffer.to_numpy().copy()
              for name, buffer in model.named_buffers()}

    model.train()
    training_logits = model(x)
    training_logits.close()
    model.eval()
    for name, buffer in model.named_buffers():
        assert not np.array_equal(buffer.to_numpy(), before[name]), name

    loss.backward()
    for name, parameter in model.named_parameters():
        assert np.allclose(parameter.grad.to_numpy(), control[name],
                           atol=1e-12), name
    for parameter in model.parameters():
        parameter.grad.close()
    _close_run(model, None, loss, logits, x)


@pytest.mark.parametrize("mutate", ["full_checkpoint", "affine_parameter"])
def test_parameter_mutation_stales_an_eval_graph_and_buffers_do_not(
    tmp_path, mutate
):
    """The distinction the design insists on: a **full** checkpoint load
    (which also replaces gamma/beta) and a direct ``copy_value_`` on a
    normalization affine parameter both stale an earlier graph through
    the unchanged v3.7 **parameter** rule — never through the buffers."""
    model = NativePhaseFClassifier()
    _nontrivial_running_state(model)
    x, targets = _inputs()
    model.eval()
    logits = model(x)
    loss = NativeCrossEntropyLoss()(logits, targets)
    resources = _saved_resources(loss)
    buffers_before = {name: buffer.to_numpy().copy()
                      for name, buffer in model.named_buffers()}
    versions = {name: p.version for name, p in model.named_parameters()}

    if mutate == "full_checkpoint":
        donor = NativePhaseFClassifier(conv_seed=7, hidden_seed=8, head_seed=9)
        path = os.path.join(str(tmp_path), "full.npz")
        save_native_checkpoint(path, donor)
        _close_module(donor)
        load_native_checkpoint(path, model)
        moved = [name for name, p in model.named_parameters()
                 if p.version != versions[name]]
        assert set(moved) == set(PARAMETER_NAMES)
    else:
        replacement = NativeTensor.from_array(
            model.layer_norm.weight.to_numpy() + 3.0
        )
        try:
            model.layer_norm.weight.copy_value_(replacement)
        finally:
            replacement.close()
        assert model.layer_norm.weight.version == versions["layer_norm.weight"] + 1

    with pytest.raises(RuntimeError, match="stale parameter value"):
        loss.backward()
    # Nothing committed anywhere...
    assert all(p.grad is None for p in model.parameters())
    # ...and the failure is attributed to a **parameter** version. A full
    # checkpoint load replaces the running buffers too, but the dedicated
    # buffer-only tests above prove that half is never a cause; the
    # affine-parameter variant isolates the cause with the buffers
    # provably untouched.
    if mutate == "affine_parameter":
        for name, buffer in model.named_buffers():
            assert np.array_equal(buffer.to_numpy(),
                                  buffers_before[name]), name
        moved = [name for name, p in model.named_parameters()
                 if p.version != versions[name]]
        assert moved == ["layer_norm.weight"]
    # The retry contract: resources stay valid until the graph is
    # released. The closed node's own saved probabilities go immediately;
    # the intermediate nodes' snapshots and winners go with the abandoned
    # chain, at a wrapper-cycle boundary.
    assert loss._graph_freed is False
    assert all(not _resource_closed(r) for r in resources)
    own = loss._graph_resources
    loss.close()
    assert all(_resource_closed(r) for r in own)
    logits.close()
    del loss, logits, own
    _collect()
    assert all(_resource_closed(r) for r in resources)

    # A fresh forward after the mutation works normally.
    fresh_logits = model(x)
    fresh_loss = NativeCrossEntropyLoss()(fresh_logits, targets)
    fresh_loss.backward()
    assert all(p.grad is not None for p in model.parameters())
    for parameter in model.parameters():
        parameter.grad.close()
    _close_run(model, None, fresh_loss, fresh_logits, x)


# ==========================================================================
# 5. The versioning archetypes meeting a normalized graph
# ==========================================================================

def _archetype_graph(model, x, targets, exp_parameter, log_parameter):
    """One graph: the integrated normalized classification loss plus the
    two Phase-E versioning archetypes."""
    model.eval()
    logits = model(x)
    classification = NativeCrossEntropyLoss()(logits, targets)
    exponentials = exp_parameter.exp()
    exp_branch = exponentials.sum()
    logarithms = log_parameter.log()
    log_branch = logarithms.sum()
    total = classification.add(exp_branch).add(log_branch)
    return {
        "logits": logits, "classification": classification,
        "exponentials": exponentials, "exp_branch": exp_branch,
        "logarithms": logarithms, "log_branch": log_branch, "total": total,
    }


def _close_archetype(graph):
    for key in ("total", "log_branch", "logarithms", "exp_branch",
                "exponentials", "classification", "logits"):
        graph[key].close()


def test_saved_state_archetypes_survive_post_forward_mutation():
    """Cross-entropy reads saved probabilities, ``exp`` reads its saved
    output, and BatchNorm eval reads forward-time snapshots — so mutating
    the running buffers and the ``exp`` parameter after the forward
    leaves every one of those edges valid, in one graph."""
    model = NativePhaseFClassifier()
    _nontrivial_running_state(model)
    x, targets = _inputs()
    exp_values = np.array([[0.25, -0.5], [1.0, 0.75]])
    log_values = np.array([[2.0, 3.0], [4.0, 5.0]])

    # The clean control: the same graph, differentiated with nothing
    # mutated in between.
    control_model = NativePhaseFClassifier()
    state = model.state_dict()
    try:
        control_model.load_state_dict(state)
    finally:
        for snapshot in state.values():
            snapshot.close()
    control_exp = NativeParameter(exp_values.copy())
    control_log = NativeParameter(log_values.copy())
    control_graph = _archetype_graph(control_model, x, targets,
                                     control_exp, control_log)
    control_graph["total"].backward()
    control = {name: parameter.grad.to_numpy().copy()
               for name, parameter in control_model.named_parameters()}
    control["exp"] = control_exp.grad.to_numpy().copy()
    control["log"] = control_log.grad.to_numpy().copy()
    _close_archetype(control_graph)
    for parameter in list(control_model.parameters()) + [control_exp,
                                                          control_log]:
        if parameter.grad is not None:
            parameter.grad.close()
    _close_run(control_model, None, control_exp, control_log)

    exp_parameter = NativeParameter(exp_values.copy())
    log_parameter = NativeParameter(log_values.copy())
    graph = _archetype_graph(model, x, targets, exp_parameter, log_parameter)
    # The cross-entropy edge records no expected version at all.
    assert graph["classification"]._expected_versions == ()
    assert graph["exponentials"]._expected_versions == ()
    assert graph["logarithms"]._expected_versions != ()

    # Mutate the saved-state inputs: all four running buffers and exp's
    # direct parameter. The log parameter is deliberately left alone.
    _load_buffers(model.batch_norm2d,
                  running_mean=[5.0] * CONV_CHANNELS,
                  running_var=[7.0] * CONV_CHANNELS)
    _load_buffers(model.batch_norm1d,
                  running_mean=[6.0] * HIDDEN_FEATURES,
                  running_var=[8.0] * HIDDEN_FEATURES)
    replacement = NativeTensor.from_array(exp_values + 10.0)
    try:
        exp_parameter.copy_value_(replacement)
    finally:
        replacement.close()

    graph["total"].backward()

    for name, parameter in model.named_parameters():
        assert np.allclose(parameter.grad.to_numpy(), control[name],
                           atol=1e-12), name
    assert np.allclose(exp_parameter.grad.to_numpy(), control["exp"],
                       atol=1e-14)
    assert np.allclose(log_parameter.grad.to_numpy(), control["log"],
                       atol=1e-14)
    _close_archetype(graph)
    for parameter in list(model.parameters()) + [exp_parameter, log_parameter]:
        if parameter.grad is not None:
            parameter.grad.close()
    _close_run(model, None, x)


def test_a_live_value_reread_stales_the_whole_normalized_graph():
    """The other half: ``log`` rereads its parent's live value, so
    mutating it invalidates the *entire* combined graph before any branch
    commits a gradient — including the normalized classification branch
    that would have been fine alone."""
    model = NativePhaseFClassifier()
    _nontrivial_running_state(model)
    x, targets = _inputs()
    exp_parameter = NativeParameter(np.array([[0.25, -0.5]]))
    log_parameter = NativeParameter(np.array([[2.0, 3.0]]))
    graph = _archetype_graph(model, x, targets, exp_parameter, log_parameter)
    resources = _saved_resources(graph["total"])

    replacement = NativeTensor.from_array(np.array([[8.0, 16.0]]))
    try:
        log_parameter.copy_value_(replacement)
    finally:
        replacement.close()

    with pytest.raises(RuntimeError, match="stale parameter value"):
        graph["total"].backward()
    # No branch committed anything.
    assert all(p.grad is None for p in model.parameters())
    assert exp_parameter.grad is None and log_parameter.grad is None
    # The saved resources are neither leaked nor silently released...
    assert all(not _resource_closed(r) for r in resources)
    own = graph["classification"]._graph_resources
    _close_archetype(graph)
    assert all(_resource_closed(r) for r in own)
    graph.clear()
    del own
    _collect()
    assert all(_resource_closed(r) for r in resources)

    # ...and a fresh graph after the mutation differentiates normally.
    again = _archetype_graph(model, x, targets, exp_parameter, log_parameter)
    again["total"].backward()
    assert all(p.grad is not None for p in model.parameters())
    assert log_parameter.grad is not None
    _close_archetype(again)
    for parameter in list(model.parameters()) + [exp_parameter, log_parameter]:
        if parameter.grad is not None:
            parameter.grad.close()
    _close_run(model, None, x)


# ==========================================================================
# 6. Shared parameters through a normalized model
# ==========================================================================

class _SharedNormalizedModel(NativeModule):
    """**Test-only.** The same ``NativeParameter`` object registered under
    two paths and used twice in one forward, with a normalization module
    in the path."""

    def __init__(self):
        super().__init__()
        self.scale = NativeParameter(np.array([[2.0, -1.0], [0.5, 1.5]]))
        self.alias = self.scale               # the *same* object, second path
        self.norm = NativeLayerNorm(2)

    def forward(self, x):
        first = x.matmul(self.scale)
        normalized = self.norm(first)
        return normalized.matmul(self.alias)


def test_a_shared_parameter_through_a_normalized_model_updates_once():
    model = _SharedNormalizedModel()
    names = [name for name, _ in model.named_parameters()]
    # One entry for the shared object, under its first-discovered name.
    assert names == ["scale", "norm.weight", "norm.bias"]
    assert len(model.parameters()) == 3
    assert model.alias is model.scale
    assert sum(1 for p in model.parameters() if p is model.scale) == 1
    assert _state_keys(model) == names

    optimizer = NativeAdam(model.parameters(), lr=0.1)
    assert len(optimizer.parameters()) == 3          # one slot, not two
    assert sum(1 for p in optimizer.parameters() if p is model.scale) == 1

    x = NativeTensor.from_array(np.array([[1.0, 2.0], [3.0, -1.0]]))
    out = model(x)
    loss = out.sum()
    loss.backward()
    assert model.scale.grad is not None
    accumulated = model.scale.grad.to_numpy().copy()

    # An independent control: the same forward differentiated with the two
    # uses split across two *distinct* parameters, whose gradients must
    # sum to the shared one.
    left = NativeParameter(np.array([[2.0, -1.0], [0.5, 1.5]]))
    right = NativeParameter(np.array([[2.0, -1.0], [0.5, 1.5]]))
    control_norm = NativeLayerNorm(2)
    control_state = model.norm.state_dict()
    try:
        control_norm.load_state_dict(control_state)
    finally:
        for snapshot in control_state.values():
            snapshot.close()
    control_x = NativeTensor.from_array(np.array([[1.0, 2.0], [3.0, -1.0]]))
    control_out = control_norm(control_x.matmul(left)).matmul(right)
    control_loss = control_out.sum()
    control_loss.backward()
    assert np.allclose(accumulated,
                       left.grad.to_numpy() + right.grad.to_numpy(),
                       atol=1e-12)
    _close_run(control_norm, None, control_loss, control_out, control_x,
               left.grad, right.grad, left, right)

    version = model.scale.version
    before = model.scale.to_numpy().copy()
    optimizer.step()
    assert model.scale.version == version + 1        # exactly one increment
    assert list(optimizer.step_counts) == [1, 1, 1]
    assert not np.array_equal(model.scale.to_numpy(), before)
    # Both aliases observe the same new value, because they are one object.
    assert model.alias is model.scale
    assert np.array_equal(model.alias.to_numpy(), model.scale.to_numpy())
    _close_run(None, optimizer, loss, out, model.scale.grad,
               model.norm.weight.grad, model.norm.bias.grad, x)
    shared = model.scale
    _close_module(model)                             # closes it exactly once
    assert shared.closed is True


def test_a_shared_parameter_survives_a_checkpoint_round_trip(tmp_path):
    model = _SharedNormalizedModel()
    replacement = NativeTensor.from_array(np.array([[9.0, 8.0], [7.0, 6.0]]))
    try:
        model.scale.copy_value_(replacement)
    finally:
        replacement.close()
    path = os.path.join(str(tmp_path), "shared.npz")
    save_native_checkpoint(path, model)
    with np.load(path, allow_pickle=False) as archive:
        manifest = archive["manifest"].tobytes().decode("utf-8")
    assert '"scale"' in manifest
    assert '"alias"' not in manifest                 # one canonical entry

    fresh = _SharedNormalizedModel()
    scale_id, alias_id = id(fresh.scale), id(fresh.alias)
    load_native_checkpoint(path, fresh)
    assert fresh.alias is fresh.scale
    assert id(fresh.scale) == scale_id and id(fresh.alias) == alias_id
    assert np.allclose(fresh.scale.to_numpy(), [[9.0, 8.0], [7.0, 6.0]])
    _close_module(model)
    _close_module(fresh)


# ==========================================================================
# 7. Frozen parameters through a normalized model
# ==========================================================================

class _FrozenNormalizedModel(NativeModule):
    """**Test-only.** A frozen parameter that really participates in the
    forward, alongside a normalization module and a trainable head."""

    def __init__(self):
        super().__init__()
        self.frozen = NativeParameter(np.array([[1.5, -0.5], [0.25, 2.0]]),
                                      requires_grad=False)
        self.norm = NativeBatchNorm1d(2, momentum=0.5)
        self.head = NativeLinear(2, 1, seed=3)

    def forward(self, x):
        return self.head(self.norm(x.matmul(self.frozen)))


def test_a_frozen_parameter_stays_registered_persisted_and_skipped(tmp_path):
    model = _FrozenNormalizedModel()
    names = [name for name, _ in model.named_parameters()]
    assert "frozen" in names                     # discoverable, not hidden
    assert model.frozen.requires_grad is False
    optimizer = NativeAdam(model.parameters(), lr=0.1)
    frozen_index = optimizer.parameters().index(model.frozen)

    x = NativeTensor.from_array(np.array([[1.0, 2.0], [3.0, -1.0],
                                          [0.5, 0.25], [-2.0, 1.0]]))
    y = NativeTensor.from_array(np.array([[1.0], [0.0], [0.5], [-1.0]]))
    before = model.frozen.to_numpy().copy()
    version = model.frozen.version
    buffers_before = {n: b.to_numpy().copy() for n, b in model.named_buffers()}
    state = optimizer.state_dict()
    frozen_m = state["m"][frozen_index].to_numpy().copy()
    frozen_v = state["v"][frozen_index].to_numpy().copy()
    for key in ("m", "v"):
        for tensor in state[key]:
            tensor.close()

    model.train()
    out = model(x)
    loss = out.subtract(y).multiply(out.subtract(y)).sum()
    # Neither the input nor the frozen parameter requires grad, so their
    # matmul builds no graph node at all — the frozen parameter is
    # genuinely absent from the graph history and backward can never
    # reach it. The trainable parameters downstream of it still are.
    reachable = _graph_object_ids(loss)
    assert id(model.frozen) not in reachable
    assert id(model.head.weight) in reachable
    assert id(model.norm.gamma) in reachable
    loss.backward()
    assert model.frozen.grad is None
    assert loss._graph_freed is True
    optimizer.step()

    assert np.array_equal(model.frozen.to_numpy(), before)
    assert model.frozen.version == version
    assert list(optimizer.step_counts)[frozen_index] == 0
    after = optimizer.state_dict()
    assert np.array_equal(after["m"][frozen_index].to_numpy(), frozen_m)
    assert np.array_equal(after["v"][frozen_index].to_numpy(), frozen_v)
    for key in ("m", "v"):
        for tensor in after[key]:
            tensor.close()
    # The active parameters trained, and the normalization buffers moved.
    assert model.head.weight.version == 1
    for name, buffer in model.named_buffers():
        assert not np.array_equal(buffer.to_numpy(), buffers_before[name]), name

    # It persists numerically and reloads still frozen.
    path = os.path.join(str(tmp_path), "frozen.npz")
    save_native_checkpoint(path, model)
    fresh = _FrozenNormalizedModel()
    load_native_checkpoint(path, fresh)
    assert fresh.frozen.requires_grad is False
    assert np.array_equal(fresh.frozen.to_numpy(), before)
    _close_run(model, optimizer, loss, out, x, y)
    _close_module(fresh)


# ==========================================================================
# 8. A non-contiguous NCHW input through the full stack
# ==========================================================================

def test_a_non_contiguous_nchw_input_runs_the_whole_integrated_stack(
    live_storages
):
    """Policy B through every layer at once: a transposed view of an
    equal-sized spatial pair reaches convolution, both normalizations,
    pooling, and the fused loss and produces exactly the contiguous
    answer, in **both** modes, without mutating the caller's base."""
    images, targets = _dataset()
    values = np.asarray(images, dtype=np.float64)
    # H == W, so swapping them is a valid NCHW shape and a real strided
    # view rather than a reshape.
    transposed = np.transpose(values, (0, 1, 3, 2))
    _collect()
    baseline = len(live_storages)

    base = NativeTensor.from_array(values)
    view = base.transpose((0, 1, 3, 2))
    assert view.shape == values.shape and view.contiguous is False
    contiguous = NativeTensor.from_array(transposed)

    for mode in ("train", "eval"):
        strided_model = NativePhaseFClassifier()
        contiguous_model = NativePhaseFClassifier()
        _nontrivial_running_state(strided_model)
        _nontrivial_running_state(contiguous_model)
        strided_model.train(mode == "train")
        contiguous_model.train(mode == "train")

        strided_logits = strided_model(view)
        contiguous_logits = contiguous_model(contiguous)
        assert strided_logits.owns_core and strided_logits.contiguous
        assert np.allclose(strided_logits.to_numpy(),
                           contiguous_logits.to_numpy(), atol=1e-12), mode
        strided_loss = NativeCrossEntropyLoss()(strided_logits, targets)
        contiguous_loss = NativeCrossEntropyLoss()(contiguous_logits, targets)
        assert float(strided_loss.to_numpy()) == pytest.approx(
            float(contiguous_loss.to_numpy()), rel=1e-12
        ), mode
        strided_loss.backward()
        contiguous_loss.backward()
        for (name, a), (_, b) in zip(strided_model.named_parameters(),
                                     contiguous_model.named_parameters()):
            assert np.allclose(a.grad.to_numpy(), b.grad.to_numpy(),
                               atol=1e-12), (mode, name)
        for (name, a), (_, b) in zip(strided_model.named_buffers(),
                                     contiguous_model.named_buffers()):
            assert np.allclose(a.to_numpy(), b.to_numpy(), atol=1e-12), (
                mode, name
            )
        # The caller's base and view are untouched, and the view is alive.
        assert np.array_equal(base.to_numpy(), values)
        assert np.array_equal(view.to_numpy(), transposed)
        assert view.closed is False and base.closed is False
        for parameter in list(strided_model.parameters()) + list(
            contiguous_model.parameters()
        ):
            if parameter.grad is not None:
                parameter.grad.close()
        _close_run(strided_model, None, strided_loss, strided_logits)
        _close_run(contiguous_model, None, contiguous_loss, contiguous_logits)

    view.close()          # a borrowing view frees only its own wrapper
    assert base.closed is False
    base.close()
    contiguous.close()
    _collect()
    assert len(live_storages) == baseline


# ==========================================================================
# 9. Stable and native lines stay separate, through normalization
# ==========================================================================

def test_the_normalization_modules_reject_stable_tensors():
    stable_2d = tensorforge.Tensor(np.zeros((2, HIDDEN_FEATURES)))
    stable_4d = tensorforge.Tensor(np.zeros((2, CONV_CHANNELS, 3, 3)))
    with pytest.raises(TypeError):
        NativeLayerNorm(HIDDEN_FEATURES)(stable_2d)
    with pytest.raises(TypeError):
        NativeBatchNorm1d(HIDDEN_FEATURES)(stable_2d)
    with pytest.raises(TypeError):
        NativeBatchNorm2d(CONV_CHANNELS)(stable_4d)


def test_the_stable_normalization_modules_reject_native_tensors():
    native = NativeTensor.from_array(np.zeros((2, HIDDEN_FEATURES)))
    for layer in (tensorforge.nn.LayerNorm(HIDDEN_FEATURES),
                  tensorforge.nn.BatchNorm1d(HIDDEN_FEATURES)):
        with pytest.raises(Exception):
            layer(native)
    native.close()


def test_no_implicit_conversion_dispatch_or_bridge_exists():
    import tensorforge.experimental as experimental

    for name in experimental.__all__:
        assert not hasattr(tensorforge, name), name
        assert not hasattr(tensorforge.nn, name), name
    for bridge in ("to_native", "to_tensor", "native", "as_native"):
        assert not hasattr(tensorforge.Tensor, bridge), bridge
        assert not hasattr(NativeTensor, bridge), bridge
    for dispatcher in ("set_backend", "use_native", "backend", "dispatch"):
        assert not hasattr(tensorforge, dispatcher), dispatcher
    for cls in (NativeLayerNorm, NativeBatchNorm1d, NativeBatchNorm2d):
        assert issubclass(cls, NativeModule)
        assert not issubclass(cls, tensorforge.nn.Module)
        assert not hasattr(cls, "to_stable")
    # Losses and optimizers reject the other line's objects.
    native_logits = NativeTensor.from_array(np.zeros((2, NUM_CLASSES)))
    stable_logits = tensorforge.Tensor(np.zeros((2, NUM_CLASSES)))
    with pytest.raises(TypeError):
        NativeCrossEntropyLoss()(stable_logits, [0, 1])
    with pytest.raises(Exception):
        tensorforge.nn.cross_entropy(native_logits, [0, 1])
    with pytest.raises(Exception):
        NativeAdam([tensorforge.nn.Parameter(np.zeros(2))], lr=0.1)
    # The stable optimizer rejects a NativeParameter too — it reaches for
    # ``.data``, which the native leaf deliberately does not have.
    native_parameter = NativeParameter(np.zeros(2))
    with pytest.raises(Exception):
        tensorforge.optim.Adam([native_parameter], lr=0.1)
    assert not hasattr(native_parameter, "data")
    native_logits.close()
    native_parameter.close()


def test_the_stable_normalization_path_still_behaves_normally():
    """F8 changes nothing in the stable framework: a representative
    stable LayerNorm + BatchNorm1d train/eval path is unaffected."""
    from tensorforge.nn import BatchNorm1d, LayerNorm, Linear, ReLU, Sequential
    from tensorforge.optim import SGD

    model = Sequential(Linear(3, 4), BatchNorm1d(4), ReLU(), LayerNorm(4),
                       Linear(4, 1))
    x = tensorforge.Tensor(np.array([[1.0, 2.0, 0.5], [0.25, -1.0, 2.0],
                                     [0.5, 0.5, 0.5], [-1.0, 0.0, 1.0]]))
    y = tensorforge.Tensor(np.array([[1.0], [0.0], [0.5], [-1.0]]))
    optimizer = SGD(model.parameters(), lr=0.05)
    model.train()
    batch_norm = model.modules[1]
    running_before = batch_norm.running_mean.copy()
    loss = tensorforge.nn.mse_loss(model(x), y)
    loss.backward()
    optimizer.step()
    assert np.isfinite(float(loss.data))
    assert not np.array_equal(batch_norm.running_mean, running_before)
    model.eval()
    evaluated = model(x)
    assert evaluated.data.shape == (4, 1)
    assert np.isfinite(evaluated.data).all()
    # Eval really used the running statistics, not the batch's own.
    model.train()
    assert not np.allclose(model(x).data, evaluated.data)


# ==========================================================================
# 10. Semantic inventory, export, and capability guardrails
# ==========================================================================

_NORMALIZATION_CLASSES = ("NativeLayerNorm", "NativeBatchNorm1d",
                          "NativeBatchNorm2d")
_NORMALIZATION_NAMES = ("layer_norm", "batch_norm", "layernorm", "batchnorm",
                        "normalize", "normalization", "layer_norm_forward",
                        "batch_norm_forward", "running_stats")


def test_every_normalization_module_is_exported_registered_and_callable():
    import tensorforge.experimental as experimental

    for name in _NORMALIZATION_CLASSES:
        assert name in experimental.__all__, name
        assert name in cpp.NATIVE_MODULES, name
        assert callable(getattr(experimental, name)), name
        assert issubclass(getattr(experimental, name), NativeModule), name
    # Every advertised module resolves to a real exported class.
    for name in cpp.NATIVE_MODULES:
        if name == "NativeModule":
            assert callable(NativeModule)
            continue
        assert name in experimental.__all__, name
        assert callable(getattr(experimental, name)), name
    # The shared private implementation stays private.
    from tensorforge.experimental import native_batchnorm

    assert issubclass(NativeBatchNorm1d, native_batchnorm._NativeBatchNorm)
    assert issubclass(NativeBatchNorm2d, native_batchnorm._NativeBatchNorm)
    for private in ("_NativeBatchNorm", "NativeBatchNorm"):
        assert private not in experimental.__all__, private
        assert not hasattr(experimental, private), private


def test_no_normalization_operation_kernel_or_abi_symbol_exists():
    """The design's single most important structural decision, checked
    against reality: the three modules are compositions, so no
    normalization operation, Core method, kernel, or ABI symbol exists."""
    for inventory in (cpp.RAW_KERNELS, cpp.TENSOR_CORE_KERNELS,
                      cpp.TENSOR_CORE_OPS, cpp.AUTOGRAD_OPS):
        for entry in inventory:
            for banned in _NORMALIZATION_NAMES:
                assert banned not in entry.lower(), (entry, banned)
    for banned in _NORMALIZATION_NAMES:
        for kernel in cpp._CHECKED_KERNELS:
            assert banned not in kernel.lower(), (kernel, banned)
    for absent in ("layer_norm", "batch_norm", "layernorm", "batchnorm"):
        assert not hasattr(NativeTensor, absent), absent
        assert not hasattr(cpp.NativeTensorCore, absent), absent
    for symbol in ("tf_core_layer_norm", "tf_core_batch_norm",
                   "tf_core_normalize", "tf_core_running_update",
                   "tf_layer_norm", "tf_batch_norm"):
        assert symbol not in cpp._CHECKED_KERNELS, symbol
        assert not hasattr(cpp, symbol), symbol
    # No functional helper appeared either.
    import tensorforge.experimental as experimental

    for helper in ("layer_norm", "batch_norm", "normalize"):
        assert not hasattr(experimental, helper), helper
        assert helper not in experimental.__all__, helper
    # The module sources build no graph node of their own.
    for name in ("native_layernorm.py", "native_batchnorm.py"):
        source = (REPO_ROOT / "src" / "tensorforge" / "experimental"
                  / name).read_text(encoding="utf-8")
        assert "_from_op(" not in source, name
        assert "import ctypes" not in source, name
        assert "argtypes" not in source and "restype" not in source, name


def test_every_state_capability_maps_to_a_real_api():
    import tensorforge.experimental as experimental

    assert cpp.STATE_SUPPORT == (
        "persistent_buffers", "state_dict", "load_state_dict",
        "generator_state",   # Phase G, milestone G1 (in-memory only)
        "save_native_checkpoint", "load_native_checkpoint",
        "checkpoint_generator_state",   # Phase G, milestone G5 (the file half)
    )
    assert callable(NativeModule.state_dict)
    assert callable(NativeModule.load_state_dict)
    for name in ("register_buffer", "buffers", "named_buffers"):
        assert callable(getattr(NativeModule, name)), name
    for name in ("save_native_checkpoint", "load_native_checkpoint"):
        assert callable(getattr(experimental, name)), name
        assert name in experimental.__all__, name
    assert native_checkpoint._FORMAT == "tensorforge.native_checkpoint"
    assert native_checkpoint._FORMAT_VERSION == 3


def test_the_remaining_capability_boundary_is_unchanged():
    import tensorforge.experimental as experimental

    assert cpp.UNSUPPORTED == ("float32", "cuda", "amp")
    assert cpp.SUPPORTED_DTYPES == ("float64",)
    assert cpp.SUPPORTED_DEVICES == ("cpu",)
    for never in ("NativeBatchNorm3d", "NativeInstanceNorm",
                  "NativeGroupNorm", "NativeRMSNorm", "NativeRNG",
                  "NativeDataLoader"):
        assert not hasattr(experimental, never), never
        assert never not in experimental.__all__, never
        assert never not in cpp.NATIVE_MODULES, never
    # "NativeDropout" left that list at Phase G milestone G4, which
    # shipped and exported it. It is a Phase-G module, it carries no
    # Phase-F capability, and the *capability* it is named after is still
    # in UNSUPPORTED above — so the boundary this test guards is
    # unchanged.
    assert hasattr(experimental, "NativeDropout")
    assert "NativeDropout" in experimental.__all__
    assert "NativeDropout" in cpp.NATIVE_MODULES
    # NativeGenerator left that list at Phase G milestone G1, which ships
    # random *state* and generates no random values. It is exported, but
    # it is not a module and carries no numerical capability — the
    # boundary this test guards is unchanged.
    assert hasattr(experimental, "NativeGenerator")
    assert "NativeGenerator" not in cpp.NATIVE_MODULES
    for never in ("rand", "randn", "manual_seed", "to", "cuda",
                  "half", "float"):
        assert not hasattr(NativeTensor, never), never
    # "dropout" left that list at Phase G milestone G3, which shipped the
    # differentiable NativeTensor.dropout. It is an *operation*, and it
    # takes an explicit NativeGenerator — there is still no global stream
    # and no manual_seed, which is the boundary this test actually
    # guards. The *capability* it is named after left UNSUPPORTED later
    # still, at the G10 closure; both are Phase-G events and neither is a
    # normalization change.
    assert hasattr(NativeTensor, "dropout")
    assert "dropout" in cpp.AUTOGRAD_OPS
    assert "dropout" not in cpp.UNSUPPORTED
    assert "dropout" not in cpp.NATIVE_MODULES
    with pytest.raises((ValueError, TypeError)):
        NativeTensor.zeros((2, 2), dtype="float32")
    with pytest.raises((ValueError, TypeError)):
        NativeTensor.zeros((2, 2), device="cuda")
    # Implemented and unsupported names are disjoint. Phase G held one
    # deliberate exception for G3-G9 (design §19) — "dropout" named both
    # a shipped operation and an unclosed capability — and the G10
    # closure ended it. No Phase-F name was ever involved, which is
    # exactly what this asserts.
    implemented = (set(cpp.TENSOR_CORE_OPS) | set(cpp.AUTOGRAD_OPS)
                   | set(cpp.RAW_KERNELS) | set(cpp.NATIVE_MODULES)
                   | set(cpp.NATIVE_LOSSES) | set(cpp.NATIVE_METRICS)
                   | set(cpp.NATIVE_OPTIMIZERS))
    assert implemented & set(cpp.UNSUPPORTED) == set()


def test_the_phase_f_artifacts_are_present_and_no_cpp_file_was_added():
    for relative in (
        "docs/native_normalization_design.md",
        "src/tensorforge/experimental/native_layernorm.py",
        "src/tensorforge/experimental/native_batchnorm.py",
        "src/tensorforge/experimental/_native_state.py",
        "examples/native_normalization_training.py",
        "benchmarks/benchmark_native_normalization.py",
        "tests/test_native_layernorm.py",
        "tests/test_native_batchnorm1d.py",
        "tests/test_native_batchnorm2d.py",
        "tests/test_native_normalization_state.py",
        "tests/test_native_normalization_training.py",
        "tests/test_native_normalization_benchmark.py",
        "tests/test_native_phase_f.py",
    ):
        assert (REPO_ROOT / relative).is_file(), relative
    # No Phase-F C++ source, header, or CTest was added.
    for directory, patterns in ((REPO_ROOT / "cpp" / "src", ("*norm*",)),
                                (REPO_ROOT / "cpp" / "include", ("*norm*",)),
                                (REPO_ROOT / "cpp" / "tests",
                                 ("*norm*", "*batch*", "*layer*"))):
        for pattern in patterns:
            assert not list(directory.glob(pattern)), (directory.name, pattern)
    cmake = (REPO_ROOT / "cpp" / "CMakeLists.txt").read_text(encoding="utf-8")
    for banned in ("normalization", "batchnorm", "layernorm"):
        assert banned not in cmake.lower(), banned


# ==========================================================================
# 11. Failure boundaries — precisely, and without over-claiming
# ==========================================================================

def _settled_model():
    """A model whose optimizer state has already been allocated and
    advanced once, so a failure test observes steady-state behavior."""
    model = NativePhaseFClassifier()
    optimizer = NativeAdam(model.parameters(), lr=LR)
    x, targets = _inputs()
    _train_step(model, NativeCrossEntropyLoss(), optimizer, x, targets)
    return model, optimizer, x, targets


def _fingerprint(model, optimizer):
    return {
        "parameters": {n: p.to_numpy().copy()
                       for n, p in model.named_parameters()},
        "versions": {n: p.version for n, p in model.named_parameters()},
        "buffers": {n: b.to_numpy().copy()
                    for n, b in model.named_buffers()},
        "parameter_ids": [id(p) for p in model.parameters()],
        "buffer_ids": [id(b) for b in model.buffers()],
        "steps": list(optimizer.step_counts),
        "optimizer": _optimizer_values(optimizer),
    }


def _assert_parameters_and_optimizer_untouched(model, optimizer, before):
    for name, parameter in model.named_parameters():
        assert np.array_equal(parameter.to_numpy(),
                              before["parameters"][name]), name
        assert parameter.version == before["versions"][name], name
    assert [id(p) for p in model.parameters()] == before["parameter_ids"]
    assert [id(b) for b in model.buffers()] == before["buffer_ids"]
    assert list(optimizer.step_counts) == before["steps"]
    assert _optimizer_values(optimizer) == before["optimizer"]


def test_boundary_a_a_batchnorm_transaction_failure_is_atomic_per_module(
    live_storages
):
    """Boundary A. A failure inside the **second** BatchNorm's
    running-state commit rolls that pair back completely — but the first
    module's transaction had already committed, and F8 reports that
    honestly rather than pretending a whole forward is transactional."""
    model, optimizer, x, targets = _settled_model()
    before = _fingerprint(model, optimizer)
    bn2d_before = {n: getattr(model.batch_norm2d, n).to_numpy().copy()
                   for n in ("running_mean", "running_var")}
    bn1d_before = {n: getattr(model.batch_norm1d, n).to_numpy().copy()
                   for n in ("running_mean", "running_var")}
    baseline = len(live_storages)

    real_install = _native_state._install_core
    calls = {"n": 0}

    def failing_install(planned, new_core):
        calls["n"] += 1
        # Calls 1-2 are BatchNorm2d's pair; 3-4 are BatchNorm1d's. Fail
        # mid-commit on the second module, after one of its swaps.
        if calls["n"] == 4:
            raise _Boom("injected BatchNorm1d commit failure")
        return real_install(planned, new_core)

    patcher = _injected(_native_state, "_install_core", failing_install)
    model.train()
    with pytest.raises(_Boom, match="injected BatchNorm1d commit failure"):
        model(x)
    patcher.undo()

    # The failing pair rolled back completely: no half-updated statistic.
    for name in ("running_mean", "running_var"):
        assert np.array_equal(
            getattr(model.batch_norm1d, name).to_numpy(), bn1d_before[name]
        ), name
    # ...while the earlier module's already-committed transaction stands.
    # That is the honest boundary: transactions are per-module.
    for name in ("running_mean", "running_var"):
        assert not np.array_equal(
            getattr(model.batch_norm2d, name).to_numpy(), bn2d_before[name]
        ), name
    # No parameter, version, gradient, or optimizer state moved.
    _assert_parameters_and_optimizer_untouched(model, optimizer, before)
    assert all(p.grad is None for p in model.parameters())
    _collect()
    assert len(live_storages) <= baseline
    assert cpp._require_library().tf_last_error_code() == cpp.TF_OK
    # A later full step succeeds.
    assert math.isfinite(
        _train_step(model, NativeCrossEntropyLoss(), optimizer, x, targets)
    )
    _close_run(model, optimizer, x)


@pytest.mark.parametrize("failure", ["invalid_targets", "backward"])
def test_boundary_b_a_failure_after_the_forward_keeps_the_committed_updates(
    monkeypatch, failure
):
    """Boundary B. Once the forward has run, the BatchNorm running
    updates are **committed**. A later loss or backward failure must not
    retroactively roll them back — and must not commit a gradient or an
    optimizer update either."""
    model, optimizer, x, targets = _settled_model()
    before = _fingerprint(model, optimizer)
    model.train()
    logits = model(x)
    # Both pairs advanced during the forward.
    for name, buffer in model.named_buffers():
        assert not np.array_equal(buffer.to_numpy(), before["buffers"][name])
    committed = {n: b.to_numpy().copy() for n, b in model.named_buffers()}

    if failure == "invalid_targets":
        with pytest.raises((TypeError, ValueError, IndexError)):
            NativeCrossEntropyLoss()(logits, [0] * (len(targets) - 1))
        with pytest.raises((TypeError, ValueError, IndexError)):
            NativeCrossEntropyLoss()(logits, [NUM_CLASSES] * len(targets))
        loss = None
    else:
        loss = NativeCrossEntropyLoss()(logits, targets)
        real_accumulate = NativeTensor._accumulate_grad
        calls = {"n": 0}

        def failing_accumulate(self, grad):
            calls["n"] += 1
            if calls["n"] == 1:      # before anything is committed anywhere
                raise _Boom("injected backward failure")
            return real_accumulate(self, grad)

        monkeypatch.setattr(NativeTensor, "_accumulate_grad",
                            failing_accumulate)
        with pytest.raises(_Boom):
            loss.backward()
        monkeypatch.undo()
        # No partial gradient committed, and the graph stays retryable.
        assert all(p.grad is None for p in model.parameters())
        assert loss._graph_freed is False

    # The forward's committed buffer updates remain — correctly.
    for name, buffer in model.named_buffers():
        assert np.array_equal(buffer.to_numpy(), committed[name]), name
    _assert_parameters_and_optimizer_untouched(model, optimizer, before)
    if loss is not None:
        loss.close()
    logits.close()
    optimizer.zero_grad()
    assert math.isfinite(
        _train_step(model, NativeCrossEntropyLoss(), optimizer, x, targets)
    )
    _close_run(model, optimizer, x)


def test_boundary_c_an_optimizer_staging_failure_commits_nothing(
    live_storages
):
    """Boundary C. After a successful forward and backward, a failure
    while staging the optimizer update leaves every parameter, version,
    moment, and counter untouched, closes every staged temporary, and
    keeps the gradients usable for a retry."""
    model, optimizer, x, targets = _settled_model()
    model.train()
    logits = model(x)
    loss = NativeCrossEntropyLoss()(logits, targets)
    loss.backward()
    before = _fingerprint(model, optimizer)
    gradients = {n: p.grad.to_numpy().copy()
                 for n, p in model.named_parameters()}
    baseline = len(live_storages)

    real_stage = NativeAdam._stage_entry
    calls = {"n": 0}

    # ``*rest`` carries the step's shared scalar-constant holder, which
    # H4 passes to every entry; forwarding it unexamined keeps this
    # injection about *when* staging fails, not about the staging
    # signature.
    def failing_stage(self, index, parameter, grad, *rest):
        calls["n"] += 1
        if calls["n"] == 3:          # after some entries are already staged
            raise _Boom("injected optimizer staging failure")
        return real_stage(self, index, parameter, grad, *rest)

    patcher = _injected(NativeAdam, "_stage_entry", failing_stage)
    with pytest.raises(_Boom, match="injected optimizer staging failure"):
        optimizer.step()
    patcher.undo()

    # Immediate, without gc: nothing committed and nothing staged leaked.
    _assert_parameters_and_optimizer_untouched(model, optimizer, before)
    assert len(live_storages) <= baseline
    # The gradients survived untouched, so the same step retries cleanly.
    for name, parameter in model.named_parameters():
        assert np.array_equal(parameter.grad.to_numpy(), gradients[name]), name
    optimizer.step()
    assert list(optimizer.step_counts) == [s + 1 for s in before["steps"]]
    for name, parameter in model.named_parameters():
        assert parameter.version == before["versions"][name] + 1, name
    optimizer.zero_grad()
    loss.close()
    logits.close()
    _close_run(model, optimizer, x)


def test_boundary_d_a_stale_parameter_backward_keeps_the_forward_update():
    """Boundary D. A version-sensitive mutation after the forward makes
    backward raise deterministically — the running buffers keep the
    update the forward legitimately committed, nothing partial lands, and
    a fresh pass works."""
    model, optimizer, x, targets = _settled_model()
    before = _fingerprint(model, optimizer)
    model.train()
    logits = model(x)
    loss = NativeCrossEntropyLoss()(logits, targets)
    resources = _saved_resources(loss)
    committed = {n: b.to_numpy().copy() for n, b in model.named_buffers()}

    replacement = NativeTensor.from_array(
        model.batch_norm1d.gamma.to_numpy() + 2.0
    )
    try:
        model.batch_norm1d.gamma.copy_value_(replacement)
    finally:
        replacement.close()

    with pytest.raises(RuntimeError, match="stale parameter value"):
        loss.backward()
    assert all(p.grad is None for p in model.parameters())
    for name, buffer in model.named_buffers():
        assert np.array_equal(buffer.to_numpy(), committed[name]), name
    assert list(optimizer.step_counts) == before["steps"]
    assert _optimizer_values(optimizer) == before["optimizer"]
    # Explicit release of the failed graph frees every saved resource:
    # the loss's own probabilities immediately, the pooling winners the
    # intermediate node owns when the abandoned chain is released.
    assert all(not _resource_closed(r) for r in resources)
    own = loss._graph_resources
    loss.close()
    assert all(_resource_closed(r) for r in own)
    logits.close()
    del loss, logits, own
    _collect()
    assert all(_resource_closed(r) for r in resources)
    assert math.isfinite(
        _train_step(model, NativeCrossEntropyLoss(), optimizer, x, targets)
    )
    _close_run(model, optimizer, x)


@pytest.mark.parametrize("call", [1, 5])
def test_boundary_e_a_checkpoint_load_failure_rolls_the_model_back(
    tmp_path, live_storages, call
):
    """Boundary E. A commit failure while loading a real integrated
    checkpoint — twelve parameters, four running buffers, and the
    NativeAdam state — restores every value, identity, and version and
    leaks no staged storage. (The loader's documented model/optimizer
    two-commit window is a separate, honestly-stated boundary; this
    failure lands inside the model commit.)"""
    model, optimizer, x, targets = _settled_model()
    path = os.path.join(str(tmp_path), "integrated.npz")
    save_native_checkpoint(path, model, optimizer=optimizer,
                           metadata={"phase": "F8"})
    _train_step(model, NativeCrossEntropyLoss(), optimizer, x, targets)
    before = _fingerprint(model, optimizer)
    training_flag = model.training
    baseline = len(live_storages)

    real_install = _native_state._install_core
    calls = {"n": 0}

    def failing_install(planned, new_core):
        calls["n"] += 1
        if calls["n"] == call:
            raise _Boom("injected checkpoint commit failure")
        return real_install(planned, new_core)

    patcher = _injected(_native_state, "_install_core", failing_install)
    with pytest.raises(_Boom, match="injected checkpoint commit failure"):
        load_native_checkpoint(path, model, optimizer=optimizer)
    patcher.undo()

    # Immediate, without gc: every value, identity, and version restored.
    _assert_parameters_and_optimizer_untouched(model, optimizer, before)
    for name, buffer in model.named_buffers():
        assert np.array_equal(buffer.to_numpy(), before["buffers"][name]), name
        assert buffer.closed is False, name
    assert model.training is training_flag
    assert len(live_storages) <= baseline
    # A valid load and a further training step both succeed afterwards.
    metadata = load_native_checkpoint(path, model, optimizer=optimizer)
    assert metadata == {"phase": "F8"}
    assert math.isfinite(
        _train_step(model, NativeCrossEntropyLoss(), optimizer, x, targets)
    )
    _close_run(model, optimizer, x)


# ==========================================================================
# 12. Native error-state recovery
# ==========================================================================

def test_handled_failures_leave_no_stale_native_error_state():
    """Python validation errors and native failures alike must leave the
    thread-local status clean, so the next normalized operation is not
    poisoned by an earlier handled error."""
    model = NativePhaseFClassifier()
    x, targets = _inputs()
    bad_shape = NativeTensor.from_array(np.zeros((2, 3, 4, 4)))
    for call in (lambda: model.batch_norm2d(bad_shape),
                 lambda: model.batch_norm1d(bad_shape),
                 lambda: model.layer_norm(bad_shape),
                 lambda: model(bad_shape),
                 lambda: NativeCrossEntropyLoss()(bad_shape, [0, 1])):
        with pytest.raises((TypeError, ValueError, RuntimeError)):
            call()
        assert cpp._require_library().tf_last_error_code() == cpp.TF_OK
    bad_shape.close()

    model.train()
    logits = model(x)
    loss = NativeCrossEntropyLoss()(logits, targets)
    assert math.isfinite(float(loss.to_numpy()))
    loss.backward()
    assert cpp._require_library().tf_last_error_code() == cpp.TF_OK
    for parameter in model.parameters():
        parameter.grad.close()
    _close_run(model, None, loss, logits, x)


@pytest.mark.skipif(
    not (cpp.is_available() and cpp.fault_injection_available()),
    reason="fault injection not compiled into the backend",
)
def test_an_injected_native_allocation_failure_recovers_cleanly(live_storages):
    model, optimizer, x, targets = _settled_model()
    before = _fingerprint(model, optimizer)
    _collect()
    baseline = len(live_storages)

    cpp._arm_alloc_failure(1)
    with pytest.raises(MemoryError):
        _train_step(model, NativeCrossEntropyLoss(), optimizer, x, targets)
    cpp._arm_alloc_failure(0)

    _assert_parameters_and_optimizer_untouched(model, optimizer, before)
    optimizer.zero_grad()
    _collect()
    assert len(live_storages) <= baseline
    assert cpp._require_library().tf_last_error_code() == cpp.TF_OK
    assert math.isfinite(
        _train_step(model, NativeCrossEntropyLoss(), optimizer, x, targets)
    )
    assert cpp._require_library().tf_last_error_code() == cpp.TF_OK
    _close_run(model, optimizer, x)


# ==========================================================================
# 13. The NumPy and conversion boundary of the integrated step
# ==========================================================================

_NUMERICAL_NUMPY = (
    "max", "amax", "argmax", "exp", "log", "logaddexp", "sum", "divide",
    "true_divide", "add", "subtract", "multiply", "matmul", "mean", "var",
    "std", "negative", "power", "square", "copyto", "sqrt", "reciprocal",
    "take", "take_along_axis", "put", "put_along_axis", "where", "choose",
    "maximum",
)
_DATA_NUMPY = ("empty", "frombuffer")


def test_one_complete_integrated_step_reaches_no_numpy(monkeypatch):
    """The whole integrated iteration — convolution, the NCHW BatchNorm
    update, ReLU, pooling, flatten, linear, the 2-D BatchNorm update,
    ReLU, LayerNorm, the head, the fused loss, backward, the NativeAdam
    step, and zero_grad — runs with every NumPy numerical routine and
    every tensor-data conversion route armed. ``native_accuracy``
    converts on purpose and stays outside."""
    model, optimizer, x, targets = _settled_model()
    running_before = model.batch_norm2d.running_mean.to_numpy().copy()

    def tripwire(*args, **kwargs):
        raise AssertionError("the integrated native step reached NumPy")

    for name in _NUMERICAL_NUMPY + _DATA_NUMPY:
        monkeypatch.setattr(np, name, tripwire)
    monkeypatch.setattr(cpp.NativeTensorCore, "to_numpy", tripwire)
    monkeypatch.setattr(cpp.NativeTensorCore, "from_array",
                        staticmethod(tripwire))
    monkeypatch.setattr(cpp.NativeTensorView, "to_numpy", tripwire)
    monkeypatch.setattr(cpp.NativeStorage, "from_array", staticmethod(tripwire))
    monkeypatch.setattr(cpp.NativeStorage, "to_numpy", tripwire)
    monkeypatch.setattr(cpp.NativeStorage, "copy_from", tripwire)
    monkeypatch.setattr(NativeTensor, "to_numpy", tripwire)

    optimizer.zero_grad()
    model.train()
    logits = model(x)
    loss = NativeCrossEntropyLoss()(logits, targets)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    with pytest.raises(AssertionError, match="reached NumPy"):
        loss.to_numpy()
    monkeypatch.undo()

    assert math.isfinite(float(loss.to_numpy()))
    assert logits.shape == (len(targets), NUM_CLASSES)
    # The whole step really happened.
    assert not np.array_equal(model.batch_norm2d.running_mean.to_numpy(),
                              running_before)
    assert list(optimizer.step_counts) == [2] * len(PARAMETER_NAMES)
    assert all(p.grad is None for p in model.parameters())
    # The reporting metric is deliberately outside the tripwire.
    accuracy = native_accuracy(logits, targets)
    assert type(accuracy) is float and 0.0 <= accuracy <= 1.0
    _close_run(model, optimizer, loss, logits, x)


# ==========================================================================
# 14. Ownership and live-storage across success and failure cycles
# ==========================================================================

def test_repeated_integrated_training_and_reporting_grow_no_storage(
    live_storages
):
    model, optimizer, x, targets = _settled_model()
    loss_fn = NativeCrossEntropyLoss()
    for _ in range(2):
        _train_step(model, loss_fn, optimizer, x, targets)
        _evaluate(model, x, targets)
    _collect()
    baseline = len(live_storages)
    assert baseline > 0                    # persistent state is honestly live
    for _ in range(4):
        _train_step(model, loss_fn, optimizer, x, targets)
        result = _evaluate(model, x, targets)
        assert 0.0 <= result["accuracy"] <= 1.0
        _collect()
        assert len(live_storages) == baseline
    _close_run(model, optimizer, x)
    _collect()
    assert len(live_storages) < baseline


def test_an_exact_resume_cycle_grows_no_storage(tmp_path, live_storages):
    def cycle():
        model = NativePhaseFClassifier()
        optimizer = NativeAdam(model.parameters(), lr=LR)
        x, targets = _inputs()
        _run_schedule(2, model, optimizer, x, targets)
        path = os.path.join(str(tmp_path), "cycle.npz")
        save_native_checkpoint(path, model, optimizer=optimizer)
        fresh = NativePhaseFClassifier()
        fresh_optimizer = NativeAdam(fresh.parameters(), lr=LR)
        load_native_checkpoint(path, fresh, optimizer=fresh_optimizer)
        _run_schedule(1, fresh, fresh_optimizer, x, targets)
        _close_run(model, optimizer)
        _close_run(fresh, fresh_optimizer, x)

    cycle()
    _collect()
    baseline = len(live_storages)
    for _ in range(2):
        cycle()
        _collect()
        assert len(live_storages) == baseline


def test_failure_cycles_leave_no_storage_behind(live_storages):
    """The failure paths that promise *immediate* deterministic cleanup
    are checked without gc; the surrounding cycle is compared at an
    established wrapper-cycle boundary."""
    model, optimizer, x, targets = _settled_model()
    _collect()
    baseline = len(live_storages)

    for _ in range(2):
        # A BatchNorm transaction failure inside the forward.
        real_install = _native_state._install_core
        calls = {"n": 0}

        def failing_install(planned, new_core):
            calls["n"] += 1
            if calls["n"] == 4:
                raise _Boom("injected commit failure")
            return real_install(planned, new_core)

        patcher = _injected(_native_state, "_install_core", failing_install)
        model.train()
        with pytest.raises(_Boom):
            model(x)
        patcher.undo()

        # A stale-parameter backward, released explicitly.
        logits = model(x)
        loss = NativeCrossEntropyLoss()(logits, targets)
        replacement = NativeTensor.from_array(
            model.layer_norm.weight.to_numpy() + 1.0
        )
        try:
            model.layer_norm.weight.copy_value_(replacement)
        finally:
            replacement.close()
        with pytest.raises(RuntimeError, match="stale parameter value"):
            loss.backward()
        loss.close()
        logits.close()

        # An abandoned eval graph.
        model.eval()
        abandoned_logits = model(x)
        abandoned_loss = NativeCrossEntropyLoss()(abandoned_logits, targets)
        abandoned_loss.close()
        abandoned_logits.close()
        del abandoned_loss, abandoned_logits, loss, logits
        _collect()
        assert len(live_storages) == baseline
    _close_run(model, optimizer, x)


def test_explicit_cleanup_closes_every_parameter_buffer_and_alias():
    """There is no ``NativeModule.close()``: the owner closes both
    traversals, identity-deduplicated, and the whole thing is
    idempotent."""
    model = NativePhaseFClassifier()
    optimizer = NativeAdam(model.parameters(), lr=LR)
    x, targets = _inputs()
    parameters = list(model.parameters())
    buffers = list(model.buffers())
    assert len(parameters) == len(PARAMETER_NAMES)
    assert len(buffers) == len(BUFFER_NAMES)
    _train_step(model, NativeCrossEntropyLoss(), optimizer, x, targets)
    for _ in range(3):
        _close_run(model, optimizer, x)
    assert all(p.closed for p in parameters)
    assert all(b.closed for b in buffers)
    assert optimizer.closed and x.closed
    # Post-close operations reject deterministically rather than crashing.
    with pytest.raises(RuntimeError):
        model(NativeTensor.from_array(np.zeros((1, 1, 6, 6))))
