"""Phase-G integration guardrails: the invariants that span the whole
native stack once Dropout and explicit random state are in it (Advanced
C++ Phase G, milestone G9).

The per-milestone suites already cover ``NativeGenerator`` (G1), the
stateless Dropout Core (G2), the differentiable operation (G3), the
module (G4), checkpoint version 2 (G5), the RNG/graph/ownership/checkpoint
hardening (G6), the deterministic stochastic training resume (G7), and the
characterization benchmark (G8) in depth. This file deliberately tests only
what those cannot: the **interactions** —

- one model carrying convolution, NCHW batch normalization, pooling, two
  Dropout layers over one shared generator, flatten, a linear stack, 1-D
  batch normalization, LayerNorm, and the fused classification loss,
  trained by ``NativeAdam`` and resumed exactly from one version-2
  checkpoint;
- **four** saved-resource families — Dropout multiplier masks, MaxPool2d
  winners, BatchNorm eval snapshots, and cross-entropy probabilities —
  alive in one graph and released exactly once;
- generator topology as a property of a real module tree: shared versus
  independent, canonical entries versus registered alias paths, identity
  rather than equal state, and a resume into a *different* topology
  rejected before anything changes;
- the buffer rule, the generator rule, and the parameter-version rule
  meeting in the same graph, so each is attributed to its right cause;
- the whole-checkpoint transaction over a realistic state volume, and its
  serializability against concurrent participants;
- non-contiguous NCHW and view inputs through the whole stack;
- and the export, inventory, and capability boundary of the phase as it
  stands — with ``"dropout"`` still unsupported, because that is G10's
  decision, not this file's.

Nothing here adds numerical behavior, registers anything, or depends on
one implementation being faster than another. Every assertion is a
property the architecture promises.

Selector: python -m pytest -q -k native_phase_g and not hardening
"""

import gc
import json
import math
import os
import threading
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
    NativeDropout,
    NativeFlatten,
    NativeGenerator,
    NativeLayerNorm,
    NativeLinear,
    NativeMaxPool2d,
    NativeModule,
    NativeParameter,
    NativeReLU,
    NativeSGD,
    NativeSequential,
    NativeTensor,
    load_native_checkpoint,
    native_accuracy,
    save_native_checkpoint,
)
from tensorforge.experimental import (
    _native_checkpoint_transaction as transaction,
    _native_state_lock as state_lock,
    native_checkpoint,
    native_generator as generator_module,
)

pytestmark = pytest.mark.skipif(
    not cpp.is_available(),
    reason="experimental C++ backend not built; " + cpp.build_instructions(),
)

REPO_ROOT = Path(__file__).resolve().parent.parent

# The integrated architecture. A 6x6 image through conv(3) is 4x4; through
# pool(2) it is 2x2; four channels give 4 * 2 * 2 = 16 flattened features.
IN_CHANNELS = 1
CONV_CHANNELS = 4
KERNEL_SIZE = 3
POOLED_SIDE = 2
FLAT_FEATURES = CONV_CHANNELS * POOLED_SIDE * POOLED_SIDE
HIDDEN_FEATURES = 8
NUM_CLASSES = 3
MOMENTUM = 0.1
DROPOUT_P = 0.5

CONV_SEED, HIDDEN_SEED, HEAD_SEED = 0, 1, 2
SHARED_GENERATOR_SEED = 0x0123456789ABCDEF
SECOND_GENERATOR_SEED = 20240707
# A deliberately different seed for every fresh restore target, so a load
# that failed to restore the stream could not possibly go unnoticed.
FRESH_GENERATOR_SEED = 999983

# The deterministic integrated schedule. Fixed, never sampled.
SAMPLES = 12
BATCH_SIZE = 6
NUM_BATCHES = SAMPLES // BATCH_SIZE
TOTAL_STEPS = 8
# Odd, so the resume lands mid-cycle in the batch schedule: a run that
# restarted the schedule at batch 0 would diverge.
SPLIT_STEP = 3
LR = 0.05

# Two Dropout modules, so one training forward consumes exactly two calls.
CALLS_PER_TRAINING_FORWARD = 2

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
GENERATOR_PATHS = ("drop2d.generator", "drop1d.generator")

JOIN_TIMEOUT = 20.0


class _Boom(RuntimeError):
    """A distinctive injected failure, so a test can never mistake an
    unrelated error for the one it injected."""


# ==========================================================================
# Fixtures and helpers
# ==========================================================================

@pytest.fixture(autouse=True)
def _disarm_after_each():
    yield
    if cpp.is_available():
        cpp._arm_alloc_failure(0)
        cpp._require_library().tf_clear_error()


@pytest.fixture
def live_storages(monkeypatch):
    """The ids of every NativeStorage currently open — the project's
    deterministic instrumentation for native-allocation lifetime (the
    Phase-C/D/E/F precedent). There is no public counter, and G9 does not
    add one."""
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


def settled(live_storages):
    """The live-storage count after a collection.

    Both the baseline and the comparison go through this. The autograd
    graph holds its parents through backward closures, which is a
    reference *cycle*, so a wrapper whose last explicit reference is gone
    is freed by the collector rather than by refcounting. Collection
    settles the count; it is never the proof that anything was released —
    every test here closes what it owns explicitly first."""
    gc.collect()
    return len(live_storages)


class patched:
    """``setattr`` with a guaranteed restore, as a context manager.

    Used instead of pytest's ``monkeypatch`` wherever a test also uses the
    ``live_storages`` fixture: that fixture patches ``NativeStorage``
    through the same ``monkeypatch`` instance, so a mid-test undo would
    silently stop the tracking and make every later baseline assertion
    meaningless."""

    def __init__(self, target, attribute, value):
        self.target = target
        self.attribute = attribute
        self.value = value

    def __enter__(self):
        self.original = getattr(self.target, self.attribute)
        setattr(self.target, self.attribute, self.value)
        return self.value

    def __exit__(self, *exc_info):
        setattr(self.target, self.attribute, self.original)
        return False


def raiser(error):
    """A callable that raises ``error``, for the failure-injection seams."""
    def boom(*args, **kwargs):
        raise error
    return boom


class Interleaver:
    """A two-thread rendezvous: one thread blocks at a seam until the
    other has provably reached the point being tested. Events and bounded
    waits only — never a sleep, so a regression fails instead of passing
    slowly."""

    def __init__(self):
        self.arrived = threading.Event()
        self.release = threading.Event()

    def block(self):
        self.arrived.set()
        assert self.release.wait(JOIN_TIMEOUT), "seam was never released"

    def wait_for_arrival(self):
        assert self.arrived.wait(JOIN_TIMEOUT), "seam was never reached"

    def let_go(self):
        self.release.set()


def run_threads(targets):
    """Start every target, join each with a bounded timeout, and re-raise
    the first exception any of them raised."""
    failures = []

    def wrap(target):
        def runner():
            try:
                target()
            except BaseException as error:      # noqa: BLE001 - reported below
                failures.append(error)
        return runner

    threads = [threading.Thread(target=wrap(t), daemon=True) for t in targets]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(JOIN_TIMEOUT)
        assert not thread.is_alive(), "a thread did not finish — likely a deadlock"
    if failures:
        raise failures[0]


# ==========================================================================
# The canonical integrated model, its fixed dataset, and its schedule
# ==========================================================================

class NativePhaseGClassifier(NativeModule):
    """**Test-only.** The design's §17 model: the Phase-D convolutional
    stack, both normalization families, LayerNorm, **two** Dropout layers,
    and a linear head whose raw logits go to the fused Phase-E loss.

    ``shared=True`` (the default) registers **one** ``NativeGenerator``
    under both Dropout modules, so the two layers draw from one
    interleaved stream; ``shared=False`` gives each its own. Named
    children give readable canonical state keys and generator paths. Not a
    production class, not exported, not a public API."""

    def __init__(self, p=DROPOUT_P, shared=True,
                 generator_seed=SHARED_GENERATOR_SEED,
                 second_seed=SECOND_GENERATOR_SEED,
                 conv_seed=CONV_SEED, hidden_seed=HIDDEN_SEED,
                 head_seed=HEAD_SEED, momentum=MOMENTUM):
        super().__init__()
        first = NativeGenerator(generator_seed)
        second = first if shared else NativeGenerator(second_seed)
        self.conv = NativeConv2d(IN_CHANNELS, CONV_CHANNELS, KERNEL_SIZE,
                                 seed=conv_seed)
        self.batch_norm2d = NativeBatchNorm2d(CONV_CHANNELS, momentum=momentum)
        self.relu2d = NativeReLU()
        self.pool = NativeMaxPool2d(2)
        self.drop2d = NativeDropout(p, generator=first)
        self.flatten = NativeFlatten()
        self.hidden = NativeLinear(FLAT_FEATURES, HIDDEN_FEATURES,
                                   seed=hidden_seed)
        self.batch_norm1d = NativeBatchNorm1d(HIDDEN_FEATURES,
                                              momentum=momentum)
        self.relu1d = NativeReLU()
        self.layer_norm = NativeLayerNorm(HIDDEN_FEATURES)
        self.drop1d = NativeDropout(p, generator=second)
        self.head = NativeLinear(HIDDEN_FEATURES, NUM_CLASSES, seed=head_seed)

    def forward(self, images):
        hidden = self.conv(images)
        hidden = self.batch_norm2d(hidden)
        hidden = self.relu2d(hidden)
        hidden = self.pool(hidden)
        hidden = self.drop2d(hidden)
        hidden = self.flatten(hidden)
        hidden = self.hidden(hidden)
        hidden = self.batch_norm1d(hidden)
        hidden = self.relu1d(hidden)
        hidden = self.layer_norm(hidden)
        hidden = self.drop1d(hidden)
        return self.head(hidden)


def _dataset():
    """The E8 fixed twelve-image, three-class task: nested float literals
    and strict host integer targets. Nothing is generated, augmented,
    shuffled, downloaded, or randomly sampled, and neither NumPy's global
    RNG nor Python's ``random`` is ever touched."""
    from examples.native_classification_training import build_dataset

    return build_dataset()


def _batches():
    """The fixed two-batch schedule: six images each, in a fixed order.
    The caller owns every returned tensor."""
    images, targets = _dataset()
    batches = []
    for index in range(NUM_BATCHES):
        start = index * BATCH_SIZE
        stop = start + BATCH_SIZE
        batches.append((NativeTensor.from_array(images[start:stop]),
                        list(targets[start:stop])))
    return batches


def batch_index_for_step(step):
    """Which batch a step uses — a pure function of the step, so the whole
    external loop position collapses to one integer."""
    return step % NUM_BATCHES


def _close_batches(batches):
    for tensor, _targets in batches:
        tensor.close()


def _inputs():
    """The whole twelve-image set as one tensor, for the graph and
    evaluation tests that do not need the batch schedule."""
    images, targets = _dataset()
    return NativeTensor.from_array(images), targets


def _close_module(module):
    """There is no ``NativeModule.close()``, so a stateful module's owner
    releases both tensor traversals explicitly. A ``NativeGenerator`` owns
    no native storage and has no ``close()``."""
    for parameter in module.parameters():
        parameter.close()
    for buffer in module.buffers():
        buffer.close()


def _close_run(model=None, optimizer=None, *tensors):
    if optimizer is not None and hasattr(optimizer, "close"):
        optimizer.close()
    if model is not None:
        for parameter in model.parameters():
            if parameter.grad is not None:
                parameter.grad.close()
        _close_module(model)
    for tensor in tensors:
        if tensor is not None and not tensor.closed:
            tensor.close()


def _train_step(model, loss_fn, optimizer, batches, step):
    """One complete integrated iteration at schedule position ``step``,
    returning the pre-update loss."""
    x, targets = batches[batch_index_for_step(step)]
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
    """A no-update reporting pass in evaluation mode, restoring the
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


def _train_logits(model, x):
    """One training-mode forward's logits as plain lists. Consumes two
    generator calls, exactly like any other training forward — callers
    that compare two runs must do it at the same point in both."""
    model.train()
    logits = model(x)
    try:
        return logits.to_numpy().tolist()
    finally:
        logits.close()


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
    """The optimizer state as plain Python values, closing every
    caller-owned snapshot the state dictionary hands back."""
    state = optimizer.state_dict()
    try:
        record = {
            "format_version": state["format_version"],
            "optimizer": state["optimizer"],
            "lr": state["lr"],
        }
        if state["optimizer"] == "NativeAdam":
            record.update({
                "betas": list(state["betas"]),
                "eps": state["eps"],
                "step_counts": list(state["step_counts"]),
                "m": [tensor.to_numpy().tolist() for tensor in state["m"]],
                "v": [tensor.to_numpy().tolist() for tensor in state["v"]],
            })
        return record
    finally:
        for key in ("m", "v"):
            for tensor in state.get(key, ()):
                tensor.close()


def _generator_values(model):
    """Every registered generator's canonical state, plus the topology."""
    return {
        "states": model.generator_state_dict(),
        "canonical": [name for name, _ in model.named_generators()],
        "paths": [path for path, _ in model._named_generator_paths()],
        "aliases": {
            path: name
            for name, _, path in _alias_rows(model)
        },
    }


def _alias_rows(model):
    """``(canonical_name, generator, registered_path)`` for every
    registered path, resolved exactly as the checkpoint's alias map is."""
    canonical = {id(generator): name
                 for name, generator in model.named_generators()}
    return [(canonical[id(generator)], generator, path)
            for path, generator in model._named_generator_paths()]


def _state_keys(model):
    state = model.state_dict()
    try:
        return list(state)
    finally:
        for snapshot in state.values():
            snapshot.close()


# -- the G2 Core oracle -----------------------------------------------------

def core_pair(values, p, seed, call_index):
    """The G2 Core's ``(output, mask)`` as NumPy arrays, everything closed.
    The oracle every layer above must reproduce exactly."""
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


def core_mask(values, p, seed, call_index):
    return core_pair(values, p, seed, call_index)[1]


# -- graph inspection -------------------------------------------------------

def _walk_graph(root):
    """Every autograd node reachable from ``root`` plus every native
    object a node's history owns."""
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

    Deliberately not the node list: retaining the nodes would keep an
    abandoned graph reachable, so the very cleanup these tests check could
    never happen."""
    return _walk_graph(root)[1]


def _resources_by_op(root):
    """``{op_name: [saved resources]}`` over the whole history.

    Classifying by the operation that saved them is the only reliable
    way to tell the families apart: a Dropout mask over the pooled
    activation and MaxPool2d's winner buffer are both cores of the *same*
    shape."""
    families = {}
    for node in _walk_graph(root)[0]:
        if node._graph_resources:
            families.setdefault(node._op, []).extend(node._graph_resources)
    return families


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
    """Give both BatchNorm modules running statistics that are neither the
    fresh zeros/ones nor the batch's own, so an eval forward really
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


def _progress(step):
    """The external loop position as validated checkpoint metadata.
    Checkpoint v2 captures TensorForge-owned state and not the data
    loader, so the schedule position is carried explicitly."""
    return {"training_step": step,
            "next_batch_index": batch_index_for_step(step),
            "lr": LR}


def _read_manifest(path):
    with np.load(path, allow_pickle=False) as archive:
        return json.loads(archive["manifest"].tobytes().decode("utf-8"))


def _archive_names(path):
    with np.load(path, allow_pickle=False) as archive:
        return list(archive.files)


# ==========================================================================
# 1. The integrated module tree and its generator topology
# ==========================================================================

def test_the_integrated_model_registers_every_state_family():
    """Four registration categories in one tree: parameters, persistent
    buffers, child modules, and generators — each in its own traversal,
    and the generator deliberately absent from ``state_dict()``."""
    model = NativePhaseGClassifier()
    assert tuple(name for name, _ in model.named_parameters()) == PARAMETER_NAMES
    assert tuple(name for name, _ in model.named_buffers()) == BUFFER_NAMES
    assert _state_keys(model) == list(PARAMETER_NAMES) + list(BUFFER_NAMES)
    # state_dict() is contractually {name: NativeTensor} — no generator.
    state = model.state_dict()
    try:
        assert all(isinstance(value, NativeTensor) for value in state.values())
        assert not any("generator" in key for key in state)
    finally:
        for snapshot in state.values():
            snapshot.close()
    # The generator lives in its own surface.
    assert tuple(name for name, _ in model.named_generators()) == (
        GENERATOR_PATHS[0],
    )
    assert tuple(path for path, _ in model._named_generator_paths()) == (
        GENERATOR_PATHS
    )
    assert set(model.generator_state_dict()) == {GENERATOR_PATHS[0]}
    # Every child class is a real, registered native module.
    for child in (NativeConv2d, NativeBatchNorm2d, NativeReLU, NativeMaxPool2d,
                  NativeDropout, NativeFlatten, NativeLinear,
                  NativeBatchNorm1d, NativeLayerNorm):
        assert child.__name__ in cpp.NATIVE_MODULES, child.__name__
    _close_run(model)


def test_shared_dropouts_hold_exactly_one_generator_object():
    """Sharing is **identity**: two registered paths, one object, one
    canonical entry, and one state to restore."""
    model = NativePhaseGClassifier(shared=True)
    assert model.drop2d.generator is model.drop1d.generator
    assert len(list(model.generators())) == 1
    canonical = list(model.named_generators())
    assert [name for name, _ in canonical] == [GENERATOR_PATHS[0]]
    rows = _alias_rows(model)
    assert [path for _, _, path in rows] == list(GENERATOR_PATHS)
    assert {name for name, _, _ in rows} == {GENERATOR_PATHS[0]}
    assert len(model.generator_state_dict()) == 1
    _close_run(model)


def test_independent_dropouts_hold_two_generators():
    model = NativePhaseGClassifier(shared=False)
    assert model.drop2d.generator is not model.drop1d.generator
    assert len(list(model.generators())) == 2
    assert [name for name, _ in model.named_generators()] == list(
        GENERATOR_PATHS
    )
    rows = _alias_rows(model)
    assert [(name, path) for name, _, path in rows] == [
        (GENERATOR_PATHS[0], GENERATOR_PATHS[0]),
        (GENERATOR_PATHS[1], GENERATOR_PATHS[1]),
    ]
    _close_run(model)


def test_independent_generators_with_equal_state_stay_two_entries(tmp_path):
    """Identity, never value: two generators built from the same seed and
    advanced to the same counter are still two entries in the model and in
    the archive."""
    model = NativePhaseGClassifier(shared=False,
                                   second_seed=SHARED_GENERATOR_SEED)
    assert model.drop2d.generator.state() == model.drop1d.generator.state()
    assert model.drop2d.generator is not model.drop1d.generator
    assert len(model.generator_state_dict()) == 2
    path = os.path.join(str(tmp_path), "twins.npz")
    save_native_checkpoint(path, model)
    section = _read_manifest(path)["generators"]
    assert list(section["keys"]) == list(GENERATOR_PATHS)
    assert section["aliases"] == {GENERATOR_PATHS[0]: GENERATOR_PATHS[0],
                                  GENERATOR_PATHS[1]: GENERATOR_PATHS[1]}
    assert section["entries"][GENERATOR_PATHS[0]] == (
        section["entries"][GENERATOR_PATHS[1]]
    )
    _close_run(model)


def test_train_eval_propagation_does_not_alter_topology():
    model = NativePhaseGClassifier()
    before = _generator_values(model)
    for _ in range(2):
        model.eval()
        assert not model.drop2d.training and not model.drop1d.training
        assert not model.batch_norm2d.training
        model.train()
        assert model.drop2d.training and model.drop1d.training
    assert _generator_values(model) == before
    assert model.drop2d.generator is model.drop1d.generator
    _close_run(model)


def test_the_shared_stream_consumes_indices_in_forward_order():
    """The two Dropout layers draw from one ordered stream, and the order
    is the **execution** order: the pooled activation gets index ``k`` and
    the normalized hidden activation gets ``k + 1``, matched against the G2
    Core."""
    model = NativePhaseGClassifier()
    x, _targets = _inputs()
    generator = model.drop2d.generator
    seed = generator.seed
    start = generator.calls
    model.train()

    # Compose the stack by hand so both Dropout inputs are observable.
    conv = model.conv(x)
    normed2d = model.batch_norm2d(conv)
    activated = model.relu2d(normed2d)
    pooled = model.pool(activated)
    pooled_values = pooled.to_numpy().copy()
    dropped2d = model.drop2d(pooled)
    assert generator.calls == start + 1
    flat = model.flatten(dropped2d)
    hidden = model.hidden(flat)
    normed1d = model.batch_norm1d(hidden)
    activated1d = model.relu1d(normed1d)
    normalized = model.layer_norm(activated1d)
    normalized_values = normalized.to_numpy().copy()
    dropped1d = model.drop1d(normalized)
    assert generator.calls == start + CALLS_PER_TRAINING_FORWARD

    expected_first, first_mask = core_pair(pooled_values, DROPOUT_P, seed,
                                           start)
    expected_second, second_mask = core_pair(normalized_values, DROPOUT_P,
                                             seed, start + 1)
    assert np.array_equal(dropped2d.to_numpy(), expected_first)
    assert np.array_equal(dropped1d.to_numpy(), expected_second)
    # Two consecutive indices really are two different streams.
    assert not np.array_equal(first_mask.reshape(-1)[:8],
                              second_mask.reshape(-1)[:8])
    _close_run(model, None, dropped1d, normalized, activated1d, normed1d,
               hidden, flat, dropped2d, pooled, activated, normed2d, conv, x)


def test_generator_state_is_not_a_parameter_buffer_or_child():
    model = NativePhaseGClassifier()
    generator = model.drop2d.generator
    assert generator not in list(model.parameters())
    assert generator not in list(model.buffers())
    assert not isinstance(generator, (NativeTensor, NativeParameter))
    assert not hasattr(generator, "close")
    child_names = [name for name, _ in model.named_modules() if name]
    assert child_names[0] == "conv"
    assert not any(name.endswith("generator") for name in child_names)
    _close_run(model)


# ==========================================================================
# 2. Training forward, backward, and the four saved-resource families
# ==========================================================================

def test_one_graph_carries_conv_norms_pooling_dropout_and_the_loss():
    """The interaction no single-module suite covers: convolution, NCHW
    batch normalization, ReLU, pooling with its winners, **two** Dropout
    layers with their masks, flatten, linear, 1-D batch normalization,
    LayerNorm, the head, and the fused loss with its probabilities — one
    forward, one backward, one NativeAdam step."""
    model = NativePhaseGClassifier()
    loss_fn = NativeCrossEntropyLoss()
    optimizer = NativeAdam(model.parameters(), lr=LR)
    x, targets = _inputs()
    generator = model.drop2d.generator

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
    assert not np.allclose(logits.to_numpy().sum(axis=1), 1.0)  # raw logits
    assert generator.calls == CALLS_PER_TRAINING_FORWARD
    loss = loss_fn(logits, targets)
    assert loss.shape == () and loss.numel == 1
    assert math.isfinite(float(loss.to_numpy()))

    families = _resources_by_op(loss)
    assert len(families["dropout"]) == 2, "both Dropout masks must be saved"
    assert len(families["maxpool2d"]) == 1
    assert len(families["cross_entropy"]) == 1
    resources = _saved_resources(loss)
    assert all(not _resource_closed(r) for r in resources)

    loss.backward()
    assert generator.calls == CALLS_PER_TRAINING_FORWARD, (
        "backward consumed a generator call"
    )
    for name, parameter in model.named_parameters():
        assert parameter.grad is not None, name
        assert parameter.grad.shape == parameter.shape, name
        assert np.isfinite(parameter.grad.to_numpy()).all(), name
    for name, buffer in model.named_buffers():
        assert buffer.grad is None, name
        assert not np.array_equal(buffer.to_numpy(), buffers_before[name]), name
    assert loss._graph_freed is True
    assert all(_resource_closed(r) for r in resources)
    assert loss._graph_resources == ()

    optimizer.step()
    for name, parameter in model.named_parameters():
        assert not np.array_equal(parameter.to_numpy(), before[name]), name
        assert parameter.version == versions[name] + 1, name
    assert list(optimizer.step_counts) == [1] * len(PARAMETER_NAMES)
    optimizer.zero_grad()
    assert all(p.grad is None for p in model.parameters())
    assert [id(p) for p in model.parameters()] == parameter_ids
    assert [id(b) for b in model.buffers()] == buffer_ids

    # A second step builds a completely fresh graph.
    second_logits = model(x)
    assert generator.calls == 2 * CALLS_PER_TRAINING_FORWARD
    second_loss = loss_fn(second_logits, targets)
    assert second_loss._graph_freed is False
    assert len(_resources_by_op(second_loss)["dropout"]) == 2
    second_loss.backward()
    assert second_loss._graph_freed is True
    _close_run(model, optimizer, second_loss, second_logits, loss, logits, x)


def test_four_saved_resource_families_coexist_in_one_integrated_graph():
    """All four families in **one** graph through the real model.

    A uniform mode cannot do it: in training the BatchNorm modules use the
    batch's own statistics and save no eval snapshot, and in evaluation
    Dropout is the identity and saves no mask. Mode is per module, so the
    honest configuration is the mixed one — normalization evaluating
    against its stored running statistics while Dropout still draws."""
    model = NativePhaseGClassifier()
    _nontrivial_running_state(model)
    model.eval()
    model.drop2d.train()
    model.drop1d.train()
    x, targets = _inputs()
    generator = model.drop2d.generator

    logits = model(x)
    loss = NativeCrossEntropyLoss()(logits, targets)
    families = _resources_by_op(loss)
    assert len(families["dropout"]) == 2
    assert len(families["maxpool2d"]) == 1
    assert len(families["cross_entropy"]) == 1
    # BatchNorm's eval snapshots are NativeTensors, and they are
    # independent objects rather than the registered buffers.
    snapshots = [resource for resources in families.values()
                 for resource in resources
                 if isinstance(resource, NativeTensor)]
    assert len(snapshots) >= 4, "both BatchNorm modules must snapshot"
    buffer_ids = {id(b) for b in model.buffers()}
    buffer_storage_ids = {id(b._core.storage) for b in model.buffers()}
    assert not ({id(s) for s in snapshots} & buffer_ids)
    assert not ({id(s._core.storage) for s in snapshots} & buffer_storage_ids)
    resources = _saved_resources(loss)
    assert len(resources) >= 8

    buffers_before = {name: buffer.to_numpy().copy()
                      for name, buffer in model.named_buffers()}
    loss.backward()
    assert generator.calls == CALLS_PER_TRAINING_FORWARD
    assert all(_resource_closed(r) for r in resources), (
        "a saved resource survived the one-shot backward"
    )
    # Evaluating normalization advanced nothing.
    for name, buffer in model.named_buffers():
        assert np.array_equal(buffer.to_numpy(), buffers_before[name]), name
    # A second backward raises rather than double-closing anything.
    with pytest.raises(RuntimeError):
        loss.backward()
    _close_run(model, None, loss, logits, x)


def test_the_phase_e_versioning_archetypes_meet_dropout_masks():
    """The three Phase-E archetypes in one graph with a Dropout mask:
    saved-output ``exp`` (backward reads its own result), live-reread
    ``log`` (backward rereads a live operand), and cross-entropy's saved
    probabilities. Mutating a directly versioned parameter afterwards
    stales only the archetype that rereads."""
    generator = NativeGenerator(SHARED_GENERATOR_SEED)
    weight = NativeParameter(NativeTensor.from_array(
        np.array([[1.5, 2.0], [0.5, 3.0]])
    ))
    x = NativeTensor.from_array(np.array([[0.25, 0.5], [0.75, 1.0]]),
                                requires_grad=True)
    dropped = x.dropout(DROPOUT_P, generator=generator)
    exponent = dropped.exp()
    logarithm = weight.log()
    logits = NativeTensor.from_array(np.array([[0.5, -0.2, 1.4]]),
                                     requires_grad=True)
    entropy = logits.cross_entropy([2], reduction="mean")
    total = exponent.sum().add(logarithm.sum()).add(entropy)

    families = _resources_by_op(total)
    assert len(families["dropout"]) == 1
    assert len(families["cross_entropy"]) == 1
    # `log` records a version expectation; `dropout` records none.
    dropout_node = next(node for node in _walk_graph(total)[0]
                        if node._op == "dropout")
    assert dropout_node._expected_versions == ()
    assert any(node._expected_versions for node in _walk_graph(total)[0])

    # Mutating the live operand stales the graph through the parameter
    # rule — a `log` effect, never a Dropout one.
    replacement = NativeTensor.from_array(np.array([[2.5, 1.0], [1.5, 2.0]]))
    weight.copy_value_(replacement)
    with pytest.raises(RuntimeError, match="stale|version"):
        total.backward()
    assert generator.calls == 1, "a stale backward consumed a generator call"
    _close_run(None, None, total, entropy, logits, logarithm, exponent,
               dropped, x, replacement, weight)


# ==========================================================================
# 3. NativeAdam integration
# ==========================================================================

def _uninterrupted_run(total_steps=TOTAL_STEPS, shared=True):
    """The reference run: a fresh model, optimizer, and generator set
    trained on the fixed schedule, reporting everything a resume must
    reproduce."""
    model = NativePhaseGClassifier(shared=shared)
    optimizer = NativeAdam(model.parameters(), lr=LR)
    loss_fn = NativeCrossEntropyLoss()
    batches = _batches()
    x, targets = _inputs()
    try:
        losses = [_train_step(model, loss_fn, optimizer, batches, step)
                  for step in range(total_steps)]
        return {
            "losses": losses,
            "final_train_logits": _train_logits(model, x),
            "final_eval": _evaluate(model, x, targets),
            "values": _values(model),
            "optimizer": _optimizer_values(optimizer),
            "generators": _generator_values(model),
        }
    finally:
        _close_run(model, optimizer, x)
        _close_batches(batches)


def test_the_integrated_stack_trains_deterministically():
    """Two independently constructed runs of the fixed schedule are
    bit-identical, and the run makes real progress on the fixed task."""
    first = _uninterrupted_run()
    second = _uninterrupted_run()
    assert first["losses"] == second["losses"]
    assert first["values"] == second["values"]
    assert first["optimizer"] == second["optimizer"]
    assert first["generators"] == second["generators"]
    assert first["final_eval"] == second["final_eval"]
    assert first["final_train_logits"] == second["final_train_logits"]

    losses = first["losses"]
    assert len(losses) == TOTAL_STEPS
    assert all(math.isfinite(value) for value in losses)
    assert losses[-1] < losses[0]
    # Exactly two calls per training step, plus the two the final
    # training-mode logits capture consumed.
    states = first["generators"]["states"]
    assert states[GENERATOR_PATHS[0]]["calls"] == (
        (TOTAL_STEPS + 1) * CALLS_PER_TRAINING_FORWARD
    )
    assert states[GENERATOR_PATHS[0]]["seed"] == SHARED_GENERATOR_SEED
    # Both BatchNorm pairs left their fresh zeros/ones behind.
    buffers = first["values"]["buffers"]
    assert buffers["batch_norm2d.running_mean"] != [0.0] * CONV_CHANNELS
    assert buffers["batch_norm1d.running_var"] != [1.0] * HIDDEN_FEATURES


def test_adam_advances_every_state_family_together():
    """Several complete steps: parameters move, Adam's moments and step
    counts advance for every parameter, both BatchNorm pairs update, the
    shared stream advances by exactly two per step, and no graph is
    retained."""
    model = NativePhaseGClassifier()
    optimizer = NativeAdam(model.parameters(), lr=LR)
    loss_fn = NativeCrossEntropyLoss()
    batches = _batches()
    generator = model.drop2d.generator
    before = _values(model)

    for step in range(3):
        _train_step(model, loss_fn, optimizer, batches, step)
        assert generator.calls == (step + 1) * CALLS_PER_TRAINING_FORWARD
        assert list(optimizer.step_counts) == [step + 1] * len(PARAMETER_NAMES)
        assert all(p.grad is None for p in model.parameters())

    after = _values(model)
    for name in PARAMETER_NAMES:
        assert after["parameters"][name] != before["parameters"][name], name
    for name in BUFFER_NAMES:
        assert after["buffers"][name] != before["buffers"][name], name
    state = optimizer.state_dict()
    try:
        assert len(state["m"]) == len(state["v"]) == len(PARAMETER_NAMES)
        for moment in state["m"] + state["v"]:
            assert np.isfinite(moment.to_numpy()).all()
    finally:
        for key in ("m", "v"):
            for tensor in state[key]:
                tensor.close()
    assert not generator._has_active_reservation()
    _close_run(model, optimizer)
    _close_batches(batches)


def test_evaluation_between_training_steps_changes_no_optimizer_or_rng_state():
    """An evaluation pass in the middle of training is inert on every axis
    that matters: no generator call, no optimizer state change, no running
    statistic change — and training then continues from the next
    unconsumed index."""
    model = NativePhaseGClassifier()
    optimizer = NativeAdam(model.parameters(), lr=LR)
    loss_fn = NativeCrossEntropyLoss()
    batches = _batches()
    x, targets = _inputs()

    _train_step(model, loss_fn, optimizer, batches, 0)
    generator = model.drop2d.generator
    calls_before = generator.calls
    optimizer_before = _optimizer_values(optimizer)
    buffers_before = {name: buffer.to_numpy().copy()
                      for name, buffer in model.named_buffers()}
    parameters_before = _values(model)["parameters"]

    first = _evaluate(model, x, targets)
    second = _evaluate(model, x, targets)
    assert first == second, "evaluation is not reproducible"
    assert generator.calls == calls_before
    assert _optimizer_values(optimizer) == optimizer_before
    assert _values(model)["parameters"] == parameters_before
    for name, buffer in model.named_buffers():
        assert np.array_equal(buffer.to_numpy(), buffers_before[name]), name
    assert model.training is True, "evaluate did not restore the mode"

    # The next training forward takes the next unconsumed indices.
    _train_step(model, loss_fn, optimizer, batches, 1)
    assert generator.calls == calls_before + CALLS_PER_TRAINING_FORWARD
    _close_run(model, optimizer, x)
    _close_batches(batches)


# ==========================================================================
# 4. NativeSGD integration
# ==========================================================================

def test_sgd_trains_the_integrated_model_and_consumes_the_stream_exactly():
    model = NativePhaseGClassifier()
    optimizer = NativeSGD(model.parameters(), lr=LR)
    loss_fn = NativeCrossEntropyLoss()
    batches = _batches()
    generator = model.drop2d.generator
    before = _values(model)["parameters"]

    losses = [_train_step(model, loss_fn, optimizer, batches, step)
              for step in range(3)]
    assert all(math.isfinite(value) for value in losses)
    assert generator.calls == 3 * CALLS_PER_TRAINING_FORWARD
    after = _values(model)["parameters"]
    # `hidden.bias` sits immediately before BatchNorm1d, whose mean
    # subtraction cancels any constant per-feature shift, so its gradient
    # is *mathematically zero* and a plain SGD update moves it by nothing.
    # That is a property of the architecture, not a broken optimizer —
    # Adam's eps division amplifies the same round-off and does move it.
    dead = "hidden.bias"
    for name in PARAMETER_NAMES:
        if name == dead:
            assert after[name] == before[name], name
        else:
            assert after[name] != before[name], name
    model.train()
    logits = model(batches[0][0])
    loss = NativeCrossEntropyLoss()(logits, batches[0][1])
    loss.backward()
    assert np.max(np.abs(model.hidden.bias.grad.to_numpy())) < 1e-12, (
        "the structurally dead bias gradient is not round-off"
    )
    model.hidden.bias.grad.close()
    loss.close()
    logits.close()
    assert not hasattr(optimizer, "close"), (
        "NativeSGD owns no native state and must not grow a close()"
    )
    _close_run(model, None)
    _close_batches(batches)


def test_sgd_and_generator_state_round_trip_through_one_checkpoint(tmp_path):
    model = NativePhaseGClassifier()
    optimizer = NativeSGD(model.parameters(), lr=LR)
    loss_fn = NativeCrossEntropyLoss()
    batches = _batches()
    for step in range(2):
        _train_step(model, loss_fn, optimizer, batches, step)
    expected_values = _values(model)
    expected_optimizer = _optimizer_values(optimizer)
    expected_generators = _generator_values(model)
    path = os.path.join(str(tmp_path), "sgd.npz")
    save_native_checkpoint(path, model, optimizer=optimizer,
                           metadata=_progress(2))

    fresh = NativePhaseGClassifier(generator_seed=FRESH_GENERATOR_SEED)
    fresh_optimizer = NativeSGD(fresh.parameters(), lr=0.9)
    generator_id = id(fresh.drop2d.generator)
    metadata = load_native_checkpoint(path, fresh, optimizer=fresh_optimizer)
    assert metadata == _progress(2)
    assert _values(fresh) == expected_values
    assert _optimizer_values(fresh_optimizer) == expected_optimizer
    assert _generator_values(fresh) == expected_generators
    assert id(fresh.drop2d.generator) == generator_id
    assert fresh.drop2d.generator is fresh.drop1d.generator
    _close_run(model, None)
    _close_run(fresh, None)
    _close_batches(batches)


@pytest.mark.parametrize("failing", ["generators", "optimizer"])
def test_a_validation_failure_cannot_partially_restore_the_other_family(
    tmp_path, failing
):
    """A generator-topology failure must not leave SGD's state loaded, and
    an SGD validation failure must not leave the generators loaded. Both
    are prevalidation failures: nothing is touched."""
    model = NativePhaseGClassifier()
    optimizer = NativeSGD(model.parameters(), lr=LR)
    loss_fn = NativeCrossEntropyLoss()
    batches = _batches()
    _train_step(model, loss_fn, optimizer, batches, 0)
    path = os.path.join(str(tmp_path), "sgd_mismatch.npz")
    save_native_checkpoint(path, model, optimizer=optimizer)

    if failing == "generators":
        # Saved shared, live independent: a topology mismatch.
        target = NativePhaseGClassifier(shared=False,
                                        generator_seed=FRESH_GENERATOR_SEED)
        target_optimizer = NativeSGD(target.parameters(), lr=0.7)
    else:
        # A saved SGD section cannot restore an Adam optimizer.
        target = NativePhaseGClassifier()
        target_optimizer = NativeAdam(target.parameters(), lr=0.7)

    values_before = _values(target)
    generators_before = _generator_values(target)
    optimizer_before = _optimizer_values(target_optimizer)
    with pytest.raises(Exception):
        load_native_checkpoint(path, target, optimizer=target_optimizer)
    assert _values(target) == values_before
    assert _generator_values(target) == generators_before
    assert _optimizer_values(target_optimizer) == optimizer_before

    _close_run(model, None)
    _close_run(target, target_optimizer if failing == "optimizer" else None)
    _close_batches(batches)


# ==========================================================================
# 5. Exact checkpoint resume across the whole integrated stack
# ==========================================================================

def test_the_integrated_stack_resumes_exactly_from_one_checkpoint(tmp_path):
    """The milestone's centre: a model combining convolution, pooling,
    **both** BatchNorm shapes, LayerNorm, two Dropout layers over one
    shared generator, the fused loss, and NativeAdam is interrupted,
    checkpointed at format version 2, reloaded into a completely fresh set
    built with a *different* generator seed, and continued — reproducing
    everything exactly."""
    reference = _uninterrupted_run()

    model = NativePhaseGClassifier()
    optimizer = NativeAdam(model.parameters(), lr=LR)
    loss_fn = NativeCrossEntropyLoss()
    batches = _batches()
    prefix = [_train_step(model, loss_fn, optimizer, batches, step)
              for step in range(SPLIT_STEP)]
    path = os.path.join(str(tmp_path), "phase_g.npz")
    save_native_checkpoint(path, model, optimizer=optimizer,
                           metadata=_progress(SPLIT_STEP))
    # The whole source run is released, so the archive is the only
    # continuation boundary.
    _close_run(model, optimizer)
    del model, optimizer

    fresh = NativePhaseGClassifier(generator_seed=FRESH_GENERATOR_SEED)
    fresh_optimizer = NativeAdam(fresh.parameters(), lr=0.9)
    parameter_ids = [id(p) for p in fresh.parameters()]
    buffer_ids = [id(b) for b in fresh.buffers()]
    generator_ids = [id(g) for g in fresh.generators()]
    fresh.eval()          # the training flag is runtime state, never saved
    metadata = load_native_checkpoint(path, fresh, optimizer=fresh_optimizer)
    assert metadata == _progress(SPLIT_STEP)
    assert fresh.training is False
    assert [id(p) for p in fresh.parameters()] == parameter_ids
    assert [id(b) for b in fresh.buffers()] == buffer_ids
    assert [id(g) for g in fresh.generators()] == generator_ids
    assert fresh.drop2d.generator is fresh.drop1d.generator

    resumed_step = metadata["training_step"]
    assert metadata["next_batch_index"] == batch_index_for_step(resumed_step)
    fresh.train()
    suffix = [_train_step(fresh, loss_fn, fresh_optimizer, batches, step)
              for step in range(resumed_step, TOTAL_STEPS)]
    x, targets = _inputs()
    final_train_logits = _train_logits(fresh, x)
    final_eval = _evaluate(fresh, x, targets)
    values = _values(fresh)
    optimizer_state = _optimizer_values(fresh_optimizer)
    generators = _generator_values(fresh)
    _close_run(fresh, fresh_optimizer, x)
    _close_batches(batches)

    # Exact equality everywhere — never a tolerance.
    assert prefix == reference["losses"][:SPLIT_STEP]
    assert suffix == reference["losses"][SPLIT_STEP:]
    assert prefix + suffix == reference["losses"]
    assert values["parameters"] == reference["values"]["parameters"]
    for name in BUFFER_NAMES:
        assert values["buffers"][name] == reference["values"]["buffers"][name], name
    assert optimizer_state == reference["optimizer"]
    assert generators == reference["generators"]
    assert generators["states"][GENERATOR_PATHS[0]]["seed"] == (
        SHARED_GENERATOR_SEED
    ), "the fresh seed survived the load"
    assert final_train_logits == reference["final_train_logits"]
    assert final_eval == reference["final_eval"]


def test_a_resume_that_restarts_the_batch_schedule_diverges(tmp_path):
    """The negative control that makes the progress metadata load-bearing:
    restoring all four state families but restarting the schedule at 0
    does **not** reproduce the reference run."""
    reference = _uninterrupted_run()
    model = NativePhaseGClassifier()
    optimizer = NativeAdam(model.parameters(), lr=LR)
    loss_fn = NativeCrossEntropyLoss()
    batches = _batches()
    for step in range(SPLIT_STEP):
        _train_step(model, loss_fn, optimizer, batches, step)
    path = os.path.join(str(tmp_path), "restart.npz")
    save_native_checkpoint(path, model, optimizer=optimizer)
    _close_run(model, optimizer)

    fresh = NativePhaseGClassifier(generator_seed=FRESH_GENERATOR_SEED)
    fresh_optimizer = NativeAdam(fresh.parameters(), lr=LR)
    load_native_checkpoint(path, fresh, optimizer=fresh_optimizer)
    wrong = [_train_step(fresh, loss_fn, fresh_optimizer, batches, step)
             for step in range(TOTAL_STEPS - SPLIT_STEP)]
    assert wrong != reference["losses"][SPLIT_STEP:]
    _close_run(fresh, fresh_optimizer)
    _close_batches(batches)


@pytest.mark.parametrize("variant", [
    "saved_shared_live_independent",
    "saved_independent_live_shared",
    "renamed_dropout_path",
    "missing_dropout_module",
    "extra_dropout_module",
])
def test_a_topology_mismatch_fails_before_any_state_family_changes(tmp_path,
                                                                   variant):
    """Every realistic topology difference is rejected in prevalidation,
    with the model, buffers, optimizer, and generators bit-identical
    afterwards."""
    source = NativePhaseGClassifier(shared=variant
                                    != "saved_independent_live_shared")
    optimizer = NativeAdam(source.parameters(), lr=LR)
    loss_fn = NativeCrossEntropyLoss()
    batches = _batches()
    _train_step(source, loss_fn, optimizer, batches, 0)
    path = os.path.join(str(tmp_path), f"{variant}.npz")
    save_native_checkpoint(path, source, optimizer=optimizer)
    _close_run(source, optimizer)
    _close_batches(batches)

    if variant == "saved_shared_live_independent":
        target = NativePhaseGClassifier(shared=False)
    elif variant == "saved_independent_live_shared":
        target = NativePhaseGClassifier(shared=True)
    else:
        target = NativePhaseGClassifier()
        if variant == "renamed_dropout_path":
            generator = target.drop1d.generator
            del target.drop1d
            target.renamed_drop = NativeDropout(DROPOUT_P,
                                                generator=generator)
        elif variant == "missing_dropout_module":
            del target.drop1d
        else:
            target.extra_drop = NativeDropout(DROPOUT_P, seed=4242)

    target_optimizer = NativeAdam(target.parameters(), lr=LR)
    values_before = _values(target)
    generators_before = _generator_values(target)
    optimizer_before = _optimizer_values(target_optimizer)
    versions_before = [p.version for p in target.parameters()]

    with pytest.raises(Exception) as excinfo:
        load_native_checkpoint(path, target, optimizer=target_optimizer)
    assert "generator" in str(excinfo.value).lower()
    assert _values(target) == values_before
    assert _generator_values(target) == generators_before
    assert _optimizer_values(target_optimizer) == optimizer_before
    assert [p.version for p in target.parameters()] == versions_before
    _close_run(target, target_optimizer)


# ==========================================================================
# 6. Evaluation and p == 0 integration
# ==========================================================================

def test_evaluation_propagates_and_consumes_no_generator_call():
    """``eval()`` reaches every nested module: BatchNorm reads its stored
    statistics, every Dropout returns its input object, no mask is
    created, and the stream does not move."""
    model = NativePhaseGClassifier()
    _nontrivial_running_state(model)
    x, targets = _inputs()
    generator = model.drop2d.generator
    generator.load_state({"algorithm": generator.algorithm,
                          "algorithm_version": generator.algorithm_version,
                          "seed": generator.seed, "calls": 5})
    model.eval()
    assert not any(child.training for name, child in model.named_modules()
                   if name)

    first = model(x)
    second = model(x)
    assert np.array_equal(first.to_numpy(), second.to_numpy())
    assert generator.calls == 5
    # Not one Dropout node exists in the graph, so no mask was saved.
    loss = NativeCrossEntropyLoss()(first, targets)
    families = _resources_by_op(loss)
    assert "dropout" not in families
    assert len(families["maxpool2d"]) == 1
    assert len(families["cross_entropy"]) == 1

    # Evaluation still validates its inputs.
    with pytest.raises(TypeError):
        model.drop1d(np.zeros((2, HIDDEN_FEATURES)))
    closed = NativeTensor.from_array(np.zeros((2, HIDDEN_FEATURES)))
    closed.close()
    with pytest.raises(RuntimeError):
        model.drop1d(closed)
    assert generator.calls == 5

    # Returning to training resumes at the exact next unconsumed indices.
    model.train()
    pooled_source = model.pool(model.relu2d(model.batch_norm2d(model.conv(x))))
    pooled_values = pooled_source.to_numpy().copy()
    dropped = model.drop2d(pooled_source)
    expected, _mask = core_pair(pooled_values, DROPOUT_P, generator.seed, 5)
    assert np.array_equal(dropped.to_numpy(), expected)
    assert generator.calls == 6
    _close_run(model, None, dropped, pooled_source, loss, second, first, x)


def test_p_zero_integrates_without_removing_the_registered_generator(tmp_path):
    """At ``p == 0`` every Dropout layer is the identity, consumes no call,
    and saves no mask — while the rest of the graph stays fully
    differentiable and the generator remains registered state."""
    model = NativePhaseGClassifier(p=0.0)
    optimizer = NativeAdam(model.parameters(), lr=LR)
    x, targets = _inputs()
    generator = model.drop2d.generator
    assert model.drop2d.p == 0.0 and model.drop1d.p == 0.0
    assert not hasattr(model.drop2d, "owns_generator")

    model.train()
    pooled = model.pool(model.relu2d(model.batch_norm2d(model.conv(x))))
    assert model.drop2d(pooled) is pooled, "p == 0 is not the identity"
    assert generator.calls == 0

    logits = model(x)
    loss = NativeCrossEntropyLoss()(logits, targets)
    families = _resources_by_op(loss)
    assert "dropout" not in families
    assert len(families["maxpool2d"]) == 1
    assert len(families["cross_entropy"]) == 1
    loss.backward()
    for name, parameter in model.named_parameters():
        assert parameter.grad is not None, name
        assert np.isfinite(parameter.grad.to_numpy()).all(), name
    optimizer.step()
    optimizer.zero_grad()
    assert generator.calls == 0

    # The generator is still registered, saved, and restored.
    assert set(model.generator_state_dict()) == {GENERATOR_PATHS[0]}
    path = os.path.join(str(tmp_path), "p_zero.npz")
    save_native_checkpoint(path, model)
    section = _read_manifest(path)["generators"]
    assert section["entries"][GENERATOR_PATHS[0]]["calls"] == "0"
    fresh = NativePhaseGClassifier(p=0.0, generator_seed=FRESH_GENERATOR_SEED)
    load_native_checkpoint(path, fresh)
    assert fresh.drop2d.generator.seed == SHARED_GENERATOR_SEED
    assert fresh.drop2d.generator.calls == 0

    # Evaluation behaves identically with respect to Dropout.
    model.eval()
    evaluated = model.drop1d(pooled)
    assert evaluated is pooled
    assert generator.calls == 0
    _close_run(model, optimizer, loss, logits, pooled, x)
    _close_run(fresh, None)


# ==========================================================================
# 7. Views and non-contiguous inputs
# ==========================================================================

def test_a_non_contiguous_nchw_input_runs_the_whole_integrated_stack(
    live_storages
):
    """Policy B through every layer at once: a transposed view of an
    equal-sized spatial pair reaches convolution, both normalizations,
    pooling, **both Dropout layers**, and the fused loss, and produces
    exactly the contiguous answer in both modes."""
    images, targets = _dataset()
    values = np.asarray(images, dtype=np.float64)
    transposed = np.transpose(values, (0, 1, 3, 2))   # H == W, a real view
    gc.collect()
    baseline = len(live_storages)

    base = NativeTensor.from_array(values)
    view = base.transpose((0, 1, 3, 2))
    assert view.shape == values.shape and view.contiguous is False
    contiguous = NativeTensor.from_array(transposed)

    for mode in ("train", "eval"):
        strided_model = NativePhaseGClassifier()
        contiguous_model = NativePhaseGClassifier()
        _nontrivial_running_state(strided_model)
        _nontrivial_running_state(contiguous_model)
        strided_model.train(mode == "train")
        contiguous_model.train(mode == "train")

        strided_logits = strided_model(view)
        contiguous_logits = contiguous_model(contiguous)
        assert strided_logits.owns_core and strided_logits.contiguous
        assert np.allclose(strided_logits.to_numpy(),
                           contiguous_logits.to_numpy(), atol=1e-12), mode
        # The same number of draws either way: layout is not a stream event.
        assert (strided_model.drop2d.generator.calls
                == contiguous_model.drop2d.generator.calls)
        strided_loss = NativeCrossEntropyLoss()(strided_logits, targets)
        contiguous_loss = NativeCrossEntropyLoss()(contiguous_logits, targets)
        strided_loss.backward()
        contiguous_loss.backward()
        for (name, a), (_, b) in zip(strided_model.named_parameters(),
                                     contiguous_model.named_parameters()):
            assert np.allclose(a.grad.to_numpy(), b.grad.to_numpy(),
                               atol=1e-12), (mode, name)
        assert np.array_equal(base.to_numpy(), values)
        assert np.array_equal(view.to_numpy(), transposed)
        _close_run(strided_model, None, strided_loss, strided_logits)
        _close_run(contiguous_model, None, contiguous_loss, contiguous_logits)

    view.close()
    assert base.closed is False
    base.close()
    contiguous.close()
    gc.collect()
    assert len(live_storages) == baseline


@pytest.mark.parametrize("layout", ["transposed", "narrowed"])
def test_a_view_through_dropout_follows_logical_indexing(live_storages,
                                                          layout):
    """A Dropout module fed a strided view: the mask follows the
    **logical** row-major order, the gradient follows the logical layout,
    exactly one call is consumed, and the view leaks nothing."""
    gc.collect()
    baseline = len(live_storages)
    logical = np.arange(1.0, 25.0).reshape(4, 6)

    if layout == "transposed":
        base = NativeTensor.from_array(np.ascontiguousarray(logical.T),
                                       requires_grad=True)
        view = base.transpose()
        assert view.contiguous is False
    else:
        padded = np.zeros((4, 10), dtype=np.float64)
        padded[:, 2:8] = logical
        base = NativeTensor.from_array(padded, requires_grad=True)
        view = base.narrow(1, 2, 6)
        assert view.contiguous is False and view._core.offset != 0
    assert np.array_equal(view.to_numpy(), logical)

    module = NativeDropout(DROPOUT_P, seed=SHARED_GENERATOR_SEED)
    generator = module.generator
    dropped = module(view)
    assert generator.calls == 1
    expected_out, expected_mask = core_pair(logical, DROPOUT_P,
                                            generator.seed, 0)
    assert np.array_equal(dropped.to_numpy(), expected_out)

    upstream = NativeTensor.from_array(np.full(logical.shape, 0.5))
    dropped.backward(gradient=upstream)
    assert np.array_equal(base.grad.to_numpy().reshape(-1).sum(),
                          (0.5 * expected_mask).sum())
    assert generator.calls == 1, "backward consumed a call"

    base.grad.close()
    base.zero_grad()
    _close_run(None, None, dropped, upstream, view, base)
    gc.collect()
    assert len(live_storages) == baseline


# ==========================================================================
# 8. Graph-resource ownership across subsystems
# ==========================================================================

def _integrated_graph(model, x, targets):
    """A mixed-mode forward whose graph owns all four families, returning
    ``(loss, logits, resources)``."""
    logits = model(x)
    loss = NativeCrossEntropyLoss()(logits, targets)
    return loss, logits, _saved_resources(loss)


def _mixed_mode(model):
    model.eval()
    model.drop2d.train()
    model.drop1d.train()


@pytest.mark.parametrize("mutation", ["generator_reseed", "generator_load",
                                      "buffer_only_load"])
def test_state_mutation_after_the_forward_cannot_change_the_gradients(
    tmp_path, mutation
):
    """Buffer-only and generator-only mutation leave an earlier graph's
    gradients **exactly** equal to a clean control: Dropout's backward
    reads only its saved mask, and BatchNorm's eval graph holds independent
    snapshots."""
    x, targets = _inputs()

    def gradients(mutate):
        model = NativePhaseGClassifier()
        _nontrivial_running_state(model)
        _mixed_mode(model)
        loss, logits, _resources = _integrated_graph(model, x, targets)
        if mutate:
            generator = model.drop2d.generator
            if mutation == "generator_reseed":
                generator.reseed(4242)
            elif mutation == "generator_load":
                model.load_generator_state_dict({
                    GENERATOR_PATHS[0]: {
                        "algorithm": generator.algorithm,
                        "algorithm_version": generator.algorithm_version,
                        "seed": 777, "calls": 99,
                    }
                })
            else:
                path = os.path.join(str(tmp_path), "buffers.npz")
                holder = _BufferHolder(
                    running_mean=model.batch_norm2d.running_mean,
                    running_var=model.batch_norm2d.running_var,
                )
                save_native_checkpoint(path, holder)
                _load_buffers(model.batch_norm2d,
                              running_mean=[9.0, 9.0, 9.0, 9.0],
                              running_var=[9.0, 9.0, 9.0, 9.0])
                load_native_checkpoint(path, holder)
        loss.backward()
        values = {name: parameter.grad.to_numpy().copy()
                  for name, parameter in model.named_parameters()}
        _close_run(model, None, loss, logits)
        return values

    control = gradients(False)
    mutated = gradients(True)
    for name in PARAMETER_NAMES:
        assert np.array_equal(control[name], mutated[name]), name
    x.close()


class _BufferHolder(NativeModule):
    """**Test-only** parameter-free module registering *existing* buffers
    as persistent aliases, so a checkpoint load can be driven over exactly
    those objects without touching any parameter."""

    def __init__(self, **buffers):
        super().__init__()
        for name, tensor in buffers.items():
            self.register_buffer(name, tensor, persistent=True)


def test_a_full_checkpoint_load_stales_an_earlier_graph_through_parameters(
    tmp_path
):
    """The contrast the design insists on: buffer-only and generator-only
    mutation are invisible to an existing graph, but a **full** load
    replaces parameters and correctly stales it — a parameter contract,
    never a Dropout effect."""
    model = NativePhaseGClassifier()
    _nontrivial_running_state(model)
    _mixed_mode(model)
    x, targets = _inputs()
    path = os.path.join(str(tmp_path), "full.npz")
    save_native_checkpoint(path, model)
    loss, logits, resources = _integrated_graph(model, x, targets)

    load_native_checkpoint(path, model)
    with pytest.raises(RuntimeError, match="stale|version"):
        loss.backward()
    # The failed backward left the graph and its saved resources intact.
    assert loss._graph_freed is False
    assert all(not _resource_closed(r) for r in resources)
    _close_run(model, None, loss, logits, x)


def test_a_checkpoint_save_between_forward_and_backward_changes_nothing(
    tmp_path
):
    """A save is a read: it takes the shared guard, snapshots, and leaves
    the live graph, its masks, and the resulting gradients untouched."""
    x, targets = _inputs()

    def gradients(save):
        model = NativePhaseGClassifier()
        optimizer = NativeAdam(model.parameters(), lr=LR)
        model.train()
        loss, logits, resources = _integrated_graph(model, x, targets)
        if save:
            path = os.path.join(str(tmp_path), "midgraph.npz")
            save_native_checkpoint(path, model, optimizer=optimizer,
                                   metadata=_progress(0))
            assert all(not _resource_closed(r) for r in resources)
        loss.backward()
        values = {name: parameter.grad.to_numpy().copy()
                  for name, parameter in model.named_parameters()}
        calls = model.drop2d.generator.calls
        _close_run(model, optimizer, loss, logits)
        return values, calls

    control, control_calls = gradients(False)
    saved, saved_calls = gradients(True)
    assert control_calls == saved_calls == CALLS_PER_TRAINING_FORWARD
    for name in PARAMETER_NAMES:
        assert np.array_equal(control[name], saved[name]), name
    x.close()


def test_a_retained_graph_keeps_its_resources_and_a_failed_backward_too():
    """``retain_graph=True`` keeps every family for a second pass, and a
    backward that fails part-way leaves them retryable."""
    model = NativePhaseGClassifier()
    _nontrivial_running_state(model)
    _mixed_mode(model)
    x, targets = _inputs()
    loss, logits, resources = _integrated_graph(model, x, targets)

    loss.backward(retain_graph=True)
    assert all(not _resource_closed(r) for r in resources)
    first = {name: parameter.grad.to_numpy().copy()
             for name, parameter in model.named_parameters()}
    for parameter in model.parameters():
        parameter.grad.close()
        parameter.zero_grad()

    # A failing native multiply makes the second pass raise; the resources
    # must survive for the retry.
    calls = {"n": 0}
    real_multiply = cpp.NativeTensorCore.multiply

    def flaky(self, other):
        calls["n"] += 1
        if calls["n"] == 2:
            raise _Boom("injected mid-backward")
        return real_multiply(self, other)

    with patched(cpp.NativeTensorCore, "multiply", flaky):
        with pytest.raises(_Boom):
            loss.backward(retain_graph=True)
    assert all(not _resource_closed(r) for r in resources)

    loss.backward()          # the retry succeeds and releases everything
    second = {name: parameter.grad.to_numpy().copy()
              for name, parameter in model.named_parameters()}
    for name in PARAMETER_NAMES:
        assert np.array_equal(first[name], second[name]), name
    assert all(_resource_closed(r) for r in resources)
    _close_run(model, None, loss, logits, x)


def test_an_abandoned_integrated_graph_releases_every_resource_once(
    live_storages
):
    """No backward at all: the closed node's own saved state goes
    immediately, the rest goes with the abandoned chain, nothing is
    released twice, and native storage returns exactly to baseline."""
    model = NativePhaseGClassifier()
    _nontrivial_running_state(model)
    _mixed_mode(model)
    x, targets = _inputs()
    gc.collect()
    baseline = len(live_storages)

    loss, logits, resources = _integrated_graph(model, x, targets)
    own = loss._graph_resources
    assert resources and all(not _resource_closed(r) for r in resources)
    assert len(resources) > len(own), "the whole graph is one node"

    # The closed node's own saved state goes immediately, without gc.
    loss.close()
    assert all(_resource_closed(r) for r in own)
    loss.close()             # idempotent: nothing is released twice
    logits.close()
    # The rest belongs to intermediate nodes the caller never held, so it
    # is released when the abandoned chain is — a wrapper-cycle boundary.
    del loss, logits, own
    gc.collect()
    assert all(_resource_closed(r) for r in resources)
    del resources
    gc.collect()
    assert len(live_storages) == baseline
    _close_run(model, None, x)


# ==========================================================================
# 9. Checkpoint topology, schema, and the whole-state transaction
# ==========================================================================

def test_the_integrated_archive_records_the_whole_state_and_the_topology(
    tmp_path
):
    """The real version-2 archive of the integrated model: format, version,
    canonical generator keys, both alias paths, canonical decimal integer
    strings, no generator arrays, and no external loop state beyond the
    metadata the caller passed."""
    model = NativePhaseGClassifier()
    optimizer = NativeAdam(model.parameters(), lr=LR)
    loss_fn = NativeCrossEntropyLoss()
    batches = _batches()
    for step in range(2):
        _train_step(model, loss_fn, optimizer, batches, step)
    path = os.path.join(str(tmp_path), "topology.npz")
    save_native_checkpoint(path, model, optimizer=optimizer,
                           metadata=_progress(2))

    manifest = _read_manifest(path)
    names = _archive_names(path)
    assert manifest["format"] == native_checkpoint._FORMAT
    assert manifest["format_version"] == native_checkpoint._FORMAT_VERSION == 3
    assert set(manifest) == {"format", "format_version", "model", "optimizer",
                             "generators", "metadata"}
    assert list(manifest["model"]["keys"]) == (list(PARAMETER_NAMES)
                                               + list(BUFFER_NAMES))
    assert manifest["optimizer"]["type"] == "NativeAdam"
    assert len(manifest["optimizer"]["m"]) == len(PARAMETER_NAMES)
    section = manifest["generators"]
    assert list(section["keys"]) == [GENERATOR_PATHS[0]]
    assert section["aliases"] == {GENERATOR_PATHS[0]: GENERATOR_PATHS[0],
                                  GENERATOR_PATHS[1]: GENERATOR_PATHS[0]}
    entry = section["entries"][GENERATOR_PATHS[0]]
    assert set(entry) == {"algorithm", "algorithm_version", "seed", "calls"}
    assert entry["algorithm"] == "tensorforge.splitmix64"
    for field in ("seed", "calls"):
        assert isinstance(entry[field], str)
        assert entry[field].isdigit() and str(int(entry[field])) == entry[field]
    assert int(entry["calls"]) == 2 * CALLS_PER_TRAINING_FORWARD
    assert manifest["metadata"] == _progress(2)
    # Generator state adds no array to the payload, and nothing external
    # to TensorForge is captured.
    assert not any("generator" in name for name in names)
    blob = (" ".join(names) + " " + json.dumps(manifest)).lower()
    for banned in ("mask", "winner", "probabilit", "snapshot", "grad",
                   "mt19937", "bit_generator", "dataloader", "scheduler",
                   "epoch", "shuffle", "random_state"):
        assert banned not in blob, banned
    _close_run(model, optimizer)
    _close_batches(batches)


COMMIT_SEAMS = ["_capture_rollback", "_commit_model", "_commit_optimizer",
                "_commit_generators", "_reach_commit_boundary"]


@pytest.mark.parametrize("seam", COMMIT_SEAMS)
def test_a_commit_failure_rolls_the_whole_integrated_state_back(tmp_path,
                                                                 seam,
                                                                 live_storages):
    """The G5 transaction over a realistic state volume: a failure at any
    commit position restores CNN parameters, all four normalization
    buffers, the optimizer, and the generators together, keeps every
    identity and every parameter version, and leaves the guard and the
    generator locks released."""
    source = NativePhaseGClassifier()
    source_optimizer = NativeAdam(source.parameters(), lr=LR)
    loss_fn = NativeCrossEntropyLoss()
    batches = _batches()
    for step in range(2):
        _train_step(source, loss_fn, source_optimizer, batches, step)
    path = os.path.join(str(tmp_path), f"{seam}.npz")
    save_native_checkpoint(path, source, optimizer=source_optimizer)
    _close_run(source, source_optimizer)

    target = NativePhaseGClassifier(generator_seed=FRESH_GENERATOR_SEED)
    target_optimizer = NativeAdam(target.parameters(), lr=LR)
    _train_step(target, loss_fn, target_optimizer, batches, 0)

    # A graph built before the load must survive it untouched.
    x, targets = _inputs()
    target.train()
    pre_loss, pre_logits, pre_resources = _integrated_graph(target, x, targets)

    values_before = _values(target)
    generators_before = _generator_values(target)
    optimizer_before = _optimizer_values(target_optimizer)
    versions_before = [p.version for p in target.parameters()]
    parameter_ids = [id(p) for p in target.parameters()]
    buffer_ids = [id(b) for b in target.buffers()]
    generator_ids = [id(g) for g in target.generators()]
    gc.collect()
    baseline = len(live_storages)

    with patched(transaction, seam, raiser(_Boom(f"injected at {seam}"))):
        with pytest.raises(_Boom):
            load_native_checkpoint(path, target, optimizer=target_optimizer)

    # The failed load allocated staged values and rollback snapshots and
    # released every one of them.
    gc.collect()
    assert len(live_storages) == baseline
    assert _values(target) == values_before
    assert _generator_values(target) == generators_before
    assert _optimizer_values(target_optimizer) == optimizer_before
    assert [p.version for p in target.parameters()] == versions_before
    assert [id(p) for p in target.parameters()] == parameter_ids
    assert [id(b) for b in target.buffers()] == buffer_ids
    assert [id(g) for g in target.generators()] == generator_ids
    # Both locks were released, and no generator kept a reservation.
    assert not state_lock.held_by_current_thread()
    for generator in target.generators():
        assert not generator._has_active_reservation()
    # The pre-load graph is intact and still produces the same gradients.
    assert all(not _resource_closed(r) for r in pre_resources)
    pre_loss.backward()
    assert all(parameter.grad is not None for parameter in target.parameters())

    for parameter in target.parameters():
        if parameter.grad is not None:
            parameter.grad.close()
    _close_run(target, target_optimizer, pre_loss, pre_logits, x)
    _close_batches(batches)


def test_a_successful_load_leaves_no_reservation_and_no_held_guard(tmp_path):
    model = NativePhaseGClassifier()
    optimizer = NativeAdam(model.parameters(), lr=LR)
    path = os.path.join(str(tmp_path), "clean.npz")
    save_native_checkpoint(path, model, optimizer=optimizer)
    load_native_checkpoint(path, model, optimizer=optimizer)
    assert not state_lock.held_by_current_thread()
    for generator in model.generators():
        assert not generator._has_active_reservation()
        assert generator._claim_serial == generator_module._NO_RESERVATION
    _close_run(model, optimizer)


def test_a_live_reservation_refuses_a_save_and_a_load(tmp_path):
    """A reservation in flight is not something a checkpoint may step
    over: both the save and the load refuse, changing nothing, and both
    succeed once the reservation settles."""
    model = NativePhaseGClassifier()
    path = os.path.join(str(tmp_path), "reserved.npz")
    save_native_checkpoint(path, model)
    generator = model.drop2d.generator
    before = _generator_values(model)

    token = generator._reserve_call()
    try:
        with pytest.raises(RuntimeError):
            save_native_checkpoint(os.path.join(str(tmp_path), "no.npz"),
                                   model)
        with pytest.raises(RuntimeError):
            load_native_checkpoint(path, model)
        assert _generator_values(model) == before
        assert not os.path.exists(os.path.join(str(tmp_path), "no.npz"))
    finally:
        generator._abandon_call(token)
    load_native_checkpoint(path, model)
    assert _generator_values(model) == before
    _close_run(model)


@pytest.mark.parametrize("mismatch", ["model_shape", "optimizer_type",
                                      "generator_topology"])
def test_prevalidation_rejects_every_section_mismatch_atomically(tmp_path,
                                                                  mismatch):
    """A load is one transaction over the whole archive: a model-section,
    optimizer-section, or generator-section mismatch is caught in
    prevalidation, with all four state families bit-identical."""
    source = NativePhaseGClassifier()
    source_optimizer = NativeAdam(source.parameters(), lr=LR)
    loss_fn = NativeCrossEntropyLoss()
    batches = _batches()
    _train_step(source, loss_fn, source_optimizer, batches, 0)
    path = os.path.join(str(tmp_path), f"{mismatch}.npz")
    save_native_checkpoint(path, source, optimizer=source_optimizer)
    _close_run(source, source_optimizer)
    _close_batches(batches)

    if mismatch == "model_shape":
        target = NativePhaseGClassifier()
        target.head = NativeLinear(HIDDEN_FEATURES, NUM_CLASSES + 1, seed=9)
        target_optimizer = NativeAdam(target.parameters(), lr=LR)
    elif mismatch == "optimizer_type":
        target = NativePhaseGClassifier()
        target_optimizer = NativeSGD(target.parameters(), lr=LR)
    else:
        target = NativePhaseGClassifier(shared=False)
        target_optimizer = NativeAdam(target.parameters(), lr=LR)

    values_before = _values(target)
    generators_before = _generator_values(target)
    optimizer_before = _optimizer_values(target_optimizer)
    versions_before = [p.version for p in target.parameters()]
    with pytest.raises(Exception):
        load_native_checkpoint(path, target, optimizer=target_optimizer)
    assert _values(target) == values_before
    assert _generator_values(target) == generators_before
    assert _optimizer_values(target_optimizer) == optimizer_before
    assert [p.version for p in target.parameters()] == versions_before
    assert not state_lock.held_by_current_thread()
    _close_run(target, target_optimizer)


# ==========================================================================
# 10. Shared and frozen parameters through the stochastic stack
# ==========================================================================

class _SharedParameterDropoutModel(NativeModule):
    """**Test-only.** One ``NativeParameter`` registered under two paths
    and used twice in one forward, with a Dropout layer between the two
    uses."""

    def __init__(self):
        super().__init__()
        self.scale = NativeParameter(np.array([[2.0, -1.0], [0.5, 1.5]]))
        self.alias = self.scale               # the *same* object
        self.drop = NativeDropout(DROPOUT_P, seed=SHARED_GENERATOR_SEED)

    def forward(self, x):
        return self.drop(x.matmul(self.scale)).matmul(self.alias)


def test_a_shared_parameter_updates_once_through_the_stochastic_stack():
    model = _SharedParameterDropoutModel()
    assert [name for name, _ in model.named_parameters()] == ["scale"]
    assert model.alias is model.scale
    optimizer = NativeAdam(model.parameters(), lr=LR)
    assert len(optimizer.parameters()) == 1

    x = NativeTensor.from_array(np.array([[1.0, 2.0], [3.0, 4.0]]))
    version = model.scale.version
    model.train()
    output = model(x)
    loss = output.sum()
    loss.backward()
    assert model.drop.generator.calls == 1
    optimizer.step()
    optimizer.zero_grad()
    # One slot, one update, one version increment — never two.
    assert list(optimizer.step_counts) == [1]
    assert model.scale.version == version + 1
    _close_run(model, optimizer, loss, output, x)


class _FrozenParameterDropoutModel(NativeModule):
    """**Test-only.** A frozen parameter that really participates in a
    stochastic forward, alongside a trainable head."""

    def __init__(self):
        super().__init__()
        self.frozen = NativeParameter(np.array([[1.5, -0.5], [0.25, 2.0]]),
                                      requires_grad=False)
        self.drop = NativeDropout(DROPOUT_P, seed=SHARED_GENERATOR_SEED)
        self.head = NativeLinear(2, 1, seed=3)

    def forward(self, x):
        return self.head(self.drop(x.matmul(self.frozen)))


def test_a_frozen_parameter_stays_registered_persisted_and_skipped(tmp_path):
    model = _FrozenParameterDropoutModel()
    assert "frozen" in [name for name, _ in model.named_parameters()]
    assert model.frozen.requires_grad is False
    optimizer = NativeAdam(model.parameters(), lr=LR)
    frozen_before = model.frozen.to_numpy().copy()
    frozen_version = model.frozen.version

    x = NativeTensor.from_array(np.array([[1.0, 2.0], [3.0, 4.0]]))
    model.train()
    output = model(x)
    loss = output.sum()
    loss.backward()
    assert model.frozen.grad is None
    optimizer.step()
    optimizer.zero_grad()
    assert np.array_equal(model.frozen.to_numpy(), frozen_before)
    assert model.frozen.version == frozen_version
    assert not np.array_equal(model.head.weight.to_numpy(),
                              np.zeros_like(model.head.weight.to_numpy()))

    # ...and it is still persisted and restored like any other parameter.
    path = os.path.join(str(tmp_path), "frozen.npz")
    save_native_checkpoint(path, model)
    assert "frozen" in _read_manifest(path)["model"]["keys"]
    fresh = _FrozenParameterDropoutModel()
    load_native_checkpoint(path, fresh)
    assert np.array_equal(fresh.frozen.to_numpy(), frozen_before)
    assert fresh.frozen.requires_grad is False
    _close_run(model, optimizer, loss, output, x)
    _close_run(fresh)


# ==========================================================================
# 11. Concurrency: the participating transactions serialize
# ==========================================================================

def _distinct_checkpoints(tmp_path):
    """Two archives whose every state family differs, plus the values each
    one must produce."""
    archives = []
    for index, steps in enumerate((1, 3)):
        model = NativePhaseGClassifier(
            generator_seed=SHARED_GENERATOR_SEED + index + 1
        )
        optimizer = NativeAdam(model.parameters(), lr=LR)
        loss_fn = NativeCrossEntropyLoss()
        batches = _batches()
        for step in range(steps):
            _train_step(model, loss_fn, optimizer, batches, step)
        path = os.path.join(str(tmp_path), f"archive_{index}.npz")
        save_native_checkpoint(path, model, optimizer=optimizer,
                               metadata=_progress(steps))
        archives.append({
            "path": path,
            "values": _values(model),
            "optimizer": _optimizer_values(optimizer),
            "generators": _generator_values(model),
        })
        _close_run(model, optimizer)
        _close_batches(batches)
    return archives


def test_two_concurrent_loads_leave_one_complete_state_never_a_hybrid(
    tmp_path
):
    """Two version-2 CNN archives loaded concurrently into one live model.
    The shared guard serializes them, so the result is one archive's state
    in full — parameters, buffers, optimizer, and generators together."""
    archives = _distinct_checkpoints(tmp_path)
    assert archives[0]["values"] != archives[1]["values"]
    assert archives[0]["generators"] != archives[1]["generators"]

    target = NativePhaseGClassifier(generator_seed=FRESH_GENERATOR_SEED)
    target_optimizer = NativeAdam(target.parameters(), lr=LR)
    interleaver = Interleaver()
    entered = threading.Event()
    real_capture = transaction._capture_rollback

    def blocking_capture(plan, targets):
        record = real_capture(plan, targets)
        if not entered.is_set():
            entered.set()
            interleaver.block()      # hold the guard, mid-transaction
        return record

    finished = []

    def loader(index):
        def run():
            load_native_checkpoint(archives[index]["path"], target,
                                   optimizer=target_optimizer)
            finished.append(index)
        return run

    with patched(transaction, "_capture_rollback", blocking_capture):
        threads = [threading.Thread(target=loader(0), daemon=True)]
        threads[0].start()
        interleaver.wait_for_arrival()
        second = threading.Thread(target=loader(1), daemon=True)
        second.start()
        # The second load cannot get past the guard while the first holds it.
        second.join(0.2)
        assert second.is_alive(), "a second load entered the guard"
        assert finished == []
        interleaver.let_go()
        for thread in threads + [second]:
            thread.join(JOIN_TIMEOUT)
            assert not thread.is_alive()

    assert sorted(finished) == [0, 1]
    winner = finished[-1]
    expected = archives[winner]
    assert _values(target) == expected["values"]
    assert _optimizer_values(target_optimizer) == expected["optimizer"]
    assert _generator_values(target) == expected["generators"]
    # ...and never a mixture of the two.
    other = archives[1 - winner]
    assert _values(target) != other["values"]
    _close_run(target, target_optimizer)


def test_a_save_racing_a_state_replacement_describes_one_serial_point(
    tmp_path
):
    """A save snapshot holds the guard until the whole payload exists, so a
    concurrent participating replacement lands strictly before or strictly
    after it — never between the model and the generator sections."""
    model = NativePhaseGClassifier()
    optimizer = NativeAdam(model.parameters(), lr=LR)
    loss_fn = NativeCrossEntropyLoss()
    batches = _batches()
    _train_step(model, loss_fn, optimizer, batches, 0)
    before = _values(model)["parameters"]

    replacement = {name: NativeTensor.from_array(
        np.full(parameter.shape, 0.125)
    ) for name, parameter in model.named_parameters()}
    interleaver = Interleaver()
    real_section = native_checkpoint._generator_section

    def blocking_section(target, where):
        section = real_section(target, where)
        interleaver.block()          # still inside the save's guard
        return section

    path = os.path.join(str(tmp_path), "race_save.npz")
    replaced = threading.Event()

    def replace():
        interleaver.wait_for_arrival()
        thread = threading.Thread(
            target=lambda: (model.load_state_dict(replacement, strict=False),
                            replaced.set()),
            daemon=True,
        )
        thread.start()
        thread.join(0.2)
        assert not replaced.is_set(), "a replacement slipped inside the save"
        interleaver.let_go()
        thread.join(JOIN_TIMEOUT)
        assert replaced.is_set()

    with patched(native_checkpoint, "_generator_section", blocking_section):
        run_threads([lambda: save_native_checkpoint(path, model,
                                                    optimizer=optimizer),
                     replace])

    # The archive describes the state *before* the replacement, in full.
    manifest = _read_manifest(path)
    with np.load(path, allow_pickle=False) as archive:
        entry = manifest["model"]["entries"]["conv.weight"]
        saved = archive[entry["array"]].reshape(entry["shape"])
    assert np.array_equal(saved, np.asarray(before["conv.weight"]))
    assert not np.array_equal(model.conv.weight.to_numpy(), saved)
    for tensor in replacement.values():
        if not tensor.closed:
            tensor.close()
    _close_run(model, optimizer)
    _close_batches(batches)


def test_a_load_racing_a_generator_replacement_is_serial(tmp_path):
    """A generator-state replacement cannot land inside a checkpoint load's
    commit: the final state is one order or the other, never a mixture."""
    model = NativePhaseGClassifier()
    optimizer = NativeAdam(model.parameters(), lr=LR)
    loss_fn = NativeCrossEntropyLoss()
    batches = _batches()
    _train_step(model, loss_fn, optimizer, batches, 0)
    path = os.path.join(str(tmp_path), "race_load.npz")
    save_native_checkpoint(path, model, optimizer=optimizer)
    archived = _generator_values(model)["states"]

    replacement_state = {
        GENERATOR_PATHS[0]: {
            "algorithm": "tensorforge.splitmix64",
            "algorithm_version": 1,
            "seed": 5150, "calls": 4096,
        }
    }
    interleaver = Interleaver()
    real_commit = transaction._commit_model
    replaced = threading.Event()

    def blocking_commit(target, staged):
        real_commit(target, staged)
        interleaver.block()

    def replace():
        interleaver.wait_for_arrival()
        thread = threading.Thread(
            target=lambda: (model.load_generator_state_dict(replacement_state),
                            replaced.set()),
            daemon=True,
        )
        thread.start()
        thread.join(0.2)
        assert not replaced.is_set(), "a replacement entered the commit"
        interleaver.let_go()
        thread.join(JOIN_TIMEOUT)

    with patched(transaction, "_commit_model", blocking_commit):
        run_threads([lambda: load_native_checkpoint(path, model,
                                                    optimizer=optimizer),
                     replace])

    # The load committed first, then the replacement — one serial order.
    final = _generator_values(model)["states"][GENERATOR_PATHS[0]]
    assert final == {"algorithm": "tensorforge.splitmix64",
                     "algorithm_version": 1, "seed": 5150, "calls": 4096}
    assert final != archived[GENERATOR_PATHS[0]]
    _close_run(model, optimizer)
    _close_batches(batches)


def test_a_reservation_and_a_load_never_overlap(tmp_path):
    """A reservation takes only its own generator's lock, so it either
    completes before a load takes that lock or begins after the load
    released it. Either way, no state is replaced under a live token."""
    model = NativePhaseGClassifier()
    path = os.path.join(str(tmp_path), "reservation.npz")
    save_native_checkpoint(path, model)
    generator = model.drop2d.generator
    x, _targets = _inputs()
    observed = []

    def draw():
        pooled = NativeTensor.from_array(np.arange(6.0).reshape(2, 3))
        try:
            result = pooled.dropout(DROPOUT_P, generator=generator)
            observed.append(("draw", generator.seed, result.closed))
            result.close()
        finally:
            pooled.close()

    def load():
        try:
            load_native_checkpoint(path, model)
            observed.append(("load", "ok", None))
        except RuntimeError as error:      # refused by a live reservation
            observed.append(("load", "refused", str(error)))

    run_threads([draw, load])
    assert len(observed) == 2
    assert not generator._has_active_reservation()
    # Whichever order ran, the generator's state is coherent: the seed is
    # the archived one and the counter never went backwards mid-draw.
    assert generator.seed == SHARED_GENERATOR_SEED
    assert generator.calls in (0, 1)
    _close_run(model, None, x)


# ==========================================================================
# 11. Earlier phases still work, and the two lines stay separate
# ==========================================================================

def test_core_arithmetic_and_basic_autograd_still_work():
    a = NativeTensor.from_array(np.array([[1.0, 2.0], [3.0, 4.0]]),
                                requires_grad=True)
    b = NativeTensor.from_array(np.array([[0.5, 0.5], [2.0, 2.0]]),
                                requires_grad=True)
    total = a.multiply(b).add(a).sum()
    total.backward()
    assert np.allclose(a.grad.to_numpy(), b.to_numpy() + 1.0)
    assert np.allclose(b.grad.to_numpy(), a.to_numpy())
    core = cpp.NativeTensorCore.from_array(np.array([[1.0, 2.0]]))
    doubled = core.add(core)
    try:
        assert np.array_equal(doubled.to_numpy(), np.array([[2.0, 4.0]]))
    finally:
        doubled.close()
        core.close()
    for tensor in (a, b):
        tensor.grad.close()
    _close_run(None, None, total, a, b)


@pytest.mark.parametrize("build", [
    "linear", "conv", "maxpool", "layernorm", "batchnorm1d", "batchnorm2d",
])
def test_earlier_native_modules_still_train(build):
    """A compact representative matrix: every pre-Phase-G module still
    completes a forward, a backward, and an optimizer step. The exhaustive
    coverage stays in the phase suites; this proves Phase G broke none of
    them."""
    shapes = {
        "linear": (2, 4), "layernorm": (2, 4), "batchnorm1d": (4, 4),
        "conv": (2, 1, 4, 4), "maxpool": (2, 1, 4, 4),
        "batchnorm2d": (2, 2, 2, 4),
    }
    builders = {
        "linear": lambda: NativeLinear(4, 3, seed=0),
        "conv": lambda: NativeConv2d(1, 2, 3, seed=0),
        "maxpool": lambda: NativeMaxPool2d(2),
        "layernorm": lambda: NativeLayerNorm(4),
        "batchnorm1d": lambda: NativeBatchNorm1d(4),
        "batchnorm2d": lambda: NativeBatchNorm2d(2),
    }
    shape = shapes[build]
    count = int(np.prod(shape))
    module = builders[build]()
    values = (np.arange(count, dtype=np.float64) % 7.0) - 3.0
    x = NativeTensor.from_array(values.reshape(shape), requires_grad=True)

    parameters = list(module.parameters())
    optimizer = NativeAdam(parameters, lr=LR) if parameters else None
    module.train()
    output = module(x)
    # A *weighted* objective, never a plain sum: normalization makes the
    # gradient of an unweighted sum structurally zero for gamma, which
    # would make "the parameter moved" vacuous.
    weights = NativeTensor.from_array(
        (np.arange(output.numel, dtype=np.float64) % 5.0 + 0.5)
        .reshape(output.shape)
    )
    loss = output.multiply(weights).sum()
    loss.backward()
    assert x.grad is not None
    if optimizer is not None:
        before = [p.to_numpy().copy() for p in parameters]
        optimizer.step()
        optimizer.zero_grad()
        for index, parameter in enumerate(parameters):
            assert not np.array_equal(parameter.to_numpy(), before[index]), (
                build, index
            )
    assert np.isfinite(output.to_numpy()).all()
    x.grad.close()
    x.zero_grad()
    _close_run(module, optimizer, loss, output, weights, x)


def test_cross_entropy_and_both_optimizers_still_work():
    for optimizer_class in (NativeSGD, NativeAdam):
        model = NativeSequential(NativeLinear(4, 3, seed=0))
        optimizer = optimizer_class(model.parameters(), lr=LR)
        x = NativeTensor.from_array(np.arange(8.0).reshape(2, 4))
        logits = model(x)
        loss = NativeCrossEntropyLoss()(logits, [0, 2])
        value = float(loss.to_numpy())
        assert math.isfinite(value) and value > 0.0
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        _close_run(model, optimizer, loss, logits, x)


def test_checkpoint_v1_and_generator_free_v2_still_work(tmp_path):
    """A generator-free model still saves a version-2 archive with an
    explicit null generator section, and a version-1 archive still loads
    into it."""
    model = NativeSequential(NativeLinear(4, 3, seed=0), NativeBatchNorm1d(3))
    path = os.path.join(str(tmp_path), "plain_v2.npz")
    save_native_checkpoint(path, model)
    manifest = _read_manifest(path)
    assert manifest["format_version"] == 3
    assert manifest["generators"] is None

    # The same archive downgraded to the version-1 field set still loads.
    v1_path = os.path.join(str(tmp_path), "plain_v1.npz")
    with np.load(path, allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}
    downgraded = dict(manifest)
    downgraded["format_version"] = 1
    downgraded.pop("generators")
    arrays["manifest"] = np.frombuffer(
        json.dumps(downgraded).encode("utf-8"), dtype=np.uint8
    )
    np.savez(v1_path, **arrays)
    load_native_checkpoint(v1_path, model)
    assert native_checkpoint._SUPPORTED_FORMAT_VERSIONS == (1, 2, 3)

    # ...but a version-1 archive cannot describe a model that has a
    # generator: it would have to fabricate a seed and a counter.
    with_generator = NativePhaseGClassifier()
    with pytest.raises(Exception):
        load_native_checkpoint(v1_path, with_generator)
    _close_run(model)
    _close_run(with_generator)


def test_the_stable_and_native_lines_stay_separate():
    """No implicit conversion in either direction, and the stable line is
    untouched by Phase G."""
    model = NativePhaseGClassifier()
    stable = tensorforge.Tensor(np.zeros((2, HIDDEN_FEATURES)))
    with pytest.raises(TypeError):
        model.drop1d(stable)
    native = NativeTensor.from_array(np.zeros((4, 3)))
    stable_dropout = tensorforge.Dropout(0.5)
    with pytest.raises((TypeError, AttributeError)):
        stable_dropout(native)
    # The stable line still works normally, and shares no state.
    stable_dropout.eval()
    assert stable_dropout(tensorforge.Tensor(np.ones((2, 2)))).data.tolist() == (
        [[1.0, 1.0], [1.0, 1.0]]
    )
    assert not hasattr(tensorforge, "NativeDropout")
    assert not hasattr(tensorforge, "NativeGenerator")
    _close_run(model, None, native)


# ==========================================================================
# 12. Public surface and the capability boundary
# ==========================================================================

def test_the_public_surface_is_exactly_separated():
    import tensorforge.experimental as experimental

    for name in ("NativeGenerator", "NativeDropout"):
        assert name in experimental.__all__, name
        assert hasattr(experimental, name), name
        assert not hasattr(tensorforge, name), name
    # Nothing private leaked into either namespace.
    for absent in ("_ReservationToken", "ReservationToken", "state_transaction",
                   "held_by_current_thread", "commit_checkpoint",
                   "replace_generator_states", "replace_native_state",
                   "snapshot_generator_states", "_dropout_forward_with_mask",
                   "_native_state_lock", "_native_checkpoint_transaction"):
        assert absent not in experimental.__all__, absent
        assert not hasattr(tensorforge, absent), absent
    # The private mask is not reachable from any public object.
    module = NativeDropout(DROPOUT_P, seed=1)
    for absent in ("mask", "masks", "last_mask", "owns_generator"):
        assert not hasattr(module, absent), absent
    assert sorted(experimental.__all__) == sorted(set(experimental.__all__))


def test_the_capability_inventories_are_exactly_what_g8_left():
    """G9 adds integration evidence, not capability. Every registry value
    is unchanged, and ``"dropout"`` stays unsupported until G10."""
    from tensorforge.experimental import native_checkpoint as checkpoint

    assert cpp.UNSUPPORTED == ("cuda", "amp")
    assert cpp.SUPPORTED_DTYPES == ("float64", "float32")
    assert cpp.SUPPORTED_DEVICES == ("cpu",)
    assert cpp.NATIVE_MODULES.count("NativeDropout") == 1
    assert "dropout" in cpp.AUTOGRAD_OPS
    assert "dropout_forward" in cpp.TENSOR_CORE_OPS
    assert "generator_state" in cpp.STATE_SUPPORT
    assert "checkpoint_generator_state" in cpp.STATE_SUPPORT
    assert checkpoint._FORMAT_VERSION == 3
    assert checkpoint._SUPPORTED_FORMAT_VERSIONS == (1, 2, 3)
    assert cpp.NATIVE_OPTIMIZERS == ("NativeSGD", "NativeAdam")
    # No integration or phase capability name was invented.
    inventories = (cpp.NATIVE_MODULES, cpp.AUTOGRAD_OPS, cpp.TENSOR_CORE_OPS,
                   cpp.STATE_SUPPORT, cpp.RAW_KERNELS, cpp.NATIVE_LOSSES,
                   cpp.NATIVE_METRICS, cpp.UNSUPPORTED)
    for inventory in inventories:
        for name in inventory:
            for banned in ("phase_g", "integration", "dropout2d", "dropout3d",
                           "rng_api"):
                assert banned not in name.lower(), name
    assert "tf_core_dropout_forward" in cpp._CHECKED_KERNELS
    for absent in ("tf_core_dropout_backward", "tf_core_random_uniform"):
        assert absent not in cpp._CHECKED_KERNELS, absent


def test_the_closure_belongs_to_g10_and_not_to_this_suite():
    """The closure milestone owns the boundary move, the sanitizer matrix,
    and the final documentation reconciliation. G10 has since run — the
    ladder says so and the boundary has moved — but **none of that work
    lives here**. G9 is integration evidence, and this file must stay
    that: no sanitizer material, no build validation, no closure claim.

    The old form of this guard asserted G10 had not begun. That premise
    expired at the closure; the *scope* separation it protected did
    not, so that is what it now checks."""
    assert "dropout" not in cpp.UNSUPPORTED
    design = (REPO_ROOT / "docs"
              / "native_rng_dropout_design.md").read_text(encoding="utf-8")
    ladder = design[design.index("| Milestone | Scope | Status |"):]
    ladder = ladder[:ladder.index("### G0")]
    import re

    row = re.search(r"\|\s*G10\s*\|[^|]*\|([^|]*)\|", ladder)
    assert row is not None
    status = re.sub(r"[*`]", "", row.group(1)).strip().lower()
    assert status.startswith("complete"), status
    # Assembled at runtime so this guard's own list is not a match for it.
    source = Path(__file__).read_text(encoding="utf-8").lower()
    for banned in ("a" + "san", "ub" + "san", "leak" + "sanitizer",
                   "sanitizer " + "closure", "phase g is " + "closed"):
        assert banned not in source, banned


# ==========================================================================
# 13. Native-storage lifecycle across the whole integrated workflow
# ==========================================================================

def test_repeated_integrated_lifecycles_return_storage_to_baseline(
    tmp_path, live_storages
):
    """The complete workflow, repeated: training with Adam and with SGD,
    evaluation, a p == 0 model, a save, a successful load, a failed load
    that rolls back, a retained graph, and an abandoned graph — each cycle
    returning native live storage exactly to the baseline."""
    gc.collect()
    baseline = len(live_storages)
    observed = []

    for cycle in range(2):
        batches = _batches()
        x, targets = _inputs()

        # Adam training, evaluation, and a checkpoint round trip.
        model = NativePhaseGClassifier()
        optimizer = NativeAdam(model.parameters(), lr=LR)
        loss_fn = NativeCrossEntropyLoss()
        for step in range(2):
            _train_step(model, loss_fn, optimizer, batches, step)
        _evaluate(model, x, targets)
        path = os.path.join(str(tmp_path), f"cycle_{cycle}.npz")
        save_native_checkpoint(path, model, optimizer=optimizer,
                               metadata=_progress(2))
        load_native_checkpoint(path, model, optimizer=optimizer)

        # A failed load that rolls back.
        with patched(transaction, "_commit_generators",
                     raiser(_Boom("cycle rollback"))):
            with pytest.raises(_Boom):
                load_native_checkpoint(path, model, optimizer=optimizer)

        # A retained graph and an abandoned graph.
        model.train()
        retained_logits = model(x)
        retained_loss = loss_fn(retained_logits, targets)
        retained_loss.backward(retain_graph=True)
        retained_loss.backward()
        abandoned_logits = model(x)
        abandoned_loss = loss_fn(abandoned_logits, targets)
        abandoned_loss.close()
        _close_run(model, optimizer, retained_loss, retained_logits,
                   abandoned_logits)
        # A closed node still holds its parents, so the abandoned chain's
        # intermediates are released at the wrapper-cycle boundary — which
        # only exists once the root is unbound.
        del retained_loss, retained_logits, abandoned_loss, abandoned_logits

        # SGD training on a fresh model.
        sgd_model = NativePhaseGClassifier(shared=False)
        sgd = NativeSGD(sgd_model.parameters(), lr=LR)
        _train_step(sgd_model, loss_fn, sgd, batches, 0)
        _close_run(sgd_model, None)

        # A p == 0 model.
        zero_model = NativePhaseGClassifier(p=0.0)
        zero_logits = zero_model(x)
        zero_loss = loss_fn(zero_logits, targets)
        zero_loss.backward()
        _close_run(zero_model, None, zero_loss, zero_logits)

        x.close()
        _close_batches(batches)
        observed.append(settled(live_storages))

    assert observed == [baseline] * len(observed), observed
    assert settled(live_storages) == baseline
    # No generator anywhere kept a reservation or a construction claim.
    probe = NativePhaseGClassifier()
    for generator in probe.generators():
        assert not generator._has_active_reservation()
        assert generator._claim_serial == generator_module._NO_RESERVATION
    assert not state_lock.held_by_current_thread()
    _close_run(probe)
    assert settled(live_storages) == baseline


def test_failure_cycles_leave_no_storage_behind(tmp_path, live_storages):
    """Repeated *failed* workflows — a rejected topology, a rolled-back
    commit, and a stale-graph backward — leave the counter exactly where
    it started and no generator holding a reservation."""
    source = NativePhaseGClassifier()
    source_optimizer = NativeAdam(source.parameters(), lr=LR)
    path = os.path.join(str(tmp_path), "failures.npz")
    save_native_checkpoint(path, source, optimizer=source_optimizer)
    _close_run(source, source_optimizer)
    gc.collect()
    baseline = len(live_storages)

    for _ in range(3):
        mismatched = NativePhaseGClassifier(shared=False)
        mismatched_optimizer = NativeAdam(mismatched.parameters(), lr=LR)
        with pytest.raises(Exception):
            load_native_checkpoint(path, mismatched,
                                   optimizer=mismatched_optimizer)
        _close_run(mismatched, mismatched_optimizer)

        target = NativePhaseGClassifier()
        with patched(transaction, "_commit_optimizer",
                     raiser(_Boom("rollback"))):
            optimizer = NativeAdam(target.parameters(), lr=LR)
            with pytest.raises(_Boom):
                load_native_checkpoint(path, target, optimizer=optimizer)
        assert not state_lock.held_by_current_thread()
        for generator in target.generators():
            assert not generator._has_active_reservation()
        _close_run(target, optimizer)

    assert settled(live_storages) == baseline


_NUMERICAL_NUMPY = (
    "add", "subtract", "multiply", "divide", "matmul", "dot", "exp", "log",
    "maximum", "sqrt", "reciprocal", "mean", "var", "sum", "where",
)


def test_one_complete_integrated_step_reaches_no_numpy(monkeypatch):
    """The conversion tripwire over the fully integrated step: with
    Dropout, both normalizations, pooling, the loss, and NativeAdam in the
    path, not one numerical NumPy entry point is reached between the
    forward and the optimizer update."""
    model = NativePhaseGClassifier()
    optimizer = NativeAdam(model.parameters(), lr=LR)
    loss_fn = NativeCrossEntropyLoss()
    x, targets = _inputs()
    tripped = []

    def tripwire(name):
        def guard(*args, **kwargs):
            tripped.append(name)
            raise AssertionError(f"the integrated step reached numpy.{name}")
        return guard

    for name in _NUMERICAL_NUMPY:
        monkeypatch.setattr(np, name, tripwire(name), raising=False)

    model.train()
    logits = model(x)
    loss = loss_fn(logits, targets)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    assert tripped == []
    monkeypatch.undo()
    assert model.drop2d.generator.calls == CALLS_PER_TRAINING_FORWARD
    _close_run(model, optimizer, loss, logits, x)


def test_the_phase_g_integration_suite_is_the_only_new_artifact():
    """G9 ships one test module. No example, benchmark, result artifact, or
    runtime file arrives with it."""
    assert (REPO_ROOT / "tests" / "test_native_phase_g.py").is_file()
    assert not (REPO_ROOT / "benchmark_results").exists()
    for absent in ("examples/native_phase_g.py",
                   "examples/native_dropout_integration.py",
                   "benchmarks/benchmark_native_phase_g.py",
                   "src/tensorforge/experimental/native_phase_g.py"):
        assert not (REPO_ROOT / absent).exists(), absent
    # The G7 example and the G8 benchmark are still exactly where they were.
    assert (REPO_ROOT / "examples" / "native_dropout_training.py").is_file()
    assert (REPO_ROOT / "benchmarks" / "benchmark_native_dropout.py").is_file()
