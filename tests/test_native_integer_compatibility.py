"""Phase K, milestone K5 — the compatibility proof.

K5 adds **zero production code**. Everything it exercises already shipped:
K1's reachability barriers, K2's one public integer door, K3's ``argmax``,
K4's ``index_select``, and — untouched by every one of them — the native
checkpoint, the in-memory optimizer state, the Phase-J loader and sampler
states, the Phase-E classification stack, and the deterministic
training/resume proof each earlier phase committed to. What was never
proved is that the *bounded* integer work of K1-K4 left all of that exactly
where it found it, so this module is that proof and nothing else.

This module owns the split ``docs/native_integer_tensors_design.md`` §30.1
assigns it: **checkpoint, state, pipeline, and classification
compatibility**. It is deliberately a *system* module. The unit-level
barrier assertions live in ``tests/test_native_integer_barriers.py`` (K1,
driven from the raw handle) and ``tests/test_native_int64_tensor.py`` (K2,
driven from a real public tensor); what is new here is the same claims
driven end to end — through a real ``.npz`` archive and the real parser,
through the real three-object data pipeline, through the real loss and
metric, and through a real interrupted-and-resumed training run.

Eleven claims, and only their conjunction is the milestone:

1. **No archive can declare an ``int64`` entry** — proved from both
   directions: the **writer** refuses to emit one, even from live state
   the public API cannot produce, before any file exists; and the
   **loader** rejects a hand-mutated archive that declares ``int64`` at a
   **parameter**, a **persistent-buffer**, an **optimizer-moment**, or an
   **optimizer-parameter** entry, before any destination is published and
   without allocating a single ``int64`` storage. The writer's half was
   added by a separate checkpoint-hardening repair that this proof
   provoked — see
   ``test_the_parameter_registration_paths_carry_no_second_dtype_authority``,
   and ``tests/test_native_checkpoint.py`` for that repair's own
   regression.
2. **The checkpoint format, current version, and accepted set are
   unmoved** — ``tensorforge.native_checkpoint``, **3**, ``(1, 2, 3)``,
   with no version 4 written, reserved, or accepted anywhere.
3. **Parameter, buffer, and optimizer state stay floating-only**, at both
   persistence values and at both optimizers, with a real ``int64`` tensor
   as the rejected operand — and the parameter role's authority is located
   exactly rather than assumed: it is ``NativeParameter.__init__`` and
   **only** that, because neither registration route re-checks a dtype,
   which is measured here by driving both of them with a genuine
   ``NativeParameter`` that already carries an ``int64`` core. A module
   forced into that state cannot reach an archive either, because the
   checkpoint writer refuses it (claim 1).
4. **The optimizer-state version is unmoved** — **1**.
5. **The loader-state version and accepted set are unmoved** — **1**,
   ``(1,)``.
6. **The sampler-state version and accepted set are unmoved** — **1**,
   ``(1,)``.
7. **Phase J still delivers the same host-label contract** — a floating
   ``NativeTensor`` feature batch and a read-only host ``numpy.ndarray``
   target batch of dtype ``int64``, at both feature dtypes, with no option
   anywhere requesting native labels.
8. **Explicit caller conversion works and needs no pipeline change** —
   a writable ``int64`` copy the caller makes from the delivered target
   batch is what ``NativeTensor.from_int64_array`` receives, the result is
   independent of that array in **both** directions, ``index_select``
   consumes it, and it is *still* refused by every state-owning surface.
   Neither the conversion nor the indexing moves the loader or the sampler
   state by so much as one field **after the delivery** — the delivery
   itself advances the position, and that is Phase J's contract, not a
   K5 claim. This is a caller's line of code after delivery; it is **not**
   loader behavior and is never described as one.
9. **``NativeCrossEntropyLoss`` is behaviorally unchanged** — the same
   accepted host-target forms, the same values and gradients, and a native
   ``int64`` target (including one ``argmax`` just produced) still
   rejected at the same host boundary.
10. **``native_accuracy`` is behaviorally unchanged** — it still
    materializes through ``to_numpy()`` and calls NumPy's ``argmax``, and
    it is proved *not* to call ``NativeTensor.argmax`` or
    ``NativeTensor.index_select`` by patching both to raise and watching it
    succeed anyway.
11. **Deterministic training, checkpointing, and resume stay
    bit-identical while ``argmax`` and ``index_select`` are used beside the
    training path** — independently at float64 and float32, with an
    observational control proving the evaluation indexing changes no
    trainable state at all.

Discipline inherited (integer design §29.6, §30.2):

* **Exact equality only** for integers, and **raw IEEE-754 bits** through
  ``uint32``/``uint64`` views for floating values wherever a bit-identical
  claim is made. A tolerance appears in exactly two places — the
  cross-entropy *value* oracle and the cross-entropy *gradient* oracle,
  each a claim about arithmetic being right rather than about bits being
  preserved — and both are labelled where they are used.
* Each dtype is proved **only against itself**. There is no cross-dtype
  numeric comparison anywhere; the only cross-dtype claims are the batch
  index sequence and the argmax index values, neither of which carries a
  dtype.
* **Every rejection is followed by a before/after fingerprint** of what it
  must have left alone, and every fingerprint and scanner has a
  **non-vacuity control** proving it can notice the change it exists for.
* **Abandonment is proved by explicit ``close()``.** No assertion depends
  on collection timing, and the live-storage tracker installs itself
  **outside** ``monkeypatch``.
* **Source scans read code, not prose** — docstrings and string literals
  are stripped through the AST first, and keyword-argument names are read
  too.
* No test starts a thread, touches the network, needs a Git ancestor, or
  depends on a total suite count.

**Not proved here, because it does not exist:** the K6 example, the K7
hardening matrix, the K8 benchmark, and the K9 closure. §7 below asserts
their absence.

Selector: python -m pytest -q tests/test_native_integer_compatibility.py
"""
import ast
import contextlib
import gc
import inspect
import json
import re
from pathlib import Path

import numpy as np
import pytest

import tensorforge.experimental as experimental
from tensorforge.backends import cpp
from tensorforge.experimental import (
    NativeAdam,
    NativeBatchNorm1d,
    NativeBatchSampler,
    NativeCrossEntropyLoss,
    NativeDataLoader,
    NativeDropout,
    NativeGenerator,
    NativeLinear,
    NativeModule,
    NativeParameter,
    NativeReLU,
    NativeSGD,
    NativeTensor,
    NativeTensorDataset,
    load_native_checkpoint,
    native_accuracy,
    native_checkpoint,
    native_optimizer_state,
    save_native_checkpoint,
)
from tensorforge.experimental import native_data_loader as loader_module
from tensorforge.experimental import native_metrics as metrics_module
from tensorforge.experimental import native_sampler as sampler_module

REPO_ROOT = Path(__file__).resolve().parent.parent
PACKAGE = "src/tensorforge/experimental"

needs_backend = pytest.mark.skipif(
    not cpp.is_available(),
    reason="experimental C++ backend not built; " + cpp.build_instructions(),
)

# ---------------------------------------------------------------------------
# The boundary K5 proves did not move. Every value is written here
# **independently** of the module that defines it, so a silent change fails
# an exact equality rather than agreeing with whatever the source now says.
# ---------------------------------------------------------------------------
FLOATING_DTYPES = ("float64", "float32")
INDEX_DTYPES = ("int64",)
INDEX_DTYPE = "int64"

CHECKPOINT_FORMAT = "tensorforge.native_checkpoint"
CHECKPOINT_VERSION = 3
CHECKPOINT_VERSIONS = (1, 2, 3)
FLOAT64_ONLY_VERSIONS = (1, 2)
OPTIMIZER_STATE_VERSION = 1
LOADER_FORMAT = "tensorforge.native_data_loader"
LOADER_VERSION = 1
LOADER_VERSIONS = (1,)
SAMPLER_FORMAT = "tensorforge.native_sampler"
SAMPLER_VERSION = 1
SAMPLER_VERSIONS = (1,)

# The inventories K5 moves by exactly nothing (design §33, K5's row).
EXPORT_COUNT = 56
CTEST_COUNT = 27
EXPERIMENTAL_EXPORTS = 25
EXAMPLE_COUNT = 16
BENCHMARK_COUNT = 9

MANIFEST_ROOT_KEYS = {"format", "format_version", "model", "optimizer",
                      "generators", "metadata"}
MODEL_ENTRY_KEYS = {"array", "shape", "dtype", "device"}

# The raw-bit view each floating dtype is compared through. Never a
# tolerance for a bit-identical claim, and each dtype only against itself.
BIT_VIEW = {"float64": np.uint64, "float32": np.uint32}

# ---------------------------------------------------------------------------
# The deterministic workload. Small enough for a routine test run and large
# enough to satisfy every structural requirement K5's proof has:
#
#   * ``TOTAL_STEPS`` > 1, so more than one optimizer update happens;
#   * ``SPLIT_STEP`` genuinely mid-epoch — ``0 < 2 < 3`` and
#     ``2 % 3 != 0`` — so the interruption is not an epoch boundary in
#     disguise and batches are still owed;
#   * the run crosses an epoch boundary (5 steps over 3 batches per epoch),
#     so the canonical ``iter(loader)`` continuation is exercised;
#   * ``EVAL_STEPS`` places at least one evaluation strictly before the
#     checkpoint and at least one strictly after the resume;
#   * ``BATCH > CLASSES``, so a batch of predicted classes is guaranteed by
#     pigeonhole to contain duplicates without the fixture distorting the
#     model to arrange them.
# ---------------------------------------------------------------------------
FEATURES = 4
HIDDEN = 5
CLASSES = 3
SAMPLES = 15
BATCH = 5
BATCHES_PER_EPOCH = 3            # SAMPLES / BATCH, drop_last=False
TOTAL_STEPS = 5
SPLIT_STEP = 2
EVAL_STEPS = (0, 1, 3, 4)
CLASS_AXIS = 1                   # the model's class dimension
LR = 0.01
SEED = 7

assert BATCH > CLASSES
assert SAMPLES == BATCH * BATCHES_PER_EPOCH      # every batch is full
assert 0 < SPLIT_STEP < BATCHES_PER_EPOCH
assert SPLIT_STEP % BATCHES_PER_EPOCH != 0
assert TOTAL_STEPS > BATCHES_PER_EPOCH
assert any(step < SPLIT_STEP for step in EVAL_STEPS)
assert any(step >= SPLIT_STEP for step in EVAL_STEPS)


# ===========================================================================
# Instruments
# ===========================================================================

@contextlib.contextmanager
def live_storage_baseline(settle=False):
    """Assert native live storage returns **exactly** to baseline.

    Installed outside ``monkeypatch`` on purpose (design §30.2): a mid-test
    ``undo()`` must not be able to disarm the tracker that proves a
    scenario leaked nothing.

    ``settle`` is the established Phase-J allowance and it is narrow.
    An autograd graph's internal nodes are framework-owned and form
    reference cycles between a node and its backward closure, so they are
    reclaimed rather than closed by a caller — a collection **settles**
    the count for a block that trained. It is never the proof that
    anything was released: every object *this module* owns is closed
    explicitly first, and
    ``test_a_collection_does_not_launder_a_real_leak`` proves a retained,
    unclosed tensor is still reported after one. Blocks that build no
    graph use the strict default."""
    live = {}
    original_init = cpp.NativeStorage.__init__
    original_close = cpp.NativeStorage.close

    def counting_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        live[id(self)] = self.dtype

    def counting_close(self):
        original_close(self)
        live.pop(id(self), None)

    cpp.NativeStorage.__init__ = counting_init
    cpp.NativeStorage.close = counting_close
    try:
        yield live
        if settle:
            gc.collect()
        assert not live, f"{len(live)} native storages were never closed"
    finally:
        cpp.NativeStorage.__init__ = original_init
        cpp.NativeStorage.close = original_close


@contextlib.contextmanager
def allocated_dtypes():
    """Record the dtype of every ``NativeStorage`` the block allocates.

    "The failed load allocated nothing at ``int64``" and "the metric built
    no native index tensor" are claims about what was *created*, which a
    live-storage baseline (which only proves what was *released*) cannot
    answer. Installed outside ``monkeypatch`` for the same reason."""
    seen = []
    original_init = cpp.NativeStorage.__init__

    def recording_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        seen.append(self.dtype)

    cpp.NativeStorage.__init__ = recording_init
    try:
        yield seen
    finally:
        cpp.NativeStorage.__init__ = original_init


def bits(array):
    """One host array's raw IEEE-754 object representations.

    The **only** way this module compares a floating value when the claim
    is bit-identity. ``==`` calls two NaNs unequal, calls ``+0.0`` and
    ``-0.0`` equal, and cannot see a NaN payload at all, so it can prove
    none of what an exact resume promises (design §29.6)."""
    array = np.ascontiguousarray(array)
    return tuple(array.view(BIT_VIEW[str(array.dtype)]).ravel().tolist())


def code_names(relative):
    """Every identifier a module's **executable code** names.

    A source-text scan would be wrong: these modules explain at length what
    they deliberately do not do, so a docstring naming ``int64`` would
    satisfy or break a substring check that is supposed to be about
    behavior. Reading the AST asks the question that was meant, and
    keyword-argument names are read too."""
    tree = ast.parse((REPO_ROOT / relative).read_text(encoding="utf-8"))
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            names.add(node.id)
        elif isinstance(node, ast.Attribute):
            names.add(node.attr)
        elif isinstance(node, (ast.ClassDef, ast.FunctionDef,
                               ast.AsyncFunctionDef)):
            names.add(node.name)
        elif isinstance(node, ast.keyword) and node.arg:
            names.add(node.arg)
        elif isinstance(node, ast.arg):
            names.add(node.arg)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                names.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            names.update(alias.name for alias in node.names)
    return names


def source_exports():
    names = set()
    for path in sorted((REPO_ROOT / "cpp" / "src").glob("*.cpp")):
        names.update(re.findall(r"TF_EXPORT[^;{]*?\b(tf_[a-z0-9_]+)\s*\(",
                                path.read_text(encoding="utf-8"), re.S))
    return names


def parameter_names(callable_object):
    return tuple(inspect.signature(callable_object).parameters)


# ===========================================================================
# The archive helpers — the mutation precedent this repository already uses
# ===========================================================================

def manifest_of(path):
    with np.load(path, allow_pickle=False) as archive:
        return json.loads(archive["manifest"].tobytes().decode("utf-8"))


def arrays_of(path):
    with np.load(path, allow_pickle=False) as archive:
        return {name: archive[name] for name in archive.files}


def rewrite(source, target, mutate):
    """A controlled malformed copy of a **real** archive.

    Every byte comes from a genuine ``save_native_checkpoint`` result; only
    what ``mutate`` changes differs, so a rejection below is about that one
    field rather than about a hand-built file the parser never sees in
    practice. No archive format is invented and no parser is bypassed."""
    arrays = arrays_of(source)
    manifest = json.loads(arrays.pop("manifest").tobytes().decode("utf-8"))
    result = mutate(manifest)
    manifest = result if result is not None else manifest
    arrays["manifest"] = np.frombuffer(
        json.dumps(manifest).encode("utf-8"), dtype=np.uint8
    )
    with open(target, "wb") as handle:
        np.savez(handle, **arrays)
    return str(target)


def downgrade_moments(manifest):
    """Version 1 and 2 wrote ``m``/``v`` as bare lists of archive names,
    with shape and dtype implied positionally by ``parameters``. Applied
    only where a historical archive is being constructed."""
    section = manifest.get("optimizer")
    if isinstance(section, dict) and section.get("type") == "NativeAdam":
        for label in ("m", "v"):
            section[label] = [
                entry["array"] if isinstance(entry, dict) else entry
                for entry in section[label]
            ]
    return manifest


# ===========================================================================
# The model, the graph, and the deterministic workload
# ===========================================================================

class CompatibilityModel(NativeModule):
    """A real native classifier with every state family non-trivial:
    trainable parameters, persistent BatchNorm buffers, and two registered
    generator streams (one **shared** between two dropouts, one owned), so
    an exact-resume claim is not recoverable from any single family."""

    def __init__(self, *, dtype=None, in_seed=1, out_seed=2,
                 shared_seed=101, own_seed=202):
        super().__init__()
        self.linear_in = NativeLinear(FEATURES, HIDDEN, seed=in_seed,
                                      dtype=dtype)
        self.norm = NativeBatchNorm1d(HIDDEN, dtype=dtype)
        self.relu = NativeReLU()
        shared = NativeGenerator(shared_seed)
        self.drop_a = NativeDropout(0.25, generator=shared)
        self.drop_b = NativeDropout(0.25, generator=shared)
        self.drop_c = NativeDropout(0.5, seed=own_seed)
        self.linear_out = NativeLinear(HIDDEN, CLASSES, seed=out_seed,
                                       dtype=dtype)

    def forward(self, x):
        hidden = self.relu(self.norm(self.linear_in(x)))
        hidden = self.drop_c(self.drop_b(self.drop_a(hidden)))
        return self.linear_out(hidden)


class Graph:
    """Model, optimizer, loss, dataset, sampler, and loader held together
    so a test can build two of them and close both explicitly. A test
    convenience; no production analogue exists or is implied."""

    __slots__ = ("model", "optimizer", "loss_fn", "dataset", "sampler",
                 "loader")

    def __init__(self, model, optimizer, loss_fn, dataset, sampler, loader):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.dataset = dataset
        self.sampler = sampler
        self.loader = loader


def host_arrays(samples=SAMPLES):
    """Deterministic source values — every row identifiable, so a batch can
    be checked against the indices that produced it."""
    values = (np.arange(samples * FEATURES, dtype=np.float64)
              .reshape(samples, FEATURES) % 7.0) - 3.0
    targets = np.arange(samples, dtype=np.int64) % CLASSES
    return values, targets


def build_graph(dtype=None, *, in_seed=1, out_seed=2, shared_seed=101,
                own_seed=202, lr=LR, batch_size=BATCH, shuffle=True,
                seed=SEED, drop_last=False):
    values, targets = host_arrays()
    dataset = NativeTensorDataset(values, targets, dtype=dtype)
    sampler = NativeBatchSampler(dataset, batch_size=batch_size,
                                 shuffle=shuffle, seed=seed,
                                 drop_last=drop_last)
    loader = NativeDataLoader(sampler)
    model = CompatibilityModel(dtype=dtype, in_seed=in_seed,
                               out_seed=out_seed, shared_seed=shared_seed,
                               own_seed=own_seed)
    optimizer = NativeAdam(model.parameters(), lr=lr)
    return Graph(model, optimizer, NativeCrossEntropyLoss(), dataset,
                 sampler, loader)


def close_graph(graph):
    """Explicit cleanup, in the established order: the loader (and so its
    iterator), then the optimizer's moments, then the dataset's host
    snapshots, then every unique parameter and persistent buffer. Nothing
    here relies on garbage collection.

    Gradients are released explicitly first: a parameter's ``close()``
    releases the parameter's own storage and not the separate tensor its
    ``.grad`` slot holds, so a run that ended between ``backward()`` and
    ``zero_grad()`` would otherwise leave one live storage per parameter."""
    graph.loader.close()
    graph.optimizer.close()
    graph.dataset.close()
    seen = set()
    for _, tensor in list(graph.model.named_parameters()) + \
            list(graph.model.named_buffers()):
        if id(tensor) not in seen:
            seen.add(id(tensor))
            if getattr(tensor, "grad", None) is not None:
                tensor.zero_grad()
            tensor.close()


class BatchStream:
    """One epoch at a time, exactly as the Phase-J caller contract requires:
    on ``StopIteration`` call ``iter(loader)`` again and continue from the
    canonical next-epoch position. The sampler is never reset and neither
    ``epoch`` nor ``cursor`` is ever touched."""

    def __init__(self, loader):
        self._loader = loader
        self._iterator = iter(loader)

    def next_batch(self):
        for _ in range(2):
            try:
                return next(self._iterator)
            except StopIteration:
                self._iterator.close()
                self._iterator = iter(self._loader)
        raise AssertionError("a fresh epoch delivered no batch")

    def close(self):
        self._iterator.close()


def evaluation_record(logits, class_axis=CLASS_AXIS):
    """The K3 + K4 evaluation path, exactly as design §32's K5 row states
    it, and **outside** the differentiable loss path.

    ``argmax`` is taken from the live (gradient-tracking) logits, because
    §17.9 promises the result is a plain leaf even then; ``index_select``
    is taken from a **detached** floating source, because §18.9 rejects a
    ``requires_grad=True`` source rather than detaching it silently.

    ``index_select`` selects **the same supplied index vector along one
    axis for every outer slice** — it is not a per-row gather — so for a
    batch of ``B`` examples the result is ``(B, B)`` and the *diagonal* is
    each example's own predicted-class logit. That is verified on the host
    here rather than asserted in prose. Every native temporary is closed
    explicitly."""
    predictions = logits.argmax(axis=class_axis)
    try:
        assert predictions.dtype == INDEX_DTYPE
        assert predictions.shape == (logits.shape[0],)
        assert predictions.requires_grad is False
        assert predictions.is_leaf is True
        assert predictions.grad is None
        index_values = tuple(predictions.tolist())

        detached = logits.detach()
        try:
            assert detached.requires_grad is False
            selected = detached.index_select(class_axis, predictions)
            try:
                assert selected.dtype == logits.dtype
                assert selected.shape == (logits.shape[0], len(index_values))
                assert selected.requires_grad is False
                assert selected.is_leaf is True
                host_selected = selected.to_numpy()
                host_logits = logits.to_numpy()
                # The diagonal is each example's predicted-class logit...
                diagonal = np.ascontiguousarray(np.diagonal(host_selected))
                expected = np.ascontiguousarray(
                    host_logits[np.arange(len(index_values)), list(index_values)]
                )
                assert bits(diagonal) == bits(expected)
                # ...and every column j is the whole predicted-class column
                # indices[j], duplicates and order preserved exactly.
                for position, index in enumerate(index_values):
                    assert bits(np.ascontiguousarray(
                        host_selected[:, position])) == \
                        bits(np.ascontiguousarray(host_logits[:, index]))
                return (index_values, bits(host_selected),
                        tuple(selected.shape))
            finally:
                selected.close()
        finally:
            detached.close()
    finally:
        predictions.close()


def gradient_bits(model):
    """Every parameter gradient as raw bits, by canonical name."""
    record = {}
    for name, parameter in model.named_parameters():
        gradient = parameter.grad
        record[name] = None if gradient is None else bits(gradient.to_numpy())
    return record


def train_step(graph, stream, *, evaluate):
    """One genuine training step, with the evaluation indexing taken from
    the step's own logits when ``evaluate`` is set.

    Reusing the step's logits is deliberate: it is what makes the
    observational control in §6 a control. A second forward pass would
    consume dropout draws and move the batch-norm running statistics, so
    "the evaluation changed nothing" would be measuring the extra forward
    rather than ``argmax`` and ``index_select``."""
    indices = graph.sampler.next_batch_indices()
    features, targets = stream.next_batch()
    try:
        assert isinstance(features, NativeTensor)
        assert isinstance(targets, np.ndarray) and targets.dtype == np.int64
        logits = graph.model(features)
        try:
            logits_bits = bits(logits.to_numpy())
            evaluation = evaluation_record(logits) if evaluate else None
            loss = graph.loss_fn(logits, targets)
            try:
                loss_bits = bits(loss.to_numpy())
                loss.backward()
            finally:
                loss.close()
            gradients = gradient_bits(graph.model)
            graph.optimizer.step()
            graph.optimizer.zero_grad()
        finally:
            logits.close()
    finally:
        features.close()
    return {
        "indices": indices,
        "targets": tuple(targets.tolist()),
        "logits": logits_bits,
        "loss": loss_bits,
        "gradients": gradients,
        "evaluation": evaluation,
    }


def run_steps(graph, stream, steps, *, evaluate_at=EVAL_STEPS,
              first_step=0):
    return [train_step(graph, stream, evaluate=(first_step + offset)
                       in evaluate_at)
            for offset in range(steps)]


def trainable_fingerprint(graph):
    """Everything an exact resume must reproduce, as one comparable value.

    Raw IEEE-754 bit patterns throughout, each dtype compared only against
    itself; no tolerance, no ``allclose``, and nothing process-local."""
    model, optimizer = graph.model, graph.optimizer
    snapshot = model.state_dict()
    try:
        tensors = {
            name: (tensor.dtype, tuple(tensor.shape), tensor.device,
                   bits(tensor.to_numpy()))
            for name, tensor in snapshot.items()
        }
        keys = list(snapshot)
    finally:
        for tensor in snapshot.values():
            tensor.close()
    state = optimizer.state_dict()
    try:
        moments = {
            label: tuple((entry.dtype, tuple(entry.shape), entry.device,
                          bits(entry.to_numpy()))
                         for entry in state[label])
            for label in ("m", "v")
        }
        parameters = tuple((tuple(entry["shape"]), entry["dtype"],
                            entry["device"])
                           for entry in state["parameters"])
        scalars = (state["format_version"], state["optimizer"],
                   optimizer.lr, tuple(optimizer.betas), optimizer.eps,
                   tuple(optimizer.step_counts))
    finally:
        for label in ("m", "v"):
            for entry in state[label]:
                entry.close()
    return {
        "keys": keys,
        "tensors": tensors,
        "moments": moments,
        "parameters": parameters,
        "scalars": scalars,
        "generators": model.generator_state_dict(),
        "generator_names": [name for name, _ in model.named_generators()],
        "loader": graph.loader.state_dict(),
        "sampler": graph.sampler.state_dict(),
    }


def training_metadata(next_step, loader_state):
    """The recommended caller convention, and nothing more: no production
    constant spells any of these three names."""
    return {"training": {"next_step": next_step,
                         "data_loader": loader_state}}


# ===========================================================================
# 0. The instruments can fail
# ===========================================================================

def test_the_bit_view_really_distinguishes_values():
    left = np.array([1.0, 2.0], dtype=np.float64)
    right = np.array([1.0, np.nextafter(2.0, 3.0)], dtype=np.float64)
    assert bits(left) != bits(right)
    assert bits(left) == bits(left.copy())
    assert bits(np.array([-0.0])) != bits(np.array([0.0]))
    narrow = np.array([1.0, 2.0], dtype=np.float32)
    assert bits(narrow) == bits(narrow.copy())
    assert bits(narrow) != bits(np.array([1.0, np.float32(2.0000005)],
                                         dtype=np.float32))


def test_the_code_scanner_reads_code_and_not_prose():
    """The control every absence scan below depends on."""
    source = ('"""a docstring naming int64 and gather."""\n'
              'def f(x):\n'
              '    return g(x, dtype="int64", trusted=True)\n')
    tree = ast.parse(source)
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            names.add(node.id)
        elif isinstance(node, ast.keyword) and node.arg:
            names.add(node.arg)
    assert "gather" not in names, "prose leaked into the code view"
    assert {"g", "x", "dtype", "trusted"} <= names
    # ...and the real reader finds the real things.
    checkpoint = code_names(f"{PACKAGE}/native_checkpoint.py")
    assert {"save_native_checkpoint", "load_native_checkpoint",
            "_validated_entry_dtype"} <= checkpoint


def test_the_export_scanner_can_actually_fail():
    source = 'TF_EXPORT void tf_core_probe(const void* a) { use(a); }'
    pattern = r"TF_EXPORT[^;{]*?\b(tf_[a-z0-9_]+)\s*\("
    assert re.findall(pattern, source, re.S) == ["tf_core_probe"]
    assert re.findall(pattern, "void tf_core_probe(void);", re.S) == []


@needs_backend
def test_the_live_storage_tracker_can_actually_fail():
    """"Nothing leaked" means something only when a leak is detectable.

    The leaked tensor is held by a strong reference for the duration, so
    the control proves the *tracker* fires rather than proving that
    reclamation happened to be prompt."""
    retained = []
    with pytest.raises(AssertionError, match="never closed"):
        with live_storage_baseline():
            retained.append(NativeTensor.from_array(np.array([1.0, 2.0])))
    retained[0].close()
    with live_storage_baseline():
        tensor = NativeTensor.from_array(np.array([1.0, 2.0]))
        tensor.close()


@needs_backend
def test_a_collection_does_not_launder_a_real_leak():
    """The control the ``settle=True`` allowance needs: a collection may
    settle a framework-owned graph node, and it must **not** be able to
    hide a tensor a test still holds and never closed."""
    retained = []
    with pytest.raises(AssertionError, match="never closed"):
        with live_storage_baseline(settle=True):
            retained.append(NativeTensor.from_array(np.array([1.0, 2.0])))
    retained[0].close()


@needs_backend
def test_the_allocation_dtype_recorder_can_actually_fail():
    with allocated_dtypes() as seen:
        floating = NativeTensor.from_array(np.array([1.0, 2.0]))
        index = NativeTensor.from_int64_array(np.array([1], dtype=np.int64))
        floating.close()
        index.close()
    assert "float64" in seen and INDEX_DTYPE in seen
    with allocated_dtypes() as quiet:
        pass
    assert quiet == []


@needs_backend
def test_the_trainable_fingerprint_notices_each_change_it_exists_for():
    """The non-vacuity control for §6's comparison: every component must be
    able to notice the change it is there for."""
    with live_storage_baseline(settle=True):
        graph = build_graph()
        try:
            before = trainable_fingerprint(graph)
            assert trainable_fingerprint(graph) == before      # pure
            stream = BatchStream(graph.loader)
            try:
                run_steps(graph, stream, 1, evaluate_at=())
            finally:
                stream.close()
            after = trainable_fingerprint(graph)
            assert after != before
            # ...and each family moved on its own, not as one lump.
            for family in ("tensors", "moments", "scalars", "generators",
                           "loader", "sampler"):
                assert after[family] != before[family], family
        finally:
            close_graph(graph)


# ===========================================================================
# 1. Group A — checkpoint schema, and the archive that cannot declare int64
# ===========================================================================

def test_the_checkpoint_constants_are_exactly_what_phase_j_left():
    assert native_checkpoint._FORMAT == CHECKPOINT_FORMAT
    assert native_checkpoint._FORMAT_VERSION == CHECKPOINT_VERSION
    assert native_checkpoint._SUPPORTED_FORMAT_VERSIONS == CHECKPOINT_VERSIONS
    assert native_checkpoint._FLOAT64_ONLY_VERSIONS == FLOAT64_ONLY_VERSIONS
    assert 4 not in native_checkpoint._SUPPORTED_FORMAT_VERSIONS
    assert native_optimizer_state.FORMAT_VERSION == OPTIMIZER_STATE_VERSION
    assert native_checkpoint._MANIFEST_KEYS == MANIFEST_ROOT_KEYS
    assert native_checkpoint._MODEL_ENTRY_KEYS == MODEL_ENTRY_KEYS


def test_no_version_four_constant_is_written_reserved_or_accepted():
    """Not merely "4 is not accepted": no module-level version constant in
    the checkpoint module holds it, and no reserved future-version name
    exists in its executable code."""
    module_versions = {
        name: value for name, value in vars(native_checkpoint).items()
        if "VERSION" in name.upper() and isinstance(value, (int, tuple))
    }
    assert module_versions, "the version-constant scan found nothing"
    for name, value in module_versions.items():
        if isinstance(value, int):
            assert value != 4, name
        else:
            assert 4 not in value, name
    names = code_names(f"{PACKAGE}/native_checkpoint.py")
    for reserved in ("_FORMAT_VERSION_4", "_NEXT_FORMAT_VERSION",
                     "_FUTURE_FORMAT_VERSION", "_RESERVED_VERSIONS",
                     "_INTEGER_SECTION", "_INDEX_SECTION"):
        assert reserved not in names, reserved


@needs_backend
@pytest.mark.parametrize("dtype", FLOATING_DTYPES)
def test_a_valid_archive_declares_floating_entries_only(tmp_path, dtype):
    """Both halves of claim 1's first side: every entry a real save writes
    is floating, at both widths, and the archive round-trips exactly."""
    with live_storage_baseline(settle=True):
        source = build_graph(dtype)
        target = build_graph(dtype, in_seed=91, out_seed=92, shared_seed=93,
                             own_seed=94, lr=0.5)
        try:
            stream = BatchStream(source.loader)
            try:
                run_steps(source, stream, 2, evaluate_at=())
            finally:
                stream.close()
            path = str(tmp_path / f"valid_{dtype}.npz")
            loader_state = source.loader.state_dict()
            save_native_checkpoint(path, source.model,
                                   optimizer=source.optimizer,
                                   metadata=training_metadata(2, loader_state))

            manifest = manifest_of(path)
            assert manifest["format"] == CHECKPOINT_FORMAT
            assert manifest["format_version"] == CHECKPOINT_VERSION
            assert set(manifest) == MANIFEST_ROOT_KEYS
            # Every declared dtype is floating — parameters, persistent
            # buffers, and both optimizer moment families alike.
            declared = [entry["dtype"]
                        for entry in manifest["model"]["entries"].values()]
            declared += [entry["dtype"]
                         for entry in manifest["optimizer"]["parameters"]]
            for label in ("m", "v"):
                declared += [entry["dtype"]
                             for entry in manifest["optimizer"][label]]
            assert declared, "the dtype sweep found no entries"
            assert set(declared) == {dtype}
            assert INDEX_DTYPE not in declared
            # The persistent buffers really are in there, so "every entry"
            # is not a claim about parameters alone.
            assert {"norm.running_mean", "norm.running_var"} <= \
                set(manifest["model"]["entries"])
            assert manifest["optimizer"]["state_format_version"] == \
                OPTIMIZER_STATE_VERSION
            # No integer field, section, reservation, or placeholder — at
            # the root or anywhere below it.
            flat = json.dumps(manifest)
            assert INDEX_DTYPE not in flat
            assert not re.search(r'"(int|integer|index)[a-z_]*"\s*:', flat)
            # ...and every stored array is at the declared width.
            for name, array in arrays_of(path).items():
                if name != "manifest":
                    assert array.dtype == np.dtype(dtype), name

            expected = trainable_fingerprint(source)
            assert trainable_fingerprint(target) != expected
            metadata = load_native_checkpoint(path, target.model,
                                              optimizer=target.optimizer)
            assert metadata["training"]["data_loader"] == loader_state
            restored = trainable_fingerprint(target)
            for family in ("tensors", "moments", "parameters", "generators"):
                assert restored[family] == expected[family], family
        finally:
            close_graph(target)
            close_graph(source)


@needs_backend
@pytest.mark.parametrize("role", ["parameter", "persistent_buffer",
                                  "optimizer_moment", "optimizer_parameter"])
def test_an_archive_that_declares_int64_is_rejected_without_mutation(
        tmp_path, role):
    """Claim 1's second side, at every entry role the format distinguishes.

    One field changes and nothing else. The load must reject **before**
    publishing any model, buffer, or optimizer mutation, must leave the
    destination byte-identical, and must not allocate an ``int64``
    checkpoint tensor on the way."""
    def mutate(manifest):
        if role == "parameter":
            manifest["model"]["entries"]["linear_in.weight"]["dtype"] = \
                INDEX_DTYPE
        elif role == "persistent_buffer":
            manifest["model"]["entries"]["norm.running_mean"]["dtype"] = \
                INDEX_DTYPE
        elif role == "optimizer_moment":
            manifest["optimizer"]["m"][0]["dtype"] = INDEX_DTYPE
        else:
            manifest["optimizer"]["parameters"][0]["dtype"] = INDEX_DTYPE
        return manifest

    with live_storage_baseline(settle=True):
        source = build_graph()
        target = build_graph(in_seed=91, out_seed=92, shared_seed=93,
                             own_seed=94, lr=0.5)
        try:
            stream = BatchStream(source.loader)
            try:
                run_steps(source, stream, 2, evaluate_at=())
            finally:
                stream.close()
            good = str(tmp_path / f"good_{role}.npz")
            save_native_checkpoint(good, source.model,
                                   optimizer=source.optimizer)
            bad = rewrite(good, tmp_path / f"bad_{role}.npz", mutate)

            before = trainable_fingerprint(target)
            identities = (
                [id(p) for _, p in target.model.named_parameters()],
                [id(b) for _, b in target.model.named_buffers()],
                [id(g) for g in target.model.generators()],
            )
            with allocated_dtypes() as seen:
                with pytest.raises(ValueError) as error:
                    load_native_checkpoint(bad, target.model,
                                           optimizer=target.optimizer)
            message = str(error.value)
            assert INDEX_DTYPE in message, message
            assert "dtype" in message, message
            # **Which** authority rejected is part of the claim: the three
            # entry roles the format describes with a declared dtype go
            # through the checkpoint's own entry validator, while the
            # optimizer's parameter *metadata* is refused by the separate
            # in-memory schema validator as a mismatch against the live
            # parameter. Two independent layers, and neither may be removed
            # because the other exists.
            if role == "optimizer_parameter":
                assert "state['parameters'][0]['dtype']" in message, message
                assert "the stored parameter is" in message, message
            else:
                assert "may declare" in message, message
                assert "['dtype']" in message, message
            # The failed load allocated no int64 storage at all.
            assert INDEX_DTYPE not in seen, seen
            # ...and published nothing: values, versions, gradients,
            # moments, counters, generator states, and every object
            # identity are exactly as they were.
            assert trainable_fingerprint(target) == before
            assert (
                [id(p) for _, p in target.model.named_parameters()],
                [id(b) for _, b in target.model.named_buffers()],
                [id(g) for g in target.model.generators()],
            ) == identities

            # The control: the same archive **without** the mutation loads,
            # so the rejection is about the declared dtype and nothing else
            # — and it moves exactly the families the rejected load left
            # alone, rather than merely differing somewhere.
            load_native_checkpoint(good, target.model,
                                   optimizer=target.optimizer)
            after = trainable_fingerprint(target)
            for family in ("tensors", "moments", "generators"):
                assert after[family] != before[family], family
        finally:
            close_graph(target)
            close_graph(source)


def test_the_entry_dtype_validator_refuses_int64_at_every_version():
    """The layer that makes the whole-archive rejection above structural
    rather than incidental, asserted at each accepted version."""
    for version in CHECKPOINT_VERSIONS:
        with pytest.raises(ValueError, match=INDEX_DTYPE):
            native_checkpoint._validated_entry_dtype(
                INDEX_DTYPE, version, "manifest['model']['entries']['w']",
                "load_native_checkpoint")
    # The control: the dtypes an archive *may* declare still pass.
    assert native_checkpoint._validated_entry_dtype(
        "float64", CHECKPOINT_VERSION, "e", "w") == "float64"
    assert native_checkpoint._validated_entry_dtype(
        "float32", CHECKPOINT_VERSION, "e", "w") == "float32"


@needs_backend
def test_historical_versions_stay_historical_and_float64_only(tmp_path):
    """Versions 1 and 2 remain accepted through their established legacy
    rules, keep their float64-only interpretation, cannot declare
    ``int64``, and are not reinterpreted; version 3 is not silently widened
    and version 4 is refused."""
    with live_storage_baseline(settle=True):
        source = build_graph()
        target = build_graph(in_seed=91, out_seed=92, shared_seed=93,
                             own_seed=94, lr=0.5)
        try:
            base = str(tmp_path / "v3.npz")
            save_native_checkpoint(base, source.model,
                                   optimizer=source.optimizer)
            expected = trainable_fingerprint(source)

            def to_version(version):
                def mutate(manifest):
                    manifest["format_version"] = version
                    downgrade_moments(manifest)
                    if version < 2:
                        manifest.pop("generators", None)
                    return manifest
                return mutate

            # A version-1 archive has no generator section, so the model it
            # restores into must have none either — the legacy rule,
            # unchanged. The version-2 and version-3 legs use the full
            # graph above.
            for version in (2, 3):
                path = rewrite(base, tmp_path / f"as_v{version}.npz",
                               to_version(version) if version != 3
                               else (lambda manifest: manifest))
                before = trainable_fingerprint(target)
                assert before != expected
                load_native_checkpoint(path, target.model,
                                       optimizer=target.optimizer)
                restored = trainable_fingerprint(target)
                for family in ("tensors", "moments", "parameters"):
                    assert restored[family] == expected[family], version
                # Put the destination back to "wrong" for the next leg.
                target_stream = BatchStream(target.loader)
                try:
                    run_steps(target, target_stream, 1, evaluate_at=())
                finally:
                    target_stream.close()

            # ...and neither historical version may declare anything but
            # float64 — a float32 entry and an int64 entry are refused with
            # different reasons, which is what makes them two rules.
            for version in FLOAT64_ONLY_VERSIONS:
                with pytest.raises(ValueError, match="float64 only"):
                    native_checkpoint._validated_entry_dtype(
                        "float32", version, "e", "w")
                with pytest.raises(ValueError, match=INDEX_DTYPE):
                    native_checkpoint._validated_entry_dtype(
                        INDEX_DTYPE, version, "e", "w")

            # Version 4 rejects through the established path, naming the
            # accepted set, and so do the malformed spellings.
            for bad_version in (4, 5, 0, -1):
                path = rewrite(base, tmp_path / f"v{bad_version}.npz",
                               lambda manifest, v=bad_version:
                               manifest.update(format_version=v) or manifest)
                with pytest.raises(ValueError, match=r"format_version"):
                    load_native_checkpoint(path, target.model,
                                           optimizer=target.optimizer)
            for bad_version in ("3", 3.0, True, None):
                path = rewrite(base, tmp_path / "vbad.npz",
                               lambda manifest, v=bad_version:
                               manifest.update(format_version=v) or manifest)
                with pytest.raises((TypeError, ValueError),
                                   match=r"format_version"):
                    load_native_checkpoint(path, target.model,
                                           optimizer=target.optimizer)
        finally:
            close_graph(target)
            close_graph(source)


@needs_backend
def test_a_version_one_archive_still_loads_under_its_legacy_rules(tmp_path):
    """The oldest format, exercised on the shape it was written for: no
    generator section at all, bare-list optimizer moments, float64 only —
    and it still cannot declare ``int64``."""
    with live_storage_baseline(settle=True):
        source = NativeLinear(FEATURES, CLASSES, seed=1)
        target = NativeLinear(FEATURES, CLASSES, seed=42)
        source_optimizer = NativeAdam(source.parameters(), lr=LR)
        target_optimizer = NativeAdam(target.parameters(), lr=0.5)
        try:
            probe = NativeTensor.from_array(
                np.arange(FEATURES, dtype=np.float64).reshape(1, FEATURES))
            try:
                out = source(probe)
                try:
                    total = out.sum()
                    try:
                        total.backward()
                    finally:
                        total.close()
                finally:
                    out.close()
            finally:
                probe.close()
            source_optimizer.step()
            source_optimizer.zero_grad()

            base = str(tmp_path / "v3_linear.npz")
            save_native_checkpoint(base, source, optimizer=source_optimizer)
            assert manifest_of(base)["generators"] is None

            def to_version_one(manifest):
                manifest["format_version"] = 1
                downgrade_moments(manifest)
                manifest.pop("generators", None)
                return manifest

            path = rewrite(base, tmp_path / "as_v1.npz", to_version_one)
            manifest = manifest_of(path)
            assert manifest["format_version"] == 1
            assert "generators" not in manifest
            assert all(entry["dtype"] == "float64"
                       for entry in manifest["model"]["entries"].values())

            expected = {name: bits(tensor.to_numpy())
                        for name, tensor in source.named_parameters()}
            before = {name: bits(tensor.to_numpy())
                      for name, tensor in target.named_parameters()}
            assert before != expected
            load_native_checkpoint(path, target, optimizer=target_optimizer)
            assert {name: bits(tensor.to_numpy())
                    for name, tensor in target.named_parameters()} == expected

            # ...and a version-1 archive that declares int64 is refused,
            # with the destination left exactly as the successful load
            # above left it.
            def to_int64_v1(manifest):
                to_version_one(manifest)
                manifest["model"]["entries"]["weight"]["dtype"] = INDEX_DTYPE
                return manifest

            bad = rewrite(base, tmp_path / "as_v1_int64.npz", to_int64_v1)
            with allocated_dtypes() as seen:
                with pytest.raises(ValueError, match=INDEX_DTYPE):
                    load_native_checkpoint(bad, target,
                                           optimizer=target_optimizer)
            assert INDEX_DTYPE not in seen, seen
            assert {name: bits(tensor.to_numpy())
                    for name, tensor in target.named_parameters()} == expected
        finally:
            source_optimizer.close()
            target_optimizer.close()
            for module in (source, target):
                for tensor in module.parameters():
                    if tensor.grad is not None:
                        tensor.zero_grad()
                    tensor.close()


# ===========================================================================
# 2. Group B — parameter, buffer, and optimizer barriers, end to end
# ===========================================================================

@contextlib.contextmanager
def index_tensor(values=(0, 1, 2)):
    """A real public ``int64`` tensor through the one public door."""
    tensor = NativeTensor.from_int64_array(np.asarray(values, dtype=np.int64))
    try:
        yield tensor
    finally:
        tensor.close()


@contextlib.contextmanager
def index_shaped_parameter(tensor):
    """A genuine ``NativeParameter`` instance carrying a real ``int64``
    core — the only way to drive a surface that takes a *parameter*.

    It has to be built through ``__new__``, and that is the point rather
    than a shortcut: the public constructor refuses an index tensor, which
    is the barrier those surfaces sit behind, so any test that reaches them
    through ``NativeParameter(tensor)`` never reaches them at all. Every
    field below is what ``__init__`` would have set, except
    ``_owns_core=False``: the core belongs to the caller's ``tensor``,
    which closes it, so this object is never a second owner and closes
    nothing.

    Disarmed on exit, in that order — ``_closed`` first, so the ``__del__``
    fallback is already a no-op by the time the core reference is dropped,
    and the caller's tensor is left untouched and still open."""
    fake = NativeParameter.__new__(NativeParameter)
    fake._core = tensor._core
    fake._owns_core = False
    fake._closed = False
    fake._requires_grad = True
    fake._grad = None
    fake._parents = ()
    fake._backward = None
    fake._op = ""
    fake._is_leaf = True
    fake._graph_freed = False
    fake._expected_versions = ()
    fake._graph_resources = ()
    fake._version = 0
    assert isinstance(fake, NativeParameter)
    assert type(fake) is NativeParameter
    assert fake.dtype == INDEX_DTYPE
    assert fake.owns_core is False
    try:
        yield fake
    finally:
        fake._closed = True
        fake._core = None
    # Disarmed, and the caller's tensor is exactly as it was handed over.
    assert fake.closed is True
    assert tensor.closed is False
    assert tensor.dtype == INDEX_DTYPE


@needs_backend
def test_the_parameter_constructor_refuses_a_real_index_tensor():
    """The **constructor** barrier, on its own and labelled as itself.

    This is `NativeParameter.__init__` → ``cpp._require_floating_dtype``
    and nothing else. It is deliberately *not* a claim about registration:
    the constructor raises before any assignment could run, so a test that
    wrote ``module.indices = NativeParameter(tensor)`` would be proving
    this line twice and the registration path zero times. Registration is
    driven for real in
    ``test_the_parameter_registration_paths_carry_no_second_dtype_authority``."""
    with live_storage_baseline():
        with index_tensor((0, 1)) as tensor:
            # Integers are compared by **exact equality**; ``bits()`` is the
            # floating instrument and is deliberately not applied here.
            before = tensor.tolist()
            with pytest.raises(ValueError, match=INDEX_DTYPE):
                NativeParameter(tensor)
            with pytest.raises(ValueError, match=INDEX_DTYPE):
                NativeParameter(tensor, dtype=INDEX_DTYPE)
            # A host ``int64`` array is refused at the same authority, so
            # the barrier is about the dtype rather than about the argument
            # being a tensor.
            with pytest.raises(ValueError, match=INDEX_DTYPE):
                NativeParameter(np.array([0, 1], dtype=np.int64),
                                dtype=INDEX_DTYPE)
            # The rejection consumed nothing: the operand is untouched and
            # still usable.
            assert tensor.closed is False
            assert tensor.tolist() == before == [0, 1]
            assert tensor.dtype == INDEX_DTYPE
        # The control: the floating half of the same constructor works.
        parameter = NativeParameter(np.array([0.0, 1.0]))
        try:
            assert parameter.dtype == "float64"
        finally:
            parameter.close()


@needs_backend
def test_the_parameter_registration_paths_carry_no_second_dtype_authority(
        tmp_path):
    """Both registration routes, **driven for real** — and the answer is
    recorded exactly as measured rather than as hoped.

    Neither ``NativeModule.__setattr__`` nor ``register_parameter``
    inspects a parameter's dtype: they delegate to
    ``NativeParameterRegistry.register``, which validates the *name* and
    the *type* and nothing else. Handed a genuine ``NativeParameter``
    instance that already carries an ``int64`` core, both routes register
    it. The parameter role therefore has **exactly one** authority —
    ``NativeParameter.__init__`` — and the registration paths are
    protected *by construction*, not by a second check.

    That is worth an executable proof in both directions, because it is
    the reason the constructor barrier is load-bearing: it is not one
    layer of several for this role, it is the layer. Nothing public can
    produce the object driven here (``__new__`` bypasses ``__init__``),
    so no public route reaches these registrations with an index tensor —
    which is what makes "no module holds an integer parameter" true.

    **What this proof found, and what was done about it.** Driving the
    registrations for real showed that the checkpoint *writer* trusted
    whatever dtype live state reported, so a forged registration produced
    an archive declaring an ``int64`` entry — one the loader then refused.
    That gap was a pre-existing defect in the writer, not something Phase K
    introduced, and it was repaired in a **separate** checkpoint-hardening
    change before this milestone: ``save_native_checkpoint`` now validates
    every persisted dtype through the same question its loader asks. This
    test asserts the repaired boundary, and
    ``tests/test_native_checkpoint.py`` owns that repair's own regression.

    Each route is driven, proved to have really registered (so the drive
    is not vacuous, and ``state_dict()`` exposing the ``int64`` entry is
    part of that evidence), proved unable to reach an archive,
    unregistered through ``del`` — the one undo that installs no ordinary
    attribute — and then the whole surrounding world is proved
    byte-identical to what it was before."""
    with live_storage_baseline():
        module = NativeModule()
        module.weight = NativeParameter(np.array([[1.5, -2.5], [0.25, 4.0]]))
        floating = NativeTensor.from_array(np.array([3.0, -1.0]))
        module.register_buffer("stat", floating, persistent=True)
        optimizer = NativeAdam([module.weight], lr=0.1)
        try:
            def world():
                """Everything a registration must not have disturbed."""
                state = module.state_dict()
                try:
                    entries = {name: (tensor.dtype, tuple(tensor.shape),
                                      bits(tensor.to_numpy()))
                               for name, tensor in state.items()}
                finally:
                    for value in state.values():
                        value.close()
                optimizer_state = optimizer.state_dict()
                try:
                    moments = {
                        label: tuple((entry.dtype, tuple(entry.shape),
                                      bits(entry.to_numpy()))
                                     for entry in optimizer_state[label])
                        for label in ("m", "v")
                    }
                    optimizer_record = (
                        optimizer_state["format_version"],
                        optimizer_state["optimizer"],
                        tuple((tuple(entry["shape"]), entry["dtype"],
                               entry["device"])
                              for entry in optimizer_state["parameters"]),
                        moments,
                        tuple(optimizer.step_counts),
                        tuple(id(p) for p in optimizer.parameters()),
                    )
                finally:
                    for label in ("m", "v"):
                        for entry in optimizer_state[label]:
                            entry.close()
                return {
                    "parameters": [(name, id(p))
                                   for name, p in module.named_parameters()],
                    "buffers": [(name, id(b))
                                for name, b in module.named_buffers()],
                    "attributes": sorted(module.__dict__),
                    "state": entries,
                    "weight": (bits(module.weight.to_numpy()),
                               module.weight.version),
                    "optimizer": optimizer_record,
                }

            before = world()
            assert world() == before          # pure
            assert "indices" not in module.__dict__
            assert "indices" not in dict(module.named_parameters())

            with index_tensor((0, 1)) as tensor:
                with index_shaped_parameter(tensor) as fake:
                    routes = (
                        ("attribute assignment",
                         lambda: module.__setattr__("indices", fake)),
                        ("register_parameter",
                         lambda: module.register_parameter("indices", fake)),
                    )
                    for label, register in routes:
                        register()
                        # Measured, not assumed: the route registered it,
                        # which is exactly what makes the drive non-vacuous
                        # — and what shows the dtype was never consulted.
                        assert dict(module.named_parameters())["indices"] \
                            is fake, label
                        assert module.indices is fake, label
                        # A registered parameter lives in the registry, not
                        # in ``__dict__`` — no ordinary attribute appeared.
                        assert "indices" not in module.__dict__, label
                        # ``state_dict()`` exposes it, which is the
                        # clearest possible evidence that the registration
                        # was real rather than silently discarded.
                        state = module.state_dict()
                        try:
                            assert state["indices"].dtype == INDEX_DTYPE, label
                        finally:
                            for value in state.values():
                                value.close()

                        # ...and the **archive** still cannot contain it,
                        # from this direction too: the checkpoint writer
                        # refuses the live state before any file exists.
                        # That authority was added by a separate
                        # checkpoint-hardening repair — this proof is what
                        # exposed its absence — and it is what makes "no
                        # archive can declare an int64 entry" true whatever
                        # hands the writer live state, rather than only for
                        # models the public API can build.
                        archive = str(tmp_path / f"forged_{label[:6]}.npz")
                        with allocated_dtypes() as seen:
                            with pytest.raises(ValueError) as error:
                                save_native_checkpoint(archive, module,
                                                       optimizer=optimizer)
                        message = str(error.value)
                        assert "save_native_checkpoint" in message, message
                        assert "indices" in message, message
                        assert INDEX_DTYPE in message, message
                        assert "floating" in message, message
                        assert not Path(archive).exists(), label
                        assert sorted(p.name for p in tmp_path.iterdir()) \
                            == [], label
                        # The refused save allocated no index storage of
                        # its own and left the operand open.
                        assert seen.count(INDEX_DTYPE) == 0, seen
                        assert tensor.closed is False
                        assert tensor.tolist() == [0, 1]

                        # ``del`` unregisters without installing the
                        # ``None`` attribute the explicit unregister leaves
                        # behind, so the world can return exactly.
                        del module.indices

                        assert "indices" not in \
                            dict(module.named_parameters()), label
                        assert "indices" not in module.__dict__, label
                        assert not hasattr(module, "indices"), label
                        assert world() == before, label

                    # ...and the surfaces that *do* carry their own dtype
                    # authority still refuse the very same object.
                    for optimizer_class in (NativeSGD, NativeAdam):
                        with pytest.raises(ValueError, match=INDEX_DTYPE):
                            optimizer_class([fake], lr=0.1)
                    assert world() == before

            # The non-vacuity control for ``world()`` itself: a real
            # registration of a *floating* parameter is visible to it.
            extra = NativeParameter(np.array([7.0]))
            try:
                module.extra = extra
                assert world() != before
                del module.extra
                assert world() == before
            finally:
                extra.close()

            # ...and the control for the save rejections: the same module
            # and optimizer, with no forgery registered, save and load,
            # and the archive declares floating entries only.
            valid = str(tmp_path / "valid.npz")
            save_native_checkpoint(valid, module, optimizer=optimizer)
            manifest = manifest_of(valid)
            declared = [entry["dtype"]
                        for entry in manifest["model"]["entries"].values()]
            declared += [entry["dtype"]
                         for entry in manifest["optimizer"]["parameters"]]
            for label in ("m", "v"):
                declared += [entry["dtype"]
                             for entry in manifest["optimizer"][label]]
            assert declared and set(declared) == {"float64"}
            assert INDEX_DTYPE not in json.dumps(manifest)
            load_native_checkpoint(valid, module, optimizer=optimizer)
        finally:
            optimizer.close()
            floating.close()
            module.weight.close()


@needs_backend
def test_no_state_owning_surface_accepts_a_real_index_tensor():
    """Every state-owning surface, driven with a tensor the public door
    produced, and each rejection followed by the proof it changed nothing.

    These barriers landed at K1 and are unit-proved there; what K5 adds is
    that K3's and K4's arrival did not open any of them."""
    with live_storage_baseline():
        module = NativeModule()
        module.weight = NativeParameter(np.array([[1.5, -2.5], [0.25, 4.0]]))
        floating = NativeTensor.from_array(np.array([3.0, -1.0]))
        module.register_buffer("stat", floating, persistent=True)
        optimizer = NativeSGD([module.weight], lr=0.1)
        try:
            def snapshot():
                state = module.state_dict()
                try:
                    keys = sorted(state)
                finally:
                    for value in state.values():
                        value.close()
                return (bits(module.weight.to_numpy()), module.weight.version,
                        bits(floating.to_numpy()),
                        sorted(name for name, _ in module.named_parameters()),
                        sorted(name for name, _ in module.named_buffers()),
                        tuple(id(p) for p in optimizer.parameters()), keys)
            before = snapshot()

            with index_tensor((0, 1)) as tensor:
                # 1. the parameter **constructor** — the one authority for
                #    the parameter role, and the only reason no module can
                #    hold an integer parameter (the registration paths
                #    themselves carry no second check, which
                #    test_the_parameter_registration_paths_carry_no_second_
                #    dtype_authority drives directly rather than implying).
                with pytest.raises(ValueError, match=INDEX_DTYPE):
                    NativeParameter(tensor)
                with pytest.raises(ValueError, match=INDEX_DTYPE):
                    NativeParameter(tensor, dtype=INDEX_DTYPE)
                # 2. both buffer kinds
                for persistent in (True, False):
                    with pytest.raises(ValueError, match=INDEX_DTYPE):
                        module.register_buffer("indices", tensor,
                                               persistent=persistent)
                # 3. both optimizers, through the per-parameter dtype check
                #    K1 added beside the pre-existing type check. Driving it
                #    needs a ``NativeParameter`` carrying the index tag,
                #    because the public constructor above already refuses to
                #    build one — which is the point.
                with index_shaped_parameter(tensor) as fake:
                    for optimizer_class in (NativeSGD, NativeAdam):
                        with pytest.raises(ValueError, match=INDEX_DTYPE):
                            optimizer_class([fake], lr=0.1)
                        # ...and the plain tensor is refused earlier still,
                        # by the type check that has always been there.
                        with pytest.raises(TypeError, match="NativeParameter"):
                            optimizer_class([tensor], lr=0.1)
                # 4. module state construction cannot contain it
                state = module.state_dict()
                try:
                    assert set(state) == {"weight", "stat"}
                    for value in state.values():
                        assert value.dtype in FLOATING_DTYPES
                finally:
                    for value in state.values():
                        value.close()

            assert snapshot() == before, "a rejection changed the world"
        finally:
            # ``NativeSGD`` owns no tensor state and so has no ``close()``;
            # ``close()`` exists exactly where something is owned.
            assert not hasattr(optimizer, "close")
            floating.close()
            module.weight.close()


@needs_backend
def test_optimizer_state_cannot_acquire_or_declare_an_index_tensor():
    """The state a live optimizer holds is floating at every position, its
    serialization declares floating only, and its version is unmoved."""
    with live_storage_baseline():
        parameter = NativeParameter(np.array([1.0, 2.0]))
        optimizer = NativeAdam([parameter], lr=0.1)
        try:
            # A real gradient through a real backward: d(sum(p * c))/dp = c.
            # No gradient is ever fabricated here.
            coefficients = NativeTensor.from_array(np.array([2.0, 3.0]))
            try:
                product = parameter.multiply(coefficients)
                try:
                    out = product.sum()
                    try:
                        out.backward()
                    finally:
                        out.close()
                finally:
                    product.close()
            finally:
                coefficients.close()
            optimizer.step()
            state = optimizer.state_dict()
            try:
                assert state["format_version"] == OPTIMIZER_STATE_VERSION
                for label in ("m", "v"):
                    for entry in state[label]:
                        assert entry.dtype in FLOATING_DTYPES
                        assert entry.dtype != INDEX_DTYPE
                for entry in state["parameters"]:
                    assert entry["dtype"] in FLOATING_DTYPES
                # An index dtype declared in the metadata is refused as a
                # mismatch against the live parameter — a second, separate
                # authority from the archive's entry validator.
                broken = dict(state)
                broken["parameters"] = (
                    {"shape": entry["shape"], "dtype": INDEX_DTYPE,
                     "device": entry["device"]}
                    for entry in state["parameters"]
                )
                broken["parameters"] = list(broken["parameters"])
                with pytest.raises(ValueError, match=INDEX_DTYPE):
                    optimizer.load_state_dict(broken)
                assert parameter.dtype == "float64"
            finally:
                for label in ("m", "v"):
                    for entry in state[label]:
                        entry.close()
        finally:
            optimizer.close()
            parameter.zero_grad()
            parameter.close()


@needs_backend
def test_a_standalone_index_tensor_beside_a_model_never_becomes_state(
        tmp_path):
    """A caller may hold an index tensor beside the model as evaluation
    metadata. It is an ordinary attribute — not a parameter, not a buffer,
    absent from ``state_dict()``, and absent from the archive."""
    with live_storage_baseline():
        module = NativeModule()
        module.weight = NativeParameter(np.array([[1.0, 2.0]]))
        try:
            with index_tensor((1, 0)) as tensor:
                module.predictions = tensor          # a plain attribute
                assert module.predictions is tensor
                assert [n for n, _ in module.named_parameters()] == ["weight"]
                assert list(module.named_buffers()) == []
                state = module.state_dict()
                try:
                    assert set(state) == {"weight"}
                finally:
                    for value in state.values():
                        value.close()
                path = str(tmp_path / "model.npz")
                save_native_checkpoint(path, module)
                manifest = manifest_of(path)
                assert set(manifest["model"]["entries"]) == {"weight"}
                assert INDEX_DTYPE not in json.dumps(manifest)
                assert sorted(arrays_of(path)) == ["manifest", "model::000000"]
                # ...and the tensor is still perfectly usable afterwards.
                assert tensor.tolist() == [1, 0]
        finally:
            module.weight.close()


@needs_backend
@pytest.mark.parametrize("dtype", FLOATING_DTYPES)
def test_floating_parameters_and_states_still_work_at_both_widths(dtype):
    """The negative control for every barrier above: the floating half of
    each surface is untouched."""
    with live_storage_baseline():
        values = np.array([1.0, 2.0], dtype=np.dtype(dtype))
        parameter = NativeParameter(values, dtype=dtype)
        buffer_tensor = NativeTensor.from_array(values, dtype=dtype)
        module = NativeModule()
        module.weight = parameter
        module.register_buffer("stat", buffer_tensor, persistent=True)
        optimizer = NativeAdam([parameter], lr=0.1)
        try:
            assert parameter.dtype == dtype
            assert set(module.state_dict()) == {"weight", "stat"}
            for tensor in module.state_dict().values():
                assert tensor.dtype == dtype
                tensor.close()
            state = optimizer.state_dict()
            try:
                assert state["format_version"] == OPTIMIZER_STATE_VERSION
                for entry in state["parameters"]:
                    assert entry["dtype"] == dtype
            finally:
                for label in ("m", "v"):
                    for entry in state[label]:
                        entry.close()
        finally:
            optimizer.close()
            buffer_tensor.close()
            parameter.close()


# ===========================================================================
# 3. Group C — loader and sampler state compatibility
# ===========================================================================

def test_the_loader_and_sampler_state_constants_are_unmoved():
    assert loader_module._FORMAT == LOADER_FORMAT
    assert loader_module._FORMAT_VERSION == LOADER_VERSION
    assert loader_module._SUPPORTED_FORMAT_VERSIONS == LOADER_VERSIONS
    assert loader_module._STATE_FIELDS == ("format", "format_version",
                                           "sampler")
    assert sampler_module._FORMAT == SAMPLER_FORMAT
    assert sampler_module._FORMAT_VERSION == SAMPLER_VERSION
    assert sampler_module._SUPPORTED_FORMAT_VERSIONS == SAMPLER_VERSIONS
    assert sampler_module._STATE_FIELDS == (
        "format", "format_version", "dataset", "seed", "shuffle",
        "batch_size", "drop_last", "epoch", "cursor")
    assert 2 not in loader_module._SUPPORTED_FORMAT_VERSIONS
    assert 2 not in sampler_module._SUPPORTED_FORMAT_VERSIONS


@needs_backend
@pytest.mark.parametrize("dtype", FLOATING_DTYPES)
def test_the_loader_state_carries_no_integer_or_dtype_expansion(dtype):
    """The state is exactly the three-key wrapper around the unchanged
    version-1 sampler state: no ``int64`` field, no tensor section, no
    dtype-registry expansion, and nothing that grows with the samples."""
    with live_storage_baseline():
        graph = build_graph(dtype)
        try:
            state = graph.loader.state_dict()
            assert set(state) == {"format", "format_version", "sampler"}
            assert state["format"] == LOADER_FORMAT
            assert state["format_version"] == LOADER_VERSION
            assert state["sampler"] == graph.sampler.state_dict()
            assert state["sampler"]["format"] == SAMPLER_FORMAT
            assert state["sampler"]["format_version"] == SAMPLER_VERSION
            # The nested key sets are exactly Phase J's, checked as **sets**
            # rather than by banning substrings: the two format tags
            # legitimately contain the words a substring ban would fire on
            # ("tensorforge.native_sampler"), which is precisely the
            # failure mode design §30.2 warns about.
            assert set(state["sampler"]) == {
                "format", "format_version", "dataset", "seed", "shuffle",
                "batch_size", "drop_last", "epoch", "cursor"}
            assert set(state["sampler"]["dataset"]) == {
                "samples", "feature_shape", "feature_dtype", "fingerprint"}
            # The only dtype-valued field anywhere in either state is the
            # dataset's feature dtype, and it is floating.
            assert state["sampler"]["dataset"]["feature_dtype"] == dtype
            assert state["sampler"]["dataset"]["feature_dtype"] \
                not in INDEX_DTYPES
            dtype_fields = [key for key in state["sampler"]["dataset"]
                            if key.endswith("dtype")]
            assert dtype_fields == ["feature_dtype"], dtype_fields
            assert INDEX_DTYPE not in json.dumps(state)
            # Nothing in the state grows with the sample count: only the
            # feature *shape* is a list, and it is the dataset's rank.
            assert state["sampler"]["dataset"]["feature_shape"] == [FEATURES]
            assert state["sampler"]["dataset"]["samples"] == SAMPLES
            # JSON-compatible and accepted unchanged by the checkpoint's
            # own metadata validator, exactly as Phase J promised.
            assert json.loads(json.dumps(state)) == state
            assert native_checkpoint._validated_metadata(
                state, "metadata", set()) == state
        finally:
            close_graph(graph)


@needs_backend
def test_the_loader_and_sampler_states_round_trip_and_reject_drift():
    """A valid round trip on both objects, the exact remaining order
    reproduced, and every malformed or unsupported version refused."""
    with live_storage_baseline(settle=True):
        graph = build_graph()
        try:
            stream = BatchStream(graph.loader)
            try:
                run_steps(graph, stream, 1, evaluate_at=())
            finally:
                stream.close()
            loader_state = graph.loader.state_dict()
            sampler_state = graph.sampler.state_dict()
            expected_next = graph.sampler.next_batch_indices()
            expected_plan = graph.sampler.plan()
            expected_remaining = graph.sampler.remaining

            # Move both away, then restore each through its own method.
            moved = json.loads(json.dumps(loader_state))
            moved["sampler"]["epoch"] = 4
            moved["sampler"]["cursor"] = 2
            graph.loader.load_state_dict(moved)
            assert graph.sampler.next_batch_indices() != expected_next
            graph.loader.load_state_dict(loader_state)
            assert graph.loader.state_dict() == loader_state
            assert graph.sampler.next_batch_indices() == expected_next
            assert graph.sampler.plan() == expected_plan
            assert graph.sampler.remaining == expected_remaining

            graph.sampler.load_state_dict(moved["sampler"])
            assert graph.sampler.next_batch_indices() != expected_next
            graph.sampler.load_state_dict(sampler_state)
            assert graph.sampler.state_dict() == sampler_state
            assert graph.sampler.next_batch_indices() == expected_next

            def unchanged(label):
                """The **complete** observable pipeline state, asserted
                immediately after a single rejection rather than once after
                a whole loop: a comparison made only at the end cannot say
                *which* rejection moved something, and a rejection that
                moved something a later one moved back would pass it."""
                assert graph.loader.state_dict() == loader_state, label
                assert graph.sampler.state_dict() == sampler_state, label
                assert graph.sampler.next_batch_indices() == expected_next, \
                    label
                assert graph.sampler.plan() == expected_plan, label
                assert graph.sampler.remaining == expected_remaining, label

            def reject(owner, state, label):
                with pytest.raises((TypeError, ValueError)):
                    owner.load_state_dict(state)
                unchanged(label)

            # Unsupported and malformed versions, on both objects.
            for bad in (2, 0, -1, True, "1", 1.0, None):
                broken = json.loads(json.dumps(loader_state))
                broken["format_version"] = bad
                reject(graph.loader, broken, f"loader version {bad!r}")
                broken = json.loads(json.dumps(sampler_state))
                broken["format_version"] = bad
                reject(graph.sampler, broken, f"sampler version {bad!r}")

            # Format-tag drift, an unexpected field, and each required
            # field missing — driven **directly on the sampler** as well as
            # on the loader wrapper. Delegation is the loader's contract,
            # not a substitute for exercising the authority it delegates
            # to: a sampler asked for its own state answers for it.
            sampler_cases = [
                ("wrong tag", lambda s: s.update(format=LOADER_FORMAT)),
                ("tag type", lambda s: s.update(format=1)),
                ("unexpected field", lambda s: s.update(int64_indices=[1, 2])),
                ("dataset unexpected field",
                 lambda s: s["dataset"].update(index_dtype=INDEX_DTYPE)),
                ("dataset index dtype",
                 lambda s: s["dataset"].update(feature_dtype=INDEX_DTYPE)),
                ("not a dict", None),
            ]
            sampler_cases += [
                (f"missing {field}", lambda s, f=field: s.pop(f))
                for field in sampler_module._STATE_FIELDS
            ]
            sampler_cases += [
                (f"dataset missing {field}",
                 lambda s, f=field: s["dataset"].pop(f))
                for field in ("samples", "feature_shape", "feature_dtype",
                              "fingerprint")
            ]
            for label, mutate in sampler_cases:
                if mutate is None:
                    reject(graph.sampler, [1, 2], f"sampler {label}")
                    continue
                broken = json.loads(json.dumps(sampler_state))
                mutate(broken)
                assert broken != sampler_state, label   # non-vacuity
                reject(graph.sampler, broken, f"sampler {label}")

            loader_cases = [
                ("wrong tag", lambda s: s.update(format=SAMPLER_FORMAT)),
                ("tag type", lambda s: s.update(format=1)),
                ("unexpected field", lambda s: s.update(int64_indices=[1, 2])),
                ("nested wrong tag",
                 lambda s: s["sampler"].update(format=LOADER_FORMAT)),
                ("nested unexpected field",
                 lambda s: s["sampler"].update(int64_indices=[1, 2])),
                ("nested missing cursor",
                 lambda s: s["sampler"].pop("cursor")),
                ("nested not a dict",
                 lambda s: s.update(sampler=[1, 2])),
                ("not a dict", None),
            ]
            loader_cases += [
                (f"missing {field}", lambda s, f=field: s.pop(f))
                for field in loader_module._STATE_FIELDS
            ]
            for label, mutate in loader_cases:
                if mutate is None:
                    reject(graph.loader, "not a state", f"loader {label}")
                    continue
                broken = json.loads(json.dumps(loader_state))
                mutate(broken)
                assert broken != loader_state, label    # non-vacuity
                reject(graph.loader, broken, f"loader {label}")

            # The control that keeps every ``unchanged()`` above honest: the
            # same comparison **does** notice a real move, so "unchanged"
            # is a measurement rather than an assertion that cannot fail.
            graph.loader.load_state_dict(moved)
            with pytest.raises(AssertionError):
                unchanged("control")
            graph.loader.load_state_dict(loader_state)
            unchanged("restored")
        finally:
            close_graph(graph)


@needs_backend
def test_a_live_index_tensor_beside_the_pipeline_changes_no_state():
    """Holding, viewing, and consuming an index tensor beside a loader
    moves neither state, neither version, and neither position."""
    with live_storage_baseline():
        graph = build_graph()
        try:
            before = (graph.loader.state_dict(), graph.sampler.state_dict(),
                      graph.sampler.next_batch_indices())
            with index_tensor((2, 0, 1)) as tensor:
                assert tensor.dtype == INDEX_DTYPE
                view = tensor.reshape((3, 1))
                try:
                    assert view.dtype == INDEX_DTYPE
                finally:
                    view.close()
                source = NativeTensor.from_array(
                    np.arange(9, dtype=np.float64).reshape(3, 3))
                try:
                    selected = source.index_select(0, tensor)
                    selected.close()
                finally:
                    source.close()
            assert (graph.loader.state_dict(), graph.sampler.state_dict(),
                    graph.sampler.next_batch_indices()) == before
        finally:
            close_graph(graph)


# ===========================================================================
# 4. Group D — Phase-J delivery, unchanged, and explicit caller conversion
# ===========================================================================

@needs_backend
@pytest.mark.parametrize("dtype", FLOATING_DTYPES)
def test_the_delivery_contract_is_still_a_native_tensor_and_host_int64(dtype):
    """``(NativeTensor, numpy.ndarray of dtype int64)`` — the Phase-J
    default, at both feature widths, with the target proved to be host
    metadata and **not** a native tensor."""
    with live_storage_baseline():
        graph = build_graph(dtype)
        try:
            planned = []
            delivered = []
            stream = BatchStream(graph.loader)
            try:
                for _ in range(BATCHES_PER_EPOCH):
                    planned.append(graph.sampler.next_batch_indices())
                    features, targets = stream.next_batch()
                    try:
                        assert type(features) is NativeTensor
                        assert features.dtype == dtype
                        assert features.shape == (BATCH, FEATURES)
                        assert features.device == "cpu"
                        # The target half: exactly host metadata.
                        assert type(targets) is np.ndarray
                        assert targets.dtype == np.dtype(np.int64)
                        assert targets.ndim == 1 and len(targets) == BATCH
                        assert not isinstance(targets, NativeTensor)
                        assert not targets.flags["WRITEABLE"]
                        assert targets.flags["C_CONTIGUOUS"]
                        delivered.append(tuple(targets.tolist()))
                    finally:
                        features.close()
            finally:
                stream.close()
            # Ordering and batching are the sampler's plan, unchanged.
            values, host_targets = host_arrays()
            assert tuple(planned) == tuple(graph.sampler.plan(epoch=0))
            for indices, labels in zip(planned, delivered):
                assert labels == tuple(host_targets[list(indices)].tolist())
        finally:
            close_graph(graph)


def test_no_dataset_or_loader_option_requests_native_labels():
    """No Phase-J surface gained a parameter that could convert a target,
    and ``target_batch`` still returns a host array."""
    assert parameter_names(NativeTensorDataset.__init__) == \
        ("self", "features", "targets", "dtype")
    assert parameter_names(NativeTensorDataset.target_batch) == \
        ("self", "indices")
    assert parameter_names(NativeTensorDataset.feature_batch) == \
        ("self", "indices")
    assert parameter_names(NativeBatchSampler.__init__) == \
        ("self", "dataset", "batch_size", "shuffle", "seed", "drop_last")
    assert parameter_names(NativeDataLoader.__init__) == ("self", "sampler")
    # ``__next__`` lives on the loader's private iterator, and it takes no
    # argument at all — there is nowhere for a delivery option to sit.
    iterator_class = loader_module._NativeBatchIterator
    assert parameter_names(iterator_class.__next__) == ("self",)
    assert parameter_names(iterator_class.__iter__) == ("self",)
    banned = ("native_targets", "native_target", "target_dtype",
              "index_dtype", "as_tensor", "collate", "collate_fn",
              "transform", "device", "workers", "num_workers", "prefetch",
              "pin_memory")
    for owner in (NativeTensorDataset, NativeBatchSampler, NativeDataLoader,
                  iterator_class):
        for name, member in vars(owner).items():
            if not callable(member) or name.startswith("__"):
                continue
            try:
                names = parameter_names(member)
            except (TypeError, ValueError):
                continue
            for absent in banned:
                assert absent not in names, (owner.__name__, name, absent)
    # ...and no pipeline module's executable code names the integer door.
    for module in ("native_dataset", "native_sampler", "native_data_loader",
                   "_native_permutation"):
        names = code_names(f"{PACKAGE}/{module}.py")
        for absent in ("from_int64_array", "INDEX_DTYPES",
                       "_normalize_index_dtype", "_from_int64_array",
                       "argmax", "index_select"):
            assert absent not in names, (module, absent)


def test_the_pipeline_scanner_can_actually_fail():
    """The control the scan above needs: it really does see the names a
    pipeline module *does* use."""
    names = code_names(f"{PACKAGE}/native_dataset.py")
    assert {"NativeTensor", "from_array", "target_batch"} <= names
    loader = code_names(f"{PACKAGE}/native_data_loader.py")
    assert {"state_dict", "load_state_dict", "feature_batch"} <= loader


def _select_predicted_columns(graph, features, indices, labels, dtype):
    """One forward, then an ``index_select`` of the labelled columns.

    A function rather than inline code so the forward's locals — and with
    them the framework-owned graph nodes it built — are gone by the time
    the caller's baseline is checked."""
    logits = graph.model(features)
    try:
        detached = logits.detach()
        try:
            selected = detached.index_select(CLASS_AXIS, indices)
            try:
                assert selected.dtype == dtype
                assert selected.shape == (len(labels), len(labels))
                host_logits = logits.to_numpy()
                host_selected = selected.to_numpy()
                for position, index in enumerate(labels):
                    assert bits(np.ascontiguousarray(
                        host_selected[:, position])) == \
                        bits(np.ascontiguousarray(host_logits[:, index]))
            finally:
                selected.close()
        finally:
            detached.close()
    finally:
        logits.close()


@needs_backend
@pytest.mark.parametrize("dtype", FLOATING_DTYPES)
def test_a_caller_may_convert_delivered_targets_explicitly(dtype):
    """The supported route, and it is a **caller's** line of code after
    delivery: nothing in the pipeline changed, and the resulting tensor is
    still refused by every state-owning surface."""
    with live_storage_baseline(settle=True):
        graph = build_graph(dtype)
        try:
            before_delivery = (graph.loader.state_dict(),
                               graph.sampler.state_dict())
            stream = BatchStream(graph.loader)
            try:
                features, targets = stream.next_batch()
                try:
                    # Captured the moment the batch is in the caller's
                    # hands. The delivery itself legitimately advances the
                    # committed position — that is Phase J's contract, not
                    # something K5 disputes — so the baseline for "the
                    # conversion moved nothing" has to be taken *after* it.
                    loader_after_delivery = graph.loader.state_dict()
                    sampler_after_delivery = graph.sampler.state_dict()
                    assert loader_after_delivery != before_delivery[0]
                    assert sampler_after_delivery != before_delivery[1]

                    # The caller's own writable array — and it is the array
                    # the constructor actually receives. Converting the
                    # delivered read-only array instead would leave this
                    # copy unrelated to the tensor, and mutating it would
                    # then prove nothing about anything.
                    host = np.array(targets, dtype=np.int64, copy=True)
                    expected = host.tolist()
                    assert expected == targets.tolist()
                    assert host.flags["WRITEABLE"]
                    assert host is not targets

                    indices = NativeTensor.from_int64_array(host)
                    try:
                        assert indices.dtype == INDEX_DTYPE
                        assert indices.shape == (BATCH,)
                        assert indices.tolist() == expected

                        # Independent of the host array it was built from:
                        # mutate that exact array and the tensor does not
                        # move. The replacement is a different **valid**
                        # class, and the mutation is proved to have landed
                        # before the independence is claimed.
                        replacement = (expected[0] + 1) % CLASSES
                        assert replacement != expected[0]
                        host[0] = replacement
                        assert int(host[0]) == replacement
                        assert host.tolist() != expected
                        assert indices.tolist() == expected

                        # ...and independent in the other direction: a host
                        # array the tensor produced is the caller's, and
                        # writing to it cannot reach native storage.
                        materialized = indices.to_numpy()
                        assert materialized is not targets
                        assert materialized is not host
                        assert materialized.tolist() == expected
                        materialized[0] = 77
                        assert int(materialized[0]) == 77
                        assert materialized.tolist() != expected
                        assert indices.tolist() == expected
                        assert indices.to_numpy().tolist() == expected

                        # Consumable by index_select where the values are in
                        # range: the class axis has CLASSES positions and
                        # every label is a class.
                        _select_predicted_columns(graph, features, indices,
                                                  expected, dtype)

                        # ...and it is still not state, in every direction.
                        with pytest.raises(ValueError, match=INDEX_DTYPE):
                            NativeParameter(indices)
                        module = NativeModule()
                        for persistent in (True, False):
                            with pytest.raises(ValueError, match=INDEX_DTYPE):
                                module.register_buffer("t", indices,
                                                       persistent=persistent)
                        plain = NativeTensor.from_array(
                            np.zeros((BATCH, CLASSES)), dtype=dtype)
                        try:
                            with pytest.raises(TypeError,
                                               match="sequence of ints"):
                                graph.loss_fn(plain, indices)
                        finally:
                            plain.close()
                    finally:
                        indices.close()

                    # The claim, in full: **nothing after the delivery**
                    # moved the pipeline. Not the epoch alone — the whole
                    # loader and sampler state, which is what would have to
                    # be equal for a resume to land in the same place.
                    assert graph.loader.state_dict() == loader_after_delivery
                    assert graph.sampler.state_dict() == sampler_after_delivery
                finally:
                    features.close()
            finally:
                stream.close()
        finally:
            close_graph(graph)


# ===========================================================================
# 5. Group E — classification compatibility
# ===========================================================================

def cross_entropy_oracle(host_logits, labels):
    """A plain host reference for the *value* claim. This is the module's
    one tolerance, and it is a claim about the arithmetic being right
    rather than about bits being preserved."""
    shifted = host_logits - host_logits.max(axis=1, keepdims=True)
    log_probabilities = shifted - np.log(
        np.exp(shifted).sum(axis=1, keepdims=True))
    rows = np.arange(len(labels))
    return float(-log_probabilities[rows, labels].mean())


ACCEPTED_TARGET_FORMS = (
    ("list of ints", lambda labels: [int(v) for v in labels]),
    ("tuple of ints", lambda labels: tuple(int(v) for v in labels)),
    ("int64 array", lambda labels: np.asarray(labels, dtype=np.int64)),
    ("int32 array", lambda labels: np.asarray(labels, dtype=np.int32)),
    ("int8 array", lambda labels: np.asarray(labels, dtype=np.int8)),
    ("uint8 array", lambda labels: np.asarray(labels, dtype=np.uint8)),
    ("read-only int64", lambda labels: _read_only(labels)),
)


def _read_only(labels):
    array = np.asarray(labels, dtype=np.int64).copy()
    array.setflags(write=False)
    return array


REJECTED_TARGET_FORMS = (
    ("bool array", lambda labels: np.asarray(labels, dtype=bool), TypeError),
    ("float array", lambda labels: np.asarray(labels, dtype=np.float64),
     TypeError),
    ("rank-2 array", lambda labels: np.asarray(labels,
                                               dtype=np.int64).reshape(-1, 1),
     ValueError),
    ("str", lambda labels: "".join(str(int(v)) for v in labels), TypeError),
    ("scalar", lambda labels: int(labels[0]), TypeError),
)


@needs_backend
@pytest.mark.parametrize("dtype", FLOATING_DTYPES)
def test_cross_entropy_accepts_exactly_the_documented_host_target_forms(dtype):
    """The accepted set, the rejected set, and the loss value — all
    unchanged, at each width independently."""
    tolerance = 1e-12 if dtype == "float64" else 1e-6
    with live_storage_baseline():
        host = np.array([[2.0, -1.0, 0.5], [0.25, 0.25, -3.0]],
                        dtype=np.dtype(dtype))
        labels = [2, 0]
        logits = NativeTensor.from_array(host.astype(np.float64), dtype=dtype)
        loss_fn = NativeCrossEntropyLoss()
        try:
            expected = cross_entropy_oracle(host.astype(np.float64),
                                            np.array(labels))
            reference = None
            for label, build in ACCEPTED_TARGET_FORMS:
                loss = loss_fn(logits, build(labels))
                try:
                    assert loss.shape == ()
                    assert loss.dtype == dtype
                    value = loss.to_numpy()
                    assert abs(float(value) - expected) < tolerance, label
                    # Every accepted form gives the *same bits*, so the
                    # accepted set really is one contract rather than
                    # several that happen to agree numerically.
                    if reference is None:
                        reference = bits(value)
                    assert bits(value) == reference, label
                finally:
                    loss.close()
            for label, build, error in REJECTED_TARGET_FORMS:
                with pytest.raises(error):
                    loss_fn(logits, build(labels))
        finally:
            logits.close()


@needs_backend
@pytest.mark.parametrize("dtype", FLOATING_DTYPES)
def test_cross_entropy_values_and_gradients_are_deterministic(dtype):
    """Two identical calls give bit-identical losses and bit-identical
    gradients — the determinism claim, with no tolerance."""
    with live_storage_baseline(settle=True):
        host = np.array([[1.0, 0.0, -0.5], [-2.0, 0.5, 0.25]])
        labels = np.array([1, 2], dtype=np.int64)
        records = []
        for _ in range(2):
            parameter = NativeParameter(host, dtype=dtype)
            loss_fn = NativeCrossEntropyLoss()
            try:
                loss = loss_fn(parameter, labels)
                try:
                    loss_bits = bits(loss.to_numpy())
                    loss.backward()
                finally:
                    loss.close()
                records.append((loss_bits, bits(parameter.grad.to_numpy())))
                # The gradient of mean cross-entropy is (softmax - onehot)/N,
                # checked once as a value claim with a labelled tolerance.
                shifted = host - host.max(axis=1, keepdims=True)
                probabilities = np.exp(shifted)
                probabilities /= probabilities.sum(axis=1, keepdims=True)
                probabilities[np.arange(len(labels)), labels] -= 1.0
                expected = probabilities / len(labels)
                assert np.allclose(parameter.grad.to_numpy(), expected,
                                   atol=1e-6 if dtype == "float32" else 1e-12)
            finally:
                # A parameter's ``close()`` releases its own storage and not
                # the separate tensor its ``.grad`` slot holds.
                parameter.zero_grad()
                parameter.close()
        assert records[0] == records[1]


@needs_backend
@pytest.mark.parametrize("dtype", FLOATING_DTYPES)
def test_a_native_index_tensor_is_still_refused_as_a_cross_entropy_target(
        dtype):
    """Three separate ways to arrive at a native ``int64`` target — the
    explicit door, a fresh ``argmax`` result, and a view of one — all
    refused at the same host-target boundary, and nothing mutated."""
    with live_storage_baseline():
        host = np.array([[2.0, -1.0, 0.5], [0.25, 0.25, -3.0]])
        logits = NativeTensor.from_array(host, dtype=dtype)
        loss_fn = NativeCrossEntropyLoss()
        try:
            before = bits(logits.to_numpy())
            predictions = logits.argmax(axis=CLASS_AXIS)
            try:
                assert predictions.dtype == INDEX_DTYPE
                candidates = {
                    "argmax result": predictions,
                    "explicit door": NativeTensor.from_int64_array(
                        np.array([1, 0], dtype=np.int64)),
                    "view of an index tensor": predictions.reshape((2,)),
                }
                try:
                    for label, target in candidates.items():
                        with pytest.raises(TypeError,
                                           match="sequence of ints"):
                            loss_fn(logits, target)
                        with pytest.raises(TypeError,
                                           match="sequence of ints"):
                            logits.cross_entropy(target)
                        # The boundary that rejected is the shared host
                        # validator, not a second rule in the module.
                        with pytest.raises(TypeError,
                                           match="sequence of ints"):
                            cpp._prepare_class_targets(target, 2, CLASSES,
                                                       label)
                finally:
                    for label, target in candidates.items():
                        if target is not predictions:
                            target.close()
            finally:
                predictions.close()
            assert bits(logits.to_numpy()) == before
            assert logits.grad is None
        finally:
            logits.close()


def test_the_classification_stack_names_no_integer_tensor_route():
    """The loss module and the shared target validator gained no route to a
    native integer target, read from executable code rather than prose."""
    loss_names = code_names(f"{PACKAGE}/native_cross_entropy_loss.py")
    for absent in ("from_int64_array", "INDEX_DTYPES", "argmax",
                   "index_select", "_normalize_index_dtype",
                   "_require_index_dtype", "_is_index_dtype"):
        assert absent not in loss_names, absent
    # The control: the module's real names are visible to the same reader.
    assert {"NativeCrossEntropyLoss", "cross_entropy", "reduction",
            "_normalize_reduction"} <= loss_names
    # ...and the module delegates rather than validating targets itself.
    assert "_prepare_class_targets" not in loss_names


@needs_backend
def test_native_accuracy_is_still_the_host_round_trip():
    """It materializes through ``to_numpy()`` exactly once, calls NumPy's
    ``argmax``, and returns the plain fraction."""
    with live_storage_baseline():
        host = np.array([[0.1, 0.9, 0.0], [2.0, 1.0, 0.5], [0.0, 0.0, 1.0]])
        logits = NativeTensor.from_array(host)
        try:
            expected = float(np.mean(np.argmax(host, axis=1) == [1, 0, 2]))
            calls = []
            original = NativeTensor.to_numpy

            def counting(self):
                calls.append(id(self))
                return original(self)

            NativeTensor.to_numpy = counting
            try:
                with allocated_dtypes() as seen:
                    value = native_accuracy(logits, [1, 0, 2])
            finally:
                NativeTensor.to_numpy = original
            assert value == expected
            assert type(value) is float
            assert calls == [id(logits)], "exactly one host materialization"
            # No native int64 result tensor is built inside the metric.
            assert INDEX_DTYPE not in seen, seen
            # It accepts the same host target forms as the loss.
            for _, build in ACCEPTED_TARGET_FORMS:
                assert native_accuracy(logits, build([1, 0, 2])) == expected
            for _, build, error in REJECTED_TARGET_FORMS:
                with pytest.raises(error):
                    native_accuracy(logits, build([1, 0, 2]))
            with pytest.raises(TypeError, match="sequence of ints"):
                with index_tensor((1, 0, 2)) as tensor:
                    native_accuracy(logits, tensor)
        finally:
            logits.close()


@needs_backend
def test_native_accuracy_calls_neither_native_argmax_nor_index_select():
    """The strong form: patch both operations to raise, and watch the
    metric succeed anyway. Every patch is restored in ``finally``, and a
    control afterwards proves the patches would have fired."""
    with live_storage_baseline():
        host = np.array([[0.1, 0.9], [2.0, 1.0]])
        logits = NativeTensor.from_array(host)
        try:
            expected = native_accuracy(logits, [1, 0])

            class Tripwire(RuntimeError):
                """Raised only if the metric reaches a native operation."""

            def trip(self, *args, **kwargs):
                raise Tripwire("native_accuracy reached a native operation")

            original_argmax = NativeTensor.argmax
            original_select = NativeTensor.index_select
            original_core_argmax = cpp.NativeTensorCore.argmax
            original_core_select = cpp.NativeTensorCore.index_select
            NativeTensor.argmax = trip
            NativeTensor.index_select = trip
            cpp.NativeTensorCore.argmax = trip
            cpp.NativeTensorCore.index_select = trip
            try:
                # The metric still works with every native index operation
                # disarmed, which is only possible if it calls none of them.
                assert native_accuracy(logits, [1, 0]) == expected
                # The control: the tripwire really is armed.
                with pytest.raises(Tripwire):
                    logits.argmax(axis=1)
                with pytest.raises(Tripwire):
                    logits.index_select(0, logits)
            finally:
                NativeTensor.argmax = original_argmax
                NativeTensor.index_select = original_select
                cpp.NativeTensorCore.argmax = original_core_argmax
                cpp.NativeTensorCore.index_select = original_core_select
            # ...and a successful control run afterwards, proving the
            # restoration worked and the operations are live again.
            predictions = logits.argmax(axis=1)
            try:
                assert predictions.tolist() == [1, 0]
            finally:
                predictions.close()
            assert native_accuracy(logits, [1, 0]) == expected
        finally:
            logits.close()


@needs_backend
@pytest.mark.parametrize("label,rows", [
    ("plain", [[0.0, 1.0, 2.0], [3.0, 1.0, 2.0]]),
    ("exact tie", [[1.0, 1.0, 0.0], [0.5, 0.5, 0.5]]),
    ("signed zeros", [[-0.0, 0.0, -1.0], [0.0, -0.0, -1.0]]),
    ("one nan", [[0.0, float("nan"), 2.0], [float("nan"), 1.0, 2.0]]),
    ("several nans", [[float("nan"), float("nan"), 2.0],
                      [1.0, float("nan"), float("nan")]]),
    ("infinities", [[float("inf"), 1.0, float("-inf")],
                    [float("-inf"), float("-inf"), float("-inf")]]),
    ("nan with infinity", [[float("inf"), float("nan"), 0.0],
                           [float("nan"), float("-inf"), 0.0]]),
])
def test_native_accuracy_still_inherits_numpys_conventions(label, rows):
    """Ties and exceptional values follow ``numpy.argmax`` unchanged.

    This is asserted against a NumPy oracle rather than against
    ``NativeTensor.argmax``: the two rules are **not** documented as
    equivalent, and a test that compared them would be inventing the
    equivalence the metric's own docstring declines to claim."""
    with live_storage_baseline():
        host = np.array(rows, dtype=np.float64)
        targets = [0, 1]
        logits = NativeTensor.from_array(host)
        try:
            expected = float(np.mean(np.argmax(host, axis=1) == targets))
            assert native_accuracy(logits, targets) == expected, label
        finally:
            logits.close()


@needs_backend
def test_native_accuracy_builds_no_graph_and_mutates_nothing():
    with live_storage_baseline(settle=True):
        parameter = NativeParameter(np.array([[0.1, 0.9], [2.0, 1.0]]))
        targets = np.array([1, 0], dtype=np.int64)
        try:
            before = (bits(parameter.to_numpy()), parameter.version,
                      parameter.grad, parameter.requires_grad,
                      parameter.is_leaf, targets.tolist())
            native_accuracy(parameter, targets)
            assert (bits(parameter.to_numpy()), parameter.version,
                    parameter.grad, parameter.requires_grad,
                    parameter.is_leaf, targets.tolist()) == before
            # A graph built before the call is still fully usable after it.
            out = parameter.sum()
            try:
                native_accuracy(parameter, targets)
                out.backward()
            finally:
                out.close()
            assert parameter.grad is not None
        finally:
            parameter.zero_grad()
            parameter.close()


def test_the_metric_module_still_states_its_host_boundary():
    """``native_accuracy`` reads NumPy's ``argmax`` in its executable code
    and names no native index operation there."""
    names = code_names(f"{PACKAGE}/native_metrics.py")
    assert {"native_accuracy", "to_numpy", "argmax", "mean",
            "_prepare_class_targets"} <= names
    assert "index_select" not in names
    source = ast.parse(
        (REPO_ROOT / PACKAGE / "native_metrics.py").read_text(encoding="utf-8"))
    native_argmax_calls = [
        node for node in ast.walk(source)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "argmax"
        and not (isinstance(node.func.value, ast.Name)
                 and node.func.value.id == "np")
    ]
    assert native_argmax_calls == [], "a non-NumPy argmax call appeared"
    # The control: the NumPy call really is there to be found.
    numpy_argmax_calls = [
        node for node in ast.walk(source)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "argmax"
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "np"
    ]
    assert len(numpy_argmax_calls) == 1


# ===========================================================================
# 6. Group F — deterministic training, checkpoint, and resume
# ===========================================================================

def strip_evaluation(records):
    return [{key: value for key, value in record.items()
             if key != "evaluation"} for record in records]


@needs_backend
@pytest.mark.parametrize("dtype", FLOATING_DTYPES)
def test_training_checkpoint_and_resume_stay_bit_identical(tmp_path, dtype):
    """**K5's exit gate.** An uninterrupted run and an
    interrupted-and-resumed run agree bit for bit in every family, while
    ``argmax`` and ``index_select`` are used beside the training path at
    fixed points on both sides of the interruption.

    Each dtype is proved only against itself, and every floating
    comparison is a raw IEEE-754 bit comparison."""
    with live_storage_baseline(settle=True):
        # --- Run A: uninterrupted -----------------------------------------
        uninterrupted = build_graph(dtype)
        try:
            stream = BatchStream(uninterrupted.loader)
            try:
                records_a = run_steps(uninterrupted, stream, TOTAL_STEPS)
            finally:
                stream.close()
            fingerprint_a = trainable_fingerprint(uninterrupted)
        finally:
            close_graph(uninterrupted)

        # The workload really did what the constants promise.
        assert len(records_a) == TOTAL_STEPS
        evaluated = [index for index, record in enumerate(records_a)
                     if record["evaluation"] is not None]
        assert tuple(evaluated) == EVAL_STEPS
        assert any(index < SPLIT_STEP for index in evaluated)
        assert any(index >= SPLIT_STEP for index in evaluated)
        # ...including at least one batch whose predicted classes repeat,
        # which BATCH > CLASSES guarantees by pigeonhole.
        duplicates = [record for record in records_a
                      if record["evaluation"] is not None
                      and len(set(record["evaluation"][0]))
                      < len(record["evaluation"][0])]
        assert duplicates, "no batch exercised duplicate predicted classes"

        # --- Run B: interrupted, saved, and resumed -----------------------
        path = str(tmp_path / f"resume_{dtype}.npz")
        first = build_graph(dtype)
        try:
            stream = BatchStream(first.loader)
            try:
                records_head = run_steps(first, stream, SPLIT_STEP)
            finally:
                stream.close()
            # A genuine mid-epoch interruption with batches still owed.
            assert first.sampler.epoch == 0
            assert 0 < first.sampler.cursor < BATCHES_PER_EPOCH
            assert first.sampler.remaining == \
                BATCHES_PER_EPOCH - SPLIT_STEP
            loader_state = first.loader.state_dict()
            saved = trainable_fingerprint(first)
            save_native_checkpoint(
                path, first.model, optimizer=first.optimizer,
                metadata=training_metadata(SPLIT_STEP, loader_state))
            assert first.loader.state_dict() == loader_state
        finally:
            close_graph(first)
        assert strip_evaluation(records_head) == \
            strip_evaluation(records_a[:SPLIT_STEP])

        # An entirely fresh graph, deliberately built wrong and **proved**
        # different before the load.
        second = build_graph(dtype, in_seed=91, out_seed=92, shared_seed=93,
                             own_seed=94, lr=0.5, batch_size=4,
                             shuffle=False, seed=1234)
        try:
            assert trainable_fingerprint(second) != saved
            assert second.loader.state_dict() != loader_state
            metadata = load_native_checkpoint(path, second.model,
                                              optimizer=second.optimizer)
            second.loader.load_state_dict(
                metadata["training"]["data_loader"])
            assert metadata["training"]["next_step"] == SPLIT_STEP
            assert trainable_fingerprint(second) == saved
            stream = BatchStream(second.loader)
            try:
                records_tail = run_steps(second, stream,
                                         TOTAL_STEPS - SPLIT_STEP,
                                         first_step=SPLIT_STEP)
            finally:
                stream.close()
            fingerprint_b = trainable_fingerprint(second)
        finally:
            close_graph(second)

        # --- the exact comparison -----------------------------------------
        assert records_tail == records_a[SPLIT_STEP:]
        assert fingerprint_b == fingerprint_a
        # ...family by family, so a failure names what drifted.
        for family in ("tensors", "moments", "parameters", "scalars",
                       "generators", "loader", "sampler", "keys"):
            assert fingerprint_b[family] == fingerprint_a[family], family
        # ...and the index and selection halves in particular.
        for tail, whole in zip(records_tail, records_a[SPLIT_STEP:]):
            if tail["evaluation"] is not None:
                assert tail["evaluation"][0] == whole["evaluation"][0]
                assert tail["evaluation"][1] == whole["evaluation"][1]


@needs_backend
@pytest.mark.parametrize("dtype", FLOATING_DTYPES)
def test_the_resume_proof_is_not_vacuous(tmp_path, dtype):
    """The negative control: **omitting** the loader restoration must make
    the continuation diverge, so the exit gate cannot pass by accident."""
    with live_storage_baseline(settle=True):
        uninterrupted = build_graph(dtype)
        try:
            stream = BatchStream(uninterrupted.loader)
            try:
                records_a = run_steps(uninterrupted, stream, TOTAL_STEPS,
                                      evaluate_at=())
            finally:
                stream.close()
        finally:
            close_graph(uninterrupted)

        path = str(tmp_path / f"vacuity_{dtype}.npz")
        first = build_graph(dtype)
        try:
            stream = BatchStream(first.loader)
            try:
                run_steps(first, stream, SPLIT_STEP, evaluate_at=())
            finally:
                stream.close()
            loader_state = first.loader.state_dict()
            save_native_checkpoint(
                path, first.model, optimizer=first.optimizer,
                metadata=training_metadata(SPLIT_STEP, loader_state))
        finally:
            close_graph(first)

        second = build_graph(dtype, batch_size=4, shuffle=False, seed=1234)
        try:
            metadata = load_native_checkpoint(path, second.model,
                                              optimizer=second.optimizer)
            # The checkpoint restored the model and touched the loader not
            # at all, which is exactly why the loader must be loaded too.
            assert second.loader.state_dict() != loader_state
            stream = BatchStream(second.loader)
            try:
                records_tail = run_steps(second, stream,
                                         TOTAL_STEPS - SPLIT_STEP,
                                         evaluate_at=())
            finally:
                stream.close()
            assert strip_evaluation(records_tail) != \
                strip_evaluation(records_a[SPLIT_STEP:])
            assert metadata["training"]["data_loader"] == loader_state
        finally:
            close_graph(second)


@needs_backend
@pytest.mark.parametrize("dtype", FLOATING_DTYPES)
def test_the_evaluation_indexing_changes_no_training_state(dtype):
    """The observational control. Two otherwise identical uninterrupted
    runs, one with the ``argmax`` / detached ``index_select`` evaluation
    path and one without, agree bit for bit in every trainable family.

    That is the proof the operations consume no RNG draw, mutate no model,
    optimizer, or pipeline state, create no graph edge, and cannot change
    what a checkpoint would contain."""
    fingerprints = {}
    records = {}
    with live_storage_baseline(settle=True):
        for evaluate in (True, False):
            graph = build_graph(dtype)
            try:
                stream = BatchStream(graph.loader)
                try:
                    records[evaluate] = run_steps(
                        graph, stream, TOTAL_STEPS,
                        evaluate_at=EVAL_STEPS if evaluate else ())
                finally:
                    stream.close()
                fingerprints[evaluate] = trainable_fingerprint(graph)
            finally:
                close_graph(graph)
    assert fingerprints[True] == fingerprints[False]
    assert strip_evaluation(records[True]) == strip_evaluation(records[False])
    # The control that keeps the comparison honest: the evaluation really
    # ran in one leg and not the other.
    assert any(record["evaluation"] is not None
               for record in records[True])
    assert all(record["evaluation"] is None for record in records[False])


def _evaluate_one_batch(graph, repeats):
    """One forward, then ``repeats`` evaluation records over its logits.

    A function rather than inline code so the forward's locals — and with
    them the framework-owned graph nodes the forward built — are gone by
    the time the caller's baseline is checked. The **strict** inner
    baseline is what proves the evaluation path itself closes everything
    it allocates."""
    stream = BatchStream(graph.loader)
    try:
        features, _ = stream.next_batch()
        try:
            logits = graph.model(features)
            try:
                with live_storage_baseline():
                    for _ in range(repeats):
                        evaluation_record(logits)
            finally:
                logits.close()
        finally:
            features.close()
    finally:
        stream.close()


@needs_backend
@pytest.mark.parametrize("dtype", FLOATING_DTYPES)
def test_the_evaluation_path_leaks_no_native_storage(dtype):
    """Every ``argmax`` result, every detached temporary, and every
    ``index_select`` result is closed explicitly.

    The inner baseline is **strict**: the evaluation path builds no graph,
    so nothing there is reclaimed rather than closed, and its return to
    baseline waits for no collection at all."""
    with live_storage_baseline(settle=True):
        graph = build_graph(dtype)
        try:
            _evaluate_one_batch(graph, repeats=3)
        finally:
            close_graph(graph)


# ===========================================================================
# 7. Absence boundaries and unmoved inventories
# ===========================================================================

def test_the_capability_registries_are_exactly_what_k4_left():
    assert cpp.SUPPORTED_DTYPES == FLOATING_DTYPES
    assert cpp.INDEX_DTYPES == INDEX_DTYPES
    assert cpp.SUPPORTED_DEVICES == ("cpu",)
    assert cpp.UNSUPPORTED == ("cuda", "amp")
    assert cpp.RAW_KERNEL_DTYPES == ("float64",)
    assert cpp.normalize_dtype(None) == "float64"
    with pytest.raises(ValueError):
        cpp.normalize_dtype(INDEX_DTYPE)
    assert set(cpp._DTYPE_CODES) == set(FLOATING_DTYPES) | set(INDEX_DTYPES)
    info = cpp.backend_info()
    assert info["supported_dtypes"] == FLOATING_DTYPES
    assert info["index_dtypes"] == INDEX_DTYPES
    assert info["dtype"] == "float64"
    assert info["stable_framework_integration"] is False
    assert cpp.NATIVE_METRICS == ("native_accuracy",)


def test_no_operation_inventory_moved_at_k5():
    for landed in ("argmax", "index_select"):
        assert cpp.TENSOR_CORE_OPS.count(landed) == 1
        assert landed not in cpp.AUTOGRAD_OPS
        assert landed not in cpp.TENSOR_CORE_KERNELS
        assert landed not in cpp.RAW_KERNELS
    for inventory in (cpp.RAW_KERNELS, cpp.TENSOR_CORE_KERNELS,
                      cpp.TENSOR_CORE_OPS, cpp.AUTOGRAD_OPS,
                      cpp.NATIVE_MODULES, cpp.NATIVE_LOSSES,
                      cpp.NATIVE_METRICS, cpp.NATIVE_OPTIMIZERS,
                      cpp.STATE_SUPPORT):
        for banned in ("argmin", "gather", "scatter", "embedding", "int64",
                       "integer", "cast", "astype", "promote", "accuracy_"):
            assert not [name for name in inventory
                        if banned in name.lower()], (banned, inventory)
    for inventory in (cpp.RAW_KERNELS, cpp.TENSOR_CORE_KERNELS,
                      cpp.TENSOR_CORE_OPS, cpp.AUTOGRAD_OPS):
        for banned in ("max", "amax", "max_with_indices", "equal", "eq"):
            assert banned not in inventory, (banned, inventory)


def test_no_k6_or_later_surface_exists():
    """K5 is a proof, so every later milestone's deliverable must still be
    absent: no example, no benchmark, no CTest, no export, no public name."""
    for owner in (NativeTensor, cpp.NativeTensorCore, cpp.NativeStorage):
        for absent in ("argmin", "gather", "scatter", "scatter_add",
                       "embedding", "max", "amax", "take", "topk", "sort",
                       "argsort", "nonzero", "where", "bincount", "cumsum",
                       "index_add", "index_put", "masked_select",
                       "index_select_backward", "confusion_matrix"):
            assert not hasattr(owner, absent), (owner.__name__, absent)
    for present in ("argmax", "index_select"):
        assert hasattr(NativeTensor, present), present
        assert hasattr(cpp.NativeTensorCore, present), present
    examples = sorted(path.name for path in
                      (REPO_ROOT / "examples").glob("*.py"))
    benchmarks = sorted(path.name for path in
                        (REPO_ROOT / "benchmarks").glob("*.py"))
    assert len(examples) == EXAMPLE_COUNT, examples
    assert len(benchmarks) == BENCHMARK_COUNT, benchmarks
    assert not [name for name in examples if "integer" in name], examples
    assert not [name for name in benchmarks if "integer" in name], benchmarks


def test_the_c_abi_and_ctest_inventories_did_not_move_at_k5():
    exports = source_exports()
    assert len(exports) == EXPORT_COUNT, sorted(exports)
    for landed in ("tf_core_argmax", "tf_core_index_select"):
        assert landed in exports, landed
    for absent in ("tf_core_gather", "tf_core_scatter",
                   "tf_core_index_select_backward", "tf_core_argmin",
                   "tf_core_max", "tf_storage_dtype", "tf_core_equal"):
        assert absent not in exports, absent
    cmake = (REPO_ROOT / "cpp" / "CMakeLists.txt").read_text(encoding="utf-8")
    assert len(re.findall(r"^\s*add_test\(", cmake, re.M)) == CTEST_COUNT
    assert len(list((REPO_ROOT / "cpp" / "tests").glob("*.cpp"))) == \
        CTEST_COUNT
    assert len(cpp._CHECKED_KERNELS) == 38


def test_the_public_python_surface_did_not_move_at_k5():
    assert len(experimental.__all__) == EXPERIMENTAL_EXPORTS
    assert len(set(experimental.__all__)) == EXPERIMENTAL_EXPORTS
    for name in experimental.__all__:
        assert hasattr(experimental, name), name
    for absent in ("INDEX_DTYPES", "from_int64_array", "native_argmax",
                   "NativeIndexTensor", "native_confusion_matrix"):
        assert absent not in experimental.__all__, absent


def test_no_checkpoint_pipeline_or_metric_module_gained_an_integer_route():
    """The absence sweep, over executable code only, with the control that
    proves the reader is looking at the right modules."""
    banned = ("from_int64_array", "_from_int64_array", "INDEX_DTYPES",
              "_normalize_index_dtype", "_require_index_dtype",
              "index_select", "argmin", "gather", "scatter", "embedding")
    for module in ("native_checkpoint", "_native_checkpoint_transaction",
                   "native_dataset", "native_sampler", "native_data_loader",
                   "native_cross_entropy_loss", "native_sgd", "native_adam",
                   "native_optimizer_state", "_native_state"):
        names = code_names(f"{PACKAGE}/{module}.py")
        for absent in banned:
            assert absent not in names, (module, absent)
    # ...and the control: each module's own real names are found.
    assert "save_native_checkpoint" in code_names(
        f"{PACKAGE}/native_checkpoint.py")
    assert "NativeBatchSampler" in code_names(f"{PACKAGE}/native_sampler.py")
    assert "FORMAT_VERSION" in code_names(
        f"{PACKAGE}/native_optimizer_state.py")


def test_the_stable_line_still_knows_nothing_about_the_native_one():
    """K5 couples nothing: the stable framework neither imports the backend
    nor mentions the index dtype in its executable code."""
    for module in ("tensor.py", "data.py"):
        names = code_names(f"src/tensorforge/{module}")
        for absent in ("cpp", "NativeTensor", "from_int64_array",
                       "INDEX_DTYPES", "index_select"):
            assert absent not in names, (module, absent)
        # The control: the same reader finds each module's own real names,
        # so "absent" is a measurement rather than an empty scan.
        assert names, module
    assert {"Tensor", "backward", "numpy"} <= \
        code_names("src/tensorforge/tensor.py")
    assert "batches" in code_names("src/tensorforge/data.py")
    assert cpp.backend_info()["stable_framework_integration"] is False
