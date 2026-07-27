"""Phase-D completion guardrails: the invariants that span several native
CNN components at once (Advanced C++ Phase D, milestone D12).

The per-component suites already cover NativeFlatten (D1), the Conv2d
line (D2–D7), the MaxPool2d line (D8–D10), and the end-to-end training +
checkpoint-resume proof (D11) in depth. This file deliberately tests only
what those files cannot: the **interactions** — one graph carrying
convolution, pooling, flatten and linear together; shared modules and
shared inputs across branches; the two operations' *different* versioning
contracts meeting in one backward; state/checkpoint integration for a
model containing both; cross-layer failure atomicity; resource lifetime
across the whole stack; and the final capability boundary.

Nothing here adds numerical behavior. Every assertion is a property the
architecture promises, not an implementation detail.

Selector: python -m pytest -q -k native_phase_d
"""

import gc
import math

import numpy as np
import pytest

from tensorforge.backends import cpp
from tensorforge.experimental import (
    NativeAdam,
    NativeConv2d,
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
    save_native_checkpoint,
)

pytestmark = pytest.mark.skipif(
    not cpp.is_available(),
    reason="experimental C++ backend not built; " + cpp.build_instructions(),
)

needs_fault_injection = pytest.mark.skipif(
    not (cpp.is_available() and cpp.fault_injection_available()),
    reason="fault injection not compiled into the backend",
)


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
    lifetime (see the Phase-C/D failure tests)."""
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
# Shared fixtures for the integrated stack
# --------------------------------------------------------------------------

BATCH, IN_CHANNELS, HEIGHT, WIDTH = 3, 2, 6, 6
CONV_CHANNELS = 3
FEATURES = CONV_CHANNELS * 2 * 2


def _model(seed=0):
    """The canonical Phase-D stack: every CNN layer in one container."""
    return NativeSequential(
        NativeConv2d(IN_CHANNELS, CONV_CHANNELS, 2, seed=seed),
        NativeReLU(),
        NativeMaxPool2d(2),
        NativeFlatten(),
        NativeLinear(FEATURES, 2, seed=seed + 1),
    )


def _inputs(seed=7, requires_grad=False):
    values = np.round(
        np.random.default_rng(seed).standard_normal(
            (BATCH, IN_CHANNELS, HEIGHT, WIDTH)
        ),
        3,
    )
    return NativeTensor.from_array(values, requires_grad=requires_grad), values


def _targets(seed=8):
    values = np.round(
        np.random.default_rng(seed).standard_normal((BATCH, 2)), 3
    )
    return NativeTensor.from_array(values), values


def _close_all(*objects):
    for item in objects:
        if item is None:
            continue
        if isinstance(item, NativeModule):
            for parameter in item.parameters():
                parameter.close()
        else:
            item.close()


def _step(model, loss_fn, optimizer, x, y):
    prediction = model(x)
    loss = loss_fn(prediction, y)
    try:
        value = float(loss.to_numpy())
        loss.backward()
        optimizer.step()
    finally:
        loss.close()
        prediction.close()
    optimizer.zero_grad()
    return value


# --------------------------------------------------------------------------
# 1. The complete module stack
# --------------------------------------------------------------------------

def test_full_stack_shape_order_and_state_keys():
    model = _model()
    x, _ = _inputs()
    out = model(x)
    assert out.shape == (BATCH, 2)
    assert out.dtype == "float64" and out.device == "cpu"
    assert [name for name, _ in model.named_parameters()] == [
        "0.weight", "0.bias", "4.weight", "4.bias"
    ]
    assert sorted(model.state_dict()) == ["0.bias", "0.weight",
                                          "4.bias", "4.weight"]
    # ReLU (1), MaxPool2d (2), and Flatten (3) hold no tensor state.
    for index in (1, 2, 3):
        assert list(model[index].parameters()) == []
        assert list(model[index].buffers()) == []
        assert model[index].state_dict() == {}
    for name, parameter in model.named_parameters():
        assert parameter.dtype == "float64" and parameter.device == "cpu", name
    _close_all(out, x, model)


def test_full_stack_accepts_a_non_contiguous_input():
    _, values = _inputs(seed=11)
    owner = NativeTensor.from_array(
        np.ascontiguousarray(values.transpose(0, 1, 3, 2))
    )
    view = owner.transpose(0, 1, 3, 2)
    assert not view.contiguous
    model = _model()
    contiguous_input = NativeTensor.from_array(values)
    strided_out = model(view)
    contiguous_out = model(contiguous_input)
    assert np.allclose(strided_out.to_numpy(), contiguous_out.to_numpy(),
                       atol=1e-12)
    _close_all(strided_out, contiguous_out, view, owner, contiguous_input, model)


def test_stack_output_survives_every_intermediate(live_storages):
    # NativeSequential drops each intermediate as it advances; the result
    # must own its storage and stay readable after the input is closed.
    gc.collect()
    baseline = len(live_storages)
    model = _model()
    x, _ = _inputs()
    out = model(x)
    x.close()
    gc.collect()
    values = out.to_numpy()          # still readable after the input closed
    assert values.shape == (BATCH, 2) and np.isfinite(values).all()
    assert out.owns_core is True
    # The parameters require grad, so this forward built a graph, and the
    # graph legitimately keeps its intermediates alive until it is released
    # or dropped (the documented one-shot/`retain_graph` contract). Once the
    # output is closed *and* dropped, every intermediate goes with it.
    out.close()
    del out
    _close_all(model)
    gc.collect()
    assert len(live_storages) == baseline


# --------------------------------------------------------------------------
# 2. End-to-end autograd through every layer
# --------------------------------------------------------------------------

def test_one_graph_produces_every_expected_gradient():
    model = _model()
    loss_fn = NativeMSELoss()
    x, _ = _inputs(requires_grad=True)
    y, _ = _targets()
    prediction = model(x)
    loss = loss_fn(prediction, y)
    loss.backward()

    expected = {
        "0.weight": (CONV_CHANNELS, IN_CHANNELS, 2, 2),
        "0.bias": (CONV_CHANNELS,),
        "4.weight": (FEATURES, 2),
        "4.bias": (2,),
    }
    for name, parameter in model.named_parameters():
        grad = parameter.grad
        assert grad is not None, name
        assert grad.shape == expected[name], name
        values = grad.to_numpy()
        assert np.isfinite(values).all(), name
        assert (values != 0.0).any(), name
    assert x.grad is not None
    assert x.grad.shape == (BATCH, IN_CHANNELS, HEIGHT, WIDTH)
    assert np.isfinite(x.grad.to_numpy()).all()
    assert (x.grad.to_numpy() != 0.0).any()
    # Parameterless layers received nothing because they own nothing.
    assert list(model[1].parameters()) == []
    assert list(model[2].parameters()) == []
    assert list(model[3].parameters()) == []
    # The default backward released the whole graph, pooling winners included.
    assert loss._graph_freed is True
    _close_all(loss, prediction, x, y, model)


def test_graph_resources_are_released_after_the_default_backward():
    model = _model()
    loss_fn = NativeMSELoss()
    x, _ = _inputs()
    y, _ = _targets()
    # Run the layers explicitly so the pooling node (which owns the private
    # winner buffer) is observable.
    conv = model[0](x)
    relu = model[1](conv)
    pooled = model[2](relu)
    winners = pooled._graph_resources
    assert winners and all(not core._closed for core in winners)
    flat = model[3](pooled)
    prediction = model[4](flat)
    loss = loss_fn(prediction, y)
    loss.backward()
    assert all(core._closed for core in winners)
    assert pooled._graph_resources == ()
    _close_all(loss, prediction, flat, pooled, relu, conv, x, y, model)


def test_retain_graph_holds_winners_until_the_final_release():
    model = _model()
    loss_fn = NativeMSELoss()
    x, _ = _inputs()
    y, _ = _targets()
    conv = model[0](x)
    relu = model[1](conv)
    pooled = model[2](relu)
    winners = pooled._graph_resources[0]
    flat = model[3](pooled)
    prediction = model[4](flat)
    loss = loss_fn(prediction, y)
    loss.backward(retain_graph=True)
    assert not winners._closed              # kept for another pass
    first = model[0].weight.grad.to_numpy().copy()
    loss.backward(retain_graph=True)
    assert not winners._closed
    assert np.allclose(model[0].weight.grad.to_numpy(), 2 * first, atol=1e-12)
    loss.backward()                          # final, one-shot
    assert winners._closed is True
    _close_all(loss, prediction, flat, pooled, relu, conv, x, y, model)


# --------------------------------------------------------------------------
# 3. Shared graphs and gradient accumulation
# --------------------------------------------------------------------------

def test_one_conv_module_used_in_two_branches_accumulates():
    conv = NativeConv2d(IN_CHANNELS, CONV_CHANNELS, 2, seed=3)
    pool = NativeMaxPool2d(2)
    x, _ = _inputs(seed=13)
    branch_a = pool(conv(x)).sum()
    branch_b = conv(x).sum()
    total = branch_a.add(branch_b)
    total.backward()
    # One shared weight object, one accumulated gradient.
    assert conv.weight.grad is not None
    assert conv.weight.grad.shape == conv.weight.shape
    assert np.isfinite(conv.weight.grad.to_numpy()).all()
    # Identity is preserved: the module was reused, not duplicated.
    assert len({id(p) for p in conv.parameters()}) == 2
    _close_all(total, branch_a, branch_b, x, conv)


def test_one_input_shared_by_conv_and_pool_branches_accumulates():
    conv = NativeConv2d(IN_CHANNELS, CONV_CHANNELS, 2, seed=4)
    pool = NativeMaxPool2d(2)
    x, _ = _inputs(seed=14, requires_grad=True)
    conv_branch = conv(x).sum()
    pool_branch = pool(x).sum()
    total = conv_branch.add(pool_branch)
    total.backward()
    assert x.grad is not None
    assert x.grad.shape == (BATCH, IN_CHANNELS, HEIGHT, WIDTH)
    grad = x.grad.to_numpy()
    assert np.isfinite(grad).all() and (grad != 0.0).any()

    # The same two branches computed separately must sum to it.
    x_conv, _ = _inputs(seed=14, requires_grad=True)
    conv(x_conv).sum().backward()
    x_pool, _ = _inputs(seed=14, requires_grad=True)
    pool(x_pool).sum().backward()
    assert np.allclose(
        grad, x_conv.grad.to_numpy() + x_pool.grad.to_numpy(), atol=1e-12
    )
    _close_all(total, conv_branch, pool_branch, x, x_conv, x_pool, conv)


def test_shared_module_traversal_and_state_are_deduplicated():
    pool = NativeMaxPool2d(2)
    conv = NativeConv2d(IN_CHANNELS, CONV_CHANNELS, 2, seed=5)
    model = NativeSequential(conv, NativeReLU(), pool, NativeFlatten())
    twin = NativeSequential(conv, pool)          # the same two objects again
    assert sum(1 for m in model.modules() if m is pool) == 1
    assert sum(1 for m in twin.modules() if m is conv) == 1
    # A shared parameter appears once under its canonical key.
    assert sorted(model.state_dict()) == ["0.bias", "0.weight"]
    assert sorted(twin.state_dict()) == ["0.bias", "0.weight"]
    assert [id(p) for p in twin.parameters()] == [id(p) for p in conv.parameters()]
    _close_all(conv)


# --------------------------------------------------------------------------
# 4. The two versioning contracts meeting in one place
# --------------------------------------------------------------------------

def test_conv_input_grad_path_detects_a_stale_weight():
    conv = NativeConv2d(IN_CHANNELS, CONV_CHANNELS, 2, seed=6)
    x, _ = _inputs(seed=15, requires_grad=True)
    out = conv(x).sum()
    replacement = NativeTensor.from_array(conv.weight.to_numpy() + 1.0)
    conv.weight.copy_value_(replacement)         # bumps the weight's version
    with pytest.raises(RuntimeError, match="stale parameter value"):
        out.backward()
    assert x.grad is None                        # nothing was committed
    _close_all(out, replacement, x, conv)


def test_conv_weight_grad_path_detects_a_stale_versioned_input():
    conv = NativeConv2d(IN_CHANNELS, CONV_CHANNELS, 2, seed=7)
    parameter_input = NativeParameter(_inputs(seed=16)[1])
    out = conv(parameter_input).sum()
    replacement = NativeTensor.from_array(parameter_input.to_numpy() * 2.0)
    parameter_input.copy_value_(replacement)     # bumps the input's version
    with pytest.raises(RuntimeError, match="stale parameter value"):
        out.backward()
    assert conv.weight.grad is None
    _close_all(out, replacement, parameter_input, conv)


def test_conv_bias_only_backward_ignores_input_and_weight_mutation():
    conv = NativeConv2d(IN_CHANNELS, CONV_CHANNELS, 2, seed=8,
                        requires_grad=False)
    conv.bias._requires_grad = True              # bias-only gradient path
    parameter_input = NativeParameter(_inputs(seed=17)[1], requires_grad=False)
    out = conv(parameter_input).sum()
    for target in (conv.weight, parameter_input):
        replacement = NativeTensor.from_array(target.to_numpy() + 0.5)
        target.copy_value_(replacement)
        replacement.close()
    out.backward()                               # must NOT raise
    assert conv.bias.grad is not None
    assert conv.weight.grad is None
    _close_all(out, parameter_input, conv)


def test_maxpool_records_no_version_and_ignores_input_mutation():
    pool = NativeMaxPool2d(2)
    values = np.arange(16, dtype=float).reshape(1, 1, 4, 4)
    parameter_input = NativeParameter(values)
    out = pool(parameter_input)
    assert out._expected_versions == ()          # deliberately unversioned
    replacement = NativeTensor.from_array(np.zeros((1, 1, 4, 4)))
    parameter_input.copy_value_(replacement)     # bumps the version
    out.backward(gradient=NativeTensor.from_array(np.ones((1, 1, 2, 2))))
    # Routing still follows the winners saved at forward time.
    assert parameter_input.grad.to_numpy().tolist() == [[[
        [0, 0, 0, 0], [0, 1, 0, 1], [0, 0, 0, 0], [0, 1, 0, 1],
    ]]]
    _close_all(out, replacement, parameter_input)


def test_mixed_graph_versioning_is_per_operation():
    """In one graph containing both operations, the version contract is
    per-operation: the value-reading edges (the linear head's matmul) make
    the graph stale when their parameter is mutated, while the pooling node
    contributes no version entry at all."""
    model = _model()
    x, _ = _inputs(seed=18)
    conv = model[0](x)
    pooled = model[2](model[1](conv))
    assert pooled._expected_versions == ()      # pooling records nothing
    flat = model[3](pooled)
    out = model[4](flat).sum()
    # Mutating the linear weight — whose value the matmul backward rereads
    # — makes this mixed graph stale, deterministically and before any
    # gradient changes.
    replacement = NativeTensor.from_array(model[4].weight.to_numpy() + 1.0)
    model[4].weight.copy_value_(replacement)
    with pytest.raises(RuntimeError, match="stale parameter value"):
        out.backward()
    assert all(p.grad is None for p in model.parameters())
    _close_all(out, replacement, flat, pooled, conv, x, model)


# --------------------------------------------------------------------------
# 5. State and checkpoint integration
# --------------------------------------------------------------------------

def test_state_dict_snapshots_are_independent_and_load_preserves_identity():
    model = _model()
    snapshot = model.state_dict()
    identities = [id(p) for p in model.parameters()]
    # Mutating the live parameters must not change the snapshot.
    replacement = NativeTensor.from_array(
        np.zeros(model[0].weight.shape)
    )
    model[0].weight.copy_value_(replacement)
    assert not np.array_equal(
        snapshot["0.weight"].to_numpy(), model[0].weight.to_numpy()
    )
    result = model.load_state_dict(snapshot)
    assert list(result.missing_keys) == [] and list(result.unexpected_keys) == []
    assert [id(p) for p in model.parameters()] == identities
    assert np.array_equal(
        model[0].weight.to_numpy(), snapshot["0.weight"].to_numpy()
    )
    _close_all(replacement, model)
    for tensor in snapshot.values():
        tensor.close()


def test_invalid_state_load_is_atomic():
    model = _model()
    before = {n: p.to_numpy().copy() for n, p in model.named_parameters()}
    versions = [p.version for p in model.parameters()]
    bad = model.state_dict()
    bad["0.weight"] = NativeTensor.from_array(np.zeros((1, 1)))  # wrong shape
    with pytest.raises(Exception):
        model.load_state_dict(bad)
    for name, parameter in model.named_parameters():
        assert np.array_equal(parameter.to_numpy(), before[name]), name
    assert [p.version for p in model.parameters()] == versions
    for tensor in bad.values():
        tensor.close()
    _close_all(model)


@pytest.mark.parametrize("optimizer_factory", [NativeAdam, NativeSGD])
def test_checkpoint_resume_restores_a_cnn_exactly(tmp_path, optimizer_factory):
    loss_fn = NativeMSELoss()
    x, _ = _inputs(seed=19)
    y, _ = _targets(seed=20)
    model = _model()
    optimizer = optimizer_factory(model.parameters(), lr=0.02)
    for _ in range(4):
        _step(model, loss_fn, optimizer, x, y)
    path = str(tmp_path / "phase_d.npz")
    save_native_checkpoint(path, model, optimizer=optimizer,
                           metadata={"steps": 4})

    fresh = _model(seed=90)                       # different initialization
    fresh_optimizer = optimizer_factory(fresh.parameters(), lr=0.9)
    identities = [id(p) for p in fresh.parameters()]
    metadata = load_native_checkpoint(path, fresh, optimizer=fresh_optimizer)
    assert metadata == {"steps": 4}
    assert [id(p) for p in fresh.parameters()] == identities

    # Continuing both runs must stay in lockstep, exactly.
    continued = [_step(model, loss_fn, optimizer, x, y) for _ in range(3)]
    resumed = [_step(fresh, loss_fn, fresh_optimizer, x, y) for _ in range(3)]
    assert continued == resumed
    for a, b in zip(model.parameters(), fresh.parameters()):
        assert np.array_equal(a.to_numpy(), b.to_numpy())
    _close_all(x, y, model, fresh)
    for opt in (optimizer, fresh_optimizer):
        if hasattr(opt, "close"):     # only the stateful optimizer owns state
            opt.close()


def test_checkpoint_holds_no_transient_cnn_state(tmp_path):
    loss_fn = NativeMSELoss()
    x, _ = _inputs(seed=21)
    y, _ = _targets(seed=22)
    model = _model()
    optimizer = NativeAdam(model.parameters(), lr=0.02)
    _step(model, loss_fn, optimizer, x, y)
    path = str(tmp_path / "transients.npz")
    save_native_checkpoint(path, model, optimizer=optimizer)
    with np.load(path, allow_pickle=False) as archive:
        blob = (" ".join(archive.files) + " "
                + archive["manifest"].tobytes().decode("utf-8")).lower()
    # No transient CNN state of any kind is serialized. ("version" is not
    # banned: the manifest legitimately carries the *schema* versions
    # `format_version` / `state_format_version`, which are checked below.)
    for banned in ("winner", "grad", "graph", "relu", "pool", "flatten",
                   "output", "prediction"):
        assert banned not in blob, banned
    assert '"format_version": 2' in blob
    _close_all(x, y, model)
    optimizer.close()


def test_phase_c_style_mlp_checkpoint_still_works(tmp_path):
    # Phase D must not have disturbed the pre-existing (Phase-C) path.
    model = NativeSequential(NativeLinear(3, 4, seed=0), NativeReLU(),
                             NativeLinear(4, 1, seed=1))
    optimizer = NativeAdam(model.parameters(), lr=0.05)
    loss_fn = NativeMSELoss()
    x = NativeTensor.from_array(np.ones((2, 3)))
    y = NativeTensor.from_array(np.zeros((2, 1)))
    _step(model, loss_fn, optimizer, x, y)
    path = str(tmp_path / "mlp.npz")
    save_native_checkpoint(path, model, optimizer=optimizer)
    fresh = NativeSequential(NativeLinear(3, 4, seed=5), NativeReLU(),
                             NativeLinear(4, 1, seed=6))
    fresh_optimizer = NativeAdam(fresh.parameters(), lr=0.5)
    load_native_checkpoint(path, fresh, optimizer=fresh_optimizer)
    for a, b in zip(model.parameters(), fresh.parameters()):
        assert np.array_equal(a.to_numpy(), b.to_numpy())
    _close_all(x, y, model, fresh)
    optimizer.close()
    fresh_optimizer.close()


# --------------------------------------------------------------------------
# 6. Cross-layer failure atomicity
# --------------------------------------------------------------------------

@needs_fault_injection
@pytest.mark.parametrize("nth", [1, 2, 3, 4])
def test_backward_allocation_failure_is_atomic_at_every_stage(nth):
    """An injected failure at successive allocation points inside one
    mixed conv/pool/linear backward must roll every leaf gradient back to
    its pre-pass value — whichever gradient path was already running."""
    model = _model()
    loss_fn = NativeMSELoss()
    x, _ = _inputs(seed=23, requires_grad=True)
    y, _ = _targets(seed=24)

    # Seed a first, complete backward so rollback has prior values.
    first = loss_fn(model(x), y)
    first.backward()
    before = {n: p.grad.to_numpy().copy() for n, p in model.named_parameters()}
    input_before = x.grad.to_numpy().copy()
    versions = [p.version for p in model.parameters()]

    prediction = model(x)
    loss = loss_fn(prediction, y)
    cpp._arm_alloc_failure(nth)
    with pytest.raises(MemoryError):
        loss.backward()
    cpp._arm_alloc_failure(0)

    # Nothing partially committed: every leaf gradient is exactly the
    # pre-pass value, and no parameter value/version moved.
    for name, parameter in model.named_parameters():
        assert np.array_equal(parameter.grad.to_numpy(), before[name]), name
    assert np.array_equal(x.grad.to_numpy(), input_before)
    assert [p.version for p in model.parameters()] == versions
    assert cpp._require_library().tf_last_error_code() == cpp.TF_OK

    # A later fresh forward/backward succeeds.
    loss.close()
    prediction.close()
    recovered = loss_fn(model(x), y)
    recovered.backward()
    assert all(p.grad is not None for p in model.parameters())
    _close_all(recovered, first, x, y, model)


@needs_fault_injection
def test_forward_failure_after_earlier_layers_leaks_nothing(live_storages):
    model = _model()
    x, _ = _inputs(seed=25)
    gc.collect()
    baseline = len(live_storages)
    # Let the convolution succeed and fail a later allocation, so earlier
    # layers have already produced temporary outputs.
    cpp._arm_alloc_failure(3)
    with pytest.raises(MemoryError):
        model(x)
    cpp._arm_alloc_failure(0)
    gc.collect()
    assert len(live_storages) == baseline    # no output/copy/winner leak
    assert cpp._require_library().tf_last_error_code() == cpp.TF_OK
    out = model(x)                            # and the stack still works
    assert out.shape == (BATCH, 2)
    _close_all(out, x, model)


@needs_fault_injection
def test_maxpool_backward_allocation_failure_preserves_gradients():
    pool = NativeMaxPool2d(2)
    x, _ = _inputs(seed=26, requires_grad=True)
    first = pool(x).sum()
    first.backward()
    before = x.grad.to_numpy().copy()

    out = pool(x).sum()
    cpp._arm_alloc_failure(1)
    with pytest.raises(MemoryError):
        out.backward()
    cpp._arm_alloc_failure(0)
    assert np.array_equal(x.grad.to_numpy(), before)
    # The graph stayed retryable and its winners alive.
    assert out._graph_freed is False
    out.backward()
    assert np.allclose(x.grad.to_numpy(), 2 * before, atol=1e-12)
    _close_all(out, first, x)


def test_optimizer_and_checkpoint_validation_failures_change_nothing(tmp_path):
    model = _model()
    before = {n: p.to_numpy().copy() for n, p in model.named_parameters()}
    with pytest.raises((TypeError, ValueError)):
        NativeAdam(model.parameters(), lr=-0.5)
    with pytest.raises(Exception):
        load_native_checkpoint(str(tmp_path / "missing.npz"), model)
    for name, parameter in model.named_parameters():
        assert np.array_equal(parameter.to_numpy(), before[name]), name
    _close_all(model)


# --------------------------------------------------------------------------
# 7. Resource lifetime across the whole stack
# --------------------------------------------------------------------------

def test_repeated_training_steps_do_not_accumulate_storage(live_storages):
    model = _model()
    loss_fn = NativeMSELoss()
    optimizer = NativeAdam(model.parameters(), lr=0.01)
    x, _ = _inputs(seed=27)
    y, _ = _targets(seed=28)
    for _ in range(3):                    # warm up the optimizer's state
        _step(model, loss_fn, optimizer, x, y)
    gc.collect()
    baseline = len(live_storages)
    for _ in range(5):
        _step(model, loss_fn, optimizer, x, y)
        gc.collect()
        assert len(live_storages) == baseline
    _close_all(x, y, model)
    optimizer.close()


def test_graph_construction_failure_releases_outputs_and_winners(
    monkeypatch, live_storages
):
    pool = NativeMaxPool2d(2)
    x, _ = _inputs(seed=29, requires_grad=True)
    gc.collect()
    baseline = len(live_storages)

    def exploding_from_op(cls, *args, **kwargs):
        raise RuntimeError("simulated graph-construction failure")

    monkeypatch.setattr(NativeTensor, "_from_op", classmethod(exploding_from_op))
    with pytest.raises(RuntimeError, match="simulated graph-construction"):
        pool(x)
    monkeypatch.undo()
    gc.collect()
    assert len(live_storages) == baseline     # output and winners released
    assert x.closed is False
    out = pool(x)                              # still usable
    out.close()
    _close_all(x)


def test_explicit_close_releases_graph_resources(live_storages):
    pool = NativeMaxPool2d(2)
    x, _ = _inputs(seed=30, requires_grad=True)
    gc.collect()
    baseline = len(live_storages)
    out = pool(x)
    winners = out._graph_resources[0]
    assert not winners._closed
    out.close()                                # deterministic release point
    assert winners._closed is True
    gc.collect()
    assert len(live_storages) == baseline
    _close_all(x)


# --------------------------------------------------------------------------
# 8. The final Phase-D capability boundary
# --------------------------------------------------------------------------

def test_native_modules_inventory_is_final():
    for name in ("NativeFlatten", "NativeConv2d", "NativeMaxPool2d"):
        assert name in cpp.NATIVE_MODULES, name
    import tensorforge.experimental as experimental

    for name in cpp.NATIVE_MODULES:
        if name == "NativeModule":
            continue
        assert name in experimental.__all__, name


def test_autograd_and_core_operations_are_final():
    for op in ("conv2d", "maxpool2d"):
        assert op in cpp.AUTOGRAD_OPS, op
    for op in ("conv2d_forward", "conv2d_input_backward",
               "conv2d_weight_backward", "maxpool2d_forward",
               "maxpool2d_backward"):
        assert op in cpp.TENSOR_CORE_OPS, op
        assert hasattr(cpp.NativeTensorCore, op), op
    # The frozen historical registry and the NumPy-buffer kernel set keep
    # their established meanings — Phase D added to neither.
    assert cpp.TENSOR_CORE_KERNELS == ("relu", "add", "subtract",
                                       "multiply", "matmul")
    for name in cpp.RAW_KERNELS:
        assert callable(getattr(cpp, name)), name
        assert "conv2d" not in name and "maxpool2d" not in name


def test_checked_kernels_remain_the_error_hook_registry():
    library = cpp._require_library()
    for name in cpp._CHECKED_KERNELS:
        assert callable(getattr(library, name)), name
    # Membership is an error-contract property, not a capability claim:
    # it also holds non-compute entries such as the storage constructor.
    assert "tf_storage_create" in cpp._CHECKED_KERNELS
    for name in ("tf_core_conv2d_forward", "tf_core_conv2d_input_backward",
                 "tf_core_conv2d_weight_backward", "tf_core_maxpool2d_forward",
                 "tf_core_maxpool2d_backward"):
        assert name in cpp._CHECKED_KERNELS, name
        assert name not in cpp.RAW_KERNELS
        assert name not in cpp.TENSOR_CORE_OPS


def test_no_out_of_scope_capability_is_advertised():
    info = cpp.backend_info()
    assert info["supported_dtypes"] == ("float64",)
    assert info["supported_devices"] == ("cpu",)
    assert info["dtype"] == "float64" and info["device"] == "cpu"
    assert info["stable_framework_integration"] is False
    # Capabilities Phase D deliberately excluded and that no later
    # milestone has shipped either. ("softmax" and "log_softmax" were in
    # this list until Phase E milestones E3 and E4 implemented them —
    # their contracts now live in tests/test_native_softmax.py and
    # tests/test_native_log_softmax.py; "cross_entropy" left it at E5,
    # which shipped its Core layer as "cross_entropy_forward"/
    # "cross_entropy_backward", and E6 then shipped the differentiable
    # operation — see tests/test_native_cross_entropy_core.py and
    # tests/test_native_cross_entropy.py.)
    # ("layernorm" left UNSUPPORTED in Phase F milestone F2 and
    # "batchnorm" in F4, once both BatchNorm shapes shipped as composed
    # modules; neither is out-of-scope work any more.)
    for absent in ("float32", "cuda", "amp"):
        assert absent in cpp.UNSUPPORTED, absent
        assert absent not in cpp.AUTOGRAD_OPS
        assert absent not in cpp.TENSOR_CORE_OPS
        assert absent not in cpp.NATIVE_MODULES
    # "dropout" is still an unsupported *capability* — the boundary Phase
    # D drew and Phase G has not yet moved — even though Phase G
    # milestones G2 and G3 shipped a Core wrapper and a differentiable
    # operation underneath it. It leaves UNSUPPORTED only at the G10
    # closure, after the phase's reproducibility matrix has run.
    assert "dropout" in cpp.UNSUPPORTED
    assert "dropout" not in cpp.NATIVE_MODULES
    # The differentiable cross-entropy operation shipped at E6 and is
    # reported as an autograd operation, not as a Core wrapper.
    assert "cross_entropy" in cpp.AUTOGRAD_OPS
    assert "cross_entropy" not in cpp.TENSOR_CORE_OPS
    assert "cross_entropy" not in cpp.UNSUPPORTED
    # E7's loss module and metric shipped too, into their own layer
    # inventories rather than into any operation inventory.
    assert "NativeCrossEntropyLoss" in cpp.NATIVE_LOSSES
    assert "native_accuracy" in cpp.NATIVE_METRICS
    for shipped in ("NativeCrossEntropyLoss", "native_accuracy"):
        assert shipped not in cpp.UNSUPPORTED, shipped
        assert shipped not in cpp.AUTOGRAD_OPS, shipped
        assert shipped not in cpp.NATIVE_MODULES, shipped
    # Nothing Phase D shipped may still be listed as unsupported.
    for shipped in ("conv2d", "maxpool2d", "flatten", "NativeConv2d",
                    "NativeMaxPool2d", "NativeFlatten"):
        assert shipped not in cpp.UNSUPPORTED, shipped


def test_support_matrix_agrees_with_the_backend_inventories():
    from pathlib import Path

    matrix = (Path(__file__).resolve().parent.parent / "docs"
              / "native_support_matrix.md").read_text(encoding="utf-8")
    for name in cpp.NATIVE_MODULES:
        assert name in matrix, name
    for op in cpp.AUTOGRAD_OPS:
        assert f"`{op}`" in matrix, op
    # Everything the registry calls unsupported is named in the doc too.
    for name in ("CUDA", "float32", "AMP"):
        assert name in matrix, name


def test_stable_and_native_lines_stay_separate():
    import tensorforge
    import tensorforge.experimental as experimental

    for name in experimental.__all__:
        assert not hasattr(tensorforge, name), name
        assert not hasattr(tensorforge.nn, name), name
    # The native CNN modules are not reachable from the stable frontend.
    assert not hasattr(tensorforge.nn.Conv2d, "maxpool2d")
    x = NativeTensor.from_array(np.ones((1, 1, 4, 4)))
    with pytest.raises(TypeError):
        NativeMaxPool2d(2)(tensorforge.Tensor(np.ones((1, 1, 4, 4))))
    x.close()


# --------------------------------------------------------------------------
# 9. Phase-D artifacts exist and stay runnable
# --------------------------------------------------------------------------

def test_phase_d_artifacts_are_present():
    from pathlib import Path

    root = Path(__file__).resolve().parent.parent
    assert (root / "examples" / "native_cnn_training.py").is_file()
    assert (root / "benchmarks" / "benchmark_native_cnn.py").is_file()
    assert (root / "cpp" / "src" / "conv2d.cpp").is_file()
    assert (root / "cpp" / "src" / "pooling.cpp").is_file()
    assert (root / "cpp" / "include" / "tf_conv2d_internal.h").is_file()
    assert (root / "cpp" / "include" / "tf_pooling_internal.h").is_file()
    for name in ("test_conv2d_forward", "test_conv2d_input_backward",
                 "test_conv2d_weight_backward", "test_maxpool2d_forward",
                 "test_maxpool2d_backward"):
        assert (root / "cpp" / "tests" / f"{name}.cpp").is_file(), name
