"""NativeMaxPool2d — the native max-pooling module (Phase D, milestone
D10).

D10 exposes the differentiable D8/D9 ``NativeTensor.maxpool2d`` operation
as a parameter-free, buffer-free ``NativeModule``. The module adds no
numerical or autograd logic: it validates and normalizes
``kernel_size``/``stride``/``padding`` in its constructor and delegates
forward to the operation. These tests cover the constructor contract and
normalized attributes, the empty parameter/buffer/state surface, forward
correctness and stable parity, inherited autograd (including the winner
lifetime that stays with the output graph, never the module),
``NativeSequential`` composition, checkpoint and optimizer compatibility,
ownership/failure behavior, and the capability split.

Backend-dependent, so the module skips cleanly when the compiled backend
is not built. Cleanup is explicit via close().

Selector: python -m pytest -q -k native_maxpool2d_module
"""

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
    live-native-allocation count for the ownership tests."""
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
# Helpers
# --------------------------------------------------------------------------

def _stable(x, kernel_size, stride=None, padding=0, g=None):
    """Stable ``tensorforge.nn.MaxPool2d`` forward (and, given ``g``, the
    input gradient) — the numerical reference."""
    from tensorforge.nn import MaxPool2d
    from tensorforge.tensor import Tensor

    xt = Tensor(np.array(x, float), requires_grad=True)
    out = MaxPool2d(kernel_size, stride=stride, padding=padding)(xt)
    if g is None:
        return out.data, None
    (out * Tensor(np.array(g, float))).sum().backward()
    return out.data, xt.grad


# --------------------------------------------------------------------------
# Constructor and attributes
# --------------------------------------------------------------------------

def test_default_stride_equals_kernel_size():
    layer = NativeMaxPool2d(2)
    assert layer.kernel_size == (2, 2)
    assert layer.stride == (2, 2)      # stride=None -> non-overlapping
    assert layer.padding == (0, 0)


def test_explicit_integer_stride_and_padding_normalize():
    layer = NativeMaxPool2d(3, stride=1, padding=1)
    assert layer.kernel_size == (3, 3)
    assert layer.stride == (1, 1)
    assert layer.padding == (1, 1)


def test_tuple_arguments_are_stored_as_normalized_pairs():
    layer = NativeMaxPool2d((3, 2), stride=(2, 1), padding=(1, 0))
    assert layer.kernel_size == (3, 2)
    assert layer.stride == (2, 1)
    assert layer.padding == (1, 0)
    assert all(isinstance(v, int) for v in layer.kernel_size + layer.stride)


def test_tuple_kernel_with_default_stride_copies_the_pair():
    layer = NativeMaxPool2d((3, 2))
    assert layer.stride == (3, 2)


def test_list_arguments_normalize_to_tuples():
    # The established native spatial helper accepts a 2-element list.
    layer = NativeMaxPool2d([2, 3], stride=[1, 2], padding=[0, 1])
    assert layer.kernel_size == (2, 3)
    assert layer.stride == (1, 2)
    assert layer.padding == (0, 1)


@pytest.mark.parametrize(
    "kernel_size",
    [0, -1, (0, 2), (2, -3), (2,), (2, 2, 2), 1.5, (2, 1.5), "2", None,
     True, (True, 2)],
)
def test_invalid_kernel_size_rejected(kernel_size):
    with pytest.raises(ValueError):
        NativeMaxPool2d(kernel_size)


@pytest.mark.parametrize(
    "stride", [0, -2, (1, 0), (1,), (1, 1, 1), 1.5, (1, 1.5), "1", True]
)
def test_invalid_stride_rejected(stride):
    with pytest.raises(ValueError):
        NativeMaxPool2d(2, stride=stride)


@pytest.mark.parametrize(
    "padding", [-1, (0, -1), (0,), (0, 0, 0), 0.5, (0, 0.5), "0", True, False]
)
def test_invalid_padding_rejected(padding):
    with pytest.raises(ValueError):
        NativeMaxPool2d(2, padding=padding)


def test_repr_reports_the_normalized_configuration():
    assert repr(NativeMaxPool2d(2)) == (
        "NativeMaxPool2d(kernel_size=(2, 2), stride=(2, 2), padding=(0, 0))"
    )
    assert repr(NativeMaxPool2d((3, 2), stride=1, padding=(1, 0))) == (
        "NativeMaxPool2d(kernel_size=(3, 2), stride=(1, 1), padding=(1, 0))"
    )
    # The repr never materializes tensor values or exposes saved winners.
    text = repr(NativeMaxPool2d(2))
    assert "winner" not in text.lower() and "NativeTensor" not in text


# --------------------------------------------------------------------------
# Module state: no parameters, no buffers, no state keys
# --------------------------------------------------------------------------

def test_is_a_native_module_with_no_trainable_state():
    layer = NativeMaxPool2d(2)
    assert isinstance(layer, NativeModule)
    assert list(layer.parameters()) == []
    assert list(layer.named_parameters()) == []
    assert list(layer.buffers()) == []
    assert list(layer.named_buffers()) == []
    assert layer.state_dict() == {}
    assert list(layer.modules()) == [layer]


def test_load_state_dict_accepts_an_empty_mapping():
    layer = NativeMaxPool2d(2)
    result = layer.load_state_dict({})
    assert list(result.missing_keys) == []
    assert list(result.unexpected_keys) == []


def test_unexpected_state_key_is_rejected_strictly():
    layer = NativeMaxPool2d(2)
    with pytest.raises(Exception):
        layer.load_state_dict({"kernel_size": NativeTensor.from_array([1.0])})
    # Non-strict loading reports it instead of raising, per existing rules.
    result = layer.load_state_dict(
        {"weight": NativeTensor.from_array([1.0])}, strict=False
    )
    assert "weight" in result.unexpected_keys
    assert layer.state_dict() == {}  # still empty


def test_configuration_is_not_serialized_as_tensor_state():
    layer = NativeMaxPool2d((3, 2), stride=(2, 1), padding=(1, 0))
    assert layer.state_dict() == {}
    # Architecture lives in the constructor, not in state.
    assert layer.kernel_size == (3, 2)


def test_train_eval_mode_changes_nothing_numerically():
    x = np.random.default_rng(1).standard_normal((1, 2, 4, 4))
    layer = NativeMaxPool2d(2)
    xi = NativeTensor.from_array(x)
    layer.train()
    assert layer.training is True
    train_out = layer(xi)
    layer.eval()
    assert layer.training is False
    eval_out = layer(xi)
    assert np.array_equal(train_out.to_numpy(), eval_out.to_numpy())
    for t in (train_out, eval_out, xi):
        t.close()


def test_registers_as_a_child_of_a_custom_module():
    class Net(NativeModule):
        def __init__(self):
            super().__init__()
            self.conv = NativeConv2d(1, 2, 2, seed=0)
            self.pool = NativeMaxPool2d(2)

        def forward(self, x):
            return self.pool(self.conv(x))

    net = Net()
    assert net.pool in net.modules()
    # Pooling contributes no keys to the parent's state.
    assert sorted(net.state_dict()) == ["conv.bias", "conv.weight"]
    assert [name for name, _ in net.named_parameters()] == [
        "conv.weight", "conv.bias"
    ]
    x = NativeTensor.from_array(np.random.default_rng(2).standard_normal((1, 1, 4, 4)))
    out = net(x)
    assert out.shape == (1, 2, 1, 1)
    out.close()
    x.close()
    for p in net.parameters():
        p.close()


def test_shared_module_instance_is_deduplicated_in_traversal():
    pool = NativeMaxPool2d(2)
    model = NativeSequential(pool, NativeReLU(), pool)
    # Identity deduplication: the shared instance appears once.
    assert sum(1 for m in model.modules() if m is pool) == 1
    assert model.state_dict() == {}


# --------------------------------------------------------------------------
# Forward
# --------------------------------------------------------------------------

def test_hand_computed_forward():
    x = np.arange(1, 17, dtype=float).reshape(1, 1, 4, 4)
    xi = NativeTensor.from_array(x)
    out = NativeMaxPool2d(2)(xi)
    assert out.to_numpy().tolist() == [[[[6.0, 8.0], [14.0, 16.0]]]]
    out.close()
    xi.close()


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
def test_forward_parity_with_stable_maxpool2d(kernel_size, stride, padding):
    x = np.round(np.random.default_rng(3).standard_normal((2, 3, 6, 5)) * 3, 3)
    xi = NativeTensor.from_array(x)
    out = NativeMaxPool2d(kernel_size, stride=stride, padding=padding)(xi)
    reference, _ = _stable(x, kernel_size, stride=stride, padding=padding)
    assert np.array_equal(out.to_numpy(), reference)  # selection is exact
    out.close()
    xi.close()


def test_integer_and_tuple_configurations_agree():
    x = np.random.default_rng(4).standard_normal((1, 2, 5, 5))
    xi = NativeTensor.from_array(x)
    a = NativeMaxPool2d(2, stride=2, padding=1)(xi)
    b = NativeMaxPool2d((2, 2), stride=(2, 2), padding=(1, 1))(xi)
    assert np.array_equal(a.to_numpy(), b.to_numpy())
    for t in (a, b, xi):
        t.close()


def test_batch_channels_negative_and_fractional_values():
    x = np.array([
        [[[-4.0, -1.5], [-3.25, -2.0]], [[0.5, 0.25], [1.75, 1.5]]],
        [[[9.5, -9.5], [0.0, 0.125]], [[-0.5, -0.25], [-1.0, -2.0]]],
    ])
    xi = NativeTensor.from_array(x)
    out = NativeMaxPool2d(2)(xi)
    assert out.to_numpy().tolist() == [
        [[[-1.5]], [[1.75]]],
        [[[9.5]], [[-0.25]]],
    ]
    out.close()
    xi.close()


def test_tie_and_negative_infinity_and_full_padding_follow_the_operation():
    # Tie: the first row-major maximum wins (checked through the gradient).
    tie = NativeParameter(np.array([[[[1.0, 5.0], [5.0, 2.0]]]]))
    y = NativeMaxPool2d(2)(tie)
    y.backward(gradient=NativeTensor.from_array(np.array([[[[1.0]]]])))
    assert tie.grad.to_numpy().tolist() == [[[[0.0, 1.0], [0.0, 0.0]]]]
    y.close()
    tie.close()
    # All -inf without padding: the output is -inf.
    neg = NativeTensor.from_array(np.full((1, 1, 2, 2), NEG_INF))
    out = NativeMaxPool2d(2)(neg)
    assert np.isneginf(out.to_numpy()[0, 0, 0, 0])
    out.close()
    neg.close()
    # Completely padded windows: -inf output, and their gradient is dropped.
    tiny = NativeParameter(np.array([[[[4.0]]]]))
    padded = NativeMaxPool2d(1, stride=1, padding=1)(tiny)
    assert padded.shape == (1, 1, 3, 3)
    assert np.isneginf(padded.to_numpy()[0, 0, 0, 0])
    padded.backward(gradient=NativeTensor.from_array(np.ones((1, 1, 3, 3))))
    assert tiny.grad.to_numpy().tolist() == [[[[1.0]]]]
    padded.close()
    tiny.close()


def test_non_contiguous_input_is_supported():
    values = np.random.default_rng(5).standard_normal((2, 2, 4, 5))
    owner = NativeTensor.from_array(
        np.ascontiguousarray(values.transpose(0, 1, 3, 2))
    )
    view = owner.transpose(0, 1, 3, 2)
    assert not view.contiguous
    out = NativeMaxPool2d(2)(view)
    reference, _ = _stable(values, 2)
    assert np.allclose(out.to_numpy(), reference, atol=1e-12)
    for t in (out, view, owner):
        t.close()


def test_output_owns_its_storage_and_outlives_the_input():
    # arange(16) -> the 2x2 window maxima are 5, 7, 13, 15.
    xi = NativeTensor.from_array(np.arange(16, dtype=float).reshape(1, 1, 4, 4))
    out = NativeMaxPool2d(2)(xi)
    assert out.owns_core is True
    assert out.contiguous
    xi.close()
    assert out.to_numpy().tolist() == [[[[5.0, 7.0], [13.0, 15.0]]]]
    out.close()


@pytest.mark.parametrize(
    "bad", [np.ones((1, 1, 4, 4)), [[1.0]], 3.0, None]
)
def test_non_native_input_rejected(bad):
    with pytest.raises(TypeError):
        NativeMaxPool2d(2)(bad)


def test_stable_tensor_input_rejected():
    from tensorforge.tensor import Tensor

    with pytest.raises(TypeError):
        NativeMaxPool2d(2)(Tensor(np.ones((1, 1, 4, 4))))


def test_closed_input_rejected():
    xi = NativeTensor.from_array(np.ones((1, 1, 4, 4)))
    xi.close()
    with pytest.raises(RuntimeError, match="closed"):
        NativeMaxPool2d(2)(xi)


@pytest.mark.parametrize("shape", [(4, 4), (1, 4, 4), (1, 1, 1, 4, 4)])
def test_rank_other_than_four_rejected(shape):
    xi = NativeTensor.from_array(np.ones(shape))
    with pytest.raises(ValueError, match="4-D NCHW"):
        NativeMaxPool2d(2)(xi)
    xi.close()


def test_window_larger_than_padded_input_rejected():
    xi = NativeTensor.from_array(np.ones((1, 1, 3, 3)))
    with pytest.raises(ValueError, match="does not fit"):
        NativeMaxPool2d(5)(xi)
    # The input is untouched and still usable.
    assert xi.closed is False
    assert xi.to_numpy().shape == (1, 1, 3, 3)
    xi.close()


# --------------------------------------------------------------------------
# Autograd (entirely inherited from the D8/D9 operation)
# --------------------------------------------------------------------------

def test_input_gradient_matches_stable():
    x = np.round(np.random.default_rng(6).standard_normal((2, 2, 5, 4)) * 3, 3)
    xi = NativeParameter(x)
    out = NativeMaxPool2d((3, 2), stride=(2, 1), padding=(1, 0))(xi)
    g = np.random.default_rng(7).standard_normal(out.shape)
    _, reference = _stable(x, (3, 2), stride=(2, 1), padding=(1, 0), g=g)
    out.backward(gradient=NativeTensor.from_array(g))
    assert np.allclose(xi.grad.to_numpy(), reference, atol=1e-12)
    out.close()
    xi.close()


def test_overlapping_windows_accumulate():
    x = np.array([[[[1.0, 2.0, 3.0], [4.0, 9.0, 5.0], [6.0, 7.0, 8.0]]]])
    xi = NativeParameter(x)
    out = NativeMaxPool2d(2, stride=1)(xi)
    out.backward(gradient=NativeTensor.from_array(np.ones((1, 1, 2, 2))))
    assert xi.grad.to_numpy().tolist() == [[[[0, 0, 0], [0, 4.0, 0], [0, 0, 0]]]]
    out.close()
    xi.close()


def test_module_defines_no_custom_backward():
    layer = NativeMaxPool2d(2)
    # The module contributes no backward machinery of its own; the graph
    # node is the operation's.
    assert not hasattr(layer, "backward")
    assert not hasattr(layer, "_backward")
    xi = NativeParameter(np.ones((1, 1, 4, 4)))
    out = layer(xi)
    assert out._op == "maxpool2d"
    assert out._parents == (xi,)
    out.close()
    xi.close()


def test_no_version_snapshot_and_mutation_does_not_change_routing():
    x = np.arange(16, dtype=float).reshape(1, 1, 4, 4)
    xi = NativeParameter(x)
    out = NativeMaxPool2d(2)(xi)
    assert out._expected_versions == ()
    replacement = NativeTensor.from_array(np.zeros((1, 1, 4, 4)))
    xi.copy_value_(replacement)          # bumps the value version
    out.backward(gradient=NativeTensor.from_array(np.ones((1, 1, 2, 2))))
    # Routing still follows the winners saved at forward time.
    assert xi.grad.to_numpy().tolist() == [[[
        [0, 0, 0, 0],
        [0, 1, 0, 1],
        [0, 0, 0, 0],
        [0, 1, 0, 1],
    ]]]
    for t in (out, replacement, xi):
        t.close()


def test_retain_graph_and_freed_history_behavior():
    xi = NativeParameter(np.random.default_rng(8).standard_normal((1, 1, 4, 4)))
    out = NativeMaxPool2d(2)(xi)
    saved = out._graph_resources[0]
    g = NativeTensor.from_array(np.ones((1, 1, 2, 2)))
    out.backward(gradient=g, retain_graph=True)
    assert not saved._closed          # winners kept for another pass
    once = xi.grad.to_numpy().copy()
    out.backward(gradient=g)          # one-shot: frees history and winners
    assert np.allclose(xi.grad.to_numpy(), 2 * once, atol=1e-12)
    assert saved._closed is True
    assert out._graph_resources == ()
    with pytest.raises(RuntimeError, match="freed autograd graph"):
        out.backward(gradient=g)
    for t in (out, g, xi):
        t.close()


def test_no_grad_input_builds_no_graph_and_keeps_no_winners():
    xi = NativeTensor.from_array(np.ones((1, 1, 4, 4)))
    assert xi.requires_grad is False
    out = NativeMaxPool2d(2)(xi)
    assert out.requires_grad is False
    assert out.is_leaf is True
    assert out._graph_resources == ()  # released immediately
    out.close()
    xi.close()


def test_shared_input_through_two_pooling_modules_accumulates():
    x = np.random.default_rng(9).standard_normal((1, 1, 4, 4))
    xi = NativeParameter(x)
    a = NativeMaxPool2d(2)(xi)
    b = NativeMaxPool2d(2, stride=1)(xi)
    total = a.sum().add(b.sum())
    total.backward()
    _, grad_a = _stable(x, 2, g=np.ones(a.shape))
    _, grad_b = _stable(x, 2, stride=1, g=np.ones(b.shape))
    assert np.allclose(xi.grad.to_numpy(), grad_a + grad_b, atol=1e-12)
    for t in (total, a, b, xi):
        t.close()


def test_zero_grad_on_the_parent_model_succeeds():
    model = NativeSequential(NativeConv2d(1, 2, 2, seed=0), NativeMaxPool2d(2))
    xi = NativeParameter(np.random.default_rng(10).standard_normal((1, 1, 4, 4)))
    out = model(xi)
    out.backward(gradient=NativeTensor.from_array(np.ones(out.shape)))
    assert model[0].weight.grad is not None
    model.zero_grad()
    assert model[0].weight.grad is None
    assert model[0].bias.grad is None
    out.close()
    xi.close()
    for p in model.parameters():
        p.close()


def test_module_never_stores_winner_state_between_calls():
    layer = NativeMaxPool2d(2)
    xi = NativeParameter(np.random.default_rng(11).standard_normal((1, 1, 4, 4)))
    first = layer(xi)
    second = layer(xi)
    # Each call owns its own winner resource; the module holds none.
    assert first._graph_resources and second._graph_resources
    assert first._graph_resources[0] is not second._graph_resources[0]
    assert not hasattr(layer, "_graph_resources")
    assert not [name for name in vars(layer) if "winner" in name.lower()]
    assert layer.state_dict() == {}
    for t in (first, second, xi):
        t.close()


# --------------------------------------------------------------------------
# NativeSequential integration
# --------------------------------------------------------------------------

def _cnn_stack(with_flatten=False, with_linear=False, seed=0):
    layers = [NativeConv2d(1, 2, 2, seed=seed), NativeReLU(), NativeMaxPool2d(2)]
    if with_flatten:
        layers.append(NativeFlatten())
    if with_linear:
        layers.append(NativeLinear(2 * 2 * 2, 3, seed=seed + 1))
    return NativeSequential(*layers)


def test_sequential_conv_relu_pool_shapes():
    model = _cnn_stack()
    xi = NativeTensor.from_array(np.random.default_rng(12).standard_normal((2, 1, 6, 6)))
    out = model(xi)
    assert out.shape == (2, 2, 2, 2)  # conv -> 5x5, pool 2 -> 2x2
    out.close()
    xi.close()
    for p in model.parameters():
        p.close()


def test_sequential_with_flatten_shapes_and_state_keys():
    model = _cnn_stack(with_flatten=True)
    xi = NativeTensor.from_array(np.random.default_rng(13).standard_normal((2, 1, 6, 6)))
    out = model(xi)
    assert out.shape == (2, 8)
    # Pooling (slot 2), ReLU, and Flatten contribute no state keys.
    assert sorted(model.state_dict()) == ["0.bias", "0.weight"]
    assert [name for name, _ in model.named_parameters()] == [
        "0.weight", "0.bias"
    ]
    out.close()
    xi.close()
    for p in model.parameters():
        p.close()


def test_full_stack_forward_backward_reaches_every_parameter():
    model = _cnn_stack(with_flatten=True, with_linear=True)
    xi = NativeParameter(np.random.default_rng(14).standard_normal((2, 1, 6, 6)))
    out = model(xi)
    assert out.shape == (2, 3)
    loss = out.sum()
    loss.backward()
    conv, linear = model[0], model[4]
    assert conv.weight.grad is not None and conv.bias.grad is not None
    assert linear.weight.grad is not None and linear.bias.grad is not None
    assert xi.grad is not None and xi.grad.shape == (2, 1, 6, 6)
    # Pooling contributed no parameters and no state keys.
    assert sorted(model.state_dict()) == [
        "0.bias", "0.weight", "4.bias", "4.weight"
    ]
    assert [name for name, _ in model.named_parameters()] == [
        "0.weight", "0.bias", "4.weight", "4.bias"
    ]
    for t in (loss, out, xi):
        t.close()
    for p in model.parameters():
        p.close()


def test_pooling_module_reused_in_two_sequential_slots():
    pool = NativeMaxPool2d(2)
    model = NativeSequential(NativeConv2d(1, 2, 3, padding=1, seed=2), pool, pool)
    xi = NativeParameter(np.random.default_rng(15).standard_normal((1, 1, 8, 8)))
    out = model(xi)
    assert out.shape == (1, 2, 2, 2)  # 8x8 -> conv 8x8 -> pool 4x4 -> pool 2x2
    out.sum().backward()
    assert xi.grad.shape == (1, 1, 8, 8)
    assert model.state_dict().keys() == {"0.weight", "0.bias"}
    out.close()
    xi.close()
    for p in model.parameters():
        p.close()


def test_outputs_stay_valid_after_sequential_rebinds_intermediates():
    # NativeSequential drops each intermediate as the loop advances; the
    # final output must not borrow any of them.
    model = _cnn_stack(with_flatten=True)
    xi = NativeTensor.from_array(np.random.default_rng(16).standard_normal((2, 1, 6, 6)))
    out = model(xi)
    xi.close()
    values = out.to_numpy()  # still readable: no dangling storage
    assert values.shape == (2, 8)
    assert np.isfinite(values).all()
    out.close()
    for p in model.parameters():
        p.close()


# --------------------------------------------------------------------------
# Checkpoint and optimizer compatibility
# --------------------------------------------------------------------------

def _tiny_cnn(seed=0):
    return NativeSequential(
        NativeConv2d(1, 2, 2, seed=seed),
        NativeReLU(),
        NativeMaxPool2d(2),
        NativeFlatten(),
        NativeLinear(2 * 2 * 2, 3, seed=seed + 1),
    )


def test_model_only_checkpoint_round_trip(tmp_path):
    model = _tiny_cnn()
    xi = NativeTensor.from_array(np.random.default_rng(17).standard_normal((2, 1, 6, 6)))
    before = model(xi)
    before_values = before.to_numpy().copy()
    path = str(tmp_path / "pool_model.npz")
    save_native_checkpoint(path, model)

    restored = _tiny_cnn(seed=50)  # different init
    identities = [id(p) for p in restored.parameters()]
    metadata = load_native_checkpoint(path, restored)
    assert metadata == {}
    assert [id(p) for p in restored.parameters()] == identities  # preserved
    after = restored(xi)
    assert np.array_equal(after.to_numpy(), before_values)

    # No pooling entries in the archive: only Conv2d/Linear parameters.
    with np.load(path, allow_pickle=False) as archive:
        blob = " ".join(archive.files) + " " + str(
            archive["manifest"].tobytes().decode("utf-8", "replace")
        )
    assert "maxpool" not in blob.lower()
    assert "winner" not in blob.lower()
    assert "pool" not in blob.lower()

    for t in (before, after, xi):
        t.close()
    for p in model.parameters():
        p.close()
    for p in restored.parameters():
        p.close()


@pytest.mark.parametrize("optimizer_factory", [NativeSGD, NativeAdam])
def test_optimizer_checkpoint_round_trip_and_step(tmp_path, optimizer_factory):
    model = _tiny_cnn(seed=3)
    optimizer = optimizer_factory(model.parameters(), lr=0.01)
    # Pooling contributes nothing to the optimizer's parameter list.
    assert len(model.parameters()) == 4  # conv w/b + linear w/b
    xi = NativeTensor.from_array(np.random.default_rng(18).standard_normal((2, 1, 6, 6)))
    out = model(xi)
    out.sum().backward()
    optimizer.step()
    optimizer.zero_grad()

    state = optimizer.state_dict()
    assert "maxpool" not in str(state).lower()
    path = str(tmp_path / "pool_opt.npz")
    save_native_checkpoint(path, model, optimizer)

    restored = _tiny_cnn(seed=60)
    restored_optimizer = optimizer_factory(restored.parameters(), lr=0.5)
    identities = [id(p) for p in restored.parameters()]
    load_native_checkpoint(path, restored, restored_optimizer)
    assert [id(p) for p in restored.parameters()] == identities
    assert restored_optimizer.state_dict()["lr"] == pytest.approx(0.01)
    a = model(xi)
    b = restored(xi)
    assert np.array_equal(a.to_numpy(), b.to_numpy())
    for t in (out, a, b, xi):
        t.close()
    for p in list(model.parameters()) + list(restored.parameters()):
        p.close()
    if hasattr(optimizer, "close"):
        optimizer.close()
    if hasattr(restored_optimizer, "close"):
        restored_optimizer.close()


def test_checkpoint_format_version_is_unchanged(tmp_path):
    model = _tiny_cnn()
    path = str(tmp_path / "version.npz")
    save_native_checkpoint(path, model)
    with np.load(path, allow_pickle=False) as archive:
        manifest = archive["manifest"].tobytes().decode("utf-8")
    assert '"tensorforge.native_checkpoint"' in manifest
    assert '"format_version": 3' in manifest  # schema unchanged by D10
    for p in model.parameters():
        p.close()


# --------------------------------------------------------------------------
# Ownership and failure behavior
# --------------------------------------------------------------------------

def test_invalid_construction_allocates_nothing(live_storages):
    baseline = len(live_storages)
    for bad in (0, -1, (2,), True, 1.5):
        with pytest.raises(ValueError):
            NativeMaxPool2d(bad)
    assert len(live_storages) == baseline


def test_failed_forward_leaves_the_input_open_and_unchanged(live_storages):
    values = np.arange(9, dtype=float).reshape(1, 1, 3, 3)
    xi = NativeTensor.from_array(values)
    baseline = len(live_storages)
    with pytest.raises(ValueError):
        NativeMaxPool2d(5)(xi)          # window larger than the input
    assert len(live_storages) == baseline
    assert xi.closed is False
    assert np.array_equal(xi.to_numpy(), values)
    xi.close()


@needs_fault_injection
def test_forward_allocation_failure_is_atomic(live_storages):
    values = np.arange(16, dtype=float).reshape(1, 1, 4, 4)
    xi = NativeParameter(values)
    layer = NativeMaxPool2d(2)
    baseline = len(live_storages)
    cpp._arm_alloc_failure(1)  # the pooled output is allocation #1
    with pytest.raises(MemoryError):
        layer(xi)
    assert len(live_storages) == baseline   # output and winners released
    assert np.array_equal(xi.to_numpy(), values)
    # A later forward/backward succeeds.
    out = layer(xi)
    out.backward(gradient=NativeTensor.from_array(np.ones((1, 1, 2, 2))))
    assert xi.grad.to_numpy().sum() == 4.0
    assert cpp._require_library().tf_last_error_code() == cpp.TF_OK
    out.close()
    xi.close()


def test_graph_construction_failure_cleanup_is_inherited(monkeypatch,
                                                         live_storages):
    xi = NativeParameter(np.arange(16, dtype=float).reshape(1, 1, 4, 4))
    layer = NativeMaxPool2d(2)
    baseline = len(live_storages)

    def exploding_from_op(cls, *args, **kwargs):
        raise RuntimeError("simulated graph-construction failure")

    monkeypatch.setattr(NativeTensor, "_from_op", classmethod(exploding_from_op))
    with pytest.raises(RuntimeError, match="simulated graph-construction"):
        layer(xi)
    # The D9 cleanup released both the pooled output and the winners.
    assert len(live_storages) == baseline
    assert xi.closed is False
    monkeypatch.undo()
    out = layer(xi)
    assert out.to_numpy().tolist() == [[[[5.0, 7.0], [13.0, 15.0]]]]
    out.close()
    xi.close()


def test_failed_backward_leaves_gradients_unchanged_and_retries():
    x = np.arange(16, dtype=float).reshape(1, 1, 4, 4)
    xi = NativeParameter(x)
    layer = NativeMaxPool2d(2)
    out = layer(xi)
    g = NativeTensor.from_array(np.ones((1, 1, 2, 2)))
    out.backward(gradient=g, retain_graph=True)
    before = xi.grad.to_numpy().copy()
    bad = NativeTensor.from_array(np.ones((1, 1, 3, 3)))  # wrong shape
    with pytest.raises(ValueError):
        out.backward(gradient=bad)
    assert np.array_equal(xi.grad.to_numpy(), before)  # unchanged
    assert not out._graph_resources[0]._closed         # still retryable
    out.backward(gradient=g)                           # retry succeeds
    assert np.allclose(xi.grad.to_numpy(), 2 * before, atol=1e-12)
    for t in (out, g, bad, xi):
        t.close()


def test_repeated_forwards_release_independently(live_storages):
    xi = NativeParameter(np.random.default_rng(19).standard_normal((1, 1, 4, 4)))
    layer = NativeMaxPool2d(2)
    baseline = len(live_storages)
    outputs = [layer(xi) for _ in range(3)]
    resources = [out._graph_resources[0] for out in outputs]
    assert len({id(r) for r in resources}) == 3  # independent buffers
    assert len(live_storages) > baseline
    for out in outputs:
        out.close()   # explicit close releases output + winners
    assert len(live_storages) == baseline
    assert all(r._closed for r in resources)
    xi.close()


# --------------------------------------------------------------------------
# Capability introspection
# --------------------------------------------------------------------------

def test_module_is_advertised_and_exported():
    import tensorforge.experimental as experimental

    assert "NativeMaxPool2d" in cpp.NATIVE_MODULES
    assert "NativeMaxPool2d" in cpp.backend_info()["native_modules"]
    assert "NativeMaxPool2d" in experimental.__all__
    assert experimental.NativeMaxPool2d is NativeMaxPool2d
    assert "NativeMaxPool2d" not in cpp.UNSUPPORTED


def test_module_does_not_leak_into_the_stable_namespace():
    import tensorforge

    assert not hasattr(tensorforge, "NativeMaxPool2d")
    assert not hasattr(tensorforge.nn, "NativeMaxPool2d")


def test_operation_support_is_unchanged_by_the_module_milestone():
    # D10 added a module only: no new Core/raw capability, no new autograd
    # op, no new checked kernel.
    assert "maxpool2d" in cpp.AUTOGRAD_OPS
    assert "maxpool2d_forward" in cpp.TENSOR_CORE_OPS
    assert "maxpool2d_backward" in cpp.TENSOR_CORE_OPS
    assert "NativeMaxPool2d" not in cpp.TENSOR_CORE_OPS
    assert "NativeMaxPool2d" not in cpp.AUTOGRAD_OPS
    assert "NativeMaxPool2d" not in cpp.RAW_KERNELS
    assert cpp.RAW_KERNELS == (
        "elementwise_add", "elementwise_subtract", "elementwise_multiply",
        "elementwise_divide", "relu", "matmul", "matmul_tiled",
    )
    assert cpp.TENSOR_CORE_KERNELS == (
        "relu", "add", "subtract", "multiply", "matmul",
    )
    assert cpp.SUPPORTED_DTYPES == ("float64", "float32")
    assert sum(1 for k in cpp._CHECKED_KERNELS if "maxpool2d" in k) == 2


def test_no_return_indices_or_public_winner_capability():
    layer = NativeMaxPool2d(2)
    public = [name for name in dir(layer) if not name.startswith("_")]
    assert not [name for name in public if "winner" in name.lower()]
    assert not [name for name in public if "indices" in name.lower()]
    assert not hasattr(layer, "return_indices")
    advertised = str(cpp.backend_info())
    assert "return_indices" not in advertised
    assert "winner" not in advertised.lower()


def test_this_module_is_part_of_the_completed_phase_d_stack():
    # D11 proved the stack trains and D12 closed the phase; this module is
    # part of both, and the artifacts that certify it exist.
    from pathlib import Path

    repo_root = Path(__file__).resolve().parent.parent
    assert (repo_root / "examples" / "native_cnn_training.py").is_file()
    assert (repo_root / "tests" / "test_native_phase_d.py").is_file()
    matrix = (repo_root / "docs" / "native_support_matrix.md").read_text(
        encoding="utf-8"
    )
    assert "NativeMaxPool2d" in matrix
    assert "D11" in matrix and "D12" in matrix
