"""Module buffers: non-trainable arrays that travel with state_dict."""

import numpy as np
import pytest

from tensorforge import (
    Tensor,
    count_parameters,
    load_checkpoint,
    load_parameters,
    save_checkpoint,
    save_parameters,
)
from tensorforge.nn import BatchNorm1d, Linear, ReLU, Sequential
from tensorforge.optim import SGD


def _model():
    return Sequential(Linear(2, 3), BatchNorm1d(3), ReLU(), Linear(3, 1))


def _run_training_forward(model, seed=0):
    """One training-mode forward so BatchNorm's running stats move."""
    rng = np.random.default_rng(seed)
    model(Tensor(rng.normal(size=(12, 2))))


def test_named_buffers_finds_running_stats():
    np.random.seed(0)
    model = _model()
    names = dict(model.named_buffers())
    assert set(names) == {"modules.1.running_mean", "modules.1.running_var"}
    assert names["modules.1.running_mean"] is model.modules[1].running_mean

    bn = BatchNorm1d(4)
    assert set(dict(bn.named_buffers())) == {"running_mean", "running_var"}
    assert len(bn.buffers()) == 2


def test_buffers_not_in_parameters_or_counts():
    np.random.seed(0)
    model = _model()
    params = model.parameters()
    buffer_ids = {id(buf) for buf in model.buffers()}
    assert all(id(p.data) not in buffer_ids for p in params)
    # Linear(2,3): 9, BatchNorm1d(3): gamma+beta = 6, Linear(3,1): 4.
    assert len(params) == 6
    assert count_parameters(model) == 9 + 6 + 4
    assert len(model.trainable_parameters()) == 6

    summary = model.summary()
    assert "running_mean" not in summary
    assert "running_var" not in summary
    assert "gamma" in summary and "beta" in summary


def test_state_dict_includes_parameters_and_buffers():
    np.random.seed(0)
    model = _model()
    state = model.state_dict()
    assert "modules.1.gamma" in state
    assert "modules.1.running_mean" in state
    assert "modules.1.running_var" in state
    assert len(state) == 6 + 2  # six parameters, two buffers

    # Buffer entries are copies.
    state["modules.1.running_mean"] += 100.0
    assert not np.allclose(model.modules[1].running_mean, state["modules.1.running_mean"])


def test_load_state_dict_restores_buffers():
    np.random.seed(0)
    model = _model()
    _run_training_forward(model)
    state = model.state_dict()
    saved_mean = state["modules.1.running_mean"].copy()

    _run_training_forward(model, seed=1)  # move the stats further
    assert not np.allclose(model.modules[1].running_mean, saved_mean)

    model.load_state_dict(state)
    assert np.allclose(model.modules[1].running_mean, saved_mean)


def test_strict_detects_buffer_key_problems():
    np.random.seed(0)
    model = _model()
    state = model.state_dict()

    missing = dict(state)
    del missing["modules.1.running_var"]
    with pytest.raises(ValueError, match="missing"):
        model.load_state_dict(missing)

    extra = dict(state)
    extra["modules.1.running_extra"] = np.zeros(3)
    with pytest.raises(ValueError, match="unexpected"):
        model.load_state_dict(extra)

    bad_shape = dict(state)
    bad_shape["modules.1.running_mean"] = np.zeros(7)
    with pytest.raises(ValueError, match="shape mismatch"):
        model.load_state_dict(bad_shape)


def test_non_strict_loads_matching_buffer_keys():
    np.random.seed(0)
    model = _model()
    state = model.state_dict()
    state["modules.1.running_mean"] = np.full(3, 42.0)
    del state["modules.1.running_var"]
    state["extra"] = np.zeros(2)

    report = model.load_state_dict(state, strict=False)
    assert report["missing_keys"] == ["modules.1.running_var"]
    assert report["unexpected_keys"] == ["extra"]
    assert np.allclose(model.modules[1].running_mean, 42.0)


def test_save_load_parameters_restores_running_stats(tmp_path):
    np.random.seed(0)
    model_a = _model()
    _run_training_forward(model_a)
    path = tmp_path / "params.npz"
    save_parameters(model_a, path)

    np.random.seed(9)
    model_b = _model()
    load_parameters(model_b, path)
    assert np.array_equal(model_b.modules[1].running_mean, model_a.modules[1].running_mean)
    assert np.array_equal(model_b.modules[1].running_var, model_a.modules[1].running_var)

    # Predictions in eval mode (which uses the running stats) match too.
    x = Tensor(np.array([[0.5, -1.0], [2.0, 0.0]]))
    assert np.allclose(model_a.eval()(x).data, model_b.eval()(x).data)


def test_checkpoint_restores_running_stats(tmp_path):
    np.random.seed(0)
    model_a = _model()
    opt_a = SGD(model_a.parameters(), lr=0.1)
    _run_training_forward(model_a)
    path = tmp_path / "ckpt.npz"
    save_checkpoint(path, model_a, opt_a, metadata={"epoch": 1})

    np.random.seed(9)
    model_b = _model()
    opt_b = SGD(model_b.parameters(), lr=0.5)
    report = load_checkpoint(path, model_b, opt_b)
    assert report["optimizer_loaded"] is True
    assert np.array_equal(model_b.modules[1].running_mean, model_a.modules[1].running_mean)
    assert np.array_equal(model_b.modules[1].running_var, model_a.modules[1].running_var)
    assert opt_b.lr == 0.1


def test_bufferless_models_unchanged():
    np.random.seed(0)
    model = Sequential(Linear(2, 3), ReLU(), Linear(3, 1))
    assert model.buffers() == []
    assert list(model.named_buffers()) == []
    assert len(model.state_dict()) == 4  # parameters only, as before