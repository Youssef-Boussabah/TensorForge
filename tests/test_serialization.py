import numpy as np
import pytest

from tensorforge import Tensor, load_parameters, save_parameters
from tensorforge.nn import Linear, Sequential, Tanh, mse_loss
from tensorforge.optim import SGD


def _make_model(seed):
    np.random.seed(seed)
    return Sequential(Linear(2, 3), Tanh(), Linear(3, 2))


def test_state_dict_basics():
    model = _make_model(seed=0)
    state = model.state_dict()

    assert isinstance(state, dict)
    assert len(state) == 4  # two Linear layers, each weight + bias
    assert all(isinstance(v, np.ndarray) for v in state.values())

    # Values are copies: mutating them must not touch the model.
    name = next(iter(state))
    state[name] += 100.0
    assert not np.allclose(state[name], dict(model.named_parameters())[name].data)


def test_state_dict_names_are_stable_and_readable():
    state = _make_model(seed=0).state_dict()
    assert set(state) == {
        "modules.0.weight",
        "modules.0.bias",
        "modules.2.weight",
        "modules.2.bias",
    }
    # Same architecture, same names.
    assert set(_make_model(seed=1).state_dict()) == set(state)


def test_load_state_dict_restores_parameters():
    model = _make_model(seed=0)
    state = model.state_dict()

    for param in model.parameters():
        param.data = param.data + 5.0  # wreck the weights

    report = model.load_state_dict(state)
    assert report == {"missing_keys": [], "unexpected_keys": []}
    for name, param in model.named_parameters():
        assert np.array_equal(param.data, state[name])


def test_load_state_dict_does_not_replace_parameter_objects():
    model = _make_model(seed=0)
    before = model.parameters()
    model.load_state_dict(model.state_dict())
    assert all(a is b for a, b in zip(before, model.parameters()))


def test_load_state_dict_shape_mismatch_raises():
    model = _make_model(seed=0)
    state = model.state_dict()
    state["modules.0.weight"] = np.zeros((3, 2))  # should be (2, 3)
    with pytest.raises(ValueError, match="shape mismatch"):
        model.load_state_dict(state)


def test_strict_missing_key_raises():
    model = _make_model(seed=0)
    state = model.state_dict()
    del state["modules.2.bias"]
    with pytest.raises(ValueError, match="missing"):
        model.load_state_dict(state, strict=True)


def test_strict_unexpected_key_raises():
    model = _make_model(seed=0)
    state = model.state_dict()
    state["not.a.real.parameter"] = np.zeros(3)
    with pytest.raises(ValueError, match="unexpected"):
        model.load_state_dict(state, strict=True)


def test_non_strict_loads_matching_keys():
    model = _make_model(seed=0)
    state = model.state_dict()
    del state["modules.2.bias"]                      # missing: tolerated
    state["extra.key"] = np.zeros(7)                 # unexpected: tolerated
    state["modules.0.weight"] = np.full((2, 3), 9.0)  # matching: loads

    report = model.load_state_dict(state, strict=False)
    assert report["missing_keys"] == ["modules.2.bias"]
    assert report["unexpected_keys"] == ["extra.key"]
    assert np.allclose(dict(model.named_parameters())["modules.0.weight"].data, 9.0)

    # Shape mismatches still raise even when not strict.
    bad = model.state_dict()
    bad["modules.0.bias"] = np.zeros(5)
    with pytest.raises(ValueError, match="shape mismatch"):
        model.load_state_dict(bad, strict=False)


def test_npz_roundtrip(tmp_path):
    model_a = _make_model(seed=0)
    path = tmp_path / "model.npz"
    save_parameters(model_a, path)

    model_b = _make_model(seed=1)  # same architecture, different init
    load_parameters(model_b, path)

    state_a = model_a.state_dict()
    state_b = model_b.state_dict()
    assert set(state_a) == set(state_b)
    for name in state_a:
        assert np.array_equal(state_a[name], state_b[name])


def test_predictions_match_after_loading(tmp_path):
    model_a = _make_model(seed=0)
    model_b = _make_model(seed=1)
    x = Tensor(np.array([[1.0, -2.0], [0.5, 0.25]]))
    assert not np.allclose(model_a(x).data, model_b(x).data)  # differ before

    path = tmp_path / "model.npz"
    save_parameters(model_a, path)
    load_parameters(model_b, path)
    assert np.allclose(model_a(x).data, model_b(x).data)


def test_trained_model_roundtrip(tmp_path):
    np.random.seed(0)
    model = Linear(1, 1)
    optimizer = SGD(model.parameters(), lr=0.05)
    x = Tensor([[0.0], [1.0], [2.0]])
    y = Tensor([[1.0], [3.0], [5.0]])
    for _ in range(30):
        optimizer.zero_grad()
        loss = mse_loss(model(x), y)
        loss.backward()
        optimizer.step()
    trained_predictions = model(x).data.copy()

    path = tmp_path / "trained.npz"
    save_parameters(model, path)

    np.random.seed(42)
    fresh = Linear(1, 1)
    load_parameters(fresh, path)
    assert np.allclose(fresh(x).data, trained_predictions)


def test_public_api_import():
    from tensorforge import load_parameters, save_parameters  # noqa: F401
    from tensorforge.serialization import (  # noqa: F401
        load_parameters as s_load,
        save_parameters as s_save,
    )
