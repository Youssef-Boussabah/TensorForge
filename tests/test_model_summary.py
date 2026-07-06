import numpy as np

from tensorforge import count_parameters, model_summary
from tensorforge.nn import Linear, ReLU, Sequential, Sigmoid, Tanh


def _state_dict_size(model):
    return sum(int(np.prod(v.shape)) for v in model.state_dict().values())


def test_linear_parameter_count():
    layer = Linear(2, 4)
    assert layer.num_parameters() == _state_dict_size(layer)
    assert isinstance(layer.num_parameters(), int)

    no_bias = Linear(3, 5, bias=False)
    assert no_bias.num_parameters() == _state_dict_size(no_bias)


def test_sequential_parameter_count():
    model = Sequential(Linear(2, 4), Tanh(), Linear(4, 3))
    expected = _state_dict_size(model)
    assert model.num_parameters() == expected
    assert count_parameters(model) == expected
    # (2*4 + 4) + (4*3 + 3) as a sanity anchor
    assert expected == 27


def test_nested_sequential_parameter_count():
    model = Sequential(
        Linear(2, 4),
        Sequential(Linear(4, 4), ReLU()),
        Linear(4, 1),
    )
    assert model.num_parameters() == _state_dict_size(model)


def test_no_parameter_module():
    for module in (Tanh(), Sigmoid(), ReLU(), Sequential(Tanh(), ReLU())):
        assert module.num_parameters() == 0
        text = module.summary()
        assert isinstance(text, str)
        assert "Total params: 0" in text


def test_trainable_only_behavior():
    model = Sequential(Linear(2, 4), Tanh(), Linear(4, 3))
    total = model.num_parameters(trainable_only=False)

    frozen = model.modules[0].bias  # freeze one parameter (4 scalars)
    frozen.requires_grad = False

    assert model.num_parameters(trainable_only=True) == total - frozen.data.size
    assert model.num_parameters(trainable_only=False) == total
    assert count_parameters(model) == total - frozen.data.size
    assert count_parameters(model, trainable_only=False) == total


def test_summary_content():
    model = Sequential(Linear(2, 4), Tanh(), Linear(4, 3))
    text = model.summary()

    assert isinstance(text, str)
    assert "TensorForge Model Summary" in text
    assert "Sequential" in text
    for name, value in model.state_dict().items():
        assert name in text
        assert str(value.shape) in text
    assert "Total params: 27" in text
    assert "Trainable params: 27" in text
    assert "Non-trainable params: 0" in text


def test_summary_reflects_frozen_parameters():
    model = Sequential(Linear(2, 4), Tanh(), Linear(4, 3))
    model.modules[0].bias.requires_grad = False
    text = model.summary()
    assert "Total params: 27" in text
    assert "Trainable params: 23" in text
    assert "Non-trainable params: 4" in text
    assert "no" in text  # the frozen row is marked


def test_model_summary_helper_matches_method():
    model = Sequential(Linear(2, 4), Tanh(), Linear(4, 3))
    assert model_summary(model) == model.summary()


def test_public_api_imports():
    from tensorforge import count_parameters, model_summary  # noqa: F401
    from tensorforge.nn import (  # noqa: F401
        count_parameters as nn_count,
        model_summary as nn_summary,
    )
