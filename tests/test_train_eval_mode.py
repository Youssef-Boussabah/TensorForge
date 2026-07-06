import numpy as np

from tensorforge import count_parameters
from tensorforge.nn import Dropout, Linear, ReLU, Sequential, Tanh


def _nested_model():
    return Sequential(
        Linear(2, 4),
        ReLU(),
        Sequential(Linear(4, 4), Dropout(p=0.5)),
        Linear(4, 1),
    )


def _all_modules(model):
    return [
        model,
        model.modules[0],
        model.modules[1],
        model.modules[2],
        model.modules[2].modules[0],
        model.modules[2].modules[1],
        model.modules[3],
    ]


def test_modules_default_to_training_mode():
    for module in (Linear(2, 3), Sequential(Linear(1, 1)), ReLU(), Tanh(), Dropout()):
        assert module.training is True


def test_eval_sets_training_false_recursively():
    model = _nested_model()
    model.eval()
    assert all(m.training is False for m in _all_modules(model))


def test_train_sets_training_true_recursively():
    model = _nested_model()
    model.eval()
    model.train()
    assert all(m.training is True for m in _all_modules(model))


def test_train_false_equals_eval():
    a = _nested_model()
    b = _nested_model()
    a.train(False)
    b.eval()
    for m_a, m_b in zip(_all_modules(a), _all_modules(b)):
        assert m_a.training is m_b.training is False


def test_train_and_eval_return_self():
    model = _nested_model()
    assert model.eval() is model
    assert model.train() is model
    assert model.train(False) is model


def test_training_flag_does_not_pollute_parameters_or_state_dict():
    model = _nested_model()
    names_before = [name for name, _ in model.named_parameters()]
    state_before = set(model.state_dict())

    model.eval()  # sets instance attributes on every module
    model.train()

    assert [name for name, _ in model.named_parameters()] == names_before
    assert set(model.state_dict()) == state_before
    assert not any("training" in name for name in state_before)


def test_counting_helpers_unaffected_by_mode():
    model = _nested_model()
    total = count_parameters(model)
    trainable = len(model.trainable_parameters())
    model.eval()
    assert count_parameters(model) == total
    assert len(model.trainable_parameters()) == trainable
    assert "Total params" in model.summary()
