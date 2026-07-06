import numpy as np
import pytest

from tensorforge import Parameter, StepLR
from tensorforge.optim import SGD, Adam


def test_decays_sgd_lr_on_schedule():
    opt = SGD([Parameter([1.0])], lr=1.0)
    scheduler = StepLR(opt, step_size=2, gamma=0.1)
    lrs = [scheduler.step() for _ in range(5)]
    # Decay happens when last_epoch hits 2 and 4.
    assert np.allclose(lrs, [1.0, 0.1, 0.1, 0.01, 0.01])
    assert np.isclose(opt.lr, 0.01)
    assert scheduler.last_epoch == 5


def test_decays_adam_lr_on_schedule():
    opt = Adam([Parameter([1.0])], lr=0.008)
    scheduler = StepLR(opt, step_size=3, gamma=0.5)
    for _ in range(6):
        scheduler.step()
    assert np.isclose(opt.lr, 0.008 * 0.5 * 0.5)  # decayed at epochs 3 and 6


def test_step_returns_current_lr_as_float():
    opt = SGD([Parameter([1.0])], lr=0.4)
    scheduler = StepLR(opt, step_size=1, gamma=0.5)
    result = scheduler.step()
    assert type(result) is float
    assert np.isclose(result, 0.2)


def test_scheduler_does_not_step_optimizer_or_touch_state():
    p = Parameter(np.array([1.0, -2.0]))
    p.grad = np.array([5.0, 5.0])
    opt = SGD([p], lr=0.1)
    scheduler = StepLR(opt, step_size=1, gamma=0.5)

    scheduler.step()

    assert np.array_equal(p.data, [1.0, -2.0])  # no optimizer step happened
    assert np.array_equal(p.grad, [5.0, 5.0])   # gradients untouched


def test_constructor_validation():
    opt = SGD([Parameter([1.0])], lr=0.1)
    for bad in (0, -2, 2.5, "3", True):
        with pytest.raises(ValueError):
            StepLR(opt, step_size=bad)
    for bad_gamma in (0, -0.5, "half"):
        with pytest.raises(ValueError):
            StepLR(opt, step_size=1, gamma=bad_gamma)


def test_state_dict_roundtrip():
    opt = SGD([Parameter([1.0])], lr=1.0)
    scheduler = StepLR(opt, step_size=2, gamma=0.1)
    scheduler.step()
    scheduler.step()
    scheduler.step()
    state = scheduler.state_dict()
    assert state == {"step_size": 2, "gamma": 0.1, "last_epoch": 3}

    other_opt = SGD([Parameter([1.0])], lr=1.0)
    other = StepLR(other_opt, step_size=99, gamma=0.9)
    other.load_state_dict(state)
    assert other.step_size == 2
    assert other.gamma == 0.1
    assert other.last_epoch == 3
    assert other.optimizer is other_opt  # optimizer not replaced

    # Resumed scheduler continues the same rhythm: epoch 4 decays.
    other.step()
    assert np.isclose(other_opt.lr, 0.1)
