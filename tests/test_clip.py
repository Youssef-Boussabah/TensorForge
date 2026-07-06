import numpy as np
import pytest

from tensorforge import Parameter, clip_grad_norm, clip_grad_value


def _param(data, grad=None, requires_grad=True):
    p = Parameter(np.asarray(data, dtype=float))
    p.requires_grad = requires_grad
    if grad is not None:
        p.grad = np.asarray(grad, dtype=float)
    return p


# ---------------------------------------------------------------------------
# clip_grad_norm
# ---------------------------------------------------------------------------


def test_norm_returns_original_total_norm():
    a = _param([0.0], grad=[3.0])
    b = _param([0.0], grad=[4.0])
    assert clip_grad_norm([a, b], max_norm=100.0) == 5.0  # sqrt(9 + 16)


def test_norm_scales_when_above_max():
    a = _param([1.0], grad=[3.0])
    b = _param([1.0], grad=[4.0])
    total = clip_grad_norm([a, b], max_norm=1.0)
    assert total == 5.0
    # Every gradient scaled by max_norm / (total + eps): direction kept.
    assert np.allclose(a.grad, 3.0 / 5.0, atol=1e-5)
    assert np.allclose(b.grad, 4.0 / 5.0, atol=1e-5)
    new_norm = np.sqrt(a.grad[0] ** 2 + b.grad[0] ** 2)
    assert new_norm <= 1.0


def test_norm_leaves_small_gradients_unchanged():
    a = _param([1.0], grad=[0.3])
    b = _param([1.0], grad=[0.4])
    total = clip_grad_norm([a, b], max_norm=1.0)
    assert np.isclose(total, 0.5)
    assert np.array_equal(a.grad, [0.3])  # bit-identical, no scaling
    assert np.array_equal(b.grad, [0.4])


def test_norm_skips_none_and_frozen():
    with_grad = _param([1.0], grad=[3.0])
    no_grad = _param([1.0])
    frozen = _param([1.0], grad=[100.0], requires_grad=False)
    total = clip_grad_norm([with_grad, no_grad, frozen], max_norm=1.0)
    assert total == 3.0  # the frozen 100.0 never entered the norm
    assert np.array_equal(frozen.grad, [100.0])  # and was not scaled
    assert no_grad.grad is None


def test_norm_returns_zero_with_no_gradients():
    assert clip_grad_norm([_param([1.0]), _param([2.0])], max_norm=1.0) == 0.0
    assert clip_grad_norm([], max_norm=1.0) == 0.0


def test_norm_does_not_touch_parameter_data():
    p = _param([1.0, -2.0], grad=[30.0, 40.0])
    data_before = p.data.copy()
    clip_grad_norm([p], max_norm=1.0)
    assert np.array_equal(p.data, data_before)


def test_norm_validates_arguments():
    p = _param([1.0], grad=[1.0])
    for bad in (0, -1.0, "big", None):
        with pytest.raises(ValueError):
            clip_grad_norm([p], max_norm=bad)
    for bad_eps in (0, -1e-6, "tiny"):
        with pytest.raises(ValueError):
            clip_grad_norm([p], max_norm=1.0, eps=bad_eps)


# ---------------------------------------------------------------------------
# clip_grad_value
# ---------------------------------------------------------------------------


def test_value_clips_both_signs():
    p = _param([0.0, 0.0, 0.0], grad=[-5.0, 0.5, 3.0])
    assert clip_grad_value([p], clip_value=1.0) is None
    assert np.array_equal(p.grad, [-1.0, 0.5, 1.0])


def test_value_leaves_in_range_values_unchanged():
    p = _param([0.0, 0.0], grad=[-0.9, 0.7])
    clip_grad_value([p], clip_value=1.0)
    assert np.array_equal(p.grad, [-0.9, 0.7])


def test_value_skips_none_and_frozen():
    no_grad = _param([1.0])
    frozen = _param([1.0], grad=[9.0], requires_grad=False)
    clip_grad_value([no_grad, frozen], clip_value=1.0)
    assert no_grad.grad is None
    assert np.array_equal(frozen.grad, [9.0])


def test_value_zero_zeroes_gradients():
    p = _param([1.0, 1.0], grad=[-3.0, 5.0])
    clip_grad_value([p], clip_value=0.0)
    assert np.array_equal(p.grad, [0.0, 0.0])


def test_value_does_not_touch_parameter_data():
    p = _param([2.0, -3.0], grad=[10.0, -10.0])
    data_before = p.data.copy()
    clip_grad_value([p], clip_value=1.0)
    assert np.array_equal(p.data, data_before)


def test_value_validates_clip_value():
    p = _param([1.0], grad=[1.0])
    for bad in (-0.1, "one", None):
        with pytest.raises(ValueError):
            clip_grad_value([p], clip_value=bad)
