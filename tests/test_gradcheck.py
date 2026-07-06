"""Finite-difference gradient checking.

Every test compares TensorForge's analytic gradients (from backward())
against central finite differences:

    numerical_grad[i] = (f(x + eps*e_i) - f(x - eps*e_i)) / (2 * eps)

Central differences have O(eps^2) truncation error, so with eps=1e-6 a
correct gradient matches to ~1e-9 and atol=1e-5 comfortably separates
"correct" from "wrong formula" without hiding real bugs.

Inputs are small, deterministic, and chosen to avoid non-differentiable
points (ReLU at exactly 0, log/div at 0).
"""

import numpy as np

from tensorforge import Tensor, cross_entropy
from tensorforge.nn import Linear, Sequential, Tanh

EPS = 1e-6


def numerical_grad(fn, x_np, eps=EPS):
    """Central finite differences of scalar-valued ``fn`` at ``x_np``.

    ``fn`` takes a NumPy array and returns a Python float. Returns an
    array of ``x_np``'s shape; the input is never mutated.
    """
    x_np = np.asarray(x_np, dtype=np.float64)
    grad = np.zeros_like(x_np)
    it = np.nditer(x_np, flags=["multi_index"])
    for _ in it:
        idx = it.multi_index
        plus = x_np.copy()
        plus[idx] += eps
        minus = x_np.copy()
        minus[idx] -= eps
        grad[idx] = (fn(plus) - fn(minus)) / (2.0 * eps)
    return grad


def assert_gradclose(analytic, numeric, atol=1e-5, rtol=1e-5):
    analytic = np.asarray(analytic)
    numeric = np.asarray(numeric)
    assert analytic.shape == numeric.shape, (
        f"shape mismatch: analytic {analytic.shape} vs numeric {numeric.shape}"
    )
    assert np.allclose(analytic, numeric, atol=atol, rtol=rtol), (
        f"gradients differ (max abs error "
        f"{np.max(np.abs(analytic - numeric)):.3e})\n"
        f"analytic:\n{analytic}\nnumeric:\n{numeric}"
    )


def test_elementwise_arithmetic():
    """f(x) = sum(x*x + 3x)"""
    x_np = np.array([1.5, -2.0, 0.7])

    x = Tensor(x_np.copy(), requires_grad=True)
    ((x * x) + (3.0 * x)).sum().backward()

    numeric = numerical_grad(lambda v: float((v * v + 3.0 * v).sum()), x_np)
    assert_gradclose(x.grad, numeric)


def test_pow_and_division():
    """f(x) = sum(x**3 / 2 + 1/x), away from x = 0."""
    x_np = np.array([1.5, -2.0, 0.7])

    x = Tensor(x_np.copy(), requires_grad=True)
    ((x ** 3) / 2.0 + 1.0 / x).sum().backward()

    numeric = numerical_grad(lambda v: float((v ** 3 / 2.0 + 1.0 / v).sum()), x_np)
    assert_gradclose(x.grad, numeric)


def test_mixed_elementwise_ops():
    """f(x) = sum(tanh(x) + sigmoid(x) + relu(x) + log(exp(x)))"""
    x_np = np.array([-1.2, 0.5, 2.0])  # keeps ReLU away from its kink at 0

    x = Tensor(x_np.copy(), requires_grad=True)
    (x.tanh() + x.sigmoid() + x.relu() + x.exp().log()).sum().backward()

    def fn(v):
        return float(
            (np.tanh(v) + 1.0 / (1.0 + np.exp(-v)) + np.maximum(v, 0.0) + v).sum()
        )

    numeric = numerical_grad(fn, x_np)
    assert_gradclose(x.grad, numeric)


def test_mean_gradient():
    """f(x) = mean(x * x), checking the 1/N scaling."""
    x_np = np.array([[1.5, -2.0], [0.7, 3.0]])

    x = Tensor(x_np.copy(), requires_grad=True)
    (x * x).mean().backward()

    numeric = numerical_grad(lambda v: float((v * v).mean()), x_np)
    assert_gradclose(x.grad, numeric)


def test_matmul_gradients():
    """f(A, B) = sum((A @ B) * W) for fixed upstream weights W."""
    a_np = np.array([[1.0, -2.0, 0.5], [3.0, 0.2, -1.5]])   # (2, 3)
    b_np = np.array([[2.0, 0.3], [-1.0, 1.5], [0.7, -0.4]])  # (3, 2)
    w_np = np.array([[1.0, -2.0], [3.0, 0.5]])                # (2, 2)

    a = Tensor(a_np.copy(), requires_grad=True)
    b = Tensor(b_np.copy(), requires_grad=True)
    ((a @ b) * w_np).sum().backward()

    numeric_a = numerical_grad(lambda v: float(((v @ b_np) * w_np).sum()), a_np)
    numeric_b = numerical_grad(lambda v: float(((a_np @ v) * w_np).sum()), b_np)
    assert_gradclose(a.grad, numeric_a)
    assert_gradclose(b.grad, numeric_b)


def test_softmax_gradient():
    """f(x) = sum(softmax(x, axis=-1) * weights)"""
    x_np = np.array([[2.0, 1.0, 0.1], [-0.5, 0.8, 1.7]])
    w_np = np.array([[1.0, -2.0, 3.0], [0.5, 2.5, -1.0]])  # non-symmetric

    x = Tensor(x_np.copy(), requires_grad=True)
    (x.softmax(axis=-1) * w_np).sum().backward()

    def fn(v):
        shifted = v - v.max(axis=-1, keepdims=True)
        e = np.exp(shifted)
        s = e / e.sum(axis=-1, keepdims=True)
        return float((s * w_np).sum())

    numeric = numerical_grad(fn, x_np)
    assert_gradclose(x.grad, numeric)


def test_cross_entropy_gradient():
    """f(logits) = cross_entropy(logits, targets)"""
    logits_np = np.array(
        [
            [2.0, 1.0, 0.1, -0.5],
            [0.3, -1.2, 1.8, 0.4],
            [-0.7, 0.9, 0.2, 1.1],
        ]
    )
    targets = [0, 2, 1]

    logits = Tensor(logits_np.copy(), requires_grad=True)
    cross_entropy(logits, targets).backward()

    def fn(v):
        shifted = v - v.max(axis=1, keepdims=True)
        log_probs = shifted - np.log(np.exp(shifted).sum(axis=1, keepdims=True))
        return float(-log_probs[np.arange(3), targets].mean())

    numeric = numerical_grad(fn, logits_np)
    assert_gradclose(logits.grad, numeric)


def test_model_parameter_gradients():
    """Every parameter of a tiny MLP, checked end to end through
    Linear -> Tanh -> Linear -> cross_entropy."""
    np.random.seed(0)
    model = Sequential(Linear(2, 3), Tanh(), Linear(3, 2))
    x_np = np.array([[0.5, -1.0], [1.5, 0.3]])
    targets = [0, 1]

    def loss_value():
        return float(cross_entropy(model(Tensor(x_np)), targets).data)

    # Analytic gradients: one backward pass through the whole model.
    cross_entropy(model(Tensor(x_np)), targets).backward()

    for param in model.parameters():
        analytic = param.grad.copy()
        numeric = np.zeros_like(param.data)
        # Perturb the parameter in place, re-run the forward pass, and
        # restore — the model itself is the function being differentiated.
        it = np.nditer(param.data, flags=["multi_index"])
        for _ in it:
            idx = it.multi_index
            original = param.data[idx]
            param.data[idx] = original + EPS
            loss_plus = loss_value()
            param.data[idx] = original - EPS
            loss_minus = loss_value()
            param.data[idx] = original
            numeric[idx] = (loss_plus - loss_minus) / (2.0 * EPS)
        assert_gradclose(analytic, numeric)
