"""Tests for native autograd (Advanced C++ v2.1–v2.4, Phase B).

v2.1 added the autograd *surface* and *engine* to NativeTensor —
requires_grad, grad, is_leaf, zero_grad, detach, backward — exercised
here through the internal graph constructor NativeTensor._from_op. v2.2
wired the core differentiable operations into that engine:
add/subtract/multiply/relu/sum/mean/matmul/reshape/transpose/T/
contiguous_copy build graph nodes when an operand requires grad, with
broadcasting handled on the way back by the _unbroadcast reduction. v2.3
makes narrow differentiable: its backward scatters the upstream gradient
into a fresh zeros tensor of the parent's shape via the native
narrow_backward kernel. The per-operation sections below verify each
backward rule against exact analytical values and, for the composite
cases, central finite differences (NumPy is used only to compute test
references — the gradient path itself is native). v2.4 adds an explicit
graph-lifetime policy: backward(gradient=None, retain_graph=False) is
one-shot by default (a successful pass frees the traversed operation
graph), retain_graph=True keeps it for another pass, leaf gradients
accumulate across passes, reusing a freed graph raises, and a failed pass
rolls back. Native-backend tests skip when the compiled library is not
built. See docs/native_autograd_design.md.
"""

import numpy as np
import pytest

from tensorforge.backends import cpp
from tensorforge.experimental import NativeTensor
from tensorforge.experimental.native_tensor import _unbroadcast

needs_native = pytest.mark.skipif(
    not cpp.is_available(),
    reason="experimental C++ backend not built",
)


# -- helpers ----------------------------------------------------------
#
# A tiny graph-construction helper for the tests. It builds a non-leaf
# node whose forward value is `value` and whose backward callback is
# supplied by the test. This mirrors exactly what the v2.2 ops will do
# via _from_op; the tests do not reach into private grad state in
# unrealistic ways.


def _leaf(values, requires_grad=True):
    return NativeTensor.from_array(values, requires_grad=requires_grad)


def _node(value, parents, backward, op="op"):
    core = cpp.NativeTensorCore.from_array(np.asarray(value, dtype=np.float64))
    return NativeTensor._from_op(core, tuple(parents), backward, op)


def _ones_like(tensor):
    return NativeTensor.full(tensor.shape, 1.0)


# ======================================================================
# Metadata
# ======================================================================


@needs_native
def test_requires_grad_defaults_false():
    for t in (
        NativeTensor.from_array([1.0, 2.0]),
        NativeTensor.zeros((2, 2)),
        NativeTensor.full((3,), 4.0),
    ):
        assert t.requires_grad is False
        t.close()


@needs_native
def test_requires_grad_explicit_true():
    for t in (
        NativeTensor.from_array([1.0, 2.0], requires_grad=True),
        NativeTensor.zeros((2, 2), requires_grad=True),
        NativeTensor.full((3,), 4.0, requires_grad=True),
    ):
        assert t.requires_grad is True
        assert t.is_leaf is True  # user-created tensors are leaves
        t.close()


@needs_native
def test_requires_grad_rejects_non_bool():
    for bad in (1, 0, "yes", 1.0, None):
        with pytest.raises(TypeError, match="requires_grad must be a bool"):
            NativeTensor.from_array([1.0], requires_grad=bad)


@needs_native
def test_grad_starts_none():
    t = NativeTensor.from_array([1.0, 2.0], requires_grad=True)
    assert t.grad is None
    t.close()


@needs_native
def test_leaf_status_of_constructed_tensors():
    a = NativeTensor.zeros((2, 2))                    # requires_grad False
    b = NativeTensor.zeros((2, 2), requires_grad=True)
    assert a.is_leaf is True
    assert b.is_leaf is True
    a.close()
    b.close()


@needs_native
def test_internal_op_result_is_non_leaf_when_grad_required():
    x = _leaf([1.0, 2.0], requires_grad=True)
    y = _node([1.0, 2.0], parents=(x,), backward=lambda u: x._accumulate_grad(u))
    assert y.is_leaf is False
    assert y.requires_grad is True
    x.close()
    y.close()


@needs_native
def test_internal_op_result_is_leaf_when_no_parent_requires_grad():
    # OR-of-parents: if no parent needs grad, the result is a plain
    # forward leaf with no recorded graph (matches the Python Tensor).
    x = _leaf([1.0, 2.0], requires_grad=False)
    y = _node([1.0, 2.0], parents=(x,), backward=lambda u: None)
    assert y.requires_grad is False
    assert y.is_leaf is True
    x.close()
    y.close()


@needs_native
def test_existing_constructors_remain_backward_compatible():
    x = np.array([[1.0, -2.0], [3.0, 4.0]])
    t = NativeTensor.from_array(x)  # no requires_grad passed
    assert t.requires_grad is False
    assert np.array_equal(t.to_numpy(), x)
    t.close()


# ======================================================================
# Gradient storage
# ======================================================================


@needs_native
def test_grad_is_native_tensor_with_matching_metadata():
    x = _leaf([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    y = _node(x.to_numpy(), parents=(x,), backward=lambda u: x._accumulate_grad(u))
    y.backward(gradient=_ones_like(y))
    assert isinstance(x.grad, NativeTensor)
    assert x.grad.shape == x.shape
    assert x.grad.dtype == x.dtype
    assert x.grad.device == x.device
    assert np.array_equal(x.grad.to_numpy(), np.ones((2, 2)))
    x.close()
    y.close()


@needs_native
def test_gradient_accumulation_uses_native_values():
    # Two independent backward contributions into the same leaf sum.
    x = _leaf([1.0, 2.0, 3.0], requires_grad=True)
    g1 = NativeTensor.from_array([1.0, 1.0, 1.0])
    g2 = NativeTensor.from_array([10.0, 20.0, 30.0])
    x._accumulate_grad(g1)
    x._accumulate_grad(g2)
    assert np.array_equal(x.grad.to_numpy(), [11.0, 21.0, 31.0])
    x.close()


@needs_native
def test_zero_grad_clears_to_none_and_is_idempotent():
    x = _leaf([1.0, 2.0], requires_grad=True)
    x._accumulate_grad(NativeTensor.from_array([5.0, 5.0]))
    assert x.grad is not None
    x.zero_grad()
    assert x.grad is None
    x.zero_grad()  # idempotent
    assert x.grad is None
    # requires_grad / leaf metadata untouched
    assert x.requires_grad is True
    assert x.is_leaf is True
    x.close()


@needs_native
def test_grad_operations_on_closed_tensor_raise():
    x = _leaf([1.0, 2.0], requires_grad=True)
    x.close()
    with pytest.raises(RuntimeError, match="closed"):
        _ = x.grad
    with pytest.raises(RuntimeError, match="closed"):
        _ = x.requires_grad
    with pytest.raises(RuntimeError, match="closed"):
        _ = x.is_leaf
    with pytest.raises(RuntimeError, match="closed"):
        x.zero_grad()
    with pytest.raises(RuntimeError, match="closed"):
        x.backward()


# ======================================================================
# detach
# ======================================================================


@needs_native
def test_detach_produces_forward_only_leaf():
    x = _leaf([[1.0, -2.0], [3.0, 4.0]], requires_grad=True)
    d = x.detach()
    assert d.requires_grad is False
    assert d.grad is None
    assert d.is_leaf is True
    assert d._parents == ()          # no graph edges
    assert d._backward is None
    x.close()
    d.close()


@needs_native
def test_detach_value_matches_and_owns_independent_storage():
    x = _leaf(np.arange(6.0).reshape(2, 3), requires_grad=True)
    d = x.detach()
    assert d.owns_core is True
    assert np.array_equal(d.to_numpy(), x.to_numpy())
    # Independent storage: closing the detached copy leaves x alive.
    d.close()
    assert x.closed is False
    assert np.array_equal(x.to_numpy(), np.arange(6.0).reshape(2, 3))
    x.close()


@needs_native
def test_detached_tensor_does_not_receive_gradient_from_original_graph():
    x = _leaf([1.0, 2.0], requires_grad=True)
    d = x.detach()
    # A graph built on x pushes gradient to x, never to the detached copy.
    y = _node(x.to_numpy(), parents=(x,), backward=lambda u: x._accumulate_grad(u))
    y.backward(gradient=_ones_like(y))
    assert x.grad is not None
    assert d.grad is None
    x.close()
    d.close()
    y.close()


# ======================================================================
# backward validation
# ======================================================================


@needs_native
def test_backward_on_non_requiring_tensor_raises():
    x = NativeTensor.from_array([1.0, 2.0])  # requires_grad False
    with pytest.raises(RuntimeError, match="does not require grad"):
        x.backward()
    x.close()


@needs_native
def test_scalar_backward_defaults_gradient_to_one():
    x = _leaf(3.0, requires_grad=True)  # scalar leaf (numel 1)
    y = _node(3.0, parents=(x,), backward=lambda u: x._accumulate_grad(u))
    assert y.numel == 1  # the "scalar output" the default seed applies to
    y.backward()  # no gradient -> seeds 1.0
    assert x.grad.numel == 1
    assert np.array_equal(x.grad.to_numpy(), np.ones(x.shape))
    x.close()
    y.close()


@needs_native
def test_non_scalar_backward_without_gradient_raises():
    x = _leaf([1.0, 2.0], requires_grad=True)
    y = _node([1.0, 2.0], parents=(x,), backward=lambda u: x._accumulate_grad(u))
    with pytest.raises(ValueError, match="non-scalar"):
        y.backward()
    x.close()
    y.close()


@needs_native
def test_backward_rejects_non_native_gradient():
    x = _leaf([1.0, 2.0], requires_grad=True)
    y = _node([1.0, 2.0], parents=(x,), backward=lambda u: x._accumulate_grad(u))
    with pytest.raises(TypeError, match="gradient must be a NativeTensor"):
        y.backward(gradient=np.array([1.0, 2.0]))
    x.close()
    y.close()


@needs_native
def test_backward_gradient_shape_mismatch_names_shapes():
    x = _leaf([1.0, 2.0, 3.0], requires_grad=True)
    y = _node([1.0, 2.0, 3.0], parents=(x,), backward=lambda u: x._accumulate_grad(u))
    bad = NativeTensor.from_array([1.0, 2.0])  # (2,) vs (3,)
    with pytest.raises(ValueError) as excinfo:
        y.backward(gradient=bad)
    msg = str(excinfo.value)
    assert "(2,)" in msg and "(3,)" in msg
    x.close()
    y.close()
    bad.close()


@needs_native
def test_backward_gradient_dtype_device_follow_metadata_contract():
    # A correctly-shaped, matching-dtype/device gradient is accepted, and
    # the resulting grad matches the leaf's dtype/device (the v1.21
    # contract native autograd builds on).
    x = _leaf([1.0, 2.0], requires_grad=True)
    y = _node([1.0, 2.0], parents=(x,), backward=lambda u: x._accumulate_grad(u))
    g = NativeTensor.from_array([2.0, 3.0])
    assert g.dtype == y.dtype and g.device == y.device
    y.backward(gradient=g)
    assert x.grad.dtype == x.dtype
    assert x.grad.device == x.device
    assert np.array_equal(x.grad.to_numpy(), [2.0, 3.0])
    x.close()
    y.close()


@needs_native
def test_backward_on_closed_output_raises():
    x = _leaf([1.0, 2.0], requires_grad=True)
    y = _node([1.0, 2.0], parents=(x,), backward=lambda u: x._accumulate_grad(u))
    y.close()
    with pytest.raises(RuntimeError, match="closed"):
        y.backward(gradient=NativeTensor.from_array([1.0, 2.0]))
    x.close()


# ======================================================================
# Traversal
# ======================================================================


@needs_native
def test_simple_chain_propagates_to_leaf():
    # x -> a -> b, each backward passes the gradient straight through.
    x = _leaf([1.0, 2.0], requires_grad=True)
    a = _node([1.0, 2.0], parents=(x,), backward=lambda u: x._accumulate_grad(u), op="a")
    b = _node([1.0, 2.0], parents=(a,), backward=lambda u: a._accumulate_grad(u), op="b")
    b.backward(gradient=NativeTensor.from_array([5.0, 7.0]))
    assert np.array_equal(x.grad.to_numpy(), [5.0, 7.0])
    # Non-leaf grads are not retained.
    assert a.grad is None
    assert b.grad is None
    for t in (x, a, b):
        t.close()


@needs_native
def test_branching_graph_shared_leaf_accumulates():
    # Diamond: x feeds a and b; c consumes both. x's gradient is the sum
    # of the two paths.
    x = _leaf([1.0, 1.0], requires_grad=True)
    a = _node([1.0, 1.0], parents=(x,), backward=lambda u: x._accumulate_grad(u), op="a")
    b = _node([1.0, 1.0], parents=(x,), backward=lambda u: x._accumulate_grad(u), op="b")

    def c_backward(u):
        a._accumulate_grad(u)
        b._accumulate_grad(u)

    c = _node([1.0, 1.0], parents=(a, b), backward=c_backward, op="c")
    c.backward(gradient=NativeTensor.from_array([1.0, 1.0]))
    # Through a and b, x receives the gradient twice.
    assert np.array_equal(x.grad.to_numpy(), [2.0, 2.0])
    for t in (x, a, b, c):
        t.close()


@needs_native
def test_duplicate_parent_reference_visited_once_but_accumulates_twice():
    # A node listing the same leaf twice as a parent: the leaf is visited
    # once in the topo build, but the node's backward accumulates into it
    # twice (that is the op's job, not the driver's).
    x = _leaf([1.0, 2.0], requires_grad=True)

    def dup_backward(u):
        x._accumulate_grad(u)
        x._accumulate_grad(u)

    y = _node([1.0, 2.0], parents=(x, x), backward=dup_backward, op="dup")
    y.backward(gradient=NativeTensor.from_array([1.0, 1.0]))
    assert np.array_equal(x.grad.to_numpy(), [2.0, 2.0])
    x.close()
    y.close()


@needs_native
def test_reverse_topological_order_and_each_callback_once():
    order = []
    x = _leaf([1.0], requires_grad=True)
    a = _node([1.0], parents=(x,),
              backward=lambda u: (order.append("a"), x._accumulate_grad(u)), op="a")
    b = _node([1.0], parents=(x,),
              backward=lambda u: (order.append("b"), x._accumulate_grad(u)), op="b")

    def c_backward(u):
        order.append("c")
        a._accumulate_grad(u)
        b._accumulate_grad(u)

    c = _node([1.0], parents=(a, b), backward=c_backward, op="c")
    c.backward(gradient=NativeTensor.from_array([1.0]))
    # c (the output) runs before its parents a and b; every callback runs
    # exactly once.
    assert order[0] == "c"
    assert sorted(order) == ["a", "b", "c"]
    assert len(order) == 3
    for t in (x, a, b, c):
        t.close()


@needs_native
def test_repeated_backward_accumulates_until_zero_grad():
    x = _leaf([1.0, 2.0], requires_grad=True)

    def make_y():
        return _node([1.0, 2.0], parents=(x,),
                     backward=lambda u: x._accumulate_grad(u))

    make_y().backward(gradient=NativeTensor.from_array([1.0, 1.0]))
    assert np.array_equal(x.grad.to_numpy(), [1.0, 1.0])
    # A second backward accumulates on top (retain-graph-free repeated
    # backward is additive, matching the Python engine before zero_grad).
    make_y().backward(gradient=NativeTensor.from_array([1.0, 1.0]))
    assert np.array_equal(x.grad.to_numpy(), [2.0, 2.0])
    # zero_grad restarts accumulation cleanly.
    x.zero_grad()
    make_y().backward(gradient=NativeTensor.from_array([3.0, 4.0]))
    assert np.array_equal(x.grad.to_numpy(), [3.0, 4.0])
    x.close()


@needs_native
def test_backward_on_a_requiring_leaf_sets_its_own_grad():
    # A leaf that is itself the backward root keeps the seed as its grad.
    x = _leaf([1.0, 2.0], requires_grad=True)
    x.backward(gradient=NativeTensor.from_array([4.0, 5.0]))
    assert np.array_equal(x.grad.to_numpy(), [4.0, 5.0])
    x.close()


# ======================================================================
# Isolation
# ======================================================================


@needs_native
def test_forward_ops_stay_forward_only_without_requiring_inputs():
    # No operand requires grad -> the op preserves the pre-autograd
    # behavior exactly: a plain forward tensor with no graph metadata.
    a = NativeTensor.from_array([[1.0, -2.0], [3.0, 4.0]])
    b = NativeTensor.from_array([[1.0, 1.0], [1.0, 1.0]])
    for result in (
        a.relu(), a.add(b), a.subtract(b), a.multiply(b), a.sum(),
        a.mean(), a.matmul(b), a.reshape((4,)), a.transpose(), a.T,
        a.narrow(1, 0, 1), a.contiguous_copy(),
    ):
        assert result.requires_grad is False
        assert result.is_leaf is True
        assert result._parents == ()
        assert result._backward is None
        if result.owns_core:
            result.close()
    a.close()
    b.close()


@needs_native
def test_compute_ops_build_graphs_when_an_input_requires_grad():
    # v2.2: with a requiring operand, each differentiable op records a
    # non-leaf node with parents, a backward closure, and its op name.
    a = NativeTensor.from_array([[1.0, -2.0], [3.0, 4.0]], requires_grad=True)
    b = NativeTensor.from_array([[1.0, 1.0], [1.0, 1.0]])
    cases = (
        (a.relu(), "relu", (a,)),
        (a.add(b), "add", (a, b)),
        (a.subtract(b), "subtract", (a, b)),
        (a.multiply(b), "multiply", (a, b)),
        (a.sum(), "sum", (a,)),
        (a.mean(), "mean", (a,)),
        (a.matmul(b), "matmul", (a, b)),
        (a.reshape((4,)), "reshape", (a,)),
        (a.T, "transpose", (a,)),
        (a.narrow(1, 0, 1), "narrow", (a,)),
        (a.contiguous_copy(), "contiguous_copy", (a,)),
    )
    for result, op, parents in cases:
        assert result.requires_grad is True
        assert result.is_leaf is False
        assert result._op == op
        assert result._parents == parents
        assert callable(result._backward)
        if result.owns_core:
            result.close()
    a.close()
    b.close()


@needs_native
def test_narrow_on_non_requiring_parent_stays_graph_free():
    # With a non-requiring parent, narrow is still a plain forward view —
    # no graph metadata, borrowing storage — exactly as before v2.3.
    x = NativeTensor.from_array([[1.0, 2.0], [3.0, 4.0]])  # requires_grad False
    n = x.narrow(1, 0, 1)
    assert n.requires_grad is False
    assert n.is_leaf is True
    assert n._parents == ()
    assert n._backward is None
    assert n.owns_core is False  # borrowing view, unchanged
    x.close()


def test_importing_tensorforge_does_not_import_experimental_autograd():
    """The framework frontend must not import the experimental autograd
    surface — a static check of tensorforge/__init__ so it cannot be
    fooled by another test importing experimental first."""
    import ast
    from pathlib import Path

    init = (
        Path(__file__).resolve().parent.parent
        / "src" / "tensorforge" / "__init__.py"
    )
    tree = ast.parse(init.read_text(encoding="utf-8"))
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)
        elif isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
    assert not any(m.startswith("tensorforge.experimental") for m in imported)


def test_tensorforge_tensor_is_unchanged_and_separate():
    from tensorforge import Tensor

    assert NativeTensor is not Tensor
    assert not issubclass(NativeTensor, Tensor)
    # The Python Tensor keeps its own NumPy-backed autograd; its grad is
    # never a NativeTensor. (The two engines never mix.)
    t = Tensor(2.0, requires_grad=True)
    (t * t).backward()
    assert t.grad is not None
    assert not isinstance(t.grad, NativeTensor)


# ======================================================================
# v2.2 — differentiable operations
# ======================================================================
#
# NumPy appears below only to build references (exact formulas and
# central finite differences); every gradient under test was computed
# natively. Tensors are tiny and deterministic.


def _grad(tensor):
    """The leaf's gradient as a NumPy array (test-side exit only)."""
    assert isinstance(tensor.grad, NativeTensor)
    return tensor.grad.to_numpy()


def _numeric_grad(f, x, eps=1e-6):
    """Central finite differences of the scalar-valued ``f`` at ``x``:
    (f(x + eps) - f(x - eps)) / (2 eps), one coordinate at a time."""
    x = np.array(x, dtype=np.float64)
    grad = np.zeros_like(x)
    flat, gflat = x.ravel(), grad.ravel()
    for i in range(flat.size):
        original = flat[i]
        flat[i] = original + eps
        f_plus = f(x)
        flat[i] = original - eps
        f_minus = f(x)
        flat[i] = original
        gflat[i] = (f_plus - f_minus) / (2 * eps)
    return grad


def _native_scalar(f, values):
    """Run ``f`` (NativeTensor -> scalar NativeTensor) on a fresh leaf
    holding ``values`` and return the float result — the forward pipeline
    finite differences perturb."""
    x = NativeTensor.from_array(values)
    out = f(x)
    result = float(out.to_numpy())
    x.close()
    return result


def _backward_grad(f, values):
    """Run ``f`` on a requiring leaf, backward from the scalar output,
    and return the leaf's native gradient as NumPy."""
    x = NativeTensor.from_array(values, requires_grad=True)
    f(x).backward()
    grad = _grad(x)
    x.close()
    return grad


# -- add / subtract ----------------------------------------------------


@needs_native
def test_add_backward_same_shape():
    a = NativeTensor.from_array([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    b = NativeTensor.from_array([[5.0, 6.0], [7.0, 8.0]], requires_grad=True)
    a.add(b).backward(gradient=NativeTensor.from_array([[1.0, 2.0], [3.0, 4.0]]))
    assert np.array_equal(_grad(a), [[1.0, 2.0], [3.0, 4.0]])
    assert np.array_equal(_grad(b), [[1.0, 2.0], [3.0, 4.0]])
    a.close()
    b.close()


@needs_native
def test_subtract_backward_same_shape():
    a = NativeTensor.from_array([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    b = NativeTensor.from_array([[5.0, 6.0], [7.0, 8.0]], requires_grad=True)
    a.subtract(b).backward(gradient=NativeTensor.from_array([[1.0, 2.0], [3.0, 4.0]]))
    assert np.array_equal(_grad(a), [[1.0, 2.0], [3.0, 4.0]])
    assert np.array_equal(_grad(b), [[-1.0, -2.0], [-3.0, -4.0]])
    a.close()
    b.close()


@needs_native
def test_add_backward_scalar_broadcasting():
    a = NativeTensor.full((), 2.0, requires_grad=True)
    b = NativeTensor.from_array([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    a.add(b).sum().backward()
    # The scalar was read into all 4 positions; its gradient sums them.
    assert a.grad.shape == ()
    assert float(_grad(a)) == 4.0
    assert np.array_equal(_grad(b), np.ones((2, 2)))
    a.close()
    b.close()


@needs_native
def test_add_backward_vector_matrix_broadcasting():
    bias = NativeTensor.from_array([10.0, 20.0, 30.0], requires_grad=True)
    x = NativeTensor.from_array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
                                requires_grad=True)
    x.add(bias).backward(
        gradient=NativeTensor.from_array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    )
    assert np.array_equal(_grad(x), [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    # (3,) was stretched over the leading axis: grad sums down it.
    assert np.array_equal(_grad(bias), [5.0, 7.0, 9.0])
    x.close()
    bias.close()


@needs_native
def test_subtract_backward_both_operands_broadcast():
    a = NativeTensor.from_array([[1.0], [2.0]], requires_grad=True)   # (2, 1)
    b = NativeTensor.from_array([1.0, 2.0, 3.0], requires_grad=True)  # (3,)
    a.subtract(b).sum().backward()  # output (2, 3)
    assert a.grad.shape == (2, 1)
    assert b.grad.shape == (3,)
    assert np.array_equal(_grad(a), [[3.0], [3.0]])
    assert np.array_equal(_grad(b), [-2.0, -2.0, -2.0])
    a.close()
    b.close()


@needs_native
def test_add_backward_duplicate_parent_accumulates_twice():
    x = NativeTensor.from_array([1.0, 2.0], requires_grad=True)
    x.add(x).backward(gradient=NativeTensor.from_array([1.0, 5.0]))
    assert np.array_equal(_grad(x), [2.0, 10.0])
    x.close()


@needs_native
def test_add_backward_skips_non_requiring_operand():
    a = NativeTensor.from_array([1.0, 2.0], requires_grad=True)
    b = NativeTensor.from_array([3.0, 4.0])  # requires_grad False
    a.add(b).backward(gradient=NativeTensor.from_array([1.0, 1.0]))
    assert np.array_equal(_grad(a), [1.0, 1.0])
    assert b.grad is None
    a.close()
    b.close()


@needs_native
def test_subtract_backward_does_not_mutate_upstream():
    # The negation for db must not touch the object da also receives.
    a = NativeTensor.from_array([1.0, 2.0], requires_grad=True)
    b = NativeTensor.from_array([3.0, 4.0], requires_grad=True)
    seed = NativeTensor.from_array([2.0, 3.0])
    a.subtract(b).backward(gradient=seed)
    assert np.array_equal(seed.to_numpy(), [2.0, 3.0])  # unchanged
    assert np.array_equal(_grad(a), [2.0, 3.0])
    assert np.array_equal(_grad(b), [-2.0, -3.0])
    a.close()
    b.close()
    seed.close()


# -- multiply ----------------------------------------------------------


@needs_native
def test_multiply_backward_same_shape():
    a_vals = np.array([[1.0, -2.0], [3.0, 4.0]])
    b_vals = np.array([[5.0, 6.0], [-7.0, 8.0]])
    a = NativeTensor.from_array(a_vals, requires_grad=True)
    b = NativeTensor.from_array(b_vals, requires_grad=True)
    a.multiply(b).sum().backward()
    assert np.array_equal(_grad(a), b_vals)  # d(ab)/da = b
    assert np.array_equal(_grad(b), a_vals)  # d(ab)/db = a
    a.close()
    b.close()


@needs_native
def test_multiply_backward_broadcasting():
    scale = NativeTensor.from_array([2.0, 3.0, 4.0], requires_grad=True)
    x_vals = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    x = NativeTensor.from_array(x_vals, requires_grad=True)
    x.multiply(scale).sum().backward()
    # d/dscale sums x down the stretched axis; d/dx repeats scale.
    assert np.array_equal(_grad(scale), x_vals.sum(axis=0))
    assert np.array_equal(_grad(x), np.broadcast_to([2.0, 3.0, 4.0], (2, 3)))
    x.close()
    scale.close()


@needs_native
def test_multiply_backward_shared_operand_is_two_edges():
    x_vals = np.array([1.0, -2.0, 3.0])
    x = NativeTensor.from_array(x_vals, requires_grad=True)
    x.multiply(x).sum().backward()  # d(x^2)/dx = 2x
    assert np.allclose(_grad(x), 2.0 * x_vals)
    x.close()


@needs_native
def test_multiply_sum_matches_finite_differences():
    values = np.array([[0.5, -1.5], [2.0, 3.0]])
    other = np.array([[1.5, 2.5], [-0.5, 1.0]])

    def pipeline(t):
        o = NativeTensor.from_array(other)
        return t.multiply(o).sum()

    analytic = _backward_grad(pipeline, values)
    numeric = _numeric_grad(lambda v: _native_scalar(pipeline, v), values)
    assert np.allclose(analytic, numeric, atol=1e-6)


# -- relu --------------------------------------------------------------


@needs_native
def test_relu_backward_positive_negative_zero():
    x = NativeTensor.from_array([-1.0, 0.0, 2.0], requires_grad=True)
    x.relu().backward(gradient=NativeTensor.from_array([10.0, 10.0, 10.0]))
    # Gradient passes only where x > 0; x == 0 blocks (the Python
    # Tensor convention: (x > 0) * grad).
    assert np.array_equal(_grad(x), [0.0, 0.0, 10.0])
    x.close()


@needs_native
def test_relu_backward_non_contiguous_input():
    x = NativeTensor.from_array([[1.0, -2.0], [-3.0, 4.0]], requires_grad=True)
    x.T.relu().sum().backward()  # relu over a transposed view
    assert np.array_equal(_grad(x), [[1.0, 0.0], [0.0, 1.0]])
    x.close()


@needs_native
def test_relu_sum_matches_finite_differences_away_from_zero():
    # All inputs far from the nondifferentiable point x == 0.
    values = np.array([[1.5, -2.5], [3.0, -0.5]])

    def pipeline(t):
        return t.relu().sum()

    analytic = _backward_grad(pipeline, values)
    numeric = _numeric_grad(lambda v: _native_scalar(pipeline, v), values)
    assert np.allclose(analytic, numeric, atol=1e-6)


# -- unbroadcast helper ------------------------------------------------


@needs_native
def test_unbroadcast_exact_shape_returns_the_gradient_unchanged():
    g = NativeTensor.from_array([[1.0, 2.0], [3.0, 4.0]])
    assert _unbroadcast(g, (2, 2)) is g
    g.close()


@needs_native
def test_unbroadcast_sums_stretched_axis_with_keepdims():
    g = NativeTensor.from_array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    out = _unbroadcast(g, (2, 1))
    assert out.shape == (2, 1)
    assert np.array_equal(out.to_numpy(), [[6.0], [15.0]])
    g.close()
    out.close()


@needs_native
def test_unbroadcast_reduces_multiple_stretched_axes():
    values = np.arange(30.0).reshape(2, 3, 5)
    g = NativeTensor.from_array(values)
    out = _unbroadcast(g, (1, 3, 1))
    assert out.shape == (1, 3, 1)
    assert np.array_equal(out.to_numpy(), values.sum(axis=(0, 2), keepdims=True))
    g.close()
    out.close()


@needs_native
def test_unbroadcast_reduces_leading_padded_axes():
    values = np.arange(12.0).reshape(3, 4)
    g = NativeTensor.from_array(values)
    out = _unbroadcast(g, (4,))
    assert out.shape == (4,)
    assert np.array_equal(out.to_numpy(), values.sum(axis=0))
    g.close()
    out.close()


@needs_native
def test_unbroadcast_reduces_all_axes_to_scalar():
    values = np.arange(12.0).reshape(3, 4)
    g = NativeTensor.from_array(values)
    out = _unbroadcast(g, ())
    assert out.shape == ()
    assert float(out.to_numpy()) == values.sum()
    g.close()
    out.close()


@needs_native
def test_unbroadcast_distinguishes_one_element_from_scalar():
    # (1,) -> (): rank drops by a reduction.
    g1 = NativeTensor.from_array([7.0])
    out1 = _unbroadcast(g1, ())
    assert out1.shape == ()
    assert float(out1.to_numpy()) == 7.0
    # () -> (1,): rank grows through a native reshape, never NumPy.
    g2 = NativeTensor.full((), 7.0)
    out2 = _unbroadcast(g2, (1,))
    assert out2.shape == (1,)
    assert np.array_equal(out2.to_numpy(), [7.0])
    for t in (g1, out1, g2, out2):
        t.close()


@needs_native
def test_unbroadcast_preserves_dtype_device():
    g = NativeTensor.from_array([[1.0, 2.0]])
    out = _unbroadcast(g, (2,))
    assert out.dtype == g.dtype == "float64"
    assert out.device == g.device == "cpu"
    g.close()
    out.close()


# -- sum / mean --------------------------------------------------------


@needs_native
def test_sum_backward_all_elements():
    x = NativeTensor.from_array([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    x.sum().backward()  # scalar output seeds 1.0
    assert np.array_equal(_grad(x), np.ones((2, 2)))
    x.close()


@needs_native
def test_sum_backward_axis0_and_axis1():
    for axis, seed, expected in (
        (0, [1.0, 2.0, 3.0], np.tile([1.0, 2.0, 3.0], (2, 1))),
        (1, [1.0, 2.0], np.array([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]])),
    ):
        x = NativeTensor.from_array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
                                    requires_grad=True)
        x.sum(axis=axis).backward(gradient=NativeTensor.from_array(seed))
        assert np.array_equal(_grad(x), expected)
        x.close()


@needs_native
def test_sum_backward_negative_axis():
    x = NativeTensor.from_array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
                                requires_grad=True)
    x.sum(axis=-1).backward(gradient=NativeTensor.from_array([1.0, 2.0]))
    assert np.array_equal(_grad(x), [[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]])
    x.close()


@needs_native
def test_sum_backward_keepdims():
    x = NativeTensor.from_array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
                                requires_grad=True)
    out = x.sum(axis=1, keepdims=True)
    assert out.shape == (2, 1)
    out.backward(gradient=NativeTensor.from_array([[3.0], [7.0]]))
    assert np.array_equal(_grad(x), [[3.0, 3.0, 3.0], [7.0, 7.0, 7.0]])
    x.close()


@needs_native
def test_sum_backward_scalar_input():
    x = NativeTensor.full((), 5.0, requires_grad=True)
    x.sum().backward()
    assert x.grad.shape == ()
    assert float(_grad(x)) == 1.0
    x.close()


@needs_native
def test_sum_backward_one_element_input_versus_scalar_upstream():
    # (1,) input: sum() has scalar shape (), backward re-expands to (1,).
    x = NativeTensor.from_array([5.0], requires_grad=True)
    out = x.sum()
    assert out.shape == ()
    out.backward()
    assert x.grad.shape == (1,)
    assert np.array_equal(_grad(x), [1.0])
    x.close()


@needs_native
def test_sum_backward_transposed_and_narrowed_input():
    x = NativeTensor.from_array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
                                requires_grad=True)
    x.T.sum(axis=0).backward(gradient=NativeTensor.from_array([1.0, 2.0]))
    # x.T is (3, 2); axis 0 sums the (former column) axis, so each x row
    # receives its output cell's seed, then the transpose inverts back.
    assert np.array_equal(_grad(x), [[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]])
    x.close()


@needs_native
def test_mean_backward_all_elements():
    x = NativeTensor.from_array([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    x.mean().backward()
    assert np.allclose(_grad(x), np.full((2, 2), 0.25))
    x.close()


@needs_native
def test_mean_backward_axis_and_keepdims():
    x = NativeTensor.from_array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
                                requires_grad=True)
    x.mean(axis=1, keepdims=True).backward(
        gradient=NativeTensor.from_array([[3.0], [6.0]])
    )
    assert np.allclose(_grad(x), [[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]])
    x.close()


@needs_native
def test_mean_backward_negative_axis_count():
    x = NativeTensor.from_array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
                                requires_grad=True)
    x.mean(axis=-1).backward(gradient=NativeTensor.from_array([3.0, 6.0]))
    # count is shape[-1] == 3, so each element gets seed / 3.
    assert np.allclose(_grad(x), [[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]])
    x.close()


@needs_native
def test_mean_matches_finite_differences():
    values = np.array([[1.0, -2.0, 0.5], [3.0, 4.0, -1.5]])

    def pipeline(t):
        return t.mean()

    analytic = _backward_grad(pipeline, values)
    numeric = _numeric_grad(lambda v: _native_scalar(pipeline, v), values)
    assert np.allclose(analytic, numeric, atol=1e-6)


@needs_native
def test_broadcast_add_multiply_sum_matches_finite_differences():
    values = np.array([1.0, -0.5, 2.0])  # broadcast over (2, 3)
    other = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

    def pipeline(t):
        o = NativeTensor.from_array(other)
        return o.multiply(t).add(t).sum()

    analytic = _backward_grad(pipeline, values)
    numeric = _numeric_grad(lambda v: _native_scalar(pipeline, v), values)
    assert np.allclose(analytic, numeric, atol=1e-6)
    # And the exact formula: d/dt sum(o*t + t) = sum_rows(o) + 2.
    assert np.allclose(analytic, other.sum(axis=0) + 2.0)


# -- matmul ------------------------------------------------------------


@needs_native
def test_matmul_backward_rectangular_exact_formulas():
    a_vals = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])          # (2, 3)
    b_vals = np.array([[1.0, 0.0, 2.0, 1.0], [0.0, 1.0, 1.0, 3.0],
                       [2.0, 1.0, 0.0, 1.0]])                       # (3, 4)
    seed = np.arange(1.0, 9.0).reshape(2, 4)
    a = NativeTensor.from_array(a_vals, requires_grad=True)
    b = NativeTensor.from_array(b_vals, requires_grad=True)
    a.matmul(b).backward(gradient=NativeTensor.from_array(seed))
    assert np.allclose(_grad(a), seed @ b_vals.T)  # da = u @ b.T
    assert np.allclose(_grad(b), a_vals.T @ seed)  # db = a.T @ u
    a.close()
    b.close()


@needs_native
def test_matmul_backward_one_requiring_operand():
    a = NativeTensor.from_array([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    b = NativeTensor.from_array([[1.0, 0.0], [0.0, 1.0]])
    a.matmul(b).sum().backward()
    assert np.allclose(_grad(a), np.ones((2, 2)) @ np.eye(2).T)
    assert b.grad is None
    a.close()
    b.close()


@needs_native
def test_matmul_backward_transposed_operand():
    a_vals = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])   # (3, 2)
    b_vals = np.arange(1.0, 13.0).reshape(3, 4)               # (3, 4)
    a = NativeTensor.from_array(a_vals, requires_grad=True)
    b = NativeTensor.from_array(b_vals, requires_grad=True)
    out = a.T.matmul(b)  # (2, 3) @ (3, 4) through a strided view
    seed = np.ones((2, 4))
    out.backward(gradient=NativeTensor.from_array(seed))
    # matmul pushes u @ b.T to the a.T view; the transpose node then
    # inverts the permutation on the way to a.
    assert np.allclose(_grad(a), (seed @ b_vals.T).T)
    assert np.allclose(_grad(b), a_vals @ seed)  # (a.T).T @ u
    a.close()
    b.close()


@needs_native
def test_matmul_sum_matches_finite_differences():
    values = np.array([[0.5, -1.0, 2.0], [1.5, 0.5, -0.5]])   # (2, 3)
    other = np.array([[1.0, 2.0], [0.0, -1.0], [3.0, 1.0]])   # (3, 2)

    def pipeline(t):
        o = NativeTensor.from_array(other)
        return t.matmul(o).sum()

    analytic = _backward_grad(pipeline, values)
    numeric = _numeric_grad(lambda v: _native_scalar(pipeline, v), values)
    assert np.allclose(analytic, numeric, atol=1e-6)


@needs_native
def test_matmul_forward_errors_are_unchanged():
    a = NativeTensor.from_array([[1.0, 2.0]], requires_grad=True)
    b = NativeTensor.from_array([[1.0, 2.0]], requires_grad=True)
    with pytest.raises(ValueError, match="inner dimensions"):
        a.matmul(b)
    a.close()
    b.close()


# -- view backward: reshape / transpose / T / contiguous_copy ----------


@needs_native
def test_reshape_backward_restores_original_shape():
    x = NativeTensor.from_array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
                                requires_grad=True)
    seed = np.arange(1.0, 7.0).reshape(3, 2)
    x.reshape((3, 2)).backward(gradient=NativeTensor.from_array(seed))
    assert x.grad.shape == (2, 3)
    assert np.array_equal(_grad(x), seed.reshape(2, 3))
    x.close()


@needs_native
def test_reshape_backward_scalar_and_one_element():
    x = NativeTensor.full((), 3.0, requires_grad=True)
    x.reshape((1,)).sum().backward()
    assert x.grad.shape == ()
    assert float(_grad(x)) == 1.0
    x.close()
    y = NativeTensor.from_array([3.0], requires_grad=True)
    y.reshape(()).backward()  # scalar output seeds 1.0
    assert y.grad.shape == (1,)
    assert np.array_equal(_grad(y), [1.0])
    y.close()


@needs_native
def test_transpose_backward_default_and_T():
    seed = np.arange(1.0, 7.0).reshape(3, 2)
    for view in ("transpose", "T"):
        x = NativeTensor.from_array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
                                    requires_grad=True)
        out = x.transpose() if view == "transpose" else x.T
        out.backward(gradient=NativeTensor.from_array(seed))
        assert x.grad.shape == (2, 3)
        assert np.array_equal(_grad(x), seed.T)
        x.close()


@needs_native
def test_transpose_backward_explicit_permutation():
    values = np.arange(24.0).reshape(2, 3, 4)
    x = NativeTensor.from_array(values, requires_grad=True)
    out = x.transpose(1, 2, 0)  # (3, 4, 2)
    seed = np.arange(24.0).reshape(3, 4, 2)
    out.backward(gradient=NativeTensor.from_array(seed))
    # The inverse of (1, 2, 0) is (2, 0, 1).
    assert np.array_equal(_grad(x), seed.transpose(2, 0, 1))
    x.close()


@needs_native
def test_chained_views_and_matmul_backward():
    a_vals = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])  # (3, 2)
    w_vals = np.array([[1.0, -1.0, 0.5], [2.0, 0.0, 1.0]])   # (2, 3)
    a = NativeTensor.from_array(a_vals, requires_grad=True)
    w = NativeTensor.from_array(w_vals, requires_grad=True)
    # (2, 3) view @ (3, 2) view -> reshape -> sum: mixes both view rules
    # with matmul in one chain.
    a.T.matmul(w.T).reshape((4,)).sum().backward()
    ones = np.ones((2, 2))
    assert np.allclose(_grad(a), (ones @ w_vals).T)
    assert np.allclose(_grad(w), (a_vals @ ones).T)
    a.close()
    w.close()


@needs_native
def test_reshape_transpose_chain_matches_finite_differences():
    values = np.array([[0.5, -1.0, 2.0], [1.5, 0.5, -0.5]])

    def pipeline(t):
        # reshape requires a contiguous source, so the transposed view
        # goes through the (differentiable) contiguous_copy first. The
        # owning copy is bound to a name so the borrowing reshape stays
        # valid in the forward-only runs too (views borrow storage).
        copied = t.T.contiguous_copy()
        flat = copied.reshape((6,))
        return flat.multiply(flat).sum()

    analytic = _backward_grad(pipeline, values)
    numeric = _numeric_grad(lambda v: _native_scalar(pipeline, v), values)
    assert np.allclose(analytic, numeric, atol=1e-6)
    assert np.allclose(analytic, 2.0 * values)  # d sum(x^2) / dx


@needs_native
def test_contiguous_copy_backward_is_identity():
    x = NativeTensor.from_array([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    seed = np.array([[5.0, 6.0], [7.0, 8.0]])
    x.contiguous_copy().backward(gradient=NativeTensor.from_array(seed))
    assert np.array_equal(_grad(x), seed)
    x.close()


@needs_native
def test_contiguous_copy_backward_through_strided_view():
    x = NativeTensor.from_array([[1.0, -2.0], [3.0, 4.0]], requires_grad=True)
    # T -> contiguous_copy -> relu -> sum exercises the identity rule in
    # the middle of a real chain over a non-contiguous parent.
    x.T.contiguous_copy().relu().sum().backward()
    assert np.array_equal(_grad(x), [[1.0, 0.0], [1.0, 1.0]])
    x.close()


# -- engine integration over real ops ----------------------------------


@needs_native
def test_branching_graph_with_real_ops():
    x_vals = np.array([1.0, -2.0, 3.0])
    x = NativeTensor.from_array(x_vals, requires_grad=True)
    x.multiply(x).add(x).sum().backward()  # d(x^2 + x)/dx = 2x + 1
    assert np.allclose(_grad(x), 2.0 * x_vals + 1.0)
    x.close()


@needs_native
def test_shared_subgraph_accumulates_through_both_consumers():
    x_vals = np.array([1.0, 2.0])
    y_vals = np.array([3.0, 4.0])
    x = NativeTensor.from_array(x_vals, requires_grad=True)
    y = NativeTensor.from_array(y_vals, requires_grad=True)
    h = x.add(y)                      # shared intermediate
    h.multiply(h).sum().backward()    # d sum(h^2) = 2h into both leaves
    expected = 2.0 * (x_vals + y_vals)
    assert np.allclose(_grad(x), expected)
    assert np.allclose(_grad(y), expected)
    x.close()
    y.close()


@needs_native
def test_repeated_backward_accumulates_and_zero_grad_resets():
    x_vals = np.array([1.0, 2.0])
    x = NativeTensor.from_array(x_vals, requires_grad=True)
    x.multiply(x).sum().backward()
    x.multiply(x).sum().backward()
    assert np.allclose(_grad(x), 4.0 * x_vals)  # two passes of 2x
    x.zero_grad()
    x.multiply(x).sum().backward()
    assert np.allclose(_grad(x), 2.0 * x_vals)
    x.close()


@needs_native
def test_detach_breaks_a_real_chain():
    x_vals = np.array([1.0, 2.0])
    x = NativeTensor.from_array(x_vals, requires_grad=True)
    d = x.multiply(x).detach()  # value x^2, no history
    assert d.requires_grad is False
    y = NativeTensor.from_array([1.0, 1.0], requires_grad=True)
    d.multiply(y).sum().backward()
    assert x.grad is None                       # gradient stopped at detach
    assert np.allclose(_grad(y), x_vals ** 2)   # d(x^2 * y)/dy = x^2
    x.close()
    d.close()
    y.close()


@needs_native
def test_closing_an_operand_before_backward_raises_clearly():
    a = NativeTensor.from_array([1.0, 2.0], requires_grad=True)
    b = NativeTensor.from_array([3.0, 4.0], requires_grad=True)
    out = a.multiply(b).sum()
    a.close()
    with pytest.raises(RuntimeError, match="closed"):
        out.backward()
    b.close()


@needs_native
def test_closing_an_intermediate_before_backward_raises_clearly():
    a = NativeTensor.from_array([1.0, 2.0], requires_grad=True)
    mid = a.multiply(a)
    out = mid.sum()
    mid.close()
    with pytest.raises(RuntimeError, match="closed"):
        out.backward()
    a.close()


@needs_native
def test_real_op_gradients_honor_the_metadata_contract():
    x = NativeTensor.from_array([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    x.multiply(x).mean().backward()
    assert isinstance(x.grad, NativeTensor)
    assert not isinstance(x.grad, np.ndarray)
    assert x.grad.shape == x.shape
    assert x.grad.dtype == x.dtype == "float64"
    assert x.grad.device == x.device == "cpu"
    x.close()


@needs_native
def test_leaf_operands_stay_open_and_unchanged_after_backward():
    a_vals = np.array([[1.0, 2.0], [3.0, 4.0]])
    a = NativeTensor.from_array(a_vals, requires_grad=True)
    b = NativeTensor.from_array(a_vals * 2.0, requires_grad=True)
    a.multiply(b).sum().backward()
    # Forward values captured by the graph were not mutated or closed.
    assert np.array_equal(a.to_numpy(), a_vals)
    assert np.array_equal(b.to_numpy(), a_vals * 2.0)
    assert a.closed is False and b.closed is False
    a.close()
    b.close()


# ======================================================================
# v2.3 — narrow backward
# ======================================================================
#
# narrow(dim, start, length) is a view; its backward scatters the upstream
# gradient into a fresh zeros tensor of the parent's shape at the narrowed
# region (via the native tf_core_narrow_backward kernel). References below
# are built with NumPy zero-padding; every gradient under test is native.


def _narrow_reference(parent_shape, dim, start, upstream):
    """A NumPy ``zeros(parent_shape)`` with ``upstream`` written into the
    narrowed region — the independent reference for narrow backward."""
    grad = np.zeros(parent_shape, dtype=np.float64)
    index = [slice(None)] * len(parent_shape)
    index[dim] = slice(start, start + upstream.shape[dim])
    grad[tuple(index)] = upstream
    return grad


# -- basic scatter -----------------------------------------------------


@needs_native
def test_narrow_backward_1d():
    x = NativeTensor.from_array([1.0, 2.0, 3.0, 4.0, 5.0], requires_grad=True)
    seed = np.array([10.0, 20.0])
    x.narrow(0, 1, 2).backward(gradient=NativeTensor.from_array(seed))
    assert x.grad.shape == (5,)
    assert np.array_equal(_grad(x), [0.0, 10.0, 20.0, 0.0, 0.0])
    x.close()


@needs_native
def test_narrow_backward_2d_axis0():
    x = NativeTensor.from_array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
                                requires_grad=True)
    seed = np.array([[7.0, 8.0], [9.0, 10.0]])
    x.narrow(0, 0, 2).backward(gradient=NativeTensor.from_array(seed))
    assert np.array_equal(_grad(x), _narrow_reference((3, 2), 0, 0, seed))
    assert np.array_equal(_grad(x), [[7.0, 8.0], [9.0, 10.0], [0.0, 0.0]])
    x.close()


@needs_native
def test_narrow_backward_2d_axis1():
    x = NativeTensor.from_array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
                                requires_grad=True)
    seed = np.array([[7.0], [8.0]])
    x.narrow(1, 2, 1).backward(gradient=NativeTensor.from_array(seed))
    assert np.array_equal(_grad(x), [[0.0, 0.0, 7.0], [0.0, 0.0, 8.0]])
    x.close()


@needs_native
def test_narrow_backward_full_length_covers_everything():
    # start=0, length=full: the gradient equals the upstream everywhere.
    x = NativeTensor.from_array([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    seed = np.array([[5.0, 6.0], [7.0, 8.0]])
    x.narrow(0, 0, 2).backward(gradient=NativeTensor.from_array(seed))
    assert np.array_equal(_grad(x), seed)
    x.close()


@needs_native
def test_narrow_backward_length_one():
    x = NativeTensor.from_array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
                                requires_grad=True)
    x.narrow(1, 1, 1).sum().backward()
    assert np.array_equal(_grad(x), [[0.0, 1.0, 0.0], [0.0, 1.0, 0.0]])
    x.close()


@needs_native
def test_narrow_backward_nonzero_start_3d():
    values = np.arange(24.0).reshape(2, 3, 4)
    x = NativeTensor.from_array(values, requires_grad=True)
    seed = np.arange(1.0, 9.0).reshape(2, 1, 4)
    x.narrow(1, 2, 1).backward(gradient=NativeTensor.from_array(seed))
    assert np.array_equal(_grad(x), _narrow_reference((2, 3, 4), 1, 2, seed))
    x.close()


@needs_native
def test_narrow_backward_zeros_outside_and_equals_upstream_inside():
    x = NativeTensor.from_array(np.arange(20.0).reshape(4, 5),
                                requires_grad=True)
    seed = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])
    x.narrow(1, 2, 2).backward(gradient=NativeTensor.from_array(seed))
    g = _grad(x)
    # Inside the [:, 2:4] window: exactly the upstream. Everywhere else: 0.
    assert np.array_equal(g[:, 2:4], seed)
    outside = np.ones((4, 5), dtype=bool)
    outside[:, 2:4] = False
    assert np.array_equal(g[outside], np.zeros(outside.sum()))
    x.close()


@needs_native
def test_narrow_backward_explicit_nonscalar_gradient():
    x = NativeTensor.from_array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
                                requires_grad=True)
    out = x.narrow(0, 1, 1)  # (1, 3)
    assert out.shape == (1, 3)
    out.backward(gradient=NativeTensor.from_array([[10.0, 20.0, 30.0]]))
    assert np.array_equal(_grad(x), [[0.0, 0.0, 0.0], [10.0, 20.0, 30.0]])
    x.close()


# -- autograd integration ----------------------------------------------


@needs_native
def test_narrow_sum_backward():
    x = NativeTensor.from_array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
                                requires_grad=True)
    x.narrow(1, 0, 2).sum().backward()
    assert np.array_equal(_grad(x), [[1.0, 1.0, 0.0], [1.0, 1.0, 0.0]])
    x.close()


@needs_native
def test_narrow_mean_backward():
    x = NativeTensor.from_array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
                                requires_grad=True)
    # mean over the (2, 2) narrowed window: 4 elements, each gets 1/4.
    x.narrow(1, 1, 2).mean().backward()
    assert np.allclose(_grad(x), [[0.0, 0.25, 0.25], [0.0, 0.25, 0.25]])
    x.close()


@needs_native
def test_narrow_multiply_sum_backward():
    x_vals = np.array([[1.0, -2.0, 3.0], [4.0, 5.0, -6.0]])
    x = NativeTensor.from_array(x_vals, requires_grad=True)
    # narrow -> square -> sum: grad is 2x inside the window, 0 outside.
    n = x.narrow(1, 1, 2)
    n.multiply(n).sum().backward()
    expected = np.zeros((2, 3))
    expected[:, 1:3] = 2.0 * x_vals[:, 1:3]
    assert np.allclose(_grad(x), expected)
    x.close()


@needs_native
def test_narrow_of_non_requiring_parent_contributes_no_gradient():
    const = NativeTensor.from_array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])  # no grad
    w = NativeTensor.from_array([[10.0, 20.0], [30.0, 40.0]], requires_grad=True)
    sliced = const.narrow(1, 0, 2)         # (2, 2), no grad
    sliced.multiply(w).sum().backward()
    assert sliced.grad is None             # narrow of a constant: no grad
    assert np.array_equal(_grad(w), [[1.0, 2.0], [4.0, 5.0]])
    const.close()
    w.close()


@needs_native
def test_narrow_backward_repeated_accumulates_and_zero_grad_resets():
    x = NativeTensor.from_array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
                                requires_grad=True)
    x.narrow(1, 0, 2).sum().backward()
    x.narrow(1, 0, 2).sum().backward()
    assert np.array_equal(_grad(x), [[2.0, 2.0, 0.0], [2.0, 2.0, 0.0]])
    x.zero_grad()
    x.narrow(1, 0, 2).sum().backward()
    assert np.array_equal(_grad(x), [[1.0, 1.0, 0.0], [1.0, 1.0, 0.0]])
    x.close()


# -- views: narrow over / under other view ops -------------------------


@needs_native
def test_narrow_on_transposed_parent():
    x_vals = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])  # (2, 3)
    x = NativeTensor.from_array(x_vals, requires_grad=True)
    # x.T is (3, 2); narrow keeps its first 2 rows -> (2, 2).
    x.T.narrow(0, 0, 2).sum().backward()
    grad_xt = np.zeros((3, 2))
    grad_xt[0:2, :] = 1.0
    assert np.array_equal(_grad(x), grad_xt.T)  # transpose backward inverts
    x.close()


@needs_native
def test_nested_narrow_backward():
    x = NativeTensor.from_array(np.arange(20.0).reshape(4, 5), requires_grad=True)
    # narrow rows 1..2, then cols 1..2 of that -> the x[1:3, 1:3] window.
    inner = x.narrow(0, 1, 2).narrow(1, 1, 2)
    seed = np.array([[1.0, 2.0], [3.0, 4.0]])
    inner.backward(gradient=NativeTensor.from_array(seed))
    expected = np.zeros((4, 5))
    expected[1:3, 1:3] = seed
    assert np.array_equal(_grad(x), expected)
    x.close()


@needs_native
def test_narrow_on_nonzero_offset_parent():
    x = NativeTensor.from_array(np.arange(15.0).reshape(3, 5), requires_grad=True)
    # The first narrow gives a nonzero-offset parent (rows 1..2, offset 5);
    # narrowing it again must still scatter into the right x positions.
    rows = x.narrow(0, 1, 2)               # x[1:3], offset 5, shape (2, 5)
    rows.narrow(1, 3, 2).sum().backward()  # cols 3..4 of that window
    expected = np.zeros((3, 5))
    expected[1:3, 3:5] = 1.0
    assert np.array_equal(_grad(x), expected)
    x.close()


@needs_native
def test_narrow_then_transpose_backward():
    x = NativeTensor.from_array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
                                requires_grad=True)
    x.narrow(1, 0, 2).T.sum().backward()  # (2, 2) window -> transpose -> sum
    assert np.array_equal(_grad(x), [[1.0, 1.0, 0.0], [1.0, 1.0, 0.0]])
    x.close()


@needs_native
def test_narrow_then_reshape_backward():
    x = NativeTensor.from_array([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]],
                                requires_grad=True)
    # The narrowed (2, 2) view is non-contiguous, so reshape goes through a
    # (differentiable) contiguous_copy first — narrow feeding two more views.
    x.narrow(1, 0, 2).contiguous_copy().reshape((4,)).sum().backward()
    assert np.array_equal(_grad(x), [[1.0, 1.0, 0.0, 0.0], [1.0, 1.0, 0.0, 0.0]])
    x.close()


@needs_native
def test_narrow_backward_leaf_grad_owns_contiguous_storage():
    x = NativeTensor.from_array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
                                requires_grad=True)
    x.narrow(1, 1, 2).sum().backward()
    assert x.grad.owns_core is True     # fresh owning storage, not a view
    assert x.grad.contiguous is True    # row-major contiguous
    assert x.grad.shape == (2, 3)
    assert np.array_equal(_grad(x), [[0.0, 1.0, 1.0], [0.0, 1.0, 1.0]])
    x.close()


# -- correctness: NumPy reference and finite differences ---------------


@needs_native
def test_narrow_backward_matches_numpy_reference():
    values = np.arange(24.0).reshape(4, 6)
    x = NativeTensor.from_array(values, requires_grad=True)
    seed = np.arange(1.0, 9.0).reshape(4, 2)
    x.narrow(1, 3, 2).backward(gradient=NativeTensor.from_array(seed))
    assert np.array_equal(_grad(x), _narrow_reference((4, 6), 1, 3, seed))
    x.close()


@needs_native
def test_narrow_sum_matches_finite_differences():
    values = np.array([[0.5, -1.0, 2.0, 1.0], [1.5, 0.5, -0.5, 3.0]])

    def pipeline(t):
        n = t.narrow(1, 1, 2)
        return n.multiply(n).sum()

    analytic = _backward_grad(pipeline, values)
    numeric = _numeric_grad(lambda v: _native_scalar(pipeline, v), values)
    assert np.allclose(analytic, numeric, atol=1e-6)
    expected = np.zeros_like(values)
    expected[:, 1:3] = 2.0 * values[:, 1:3]  # 2x inside, 0 outside
    assert np.allclose(analytic, expected)


# -- errors / lifetime -------------------------------------------------


@needs_native
def test_narrow_backward_closed_parent_raises():
    x = NativeTensor.from_array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
                                requires_grad=True)
    out = x.narrow(1, 0, 2).sum()
    x.close()  # the narrow node borrows x's storage
    with pytest.raises(RuntimeError, match="closed"):
        out.backward()


@needs_native
def test_narrow_backward_closed_output_raises():
    x = NativeTensor.from_array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
                                requires_grad=True)
    out = x.narrow(1, 0, 2)
    out.close()
    with pytest.raises(RuntimeError, match="closed"):
        out.backward(gradient=NativeTensor.from_array([[1.0, 1.0], [1.0, 1.0]]))
    x.close()


@needs_native
def test_narrow_backward_wrong_upstream_shape_raises():
    x = NativeTensor.from_array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
                                requires_grad=True)
    out = x.narrow(1, 0, 2)  # output shape (2, 2)
    bad = NativeTensor.from_array([1.0, 2.0, 3.0])  # (3,)
    with pytest.raises(ValueError) as excinfo:
        out.backward(gradient=bad)
    assert "(2, 2)" in str(excinfo.value)
    x.close()
    bad.close()


@needs_native
def test_narrow_backward_core_validates_shape_compatibility():
    # Direct core-level check: a non-narrowed extent that disagrees with
    # the original shape is rejected before any scatter runs.
    upstream = cpp.NativeTensorCore.from_array([[1.0, 2.0]])  # (1, 2)
    with pytest.raises(ValueError, match="compatible"):
        upstream.narrow_backward(1, 0, (2, 3))  # axis-0 extent 1 != 2
    upstream.close()


@needs_native
def test_narrow_backward_core_validates_bounds():
    upstream = cpp.NativeTensorCore.from_array([[1.0, 2.0], [3.0, 4.0]])  # (2, 2)
    with pytest.raises(ValueError, match="out of bounds"):
        upstream.narrow_backward(1, 2, (2, 3))  # start 2 + length 2 > 3
    upstream.close()


@needs_native
def test_narrow_forward_errors_unchanged_for_requiring_input():
    x = NativeTensor.from_array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
                                requires_grad=True)
    with pytest.raises(ValueError, match="out of bounds"):
        x.narrow(1, 2, 5)          # start + length > size
    with pytest.raises(ValueError, match="dim must be in"):
        x.narrow(-1, 0, 1)         # negative dim unsupported (forward error)
    with pytest.raises(TypeError, match="must be an int"):
        x.narrow(0, 0.5, 1)        # non-int start
    x.close()


# -- isolation ---------------------------------------------------------


@needs_native
def test_narrow_backward_grad_is_native_not_numpy():
    x = NativeTensor.from_array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
                                requires_grad=True)
    x.narrow(1, 0, 2).sum().backward()
    assert isinstance(x.grad, NativeTensor)
    assert not isinstance(x.grad, np.ndarray)
    assert x.grad.dtype == x.dtype == "float64"
    assert x.grad.device == x.device == "cpu"
    x.close()


@needs_native
def test_narrow_backward_leaves_parent_open_and_unchanged():
    x_vals = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    x = NativeTensor.from_array(x_vals, requires_grad=True)
    x.narrow(1, 0, 2).sum().backward()
    assert x.closed is False
    assert np.array_equal(x.to_numpy(), x_vals)  # forward value untouched
    x.close()


# ======================================================================
# v2.4 — graph lifetime policy (retain_graph, one-shot free, failure safety)
# ======================================================================
#
# backward(gradient=None, retain_graph=False): the default is one-shot and
# releases the traversed operation graph; retain_graph=True keeps it for
# another pass. Leaf gradients accumulate across successful passes; a freed
# graph raises deterministically; a failed pass rolls back. Selector:
#   -k "graph_lifetime or retain_graph or freed_graph"


# -- retain_graph argument validation ----------------------------------


@needs_native
def test_retain_graph_defaults_to_false():
    x = NativeTensor.from_array([1.0, 2.0, 3.0], requires_grad=True)
    out = x.multiply(x).sum()
    out.backward()  # no retain_graph -> default False -> one-shot free
    assert out._graph_freed is True
    with pytest.raises(RuntimeError, match="freed"):
        out.backward()
    x.close()


@needs_native
def test_retain_graph_true_accepted():
    x_vals = np.array([1.0, 2.0])
    x = NativeTensor.from_array(x_vals, requires_grad=True)
    out = x.multiply(x).sum()
    out.backward(retain_graph=True)
    assert out._graph_freed is False
    assert np.allclose(_grad(x), 2.0 * x_vals)
    x.close()


@needs_native
def test_retain_graph_false_accepted():
    x_vals = np.array([1.0, 2.0])
    x = NativeTensor.from_array(x_vals, requires_grad=True)
    out = x.multiply(x).sum()
    out.backward(retain_graph=False)
    assert out._graph_freed is True
    assert np.allclose(_grad(x), 2.0 * x_vals)
    x.close()


@needs_native
def test_retain_graph_rejects_non_bool():
    x = NativeTensor.from_array([1.0, 2.0], requires_grad=True)
    for bad in (1, 0, "true", 1.0, None, [], object()):
        out = x.multiply(x).sum()
        with pytest.raises(TypeError, match="retain_graph must be a bool"):
            out.backward(retain_graph=bad)
    x.close()


@needs_native
def test_retain_graph_validated_before_any_mutation():
    x_vals = np.array([1.0, 2.0])
    x = NativeTensor.from_array(x_vals, requires_grad=True)
    out = x.multiply(x).sum()
    with pytest.raises(TypeError, match="retain_graph must be a bool"):
        out.backward(retain_graph="yes")
    # No gradient produced, graph still valid (nothing mutated or freed).
    assert x.grad is None
    assert out._graph_freed is False
    assert callable(out._backward)
    # A real backward still works afterward.
    out.backward()
    assert np.allclose(_grad(x), 2.0 * x_vals)
    x.close()


# -- one-shot free and freed-graph errors ------------------------------


@needs_native
def test_freed_graph_default_backward_computes_and_frees():
    x_vals = np.array([1.0, -2.0, 3.0])
    x = NativeTensor.from_array(x_vals, requires_grad=True)
    mid = x.multiply(x)
    out = mid.sum()
    out.backward()
    assert np.allclose(_grad(x), 2.0 * x_vals)      # correct gradient
    assert out._graph_freed is True and mid._graph_freed is True
    assert out._parents == () and mid._parents == ()
    assert out._backward is None and mid._backward is None
    assert x._graph_freed is False and x.requires_grad is True  # leaf untouched
    x.close()


@needs_native
def test_freed_graph_second_backward_raises_and_names_retain_graph():
    x = NativeTensor.from_array([1.0, 2.0], requires_grad=True)
    out = x.multiply(x).sum()
    out.backward()
    with pytest.raises(RuntimeError) as excinfo:
        out.backward()
    msg = str(excinfo.value)
    assert "freed" in msg and "retain_graph=True" in msg
    x.close()


@needs_native
def test_freed_graph_second_backward_leaves_grad_unchanged():
    x_vals = np.array([1.0, 2.0, 3.0])
    x = NativeTensor.from_array(x_vals, requires_grad=True)
    out = x.multiply(x).sum()
    out.backward()
    before = x.grad.to_numpy().copy()
    with pytest.raises(RuntimeError, match="freed"):
        out.backward()
    assert np.array_equal(x.grad.to_numpy(), before)  # failed call changed nothing
    x.close()


@needs_native
def test_freed_graph_shared_intermediate_detected_via_other_output():
    x_vals = np.array([1.0, 2.0, 3.0])
    x = NativeTensor.from_array(x_vals, requires_grad=True)
    shared = x.multiply(x)
    out_a = shared.sum()
    out_b = shared.mean()
    out_a.backward()                 # frees the shared intermediate
    assert shared._graph_freed is True
    before = x.grad.to_numpy().copy()
    with pytest.raises(RuntimeError, match="freed"):
        out_b.backward()             # reaches the freed shared node
    assert np.array_equal(x.grad.to_numpy(), before)
    x.close()


@needs_native
def test_freed_graph_new_op_from_freed_value_raises():
    x_vals = np.array([1.0, 2.0, 3.0])
    x = NativeTensor.from_array(x_vals, requires_grad=True)
    shared = x.multiply(x)
    shared.sum().backward()          # frees shared
    # Forward still works on the freed tensor's stored value...
    y = NativeTensor.from_array([1.0, 1.0, 1.0], requires_grad=True)
    new_out = shared.add(y).sum()
    assert np.allclose(new_out.to_numpy(), (x_vals ** 2 + 1.0).sum())
    # ...but backward must not silently cross the freed history.
    with pytest.raises(RuntimeError, match="freed"):
        new_out.backward()
    assert y.grad is None            # nothing committed
    x.close()
    y.close()


@needs_native
def test_freed_graph_cleanup_clears_graph_metadata():
    a = NativeTensor.from_array([1.0, 2.0], requires_grad=True)
    b = NativeTensor.from_array([3.0, 4.0], requires_grad=True)
    mid = a.multiply(b)
    out = mid.sum()
    assert out._parents == (mid,) and mid._parents == (a, b)
    assert callable(out._backward) and callable(mid._backward)
    out.backward()
    for node in (out, mid):
        assert node._parents == ()
        assert node._backward is None
        assert node._graph_freed is True
        assert node._grad is None        # transient grad dropped
        assert node.is_leaf is False     # not converted to a leaf
    assert np.allclose(mid.to_numpy(), [3.0, 8.0])  # value still usable forward
    a.close()
    b.close()


# -- retain_graph reuse -------------------------------------------------


@needs_native
def test_retain_graph_two_passes_accumulate_twice():
    x_vals = np.array([1.0, 2.0, 3.0])
    x = NativeTensor.from_array(x_vals, requires_grad=True)
    out = x.multiply(x).sum()        # per-pass grad = 2x
    out.backward(retain_graph=True)
    out.backward(retain_graph=True)
    assert np.allclose(_grad(x), 4.0 * x_vals)   # exactly two 2x contributions
    assert out._graph_freed is False
    x.close()


@needs_native
def test_retain_graph_then_default_frees_then_reuse_raises():
    x_vals = np.array([1.0, 2.0])
    x = NativeTensor.from_array(x_vals, requires_grad=True)
    out = x.multiply(x).sum()
    out.backward(retain_graph=True)  # 2x
    out.backward(retain_graph=True)  # 4x
    out.backward()                   # 6x, then free
    assert np.allclose(_grad(x), 6.0 * x_vals)
    assert out._graph_freed is True
    with pytest.raises(RuntimeError, match="freed"):
        out.backward()
    x.close()


@needs_native
def test_retain_graph_keeps_graph_metadata():
    a = NativeTensor.from_array([1.0, 2.0], requires_grad=True)
    b = NativeTensor.from_array([3.0, 4.0], requires_grad=True)
    mid = a.multiply(b)
    out = mid.sum()
    out.backward(retain_graph=True)
    assert out._parents == (mid,) and mid._parents == (a, b)
    assert callable(out._backward) and callable(mid._backward)
    assert out._graph_freed is False and mid._graph_freed is False
    assert out._grad is None and mid._grad is None  # transient grads still dropped
    a.close()
    b.close()


@needs_native
def test_retain_graph_zero_grad_between_passes():
    x_vals = np.array([1.0, 2.0, 3.0])
    x = NativeTensor.from_array(x_vals, requires_grad=True)
    out = x.multiply(x).sum()
    out.backward(retain_graph=True)
    assert np.allclose(_grad(x), 2.0 * x_vals)
    x.zero_grad()
    assert x.grad is None
    assert out._graph_freed is False       # zero_grad did not damage the graph
    out.backward(retain_graph=True)
    assert np.allclose(_grad(x), 2.0 * x_vals)  # fresh accumulation from None
    x.close()


@needs_native
def test_freed_graph_zero_grad_does_not_resurrect():
    x = NativeTensor.from_array([1.0, 2.0], requires_grad=True)
    out = x.multiply(x).sum()
    out.backward()                 # frees graph
    x.zero_grad()                  # clears leaf grad only
    assert out._graph_freed is True
    with pytest.raises(RuntimeError, match="freed"):
        out.backward()
    x.close()


# -- leaf behavior ------------------------------------------------------


@needs_native
def test_graph_lifetime_scalar_leaf_repeated_backward_accumulates():
    x = NativeTensor.full((), 5.0, requires_grad=True)  # scalar leaf, no graph
    x.backward()                    # seed 1.0
    x.backward()                    # +1.0
    x.backward(retain_graph=True)   # +1.0; retain has no effect on a leaf
    assert x.grad.shape == ()
    assert float(_grad(x)) == 3.0
    assert x._graph_freed is False  # a leaf is never marked freed
    assert x.is_leaf is True
    x.close()


@needs_native
def test_graph_lifetime_detach_is_graph_free():
    x = NativeTensor.from_array([1.0, 2.0], requires_grad=True)
    d = x.multiply(x).detach()
    assert d.requires_grad is False
    assert d.is_leaf is True
    assert d._graph_freed is False  # detached leaf, never part of a graph
    assert d._parents == ()
    assert d._backward is None
    x.close()
    d.close()


# -- graph shapes still correct under retain_graph ---------------------


@needs_native
def test_retain_graph_branching_accumulates():
    x_vals = np.array([1.0, 2.0])
    y_vals = np.array([3.0, 4.0])
    x = NativeTensor.from_array(x_vals, requires_grad=True)
    y = NativeTensor.from_array(y_vals, requires_grad=True)
    h = x.add(y)
    out = h.multiply(h).sum()       # per-pass grad = 2h into both leaves
    out.backward(retain_graph=True)
    out.backward(retain_graph=True)
    expected = 2.0 * 2.0 * (x_vals + y_vals)
    assert np.allclose(_grad(x), expected)
    assert np.allclose(_grad(y), expected)
    x.close()
    y.close()


@needs_native
def test_retain_graph_duplicate_parent_accumulates():
    x_vals = np.array([1.0, 2.0, 3.0])
    x = NativeTensor.from_array(x_vals, requires_grad=True)
    out = x.add(x).sum()            # parents (x, x); per-pass grad = 2
    out.backward(retain_graph=True)
    out.backward(retain_graph=True)
    assert np.allclose(_grad(x), np.full(3, 4.0))  # 2 per pass, twice
    x.close()


@needs_native
def test_retain_graph_explicit_gradient():
    a_vals = np.array([[1.0, 2.0], [3.0, 4.0]])
    a = NativeTensor.from_array(a_vals, requires_grad=True)
    out = a.multiply(a)            # non-scalar output -> explicit gradient
    seed = np.array([[1.0, 1.0], [1.0, 1.0]])
    out.backward(gradient=NativeTensor.from_array(seed), retain_graph=True)
    out.backward(gradient=NativeTensor.from_array(seed), retain_graph=True)
    assert np.allclose(_grad(a), 2.0 * 2.0 * a_vals)  # d(a^2)=2a, ones seed, twice
    a.close()


@needs_native
def test_retain_graph_broadcasting():
    bias_vals = np.array([1.0, 2.0, 3.0])
    x_vals = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    bias = NativeTensor.from_array(bias_vals, requires_grad=True)
    x = NativeTensor.from_array(x_vals, requires_grad=True)
    out = x.add(bias).sum()
    out.backward(retain_graph=True)
    out.backward(retain_graph=True)
    assert np.allclose(_grad(bias), np.full(3, 4.0))  # (3,) grad summed rows, twice
    assert np.allclose(_grad(x), np.full((2, 3), 2.0))
    bias.close()
    x.close()


@needs_native
def test_retain_graph_matmul():
    a_vals = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])    # (2, 3)
    b_vals = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])  # (3, 2)
    a = NativeTensor.from_array(a_vals, requires_grad=True)
    b = NativeTensor.from_array(b_vals, requires_grad=True)
    out = a.matmul(b).sum()
    out.backward(retain_graph=True)
    out.backward(retain_graph=True)
    ones = np.ones((2, 2))
    assert np.allclose(_grad(a), 2.0 * (ones @ b_vals.T))
    assert np.allclose(_grad(b), 2.0 * (a_vals.T @ ones))
    a.close()
    b.close()


@needs_native
def test_graph_lifetime_view_chain_reshape_transpose_contiguous_narrow():
    x_vals = np.arange(6.0).reshape(2, 3)
    x = NativeTensor.from_array(x_vals, requires_grad=True)
    # A chain mixing every view op; obeys the same lifetime policy.
    out = x.T.contiguous_copy().reshape((6,)).narrow(0, 1, 4).sum()
    expected_pass = np.array([[0.0, 1.0, 1.0], [1.0, 1.0, 0.0]])
    out.backward(retain_graph=True)
    assert np.allclose(_grad(x), expected_pass)
    out.backward()                 # default -> accumulate once more, then free
    assert np.allclose(_grad(x), 2.0 * expected_pass)
    assert out._graph_freed is True
    with pytest.raises(RuntimeError, match="freed"):
        out.backward()
    x.close()


# -- failure safety -----------------------------------------------------


@needs_native
def test_freed_graph_failed_backward_does_not_free_or_partially_commit():
    a = NativeTensor.from_array([1.0, 2.0], requires_grad=True)
    b = NativeTensor.from_array([3.0, 4.0], requires_grad=True)
    add = a.add(b)
    out = add.sum()
    assert a.grad is None
    # add's backward commits a's contribution, then b's branch raises
    # (b closed) — the staged pass must roll a's back and leave the graph.
    b.close()
    with pytest.raises(RuntimeError, match="closed"):
        out.backward()
    assert a.grad is None                         # no partial commit
    assert out._graph_freed is False              # no partial free
    assert add._graph_freed is False
    assert out._parents == (add,) and add._parents == (a, b)
    assert callable(out._backward) and callable(add._backward)
    a.close()
