"""Cross-cutting Phase B guardrail tests (Advanced C++ v2.6).

This is the **completion** milestone for Phase B — native autograd. Where
``tests/test_native_autograd.py`` verifies each backward *rule* (v2.1–v2.4)
and ``tests/test_native_autograd_benchmark.py`` verifies the *benchmark
harness* (v2.5), this file locks several completed Phase B invariants
together as cross-cutting guardrails, so they cannot silently erode:

1.  the native gradient path never falls back to NumPy compute;
2.  ``NativeTensor`` and ``tensorforge.Tensor`` stay fully isolated;
3.  the native backend is reached only through the explicit experimental
    API — no implicit dispatch, no silent fallback;
4.  gradient ownership (leaf-only retention, native/float64/cpu grads,
    accumulation, ``zero_grad``);
5.  the graph lifetime policy holds across a realistic mixed graph;
6.  ``detach`` returns an independent, graph-free owning copy;
7.  view + offset backward (transpose→narrow) scatters correctly and owns
    contiguous storage;
8.  an invalid/closed operand fails without partial commit or partial free;
9.  the public raw-buffer kernel registry does not leak the internal fused
    backward kernels;
10. the v2.5 benchmark modes keep their documented meanings.

The tests are named so this selector runs exactly this cross-cutting set:

    python -m pytest -q -k "phase_b_guardrail or native_autograd_guardrail or native_backend_isolation"

Native-backend tests skip when the compiled library is not built, matching
the rest of the native suite. See docs/native_autograd_design.md.
"""

import ast
import contextlib
import sys
from pathlib import Path

import numpy as np
import pytest

from tensorforge.backends import cpp
from tensorforge.experimental import NativeTensor

needs_native = pytest.mark.skipif(
    not cpp.is_available(),
    reason="experimental C++ backend not built",
)

REPO_ROOT = Path(__file__).resolve().parent.parent


# ======================================================================
# 1. The native gradient path does not use NumPy compute
# ======================================================================
#
# The native backward rules run entirely on C++ kernels. NumPy is used
# *only* to marshal small shape/stride arrays across the ctypes boundary
# (np.asarray) and to materialize copies out (np.empty, in to_numpy) — it
# never computes a gradient value. This guard proves that invariant at
# runtime: during a guarded backward() we replace NumPy's *numerical*
# functions (the ones a NumPy-fallback implementation of these backward
# rules would call — add/multiply/matmul/sum/mean/maximum/where/...) with
# tripwires that raise. The marshalling helpers (asarray/empty/...) are
# deliberately left intact, so a correct native pass still completes; only
# a smuggled-in NumPy gradient computation would trip a wire.
#
# The graph is fully built *before* the guard is entered, and gradients are
# inspected with to_numpy() *after* it is left, so neither construction nor
# inspection is affected by the tripwires.

# The NumPy functions a NumPy-backed implementation of the native backward
# rules would reach for. None of them is used by the native gradient path
# (verified: the path calls only np.asarray/np.empty for marshalling).
_GUARDED_NUMPY_FUNCS = (
    "add", "subtract", "multiply", "divide", "true_divide", "negative",
    "matmul", "dot", "vdot", "inner", "outer", "tensordot", "einsum",
    "sum", "mean", "prod", "cumsum",
    "maximum", "minimum", "where", "clip", "sign", "abs", "absolute",
    "broadcast_to", "tile", "repeat", "power", "square",
)


class _NumpyComputeUsed(AssertionError):
    """Raised by the guard when the native gradient path reaches for a
    NumPy numerical function — i.e. a silent NumPy fallback."""


@contextlib.contextmanager
def _numpy_compute_guard():
    """Temporarily replace NumPy's numerical functions with tripwires.

    Anything the native gradient path legitimately uses (np.asarray for
    stride marshalling, np.empty for materialization) is left untouched, so
    a correct native backward passes cleanly; a NumPy gradient computation
    would raise ``_NumpyComputeUsed``. Restores every patched attribute on
    exit, even on error."""
    originals = {}
    for name in _GUARDED_NUMPY_FUNCS:
        originals[name] = getattr(np, name)

        def _tripwire(*args, _name=name, **kwargs):
            raise _NumpyComputeUsed(
                f"native gradient path used NumPy compute: np.{_name}(...)"
            )

        setattr(np, name, _tripwire)
    try:
        yield
    finally:
        for name, original in originals.items():
            setattr(np, name, original)


@needs_native
def test_native_autograd_guardrail_numpy_guard_actually_bites():
    # Prove the guard has teeth: while active, a NumPy compute op raises,
    # while a marshalling call (np.asarray) still works.
    with _numpy_compute_guard():
        with pytest.raises(_NumpyComputeUsed):
            np.multiply(np.array([1.0]), np.array([2.0]))
        with pytest.raises(_NumpyComputeUsed):
            np.matmul(np.ones((2, 2)), np.ones((2, 2)))
        assert np.asarray((2, 3), dtype=np.int64).tolist() == [2, 3]
    # Restored afterwards.
    assert np.multiply(2.0, 3.0) == 6.0


@needs_native
def test_phase_b_guardrail_backward_no_numpy_elementwise():
    x = NativeTensor.from_array([[1.0, -2.0], [3.0, 4.0]], requires_grad=True)
    scale = NativeTensor.from_array([[2.0, 2.0], [2.0, 2.0]], requires_grad=True)
    out = x.multiply(scale).relu().sum()   # graph built before the guard
    with _numpy_compute_guard():
        out.backward()
    # x*scale = [[2,-4],[6,8]]; relu passes the gradient where that is > 0,
    # so the relu mask is [[1,0],[1,1]]. dx = mask*scale, dscale = mask*x.
    assert np.array_equal(x.grad.to_numpy(), [[2.0, 0.0], [2.0, 2.0]])
    assert np.array_equal(scale.grad.to_numpy(), [[1.0, 0.0], [3.0, 4.0]])
    x.close()
    scale.close()


@needs_native
def test_phase_b_guardrail_backward_no_numpy_broadcasting():
    x = NativeTensor.from_array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
                                requires_grad=True)
    bias = NativeTensor.from_array([10.0, 20.0, 30.0], requires_grad=True)  # (3,)
    out = x.add(bias).sum()
    with _numpy_compute_guard():
        out.backward()
    assert bias.grad.shape == (3,)
    assert np.array_equal(bias.grad.to_numpy(), [2.0, 2.0, 2.0])  # summed rows
    assert np.array_equal(x.grad.to_numpy(), np.ones((2, 3)))
    x.close()
    bias.close()


@needs_native
def test_phase_b_guardrail_backward_no_numpy_reduction():
    x = NativeTensor.from_array(np.arange(24.0).reshape(2, 3, 4),
                                requires_grad=True)
    out = x.mean(axis=1).sum()          # 3-D reduction chain
    with _numpy_compute_guard():
        out.backward()
    assert x.grad.shape == (2, 3, 4)
    assert np.allclose(x.grad.to_numpy(), np.full((2, 3, 4), 1.0 / 3.0))
    x.close()


@needs_native
def test_phase_b_guardrail_backward_no_numpy_matmul():
    a = NativeTensor.from_array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
                                requires_grad=True)
    b = NativeTensor.from_array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]],
                                requires_grad=True)
    out = a.matmul(b).sum()
    with _numpy_compute_guard():
        out.backward()
    ones = np.ones((2, 2))
    assert np.allclose(a.grad.to_numpy(), ones @ np.array(
        [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]).T)
    assert np.allclose(b.grad.to_numpy(), np.array(
        [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]).T @ ones)
    a.close()
    b.close()


@needs_native
def test_phase_b_guardrail_backward_no_numpy_view_chain():
    # transpose -> narrow -> contiguous_copy -> reshape -> sum: every view
    # backward rule (inverse permutation, scatter, identity, inverse
    # reshape) exercised under the guard in one chain.
    x = NativeTensor.from_array(np.arange(6.0).reshape(2, 3), requires_grad=True)
    out = x.T.narrow(0, 0, 2).contiguous_copy().reshape((4,)).sum()
    with _numpy_compute_guard():
        out.backward()
    # x.T is (3, 2); keeping its first two rows and summing gives ones in
    # the transposed positions that map back to x's first two columns.
    expected = np.zeros((2, 3))
    expected[:, 0:2] = 1.0
    assert np.array_equal(x.grad.to_numpy(), expected)
    x.close()


# ======================================================================
# 2. NativeTensor / tensorforge.Tensor isolation
# ======================================================================


@needs_native
def test_native_backend_isolation_ops_return_native_tensors():
    a = NativeTensor.from_array([[1.0, -2.0], [3.0, 4.0]], requires_grad=True)
    b = NativeTensor.from_array([[1.0, 1.0], [1.0, 1.0]])
    results = [
        a.relu(), a.add(b), a.subtract(b), a.multiply(b), a.sum(), a.mean(),
        a.matmul(b), a.reshape((4,)), a.transpose(), a.T, a.narrow(1, 0, 1),
        a.contiguous_copy(),
    ]
    for result in results:
        assert isinstance(result, NativeTensor)
        assert not isinstance(result, np.ndarray)
        if result.owns_core:
            result.close()
    a.close()
    b.close()


@needs_native
def test_native_backend_isolation_grad_is_native_backed():
    x = NativeTensor.from_array([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    x.multiply(x).sum().backward()
    assert isinstance(x.grad, NativeTensor)
    assert not isinstance(x.grad, np.ndarray)
    assert x.grad.dtype == "float64" and x.grad.device == "cpu"
    x.close()


def test_native_backend_isolation_tensor_stays_numpy_backed():
    # tensorforge.Tensor keeps its own NumPy-backed autograd; its grad is a
    # NumPy array and never a NativeTensor. No native backend involved.
    from tensorforge import Tensor

    t = Tensor([2.0, 3.0], requires_grad=True)
    (t * t).sum().backward()
    assert isinstance(t.grad, np.ndarray)
    assert not isinstance(t.grad, NativeTensor)
    assert np.allclose(t.grad, [4.0, 6.0])


@needs_native
def test_native_backend_isolation_native_backward_does_not_touch_tensor():
    # Running a native backward must not create or modify any
    # tensorforge.Tensor. A Tensor built beside it is untouched.
    from tensorforge import Tensor

    witness = Tensor([1.0, 2.0], requires_grad=True)
    x = NativeTensor.from_array([1.0, 2.0, 3.0], requires_grad=True)
    x.multiply(x).sum().backward()
    assert witness.grad is None          # native backward never touched it
    assert isinstance(x.grad, NativeTensor)
    x.close()


def test_native_backend_isolation_tensor_backward_creates_no_native():
    # A pure tensorforge.Tensor backward produces only NumPy-backed grads,
    # never a NativeTensor — the two engines never cross.
    from tensorforge import Tensor

    a = Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    b = Tensor([[5.0, 6.0], [7.0, 8.0]], requires_grad=True)
    (a * b).sum().backward()
    for grad in (a.grad, b.grad):
        assert isinstance(grad, np.ndarray)
        assert not isinstance(grad, NativeTensor)


@needs_native
def test_native_backend_isolation_mixed_operands_fail_clearly():
    # Mixed NativeTensor / tensorforge.Tensor operands are unsupported and
    # must fail clearly rather than dispatch implicitly, in both directions.
    from tensorforge import Tensor

    nt = NativeTensor.from_array([1.0, 2.0], requires_grad=True)
    t = Tensor([1.0, 2.0], requires_grad=True)
    for op in ("add", "subtract", "multiply", "matmul"):
        with pytest.raises(TypeError):
            getattr(nt, op)(t)          # NativeTensor rejects a Tensor operand
    # And the reverse: Tensor arithmetic does not silently absorb a
    # NativeTensor (or route to the native backend); it raises.
    with pytest.raises((TypeError, ValueError)):
        _ = t + nt
    nt.close()


# ======================================================================
# 3. Explicit backend behavior — no implicit dispatch or silent fallback
# ======================================================================


def test_native_backend_isolation_reached_only_through_experimental_api():
    # The wrapper lives under tensorforge.experimental, the explicit,
    # opt-in entry point. It is not re-exported from the framework root.
    import tensorforge
    import tensorforge.experimental as experimental

    assert experimental.NativeTensor is NativeTensor
    assert not hasattr(tensorforge, "NativeTensor")


def test_phase_b_guardrail_framework_import_does_not_route_through_native():
    # A static check that tensorforge's __init__ imports neither the
    # experimental wrapper nor the backends package — so importing or using
    # tensorforge.Tensor cannot silently route numerical work through the
    # native backend. (Cannot be fooled by another test importing them.)
    init = REPO_ROOT / "src" / "tensorforge" / "__init__.py"
    tree = ast.parse(init.read_text(encoding="utf-8"))
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)
        elif isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
    assert not any(m.startswith("tensorforge.experimental") for m in imported)
    assert not any(m.startswith("tensorforge.backends") for m in imported)


def test_phase_b_guardrail_no_automatic_backend_selection():
    # Backend selection is always explicit: the registry lists names, and
    # get_backend requires one — nothing chooses a backend on its own.
    from tensorforge.backends import available_backends, get_backend

    assert set(available_backends()) == {"numpy", "native"}
    with pytest.raises(ValueError, match="unknown backend"):
        get_backend("auto")
    with pytest.raises(TypeError):
        get_backend()  # no implicit default


def test_phase_b_guardrail_native_unavailable_fails_explicitly(monkeypatch):
    # When the native runtime is unavailable, a constructor must raise the
    # explicit ImportError rather than silently falling back to NumPy.
    # Simulated so the test runs even where the backend is built.
    def _unbuilt(*args, **kwargs):
        raise ImportError("The experimental C++ backend is not built ...")

    monkeypatch.setattr(cpp.NativeTensorCore, "from_array", staticmethod(_unbuilt))
    with pytest.raises(ImportError, match="not built"):
        NativeTensor.from_array([1.0, 2.0])


# ======================================================================
# 4. Gradient ownership invariants
# ======================================================================


@needs_native
def test_phase_b_guardrail_gradient_ownership_invariants():
    # A branching graph: x feeds two paths that recombine. Leaves retain
    # gradients; non-leaves do not; grads are native/float64/cpu and shaped
    # like their leaf.
    x_vals = np.array([1.0, 2.0, 3.0])
    x = NativeTensor.from_array(x_vals, requires_grad=True)
    a = x.multiply(x)          # non-leaf
    b = x.add(x)               # non-leaf
    out = a.add(b).sum()       # d/dx (x^2 + 2x) = 2x + 2
    out.backward(retain_graph=True)

    # Leaf retains a native, correctly-typed, correctly-shaped grad.
    assert isinstance(x.grad, NativeTensor)
    assert x.grad.shape == x.shape == (3,)
    assert x.grad.dtype == "float64"
    assert x.grad.device == "cpu"
    assert np.allclose(x.grad.to_numpy(), 2.0 * x_vals + 2.0)
    # Non-leaves keep no gradient after the pass.
    assert a.grad is None and b.grad is None and out.grad is None

    # Repeated successful backward accumulates by native addition.
    out.backward(retain_graph=True)
    assert np.allclose(x.grad.to_numpy(), 2.0 * (2.0 * x_vals + 2.0))

    # zero_grad returns the leaf grad to None without altering data or graph.
    out_backward = out._backward
    x.zero_grad()
    assert x.grad is None
    assert np.array_equal(x.to_numpy(), x_vals)     # data untouched
    assert out._backward is out_backward            # graph untouched
    assert out._graph_freed is False
    x.close()


@needs_native
def test_phase_b_guardrail_non_leaf_never_retains_grad():
    x = NativeTensor.from_array([1.0, 2.0], requires_grad=True)
    mid = x.multiply(x)
    out = mid.sum()
    out.backward(retain_graph=True)
    # Only the leaf keeps a gradient; the intermediate and the output do not.
    assert isinstance(x.grad, NativeTensor)
    assert mid.grad is None
    assert out.grad is None
    x.close()


# ======================================================================
# 5. Graph-lifetime invariants across a realistic mixed graph
# ======================================================================


@needs_native
def test_phase_b_guardrail_graph_lifetime_mixed_graph():
    # One consolidated graph containing a shared intermediate, a broadcast
    # op, and a view op. retain_graph=True reuses it; a later default
    # backward accumulates once more and then frees it; a subsequent reuse
    # raises deterministically and leaves the leaf gradients unchanged.
    x_vals = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    bias_vals = np.array([10.0, 20.0, 30.0])
    x = NativeTensor.from_array(x_vals, requires_grad=True)
    bias = NativeTensor.from_array(bias_vals, requires_grad=True)

    shared = x.add(bias)                 # broadcast (3,) over (2, 3); shared
    left = shared.narrow(1, 0, 2).sum()  # view branch
    # Build the whole graph from `shared` so both branches share it.
    out = shared.sum().add(left)

    out.backward(retain_graph=True)
    grad_x_once = x.grad.to_numpy().copy()
    grad_bias_once = bias.grad.to_numpy().copy()
    assert out._graph_freed is False and shared._graph_freed is False

    out.backward()                       # default -> accumulate once more, free
    assert np.allclose(x.grad.to_numpy(), 2.0 * grad_x_once)
    assert np.allclose(bias.grad.to_numpy(), 2.0 * grad_bias_once)
    assert out._graph_freed is True and shared._graph_freed is True
    assert out._parents == () and out._backward is None

    # Reusing the freed graph raises, and changes nothing.
    before_x = x.grad.to_numpy().copy()
    before_bias = bias.grad.to_numpy().copy()
    with pytest.raises(RuntimeError, match="freed"):
        out.backward()
    assert np.array_equal(x.grad.to_numpy(), before_x)
    assert np.array_equal(bias.grad.to_numpy(), before_bias)
    x.close()
    bias.close()


# ======================================================================
# 6. Detached tensor invariants
# ======================================================================


@needs_native
def test_phase_b_guardrail_detach_invariants():
    x_vals = np.arange(6.0).reshape(2, 3)
    x = NativeTensor.from_array(x_vals, requires_grad=True)
    d = x.multiply(x).detach()           # value x^2, detached from the graph

    # Independent, graph-free, non-requiring owning copy.
    assert d.requires_grad is False
    assert d.is_leaf is True
    assert d.owns_core is True
    assert d._parents == ()
    assert d._backward is None
    assert d._graph_freed is False
    assert np.allclose(d.to_numpy(), x_vals ** 2)

    # A backward on the original graph never reaches the detached copy.
    x.multiply(x).sum().backward()
    assert d.grad is None

    # It does not share mutable storage with the source, and stays usable
    # after the source is closed (owning copy, independent lifetime).
    x.close()
    assert d.closed is False
    assert np.allclose(d.to_numpy(), x_vals ** 2)

    # It cannot reconnect to or resurrect the original graph: a fresh op on
    # the detached leaf builds a graph rooted only at the detached leaf.
    y = NativeTensor.from_array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0], requires_grad=True)
    d.reshape((6,)).multiply(y).sum().backward()
    assert d.grad is None                # detached: still no gradient
    assert np.allclose(y.grad.to_numpy(), (x_vals ** 2).ravel())
    d.close()
    y.close()


# ======================================================================
# 7. View and offset invariants (nonzero-offset / non-contiguous parent)
# ======================================================================


@needs_native
def test_phase_b_guardrail_transpose_narrow_offset_backward():
    # A transpose introduces non-contiguity and a narrow introduces a
    # nonzero offset; backward must still produce a grad at the original
    # parent shape, scatter correctly, own contiguous storage, use no NumPy
    # compute, and leave a cleanly freed graph afterward.
    x_vals = np.arange(12.0).reshape(3, 4)
    x = NativeTensor.from_array(x_vals, requires_grad=True)
    # x.T is (4, 3), non-contiguous; narrow to rows 1..2 -> nonzero offset.
    out = x.T.narrow(0, 1, 2).sum()
    with _numpy_compute_guard():
        out.backward()

    # Reference: ones scattered into x.T's rows 1..2, then transposed back.
    grad_xt = np.zeros((4, 3))
    grad_xt[1:3, :] = 1.0
    assert x.grad.shape == x.shape == (3, 4)
    assert np.array_equal(x.grad.to_numpy(), grad_xt.T)

    # The retained leaf grad owns fresh contiguous storage.
    assert x.grad.owns_core is True
    assert x.grad.contiguous is True

    # Graph cleanup still worked (default one-shot backward).
    assert out._graph_freed is True
    with pytest.raises(RuntimeError, match="freed"):
        out.backward()
    x.close()


# ======================================================================
# 8. Closed / invalid native operand safety
# ======================================================================


@needs_native
def test_phase_b_guardrail_closed_operand_does_not_corrupt_graph():
    # Closing an operand before backward must raise, commit no leaf
    # gradient, free no graph, and leave the graph state intact and reusable
    # once the failure condition is resolved (here: nothing to resolve, but
    # the invariant is that the failed pass changed nothing).
    a = NativeTensor.from_array([1.0, 2.0], requires_grad=True)
    b = NativeTensor.from_array([3.0, 4.0], requires_grad=True)
    add = a.add(b)
    out = add.sum()
    b.close()                            # b is needed by add's backward
    with pytest.raises(RuntimeError, match="closed"):
        out.backward()
    # No partial commit, no partial free, no corrupted graph.
    assert a.grad is None
    assert out._graph_freed is False and add._graph_freed is False
    assert out._parents == (add,) and add._parents == (a, b)
    assert callable(out._backward) and callable(add._backward)
    a.close()


@needs_native
def test_phase_b_guardrail_closed_intermediate_before_backward_raises():
    a = NativeTensor.from_array([1.0, 2.0], requires_grad=True)
    mid = a.multiply(a)
    out = mid.sum()
    mid.close()
    with pytest.raises(RuntimeError, match="closed"):
        out.backward()
    assert a.grad is None
    assert out._graph_freed is False
    a.close()


# ======================================================================
# 9. Kernel registry boundary
# ======================================================================


def test_native_backend_isolation_registry_excludes_internal_backward_kernels():
    # list_kernels() and backend_info()['tensor_core_kernels'] are the
    # public raw-buffer / TensorCore-method registries. The internal fused
    # backward kernels (surfaced only as forward-shaped numerical methods,
    # by design) must not leak into either — nor may backend/TensorCore
    # object names. This is the intentional stable contract; incidental
    # ordering is deliberately not frozen.
    kernels = set(cpp.list_kernels())
    tensor_core_kernels = set(cpp.backend_info()["tensor_core_kernels"])

    internal_backward = {
        "tf_core_relu_backward", "relu_backward",
        "tf_core_narrow_backward", "narrow_backward",
        "tf_core_sum", "tf_storage_scale",
    }
    for name in internal_backward:
        assert name not in kernels, f"{name} leaked into list_kernels()"
        assert name not in tensor_core_kernels, (
            f"{name} leaked into tensor_core_kernels"
        )

    # And the registry still describes raw buffer kernels, not objects.
    assert "elementwise_add" in kernels
    for not_a_kernel in ("NativeTensor", "NativeTensorCore", "backward"):
        assert not_a_kernel not in kernels


def test_native_backend_isolation_backend_info_advertises_no_integration():
    # Native autograd stays experimental and unintegrated with the STABLE
    # framework: backend_info() advertises no wiring into tensorforge.Tensor
    # (the deliberate architectural separation). It does, accurately, report
    # that the native line has its own autograd and optimizers.
    info = cpp.backend_info()
    assert info["stable_framework_integration"] is False
    assert info["native_autograd"] is True


# ======================================================================
# 10. Benchmark contract guardrails
# ======================================================================
#
# The v2.5 benchmark tests already cover schema, selection, the correctness
# gate, and the no-speed-verdict rule. These add the mode-*meaning*
# protections the completion milestone locks down, using the benchmark's own
# case builders (no timing anywhere).

sys.path.insert(0, str(REPO_ROOT))
from benchmarks.benchmark_native_autograd import CASES, MODES  # noqa: E402


@needs_native
def test_phase_b_guardrail_benchmark_forward_native_builds_no_graph():
    # forward_native uses non-requiring inputs, so the forward result builds
    # no autograd graph — the mode's defining property.
    for name, spec in CASES.items():
        cfg = spec["shapes"]["smoke"]
        inp = spec["make_inputs"](cfg, requires_grad=False)
        assert inp["leaves"] == []
        out = spec["forward"](inp)
        assert out.requires_grad is False
        assert out.is_leaf is True
        assert out._parents == ()
        for operand in inp["operands"]:
            operand.close()
        if out.owns_core:
            out.close()


@needs_native
def test_phase_b_guardrail_benchmark_grad_modes_build_graph():
    # The grad-tracking modes use requiring inputs, so the forward result is
    # a non-leaf graph node with a backward closure.
    for name, spec in CASES.items():
        cfg = spec["shapes"]["smoke"]
        inp = spec["make_inputs"](cfg, requires_grad=True)
        assert inp["leaves"], f"{name}: requiring mode produced no leaves"
        out = spec["forward"](inp)
        assert out.requires_grad is True
        assert out.is_leaf is False
        assert callable(out._backward)
        assert out.shape == spec["out_shape"]  # scalar loss
        for operand in inp["operands"]:
            operand.close()


@needs_native
def test_phase_b_guardrail_benchmark_fresh_mode_frees_graph():
    # forward_backward_fresh builds and frees a new graph each iteration:
    # one default backward() leaves the graph one-shot freed.
    spec = CASES["elementwise"]
    inp = spec["make_inputs"](spec["shapes"]["smoke"], requires_grad=True)
    out = spec["forward"](inp)
    out.backward()
    assert out._graph_freed is True
    for leaf in inp["leaves"]:
        assert leaf.grad is not None
    with pytest.raises(RuntimeError, match="freed"):
        out.backward()
    for operand in inp["operands"]:
        operand.close()


@needs_native
def test_phase_b_guardrail_benchmark_retained_mode_reuses_one_graph():
    # backward_retained builds the graph once and re-runs backward over it
    # (retain_graph=True), so the graph is never freed and gradients
    # accumulate across passes.
    spec = CASES["matmul"]
    inp = spec["make_inputs"](spec["shapes"]["smoke"], requires_grad=True)
    retained = spec["forward"](inp)
    retained.backward(retain_graph=True)
    first = [leaf.grad.to_numpy().copy() for leaf in inp["leaves"]]
    retained.backward(retain_graph=True)
    for leaf, once in zip(inp["leaves"], first):
        assert np.allclose(leaf.grad.to_numpy(), 2.0 * once)
    assert retained._graph_freed is False
    for operand in inp["operands"]:
        operand.close()


def test_phase_b_guardrail_benchmark_modes_are_the_documented_four():
    # The four modes keep their documented identity and order; forward_native
    # is first (the graph-free baseline).
    assert MODES == (
        "forward_native",
        "forward_graph",
        "forward_backward_fresh",
        "backward_retained",
    )
