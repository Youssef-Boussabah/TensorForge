# How the autograd engine works

The whole engine lives in `src/tensorforge/tensor.py` and is a few
hundred lines. This page is the guided tour.

## What a Tensor stores

```python
x = Tensor(4.0, requires_grad=True)
```

A Tensor holds:

- `data` — the actual values, as a NumPy float64 array
- `grad` — the gradient, filled in by `backward()` (starts as `None`)
- `requires_grad` — whether this tensor wants gradients at all
- `_prev` — the tensors that produced this one (its "parents")
- `_backward` — a closure that knows how to push gradient back to
  those parents

The last two are the computation graph. You never set them yourself —
operations do.

## Operations build the graph

Every operation computes its result immediately with NumPy, then
attaches the bookkeeping:

```python
y = x * x    # y.data is 16.0, computed right away
```

`y` remembers that it came from `(x, x)` and carries a `_backward`
closure containing the local derivative of multiplication: "each input
receives the other input times my gradient". That's all an operation
needs to know — its own local rule, nothing about the rest of the
graph.

Some operations don't even need their own rule. Subtraction is defined
as `a + (-b)` and division as `a * b**-1`, so they inherit correct
gradients from the operations they're built from. `softmax` inside
`Tensor`, and the BatchNorm1d and LayerNorm normalizations, work the
same way: compositions of existing ops, gradients for free.

## What backward() does

```python
y.backward()
print(x.grad)  # 8.0
```

`backward()` does three things:

1. **Topologically sorts** the graph, so every tensor comes after all
   the tensors that were computed from it.
2. **Seeds** the output's gradient with ones (dy/dy = 1).
3. Walks the sorted list **in reverse**, calling each tensor's
   `_backward` to push gradient one step further toward the inputs.

The topological order matters because a tensor may feed several later
operations. It must wait until *all* of them have contributed gradient
before passing its total on — visiting in reverse topological order
guarantees exactly that.

## Gradients accumulate

If a tensor is used twice, both uses contribute gradient, and the
contributions **add**:

```python
y = x * x + x   # x appears three times
y.backward()    # x.grad = 2x + 1 — the sum of all contributions
```

This is why optimizers call `zero_grad()` before each backward pass:
without it, gradients from the previous step would still be there and
the new ones would pile on top.

One more wrinkle: NumPy broadcasting. When a `(3,)` bias is added to a
`(32, 3)` matrix, the bias participates 32 times, so its gradient must
be summed back down to shape `(3,)`. A helper (`_unbroadcast`) handles
this whenever gradients flow into a broadcast input.

## Fused losses

`cross_entropy` and `binary_cross_entropy` are *not* compositions —
they are single fused operations with hand-written backward passes.
The reason is numerical stability: computing `log(softmax(x))` naively
underflows to `log(0)` when a probability saturates, while the fused
log-sum-exp form stays finite for any input. Their backward passes use
the classic closed forms (`(softmax − one_hot) / batch` and
`(sigmoid − target) / n`), which are also cheaper than backpropagating
through the composition.

All gradients — composed and fused — are verified against central
finite differences in `tests/test_gradcheck.py`.
