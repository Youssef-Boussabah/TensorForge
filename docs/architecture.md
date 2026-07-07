# Architecture

TensorForge is a from-scratch deep learning framework written in
Python and NumPy. It reimplements how a framework like PyTorch works
under the hood, with every piece kept deliberately small and readable.
The core framework has no dependency besides NumPy and no GPU code;
an experimental native C++ backend lives in its own namespace (see
[backend_experiments.md](backend_experiments.md)).

## Package layout

```
src/tensorforge/
  tensor.py          Tensor + the autograd engine
  data.py            batches() and train_test_split()
  serialization.py   save/load parameters, save/load checkpoints
  nn/                everything model-related
    parameter.py     Parameter (a trainable Tensor)
    module.py        Module base class: parameters, buffers,
                     train/eval mode, state_dict, summary
    linear.py        Linear layer
    activations.py   ReLU, Sigmoid, Tanh
    dropout.py       Dropout
    batchnorm.py     BatchNorm1d
    layernorm.py     LayerNorm
    conv.py          Conv2d (NCHW image-shaped inputs)
    pool.py          MaxPool2d
    flatten.py       Flatten
    sequential.py    Sequential container
    losses.py        mse_loss, cross_entropy, binary_cross_entropy
    metrics.py       accuracy, binary_accuracy, evaluators
  optim/             optimizers and training control
    sgd.py           SGD
    adam.py          Adam
    clip.py          clip_grad_norm, clip_grad_value
    lr_scheduler.py  StepLR
examples/            runnable training scripts
tests/               pytest suite (one test file per feature area)
```

## How the pieces fit together

**Tensor** (`tensor.py`) is the foundation. It wraps a NumPy array and
records enough information to run backpropagation. Everything else in
the framework is built on top of Tensor operations — see
[autograd.md](autograd.md) for how that works.

**Parameter** is a Tensor subclass that always has
`requires_grad=True`. Being a distinct class is what lets the rest of
the framework find trainable state by type.

**Module** is the base class for layers and models. It discovers
Parameters by walking its own attributes (including child modules and
lists of modules), which gives every model `parameters()`,
`state_dict()`, `summary()`, and train/eval mode for free — no
registration calls needed. Modules can also declare non-trainable
*buffers* (like BatchNorm's running statistics) that travel with
`state_dict()` but are never optimized.

**Layers** (Linear, activations, Dropout, BatchNorm1d, LayerNorm,
Conv2d, Flatten) are small Module subclasses. The two normalizations
differ in what they average over: BatchNorm1d normalizes each feature
*across the batch* (so it keeps running statistics and changes behavior
in eval mode), while LayerNorm normalizes each sample over *its own*
trailing dimensions (no buffers, identical in train and eval mode). Each one implements `forward()`
using Tensor operations, so gradients flow through them automatically
(Conv2d and MaxPool2d are the exceptions — fused ops with explicit
backwards, because composing them from elementwise ops would be
unreadable). Conv2d and MaxPool2d work on image-shaped `(batch,
channels, height, width)` input; MaxPool2d has no Parameters at all and
its backward routes each window's gradient only to the position that
won the max. Flatten bridges image-shaped activations back to the
`(batch, features)` shape Linear expects. Sequential chains layers so
the output of one feeds the next.

**Losses** are either plain Tensor expressions (`mse_loss`) or fused
operations with a hand-written backward pass where numerical stability
demands it (`cross_entropy`, `binary_cross_entropy`).

**Optimizers** (SGD, Adam) are plain classes, not Modules. They hold a
list of Parameters and update `param.data` from `param.grad` in
`step()`. They skip parameters with no gradient and frozen parameters
(`requires_grad=False`). Gradient clipping and the StepLR scheduler
sit alongside them in `optim/`.

**Examples** tie it all together. Each one is a standalone script with
a `train()` function (so tests can import and run it) and a `main()`
that prints progress. See [examples.md](examples.md).

## Design habits

A few conventions hold across the codebase:

- Forward computation is eager NumPy; only the gradient bookkeeping is
  deferred.
- Comments explain math and autograd reasoning, not obvious Python.
- Randomness is always seedable: helpers take a `seed` argument and
  use their own `np.random.default_rng`. The one deliberate exception
  is unseeded Dropout, which draws from NumPy's global RNG so that
  checkpoint RNG save/restore can make dropout training resume
  bit-for-bit.
- Every feature ships with tests, and training tests assert learning
  (loss decreased, accuracy above a threshold) rather than exact
  floating-point values.
