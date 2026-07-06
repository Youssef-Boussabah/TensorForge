# Training with TensorForge

## The basic loop

Every training loop in the examples follows the same shape:

```python
from tensorforge import Tensor, train_test_split
from tensorforge.nn import Linear, ReLU, Sequential, cross_entropy
from tensorforge.optim import Adam, StepLR, clip_grad_norm

x_train, x_val, y_train, y_val = train_test_split(X, y, test_size=0.25, seed=0)

model = Sequential(Linear(2, 16), ReLU(), Linear(16, 3))
optimizer = Adam(model.parameters(), lr=0.01)
scheduler = StepLR(optimizer, step_size=100, gamma=0.5)   # optional

for epoch in range(epochs):
    logits = model(Tensor(x_train))              # 1. forward pass
    loss = cross_entropy(logits, y_train)        # 2. loss

    optimizer.zero_grad()                        # 3. clear old gradients
    loss.backward()                              # 4. backprop
    clip_grad_norm(model.parameters(), 5.0)      # 5. optional clipping
    optimizer.step()                             # 6. update parameters
    scheduler.step()                             # 7. optional lr decay
```

The order matters in two places: `zero_grad()` must come before
`backward()` (gradients accumulate — see
[autograd.md](autograd.md)), and clipping must sit between
`backward()` and `step()`, because it edits the gradients the step is
about to use.

For datasets too big to process at once, `batches()` slices the data
into mini-batches and you run steps 1–6 once per batch instead of once
per epoch. `examples/train_multiclass.py` shows both variants.

## Train mode vs eval mode

Some layers behave differently while training:

- **Dropout** randomly zeroes activations in training mode (that noise
  is the regularization) and is the identity in eval mode.
- **BatchNorm1d** normalizes with the current batch's statistics in
  training mode and updates its running averages; in eval mode it uses
  the stored running averages and updates nothing.

Not every layer cares: **LayerNorm** normalizes each sample over its
own features, uses no batch statistics, and behaves identically in
both modes.

`model.train()` and `model.eval()` switch the whole model, recursing
through all children. The rule of thumb: **optimize in train mode,
measure in eval mode.** A loss measured with dropout active jumps
around randomly and understates the model's real performance.

## Evaluating

`evaluate_classifier(model, X, y)` and
`evaluate_binary_classifier(model, X, y)` return
`{"loss": ..., "accuracy": ...}` in one call. They handle the mode
problem for you: the model is temporarily switched to eval mode for
the measurement and restored afterwards, so you can call them
mid-training without thinking about it. They never touch gradients or
parameters.

The usual setup is to hold out a validation split with
`train_test_split`, train on one part, and track both splits' metrics
per epoch — `examples/train_mlp_with_dropout.py` is the template.
A validation accuracy well below training accuracy is the classic sign
of overfitting.

## Saving: parameters vs checkpoints

Two levels, both plain NumPy `.npz` files (no pickle):

- `save_parameters(model, path)` / `load_parameters(model, path)`
  save **weights only** (including buffers like BatchNorm running
  stats). Use this to reuse a trained model.
- `save_checkpoint(path, model, optimizer, metadata=..., scheduler=...)` /
  `load_checkpoint(path, model, optimizer, scheduler=...)` additionally
  save the **optimizer's state** — for Adam that means the step count
  and moment estimates — the **scheduler's state** if you pass one
  (StepLR's epoch counter and decay settings), plus any JSON metadata.
  Use this to *resume training*: a run resumed from a checkpoint
  continues exactly as if it had never stopped, which weights alone
  cannot guarantee. Without the scheduler state, a resumed run would
  restart the learning-rate schedule from the wrong place.

Loading requires a model built with the same architecture; only the
values move.

One more optional piece: randomness. If training uses unseeded Dropout
(or anything else drawing from NumPy's global RNG), the sequence of
random masks is part of the training trajectory. Pass
``rng_state=True`` when saving and ``restore_rng_state=True`` when
loading, and the resumed run replays the exact same masks the
uninterrupted run would have seen — making the resume bit-for-bit.
Both flags are off by default: without them checkpoints behave exactly
as before, and a resumed dropout run is statistically equivalent but
not identical.
