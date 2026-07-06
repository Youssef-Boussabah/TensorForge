# Examples

Each example is a standalone script in `examples/`. They all follow
the same pattern: a `train()` function that returns a stats dictionary
(so the tests can import and verify it) and a `main()` that prints
progress. Every one uses a fixed seed, so the output below is what you
should actually see.

## Linear regression

```
uv run python examples/train_linear_regression.py
```

The "hello world": fit `Linear(1, 1)` to the line `y = 2x + 1` with
SGD and MSE loss. Teaches the bare training loop — forward, loss,
zero_grad, backward, step. Ends with weight ≈ 2.0 and bias ≈ 1.0.

## XOR

```
uv run python examples/train_xor.py
```

Why hidden layers exist: no single linear layer can separate XOR, and
a small `Linear → Tanh → Linear → Sigmoid` network can. Final
predictions land near 0, 1, 1, 0 for the four inputs.

## Multi-class spiral

```
uv run python examples/train_multiclass.py
```

A real classifier: a two-hidden-layer MLP on a 3-class spiral, trained
with `cross_entropy` on integer class labels. Also demonstrates
mini-batch training (`train(batch_size=16)`) and an optional
validation split (`train(validation_split=0.25)`). Reaches 100%
training accuracy at the default settings.

## Binary classification

```
uv run python examples/train_binary_classification.py
```

Logistic regression from TensorForge parts: `Linear(2, 1)` producing
one raw logit, `binary_cross_entropy` (no Sigmoid layer needed — the
loss works on logits directly, which is what keeps it numerically
stable), a train/validation split, and `binary_accuracy`. Reaches
about 97% train / 93% validation accuracy on two overlapping point
clouds.

## Tiny CNN

```
uv run python examples/train_tiny_cnn.py
```

Why convolutions exist: a `Conv2d -> ReLU -> Flatten -> Linear` model
classifies synthetic 6x6 images containing a vertical or horizontal
bar at a *random position*. One small kernel slid across the image
detects a bar wherever it appears — which a Linear layer on raw pixels
can't do without learning every position separately. Reaches 100%
accuracy in about 50 epochs.

## MLP with Dropout

```
uv run python examples/train_mlp_with_dropout.py
```

The train/eval distinction in action: a deeper MLP with two Dropout
layers on concentric circles. Optimization steps run in train mode
(dropout active), every metric is measured through
`evaluate_binary_classifier` in eval mode (dropout off). Reaches 100%
on both splits. Its `train()` also accepts `max_grad_norm=...` for
gradient clipping and `scheduler_step_size=...` /
`scheduler_gamma=...` for StepLR learning-rate decay.
