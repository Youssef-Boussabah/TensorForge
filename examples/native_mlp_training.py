"""The first complete native CPU training proof (Advanced C++ v3.9).

A small multilayer perceptron — ``NativeSequential(NativeLinear(2, 8,
seed=0), NativeReLU(), NativeLinear(8, 1, seed=1))`` — trains on a
fixed synthetic regression dataset for 25 deterministic steps of
``NativeSGD(lr=0.1)``, entirely through the experimental native stack:
native forward, native ``NativeMSELoss``, native ``backward()``,
graph-free native optimizer updates through the v3.7 ``copy_value_``
contract, and ``zero_grad()`` between iterations. Every iteration
builds a completely fresh graph (no ``retain_graph``, no graph reuse,
so the v3.7 stale-value guard never triggers), every parameter's value
version advances by exactly one per step, parameter identities stay
stable throughout, and the loss decreases monotonically — from about
2.1079 to about 0.0095.

The input and target are **fixed Python literals** handed once to
``NativeTensor.from_array`` — that is data construction at the explicit
entry boundary, not native computation; every piece of training math
(forward, activation, loss, gradients, updates) runs on the native
kernels. NumPy appears only to read scalars and snapshots back out
through ``to_numpy()``, the established inspection exit. The stable
framework's tensor and optimizers are involved nowhere.

Lifetime is explicit: the model parameters, the optimizer, and the
fixed data tensors live for the whole run; the per-iteration prediction
and loss tensors are closed every iteration after their one-shot
``backward()`` has released the operation graph; and everything the run
created is closed on the way out, success or failure. ``train()``
returns plain Python values only — never live native tensors.

This is a training proof for the experimental native training stack —
not a benchmark, and no performance is claimed. It needs the
experimental C++ backend to be built — run:

    uv run python examples/native_mlp_training.py

``train()`` returns its results as a dict of Python scalars/lists so
the tests can import and verify it; ``main()`` prints them.
"""

from tensorforge.backends import cpp
from tensorforge.experimental import (
    NativeLinear,
    NativeMSELoss,
    NativeReLU,
    NativeSequential,
    NativeSGD,
    NativeTensor,
)

# Fixed synthetic regression data: 8 samples, 2 input features, 1
# target feature — values chosen once from y = 0.5*x0 - x1 + 0.25 and
# frozen as literals. Nothing is generated, loaded, or preprocessed.
X_VALUES = [
    [0.0, 0.5],
    [1.0, -1.0],
    [-0.5, 1.5],
    [2.0, 0.0],
    [-1.5, -0.5],
    [0.5, 2.0],
    [1.5, 1.0],
    [-2.0, 1.0],
]
Y_VALUES = [
    [-0.25],
    [1.75],
    [-1.5],
    [1.25],
    [0.0],
    [-1.5],
    [0.0],
    [-1.75],
]

IN_FEATURES = 2
HIDDEN_FEATURES = 8
OUT_FEATURES = 1
HIDDEN_SEED = 0
OUTPUT_SEED = 1
DEFAULT_STEPS = 25
DEFAULT_LR = 0.1


def _build_model():
    """The deterministic MLP: 2 → 8 → ReLU → 1, seeded initialization."""
    return NativeSequential(
        NativeLinear(IN_FEATURES, HIDDEN_FEATURES, seed=HIDDEN_SEED),
        NativeReLU(),
        NativeLinear(HIDDEN_FEATURES, OUT_FEATURES, seed=OUTPUT_SEED),
    )


def _forward_backward(model, loss_fn, x, y):
    """One fresh forward → loss → record → backward, then close the
    per-iteration tensors this call created. The one-shot backward
    releases the operation graph (and with it the last references to
    the graph-internal intermediates); the prediction and loss wrappers
    are the only graph tensors returned to Python, and they are closed
    here. Returns the recorded scalar loss as a Python float."""
    prediction = model(x)
    loss = loss_fn(prediction, y)
    try:
        value = float(loss.to_numpy())  # inspection exit, before release
        loss.backward()
    finally:
        loss.close()
        prediction.close()
    return value


def train(steps=DEFAULT_STEPS, lr=DEFAULT_LR):
    """Train the deterministic native MLP for ``steps`` SGD updates and
    return the run's evidence as plain Python values.

    ``steps`` must be a real positive int (bool rejected); ``lr`` is
    validated by NativeSGD itself. The returned dict contains:

    - ``"steps"``, ``"lr"`` — the configuration actually used;
    - ``"initial_loss"``, ``"final_loss"`` — the loss before the first
      update and after the last one (a fresh evaluation pass);
    - ``"loss_history"`` — the pre-update loss of every iteration
      (``steps`` floats; ``final_loss`` extends it by one entry);
    - ``"parameter_names"`` — the canonical ``named_parameters()``
      names, recorded before training and verified unchanged after;
    - ``"initial_versions"``, ``"final_versions"``,
      ``"version_history"`` — per-parameter value versions, snapshotted
      after every step;
    - ``"initial_parameters"``, ``"final_parameters"`` — name → nested
      lists of the parameter values (independent Python copies);
    - ``"identity_stable"`` — True iff every parameter was the same
      object at every checkpoint (checked with in-process identity,
      never serialized);
    - ``"names_stable"`` — True iff the canonical names and order never
      changed;
    - ``"gradient_lifecycle_ok"`` — True iff, on every iteration,
      gradients were absent before forward, present on every parameter
      after backward, still present after ``step()``, and cleared
      after ``zero_grad()``;
    - ``"gradients_cleared"`` — True iff no parameter holds a gradient
      at return.

    Everything the run creates — the fixed data tensors and the model's
    parameters included — is closed before returning, success or
    failure; the caller receives Python values only."""
    if isinstance(steps, bool) or not isinstance(steps, int):
        raise TypeError(f"steps must be an int, got {type(steps).__name__}")
    if steps <= 0:
        raise ValueError(f"steps must be strictly positive, got {steps}")

    model = _build_model()
    loss_fn = NativeMSELoss()
    optimizer = NativeSGD(model.parameters(), lr=lr)  # validates lr

    named = list(model.named_parameters())
    names = [name for name, _ in named]
    parameters = [parameter for _, parameter in named]
    identities = [id(parameter) for parameter in parameters]

    x = NativeTensor.from_array(X_VALUES)
    y = NativeTensor.from_array(Y_VALUES)
    try:
        results = {
            "steps": steps,
            "lr": optimizer.lr,
            "parameter_names": names,
            "initial_versions": [p.version for p in parameters],
            "initial_parameters": {
                name: parameter.to_numpy().tolist()
                for name, parameter in named
            },
        }

        loss_history = []
        version_history = []
        identity_stable = True
        names_stable = True
        gradient_lifecycle_ok = True
        for _ in range(steps):
            # 1. The previous iteration's gradients are cleared.
            gradient_lifecycle_ok &= all(p.grad is None for p in parameters)
            # 2-6. Fresh graph, forward, scalar loss, record, backward.
            loss_history.append(_forward_backward(model, loss_fn, x, y))
            # 7. Every trainable parameter received a gradient.
            gradient_lifecycle_ok &= all(p.grad is not None for p in parameters)
            # 8. Commit the SGD update through the v3.7 mutation path.
            optimizer.step()
            # 9. Identities stable, versions advanced, gradients kept.
            identity_stable &= (
                [id(p) for _, p in model.named_parameters()] == identities
            )
            names_stable &= (
                [name for name, _ in model.named_parameters()] == names
            )
            gradient_lifecycle_ok &= all(p.grad is not None for p in parameters)
            version_history.append([p.version for p in parameters])
            # 10-11. Clear gradients for the next fresh iteration.
            optimizer.zero_grad()
            gradient_lifecycle_ok &= all(p.grad is None for p in parameters)

        # Final evaluation: the same fresh-graph lifecycle without an
        # optimizer step — its backward releases the evaluation graph,
        # and zero_grad clears the gradients it produced. Versions are
        # untouched: forward/backward/zero_grad never count.
        final_loss = _forward_backward(model, loss_fn, x, y)
        optimizer.zero_grad()

        results.update(
            loss_history=loss_history,
            initial_loss=loss_history[0],
            final_loss=final_loss,
            version_history=version_history,
            final_versions=[p.version for p in parameters],
            final_parameters={
                name: parameter.to_numpy().tolist()
                for name, parameter in named
            },
            identity_stable=identity_stable,
            names_stable=names_stable,
            gradient_lifecycle_ok=gradient_lifecycle_ok,
            gradients_cleared=all(p.grad is None for p in parameters),
        )
        return results
    finally:
        # Explicit release of everything this run created (the demo
        # convention): the fixed data and the model's parameters. close()
        # is idempotent, and nothing closed is ever returned.
        x.close()
        y.close()
        for parameter in parameters:
            parameter.close()


def main():
    if not cpp.is_available():
        print("The experimental C++ backend is not built.")
        print(cpp.build_instructions())
        return

    r = train()
    reduction = 100.0 * (1.0 - r["final_loss"] / r["initial_loss"])
    print("Native MLP training proof (Advanced C++ v3.9)")
    print("=" * 50)
    print(
        f"model: NativeLinear({IN_FEATURES}, {HIDDEN_FEATURES}, "
        f"seed={HIDDEN_SEED}) -> NativeReLU() -> "
        f"NativeLinear({HIDDEN_FEATURES}, {OUT_FEATURES}, seed={OUTPUT_SEED})"
    )
    print(f"data: {len(X_VALUES)} fixed samples, "
          f"{IN_FEATURES} features -> {OUT_FEATURES} target")
    print(f"optimizer: NativeSGD(lr={r['lr']}), {r['steps']} steps")
    print(f"initial loss: {r['initial_loss']:.6f}")
    for step in (5, 10, 15, 20):
        print(f"loss after step {step:2d}: {r['loss_history'][step]:.6f}")
    print(f"final loss: {r['final_loss']:.6f} ({reduction:.1f}% reduction)")
    print(f"parameter versions: {dict(zip(r['parameter_names'], r['final_versions']))}")
    print(f"gradients cleared: {r['gradients_cleared']}")
    print("native MLP training ok")


if __name__ == "__main__":
    main()
