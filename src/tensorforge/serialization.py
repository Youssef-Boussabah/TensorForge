"""Save and load model parameters and training checkpoints.

Everything travels as plain NumPy ``.npz`` archives — no pickle, no
code, just arrays and JSON strings. Loading requires a model with the
same architecture; only the values move.

Two levels of saving:

- ``save_parameters`` / ``load_parameters``: model weights only.
- ``save_checkpoint`` / ``load_checkpoint``: model weights plus
  (optionally) optimizer state and JSON metadata, so training can
  *resume* exactly — for Adam that means the step count and moment
  estimates, not just the weights.
"""

import json
from pathlib import Path

import numpy as np


def save_parameters(model, path):
    """Save ``model.state_dict()`` to a ``.npz`` file at ``path``."""
    np.savez(Path(path), **model.state_dict())


def load_parameters(model, path, strict=True):
    """Load a ``.npz`` file saved by ``save_parameters`` into ``model``.

    Returns the same report dict as ``model.load_state_dict``.
    """
    with np.load(Path(path)) as archive:
        state_dict = {name: archive[name] for name in archive.files}
    return model.load_state_dict(state_dict, strict=strict)


def save_checkpoint(path, model, optimizer=None, metadata=None):
    """Save a training checkpoint: model weights, optional optimizer
    state, and optional JSON-serializable metadata.

    Archive layout (all pickle-free):
      model::<param_name>   parameter arrays
      checkpoint::meta      metadata as a JSON string
      optimizer::meta       optimizer class + scalar state as JSON
      optimizer::m::<i>     Adam's moment arrays, one entry each
      optimizer::v::<i>
    """
    arrays = {f"model::{name}": value for name, value in model.state_dict().items()}
    arrays["checkpoint::meta"] = json.dumps(metadata if metadata is not None else {})

    if optimizer is not None:
        state = optimizer.state_dict()
        # Arrays (Adam's moment estimates) become their own npz entries;
        # everything left is plain scalars, stored as one JSON string.
        for i, m in enumerate(state.pop("m", [])):
            arrays[f"optimizer::m::{i}"] = m
        for i, v in enumerate(state.pop("v", [])):
            arrays[f"optimizer::v::{i}"] = v
        arrays["optimizer::meta"] = json.dumps(
            {"class": type(optimizer).__name__, "state": state}
        )

    np.savez(Path(path), **arrays)


def load_checkpoint(path, model, optimizer=None, strict=True):
    """Load a checkpoint saved by ``save_checkpoint``.

    Model weights always load (``strict`` is passed through to
    ``model.load_state_dict``). Optimizer state loads only when an
    ``optimizer`` is passed; it must be the same class the checkpoint
    was saved with. Passing an optimizer when the checkpoint has no
    optimizer state is an error. Returns::

        {"model": <load_state_dict report>,
         "optimizer_loaded": bool,
         "metadata": dict}
    """
    with np.load(Path(path), allow_pickle=False) as archive:
        files = set(archive.files)
        model_state = {
            name[len("model::"):]: archive[name]
            for name in files
            if name.startswith("model::")
        }
        metadata = json.loads(archive["checkpoint::meta"].item())
        optimizer_meta = None
        moment_arrays = {"m": {}, "v": {}}
        if "optimizer::meta" in files:
            optimizer_meta = json.loads(archive["optimizer::meta"].item())
            for name in files:
                for label in ("m", "v"):
                    prefix = f"optimizer::{label}::"
                    if name.startswith(prefix):
                        moment_arrays[label][int(name[len(prefix):])] = archive[name]

    model_report = model.load_state_dict(model_state, strict=strict)

    optimizer_loaded = False
    if optimizer is not None:
        if optimizer_meta is None:
            raise ValueError(
                "an optimizer was passed, but this checkpoint contains "
                "no optimizer state"
            )
        expected = optimizer_meta["class"]
        actual = type(optimizer).__name__
        if expected != actual:
            raise ValueError(
                f"checkpoint was saved with optimizer {expected!r}, "
                f"but got {actual!r}"
            )
        state = dict(optimizer_meta["state"])
        for label in ("m", "v"):
            if moment_arrays[label]:
                state[label] = [
                    moment_arrays[label][i]
                    for i in sorted(moment_arrays[label])
                ]
        optimizer.load_state_dict(state)
        optimizer_loaded = True

    return {
        "model": model_report,
        "optimizer_loaded": optimizer_loaded,
        "metadata": metadata,
    }
