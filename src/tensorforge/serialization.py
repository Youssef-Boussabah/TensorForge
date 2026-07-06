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


def save_checkpoint(path, model, optimizer=None, metadata=None, scheduler=None, rng_state=None):
    """Save a training checkpoint: model weights, optional optimizer
    state, optional scheduler state, optional RNG state, and optional
    JSON metadata.

    ``rng_state=True`` captures NumPy's current global RNG state
    (``np.random.get_state()``); an explicit state tuple is also
    accepted. Saving it lets a resumed run replay the exact same
    randomness — dropout masks, shuffles — as the uninterrupted run.

    Archive layout (all pickle-free):
      model::<param_name>   parameter arrays
      checkpoint::meta      metadata as a JSON string
      optimizer::meta       optimizer class + scalar state as JSON
      optimizer::m::<i>     Adam's moment arrays, one entry each
      optimizer::v::<i>
      scheduler::meta       scheduler class + state as JSON
      rng::meta             RNG scalars as JSON
      rng::keys             the RNG key array
    """
    if scheduler is not None and optimizer is None:
        raise ValueError(
            "saving scheduler state requires the optimizer too — a "
            "scheduler is meaningless without the optimizer it drives"
        )
    arrays = {f"model::{name}": value for name, value in model.state_dict().items()}
    arrays["checkpoint::meta"] = json.dumps(metadata if metadata is not None else {})

    if rng_state is not None and rng_state is not False:
        state = np.random.get_state() if rng_state is True else rng_state
        # The legacy global state is a 5-tuple:
        # (bit_generator_name, keys_array, pos, has_gauss, cached_gaussian)
        try:
            name, keys, pos, has_gauss, cached = state
            arrays["rng::keys"] = np.asarray(keys, dtype=np.uint32)
            arrays["rng::meta"] = json.dumps(
                {
                    "bit_generator": str(name),
                    "pos": int(pos),
                    "has_gauss": int(has_gauss),
                    "cached_gaussian": float(cached),
                }
            )
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"rng_state must be True or a NumPy np.random.get_state() "
                f"tuple, got {rng_state!r}"
            ) from error

    if scheduler is not None:
        arrays["scheduler::meta"] = json.dumps(
            {"class": type(scheduler).__name__, "state": scheduler.state_dict()}
        )

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


def load_checkpoint(path, model, optimizer=None, strict=True, scheduler=None, restore_rng_state=False):
    """Load a checkpoint saved by ``save_checkpoint``.

    Model weights always load (``strict`` is passed through to
    ``model.load_state_dict``). Optimizer state loads only when an
    ``optimizer`` is passed, and scheduler state only when a
    ``scheduler`` is passed (which requires the optimizer too, loaded
    first). Each must be the same class the checkpoint was saved with,
    and passing one when the checkpoint holds no matching state is an
    error. Scheduler state in a checkpoint is ignored when no scheduler
    is passed.

    ``restore_rng_state=True`` additionally restores NumPy's global RNG
    from the checkpoint (error if the checkpoint has none); by default
    the RNG is left untouched. Returns::

        {"model": <load_state_dict report>,
         "optimizer_loaded": bool,
         "scheduler_loaded": bool,
         "rng_loaded": bool,
         "metadata": dict}
    """
    if scheduler is not None and optimizer is None:
        raise ValueError(
            "loading scheduler state requires the optimizer too — a "
            "scheduler is meaningless without the optimizer it drives"
        )
    with np.load(Path(path), allow_pickle=False) as archive:
        files = set(archive.files)
        model_state = {
            name[len("model::"):]: archive[name]
            for name in files
            if name.startswith("model::")
        }
        metadata = json.loads(archive["checkpoint::meta"].item())
        rng_meta = None
        rng_keys = None
        if "rng::meta" in files:
            rng_meta = json.loads(archive["rng::meta"].item())
            rng_keys = archive["rng::keys"]
        scheduler_meta = None
        if "scheduler::meta" in files:
            scheduler_meta = json.loads(archive["scheduler::meta"].item())
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

    # Scheduler state loads after the optimizer, so a restored lr is
    # not overwritten by a stale scheduler-driven value (and vice versa
    # the scheduler's epoch counter lines up with the restored lr).
    scheduler_loaded = False
    if scheduler is not None:
        if scheduler_meta is None:
            raise ValueError(
                "a scheduler was passed, but this checkpoint contains "
                "no scheduler state"
            )
        expected = scheduler_meta["class"]
        actual = type(scheduler).__name__
        if expected != actual:
            raise ValueError(
                f"checkpoint was saved with scheduler {expected!r}, "
                f"but got {actual!r}"
            )
        scheduler.load_state_dict(scheduler_meta["state"])
        scheduler_loaded = True

    rng_loaded = False
    if restore_rng_state:
        if rng_meta is None:
            raise ValueError(
                "restore_rng_state=True, but this checkpoint contains "
                "no RNG state"
            )
        np.random.set_state(
            (
                rng_meta["bit_generator"],
                rng_keys,
                rng_meta["pos"],
                rng_meta["has_gauss"],
                rng_meta["cached_gaussian"],
            )
        )
        rng_loaded = True

    return {
        "model": model_report,
        "optimizer_loaded": optimizer_loaded,
        "scheduler_loaded": scheduler_loaded,
        "rng_loaded": rng_loaded,
        "metadata": metadata,
    }
