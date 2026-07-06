"""Save and load model parameters.

Parameters travel as plain NumPy ``.npz`` archives keyed by the model's
state-dict names — no pickle, no code, just arrays. Loading requires a
model with the same architecture; only the values move.
"""

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
