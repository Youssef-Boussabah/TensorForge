"""TensorForge — a mini deep learning framework built from scratch."""

from tensorforge.nn.losses import cross_entropy
from tensorforge.nn.metrics import accuracy
from tensorforge.tensor import Tensor

__all__ = ["Tensor", "cross_entropy", "accuracy"]
