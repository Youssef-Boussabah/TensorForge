"""TensorForge — a mini deep learning framework built from scratch."""

from tensorforge.data import batches, train_test_split
from tensorforge.nn.losses import binary_cross_entropy, cross_entropy
from tensorforge.nn.metrics import (
    accuracy,
    binary_accuracy,
    evaluate_binary_classifier,
    evaluate_classifier,
)
from tensorforge.nn.batchnorm import BatchNorm1d
from tensorforge.nn.conv import Conv2d
from tensorforge.nn.dropout import Dropout
from tensorforge.nn.flatten import Flatten
from tensorforge.nn.module import count_parameters, model_summary
from tensorforge.nn.parameter import Parameter
from tensorforge.nn.pool import MaxPool2d
from tensorforge.optim import SGD, Adam, StepLR, clip_grad_norm, clip_grad_value
from tensorforge.serialization import (
    load_checkpoint,
    load_parameters,
    save_checkpoint,
    save_parameters,
)
from tensorforge.tensor import Tensor

__all__ = [
    "Tensor",
    "Parameter",
    "Dropout",
    "BatchNorm1d",
    "Conv2d",
    "MaxPool2d",
    "Flatten",
    "cross_entropy",
    "binary_cross_entropy",
    "accuracy",
    "binary_accuracy",
    "SGD",
    "Adam",
    "StepLR",
    "clip_grad_norm",
    "clip_grad_value",
    "batches",
    "train_test_split",
    "evaluate_classifier",
    "evaluate_binary_classifier",
    "save_parameters",
    "load_parameters",
    "save_checkpoint",
    "load_checkpoint",
    "count_parameters",
    "model_summary",
]
