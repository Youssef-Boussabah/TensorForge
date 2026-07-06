"""Locks in the intended public API: these imports must keep working."""


def test_root_package_exports():
    from tensorforge import (  # noqa: F401
        Adam,
        BatchNorm1d,
        Conv2d,
        Dropout,
        Flatten,
        MaxPool2d,
        Parameter,
        SGD,
        Tensor,
        accuracy,
        batches,
        binary_accuracy,
        binary_cross_entropy,
        count_parameters,
        cross_entropy,
        evaluate_binary_classifier,
        evaluate_classifier,
        load_checkpoint,
        load_parameters,
        model_summary,
        save_checkpoint,
        save_parameters,
        StepLR,
        clip_grad_norm,
        clip_grad_value,
        train_test_split,
    )


def test_nn_exports():
    from tensorforge.nn import (  # noqa: F401
        BatchNorm1d,
        Conv2d,
        Dropout,
        Flatten,
        Linear,
        MaxPool2d,
        Module,
        Parameter,
        ReLU,
        Sequential,
        Sigmoid,
        Tanh,
        accuracy,
        binary_accuracy,
        binary_cross_entropy,
        cross_entropy,
        evaluate_binary_classifier,
        evaluate_classifier,
        mse_loss,
    )


def test_optim_exports():
    from tensorforge.optim import (  # noqa: F401
        SGD,
        Adam,
        StepLR,
        clip_grad_norm,
        clip_grad_value,
    )


def test_data_exports():
    from tensorforge.data import batches, train_test_split  # noqa: F401


def test_root_names_are_the_same_objects_as_submodule_names():
    """The root exports are conveniences, not copies."""
    import tensorforge
    import tensorforge.data
    import tensorforge.nn
    import tensorforge.optim

    assert tensorforge.SGD is tensorforge.optim.SGD
    assert tensorforge.Adam is tensorforge.optim.Adam
    assert tensorforge.batches is tensorforge.data.batches
    assert tensorforge.cross_entropy is tensorforge.nn.cross_entropy
    assert tensorforge.accuracy is tensorforge.nn.accuracy
    assert tensorforge.Parameter is tensorforge.nn.Parameter
    assert tensorforge.Conv2d is tensorforge.nn.Conv2d
    assert tensorforge.Flatten is tensorforge.nn.Flatten
    assert tensorforge.MaxPool2d is tensorforge.nn.MaxPool2d
