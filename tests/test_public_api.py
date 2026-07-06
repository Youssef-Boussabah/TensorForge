"""Locks in the intended public API: these imports must keep working."""


def test_root_package_exports():
    from tensorforge import (  # noqa: F401
        Adam,
        Parameter,
        SGD,
        Tensor,
        accuracy,
        batches,
        count_parameters,
        cross_entropy,
        evaluate_classifier,
        load_parameters,
        model_summary,
        save_parameters,
        train_test_split,
    )


def test_nn_exports():
    from tensorforge.nn import (  # noqa: F401
        Linear,
        Module,
        Parameter,
        ReLU,
        Sequential,
        Sigmoid,
        Tanh,
        accuracy,
        cross_entropy,
        evaluate_classifier,
        mse_loss,
    )


def test_optim_exports():
    from tensorforge.optim import SGD, Adam  # noqa: F401


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
