import numpy as np
import pytest

from tensorforge import (
    Tensor,
    load_checkpoint,
    load_parameters,
    save_checkpoint,
    save_parameters,
)
from tensorforge.nn import Linear, Parameter, mse_loss
from tensorforge.optim import SGD, Adam, StepLR


def _train_steps(model, optimizer, steps):
    """A tiny fixed regression problem: y = 2x + 1."""
    x = Tensor([[0.0], [1.0], [2.0]])
    y = Tensor([[1.0], [3.0], [5.0]])
    for _ in range(steps):
        optimizer.zero_grad()
        mse_loss(model(x), y).backward()
        optimizer.step()


# ---------------------------------------------------------------------------
# Optimizer state_dict / load_state_dict
# ---------------------------------------------------------------------------


def test_sgd_state_roundtrip():
    p = Parameter([1.0])
    opt = SGD([p], lr=0.5)
    state = opt.state_dict()
    assert state == {"lr": 0.5}

    other = SGD([p], lr=0.001)
    assert other.load_state_dict(state) is None
    assert other.lr == 0.5


def test_adam_state_dict_contents():
    p = Parameter(np.array([1.0, 2.0]))
    opt = Adam([p], lr=0.01, betas=(0.8, 0.99), eps=1e-7)
    p.grad = np.array([0.1, -0.1])
    opt.step()

    state = opt.state_dict()
    assert set(state) == {"lr", "beta1", "beta2", "eps", "t", "m", "v"}
    assert state["lr"] == 0.01
    assert state["beta1"] == 0.8
    assert state["beta2"] == 0.99
    assert state["eps"] == 1e-7
    assert state["t"] == 1
    assert len(state["m"]) == 1 and len(state["v"]) == 1
    # The returned arrays are copies of the live moment estimates.
    state["m"][0] += 100.0
    assert not np.allclose(opt.m[0], state["m"][0])


def test_adam_load_state_dict_restores_t_and_moments():
    p = Parameter(np.array([1.0, 2.0]))
    source = Adam([p], lr=0.01)
    p.grad = np.array([0.1, -0.1])
    source.step()
    state = source.state_dict()

    target = Adam([p], lr=0.9)
    assert target.load_state_dict(state) is None
    assert target.t == 1
    assert target.lr == 0.01
    assert np.array_equal(target.m[0], source.m[0])
    assert np.array_equal(target.v[0], source.v[0])
    # Loaded arrays are copies, not the checkpoint's own objects.
    state["m"][0] += 100.0
    assert not np.allclose(target.m[0], state["m"][0])


def test_adam_load_state_dict_rejects_wrong_length():
    p = Parameter(np.array([1.0, 2.0]))
    opt = Adam([p], lr=0.01)
    state = opt.state_dict()
    state["m"] = []
    with pytest.raises(ValueError, match="moment arrays"):
        opt.load_state_dict(state)


def test_adam_load_state_dict_rejects_wrong_shape():
    p = Parameter(np.array([1.0, 2.0]))
    opt = Adam([p], lr=0.01)
    state = opt.state_dict()
    state["v"] = [np.zeros((3, 3))]
    with pytest.raises(ValueError, match="shape mismatch"):
        opt.load_state_dict(state)


def test_loading_optimizer_state_keeps_parameter_objects():
    p = Parameter(np.array([1.0]))
    opt = Adam([p], lr=0.01)
    before = list(opt.parameters)
    opt.load_state_dict(opt.state_dict())
    assert all(a is b for a, b in zip(before, opt.parameters))
    assert opt.parameters[0] is p


# ---------------------------------------------------------------------------
# save_checkpoint / load_checkpoint
# ---------------------------------------------------------------------------


def test_checkpoint_restores_model_predictions(tmp_path):
    np.random.seed(0)
    model_a = Linear(2, 1)
    x = Tensor([[1.0, -2.0], [0.5, 0.25]])
    path = tmp_path / "ckpt.npz"
    save_checkpoint(path, model_a)

    np.random.seed(99)
    model_b = Linear(2, 1)
    report = load_checkpoint(path, model_b)
    assert np.allclose(model_a(x).data, model_b(x).data)
    assert report["optimizer_loaded"] is False
    assert report["metadata"] == {}
    assert report["model"] == {"missing_keys": [], "unexpected_keys": []}


def test_metadata_roundtrip(tmp_path):
    np.random.seed(0)
    model = Linear(1, 1)
    metadata = {"epoch": 12, "loss": 0.25, "tags": ["baseline", "v1"]}
    path = tmp_path / "ckpt.npz"
    save_checkpoint(path, model, metadata=metadata)
    report = load_checkpoint(path, model)
    assert report["metadata"] == metadata


def test_checkpoint_is_pickle_free(tmp_path):
    np.random.seed(0)
    model = Linear(2, 1)
    opt = Adam(model.parameters(), lr=0.01)
    path = tmp_path / "ckpt.npz"
    save_checkpoint(path, model, opt, metadata={"epoch": 1})

    with np.load(path, allow_pickle=False) as archive:
        assert any(name.startswith("model::") for name in archive.files)
        assert "optimizer::meta" in archive.files
        assert "checkpoint::meta" in archive.files


def test_optimizer_arg_but_no_optimizer_state_raises(tmp_path):
    np.random.seed(0)
    model = Linear(1, 1)
    path = tmp_path / "ckpt.npz"
    save_checkpoint(path, model)  # no optimizer saved
    with pytest.raises(ValueError, match="no optimizer state"):
        load_checkpoint(path, model, optimizer=SGD(model.parameters(), lr=0.1))


def test_checkpoint_with_optimizer_loads_model_only_without_optimizer(tmp_path):
    np.random.seed(0)
    model = Linear(1, 1)
    opt = Adam(model.parameters(), lr=0.01)
    _train_steps(model, opt, steps=3)
    path = tmp_path / "ckpt.npz"
    save_checkpoint(path, model, opt)

    np.random.seed(5)
    fresh = Linear(1, 1)
    report = load_checkpoint(path, fresh)  # no optimizer passed
    assert report["optimizer_loaded"] is False
    assert np.array_equal(fresh.weight.data, model.weight.data)


def test_optimizer_class_mismatch_raises(tmp_path):
    np.random.seed(0)
    model = Linear(1, 1)
    path = tmp_path / "ckpt.npz"
    save_checkpoint(path, model, SGD(model.parameters(), lr=0.1))
    with pytest.raises(ValueError, match="SGD"):
        load_checkpoint(path, model, optimizer=Adam(model.parameters(), lr=0.1))


def test_adam_resume_matches_uninterrupted_training(tmp_path):
    """The point of checkpoints: save mid-training, resume elsewhere,
    and land exactly where uninterrupted training would have."""
    np.random.seed(0)
    model_a = Linear(1, 1)
    opt_a = Adam(model_a.parameters(), lr=0.05)
    _train_steps(model_a, opt_a, steps=5)

    path = tmp_path / "ckpt.npz"
    save_checkpoint(path, model_a, opt_a, metadata={"step": 5})
    _train_steps(model_a, opt_a, steps=7)  # A trains straight through

    np.random.seed(123)  # B starts from unrelated random weights
    model_b = Linear(1, 1)
    opt_b = Adam(model_b.parameters(), lr=0.9)  # wrong lr, must be restored
    report = load_checkpoint(path, model_b, opt_b)
    assert report["optimizer_loaded"] is True
    assert report["metadata"] == {"step": 5}
    assert opt_b.t == opt_a.t - 7  # 5 steps at save time

    _train_steps(model_b, opt_b, steps=7)  # B resumes the same 7 steps

    assert np.allclose(model_a.weight.data, model_b.weight.data, atol=1e-12)
    assert np.allclose(model_a.bias.data, model_b.bias.data, atol=1e-12)


def test_strict_false_passes_through(tmp_path):
    np.random.seed(0)
    model = Linear(2, 1)  # has weight + bias
    path = tmp_path / "ckpt.npz"
    save_checkpoint(path, model)

    target = Linear(2, 1, bias=False)  # checkpoint's bias is unexpected
    with pytest.raises(ValueError):
        load_checkpoint(path, target, strict=True)
    report = load_checkpoint(path, target, strict=False)
    assert report["model"]["unexpected_keys"] == ["bias"]
    assert np.array_equal(target.weight.data, model.weight.data)


# ---------------------------------------------------------------------------
# Scheduler checkpointing
# ---------------------------------------------------------------------------


def _train_steps_with_scheduler(model, optimizer, scheduler, steps):
    x = Tensor([[0.0], [1.0], [2.0]])
    y = Tensor([[1.0], [3.0], [5.0]])
    for _ in range(steps):
        optimizer.zero_grad()
        mse_loss(model(x), y).backward()
        optimizer.step()
        scheduler.step()


def test_save_scheduler_without_optimizer_raises(tmp_path):
    np.random.seed(0)
    model = Linear(1, 1)
    scheduler = StepLR(SGD(model.parameters(), lr=0.1), step_size=2)
    with pytest.raises(ValueError, match="requires the optimizer"):
        save_checkpoint(tmp_path / "ckpt.npz", model, scheduler=scheduler)


def test_load_scheduler_without_optimizer_raises(tmp_path):
    np.random.seed(0)
    model = Linear(1, 1)
    opt = SGD(model.parameters(), lr=0.1)
    scheduler = StepLR(opt, step_size=2)
    path = tmp_path / "ckpt.npz"
    save_checkpoint(path, model, opt, scheduler=scheduler)
    with pytest.raises(ValueError, match="requires the optimizer"):
        load_checkpoint(path, model, scheduler=scheduler)


def test_checkpoint_without_scheduler_reports_false(tmp_path):
    np.random.seed(0)
    model = Linear(1, 1)
    opt = SGD(model.parameters(), lr=0.1)
    path = tmp_path / "ckpt.npz"
    save_checkpoint(path, model, opt)
    report = load_checkpoint(path, model, opt)
    assert report["scheduler_loaded"] is False
    assert report["optimizer_loaded"] is True


def test_scheduler_checkpoint_loads_without_scheduler(tmp_path):
    np.random.seed(0)
    model = Linear(1, 1)
    opt = SGD(model.parameters(), lr=0.1)
    scheduler = StepLR(opt, step_size=2)
    scheduler.step()
    path = tmp_path / "ckpt.npz"
    save_checkpoint(path, model, opt, metadata={"epoch": 1}, scheduler=scheduler)

    report = load_checkpoint(path, model, opt)  # no scheduler passed
    assert report["scheduler_loaded"] is False
    assert report["optimizer_loaded"] is True
    assert report["metadata"] == {"epoch": 1}


def test_loading_scheduler_from_schedulerless_checkpoint_raises(tmp_path):
    np.random.seed(0)
    model = Linear(1, 1)
    opt = SGD(model.parameters(), lr=0.1)
    path = tmp_path / "ckpt.npz"
    save_checkpoint(path, model, opt)  # no scheduler saved
    scheduler = StepLR(opt, step_size=2)
    with pytest.raises(ValueError, match="no scheduler state"):
        load_checkpoint(path, model, opt, scheduler=scheduler)


def test_scheduler_class_mismatch_raises(tmp_path):
    class OtherLR(StepLR):
        pass

    np.random.seed(0)
    model = Linear(1, 1)
    opt = SGD(model.parameters(), lr=0.1)
    path = tmp_path / "ckpt.npz"
    save_checkpoint(path, model, opt, scheduler=StepLR(opt, step_size=2))
    with pytest.raises(ValueError, match="StepLR"):
        load_checkpoint(path, model, opt, scheduler=OtherLR(opt, step_size=2))


def test_scheduler_state_roundtrip_and_identity(tmp_path):
    np.random.seed(0)
    model = Linear(1, 1)
    opt = Adam(model.parameters(), lr=0.05)
    scheduler = StepLR(opt, step_size=3, gamma=0.5)
    _train_steps_with_scheduler(model, opt, scheduler, steps=4)  # decayed once
    path = tmp_path / "ckpt.npz"
    save_checkpoint(path, model, opt, scheduler=scheduler)

    np.random.seed(7)
    model_b = Linear(1, 1)
    opt_b = Adam(model_b.parameters(), lr=0.9)
    scheduler_b = StepLR(opt_b, step_size=99, gamma=0.9)  # wrong config
    report = load_checkpoint(path, model_b, opt_b, scheduler=scheduler_b)

    assert report["scheduler_loaded"] is True
    assert scheduler_b.last_epoch == 4
    assert scheduler_b.step_size == 3
    assert scheduler_b.gamma == 0.5
    assert scheduler_b.optimizer is opt_b  # neither object was replaced
    assert np.isclose(opt_b.lr, opt.lr)


def test_scheduler_checkpoint_is_pickle_free(tmp_path):
    np.random.seed(0)
    model = Linear(1, 1)
    opt = SGD(model.parameters(), lr=0.1)
    path = tmp_path / "ckpt.npz"
    save_checkpoint(path, model, opt, scheduler=StepLR(opt, step_size=2))
    with np.load(path, allow_pickle=False) as archive:
        assert "scheduler::meta" in archive.files


def test_adam_steplr_resume_matches_uninterrupted_training(tmp_path):
    """The full resume story: model + optimizer + scheduler. This fails
    if scheduler state is not restored, because B's scheduler starts
    with a schedule that would never decay."""
    np.random.seed(0)
    model_a = Linear(1, 1)
    opt_a = Adam(model_a.parameters(), lr=0.05)
    sched_a = StepLR(opt_a, step_size=3, gamma=0.5)
    _train_steps_with_scheduler(model_a, opt_a, sched_a, steps=5)

    path = tmp_path / "ckpt.npz"
    save_checkpoint(path, model_a, opt_a, metadata={"epoch": 5}, scheduler=sched_a)
    _train_steps_with_scheduler(model_a, opt_a, sched_a, steps=7)  # A runs on

    np.random.seed(123)  # B starts from unrelated weights and settings
    model_b = Linear(1, 1)
    opt_b = Adam(model_b.parameters(), lr=0.9)
    sched_b = StepLR(opt_b, step_size=1000, gamma=0.1)  # would never decay
    report = load_checkpoint(path, model_b, opt_b, scheduler=sched_b)
    assert report["optimizer_loaded"] and report["scheduler_loaded"]

    _train_steps_with_scheduler(model_b, opt_b, sched_b, steps=7)  # B resumes

    assert np.allclose(model_a.weight.data, model_b.weight.data, atol=1e-12)
    assert np.allclose(model_a.bias.data, model_b.bias.data, atol=1e-12)
    assert np.isclose(opt_a.lr, opt_b.lr)
    assert sched_a.last_epoch == sched_b.last_epoch == 12


def test_save_load_parameters_still_work(tmp_path):
    np.random.seed(0)
    model_a = Linear(2, 1)
    path = tmp_path / "params.npz"
    save_parameters(model_a, path)
    np.random.seed(7)
    model_b = Linear(2, 1)
    load_parameters(model_b, path)
    assert np.array_equal(model_a.weight.data, model_b.weight.data)
    assert np.array_equal(model_a.bias.data, model_b.bias.data)