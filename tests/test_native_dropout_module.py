"""NativeDropout — the native Dropout module (Phase G, milestone G4).

G4 is the public layer over the completed G3 operation and adds no
numerical surface of its own: the module is argument validation, one
registered ``NativeGenerator``, a train/eval dispatch, and a delegation
to ``NativeTensor.dropout``. These tests cover the constructor and its
mutually-exclusive ``seed``/``generator`` rule, generator ownership and
registration as Phase G's fourth state category — read from **identity
and the registered topology**, never from a stored ownership flag, of
which the module deliberately has none — the three forward cases
(training stochastic, evaluation identity, ``p == 0`` identity), the
call-consumption contract across mode switches, shared versus independent
streams, ``NativeSequential`` composition, construction and forward
failure atomicity, the module generator's **checkpoint-v2 persistence**
(the gap G4 left open and G5 closed), and the capability boundary G4
deliberately does *not* move — ``"dropout"`` stays unsupported until G10.

Backend-dependent, so the module skips cleanly when the compiled backend
is not built. Cleanup is explicit via close().

Selector: python -m pytest -q -k native_dropout_module
"""

import gc

import numpy as np
import pytest

from tensorforge.backends import cpp
from tensorforge.experimental import (
    NativeDropout, NativeGenerator, NativeLinear, NativeModule,
    NativeParameter, NativeReLU, NativeSequential, NativeTensor,
)
from tensorforge.experimental import native_tensor as native_tensor_module

pytestmark = pytest.mark.skipif(
    not cpp.is_available(),
    reason="experimental C++ backend not built; " + cpp.build_instructions(),
)

# The same committed G2 known-answer vector the Core and operation suites
# assert, restated rather than imported: a known answer one suite can
# redefine for another is not a known answer.
VECTOR_SEED = 0x0123456789ABCDEF
VECTOR_P = 0.25
VECTOR_KEEP = "011011111010"          # 12 logical elements, row-major
VECTOR_SCALE = 1.0 / (1.0 - VECTOR_P)

VALUES = np.arange(1.0, 13.0)


@pytest.fixture
def live_storages(monkeypatch):
    """The ids of every NativeStorage currently open — a real
    live-native-allocation count, so an ownership test can prove the count
    returns exactly to its baseline instead of trusting collection."""
    open_ids = set()
    original_init = cpp.NativeStorage.__init__
    original_close = cpp.NativeStorage.close

    def tracked_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        open_ids.add(id(self))

    def tracked_close(self):
        original_close(self)
        open_ids.discard(id(self))

    monkeypatch.setattr(cpp.NativeStorage, "__init__", tracked_init)
    monkeypatch.setattr(cpp.NativeStorage, "close", tracked_close)
    return open_ids


def committed_mask():
    values = [VECTOR_SCALE if keep == "1" else 0.0 for keep in VECTOR_KEEP]
    return np.array(values, dtype=np.float64)


def core_reference(values, p, seed, call_index):
    """The G2 Core's ``(output, mask)``, as NumPy arrays, everything
    native closed again. The module's oracle: G4 adds a wrapper and must
    change the numbers by exactly nothing."""
    source = cpp.NativeTensorCore.from_array(np.asarray(values, dtype=float))
    try:
        out, mask = source._dropout_forward_with_mask(
            p, seed=seed, call_index=call_index
        )
        try:
            return out.to_numpy().copy(), mask.to_numpy().copy()
        finally:
            out.close()
            mask.close()
    finally:
        source.close()


def ones_like(tensor):
    return NativeTensor.from_array(np.ones(tensor.shape, dtype=np.float64))


# ==========================================================================
# 1. Constructor
# ==========================================================================

def test_default_construction():
    module = NativeDropout()
    assert module.p == 0.5
    assert module.training is True
    assert isinstance(module.generator, NativeGenerator)
    assert module.generator.calls == 0
    assert module.generator.algorithm == "tensorforge.splitmix64"


def test_the_signature_is_the_locked_one():
    import inspect

    signature = inspect.signature(NativeDropout.__init__)
    assert list(signature.parameters) == ["self", "p", "seed", "generator"]
    assert signature.parameters["p"].default == 0.5
    assert signature.parameters["seed"].default is None
    assert signature.parameters["generator"].default is None


@pytest.mark.parametrize("p", [0.0, 0, 0.25, 0.5, 0.999, np.float64(0.3)])
def test_valid_probabilities_are_accepted_and_normalized(p):
    module = NativeDropout(p, seed=1)
    assert type(module.p) is float
    assert module.p == float(p)


@pytest.mark.parametrize("p", [1.0, 1, 1.5, -0.001, float("nan"),
                               float("inf"), float("-inf")])
def test_out_of_range_probabilities_are_rejected(p):
    with pytest.raises(ValueError):
        NativeDropout(p, seed=1)


@pytest.mark.parametrize("p", [True, False, np.bool_(True)])
def test_a_bool_probability_is_rejected(p):
    with pytest.raises(TypeError, match="bool"):
        NativeDropout(p, seed=1)


@pytest.mark.parametrize("p", [None, "0.5", [0.5], complex(0.5, 0)])
def test_a_non_real_probability_is_rejected(p):
    with pytest.raises(TypeError):
        NativeDropout(p, seed=1)


def test_the_module_uses_the_shared_probability_validator():
    """Not a third rule: the module, the operation, and the Core accept
    and reject exactly the same values, checked by running all three over
    the same set."""
    x = NativeTensor.from_array(VALUES)
    core = cpp.NativeTensorCore.from_array(VALUES)
    candidates = [0.0, 0, 0.5, np.float64(0.25), 1.0, 1, -0.5, 2.0,
                  float("nan"), float("inf"), True, None, "0.5"]
    for candidate in candidates:
        outcomes = []
        for attempt in (
            lambda: NativeDropout(candidate, seed=1),
            lambda: x.dropout(candidate, generator=NativeGenerator(1)),
            lambda: core.dropout_forward(candidate, seed=1, call_index=0),
        ):
            try:
                produced = attempt()
            except BaseException as error:
                outcomes.append(type(error).__name__)
            else:
                outcomes.append("accepted")
                if isinstance(produced, cpp.NativeTensorCore):
                    produced.close()
                elif isinstance(produced, NativeTensor) and produced is not x:
                    produced.close()
        assert len(set(outcomes)) == 1, (candidate, outcomes)
    core.close()
    x.close()


def test_an_explicit_seed_builds_a_generator_with_that_seed():
    """A created generator is an ordinary registered generator: the
    construction path is visible in what exists — a fresh object carrying
    the requested seed, registered under the canonical name — and nowhere
    else."""
    module = NativeDropout(0.5, seed=12345)
    assert isinstance(module.generator, NativeGenerator)
    assert module.generator.seed == 12345
    assert module.generator.calls == 0
    assert [name for name, _ in module.named_generators()] == ["generator"]
    assert module.generators() == [module.generator]


@pytest.mark.parametrize(
    "seed, error",
    [(True, TypeError), (1.0, TypeError), ("7", TypeError),
     (np.int64(7), TypeError), (-1, ValueError), (2 ** 64, ValueError)],
)
def test_an_invalid_seed_is_rejected(seed, error):
    with pytest.raises(error):
        NativeDropout(0.5, seed=seed)


def test_seed_none_draws_entropy_once_and_two_modules_differ():
    first = NativeDropout(0.5)
    second = NativeDropout(0.5)
    assert 0 <= first.generator.seed <= 2 ** 64 - 1
    assert 0 <= second.generator.seed <= 2 ** 64 - 1
    assert first.generator is not second.generator
    # Two 64-bit OS draws colliding is not a realistic outcome.
    assert first.generator.seed != second.generator.seed


def test_an_explicit_generator_is_registered_as_the_exact_object():
    generator = NativeGenerator(99)
    before = generator.state()
    module = NativeDropout(0.5, generator=generator)
    assert module.generator is generator, "the generator was copied"
    assert module.generators() == [generator]
    # Adopting changes nothing about it.
    assert generator.state() == before


def test_seed_and_generator_are_mutually_exclusive():
    generator = NativeGenerator(7)
    with pytest.raises(TypeError, match="not both"):
        NativeDropout(0.5, seed=1, generator=generator)
    # Neither argument was silently applied: the generator is untouched
    # and no module escaped.
    assert generator.seed == 7 and generator.calls == 0


@pytest.mark.parametrize(
    "bad", [0, 1.5, "generator", object(), np.random.default_rng(0),
            NativeTensor.from_array(np.zeros(2))],
)
def test_a_non_generator_is_rejected(bad):
    with pytest.raises(TypeError, match="NativeGenerator"):
        NativeDropout(0.5, generator=bad)


def test_repr_reports_stable_configuration_only():
    module = NativeDropout(0.25, seed=3)
    text = repr(module)
    assert "NativeDropout" in text
    assert "0.25" in text
    before = text
    # A forward moves the generator's counter; the repr must not move
    # with it, or it would be useless for identifying a layer.
    x = NativeTensor.from_array(VALUES)
    y = module(x)
    assert repr(module) == before
    # ...and it never leaks the transaction machinery, the seed's live
    # state, or any ownership bookkeeping.
    for leaked in ("token", "reservation", "claim", "lock", "mask",
                   "calls", "_outcome", "owns", "own", "shared"):
        assert leaked not in text, leaked
    y.close()
    x.close()


def test_repr_is_identical_for_an_owned_and_a_shared_generator():
    """The repr is a function of stable configuration only, so the two
    construction paths — which are not durable facts — cannot be read out
    of it."""
    shared = NativeGenerator(3)
    created = NativeDropout(0.25, seed=3)
    adopted = NativeDropout(0.25, generator=shared)
    assert repr(created) == repr(adopted) == "NativeDropout(p=0.25)"
    # ...and sharing the created module's generator afterwards changes
    # neither repr.
    third = NativeDropout(0.25, generator=created.generator)
    assert repr(created) == repr(third) == "NativeDropout(p=0.25)"


# ==========================================================================
# 1b. Ownership is identity and registration — never a stored flag
# ==========================================================================

def test_there_is_no_public_ownership_flag():
    """`owns_generator` was removed deliberately.

    It was never a durable truth: a module that creates its own generator
    can share it a line later, and a public mutable Boolean would then
    assert an exclusivity the object graph contradicts — or be rewritten
    by a caller without changing any registration, lifetime, or behavior.
    The authoritative state is generator identity and the registered
    topology."""
    created = NativeDropout(0.5, seed=1)
    adopted = NativeDropout(0.5, generator=NativeGenerator(2))
    for module in (created, adopted):
        assert not hasattr(module, "owns_generator")
        assert "owns_generator" not in vars(module)
        assert "owns_generator" not in dir(module)
        # Not smuggled back in as a property or descriptor either.
        assert not hasattr(type(module), "owns_generator")


def test_no_public_ownership_attribute_of_any_name_exists():
    """The public surface is `p`, `generator`, `training`, and ordinary
    NativeModule methods — checked against a real NativeModule baseline so
    the assertion is about what *this* class adds, not a hand-copied
    list."""
    module = NativeDropout(0.5, seed=5)
    baseline = set(dir(NativeReLU()))
    added = {name for name in dir(module) if not name.startswith("_")}
    assert added - baseline - {"p", "generator"} == set()
    # And no private ownership bookkeeping was reintroduced under a
    # different spelling.
    for name in vars(module):
        assert "own" not in name.lower(), name


def test_a_created_generator_is_registered_exactly_like_a_supplied_one():
    """Both construction paths produce the *same* registered state: one
    generator, one canonical name, one state-dict entry. Nothing about
    the module distinguishes them, which is the point."""
    created = NativeDropout(0.5, seed=7)
    adopted = NativeDropout(0.5, generator=NativeGenerator(7))

    assert vars(created).keys() == vars(adopted).keys()
    for module in (created, adopted):
        assert [name for name, _ in module.named_generators()] == ["generator"]
        assert module.generators() == [module.generator]
        assert module._generators == {"generator": module.generator}
        assert module.state_dict() == {}
    assert created.generator_state_dict() == adopted.generator_state_dict()
    # ...and they behave identically, from equal starting state.
    x = NativeTensor.from_array(VALUES)
    a, b = created(x), adopted(x)
    assert np.array_equal(a.to_numpy(), b.to_numpy())
    for tensor in (a, b, x):
        tensor.close()


def test_sharing_a_created_generator_leaves_no_stale_ownership_claim():
    """The exact scenario the removed flag got wrong: a module creates a
    generator and *then* shares it. Ownership is no longer exclusive, and
    because nothing recorded the construction path there is no stale claim
    to contradict the object graph."""
    creator = NativeDropout(0.5, seed=263)
    shared = creator.generator
    borrower = NativeDropout(0.5, generator=shared)

    # Identity — the authoritative record — says exactly what is true.
    assert borrower.generator is shared is creator.generator
    assert not hasattr(creator, "owns_generator")
    assert not hasattr(borrower, "owns_generator")

    class Model(NativeModule):
        def __init__(self):
            super().__init__()
            self.creator = creator
            self.borrower = borrower

        def forward(self, x):
            return self.borrower(self.creator(x))

    # The registered topology deduplicates by identity: one object, one
    # entry — which is what a checkpoint alias section would persist.
    model = Model()
    assert model.generators() == [shared]
    assert [name for name, _ in model.named_generators()] == [
        "creator.generator"
    ]

    # ...and the shared stream really is one ordered stream.
    x = NativeTensor.from_array(VALUES)
    y = model(x)
    assert shared.calls == 2
    y.close()
    x.close()


def test_reassigning_the_generator_keeps_identity_authoritative():
    """A module's generator can be replaced by assignment (ordinary fourth
    -category registration). Nothing stale survives, because nothing about
    the construction path was stored."""
    module = NativeDropout(0.5, seed=269)
    original = module.generator
    replacement = NativeGenerator(271)
    module.generator = replacement
    assert module.generator is replacement
    assert module.generators() == [replacement]
    assert module.generator_state_dict()["generator"]["seed"] == 271
    assert not hasattr(module, "owns_generator")
    # The displaced generator is untouched — the module never owned
    # anything closable about it.
    assert original.seed == 269 and original.calls == 0
    x = NativeTensor.from_array(VALUES)
    module(x).close()
    assert replacement.calls == 1 and original.calls == 0
    x.close()


# ==========================================================================
# 2. Registration and state
# ==========================================================================

def test_the_generator_is_registered_under_the_canonical_name():
    module = NativeDropout(0.5, seed=11)
    assert [name for name, _ in module.named_generators()] == ["generator"]
    assert module.generators() == [module.generator]
    assert module.generator_state_dict() == {
        "generator": {
            "algorithm": "tensorforge.splitmix64",
            "algorithm_version": 1,
            "seed": 11,
            "calls": 0,
        }
    }


def test_the_generator_is_not_a_parameter_buffer_or_child():
    module = NativeDropout(0.5, seed=13)
    assert module.state_dict() == {}
    assert list(module.parameters()) == []
    assert list(module.named_parameters()) == []
    assert list(module.named_buffers()) == []
    assert [name for name, _ in module.named_modules()] == [""]
    assert "generator" not in module._parameters
    assert "generator" not in module._buffers
    assert "generator" not in module._modules
    assert "generator" in module._generators


def test_the_generator_is_discovered_through_a_parent_module():
    class Model(NativeModule):
        def __init__(self):
            super().__init__()
            self.linear = NativeLinear(4, 4)
            self.drop = NativeDropout(0.5, seed=17)

        def forward(self, x):
            return self.drop(self.linear(x))

    model = Model()
    assert [name for name, _ in model.named_generators()] == [
        "drop.generator"
    ]
    assert model.generators() == [model.drop.generator]
    assert set(model.generator_state_dict()) == {"drop.generator"}
    # ...and the ordinary tensor state dict is still exactly the linear's.
    assert set(model.state_dict()) == {"linear.weight", "linear.bias"}
    for value in model.state_dict().values():
        assert isinstance(value, NativeTensor)
    for _, parameter in model.named_parameters():
        parameter.close()


def test_recursive_and_non_recursive_traversal():
    class Model(NativeModule):
        def __init__(self):
            super().__init__()
            self.drop = NativeDropout(0.5, seed=19)

        def forward(self, x):
            return self.drop(x)

    model = Model()
    assert model.generators(recurse=True) == [model.drop.generator]
    assert model.generators(recurse=False) == []
    assert [n for n, _ in model.named_generators(recurse=False)] == []


def test_load_generator_state_dict_preserves_identity():
    module = NativeDropout(0.5, seed=23)
    original = module.generator
    module.load_generator_state_dict({
        "generator": {
            "algorithm": "tensorforge.splitmix64",
            "algorithm_version": 1,
            "seed": 4242,
            "calls": 7,
        }
    })
    assert module.generator is original, "the load replaced the object"
    assert module.generator.seed == 4242
    assert module.generator.calls == 7
    # ...and the next forward continues from the loaded index.
    x = NativeTensor.from_array(VALUES)
    y = module(x)
    expected, _ = core_reference(VALUES, 0.5, 4242, 7)
    assert np.array_equal(y.to_numpy(), expected)
    assert module.generator.calls == 8
    y.close()
    x.close()


def test_train_and_eval_transitions_do_not_touch_generator_state():
    module = NativeDropout(0.5, seed=29)
    before = module.generator.state()
    for _ in range(3):
        module.eval()
        assert module.generator.state() == before
        module.train()
        assert module.generator.state() == before
        module.train(False)
        assert module.generator.state() == before
        module.train(True)
        assert module.generator.state() == before


def test_dropping_the_module_does_not_reset_or_mutate_a_shared_generator():
    generator = NativeGenerator(31)
    module = NativeDropout(0.5, generator=generator)
    x = NativeTensor.from_array(VALUES)
    module(x).close()
    assert generator.calls == 1
    del module
    gc.collect()
    # The module owns nothing about it: no close(), no reset, no mutation.
    assert generator.calls == 1
    assert generator.seed == 31
    assert generator._has_active_reservation() is False
    x.close()


def test_unregistering_another_alias_leaves_the_module_registration():
    generator = NativeGenerator(37)
    module = NativeDropout(0.5, generator=generator)

    class Holder(NativeModule):
        def __init__(self):
            super().__init__()
            self.other = generator

        def forward(self, x):
            return x

    holder = Holder()
    assert holder.other is generator
    holder.other = None                      # unregister that alias
    assert list(holder.named_generators()) == []
    # The module's own registration is untouched, and so is the object.
    assert module.generator is generator
    assert [n for n, _ in module.named_generators()] == ["generator"]
    assert generator.calls == 0


# ==========================================================================
# 3. Training behavior
# ==========================================================================

def test_training_forward_reproduces_the_committed_g2_vector():
    module = NativeDropout(VECTOR_P, seed=VECTOR_SEED)
    x = NativeTensor.from_array(VALUES, requires_grad=True)
    y = module(x)
    mask = committed_mask()
    assert np.array_equal(y.to_numpy(), VALUES * mask)
    assert np.array_equal(y._graph_resources[0].to_numpy(), mask)
    y.backward(gradient=ones_like(y))
    assert np.array_equal(x.grad.to_numpy(), mask)
    assert module.generator.calls == 1
    y.close()
    x.close()


def test_training_forward_equals_the_operation_on_the_same_state():
    """The module is a wrapper, so it must produce *exactly* what the G3
    operation produces from the same generator state."""
    module = NativeDropout(0.4, seed=101)
    direct_generator = NativeGenerator(101)
    x = NativeTensor.from_array(VALUES, requires_grad=True)
    through_module = module(x)
    through_operation = x.dropout(0.4, generator=direct_generator)
    assert np.array_equal(through_module.to_numpy(),
                          through_operation.to_numpy())
    assert module.generator.calls == direct_generator.calls == 1
    for tensor in (through_module, through_operation, x):
        tensor.close()


def test_a_successful_forward_consumes_exactly_one_call():
    module = NativeDropout(0.5, seed=103)
    x = NativeTensor.from_array(VALUES)
    for expected in (1, 2, 3):
        module(x).close()
        assert module.generator.calls == expected


def test_repeated_forwards_use_consecutive_call_indices():
    module = NativeDropout(0.5, seed=107)
    x = NativeTensor.from_array(VALUES)
    for index in range(4):
        y = module(x)
        expected, _ = core_reference(VALUES, 0.5, 107, index)
        assert np.array_equal(y.to_numpy(), expected), index
        y.close()
    assert module.generator.calls == 4
    x.close()


def test_a_no_grad_forward_consumes_once_and_retains_no_mask(live_storages):
    module = NativeDropout(0.5, seed=109)
    x = NativeTensor.from_array(VALUES)      # requires_grad=False
    baseline = len(live_storages)
    y = module(x)
    assert module.generator.calls == 1
    assert y.requires_grad is False
    assert y._graph_resources == ()
    assert len(live_storages) == baseline + 1     # only the output
    y.close()
    assert len(live_storages) == baseline
    x.close()


@pytest.mark.parametrize(
    "shape", [(1,), (12,), (3, 4), (2, 3, 4), (2, 2, 2, 2)],
)
def test_ranks_are_preserved(shape):
    values = np.arange(1.0, 1.0 + int(np.prod(shape))).reshape(shape)
    module = NativeDropout(0.5, seed=113)
    x = NativeTensor.from_array(values, requires_grad=True)
    y = module(x)
    assert y.shape == shape
    expected, _ = core_reference(values, 0.5, 113, 0)
    assert np.array_equal(y.to_numpy(), expected)
    y.close()
    x.close()


def test_a_scalar_input_works():
    module = NativeDropout(0.5, seed=127)
    x = NativeTensor.full((), 3.0, requires_grad=True)
    y = module(x)
    assert y.shape == ()
    expected, mask = core_reference([3.0], 0.5, 127, 0)
    assert float(y.to_numpy()) == float(expected[0])
    y.backward()
    assert float(x.grad.to_numpy()) == float(mask[0])
    assert module.generator.calls == 1
    y.close()
    x.close()


@pytest.mark.parametrize("build", ["transpose", "narrow", "offset"])
def test_noncontiguous_inputs_work_and_keep_the_logical_mask(build):
    base = np.arange(1.0, 13.0).reshape(3, 4)
    x = NativeTensor.from_array(base, requires_grad=True)
    view = {"transpose": lambda: x.T,
            "narrow": lambda: x.narrow(1, 1, 2),
            "offset": lambda: x.narrow(0, 1, 2)}[build]()
    module = NativeDropout(0.5, seed=131)
    y = module(view)
    expected, _ = core_reference(view.to_numpy(), 0.5, 131, 0)
    assert np.array_equal(y.to_numpy(), expected)
    assert module.generator.calls == 1
    y.backward(gradient=ones_like(y))
    assert x.grad is not None
    for tensor in (y, view, x):
        tensor.close()


def test_a_parameter_input_receives_a_gradient():
    parameter = NativeParameter(VALUES.reshape(3, 4))
    module = NativeDropout(0.5, seed=137)
    y = module(parameter)
    _, mask = core_reference(VALUES.reshape(3, 4), 0.5, 137, 0)
    y.backward(gradient=ones_like(y))
    assert np.array_equal(parameter.grad.to_numpy(), mask)
    y.close()
    parameter.close()


def test_composition_inside_native_sequential():
    model = NativeSequential(
        NativeLinear(4, 4), NativeReLU(), NativeDropout(0.5, seed=139)
    )
    x = NativeTensor.from_array(VALUES.reshape(3, 4), requires_grad=True)
    y = model(x)
    assert y.shape == (3, 4)
    generator = model[2].generator
    assert generator.calls == 1
    assert [name for name, _ in model.named_generators()] == ["2.generator"]
    y.backward(gradient=ones_like(y))
    assert x.grad is not None
    y.close()
    x.close()
    for _, parameter in model.named_parameters():
        parameter.close()


# ==========================================================================
# 4. Evaluation behavior
# ==========================================================================

def test_evaluation_returns_the_exact_input_object():
    module = NativeDropout(0.5, seed=149)
    module.eval()
    assert module.training is False
    x = NativeTensor.from_array(VALUES, requires_grad=True)
    result = module(x)
    assert result is x
    assert result.requires_grad is True
    assert result.is_leaf is True
    assert module.generator.calls == 0
    assert np.array_equal(x.to_numpy(), VALUES)
    x.close()


def test_evaluation_reaches_neither_the_operation_nor_the_core(monkeypatch,
                                                               live_storages):
    """Proved with tripwires, not by inference."""
    reached = []

    def tripwire(*args, **kwargs):
        reached.append(1)
        raise AssertionError("evaluation must do no work")

    monkeypatch.setattr(NativeTensor, "dropout", tripwire)
    monkeypatch.setattr(
        cpp.NativeTensorCore, "_dropout_forward_with_mask", tripwire
    )
    monkeypatch.setattr(cpp.NativeTensorCore, "dropout_forward", tripwire)
    monkeypatch.setattr(NativeGenerator, "_reserve_call", tripwire)

    module = NativeDropout(0.5, seed=151)
    module.eval()
    x = NativeTensor.from_array(VALUES)
    baseline = len(live_storages)
    assert module(x) is x
    assert reached == []
    assert len(live_storages) == baseline, "evaluation allocated"
    monkeypatch.undo()
    x.close()


def test_repeated_evaluation_leaves_no_gap_in_the_stream():
    """The stream contract: a training forward, any number of eval
    forwards, then a training forward — consecutive indices, tied to G2
    reference masks rather than to "these two look different"."""
    module = NativeDropout(0.5, seed=157)
    x = NativeTensor.from_array(VALUES)

    first = module(x)
    expected_0, _ = core_reference(VALUES, 0.5, 157, 0)
    assert np.array_equal(first.to_numpy(), expected_0)
    assert module.generator.calls == 1
    first.close()

    module.eval()
    for _ in range(5):
        assert module(x) is x
    assert module.generator.calls == 1
    assert module.generator.seed == 157

    module.train()
    second = module(x)
    expected_1, _ = core_reference(VALUES, 0.5, 157, 1)
    assert np.array_equal(second.to_numpy(), expected_1), (
        "evaluation opened a gap in the random stream"
    )
    assert module.generator.calls == 2
    second.close()
    x.close()


def test_train_false_matches_eval_and_train_true_restores():
    module = NativeDropout(0.5, seed=163)
    x = NativeTensor.from_array(VALUES)
    module.train(False)
    assert module.training is False
    assert module(x) is x
    assert module.generator.calls == 0
    module.train(True)
    assert module.training is True
    y = module(x)
    assert y is not x
    assert module.generator.calls == 1
    y.close()
    x.close()


def test_mode_propagates_through_native_sequential():
    model = NativeSequential(NativeReLU(), NativeDropout(0.5, seed=167))
    dropout = model[1]
    assert dropout.training is True
    model.eval()
    assert dropout.training is False
    x = NativeTensor.from_array(VALUES)
    relu_out = model(x)
    # The dropout is identity in eval, so the sequence's output is the
    # ReLU's own output object.
    assert dropout.generator.calls == 0
    relu_out.close()
    model.train()
    assert dropout.training is True
    out = model(x)
    assert dropout.generator.calls == 1
    out.close()
    x.close()


def test_evaluation_still_validates_its_input():
    """Identity is the result, not a bypass of the contract."""
    module = NativeDropout(0.5, seed=173)
    module.eval()
    closed = NativeTensor.from_array(VALUES)
    closed.close()
    with pytest.raises(RuntimeError, match="closed"):
        module(closed)
    with pytest.raises(TypeError, match="NativeTensor"):
        module(np.zeros(4))
    assert module.generator.calls == 0


# ==========================================================================
# 5. p == 0
# ==========================================================================

@pytest.mark.parametrize("mode", ["train", "eval"])
def test_p_zero_is_identity_in_both_modes(mode):
    module = NativeDropout(0.0, seed=179)
    if mode == "eval":
        module.eval()
    x = NativeTensor.from_array(VALUES, requires_grad=True)
    result = module(x)
    assert result is x
    assert module.generator.calls == 0
    assert module.generator.state() == {
        "algorithm": "tensorforge.splitmix64", "algorithm_version": 1,
        "seed": 179, "calls": 0,
    }
    assert x.requires_grad is True and x.is_leaf is True
    assert np.array_equal(x.to_numpy(), VALUES)
    x.close()


def test_p_zero_reserves_nothing_and_reaches_no_kernel(monkeypatch,
                                                       live_storages):
    """``p == 0`` identity belongs to the G3 operation (design §6.2), so
    the module deliberately does not duplicate the rule — but the
    observable behavior is the same either way, and that is what this
    proves: the operation really is reached, and it really does no work."""
    delegated = []
    original = NativeTensor.dropout

    def counting_dropout(self, p, *, generator):
        delegated.append(p)
        return original(self, p, generator=generator)

    reached = []

    def tripwire(*args, **kwargs):
        reached.append(1)
        raise AssertionError("p == 0 must reserve nothing and draw nothing")

    monkeypatch.setattr(NativeTensor, "dropout", counting_dropout)
    monkeypatch.setattr(
        cpp.NativeTensorCore, "_dropout_forward_with_mask", tripwire
    )
    monkeypatch.setattr(NativeGenerator, "_reserve_call", tripwire)

    module = NativeDropout(0.0, seed=181)
    x = NativeTensor.from_array(VALUES, requires_grad=True)
    baseline = len(live_storages)
    assert module(x) is x
    # The module delegated (it did not short-circuit)...
    assert delegated == [0.0]
    # ...and the operation reserved nothing, drew nothing, allocated
    # nothing, and built no graph.
    assert reached == []
    assert len(live_storages) == baseline
    assert x._graph_resources == () and x.is_leaf is True
    monkeypatch.undo()
    x.close()


# ==========================================================================
# 6. Shared and independent generators
# ==========================================================================

def test_two_modules_sharing_one_generator_consume_one_ordered_stream():
    shared = NativeGenerator(191)
    first = NativeDropout(0.5, generator=shared)
    second = NativeDropout(0.5, generator=shared)
    assert first.generator is second.generator is shared

    x = NativeTensor.from_array(VALUES)
    a = first(x)
    b = second(x)
    expected_0, _ = core_reference(VALUES, 0.5, 191, 0)
    expected_1, _ = core_reference(VALUES, 0.5, 191, 1)
    assert np.array_equal(a.to_numpy(), expected_0)
    assert np.array_equal(b.to_numpy(), expected_1)
    assert shared.calls == 2
    # Order follows successful-forward order, not construction order.
    c = second(x)
    expected_2, _ = core_reference(VALUES, 0.5, 191, 2)
    assert np.array_equal(c.to_numpy(), expected_2)
    assert shared.calls == 3
    for tensor in (a, b, c, x):
        tensor.close()


def test_a_parent_deduplicates_a_shared_generator_by_identity():
    shared = NativeGenerator(193)

    class Model(NativeModule):
        def __init__(self):
            super().__init__()
            self.first = NativeDropout(0.5, generator=shared)
            self.second = NativeDropout(0.5, generator=shared)

        def forward(self, x):
            return self.second(self.first(x))

    model = Model()
    # One object, so one entry, under its first-discovered canonical name.
    assert model.generators() == [shared]
    assert [name for name, _ in model.named_generators()] == [
        "first.generator"
    ]
    assert set(model.generator_state_dict()) == {"first.generator"}
    x = NativeTensor.from_array(VALUES)
    y = model(x)
    assert shared.calls == 2, "one shared stream, two successful forwards"
    y.close()
    x.close()


def test_independent_modules_with_equal_seeds_stay_independent():
    first = NativeDropout(0.5, seed=197)
    second = NativeDropout(0.5, seed=197)
    assert first.generator is not second.generator
    assert first.generator.state() == second.generator.state()

    x = NativeTensor.from_array(VALUES)
    a = first(x)
    b = second(x)
    assert np.array_equal(a.to_numpy(), b.to_numpy()), (
        "equal initial state must give equal first outputs"
    )
    assert first.generator.calls == second.generator.calls == 1

    # Advancing one does not advance the other.
    c = first(x)
    assert first.generator.calls == 2
    assert second.generator.calls == 1
    d = second(x)
    assert np.array_equal(c.to_numpy(), d.to_numpy())
    for tensor in (a, b, c, d, x):
        tensor.close()


def test_the_default_gives_every_module_its_own_generator():
    modules = [NativeDropout(0.5) for _ in range(3)]
    identities = {id(module.generator) for module in modules}
    assert len(identities) == 3
    x = NativeTensor.from_array(VALUES)
    for module in modules:
        module(x).close()
    assert all(module.generator.calls == 1 for module in modules)
    x.close()


def test_one_module_registered_twice_owns_exactly_one_generator():
    dropout = NativeDropout(0.5, seed=199)

    class Model(NativeModule):
        def __init__(self):
            super().__init__()
            self.a = dropout
            self.b = dropout                 # the same module object

        def forward(self, x):
            return self.b(self.a(x))

    model = Model()
    assert model.a is model.b is dropout
    # One module, one generator, one entry.
    assert model.generators() == [dropout.generator]
    assert [name for name, _ in model.named_generators()] == ["a.generator"]
    x = NativeTensor.from_array(VALUES)
    y = model(x)
    assert dropout.generator.calls == 2, "two forwards, one stream"
    y.close()
    x.close()


# ==========================================================================
# 7. Failure atomicity
# ==========================================================================

@pytest.mark.parametrize(
    "kwargs, error",
    [
        ({"p": 1.0}, ValueError),
        ({"p": True}, TypeError),
        ({"p": None}, TypeError),
        ({"p": 0.5, "seed": -1}, ValueError),
        ({"p": 0.5, "seed": "x"}, TypeError),
        ({"p": 0.5, "generator": 7}, TypeError),
    ],
)
def test_construction_failures_allocate_nothing(kwargs, error, live_storages):
    baseline = len(live_storages)
    with pytest.raises(error):
        NativeDropout(**kwargs)
    assert len(live_storages) == baseline


def test_an_invalid_probability_fails_before_any_generator_is_built(
    monkeypatch,
):
    """Ordering, not just outcome: a rejected probability must never draw
    entropy or construct a generator."""
    built = []
    original = NativeGenerator.__init__

    def counting_init(self, seed=None):
        built.append(seed)
        original(self, seed)

    monkeypatch.setattr(NativeGenerator, "__init__", counting_init)
    with pytest.raises(ValueError):
        NativeDropout(1.0, seed=5)
    with pytest.raises(TypeError):
        NativeDropout(True, seed=5)
    assert built == []
    monkeypatch.undo()


def test_a_conflicting_argument_pair_fails_before_any_generator_is_built(
    monkeypatch,
):
    supplied = NativeGenerator(211)
    built = []
    original = NativeGenerator.__init__

    def counting_init(self, seed=None):
        built.append(seed)
        original(self, seed)

    monkeypatch.setattr(NativeGenerator, "__init__", counting_init)
    with pytest.raises(TypeError):
        NativeDropout(0.5, seed=1, generator=supplied)
    assert built == [], "a conflicting call still built a generator"
    assert supplied.seed == 211 and supplied.calls == 0
    monkeypatch.undo()


def test_a_failing_entropy_draw_leaves_nothing_behind(monkeypatch,
                                                      live_storages):
    import secrets

    def failing_randbits(bits):
        raise OSError("injected entropy failure")

    monkeypatch.setattr(secrets, "randbits", failing_randbits)
    baseline = len(live_storages)
    with pytest.raises(OSError, match="injected entropy"):
        NativeDropout(0.5)
    assert len(live_storages) == baseline
    monkeypatch.undo()
    # ...and construction works again immediately.
    module = NativeDropout(0.5)
    assert isinstance(module.generator, NativeGenerator)


def test_a_failing_generator_construction_leaves_nothing_behind(monkeypatch):
    def exploding_init(self, seed=None):
        raise RuntimeError("injected generator construction failure")

    monkeypatch.setattr(NativeGenerator, "__init__", exploding_init)
    with pytest.raises(RuntimeError, match="injected generator"):
        NativeDropout(0.5, seed=3)
    monkeypatch.undo()


def test_a_failing_generator_registration_leaves_the_generator_unchanged(
    monkeypatch,
):
    """Injected at the generator registration specifically, not at any
    attribute write, so the failure really is the one being tested."""
    supplied = NativeGenerator(223)
    before = supplied.state()
    original = NativeModule.__setattr__

    def exploding_setattr(self, name, value):
        if isinstance(value, NativeGenerator):
            raise RuntimeError("injected generator registration failure")
        return original(self, name, value)

    monkeypatch.setattr(NativeModule, "__setattr__", exploding_setattr)
    with pytest.raises(RuntimeError, match="injected generator registration"):
        NativeDropout(0.5, generator=supplied)
    monkeypatch.undo()
    # The supplied generator is bit-identical and still usable.
    assert supplied.state() == before
    assert supplied._has_active_reservation() is False
    module = NativeDropout(0.5, generator=supplied)
    assert module.generator is supplied


def test_a_delegated_forward_failure_consumes_no_call(monkeypatch,
                                                      live_storages):
    """The wrapper adds no failure hole: a G3-level failure behaves
    through the module exactly as it does directly."""
    module = NativeDropout(0.5, seed=227)
    x = NativeTensor.from_array(VALUES, requires_grad=True)
    baseline = len(live_storages)

    def boom(result):
        raise RuntimeError("injected delivery failure")

    monkeypatch.setattr(
        native_tensor_module, "_deliver_dropout_result", boom
    )
    with pytest.raises(RuntimeError, match="injected delivery"):
        module(x)
    monkeypatch.undo()

    assert module.generator.calls == 0
    assert module.generator._has_active_reservation() is False
    assert len(live_storages) == baseline
    assert x.closed is False and np.array_equal(x.to_numpy(), VALUES)
    assert x.grad is None
    # The unconsumed index is reused by the next successful forward.
    y = module(x)
    expected, _ = core_reference(VALUES, 0.5, 227, 0)
    assert np.array_equal(y.to_numpy(), expected)
    assert module.generator.calls == 1
    y.close()
    x.close()


def test_repeated_delegated_failures_do_not_accumulate(monkeypatch,
                                                       live_storages):
    module = NativeDropout(0.5, seed=229)
    x = NativeTensor.from_array(VALUES, requires_grad=True)
    baseline = len(live_storages)

    def boom(result):
        raise RuntimeError("injected")

    monkeypatch.setattr(
        native_tensor_module, "_deliver_dropout_result", boom
    )
    for _ in range(10):
        with pytest.raises(RuntimeError):
            module(x)
        assert module.generator.calls == 0
        assert len(live_storages) == baseline
    monkeypatch.undo()
    x.close()


# ==========================================================================
# 8. Input validation
# ==========================================================================

@pytest.mark.parametrize(
    "bad", [None, 0, 1.5, "tensor", [1.0, 2.0], np.zeros(4), object()],
)
def test_a_non_native_tensor_input_is_rejected_without_consuming(bad):
    module = NativeDropout(0.5, seed=233)
    with pytest.raises(TypeError, match="NativeTensor"):
        module(bad)
    assert module.generator.calls == 0
    assert module.generator._has_active_reservation() is False


def test_a_stable_tensor_is_rejected():
    """No implicit conversion in either direction."""
    from tensorforge.tensor import Tensor

    module = NativeDropout(0.5, seed=239)
    with pytest.raises(TypeError):
        module(Tensor(np.zeros(4)))
    assert module.generator.calls == 0


def test_a_closed_input_is_rejected_without_consuming():
    module = NativeDropout(0.5, seed=241)
    x = NativeTensor.from_array(VALUES)
    x.close()
    with pytest.raises(RuntimeError, match="closed"):
        module(x)
    assert module.generator.calls == 0
    assert module.generator._has_active_reservation() is False


def test_live_storage_returns_to_baseline_across_a_full_lifecycle(
    live_storages,
):
    baseline = len(live_storages)
    module = NativeDropout(0.5, seed=251)
    # A module owns no native storage: constructing it moves nothing.
    assert len(live_storages) == baseline

    x = NativeParameter(VALUES)
    y = module(x)
    g = ones_like(y)
    y.backward(gradient=g)
    y.close()
    g.close()
    x.zero_grad()
    x.close()
    gc.collect()
    assert len(live_storages) == baseline


# ==========================================================================
# 9. The version-1 checkpoint boundary (the honest G4 limitation)
# ==========================================================================

def _dropout_model():
    class Model(NativeModule):
        def __init__(self):
            super().__init__()
            self.linear = NativeLinear(4, 4, seed=1)
            self.drop = NativeDropout(0.5, seed=257)

        def forward(self, x):
            return self.drop(self.linear(x))

    return Model()


def test_state_dict_stays_tensor_only_for_a_model_containing_dropout():
    model = _dropout_model()
    state = model.state_dict()
    assert set(state) == {"linear.weight", "linear.bias"}
    for value in state.values():
        assert isinstance(value, NativeTensor)
    # The generator is reachable only through the generator surface.
    assert set(model.generator_state_dict()) == {"drop.generator"}
    for _, parameter in model.named_parameters():
        parameter.close()


def test_a_version_2_checkpoint_carries_the_module_generator(tmp_path):
    """The gap G4 left open and G5 closed, from the module's side.

    G4 shipped the module over a version-1 format that had no generator
    section, so a save preserved parameters and buffers and silently
    omitted the random stream. G5 moved the format to version 2: the
    module's registered generator is now written under its canonical path
    and restored exactly, in place, into the same object."""
    import json

    from tensorforge.experimental import (
        NativeSGD, load_native_checkpoint, native_checkpoint,
        save_native_checkpoint,
    )

    model = _dropout_model()
    optimizer = NativeSGD(model.parameters(), lr=0.1)
    x = NativeTensor.from_array(VALUES.reshape(3, 4))
    model(x).close()
    assert model.drop.generator.calls == 1

    path = tmp_path / "checkpoint.npz"
    save_native_checkpoint(path, model, optimizer)

    assert native_checkpoint._FORMAT_VERSION == 3
    with np.load(path, allow_pickle=False) as archive:
        manifest = json.loads(archive["manifest"].tobytes().decode("utf-8"))
        # Generator state rides the manifest only — never an NPZ array.
        assert not any("generator" in name for name in archive.files)
    assert manifest["format_version"] == 3
    assert sorted(manifest) == ["format", "format_version", "generators",
                                "metadata", "model", "optimizer"]
    assert manifest["generators"] == {
        "keys": ["drop.generator"],
        "entries": {
            "drop.generator": {
                "algorithm": "tensorforge.splitmix64",
                "algorithm_version": 1,
                "seed": "257",
                "calls": "1",
            }
        },
        "aliases": {"drop.generator": "drop.generator"},
    }

    # A load restores the exact stream, in place.
    generator = model.drop.generator
    generator.reseed(999999)
    load_native_checkpoint(path, model, optimizer)
    assert model.drop.generator is generator, "the load replaced the object"
    assert generator.state() == {
        "algorithm": "tensorforge.splitmix64", "algorithm_version": 1,
        "seed": 257, "calls": 1,
    }
    # ...so the next forward is the mask the saved run would have drawn.
    y = model.drop(x)
    expected, _ = core_reference(x.to_numpy(), 0.5, 257, 1)
    assert np.array_equal(y.to_numpy(), expected)
    y.close()

    x.close()
    for _, parameter in model.named_parameters():
        parameter.close()


def test_a_load_never_fabricates_generator_state(tmp_path):
    """The half that matters most: no seed and no counter is ever
    invented. A mid-reservation load is refused outright rather than
    capturing or overwriting an ambiguous in-flight stream, and the live
    generator is left exactly as it was found."""
    from tensorforge.experimental import (
        load_native_checkpoint, save_native_checkpoint,
    )

    model = _dropout_model()
    path = tmp_path / "checkpoint.npz"
    save_native_checkpoint(path, model)

    generator = model.drop.generator
    token = generator._reserve_call()
    before = generator.state()
    with pytest.raises(RuntimeError, match="reservation"):
        load_native_checkpoint(path, model)
    assert generator.state() == before
    assert generator._has_active_reservation() is True
    # ...and a save is refused for the same reason, changing nothing.
    with pytest.raises(RuntimeError, match="reservation"):
        save_native_checkpoint(tmp_path / "mid-draw.npz", model)
    assert not (tmp_path / "mid-draw.npz").exists()
    assert generator.state() == before

    generator._abandon_call(token)
    load_native_checkpoint(path, model)             # recovers immediately
    assert model.drop.generator is generator
    for _, parameter in model.named_parameters():
        parameter.close()


# ==========================================================================
# 10. Separation: what G4 did and did not move
# ==========================================================================

def test_the_public_export_exists_experimentally_and_not_at_top_level():
    import tensorforge
    import tensorforge.experimental as experimental

    assert experimental.NativeDropout is NativeDropout
    assert "NativeDropout" in experimental.__all__
    assert experimental.__all__.count("NativeDropout") == 1
    assert not hasattr(tensorforge, "NativeDropout")
    assert "NativeDropout" not in getattr(tensorforge, "__all__", [])


def test_native_modules_gained_exactly_one_entry():
    assert "NativeDropout" in cpp.NATIVE_MODULES
    assert cpp.NATIVE_MODULES.count("NativeDropout") == 1
    assert cpp.NATIVE_MODULES[-1] == "NativeDropout"
    assert tuple(cpp.backend_info()["native_modules"]) == cpp.NATIVE_MODULES
    # A module is not an operation, a loss, a metric, or an optimizer.
    for inventory in (cpp.AUTOGRAD_OPS, cpp.TENSOR_CORE_OPS, cpp.RAW_KERNELS,
                      cpp.TENSOR_CORE_KERNELS, cpp.NATIVE_LOSSES,
                      cpp.NATIVE_METRICS, cpp.NATIVE_OPTIMIZERS,
                      cpp.STATE_SUPPORT, cpp.UNSUPPORTED):
        assert "NativeDropout" not in inventory, inventory


def test_the_capability_boundary_did_not_move():
    from tensorforge.experimental import native_checkpoint

    assert cpp.UNSUPPORTED == ("float32", "cuda", "amp")
    assert cpp.SUPPORTED_DTYPES == ("float64",)
    assert cpp.SUPPORTED_DEVICES == ("cpu",)
    assert native_checkpoint._FORMAT_VERSION == 3
    assert cpp.STATE_SUPPORT == (
        "persistent_buffers", "state_dict", "load_state_dict",
        "generator_state",
        "save_native_checkpoint", "load_native_checkpoint",
        "checkpoint_generator_state",
    )
    # G3's operation and G2's Core are exactly where they were.
    assert "dropout" in cpp.AUTOGRAD_OPS
    assert "dropout_forward" in cpp.TENSOR_CORE_OPS
    dropout_symbols = [name for name in cpp._CHECKED_KERNELS
                       if "dropout" in name or "random" in name]
    assert dropout_symbols == ["tf_core_dropout_forward"]


def test_g4_added_no_operation_kernel_or_generic_rng_surface():
    import tensorforge.experimental as experimental

    for absent in ("NativeDropout2d", "NativeDropout3d", "dropout",
                   "default_generator", "manual_seed", "rand", "randn"):
        assert not hasattr(experimental, absent), absent
    assert not hasattr(cpp.NativeTensorCore, "dropout")
    assert not hasattr(cpp.NativeTensorCore, "dropout_backward")
    # The module's forward is a delegation, not a second implementation.
    import inspect

    source = inspect.getsource(NativeDropout.forward)
    assert "dropout(" in source
    for forbidden in ("_dropout_forward_with_mask", "dropout_forward",
                      "_reserve_call", "_commit_call", "_abandon_call",
                      "graph_resources", "np.random", "seed="):
        assert forbidden not in source, forbidden


def test_the_boundary_move_belongs_to_g10_not_to_g4():
    """G4 shipped the module without moving the capability boundary, and
    that ordering is the whole point of the milestone. The boundary has
    since moved at **G10**, so what stays durable is the attribution —
    plus the standing rule that no machine-specific benchmark artifact is
    ever committed."""
    from pathlib import Path

    repo_root = Path(__file__).resolve().parent.parent
    assert not (repo_root / "benchmark_results").exists()
    from tensorforge.backends import cpp

    assert "dropout" not in cpp.UNSUPPORTED
    assert cpp.UNSUPPORTED == ("float32", "cuda", "amp")
    # The module G4 shipped is still exactly one entry, in one inventory.
    assert cpp.NATIVE_MODULES.count("NativeDropout") == 1
