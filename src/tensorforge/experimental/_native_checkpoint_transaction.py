"""Private, whole-checkpoint load transaction (Phase G, milestone G5 —
see docs/native_rng_dropout_design.md §10.7 and §10.8).

**This module is private.** It is deliberately absent from
``tensorforge.experimental.__all__`` and must stay that way: it is not a
public transaction API, and nothing outside ``tensorforge.experimental``
may depend on it. Its one caller is
``native_checkpoint.load_native_checkpoint``.

What it does
------------

One narrow thing: run the **commit** phase of a checkpoint load — model
state, optimizer state, and registered generator state — as a *single*
transaction under *one* rollback guard and *one* concurrency guard, so
that neither an exception nor another thread can leave those four state
families disagreeing with each other.

Before G5 the loader committed two components through two independent
public loaders and documented the gap between them honestly: an
interruption there could leave the model restored while the optimizer
stayed stale. Adding a third component (generators) would have made that
gap wider and the resume story unprovable, so §10.7 replaces the wording
with a guarantee, and this module is where the guarantee lives.

The two guarantees are separate and both are needed
---------------------------------------------------

**Atomicity** (§10.7) is about *this* transaction: any exception during
the commit restores every component. **Serializability** (§10.8) is about
*other* transactions: while this one commits, no other participating
state load may run. Atomic without serializable still permits two
concurrent loads that each succeed and leave the model from one archive
beside the optimizer from the other — a hybrid state no per-component
guarantee can see. So the commit runs under the shared state-transaction
guard, in the universal order:

1. the shared ``_native_state_lock`` guard;
2. every unique target generator's lock, in the global ``id()`` order.

Both are acquired by ``locked_generators`` and held across the reservation
recheck, the rollback snapshots, the commit, and any rollback — released
only once the transaction has fully finished, one way or the other. The
components' own loaders are then called *inside* that, and each re-enters
the guard through the ``RLock`` rather than deadlocking on it.

The transaction model
---------------------

The loader has already finished prevalidation (§10.7 Phase 1) and the
ordinary part of staging (Phase 2) before anything here runs: the archive
is parsed, validated against the live model, and decoded into staged
``NativeTensor`` values, none of which touches live state.

**Rollback snapshots are taken here, not there**, because they must
reflect the state at the *actual* commit boundary. A snapshot captured
before the guard was held could describe a model that another transaction
has since replaced, and rolling back to it would undo someone else's
committed work. So, with both locks held, this module captures:

- for each registered parameter and persistent buffer, an owning
  ``NativeTensor`` copy of its current value plus its current version;
- for the optimizer, its complete ``state_dict()`` (owning moment copies
  for ``NativeAdam``);
- for each unique registered generator, its ``(seed, calls)`` pair.

Every allocation the rollback could need therefore happens *before* the
first commit step and *after* the locks are held, which is what lets the
rollback be **unfailable**: it restores already-allocated cores and
already-computed immutable values with plain attribute assignments, and
calls nothing that can raise. A snapshot failure aborts with nothing
committed.

**Commit** runs in the locked order model → optimizer → generators, each
step through the component's own existing loader, and appends to a record
of what has actually succeeded. Each of the three is already internally
atomic (``replace_native_state`` for the model, a validated assignment /
staged swap for the optimizer, ``replace_generator_states`` for the
generators), so a failure *inside* a step leaves that step's own component
untouched and nothing is recorded for it.

**Rollback** unwinds in the reverse order, and only the steps that
actually completed:

- generators are written back through ``_assign_state``, the same
  non-failing integer write seam the multi-generator transaction commits
  and rolls back with;
- the optimizer's scalars and step counts are reassigned, and each live
  moment tensor **swaps cores** with its rollback snapshot;
- every parameter and persistent buffer swaps cores with its snapshot the
  same way, and each parameter's value version is written straight back,
  so a rolled-back load moves no version at all and stales no graph.

The core **swap** (rather than a hand-off) is deliberate: it keeps the
ownership rule trivially true on every path — the live object always owns
exactly one core, the caller's snapshot wrapper always owns exactly one
core, and the caller's existing ``finally`` closes every snapshot exactly
once whether the transaction committed or rolled back. Nothing is ever
released here; snapshots are appended to the caller-owned
``plan.owned_snapshots`` list as they are created, so a failure *while
snapshotting* still leaves the caller everything it must close.

    THE COMMIT BOUNDARY is ``_reach_commit_boundary()``, the point at
    which all three components have committed. Before it the transaction
    is fully reversible; after it the caller releases the snapshots and
    there is no way back.

What identity survives
----------------------

Everything the components' own contracts preserve. Every
``NativeParameter``, persistent buffer, ``NativeGenerator``, and the
optimizer object itself are the same objects afterwards, committed or
rolled back — this module never constructs or substitutes one. The one
thing a rollback restores *by value rather than by identity* is the
optimizer's private moment buffers: ``NativeAdam.load_state_dict``
releases the buffers it replaces, so a rolled-back load restores their
**values** into the optimizer's current buffer objects. Those buffers are
private optimizer internals with no public identity contract (a
*successful* load replaces them outright), while every publicly
identified object is untouched.

Failure-injection seams
-----------------------

The module-level ``_capture_rollback`` / ``_commit_model`` /
``_commit_optimizer`` / ``_commit_generators`` / ``_reach_commit_boundary``
functions exist so tests can simulate a failure at each position — and so
a test can *block* inside one to force an interleaving — and the three
``_rollback_*`` functions are separated for the same reason. They are
private seams, not production flags: nothing in the library ever changes
them, and there is no user-facing failure control.
"""

from collections import namedtuple

from ._native_state_lock import state_transaction
from .native_adam import NativeAdam
from .native_generator import locked_generators, replace_generator_states


# One live model destination and everything its rollback needs, captured
# at the commit boundary.
#
# ``destination``  the registered NativeParameter or persistent-buffer
#                  NativeTensor the commit overwrites. Preserved by
#                  identity; only its core and version move.
# ``snapshot``     a caller-owned independent owning NativeTensor holding
#                  the destination's pre-load value. Rollback swaps cores
#                  with it, so afterwards the snapshot holds the core the
#                  commit installed and the caller closes it as usual.
# ``version``      the destination's pre-load value version (parameters
#                  only; buffers carry none).
ModelRollback = namedtuple(
    "ModelRollback", ("label", "destination", "snapshot", "is_parameter",
                      "version"),
)

# The optimizer and its pre-load ``state_dict()`` — scalars, step counts,
# and (NativeAdam only) owning moment snapshots the caller closes.
OptimizerRollback = namedtuple("OptimizerRollback", ("optimizer", "state"))

# One unique generator's pre-load ``(seed, calls)``, read under its lock.
GeneratorRollback = namedtuple(
    "GeneratorRollback", ("label", "generator", "seed", "calls"),
)

# Everything the rollback needs, captured together at the boundary.
Rollback = namedtuple("Rollback", ("model", "optimizer", "generators"))

# The staged transaction handed to :func:`commit_checkpoint`.
#
# ``staged_optimizer`` and ``generator_entries`` may be None/empty: a
# checkpoint without optimizer state, or a model with no registered
# generators, simply has fewer components — never a different transaction
# shape.
#
# ``owned_snapshots`` is a caller-owned mutable list. Rollback snapshots
# are appended to it as they are created, so the caller's ``finally``
# closes exactly what exists, on every path.
CheckpointPlan = namedtuple(
    "CheckpointPlan",
    ("model", "staged_model", "optimizer", "staged_optimizer",
     "generator_entries", "owned_snapshots"),
)


# ---------------------------------------------------------------------------
# Private seams. Each is one indivisible step of the transaction, factored
# out so a test can monkeypatch exactly one of them — to fail there, or to
# block there and force an interleaving. Keep them tiny: the transaction's
# correctness argument assumes each either fully happens or raises leaving
# its own component untouched.
# ---------------------------------------------------------------------------

def _capture_rollback(plan, targets):
    """Snapshot every live target the commit will overwrite, **at the
    commit boundary** — both locks are already held, so nothing can move
    between this and the first commit step.

    Every snapshot is appended to ``plan.owned_snapshots`` as it is
    created, so a failure partway through still leaves the caller
    everything it has to close."""
    model_snapshots = plan.model.state_dict()
    plan.owned_snapshots.extend(model_snapshots.values())
    model_records = [
        ModelRollback(
            label=name,
            destination=destination,
            snapshot=model_snapshots[name],
            is_parameter=hasattr(destination, "_version"),
            version=getattr(destination, "_version", None),
        )
        for name, destination in plan.model._state_named_tensors()
    ]

    optimizer_record = None
    if plan.staged_optimizer is not None:
        state = plan.optimizer.state_dict()
        if isinstance(plan.optimizer, NativeAdam):
            plan.owned_snapshots.extend(state["m"])
            plan.owned_snapshots.extend(state["v"])
        optimizer_record = OptimizerRollback(
            optimizer=plan.optimizer, state=state,
        )

    # Read through the generator's own private snapshot seam; the lock it
    # takes is one this transaction already holds, so this is a reentrant
    # read of state nothing else can be changing.
    labels = {id(entry.generator): entry.label
              for entry in plan.generator_entries}
    generator_records = []
    for generator in targets:
        seed, calls = generator._snapshot_state()
        generator_records.append(GeneratorRollback(
            label=labels.get(id(generator), "<generator>"),
            generator=generator, seed=seed, calls=calls,
        ))
    return Rollback(model_records, optimizer_record, generator_records)


def _commit_model(model, staged_state):
    """Install the staged model state through the module's own loader.

    ``NativeModule.load_state_dict`` runs the F1 state transaction, which
    is internally all-or-nothing: parameters and persistent buffers are
    swapped together, every affected version moves exactly once, and any
    failure leaves the model completely untouched. So a failure here needs
    no rollback of its own. It re-enters the shared guard this transaction
    already holds."""
    model.load_state_dict(staged_state)


def _commit_optimizer(optimizer, staged_state):
    """Install the staged optimizer state through the optimizer's own
    loader. Preflighted by the checkpoint's prevalidation, so it has no
    ordinary public failure path left; it stays a seam because an
    *injected* or asynchronous failure here is exactly what the
    whole-checkpoint rollback exists for."""
    optimizer.load_state_dict(staged_state)


def _commit_generators(entries):
    """Install every generator state through the shared multi-generator
    transaction, which owns the reservation recheck and its own internal
    rollback. It re-acquires the guard and the very same generator locks
    this transaction already holds — reentrantly, in the same order."""
    replace_generator_states(entries)


def _reach_commit_boundary():
    """The commit boundary itself: a deliberate no-op marking the last
    point at which the whole transaction is still reversible.

    It exists so that "everything committed, nothing released yet" is an
    *addressable* position rather than an argued one — a test raises here
    to prove that a failure after the final component still restores all
    four state families, and blocks here to prove another thread's load
    cannot slip in."""


def _rollback_model(records):
    """Put every model destination back. Non-failing by construction:
    core swaps and integer assignments only, over objects that were live a
    moment ago."""
    for record in records:
        destination = record.destination
        snapshot = record.snapshot
        destination._core, snapshot._core = snapshot._core, destination._core
        if record.is_parameter:
            destination._version = record.version


def _rollback_optimizer(record):
    """Put the optimizer's state back. Non-failing by construction: the
    scalars and counters are immutable values captured before the commit,
    and each moment buffer swaps cores with its snapshot."""
    optimizer = record.optimizer
    state = record.state
    optimizer._lr = state["lr"]
    if isinstance(optimizer, NativeAdam):
        optimizer._betas = state["betas"]
        optimizer._eps = state["eps"]
        optimizer._steps = list(state["step_counts"])
        for label in ("m", "v"):
            live = optimizer._m if label == "m" else optimizer._v
            for index, snapshot in enumerate(state[label]):
                live[index]._core, snapshot._core = (
                    snapshot._core, live[index]._core
                )


def _rollback_generators(records):
    """Put every generator's seed and counter back through the same
    non-failing write seam the multi-generator transaction uses. The lock
    each write takes is one this transaction is still holding."""
    for record in records:
        record.generator._assign_state(record.seed, record.calls)


# ---------------------------------------------------------------------------
# The transaction
# ---------------------------------------------------------------------------

def commit_checkpoint(plan):
    """Commit a fully staged checkpoint as one serialized transaction.

    Takes the universal state-replacement lock order — the shared
    state-transaction guard, then every unique target generator's lock in
    global ``id()`` order — rechecks that no target has a reservation in
    flight, snapshots every live target at the boundary, and commits
    model → optimizer → generators through each component's own loader.
    Returns the ordered list of components that committed.

    While it holds those locks, **no other participating state load can
    run**: not another checkpoint load, not ``load_state_dict``, not
    ``load_generator_state_dict``, not an optimizer state load. So two
    concurrent loads produce one archive's state followed by the other's,
    never a mixture of both.

    On **any** exception — ordinary or a deliverable ``BaseException``
    such as ``KeyboardInterrupt`` — every component that had committed is
    rolled back **before either lock is released**, and the original
    exception propagates unchanged. No other thread can observe the
    partial state, because no other participating operation can be
    running. Afterwards the model, its persistent buffers, the optimizer,
    and every generator hold exactly their pre-load state; no object
    identity has changed; no parameter version has moved.

    Nothing is released here. The caller owns every staged tensor and
    every rollback snapshot — the latter appended to
    ``plan.owned_snapshots`` — and closes them in its own ``finally``, on
    both paths.
    """
    committed = []
    targets_entries = [
        (entry.label, entry.generator) for entry in plan.generator_entries
    ]
    # Order item 1 then item 2, together, for the whole transaction. An
    # empty generator set still takes the guard: serializability is not
    # conditional on the model having generators.
    with state_transaction():
        with locked_generators(
            targets_entries, "load generator state", "load state into",
        ) as targets:
            rollback = _capture_rollback(plan, targets)
            try:
                _commit_model(plan.model, plan.staged_model)
                committed.append("model")
                if plan.staged_optimizer is not None:
                    _commit_optimizer(plan.optimizer, plan.staged_optimizer)
                    committed.append("optimizer")
                if plan.generator_entries:
                    _commit_generators(plan.generator_entries)
                    committed.append("generators")
                # --- COMMIT BOUNDARY. Everything above is reversible.
                _reach_commit_boundary()
            except BaseException:
                # Unwind in reverse, and only what actually committed.
                # Each component's own loader is internally atomic, so a
                # step that raised left its component untouched and was
                # never recorded. This completes while both locks are
                # still held.
                if "generators" in committed:
                    _rollback_generators(rollback.generators)
                if "optimizer" in committed:
                    _rollback_optimizer(rollback.optimizer)
                if "model" in committed:
                    _rollback_model(rollback.model)
                raise
    return committed
