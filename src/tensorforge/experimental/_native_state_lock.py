"""The one shared native state-transaction guard (Phase G, milestone G5 —
see docs/native_rng_dropout_design.md §10.8).

**This module is private.** It is deliberately absent from
``tensorforge.experimental.__all__``, exposes no public API, and must stay
that way: nothing outside ``tensorforge.experimental`` may acquire this
lock, and no caller may be given the ability to hold it across arbitrary
work.

Why it exists
-------------

Every native state-replacement path was individually atomic before G5 —
``replace_native_state`` for parameters and persistent buffers,
``replace_generator_states`` for generators, each optimizer's
``load_state_dict``, and the whole-checkpoint transaction over all of
them. Atomic **individually** is not the same as **serializable**: two
concurrent checkpoint loads could each be internally all-or-nothing and
still finish with the model from one archive and the optimizer or the
generators from the other, because nothing forced one whole transaction
to happen before or after the other. Deadlock freedom is not enough; a
hybrid final state assembled from two checkpoints is a corruption that no
per-component guarantee can see.

So every participating replacement runs under **one** guard, and the
resulting execution has a valid serial order: after two concurrent
operations finish, the complete live state equals one of them followed by
the other, never a mixture.

The universal state-replacement lock order
------------------------------------------

1. **this guard**, always first;
2. then, when generators are involved, every unique target's lock in the
   existing global ``id()``-sorted order (§9.6);
3. nothing acquires them in the opposite order — ever.

Generator **reservations** deliberately do not participate: ``_reserve_call``
takes only its own generator's lock and never this guard. That is what
keeps the two systems from inverting. A reservation racing a transaction
therefore has exactly two outcomes — it wins its generator's lock first
and completes before the transaction can take it, or it waits and begins
after the transaction has released it — so no state replacement ever
happens underneath a live token.

Why a global ``RLock``
----------------------

- **Reentrancy is required, not incidental.** The checkpoint transaction
  holds the guard and then calls the components' own public loaders,
  each of which takes it again. A plain ``Lock`` would self-deadlock on
  the first nested call.
- **One universal outer order.** A per-model or per-object lock would
  need a registry, a lifetime, and an ordering rule between unrelated
  models — and two transactions whose target sets overlap partially are
  exactly the case that deadlocks. One process-wide lock has one order
  by construction.
- **Correctness over unrelated-model parallelism.** The critical sections
  are state replacement and snapshotting, not training. Serializing them
  costs nothing a native training loop notices, and the alternative is a
  class of bug that only appears under load and cannot be reproduced.

**What this does not claim.** It serializes *state replacement and
checkpoint snapshotting*, and nothing else. Ordinary training mutation —
an optimizer ``step()``, a ``copy_value_``, a backward accumulating
gradients — does **not** take this guard, so concurrent training against
a model being checkpointed is still not a supported thing to do. The
guarantee is precisely: participating operations serialize with respect
to each other.

One nuance, stated rather than left to be discovered: a BatchNorm
training forward commits its running statistics through
``replace_native_state``, so that particular training-time mutation *does*
participate and cannot tear a concurrent save. That falls out of routing
every registered-state replacement through one primitive; it does not
widen the claim, because ``step()``, ``copy_value_``, and gradient
accumulation still do not participate.
"""

import threading

# The single process-wide guard. Module-level and never rebound: rebinding
# it would silently split the world into two lock domains, which is the
# one failure this module exists to prevent.
_STATE_TRANSACTION_LOCK = threading.RLock()


def state_transaction():
    """The shared guard, as a context manager.

    ``with state_transaction():`` is the only intended way in. Reentrant,
    so a transaction that calls a component's own loader (which takes it
    again) proceeds instead of deadlocking."""
    return _STATE_TRANSACTION_LOCK


def held_by_current_thread():
    """Whether the calling thread already holds the guard.

    Diagnostics and guardrail tests only — production code never branches
    on it. It exists so "no participating path takes a generator lock
    before this one" can be asserted from live behavior rather than
    argued from source."""
    return _STATE_TRANSACTION_LOCK._is_owned()
