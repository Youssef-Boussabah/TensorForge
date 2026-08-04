"""NativeBatchSampler — the deterministic native batch planner (Phase J,
milestone J2; see docs/native_data_pipeline_design.md §3.4, §7, §8, §11.2,
§12.3, §12.4, §15.2, and §16), carrying the **integer half** of J3's
batch-delivery transaction (§9.4).

The second piece of Phase-J runtime, and a **planner and state holder**,
not an iterator. It answers *"which indices, in which groups, in which
order"* and nothing else: it materializes no batch, allocates nothing
native, constructs no ``NativeTensor``, and has no ``__iter__``,
``__next__``, or public advance of any kind. Iteration and batch delivery
belong to ``NativeDataLoader`` (J3), which reaches the position **only**
through the private transaction primitives below.

The three properties that shape every line below
-------------------------------------------------

**1. There is no consumable stream, by construction.** A permutation is a
**pure function** of ``(seed, epoch, length)`` (§7.7), so there is no
reservation, no counter of draws, no partially consumed sequence, and
nothing to roll back. ``epoch_permutation()``, ``plan()``, and
``next_batch_indices()`` may be called any number of times, in any order,
at any epoch, and consume nothing — which is *why* a rejected state load
and (at J3) an abandoned iterator or a failed delivery can consume
nothing either. The only state that ever moves is the ``(seed, shuffle,
batch_size, drop_last, epoch, cursor)`` tuple.

**2. It owns nothing releasable, so it has no ``close()``.** Six
configuration-and-position scalars, one reference to its dataset, an
optional private tuple cache, and — while a J3 batch handoff is in
flight — the **integer half** of one delivery record: a serial, an owning
iterator token, two position tuples, and the claimed indices. No native
storage, no host snapshot, no batch, no tensor, no file, no generator, no
thread, no lock, no worker, and no queue: **the feature tensor and the
target array are the iterator's**, which is exactly why the sampler still
owns nothing releasable. Inventing a ``close()`` would advertise a lifetime this
object does not have — ``NativeGenerator``'s stated reason, applied
rather than replaced (§15.1). A sampler is *always* usable; there is no
closed-sampler state to reason about and no lifecycle question in its
state schema. **A closed dataset changes none of that**: planning needs
only ``samples`` and the identity metadata, both of which survive
``NativeTensorDataset.close()``.

**3. It is not coupled to a live ``NativeGenerator``, deliberately**
(§8.3). It is not one, does not hold one, does not accept one, creates
none, and consumes no call from one. Three independent reasons: coupling
would entangle the data order with Dropout's stream, so changing the
batch size would silently change every mask; a permutation is indexed by
an **epoch**, not by a monotonic call count, and a restored sampler must
reproduce epoch 9's order without having consumed epochs 0-8; and
``NativeGenerator`` exposes no bit derivation at all, so coupling would
require inventing a public random surface. It holds a plain ``seed``
integer in the **same** unsigned 64-bit domain, through the **same**
validator, so the phase invents no second seed contract — that shared
import is a *validation rule*, and carries no generator, no state, and no
stream with it.

Position, and who may move it
-----------------------------

``epoch`` is the **active** epoch — the one whose permutation is being
consumed — and ``cursor`` is the number of batches **already
successfully delivered** in it, always in ``[0, batches_per_epoch)``.
The canonical transition, committed once per delivered batch, is
``cursor += 1``, and then ``epoch += 1; cursor = 0`` when the cursor
reaches the batch count — canonicalized *immediately*, so every position
has exactly one representation and two runs that consumed the same number
of batches always have byte-identical state (§7.4).

**No public method here applies it.** A position other than ``(0, 0)``
arrives through exactly two audited paths: a validated
``load_state_dict``, and the loader's **successful** batch delivery.
``_next_position`` computes the candidate without mutating anything, and
``_assign_state`` is the single non-failing write seam — used in **both**
directions, to commit a delivery and to roll one back, which is what
makes a rollback structurally unable to fail.

**No public advance exists**, and none may be added: a caller who could
move the cursor could desynchronize an active iteration or strand a
pending delivery. ``_claim_batch``, ``_publish_pending``,
``_commit_pending``, ``_rollback_pending``, and ``_complete_pending`` are
private, are reachable only from the loader's iterator, and every one of
them matches on the transaction's **never-reused serial** *and* its
owning iterator token, so a stale, foreign, completed, or already
rolled-back record is left strictly alone.

Adds no kernel, C ABI symbol, ctypes declaration, checkpoint field or
version, optimizer-state version, capability registry value, dtype,
device, or dependency. It has no ``dtype`` argument: it owns no
dtype-bearing numeric state, and a second authority on the dataset's
dtype could disagree with the data.
"""

from . import _native_permutation as _perm
# The seed's domain and its validation, **shared rather than restated**
# (§8.3): a duplicated validator would be a second seed contract that
# could drift. This imports one pure integer check and a bound — no
# generator is imported, held, created, accepted, or consulted anywhere in
# this module.
from .native_generator import UINT64_MAX, _validate_uint64
from .native_dataset import NativeTensorDataset

# The state schema's identity (§11.2). The format tag is what makes a
# sampler state distinguishable from any other JSON object a caller might
# hand back — including J4's loader state, which wraps this one under its
# own tag precisely so the two cannot be confused. This schema and this
# version do not move when the wrapper lands around them.
_FORMAT = "tensorforge.native_sampler"
_FORMAT_VERSION = 1
_SUPPORTED_FORMAT_VERSIONS = (1,)

# The exact key sets. Validation is by exact **set** equality in both
# directions, so a missing key is not a default and an unknown key is not
# ignored. Nothing depends on the emission order below.
_STATE_FIELDS = ("format", "format_version", "dataset", "seed", "shuffle",
                 "batch_size", "drop_last", "epoch", "cursor")
_DATASET_FIELDS = ("samples", "feature_shape", "feature_dtype", "fingerprint")

# The dtype names a dataset identity may carry. Deliberately a literal
# pair rather than a read of ``cpp.SUPPORTED_DTYPES``: this is a *schema*
# statement about what a version-1 sampler state may spell, and it must
# not silently widen the day a registry does. Widening it is a milestone
# decision, exactly as widening the registry is.
_IDENTITY_DTYPES = ("float64", "float32")

# A fingerprint is 64 lowercase hexadecimal characters (§6.3). Checked as
# a set membership rather than with ``int(value, 16)``, which would accept
# uppercase, a leading sign, and underscores.
_FINGERPRINT_LENGTH = 64
_HEX_DIGITS = frozenset("0123456789abcdef")

# The three phases one batch-delivery record passes through (§9.4). They are
# ordinary private strings: they are never serialized, never appear in
# ``state_dict()``, never appear in ``__repr__`` of the sampler, and carry no
# C ABI meaning.
_CLAIMED = "claim"          # phase 1 done; nothing constructed, nothing moved
_PENDING = "pending"        # phase 3 done; both batches exist, nothing moved
_COMMITTED = "committed"    # phase 4 step 1 done; the candidate is applied

# The value that is never a live serial, so "no transaction" needs no second
# flag and a cleanup called with it can match nothing.
_NO_SERIAL = 0


class _BatchTransaction:
    """The **integer half** of one in-flight batch handoff (§9.4, Phase 3).

    The record is split across two owners by what each *owns*: the sampler
    keeps this half — the serial, the owning iterator's token, the exact
    pre-delivery position, the candidate post-delivery position, and the
    claimed indices — because that is what its own ``state_dict()`` and
    ``load_state_dict()`` guards must be able to see. The **iterator**
    keeps the resource half, because §15.1 puts an owned resource on the
    object whose ``close()`` releases it, and a sampler that transiently
    owned native storage would contradict §15.2.

    Nothing here is releasable, nothing here is state, and nothing here is
    public: it is never serialized, never compared, never in a repr of the
    sampler, and never reachable from a public attribute.
    """

    __slots__ = ("serial", "owner", "status", "before", "after", "indices")

    def __init__(self, serial, owner, before, after, indices):
        self.serial = serial
        # The owning iterator's participation token — a process-local
        # integer minted by ``_begin_iteration`` and never reused. It is
        # internal ownership matching only: it is not an ``id()``, it never
        # leaves the process, and it never enters state.
        self.owner = owner
        self.status = _CLAIMED
        self.before = before
        self.after = after
        self.indices = indices

    def __repr__(self):
        return (f"<NativeBatchSampler batch transaction {self.serial} "
                f"({self.status})>")


def _require_exact_int(value, what):
    """Reject anything that is not exactly a Python ``int``.

    ``bool`` is not an ``int`` here and a NumPy integer scalar is not one
    either — the exact-type discipline ``_validate_uint64``,
    ``_validated_metadata``, and the J1 dataset all use. ``True`` is not
    ``1``, ``1.0`` is not ``1``, and ``"1"`` is not ``1``; each is a
    ``TypeError`` rather than a conversion (§11.5).
    """
    if type(value) is not int:
        raise TypeError(f"{what} must be an int, got {type(value).__name__}")
    return value


def _require_exact_bool(value, what):
    """Reject anything that is not exactly a Python ``bool``.

    ``0``, ``1``, ``""``, ``None``, and a NumPy ``bool_`` are each a
    ``TypeError``. Accepting a truthy value here would let ``shuffle=1``
    and ``shuffle=True`` mean the same thing in a *state dict*, and then a
    round trip through JSON could not be proved lossless.
    """
    if type(value) is not bool:
        raise TypeError(f"{what} must be a bool, got {type(value).__name__}")
    return value


def _require_exact_keys(mapping, expected, what):
    """Exact key-set equality in both directions, naming what is missing
    and what is unexpected in a deterministic sorted order.

    Module-level rather than a method so the dataset block and the root
    object share one rule; no exact message text is a contract.
    """
    keys = set(mapping.keys())
    wanted = set(expected)
    if keys != wanted:
        missing = sorted(wanted - keys)
        unexpected = sorted(str(key) for key in keys - wanted)
        raise ValueError(
            f"{what} must have exactly the keys {list(expected)}: missing "
            f"{missing}, unexpected {unexpected}"
        )


class NativeBatchSampler:
    """Deterministic sample order and batch planning over a dataset.

    ``NativeBatchSampler(dataset, *, batch_size, shuffle=False, seed=0,
    drop_last=False)`` — ``dataset`` must be a ``NativeTensorDataset``
    (a **closed** one is fine); ``batch_size`` is required, keyword-only,
    an exact ``int`` at least 1; ``shuffle`` and ``drop_last`` are exact
    ``bool``s; ``seed`` is an exact ``int`` in ``[0, 2**64 - 1]``.
    ``drop_last=True`` with ``batch_size > dataset.samples`` is refused,
    because dropping the only partial batch would leave no batches at all.

    A new sampler starts at ``epoch == 0``, ``cursor == 0``; neither is a
    constructor argument. ``epoch_permutation()``, ``plan()``, and
    ``next_batch_indices()`` are pure and change nothing.

    See the module docstring for the ownership, position, and
    no-consumption contracts.
    """

    # No ``__dict__``: the configuration and position are read-only
    # properties, so ``sampler.cursor = 3`` raises rather than shadowing
    # one, and no attribute can be injected onto a sampler. There is
    # deliberately **no ``_closed``** — a sampler owns nothing releasable
    # (§15.2).
    __slots__ = ("_dataset", "_batch_size", "_shuffle", "_seed", "_drop_last",
                 "_epoch", "_cursor", "_cache_key", "_cache_order",
                 # J3's bookkeeping. ``_transaction`` is the integer half of
                 # at most one in-flight batch handoff; ``_active_iterations``
                 # is the set of live iterator tokens behind the
                 # ``load_state_dict`` refusal. Neither is state: neither is
                 # serialized, compared, or reported anywhere, and neither
                 # holds a native resource — so the sampler still needs no
                 # ``close()``.
                 "_transaction", "_next_serial",
                 "_active_iterations", "_next_iteration_token")

    def __init__(self, dataset, *, batch_size, shuffle=False, seed=0,
                 drop_last=False):
        # --- §12.3 validation order. Every individual configuration field
        # is checked before the joint rule, so a caller who passed
        # ``batch_size=0`` is told *that*, not told about ``drop_last``.

        # 1. The dataset. ``isinstance`` rather than an exact type check:
        #    unlike the J1 input contract — where a subclass could make a
        #    gather mean something other than a gather — the sampler reads
        #    only ``samples`` and ``identity()``, which any subclass
        #    inherits intact. A **closed** dataset is accepted on purpose:
        #    ``samples`` and the identity metadata survive close (§5.5),
        #    the sampler needs nothing else, and refusing here would
        #    invent a lifecycle rule with no purpose.
        if not isinstance(dataset, NativeTensorDataset):
            raise TypeError(
                f"dataset must be a NativeTensorDataset, got "
                f"{type(dataset).__name__}"
            )
        # 2. The batch size: type (bool first), then range. No upper bound
        #    is imposed — ``batch_size > samples`` with ``drop_last=False``
        #    is legal and simply gives one short batch per epoch. The
        #    platform limit is reached where the *native batch* is
        #    allocated, and ``MemoryError`` is the honest answer there;
        #    inventing a threshold here would be a performance control,
        #    which this project does not have.
        _require_exact_int(batch_size, "batch_size")
        if batch_size < 1:
            raise ValueError(f"batch_size must be at least 1, got {batch_size}")
        # 3. shuffle, exactly a bool.
        _require_exact_bool(shuffle, "shuffle")
        # 4. The seed: the identical domain and the identical validation
        #    NativeGenerator uses, through the identical function.
        _validate_uint64(seed, "seed")
        # 5. drop_last, exactly a bool.
        _require_exact_bool(drop_last, "drop_last")
        # 6. The §7.5 joint rule, last, because it is the only one that
        #    depends on two fields and the dataset at once. A zero-batch
        #    epoch would break the §7.4 transition outright: no batch is
        #    ever delivered, so the epoch never advances, so iteration
        #    would spin forever on a position that cannot move. Refusing
        #    it here makes ``batches_per_epoch >= 1`` an invariant and
        #    lets every rule downstream be total.
        samples = dataset.samples
        if drop_last and batch_size > samples:
            raise ValueError(
                f"batch_size {batch_size} exceeds the dataset's {samples} "
                f"samples with drop_last=True: dropping the only partial "
                f"batch would leave no batches at all in an epoch"
            )

        # --- Publish. Plain assignments that cannot fail, so no partially
        # initialized sampler is ever observable. The dataset is stored by
        # **identity**: not copied, cloned, wrapped, reconstructed, or
        # replaced, here or by ``load_state_dict``.
        self._dataset = dataset
        self._batch_size = batch_size
        self._shuffle = shuffle
        self._seed = seed
        self._drop_last = drop_last
        # ``epoch`` and ``cursor`` are not constructor arguments (§7.2):
        # every other position arrives through ``load_state_dict``, which
        # is the one audited path into one.
        self._epoch = 0
        self._cursor = 0
        # The permutation cache (§7.8). Not state, never serialized, never
        # compared, and droppable at any moment without observable change.
        self._cache_key = None
        self._cache_order = None
        # The J3 transaction bookkeeping, all inert until a loader iterates.
        # Serials and iteration tokens start at 1 and only ever increase, so
        # ``_NO_SERIAL`` can never name a live record and a released token
        # can never be handed out again.
        self._transaction = None
        self._next_serial = 1
        self._active_iterations = set()
        self._next_iteration_token = 1

    # -- configuration and position (read-only) ------------------------

    @property
    def dataset(self):
        """The dataset this sampler was constructed with — **identity**,
        not a copy. ``load_state_dict`` never replaces it."""
        return self._dataset

    @property
    def batch_size(self):
        """Batches this sampler emits are this many indices, except a
        final short one when ``drop_last`` is false. May be **replaced**
        by ``load_state_dict`` (§12.4)."""
        return self._batch_size

    @property
    def shuffle(self):
        """Whether each epoch takes a derived permutation or the identity
        order. May be replaced by ``load_state_dict``."""
        return self._shuffle

    @property
    def seed(self):
        """The unsigned 64-bit permutation seed, an exact Python int. It
        is **not** a ``NativeGenerator``, and drives no other stream."""
        return self._seed

    @property
    def drop_last(self):
        """Whether an epoch's final short batch is omitted."""
        return self._drop_last

    @property
    def epoch(self):
        """The **active** epoch — the one whose permutation is currently
        being consumed, not the next one. A fresh sampler reads 0, and its
        first batch comes from epoch 0's order."""
        return self._epoch

    @property
    def cursor(self):
        """Batches already **successfully delivered** in the active epoch;
        always ``0 <= cursor < batches_per_epoch``. No public method
        advances it: it moves only when a ``NativeDataLoader`` iterator
        hands a batch to its caller, or through a validated state load."""
        return self._cursor

    @property
    def batches_per_epoch(self):
        """Batches in one epoch; always ``>= 1``.

        ``samples // batch_size`` with ``drop_last``, otherwise
        ``ceil(samples / batch_size)``, in integer arithmetic. It depends
        only on ``(samples, batch_size, drop_last)``, so it is the same
        for every epoch — which is what lets the cursor mean one thing.
        """
        return _perm.batches_per_epoch(self._dataset.samples,
                                       self._batch_size, self._drop_last)

    @property
    def remaining(self):
        """Batches left in the active epoch: ``batches_per_epoch -
        cursor``, always in ``[1, batches_per_epoch]``, because a
        canonical position never stores ``cursor == batches_per_epoch``.
        A fresh sampler reads the whole epoch."""
        return self.batches_per_epoch - self._cursor

    # -- planning (all pure) -------------------------------------------

    def _resolved_epoch(self, epoch):
        """``self.epoch`` for ``None``, or a validated explicit epoch.

        An explicit epoch is held to exactly the domain the stored one is:
        an exact ``int`` (``bool`` rejected) in ``[0, 2**64 - 1]``. A
        rejected argument mutates nothing at all — the caller simply gets
        an exception and the sampler is untouched.
        """
        if epoch is None:
            return self._epoch
        return _validate_uint64(epoch, "epoch")

    def _sample_order(self, epoch):
        """The already-validated epoch's full order, as a tuple of ints.

        Sequential order is returned **directly**, with no cache and no
        derivation: ``shuffle=False`` is a different, cheaper branch, not
        a shuffle with a fixed seed.

        The cache (§7.8) holds only the **active** epoch's permutation,
        keyed on ``(seed, epoch, samples)``. That is the only epoch a
        batch is ever taken from, so it is the only one worth keeping, and
        it makes "an arbitrary-epoch inspection never touches the cache" a
        property rather than a hope. The value is a pure function of the
        key, so dropping the cache at any moment changes nothing.
        """
        samples = self._dataset.samples
        if not self._shuffle:
            return tuple(range(samples))
        key = (self._seed, epoch, samples)
        if epoch != self._epoch:
            return _perm.permutation(self._seed, epoch, samples)
        if self._cache_key != key:
            self._cache_order = _perm.permutation(self._seed, epoch, samples)
            self._cache_key = key
        return self._cache_order

    def epoch_permutation(self, epoch=None):
        """One epoch's complete sample order, as a ``tuple`` of ``int``.

        ``epoch=None`` means the active epoch. The result has exactly
        ``dataset.samples`` entries and contains every index exactly once;
        with ``shuffle=False`` it is the identity order at every seed and
        every epoch.

        **Pure.** It changes no field, consumes no draw, and may be called
        any number of times in any order, at the current epoch or an
        arbitrary one. The returned tuple is safe to share and to keep: a
        tuple of ints is immutable, so a caller cannot reach the sampler's
        order through it.
        """
        return self._sample_order(self._resolved_epoch(epoch))

    def plan(self, epoch=None):
        """One epoch's complete batch plan: a tuple of index tuples.

        Exactly ``batches_per_epoch`` groups, batch ``k`` being the slice
        ``[k * batch_size, (k + 1) * batch_size)`` of that epoch's order.
        With ``drop_last=False`` the last group may be short; with
        ``drop_last=True`` the tail is not emitted. No index appears
        twice, because an order is a permutation and a batch is a
        contiguous slice of it.

        **Pure**, at the current epoch and at an arbitrary one: no state
        moves, no draw is consumed, nothing native is allocated, and no
        NumPy array or list appears anywhere in the result.
        """
        order = self._sample_order(self._resolved_epoch(epoch))
        batch_size = self._batch_size
        count = _perm.batches_per_epoch(len(order), batch_size,
                                        self._drop_last)
        return tuple(order[k * batch_size:(k + 1) * batch_size]
                     for k in range(count))

    def next_batch_indices(self):
        """The indices the next batch will use, as a ``tuple`` of ``int``.

        Exactly ``plan(self.epoch)[self.cursor]``, computed without
        building the rest of the plan.

        **Pure, and it does not consume.** It advances no cursor, no
        epoch, and no draw stream; repeated calls return the same tuple
        until the state actually changes. It is valid at every canonical
        position, and works with a **closed** dataset, because it needs
        only the surviving ``samples`` metadata — materializing the batch
        is the loader's job (J3), and it is the only step a closed dataset
        refuses.
        """
        order = self._sample_order(self._epoch)
        start = self._cursor * self._batch_size
        # Python slicing truncates, which *is* the specified behavior for
        # a short final batch; with ``drop_last`` the cursor can never
        # reach a short slice, so the same expression covers both.
        return order[start:start + self._batch_size]

    # -- state ---------------------------------------------------------

    def state_dict(self):
        """A fresh, plain, JSON-compatible snapshot of §11.2's schema.

        Every container is new at every call — the root dict, the nested
        ``dataset`` dict, and the ``feature_shape`` list — so the result
        shares nothing mutable with the sampler, the dataset, the cache,
        or a previous result, and a caller may edit what they are given
        without reaching anything.

        It describes the position after the last successfully delivered
        batch, which is always the **exact next batch**. It carries no
        permutation, no dataset content, no NumPy object, no bytes, no
        object id, and nothing executable: the permutation is a pure
        function of ``(seed, epoch, samples)``, all three of which are
        already here, so serializing an array the size of the dataset
        would carry information the eight bytes of ``seed`` already carry
        exactly.

        Pure, allocation-free natively, and available with a **closed**
        dataset. Every field passes the checkpoint's
        ``_validated_metadata`` unchanged, which is what lets a caller
        carry it through the existing version-3 metadata channel without
        the archive growing a field.

        **Refused with ``RuntimeError`` while a §9.4 batch transaction is
        in flight**, reading nothing and changing nothing. Inside Phase 4
        the candidate position has been applied but the batch has not been
        delivered, so there is no honest single answer: reporting the
        candidate would expose a committed cursor that skipped an
        undelivered batch, and reporting the pre-delivery one would
        contradict the object's own fields. **A snapshot must never be
        able to observe a skipped-but-undelivered position**, so the
        ambiguous window is refused rather than captured — exactly
        ``snapshot_generator_states``' rule. Every other time, including
        between batches while an iterator is active, it is allowed.
        """
        self._require_no_transaction("take a state snapshot of this sampler")
        return {
            "format": _FORMAT,
            "format_version": _FORMAT_VERSION,
            # ``identity()`` already returns a fresh dict with a fresh
            # ``feature_shape`` **list**, which is what a JSON round trip
            # returns, so a saved-and-reloaded state compares equal to a
            # freshly produced one without normalization.
            "dataset": self._dataset.identity(),
            "seed": self._seed,
            "shuffle": self._shuffle,
            "batch_size": self._batch_size,
            "drop_last": self._drop_last,
            "epoch": self._epoch,
            "cursor": self._cursor,
        }

    def load_state_dict(self, state):
        """Restore configuration and position from a §11.2 state. Returns
        ``None``.

        **Transactional.** Everything is validated first, against the
        live dataset and against the schema, and only then are six
        already-validated ``int``s and ``bool``s assigned. The commit
        therefore **cannot fail**, which is what makes the transaction
        exact without a rollback path — ``NativeGenerator._assign_state``'s
        property, deliberately reproduced. A rejected state leaves every
        observable property, the position, and the cached behavior exactly
        as they were.

        **Dataset identity is validated, never adopted**, and the sampler
        keeps the exact dataset object it already had. **Configuration is
        adopted**: ``seed``, ``shuffle``, ``batch_size``, ``drop_last``,
        ``epoch``, and ``cursor`` all come from the state, so a restored
        sampler may legitimately report a different ``batch_size`` than
        its constructor was given — exactly as a restored ``NativeAdam``
        may report a different ``lr`` (§2.7, §12.4). Object identity is
        preserved absolutely: nothing is replaced, rebound, or recreated.

        Nothing is cast, coerced, truncated, clamped, wrapped, rounded,
        defaulted, or ignored; every one of those is a rejection naming
        the field.

        Two lifecycle guards run **before** ``state`` is inspected at all,
        in §12.4's order. A live §9.4 transaction is refused first,
        because replacing a position underneath one would make its
        pre/post pair describe a stream that no longer exists. An **active
        iteration** is refused next, because an iterator captures its
        epoch's remaining batch count when it is created (§9.3), so a load
        would leave that countdown describing a position that is gone.
        Neither reads the argument, so a malformed state cannot even be
        parsed while either holds.
        """
        # 1. Transaction guard, first of all — ahead of the container check
        #    and ahead of reading a single field of ``state``.
        self._require_no_transaction("load state into this sampler")
        # 2. Active-iteration guard, next.
        self._require_no_active_iteration("load state into this sampler")
        # 3 onward: J2's validation, unchanged.
        values = self._validate_state(state)
        self._assign_state(*values)
        return None

    # -- private state helpers -----------------------------------------
    #
    # Private, unexported, and not a second public state API. The split
    # exists because J3's loader must be able to validate this object's
    # inner state *without* mutating it and then commit through the same
    # non-failing seam the delivery transaction uses in both directions —
    # so a rollback and a load share one write path rather than two
    # spellings of one idea.

    def _validate_state(self, state):
        """Validate a candidate state completely and return the six values
        it carries. **Mutates nothing**, reads no field it does not
        compare, and allocates nothing native.

        The order is §12.4's exactly: container, key set, format, version,
        the dataset block's shape, the dataset block against **live**
        reality, then the configuration's types, its ranges, the
        zero-batch joint rule, and the cursor last — because the cursor is
        the only rule that depends on several other fields being valid
        first.
        """
        # 3. Container.
        if type(state) is not dict:
            raise TypeError(
                f"sampler state must be a dict, got {type(state).__name__}"
            )
        # 4. Exact root key set, in both directions: a missing key is not a
        #    default and an unknown key is not ignored. Sorted so the
        #    message is deterministic.
        _require_exact_keys(state, _STATE_FIELDS, "sampler state")
        # 5/6. The format tag, then the version. The tag is what
        #    distinguishes this from any other JSON object — including
        #    J4's loader state, which carries its own.
        format_tag = state["format"]
        if type(format_tag) is not str:
            raise TypeError(
                f"sampler state 'format' must be a str, got "
                f"{type(format_tag).__name__}"
            )
        if format_tag != _FORMAT:
            raise ValueError(
                f"sampler state format mismatch: expected {_FORMAT!r}, got "
                f"{format_tag!r}"
            )
        version = _require_exact_int(state["format_version"],
                                     "sampler state 'format_version'")
        if version not in _SUPPORTED_FORMAT_VERSIONS:
            raise ValueError(
                f"unsupported sampler state format version {version}; "
                f"supported: {list(_SUPPORTED_FORMAT_VERSIONS)}"
            )
        # 7-11. The dataset identity block: its own shape first, then its
        #    values against live reality.
        self._validate_dataset_identity(state["dataset"])
        # 12. Configuration **types**, in schema order, all before any
        #    range check — so a state with a float seed and an out-of-range
        #    epoch reports the more basic fault.
        seed = _require_exact_int(state["seed"], "sampler state 'seed'")
        shuffle = _require_exact_bool(state["shuffle"],
                                      "sampler state 'shuffle'")
        batch_size = _require_exact_int(state["batch_size"],
                                        "sampler state 'batch_size'")
        drop_last = _require_exact_bool(state["drop_last"],
                                        "sampler state 'drop_last'")
        epoch = _require_exact_int(state["epoch"], "sampler state 'epoch'")
        cursor = _require_exact_int(state["cursor"], "sampler state 'cursor'")
        # 13. Configuration ranges: seed, batch size, epoch.
        if not 0 <= seed <= UINT64_MAX:
            raise ValueError(
                f"sampler state 'seed' must be in [0, {UINT64_MAX}], got {seed}"
            )
        if batch_size < 1:
            raise ValueError(
                f"sampler state 'batch_size' must be at least 1, got "
                f"{batch_size}"
            )
        if not 0 <= epoch <= UINT64_MAX:
            raise ValueError(
                f"sampler state 'epoch' must be in [0, {UINT64_MAX}], got "
                f"{epoch}"
            )
        # 14. The §7.5 joint rule, against the **state's own** batch size
        #    and drop-last and the live sample count. A state that would
        #    produce zero batches is refused here, before anything moves,
        #    for the same reason the constructor refuses one.
        samples = self._dataset.samples
        if drop_last and batch_size > samples:
            raise ValueError(
                f"sampler state 'batch_size' {batch_size} exceeds the "
                f"dataset's {samples} samples with drop_last=True: the "
                f"restored configuration would leave no batches in an epoch"
            )
        # 15. The cursor, against the batch count the *candidate*
        #     configuration implies. Half-open, with no special case at the
        #     top: a cursor equal to the batch count is unambiguously
        #     invalid rather than ambiguously terminal, and is never
        #     clamped to one that is valid.
        count = _perm.batches_per_epoch(samples, batch_size, drop_last)
        if not 0 <= cursor < count:
            raise ValueError(
                f"sampler state 'cursor' must be in [0, {count}), got "
                f"{cursor}"
            )
        return seed, shuffle, batch_size, drop_last, epoch, cursor

    def _validate_dataset_identity(self, identity):
        """The state's dataset block: shape, then compatibility.

        Structural facts are compared against the **live** dataset and are
        never adopted — ``NativeAdam.load_state_dict``'s rule, applied
        unchanged. The order is ``samples`` → ``feature_shape`` →
        ``feature_dtype`` → ``fingerprint``: structural first, digest
        last, because a structural mismatch has an understandable message
        while "the fingerprints differ" is only useful once the shapes
        agree.
        """
        # 7/8. Container and exact key set.
        if type(identity) is not dict:
            raise TypeError(
                f"sampler state 'dataset' must be a dict, got "
                f"{type(identity).__name__}"
            )
        _require_exact_keys(identity, _DATASET_FIELDS,
                            "sampler state 'dataset'")
        # 9. Field types. ``feature_shape`` accepts a tuple as well as a
        #    list, following the v3.13 precedent that a caller may
        #    legitimately have rebuilt the container; ``state_dict()``
        #    always emits a list, which is what JSON returns.
        samples = _require_exact_int(identity["samples"],
                                     "sampler state 'dataset.samples'")
        shape = identity["feature_shape"]
        if type(shape) is not list and type(shape) is not tuple:
            raise TypeError(
                f"sampler state 'dataset.feature_shape' must be a list or "
                f"tuple, got {type(shape).__name__}"
            )
        dtype = identity["feature_dtype"]
        if type(dtype) is not str:
            raise TypeError(
                f"sampler state 'dataset.feature_dtype' must be a str, got "
                f"{type(dtype).__name__}"
            )
        fingerprint = identity["fingerprint"]
        if type(fingerprint) is not str:
            raise TypeError(
                f"sampler state 'dataset.fingerprint' must be a str, got "
                f"{type(fingerprint).__name__}"
            )
        # 10. Ranges, and the per-element shape rules.
        if samples < 1:
            raise ValueError(
                f"sampler state 'dataset.samples' must be at least 1, got "
                f"{samples}"
            )
        for axis, dimension in enumerate(shape):
            _require_exact_int(
                dimension,
                f"sampler state 'dataset.feature_shape[{axis}]'"
            )
            if dimension < 1:
                raise ValueError(
                    f"sampler state 'dataset.feature_shape[{axis}]' must be "
                    f"at least 1, got {dimension}"
                )
        if dtype not in _IDENTITY_DTYPES:
            raise ValueError(
                f"sampler state 'dataset.feature_dtype' must be one of "
                f"{list(_IDENTITY_DTYPES)}, got {dtype!r}"
            )
        if (len(fingerprint) != _FINGERPRINT_LENGTH
                or not _HEX_DIGITS.issuperset(fingerprint)):
            raise ValueError(
                f"sampler state 'dataset.fingerprint' must be exactly "
                f"{_FINGERPRINT_LENGTH} lowercase hexadecimal characters, "
                f"got a {len(fingerprint)}-character string"
            )
        # 11. Compatibility against live reality, structural first.
        live = self._dataset
        if samples != live.samples:
            raise ValueError(
                f"sampler state was written for a dataset of {samples} "
                f"samples; this dataset has {live.samples}"
            )
        if tuple(shape) != live.feature_shape:
            raise ValueError(
                f"sampler state was written for per-sample shape "
                f"{tuple(shape)}; this dataset has {live.feature_shape}"
            )
        if dtype != live.dtype:
            raise ValueError(
                f"sampler state was written for feature dtype {dtype!r}; "
                f"this dataset has {live.dtype!r}"
            )
        if fingerprint != live.fingerprint:
            raise ValueError(
                "sampler state was written for a different dataset: the "
                "content fingerprints differ, so the stored position would "
                "describe rows this dataset does not hold"
            )

    def _assign_state(self, seed, shuffle, batch_size, drop_last, epoch,
                      cursor):
        """Write six already-validated values. **Non-failing by
        construction**, and the single write seam.

        Six ``__slots__`` assignments of ``int``s and ``bool``s the caller
        has already validated, plus a cache invalidation that cannot fail
        either. That is what makes ``load_state_dict`` exact without a
        rollback path, and it is what J3 will use in **both** directions —
        applying a candidate position at commit and restoring the
        pre-delivery one at rollback — so a rollback that could itself
        raise is structurally impossible.
        """
        self._seed = seed
        self._shuffle = shuffle
        self._batch_size = batch_size
        self._drop_last = drop_last
        self._epoch = epoch
        self._cursor = cursor
        # The cache is not state, so this is bookkeeping rather than part
        # of the transaction — but it must happen at the commit, because a
        # changed seed or epoch would otherwise leave a stale order
        # readable through a key that no longer describes it.
        self._cache_key = None
        self._cache_order = None

    def _snapshot_state(self):
        """The six values ``_assign_state`` would restore. Private, pure,
        and the rollback half of J3's delivery transaction."""
        return (self._seed, self._shuffle, self._batch_size, self._drop_last,
                self._epoch, self._cursor)

    def _next_position(self, epoch, cursor):
        """The committed position §7.4 reaches **after** one more batch is
        successfully delivered from ``(epoch, cursor)``.

        Pure: it computes a candidate and **mutates nothing**. J2 has no
        delivery and calls it nowhere in a public path; it exists so that
        J3 applies the canonical rule through this object rather than
        redesigning the position semantics, and so that the rule is
        testable now.

        The epoch boundary is canonicalized **immediately** — the moment
        the last batch of an epoch is delivered, not lazily on the next
        request — so every position has exactly one representation and the
        cursor range stays the half-open ``[0, batches_per_epoch)``.

        ``epoch`` is bounded by the same unsigned 64-bit domain as the
        seed. An advance past ``2**64 - 1`` raises ``RuntimeError`` and
        moves nothing, exactly as ``NativeGenerator`` refuses at an
        exhausted counter. Unreachable in practice, and specified so that
        it is not undefined. Because ``_claim_batch`` calls it **before**
        it mints a serial or publishes anything, that refusal leaves no
        claim, allocates no batch, and permits no wrapped epoch.
        """
        cursor += 1
        if cursor == _perm.batches_per_epoch(self._dataset.samples,
                                             self._batch_size,
                                             self._drop_last):
            if epoch >= UINT64_MAX:
                raise RuntimeError(
                    f"epoch would advance past {UINT64_MAX}, the unsigned "
                    f"64-bit domain it shares with the seed; nothing moved"
                )
            epoch += 1
            cursor = 0
        return epoch, cursor

    # -- active iterations (J3) -----------------------------------------
    #
    # Not public state, never serialized, never compared, and absent from
    # every property, repr, identity, and cache. Tokens are minted from a
    # monotonic counter and **never reused**, so a stale iterator cannot
    # release another iterator's participation and a released token can
    # never be handed out again. Nothing here allocates native storage and
    # nothing here takes a lock: this is reentrancy bookkeeping, not a
    # concurrency mechanism (§16.2).

    def _begin_iteration(self):
        """Register one live iterator and return its never-reused token."""
        token = self._next_iteration_token
        self._next_iteration_token = token + 1
        self._active_iterations.add(token)
        return token

    def _end_iteration(self, token):
        """Release exactly the participation ``token`` names, if it is still
        held.

        ``discard`` rather than ``remove``, so the release is idempotent:
        an iterator that exhausts, is closed, and is then finalized
        releases **once**, and the second and third calls find nothing.
        Matching on the token is what makes it exact — a stale iterator
        can never decrement a *different* iterator's participation.
        """
        self._active_iterations.discard(token)

    def _iteration_is_active(self):
        """Whether any iterator is live over this sampler."""
        return bool(self._active_iterations)

    def _has_transaction(self):
        """Whether a §9.4 claim or pending-delivery record is in flight."""
        return self._transaction is not None

    def _require_no_transaction(self, what):
        """Refuse an operation that cannot be answered honestly mid-handoff."""
        transaction = self._transaction
        if transaction is not None:
            raise RuntimeError(
                f"cannot {what} while batch-delivery transaction "
                f"{transaction.serial} is in flight ({transaction.status}): "
                f"the candidate position it carries has not been delivered, "
                f"so there is no single honest answer. Let the handoff "
                f"finish, or close the iterator to roll it back."
            )

    def _require_no_active_iteration(self, what):
        """Refuse a position replacement while a countdown depends on one."""
        active = len(self._active_iterations)
        if active:
            raise RuntimeError(
                f"cannot {what} while {active} batch iterator(s) are active "
                f"over it: an iterator captures its epoch's remaining batch "
                f"count when it is created, so replacing the position "
                f"underneath it would leave that countdown describing a "
                f"position that no longer exists. Close the iterator(s) "
                f"first."
            )

    # -- the batch-delivery transaction (J3) ----------------------------
    #
    # Five phases, and the integer half of each. Private, unexported, and
    # reachable only from the loader's iterator: a public advance would let
    # a caller desynchronize a live iteration or strand a pending delivery.
    #
    # Every one of the four resolution routes below is **exact-match**: it
    # acts only on a live record whose serial *and* owning token are this
    # transaction's, and whose status is the one it expects. A newer
    # transaction, a foreign iterator's transaction, a completed one, and
    # an already rolled-back one are each left strictly alone — which is
    # what makes a rollback idempotent, so a reentrant ``close()`` racing
    # the delivery's own ``finally`` cannot double-roll.

    def _matching_transaction(self, serial, owner, status):
        """The live record iff it is exactly this one; otherwise ``None``."""
        transaction = self._transaction
        if (transaction is None
                or transaction.serial != serial
                or transaction.owner != owner
                or (status is not None and transaction.status != status)):
            return None
        return transaction

    def _claim_batch(self, owner):
        """Phase 1: decide the next batch and publish **only the claim**.

        Returns ``(serial, indices)``. The committed epoch and cursor do
        **not** move here, and neither does anything else: ``before``,
        ``indices``, and ``after`` are all pure functions of committed
        state, so computing them mutates nothing.

        The claim is written **last**, so a failure anywhere before it —
        including the uint64 epoch-overflow refusal inside
        ``_next_position`` — leaves no record, mints no serial, and
        changes nothing at all.
        """
        # Reject another in-flight transaction, whoever owns it. This is
        # what makes a reentrant ``__next__`` (from a finalizer, a
        # callback, or a signal handler) a deterministic refusal rather
        # than two interleaved traversals over one cursor.
        self._require_no_transaction("claim a batch from this sampler")
        before = self._snapshot_state()
        indices = self.next_batch_indices()
        epoch, cursor = self._next_position(self._epoch, self._cursor)
        after = (self._seed, self._shuffle, self._batch_size,
                 self._drop_last, epoch, cursor)
        # A never-reused serial: it advances here, at the claim, so even a
        # discarded claim's serial is never handed out twice for the
        # lifetime of this sampler.
        serial = self._next_serial
        self._next_serial = serial + 1
        self._transaction = _BatchTransaction(serial, owner, before, after,
                                              indices)
        return serial, indices

    def _publish_pending(self, serial, owner):
        """Phase 3: turn this claim into a pending-delivery record.

        The claim is rechecked rather than assumed: only its owner can
        resolve it, so a mismatch already means internal state was broken,
        and failing loudly is worth more than proceeding. The committed
        epoch and cursor still do not move.
        """
        transaction = self._matching_transaction(serial, owner, _CLAIMED)
        if transaction is None:
            raise RuntimeError(
                f"batch-delivery transaction {serial} lost its claim before "
                f"it could be published; this sampler's internal transaction "
                f"state is inconsistent"
            )
        transaction.status = _PENDING

    def _commit_pending(self, serial, owner):
        """Phase 4, step 1: apply the candidate position.

        Through ``_assign_state`` — the one non-failing write seam, shared
        with ``load_state_dict`` and with the rollback below, so a
        rollback that could itself raise is structurally impossible.

        The status is advanced **before** the write rather than after. The
        write cannot fail, so the order is unobservable in production; it
        matters under the J3 injection that makes it fail anyway, where
        marking first is what keeps the rollback's restore correct instead
        of concluding that nothing had been applied.
        """
        transaction = self._matching_transaction(serial, owner, _PENDING)
        if transaction is None:
            raise RuntimeError(
                f"batch-delivery transaction {serial} is not pending on this "
                f"sampler; it cannot be committed"
            )
        transaction.status = _COMMITTED
        self._assign_state(*transaction.after)

    def _rollback_pending(self, serial, owner):
        """Phase 5: restore the exact pre-delivery position and clear the
        record. Returns whether it matched.

        Restoring comes **first**, and through the non-failing seam,
        precisely so the committed state is already correct before any
        step that could conceivably raise. A record still in its claim or
        pending phase never had the candidate applied, so there is nothing
        to restore and only the record is cleared.

        Exact-match and therefore **idempotent**: a second attempt finds
        no matching live record and does nothing.
        """
        transaction = self._matching_transaction(serial, owner, None)
        if transaction is None:
            return False
        if transaction.status == _COMMITTED:
            self._assign_state(*transaction.before)
        self._transaction = None
        return True

    def _complete_pending(self, serial, owner):
        """Phase 4, step 3: mark this transaction delivered. Returns
        whether it matched.

        Only a **committed** record owned by this exact serial and token
        can complete, so a transaction that a reentrant ``close()``
        already rolled back cannot be completed afterwards, and no
        transaction can be completed twice. The serial is never reused, so
        no later cleanup can reach a completed one.
        """
        transaction = self._matching_transaction(serial, owner, _COMMITTED)
        if transaction is None:
            return False
        self._transaction = None
        return True

    # -- representation -------------------------------------------------

    def __repr__(self):
        """Configuration and position only, and valid when the dataset is
        closed. Never a feature value, a target, a permutation, the
        cache, a fingerprint, an address, or an object id."""
        return (
            f"NativeBatchSampler(samples={self._dataset.samples}, "
            f"batch_size={self._batch_size}, shuffle={self._shuffle}, "
            f"seed={self._seed}, drop_last={self._drop_last}, "
            f"epoch={self._epoch}, cursor={self._cursor}, "
            f"batches_per_epoch={self.batches_per_epoch}, "
            f"remaining={self.remaining})"
        )
