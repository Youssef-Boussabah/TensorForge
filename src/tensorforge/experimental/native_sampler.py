"""NativeBatchSampler — the deterministic native batch planner (Phase J,
milestone J2; see docs/native_data_pipeline_design.md §3.4, §7, §8, §11.2,
§12.3, §12.4, §15.2, and §16).

The second piece of Phase-J runtime, and a **planner and state holder**,
not an iterator. It answers *"which indices, in which groups, in which
order"* and nothing else: it materializes no batch, allocates nothing
native, constructs no ``NativeTensor``, and has no ``__iter__``,
``__next__``, or public advance of any kind. ``NativeDataLoader`` (J3)
does not exist yet, and iteration and successful-delivery cursor
advancement are its milestone, not this one.

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
configuration-and-position scalars, one reference to its dataset, and an
optional private tuple cache. No native storage, no host snapshot, no
batch, no tensor, no file, no generator, no thread, no lock, no worker,
and no queue. Inventing a ``close()`` would advertise a lifetime this
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

**No public method here applies it.** At J2 a position other than
``(0, 0)`` arrives only through a validated ``load_state_dict``, which is
the one audited path into a position. ``_next_position`` computes the
candidate without mutating anything, and ``_assign_state`` is the single
non-failing write seam; J3 uses both to commit a delivery and to roll one
back, which is what keeps a rollback exact.

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
# hand back — including, once J3 exists, a loader state, which wraps this
# one under its own tag precisely so the two cannot be confused.
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
                 "_epoch", "_cursor", "_cache_key", "_cache_order")

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
        advances it: at J2 nothing delivers a batch, and at J3 only a
        successful delivery will."""
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
        """
        # (J3 adds the §9.5 in-flight-transaction refusal here, ahead of
        # everything else. At J2 no loader, iterator, or batch transaction
        # exists, so no such transaction can exist to refuse; inventing a
        # flag to advertise the behavior early would be a fiction. The
        # public schema below does not change when the guard arrives.)
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
        """
        # (J3 adds the §12.4 transaction and active-iteration guards here,
        # ahead of the container check, exactly as the design orders them.
        # Neither can exist at J2: there is no iterator and no batch
        # transaction to be in flight.)
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
        #    distinguishes this from any other JSON object — including,
        #    once J3 exists, a loader state.
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
        it is not undefined.
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
