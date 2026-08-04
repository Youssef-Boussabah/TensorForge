"""NativeTensorDataset — the finite, host-backed native dataset (Phase J,
milestone J1; see docs/native_data_pipeline_design.md §3.3, §4, §5, §6,
§10.4, §12.2, §12.6, §15, and §17.2).

The first piece of Phase-J runtime. It holds **one owned host snapshot of
the features and one of the class targets**, at an *explicitly chosen*
native feature dtype, and turns any index sequence into a fresh owning
``NativeTensor`` feature batch beside a fresh read-only host ``int64``
target batch.

What it is, and what it deliberately is not
-------------------------------------------

It is **input**, not training state: it has no ``state_dict``, no
``load_state_dict``, no cursor, no epoch, and no shuffle. There is **no
sampler and no loader** — ``NativeBatchSampler`` (J2) and
``NativeDataLoader`` (J3) do not exist yet, and nothing here plans,
orders, or groups batches. This class answers exactly one question:
*given these indices, what is the batch?*

The three rules that shape every line below
-------------------------------------------

**1. The snapshots are copies, taken once, and nothing survives from the
caller.** Both are built with ``numpy.array(..., copy=True)`` rather than
``numpy.ascontiguousarray``, which returns an already-contiguous input
*unchanged* and would therefore alias caller memory in exactly the common
case (§5.1). After construction the caller may mutate, resize, or delete
their arrays freely; the dataset, its fingerprint, and every batch it will
ever produce are unaffected. Hidden aliasing is the exact negation of the
determinism this phase exists to provide.

**2. The native dtype is chosen, never inferred.** ``dtype`` accepts
``None``, ``"float64"``, and ``"float32"`` through the one shared
``_native_dtype.normalize_module_dtype`` route; ``None`` means
``"float64"``. A ``float32`` NumPy feature array with ``dtype`` omitted
gives a **float64** dataset and float64 batches — the Phase-I rule (dtype
design §9.4) applied without exception. Converting the host values into
the chosen dtype happens **once, here**, at the same explicit host→native
boundary ``from_array`` has always used, so every later batch transfer
copies matching bits and no per-batch conversion exists. That is not a
casting feature: no *native* tensor ever changes dtype, and none can.

**3. The dataset owns no native storage.** Between calls it holds two
NumPy arrays and nothing else. Native storage is allocated only *inside*
``feature_batch``, and every byte of it belongs to the returned tensor,
which **the caller closes**. Constructing, holding, inspecting, and
discarding a dataset therefore leaves the native live-storage count
exactly where it was.

Identity, and why it is a content fingerprint
---------------------------------------------

``identity()`` reports four JSON-compatible fields — ``samples``,
``feature_shape``, ``feature_dtype``, and a SHA-256 ``fingerprint`` (§6).
The first three catch a structural mismatch; the fourth catches *"the
same shape, the same dtype, different data"* — a different split, a
re-shuffled source, a changed preprocessing step — which is precisely the
case a resume proof must exclude. The digest is over a canonical,
explicitly little-endian byte stream (§6.3), so identical logical data
fingerprints identically regardless of the input's byte order or the
host's. It detects **accidents**; it is not an adversarial integrity
check and no document may describe it as a security property.

Adds no C++, no kernel, no C ABI symbol, no ctypes declaration, no
checkpoint field or version, no optimizer-state version, no capability
registry value, and no dependency — ``hashlib`` is the standard library.
Fully separate from the stable line: ``tensorforge.data.batches`` is
neither used, wrapped, imported, extended, nor changed, and a stable
``tensorforge.Tensor`` is rejected exactly where any other non-``ndarray``
is (§18).
"""

import hashlib

import numpy as np

from tensorforge.backends import cpp

from ._native_dtype import normalize_module_dtype
from .native_tensor import NativeTensor

# The one device this line has. There is no ``device`` argument anywhere in
# Phase J and none may be added (§19.7); this is a read-only report, present
# so the dataset answers the same metadata questions every other native
# object does.
_DEVICE = "cpu"

# The fingerprint domain and schema tags (§6.3, rows 1, 2, and 7). Changing
# any of them changes every digest, so they are spelled here once, as bytes,
# and are part of the locked contract rather than an implementation detail.
_FINGERPRINT_DOMAIN = b"tensorforge.native_dataset\x00"
_FINGERPRINT_SCHEMA = b"fingerprint-v1\x00"
_FINGERPRINT_TARGET_MARKER = b"targets\x00"

# How many elements are handed to the hasher at a time. Purely an
# implementation detail of *peak host memory*: SHA-256 is a streaming
# construction, so feeding the identical byte sequence in any chunking
# produces the identical digest. Hashing through one whole-array
# ``tobytes()`` would double the dataset's peak memory for no benefit.
_HASH_CHUNK_ELEMENTS = 1 << 20

# The int64 window a target value must fit, checked as Python ints so no
# NumPy arithmetic decides an acceptance (the ``_prepare_class_targets``
# precedent).
_INT64_MIN = -(2 ** 63)
_INT64_MAX = 2 ** 63 - 1


def _hash_values(hasher, array):
    """Feed ``array``'s elements to ``hasher`` in C order, little-endian.

    The snapshots are stored in *native* byte order, so on a big-endian
    host their raw bytes would differ from a little-endian host's for
    identical values. ``astype(newbyteorder("<"), copy=False)`` is the
    explicit normalization: a no-op returning the same object on a
    little-endian host, and a byte-swapped copy on a big-endian one. This
    is why the digest is a property of the *values*, not of the machine.
    """
    little = array.astype(array.dtype.newbyteorder("<"), copy=False)
    # Both snapshots are C-contiguous, so this reshape is a view and each
    # slice below is contiguous — ``tobytes()`` on it is exactly the bytes
    # the stream calls for, with no whole-array temporary.
    flat = little.reshape(-1)
    start = 0
    total = flat.size
    while start < total:
        stop = min(start + _HASH_CHUNK_ELEMENTS, total)
        hasher.update(flat[start:stop].tobytes())
        start = stop


def _fingerprint(dtype, features, targets):
    """The §6.3 digest over the two snapshots: 64 lowercase hex characters.

    The byte stream, in this exact order: the domain tag; the schema tag;
    the selected native dtype name and a NUL; the feature rank; every full
    feature dimension **including the sample axis**; the feature values;
    the target marker; the target count; the target values. Shape is
    hashed *before* the values so ``(6, 2)`` and ``(4, 3)`` holding the
    same numbers cannot collide, and the dtype name is hashed even though
    a float32 and a float64 snapshot already differ in bytes, so the
    digest is self-describing rather than accidentally distinct.

    No Python ``hash()``, no ``pickle``, no ``repr``/``str`` of an array,
    no native-order ``tobytes()``, and no floating-point arithmetic enters
    it: every length and dimension is an 8-byte little-endian unsigned
    integer written by ``int.to_bytes``.
    """
    hasher = hashlib.sha256()
    hasher.update(_FINGERPRINT_DOMAIN)
    hasher.update(_FINGERPRINT_SCHEMA)
    hasher.update(dtype.encode("utf-8"))
    hasher.update(b"\x00")
    hasher.update(len(features.shape).to_bytes(8, "little"))
    for dimension in features.shape:
        hasher.update(int(dimension).to_bytes(8, "little"))
    _hash_values(hasher, features)
    hasher.update(_FINGERPRINT_TARGET_MARKER)
    hasher.update(int(targets.shape[0]).to_bytes(8, "little"))
    _hash_values(hasher, targets)
    return hasher.hexdigest()


class NativeTensorDataset:
    """A finite dataset over one owned host snapshot of features and targets.

    ``NativeTensorDataset(features, targets, *, dtype=None)`` — both
    arguments must be exactly ``numpy.ndarray`` (subclasses included,
    rejected); ``features`` floating with rank ≥ 1 and every dimension
    ≥ 1; ``targets`` a 1-D non-``bool`` integer array of the same length
    whose every value is a non-negative int64. ``dtype`` selects the
    **native** feature dtype and defaults to ``"float64"`` — it is never
    inferred from ``features.dtype``.

    ``feature_batch(indices)`` returns an owning ``NativeTensor`` **the
    caller closes**; ``target_batch(indices)`` returns a fresh read-only
    ``int64`` array needing no close. Both preserve index order and
    duplicates exactly and reject an empty request, because the native
    runtime cannot represent a zero-element tensor.

    See the module docstring for the ownership, dtype, and fingerprint
    contracts.
    """

    # No attribute may be added to a dataset after construction, and the two
    # snapshots are reachable only from the two methods that gather from
    # them: there is deliberately **no public snapshot accessor** (§3.3), as
    # returning one would either alias the dataset's memory into caller
    # hands or copy the whole dataset on a property read.
    __slots__ = ("_features", "_targets", "_samples", "_feature_shape",
                 "_dtype", "_fingerprint", "_closed")

    def __init__(self, features, targets, *, dtype=None):
        # --- §4.8 / §12.2 validation order. Every step below precedes every
        # allocation, so a caller who got two things wrong is told about the
        # more basic one and nothing is half-built when they are.

        # 1. The requested native dtype, first — before NumPy is asked to do
        #    any work at all, through the one shared validator so this
        #    constructor invents no dtype rule of its own.
        canonical = normalize_module_dtype(dtype)

        # 2. Exact feature type. ``type(...) is`` rather than isinstance:
        #    a masked array is an ndarray subclass whose mask a snapshot
        #    would silently discard, and a subclass overriding indexing
        #    could make the gather mean something other than a gather.
        if type(features) is not np.ndarray:
            raise TypeError(
                f"features must be exactly a numpy.ndarray, got "
                f"{type(features).__name__} (ndarray subclasses are "
                f"rejected; convert explicitly on your own side)"
            )
        # 3. Feature rank. A 0-d array has no sample axis at all.
        if features.ndim < 1:
            raise ValueError(
                "features must have at least one dimension (axis 0 is the "
                f"sample axis), got shape {features.shape}"
            )
        # 4. Feature dtype kind. Integer feature arrays are rejected on
        #    purpose: int64 -> float32 silently loses exactness above 2**24,
        #    and an integer column is as often categorical as numeric.
        if not np.issubdtype(features.dtype, np.floating):
            raise TypeError(
                f"features must be a floating-point array, got dtype "
                f"{features.dtype} (bool, integer, complex, object, "
                f"structured, string, datetime, and timedelta arrays are "
                f"rejected outright — nothing is reinterpreted)"
            )
        # 5. Sample count, then every trailing dimension. The runtime
        #    cannot represent a zero-size dimension, so a dataset that
        #    could never produce a batch is a construction error rather
        #    than a surprise in the middle of a training loop.
        samples = int(features.shape[0])
        if samples < 1:
            raise ValueError(
                f"features must hold at least one sample, got {samples} "
                f"(an empty dataset could never produce a batch)"
            )
        for axis, dimension in enumerate(features.shape[1:], start=1):
            if int(dimension) < 1:
                raise ValueError(
                    f"every feature dimension must be at least 1, got "
                    f"{dimension} at axis {axis} of shape {features.shape}"
                )

        # 6. Exact target type, on the same discipline as the features.
        if type(targets) is not np.ndarray:
            raise TypeError(
                f"targets must be exactly a numpy.ndarray, got "
                f"{type(targets).__name__} (ndarray subclasses are rejected)"
            )
        # 7. Target rank: exactly one dimension. An (N, 1) column is not a
        #    label vector here, and is not silently squeezed into one.
        if targets.ndim != 1:
            raise ValueError(
                f"targets must be one-dimensional, got shape {targets.shape}"
            )
        # 8. Target dtype kind, with bool rejected first — ``True`` is not
        #    class 1 here (the Phase-E ``_prepare_class_targets`` rule, at
        #    the same strictness and for the same reason).
        if targets.dtype == np.bool_ or not np.issubdtype(targets.dtype,
                                                          np.integer):
            raise TypeError(
                f"targets must be integer class labels, got a NumPy array "
                f"of dtype {targets.dtype} (bool and floating-point targets "
                f"are rejected outright — nothing is truncated or "
                f"reinterpreted)"
            )
        # 9. Length against the sample count.
        if int(targets.shape[0]) != samples:
            raise ValueError(
                f"targets must hold one label per sample: features have "
                f"{samples} samples, targets have {int(targets.shape[0])}"
            )
        # 10/11. Target values, as exact Python ints so no NumPy arithmetic
        #    decides an acceptance. ``.tolist()`` yields exact ints for
        #    every integer width, including uint64 values above the int64
        #    maximum — which is what makes step 10 checkable at all.
        #    Representability first, then non-negativity, each naming the
        #    first offending index.
        values = targets.tolist()
        for index, value in enumerate(values):
            if value < _INT64_MIN or value > _INT64_MAX:
                raise ValueError(
                    f"target at index {index} is {value}, outside the int64 "
                    f"range the native runtime addresses"
                )
        for index, value in enumerate(values):
            if value < 0:
                raise ValueError(
                    f"target at index {index} is {value}, but class labels "
                    f"must be non-negative"
                )
        # There is deliberately **no upper class bound and no num_classes
        # argument** (§4.4): the number of classes is the model's fact, and
        # ``cross_entropy`` already checks ``0 <= value < num_classes`` on
        # every call. A dataset that also held a class count would be a
        # second authority that could disagree with the model. ``>= 0`` is
        # kept only because it is a strict subset of that check at every
        # possible class count, so it can never disagree with it.

        # --- 12/13/14. Allocation, and only now. Both snapshots are
        # unconditional copies: ``ascontiguousarray`` would return an
        # already-contiguous, already-typed input unchanged and alias
        # caller memory in exactly the common case (§5.1). The feature
        # snapshot lands at the **chosen** dtype's NumPy counterpart, so
        # every later batch is a same-dtype row gather and every transfer
        # copies matching bits.
        feature_snapshot = None
        target_snapshot = None
        try:
            feature_snapshot = np.array(
                features, dtype=cpp._DTYPE_NUMPY[canonical], order="C",
                copy=True,
            )
            target_snapshot = np.array(
                targets, dtype=np.int64, order="C", copy=True,
            )
            fingerprint = _fingerprint(canonical, feature_snapshot,
                                       target_snapshot)
        except BaseException:
            # §17.2: whatever was allocated is released *before* the
            # exception leaves the constructor, so a traceback that keeps
            # this frame alive cannot keep a half-built dataset's snapshots
            # alive with it. No native storage exists at any point here, so
            # a failed construction cannot move the live-storage count.
            feature_snapshot = None
            target_snapshot = None
            raise

        # --- 15. Publish. Plain attribute assignments that cannot fail, so
        # no partially initialized dataset is ever observable.
        self._features = feature_snapshot
        self._targets = target_snapshot
        self._samples = samples
        self._feature_shape = tuple(int(dimension)
                                    for dimension in feature_snapshot.shape[1:])
        self._dtype = canonical
        self._fingerprint = fingerprint
        self._closed = False

    # -- metadata (all of it readable after close) ---------------------
    #
    # These are what a state comparison needs, and they are exactly what
    # survives ``close()`` — the ``NativeStorage.dtype`` precedent, where
    # metadata is explicitly readable after the storage is gone.

    @property
    def samples(self):
        """The number of samples; always ``>= 1``."""
        return self._samples

    @property
    def feature_shape(self):
        """The **per-sample** shape; ``()`` for scalar samples."""
        return self._feature_shape

    @property
    def dtype(self):
        """The native dtype every feature batch carries: ``"float64"`` or
        ``"float32"``. Chosen at construction and never inferred from the
        host array."""
        return self._dtype

    @property
    def device(self):
        """``"cpu"``. There is no ``device`` argument and no device
        movement (§19.7)."""
        return _DEVICE

    @property
    def fingerprint(self):
        """The §6.3 content digest: 64 lowercase hexadecimal characters."""
        return self._fingerprint

    @property
    def closed(self):
        return self._closed

    def __len__(self):
        return self._samples

    def identity(self):
        """The four JSON-compatible identity fields, as a **fresh** dict.

        Nothing mutable is shared with the dataset or with a previous
        result, so a caller may edit what they are given without reaching
        anything. Carries **no dataset content** — no value, no array, no
        byte string, and nothing process-local: a count, a shape, a dtype
        name, and a digest. Every field passes the checkpoint's
        ``_validated_metadata`` unchanged, which is what lets a caller
        record it through the existing version-3 metadata channel without
        the archive growing a field.

        Available after ``close()``, because it is metadata.
        """
        return {
            "samples": self._samples,
            "feature_shape": [int(dimension)
                              for dimension in self._feature_shape],
            "feature_dtype": self._dtype,
            "fingerprint": self._fingerprint,
        }

    # -- batches -------------------------------------------------------

    def _validated_indices(self, indices, method):
        """The §12.6 index contract, in order: closed → container →
        non-empty → bounds. Returns a plain ``list`` of exact Python ints
        in the caller's order.

        Order is significant and duplicates are legal: a gather is a
        gather. Nothing is sorted, deduplicated, set-converted, clamped,
        wrapped, or otherwise normalized — every one of those would change
        which row lands in which batch position.
        """
        # 1. Lifecycle first, before the request is even parsed.
        if self._closed:
            raise RuntimeError(
                f"{method} is unavailable on a closed NativeTensorDataset "
                f"(its host snapshots were released by close())"
            )
        # 2. Container and element types.
        if type(indices) is np.ndarray:
            if indices.ndim != 1:
                raise ValueError(
                    f"{method} indices must be one-dimensional, got shape "
                    f"{indices.shape}"
                )
            if indices.dtype == np.bool_ or not np.issubdtype(indices.dtype,
                                                              np.integer):
                raise TypeError(
                    f"{method} indices must be an integer array, got dtype "
                    f"{indices.dtype} (bool arrays are rejected — they are "
                    f"not positions)"
                )
            # Exact Python ints at every width, so the bounds check below
            # is total even for a uint64 above the int64 maximum.
            wanted = indices.tolist()
        elif type(indices) is tuple or type(indices) is list:
            wanted = list(indices)
            for position, value in enumerate(wanted):
                # ``type(value) is int`` excludes bool and every int
                # subclass. A caller holding NumPy integers passes the
                # array itself, which the branch above accepts.
                if type(value) is not int:
                    raise TypeError(
                        f"{method} indices must contain exact ints, got "
                        f"{value!r} at position {position}"
                    )
        else:
            raise TypeError(
                f"{method} indices must be a tuple or list of ints, or a "
                f"one-dimensional integer numpy.ndarray, got "
                f"{type(indices).__name__}"
            )
        # 3. Non-empty. A zero-row batch cannot become a native tensor, so
        #    it is refused here rather than deep inside an allocation.
        if not wanted:
            raise ValueError(
                f"{method} needs at least one index: a zero-row batch "
                f"cannot be represented as a native tensor"
            )
        # 4. Bounds, naming the position and the value.
        for position, value in enumerate(wanted):
            if value < 0 or value >= self._samples:
                raise ValueError(
                    f"{method} index at position {position} is {value}, "
                    f"outside the valid range [0, {self._samples})"
                )
        return wanted

    def feature_batch(self, indices):
        """An **owning** ``NativeTensor`` of shape ``(len(indices),) +
        feature_shape`` at ``self.dtype`` on ``"cpu"``. **The caller
        closes it.**

        One NumPy fancy-index gather produces a fresh C-contiguous host
        array — already the target dtype's counterpart, because the
        snapshot is — and it goes through the public ``from_array``
        host→native boundary with ``dtype`` passed **explicitly**, so the
        gathered array's dtype never selects the native one. The transfer
        therefore copies matching bits with no conversion, no intermediate
        native tensor, no reshape, and no view chain.

        The result aliases nothing: not the dataset, not a previous batch.
        Two calls with equal indices give two independent tensors with
        identical contents, and closing one changes nothing anywhere else.
        The dataset keeps no reference to it and cannot close it for you.
        """
        wanted = self._validated_indices(indices, "feature_batch")
        host = self._features[wanted]
        return NativeTensor.from_array(host, dtype=self._dtype)

    def target_batch(self, indices):
        """A fresh, independently owned, C-contiguous, **read-only**
        ``int64`` array of shape ``(len(indices),)``. No close needed.

        Copied rather than viewed so a caller cannot mutate the dataset
        through a batch, and read-only so the object they hold cannot be
        edited in place and re-used with a different meaning — the same
        stance the Phase-E forward takes with its own saved copy. A
        read-only array is still a perfectly good ``cross_entropy``
        argument, which re-validates and re-copies it.

        Targets are host metadata at every width; **no native integer
        tensor exists**, is needed, or is implied.
        """
        wanted = self._validated_indices(indices, "target_batch")
        batch = np.ascontiguousarray(self._targets[wanted], dtype=np.int64)
        batch.setflags(write=False)
        return batch

    # -- lifecycle -----------------------------------------------------

    def close(self):
        """Release both host snapshots. Idempotent; returns ``None``.

        Allocates nothing, frees no native storage (there is none to
        free), and leaves every metadata property, ``identity()``, and
        ``__repr__`` working. ``feature_batch`` and ``target_batch``
        afterwards raise ``RuntimeError`` before validating anything.

        There is no ``__del__``: the dataset owns two NumPy arrays and no
        native resource, so ordinary Python reclamation is already
        correct, and inventing a finalizer would advertise a lifetime this
        object does not have (``NativeGenerator``'s stated reason for
        having no ``close()`` at all, applied rather than replaced).
        """
        self._features = None
        self._targets = None
        self._closed = True
        return None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
        return False

    def __repr__(self):
        """Metadata only, and valid after close. Never a feature value, a
        target, a snapshot, an address, or an object id — and only the
        fingerprint's first 12 characters, which is enough to recognize a
        dataset in a log without putting a full identity there."""
        return (
            f"NativeTensorDataset(samples={self._samples}, "
            f"feature_shape={self._feature_shape}, dtype='{self._dtype}', "
            f"device='{_DEVICE}', closed={self._closed}, "
            f"fingerprint='{self._fingerprint[:12]}...')"
        )
