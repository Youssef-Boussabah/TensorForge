"""Native checkpoint files (Advanced C++ v3.14; format version 3 as of
Phase I milestone I8 — see docs/backend_experiments.md,
docs/native_rng_dropout_design.md §10, and
docs/native_dtype_float32_design.md §16-§17).

``save_native_checkpoint(path, model, optimizer=None, metadata=None)``
and ``load_native_checkpoint(path, model, optimizer=None)`` persist a
``NativeModule``'s parameters and persistent buffers, optionally one
native optimizer's v3.13 state (``NativeSGD`` or ``NativeAdam``), every
registered ``NativeGenerator``'s random state **and sharing topology**,
and optional JSON-compatible metadata to **one explicit, pickle-free NPZ
archive** — and restore them deterministically, so a saved native
training run resumes bit-for-bit. Everything follows the stable
framework's no-pickle serialization philosophy
(src/tensorforge/serialization.py): arrays and JSON only,
``allow_pickle=False`` on load, no code and no object reconstruction.

The archive (format ``"tensorforge.native_checkpoint"``, format
version 3) contains:

- ``manifest`` — one JSON document encoded as UTF-8 bytes in a 1-D
  ``uint8`` array (never an object array, never pickle):
  ``{"format", "format_version", "model": {"keys", "entries"},
  "optimizer": null | {...}, "generators": null | {...},
  "metadata": {...}}``. Model entries map each canonical state key to
  its archive array name plus exact shape/dtype/device; the optimizer
  section reproduces the v3.13 in-memory schema (type tag, state format
  version, hyperparameters, positional parameter metadata, and — for
  NativeAdam — step counts and the m/v archive array names). No Python
  ``id()``, pointer, repr, gradient, parameter version, graph data, or
  closed flag is ever written.
- one array per model parameter/persistent buffer and per Adam moment,
  under deterministic zero-padded indexed names: ``model::000000`` … in
  model-key order, ``optimizer::m::000000`` … / ``optimizer::v::000000``
  … in the optimizer's positional order. The manifest maps semantic
  entries to array names explicitly; duplicate references, missing
  arrays, and unreferenced extra entries are all rejected on load.

**Dtype (v3).** Every numeric entry — model parameter, persistent
buffer, and Adam moment alike — declares its dtype explicitly as the
canonical string ``"float64"`` or ``"float32"``, and the loader checks
that declaration against *both* the stored array (exact NumPy dtype, in
native byte order) and the live destination (exact equality). A
disagreement in either direction is a ``ValueError`` naming the entry,
the expected value, and the found one. **There is no cast, no
promotion, no ``map_location``, no device movement, and no "closest
match" anywhere on either path** — a float32 checkpoint loads into a
float32 model or it does not load. Version 3 is where Adam's ``"m"`` and
``"v"`` became lists of entry objects (``{"array", "shape", "dtype",
"device"}``) instead of bare archive names, so a moment's metadata is
carried rather than inferred positionally from ``"parameters"``: that
inference holds only while the two lists agree, which is exactly what a
malformed archive violates.

**Versions 1 and 2 are float64-only formats, permanently.** They predate
float32 and have no way to say otherwise, so loading one requires every
declared dtype *and* every stored array to be float64, and a payload is
**never guessed** to be float32 — not from an array's dtype, not from a
byte length, not from a shape. A v1/v2 archive therefore loads into a
float64 model or it fails, which is the correct behavior rather than a
limitation. The **in-memory** optimizer state schema
(``native_optimizer_state.FORMAT_VERSION``) is a different thing and
stays at 1: only the file encoding of the moments moved.

**The generator section (v2).** ``"generators"`` is ``null`` for a model
with no registered generators — absence is explicit, never inferred from
a missing field — or an object with exactly three fields:

- ``keys`` — the ordered canonical generator names, from the
  identity-deduplicated ``named_generators()`` walk;
- ``entries`` — one ``{algorithm, algorithm_version, seed, calls}``
  object per canonical name, mapping exactly ``keys`` in the same order.
  ``seed`` and ``calls`` are **canonical decimal strings**
  (``^(0|[1-9][0-9]*)$``, ≤ 20 digits, in ``[0, 2**64 - 1]``), because a
  ``uint64`` above ``2**53`` is not representable in the IEEE double most
  JSON readers use and a checkpoint that silently loses the low bits of a
  seed is worse than one that will not parse;
- ``aliases`` — the complete **registered path → canonical name** map, in
  full traversal order, *including* each canonical name mapped to itself.

Generator state adds **no array** to the NPZ payload: it is four scalar
fields per generator and lives entirely in the manifest. A shared
generator's state is written **once**, under its canonical name, exactly
as a shared parameter's tensor is — but unlike a shared parameter, its
*sharing topology is itself semantic state*: two Dropout layers on one
generator consume one interleaved stream, two independent generators
consume two, and restoring the states without the topology would accept a
model whose stochastic behavior after the resume differs from the one
that was saved. So ``aliases`` records the topology explicitly and a load
compares it, strictly and in both directions, against a real
``named_generators()`` traversal of the live model.

**Version-1 compatibility.** New saves always write version 3 — the
version describes the *schema*, not the content, so a float64
generator-free model still writes 3. A
version-1 archive (no ``"generators"`` field) remains loadable into a
model with **no** registered generators and behaves exactly as it always
did; loading one into a model that **has** generators fails, naming them,
because no seed and no counter is ever fabricated — not zero, not fresh
entropy, not the generator's current value. A version-2 or version-3 archive with a
non-null generator section loaded into a generator-free model fails as an
unexpected-generator error. Any other ``format_version`` fails. There is
no "latest wins", no upgrade in place, and no silent rewrite of an old
file.

**Strict optimizer presence.** Unlike the stable ``load_checkpoint``
(which silently ignores archive optimizer state when no optimizer is
passed), the native contract is strict in both directions: an archive
with optimizer state requires a compatible optimizer of the same type,
and an archive without one rejects a supplied optimizer — so a resume
can never silently discard or invent optimizer state. There are no
``model_only``/ignore flags, no ``strict=False``, and no
``map_location``.

**Saving** validates everything first (path, open ``NativeModule``,
``None``/``NativeSGD``/open ``NativeAdam`` optimizer whose unique
parameter sequence is positionally *identical* to the model's, and
recursively JSON-compatible metadata), snapshots the model and
optimizer through their existing ``state_dict()`` contracts, reads every
registered generator's state as **one locked, reservation-checked
snapshot** (a save is refused, changing nothing, while any target
generator has a call reservation in flight or under construction —
a generator whose next index has been decided but not committed has no
single honest state to record), converts each caller-owned snapshot
through the explicit ``to_numpy()`` serialization boundary, and closes
every snapshot in a ``finally`` — then writes through a collision-safe
temporary file in the destination directory (``np.savez`` onto an
explicitly opened handle, so NumPy can never silently rename the file)
and commits with ``os.replace``: an existing destination is replaced
atomically on success and remains byte-intact on failure, and no
temporary file survives either way (same-filesystem atomicity only; no
directory creation).

**Loading is one transaction over the whole archive** (§10.7), in four
phases.

*Phase 1 — prevalidation, nothing touched.* The live model and
optimizer, then the complete archive: opened with ``allow_pickle=False``;
manifest presence, representation, UTF-8, JSON (duplicate object keys
rejected), root type; exact format identity, a supported version, and the
version's exact field set; strict optimizer presence/type; model keys
against the live model and every array's exact declared dtype (float64
only at versions 1 and 2) and shape against both manifest and live
destination; optimizer
scalars/metadata/counters through the same validators the optimizer
constructors use; the **complete generator topology** — section shape,
canonical keys, entries mapping them in order, algorithm and version
equality against the *live* generator, canonical ``uint64`` strings and
ranges, and the alias map compared path-by-path against the live
traversal; and the full array cross-reference. If anything fails,
nothing whatsoever has changed.

*Phase 2 — staging, everything that can allocate or raise.* Independent
``NativeTensor`` copies of every model value and Adam moment; the
validated generator states; and, for rollback, an independent owning
snapshot of **every live target the commit will overwrite** — the current
model and buffer values with their versions, the optimizer's complete
current state, and each generator's ``(seed, calls)`` read under its
lock with no reservation in flight. A staging failure closes everything
staged and leaves every live component untouched.

*Phase 3 — commit, atomic under any synchronous exception.* Model →
optimizer → generators, each through its component's own loader, all
inside **one** rollback guard (``_native_checkpoint_transaction``). If
any exception is raised anywhere in the commit — including a deliverable
``KeyboardInterrupt`` — every component that had committed is rolled
back, so a caller that catches it sees exactly the pre-load model,
buffers, optimizer, and generators: no parameter version moved, no
identity changed, no partially loaded component observable, and
graph-owned saved state (Dropout masks, MaxPool2d winners, cross-entropy
probabilities) from graphs built before the load untouched, as it is on
every path. Staged tensors and rollback snapshots are closed in a
``finally``, so native live storage returns to its baseline.

*Phase 4 — the one honest exception.* Only **external** asynchronous
termination of the process or death of the interpreter (``SIGKILL``, a
power loss, a hard crash) is outside the guarantee, because no in-process
rollback can survive it. An asynchronous exception that is nevertheless
deliverable to Python is **not** an exception to it.

Committed behavior is exactly the components' documented contracts: model
loading increments each parameter version once and makes old
value-sensitive retained graphs stale; optimizer loading and generator
loading move no versions and leave graphs valid. Generators are loaded
**in place** — the archive never constructs a ``NativeGenerator`` — so
every registered object, canonical name, and sharing relationship
survives a load unchanged.

``load_native_checkpoint`` returns the checkpoint's metadata as an
independent plain-Python dictionary (a fresh JSON parse — mutating it
affects nothing), validated by the **same** ``_validated_metadata``
authority the saver uses, during archive prevalidation and before
anything is staged or mutated. That symmetry matters because
``json.loads`` accepts the non-standard ``NaN``/``Infinity``/
``-Infinity`` literals: without it, a hand-written archive could hand
back a value ``save_native_checkpoint`` would have refused to write.
Deliberately **not** implemented: scheduler state,
Python or NumPy global random-state capture, dataloader/shuffle position,
multiple models/optimizers, partial loading, name-based remapping,
checkpoint merging, sharding, compression, encryption, URLs, dtype
casting, or device movement. Reproducibility is exact **for the state
actually captured**; full-program determinism is not claimed. Still
float64/cpu only, experimental, and fully separate from
``tensorforge.serialization``.
"""

import json
import math
import os
import re
import tempfile

import numpy as np

from ..backends import cpp
from . import _native_checkpoint_transaction as _transaction
from ._native_state_lock import state_transaction
from .native_adam import (
    NativeAdam,
    _validated_betas,
    _validated_positive_real,
)
from .native_generator import (
    ALGORITHM as _GENERATOR_ALGORITHM,
    ALGORITHM_VERSION as _GENERATOR_ALGORITHM_VERSION,
    GeneratorStateEntry,
    UINT64_MAX as _UINT64_MAX,
    snapshot_generator_states,
)
from .native_module import NativeModule
from .native_optimizer_state import (
    FORMAT_VERSION as _STATE_FORMAT_VERSION,
    validate_parameter_metadata,
    validate_step_counts,
)
from .native_sgd import NativeSGD, _validated_lr
from .native_tensor import NativeTensor

_FORMAT = "tensorforge.native_checkpoint"
# The schema version, not a statement about what a particular model
# contains: every new save writes 3, whether or not the model holds a
# float32 tensor and whether or not it has generators — exactly as v2 was
# written for generator-free models (Phase I, milestone I8; design §16.1).
# The format **name** never moves; that is the rule G5 established.
_FORMAT_VERSION = 3
# All three versions load; the loader dispatches on the value. There is no
# "latest wins", no upgrade in place, and no silent rewrite of an old file.
_SUPPORTED_FORMAT_VERSIONS = (1, 2, 3)
# Versions 1 and 2 are float64-only formats, permanently (design §16.5):
# they predate float32 and have no way to say otherwise, so a payload is
# never *guessed* to be float32 — not from an array's dtype, not from a
# byte length, not from a shape.
_FLOAT64_ONLY_VERSIONS = (1, 2)
_MANIFEST_ENTRY = "manifest"
_MANIFEST_KEYS = {
    "format", "format_version", "model", "optimizer", "generators",
    "metadata",
}
# Version 1 had no "generators" field at all — its absence is what marks
# an archive as pre-G5, and it is validated as a strict field set so a v1
# archive carrying a generator section is rejected rather than half-read.
_MANIFEST_KEYS_V1 = _MANIFEST_KEYS - {"generators"}
_MODEL_SECTION_KEYS = {"keys", "entries"}
_MODEL_ENTRY_KEYS = {"array", "shape", "dtype", "device"}
# A version-3 optimizer moment is described by an entry object of exactly
# the same shape as a model entry (design §16.2 item 3). In v1/v2 "m" and
# "v" were bare lists of archive names whose shape and dtype were implied
# positionally by the "parameters" list — which works only while the two
# lists agree, precisely what a malformed archive violates.
_MOMENT_ENTRY_KEYS = _MODEL_ENTRY_KEYS
_SGD_SECTION_KEYS = {"type", "state_format_version", "lr", "parameters"}
_ADAM_SECTION_KEYS = _SGD_SECTION_KEYS | {"betas", "eps", "step_counts", "m", "v"}
_GENERATOR_SECTION_KEYS = {"keys", "entries", "aliases"}
_GENERATOR_ENTRY_KEYS = {"algorithm", "algorithm_version", "seed", "calls"}

# A canonical unsigned-64-bit decimal string: no sign, no leading zeros
# (except "0" itself), no separators, no decimal point, no exponent. The
# 20-digit cap is 2**64 - 1's width; the range is still checked after
# parsing, so "99999999999999999999" is rejected on value, not on shape.
_CANONICAL_UINT64 = re.compile(r"(?:0|[1-9][0-9]*)")
_UINT64_MAX_DIGITS = 20


# ---------------------------------------------------------------------------
# Shared validation helpers
# ---------------------------------------------------------------------------


def _validated_path(path, where):
    """``path`` as a plain string: str or os.PathLike only."""
    if isinstance(path, str):
        return path
    if isinstance(path, os.PathLike):
        fspath = os.fspath(path)
        if isinstance(fspath, str):
            return fspath
    raise TypeError(
        f"{where}: path must be a str or os.PathLike, got "
        f"{type(path).__name__}"
    )


def _validate_model(model, where):
    """``model`` must be a NativeModule whose parameters and persistent
    buffers are all open. Stable framework modules are rejected by the
    type check — nothing is converted."""
    if not isinstance(model, NativeModule):
        raise TypeError(
            f"{where}: model must be a NativeModule, got "
            f"{type(model).__name__}"
        )
    for name, tensor in model._state_named_tensors():
        if tensor.closed:
            raise RuntimeError(
                f"{where}: model state entry {name!r} has been closed"
            )


def _validate_optimizer(optimizer, model, where):
    """``optimizer`` must be None, a NativeSGD, or an open NativeAdam
    whose unique parameter sequence is positionally identical (by
    object identity) to the model's — the structural-compatibility
    proof that this optimizer really drives this model. Stable
    optimizers are rejected by the type check."""
    if optimizer is None:
        return
    if not isinstance(optimizer, (NativeSGD, NativeAdam)):
        raise TypeError(
            f"{where}: optimizer must be None, a NativeSGD, or a "
            f"NativeAdam, got {type(optimizer).__name__}"
        )
    if isinstance(optimizer, NativeAdam) and optimizer.closed:
        raise RuntimeError(f"{where}: the optimizer has been closed")
    model_parameters = model.parameters()
    optimizer_parameters = optimizer.parameters()
    if len(optimizer_parameters) != len(model_parameters):
        raise ValueError(
            f"{where}: the optimizer stores {len(optimizer_parameters)} "
            f"unique parameters, the model has {len(model_parameters)}"
        )
    for index, (model_parameter, optimizer_parameter) in enumerate(
        zip(model_parameters, optimizer_parameters)
    ):
        if model_parameter is not optimizer_parameter:
            raise ValueError(
                f"{where}: optimizer parameter {index} is not the "
                f"model's parameter at the same position — the "
                f"optimizer must be built over this model's "
                f"parameters()"
            )


def _validated_metadata(value, path, seen):
    """Validate one metadata value as recursively JSON-compatible and
    return its normalized plain-Python form. Exact scalar types only —
    ``type() is`` checks, so NumPy scalars (``np.float64`` subclasses
    ``float``) are rejected; floats must be finite; tuples normalize to
    lists (the stable ``json.dumps`` convention); dict keys must be
    str; cyclic containers are rejected via the ``seen`` id set.

    **One authority, used on both sides** (Phase I, milestone I10).
    ``save_native_checkpoint`` runs this over caller-supplied metadata and
    ``load_native_checkpoint`` runs it over the parsed manifest's, so the
    set a load accepts is exactly the set a save could have written. Some
    of the rules are vacuous on the load side and deliberately kept
    anyway: decoded JSON has no non-str object keys, no cycles, and no
    arbitrary Python objects, so those branches cannot fire there. The one
    that genuinely fires on load is the finite-float rule, because
    ``json.loads`` accepts ``NaN``, ``Infinity``, and ``-Infinity`` as a
    non-standard extension."""
    if value is None:
        return None
    kind = type(value)
    if kind is bool:
        return value
    if kind is int:
        return value
    if kind is float:
        if not math.isfinite(value):
            raise ValueError(
                f"{path} must be a finite number, got {value!r}"
            )
        return value
    if kind is str:
        return value
    if kind in (list, tuple):
        if id(value) in seen:
            raise ValueError(f"{path} is part of a cyclic container")
        seen = seen | {id(value)}
        return [
            _validated_metadata(item, f"{path}[{index}]", seen)
            for index, item in enumerate(value)
        ]
    if kind is dict:
        if id(value) in seen:
            raise ValueError(f"{path} is part of a cyclic container")
        seen = seen | {id(value)}
        normalized = {}
        for key, item in value.items():
            if type(key) is not str:
                raise TypeError(
                    f"{path} keys must be str, got {type(key).__name__}"
                )
            normalized[key] = _validated_metadata(
                item, f"{path}[{key!r}]", seen
            )
        return normalized
    raise TypeError(
        f"{path} has unsupported type {type(value).__name__}: metadata "
        f"must be recursively JSON-compatible (None, bool, int, finite "
        f"float, str, list/tuple, str-keyed dict)"
    )


def _checkpoint_error(where, detail, cause=None):
    error = ValueError(f"{where}: {detail}")
    if cause is not None:
        raise error from cause
    raise error


# ---------------------------------------------------------------------------
# Generator topology (format version 2; design §10.3-§10.5)
# ---------------------------------------------------------------------------


def _encoded_uint64(value):
    """One ``uint64`` as its canonical decimal string. ``str()`` on a
    non-negative Python int already produces exactly the canonical form
    the loader's pattern accepts; the function exists so the encode and
    decode rules sit next to each other."""
    return str(value)


def _decoded_uint64(text, path, where):
    """Parse one canonical decimal ``uint64`` string, or fail.

    Deliberately narrow, because every looser reading loses information
    silently somewhere: a JSON number cannot carry ``2**64 - 1`` through
    an IEEE double, ``int()`` would accept ``" +7 "``, ``"0x1f"``, and
    ``"1_000"``, and a float string would round. Exactly one spelling per
    value also means two archives of the same state are byte-identical."""
    if not isinstance(text, str):
        _checkpoint_error(
            where,
            f"{path} must be a canonical decimal string, got "
            f"{type(text).__name__}",
        )
    if (len(text) > _UINT64_MAX_DIGITS
            or _CANONICAL_UINT64.fullmatch(text) is None):
        _checkpoint_error(
            where,
            f"{path} must be a canonical decimal integer string (no sign, "
            f"no leading zeros, no separators, no exponent), got {text!r}",
        )
    value = int(text)
    if not 0 <= value <= _UINT64_MAX:
        _checkpoint_error(
            where,
            f"{path} must be in [0, {_UINT64_MAX}], got {value}",
        )
    return value


def _valid_generator_path(name):
    """Whether ``name`` is a well-formed dotted registration path — the
    shape ``named_generators()`` produces. Non-empty, no empty segment,
    so ``""``, ``"."``, ``"a."``, and ``".a"`` are all rejected before
    they can be compared against a live traversal."""
    return (
        isinstance(name, str)
        and name != ""
        and all(segment != "" for segment in name.split("."))
    )


def _live_generator_topology(model):
    """The live model's generator topology, as the loader and the saver
    both see it: the ordered canonical ``(name, generator)`` pairs, the
    ``id(generator) -> canonical name`` map, and the complete
    ``path -> canonical name`` alias map in full traversal order.

    Both walks come from the same deterministic cycle-safe pre-order
    traversal, so the canonical names, the alias order, and therefore the
    serialized manifest are functions of the model alone."""
    canonical = list(model.named_generators())
    canonical_by_id = {id(generator): name for name, generator in canonical}
    aliases = {
        path: canonical_by_id[id(generator)]
        for path, generator in model._named_generator_paths()
    }
    return canonical, canonical_by_id, aliases


def _generator_section(model, where):
    """The manifest's ``"generators"`` value for ``model``: ``None`` when
    it registers none, otherwise the ``keys``/``entries``/``aliases``
    object of §10.1.

    Every state is read in **one** locked snapshot, so the states written
    were true together, and a generator with a call reservation in flight
    (published or under construction) refuses the whole save rather than
    being captured mid-draw."""
    canonical, _, aliases = _live_generator_topology(model)
    if not canonical:
        return None
    try:
        snapshots = snapshot_generator_states(canonical)
    except RuntimeError as error:
        raise RuntimeError(f"{where}: {error}") from None
    entries = {
        name: {
            "algorithm": state["algorithm"],
            "algorithm_version": state["algorithm_version"],
            "seed": _encoded_uint64(state["seed"]),
            "calls": _encoded_uint64(state["calls"]),
        }
        for name, _, state in snapshots
    }
    return {
        "keys": [name for name, _ in canonical],
        "entries": entries,
        "aliases": aliases,
    }


# ---------------------------------------------------------------------------
# Saving
# ---------------------------------------------------------------------------


def _coherent_snapshot(model, optimizer, metadata, where):
    """Every live state family, captured as **one** snapshot, and encoded
    into the archive's ``{array_name: ndarray}`` payload (the manifest
    included, as its UTF-8 JSON ``uint8`` entry).

    The whole capture runs under the shared state-transaction guard
    (§10.8). Without it a concurrent participating state replacement could
    land between the model snapshot and the optimizer or generator
    snapshot, and the archive would describe a model that never existed —
    the save-side twin of the mixed-state problem the load transaction
    solves. The guard is held only for as long as it takes to build the
    complete immutable payload; NPZ encoding and the disk write happen
    after it is released, because they touch no live state.

    Every caller-owned snapshot goes through the explicit ``to_numpy()``
    serialization boundary and is closed in the ``finally`` — the arrays
    are independent copies, so nothing here aliases live state, and a
    failure leaves the model, optimizer, generators, and filesystem
    untouched."""
    arrays = {}
    model_state = None
    optimizer_state = None
    with state_transaction():
        try:
            model_state = model.state_dict()
            keys = list(model_state)
            entries = {}
            for index, key in enumerate(keys):
                snapshot = model_state[key]
                # Phase I, milestone I8 removed the version-2 float32
                # rejection that used to stand here. Version 3 declares
                # every entry's dtype explicitly and its loader validates
                # the declaration against both the payload and the live
                # destination, so a float32 parameter now serializes as a
                # float32 array under a manifest that says so — and reads
                # back bit for bit. Nothing is cast, widened, narrowed, or
                # guessed on either side; ``to_numpy()`` materializes at
                # the snapshot's own dtype and the entry records it.
                array_name = f"model::{index:06d}"
                arrays[array_name] = snapshot.to_numpy()
                entries[key] = {
                    "array": array_name,
                    "shape": list(snapshot.shape),
                    "dtype": snapshot.dtype,
                    "device": snapshot.device,
                }
            optimizer_section = None
            if optimizer is not None:
                optimizer_state = optimizer.state_dict()
                optimizer_section = {
                    "type": optimizer_state["optimizer"],
                    "state_format_version": optimizer_state["format_version"],
                    "lr": optimizer_state["lr"],
                    "parameters": [
                        {
                            "shape": list(entry["shape"]),
                            "dtype": entry["dtype"],
                            "device": entry["device"],
                        }
                        for entry in optimizer_state["parameters"]
                    ],
                }
                if isinstance(optimizer, NativeAdam):
                    optimizer_section["betas"] = list(
                        optimizer_state["betas"]
                    )
                    optimizer_section["eps"] = optimizer_state["eps"]
                    optimizer_section["step_counts"] = list(
                        optimizer_state["step_counts"]
                    )
                    # v3: each moment is an explicit entry object carrying
                    # its own shape/dtype/device, rather than a bare
                    # archive name whose metadata the loader would have to
                    # infer positionally from "parameters" (design §16.2).
                    for label in ("m", "v"):
                        entries_out = []
                        for index, snapshot in enumerate(
                            optimizer_state[label]
                        ):
                            array_name = f"optimizer::{label}::{index:06d}"
                            arrays[array_name] = snapshot.to_numpy()
                            entries_out.append({
                                "array": array_name,
                                "shape": list(snapshot.shape),
                                "dtype": snapshot.dtype,
                                "device": snapshot.device,
                            })
                        optimizer_section[label] = entries_out
            # The generator section is built last but is not "extra": a
            # reservation in flight refuses the whole save here, before
            # the temporary file exists, so an ambiguous mid-draw state
            # can never reach an archive and the destination is never
            # touched. Its generator locks are taken *under* the guard
            # this function already holds — the universal order.
            generator_section = _generator_section(model, where)
            manifest = {
                "format": _FORMAT,
                "format_version": _FORMAT_VERSION,
                "model": {"keys": keys, "entries": entries},
                "optimizer": optimizer_section,
                "generators": generator_section,
                "metadata": metadata,
            }
            manifest_bytes = json.dumps(
                manifest, allow_nan=False
            ).encode("utf-8")
            arrays[_MANIFEST_ENTRY] = np.frombuffer(
                manifest_bytes, dtype=np.uint8
            )
        finally:
            if model_state is not None:
                for snapshot in model_state.values():
                    snapshot.close()
            if (optimizer_state is not None
                    and isinstance(optimizer, NativeAdam)):
                for label in ("m", "v"):
                    for snapshot in optimizer_state[label]:
                        snapshot.close()
    return arrays


def save_native_checkpoint(path, model, optimizer=None, metadata=None):
    """Save ``model`` (and optionally ``optimizer`` state and JSON
    ``metadata``) to one pickle-free native checkpoint archive at
    ``path``, written atomically through a temporary file. See the
    module docstring for the complete format and failure contract.
    Returns None."""
    where = "save_native_checkpoint()"
    path = _validated_path(path, where)
    _validate_model(model, where)
    _validate_optimizer(optimizer, model, where)
    metadata = _validated_metadata(
        {} if metadata is None else metadata, "metadata", frozenset()
    )
    if not isinstance(metadata, dict):
        raise TypeError(
            f"{where}: metadata must be None or a dict, got "
            f"{type(metadata).__name__}"
        )
    parent = os.path.dirname(path) or "."
    if not os.path.isdir(parent):
        raise ValueError(
            f"{where}: the destination directory does not exist: "
            f"{parent!r}"
        )
    if os.path.isdir(path):
        raise ValueError(
            f"{where}: the destination is a directory: {path!r}"
        )

    # One coherent snapshot of every live state family, then the write.
    arrays = _coherent_snapshot(model, optimizer, metadata, where)

    # Atomic write: a collision-safe temporary file in the destination
    # directory (same filesystem), np.savez onto the explicitly opened
    # handle (a file object, so NumPy can never silently append
    # ".npz"), flush + close, then one os.replace. On failure the
    # temporary file is removed and an existing destination is never
    # touched; on success no temporary file remains.
    fd, temporary_path = tempfile.mkstemp(
        dir=parent, prefix=os.path.basename(path) + ".", suffix=".tmp"
    )
    try:
        with os.fdopen(fd, "wb") as handle:
            np.savez(handle, **arrays)
            handle.flush()
        os.replace(temporary_path, path)
    except BaseException:
        try:
            os.remove(temporary_path)
        except OSError:
            pass
        raise


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


class _DuplicateManifestKey(Exception):
    """A repeated key in one JSON object of the manifest."""

    def __init__(self, key):
        super().__init__(key)
        self.key = key


def _no_duplicate_keys(pairs):
    """``json.loads`` object hook rejecting a repeated key.

    Python's default keeps the *last* occurrence silently, which for the
    alias map (§10.5) would turn "this archive names one path twice, with
    two different canonical targets" into "this archive is fine" — a
    topology corruption that reads as valid. Applied to the whole
    manifest, since no section benefits from a silently dropped key."""
    seen = {}
    for key, value in pairs:
        if key in seen:
            raise _DuplicateManifestKey(key)
        seen[key] = value
    return seen


def _parse_manifest(archive, where):
    """The manifest as a validated plain dict: the entry must exist,
    be a 1-D uint8 array, decode as UTF-8, parse as JSON with no repeated
    object key, and be a JSON object carrying exactly the top-level
    fields of a **supported** format version. Returns the manifest; its
    ``format_version`` selects the field set and the loader's later
    generator behavior."""
    if _MANIFEST_ENTRY not in archive.files:
        _checkpoint_error(where, "the archive has no 'manifest' entry")
    manifest_array = archive[_MANIFEST_ENTRY]
    if manifest_array.dtype != np.uint8 or manifest_array.ndim != 1:
        _checkpoint_error(
            where,
            f"the 'manifest' entry must be a 1-D uint8 array of UTF-8 "
            f"JSON, got {manifest_array.dtype}/{manifest_array.ndim}-D",
        )
    try:
        manifest_text = manifest_array.tobytes().decode("utf-8")
    except UnicodeDecodeError as error:
        _checkpoint_error(
            where, "the manifest is not valid UTF-8", cause=error
        )
    try:
        manifest = json.loads(
            manifest_text, object_pairs_hook=_no_duplicate_keys
        )
    except json.JSONDecodeError as error:
        _checkpoint_error(
            where, "the manifest is not valid JSON", cause=error
        )
    except _DuplicateManifestKey as error:
        _checkpoint_error(
            where,
            f"the manifest repeats the object key {error.key!r}",
            cause=error,
        )
    if not isinstance(manifest, dict):
        _checkpoint_error(
            where,
            f"the manifest root must be a JSON object, got "
            f"{type(manifest).__name__}",
        )
    if "format" not in manifest or manifest["format"] != _FORMAT:
        _checkpoint_error(
            where,
            f"manifest['format'] must be {_FORMAT!r}, got "
            f"{manifest.get('format')!r}",
        )
    if "format_version" not in manifest:
        _checkpoint_error(
            where, "the manifest has no 'format_version' field"
        )
    version = manifest["format_version"]
    if isinstance(version, bool) or not isinstance(version, int):
        _checkpoint_error(
            where,
            f"manifest['format_version'] must be an int, got "
            f"{type(version).__name__}",
        )
    if version not in _SUPPORTED_FORMAT_VERSIONS:
        _checkpoint_error(
            where,
            f"manifest['format_version'] must be one of "
            f"{list(_SUPPORTED_FORMAT_VERSIONS)}, got {version!r}",
        )
    expected_keys = _MANIFEST_KEYS if version >= 2 else _MANIFEST_KEYS_V1
    if set(manifest) != expected_keys:
        _checkpoint_error(
            where,
            f"manifest fields do not match format version {version}: "
            f"missing {sorted(expected_keys - set(manifest))}, unexpected "
            f"{sorted(set(manifest) - expected_keys)}",
        )
    return manifest


def _validated_entry_dtype(declared, version, entry_path, where):
    """One declared ``"dtype"`` field, validated for ``version``.

    Two rules, in order (design §16.4, §16.5):

    1. it must be a **floating compute** dtype — ``cpp.normalize_dtype`` is
       the authority, so this file keeps no dtype table of its own and
       cannot drift from the storage layer;
    2. in a version-1 or version-2 archive it must additionally be exactly
       ``"float64"``, because those formats *are* float64 by definition.

    Rule 1 measured against ``cpp._normalize_internal_dtype`` — the
    **representation** table — until **Phase K, milestone K1 narrowed it**
    to the public floating registry (docs/native_integer_tensors_design.md
    §5.4, §21.2). The narrowing is behavior-preserving (no archive any
    shipped code could write has ever contained a non-floating entry, and
    the two validators accepted the same set on the day it landed) and it
    is the layer that makes a hand-written archive unable to declare an
    integer model, buffer, or optimizer entry. It is **one of three
    independent layers**, not the only one: a parameter cannot be
    non-floating, a buffer cannot be non-floating, and an archive entry
    cannot declare it — each with a different first authority and a
    different error, because a single layer is a single point of failure.

    Returns the canonical dtype string. Raises a checkpoint ``ValueError``
    naming the field, the value, and what was expected — before anything
    live is touched."""
    if not isinstance(declared, str):
        _checkpoint_error(
            where,
            f"{entry_path}['dtype'] must be a string, got "
            f"{type(declared).__name__}",
        )
    try:
        canonical = cpp.normalize_dtype(declared)
    except (TypeError, ValueError) as error:
        _checkpoint_error(
            where,
            f"{entry_path}['dtype'] is {declared!r}, which is not a dtype "
            f"a native checkpoint entry may declare: {error}",
            cause=error,
        )
    if version in _FLOAT64_ONLY_VERSIONS and canonical != "float64":
        _checkpoint_error(
            where,
            f"{entry_path}['dtype'] is {declared!r}, but native checkpoint "
            f"format version {version} stores float64 only. Versions "
            f"{list(_FLOAT64_ONLY_VERSIONS)} predate float32 and are "
            f"float64 formats permanently; nothing is cast, widened, or "
            f"guessed",
        )
    return canonical


def _validate_model_section(section, model, version, where):
    """Validate manifest['model'] against the live model: exact
    section shape, ordered string keys consistent with the entries
    mapping, key set equal to the model's canonical parameter names,
    and per-key shape/dtype/device equal to the live open parameter.
    Returns the ordered ``(key, entry, canonical_dtype)`` triples.

    The declared dtype is validated twice over, and both matter: once as a
    dtype the runtime represents and that this format version may carry
    (``_validated_entry_dtype``), and once as *exactly* the live
    destination's dtype. The second is the API contract — a checkpoint
    whose dtype differs from the model's is a rejection naming both, never
    a cast, a ``map_location``, or a closest match (design §16.4)."""
    if not isinstance(section, dict) or set(section) != _MODEL_SECTION_KEYS:
        _checkpoint_error(
            where,
            "manifest['model'] must be an object with exactly the "
            "fields 'keys' and 'entries'",
        )
    keys = section["keys"]
    entries = section["entries"]
    if not isinstance(keys, list) or not all(
        isinstance(key, str) for key in keys
    ):
        _checkpoint_error(
            where, "manifest['model']['keys'] must be a list of strings"
        )
    if not isinstance(entries, dict) or list(entries) != keys:
        _checkpoint_error(
            where,
            "manifest['model']['entries'] must map exactly the keys in "
            "manifest['model']['keys'], in the same order",
        )
    # The live state key space is parameters *and* persistent buffers
    # (v3.15): exactly what NativeModule.state_dict() snapshots. A
    # bufferless model yields only its parameters, so a pre-buffer
    # (Phase-C) checkpoint still validates unchanged.
    live = dict(model._state_named_tensors())
    missing = sorted(set(live) - set(keys))
    unexpected = sorted(set(keys) - set(live))
    if missing or unexpected:
        _checkpoint_error(
            where,
            f"checkpoint model keys do not match the model: missing "
            f"{missing}, unexpected {unexpected}",
        )
    dtypes = {}
    for key in keys:
        entry = entries[key]
        entry_path = f"manifest['model']['entries'][{key!r}]"
        if not isinstance(entry, dict) or set(entry) != _MODEL_ENTRY_KEYS:
            _checkpoint_error(
                where,
                f"{entry_path} must have exactly the fields 'array', "
                f"'shape', 'dtype', and 'device'",
            )
        if not isinstance(entry["array"], str):
            _checkpoint_error(
                where, f"{entry_path}['array'] must be a string"
            )
        shape = entry["shape"]
        if not isinstance(shape, list) or any(
            isinstance(dim, bool) or not isinstance(dim, int) or dim < 0
            for dim in shape
        ):
            _checkpoint_error(
                where,
                f"{entry_path}['shape'] must be a list of non-negative "
                f"ints, got {shape!r}",
            )
        destination = live[key]
        if tuple(shape) != destination.shape:
            _checkpoint_error(
                where,
                f"{entry_path}['shape'] is {tuple(shape)}, the model "
                f"state entry is {destination.shape}",
            )
        dtype = _validated_entry_dtype(
            entry["dtype"], version, entry_path, where
        )
        if dtype != destination.dtype:
            _checkpoint_error(
                where,
                f"{entry_path}['dtype'] is {entry['dtype']!r}, the "
                f"model state entry is {destination.dtype!r}",
            )
        if entry["device"] != destination.device:
            _checkpoint_error(
                where,
                f"{entry_path}['device'] is {entry['device']!r}, the "
                f"model state entry is {destination.device!r}",
            )
        dtypes[key] = dtype
    return [(key, entries[key], dtypes[key]) for key in keys]


def _validate_optimizer_section(section, optimizer, version, where):
    """Validate manifest['optimizer'] against the live optimizer under
    the strict presence rules, using the same validators the optimizer
    constructors and v3.13 loaders use — so after this passes (and the
    arrays validate), the final optimizer.load_state_dict() commit has
    no ordinary public failure path left.

    Returns the ordered per-moment ``(archive_name, shape, dtype)`` triples
    as ``{"m": [...], "v": [...]}`` — empty for NativeSGD and for an
    absent section. The **in-memory** optimizer state schema
    (``native_optimizer_state.FORMAT_VERSION``) is untouched at 1 and this
    function still validates against it; only the *file* encoding of the
    moments differs between archive versions (design §16.2 item 3)."""
    if section is None:
        if optimizer is not None:
            _checkpoint_error(
                where,
                "an optimizer was supplied, but this checkpoint "
                "contains no optimizer state",
            )
        return {"m": [], "v": []}
    if optimizer is None:
        _checkpoint_error(
            where,
            "this checkpoint contains optimizer state, but no "
            "optimizer was supplied — pass the optimizer to restore "
            "into (optimizer state is never silently discarded)",
        )
    if not isinstance(section, dict):
        _checkpoint_error(
            where,
            f"manifest['optimizer'] must be null or an object, got "
            f"{type(section).__name__}",
        )
    expected_tag = type(optimizer).__name__
    saved_tag = section.get("type")
    if saved_tag != expected_tag:
        _checkpoint_error(
            where,
            f"the checkpoint was saved with optimizer {saved_tag!r}, "
            f"but a {expected_tag!r} was supplied",
        )
    expected_keys = (
        _ADAM_SECTION_KEYS if expected_tag == "NativeAdam"
        else _SGD_SECTION_KEYS
    )
    if set(section) != expected_keys:
        _checkpoint_error(
            where,
            f"manifest['optimizer'] fields do not match a {expected_tag} "
            f"state: missing {sorted(expected_keys - set(section))}, "
            f"unexpected {sorted(set(section) - expected_keys)}",
        )
    state_version = section["state_format_version"]
    if (
        isinstance(state_version, bool)
        or not isinstance(state_version, int)
        or state_version != _STATE_FORMAT_VERSION
    ):
        _checkpoint_error(
            where,
            f"manifest['optimizer']['state_format_version'] must be "
            f"{_STATE_FORMAT_VERSION}, got {state_version!r}",
        )
    parameters = optimizer.parameters()
    try:
        if expected_tag == "NativeAdam":
            _validated_positive_real(section["lr"], "lr")
            _validated_betas(section["betas"])
            _validated_positive_real(section["eps"], "eps")
        else:
            _validated_lr(section["lr"])
        validate_parameter_metadata(section["parameters"], parameters, where)
        if expected_tag == "NativeAdam":
            validate_step_counts(
                section["step_counts"], len(parameters), where
            )
    except (TypeError, ValueError) as error:
        _checkpoint_error(
            where,
            f"invalid optimizer state in the manifest: {error}",
            cause=error,
        )
    moments = {"m": [], "v": []}
    if expected_tag != "NativeAdam":
        return moments
    for label in ("m", "v"):
        listed = section[label]
        field = f"manifest['optimizer'][{label!r}]"
        if not isinstance(listed, list):
            _checkpoint_error(
                where,
                f"{field} must be a list of "
                + ("moment entry objects" if version >= 3
                   else "archive array names"),
            )
        if len(listed) != len(parameters):
            _checkpoint_error(
                where,
                f"{field} names {len(listed)} arrays, the optimizer "
                f"stores {len(parameters)} parameters",
            )
        for index, item in enumerate(listed):
            entry_path = f"{field}[{index}]"
            parameter = parameters[index]
            if version >= 3:
                # v3: an explicit entry object, so a mismatch is a
                # rejection rather than a plausible read. Its dtype is
                # carried, never inferred from parameters[index] — that
                # inference works only while the two lists agree, which is
                # exactly what a malformed archive violates (design §16.2).
                if (not isinstance(item, dict)
                        or set(item) != _MOMENT_ENTRY_KEYS):
                    _checkpoint_error(
                        where,
                        f"{entry_path} must have exactly the fields "
                        f"'array', 'shape', 'dtype', and 'device'",
                    )
                if not isinstance(item["array"], str):
                    _checkpoint_error(
                        where, f"{entry_path}['array'] must be a string"
                    )
                shape = item["shape"]
                if not isinstance(shape, list) or any(
                    isinstance(dim, bool) or not isinstance(dim, int)
                    or dim < 0 for dim in shape
                ):
                    _checkpoint_error(
                        where,
                        f"{entry_path}['shape'] must be a list of "
                        f"non-negative ints, got {shape!r}",
                    )
                if tuple(shape) != parameter.shape:
                    _checkpoint_error(
                        where,
                        f"{entry_path}['shape'] is {tuple(shape)}, the "
                        f"optimizer's parameter is {parameter.shape}",
                    )
                dtype = _validated_entry_dtype(
                    item["dtype"], version, entry_path, where
                )
                if dtype != parameter.dtype:
                    _checkpoint_error(
                        where,
                        f"{entry_path}['dtype'] is {item['dtype']!r}, the "
                        f"optimizer's parameter is {parameter.dtype!r}",
                    )
                if item["device"] != parameter.device:
                    _checkpoint_error(
                        where,
                        f"{entry_path}['device'] is {item['device']!r}, "
                        f"the optimizer's parameter is "
                        f"{parameter.device!r}",
                    )
                moments[label].append(
                    (item["array"], parameter.shape, dtype)
                )
            else:
                # v1/v2: a bare archive name, with shape implied
                # positionally by "parameters" and dtype float64 by the
                # format's definition — never guessed from the payload.
                if not isinstance(item, str):
                    _checkpoint_error(
                        where,
                        f"{field} must be a list of archive array names",
                    )
                moments[label].append((
                    item,
                    tuple(section["parameters"][index]["shape"]),
                    "float64",
                ))
    return moments


def _validate_generator_section(manifest, model, where):
    """Validate the archive's generator topology against the **live**
    model, and return the ordered ``[(canonical_name, generator, state)]``
    the commit will load — empty when there is nothing to restore.

    Every check here is prevalidation (§10.5): it runs before any
    staging, so a topology mismatch is detected while the model, its
    buffers, the optimizer, and every generator are still completely
    untouched. Matching is **strict in both directions** — an archive may
    neither omit a generator or a registered path the model has, nor
    carry one it does not — and the comparison is against a real
    ``named_generators()`` traversal of the live model, never against a
    name list the caller supplies.

    ``state`` is the four-field mapping ``NativeGenerator.load_state``
    accepts, with ``seed`` and ``calls`` already parsed to exact ints."""
    version = manifest["format_version"]
    canonical, _, live_aliases = _live_generator_topology(model)
    live_names = [name for name, _ in canonical]
    live_by_name = dict(canonical)

    if version == 1:
        # §10.6: a v1 archive has no generator section and never will.
        # Loading one into a generator-bearing model must fail rather
        # than fabricate a seed, reset a counter to zero, or keep the
        # live stream while claiming an exact restore.
        if canonical:
            _checkpoint_error(
                where,
                f"this is a format-version 1 checkpoint, which carries no "
                f"generator state, but the model registers {live_names} — "
                f"a version-1 archive cannot restore them and no seed or "
                f"call counter is ever invented. Re-save this model to "
                f"format version {_FORMAT_VERSION}",
            )
        return []

    section = manifest["generators"]
    if section is None:
        if canonical:
            _checkpoint_error(
                where,
                f"this checkpoint records no generator state, but the "
                f"model registers {live_names}",
            )
        return []
    if not isinstance(section, dict) or set(section) != _GENERATOR_SECTION_KEYS:
        _checkpoint_error(
            where,
            "manifest['generators'] must be null or an object with "
            "exactly the fields 'keys', 'entries', and 'aliases'",
        )

    keys = section["keys"]
    entries = section["entries"]
    aliases = section["aliases"]
    if not isinstance(keys, list) or not all(
        _valid_generator_path(key) for key in keys
    ):
        _checkpoint_error(
            where,
            "manifest['generators']['keys'] must be a list of dotted "
            "generator names",
        )
    if not canonical:
        _checkpoint_error(
            where,
            f"this checkpoint carries generator state for {sorted(keys)}, "
            f"but the model registers none — generator state is never "
            f"silently discarded",
        )
    if len(set(keys)) != len(keys):
        duplicates = sorted({key for key in keys if keys.count(key) > 1})
        _checkpoint_error(
            where,
            f"manifest['generators']['keys'] repeats {duplicates}",
        )
    if not isinstance(entries, dict) or list(entries) != keys:
        _checkpoint_error(
            where,
            "manifest['generators']['entries'] must map exactly the keys "
            "in manifest['generators']['keys'], in the same order",
        )
    if not isinstance(aliases, dict):
        _checkpoint_error(
            where,
            f"manifest['generators']['aliases'] must be an object, got "
            f"{type(aliases).__name__}",
        )
    for path, target in aliases.items():
        if not _valid_generator_path(path):
            _checkpoint_error(
                where,
                f"manifest['generators']['aliases'] has the malformed "
                f"generator path {path!r}",
            )
        if not isinstance(target, str):
            _checkpoint_error(
                where,
                f"manifest['generators']['aliases'][{path!r}] must be a "
                f"canonical generator name, got {type(target).__name__}",
            )
        if target not in entries:
            _checkpoint_error(
                where,
                f"manifest['generators']['aliases'][{path!r}] names the "
                f"canonical generator {target!r}, which has no entry",
            )
    for key in keys:
        if key not in aliases:
            _checkpoint_error(
                where,
                f"manifest['generators']['aliases'] is missing the "
                f"canonical generator {key!r}; every canonical name must "
                f"appear, mapped to itself",
            )
        if aliases[key] != key:
            _checkpoint_error(
                where,
                f"manifest['generators']['aliases'][{key!r}] is "
                f"{aliases[key]!r}; a canonical generator must map to "
                f"itself",
            )
    # Defensive: with every canonical name present and self-mapped, an
    # unreferenced entry is unreachable. Asserted anyway rather than
    # argued, so relaxing either check above cannot silently admit a
    # canonical entry the topology never mentions.
    referenced = set(aliases.values())
    unreferenced = [key for key in keys if key not in referenced]
    if unreferenced:                            # pragma: no cover - defensive
        _checkpoint_error(
            where,
            f"manifest['generators'] carries canonical entries no alias "
            f"references: {unreferenced}",
        )
    # The relation must be one step. It cannot express a cycle by
    # construction (every alias target is a canonical key and every
    # canonical key maps to itself), and asserting it directly is
    # cheaper than relying on that argument staying true.
    for path, target in aliases.items():
        if aliases[target] != target:
            _checkpoint_error(
                where,
                f"manifest['generators']['aliases'] is not a one-step "
                f"map: {path!r} -> {target!r} -> {aliases[target]!r}",
            )

    # --- topology, against the live traversal.
    missing = sorted(set(live_names) - set(keys))
    unexpected = sorted(set(keys) - set(live_names))
    if missing or unexpected:
        _checkpoint_error(
            where,
            f"checkpoint canonical generator names do not match the "
            f"model: missing {missing}, unexpected {unexpected}",
        )
    missing_paths = sorted(set(live_aliases) - set(aliases))
    unexpected_paths = sorted(set(aliases) - set(live_aliases))
    if missing_paths or unexpected_paths:
        _checkpoint_error(
            where,
            f"checkpoint generator paths do not match the model: missing "
            f"{missing_paths}, unexpected {unexpected_paths}",
        )
    mismatched = [
        (path, aliases[path], live_aliases[path])
        for path in live_aliases
        if aliases[path] != live_aliases[path]
    ]
    if mismatched:
        detail = ", ".join(
            f"{path!r} is shared with {saved!r} in the checkpoint but "
            f"with {live!r} in the model"
            for path, saved, live in mismatched
        )
        _checkpoint_error(
            where,
            f"checkpoint generator sharing topology does not match the "
            f"model: {detail}. Two paths draw from one stream in the "
            f"archive exactly when they name the same canonical "
            f"generator, and restoring the states without the topology "
            f"would resume a different model",
        )

    # --- per-entry state, against the live generator.
    validated = []
    for name in keys:
        entry = entries[name]
        entry_path = f"manifest['generators']['entries'][{name!r}]"
        if not isinstance(entry, dict) or set(entry) != _GENERATOR_ENTRY_KEYS:
            _checkpoint_error(
                where,
                f"{entry_path} must have exactly the fields 'algorithm', "
                f"'algorithm_version', 'seed', and 'calls'",
            )
        generator = live_by_name[name]
        if entry["algorithm"] != _GENERATOR_ALGORITHM:
            _checkpoint_error(
                where,
                f"{entry_path}['algorithm'] is {entry['algorithm']!r}, "
                f"the model's generator uses {_GENERATOR_ALGORITHM!r}",
            )
        entry_version = entry["algorithm_version"]
        if (isinstance(entry_version, bool)
                or not isinstance(entry_version, int)
                or entry_version != _GENERATOR_ALGORITHM_VERSION):
            _checkpoint_error(
                where,
                f"{entry_path}['algorithm_version'] is "
                f"{entry_version!r}, the model's generator uses "
                f"{_GENERATOR_ALGORITHM_VERSION}",
            )
        seed = _decoded_uint64(entry["seed"], f"{entry_path}['seed']", where)
        calls = _decoded_uint64(
            entry["calls"], f"{entry_path}['calls']", where
        )
        validated.append((name, generator, {
            "algorithm": _GENERATOR_ALGORITHM,
            "algorithm_version": _GENERATOR_ALGORITHM_VERSION,
            "seed": seed,
            "calls": calls,
        }))

    # The reservation rule, checked here so a mid-draw generator refuses
    # the load before anything is staged. The commit rechecks it while
    # holding every target's lock, which is the check that cannot be
    # raced; this one gives the caller the early, unambiguous failure.
    try:
        snapshot_generator_states(
            [(name, generator) for name, generator, _ in validated]
        )
    except RuntimeError as error:
        raise RuntimeError(f"{where}: {error}") from None
    return validated


def _read_arrays(archive, references, where):
    """Read every referenced array, enforcing the cross-reference
    rules: no duplicate references, no missing arrays, no unreferenced
    extra entries, and every array exactly its **declared** dtype in
    native byte order (which also rules out object dtype) with the
    expected shape. ``references`` is an ordered list of
    ``(archive_name, expected_shape, expected_dtype, described_as)``.
    Returns ``{archive_name: ndarray}``.

    The dtype comes from the manifest rather than from the payload, so a
    dtype/payload disagreement is a rejection **in either direction** — a
    declared float32 backed by a float64 array and the reverse both fail
    here (design §16.4). Byte order is compared as part of the dtype
    identity, so a foreign-endian array fails too: TensorForge does not
    claim cross-endian portability and never silently byte-swaps
    (design §17.2)."""
    seen = set()
    for name, _, _, described_as in references:
        if name in seen:
            _checkpoint_error(
                where,
                f"archive entry {name!r} is referenced more than once "
                f"(also by {described_as})",
            )
        seen.add(name)
    files = set(archive.files)
    expected_files = seen | {_MANIFEST_ENTRY}
    missing = sorted(seen - files)
    unexpected = sorted(files - expected_files)
    if missing or unexpected:
        _checkpoint_error(
            where,
            f"archive entries do not match the manifest: missing "
            f"{missing}, unexpected {unexpected}",
        )
    arrays = {}
    for name, expected_shape, expected_dtype, described_as in references:
        try:
            array = archive[name]
        except Exception as error:  # e.g. a pickled/object entry
            _checkpoint_error(
                where,
                f"archive entry {name!r} ({described_as}) could not be "
                f"read without pickle",
                cause=error,
            )
        if array.dtype != np.dtype(expected_dtype):
            _checkpoint_error(
                where,
                f"archive entry {name!r} ({described_as}) must be "
                f"{expected_dtype}, got {array.dtype}",
            )
        if array.shape != expected_shape:
            _checkpoint_error(
                where,
                f"archive entry {name!r} ({described_as}) has shape "
                f"{array.shape}, expected {expected_shape}",
            )
        arrays[name] = array
    return arrays


def load_native_checkpoint(path, model, optimizer=None):
    """Load a native checkpoint saved by ``save_native_checkpoint``
    into ``model`` (and ``optimizer``, under the strict presence
    rules), and return the checkpoint's metadata as an independent
    plain dict. See the module docstring for the complete validation,
    staging/commit, cleanup, and failure contract."""
    where = "load_native_checkpoint()"
    path = _validated_path(path, where)
    _validate_model(model, where)
    _validate_optimizer(optimizer, model, where)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{where}: no checkpoint file at {path!r}"
        )
    if not os.path.isfile(path):
        raise ValueError(
            f"{where}: {path!r} is not a regular file"
        )

    # Phase 1 — parse and validate the complete archive, reading every
    # array, with no live mutation. The archive handle is always
    # closed; pickle is never enabled.
    try:
        archive = np.load(path, allow_pickle=False)
    except Exception as error:
        _checkpoint_error(
            where,
            f"{path!r} is not a valid native checkpoint archive",
            cause=error,
        )
    try:
        manifest = _parse_manifest(archive, where)
        version = manifest["format_version"]
        model_entries = _validate_model_section(
            manifest["model"], model, version, where
        )
        moment_entries = _validate_optimizer_section(
            manifest["optimizer"], optimizer, version, where
        )
        generator_entries = _validate_generator_section(manifest, model, where)
        references = [
            (entry["array"], tuple(entry["shape"]), dtype,
             f"model key {key!r}")
            for key, entry, dtype in model_entries
        ]
        optimizer_section = manifest["optimizer"]
        for label in ("m", "v"):
            for index, (name, shape, dtype) in enumerate(
                moment_entries[label]
            ):
                references.append(
                    (name, shape, dtype, f"optimizer {label}[{index}]")
                )
        arrays = _read_arrays(archive, references, where)
        # Metadata is validated by the **same** authority the saver uses
        # (Phase I, milestone I10). Until I10 this was a root-type check
        # only, which let one class of archive through: ``json.loads``
        # accepts the non-standard ``NaN``/``Infinity``/``-Infinity``
        # literals, so a hand-written manifest carrying one parsed cleanly
        # and was handed straight back to the caller — a value
        # ``save_native_checkpoint`` would have refused to write. Running
        # ``_validated_metadata`` here makes "what a load accepts" exactly
        # "what a save would have produced", by construction rather than by
        # two rule sets kept in step by hand.
        #
        # It stays in Phase 1, at the position the root-type check already
        # occupied, so every established error precedence is unchanged and
        # the rejection still happens before any staged NativeTensor, any
        # rollback snapshot, and any model, optimizer, or generator
        # mutation exists.
        metadata = manifest["metadata"]
        if not isinstance(metadata, dict):
            _checkpoint_error(
                where,
                f"manifest['metadata'] must be an object, got "
                f"{type(metadata).__name__}",
            )
        try:
            metadata = _validated_metadata(
                metadata, "manifest['metadata']", frozenset()
            )
        except (TypeError, ValueError) as error:
            # Re-raised in the loader's own style so the message names the
            # failing path and value and still identifies the operation.
            _checkpoint_error(where, str(error), cause=error)
    finally:
        archive.close()

    # Phase 2 — stage. Everything that can allocate or raise happens
    # here: the independent NativeTensor state (through the explicit
    # from_array entry boundary), the staged optimizer schema, and — for
    # the whole-checkpoint rollback — an independent owning snapshot of
    # every live target the commit will overwrite. A failure closes
    # everything staged; nothing live has been touched.
    staged_model = {}
    staged_moments = {"m": [], "v": []}
    rollback_snapshots = []
    # Every staged tensor is built at its **declared** dtype, which
    # ``_read_arrays`` has already proved the payload matches exactly. The
    # typed ingress is therefore a bit-for-bit copy of matching data, not a
    # conversion: there is no cast anywhere on this path, and a float64
    # archive still takes the identical route it always did (design §9.4).
    try:
        for key, entry, dtype in model_entries:
            staged_model[key] = NativeTensor._typed_from_array(
                arrays[entry["array"]], dtype
            )
        staged_optimizer = None
        if optimizer_section is not None:
            if isinstance(optimizer, NativeAdam):
                for label in ("m", "v"):
                    for name, _, dtype in moment_entries[label]:
                        staged_moments[label].append(
                            NativeTensor._typed_from_array(
                                arrays[name], dtype
                            )
                        )
                staged_optimizer = {
                    "format_version": _STATE_FORMAT_VERSION,
                    "optimizer": "NativeAdam",
                    "lr": float(optimizer_section["lr"]),
                    "betas": tuple(
                        float(beta) for beta in optimizer_section["betas"]
                    ),
                    "eps": float(optimizer_section["eps"]),
                    "parameters": tuple(
                        {
                            "shape": tuple(entry["shape"]),
                            "dtype": entry["dtype"],
                            "device": entry["device"],
                        }
                        for entry in optimizer_section["parameters"]
                    ),
                    "step_counts": tuple(
                        int(count)
                        for count in optimizer_section["step_counts"]
                    ),
                    "m": staged_moments["m"],
                    "v": staged_moments["v"],
                }
            else:
                staged_optimizer = {
                    "format_version": _STATE_FORMAT_VERSION,
                    "optimizer": "NativeSGD",
                    "lr": float(optimizer_section["lr"]),
                    "parameters": tuple(
                        {
                            "shape": tuple(entry["shape"]),
                            "dtype": entry["dtype"],
                            "device": entry["device"],
                        }
                        for entry in optimizer_section["parameters"]
                    ),
                }

        # --- rollback snapshots (§10.7 Phase 2). Every allocation the
        # rollback could ever need happens here, in the phase that is
        # allowed to fail, which is precisely what lets the rollback
        # itself be unfailable.
        #
        # The **rollback snapshots** are deliberately *not* taken here.
        # They must reflect the state at the real commit boundary, and
        # that boundary is inside the transaction's locks (§10.8): a
        # snapshot captured now could describe a model another thread
        # replaces before this commit starts, and rolling back to it would
        # undo work this load never touched. The transaction captures them
        # with both locks held and appends them to ``rollback_snapshots``,
        # which stays this function's to close on every path.
        staged_generators = [
            GeneratorStateEntry(label=name, generator=generator, state=state)
            for name, generator, state in generator_entries
        ]
    except BaseException:
        for staged in staged_model.values():
            staged.close()
        for label in ("m", "v"):
            for staged in staged_moments[label]:
                staged.close()
        for snapshot in rollback_snapshots:
            snapshot.close()
        raise

    # Phase 3 — commit, serialized. The transaction takes the universal
    # state-replacement lock order (the shared state-transaction guard,
    # then every unique target generator's lock in global id() order),
    # rechecks reservations, snapshots every live target at the boundary,
    # and commits model → optimizer → generators through their own
    # existing loaders inside **one** rollback guard. So any ordinary
    # synchronous exception (and any deliverable asynchronous one)
    # restores all four state families, and no other participating state
    # load — another checkpoint load, load_state_dict,
    # load_generator_state_dict, or an optimizer state load — can
    # interleave with it. Every staged tensor and every rollback snapshot
    # is closed in the finally on both paths, so native live storage
    # returns to its baseline whether the load committed or rolled back.
    try:
        _transaction.commit_checkpoint(_transaction.CheckpointPlan(
            model=model,
            staged_model=staged_model,
            optimizer=optimizer,
            staged_optimizer=staged_optimizer,
            generator_entries=staged_generators,
            owned_snapshots=rollback_snapshots,
        ))
    finally:
        for staged in staged_model.values():
            staged.close()
        for label in ("m", "v"):
            for staged in staged_moments[label]:
                staged.close()
        for snapshot in rollback_snapshots:
            snapshot.close()
    return metadata
