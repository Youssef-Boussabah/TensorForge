"""Native checkpoint files (Advanced C++ v3.14; see
docs/backend_experiments.md).

``save_native_checkpoint(path, model, optimizer=None, metadata=None)``
and ``load_native_checkpoint(path, model, optimizer=None)`` persist a
``NativeModule``'s parameters, optionally one native optimizer's v3.13
state (``NativeSGD`` or ``NativeAdam``), and optional JSON-compatible
metadata to **one explicit, pickle-free NPZ archive** — and restore
them deterministically, so a saved native training run resumes
bit-for-bit. Everything follows the stable framework's no-pickle
serialization philosophy (src/tensorforge/serialization.py): arrays
and JSON only, ``allow_pickle=False`` on load, no code and no object
reconstruction.

The archive (format ``"tensorforge.native_checkpoint"``, format
version 1) contains:

- ``manifest`` — one JSON document encoded as UTF-8 bytes in a 1-D
  ``uint8`` array (never an object array, never pickle):
  ``{"format", "format_version", "model": {"keys", "entries"},
  "optimizer": null | {...}, "metadata": {...}}``. Model entries map
  each canonical state key to its archive array name plus exact
  shape/dtype/device; the optimizer section reproduces the v3.13
  in-memory schema (type tag, state format version, hyperparameters,
  positional parameter metadata, and — for NativeAdam — step counts
  and the m/v archive array names). No Python ``id()``, pointer,
  repr, gradient, parameter version, graph data, or closed flag is
  ever written.
- one float64 array per model parameter and per Adam moment, under
  deterministic zero-padded indexed names: ``model::000000`` … in
  model-key order, ``optimizer::m::000000`` … / ``optimizer::v::000000``
  … in the optimizer's positional order. The manifest maps semantic
  entries to array names explicitly; duplicate references, missing
  arrays, and unreferenced extra entries are all rejected on load.

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
optimizer through their existing ``state_dict()`` contracts, converts
each caller-owned snapshot through the explicit ``to_numpy()``
serialization boundary, and closes every snapshot in a ``finally`` —
then writes through a collision-safe temporary file in the destination
directory (``np.savez`` onto an explicitly opened handle, so NumPy can
never silently rename the file) and commits with ``os.replace``: an
existing destination is replaced atomically on success and remains
byte-intact on failure, and no temporary file survives either way
(same-filesystem atomicity only; no directory creation).

**Loading** is validate → stage → commit. Phase 1 validates with no
mutation: the live model and optimizer, then the complete archive —
opened with ``allow_pickle=False``; manifest presence, representation,
UTF-8, JSON, root type; exact format identity, version, and field
sets; strict optimizer presence/type; model keys against the live
model and every array's exact float64 dtype and shape against both
manifest and live destination; optimizer scalars/metadata/counters
through the same validators the optimizer constructors use; and the
full array cross-reference. Phase 2 stages independent ``NativeTensor``
copies (a failure closes them all). Phase 3 commits through the
existing public loaders only — ``NativeModule.load_state_dict`` then
``optimizer.load_state_dict`` — and closes every staged tensor in a
``finally``. Every ordinary failure therefore happens before any live
mutation, preserving model values, parameter versions, gradients,
optimizer hyperparameters, moments (by identity and value), counters,
registrations, and usability. Committed behavior is exactly the
components' documented contracts: model loading increments each
parameter version once and makes old value-sensitive retained graphs
stale; optimizer loading moves no versions and leaves graphs valid.
One narrow, honest limitation: the model commit and the optimizer
commit are two separate Python operations — an asynchronous
interruption (e.g. KeyboardInterrupt) between them can leave the model
restored while optimizer state remains old, and an interruption inside
either component keeps that component's own documented window. No
private rollback is manufactured.

``load_native_checkpoint`` returns the checkpoint's metadata as an
independent plain-Python dictionary (a fresh JSON parse — mutating it
affects nothing). Deliberately **not** implemented: scheduler state,
random-state capture or restoration, dataloader state, multiple
models/optimizers, partial loading, name-based remapping, checkpoint
merging, sharding, compression, encryption, URLs, dtype casting, or
device movement. Still float64/cpu only, experimental, and fully
separate from ``tensorforge.serialization``.
"""

import json
import math
import os
import tempfile

import numpy as np

from .native_adam import (
    NativeAdam,
    _validated_betas,
    _validated_positive_real,
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
_FORMAT_VERSION = 1
_MANIFEST_ENTRY = "manifest"
_MANIFEST_KEYS = {"format", "format_version", "model", "optimizer", "metadata"}
_MODEL_SECTION_KEYS = {"keys", "entries"}
_MODEL_ENTRY_KEYS = {"array", "shape", "dtype", "device"}
_SGD_SECTION_KEYS = {"type", "state_format_version", "lr", "parameters"}
_ADAM_SECTION_KEYS = _SGD_SECTION_KEYS | {"betas", "eps", "step_counts", "m", "v"}


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
    """``model`` must be a NativeModule whose parameters are all open.
    Stable framework modules are rejected by the type check — nothing
    is converted."""
    if not isinstance(model, NativeModule):
        raise TypeError(
            f"{where}: model must be a NativeModule, got "
            f"{type(model).__name__}"
        )
    for name, parameter in model.named_parameters():
        if parameter.closed:
            raise RuntimeError(
                f"{where}: model parameter {name!r} has been closed"
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
    str; cyclic containers are rejected via the ``seen`` id set."""
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
# Saving
# ---------------------------------------------------------------------------


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

    # Snapshot the model and optimizer through their existing state
    # contracts, convert every caller-owned snapshot through the
    # explicit to_numpy() serialization boundary, and close every
    # snapshot in the finally — the arrays are independent copies, so
    # nothing here aliases live state, and a failure leaves the model,
    # optimizer, and filesystem untouched.
    arrays = {}
    model_state = None
    optimizer_state = None
    try:
        model_state = model.state_dict()
        keys = list(model_state)
        entries = {}
        for index, key in enumerate(keys):
            snapshot = model_state[key]
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
                optimizer_section["betas"] = list(optimizer_state["betas"])
                optimizer_section["eps"] = optimizer_state["eps"]
                optimizer_section["step_counts"] = list(
                    optimizer_state["step_counts"]
                )
                for label in ("m", "v"):
                    names = []
                    for index, snapshot in enumerate(optimizer_state[label]):
                        array_name = f"optimizer::{label}::{index:06d}"
                        arrays[array_name] = snapshot.to_numpy()
                        names.append(array_name)
                    optimizer_section[label] = names
        manifest = {
            "format": _FORMAT,
            "format_version": _FORMAT_VERSION,
            "model": {"keys": keys, "entries": entries},
            "optimizer": optimizer_section,
            "metadata": metadata,
        }
        manifest_bytes = json.dumps(manifest, allow_nan=False).encode("utf-8")
        arrays[_MANIFEST_ENTRY] = np.frombuffer(manifest_bytes, dtype=np.uint8)
    finally:
        if model_state is not None:
            for snapshot in model_state.values():
                snapshot.close()
        if optimizer_state is not None and isinstance(optimizer, NativeAdam):
            for label in ("m", "v"):
                for snapshot in optimizer_state[label]:
                    snapshot.close()

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


def _parse_manifest(archive, where):
    """The manifest as a validated plain dict: the entry must exist,
    be a 1-D uint8 array, decode as UTF-8, parse as JSON, and be a
    JSON object with exactly the format-1 top-level fields."""
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
        manifest = json.loads(manifest_text)
    except json.JSONDecodeError as error:
        _checkpoint_error(
            where, "the manifest is not valid JSON", cause=error
        )
    if not isinstance(manifest, dict):
        _checkpoint_error(
            where,
            f"the manifest root must be a JSON object, got "
            f"{type(manifest).__name__}",
        )
    if set(manifest) != _MANIFEST_KEYS:
        _checkpoint_error(
            where,
            f"manifest fields do not match the format: missing "
            f"{sorted(_MANIFEST_KEYS - set(manifest))}, unexpected "
            f"{sorted(set(manifest) - _MANIFEST_KEYS)}",
        )
    if manifest["format"] != _FORMAT:
        _checkpoint_error(
            where,
            f"manifest['format'] must be {_FORMAT!r}, got "
            f"{manifest['format']!r}",
        )
    version = manifest["format_version"]
    if isinstance(version, bool) or not isinstance(version, int):
        _checkpoint_error(
            where,
            f"manifest['format_version'] must be an int, got "
            f"{type(version).__name__}",
        )
    if version != _FORMAT_VERSION:
        _checkpoint_error(
            where,
            f"manifest['format_version'] must be {_FORMAT_VERSION}, "
            f"got {version!r}",
        )
    return manifest


def _validate_model_section(section, model, where):
    """Validate manifest['model'] against the live model: exact
    section shape, ordered string keys consistent with the entries
    mapping, key set equal to the model's canonical parameter names,
    and per-key shape/dtype/device equal to the live open parameter.
    Returns the ordered (key, entry) pairs."""
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
    live = dict(model.named_parameters())
    missing = sorted(set(live) - set(keys))
    unexpected = sorted(set(keys) - set(live))
    if missing or unexpected:
        _checkpoint_error(
            where,
            f"checkpoint model keys do not match the model: missing "
            f"{missing}, unexpected {unexpected}",
        )
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
        parameter = live[key]
        if tuple(shape) != parameter.shape:
            _checkpoint_error(
                where,
                f"{entry_path}['shape'] is {tuple(shape)}, the model "
                f"parameter is {parameter.shape}",
            )
        if entry["dtype"] != parameter.dtype:
            _checkpoint_error(
                where,
                f"{entry_path}['dtype'] is {entry['dtype']!r}, the "
                f"model parameter is {parameter.dtype!r}",
            )
        if entry["device"] != parameter.device:
            _checkpoint_error(
                where,
                f"{entry_path}['device'] is {entry['device']!r}, the "
                f"model parameter is {parameter.device!r}",
            )
    return [(key, entries[key]) for key in keys]


def _validate_optimizer_section(section, optimizer, where):
    """Validate manifest['optimizer'] against the live optimizer under
    the strict presence rules, using the same validators the optimizer
    constructors and v3.13 loaders use — so after this passes (and the
    arrays validate), the final optimizer.load_state_dict() commit has
    no ordinary public failure path left."""
    if section is None:
        if optimizer is not None:
            _checkpoint_error(
                where,
                "an optimizer was supplied, but this checkpoint "
                "contains no optimizer state",
            )
        return
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
    if expected_tag == "NativeAdam":
        for label in ("m", "v"):
            names = section[label]
            if not isinstance(names, list) or not all(
                isinstance(name, str) for name in names
            ):
                _checkpoint_error(
                    where,
                    f"manifest['optimizer'][{label!r}] must be a list "
                    f"of archive array names",
                )
            if len(names) != len(parameters):
                _checkpoint_error(
                    where,
                    f"manifest['optimizer'][{label!r}] names "
                    f"{len(names)} arrays, the optimizer stores "
                    f"{len(parameters)} parameters",
                )


def _read_arrays(archive, references, where):
    """Read every referenced array, enforcing the cross-reference
    rules: no duplicate references, no missing arrays, no unreferenced
    extra entries, and every array exactly float64 (which also rules
    out object dtype) with the expected shape. ``references`` is an
    ordered list of ``(archive_name, expected_shape, described_as)``.
    Returns ``{archive_name: ndarray}``."""
    seen = set()
    for name, _, described_as in references:
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
    for name, expected_shape, described_as in references:
        try:
            array = archive[name]
        except Exception as error:  # e.g. a pickled/object entry
            _checkpoint_error(
                where,
                f"archive entry {name!r} ({described_as}) could not be "
                f"read without pickle",
                cause=error,
            )
        if array.dtype != np.float64:
            _checkpoint_error(
                where,
                f"archive entry {name!r} ({described_as}) must be "
                f"float64, got {array.dtype}",
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
        model_entries = _validate_model_section(
            manifest["model"], model, where
        )
        _validate_optimizer_section(manifest["optimizer"], optimizer, where)
        references = [
            (entry["array"], tuple(entry["shape"]), f"model key {key!r}")
            for key, entry in model_entries
        ]
        optimizer_section = manifest["optimizer"]
        if optimizer_section is not None and isinstance(optimizer, NativeAdam):
            for label in ("m", "v"):
                for index, name in enumerate(optimizer_section[label]):
                    shape = tuple(
                        optimizer_section["parameters"][index]["shape"]
                    )
                    references.append(
                        (name, shape, f"optimizer {label}[{index}]")
                    )
        arrays = _read_arrays(archive, references, where)
        metadata = manifest["metadata"]
        if not isinstance(metadata, dict):
            _checkpoint_error(
                where,
                f"manifest['metadata'] must be an object, got "
                f"{type(metadata).__name__}",
            )
    finally:
        archive.close()

    # Phase 2 — stage independent NativeTensor state through the
    # explicit from_array entry boundary. A failure closes every
    # staged tensor; nothing live has been touched.
    staged_model = {}
    staged_moments = {"m": [], "v": []}
    try:
        for key, entry in model_entries:
            staged_model[key] = NativeTensor.from_array(arrays[entry["array"]])
        staged_optimizer = None
        if optimizer_section is not None:
            if isinstance(optimizer, NativeAdam):
                for label in ("m", "v"):
                    for name in optimizer_section[label]:
                        staged_moments[label].append(
                            NativeTensor.from_array(arrays[name])
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
    except BaseException:
        for staged in staged_model.values():
            staged.close()
        for label in ("m", "v"):
            for staged in staged_moments[label]:
                staged.close()
        raise

    # Phase 3 — commit through the existing public loaders only, then
    # close every staged tensor. After the preflight above neither
    # loader has an ordinary public failure path left; the honest
    # asynchronous-interruption window between the two commits is
    # documented in the module docstring.
    try:
        model.load_state_dict(staged_model)
        if staged_optimizer is not None:
            optimizer.load_state_dict(staged_optimizer)
    finally:
        for staged in staged_model.values():
            staged.close()
        for label in ("m", "v"):
            for staged in staged_moments[label]:
                staged.close()
    return metadata
