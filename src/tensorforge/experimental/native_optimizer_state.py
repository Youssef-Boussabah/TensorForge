"""The shared native optimizer state schema (Advanced C++ v3.13; see
docs/backend_experiments.md).

Private helpers behind ``NativeSGD.state_dict()`` /
``NativeAdam.state_dict()`` and their ``load_state_dict()`` twins —
**not** a public API, not exported from ``tensorforge.experimental``,
and deliberately not an optimizer base class: plain functions the two
optimizers call so the schema can never drift between them.

The schema (format version 1) is a plain in-memory Python dict:

- ``"format_version"`` — the int ``1``.
- ``"optimizer"`` — the exact type tag, ``"NativeSGD"`` or
  ``"NativeAdam"``, so a state can never load into the wrong optimizer.
- validated hyperparameters (each optimizer's own keys).
- ``"parameters"`` — a tuple of ``{"shape", "dtype", "device"}`` dicts,
  one per unique stored parameter, **in the optimizer's deterministic
  identity-deduplicated first-occurrence order**. Mapping across
  optimizer instances is purely positional: entry *i* describes — and
  on loading is validated against — the loading optimizer's *i*-th
  stored parameter. No Python ``id()``, pointer, memory address,
  module name, or object repr is ever serialized, and no parameter
  values, gradients, graph data, or closed-state flags appear.

Loading accepts a tuple **or** list wherever the schema emits a
sequence (a caller may legitimately rebuild the containers); every
other validation is exact — exact key set, exact format version, exact
tag, exact per-position shape/dtype/device — with deterministic errors
naming stable field paths (``state['parameters'][1]['shape']``), never
memory addresses. There is no ``strict=False``, no compatibility mode,
no casting, and no device movement. File serialization does not exist
at this layer (native checkpoint archives are v3.14).
"""

FORMAT_VERSION = 1


def parameter_metadata(parameters):
    """The ordered parameter metadata for ``state_dict()``: one fresh
    ``{"shape", "dtype", "device"}`` dict per stored parameter, as a
    tuple in stored order. The caller has preflighted every parameter
    as open."""
    return tuple(
        {
            "shape": parameter.shape,
            "dtype": parameter.dtype,
            "device": parameter.device,
        }
        for parameter in parameters
    )


def validate_state_schema(state, optimizer_tag, required_keys, where):
    """Validate the state container itself: a plain dict with exactly
    ``required_keys``, ``format_version == FORMAT_VERSION`` (an int,
    never bool), and the exact ``optimizer`` tag. Raises TypeError /
    ValueError naming ``where`` and the failing field; touches
    nothing."""
    if not isinstance(state, dict):
        raise TypeError(
            f"{where}: state must be a dict, got {type(state).__name__}"
        )
    provided = set(state)
    required = set(required_keys)
    if provided != required:
        missing = sorted(required - provided)
        unexpected = sorted(provided - required)
        raise ValueError(
            f"{where}: state keys do not match the schema: "
            f"missing {missing}, unexpected {unexpected}"
        )
    version = state["format_version"]
    if isinstance(version, bool) or not isinstance(version, int):
        raise TypeError(
            f"{where}: state['format_version'] must be an int, got "
            f"{type(version).__name__}"
        )
    if version != FORMAT_VERSION:
        raise ValueError(
            f"{where}: state['format_version'] must be {FORMAT_VERSION}, "
            f"got {version}"
        )
    tag = state["optimizer"]
    if tag != optimizer_tag:
        raise ValueError(
            f"{where}: state['optimizer'] must be {optimizer_tag!r}, "
            f"got {tag!r}"
        )


def validate_parameter_metadata(entries, parameters, where):
    """Validate ``state['parameters']`` against the optimizer's stored
    parameters, position by position: a tuple/list of exactly
    ``len(parameters)`` well-formed metadata dicts whose shape (a
    tuple/list of non-bool ints), dtype, and device exactly match the
    stored parameter at the same position — no casting, reshaping,
    broadcasting, or device movement. Raises naming the failing index
    and field; touches nothing. The caller has preflighted every
    stored parameter as open."""
    if not isinstance(entries, (tuple, list)):
        raise TypeError(
            f"{where}: state['parameters'] must be a tuple or list, got "
            f"{type(entries).__name__}"
        )
    if len(entries) != len(parameters):
        raise ValueError(
            f"{where}: state['parameters'] describes {len(entries)} "
            f"parameters, this optimizer stores {len(parameters)}"
        )
    for index, (entry, parameter) in enumerate(zip(entries, parameters)):
        if not isinstance(entry, dict):
            raise TypeError(
                f"{where}: state['parameters'][{index}] must be a dict, "
                f"got {type(entry).__name__}"
            )
        if set(entry) != {"shape", "dtype", "device"}:
            raise ValueError(
                f"{where}: state['parameters'][{index}] must have exactly "
                f"the keys 'shape', 'dtype', and 'device', got "
                f"{sorted(entry)}"
            )
        shape = entry["shape"]
        if not isinstance(shape, (tuple, list)) or any(
            isinstance(dim, bool) or not isinstance(dim, int)
            for dim in shape
        ):
            raise TypeError(
                f"{where}: state['parameters'][{index}]['shape'] must be "
                f"a tuple of ints, got {shape!r}"
            )
        if tuple(shape) != parameter.shape:
            raise ValueError(
                f"{where}: state['parameters'][{index}]['shape'] is "
                f"{tuple(shape)}, the stored parameter is "
                f"{parameter.shape}"
            )
        if entry["dtype"] != parameter.dtype:
            raise ValueError(
                f"{where}: state['parameters'][{index}]['dtype'] is "
                f"{entry['dtype']!r}, the stored parameter is "
                f"{parameter.dtype!r}"
            )
        if entry["device"] != parameter.device:
            raise ValueError(
                f"{where}: state['parameters'][{index}]['device'] is "
                f"{entry['device']!r}, the stored parameter is "
                f"{parameter.device!r}"
            )


def validate_step_counts(counts, parameter_count, where):
    """Validate a step-count collection: a tuple/list of exactly
    ``parameter_count`` non-bool non-negative ints. Raises naming the
    failing index; touches nothing."""
    if not isinstance(counts, (tuple, list)):
        raise TypeError(
            f"{where}: state['step_counts'] must be a tuple or list, got "
            f"{type(counts).__name__}"
        )
    if len(counts) != parameter_count:
        raise ValueError(
            f"{where}: state['step_counts'] holds {len(counts)} counts, "
            f"this optimizer stores {parameter_count} parameters"
        )
    for index, count in enumerate(counts):
        if isinstance(count, bool) or not isinstance(count, int):
            raise TypeError(
                f"{where}: state['step_counts'][{index}] must be an int, "
                f"got {type(count).__name__}"
            )
        if count < 0:
            raise ValueError(
                f"{where}: state['step_counts'][{index}] must be >= 0, "
                f"got {count}"
            )
