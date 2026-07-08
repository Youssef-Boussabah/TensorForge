"""Explicit backend selection.

The Stage-1 entry point from docs/dispatch_design.md: name a backend,
get its object. Nothing selects a backend on the caller's behalf, and
nothing here loads the compiled native library (the native backend
object is constructible whether or not it is built).
"""

from tensorforge.backends.native_backend import NativeBackend
from tensorforge.backends.numpy_backend import NumpyBackend

# Stateless singletons — backend objects hold no per-call state.
_BACKENDS = {
    "numpy": NumpyBackend(),
    "native": NativeBackend(),
}


def available_backends():
    """The names that ``get_backend`` accepts. This is *registration*,
    not readiness: a name appears here even if that backend's compiled
    library is not built (call ``backend.available()`` for readiness)."""
    return tuple(_BACKENDS)


def get_backend(name):
    """Return the backend object registered under ``name``.

    Raises ValueError for an unknown name. The returned object is
    always usable for introspection (``name``, ``available()``,
    ``backend_info()``); operations may still raise at call time if
    the backend is unavailable.
    """
    try:
        return _BACKENDS[name]
    except KeyError:
        raise ValueError(
            f"unknown backend {name!r}; available: "
            f"{', '.join(available_backends())}"
        ) from None
