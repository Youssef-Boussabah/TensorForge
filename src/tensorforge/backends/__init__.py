"""tensorforge.backends — experimental compiled backends.

Nothing here is used by the main framework (importing tensorforge
itself never touches this package). It provides the explicit backend
API for backend experiments — name a backend, get its object:

    from tensorforge.backends import get_backend, available_backends

    numpy_backend = get_backend("numpy")
    native_backend = get_backend("native")

Selection is always explicit; nothing routes operations implicitly.
See docs/dispatch_design.md.
"""

from tensorforge.backends.registry import available_backends, get_backend

__all__ = ["get_backend", "available_backends"]
