"""Experimental C++ backend.

A ctypes wrapper around tiny compiled kernels (see cpp/ at the repo
root). This is a proof of concept that Python TensorForge can call
compiled code — it is NOT wired into Tensor or autograd, and the
normal framework never imports it.

All kernels are float64-only. Binary operations require identical
shapes; broadcasting is deliberately not supported.

Importing this module always succeeds. The compiled library is loaded
lazily: check ``is_available()`` to see whether it can be used, and
call ``backend_info()`` for a summary. Calling a math kernel while the
backend is unbuilt raises ImportError with build instructions.
"""

import ctypes
import platform
from pathlib import Path

import numpy as np

_SUFFIX = {"Windows": ".dll", "Darwin": ".dylib"}.get(platform.system(), ".so")
_LIBRARY_PATH = Path(__file__).with_name("_tensorforge_cpp" + _SUFFIX)

# The supported kernels, in the order they were added.
KERNELS = (
    "elementwise_add",
    "elementwise_subtract",
    "elementwise_multiply",
    "elementwise_divide",
    "relu",
    "matmul",
    "matmul_tiled",
)

_BINARY_KERNELS = (
    "tf_elementwise_add",
    "tf_elementwise_subtract",
    "tf_elementwise_multiply",
    "tf_elementwise_divide",
)

# Operations available as NativeTensorCore methods — distinct from
# KERNELS, which lists the raw NumPy-buffer kernels.
TENSOR_CORE_KERNELS = ("relu", "add", "subtract", "multiply", "matmul")

# Supported native dtype/device metadata (v1.21). The native kernels are
# float64 CPU only, so these are the single legal values today. The tags
# are explicit and validated — a native tensor never claims a dtype/device
# the kernels cannot actually compute, and unsupported values are rejected
# at construction rather than silently coerced (see
# docs/native_dtype_device_metadata_design.md).
SUPPORTED_DTYPES = ("float64",)
SUPPORTED_DEVICES = ("cpu",)

_lib = None  # loaded lazily by _require_library()


def normalize_dtype(dtype=None):
    """Validate and canonicalize a native dtype tag.

    ``None`` means the default ``"float64"`` (the only supported dtype
    today). A non-string raises TypeError; a string outside
    ``SUPPORTED_DTYPES`` raises ValueError naming the offending value and
    the supported set. Pure Python — never touches the compiled library,
    so it is safe whether or not the backend is built."""
    if dtype is None:
        return "float64"
    if not isinstance(dtype, str):
        raise TypeError(f"dtype must be a string or None, got {dtype!r}")
    if dtype not in SUPPORTED_DTYPES:
        raise ValueError(
            f"unsupported dtype {dtype!r}; the native runtime supports "
            f"{SUPPORTED_DTYPES}"
        )
    return dtype


def normalize_device(device="cpu"):
    """Validate and canonicalize a native device tag.

    ``None`` means the default ``"cpu"`` (the only supported device
    today). A non-string raises TypeError; a string outside
    ``SUPPORTED_DEVICES`` raises ValueError naming the offending value and
    the supported set. Pure Python — never touches the compiled library."""
    if device is None:
        return "cpu"
    if not isinstance(device, str):
        raise TypeError(f"device must be a string or None, got {device!r}")
    if device not in SUPPORTED_DEVICES:
        raise ValueError(
            f"unsupported device {device!r}; the native runtime supports "
            f"{SUPPORTED_DEVICES}"
        )
    return device


def build_instructions():
    """The commands needed to build the experimental backend."""
    return (
        "The C++ backend is experimental. Build it from the repo root:\n"
        "    uv sync --group cpp   # only if you have no C++ compiler\n"
        "    uv run python cpp/build.py"
    )


def _load_library():
    if not _LIBRARY_PATH.exists():
        raise ImportError(
            f"The experimental C++ backend is not built "
            f"(missing {_LIBRARY_PATH.name}).\n" + build_instructions()
        )
    try:
        library = ctypes.CDLL(str(_LIBRARY_PATH))
    except OSError as error:
        raise ImportError(
            f"The experimental C++ backend library exists but failed to "
            f"load ({error}). Try rebuilding it.\n" + build_instructions()
        ) from error
    f64_array = np.ctypeslib.ndpointer(dtype=np.float64, flags="C_CONTIGUOUS")
    for name in _BINARY_KERNELS:
        kernel = getattr(library, name)
        kernel.argtypes = [f64_array, f64_array, f64_array, ctypes.c_int64]
        kernel.restype = None
    library.tf_relu.argtypes = [f64_array, f64_array, ctypes.c_int64]
    library.tf_relu.restype = None
    library.tf_matmul.argtypes = [
        f64_array, f64_array, f64_array,
        ctypes.c_int64, ctypes.c_int64, ctypes.c_int64,
    ]
    library.tf_matmul.restype = None
    library.tf_matmul_tiled.argtypes = [
        f64_array, f64_array, f64_array,
        ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64,
    ]
    library.tf_matmul_tiled.restype = None
    library.tf_storage_create.argtypes = [ctypes.c_int64]
    library.tf_storage_create.restype = ctypes.c_void_p
    library.tf_storage_destroy.argtypes = [ctypes.c_void_p]
    library.tf_storage_destroy.restype = None
    library.tf_storage_size.argtypes = [ctypes.c_void_p]
    library.tf_storage_size.restype = ctypes.c_int64
    library.tf_storage_fill.argtypes = [ctypes.c_void_p, ctypes.c_double]
    library.tf_storage_fill.restype = None
    library.tf_storage_copy_from.argtypes = [ctypes.c_void_p, f64_array]
    library.tf_storage_copy_from.restype = None
    library.tf_storage_copy_to.argtypes = [ctypes.c_void_p, f64_array]
    library.tf_storage_copy_to.restype = None
    i64_array = np.ctypeslib.ndpointer(dtype=np.int64, flags="C_CONTIGUOUS")
    library.tf_storage_materialize.argtypes = [
        ctypes.c_void_p, f64_array, i64_array, i64_array,
        ctypes.c_int64, ctypes.c_int64,
    ]
    library.tf_storage_materialize.restype = None
    library.tf_core_relu.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, i64_array, i64_array,
        ctypes.c_int64, ctypes.c_int64,
    ]
    library.tf_core_relu.restype = None
    # tf_core_relu_backward shares the binary-kernel signature: it walks
    # the forward input and the upstream gradient in lockstep.
    for name in (
        "tf_core_add", "tf_core_subtract", "tf_core_multiply",
        "tf_core_relu_backward",
    ):
        kernel = getattr(library, name)
        kernel.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
            i64_array, i64_array, i64_array,
            ctypes.c_int64, ctypes.c_int64, ctypes.c_int64,
        ]
        kernel.restype = None
    # Contiguous fast-path kernels (v1.14): flat, index-free loops that
    # take numel + offsets instead of shape/strides. Selected by
    # NativeTensorCore when the operands are row-major contiguous.
    library.tf_core_relu_contiguous.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64, ctypes.c_int64,
    ]
    library.tf_core_relu_contiguous.restype = None
    for name in (
        "tf_core_add_contiguous",
        "tf_core_subtract_contiguous",
        "tf_core_multiply_contiguous",
    ):
        kernel = getattr(library, name)
        kernel.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_int64, ctypes.c_int64, ctypes.c_int64,
        ]
        kernel.restype = None
    library.tf_core_matmul.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
    ] + [ctypes.c_int64] * 9
    library.tf_core_matmul.restype = None
    # Reduction kernels (v1.19): tf_core_sum scatter-accumulates a strided
    # input into a fresh contiguous output using per-input-axis output
    # write-strides (0 on reduced axes); tf_storage_scale does mean's
    # in-place 1/count scaling.
    library.tf_core_sum.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, i64_array, i64_array, i64_array,
        ctypes.c_int64, ctypes.c_int64,
    ]
    library.tf_core_sum.restype = None
    library.tf_storage_scale.argtypes = [ctypes.c_void_p, ctypes.c_double]
    library.tf_storage_scale.restype = None
    return library


def _require_library():
    """Load the compiled library on first use; raise helpfully if missing."""
    global _lib
    if _lib is None:
        _lib = _load_library()
    return _lib


def is_available():
    """True if the compiled backend can actually be loaded.

    This attempts a real load (cached after the first success), not
    just a file-existence check. Never raises.
    """
    try:
        _require_library()
    except ImportError:
        return False
    return True


def list_kernels():
    """The experimental kernels this backend provides, in stable order."""
    return KERNELS


def backend_info():
    """A small metadata dictionary describing the experimental backend."""
    return {
        "name": "cpp",
        "experimental": True,
        "available": is_available(),
        "kernels": KERNELS,
        "storage_object": "NativeStorage",
        "tensor_view": "NativeTensorView",
        "tensor_core": "NativeTensorCore",
        "tensor_core_kernels": TENSOR_CORE_KERNELS,
        "dtype": "float64",
        "device": "cpu",
        "supported_dtypes": SUPPORTED_DTYPES,
        "supported_devices": SUPPORTED_DEVICES,
        "tensor_integration": False,
        "autograd_integration": False,
        "build_instructions": build_instructions(),
    }


class NativeStorage:
    """A C++-owned float64 buffer — the storage half of a future
    tensor runtime prototype.

    Not a Tensor: it has a size but no shape, no strides, and no
    connection to Tensor/autograd. Data moves in and out by copy
    (``copy_from`` / ``to_numpy``); the raw native pointer is never
    exposed. Call ``close()`` (or use it as a context manager) to
    release the native memory; operations on a closed storage raise
    RuntimeError, and closing twice is safe.
    """

    def __init__(self, size, dtype=None, device="cpu"):
        self._handle = None  # so a failed __init__ still __del__s safely
        if not isinstance(size, (int, np.integer)) or isinstance(size, bool) or size <= 0:
            raise ValueError(f"size must be a positive int, got {size!r}")
        # dtype/device are validated *before* allocation, so an
        # unsupported request never leaks native memory.
        dtype = normalize_dtype(dtype)
        device = normalize_device(device)
        lib = _require_library()
        handle = lib.tf_storage_create(int(size))
        if not handle:
            raise MemoryError(f"could not allocate native storage of size {size}")
        self._lib = lib
        self._handle = handle
        self._size = int(size)
        self._dtype = dtype
        self._device = device

    @classmethod
    def from_array(cls, values, dtype=None, device="cpu"):
        """Create storage sized to ``values`` and copy them in.

        The input is always converted to contiguous float64 and flattened
        (the only element type the kernels compute); ``dtype``/``device``
        record the metadata and default to ``"float64"``/``"cpu"``.
        """
        array = np.ascontiguousarray(values, dtype=np.float64).ravel()
        # empty input fails size validation; dtype/device validated too
        storage = cls(int(array.size), dtype=dtype, device=device)
        storage.copy_from(array)
        return storage

    @property
    def size(self):
        """Number of float64 elements the storage holds."""
        return self._size

    @property
    def dtype(self):
        """The element type tag (``"float64"``). Readable after close."""
        return self._dtype

    @property
    def device(self):
        """The device tag (``"cpu"``). Readable after close."""
        return self._device

    def _require_open(self):
        if self._handle is None:
            raise RuntimeError("this NativeStorage has been closed")
        return self._handle

    def fill(self, value):
        """Set every element to ``value``."""
        self._lib.tf_storage_fill(self._require_open(), float(value))

    def copy_from(self, values):
        """Copy ``values`` into the storage.

        The input is converted to contiguous float64 and flattened; it
        must contain exactly ``size`` elements.
        """
        handle = self._require_open()
        array = np.ascontiguousarray(values, dtype=np.float64).ravel()
        if array.size != self._size:
            raise ValueError(
                f"copy_from needs exactly {self._size} values, got {array.size}"
            )
        self._lib.tf_storage_copy_from(handle, array)

    def to_numpy(self):
        """Return a new, independent 1-D float64 copy of the contents."""
        handle = self._require_open()
        out = np.empty(self._size, dtype=np.float64)
        self._lib.tf_storage_copy_to(handle, out)
        return out

    def close(self):
        """Release the native memory. Safe to call more than once."""
        if self._handle is not None:
            self._lib.tf_storage_destroy(self._handle)
            self._handle = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
        return False

    def __del__(self):
        # Defensive cleanup only — correctness never depends on when
        # (or whether) the garbage collector runs; use close().
        try:
            self.close()
        except Exception:
            pass

    def __repr__(self):
        state = "closed" if self._handle is None else f"size={self._size}"
        return f"NativeStorage({state})"


class NativeTensorView:
    """A logical view: NativeStorage plus shape/stride/offset metadata.

    This binds the two halves of the backend runtime prototype — the
    v0.7 layout contract and the v0.8 C++-owned buffer — into one
    object that knows which storage element each logical index means.
    Its first (and so far only) operation is contiguous
    materialization. Not a full Tensor: no math ops, no autograd, no
    connection to tensorforge.Tensor.

    Views never own the storage; closing the storage makes the view's
    operations raise. Reachable offsets are bounds-checked at
    construction, so a valid view can never read outside its storage
    (negative strides included).
    """

    def __init__(self, storage, shape, strides=None, offset=0):
        if not isinstance(storage, NativeStorage):
            raise TypeError(
                f"storage must be a NativeStorage, got {type(storage).__name__}"
            )
        storage._require_open()  # a closed storage cannot back a view
        info = shape_info(shape, strides=strides, offset=offset)

        # Bounds: each dimension contributes between 0 and
        # (dim - 1) * stride to the offset — negative strides make the
        # low end move. The whole reachable range must fit in storage.
        low = high = info["offset"]
        for dim, stride in zip(info["shape"], info["strides"]):
            contribution = (dim - 1) * stride
            if contribution >= 0:
                high += contribution
            else:
                low += contribution
        if low < 0 or high > storage.size - 1:
            raise ValueError(
                f"view reaches storage offsets [{low}, {high}], outside "
                f"the valid range [0, {storage.size - 1}]"
            )

        self._storage = storage
        self._shape = info["shape"]
        self._strides = info["strides"]
        self._offset = info["offset"]
        self._numel = info["numel"]
        self._contiguous = info["contiguous"]

    @classmethod
    def from_array(cls, values):
        """Create a contiguous view over new storage holding ``values``.

        The array's shape is preserved; its data is copied into a fresh
        NativeStorage.
        """
        array = np.ascontiguousarray(values, dtype=np.float64)
        storage = NativeStorage.from_array(array)
        return cls(storage, array.shape)

    @property
    def shape(self):
        return self._shape

    @property
    def strides(self):
        return self._strides

    @property
    def offset(self):
        return self._offset

    @property
    def ndim(self):
        return len(self._shape)

    @property
    def numel(self):
        return self._numel

    @property
    def contiguous(self):
        return self._contiguous

    def to_numpy(self):
        """Materialize the logical view into a fresh NumPy array of
        shape ``self.shape`` (row-major), copied element by element by
        the native materialization kernel."""
        handle = self._storage._require_open()
        out = np.empty(self._numel, dtype=np.float64)
        self._storage._lib.tf_storage_materialize(
            handle,
            out,
            np.asarray(self._shape, dtype=np.int64),
            np.asarray(self._strides, dtype=np.int64),
            self._offset,
            len(self._shape),
        )
        return out.reshape(self._shape)

    def contiguous_copy(self):
        """Materialize into a new NativeStorage in row-major order.

        Always copies, even if the view is already contiguous — the
        result is an independent storage the caller owns (and must
        close).
        """
        return NativeStorage.from_array(self.to_numpy().ravel())

    def __repr__(self):
        return (
            f"NativeTensorView(shape={self._shape}, strides={self._strides}, "
            f"offset={self._offset}, contiguous={self._contiguous})"
        )


class NativeTensorCore:
    """The first native tensor runtime object: an owned NativeStorage
    plus a NativeTensorView describing its layout, composed into one
    thing you can create, inspect, materialize, and release.

    Still not tensorforge.Tensor: no math operations, no autograd, no
    backend dispatch. It is the foundation those would build on.

    Ownership model: a core created by from_array/zeros/full OWNS its
    storage — its ``close()`` releases the native memory. View
    operations (reshape, transpose, T, narrow) return cores that
    BORROW the same storage: closing a borrowing view closes only that
    view, never the shared memory, so sibling views stay usable.
    Closing the owner releases the memory for every core sharing it,
    after which their data operations raise too. close() is
    idempotent everywhere; metadata properties stay readable after
    close, matching NativeStorage.size.
    """

    def __init__(self, storage, view, owns_storage=True):
        """Advanced constructor; prefer from_array / zeros / full.

        ``view`` must describe ``storage``. With ``owns_storage=True``
        this core's close() releases the storage; view operations pass
        False so shared storage is never freed by a view.
        """
        if not isinstance(storage, NativeStorage):
            raise TypeError(
                f"storage must be a NativeStorage, got {type(storage).__name__}"
            )
        if not isinstance(view, NativeTensorView) or view._storage is not storage:
            raise ValueError("view must be a NativeTensorView over the given storage")
        self._storage = storage
        self._view = view
        self._owns_storage = bool(owns_storage)
        self._closed = False

    # -- constructors -------------------------------------------------

    @classmethod
    def from_array(cls, values, dtype=None, device="cpu"):
        """A contiguous tensor holding a copy of ``values``, with the
        array's shape preserved. ``dtype``/``device`` record metadata and
        default to ``"float64"``/``"cpu"`` (the values are still coerced to
        float64); unsupported values are rejected."""
        array = np.ascontiguousarray(values, dtype=np.float64)
        # empty input fails here; dtype/device validated in the storage
        storage = NativeStorage.from_array(array, dtype=dtype, device=device)
        return cls(storage, NativeTensorView(storage, array.shape))

    @classmethod
    def zeros(cls, shape, dtype="float64", device="cpu"):
        """A row-major contiguous tensor of ``shape``, all zeros
        (native storage is zero-initialized, so no fill pass runs).
        ``dtype``/``device`` default to ``"float64"``/``"cpu"``;
        unsupported values are rejected."""
        count = numel(shape)  # validates shape by the v0.7 rules
        storage = NativeStorage(count, dtype=dtype, device=device)
        return cls(storage, NativeTensorView(storage, shape))

    @classmethod
    def full(cls, shape, value, dtype="float64", device="cpu"):
        """A row-major contiguous tensor of ``shape`` filled with
        ``value`` (anything float() accepts). ``dtype``/``device`` default
        to ``"float64"``/``"cpu"``; unsupported values are rejected."""
        tensor = cls.zeros(shape, dtype=dtype, device=device)
        tensor._storage.fill(float(value))
        return tensor

    # -- metadata (readable even after close) --------------------------

    @property
    def shape(self):
        return self._view.shape

    @property
    def strides(self):
        return self._view.strides

    @property
    def offset(self):
        return self._view.offset

    @property
    def ndim(self):
        return self._view.ndim

    @property
    def numel(self):
        return self._view.numel

    @property
    def contiguous(self):
        return self._view.contiguous

    @property
    def dtype(self):
        """The element type tag, delegated to this core's storage
        (``"float64"``). A view shares its storage, so it reports the same
        dtype as its owner. Readable after close, like ``shape``."""
        return self._storage.dtype

    @property
    def device(self):
        """The device tag, delegated to this core's storage (``"cpu"``). A
        view shares its storage, so it reports the same device as its
        owner. Readable after close, like ``shape``."""
        return self._storage.device

    @property
    def storage(self):
        """The owned NativeStorage (read-only access to the object)."""
        return self._storage

    @property
    def view(self):
        """The NativeTensorView describing this tensor's layout."""
        return self._view

    # -- data operations ------------------------------------------------

    def _require_open(self):
        if self._closed:
            raise RuntimeError("this NativeTensorCore has been closed")

    def to_numpy(self):
        """Materialize into a fresh, independent NumPy array of
        ``self.shape``."""
        self._require_open()
        return self._view.to_numpy()

    def contiguous_copy(self):
        """A new, independent NativeTensorCore with the same values in
        row-major contiguous storage. Always copies, even when this
        tensor is already contiguous."""
        self._require_open()
        return NativeTensorCore.from_array(self.to_numpy())

    # -- native compute (arithmetic happens in C++ over storage) ---------

    def _layout_arrays(self):
        return (
            np.asarray(self.shape, dtype=np.int64),
            np.asarray(self.strides, dtype=np.int64),
        )

    def relu(self):
        """max(x, 0) elementwise, computed by the native kernel reading
        this tensor's (possibly strided) view directly. Returns a new
        row-major contiguous NativeTensorCore.

        A contiguous input takes the flat fast-path kernel (a plain
        pointer loop); a strided view takes the generic odometer kernel.
        Both produce bit-for-bit identical results — the fast path is
        purely a traversal choice."""
        self._require_open()
        out = NativeTensorCore.zeros(self.shape, dtype=self.dtype, device=self.device)
        if self.contiguous:
            self._storage._lib.tf_core_relu_contiguous(
                self._storage._require_open(),
                out._storage._require_open(),
                self.numel, self.offset,
            )
            return out
        shape_arr, strides_arr = self._layout_arrays()
        self._storage._lib.tf_core_relu(
            self._storage._require_open(),
            out._storage._require_open(),
            shape_arr, strides_arr, self.offset, self.ndim,
        )
        return out

    def relu_backward(self, upstream):
        """The gradient of ``relu`` at this tensor's forward value:
        ``upstream`` where this tensor's element is ``> 0``, else ``0``
        (``x == 0`` blocks the gradient, the Python Tensor convention).

        A forward-shaped numerical kernel, not graph machinery — the core
        stays autograd-unaware; the NativeTensor layer calls this from its
        relu backward closure. Both operands may be strided views (each is
        read through its own strides/offset); shapes must match exactly
        (no broadcasting — the upstream gradient of an op always has the
        op's output shape). Returns a new row-major contiguous
        NativeTensorCore of this tensor's shape."""
        self._require_open()
        if not isinstance(upstream, NativeTensorCore):
            raise TypeError(
                f"relu_backward requires a NativeTensorCore upstream "
                f"gradient, got {type(upstream).__name__}"
            )
        upstream._require_open()
        self._require_matching_metadata(upstream, "relu_backward")
        if self.shape != upstream.shape:
            raise ValueError(
                f"relu_backward requires the upstream gradient shape to "
                f"match the input shape, got {upstream.shape} and {self.shape}"
            )
        out = NativeTensorCore.zeros(self.shape, dtype=self.dtype, device=self.device)
        shape_arr, x_strides = self._layout_arrays()
        u_strides = np.asarray(upstream.strides, dtype=np.int64)
        self._storage._lib.tf_core_relu_backward(
            self._storage._require_open(),
            upstream._storage._require_open(),
            out._storage._require_open(),
            shape_arr, x_strides, u_strides,
            self.offset, upstream.offset, self.ndim,
        )
        return out

    def _require_matching_metadata(self, other, op_name):
        """Both operands of a binary/matmul op must share dtype and
        device; there is no implicit promotion and no automatic device
        move (see docs/native_dtype_device_metadata_design.md §8). Raises
        ValueError naming both dtype/device pairs on a mismatch. With only
        float64/cpu constructible today this guard cannot yet fire, but it
        is the enforced contract native autograd (Phase B) builds on."""
        if self.dtype != other.dtype or self.device != other.device:
            raise ValueError(
                f"{op_name} requires matching dtype and device, got "
                f"{self.dtype}/{self.device} and {other.dtype}/{other.device}"
            )

    def _binary_core_op(self, other, kernel_name, op_name):
        """Shared plumbing for add/subtract/multiply over tensor cores,
        with a three-way traversal dispatch (v1.17):

        A. **Same shape, both contiguous** — the v1.14 flat fast-path
           kernel (``<kernel_name>_contiguous``): a plain pointer loop
           over numel.
        B. **Same shape, either strided** — the generic odometer kernel
           (``<kernel_name>``): walks the shared shape with each
           operand's real strides.
        C. **Differing but broadcast-compatible shapes** — NumPy-style
           broadcasting. The output shape is inferred by
           ``broadcast_shapes`` and each operand is read through
           *broadcast strides* (§ native_broadcasting_design.md): a real
           axis keeps its stride, a stretched or left-padded size-1 axis
           gets stride 0 so the odometer re-reads one element instead of
           advancing. The very same generic odometer kernel from path B
           consumes those strides — a zero stride is broadcasting, no
           expanded operand is materialized. Incompatible shapes raise a
           clear ``ValueError`` naming both shapes.

        Paths A and B are bit-for-bit unchanged from before; only when the
        shapes actually differ does broadcasting engage. The output is
        always freshly allocated row-major contiguous storage."""
        self._require_open()
        if not isinstance(other, NativeTensorCore):
            raise TypeError(
                f"{op_name} requires a NativeTensorCore operand, "
                f"got {type(other).__name__}"
            )
        other._require_open()
        self._require_matching_metadata(other, op_name)
        lib = self._storage._lib

        # Same-shape paths (A and B) — the exact-shape behavior, unchanged.
        if self.shape == other.shape:
            out = NativeTensorCore.zeros(self.shape, dtype=self.dtype, device=self.device)
            if self.contiguous and other.contiguous:
                getattr(lib, kernel_name + "_contiguous")(
                    self._storage._require_open(),
                    other._storage._require_open(),
                    out._storage._require_open(),
                    self.numel, self.offset, other.offset,
                )
                return out
            shape_arr, a_strides = self._layout_arrays()
            b_strides = np.asarray(other.strides, dtype=np.int64)
            getattr(lib, kernel_name)(
                self._storage._require_open(),
                other._storage._require_open(),
                out._storage._require_open(),
                shape_arr, a_strides, b_strides,
                self.offset, other.offset, self.ndim,
            )
            return out

        # Broadcasting path (C) — differing shapes. broadcast_shapes
        # raises (naming both shapes) if they are incompatible, before
        # any output is allocated.
        out_shape = broadcast_shapes(self.shape, other.shape)
        out = NativeTensorCore.zeros(out_shape, dtype=self.dtype, device=self.device)
        out_ndim = len(out_shape)
        shape_arr = np.asarray(out_shape, dtype=np.int64)
        a_strides = np.asarray(
            _broadcast_strides(self.shape, self.strides, out_shape),
            dtype=np.int64,
        )
        b_strides = np.asarray(
            _broadcast_strides(other.shape, other.strides, out_shape),
            dtype=np.int64,
        )
        getattr(lib, kernel_name)(
            self._storage._require_open(),
            other._storage._require_open(),
            out._storage._require_open(),
            shape_arr, a_strides, b_strides,
            self.offset, other.offset, out_ndim,
        )
        return out

    def add(self, other):
        """self + other elementwise, natively. Identical shapes, or
        NumPy-style broadcasting for compatible shapes (v1.17)."""
        return self._binary_core_op(other, "tf_core_add", "add")

    def subtract(self, other):
        """self - other elementwise, natively. Identical shapes, or
        NumPy-style broadcasting for compatible shapes (v1.17)."""
        return self._binary_core_op(other, "tf_core_subtract", "subtract")

    def multiply(self, other):
        """self * other elementwise, natively. Identical shapes, or
        NumPy-style broadcasting for compatible shapes (v1.17)."""
        return self._binary_core_op(other, "tf_core_multiply", "multiply")

    def matmul(self, other):
        """(m, n) @ (n, p) matrix multiplication over native storage.

        Both operands must be 2-D tensor cores; either may be a
        non-contiguous view (transposed, narrowed) — the kernel
        addresses each source through its own strides, so nothing is
        materialized first. Returns a new (m, p) row-major contiguous
        NativeTensorCore. The naive triple loop; no broadcasting.
        """
        self._require_open()
        if not isinstance(other, NativeTensorCore):
            raise TypeError(
                f"matmul requires a NativeTensorCore operand, "
                f"got {type(other).__name__}"
            )
        other._require_open()
        self._require_matching_metadata(other, "matmul")
        if self.ndim != 2:
            raise ValueError(
                f"matmul requires a 2-D left operand, got shape {self.shape}"
            )
        if other.ndim != 2:
            raise ValueError(
                f"matmul requires a 2-D right operand, got shape {other.shape}"
            )
        if self.shape[1] != other.shape[0]:
            raise ValueError(
                f"inner dimensions do not match: "
                f"{self.shape} @ {other.shape} (need (m, n) @ (n, p))"
            )
        m, n = self.shape
        p = other.shape[1]
        out = NativeTensorCore.zeros((m, p), dtype=self.dtype, device=self.device)
        self._storage._lib.tf_core_matmul(
            self._storage._require_open(),
            other._storage._require_open(),
            out._storage._require_open(),
            m, n, p,
            self.strides[0], self.strides[1],
            other.strides[0], other.strides[1],
            self.offset, other.offset,
        )
        return out

    # -- reductions (v1.19) ---------------------------------------------

    def sum(self, axis=None, keepdims=False):
        """Sum over ``axis`` (``None`` = all elements) natively, reading
        this tensor's (possibly strided) view directly. Returns a new
        owning row-major contiguous NativeTensorCore whose shape is
        ``reduce_shape(self.shape, axis, keepdims)``. Single integer or
        negative axis only; no broadcasting of the result, no autograd.

        Deterministic row-major accumulation order over the input; float
        sums are order-sensitive, so results match NumPy to a tolerance,
        not bit-for-bit (see docs/native_reductions_design.md)."""
        self._require_open()
        out_shape = reduce_shape(self.shape, axis, keepdims)  # validates axis/keepdims
        out = NativeTensorCore.zeros(out_shape, dtype=self.dtype, device=self.device)
        if axis is None:
            reduced = set(range(self.ndim))
        else:
            reduced = {_normalize_axis(axis, self.shape)}
        out_strides = _reduce_out_strides(self.shape, reduced, bool(keepdims), out_shape)
        self._storage._lib.tf_core_sum(
            self._storage._require_open(),
            out._storage._require_open(),
            np.asarray(self.shape, dtype=np.int64),
            np.asarray(self.strides, dtype=np.int64),
            np.asarray(out_strides, dtype=np.int64),
            self.offset, self.ndim,
        )
        return out

    def mean(self, axis=None, keepdims=False):
        """Mean over ``axis`` (``None`` = all elements) natively: the
        native ``sum`` scaled in place by ``1/count`` in float64, where
        ``count`` is ``numel`` for ``axis=None`` or ``shape[axis]`` for a
        single axis. Returns a new owning row-major contiguous
        NativeTensorCore. No NumPy touches the data; no autograd."""
        self._require_open()
        result = self.sum(axis=axis, keepdims=keepdims)
        if axis is None:
            count = self.numel
        else:
            count = self.shape[_normalize_axis(axis, self.shape)]
        # In-place native scale of the freshly summed output — no copy,
        # no NumPy round trip. count >= 1 always (dims are positive).
        result._storage._lib.tf_storage_scale(
            result._storage._require_open(), 1.0 / count
        )
        return result

    # -- view operations (metadata only: no data is copied) --------------

    def _view_core(self, shape, strides, offset):
        """A new core borrowing this core's storage with new layout."""
        view = NativeTensorView(self._storage, shape, strides=strides, offset=offset)
        return NativeTensorCore(self._storage, view, owns_storage=False)

    def reshape(self, new_shape):
        """A view of the same storage with ``new_shape`` (row-major).

        Metadata only — no copy. Requires a contiguous tensor (a
        non-contiguous layout cannot be reinterpreted by strides
        alone; materialize with contiguous_copy() first) and the same
        total number of elements.
        """
        self._require_open()
        if not self.contiguous:
            raise ValueError(
                "reshape requires a contiguous tensor; call "
                "contiguous_copy() first"
            )
        count = numel(new_shape)  # validates the shape by the v0.7 rules
        if count != self.numel:
            raise ValueError(
                f"cannot reshape {self.shape} ({self.numel} elements) "
                f"into {tuple(new_shape)} ({count} elements)"
            )
        return self._view_core(new_shape, None, self.offset)

    def transpose(self, *axes):
        """A view with permuted axes. Metadata only — no copy.

        With no arguments, all axes are reversed (NumPy behavior; a
        no-op for scalars and 1-D tensors). Explicit axes must be a
        complete permutation of range(ndim).
        """
        self._require_open()
        if len(axes) == 1 and isinstance(axes[0], (tuple, list)):
            axes = tuple(axes[0])
        if not axes:
            axes = tuple(reversed(range(self.ndim)))
        axes = _as_int_tuple(axes, "axes")
        if sorted(axes) != list(range(self.ndim)):
            raise ValueError(
                f"axes must be a permutation of range({self.ndim}), got {axes}"
            )
        new_shape = tuple(self.shape[axis] for axis in axes)
        new_strides = tuple(self.strides[axis] for axis in axes)
        return self._view_core(new_shape, new_strides, self.offset)

    @property
    def T(self):
        """transpose() with all axes reversed — NumPy's .T semantics,
        so (1, 0) for 2-D and a no-op for scalars and 1-D tensors."""
        return self.transpose()

    def narrow(self, dim, start, length):
        """A view keeping ``length`` positions of dimension ``dim``,
        beginning at ``start``. Metadata only — no copy: the shape
        shrinks in one dimension and the offset advances by
        ``start * strides[dim]``; strides are unchanged.

        ``length`` must be at least 1 (zero-size shapes are not
        supported). No step parameter in v1.1.
        """
        self._require_open()
        for name, value in (("dim", dim), ("start", start), ("length", length)):
            if not isinstance(value, (int, np.integer)) or isinstance(value, bool):
                raise TypeError(f"{name} must be an int, got {value!r}")
        if not 0 <= dim < self.ndim:
            raise ValueError(f"dim must be in [0, {self.ndim}), got {dim}")
        if start < 0 or length < 1 or start + length > self.shape[dim]:
            raise ValueError(
                f"narrow(dim={dim}, start={start}, length={length}) is out "
                f"of bounds for dimension size {self.shape[dim]}"
            )
        new_shape = tuple(
            length if axis == dim else size for axis, size in enumerate(self.shape)
        )
        new_offset = self.offset + start * self.strides[dim]
        return self._view_core(new_shape, self.strides, new_offset)

    # -- lifetime -------------------------------------------------------

    def close(self):
        """Close this core. Owners release the native storage; views
        only close themselves and leave shared storage untouched.
        Safe to call repeatedly."""
        self._closed = True
        if self._owns_storage:
            self._storage.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
        return False

    def __del__(self):
        # Defensive cleanup only — never rely on GC timing; use close().
        try:
            self.close()
        except Exception:
            pass

    def __repr__(self):
        state = ", closed" if self._closed else ""
        return (
            f"NativeTensorCore(shape={self.shape}, strides={self.strides}, "
            f"contiguous={self.contiguous}{state})"
        )


# ---------------------------------------------------------------------------
# Shape/stride metadata
#
# The Python-facing metadata layer that prepares the path for a later
# native tensor storage object: shape, strides, element counts,
# contiguity checks, and flat-offset math. Implemented in Python on
# purpose — the deliverable is the *contract* (what shapes/strides
# mean), and integer arithmetic gains nothing from a ctypes round
# trip. These helpers never touch the compiled library, so they are
# safe whether or not the backend is built.
#
# Conventions:
# - strides count ELEMENTS, not bytes (unlike numpy.ndarray.strides);
#   row_major_strides((2, 3, 4)) == (12, 4, 1).
# - dimensions must be positive ints; zero-size dimensions are
#   rejected in v0.7 (their stride conventions deserve their own
#   tested milestone).
# - the scalar shape () has strides (), ndim 0, and numel 1.
# ---------------------------------------------------------------------------


def _as_int_tuple(values, name):
    try:
        items = tuple(values)
    except TypeError:
        raise TypeError(
            f"{name} must be a sequence of ints, got {values!r}"
        ) from None
    for value in items:
        if not isinstance(value, (int, np.integer)) or isinstance(value, bool):
            raise TypeError(f"{name} must contain only ints, got {value!r}")
    return tuple(int(value) for value in items)


def _as_shape(shape):
    dims = _as_int_tuple(shape, "shape")
    for dim in dims:
        if dim <= 0:
            raise ValueError(
                f"shape dimensions must be positive ints, got {dims} "
                f"(zero-size dimensions are not supported in v0.7)"
            )
    return dims


def _as_offset(offset):
    if not isinstance(offset, (int, np.integer)) or isinstance(offset, bool):
        raise TypeError(f"offset must be an int, got {offset!r}")
    return int(offset)


def row_major_strides(shape):
    """Element strides for a row-major contiguous layout of ``shape``.

    The last dimension varies fastest: row_major_strides((2, 3, 4))
    is (12, 4, 1). The scalar shape () gives ().
    """
    dims = _as_shape(shape)
    strides = []
    running = 1
    for dim in reversed(dims):
        strides.append(running)
        running *= dim
    return tuple(reversed(strides))


def numel(shape):
    """Number of elements in ``shape``; 1 for the scalar shape ()."""
    dims = _as_shape(shape)
    count = 1
    for dim in dims:
        count *= dim
    return count


def is_contiguous_shape(shape, strides):
    """True if ``strides`` is exactly the row-major contiguous layout
    for ``shape``. A scalar () with strides () is contiguous."""
    dims = _as_shape(shape)
    stride_tuple = _as_int_tuple(strides, "strides")
    if len(stride_tuple) != len(dims):
        raise ValueError(
            f"shape and strides must have the same length, "
            f"got {len(dims)} and {len(stride_tuple)}"
        )
    return stride_tuple == row_major_strides(dims)


def flat_offset(indices, strides, offset=0):
    """Flat storage position of a logical index: offset + sum(i * s).

    Pure stride math — no shape is involved, so no bounds checking is
    performed, and negative indices or strides are allowed (real
    strided views use negative strides).
    """
    index_tuple = _as_int_tuple(indices, "indices")
    stride_tuple = _as_int_tuple(strides, "strides")
    if len(index_tuple) != len(stride_tuple):
        raise ValueError(
            f"indices and strides must have the same length, "
            f"got {len(index_tuple)} and {len(stride_tuple)}"
        )
    return _as_offset(offset) + sum(
        index * stride for index, stride in zip(index_tuple, stride_tuple)
    )


def shape_info(shape, strides=None, offset=0):
    """A small metadata dictionary describing one array layout.

    With ``strides=None`` the row-major contiguous strides are used
    (and ``contiguous`` is True by construction). Explicit strides are
    validated against the shape's length and checked for contiguity.
    """
    dims = _as_shape(shape)
    if strides is None:
        stride_tuple = row_major_strides(dims)
    else:
        stride_tuple = _as_int_tuple(strides, "strides")
        if len(stride_tuple) != len(dims):
            raise ValueError(
                f"shape and strides must have the same length, "
                f"got {len(dims)} and {len(stride_tuple)}"
            )
    return {
        "shape": dims,
        "strides": stride_tuple,
        "ndim": len(dims),
        "numel": numel(dims),
        "offset": _as_offset(offset),
        "contiguous": stride_tuple == row_major_strides(dims),
    }


def broadcast_shapes(shape_a, shape_b):
    """The NumPy-style broadcast of two shapes, or a clear ValueError.

    Shapes are aligned from the trailing axis; the shorter one is
    conceptually left-padded with leading 1s. Two extents are compatible
    when they are equal or one of them is 1, and the result extent is
    their max. The scalar shape () broadcasts against anything.

        broadcast_shapes((), (3, 4))          # (3, 4)
        broadcast_shapes((3, 1), (1, 4))      # (3, 4)
        broadcast_shapes((4,), (3, 4))        # (3, 4)  (left-pad to (1, 4))
        broadcast_shapes((1, 3, 1), (2, 1, 5))# (2, 3, 5)
        broadcast_shapes((2, 3), (4, 3))      # ValueError (2 vs 4)

    Pure Python — it never calls NumPy and never touches the compiled
    library, so it is safe and testable whether or not the backend is
    built. Incompatible shapes raise a ValueError naming both original
    shapes and the conflicting extents.
    """
    a = _as_shape(shape_a)  # validates positive-int dims (v0.7 rules)
    b = _as_shape(shape_b)
    rank = max(len(a), len(b))
    pa = (1,) * (rank - len(a)) + a  # left-pad with leading 1s
    pb = (1,) * (rank - len(b)) + b
    out = []
    for da, db in zip(pa, pb):
        if da == db or da == 1 or db == 1:
            out.append(max(da, db))
        else:
            raise ValueError(
                f"cannot broadcast shapes {a} and {b}: incompatible "
                f"dimensions {da} and {db} (neither is 1)"
            )
    return tuple(out)


def _broadcast_strides(shape, strides, out_shape):
    """Read-strides that stretch a (real) ``shape``/``strides`` operand
    over ``out_shape`` without materializing it.

    For each output axis: a real axis that carries a genuine extent keeps
    its real stride; a size-1 axis (or a leading axis introduced by
    left-padding) gets stride 0, so the odometer re-reads the same
    element instead of advancing — that is exactly broadcasting. Assumes
    ``out_shape`` is the broadcast of ``shape`` with the other operand
    (compatibility already checked by ``broadcast_shapes``)."""
    rank = len(out_shape)
    pad = rank - len(shape)
    result = [0] * rank  # leading padded axes stay 0
    for i, dim in enumerate(shape):
        if dim != 1:  # a stretched size-1 axis keeps stride 0
            result[pad + i] = strides[i]
    return result


def _normalize_axis(axis, shape):
    """Normalize a single reduction ``axis`` against ``shape``.

    Accepts a plain int (negative allowed, NumPy-style: ``axis + ndim``),
    validates its type and bounds, and returns the non-negative axis.
    Raises ``TypeError`` for a non-int axis and ``ValueError`` naming both
    the axis and the shape when out of bounds (including any integer axis
    on a scalar). Pure Python — no NumPy, no compiled library."""
    dims = _as_shape(shape)
    ndim = len(dims)
    if not isinstance(axis, (int, np.integer)) or isinstance(axis, bool):
        raise TypeError(f"axis must be None or an int, got {axis!r}")
    value = int(axis)
    normalized = value + ndim if value < 0 else value
    if normalized < 0 or normalized >= ndim:
        raise ValueError(
            f"axis {value} is out of bounds for a tensor of shape {dims} "
            f"(ndim {ndim})"
        )
    return normalized


def reduce_shape(shape, axis=None, keepdims=False):
    """The output shape of reducing ``shape`` over ``axis``.

    ``axis=None`` reduces every element; a single integer ``axis``
    (negative allowed) reduces one dimension. ``keepdims=True`` leaves
    each reduced axis as size 1, ``keepdims=False`` (default) removes it.

        reduce_shape((2, 3))                       # ()
        reduce_shape((2, 3), keepdims=True)        # (1, 1)
        reduce_shape((2, 3), axis=0)               # (3,)
        reduce_shape((2, 3), axis=1, keepdims=True)# (2, 1)
        reduce_shape((), axis=None)                # ()
        reduce_shape((2, 3, 4), axis=-1)           # (2, 3)

    Pure Python — never calls NumPy, never touches the compiled library,
    so it is testable whether or not the backend is built. Tuple/multiple
    axes are not supported yet (a single int or None only). Raises
    ``TypeError`` for a non-bool ``keepdims`` or non-int ``axis``, and
    ``ValueError`` naming both axis and shape for an out-of-bounds axis.
    """
    dims = _as_shape(shape)
    if not isinstance(keepdims, bool):
        raise TypeError(f"keepdims must be a bool, got {keepdims!r}")
    ndim = len(dims)
    if axis is None:
        return (1,) * ndim if keepdims else ()
    normalized = _normalize_axis(axis, dims)
    if keepdims:
        return tuple(1 if d == normalized else dims[d] for d in range(ndim))
    return tuple(dims[d] for d in range(ndim) if d != normalized)


def _reduce_out_strides(in_shape, reduced_axes, keepdims, out_shape):
    """Per-input-axis output write-strides for the sum kernel.

    For each input axis: 0 if it is reduced (so those elements
    accumulate into one output cell), otherwise the row-major stride of
    the axis it maps to in ``out_shape``. With ``keepdims`` the output
    keeps the reduced axes (as size 1), so input axis ``d`` maps to
    output axis ``d``; without it, the kept input axes map in order to the
    surviving output axes. Assumes ``out_shape == reduce_shape(in_shape,
    ...)`` for the same reduction."""
    out_full = row_major_strides(out_shape)
    result = [0] * len(in_shape)
    if keepdims:
        for d in range(len(in_shape)):
            result[d] = 0 if d in reduced_axes else out_full[d]
    else:
        out_index = 0
        for d in range(len(in_shape)):
            if d in reduced_axes:
                continue  # reduced axis: stays 0, no output axis consumed
            result[d] = out_full[out_index]
            out_index += 1
    return result


def _binary_op(kernel_name, a, b, name):
    """Shared plumbing for the binary kernels: convert both inputs to
    contiguous float64, require identical shapes, call the kernel."""
    lib = _require_library()
    a = np.ascontiguousarray(a, dtype=np.float64)
    b = np.ascontiguousarray(b, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError(
            f"the experimental C++ {name} requires identical "
            f"shapes (no broadcasting), got {a.shape} and {b.shape}"
        )
    out = np.empty_like(a)
    getattr(lib, kernel_name)(a, b, out, a.size)
    return out


def elementwise_add(a, b):
    """a + b elementwise, using the compiled C++ kernel."""
    return _binary_op("tf_elementwise_add", a, b, "elementwise_add")


def elementwise_subtract(a, b):
    """a - b elementwise, using the compiled C++ kernel."""
    return _binary_op("tf_elementwise_subtract", a, b, "elementwise_subtract")


def elementwise_multiply(a, b):
    """a * b elementwise, using the compiled C++ kernel."""
    return _binary_op("tf_elementwise_multiply", a, b, "elementwise_multiply")


def elementwise_divide(a, b):
    """a / b elementwise, using the compiled C++ kernel.

    IEEE float64 division: dividing by zero yields +-inf (or NaN for
    0/0), the same values NumPy produces — but without NumPy's runtime
    warning.
    """
    return _binary_op("tf_elementwise_divide", a, b, "elementwise_divide")


def relu(a):
    """max(a, 0) elementwise, using the compiled C++ kernel.

    Unary: accepts any shape, returns a new float64 array of the same
    shape.
    """
    lib = _require_library()
    a = np.ascontiguousarray(a, dtype=np.float64)
    out = np.empty_like(a)
    lib.tf_relu(a, out, a.size)
    return out


def _prepare_matmul_inputs(a, b, name):
    """Shared matmul validation: contiguous float64, strictly 2-D,
    compatible inner dimensions. Returns (a, b)."""
    a = np.ascontiguousarray(a, dtype=np.float64)
    b = np.ascontiguousarray(b, dtype=np.float64)
    if a.ndim != 2:
        raise ValueError(
            f"the experimental C++ {name} requires a 2-D left input, "
            f"got shape {a.shape}"
        )
    if b.ndim != 2:
        raise ValueError(
            f"the experimental C++ {name} requires a 2-D right input, "
            f"got shape {b.shape}"
        )
    if a.shape[1] != b.shape[0]:
        raise ValueError(
            f"inner dimensions do not match: "
            f"{a.shape} @ {b.shape} (need (m, n) @ (n, p))"
        )
    return a, b


def matmul(a, b):
    """(m, n) @ (n, p) matrix multiplication using the compiled C++
    kernel — the naive triple loop, kept as the reference that
    matmul_tiled is measured against. Correct but much slower than
    NumPy's BLAS-backed matmul.

    Strictly 2-D: vectors must be passed as (1, n) or (n, 1) matrices.
    Returns a new (m, p) float64 array.
    """
    lib = _require_library()
    a, b = _prepare_matmul_inputs(a, b, "matmul")
    m, n = a.shape
    p = b.shape[1]
    out = np.empty((m, p), dtype=np.float64)
    lib.tf_matmul(a, b, out, m, n, p)
    return out


def matmul_tiled(a, b, block_size=32):
    """(m, n) @ (n, p) matrix multiplication using the tiled C++
    kernel — an optimization experiment in cache blocking. Same
    contract as ``matmul``; ``block_size`` sets the tile edge length
    and any positive int works, including sizes that don't divide the
    matrix dimensions.

    Blocking improves memory locality over the naive loop, but this is
    still single-threaded scalar code: NumPy's BLAS may well remain
    faster. Returns a new (m, p) float64 array.
    """
    if not isinstance(block_size, int) or isinstance(block_size, bool) or block_size <= 0:
        raise ValueError(
            f"block_size must be a positive int, got {block_size!r}"
        )
    lib = _require_library()
    a, b = _prepare_matmul_inputs(a, b, "matmul_tiled")
    m, n = a.shape
    p = b.shape[1]
    out = np.empty((m, p), dtype=np.float64)
    lib.tf_matmul_tiled(a, b, out, m, n, p, block_size)
    return out
