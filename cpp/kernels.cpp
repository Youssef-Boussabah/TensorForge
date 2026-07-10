// TensorForge experimental C++ backend — the kernels.
//
// Deliberately tiny: plain C-ABI functions over raw float64 buffers,
// so Python can load them with ctypes. No Python C-API, no pybind11,
// no NumPy headers — the wrapper on the Python side handles arrays,
// shapes, and validation; this file only does the arithmetic.

#include <cstdint>
#include <new>

#ifdef _WIN32
#define TF_EXPORT extern "C" __declspec(dllexport)
#else
#define TF_EXPORT extern "C"
#endif

// ---------------------------------------------------------------------------
// Native storage: a C++-owned float64 buffer behind an opaque handle.
//
// Python never sees the pointer inside — it holds the handle, moves
// data in and out through copy_from/copy_to, and must destroy the
// handle when done (the Python wrapper does this in close()). This is
// the storage half of a future tensor runtime; the shape/stride
// metadata layer is the other half.
// ---------------------------------------------------------------------------

namespace {
struct TfStorage {
    double* data;
    int64_t size;
};
}  // namespace

// Returns an opaque handle, or null if allocation fails.
// The buffer is zero-initialized for predictable behavior.
TF_EXPORT void* tf_storage_create(int64_t size) {
    if (size <= 0) {
        return nullptr;
    }
    double* data = new (std::nothrow) double[size]();
    if (data == nullptr) {
        return nullptr;
    }
    return new TfStorage{data, size};
}

// Safe to call with null (does nothing), and must be called exactly
// once per successful create — the Python wrapper guarantees that.
TF_EXPORT void tf_storage_destroy(void* handle) {
    if (handle == nullptr) {
        return;
    }
    TfStorage* storage = static_cast<TfStorage*>(handle);
    delete[] storage->data;
    delete storage;
}

TF_EXPORT int64_t tf_storage_size(const void* handle) {
    return static_cast<const TfStorage*>(handle)->size;
}

TF_EXPORT void tf_storage_fill(void* handle, double value) {
    TfStorage* storage = static_cast<TfStorage*>(handle);
    for (int64_t i = 0; i < storage->size; ++i) {
        storage->data[i] = value;
    }
}

// Multiply every element by a scalar factor, in place. A small storage
// primitive alongside tf_storage_fill; used by the mean reduction to
// scale a freshly summed output by 1/count in float64.
TF_EXPORT void tf_storage_scale(void* handle, double factor) {
    TfStorage* storage = static_cast<TfStorage*>(handle);
    for (int64_t i = 0; i < storage->size; ++i) {
        storage->data[i] *= factor;
    }
}

TF_EXPORT void tf_storage_copy_from(void* handle, const double* src) {
    TfStorage* storage = static_cast<TfStorage*>(handle);
    for (int64_t i = 0; i < storage->size; ++i) {
        storage->data[i] = src[i];
    }
}

TF_EXPORT void tf_storage_copy_to(const void* handle, double* dst) {
    const TfStorage* storage = static_cast<const TfStorage*>(handle);
    for (int64_t i = 0; i < storage->size; ++i) {
        dst[i] = storage->data[i];
    }
}

// ---------------------------------------------------------------------------
// Kernels over native tensor cores: read strided views directly from
// NativeStorage and write into fresh contiguous storage. Same odometer
// traversal as materialization, so contiguous and non-contiguous
// inputs (transposes, narrows) work identically.
// ---------------------------------------------------------------------------

namespace {

// Plain function pointers keep the binary walker generic without
// templates or registries.
typedef double (*TfBinaryOp)(double, double);
double tf_op_add(double x, double y) { return x + y; }
double tf_op_subtract(double x, double y) { return x - y; }
double tf_op_multiply(double x, double y) { return x * y; }
// ReLU's backward as a binary op over (input, upstream gradient): the
// gradient passes through where the forward input was positive and is
// blocked where relu clamped to zero. x == 0 blocks, matching the
// Python Tensor's (x > 0) * grad convention exactly.
double tf_op_relu_backward(double x, double u) { return x > 0.0 ? u : 0.0; }

// Contiguous fast path: when a logical view's strides are exactly the
// row-major strides for its shape, the odometer's source position
// sequence degenerates to offset, offset+1, ..., offset+numel-1. So a
// flat, index-free pointer loop reads the same elements in the same
// order as tf_core_binary above — bit-for-bit identical results, no
// per-axis carry loop and no counter allocation. Nonzero offsets are
// handled by starting from data + offset (a row slice keeps row-major
// strides but shifts the offset). Scalars fall out as numel == 1.
void tf_core_binary_contiguous(
    const void* a_handle, const void* b_handle, void* dst_handle,
    int64_t numel, int64_t a_offset, int64_t b_offset, TfBinaryOp op
) {
    const double* a = static_cast<const TfStorage*>(a_handle)->data + a_offset;
    const double* b = static_cast<const TfStorage*>(b_handle)->data + b_offset;
    double* dst = static_cast<TfStorage*>(dst_handle)->data;
    for (int64_t i = 0; i < numel; ++i) {
        dst[i] = op(a[i], b[i]);
    }
}

// Walk two strided sources in lockstep (same logical shape, separate
// strides/offsets) and write row-major contiguous output.
void tf_core_binary(
    const void* a_handle, const void* b_handle, void* dst_handle,
    const int64_t* shape, const int64_t* a_strides, const int64_t* b_strides,
    int64_t a_offset, int64_t b_offset, int64_t ndim, TfBinaryOp op
) {
    const double* a = static_cast<const TfStorage*>(a_handle)->data;
    const double* b = static_cast<const TfStorage*>(b_handle)->data;
    double* dst = static_cast<TfStorage*>(dst_handle)->data;
    if (ndim == 0) {
        dst[0] = op(a[a_offset], b[b_offset]);
        return;
    }
    int64_t total = 1;
    for (int64_t d = 0; d < ndim; ++d) {
        total *= shape[d];
    }
    int64_t* counter = new int64_t[ndim]();
    int64_t a_pos = a_offset;
    int64_t b_pos = b_offset;
    for (int64_t out = 0; out < total; ++out) {
        dst[out] = op(a[a_pos], b[b_pos]);
        for (int64_t d = ndim - 1; d >= 0; --d) {
            ++counter[d];
            a_pos += a_strides[d];
            b_pos += b_strides[d];
            if (counter[d] < shape[d]) {
                break;
            }
            counter[d] = 0;
            a_pos -= shape[d] * a_strides[d];
            b_pos -= shape[d] * b_strides[d];
        }
    }
    delete[] counter;
}

}  // namespace

TF_EXPORT void tf_core_relu(
    const void* src_handle, void* dst_handle,
    const int64_t* shape, const int64_t* strides,
    int64_t offset, int64_t ndim
) {
    const double* src = static_cast<const TfStorage*>(src_handle)->data;
    double* dst = static_cast<TfStorage*>(dst_handle)->data;
    if (ndim == 0) {
        dst[0] = src[offset] > 0.0 ? src[offset] : 0.0;
        return;
    }
    int64_t total = 1;
    for (int64_t d = 0; d < ndim; ++d) {
        total *= shape[d];
    }
    int64_t* counter = new int64_t[ndim]();
    int64_t src_pos = offset;
    for (int64_t out = 0; out < total; ++out) {
        dst[out] = src[src_pos] > 0.0 ? src[src_pos] : 0.0;
        for (int64_t d = ndim - 1; d >= 0; --d) {
            ++counter[d];
            src_pos += strides[d];
            if (counter[d] < shape[d]) {
                break;
            }
            counter[d] = 0;
            src_pos -= shape[d] * strides[d];
        }
    }
    delete[] counter;
}

// Contiguous fast path for relu: a flat pointer loop equivalent to
// tf_core_relu on a contiguous input, starting at data + offset. Same
// reasoning as tf_core_binary_contiguous; scalars are numel == 1.
TF_EXPORT void tf_core_relu_contiguous(
    const void* src_handle, void* dst_handle,
    int64_t numel, int64_t offset
) {
    const double* src = static_cast<const TfStorage*>(src_handle)->data + offset;
    double* dst = static_cast<TfStorage*>(dst_handle)->data;
    for (int64_t i = 0; i < numel; ++i) {
        dst[i] = src[i] > 0.0 ? src[i] : 0.0;
    }
}

TF_EXPORT void tf_core_add(
    const void* a, const void* b, void* dst,
    const int64_t* shape, const int64_t* a_strides, const int64_t* b_strides,
    int64_t a_offset, int64_t b_offset, int64_t ndim
) {
    tf_core_binary(a, b, dst, shape, a_strides, b_strides,
                   a_offset, b_offset, ndim, tf_op_add);
}

TF_EXPORT void tf_core_subtract(
    const void* a, const void* b, void* dst,
    const int64_t* shape, const int64_t* a_strides, const int64_t* b_strides,
    int64_t a_offset, int64_t b_offset, int64_t ndim
) {
    tf_core_binary(a, b, dst, shape, a_strides, b_strides,
                   a_offset, b_offset, ndim, tf_op_subtract);
}

TF_EXPORT void tf_core_multiply(
    const void* a, const void* b, void* dst,
    const int64_t* shape, const int64_t* a_strides, const int64_t* b_strides,
    int64_t a_offset, int64_t b_offset, int64_t ndim
) {
    tf_core_binary(a, b, dst, shape, a_strides, b_strides,
                   a_offset, b_offset, ndim, tf_op_multiply);
}

// ReLU backward over tensor cores: dst = upstream where x > 0, else 0.
// The one genuinely new kernel native autograd's first scope needs (the
// runtime has no compare/where to compose it from). It is just the
// generic binary odometer walking the forward input x and the upstream
// gradient in lockstep — same logical shape, each through its own
// strides/offset, so transposed/narrowed/nonzero-offset inputs work
// without materializing — writing a fresh row-major contiguous output.
TF_EXPORT void tf_core_relu_backward(
    const void* x, const void* upstream, void* dst,
    const int64_t* shape, const int64_t* x_strides, const int64_t* u_strides,
    int64_t x_offset, int64_t u_offset, int64_t ndim
) {
    tf_core_binary(x, upstream, dst, shape, x_strides, u_strides,
                   x_offset, u_offset, ndim, tf_op_relu_backward);
}

// Contiguous fast-path binary kernels: flat, index-free loops used when
// both operands are row-major contiguous. Each is the exact equivalent
// of its odometer counterpart above (same op, same element order),
// selected on the Python side by the contiguity metadata.
TF_EXPORT void tf_core_add_contiguous(
    const void* a, const void* b, void* dst,
    int64_t numel, int64_t a_offset, int64_t b_offset
) {
    tf_core_binary_contiguous(a, b, dst, numel, a_offset, b_offset, tf_op_add);
}

TF_EXPORT void tf_core_subtract_contiguous(
    const void* a, const void* b, void* dst,
    int64_t numel, int64_t a_offset, int64_t b_offset
) {
    tf_core_binary_contiguous(a, b, dst, numel, a_offset, b_offset, tf_op_subtract);
}

TF_EXPORT void tf_core_multiply_contiguous(
    const void* a, const void* b, void* dst,
    int64_t numel, int64_t a_offset, int64_t b_offset
) {
    tf_core_binary_contiguous(a, b, dst, numel, a_offset, b_offset, tf_op_multiply);
}

// Matrix multiplication over tensor cores: out (m x p, contiguous
// row-major) = a (m x n) @ b (n x p), where each source element is
// addressed through its own strides and offset. That is what lets a
// transposed or narrowed view multiply directly — no materialization:
// a[i, k] lives at a_offset + i*a_stride0 + k*a_stride1, whatever the
// layout. The naive triple loop, matching the reference matmul.
TF_EXPORT void tf_core_matmul(
    const void* a_handle, const void* b_handle, void* dst_handle,
    int64_t m, int64_t n, int64_t p,
    int64_t a_stride0, int64_t a_stride1,
    int64_t b_stride0, int64_t b_stride1,
    int64_t a_offset, int64_t b_offset
) {
    const double* a = static_cast<const TfStorage*>(a_handle)->data;
    const double* b = static_cast<const TfStorage*>(b_handle)->data;
    double* dst = static_cast<TfStorage*>(dst_handle)->data;
    for (int64_t i = 0; i < m; ++i) {
        for (int64_t j = 0; j < p; ++j) {
            double sum = 0.0;
            for (int64_t k = 0; k < n; ++k) {
                sum += a[a_offset + i * a_stride0 + k * a_stride1]
                     * b[b_offset + k * b_stride0 + j * b_stride1];
            }
            dst[i * p + j] = sum;
        }
    }
}

// Sum reduction over a tensor core: the dual of broadcasting. Where a
// broadcast reads one element into many output positions via zero READ
// strides, a reduction writes many input elements into one output cell
// via zero WRITE strides. The input is walked with the usual odometer
// (its shape/strides/offset, so transposed/narrowed/offset views work
// without materializing); alongside the input position we advance an
// output position by out_strides — the row-major stride of each KEPT
// axis in the output, or 0 for each REDUCED axis, so reduced axes
// accumulate into the same cell. dst is fresh zero-initialized storage
// (the additive identity), so a plain += accumulates. For axis=None
// every out_stride is 0 and everything lands in dst[0]. Deterministic
// row-major (input) order; no SIMD/FMA/Kahan/pairwise. Python chooses
// the output shape/strides (keepdims included) and does mean's scaling.
TF_EXPORT void tf_core_sum(
    const void* src_handle, void* dst_handle,
    const int64_t* shape, const int64_t* in_strides, const int64_t* out_strides,
    int64_t offset, int64_t ndim
) {
    const double* src = static_cast<const TfStorage*>(src_handle)->data;
    double* dst = static_cast<TfStorage*>(dst_handle)->data;
    if (ndim == 0) {  // scalar input: its single element is the whole sum
        dst[0] += src[offset];
        return;
    }
    int64_t total = 1;
    for (int64_t d = 0; d < ndim; ++d) {
        total *= shape[d];
    }
    int64_t* counter = new int64_t[ndim]();
    int64_t in_pos = offset;
    int64_t out_pos = 0;
    for (int64_t i = 0; i < total; ++i) {
        dst[out_pos] += src[in_pos];
        for (int64_t d = ndim - 1; d >= 0; --d) {
            ++counter[d];
            in_pos += in_strides[d];
            out_pos += out_strides[d];
            if (counter[d] < shape[d]) {
                break;
            }
            counter[d] = 0;
            in_pos -= shape[d] * in_strides[d];
            out_pos -= shape[d] * out_strides[d];
        }
    }
    delete[] counter;
}

// Materialize a strided view of the storage into a contiguous output
// buffer — the canonical tensor-runtime loop. The logical indices are
// walked like an odometer (last dimension fastest, i.e. row-major
// output order) while the source position is updated incrementally:
// stepping dimension d moves by strides[d], and when a dimension
// wraps, its full extent is subtracted before the next dimension
// steps. Bounds are validated on the Python side before this is
// called.
TF_EXPORT void tf_storage_materialize(
    const void* handle, double* dst,
    const int64_t* shape, const int64_t* strides,
    int64_t offset, int64_t ndim
) {
    const double* src = static_cast<const TfStorage*>(handle)->data;
    if (ndim == 0) {  // scalar view: one element at the offset
        dst[0] = src[offset];
        return;
    }
    int64_t total = 1;
    for (int64_t d = 0; d < ndim; ++d) {
        total *= shape[d];
    }
    int64_t* counter = new int64_t[ndim]();
    int64_t src_pos = offset;
    for (int64_t out = 0; out < total; ++out) {
        dst[out] = src[src_pos];
        for (int64_t d = ndim - 1; d >= 0; --d) {
            ++counter[d];
            src_pos += strides[d];
            if (counter[d] < shape[d]) {
                break;  // this dimension advanced; done
            }
            counter[d] = 0;  // wrap and carry into the next dimension
            src_pos -= shape[d] * strides[d];
        }
    }
    delete[] counter;
}

TF_EXPORT void tf_elementwise_add(
    const double* a, const double* b, double* out, int64_t n
) {
    for (int64_t i = 0; i < n; ++i) {
        out[i] = a[i] + b[i];
    }
}

TF_EXPORT void tf_elementwise_subtract(
    const double* a, const double* b, double* out, int64_t n
) {
    for (int64_t i = 0; i < n; ++i) {
        out[i] = a[i] - b[i];
    }
}

TF_EXPORT void tf_elementwise_multiply(
    const double* a, const double* b, double* out, int64_t n
) {
    for (int64_t i = 0; i < n; ++i) {
        out[i] = a[i] * b[i];
    }
}

// IEEE float64 division: x/0 gives +-inf and 0/0 gives NaN, the same
// values NumPy produces (NumPy additionally warns; this kernel does not).
TF_EXPORT void tf_elementwise_divide(
    const double* a, const double* b, double* out, int64_t n
) {
    for (int64_t i = 0; i < n; ++i) {
        out[i] = a[i] / b[i];
    }
}

TF_EXPORT void tf_relu(const double* a, double* out, int64_t n) {
    for (int64_t i = 0; i < n; ++i) {
        out[i] = a[i] > 0.0 ? a[i] : 0.0;
    }
}

// Naive matrix multiplication: out (m x p) = a (m x n) @ b (n x p),
// all row-major. The textbook triple loop — deliberately unoptimized
// (no blocking, no SIMD, no BLAS); this stays as the reference kernel
// that tf_matmul_tiled is measured against.
TF_EXPORT void tf_matmul(
    const double* a, const double* b, double* out,
    int64_t m, int64_t n, int64_t p
) {
    for (int64_t i = 0; i < m; ++i) {
        for (int64_t j = 0; j < p; ++j) {
            double sum = 0.0;
            for (int64_t k = 0; k < n; ++k) {
                sum += a[i * n + k] * b[k * p + j];
            }
            out[i * p + j] = sum;
        }
    }
}

// Tiled (blocked) matrix multiplication — the classic cache-locality
// optimization. The naive kernel streams entire rows/columns for every
// output value, so for matrices bigger than cache the same data is
// reloaded from memory over and over. Tiling processes small
// block x block sub-matrices that stay resident in cache while they
// are reused. Within a block, the i-k-j loop order walks b and out
// row-by-row (sequential memory), instead of striding down b's
// columns like the naive j-inner order does.
//
// Still single-threaded, no SIMD, no BLAS: an experiment in one idea,
// not a performance product.
TF_EXPORT void tf_matmul_tiled(
    const double* a, const double* b, double* out,
    int64_t m, int64_t n, int64_t p, int64_t block
) {
    for (int64_t i = 0; i < m * p; ++i) {
        out[i] = 0.0;  // accumulate into out, so it must start at zero
    }
    for (int64_t i0 = 0; i0 < m; i0 += block) {
        const int64_t i_end = i0 + block < m ? i0 + block : m;
        for (int64_t k0 = 0; k0 < n; k0 += block) {
            const int64_t k_end = k0 + block < n ? k0 + block : n;
            for (int64_t j0 = 0; j0 < p; j0 += block) {
                const int64_t j_end = j0 + block < p ? j0 + block : p;
                // One block x block tile; the min-clamped ends handle
                // dimensions that are not multiples of the block size.
                for (int64_t i = i0; i < i_end; ++i) {
                    for (int64_t k = k0; k < k_end; ++k) {
                        const double a_ik = a[i * n + k];
                        for (int64_t j = j0; j < j_end; ++j) {
                            out[i * p + j] += a_ik * b[k * p + j];
                        }
                    }
                }
            }
        }
    }
}
