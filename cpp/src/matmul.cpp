// Matrix multiplication kernels: the strided tensor-core matmul native
// autograd uses, and the legacy raw-buffer naive/tiled pair the
// benchmarks compare. All are deliberately unoptimized reference loops
// (no blocking beyond the explicit tile, no SIMD, no BLAS).

#include "tf_internal.h"

using tf::Storage;
using tf::as_storage;

// out (m x p, contiguous row-major) = a (m x n) @ b (n x p), where each
// source element is addressed through its own strides and offset — which
// is what lets a transposed or narrowed view multiply directly without
// materializing: a[i, k] lives at a_offset + i*a_stride0 + k*a_stride1,
// whatever the layout. The naive triple loop, matching the reference.
TF_EXPORT void tf_core_matmul(
    const void* a_handle, const void* b_handle, void* dst_handle,
    int64_t m, int64_t n, int64_t p,
    int64_t a_stride0, int64_t a_stride1,
    int64_t b_stride0, int64_t b_stride1,
    int64_t a_offset, int64_t b_offset
) {
    TF_GUARD_BEGIN
    const double* a = as_storage(a_handle)->data;
    const double* b = as_storage(b_handle)->data;
    double* dst = as_storage(dst_handle)->data;
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
    TF_GUARD_END_VOID()
}

// Legacy naive matmul over plain row-major buffers: out (m x p) =
// a (m x n) @ b (n x p). The textbook triple loop, kept as the reference
// tf_matmul_tiled is measured against. No Storage handle, no allocation.
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

// Tiled (blocked) matmul — the classic cache-locality experiment. The
// naive kernel streams whole rows/columns per output value, reloading the
// same data for matrices bigger than cache; tiling processes small
// block x block sub-matrices that stay resident while reused, and the
// i-k-j loop order walks b and out sequentially. Still single-threaded,
// no SIMD, no BLAS: one idea, not a performance product.
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
