// Matrix multiplication kernels: the strided tensor-core matmul native
// autograd uses — a generic reference path plus the Phase-H (H2)
// row-sweep fast path it dispatches to on layouts that qualify — and the
// legacy raw-buffer naive/tiled pair the benchmarks compare.
//
// The two raw-buffer kernels at the bottom (tf_matmul / tf_matmul_tiled)
// are **not** on any production path and H2 deliberately did not route
// production through them: they take plain row-major buffers rather than
// storage handles, cannot read a strided view, carry no guard or error
// contract, and tf_matmul_tiled accumulates into its destination after
// zeroing it — a full extra write pass over the output, which is exactly
// what H1 removed. They stay as the standing cache-blocking experiment
// their benchmark and tests measure.

#include "tf_internal.h"
#include "tf_matmul_internal.h"

using tf::Storage;
using tf::as_storage;

namespace tf {

bool matmul_prefers_row_sweep(int64_t m, int64_t n, int64_t p,
                              int64_t b_stride1) noexcept {
    (void)m;  // the row sweep is correct and useful for every m >= 0
    return b_stride1 == 1 && n >= 1 && p >= MATMUL_MIN_COLUMNS;
}

// The pre-H2 kernel, unchanged. Each source element is addressed through
// its own strides and offset — which is what lets a transposed or
// narrowed view multiply directly without materializing: a[i, k] lives at
// a_offset + i*a_stride0 + k*a_stride1, whatever the layout. The naive
// triple loop, and the reference every optimized result is compared
// against.
void matmul_generic_strided(
    const double* a, const double* b, double* dst,
    int64_t m, int64_t n, int64_t p,
    int64_t a_stride0, int64_t a_stride1,
    int64_t b_stride0, int64_t b_stride1,
    int64_t a_offset, int64_t b_offset
) noexcept {
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

// The H2 fast path: i-k-j, sweeping MATMUL_ROW_BLOCK destination rows at
// once. See tf_matmul_internal.h for the four-part numerical contract
// this shares with the generic kernel — identical accumulation order,
// bit identity on every non-NaN result, NaN-class equivalence, and NaN
// payload bits deliberately left outside the contract — and for why it is
// safe against an uninitialized destination.
void matmul_row_sweep(
    const double* a, const double* b, double* dst,
    int64_t m, int64_t n, int64_t p,
    int64_t a_stride0, int64_t a_stride1,
    int64_t b_stride0,
    int64_t a_offset, int64_t b_offset
) noexcept {
    for (int64_t i0 = 0; i0 < m; i0 += MATMUL_ROW_BLOCK) {
        const int64_t rows =
            (m - i0 < MATMUL_ROW_BLOCK) ? (m - i0) : MATMUL_ROW_BLOCK;
        // k == 0 — the assigning pass. Every element of every row in the
        // group is written here, before any accumulation reads it, which
        // is what makes an uninitialized destination safe (H1). The
        // explicit `0.0 +` reproduces the generic kernel's
        // `double sum = 0.0; sum += ...` exactly: it is not redundant,
        // because 0.0 + (-0.0) is +0.0 while -0.0 alone is not.
        {
            const double* b_row = b + b_offset;
            for (int64_t r = 0; r < rows; ++r) {
                double* out = dst + (i0 + r) * p;
                const double a_ik = a[a_offset + (i0 + r) * a_stride0];
                for (int64_t j = 0; j < p; ++j) {
                    out[j] = 0.0 + a_ik * b_row[j];
                }
            }
        }
        // k >= 1 — the accumulating passes, ascending, so every output
        // takes its products in the generic kernel's order.
        for (int64_t k = 1; k < n; ++k) {
            const double* b_row = b + b_offset + k * b_stride0;
            for (int64_t r = 0; r < rows; ++r) {
                double* out = dst + (i0 + r) * p;
                const double a_ik =
                    a[a_offset + (i0 + r) * a_stride0 + k * a_stride1];
                for (int64_t j = 0; j < p; ++j) {
                    out[j] += a_ik * b_row[j];
                }
            }
        }
    }
}

}  // namespace tf

// out (m x p, contiguous row-major) = a (m x n) @ b (n x p), each source
// element addressed through its own strides and offset.
//
// Phase H (H2): the layout decides which of the two shipped compute paths
// runs. The decision is made here, from the metadata this function was
// already given — it allocates nothing, mutates nothing, reads no global
// or environment state, and cannot fail; a layout that does not qualify
// simply takes the generic path. There is no exported selector and no way
// for a caller to override the choice.
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
    if (tf::matmul_prefers_row_sweep(m, n, p, b_stride1)) {
        tf::matmul_row_sweep(a, b, dst, m, n, p,
                             a_stride0, a_stride1, b_stride0,
                             a_offset, b_offset);
    } else {
        tf::matmul_generic_strided(a, b, dst, m, n, p,
                                   a_stride0, a_stride1,
                                   b_stride0, b_stride1,
                                   a_offset, b_offset);
    }
    TF_GUARD_END_VOID()
}

// Legacy naive matmul over plain row-major buffers: out (m x p) =
// a (m x n) @ b (n x p). The textbook triple loop, kept as the reference
// tf_matmul_tiled is measured against. No Storage handle, no allocation.
// Not a production path — see the file header.
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

// Tiled (blocked) matmul — the classic cache-locality experiment, kept
// exactly as it was. The naive kernel streams whole rows/columns per
// output value, reloading the same data for matrices bigger than cache;
// tiling processes small block x block sub-matrices that stay resident
// while reused, and the i-k-j loop order walks b and out sequentially.
// Still single-threaded, no SIMD, no BLAS: one idea, not a performance
// product.
//
// H2 measured this shape against the row sweep it adopted for production
// and found the blocking *slower* at every size tested; the win is the
// i-k-j order, not the tiles. It is retained here as the measured
// alternative, on no production path.
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
