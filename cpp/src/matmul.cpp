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

// (The two compute paths that used to be defined here — the pre-H2 generic
// i-j-k triple loop and the H2 i-k-j row sweep — are now templates over the
// element type in tf_matmul_internal.h, beside the four-part numerical
// contract they answer to. Nothing about either loop nest, either ``k``
// order, or either accumulator changed in the move: ``T = double`` is the
// code Phase H measured, statement for statement. The definitions moved for
// the ordinary reason a template must, so both instantiations are available
// to the export below *and* to the CTests that compile this file directly.)

}  // namespace tf

namespace {

// -- Phase I, milestone I4: the one dispatch per exported call --------------
//
// One helper holding the **single** ``switch`` ``tf_core_matmul`` performs
// (design §8.1). It sits above the two compute paths and below the export,
// so the dtype decision is made exactly once — after validation and before
// any compute — and nothing beneath it branches on dtype again: not the H2
// predicate, not the row grouping, not the ``k`` loop, not the accumulator,
// and emphatically not any element.
//
// The predicate is untouched by dtype: it reads ``int64`` metadata only,
// which is exactly why it carries over unchanged and why both widths take
// the *same* path for the same layout.
template <class T>
void matmul_dispatch(
    const void* a_handle, const void* b_handle, void* dst_handle,
    int64_t m, int64_t n, int64_t p,
    int64_t a_stride0, int64_t a_stride1,
    int64_t b_stride0, int64_t b_stride1,
    int64_t a_offset, int64_t b_offset
) {
    const T* a = tf::storage_typed<T>(a_handle);
    const T* b = tf::storage_typed<T>(b_handle);
    T* dst = tf::storage_typed<T>(dst_handle);
    if (tf::matmul_prefers_row_sweep(m, n, p, b_stride1)) {
        tf::matmul_row_sweep(a, b, dst, m, n, p,
                             a_stride0, a_stride1, b_stride0,
                             a_offset, b_offset);
        return;
    }
    tf::matmul_generic_strided(a, b, dst, m, n, p,
                               a_stride0, a_stride1,
                               b_stride0, b_stride1,
                               a_offset, b_offset);
}

}  // namespace

// out (m x p, contiguous row-major) = a (m x n) @ b (n x p), each source
// element addressed through its own strides and offset.
//
// Phase H (H2): the layout decides which of the two shipped compute paths
// runs. The decision is made here, from the metadata this function was
// already given — it allocates nothing, mutates nothing, reads no global
// or environment state, and cannot fail; a layout that does not qualify
// simply takes the generic path. There is no exported selector and no way
// for a caller to override the choice.
//
// Phase I, milestone I4: dtype-general. All three handles must carry the
// **same** dtype — there is no casting, no promotion, and no mixed-dtype
// arithmetic anywhere in the runtime (design §9), so a float32 left operand
// with a float64 right one is an invalid *request* rather than a conversion
// opportunity, in any of the three positions. The rejection is recorded
// before anything is read or written, so a rejected call leaves the
// destination byte-for-byte unchanged.
//
// **float32 accumulates in float32** (design §10.1). The per-output ``k``
// order is preserved exactly at both widths, on both paths, so H2's
// four-part contract restates rather than weakens at binary32.
TF_EXPORT void tf_core_matmul(
    const void* a_handle, const void* b_handle, void* dst_handle,
    int64_t m, int64_t n, int64_t p,
    int64_t a_stride0, int64_t a_stride1,
    int64_t b_stride0, int64_t b_stride1,
    int64_t a_offset, int64_t b_offset
) {
    TF_GUARD_BEGIN
    // K1: the dtype-role guard runs first — an int64 operand is
    // a role error, never a promotion opportunity (§22.4).
    if (!tf::require_floating(
            "tf_core_matmul",
            {a_handle, b_handle, dst_handle})) {
        return;
    }
    if (!tf::require_matching_dtype(
            "tf_core_matmul", {a_handle, b_handle, dst_handle})) {
        return;
    }
    switch (tf::dispatch_dtype({a_handle, b_handle, dst_handle})) {
        case tf::Dtype::Float32:
            matmul_dispatch<float>(a_handle, b_handle, dst_handle, m, n, p,
                                   a_stride0, a_stride1, b_stride0, b_stride1,
                                   a_offset, b_offset);
            return;
        case tf::Dtype::Int64:
            // Unreachable: require_floating rejected an int64 operand
            // above; a return so int64 never reads as double.
            return;
        case tf::Dtype::Float64:
            break;
    }
    matmul_dispatch<double>(a_handle, b_handle, dst_handle, m, n, p,
                            a_stride0, a_stride1, b_stride0, b_stride1,
                            a_offset, b_offset);
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
