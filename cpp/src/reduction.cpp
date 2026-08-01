// Reductions over native tensor cores: the sum reduction (the dual of
// broadcasting) and narrow's backward scatter (the dual of the sum).
// Python chooses every shape/stride/offset (keepdims, mean's scaling, and
// the narrow base offset included); these kernels only do the traversal
// and arithmetic.
//
// Phase H, milestone H6: the sum now ships **two** traversals behind the
// unchanged ``tf_core_sum`` export — the retained generic odometer, which
// is the reference path and the only one that can address a strided
// source, and a flat block traversal for the row-major layouts whose
// factorization is provable from the metadata the export already
// receives. The choice is made by ``tf::reduce_prefers_contiguous_blocks``
// and is a function of that metadata alone; a false answer is a fallback,
// never an error. See tf_reduction_internal.h and
// docs/native_cpu_performance_design.md §16.6. ``tf_core_narrow_backward``
// is deliberately outside H6's scope and keeps the odometer unchanged.

#include "tf_internal.h"
#include "tf_reduction_internal.h"

using tf::Storage;
using tf::as_storage;

namespace tf {

// -- the dispatch predicate (see tf_reduction_internal.h) -------------------

bool reduce_prefers_contiguous_blocks(
    const int64_t* shape, const int64_t* in_strides,
    const int64_t* out_strides, int64_t ndim,
    int64_t* outer, int64_t* mid, int64_t* inner
) noexcept {
    // 1. The source must be exactly row-major contiguous, right to left.
    int64_t expected = 1;
    for (int64_t d = ndim - 1; d >= 0; --d) {
        if (in_strides[d] != expected) {
            return false;
        }
        expected *= shape[d];
    }
    // 2. The reduced axes are those with a zero WRITE stride. They must
    //    form one contiguous, non-empty run of axis indices.
    int64_t first = ndim;
    int64_t last = -1;
    for (int64_t d = 0; d < ndim; ++d) {
        if (out_strides[d] == 0) {
            if (first == ndim) {
                first = d;
            }
            last = d;
        }
    }
    if (last < 0) {
        return false;  // nothing is reduced; not this path's business
    }
    for (int64_t d = first; d <= last; ++d) {
        if (out_strides[d] != 0) {
            return false;  // a kept axis interrupts the run
        }
    }
    // 3. The kept axes must carry exactly the row-major strides of the
    //    output formed by dropping that run.
    int64_t kept_expected = 1;
    for (int64_t d = ndim - 1; d >= 0; --d) {
        if (d >= first && d <= last) {
            continue;
        }
        if (out_strides[d] != kept_expected) {
            return false;
        }
        kept_expected *= shape[d];
    }
    // 4. Factorize. Adjacent axes of the same class collapse by
    //    multiplication, which conditions 1 and 3 have just proved sound.
    int64_t outer_extent = 1;
    int64_t mid_extent = 1;
    int64_t inner_extent = 1;
    for (int64_t d = 0; d < first; ++d) {
        outer_extent *= shape[d];
    }
    for (int64_t d = first; d <= last; ++d) {
        mid_extent *= shape[d];
    }
    for (int64_t d = last + 1; d < ndim; ++d) {
        inner_extent *= shape[d];
    }
    *outer = outer_extent;
    *mid = mid_extent;
    *inner = inner_extent;
    return true;
}

// -- the retained generic reference traversal -------------------------------

void sum_generic_strided(
    const double* src, double* dst,
    const int64_t* shape, const int64_t* in_strides,
    const int64_t* out_strides, int64_t offset, int64_t ndim,
    int64_t* counter
) noexcept {
    int64_t total = 1;
    for (int64_t d = 0; d < ndim; ++d) {
        total *= shape[d];
        counter[d] = 0;
    }
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
}

// -- the optimized flat block traversal -------------------------------------

void sum_contiguous_blocks(
    const double* src, double* dst,
    int64_t outer, int64_t mid, int64_t inner, int64_t offset
) noexcept {
    if (inner == 1) {
        // One contiguous ascending source run per destination cell. The
        // accumulator is seeded from the destination, so the
        // accumulate-into contract and the signed-zero behavior are
        // exactly the generic path's, and the running total never makes a
        // round trip through memory.
        const double* run = src + offset;
        for (int64_t o = 0; o < outer; ++o) {
            double accumulator = dst[o];
            for (int64_t m = 0; m < mid; ++m) {
                accumulator += run[m];
            }
            dst[o] = accumulator;
            run += mid;
        }
        return;
    }
    // A contiguous source row added elementwise into a contiguous
    // destination row, ``mid`` times per outer block. Distinct ``i`` are
    // distinct destination cells, so nothing here is a horizontal
    // reduction and no addition is reassociated.
    const double* in_row = src + offset;
    double* out_row = dst;
    for (int64_t o = 0; o < outer; ++o) {
        for (int64_t m = 0; m < mid; ++m) {
            for (int64_t i = 0; i < inner; ++i) {
                out_row[i] += in_row[i];
            }
            in_row += inner;
        }
        out_row += inner;
    }
}

}  // namespace tf

// Sum reduction: where a broadcast reads one element into many output
// positions via zero READ strides, a reduction writes many input elements
// into one output cell via zero WRITE strides. dst is fresh
// zero-initialized storage (the additive identity), so a plain +=
// accumulates; for axis=None every out_stride is 0 and everything lands
// in dst[0]. Deterministic row-major (input) order; no SIMD horizontal
// reduction, no FMA, no Kahan, no pairwise or tree summation.
//
// H6: the traversal is chosen from the metadata already in hand. A
// row-major source whose reduced axes form one contiguous run — every
// reduction the Python layer can currently express over a contiguous
// tensor — is walked with the flat block traversal; anything else keeps
// the generic odometer, which is the retained reference traversal and the
// only one that can address a transposed, narrowed, non-unit-strided, or
// broadcast source at all. Both accumulate the same source values into
// each destination cell in the same ascending order starting from the same
// value, so they are bit-identical — exceptional values and NaN payloads
// included.
TF_EXPORT void tf_core_sum(
    const void* src_handle, void* dst_handle,
    const int64_t* shape, const int64_t* in_strides, const int64_t* out_strides,
    int64_t offset, int64_t ndim
) {
    TF_GUARD_BEGIN
    if (!tf::require_float64("tf_core_sum", {src_handle, dst_handle})) {
        return;
    }
    const double* src = tf::storage_f64(src_handle);
    double* dst = tf::storage_f64(dst_handle);
    if (ndim == 0) {  // scalar input: its single element is the whole sum
        dst[0] += src[offset];
        return;
    }
    int64_t outer = 0;
    int64_t mid = 0;
    int64_t inner = 0;
    if (tf::reduce_prefers_contiguous_blocks(shape, in_strides, out_strides,
                                             ndim, &outer, &mid, &inner)) {
        tf::sum_contiguous_blocks(src, dst, outer, mid, inner, offset);
        return;
    }
    // The odometer's counter is allocated here rather than in the
    // traversal so the guard can map an allocation failure onto the C ABI
    // error contract, and so the fault-injection hook keeps its existing
    // meaning for this kernel.
    std::vector<int64_t> counter = tf::make_counter(ndim);
    tf::sum_generic_strided(src, dst, shape, in_strides, out_strides, offset,
                            ndim, counter.data());
    TF_GUARD_END_VOID()
}

// Narrow's backward: scatter an upstream gradient into a zero output —
// the odometer dual of tf_core_sum. It walks the (smaller) narrowed shape
// and places each upstream element into its own output cell: the output
// write position advances by out_strides (the row-major strides of the
// FULL parent shape, none reduced) from a base out_offset that skips the
// leading ``start`` slabs along the narrowed dimension. The upstream is
// read through its own shape/strides/offset. dst is fresh zero-initialized
// parent-shaped storage; each upstream element maps to exactly one
// distinct cell (narrow regions never overlap), so a plain assignment is
// correct and every un-narrowed cell keeps its zero.
//
// This is a **scatter**, not a reduction, and H6 deliberately left it
// alone: its destination stride vector has no zeros, so it has no
// accumulation to preserve and no reduced run to factorize, and widening
// H6 to cover it would have made the milestone a general scatter one.
TF_EXPORT void tf_core_narrow_backward(
    const void* upstream_handle, void* dst_handle,
    const int64_t* shape, const int64_t* u_strides, const int64_t* out_strides,
    int64_t u_offset, int64_t out_offset, int64_t ndim
) {
    TF_GUARD_BEGIN
    if (!tf::require_float64(
            "tf_core_narrow_backward", {upstream_handle, dst_handle})) {
        return;
    }
    const double* upstream = tf::storage_f64(upstream_handle);
    double* dst = tf::storage_f64(dst_handle);
    if (ndim == 0) {  // scalar upstream: one element at the base offset
        dst[out_offset] = upstream[u_offset];
        return;
    }
    int64_t total = 1;
    for (int64_t d = 0; d < ndim; ++d) {
        total *= shape[d];
    }
    std::vector<int64_t> counter = tf::make_counter(ndim);
    int64_t u_pos = u_offset;
    int64_t out_pos = out_offset;
    for (int64_t i = 0; i < total; ++i) {
        dst[out_pos] = upstream[u_pos];
        for (int64_t d = ndim - 1; d >= 0; --d) {
            ++counter[d];
            u_pos += u_strides[d];
            out_pos += out_strides[d];
            if (counter[d] < shape[d]) {
                break;
            }
            counter[d] = 0;
            u_pos -= shape[d] * u_strides[d];
            out_pos -= shape[d] * out_strides[d];
        }
    }
    TF_GUARD_END_VOID()
}
