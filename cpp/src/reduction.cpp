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
//
// Phase I, milestone I4 made both exports below dtype-general. Each keeps
// its symbol, its argument list, its calling convention, its traversal
// tiers, its validation, and its ownership contract; the only changes are
// that ``tf::require_float64`` ("this operation has not been generalized")
// became ``tf::require_matching_dtype`` ("it has been, and its operands must
// agree"), and that one ``switch`` per exported call now selects the
// instantiation. The traversals themselves live in tf_reduction_internal.h
// as templates over the element type, so the optimized path, the retained
// reference path, and the scatter are the *same source* at both widths.

#include "tf_internal.h"
#include "tf_reduction_internal.h"

using tf::Storage;
using tf::as_storage;

namespace {

// -- Phase I, milestone I4: the one dispatch per exported call --------------
//
// Two helpers, one per export, each holding the **single** ``switch`` its
// export performs (design §8.1). They sit above the traversals and below the
// exports, so the dtype decision is made exactly once — after the export's
// validation and before any compute — and nothing beneath them branches on
// dtype again: not the predicate, not the factorization, not the odometer
// carry, not the accumulator, and emphatically not any element.
//
// Both arms take the *same source*. ``T = double`` is the pre-I4 call
// statement for statement, so Phase H's measured float64 traversal is
// preserved rather than re-derived, and ``T = float`` cannot drift from it.
//
// The dtype comes from ``tf::dispatch_dtype``, which reads the storage tag —
// layout and operand metadata only, never a pointer value, an alignment, a
// clock, an environment variable, or a CPU-feature probe. Neither ``switch``
// has a ``default:`` label, so a future dtype without an instantiation is a
// compile-time warning rather than a silent misread.
//
// The counter allocation stays *inside* each helper rather than above the
// switch, so an injected allocation failure keeps its existing meaning for
// this kernel and the guard still maps it onto the C ABI error contract.

template <class T>
void sum_dispatch(
    const void* src_handle, void* dst_handle,
    const int64_t* shape, const int64_t* in_strides, const int64_t* out_strides,
    int64_t offset, int64_t ndim
) {
    const T* src = tf::storage_typed<T>(src_handle);
    T* dst = tf::storage_typed<T>(dst_handle);
    if (ndim == 0) {  // scalar input: its single element is the whole sum
        dst[0] += src[offset];
        return;
    }
    int64_t outer = 0;
    int64_t mid = 0;
    int64_t inner = 0;
    // The H6 predicate is untouched by dtype: it reads ``int64`` layout
    // metadata only, which is exactly why it carries over unchanged.
    if (tf::reduce_prefers_contiguous_blocks(shape, in_strides, out_strides,
                                             ndim, &outer, &mid, &inner)) {
        tf::sum_contiguous_blocks(src, dst, outer, mid, inner, offset);
        return;
    }
    std::vector<int64_t> counter = tf::make_counter(ndim);
    tf::sum_generic_strided(src, dst, shape, in_strides, out_strides, offset,
                            ndim, counter.data());
}

template <class T>
void narrow_backward_dispatch(
    const void* upstream_handle, void* dst_handle,
    const int64_t* shape, const int64_t* u_strides, const int64_t* out_strides,
    int64_t u_offset, int64_t out_offset, int64_t ndim
) {
    const T* upstream = tf::storage_typed<T>(upstream_handle);
    T* dst = tf::storage_typed<T>(dst_handle);
    if (ndim == 0) {  // scalar upstream: one element at the base offset
        dst[out_offset] = upstream[u_offset];
        return;
    }
    std::vector<int64_t> counter = tf::make_counter(ndim);
    tf::narrow_backward_scatter(upstream, dst, shape, u_strides, out_strides,
                                u_offset, out_offset, ndim, counter.data());
}

}  // namespace

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

// (The two traversals that used to be defined here — the retained generic
// odometer and the optimized flat block walk — are now templates over the
// element type in tf_reduction_internal.h, beside the contract they answer
// to. Nothing about either loop, either carry, or either accumulator changed
// in the move: ``T = double`` is the code Phase H measured, statement for
// statement. The definitions moved for the ordinary reason a template must,
// so both instantiations are available to the exports below *and* to the
// CTests that compile this file directly.)

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
//
// I4: dtype-general. Source and destination dtypes must **agree** — there is
// no casting, no promotion, and no mixed-dtype arithmetic anywhere in the
// runtime (design §9), so summing a float32 source into a float64
// destination is an invalid *request* rather than a conversion opportunity.
// The rejection is recorded before anything is read or written, so a
// rejected call leaves the destination byte-for-byte unchanged — which
// matters more here than in most exports, because this destination carries
// the caller's accumulated partial sums rather than a write-only result.
//
// **float32 accumulates in float32** (design §10.1): the accumulator type
// follows the element type on both traversals, so no hidden binary64
// intermediate exists on either, and the two paths therefore stay the same
// reduction at binary32 exactly as they are at binary64.
TF_EXPORT void tf_core_sum(
    const void* src_handle, void* dst_handle,
    const int64_t* shape, const int64_t* in_strides, const int64_t* out_strides,
    int64_t offset, int64_t ndim
) {
    TF_GUARD_BEGIN
    if (!tf::require_matching_dtype("tf_core_sum", {src_handle, dst_handle})) {
        return;
    }
    // The one dispatch: before any compute, reading the dtype from the
    // handles the caller already passed. Nothing below this point branches
    // on dtype again — not the H6 predicate, not the factorization, not the
    // odometer carry, not the accumulator, and emphatically not any element.
    // The counter allocation the guard maps onto the error contract lives
    // inside the helper, so an injected allocation failure keeps its
    // existing meaning for this kernel at both widths.
    switch (tf::dispatch_dtype({src_handle, dst_handle})) {
        case tf::Dtype::Float32:
            sum_dispatch<float>(src_handle, dst_handle, shape, in_strides,
                                out_strides, offset, ndim);
            return;
        case tf::Dtype::Float64:
            break;
    }
    sum_dispatch<double>(src_handle, dst_handle, shape, in_strides,
                         out_strides, offset, ndim);
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
//
// I4: dtype-general, and the traversal moved to
// ``tf::narrow_backward_scatter`` in tf_reduction_internal.h so both
// instantiations come from one source. Upstream and destination dtypes must
// **agree**; the rejection is recorded before anything is written, so a
// rejected call leaves the destination — including every zero that *is* the
// gradient outside the narrowed region — byte-for-byte unchanged.
//
// It stays a scatter and not an identity copy: the destination must still be
// zero-initialized at both widths, because the untouched cells are the
// answer rather than an initialization detail.
TF_EXPORT void tf_core_narrow_backward(
    const void* upstream_handle, void* dst_handle,
    const int64_t* shape, const int64_t* u_strides, const int64_t* out_strides,
    int64_t u_offset, int64_t out_offset, int64_t ndim
) {
    TF_GUARD_BEGIN
    if (!tf::require_matching_dtype(
            "tf_core_narrow_backward", {upstream_handle, dst_handle})) {
        return;
    }
    switch (tf::dispatch_dtype({upstream_handle, dst_handle})) {
        case tf::Dtype::Float32:
            narrow_backward_dispatch<float>(upstream_handle, dst_handle, shape,
                                            u_strides, out_strides, u_offset,
                                            out_offset, ndim);
            return;
        case tf::Dtype::Float64:
            break;
    }
    narrow_backward_dispatch<double>(upstream_handle, dst_handle, shape,
                                     u_strides, out_strides, u_offset,
                                     out_offset, ndim);
    TF_GUARD_END_VOID()
}
