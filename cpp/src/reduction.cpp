// Reductions over native tensor cores: the sum reduction (the dual of
// broadcasting) and narrow's backward scatter (the dual of the sum).
// Both walk a strided source with the standard odometer and accumulate or
// place into a fresh output the caller owns. Python chooses every
// shape/stride/offset (keepdims, mean's scaling, and the narrow base
// offset included); these kernels only do the traversal and arithmetic.

#include "tf_internal.h"

using tf::Storage;
using tf::as_storage;

// Sum reduction: where a broadcast reads one element into many output
// positions via zero READ strides, a reduction writes many input elements
// into one output cell via zero WRITE strides. The input is walked with
// the usual odometer; alongside it an output position advances by
// out_strides — the row-major stride of each KEPT axis, or 0 for each
// REDUCED axis, so reduced axes accumulate into the same cell. dst is
// fresh zero-initialized storage (the additive identity), so a plain +=
// accumulates. For axis=None every out_stride is 0 and everything lands
// in dst[0]. Deterministic row-major (input) order; no SIMD/FMA/Kahan.
TF_EXPORT void tf_core_sum(
    const void* src_handle, void* dst_handle,
    const int64_t* shape, const int64_t* in_strides, const int64_t* out_strides,
    int64_t offset, int64_t ndim
) {
    TF_GUARD_BEGIN
    const double* src = as_storage(src_handle)->data;
    double* dst = as_storage(dst_handle)->data;
    if (ndim == 0) {  // scalar input: its single element is the whole sum
        dst[0] += src[offset];
        return;
    }
    int64_t total = 1;
    for (int64_t d = 0; d < ndim; ++d) {
        total *= shape[d];
    }
    std::vector<int64_t> counter = tf::make_counter(ndim);
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
TF_EXPORT void tf_core_narrow_backward(
    const void* upstream_handle, void* dst_handle,
    const int64_t* shape, const int64_t* u_strides, const int64_t* out_strides,
    int64_t u_offset, int64_t out_offset, int64_t ndim
) {
    TF_GUARD_BEGIN
    const double* upstream = as_storage(upstream_handle)->data;
    double* dst = as_storage(dst_handle)->data;
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
