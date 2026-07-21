// Phase-E classification kernels (milestone E3: softmax forward).
//
// The source unit locked by docs/native_classification_design.md §9.1:
// softmax and log_softmax forwards and, later, the fused cross-entropy
// forward/backward live here — deliberately NOT in elementwise.cpp,
// because these are axis-wise fused reductions, not elementwise maps.
//
// Two layers, the Phase-D split (conv2d.cpp / pooling.cpp):
//   1. tf::softmax_forward_contiguous — the internal, hidden, noexcept
//      compute kernel (declared in tf_classification_internal.h), which
//      assumes a fully validated contiguous (outer, axis_length, inner)
//      decomposition and allocates nothing.
//   2. tf_core_softmax_forward — the exported, exception-guarded C ABI
//      wrapper that revalidates every trust-boundary argument itself
//      (handles, dimensions, offsets, overflow, spans, destination
//      capacity) before a single destination element is written.
//
// The ABI is **contiguous-only** by design: the Python NativeTensorCore
// layer applies the existing Policy-B copy-then-compute rule, so a
// strided view is materialized into a private contiguous copy before the
// call and no stride metadata ever crosses the boundary.
//
// Numerical contract: the standard maximum shift, entirely in float64.
// Exceptional values follow plain IEEE arithmetic with no special-casing
// — a NaN or +inf in a slice propagates through that slice's shift and
// sum, so the slice becomes NaN. Those are *values*, never ABI errors:
// the error slot stays TF_OK.

#include <cmath>

#include "tf_classification_internal.h"
#include "tf_internal.h"  // export macro, Storage/as_storage, TF_GUARD, set_error

namespace tf {

void softmax_forward_contiguous(
    const double* src, double* dst,
    int64_t outer, int64_t axis_length, int64_t inner) noexcept {
    // One reduction slice per (outer, inner) pair; the axis is walked
    // with stride ``inner``. Loop order is fixed, so results are
    // reproducible run to run.
    for (int64_t o = 0; o < outer; ++o) {
        const int64_t plane = o * axis_length * inner;
        for (int64_t i = 0; i < inner; ++i) {
            const int64_t base = plane + i;

            // Pass 1 — the slice maximum. A strict `>` comparison means a
            // NaN never becomes the maximum (NaN > m is false), matching
            // the pooling kernel's convention; a NaN still poisons the
            // slice through the shift below, which is the honest IEEE
            // outcome rather than a special case.
            double maximum = src[base];
            for (int64_t k = 1; k < axis_length; ++k) {
                const double value = src[base + k * inner];
                if (value > maximum) {
                    maximum = value;
                }
            }

            // Pass 2 — shifted exponentials, written straight into the
            // destination (no second buffer is allocated) and accumulated
            // in float64. Every exponent is <= 0 for finite input, so a
            // large common offset cannot overflow.
            double total = 0.0;
            for (int64_t k = 0; k < axis_length; ++k) {
                const double shifted = std::exp(src[base + k * inner] - maximum);
                dst[base + k * inner] = shifted;
                total += shifted;
            }

            // Pass 3 — normalize in place. For finite input ``total`` is
            // >= 1 (the maximum contributes exp(0) == 1), so this never
            // divides by zero; for a NaN/inf slice the division simply
            // propagates the IEEE result.
            for (int64_t k = 0; k < axis_length; ++k) {
                dst[base + k * inner] /= total;
            }
        }
    }
}

}  // namespace tf

namespace {

// Checked int64 multiply/add for non-negative operands. File-local, like
// the equivalents in conv2d.cpp, pooling.cpp, and elementwise.cpp (each
// compute unit keeps its own rather than growing a premature shared
// surface — design §14).
bool checked_mul(int64_t a, int64_t b, int64_t& out) {
    if (a == 0 || b == 0) {
        out = 0;
        return true;
    }
    if (a > INT64_MAX / b) {
        return false;
    }
    out = a * b;
    return true;
}

bool checked_add(int64_t a, int64_t b, int64_t& out) {
    if (a > INT64_MAX - b) {
        return false;
    }
    out = a + b;
    return true;
}

// A contiguous run of ``count`` elements starting at ``offset`` must fit
// inside a storage holding ``size`` doubles.
bool span_within(int64_t offset, int64_t count, int64_t size) {
    int64_t end;
    if (!checked_add(offset, count, end)) {
        return false;
    }
    return end <= size;
}

}  // namespace

TF_EXPORT void tf_core_softmax_forward(
    const void* src_handle, int64_t src_offset,
    void* dst_handle,
    int64_t outer, int64_t axis_length, int64_t inner) {
    TF_GUARD_BEGIN
    // -- required handles --
    if (src_handle == nullptr || dst_handle == nullptr) {
        tf::set_error(TF_ERROR_INVALID,
                      "softmax_forward: null storage handle");
        return;
    }
    // -- dimensional factors: every extent is a real, positive count --
    if (outer < 1 || axis_length < 1 || inner < 1) {
        tf::set_error(
            TF_ERROR_INVALID,
            "softmax_forward: outer, axis_length, and inner must each be >= 1");
        return;
    }
    if (src_offset < 0) {
        tf::set_error(TF_ERROR_INVALID,
                      "softmax_forward: negative source offset");
        return;
    }
    // -- element count, overflow-checked so a bogus factor cannot wrap
    //    int64 into a small span that would pass the bounds test below --
    int64_t numel;
    if (!checked_mul(outer, axis_length, numel) ||
        !checked_mul(numel, inner, numel)) {
        tf::set_error(TF_ERROR_INVALID,
                      "softmax_forward: shape product overflows int64");
        return;
    }
    // -- spans must fit their allocations --
    if (!span_within(src_offset, numel, tf::as_storage(src_handle)->size)) {
        tf::set_error(TF_ERROR_INVALID,
                      "softmax_forward: source span exceeds its storage");
        return;
    }
    if (tf::as_storage(dst_handle)->size < numel) {
        tf::set_error(
            TF_ERROR_INVALID,
            "softmax_forward: destination storage smaller than the element count");
        return;
    }
    // -- validated: run the internal noexcept kernel. Nothing above this
    //    point writes to the destination, so a rejected call leaves it
    //    byte-for-byte unchanged. --
    tf::softmax_forward_contiguous(
        tf::as_storage(src_handle)->data + src_offset,
        tf::as_storage(dst_handle)->data,
        outer, axis_length, inner);
    TF_GUARD_END_VOID()
}
