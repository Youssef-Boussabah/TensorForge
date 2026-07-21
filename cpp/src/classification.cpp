// Phase-E classification kernels (milestone E3: softmax forward;
// milestone E4: log-softmax forward).
//
// The source unit locked by docs/native_classification_design.md §9.1:
// softmax and log_softmax forwards and, later, the fused cross-entropy
// forward/backward live here — deliberately NOT in elementwise.cpp,
// because these are axis-wise fused reductions, not elementwise maps.
//
// Two layers, the Phase-D split (conv2d.cpp / pooling.cpp):
//   1. tf::softmax_forward_contiguous / tf::log_softmax_forward_contiguous
//      — the internal, hidden, noexcept compute kernels (declared in
//      tf_classification_internal.h), which assume a fully validated
//      contiguous (outer, axis_length, inner) decomposition and allocate
//      nothing.
//   2. tf_core_softmax_forward / tf_core_log_softmax_forward — the
//      exported, exception-guarded C ABI wrappers that revalidate every
//      trust-boundary argument themselves (handles, dimensions, offsets,
//      overflow, spans, destination capacity) before a single destination
//      element is written. Both take the identical call shape, so they
//      share one file-local validator (``forward_argument_error``) and
//      differ only in the operation name their message carries.
//
// The ABI is **contiguous-only** by design: the Python NativeTensorCore
// layer applies the existing Policy-B copy-then-compute rule, so a
// strided view is materialized into a private contiguous copy before the
// call and no stride metadata ever crosses the boundary.
//
// Numerical contract: the standard maximum shift, entirely in float64 —
// and, for log-softmax, the fused log-sum-exp that follows from it,
// never log(softmax(x)). Exceptional values follow plain IEEE arithmetic
// with no special-casing — a NaN or +inf in a slice propagates through
// that slice's shift and sum, so the slice becomes NaN. Those are
// *values*, never ABI errors: the error slot stays TF_OK.

#include <cmath>
#include <cstdio>

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

void log_softmax_forward_contiguous(
    const double* src, double* dst,
    int64_t outer, int64_t axis_length, int64_t inner) noexcept {
    // The same slice decomposition softmax uses; only the arithmetic per
    // slice differs. Deliberately NOT softmax followed by a logarithm:
    // no probability is ever formed, so a tiny probability never rounds
    // to 0 before its logarithm is taken (log(0) == -inf), which is the
    // whole reason this operation exists as its own kernel.
    for (int64_t o = 0; o < outer; ++o) {
        const int64_t plane = o * axis_length * inner;
        for (int64_t i = 0; i < inner; ++i) {
            const int64_t base = plane + i;

            // Pass 1 — the slice maximum, with the same strict `>` that
            // keeps a NaN from becoming the maximum. A NaN still poisons
            // the slice through the accumulation below (the honest IEEE
            // outcome), so nothing is special-cased here either.
            double maximum = src[base];
            for (int64_t k = 1; k < axis_length; ++k) {
                const double value = src[base + k * inner];
                if (value > maximum) {
                    maximum = value;
                }
            }

            // Pass 2 — the shifted logits go straight into the
            // destination while their exponentials are accumulated in
            // float64. Every exponent is <= 0 for finite input, so a
            // large common offset cannot overflow, and the maximum
            // itself contributes exp(0) == 1, so ``sum_exp`` is >= 1 and
            // its logarithm is well defined.
            double sum_exp = 0.0;
            for (int64_t k = 0; k < axis_length; ++k) {
                const double shifted = src[base + k * inner] - maximum;
                dst[base + k * inner] = shifted;
                sum_exp += std::exp(shifted);
            }

            // Pass 3 — subtract the log-normalizer in place. For a
            // length-1 slice this is exactly 0 - log(1) == 0; for equal
            // logits it is exactly -log(axis_length).
            const double log_denominator = std::log(sum_exp);
            for (int64_t k = 0; k < axis_length; ++k) {
                dst[base + k * inner] -= log_denominator;
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

// Trust-boundary validation shared by the fused axis-wise classification
// forwards. Both exports take the identical contiguous (outer,
// axis_length, inner) call shape and must re-prove the identical
// preconditions, so they share one validator rather than keeping two
// copies that could drift apart. Returns nullptr when every argument is
// sound, or a short description of the FIRST failure, in the order the
// E3 softmax export established: handles, dimensional factors, offset
// sign, product overflow, source span, destination capacity.
//
// The caller prefixes its own operation name (below), so each export's
// message text is exactly what it was before this was factored out.
const char* forward_argument_error(
    const void* src_handle, int64_t src_offset, const void* dst_handle,
    int64_t outer, int64_t axis_length, int64_t inner) {
    // -- required handles --
    if (src_handle == nullptr || dst_handle == nullptr) {
        return "null storage handle";
    }
    // -- dimensional factors: every extent is a real, positive count --
    if (outer < 1 || axis_length < 1 || inner < 1) {
        return "outer, axis_length, and inner must each be >= 1";
    }
    if (src_offset < 0) {
        return "negative source offset";
    }
    // -- element count, overflow-checked so a bogus factor cannot wrap
    //    int64 into a small span that would pass the bounds test below --
    int64_t numel;
    if (!checked_mul(outer, axis_length, numel) ||
        !checked_mul(numel, inner, numel)) {
        return "shape product overflows int64";
    }
    // -- spans must fit their allocations --
    if (!span_within(src_offset, numel, tf::as_storage(src_handle)->size)) {
        return "source span exceeds its storage";
    }
    if (tf::as_storage(dst_handle)->size < numel) {
        return "destination storage smaller than the element count";
    }
    return nullptr;
}

// Validate, and on failure record ``op: reason`` in the thread-local slot
// and report that the call must be rejected. Nothing has touched the
// destination by this point, so a rejected call leaves it byte-for-byte
// unchanged.
bool reject_forward_arguments(
    const char* op,
    const void* src_handle, int64_t src_offset, const void* dst_handle,
    int64_t outer, int64_t axis_length, int64_t inner) {
    const char* reason = forward_argument_error(
        src_handle, src_offset, dst_handle, outer, axis_length, inner);
    if (reason == nullptr) {
        return false;
    }
    char message[160];
    std::snprintf(message, sizeof(message), "%s: %s", op, reason);
    tf::set_error(TF_ERROR_INVALID, message);
    return true;
}

}  // namespace

TF_EXPORT void tf_core_softmax_forward(
    const void* src_handle, int64_t src_offset,
    void* dst_handle,
    int64_t outer, int64_t axis_length, int64_t inner) {
    TF_GUARD_BEGIN
    if (reject_forward_arguments("softmax_forward", src_handle, src_offset,
                                 dst_handle, outer, axis_length, inner)) {
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

TF_EXPORT void tf_core_log_softmax_forward(
    const void* src_handle, int64_t src_offset,
    void* dst_handle,
    int64_t outer, int64_t axis_length, int64_t inner) {
    TF_GUARD_BEGIN
    if (reject_forward_arguments("log_softmax_forward", src_handle, src_offset,
                                 dst_handle, outer, axis_length, inner)) {
        return;
    }
    // -- validated: run the internal noexcept kernel. As above, nothing
    //    before this point writes to the destination. A NaN or infinity
    //    in the RESULT is a value, not an ABI failure: the error slot the
    //    guard cleared on entry stays TF_OK. --
    tf::log_softmax_forward_contiguous(
        tf::as_storage(src_handle)->data + src_offset,
        tf::as_storage(dst_handle)->data,
        outer, axis_length, inner);
    TF_GUARD_END_VOID()
}
