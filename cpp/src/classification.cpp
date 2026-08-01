// Phase-E classification kernels (milestone E3: softmax forward;
// milestone E4: log-softmax forward; milestone E5: the fused
// cross-entropy forward and backward).
//
// The source unit locked by docs/native_classification_design.md §9.1:
// the softmax and log_softmax forwards and the fused cross-entropy
// forward/backward live here — deliberately NOT in elementwise.cpp,
// because these are axis-wise fused reductions, not elementwise maps.
//
// Two layers, the Phase-D split (conv2d.cpp / pooling.cpp):
//   1. tf::softmax_forward_contiguous, tf::log_softmax_forward_contiguous,
//      tf::cross_entropy_forward_contiguous, and
//      tf::cross_entropy_backward_contiguous — the internal, hidden,
//      noexcept compute kernels (declared in
//      tf_classification_internal.h), which assume fully validated
//      contiguous arguments and allocate nothing.
//   2. tf_core_softmax_forward, tf_core_log_softmax_forward,
//      tf_core_cross_entropy_forward, and tf_core_cross_entropy_backward
//      — the exported, exception-guarded C ABI wrappers that revalidate
//      every trust-boundary argument themselves (handles, dimensions,
//      offsets, overflow, spans, destination capacity, and every target
//      index) before a single destination element is written. The two
//      axis-wise forwards take the identical call shape, so they share
//      one file-local validator (``forward_argument_error``) and differ
//      only in the operation name their message carries; the two
//      cross-entropy exports likewise share
//      ``cross_entropy_common_error`` and ``reject_target_range``.
//
// The ABI is **contiguous-only** for tensor data by design: the Python
// NativeTensorCore layer applies the existing Policy-B copy-then-compute
// rule, so a strided view is materialized into a private contiguous copy
// before the call and no stride metadata ever crosses the boundary. The
// cross-entropy targets are the one non-tensor operand — a host int64
// span, because the native runtime has no integer dtype (design §6).
//
// Numerical contract: the standard maximum shift, entirely in float64 —
// and, for log-softmax and cross-entropy, the fused log-sum-exp that
// follows from it, never log(softmax(x)) and never
// -log(probability[target]). Exceptional values follow plain IEEE
// arithmetic with no special-casing — a NaN or +inf in a slice
// propagates through that slice's shift and sum, so the slice becomes
// NaN. Those are *values*, never ABI errors: the error slot stays TF_OK.

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

void cross_entropy_forward_contiguous(
    const double* logits, const int64_t* targets,
    double* loss, double* probabilities,
    int64_t batch_size, int64_t num_classes,
    int64_t reduction_code) noexcept {
    // One row per example; the class axis is fixed at axis 1, so a row is
    // a contiguous run of ``num_classes`` doubles. Fixed traversal order,
    // so the accumulation is reproducible run to run.
    double total = 0.0;
    for (int64_t n = 0; n < batch_size; ++n) {
        const int64_t base = n * num_classes;

        // Pass 1 — the row maximum, with the same strict `>` the softmax
        // and log-softmax kernels use, so a NaN never becomes the
        // maximum. A NaN still poisons its row through the shift below,
        // which is the honest IEEE outcome rather than a special case.
        double maximum = logits[base];
        for (int64_t c = 1; c < num_classes; ++c) {
            const double value = logits[base + c];
            if (value > maximum) {
                maximum = value;
            }
        }

        // Pass 2 — shifted exponentials, written straight into the saved
        // probability destination (no second buffer is allocated) and
        // accumulated in float64. Every exponent is <= 0 for finite
        // input, so a large common offset cannot overflow.
        double sum_exp = 0.0;
        for (int64_t c = 0; c < num_classes; ++c) {
            const double shifted = std::exp(logits[base + c] - maximum);
            probabilities[base + c] = shifted;
            sum_exp += shifted;
        }

        // Pass 3 — normalize the row in place. For finite input
        // ``sum_exp`` is >= 1 (the maximum contributes exp(0) == 1), so
        // this never divides by zero.
        for (int64_t c = 0; c < num_classes; ++c) {
            probabilities[base + c] /= sum_exp;
        }

        // The per-example loss comes from the log-sum-exp directly:
        //     log(Σ exp(x - m)) - (x[target] - m)
        // NOT -log(probabilities[target]), which would first round a tiny
        // probability to 0 and then report an infinite loss. The target
        // index was proved in [0, num_classes) by the guarded wrapper.
        const double log_denominator = std::log(sum_exp);
        const double shifted_target = logits[base + targets[n]] - maximum;
        total += log_denominator - shifted_target;
    }

    // "sum" takes the accumulated total unscaled; "mean" divides ONCE by
    // the batch size — never by num_classes.
    *loss = (reduction_code == kCrossEntropyReductionMean)
                ? total / static_cast<double>(batch_size)
                : total;
}

void cross_entropy_backward_contiguous(
    const double* probabilities, const int64_t* targets,
    const double* upstream, double* grad_logits,
    int64_t batch_size, int64_t num_classes,
    int64_t reduction_code) noexcept {
    // The whole gradient comes from the SAVED probabilities and the
    // copied targets: no logit is reread, no maximum is recomputed, and
    // no exponential is evaluated here.
    const double scale = *upstream;
    const bool mean = (reduction_code == kCrossEntropyReductionMean);
    const double count = static_cast<double>(batch_size);
    for (int64_t n = 0; n < batch_size; ++n) {
        const int64_t base = n * num_classes;
        const int64_t target = targets[n];
        for (int64_t c = 0; c < num_classes; ++c) {
            // d(loss_n)/d(x[n, c]) = p[n, c] - [c == target]
            double contribution = probabilities[base + c];
            if (c == target) {
                contribution -= 1.0;
            }
            if (mean) {
                contribution /= count;
            }
            grad_logits[base + c] = scale * contribution;
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

// -- cross-entropy trust-boundary validation (E5) ----------------------
//
// The two cross-entropy exports share the same source-plus-targets call
// shape (contiguous float64 block + offset, a host int64 target span,
// dimensions, a reduction code), so they share these checks and add only
// their own destination capacities. Everything below runs BEFORE a
// single destination element is written, so a rejected call leaves every
// destination byte-for-byte unchanged.

// Record ``op: reason`` in the thread-local slot and report rejection.
bool reject_reason(const char* op, const char* reason) {
    char message[192];
    std::snprintf(message, sizeof(message), "%s: %s", op, reason);
    tf::set_error(TF_ERROR_INVALID, message);
    return true;
}

// The checks common to the forward (source = logits) and the backward
// (source = saved probabilities), in a fixed order. Returns nullptr when
// every argument is sound, or a short description of the FIRST failure,
// and sets ``numel`` to the overflow-checked batch_size * num_classes.
// The caller has already proved the handles and the target pointer are
// non-null.
const char* cross_entropy_common_error(
    const void* src_handle, int64_t src_offset,
    int64_t target_count, int64_t batch_size, int64_t num_classes,
    int64_t reduction_code, int64_t& numel) {
    if (src_offset < 0) {
        return "negative source offset";
    }
    if (batch_size < 1 || num_classes < 1) {
        return "batch_size and num_classes must each be >= 1";
    }
    // The target span is host metadata, not storage: its only structural
    // guarantee is its length, so that must match the batch exactly.
    if (target_count != batch_size) {
        return "target count does not equal batch_size";
    }
    if (reduction_code != tf::kCrossEntropyReductionMean &&
        reduction_code != tf::kCrossEntropyReductionSum) {
        return "unknown reduction code (0 = mean, 1 = sum)";
    }
    // Element count, overflow-checked so a bogus dimension cannot wrap
    // int64 into a small span that would pass the bounds tests below.
    if (!checked_mul(batch_size, num_classes, numel)) {
        return "shape product overflows int64";
    }
    if (!span_within(src_offset, numel, tf::as_storage(src_handle)->size)) {
        return "source span exceeds its storage";
    }
    return nullptr;
}

// Every target index the kernel will dereference must be in range. The
// Python layer already proved this, but the exports are reachable by any
// ctypes caller, so C++ re-proves it for the whole span before the
// kernel writes anything — the same discipline the pooling backward
// applies to its saved winners.
bool reject_target_range(const char* op, const int64_t* targets,
                         int64_t count, int64_t num_classes) {
    for (int64_t index = 0; index < count; ++index) {
        const int64_t target = targets[index];
        if (target < 0 || target >= num_classes) {
            char message[192];
            std::snprintf(
                message, sizeof(message),
                "%s: target %lld at index %lld is outside the class range "
                "[0, %lld)", op, static_cast<long long>(target),
                static_cast<long long>(index),
                static_cast<long long>(num_classes));
            tf::set_error(TF_ERROR_INVALID, message);
            return true;
        }
    }
    return false;
}

}  // namespace

TF_EXPORT void tf_core_softmax_forward(
    const void* src_handle, int64_t src_offset,
    void* dst_handle,
    int64_t outer, int64_t axis_length, int64_t inner) {
    TF_GUARD_BEGIN
    if (!tf::require_float64(
            "tf_core_softmax_forward",
            {src_handle, dst_handle})) {
        return;
    }
    if (reject_forward_arguments("softmax_forward", src_handle, src_offset,
                                 dst_handle, outer, axis_length, inner)) {
        return;
    }
    // -- validated: run the internal noexcept kernel. Nothing above this
    //    point writes to the destination, so a rejected call leaves it
    //    byte-for-byte unchanged. --
    tf::softmax_forward_contiguous(
        tf::storage_f64(src_handle) + src_offset,
        tf::storage_f64(dst_handle),
        outer, axis_length, inner);
    TF_GUARD_END_VOID()
}

TF_EXPORT void tf_core_log_softmax_forward(
    const void* src_handle, int64_t src_offset,
    void* dst_handle,
    int64_t outer, int64_t axis_length, int64_t inner) {
    TF_GUARD_BEGIN
    if (!tf::require_float64(
            "tf_core_log_softmax_forward",
            {src_handle, dst_handle})) {
        return;
    }
    if (reject_forward_arguments("log_softmax_forward", src_handle, src_offset,
                                 dst_handle, outer, axis_length, inner)) {
        return;
    }
    // -- validated: run the internal noexcept kernel. As above, nothing
    //    before this point writes to the destination. A NaN or infinity
    //    in the RESULT is a value, not an ABI failure: the error slot the
    //    guard cleared on entry stays TF_OK. --
    tf::log_softmax_forward_contiguous(
        tf::storage_f64(src_handle) + src_offset,
        tf::storage_f64(dst_handle),
        outer, axis_length, inner);
    TF_GUARD_END_VOID()
}

// Fused cross-entropy forward (E5). Contiguous-only for the tensor data,
// like the two transforms above: the Core layer applies Policy-B
// copy-then-compute to a strided view, so no stride metadata crosses the
// boundary. The targets are the one non-tensor operand — a caller-owned
// span of ``target_count`` host int64 class labels, since the native
// runtime has no integer dtype (design §6).
//
// Both destinations (the scalar loss and the saved probabilities) are
// caller-allocated at offset 0. Every argument is revalidated here
// before a single destination element is written.
TF_EXPORT void tf_core_cross_entropy_forward(
    const void* logits_handle, int64_t logits_offset,
    const int64_t* targets, int64_t target_count,
    void* loss_handle, void* probabilities_handle,
    int64_t batch_size, int64_t num_classes, int64_t reduction_code) {
    TF_GUARD_BEGIN
    if (!tf::require_float64(
            "tf_core_cross_entropy_forward",
            {logits_handle, loss_handle, probabilities_handle})) {
        return;
    }
    const char* op = "cross_entropy_forward";
    if (logits_handle == nullptr || loss_handle == nullptr ||
        probabilities_handle == nullptr) {
        reject_reason(op, "null storage handle");
        return;
    }
    if (targets == nullptr) {
        reject_reason(op, "null target pointer");
        return;
    }
    int64_t numel = 0;
    const char* reason = cross_entropy_common_error(
        logits_handle, logits_offset, target_count, batch_size, num_classes,
        reduction_code, numel);
    if (reason != nullptr) {
        reject_reason(op, reason);
        return;
    }
    // -- destination capacities: one double for the scalar loss, a full
    //    (batch_size, num_classes) block for the saved probabilities --
    if (tf::as_storage(loss_handle)->size < 1) {
        reject_reason(op, "loss storage cannot hold a scalar");
        return;
    }
    if (tf::as_storage(probabilities_handle)->size < numel) {
        reject_reason(op,
                      "probability storage smaller than the element count");
        return;
    }
    // -- no destination may alias the logits: the row loss reads a logit
    //    AFTER the row's probabilities have been written, so an aliasing
    //    call would silently compute a wrong loss rather than fail --
    if (probabilities_handle == logits_handle ||
        loss_handle == logits_handle ||
        loss_handle == probabilities_handle) {
        reject_reason(op, "destination storage aliases another operand");
        return;
    }
    if (reject_target_range(op, targets, target_count, num_classes)) {
        return;
    }
    // -- validated: run the internal noexcept kernel. Nothing above this
    //    point writes to either destination, so a rejected call leaves
    //    both byte-for-byte unchanged. A NaN or infinity in the RESULT is
    //    a value, not an ABI failure: the error slot the guard cleared on
    //    entry stays TF_OK. --
    tf::cross_entropy_forward_contiguous(
        tf::storage_f64(logits_handle) + logits_offset,
        targets,
        tf::storage_f64(loss_handle),
        tf::storage_f64(probabilities_handle),
        batch_size, num_classes, reduction_code);
    TF_GUARD_END_VOID()
}

// Fused cross-entropy backward (E5). Reads the SAVED probabilities, the
// copied targets, and one upstream value; the logits are neither passed
// nor reachable, which is the structural half of "backward never rereads
// the logits". The gradient destination is caller-allocated at offset 0.
TF_EXPORT void tf_core_cross_entropy_backward(
    const void* probabilities_handle, int64_t probabilities_offset,
    const int64_t* targets, int64_t target_count,
    const void* upstream_handle, int64_t upstream_offset,
    void* grad_logits_handle,
    int64_t batch_size, int64_t num_classes, int64_t reduction_code) {
    TF_GUARD_BEGIN
    if (!tf::require_float64(
            "tf_core_cross_entropy_backward",
            {probabilities_handle, upstream_handle, grad_logits_handle})) {
        return;
    }
    const char* op = "cross_entropy_backward";
    if (probabilities_handle == nullptr || upstream_handle == nullptr ||
        grad_logits_handle == nullptr) {
        reject_reason(op, "null storage handle");
        return;
    }
    if (targets == nullptr) {
        reject_reason(op, "null target pointer");
        return;
    }
    int64_t numel = 0;
    const char* reason = cross_entropy_common_error(
        probabilities_handle, probabilities_offset, target_count, batch_size,
        num_classes, reduction_code, numel);
    if (reason != nullptr) {
        reject_reason(op, reason);
        return;
    }
    // -- the upstream is a single value: one element must fit at its
    //    offset, whatever the caller's logical shape was --
    if (upstream_offset < 0) {
        reject_reason(op, "negative upstream offset");
        return;
    }
    if (!span_within(upstream_offset, 1,
                     tf::as_storage(upstream_handle)->size)) {
        reject_reason(op, "upstream span exceeds its storage");
        return;
    }
    if (tf::as_storage(grad_logits_handle)->size < numel) {
        reject_reason(op,
                      "gradient storage smaller than the element count");
        return;
    }
    // -- the gradient destination is written element by element while the
    //    probabilities and the upstream are read, so it must alias
    //    neither --
    if (grad_logits_handle == probabilities_handle ||
        grad_logits_handle == upstream_handle) {
        reject_reason(op, "destination storage aliases another operand");
        return;
    }
    if (reject_target_range(op, targets, target_count, num_classes)) {
        return;
    }
    // -- validated: nothing above this point writes to the destination. --
    tf::cross_entropy_backward_contiguous(
        tf::storage_f64(probabilities_handle) + probabilities_offset,
        targets,
        tf::storage_f64(upstream_handle) + upstream_offset,
        tf::storage_f64(grad_logits_handle),
        batch_size, num_classes, reduction_code);
    TF_GUARD_END_VOID()
}
