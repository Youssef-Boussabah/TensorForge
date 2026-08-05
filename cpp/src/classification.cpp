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
//      noexcept compute kernels, which assume fully validated contiguous
//      arguments and allocate nothing. Since Phase I milestone I6 they are
//      **templates over the element type**, so their definitions live in
//      tf_classification_internal.h (a template must be visible where it is
//      instantiated, and both instantiations must reach the wrappers below
//      *and* the CTests that compile this file directly).
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
// Numerical contract: the standard maximum shift, entirely **in the
// element type** (Phase I, milestone I6 — it read "entirely in float64"
// through I5, when float64 was the only element type there was) — and, for
// log-softmax and cross-entropy, the fused log-sum-exp that follows from
// it, never log(softmax(x)) and never -log(probability[target]).
// Exceptional values follow plain IEEE arithmetic with no special-casing —
// a NaN or +inf in a slice propagates through that slice's shift and sum,
// so the slice becomes NaN. Those are *values*, never ABI errors: the
// error slot stays TF_OK.
//
// Dtype discipline (design §8.1, §9.1): each export validates that its
// participating numeric handles carry the **same** dtype, then performs
// **one** dispatch into the templated kernel. The cross-entropy target
// span is host int64 metadata with no tensor dtype, so it takes part in
// neither. Nothing below the switch branches on dtype again — not per
// slice, not per class, not per element.

#include <cstdio>

#include "tf_classification_internal.h"
#include "tf_internal.h"  // export macro, Storage/as_storage, TF_GUARD, set_error

namespace {

// -- dtype dispatch arms (Phase I, milestone I6) -----------------------
//
// One arm per export, mirroring the I5 conv2d/pooling arms: recover the
// typed pointers through the single ``tf::storage_typed<T>`` accessor and
// call the one templated kernel. They exist so each export's ``switch``
// stays two short branches and the pointer recovery is written once per
// operation rather than once per dtype. Every arm is reached only after
// ``tf::require_matching_dtype`` has proved the participating handles
// agree, which is what makes the recovery sound.
//
// The int64 target span crosses these unchanged: it is host metadata, not
// storage, so it has no dtype to dispatch on and is not a template
// parameter of anything.

template <class T>
void softmax_forward_dispatch(
    const void* src_handle, int64_t src_offset, void* dst_handle,
    int64_t outer, int64_t axis_length, int64_t inner) {
    tf::softmax_forward_contiguous(
        tf::storage_typed<T>(src_handle) + src_offset,
        tf::storage_typed<T>(dst_handle),
        outer, axis_length, inner);
}

template <class T>
void log_softmax_forward_dispatch(
    const void* src_handle, int64_t src_offset, void* dst_handle,
    int64_t outer, int64_t axis_length, int64_t inner) {
    tf::log_softmax_forward_contiguous(
        tf::storage_typed<T>(src_handle) + src_offset,
        tf::storage_typed<T>(dst_handle),
        outer, axis_length, inner);
}

template <class T>
void cross_entropy_forward_dispatch(
    const void* logits_handle, int64_t logits_offset,
    const int64_t* targets, void* loss_handle, void* probabilities_handle,
    int64_t batch_size, int64_t num_classes, int64_t reduction_code) {
    tf::cross_entropy_forward_contiguous(
        tf::storage_typed<T>(logits_handle) + logits_offset,
        targets,
        tf::storage_typed<T>(loss_handle),
        tf::storage_typed<T>(probabilities_handle),
        batch_size, num_classes, reduction_code);
}

template <class T>
void cross_entropy_backward_dispatch(
    const void* probabilities_handle, int64_t probabilities_offset,
    const int64_t* targets,
    const void* upstream_handle, int64_t upstream_offset,
    void* grad_logits_handle,
    int64_t batch_size, int64_t num_classes, int64_t reduction_code) {
    tf::cross_entropy_backward_contiguous(
        tf::storage_typed<T>(probabilities_handle) + probabilities_offset,
        targets,
        tf::storage_typed<T>(upstream_handle) + upstream_offset,
        tf::storage_typed<T>(grad_logits_handle),
        batch_size, num_classes, reduction_code);
}

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
// inside a storage holding ``size`` elements. Spans, offsets, and
// capacities are counted in **logical elements** at every dtype (design
// §4.3), so none of this arithmetic changed at I6.
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
// shape (a contiguous block + offset at either dtype, a host int64 target span,
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
    // The dtype guard runs first (I6, matching the I3/I4/I5 exports): the
    // source and the destination must agree, and a call that is both
    // mixed-dtype and otherwise malformed reports the dtype. There is no
    // promotion and no narrowing, so a float32 source with a float64
    // destination is an invalid request, not a conversion opportunity.
    // K1: the dtype-role guard runs first — an int64 operand is
    // a role error, never a promotion opportunity (§22.4).
    if (!tf::require_floating(
            "tf_core_softmax_forward",
            {src_handle, dst_handle})) {
        return;
    }
    if (!tf::require_matching_dtype(
            "tf_core_softmax_forward",
            {src_handle, dst_handle})) {
        return;
    }
    if (reject_forward_arguments("softmax_forward", src_handle, src_offset,
                                 dst_handle, outer, axis_length, inner)) {
        return;
    }
    // -- validated: one dtype dispatch into the internal noexcept kernel.
    //    Nothing above this point writes to the destination, so a rejected
    //    call leaves it byte-for-byte unchanged. --
    switch (tf::dispatch_dtype({src_handle, dst_handle})) {
        case tf::Dtype::Float32:
            softmax_forward_dispatch<float>(
                src_handle, src_offset, dst_handle,
                outer, axis_length, inner);
            break;
        case tf::Dtype::Float64:
            softmax_forward_dispatch<double>(
                src_handle, src_offset, dst_handle,
                outer, axis_length, inner);
            break;
    }
    TF_GUARD_END_VOID()
}

TF_EXPORT void tf_core_log_softmax_forward(
    const void* src_handle, int64_t src_offset,
    void* dst_handle,
    int64_t outer, int64_t axis_length, int64_t inner) {
    TF_GUARD_BEGIN
    // Same dtype rule and same ordering as the softmax export above.
    // K1: the dtype-role guard runs first — an int64 operand is
    // a role error, never a promotion opportunity (§22.4).
    if (!tf::require_floating(
            "tf_core_log_softmax_forward",
            {src_handle, dst_handle})) {
        return;
    }
    if (!tf::require_matching_dtype(
            "tf_core_log_softmax_forward",
            {src_handle, dst_handle})) {
        return;
    }
    if (reject_forward_arguments("log_softmax_forward", src_handle, src_offset,
                                 dst_handle, outer, axis_length, inner)) {
        return;
    }
    // -- validated: one dtype dispatch into the internal noexcept kernel.
    //    As above, nothing before this point writes to the destination. A
    //    NaN or infinity in the RESULT is a value, not an ABI failure: the
    //    error slot the guard cleared on entry stays TF_OK. That covers the
    //    -inf a slice gets when its spread exceeds the element type's
    //    finite range, which is an IEEE overflow of a representable
    //    request, not a malformed one. --
    switch (tf::dispatch_dtype({src_handle, dst_handle})) {
        case tf::Dtype::Float32:
            log_softmax_forward_dispatch<float>(
                src_handle, src_offset, dst_handle,
                outer, axis_length, inner);
            break;
        case tf::Dtype::Float64:
            log_softmax_forward_dispatch<double>(
                src_handle, src_offset, dst_handle,
                outer, axis_length, inner);
            break;
    }
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
    // All **three** numeric handles must agree: the logits, the scalar
    // loss, and the saved probabilities. The target span is not among them
    // — it is host int64 metadata, so there is no target dtype to check, no
    // target dtype to dispatch on, and nothing is ever inferred from it.
    // K1: the dtype-role guard runs first — an int64 operand is
    // a role error, never a promotion opportunity (§22.4).
    if (!tf::require_floating(
            "tf_core_cross_entropy_forward",
            {logits_handle, loss_handle, probabilities_handle})) {
        return;
    }
    if (!tf::require_matching_dtype(
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
    // -- destination capacities: one element for the scalar loss, a full
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
    // -- validated: one dtype dispatch into the internal noexcept kernel.
    //    Nothing above this point writes to either destination, so a
    //    rejected call leaves both byte-for-byte unchanged. A NaN or
    //    infinity in the RESULT is a value, not an ABI failure: the error
    //    slot the guard cleared on entry stays TF_OK. --
    switch (tf::dispatch_dtype(
                {logits_handle, loss_handle, probabilities_handle})) {
        case tf::Dtype::Float32:
            cross_entropy_forward_dispatch<float>(
                logits_handle, logits_offset, targets,
                loss_handle, probabilities_handle,
                batch_size, num_classes, reduction_code);
            break;
        case tf::Dtype::Float64:
            cross_entropy_forward_dispatch<double>(
                logits_handle, logits_offset, targets,
                loss_handle, probabilities_handle,
                batch_size, num_classes, reduction_code);
            break;
    }
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
    // All **three** numeric handles must agree: the saved probabilities,
    // the scalar upstream, and the logits-gradient destination. As in the
    // forward, the target span is host int64 metadata and takes part in
    // neither the agreement check nor the dispatch. The logits are not a
    // parameter of this export at all, so there is no fourth handle here
    // and never could be.
    // K1: the dtype-role guard runs first — an int64 operand is
    // a role error, never a promotion opportunity (§22.4).
    if (!tf::require_floating(
            "tf_core_cross_entropy_backward",
            {probabilities_handle, upstream_handle, grad_logits_handle})) {
        return;
    }
    if (!tf::require_matching_dtype(
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
    // -- validated: one dtype dispatch into the internal noexcept kernel.
    //    Nothing above this point writes to the destination. --
    switch (tf::dispatch_dtype(
                {probabilities_handle, upstream_handle, grad_logits_handle})) {
        case tf::Dtype::Float32:
            cross_entropy_backward_dispatch<float>(
                probabilities_handle, probabilities_offset, targets,
                upstream_handle, upstream_offset, grad_logits_handle,
                batch_size, num_classes, reduction_code);
            break;
        case tf::Dtype::Float64:
            cross_entropy_backward_dispatch<double>(
                probabilities_handle, probabilities_offset, targets,
                upstream_handle, upstream_offset, grad_logits_handle,
                batch_size, num_classes, reduction_code);
            break;
    }
    TF_GUARD_END_VOID()
}
