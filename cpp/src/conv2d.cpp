// Conv2d (Phase D, milestones D2-D6): the H9 dispatch predicates and the
// exported, exception-guarded C ABI wrappers over the internal noexcept
// cross-correlation kernels.
//
// The compute kernels themselves — three retained Phase-D generic loops,
// three H9 optimized traversals, and the three dispatchers that choose
// between them — live in cpp/include/tf_conv2d_internal.h as templates
// over the element type (Phase I, milestone I5), so both the ``float`` and
// ``double`` instantiations are available to the exported wrappers here
// *and* to the CTests that compile this file directly. This file holds
// what is not a template: the geometry predicates, the validation, and the
// dtype dispatch.
//
// Cross-correlation (NOT flipped convolution), matching the reference
// stable tensorforge.nn.Conv2d:
//   out[n,o,i,j] = bias[o] + sum_{c,p,q} x_pad[n,c,i*sh+p,j*sw+q] * w[o,c,p,q]
// Symmetric zero padding is realized by skipping out-of-bounds source
// coordinates (a padded cell contributes 0*w = 0), so no padded copy of
// the input is materialized. See docs/native_cnn_design.md §6.

#include "tf_conv2d_internal.h"
#include "tf_internal.h"  // export macro, Storage/as_storage, TF_GUARD, set_error

namespace tf {

// ---------------------------------------------------------------------------
// H9 dispatch predicates. Total, pure, allocation-free, and functions of the
// integer geometry alone; a false answer selects the retained generic path and
// is never an error. See tf_conv2d_internal.h for the measured justification
// of kConv2dMinSweptExtent. They read no dtype — which is exactly why the
// same layout takes the same traversal at both element widths (I5).
// ---------------------------------------------------------------------------

bool conv2d_sweep_extent_is_worthwhile(
    int64_t input_width, int64_t output_width) noexcept {
    const int64_t swept =
        (input_width < output_width) ? input_width : output_width;
    return swept >= kConv2dMinSweptExtent;
}

bool conv2d_forward_prefers_row_sweep(
    int64_t input_width, int64_t output_width) noexcept {
    return conv2d_sweep_extent_is_worthwhile(input_width, output_width);
}

bool conv2d_input_backward_prefers_gather(
    int64_t stride_height, int64_t stride_width,
    int64_t input_width, int64_t output_width) noexcept {
    return stride_height == 1 && stride_width == 1 &&
           conv2d_sweep_extent_is_worthwhile(input_width, output_width);
}

bool conv2d_weight_backward_prefers_gather(
    int64_t input_width, int64_t output_width) noexcept {
    return conv2d_sweep_extent_is_worthwhile(input_width, output_width);
}

}  // namespace tf

// ---------------------------------------------------------------------------
// Exported C ABI wrappers (Phase D, milestones D3/D6).
//
// ``tf_core_conv2d_forward`` and the two backward wrappers are the
// exception-guarded boundaries between the Python/Core layer and the
// internal noexcept arithmetic. The Core wrapper (backends/cpp.py) owns the
// *shape/argument* validation and, by Policy B (docs/native_cnn_design.md
// §5), guarantees every operand it passes is **row-major contiguous** — so
// the raw ABI takes only storage handles + per-operand offsets + the
// integer dimensions, never stride arrays, and computes row-major offsets
// itself.
//
// Even though the Core layer validates first, the boundary still defends
// itself against a direct (mis)call: mixed operand dtypes, null required
// handles, non-positive dimensions, negative padding/offsets, an output
// shape that disagrees with the locked floor formula, integer overflow in
// the shape/span products, and any storage span that would read or write
// outside its allocation are all rejected with TF_ERROR_INVALID *before*
// the internal kernel runs. The internal kernels neither allocate nor
// throw, so the only failure these wrappers can surface is invalid
// metadata; the guard is present regardless, per the ABI error contract.
//
// Phase I, milestone I5 made all three exports dtype-general. Each keeps
// its symbol, its argument list, its calling convention, its traversal
// tiers, its validation, and its ownership contract; the only changes are
// that ``tf::require_float64`` ("this operation has not been generalized")
// became ``tf::require_matching_dtype`` ("it has been, and its operands
// must agree"), and that one ``switch`` per exported call now selects the
// instantiation. Every participating handle — bias included, when present —
// must carry the same dtype; there is no casting and no promotion.
// ---------------------------------------------------------------------------

using tf::as_storage;

namespace {

// Checked int64 multiply/add for non-negative operands: return false on
// overflow instead of wrapping, so a bogus dimension can never turn into a
// small (passing) span through silent int64 overflow.
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

// A contiguous operand of ``count`` elements beginning at ``offset`` must
// fit inside a storage holding ``size`` elements. ``offset``/``count`` are
// already known non-negative here.
bool span_within(int64_t offset, int64_t count, int64_t size) {
    int64_t end;
    if (!checked_add(offset, count, end)) {
        return false;  // offset + count overflowed -> cannot fit
    }
    return end <= size;
}

// Product of four non-negative extents with overflow detection.
bool numel4(int64_t a, int64_t b, int64_t c, int64_t d, int64_t& out) {
    return checked_mul(a, b, out) && checked_mul(out, c, out) &&
           checked_mul(out, d, out);
}

// Recompute one spatial output extent from the locked floor formula and
// confirm the caller's claimed value agrees. ``num`` (padded - kernel) is
// kept non-negative: a negative value means the kernel does not fit the
// padded input, which would make the output extent < 1. With ``num >= 0``
// and ``stride >= 1``, C++ integer division equals the floor the Python
// side computes with ``//``. Returns nullptr on success, else an error
// message.
const char* check_output_dim(
    int64_t size, int64_t kernel, int64_t stride, int64_t pad, int64_t claimed) {
    int64_t two_pad, padded;
    if (!checked_mul(2, pad, two_pad) || !checked_add(size, two_pad, padded)) {
        return "conv2d_forward: output-shape arithmetic overflow";
    }
    const int64_t num = padded - kernel;
    if (num < 0) {
        return "conv2d_forward: kernel does not fit the padded input";
    }
    const int64_t expected = num / stride + 1;  // floor: num >= 0, stride >= 1
    if (claimed != expected) {
        return "conv2d_forward: output dimension disagrees with the computed "
               "shape";
    }
    return nullptr;
}

// -- Phase I, milestone I5: the one dispatch per exported call --------------
//
// Three helpers, one per export, each holding the typed body its export's
// **single** ``switch`` selects (design §8.1). They sit above the
// traversals and below the exports, so the dtype decision is made exactly
// once — after the export's validation and before any compute — and nothing
// beneath them branches on dtype again: not the geometry predicate, not the
// tap ranges, not the accumulator, and emphatically not any element.
//
// Both arms take the *same source*. ``T = double`` is the pre-I5 call
// statement for statement, so Phase H's measured H9 traversals are
// preserved rather than re-derived, and ``T = float`` cannot drift from
// them. The typed pointers are formed here, after validation, from the tag
// the dispatch already proved — ``tf::storage_typed<T>`` is the one sound
// recovery of a typed buffer.

template <class T>
void conv2d_forward_dispatch(
    const void* input_handle, int64_t input_offset,
    const void* weight_handle, int64_t weight_offset,
    const void* bias_handle, int64_t bias_offset,
    void* output_handle,
    int64_t batch, int64_t in_channels,
    int64_t input_height, int64_t input_width,
    int64_t out_channels, int64_t kernel_height, int64_t kernel_width,
    int64_t stride_height, int64_t stride_width,
    int64_t pad_height, int64_t pad_width,
    int64_t output_height, int64_t output_width
) {
    const T* bias = nullptr;
    if (bias_handle != nullptr) {
        bias = tf::storage_typed<T>(bias_handle) + bias_offset;
    }
    const T* input = tf::storage_typed<T>(input_handle) + input_offset;
    const T* weight = tf::storage_typed<T>(weight_handle) + weight_offset;
    T* output = tf::storage_typed<T>(output_handle);
    tf::conv2d_forward_contiguous(
        input, weight, bias, output,
        batch, in_channels, input_height, input_width,
        out_channels, kernel_height, kernel_width,
        stride_height, stride_width, pad_height, pad_width,
        output_height, output_width);
}

template <class T>
void conv2d_input_backward_dispatch(
    const void* grad_output_handle, int64_t grad_output_offset,
    const void* weight_handle, int64_t weight_offset,
    void* grad_input_handle,
    int64_t batch, int64_t in_channels,
    int64_t input_height, int64_t input_width,
    int64_t out_channels, int64_t kernel_height, int64_t kernel_width,
    int64_t stride_height, int64_t stride_width,
    int64_t pad_height, int64_t pad_width,
    int64_t output_height, int64_t output_width
) {
    const T* grad_output =
        tf::storage_typed<T>(grad_output_handle) + grad_output_offset;
    const T* weight = tf::storage_typed<T>(weight_handle) + weight_offset;
    T* grad_input = tf::storage_typed<T>(grad_input_handle);
    tf::conv2d_input_backward_contiguous(
        grad_output, weight, grad_input,
        batch, in_channels, input_height, input_width,
        out_channels, kernel_height, kernel_width,
        stride_height, stride_width, pad_height, pad_width,
        output_height, output_width);
}

template <class T>
void conv2d_weight_backward_dispatch(
    const void* grad_output_handle, int64_t grad_output_offset,
    const void* input_handle, int64_t input_offset,
    void* grad_weight_handle,
    int64_t batch, int64_t in_channels,
    int64_t input_height, int64_t input_width,
    int64_t out_channels, int64_t kernel_height, int64_t kernel_width,
    int64_t stride_height, int64_t stride_width,
    int64_t pad_height, int64_t pad_width,
    int64_t output_height, int64_t output_width
) {
    const T* grad_output =
        tf::storage_typed<T>(grad_output_handle) + grad_output_offset;
    const T* input = tf::storage_typed<T>(input_handle) + input_offset;
    T* grad_weight = tf::storage_typed<T>(grad_weight_handle);
    tf::conv2d_weight_backward_contiguous(
        grad_output, input, grad_weight,
        batch, in_channels, input_height, input_width,
        out_channels, kernel_height, kernel_width,
        stride_height, stride_width, pad_height, pad_width,
        output_height, output_width);
}

}  // namespace

TF_EXPORT void tf_core_conv2d_forward(
    const void* input_handle, int64_t input_offset,
    const void* weight_handle, int64_t weight_offset,
    const void* bias_handle, int64_t bias_offset,
    void* output_handle,
    int64_t batch,
    int64_t in_channels,
    int64_t input_height,
    int64_t input_width,
    int64_t out_channels,
    int64_t kernel_height,
    int64_t kernel_width,
    int64_t stride_height,
    int64_t stride_width,
    int64_t pad_height,
    int64_t pad_width,
    int64_t output_height,
    int64_t output_width) {
    TF_GUARD_BEGIN
    // The dtype guard runs first (I5, matching the I3/I4 exports): a call
    // that is both mixed-dtype and otherwise malformed reports the dtype.
    // Every participating handle — the nullable bias included when it is
    // present — must agree; the list form skips nulls, so the export's own
    // null validation below keeps its message and its ordering.
    if (!tf::require_matching_dtype(
            "tf_core_conv2d_forward",
            {input_handle, weight_handle, bias_handle, output_handle})) {
        return;
    }
    // -- required handles (bias is the one nullable handle: null = no bias) --
    if (input_handle == nullptr || weight_handle == nullptr ||
        output_handle == nullptr) {
        tf::set_error(TF_ERROR_INVALID,
                      "conv2d_forward: null required storage handle");
        return;
    }
    // -- dimension / kernel / stride bounds --
    if (batch < 1 || in_channels < 1 || input_height < 1 || input_width < 1 ||
        out_channels < 1 || kernel_height < 1 || kernel_width < 1 ||
        stride_height < 1 || stride_width < 1) {
        tf::set_error(
            TF_ERROR_INVALID,
            "conv2d_forward: batch, channels, spatial, kernel, and stride "
            "extents must each be >= 1");
        return;
    }
    if (pad_height < 0 || pad_width < 0) {
        tf::set_error(TF_ERROR_INVALID,
                      "conv2d_forward: padding must be >= 0");
        return;
    }
    if (input_offset < 0 || weight_offset < 0 || bias_offset < 0 ||
        output_height < 1 || output_width < 1) {
        tf::set_error(TF_ERROR_INVALID,
                      "conv2d_forward: negative offset or non-positive output "
                      "extent");
        return;
    }
    // -- output shape must equal the locked floor formula --
    if (const char* err = check_output_dim(
            input_height, kernel_height, stride_height, pad_height,
            output_height)) {
        tf::set_error(TF_ERROR_INVALID, err);
        return;
    }
    if (const char* err = check_output_dim(
            input_width, kernel_width, stride_width, pad_width,
            output_width)) {
        tf::set_error(TF_ERROR_INVALID, err);
        return;
    }
    // -- storage spans must fit their allocations (overflow-checked) --
    int64_t input_count, weight_count, output_count;
    if (!numel4(batch, in_channels, input_height, input_width, input_count) ||
        !numel4(out_channels, in_channels, kernel_height, kernel_width,
                weight_count) ||
        !numel4(batch, out_channels, output_height, output_width,
                output_count)) {
        tf::set_error(TF_ERROR_INVALID,
                      "conv2d_forward: shape product overflows int64");
        return;
    }
    if (!span_within(input_offset, input_count,
                     as_storage(input_handle)->size)) {
        tf::set_error(TF_ERROR_INVALID,
                      "conv2d_forward: input span exceeds its storage");
        return;
    }
    if (!span_within(weight_offset, weight_count,
                     as_storage(weight_handle)->size)) {
        tf::set_error(TF_ERROR_INVALID,
                      "conv2d_forward: weight span exceeds its storage");
        return;
    }
    if (!span_within(0, output_count, as_storage(output_handle)->size)) {
        tf::set_error(TF_ERROR_INVALID,
                      "conv2d_forward: output span exceeds its storage");
        return;
    }
    if (bias_handle != nullptr &&
        !span_within(bias_offset, out_channels,
                     as_storage(bias_handle)->size)) {
        tf::set_error(TF_ERROR_INVALID,
                      "conv2d_forward: bias span exceeds its storage");
        return;
    }
    // -- validated: one dtype dispatch into the internal noexcept kernels --
    switch (tf::dispatch_dtype(
        {input_handle, weight_handle, bias_handle, output_handle})) {
        case tf::Dtype::Float32:
            conv2d_forward_dispatch<float>(
                input_handle, input_offset, weight_handle, weight_offset,
                bias_handle, bias_offset, output_handle,
                batch, in_channels, input_height, input_width,
                out_channels, kernel_height, kernel_width,
                stride_height, stride_width, pad_height, pad_width,
                output_height, output_width);
            break;
        case tf::Dtype::Float64:
            conv2d_forward_dispatch<double>(
                input_handle, input_offset, weight_handle, weight_offset,
                bias_handle, bias_offset, output_handle,
                batch, in_channels, input_height, input_width,
                out_channels, kernel_height, kernel_width,
                stride_height, stride_width, pad_height, pad_width,
                output_height, output_width);
            break;
    }
    TF_GUARD_END_VOID()
}

// ---------------------------------------------------------------------------
// Exported C ABI backward wrappers (Phase D, milestone D6).
//
// ``tf_core_conv2d_input_backward`` and ``tf_core_conv2d_weight_backward``
// are the exception-guarded boundaries over the D4/D5 internal noexcept
// gradient kernels. They share the forward wrapper's contract exactly (same
// file-local checked-arithmetic helpers, same validation classes, same
// error style): the raw ABI takes storage handles + per-operand offsets +
// the integer dimensions, **never stride arrays**, so it interprets each
// span as canonical contiguous data and bounds-checks it — logical
// contiguity remains the caller precondition the NativeTensorCore layer
// guarantees by Policy-B copy-then-compute. Neither allocates or frees
// caller-owned storage, and neither mutates its read-only operands. There
// is **no** bias-gradient C ABI symbol — that gradient composes from the
// existing native ``sum`` reduction (docs/native_cnn_design.md §7.3),
// which I4 already generalized, so a float32 bias gradient needs nothing
// from this file.
// ---------------------------------------------------------------------------

// Shared validation for both backward wrappers: the dimension/kernel/stride/
// padding/offset bounds and the output-shape agreement identical to the
// forward wrapper. Returns nullptr on success, else an error message. The
// caller checks required handles and storage spans separately (the operand
// roles differ between the two backward directions).
namespace {

const char* validate_conv2d_backward_dims(
    int64_t batch, int64_t in_channels,
    int64_t input_height, int64_t input_width,
    int64_t out_channels, int64_t kernel_height, int64_t kernel_width,
    int64_t stride_height, int64_t stride_width,
    int64_t pad_height, int64_t pad_width,
    int64_t output_height, int64_t output_width,
    int64_t grad_output_offset, int64_t other_offset) {
    if (batch < 1 || in_channels < 1 || input_height < 1 || input_width < 1 ||
        out_channels < 1 || kernel_height < 1 || kernel_width < 1 ||
        stride_height < 1 || stride_width < 1) {
        return "conv2d backward: batch, channels, spatial, kernel, and stride "
               "extents must each be >= 1";
    }
    if (pad_height < 0 || pad_width < 0) {
        return "conv2d backward: padding must be >= 0";
    }
    if (grad_output_offset < 0 || other_offset < 0 ||
        output_height < 1 || output_width < 1) {
        return "conv2d backward: negative offset or non-positive output extent";
    }
    if (const char* err = check_output_dim(
            input_height, kernel_height, stride_height, pad_height,
            output_height)) {
        return err;
    }
    if (const char* err = check_output_dim(
            input_width, kernel_width, stride_width, pad_width,
            output_width)) {
        return err;
    }
    return nullptr;
}

}  // namespace

// Gradient w.r.t. the Conv2d input. grad_output (NCHW) + weight (OIHW) ->
// grad_input (NCHW, caller-allocated, offset 0).
TF_EXPORT void tf_core_conv2d_input_backward(
    const void* grad_output_handle, int64_t grad_output_offset,
    const void* weight_handle, int64_t weight_offset,
    void* grad_input_handle,
    int64_t batch,
    int64_t in_channels,
    int64_t input_height,
    int64_t input_width,
    int64_t out_channels,
    int64_t kernel_height,
    int64_t kernel_width,
    int64_t stride_height,
    int64_t stride_width,
    int64_t pad_height,
    int64_t pad_width,
    int64_t output_height,
    int64_t output_width) {
    TF_GUARD_BEGIN
    if (!tf::require_matching_dtype(
            "tf_core_conv2d_input_backward",
            {grad_output_handle, weight_handle, grad_input_handle})) {
        return;
    }
    if (grad_output_handle == nullptr || weight_handle == nullptr ||
        grad_input_handle == nullptr) {
        tf::set_error(TF_ERROR_INVALID,
                      "conv2d_input_backward: null required storage handle");
        return;
    }
    if (const char* err = validate_conv2d_backward_dims(
            batch, in_channels, input_height, input_width,
            out_channels, kernel_height, kernel_width,
            stride_height, stride_width, pad_height, pad_width,
            output_height, output_width, grad_output_offset, weight_offset)) {
        tf::set_error(TF_ERROR_INVALID, err);
        return;
    }
    int64_t grad_output_count, weight_count, grad_input_count;
    if (!numel4(batch, out_channels, output_height, output_width,
                grad_output_count) ||
        !numel4(out_channels, in_channels, kernel_height, kernel_width,
                weight_count) ||
        !numel4(batch, in_channels, input_height, input_width,
                grad_input_count)) {
        tf::set_error(TF_ERROR_INVALID,
                      "conv2d_input_backward: shape product overflows int64");
        return;
    }
    if (!span_within(grad_output_offset, grad_output_count,
                     as_storage(grad_output_handle)->size)) {
        tf::set_error(TF_ERROR_INVALID,
                      "conv2d_input_backward: grad_output span exceeds storage");
        return;
    }
    if (!span_within(weight_offset, weight_count,
                     as_storage(weight_handle)->size)) {
        tf::set_error(TF_ERROR_INVALID,
                      "conv2d_input_backward: weight span exceeds storage");
        return;
    }
    if (!span_within(0, grad_input_count,
                     as_storage(grad_input_handle)->size)) {
        tf::set_error(TF_ERROR_INVALID,
                      "conv2d_input_backward: grad_input span exceeds storage");
        return;
    }
    switch (tf::dispatch_dtype(
        {grad_output_handle, weight_handle, grad_input_handle})) {
        case tf::Dtype::Float32:
            conv2d_input_backward_dispatch<float>(
                grad_output_handle, grad_output_offset,
                weight_handle, weight_offset, grad_input_handle,
                batch, in_channels, input_height, input_width,
                out_channels, kernel_height, kernel_width,
                stride_height, stride_width, pad_height, pad_width,
                output_height, output_width);
            break;
        case tf::Dtype::Float64:
            conv2d_input_backward_dispatch<double>(
                grad_output_handle, grad_output_offset,
                weight_handle, weight_offset, grad_input_handle,
                batch, in_channels, input_height, input_width,
                out_channels, kernel_height, kernel_width,
                stride_height, stride_width, pad_height, pad_width,
                output_height, output_width);
            break;
    }
    TF_GUARD_END_VOID()
}

// Gradient w.r.t. the Conv2d weight. grad_output (NCHW) + input (NCHW) ->
// grad_weight (OIHW, caller-allocated, offset 0).
TF_EXPORT void tf_core_conv2d_weight_backward(
    const void* grad_output_handle, int64_t grad_output_offset,
    const void* input_handle, int64_t input_offset,
    void* grad_weight_handle,
    int64_t batch,
    int64_t in_channels,
    int64_t input_height,
    int64_t input_width,
    int64_t out_channels,
    int64_t kernel_height,
    int64_t kernel_width,
    int64_t stride_height,
    int64_t stride_width,
    int64_t pad_height,
    int64_t pad_width,
    int64_t output_height,
    int64_t output_width) {
    TF_GUARD_BEGIN
    if (!tf::require_matching_dtype(
            "tf_core_conv2d_weight_backward",
            {grad_output_handle, input_handle, grad_weight_handle})) {
        return;
    }
    if (grad_output_handle == nullptr || input_handle == nullptr ||
        grad_weight_handle == nullptr) {
        tf::set_error(TF_ERROR_INVALID,
                      "conv2d_weight_backward: null required storage handle");
        return;
    }
    if (const char* err = validate_conv2d_backward_dims(
            batch, in_channels, input_height, input_width,
            out_channels, kernel_height, kernel_width,
            stride_height, stride_width, pad_height, pad_width,
            output_height, output_width, grad_output_offset, input_offset)) {
        tf::set_error(TF_ERROR_INVALID, err);
        return;
    }
    int64_t grad_output_count, input_count, grad_weight_count;
    if (!numel4(batch, out_channels, output_height, output_width,
                grad_output_count) ||
        !numel4(batch, in_channels, input_height, input_width,
                input_count) ||
        !numel4(out_channels, in_channels, kernel_height, kernel_width,
                grad_weight_count)) {
        tf::set_error(TF_ERROR_INVALID,
                      "conv2d_weight_backward: shape product overflows int64");
        return;
    }
    if (!span_within(grad_output_offset, grad_output_count,
                     as_storage(grad_output_handle)->size)) {
        tf::set_error(TF_ERROR_INVALID,
                      "conv2d_weight_backward: grad_output span exceeds storage");
        return;
    }
    if (!span_within(input_offset, input_count,
                     as_storage(input_handle)->size)) {
        tf::set_error(TF_ERROR_INVALID,
                      "conv2d_weight_backward: input span exceeds storage");
        return;
    }
    if (!span_within(0, grad_weight_count,
                     as_storage(grad_weight_handle)->size)) {
        tf::set_error(TF_ERROR_INVALID,
                      "conv2d_weight_backward: grad_weight span exceeds storage");
        return;
    }
    switch (tf::dispatch_dtype(
        {grad_output_handle, input_handle, grad_weight_handle})) {
        case tf::Dtype::Float32:
            conv2d_weight_backward_dispatch<float>(
                grad_output_handle, grad_output_offset,
                input_handle, input_offset, grad_weight_handle,
                batch, in_channels, input_height, input_width,
                out_channels, kernel_height, kernel_width,
                stride_height, stride_width, pad_height, pad_width,
                output_height, output_width);
            break;
        case tf::Dtype::Float64:
            conv2d_weight_backward_dispatch<double>(
                grad_output_handle, grad_output_offset,
                input_handle, input_offset, grad_weight_handle,
                batch, in_channels, input_height, input_width,
                out_channels, kernel_height, kernel_width,
                stride_height, stride_width, pad_height, pad_width,
                output_height, output_width);
            break;
    }
    TF_GUARD_END_VOID()
}
