// MaxPool2d (Phase D, milestones D8/D9): the exported, exception-guarded C
// ABI wrappers over the internal noexcept window-maximum and scatter-add
// kernels.
//
// The compute kernels themselves live in cpp/include/tf_pooling_internal.h
// as templates over the *value* element type (Phase I, milestone I5), so
// both the ``float`` and ``double`` instantiations are available to the
// exported wrappers here *and* to the CTests that compile this file
// directly. The winner buffer is not templated and never will be: it is an
// internal **float64** buffer of flat input-plane offsets with a -1
// sentinel for "padding won" (docs/native_cnn_design.md §12,
// docs/native_dtype_float32_design.md §13.3), at every value dtype. It
// never becomes a public tensor and introduces no new dtype.
//
// Backward consumes the saved winners only — it never rereads an input
// value and never recomputes a window maximum. The Python-managed
// ``NativeTensor.maxpool2d`` graph node lives in
// experimental/native_tensor.py; no graph state enters C++.

#include <cmath>

#include "tf_internal.h"  // export macro, Storage/as_storage, TF_GUARD, set_error
#include "tf_pooling_internal.h"

using tf::as_storage;

namespace {

// Checked int64 multiply/add for non-negative operands: return false on
// overflow instead of wrapping, so a bogus dimension can never turn into a
// small (passing) span through silent int64 overflow. File-local, matching
// the equivalent helpers in conv2d.cpp (each compute unit keeps its own —
// design §14 prefers file-local helpers over premature shared surface).
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
// kept non-negative: a negative value means the window does not fit the
// padded input, which would make the output extent < 1. With ``num >= 0``
// and ``stride >= 1``, C++ integer division equals the floor the Python
// side computes with ``//``. Returns nullptr on success, else a message.
const char* check_output_dim(
    int64_t size, int64_t kernel, int64_t stride, int64_t pad, int64_t claimed) {
    int64_t two_pad, padded;
    if (!checked_mul(2, pad, two_pad) || !checked_add(size, two_pad, padded)) {
        return "maxpool2d_forward: output-shape arithmetic overflow";
    }
    const int64_t num = padded - kernel;
    if (num < 0) {
        return "maxpool2d_forward: kernel does not fit the padded input";
    }
    const int64_t expected = num / stride + 1;  // floor: num >= 0, stride >= 1
    if (claimed != expected) {
        return "maxpool2d_forward: output dimension disagrees with the "
               "computed shape";
    }
    return nullptr;
}

// Every winner is stored as a float64, so a flat plane offset is exact
// only while input_height*input_width <= 2^53 (docs/native_cnn_design.md
// §12). Python proves this in arbitrary precision before allocating; the
// ABI re-proves it here in overflow-checked int64 so a direct caller
// cannot smuggle a plane whose offsets would round. The bound is float64's
// at every value dtype — it does not shrink to 2^24 for a float32 pool,
// because the winner buffer does not follow the value dtype (design §13.3).
const int64_t kMaxExactPlane = static_cast<int64_t>(1) << 53;

// Phase I, milestone I5: the winner-buffer dtype guard.
//
// The winner handle is not a numeric operand and takes no part in the value
// dispatch; it must be **exactly float64** whatever the pool's value dtype,
// because its contents are plane offsets whose exactness proof is stated at
// binary64. A float32 winner buffer is refused before anything is read or
// written — reading 4-byte winners through the ``double*`` the kernels use
// would overrun the buffer by a factor of two.
//
// Null passes for exactly ``require_matching_dtype``'s reason: each export
// keeps its own null validation, with its own message and ordering, and
// this guard must not pre-empt it.
bool require_winner_float64(const char* operation, const void* handle) {
    if (handle == nullptr) {
        return true;
    }
    const tf::Dtype dtype = tf::storage_dtype(handle);
    if (tf::dtype_is_float64(dtype)) {
        return true;
    }
    char message[192];
    std::snprintf(message, sizeof message,
                  "%s: the winner buffer must be float64 storage at every "
                  "value dtype (winner offsets are exact float64 plane "
                  "indices); got %s",
                  operation, tf::dtype_name(dtype));
    tf::set_error(TF_ERROR_INVALID, message);
    return false;
}

// -- Phase I, milestone I5: the one dispatch per exported call --------------
//
// Two helpers, one per export, each holding the typed body its export's
// **single** ``switch`` selects (design §8.1). The dtype decision is made
// exactly once, on the *value* handles alone — the winner handle carries
// index metadata, was proved float64 above, and is reached through the
// unchanged ``storage_f64`` accessor in both arms. Nothing beneath these
// branches on dtype again: not the window scan, not the comparison, not
// the scatter, and emphatically not any element.

template <class T>
void maxpool2d_forward_dispatch(
    const void* input_handle, int64_t input_offset,
    void* output_handle, void* winners_handle,
    int64_t batch, int64_t channels,
    int64_t input_height, int64_t input_width,
    int64_t kernel_height, int64_t kernel_width,
    int64_t stride_height, int64_t stride_width,
    int64_t pad_height, int64_t pad_width,
    int64_t output_height, int64_t output_width
) {
    const T* input = tf::storage_typed<T>(input_handle) + input_offset;
    T* output = tf::storage_typed<T>(output_handle);
    double* winners = tf::storage_f64(winners_handle);
    tf::maxpool2d_forward_contiguous(
        input, output, winners,
        batch, channels, input_height, input_width,
        kernel_height, kernel_width, stride_height, stride_width,
        pad_height, pad_width, output_height, output_width);
}

template <class T>
void maxpool2d_backward_dispatch(
    const void* grad_output_handle, int64_t grad_output_offset,
    const void* winners_handle, int64_t winners_offset,
    void* grad_input_handle,
    int64_t batch, int64_t channels,
    int64_t input_height, int64_t input_width,
    int64_t output_height, int64_t output_width
) {
    const T* grad_output =
        tf::storage_typed<T>(grad_output_handle) + grad_output_offset;
    const double* winners =
        tf::storage_f64(winners_handle) + winners_offset;
    T* grad_input = tf::storage_typed<T>(grad_input_handle);
    tf::maxpool2d_backward_contiguous(
        grad_output, winners, grad_input,
        batch, channels, input_height, input_width,
        output_height, output_width);
}

}  // namespace

// ---------------------------------------------------------------------------
// Exported C ABI wrapper (Phase D, milestone D8).
//
// ``tf_core_maxpool2d_forward`` is the exception-guarded boundary between
// the Python/Core layer and the internal noexcept arithmetic. It follows
// the Conv2d wrappers' contract exactly: the Core wrapper (backends/cpp.py)
// owns the public shape/argument validation and, by Policy B
// (docs/native_cnn_design.md §5), guarantees the input it passes is
// **row-major contiguous** — so the raw ABI takes only storage handles +
// the input offset + the integer dimensions, never stride arrays, and
// computes row-major offsets itself. Because no stride metadata crosses the
// boundary, the ABI cannot and does not inspect *logical* contiguity: it
// interprets each (handle, offset, dims) span as canonical contiguous data
// and independently guarantees only that the span lies inside its
// allocation, alongside the metadata checks below.
//
// Even though the Core layer validates first, the boundary still defends
// itself against a direct (mis)call: mixed value dtypes, a non-float64
// winner buffer, null handles, non-positive extents, negative
// padding/offsets, an output shape disagreeing with the locked floor
// formula, a plane too large for exact float64 winner indices, and — using
// overflow-checked int64 arithmetic — any span that would fall outside its
// allocation are all rejected with TF_ERROR_INVALID before the kernel runs.
// The internal kernel neither allocates nor throws; it writes only the
// caller-allocated output and winner storage and frees nothing.
// ---------------------------------------------------------------------------

TF_EXPORT void tf_core_maxpool2d_forward(
    const void* input_handle, int64_t input_offset,
    void* output_handle,
    void* winners_handle,
    int64_t batch,
    int64_t channels,
    int64_t input_height,
    int64_t input_width,
    int64_t kernel_height,
    int64_t kernel_width,
    int64_t stride_height,
    int64_t stride_width,
    int64_t pad_height,
    int64_t pad_width,
    int64_t output_height,
    int64_t output_width) {
    TF_GUARD_BEGIN
    // The dtype guards run first (I5, matching the I3/I4 exports): the
    // *value* handles must agree, and the winner buffer must be float64 at
    // every value dtype (design §13.3). A call that is both mixed-dtype and
    // otherwise malformed reports the dtype.
    // K1: the dtype-role guard runs first — an int64 operand is
    // a role error, never a promotion opportunity (§22.4).
    if (!tf::require_floating(
            "tf_core_maxpool2d_forward",
            {input_handle, output_handle})) {
        return;
    }
    if (!tf::require_matching_dtype("tf_core_maxpool2d_forward",
                                    {input_handle, output_handle})) {
        return;
    }
    if (!require_winner_float64("maxpool2d_forward", winners_handle)) {
        return;
    }
    // -- required handles (none is nullable for pooling) --
    if (input_handle == nullptr || output_handle == nullptr ||
        winners_handle == nullptr) {
        tf::set_error(TF_ERROR_INVALID,
                      "maxpool2d_forward: null required storage handle");
        return;
    }
    // -- dimension / kernel / stride bounds --
    if (batch < 1 || channels < 1 || input_height < 1 || input_width < 1 ||
        kernel_height < 1 || kernel_width < 1 || stride_height < 1 ||
        stride_width < 1) {
        tf::set_error(
            TF_ERROR_INVALID,
            "maxpool2d_forward: batch, channels, spatial, kernel, and stride "
            "extents must each be >= 1");
        return;
    }
    if (pad_height < 0 || pad_width < 0) {
        tf::set_error(TF_ERROR_INVALID,
                      "maxpool2d_forward: padding must be >= 0");
        return;
    }
    if (input_offset < 0 || output_height < 1 || output_width < 1) {
        tf::set_error(TF_ERROR_INVALID,
                      "maxpool2d_forward: negative offset or non-positive "
                      "output extent");
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
    // -- winner indices must be exactly representable in float64 --
    int64_t plane;
    if (!checked_mul(input_height, input_width, plane) ||
        plane > kMaxExactPlane) {
        tf::set_error(TF_ERROR_INVALID,
                      "maxpool2d_forward: input plane too large for exact "
                      "float64 winner indices");
        return;
    }
    // -- storage spans must fit their allocations (overflow-checked) --
    int64_t input_count, output_count;
    if (!numel4(batch, channels, input_height, input_width, input_count) ||
        !numel4(batch, channels, output_height, output_width, output_count)) {
        tf::set_error(TF_ERROR_INVALID,
                      "maxpool2d_forward: shape product overflows int64");
        return;
    }
    if (!span_within(input_offset, input_count,
                     as_storage(input_handle)->size)) {
        tf::set_error(TF_ERROR_INVALID,
                      "maxpool2d_forward: input span exceeds its storage");
        return;
    }
    if (!span_within(0, output_count, as_storage(output_handle)->size)) {
        tf::set_error(TF_ERROR_INVALID,
                      "maxpool2d_forward: output span exceeds its storage");
        return;
    }
    if (!span_within(0, output_count, as_storage(winners_handle)->size)) {
        tf::set_error(TF_ERROR_INVALID,
                      "maxpool2d_forward: winner span exceeds its storage");
        return;
    }
    // -- validated: one value-dtype dispatch into the internal kernel --
    switch (tf::dispatch_dtype({input_handle, output_handle})) {
        case tf::Dtype::Float32:
            maxpool2d_forward_dispatch<float>(
                input_handle, input_offset, output_handle, winners_handle,
                batch, channels, input_height, input_width,
                kernel_height, kernel_width, stride_height, stride_width,
                pad_height, pad_width, output_height, output_width);
            break;
        case tf::Dtype::Float64:
            maxpool2d_forward_dispatch<double>(
                input_handle, input_offset, output_handle, winners_handle,
                batch, channels, input_height, input_width,
                kernel_height, kernel_width, stride_height, stride_width,
                pad_height, pad_width, output_height, output_width);
            break;
        case tf::Dtype::Int64:
            // Unreachable: require_floating rejected an int64 operand above.
            break;
    }
    TF_GUARD_END_VOID()
}

// ---------------------------------------------------------------------------
// Exported C ABI backward wrapper (Phase D, milestone D9).
//
// ``tf_core_maxpool2d_backward`` is the exception-guarded boundary over the
// internal noexcept scatter-add. It shares the forward wrapper's contract
// exactly (same file-local checked arithmetic, same validation classes,
// same error style): storage handles + per-operand offsets + integer
// dimensions, **never stride arrays**, so it interprets each span as
// canonical contiguous data and bounds-checks it — logical contiguity stays
// the caller precondition the NativeTensorCore layer guarantees by
// Policy-B copy-then-compute. It allocates and frees nothing and mutates
// only the caller-allocated grad_input.
//
// It takes **no kernel/stride/padding metadata**: the saved winners fully
// determine the gradient routing, so backward never reconstructs window
// geometry (docs/native_cnn_design.md §11).
//
// Beyond the usual metadata checks it **validates every winner value**
// before the kernel runs. The winner buffer is private and produced by the
// D8 forward, but this boundary is correctness-first: a value that is not
// exactly -1.0 and not an exact, finite, in-range non-negative integer is
// rejected with TF_ERROR_INVALID rather than rounded, truncated, or used
// to scatter outside grad_input.
// ---------------------------------------------------------------------------

TF_EXPORT void tf_core_maxpool2d_backward(
    const void* grad_output_handle, int64_t grad_output_offset,
    const void* winners_handle, int64_t winners_offset,
    void* grad_input_handle,
    int64_t batch,
    int64_t channels,
    int64_t input_height,
    int64_t input_width,
    int64_t output_height,
    int64_t output_width) {
    TF_GUARD_BEGIN
    // The dtype guards run first (I5): the upstream gradient and the
    // gradient destination must agree, and the winner buffer must be
    // float64 at every value dtype (design §13.3) — it is index metadata,
    // not a numeric operand, so it takes no part in the value dispatch.
    // K1: the dtype-role guard runs first — an int64 operand is
    // a role error, never a promotion opportunity (§22.4).
    if (!tf::require_floating(
            "tf_core_maxpool2d_backward",
            {grad_output_handle, grad_input_handle})) {
        return;
    }
    if (!tf::require_matching_dtype("tf_core_maxpool2d_backward",
                                    {grad_output_handle, grad_input_handle})) {
        return;
    }
    if (!require_winner_float64("maxpool2d_backward", winners_handle)) {
        return;
    }
    if (grad_output_handle == nullptr || winners_handle == nullptr ||
        grad_input_handle == nullptr) {
        tf::set_error(TF_ERROR_INVALID,
                      "maxpool2d_backward: null required storage handle");
        return;
    }
    if (batch < 1 || channels < 1 || input_height < 1 || input_width < 1 ||
        output_height < 1 || output_width < 1) {
        tf::set_error(TF_ERROR_INVALID,
                      "maxpool2d_backward: batch, channels, and spatial "
                      "extents must each be >= 1");
        return;
    }
    if (grad_output_offset < 0 || winners_offset < 0) {
        tf::set_error(TF_ERROR_INVALID,
                      "maxpool2d_backward: negative storage offset");
        return;
    }
    // The winner domain is [0, H*W - 1] plus the -1 sentinel, so the same
    // float64 exactness bound the forward proves must hold here too.
    int64_t plane;
    if (!checked_mul(input_height, input_width, plane) ||
        plane > kMaxExactPlane) {
        tf::set_error(TF_ERROR_INVALID,
                      "maxpool2d_backward: input plane too large for exact "
                      "float64 winner indices");
        return;
    }
    int64_t grad_output_count, grad_input_count;
    if (!numel4(batch, channels, output_height, output_width,
                grad_output_count) ||
        !numel4(batch, channels, input_height, input_width,
                grad_input_count)) {
        tf::set_error(TF_ERROR_INVALID,
                      "maxpool2d_backward: shape product overflows int64");
        return;
    }
    if (!span_within(grad_output_offset, grad_output_count,
                     as_storage(grad_output_handle)->size)) {
        tf::set_error(TF_ERROR_INVALID,
                      "maxpool2d_backward: grad_output span exceeds storage");
        return;
    }
    if (!span_within(winners_offset, grad_output_count,
                     as_storage(winners_handle)->size)) {
        tf::set_error(TF_ERROR_INVALID,
                      "maxpool2d_backward: winner span exceeds storage");
        return;
    }
    if (!span_within(0, grad_input_count,
                     as_storage(grad_input_handle)->size)) {
        tf::set_error(TF_ERROR_INVALID,
                      "maxpool2d_backward: grad_input span exceeds storage");
        return;
    }
    // Winner validation: exactly -1.0, or a finite, non-negative, exactly
    // integral offset no larger than plane - 1. Nothing else is accepted,
    // and nothing is silently rounded. Checked for the whole buffer before
    // the kernel writes anything, so a malformed winner leaves grad_input
    // untouched. The winners are float64 — proved above — at every value
    // dtype, so this loop reads them through the unchanged float64
    // accessor and needs no dispatch.
    const double* winners = tf::storage_f64(winners_handle) + winners_offset;
    for (int64_t idx = 0; idx < grad_output_count; ++idx) {
        const double winner = winners[idx];
        if (winner == -1.0) {
            continue;  // padding won: backward drops this gradient
        }
        const bool valid = std::isfinite(winner) && winner >= 0.0 &&
                           std::floor(winner) == winner &&
                           winner <= static_cast<double>(plane - 1);
        if (!valid) {
            tf::set_error(
                TF_ERROR_INVALID,
                "maxpool2d_backward: winner buffer holds an invalid index "
                "(each entry must be -1 or an exact integer in "
                "[0, input_height*input_width - 1])");
            return;
        }
    }
    // -- validated: one value-dtype dispatch into the internal kernel --
    switch (tf::dispatch_dtype({grad_output_handle, grad_input_handle})) {
        case tf::Dtype::Float32:
            maxpool2d_backward_dispatch<float>(
                grad_output_handle, grad_output_offset,
                winners_handle, winners_offset, grad_input_handle,
                batch, channels, input_height, input_width,
                output_height, output_width);
            break;
        case tf::Dtype::Float64:
            maxpool2d_backward_dispatch<double>(
                grad_output_handle, grad_output_offset,
                winners_handle, winners_offset, grad_input_handle,
                batch, channels, input_height, input_width,
                output_height, output_width);
            break;
        case tf::Dtype::Int64:
            // Unreachable: require_floating rejected an int64 operand above.
            break;
    }
    TF_GUARD_END_VOID()
}
