// MaxPool2d forward (Phase D, milestone D8): the internal CPU float64
// window-maximum compute kernel plus its exported, exception-guarded C ABI
// wrapper.
//
// The kernel writes the pooled output and the parallel *winner* buffer in
// one pass, so the forward value and the saved winner are always produced
// by the same decision (docs/native_cnn_design.md §10). The winner buffer
// is an internal float64 buffer of flat input-plane offsets with a -1
// sentinel for "padding won" (§12); it never becomes a public tensor and
// introduces no new dtype.
//
// The pooling *backward* scatter kernel, its C ABI symbol, and the
// NativeTensor autograd integration are milestone D9 — none of that lives
// here yet.

#include <limits>

#include "tf_internal.h"  // export macro, Storage/as_storage, TF_GUARD, set_error
#include "tf_pooling_internal.h"

namespace {

// Row-major flat offset into a 4-D array of extents (·, dim1, dim2, dim3)
// at index (i0, i1, i2, i3). File-local (never part of the ABI); the
// leading extent is not needed for the offset. Both the NCHW input and the
// NCHW output/winner buffers share this layout. Deliberately duplicated
// from conv2d.cpp rather than promoted to a shared header: each compute
// translation unit keeps its indexing self-contained (design §14).
inline int64_t index4d(
    int64_t i0, int64_t i1, int64_t i2, int64_t i3,
    int64_t dim1, int64_t dim2, int64_t dim3) {
    return ((i0 * dim1 + i1) * dim2 + i2) * dim3 + i3;
}

}  // namespace

namespace tf {

void maxpool2d_forward_contiguous(
    const double* input,
    double* output,
    double* winners,
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
    int64_t output_width) noexcept {
    // Direct nested loops, correctness first: n, c, i, j outer; p, q inner
    // (row-major over the kernel window). Padded coordinates are computed
    // in *signed* int64 so an out-of-bounds position is genuinely negative,
    // never an unsigned wrap; no padded copy of the input is materialized.
    const double negative_infinity = -std::numeric_limits<double>::infinity();
    for (int64_t n = 0; n < batch; ++n) {
        for (int64_t c = 0; c < channels; ++c) {
            for (int64_t i = 0; i < output_height; ++i) {
                for (int64_t j = 0; j < output_width; ++j) {
                    // best_* holds the current selection. ``seen_any`` marks
                    // the fallback anchor (the window's first candidate,
                    // used only when every candidate is NaN); ``seen_number``
                    // marks the first non-NaN candidate, which seeds the
                    // real scan. Afterwards only a strictly greater value
                    // replaces the selection, so ties keep the first
                    // occurrence in row-major window order.
                    double best_value = 0.0;
                    double best_winner = -1.0;
                    bool seen_any = false;
                    bool seen_number = false;
                    for (int64_t p = 0; p < kernel_height; ++p) {
                        const int64_t ih = i * stride_height + p - pad_height;
                        const bool row_inside =
                            (ih >= 0 && ih < input_height);
                        for (int64_t q = 0; q < kernel_width; ++q) {
                            const int64_t iw = j * stride_width + q - pad_width;
                            const bool inside =
                                row_inside && iw >= 0 && iw < input_width;
                            // A padded position conceptually holds -inf and
                            // participates in the selection with winner -1.
                            const double candidate =
                                inside
                                    ? input[index4d(n, c, ih, iw, channels,
                                                    input_height, input_width)]
                                    : negative_infinity;
                            const double candidate_winner =
                                inside
                                    ? static_cast<double>(ih * input_width + iw)
                                    : -1.0;
                            if (!seen_any) {
                                // Deterministic all-NaN fallback anchor.
                                seen_any = true;
                                best_value = candidate;
                                best_winner = candidate_winner;
                            }
                            if (candidate != candidate) {
                                continue;  // NaN never wins (see the header)
                            }
                            if (!seen_number) {
                                seen_number = true;
                                best_value = candidate;
                                best_winner = candidate_winner;
                            } else if (candidate > best_value) {
                                // Strict >: an equal later candidate never
                                // displaces the earlier winner.
                                best_value = candidate;
                                best_winner = candidate_winner;
                            }
                        }
                    }
                    const int64_t out_index = index4d(
                        n, c, i, j, channels, output_height, output_width);
                    output[out_index] = best_value;
                    winners[out_index] = best_winner;
                }
            }
        }
    }
}

}  // namespace tf

// ---------------------------------------------------------------------------
// Exported C ABI wrapper (Phase D, milestone D8).
//
// ``tf_core_maxpool2d_forward`` is the exception-guarded boundary between
// the Python/Core layer and the internal noexcept arithmetic above. It
// follows the Conv2d wrappers' contract exactly: the Core wrapper
// (backends/cpp.py) owns the public shape/argument validation and, by
// Policy B (docs/native_cnn_design.md §5), guarantees the input it passes
// is **row-major contiguous** — so the raw ABI takes only storage handles +
// the input offset + the integer dimensions, never stride arrays, and
// computes row-major offsets itself. Because no stride metadata crosses the
// boundary, the ABI cannot and does not inspect *logical* contiguity: it
// interprets each (handle, offset, dims) span as canonical contiguous data
// and independently guarantees only that the span lies inside its
// allocation, alongside the metadata checks below.
//
// Even though the Core layer validates first, the boundary still defends
// itself against a direct (mis)call: null handles, non-positive extents,
// negative padding/offsets, an output shape disagreeing with the locked
// floor formula, a plane too large for exact float64 winner indices, and —
// using overflow-checked int64 arithmetic — any span that would fall
// outside its allocation are all rejected with TF_ERROR_INVALID before the
// kernel runs. The internal kernel neither allocates nor throws; it writes
// only the caller-allocated output and winner storage and frees nothing.
// ---------------------------------------------------------------------------

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
// fit inside a storage holding ``size`` doubles. ``offset``/``count`` are
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
// cannot smuggle a plane whose offsets would round.
const int64_t kMaxExactPlane = static_cast<int64_t>(1) << 53;

}  // namespace

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
    // -- validated: run the internal noexcept pooling kernel --
    const double* input = as_storage(input_handle)->data + input_offset;
    double* output = as_storage(output_handle)->data;
    double* winners = as_storage(winners_handle)->data;
    tf::maxpool2d_forward_contiguous(
        input, output, winners,
        batch, channels, input_height, input_width,
        kernel_height, kernel_width, stride_height, stride_width,
        pad_height, pad_width, output_height, output_width);
    TF_GUARD_END_VOID()
}
