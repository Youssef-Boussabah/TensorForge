// Internal (non-ABI) declaration of the Conv2d forward compute kernel
// (Phase D, milestone D2). See docs/native_cnn_design.md §6 and §14.
//
// This is deliberately NOT part of the public C ABI: it is a plain C++
// function in ``namespace tf`` with hidden visibility, holding only the
// contiguous CPU float64 cross-correlation arithmetic. It performs no
// allocation, no error reporting, and no mutation of its inputs. The D3
// milestone will add the exported ``tf_core_conv2d_forward`` wrapper
// (TF_GUARD error contract + Storage handles) plus the Python/Core
// validation and output allocation; none of that lives here.
//
// Because the shared library exports only TF_EXPORT symbols, this
// internal function is not reachable from a separately linked test
// binary — so the D2 test (cpp/tests/test_conv2d_forward.cpp) compiles
// cpp/src/conv2d.cpp directly rather than linking the shared library.
//
// ---------------------------------------------------------------------------
// Phase H, milestone H9 — two compute paths per direction, one export each.
//
// Each of the three ``*_contiguous`` entry points below is now a *dispatcher*
// that picks between the retained Phase-D direct loop (the generic reference
// path, still reachable for every supported geometry) and one H9 optimized
// traversal, from the integer geometry it already receives. This reuses the
// dispatch shape H2 (matmul), H5 (copy), H6 (reduction), and H8 (elementwise)
// each proved: no new exported C ABI symbol, no new translation unit, and no
// public control of any kind — no path selector, block-size setter, dispatch
// tracer, profiling counter, or environment-variable mode.
//
// The predicates are total, pure, allocation-free, and functions of the
// integer geometry alone — never of a pointer value, an alignment, a clock,
// an environment variable, or a CPU-feature probe. A false answer selects the
// generic path and is **never an error**.
//
// **Per-destination accumulation order is preserved exactly** on both paths,
// for all three directions; see the proof comments on each optimized kernel.
// ---------------------------------------------------------------------------
#pragma once

#include <cstdint>

namespace tf {

// The minimum swept extent an H9 optimized traversal requires. Below this the
// swept inner loop is too short to pay for its own setup and the retained
// generic path is genuinely faster — measured, not assumed: at a swept extent
// of 1 the optimized forms ran 0.57x-0.93x, at 2 they ran 1.04x-1.38x, and at
// 4 they ran 1.91x-2.40x. Four is the smallest extent at which all three
// directions are clearly ahead. It is a structural minimum on the loop, not a
// tuned constant: it does not vary with the machine, the cache, or the build.
inline constexpr int64_t kConv2dMinSweptExtent = 4;

// All three optimized paths replace a kernel-tap inner loop with a sweep
// along one spatial row, and in every direction that sweep is bounded by
// **both** widths at once: the run of output columns it can cover is limited
// by output_width, and the run of input columns those map to is limited by
// input_width. So ``min(input_width, output_width)`` is the one honest bound
// on all three inner loops, and one shared rule keeps the three predicates
// from drifting apart. (Keying the input gradient on input_width alone was
// measured wrong: a 5-wide input with a 1-wide output sweeps a single element
// and ran 0.73x.)
bool conv2d_sweep_extent_is_worthwhile(
    int64_t input_width, int64_t output_width) noexcept;

// Does the forward take the H9 row sweep?
bool conv2d_forward_prefers_row_sweep(
    int64_t input_width, int64_t output_width) noexcept;

// Does the input gradient take the H9 gather? Besides the shared extent rule
// it requires **unit stride in both axes**: it visits the taps contributing to
// one destination by walking the kernel offsets downward, which reproduces
// the reference's ascending output-position order only when each output
// position maps to a source position one-for-one (see the kernel comment).
bool conv2d_input_backward_prefers_gather(
    int64_t stride_height, int64_t stride_width,
    int64_t input_width, int64_t output_width) noexcept;

// Does the weight gradient take the H9 gather?
bool conv2d_weight_backward_prefers_gather(
    int64_t input_width, int64_t output_width) noexcept;

// Two-dimensional cross-correlation (NOT flipped convolution), CPU
// float64, direct nested loops. Matches the stable tensorforge.nn.Conv2d
// numerically (to a floating-point tolerance; the summation order is
// deterministic but not guaranteed bit-identical to NumPy's einsum).
//
//   out[n, o, i, j] = (bias ? bias[o] : 0)
//                   + sum_{c, p, q} in[n, c, i*sh + p - ph, j*sw + q - pw]
//                                     * weight[o, c, p, q]
//
// with symmetric zero padding applied by skipping source coordinates that
// fall outside the real input (a padded cell contributes 0). Bias, when
// present, is added exactly once per output element. Accumulation runs in
// deterministic c -> p -> q order into a double accumulator.
//
// Layouts (all row-major contiguous float64):
//   input  : NCHW  (batch, in_channels, input_height, input_width)
//   weight : OIHW  (out_channels, in_channels, kernel_height, kernel_width)
//   bias   : (out_channels,), or nullptr for no bias
//   output : NCHW  (batch, out_channels, output_height, output_width) — written
//
// Preconditions (guaranteed by the future D3 wrapper; NOT re-validated
// here — this routine is the inner math, not a validation boundary):
//   * input / weight / output are non-null and each point to contiguous
//     storage of exactly batch*in_channels*input_height*input_width,
//     out_channels*in_channels*kernel_height*kernel_width, and
//     batch*out_channels*output_height*output_width doubles;
//   * bias is nullptr or points to out_channels doubles;
//   * every dimension is positive; stride/kernel >= 1; padding >= 0;
//   * output_height / output_width equal the floor formula
//       floor((input_dim + 2*pad - kernel) / stride) + 1
//     so every output coordinate maps inside the padded grid.
//
// The routine allocates no heap memory and cannot throw (noexcept): it is
// pure arithmetic over caller-owned buffers.
void conv2d_forward_contiguous(
    const double* input,
    const double* weight,
    const double* bias,
    double* output,
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
    int64_t output_width) noexcept;

// Gradient of the Conv2d forward with respect to its input (Phase D,
// milestone D4). The internal CPU float64 scatter-add that is the adjoint
// of the cross-correlation above; pure arithmetic only. Like the forward
// kernel this is deliberately NOT part of the public C ABI: it is a plain
// C++ function in ``namespace tf`` with hidden visibility. The exported
// ``tf_core_conv2d_input_backward`` wrapper (TF_GUARD error contract +
// Storage handles), its ctypes registration, the NativeTensorCore backward
// method, and the NativeTensor.conv2d autograd node are all later
// milestones (D6) — none of that lives here.
//
// Given the forward relation
//   out[n,o,i,j] = bias[o] + sum_{c,p,q} in[n,c, i*sh+p-ph, j*sw+q-pw]
//                                          * weight[o,c,p,q],
// the input gradient scatters each upstream value back through the same
// coordinate map:
//   grad_input[n,c, i*sh+p-ph, j*sw+q-pw] += grad_output[n,o,i,j]
//                                            * weight[o,c,p,q]
// summed over o,i,j (and the kernel taps c,p,q), with padded source
// coordinates that fall outside the real input skipped (the gradient that
// would land on the zero pad border is discarded, matching the stable
// framework). Overlapping windows accumulate through ``+=``. Bias does not
// affect the input gradient, so this kernel neither receives nor reads a
// bias. Accumulation runs in deterministic n -> o -> i -> j -> c -> p -> q
// order into the output span.
//
// Layouts (all row-major contiguous float64):
//   grad_output : NCHW  (batch, out_channels, output_height, output_width)
//   weight      : OIHW  (out_channels, in_channels, kernel_height, kernel_width)
//   grad_input  : NCHW  (batch, in_channels, input_height, input_width) — written
//
// Output initialization: the routine **zero-initializes the entire
// grad_input span itself** (batch*in_channels*input_height*input_width
// doubles) before accumulating, so the caller need NOT pre-zero it; any
// prior contents are fully overwritten/defined.
//
// Preconditions (guaranteed by the future D6 wrapper; NOT re-validated here
// — this routine is the inner math, not a validation boundary):
//   * grad_output / weight / grad_input are non-null and each point to
//     contiguous storage of exactly
//     batch*out_channels*output_height*output_width,
//     out_channels*in_channels*kernel_height*kernel_width, and
//     batch*in_channels*input_height*input_width doubles;
//   * every dimension is positive; stride/kernel >= 1; padding >= 0;
//   * output_height / output_width equal the floor formula
//       floor((input_dim + 2*pad - kernel) / stride) + 1
//     so every output coordinate maps inside the padded grid;
//   * all integer products and offsets are representable in int64.
//
// The routine allocates no heap memory and cannot throw (noexcept): it is
// pure arithmetic over caller-owned buffers. It reads grad_output and
// weight without modifying them and writes only inside the grad_input span.
void conv2d_input_backward_contiguous(
    const double* grad_output,
    const double* weight,
    double* grad_input,
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
    int64_t output_width) noexcept;

// Gradient of the Conv2d forward with respect to its weight (Phase D,
// milestone D5). The internal CPU float64 accumulation that pairs each
// upstream value with the input pixel it multiplied in the forward; pure
// arithmetic only. Like the forward and input-gradient kernels this is
// deliberately NOT part of the public C ABI: a plain C++ function in
// ``namespace tf`` with hidden visibility. The exported
// ``tf_core_conv2d_weight_backward`` wrapper (TF_GUARD error contract +
// Storage handles), its ctypes registration, the NativeTensorCore backward
// method, and the NativeTensor.conv2d autograd node are all later
// milestones (D6) — none of that lives here.
//
// Given the forward relation
//   out[n,o,i,j] = bias[o] + sum_{c,p,q} in[n,c, i*sh+p-ph, j*sw+q-pw]
//                                          * weight[o,c,p,q],
// the weight gradient accumulates, for every valid forward contribution,
//   grad_weight[o,c,p,q] += grad_output[n,o,i,j]
//                           * in[n,c, i*sh+p-ph, j*sw+q-pw]
// summed over n, i, j (matching the stable einsum "no,nckl->ockl"), with
// padded source coordinates that fall outside the real input skipped (they
// contributed 0 in the forward, so they contribute 0 to the gradient — the
// pad border never accumulates into grad_weight). Accumulation runs in
// deterministic n -> o -> i -> j -> c -> p -> q order into the output span.
//
// Layouts (all row-major contiguous float64):
//   grad_output : NCHW  (batch, out_channels, output_height, output_width)
//   input       : NCHW  (batch, in_channels, input_height, input_width)
//   grad_weight : OIHW  (out_channels, in_channels, kernel_height, kernel_width) — written
//
// Output initialization: the routine **zero-initializes the entire
// grad_weight span itself** (out_channels*in_channels*kernel_height*
// kernel_width doubles) before accumulating, so the caller need NOT pre-zero
// it; any prior contents are fully overwritten/defined.
//
// Bias does not appear in the weight gradient, so this kernel neither
// receives nor reads a bias, and grad_weight is independent of whether the
// forward used a bias.
//
// Preconditions (guaranteed by the future D6 wrapper; NOT re-validated here
// — this routine is the inner math, not a validation boundary):
//   * grad_output / input / grad_weight are non-null and each point to
//     contiguous storage of exactly
//     batch*out_channels*output_height*output_width,
//     batch*in_channels*input_height*input_width, and
//     out_channels*in_channels*kernel_height*kernel_width doubles;
//   * every dimension is positive; stride/kernel >= 1; padding >= 0;
//   * output_height / output_width equal the floor formula
//       floor((input_dim + 2*pad - kernel) / stride) + 1
//     so every output coordinate maps inside the padded grid;
//   * all integer products and offsets are representable in int64.
//
// The routine allocates no heap memory and cannot throw (noexcept): it is
// pure arithmetic over caller-owned buffers. It reads grad_output and input
// without modifying them and writes only inside the grad_weight span.
void conv2d_weight_backward_contiguous(
    const double* grad_output,
    const double* input,
    double* grad_weight,
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
    int64_t output_width) noexcept;

// ---------------------------------------------------------------------------
// H9 compute paths. Each ``*_contiguous`` entry above dispatches between the
// ``*_generic`` twin — the Phase-D direct loop, retained verbatim as the
// shipped generic reference path — and the H9 optimized traversal. Both twins
// take the identical argument list as their dispatcher and honor the identical
// contract (same preconditions, same output initialization, same
// per-destination accumulation order, no allocation, noexcept).
//
// The generic paths are what every geometry that fails a predicate runs, and
// they remain the oracle every optimized result is compared against.
// ---------------------------------------------------------------------------

#define TF_CONV2D_KERNEL_PARAMS                                              \
    int64_t batch, int64_t in_channels, int64_t input_height,                \
    int64_t input_width, int64_t out_channels, int64_t kernel_height,        \
    int64_t kernel_width, int64_t stride_height, int64_t stride_width,       \
    int64_t pad_height, int64_t pad_width, int64_t output_height,            \
    int64_t output_width

void conv2d_forward_generic(
    const double* input, const double* weight, const double* bias,
    double* output, TF_CONV2D_KERNEL_PARAMS) noexcept;

void conv2d_forward_row_sweep(
    const double* input, const double* weight, const double* bias,
    double* output, TF_CONV2D_KERNEL_PARAMS) noexcept;

void conv2d_input_backward_generic(
    const double* grad_output, const double* weight, double* grad_input,
    TF_CONV2D_KERNEL_PARAMS) noexcept;

void conv2d_input_backward_gather(
    const double* grad_output, const double* weight, double* grad_input,
    TF_CONV2D_KERNEL_PARAMS) noexcept;

void conv2d_weight_backward_generic(
    const double* grad_output, const double* input, double* grad_weight,
    TF_CONV2D_KERNEL_PARAMS) noexcept;

void conv2d_weight_backward_gather(
    const double* grad_output, const double* input, double* grad_weight,
    TF_CONV2D_KERNEL_PARAMS) noexcept;

}  // namespace tf
