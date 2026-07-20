// Dependency-free C++ test for the MaxPool2d backward path (Phase D,
// milestone D9). No GoogleTest/Catch2 — a plain executable that prints
// failures and returns a nonzero exit code if any check fails, so CTest
// reports pass/fail.
//
// This binary compiles cpp/src/pooling.cpp (and error.cpp) directly (see
// the CMake TF_BUILD_TESTS target). That gives it both layers:
//   * the internal, hidden tf::maxpool2d_backward_contiguous kernel, which
//     is exercised only with VALID winners (it is the inner math, not a
//     validation boundary); and
//   * the exported, guarded tf_core_maxpool2d_backward wrapper, which is
//     where malformed winner values must be rejected — tested here through
//     plain tf::Storage handles plus the thread-local error slot.
//
// The stable-framework parity case embeds a gradient generated once from
// tensorforge.nn.MaxPool2d; the test never touches Python at runtime.

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <vector>

#include "tf_internal.h"  // TF_EXPORT, tf::Storage, error accessors, TfStatus
#include "tf_pooling_internal.h"

// The exported wrapper under test. Declared with the same TF_EXPORT macro
// the definition uses, so the linkage matches exactly.
TF_EXPORT void tf_core_maxpool2d_backward(
    const void* grad_output_handle, int64_t grad_output_offset,
    const void* winners_handle, int64_t winners_offset,
    void* grad_input_handle,
    int64_t batch, int64_t channels,
    int64_t input_height, int64_t input_width,
    int64_t output_height, int64_t output_width);

namespace {

int g_failures = 0;

const double kNegInf = -std::numeric_limits<double>::infinity();
const double kPosInf = std::numeric_limits<double>::infinity();
const double kNaN = std::numeric_limits<double>::quiet_NaN();

void check(bool condition, const char* what) {
    if (!condition) {
        std::printf("FAIL: %s\n", what);
        ++g_failures;
    }
}

// Strict equality — every hand-computed case uses exactly representable
// values, and the scatter is a plain ordered sum of them.
void check_eq(double got, double want, const char* what) {
    if (got != want) {
        std::printf("FAIL: %s (got %.17g, want %.17g)\n", what, got, want);
        ++g_failures;
    }
}

void check_close(double got, double want, double tol, const char* what) {
    if (!(std::fabs(got - want) <= tol)) {
        std::printf("FAIL: %s (got %.17g, want %.17g, |diff|=%.3g)\n",
                    what, got, want, std::fabs(got - want));
        ++g_failures;
    }
}

void expect(const std::vector<double>& got, const std::vector<double>& want,
            const char* what) {
    if (got.size() != want.size()) {
        std::printf("FAIL: %s (size %zu != %zu)\n", what, got.size(),
                    want.size());
        ++g_failures;
        return;
    }
    for (size_t i = 0; i < want.size(); ++i) {
        char label[128];
        std::snprintf(label, sizeof(label), "%s[%zu]", what, i);
        check_eq(got[i], want[i], label);
    }
}

int64_t out_dim(int64_t in, int64_t k, int64_t s, int64_t p) {
    return (in + 2 * p - k) / s + 1;
}

// Run the internal kernel. The grad_input buffer is pre-filled with
// garbage on every call, so "the kernel zero-initializes its own output"
// is exercised everywhere, not only in the dedicated case.
std::vector<double> run(const std::vector<double>& grad_output,
                        const std::vector<double>& winners,
                        int64_t N, int64_t C, int64_t H, int64_t W,
                        int64_t oh, int64_t ow) {
    std::vector<double> grad_input(static_cast<size_t>(N * C * H * W), 7777.5);
    tf::maxpool2d_backward_contiguous(
        grad_output.data(), winners.data(), grad_input.data(),
        N, C, H, W, oh, ow);
    return grad_input;
}

// Run the exported checked wrapper over plain storage structs and report
// the resulting status code (TF_OK on success).
int run_checked(const std::vector<double>& grad_output,
                const std::vector<double>& winners,
                std::vector<double>& grad_input,
                int64_t N, int64_t C, int64_t H, int64_t W,
                int64_t oh, int64_t ow) {
    tf::Storage go{const_cast<double*>(grad_output.data()),
                   static_cast<int64_t>(grad_output.size())};
    tf::Storage wn{const_cast<double*>(winners.data()),
                   static_cast<int64_t>(winners.size())};
    tf::Storage gi{grad_input.data(),
                   static_cast<int64_t>(grad_input.size())};
    tf_core_maxpool2d_backward(&go, 0, &wn, 0, &gi, N, C, H, W, oh, ow);
    return tf::last_error_code();
}

// -- individual cases -------------------------------------------------

// 1. One window, one winner: the whole upstream lands on that element.
void test_simple_scatter() {
    std::vector<double> g = {2.5};
    std::vector<double> winners = {3};  // (1,1) of a 2x2 plane
    auto grad = run(g, winners, 1, 1, 2, 2, 1, 1);
    expect(grad, {0, 0, 0, 2.5}, "simple_scatter");
}

// 2. Multiple non-overlapping windows over a 4x4 plane.
void test_multiple_windows() {
    std::vector<double> g = {1, 2, 3, 4};
    std::vector<double> winners = {5, 7, 13, 15};  // the 2x2-pool winners
    auto grad = run(g, winners, 1, 1, 4, 4, 2, 2);
    expect(grad, {0, 0, 0, 0,
                  0, 1, 0, 2,
                  0, 0, 0, 0,
                  0, 3, 0, 4}, "multiple_windows");
}

// 3. Overlapping windows (stride < kernel) accumulate into one element.
void test_overlapping_accumulation() {
    // Four 2x2 windows over a 3x3 plane all selected the centre (offset 4).
    std::vector<double> g = {1, 2, 3, 4};
    std::vector<double> winners = {4, 4, 4, 4};
    auto grad = run(g, winners, 1, 1, 3, 3, 2, 2);
    expect(grad, {0, 0, 0,
                  0, 10, 0,   // 1 + 2 + 3 + 4
                  0, 0, 0}, "overlapping_accumulation");
}

// 4. Channels are independent; the same offset in another channel is a
// different element.
void test_multiple_channels() {
    std::vector<double> g = {1.5,   -2.5};   // N=1, C=2, 1x1 output each
    std::vector<double> winners = {3,   3};
    auto grad = run(g, winners, 1, 2, 2, 2, 1, 1);
    expect(grad, {0, 0, 0, 1.5,   0, 0, 0, -2.5}, "multiple_channels");
}

// 5. Batch greater than one — each sample scatters into its own plane.
void test_batch_greater_than_one() {
    std::vector<double> g = {3.0,   4.0};
    std::vector<double> winners = {0,   2};
    auto grad = run(g, winners, 2, 1, 2, 2, 1, 1);
    expect(grad, {3, 0, 0, 0,   0, 0, 4, 0}, "batch_greater_than_one");
}

// 6. Rectangular input and rectangular output.
void test_rectangular() {
    // 2x3 input, 1x2 output (2x2 kernel, stride 1).
    std::vector<double> g = {5.0, 6.0};
    std::vector<double> winners = {4, 5};
    auto grad = run(g, winners, 1, 1, 2, 3, 1, 2);
    expect(grad, {0, 0, 0,   0, 5, 6}, "rectangular");
}

// 7. Stride is expressed entirely through the saved winners: backward
// takes no stride argument and still routes to the strided positions.
void test_stride_through_winners() {
    // 5x5 plane pooled 2x2 with stride 2 -> winners 6, 8, 16, 18.
    std::vector<double> g = {1, 2, 3, 4};
    std::vector<double> winners = {6, 8, 16, 18};
    auto grad = run(g, winners, 1, 1, 5, 5, 2, 2);
    std::vector<double> want(25, 0.0);
    want[6] = 1; want[8] = 2; want[16] = 3; want[18] = 4;
    expect(grad, want, "stride_through_winners");
}

// 8. A -1 winner (padding won) drops that window's gradient entirely.
void test_padding_sentinel_drops_gradient() {
    std::vector<double> g = {1, 2, 3, 4};
    std::vector<double> winners = {-1, 1, -1, 3};
    auto grad = run(g, winners, 1, 1, 2, 2, 2, 2);
    expect(grad, {0, 2, 0, 4}, "padding_sentinel_drops_gradient");
}

// 9. A completely padded output (every winner -1) contributes nothing:
// grad_input is all zeros, still fully written.
void test_all_padding_winners() {
    std::vector<double> g = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    std::vector<double> winners(9, -1.0);
    auto grad = run(g, winners, 1, 1, 1, 1, 3, 3);
    expect(grad, {0}, "all_padding_winners");
}

// 10. Repeated winner offsets across several windows sum, and untouched
// positions stay exactly zero.
void test_repeated_winner_offsets() {
    std::vector<double> g = {1, 1, 1, 1, 1, 1};
    std::vector<double> winners = {2, 2, 2, 5, 5, 2};
    auto grad = run(g, winners, 1, 1, 2, 3, 2, 3);
    expect(grad, {0, 0, 4,   0, 0, 2}, "repeated_winner_offsets");
}

// 11. Negative upstream gradients propagate with their sign.
void test_negative_gradients() {
    std::vector<double> g = {-1.5, -2.5};
    std::vector<double> winners = {0, 0};
    auto grad = run(g, winners, 1, 1, 2, 2, 1, 2);
    expect(grad, {-4.0, 0, 0, 0}, "negative_gradients");
}

// 12. Fractional upstream gradients accumulate exactly.
void test_fractional_gradients() {
    std::vector<double> g = {0.25, 0.5, 1.25};
    std::vector<double> winners = {1, 1, 3};
    auto grad = run(g, winners, 1, 1, 2, 2, 1, 3);
    expect(grad, {0, 0.75, 0, 1.25}, "fractional_gradients");
}

// 13. grad_input is fully zero-initialized from garbage before scattering.
void test_zero_initialization_from_garbage() {
    std::vector<double> g = {9.0};
    std::vector<double> winners = {2};
    std::vector<double> grad_input(4, 12345.0);
    tf::maxpool2d_backward_contiguous(
        g.data(), winners.data(), grad_input.data(), 1, 1, 2, 2, 1, 1);
    expect(grad_input, {0, 0, 9.0, 0}, "zero_initialization_from_garbage");
}

// 14. grad_output is never modified.
void test_grad_output_unmodified() {
    std::vector<double> g = {1, 2, 3, 4};
    const std::vector<double> before = g;
    std::vector<double> winners = {5, 7, 13, 15};
    (void)run(g, winners, 1, 1, 4, 4, 2, 2);
    check(g == before, "grad_output unmodified");
}

// 15. The winner buffer is never modified (backward only reads it).
void test_winners_unmodified() {
    std::vector<double> g = {1, 2, 3, 4};
    std::vector<double> winners = {5, -1, 13, 15};
    const std::vector<double> before = winners;
    (void)run(g, winners, 1, 1, 4, 4, 2, 2);
    check(winners == before, "winners unmodified");
}

// 16. Determinism: repeated execution is bit-identical.
void test_determinism() {
    std::vector<double> g = {-1.5, 2.25, 0.125, -3.0, 7.5, 0.75};
    std::vector<double> winners = {4, 4, 0, 8, 4, 7};
    auto a = run(g, winners, 1, 1, 3, 3, 2, 3);
    auto b = run(g, winners, 1, 1, 3, 3, 2, 3);
    expect(a, b, "determinism");
}

// 17. Forward-to-backward integration: the winners the forward kernel
// saved route the gradient to exactly the cells that won, including the
// padding sentinel — no window geometry is recomputed anywhere.
void test_forward_winner_integration() {
    // 3x3 input, 2x2 kernel, stride 1, padding 1 -> 4x4 output with padded
    // border windows.
    const int64_t H = 3, W = 3, kh = 2, kw = 2, sh = 1, sw = 1, ph = 1, pw = 1;
    const int64_t oh = out_dim(H, kh, sh, ph);
    const int64_t ow = out_dim(W, kw, sw, pw);
    std::vector<double> in = {1, 2, 3, 4, 9, 5, 6, 7, 8};
    std::vector<double> out(static_cast<size_t>(oh * ow), 0.0);
    std::vector<double> winners(static_cast<size_t>(oh * ow), 0.0);
    tf::maxpool2d_forward_contiguous(
        in.data(), out.data(), winners.data(),
        1, 1, H, W, kh, kw, sh, sw, ph, pw, oh, ow);
    // Feed a unit upstream: every window donates 1.0 to its winner, so the
    // gradient counts how many windows each cell won.
    std::vector<double> g(static_cast<size_t>(oh * ow), 1.0);
    auto grad = run(g, winners, 1, 1, H, W, oh, ow);
    // Independent oracle: tally the saved winners directly.
    std::vector<double> want(static_cast<size_t>(H * W), 0.0);
    double dropped = 0.0;
    for (size_t i = 0; i < winners.size(); ++i) {
        if (winners[i] < 0) {
            dropped += 1.0;
        } else {
            want[static_cast<size_t>(winners[i])] += 1.0;
        }
    }
    expect(grad, want, "forward_winner_integration");
    // Every window here contains a real cell that beats the padded -inf,
    // so nothing is dropped and the whole upstream reaches the input.
    check_eq(dropped, 0.0, "no sentinel for finite padded data");
    double total = 0.0;
    for (double value : grad) {
        total += value;
    }
    check_eq(total, static_cast<double>(oh * ow), "gradient conservation");

    // Second geometry: a 1x1 input with 1x1 kernel and padding 1 gives a
    // 3x3 output whose eight border windows are ENTIRELY padding, so the
    // forward saves -1 for them and backward must drop exactly those.
    std::vector<double> tiny_in = {4.0};
    std::vector<double> tiny_out(9, 0.0);
    std::vector<double> tiny_winners(9, 0.0);
    tf::maxpool2d_forward_contiguous(
        tiny_in.data(), tiny_out.data(), tiny_winners.data(),
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 3, 3);
    double sentinels = 0.0;
    for (double winner : tiny_winners) {
        if (winner == -1.0) {
            sentinels += 1.0;
        }
    }
    check_eq(sentinels, 8.0, "eight completely padded windows saved -1");
    std::vector<double> tiny_g(9, 1.0);
    auto tiny_grad = run(tiny_g, tiny_winners, 1, 1, 1, 1, 3, 3);
    // Only the single real window contributes; the eight padded ones drop.
    expect(tiny_grad, {1.0}, "forward_winner_integration_padded");
}

// 18. Stable-framework parity: grad_input generated once from
// tensorforge.nn.MaxPool2d (N=2, C=2, H=5, W=4, 3x2 kernel, stride (2,1),
// padding (1,0)) for the embedded upstream gradient.
void test_stable_parity() {
    const std::vector<double> g = {
        -0.235, -1.908, -4.007, 5.299, -0.039, 1.196, 0.452, 1.984, -2.949,
        -2.907, -2.04, -0.926, -0.202, -0.062, -2.027, -2.809, 2.181, 1.465,
        2.951, 0.139, 1.397, 3.003, -0.512, -0.447, 1.902, 2.356, 0.127,
        -0.626, 1.173, 0.369, 1.369, -1.497, -0.959, 0.702, 1.083, -3.401};
    const std::vector<double> winners = {
        4.0, 2.0, 2.0, 9.0, 9.0, 11.0, 17.0, 17.0, 19.0, 5.0, 5.0, 7.0, 8.0,
        5.0, 7.0, 17.0, 17.0, 15.0, 5.0, 5.0, 7.0, 5.0, 14.0, 14.0, 12.0,
        14.0, 14.0, 0.0, 1.0, 3.0, 5.0, 5.0, 7.0, 16.0, 13.0, 19.0};
    const std::vector<double> want = {
        0.0, 0.0, -5.914999999999999, 0.0, -0.235, 0.0, 0.0, 0.0, 0.0,
        5.260000000000001, 0.0, 1.196, 0.0, 0.0, 0.0, 0.0, 0.0, 2.436, 0.0,
        -2.949, 0.0, 0.0, 0.0, 0.0, 0.0, -5.009, 0.0, -2.9530000000000003,
        -0.202, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.465, 0.0,
        -0.6280000000000001, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 6.093, 0.0,
        1.397, 0.0, 0.0, 0.0, 0.0, 1.902, 0.0, 1.5239999999999998, 0.0, 0.0,
        0.0, 0.0, 0.0, -0.626, 1.173, 0.0, 0.369, 0.0, -0.1280000000000001,
        0.0, -0.959, 0.0, 0.0, 0.0, 0.0, 0.0, 1.083, 0.0, 0.0, 0.702, 0.0,
        0.0, -3.401};
    auto grad = run(g, winners, 2, 2, 5, 4, 3, 3);
    if (grad.size() != want.size()) {
        check(false, "stable_parity size");
        return;
    }
    for (size_t i = 0; i < want.size(); ++i) {
        char label[64];
        std::snprintf(label, sizeof(label), "stable_parity[%zu]", i);
        check_close(grad[i], want[i], 1e-12, label);
    }
}

// 19. Malformed winners are rejected by the CHECKED WRAPPER (never by the
// inner kernel, and never silently rounded), and grad_input is left
// untouched when they are.
void test_checked_wrapper_rejects_invalid_winners() {
    const std::vector<double> g = {1.0, 2.0, 3.0, 4.0};
    // A 2x2 plane: valid winners are -1 and 0..3.
    const double bad_values[] = {
        0.5,          // fractional
        kNaN,         // NaN
        kPosInf,      // +inf
        kNegInf,      // -inf
        -2.0,         // below the -1 sentinel
        4.0,          // == H*W, one past the end
        99.0,         // far out of range
    };
    for (double bad : bad_values) {
        std::vector<double> winners = {0, 1, 2, 3};
        winners[2] = bad;
        std::vector<double> grad_input(4, 555.0);
        const int status = run_checked(g, winners, grad_input, 1, 1, 2, 2, 2, 2);
        char label[96];
        std::snprintf(label, sizeof(label),
                      "invalid winner %.17g rejected", bad);
        check(status == TF_ERROR_INVALID, label);
        std::snprintf(label, sizeof(label),
                      "grad_input untouched for winner %.17g", bad);
        bool untouched = true;
        for (double value : grad_input) {
            untouched = untouched && (value == 555.0);
        }
        check(untouched, label);
    }
    tf::clear_error();
    // The valid domain is accepted: the sentinel and both boundaries.
    std::vector<double> winners = {-1.0, 0.0, 3.0, 2.0};
    std::vector<double> grad_input(4, 555.0);
    const int status = run_checked(g, winners, grad_input, 1, 1, 2, 2, 2, 2);
    check(status == TF_OK, "valid winners accepted");
    expect(grad_input, {2.0, 0.0, 4.0, 3.0}, "valid_winner_scatter");
    tf::clear_error();
}

// 20. Boundary: winner offset 0 (the plane's first element).
void test_boundary_winner_zero() {
    std::vector<double> g = {6.0};
    std::vector<double> winners = {0};
    auto grad = run(g, winners, 1, 1, 3, 3, 1, 1);
    std::vector<double> want(9, 0.0);
    want[0] = 6.0;
    expect(grad, want, "boundary_winner_zero");
}

// 21. Boundary: winner offset H*W-1 (the plane's last element), through
// the checked wrapper so the range check is exercised at its edge.
void test_boundary_winner_last() {
    const int64_t H = 4, W = 5;
    std::vector<double> g = {2.0};
    std::vector<double> winners = {static_cast<double>(H * W - 1)};
    std::vector<double> grad_input(static_cast<size_t>(H * W), 999.0);
    const int status = run_checked(g, winners, grad_input, 1, 1, H, W, 1, 1);
    check(status == TF_OK, "boundary winner H*W-1 accepted");
    std::vector<double> want(static_cast<size_t>(H * W), 0.0);
    want[static_cast<size_t>(H * W - 1)] = 2.0;
    expect(grad_input, want, "boundary_winner_last");
    tf::clear_error();
}

}  // namespace

int main() {
    test_simple_scatter();
    test_multiple_windows();
    test_overlapping_accumulation();
    test_multiple_channels();
    test_batch_greater_than_one();
    test_rectangular();
    test_stride_through_winners();
    test_padding_sentinel_drops_gradient();
    test_all_padding_winners();
    test_repeated_winner_offsets();
    test_negative_gradients();
    test_fractional_gradients();
    test_zero_initialization_from_garbage();
    test_grad_output_unmodified();
    test_winners_unmodified();
    test_determinism();
    test_forward_winner_integration();
    test_stable_parity();
    test_checked_wrapper_rejects_invalid_winners();
    test_boundary_winner_zero();
    test_boundary_winner_last();

    if (g_failures == 0) {
        std::printf("OK: all maxpool2d_backward tests passed\n");
        return 0;
    }
    std::printf("FAILED: %d maxpool2d_backward check(s)\n", g_failures);
    return 1;
}
