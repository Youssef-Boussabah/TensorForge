// Dependency-free C++ test for the internal MaxPool2d forward kernel
// (Phase D, milestone D8). No GoogleTest/Catch2 — a plain executable that
// prints failures and returns a nonzero exit code if any check fails, so
// CTest reports pass/fail.
//
// This binary compiles cpp/src/pooling.cpp directly (see the CMake
// TF_BUILD_TESTS target): the kernel is an internal, hidden-visibility
// symbol, so it is not reachable through the shared library's public ABI.
// The test never touches Python at runtime; the stable-framework parity
// case embeds values generated once from tensorforge.nn.MaxPool2d (with
// its window-local argmax winners converted to the native flat-plane-offset
// representation) and compares them exactly — max pooling *selects* a
// value rather than summing, so there is no accumulation order to make
// parity approximate.
//
// Every case checks BOTH the pooled output and the saved winner buffer.

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <vector>

#include "tf_pooling_internal.h"

namespace {

int g_failures = 0;

const double kNegInf = -std::numeric_limits<double>::infinity();

void check(bool condition, const char* what) {
    if (!condition) {
        std::printf("FAIL: %s\n", what);
        ++g_failures;
    }
}

// Strict equality — pooling selects an input value verbatim and writes
// integral winner indices, so every expectation here is exact.
void check_eq(double got, double want, const char* what) {
    if (got != want) {
        std::printf("FAIL: %s (got %.17g, want %.17g)\n", what, got, want);
        ++g_failures;
    }
}

int64_t out_dim(int64_t in, int64_t k, int64_t s, int64_t p) {
    return (in + 2 * p - k) / s + 1;
}

// One pooled result: the output values and the parallel winner buffer.
struct Pooled {
    std::vector<double> output;
    std::vector<double> winners;
};

// Run the kernel; sizes both buffers from the floor formula. The Core
// wrapper owns these allocations in the real stack — here the test does.
// Both buffers are pre-filled with garbage so "fully defined output" is
// exercised on every call, not just in the dedicated case.
Pooled run(const std::vector<double>& input,
           int64_t N, int64_t C, int64_t H, int64_t W,
           int64_t kh, int64_t kw, int64_t sh, int64_t sw,
           int64_t ph, int64_t pw) {
    const int64_t oh = out_dim(H, kh, sh, ph);
    const int64_t ow = out_dim(W, kw, sw, pw);
    const size_t count = static_cast<size_t>(N * C * oh * ow);
    Pooled result{std::vector<double>(count, 1234.5),
                  std::vector<double>(count, -98765.25)};
    tf::maxpool2d_forward_contiguous(
        input.data(), result.output.data(), result.winners.data(),
        N, C, H, W, kh, kw, sh, sw, ph, pw, oh, ow);
    return result;
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

void expect_pooled(const Pooled& got, const std::vector<double>& values,
                   const std::vector<double>& winners, const char* what) {
    char label[160];
    std::snprintf(label, sizeof(label), "%s/output", what);
    expect(got.output, values, label);
    std::snprintf(label, sizeof(label), "%s/winners", what);
    expect(got.winners, winners, label);
}

// -- individual cases -------------------------------------------------

// 1. The simplest possible pooling: one 2x2 window over a 2x2 input.
void test_simple_2x2() {
    std::vector<double> in = {1, 2, 3, 4};  // 1x1x2x2
    auto got = run(in, 1, 1, 2, 2, 2, 2, 2, 2, 0, 0);
    expect_pooled(got, {4}, {3}, "simple_2x2");
}

// 2. Multiple non-overlapping windows over a 4x4 plane.
void test_multiple_windows() {
    std::vector<double> in = {
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
    auto got = run(in, 1, 1, 4, 4, 2, 2, 2, 2, 0, 0);
    expect_pooled(got, {6, 8, 14, 16}, {5, 7, 13, 15}, "multiple_windows");
}

// 3. Channels are pooled independently; the winner is a per-plane offset,
// so the same offset may repeat across channels.
void test_multiple_channels() {
    std::vector<double> in = {1, 2, 3, 4,   8, 7, 6, 5};  // N=1, C=2, 2x2
    auto got = run(in, 1, 2, 2, 2, 2, 2, 2, 2, 0, 0);
    expect_pooled(got, {4, 8}, {3, 0}, "multiple_channels");
}

// 4. Batch greater than one — each sample is independent.
void test_batch_greater_than_one() {
    std::vector<double> in = {1, 2, 3, 4,   5, 8, 7, 6};  // N=2, C=1, 2x2
    auto got = run(in, 2, 1, 2, 2, 2, 2, 2, 2, 0, 0);
    expect_pooled(got, {4, 8}, {3, 1}, "batch_greater_than_one");
}

// 5. Rectangular input (H != W).
void test_rectangular_input() {
    std::vector<double> in = {1, 2, 3, 4, 5, 6};  // 1x1x2x3
    auto got = run(in, 1, 1, 2, 3, 2, 2, 1, 1, 0, 0);
    // out_h = 1, out_w = 2; winners are offsets into the 2x3 plane.
    expect_pooled(got, {5, 6}, {4, 5}, "rectangular_input");
}

// 6. Rectangular kernel (kh != kw): a 1x3 window pools whole rows.
void test_rectangular_kernel() {
    std::vector<double> in = {1, 2, 3, 4, 5, 6, 7, 8, 9};  // 1x1x3x3
    auto got = run(in, 1, 1, 3, 3, 1, 3, 1, 1, 0, 0);
    expect_pooled(got, {3, 6, 9}, {2, 5, 8}, "rectangular_kernel");
}

// 7. Stride greater than one, with an input the windows do not tile
// exactly (the trailing row/column is dropped by the floor formula).
void test_stride_greater_than_one() {
    std::vector<double> in(25);
    for (int i = 0; i < 25; ++i) {
        in[static_cast<size_t>(i)] = i + 1;  // 1..25 over a 5x5 plane
    }
    auto got = run(in, 1, 1, 5, 5, 2, 2, 2, 2, 0, 0);
    expect_pooled(got, {7, 9, 17, 19}, {6, 8, 16, 18},
                  "stride_greater_than_one");
}

// 8. Separate height/width stride.
void test_asymmetric_stride() {
    std::vector<double> in = {
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
    auto got = run(in, 1, 1, 4, 4, 2, 2, 2, 1, 0, 0);
    expect_pooled(got, {6, 7, 8, 14, 15, 16}, {5, 6, 7, 13, 14, 15},
                  "asymmetric_stride");
}

// 9. Symmetric padding on both axes: padded cells lose to every real
// value, so each window reports its single in-bounds element.
void test_symmetric_padding() {
    std::vector<double> in = {1, 2, 3, 4};  // 1x1x2x2
    auto got = run(in, 1, 1, 2, 2, 2, 2, 2, 2, 1, 1);
    expect_pooled(got, {1, 2, 3, 4}, {0, 1, 2, 3}, "symmetric_padding");
}

// 10. Separate height/width padding (padded rows only).
void test_asymmetric_padding() {
    std::vector<double> in = {1, 2, 3, 4, 5, 6};  // 1x1x3x2
    auto got = run(in, 1, 1, 3, 2, 2, 2, 2, 2, 1, 0);
    expect_pooled(got, {2, 6}, {1, 5}, "asymmetric_padding");
}

// 11. Combined stride and padding.
void test_stride_and_padding() {
    std::vector<double> in = {1, 2, 3, 4, 5, 6, 7, 8, 9};  // 1x1x3x3
    auto got = run(in, 1, 1, 3, 3, 2, 2, 2, 2, 1, 1);
    expect_pooled(got, {1, 3, 7, 9}, {0, 2, 6, 8}, "stride_and_padding");
}

// 12. All-negative values: the maximum is the least negative one (a
// zero-initialized accumulator would wrongly report 0).
void test_negative_values() {
    std::vector<double> in = {-4, -1, -3, -2};
    auto got = run(in, 1, 1, 2, 2, 2, 2, 2, 2, 0, 0);
    expect_pooled(got, {-1}, {1}, "negative_values");
}

// 13. Fractional values pass through bit-exactly.
void test_fractional_values() {
    std::vector<double> in = {0.5, -0.25, 2.25, 1.75};  // 1x1x1x4
    auto got = run(in, 1, 1, 1, 4, 1, 2, 1, 2, 0, 0);
    expect_pooled(got, {0.5, 2.25}, {0, 2}, "fractional_values");
}

// 14. A unique interior maximum is found (not a corner artifact).
void test_unique_maximum() {
    std::vector<double> in = {1, 2, 3, 4, 9, 5, 6, 7, 8};
    auto got = run(in, 1, 1, 3, 3, 3, 3, 1, 1, 0, 0);
    expect_pooled(got, {9}, {4}, "unique_maximum");
}

// 15. Ties select the FIRST occurrence in row-major window order.
void test_tie_selects_first() {
    std::vector<double> all_equal = {5, 5, 5, 5};
    expect_pooled(run(all_equal, 1, 1, 2, 2, 2, 2, 2, 2, 0, 0),
                  {5}, {0}, "tie_all_equal");
    // A later equal value must not displace the earlier winner.
    std::vector<double> partial = {1, 5, 5, 2};
    expect_pooled(run(partial, 1, 1, 2, 2, 2, 2, 2, 2, 0, 0),
                  {5}, {1}, "tie_first_of_two_maxima");
}

// 16. Padding versus a real -inf: whichever comes first in row-major
// window order wins. Every value here is -inf, so the tie rule decides.
void test_padding_versus_real_negative_infinity() {
    std::vector<double> in(4, kNegInf);  // 1x1x2x2, all -inf
    auto got = run(in, 1, 1, 2, 2, 2, 2, 2, 2, 1, 1);
    // Windows 0-2 begin on a padded cell (-1 sentinel); window 3 begins on
    // the real (1,1) cell, whose flat offset is 3.
    expect_pooled(got, {kNegInf, kNegInf, kNegInf, kNegInf},
                  {-1, -1, -1, 3}, "padding_vs_real_neg_inf");
}

// 17. All valid values -inf with NO padding: the first real cell wins, so
// the gradient still flows to a genuine input element.
void test_all_valid_negative_infinity() {
    std::vector<double> in(4, kNegInf);
    auto got = run(in, 1, 1, 2, 2, 2, 2, 2, 2, 0, 0);
    expect_pooled(got, {kNegInf}, {0}, "all_valid_neg_inf");
}

// 18. Completely padded windows are allowed: output -inf, winner -1.
void test_completely_padded_windows() {
    std::vector<double> in = {7.0};  // 1x1x1x1
    auto got = run(in, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1);  // 3x3 output
    expect_pooled(
        got,
        {kNegInf, kNegInf, kNegInf, kNegInf, 7.0, kNegInf, kNegInf, kNegInf,
         kNegInf},
        {-1, -1, -1, -1, 0, -1, -1, -1, -1},
        "completely_padded_windows");
}

// 19. Overlapping windows may independently save the same input offset.
void test_repeated_winner_offsets() {
    std::vector<double> in = {1, 2, 3, 4, 9, 5, 6, 7, 8};  // max at (1,1)
    auto got = run(in, 1, 1, 3, 3, 2, 2, 1, 1, 0, 0);
    expect_pooled(got, {9, 9, 9, 9}, {4, 4, 4, 4}, "repeated_winner_offsets");
}

// 20. The kernel never mutates its input.
void test_input_unmodified() {
    std::vector<double> in = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    const std::vector<double> before = in;
    (void)run(in, 1, 1, 3, 3, 2, 2, 1, 1, 0, 0);
    check(in == before, "input unmodified");
}

// 21. Both buffers are fully defined from garbage — no element of the
// output or the winner buffer survives the call uninitialized.
void test_full_initialization_from_garbage() {
    std::vector<double> in = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    const int64_t oh = 2, ow = 2;
    std::vector<double> output(static_cast<size_t>(oh * ow), 42.0);
    std::vector<double> winners(static_cast<size_t>(oh * ow), 42.0);
    tf::maxpool2d_forward_contiguous(
        in.data(), output.data(), winners.data(),
        1, 1, 3, 3, 2, 2, 1, 1, 0, 0, oh, ow);
    for (size_t i = 0; i < output.size(); ++i) {
        char label[96];
        std::snprintf(label, sizeof(label), "garbage_output[%zu] overwritten",
                      i);
        check(output[i] != 42.0, label);
        std::snprintf(label, sizeof(label), "garbage_winner[%zu] overwritten",
                      i);
        check(winners[i] != 42.0, label);
    }
    expect(output, {5, 6, 8, 9}, "garbage_output_values");
    expect(winners, {4, 5, 7, 8}, "garbage_winner_values");
}

// 22. Determinism: repeated execution is bit-identical in both buffers.
void test_determinism() {
    std::vector<double> in = {
        -1.5, 2.5, 0.5, -3.0, 7.25, -0.125, 4.0, 9.5, 6.0};
    auto a = run(in, 1, 1, 3, 3, 2, 2, 1, 1, 1, 1);
    auto b = run(in, 1, 1, 3, 3, 2, 2, 1, 1, 1, 1);
    expect(a.output, b.output, "determinism_output");
    expect(a.winners, b.winners, "determinism_winners");
}

// 23. Stable-framework parity (supported, non-NaN data): values generated
// once from tensorforge.nn.MaxPool2d (N=2, C=3, H=5, W=4, 3x2 kernel,
// stride (2,1), padding (1,0)), with its window-local argmax winners
// converted to flat plane offsets. Compared exactly.
void test_stable_parity() {
    const std::vector<double> in = {
        -7.161, 0.04, 1.787, 2.584, 1.974, -4.589, 6.368, -2.633, 0.45,
        0.079, -3.617, -2.232, 2.378, 2.983, -2.347, 4.369, 0.479, 5.557,
        -1.85, -2.181, 1.487, -1.757, -0.194, 2.654, -0.638, 6.496, -2.406,
        -3.529, -0.103, 1.989, 6.959, -3.731, 2.362, 4.919, -0.45, 5.728,
        3.759, 3.857, -2.909, -7.206, 5.765, -3.263, -0.082, -0.529, -5.834,
        -4.7, 0.854, 3.247, -10.013, 4.887, 5.586, 2.268, 0.133, 0.694,
        -11.06, -1.55, -1.927, -4.186, -2.474, -5.856, 2.15, 1.999, -1.218,
        0.729, -3.523, 3.066, -4.149, -2.401, 4.244, -1.499, 2.735, -7.469,
        -2.519, 6.285, 5.31, 3.861, -3.161, 0.927, -1.047, -2.892, -3.794,
        2.812, -3.78, -0.147, 3.843, 5.518, -1.208, -3.334, -3.589, -0.854,
        -7.801, -4.321, -3.03, -2.989, 1.357, -0.83, -1.075, -8.088, 0.46,
        5.643, -2.928, -11.58, -7.434, -1.214, -0.685, -2.987, -2.917,
        3.393, 6.575, -0.071, 0.726, -1.277, 1.297, -4.162, -5.474, 1.414,
        -1.565, -1.664, 0.844, -5.725};
    const std::vector<double> want_output = {
        1.974, 6.368, 6.368, 2.983, 6.368, 6.368, 5.557, 5.557, 4.369,
        6.496, 6.496, 2.654, 6.496, 6.959, 6.959, 4.919, 4.919, 5.728,
        5.765, 0.854, 3.247, 4.887, 5.586, 5.586, 0.694, 0.694, -1.55,
        3.066, 3.066, 0.729, 6.285, 6.285, 5.31, 6.285, 6.285, 5.31, 5.518,
        5.518, -0.147, 5.518, 5.518, 1.357, -1.075, 1.357, 5.643, -0.685,
        -2.917, 3.393, 6.575, 0.726, 3.393, 1.297, 0.844, 1.414};
    const std::vector<double> want_winners = {
        4.0, 6.0, 6.0, 13.0, 6.0, 6.0, 17.0, 17.0, 15.0, 5.0, 5.0, 3.0, 5.0,
        10.0, 10.0, 13.0, 13.0, 15.0, 0.0, 6.0, 7.0, 9.0, 10.0, 10.0, 13.0,
        13.0, 15.0, 5.0, 5.0, 3.0, 13.0, 13.0, 14.0, 13.0, 13.0, 14.0, 5.0,
        5.0, 3.0, 5.0, 5.0, 14.0, 16.0, 14.0, 19.0, 4.0, 6.0, 7.0, 8.0,
        10.0, 7.0, 12.0, 18.0, 15.0};
    auto got = run(in, 2, 3, 5, 4, 3, 2, 2, 1, 1, 0);
    expect_pooled(got, want_output, want_winners, "stable_parity");
}

// 24. Winner exactness at the largest flat offset this suite exercises: a
// 200x200 plane pooled by one 200x200 window. The maximum sits in the last
// cell, so the winner is 39999 — written as an exact integral float64.
void test_winner_exactness_large_offset() {
    const int64_t H = 200, W = 200;
    std::vector<double> in(static_cast<size_t>(H * W));
    for (size_t i = 0; i < in.size(); ++i) {
        in[i] = static_cast<double>(i);  // strictly increasing
    }
    auto got = run(in, 1, 1, H, W, H, W, 1, 1, 0, 0);
    check(got.output.size() == 1, "large_offset single output");
    expect_pooled(got, {static_cast<double>(H * W - 1)},
                  {static_cast<double>(H * W - 1)}, "winner_exactness");
    const double winner = got.winners[0];
    check(winner == std::floor(winner), "winner is integral");
    check(static_cast<double>(static_cast<int64_t>(winner)) == winner,
          "winner round-trips through int64 exactly");
}

// 25. NaN behavior (documented, deliberately NOT claimed as stable
// parity): a NaN never wins merely by appearing first, and a window with
// no non-NaN candidate falls back deterministically to its first
// candidate, so the output value and the winner still agree.
void test_nan_documented_behavior() {
    const double nan_value = std::numeric_limits<double>::quiet_NaN();
    // A leading NaN does not become the winner; the first non-NaN seeds
    // the scan and the strict-> rule picks the real maximum.
    std::vector<double> leading_nan = {nan_value, 1.0, 3.0, 2.0};
    auto got = run(leading_nan, 1, 1, 1, 4, 1, 4, 1, 1, 0, 0);
    expect_pooled(got, {3.0}, {2.0}, "nan_never_wins");
    // An all-NaN window (no padding, so no -inf candidate exists) falls
    // back to the FIRST candidate: value NaN, winner offset 0.
    std::vector<double> all_nan = {nan_value, nan_value};
    auto fallback = run(all_nan, 1, 1, 1, 2, 1, 2, 1, 1, 0, 0);
    check(fallback.output.size() == 1 && std::isnan(fallback.output[0]),
          "all-NaN window outputs NaN");
    check(fallback.winners.size() == 1 && fallback.winners[0] == 0.0,
          "all-NaN window falls back to the first candidate's winner");
}

}  // namespace

int main() {
    test_simple_2x2();
    test_multiple_windows();
    test_multiple_channels();
    test_batch_greater_than_one();
    test_rectangular_input();
    test_rectangular_kernel();
    test_stride_greater_than_one();
    test_asymmetric_stride();
    test_symmetric_padding();
    test_asymmetric_padding();
    test_stride_and_padding();
    test_negative_values();
    test_fractional_values();
    test_unique_maximum();
    test_tie_selects_first();
    test_padding_versus_real_negative_infinity();
    test_all_valid_negative_infinity();
    test_completely_padded_windows();
    test_repeated_winner_offsets();
    test_input_unmodified();
    test_full_initialization_from_garbage();
    test_determinism();
    test_stable_parity();
    test_winner_exactness_large_offset();
    test_nan_documented_behavior();

    if (g_failures == 0) {
        std::printf("OK: all maxpool2d_forward tests passed\n");
        return 0;
    }
    std::printf("FAILED: %d maxpool2d_forward check(s)\n", g_failures);
    return 1;
}
