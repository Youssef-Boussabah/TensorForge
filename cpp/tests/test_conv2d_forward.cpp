// Dependency-free C++ test for the internal Conv2d forward kernel
// (Phase D, milestone D2). No GoogleTest/Catch2 — a plain executable that
// prints failures and returns a nonzero exit code if any check fails, so
// CTest reports pass/fail.
//
// This binary compiles cpp/src/conv2d.cpp directly (see the CMake
// TF_BUILD_TESTS target): the kernel is an internal, hidden-visibility
// symbol, so it is not reachable through the shared library's public ABI.
// The test never touches Python at runtime; the stable-framework parity
// case embeds output values generated once from tensorforge.nn.Conv2d and
// is compared to a small tolerance (the summation order is deterministic
// but not bit-identical to NumPy's einsum).

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <vector>

#include "tf_conv2d_internal.h"

namespace {

int g_failures = 0;

void check(bool condition, const char* what) {
    if (!condition) {
        std::printf("FAIL: %s\n", what);
        ++g_failures;
    }
}

// Strict equality — the hand-computed cases use only exactly
// representable float64 values, so any drift is a real bug.
void check_eq(double got, double want, const char* what) {
    if (got != want) {
        std::printf("FAIL: %s (got %.17g, want %.17g)\n", what, got, want);
        ++g_failures;
    }
}

// Tolerance comparison for the stable-framework parity case.
void check_close(double got, double want, double tol, const char* what) {
    if (!(std::fabs(got - want) <= tol)) {
        std::printf("FAIL: %s (got %.17g, want %.17g, |diff|=%.3g)\n",
                    what, got, want, std::fabs(got - want));
        ++g_failures;
    }
}

int64_t out_dim(int64_t in, int64_t k, int64_t s, int64_t p) {
    return (in + 2 * p - k) / s + 1;
}

// Run the kernel; sizes the output from the floor formula and zeroes it
// first (the wrapper will own this in D3 — here the test owns the buffer).
std::vector<double> run(
    const std::vector<double>& input, const std::vector<double>& weight,
    const double* bias,
    int64_t N, int64_t C, int64_t H, int64_t W, int64_t O,
    int64_t kh, int64_t kw, int64_t sh, int64_t sw, int64_t ph, int64_t pw) {
    const int64_t oh = out_dim(H, kh, sh, ph);
    const int64_t ow = out_dim(W, kw, sw, pw);
    std::vector<double> output(static_cast<size_t>(N * O * oh * ow), 0.0);
    tf::conv2d_forward_contiguous(
        input.data(), weight.data(), bias, output.data(),
        N, C, H, W, O, kh, kw, sh, sw, ph, pw, oh, ow);
    return output;
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

// -- individual cases -------------------------------------------------

// 1. Single-channel simple kernel, no bias, stride 1, padding 0.
void test_single_channel_no_bias() {
    std::vector<double> in = {1, 2, 3, 4, 5, 6, 7, 8, 9};  // 1x1x3x3
    std::vector<double> w = {1, 0, 0, 1};                  // 1x1x2x2 (diag)
    auto out = run(in, w, nullptr, 1, 1, 3, 3, 1, 2, 2, 1, 1, 0, 0);
    // Cross-correlation with the identity-diagonal 2x2 kernel.
    expect(out, {6, 8, 12, 14}, "single_channel_no_bias");
}

// 2a. Bias is added once per output element.
void test_bias_added_once() {
    std::vector<double> in = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    std::vector<double> w = {1, 0, 0, 1};
    double bias[1] = {10.0};
    auto out = run(in, w, bias, 1, 1, 3, 3, 1, 2, 2, 1, 1, 0, 0);
    expect(out, {16, 18, 22, 24}, "bias_added_once");
}

// 2b. With an all-zero kernel every output equals the bias exactly —
// proves the bias is added exactly once (not zero times, not twice).
void test_bias_only_with_zero_kernel() {
    std::vector<double> in = {5, -2, 7, 0, 3, -9, 1, 4, 8};
    std::vector<double> w(4, 0.0);  // 1x1x2x2 all zeros
    double bias[1] = {3.5};
    auto out = run(in, w, bias, 1, 1, 3, 3, 1, 2, 2, 1, 1, 0, 0);
    expect(out, {3.5, 3.5, 3.5, 3.5}, "bias_only_zero_kernel");
}

// 3. Multiple input channels accumulate.
void test_multiple_input_channels() {
    // N=1, C=2, H=2, W=2; 1x1 kernel per channel: out = c0*1 + c1*2.
    std::vector<double> in = {1, 2, 3, 4,   10, 20, 30, 40};
    std::vector<double> w = {1, 2};  // O=1, C=2, 1x1
    auto out = run(in, w, nullptr, 1, 2, 2, 2, 1, 1, 1, 1, 1, 0, 0);
    expect(out, {21, 42, 63, 84}, "multiple_input_channels");
}

// 4. Multiple output channels have distinct kernels and biases.
void test_multiple_output_channels() {
    std::vector<double> in = {1, 2, 3, 4};  // N=1,C=1,2x2
    std::vector<double> w = {2, -1};        // O=2,C=1,1x1: ch0=2x, ch1=-1x
    double bias[2] = {0.5, -0.5};
    auto out = run(in, w, bias, 1, 1, 2, 2, 2, 1, 1, 1, 1, 0, 0);
    // ch0 = 2*in + 0.5 ; ch1 = -in - 0.5
    expect(out, {2.5, 4.5, 6.5, 8.5,   -1.5, -2.5, -3.5, -4.5},
           "multiple_output_channels");
}

// 5. Symmetric padding: out-of-bounds coordinates contribute zero.
void test_padding() {
    std::vector<double> in = {1, 2, 3, 4, 5, 6, 7, 8, 9};  // 3x3
    std::vector<double> w(9, 1.0);  // 3x3 all-ones
    auto out = run(in, w, nullptr, 1, 1, 3, 3, 1, 3, 3, 1, 1, 1, 1);
    expect(out, {12, 21, 16,  27, 45, 33,  24, 39, 28}, "padding");
}

// 6. Stride greater than one.
void test_stride() {
    std::vector<double> in = {
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};  // 4x4
    std::vector<double> w(4, 1.0);  // 2x2 all-ones
    auto out = run(in, w, nullptr, 1, 1, 4, 4, 1, 2, 2, 2, 2, 0, 0);
    expect(out, {14, 22, 46, 54}, "stride");
}

// 7. Rectangular input and rectangular kernel (separate h/w dims).
void test_rectangular() {
    std::vector<double> in = {1, 2, 3, 4, 5, 6};  // 1x1x2x3
    std::vector<double> w = {1, 1, 1, 1};          // 1x1x2x2
    auto out = run(in, w, nullptr, 1, 1, 2, 3, 1, 2, 2, 1, 1, 0, 0);
    expect(out, {12, 16}, "rectangular");  // out_h=1, out_w=2
}

// 8. Combined stride and padding.
void test_stride_and_padding() {
    std::vector<double> in = {1, 2, 3, 4, 5, 6, 7, 8, 9};  // 3x3
    std::vector<double> w(4, 1.0);  // 2x2 all-ones
    auto out = run(in, w, nullptr, 1, 1, 3, 3, 1, 2, 2, 2, 2, 1, 1);
    expect(out, {1, 5, 11, 28}, "stride_and_padding");  // 2x2 output
}

// 9. Batch size greater than one — each sample is independent.
void test_batch() {
    // N=2, C=1, 2x2 each; 1x1 kernel weight 2, no bias.
    std::vector<double> in = {1, 2, 3, 4,   5, 6, 7, 8};
    std::vector<double> w = {2};
    auto out = run(in, w, nullptr, 2, 1, 2, 2, 1, 1, 1, 1, 1, 0, 0);
    expect(out, {2, 4, 6, 8,   10, 12, 14, 16}, "batch");
}

// 10. Negative and fractional values.
void test_negative_and_fractional() {
    std::vector<double> in = {-1.5, 2.5, 0.5, -3.0};  // 2x2
    std::vector<double> w = {2.0};                    // 1x1
    double bias[1] = {-0.5};
    auto out = run(in, w, bias, 1, 1, 2, 2, 1, 1, 1, 1, 1, 0, 0);
    // 2*in - 0.5, all exactly representable.
    expect(out, {-3.5, 4.5, 0.5, -6.5}, "negative_and_fractional");
}

// 11. Null bias pointer must behave as "no bias" and equal a zero-bias run.
void test_null_bias_matches_zero_bias() {
    std::vector<double> in = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    std::vector<double> w = {1, 0, 0, 1};
    double zero_bias[1] = {0.0};
    auto with_null = run(in, w, nullptr, 1, 1, 3, 3, 1, 2, 2, 1, 1, 0, 0);
    auto with_zero = run(in, w, zero_bias, 1, 1, 3, 3, 1, 2, 2, 1, 1, 0, 0);
    expect(with_null, with_zero, "null_bias_matches_zero_bias");
}

// 12. The kernel does not mutate input, weight, or bias.
void test_inputs_unmodified() {
    std::vector<double> in = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    std::vector<double> w = {1, 0, 0, 1};
    double bias[1] = {2.0};
    const std::vector<double> in_before = in;
    const std::vector<double> w_before = w;
    const double bias_before = bias[0];
    (void)run(in, w, bias, 1, 1, 3, 3, 1, 2, 2, 1, 1, 0, 0);
    check(in == in_before, "input unmodified");
    check(w == w_before, "weight unmodified");
    check(bias[0] == bias_before, "bias unmodified");
}

// 13. Determinism: repeated execution is bit-identical.
void test_determinism() {
    std::vector<double> in = {-1.5, 2.5, 0.5, -3.0, 7.25, -0.125, 4.0, 9.5, 6.0};
    std::vector<double> w = {0.5, -2.0, 1.25, 3.0};
    double bias[1] = {-0.75};
    auto a = run(in, w, bias, 1, 1, 3, 3, 1, 2, 2, 1, 1, 0, 0);
    auto b = run(in, w, bias, 1, 1, 3, 3, 1, 2, 2, 1, 1, 0, 0);
    expect(a, b, "determinism");  // strict equality across runs
}

// 14. Stable-framework parity: values generated once from the reference
// tensorforge.nn.Conv2d (N=2, C=3, H=5, W=4, O=4, 3x2 kernel, stride (2,1),
// padding (1,0), with bias). Compared to a small tolerance.
void test_stable_parity() {
    const std::vector<double> in = {
        5.292, 1.2, 2.936, 6.723, 5.603, -2.932, 2.85, -0.454, -0.31, 1.232,
        0.432, 4.363, 2.283, 0.365, 1.332, 1.001, 4.482, -0.615, 0.939, -2.562,
        -7.659, 1.961, 2.593, -2.226, 6.809, -4.363, 0.137, -0.562, 4.598,
        4.408, 0.465, 1.134, -2.663, -5.942, -1.044, 0.469, 3.691, 3.607,
        -1.162, -0.907, -3.146, -4.26, -5.119, 5.852, -1.529, -1.314, -3.758,
        2.332, -4.842, -0.638, -2.686, 1.161, -1.532, -3.542, -0.085, 1.285,
        0.2, 0.907, -1.903, -1.088, -2.017, -1.079, -2.439, -5.179, 0.532,
        -1.205, -4.891, 1.388, -2.722, 0.156, 2.187, 0.387, 3.418, -3.704,
        1.207, -2.054, -2.612, -1.737, -0.935, 0.168, -3.495, 2.702, 1.397,
        -4.609, 4.465, 5.688, 3.536, -0.54, -3.212, 3.163, -1.21, 3.667,
        0.625, 2.93, 1.069, 2.12, 0.032, 5.358, 0.381, 1.206, 5.649, -4.043,
        -3.811, 2.908, -3.519, 5.831, -1.241, -2.242, 5.769, 4.442, 5.603,
        2.718, -2.584, 5.73, -0.804, 2.407, 2.842, -0.465, 1.842, 2.767};
    const std::vector<double> w = {
        0.753, -2.199, 0.596, 2.653, -1.389, -0.299, -0.87, 3.699, 1.345,
        0.815, -1.54, 1.078, -1.349, 0.064, -1.272, 1.353, 1.153, -0.417,
        0.792, -2.186, -2.983, 0.879, 0.333, 1.27, 4.766, 1.889, -1.826,
        2.234, -2.632, -0.923, -0.136, 3.427, -1.49, -1.653, -0.197, -1.327,
        2.253, -2.16, -2.295, -0.876, -0.996, 3.859, 1.899, 0.175, -2.451,
        1.689, -2.0, -3.09, 2.376, 0.634, 1.842, 0.637, 1.714, -1.302, -2.068,
        1.363, -1.607, -1.379, -0.911, 0.035, -0.708, -2.75, -1.287, -4.447,
        1.25, -3.204, -2.209, 0.104, -1.479, 3.086, -2.586, 0.534};
    double bias[4] = {0.5, -1.0, 2.0, 0.25};
    const std::vector<double> want = {
        -26.937648, 22.387109000000002, 26.243356000000002, 3.3648009999999964,
        6.296696000000004, 24.364774000000004, -6.489656000000002,
        6.226827999999999, -5.449071999999999, 0.6571030000000047,
        34.268060000000006, -17.425945000000002, 60.20946599999999,
        -28.47570900000001, 18.998810999999996, -50.225522, -38.452078,
        -2.1312770000000008, -15.561937999999998, 10.001319000000004,
        -39.049581, 39.947010000000006, -26.712177000000004, -4.776672,
        -17.383056999999997, -29.747046, -1.2418509999999996, 3.269763000000001,
        -31.00938, 28.776068999999996, -10.730542999999999, 2.505880000000001,
        -4.146955, -7.373958000000002, 9.208587999999999, 5.640510999999999,
        -26.331094000000004, 3.574489, -7.9337170000000015, 14.734358999999994,
        29.993514999999988, -11.890546000000004, 19.366334000000002,
        -4.234476000000001, 16.860616999999998, -10.647115, -13.763804,
        -15.707429999999997, 44.781447, 6.116971999999993, -18.226034,
        53.015832, -1.0478610000000002, 19.520498999999997, -16.775206,
        -35.89132600000001, 1.3745329999999996, 3.501646000000001, 44.070724,
        -16.243322999999997, 38.320396, 2.5370070000000005, 19.430079,
        -24.326079000000004, -29.158181999999996, 57.28659400000001,
        -15.621005999999998, -33.513779, -8.214220000000001, -36.976710999999995,
        -6.3375449999999995, -8.374078};
    auto out = run(in, w, bias, 2, 3, 5, 4, 4, 3, 2, 2, 1, 1, 0);
    if (out.size() != want.size()) {
        check(false, "stable_parity size");
        return;
    }
    for (size_t i = 0; i < want.size(); ++i) {
        char label[64];
        std::snprintf(label, sizeof(label), "stable_parity[%zu]", i);
        check_close(out[i], want[i], 1e-9, label);
    }
}

}  // namespace

int main() {
    test_single_channel_no_bias();
    test_bias_added_once();
    test_bias_only_with_zero_kernel();
    test_multiple_input_channels();
    test_multiple_output_channels();
    test_padding();
    test_stride();
    test_rectangular();
    test_stride_and_padding();
    test_batch();
    test_negative_and_fractional();
    test_null_bias_matches_zero_bias();
    test_inputs_unmodified();
    test_determinism();
    test_stable_parity();

    if (g_failures == 0) {
        std::printf("OK: all conv2d_forward tests passed\n");
        return 0;
    }
    std::printf("FAILED: %d conv2d_forward check(s)\n", g_failures);
    return 1;
}
