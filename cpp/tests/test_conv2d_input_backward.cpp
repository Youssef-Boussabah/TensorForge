// Dependency-free C++ test for the internal Conv2d input-gradient kernel
// (Phase D, milestone D4). No GoogleTest/Catch2 — a plain executable that
// prints failures and returns a nonzero exit code if any check fails, so
// CTest reports pass/fail.
//
// This binary compiles cpp/src/conv2d.cpp directly (see the CMake
// TF_BUILD_TESTS target): the kernels are internal, hidden-visibility
// symbols, not reachable through the shared library's public ABI. It never
// touches Python at runtime. The stable-framework parity case embeds the
// input gradient generated once from tensorforge.nn.Conv2d (compared to a
// small tolerance — the summation order is deterministic but not
// bit-identical to NumPy's einsum). The finite-difference cases build their
// own scalar objective locally from the already-tested internal forward
// kernel (tf::conv2d_forward_contiguous), so nothing is circular.

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

// Tolerance comparison for parity / finite-difference cases.
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

// Run the input-gradient kernel; the output is sized (N, C, H, W) and the
// kernel zero-initializes it (we deliberately do NOT pre-zero here, except
// where a case wants to prove the reset). Returns grad_input.
std::vector<double> run_backward(
    const std::vector<double>& grad_output, const std::vector<double>& weight,
    int64_t N, int64_t C, int64_t H, int64_t W, int64_t O,
    int64_t kh, int64_t kw, int64_t sh, int64_t sw, int64_t ph, int64_t pw) {
    const int64_t oh = out_dim(H, kh, sh, ph);
    const int64_t ow = out_dim(W, kw, sw, pw);
    (void)oh;
    (void)ow;
    std::vector<double> grad_input(
        static_cast<size_t>(N * C * H * W), 12345.678);  // garbage on purpose
    tf::conv2d_input_backward_contiguous(
        grad_output.data(), weight.data(), grad_input.data(),
        N, C, H, W, O, kh, kw, sh, sw, ph, pw,
        out_dim(H, kh, sh, ph), out_dim(W, kw, sw, pw));
    return grad_input;
}

// Forward pass used only by the finite-difference / no-bias objectives.
std::vector<double> run_forward(
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

// Scalar objective L(input) = sum_k grad_output[k] * forward(input)[k].
// Its exact gradient w.r.t. input is precisely what the backward kernel
// computes, which is what the finite-difference cases check.
double objective(
    const std::vector<double>& input, const std::vector<double>& weight,
    const double* bias, const std::vector<double>& grad_output,
    int64_t N, int64_t C, int64_t H, int64_t W, int64_t O,
    int64_t kh, int64_t kw, int64_t sh, int64_t sw, int64_t ph, int64_t pw) {
    std::vector<double> out =
        run_forward(input, weight, bias, N, C, H, W, O, kh, kw, sh, sw, ph, pw);
    double sum = 0.0;
    for (size_t k = 0; k < out.size(); ++k) {
        sum += grad_output[k] * out[k];
    }
    return sum;
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

// Central finite-difference check of the analytic input gradient against
// the local scalar objective, for one (input, weight, bias, grad_output)
// configuration. eps 1e-5, absolute tolerance 1e-6 (small O(1) values).
void finite_difference_check(
    std::vector<double> input, const std::vector<double>& weight,
    const double* bias, const std::vector<double>& grad_output,
    int64_t N, int64_t C, int64_t H, int64_t W, int64_t O,
    int64_t kh, int64_t kw, int64_t sh, int64_t sw, int64_t ph, int64_t pw,
    const char* what) {
    const std::vector<double> analytic = run_backward(
        grad_output, weight, N, C, H, W, O, kh, kw, sh, sw, ph, pw);
    const double eps = 1e-5;
    for (size_t idx = 0; idx < input.size(); ++idx) {
        const double original = input[idx];
        input[idx] = original + eps;
        const double plus = objective(input, weight, bias, grad_output,
                                      N, C, H, W, O, kh, kw, sh, sw, ph, pw);
        input[idx] = original - eps;
        const double minus = objective(input, weight, bias, grad_output,
                                       N, C, H, W, O, kh, kw, sh, sw, ph, pw);
        input[idx] = original;
        const double numeric = (plus - minus) / (2.0 * eps);
        char label[160];
        std::snprintf(label, sizeof(label), "%s[fd %zu]", what, idx);
        check_close(analytic[idx], numeric, 1e-6, label);
    }
}

// -- individual cases -------------------------------------------------

// 1. Simple 1x1 kernel: grad_input = grad_output * weight.
void test_simple_1x1() {
    std::vector<double> g = {1, 2, 3, 4};  // N1 O1 2x2
    std::vector<double> w = {2.0};         // O1 C1 1x1
    auto grad = run_backward(g, w, 1, 1, 2, 2, 1, 1, 1, 1, 1, 0, 0);
    expect(grad, {2, 4, 6, 8}, "simple_1x1");
}

// 2. Small hand-computed 2x2 kernel scatter (diagonal weight).
void test_hand_computed_2x2() {
    std::vector<double> g = {1, 2, 3, 4};  // N1 O1 2x2
    std::vector<double> w = {1, 0, 0, 1};  // O1 C1 2x2 diagonal
    auto grad = run_backward(g, w, 1, 1, 3, 3, 1, 2, 2, 1, 1, 0, 0);
    // Derived directly from the scatter relation on a 3x3 input.
    expect(grad, {1, 2, 0,  3, 5, 2,  0, 3, 4}, "hand_computed_2x2");
}

// 3. Overlapping windows accumulate (2x2 all-ones, stride 1) -> coverage
// counts of a 3x3 input.
void test_overlapping_windows() {
    std::vector<double> g = {1, 1, 1, 1};  // N1 O1 2x2, all ones
    std::vector<double> w(4, 1.0);         // 2x2 all-ones
    auto grad = run_backward(g, w, 1, 1, 3, 3, 1, 2, 2, 1, 1, 0, 0);
    expect(grad, {1, 2, 1,  2, 4, 2,  1, 2, 1}, "overlapping_windows");
}

// 4. Multiple input channels: each channel scaled by its own 1x1 weight.
void test_multiple_input_channels() {
    std::vector<double> g = {1, 2, 3, 4};    // N1 O1 2x2
    std::vector<double> w = {1, 2};          // O1 C2 1x1: ch0=1, ch1=2
    auto grad = run_backward(g, w, 1, 2, 2, 2, 1, 1, 1, 1, 1, 0, 0);
    expect(grad, {1, 2, 3, 4,   2, 4, 6, 8}, "multiple_input_channels");
}

// 5. Multiple output channels accumulate into one input channel.
void test_multiple_output_channels() {
    // O=2, C=1, 1x1; weights 1 and 10. g_o0=[1,2,3,4], g_o1=[1,1,1,1].
    std::vector<double> g = {1, 2, 3, 4,   1, 1, 1, 1};  // N1 O2 2x2
    std::vector<double> w = {1, 10};                     // O2 C1 1x1
    auto grad = run_backward(g, w, 1, 1, 2, 2, 2, 1, 1, 1, 1, 0, 0);
    expect(grad, {11, 12, 13, 14}, "multiple_output_channels");
}

// 6. Batch greater than one — each sample independent.
void test_batch() {
    std::vector<double> g = {1, 2, 3, 4,   5, 6, 7, 8};  // N2 O1 2x2
    std::vector<double> w = {2.0};                        // 1x1
    auto grad = run_backward(g, w, 2, 1, 2, 2, 1, 1, 1, 1, 1, 0, 0);
    expect(grad, {2, 4, 6, 8,   10, 12, 14, 16}, "batch");
}

// 7. Stride greater than one, non-tiling: untouched input row/col stay 0.
void test_stride_leaves_untouched_zero() {
    // H=W=5, 2x2 all-ones, stride 2 -> out 2x2; row/col 4 never covered.
    std::vector<double> g = {1, 2, 3, 4};  // N1 O1 2x2
    std::vector<double> w(4, 1.0);
    auto grad = run_backward(g, w, 1, 1, 5, 5, 1, 2, 2, 2, 2, 0, 0);
    expect(grad, {
        1, 1, 2, 2, 0,
        1, 1, 2, 2, 0,
        3, 3, 4, 4, 0,
        3, 3, 4, 4, 0,
        0, 0, 0, 0, 0}, "stride_untouched_zero");
}

// 8. Symmetric padding: 3x3 all-ones kernel, pad 1 -> coverage counts.
void test_symmetric_padding() {
    std::vector<double> g(9, 1.0);   // N1 O1 3x3, all ones
    std::vector<double> w(9, 1.0);   // 3x3 all-ones
    auto grad = run_backward(g, w, 1, 1, 3, 3, 1, 3, 3, 1, 1, 1, 1);
    expect(grad, {4, 6, 4,  6, 9, 6,  4, 6, 4}, "symmetric_padding");
}

// 9. Rectangular input (2x3), 2x2 diagonal weight.
void test_rectangular_input() {
    std::vector<double> g = {1, 2};        // N1 O1 out (1x2)
    std::vector<double> w = {1, 0, 0, 1};  // 2x2 diagonal
    auto grad = run_backward(g, w, 1, 1, 2, 3, 1, 2, 2, 1, 1, 0, 0);
    expect(grad, {1, 2, 0,   0, 1, 2}, "rectangular_input");
}

// 10. Rectangular kernel (2x3): validated by finite differences.
void test_rectangular_kernel() {
    std::vector<double> in = {  // N1 C1 3x4
        0.5, -1.5, 2.0, 0.25, -0.75, 1.25, 3.0, -2.0, 0.1, -0.2, 0.3, -0.4};
    std::vector<double> w = {0.7, -1.1, 0.3, -0.9, 1.4, 0.6};  // O1 C1 2x3
    // out (3-2+1) x (4-3+1) = 2 x 2
    std::vector<double> g = {1.0, -2.0, 0.5, 3.0};
    finite_difference_check(in, w, nullptr, g,
                            1, 1, 3, 4, 1, 2, 3, 1, 1, 0, 0,
                            "rectangular_kernel");
}

// 11. Tuple-style asymmetric spatial values: stride (2,1), padding (1,0).
void test_asymmetric_spatial() {
    std::vector<double> in = {  // N1 C1 4x3
        1, -2, 3, -4, 5, -6, 0.5, -0.5, 1.5, -1.5, 2.5, -2.5};
    std::vector<double> w = {0.5, -1.0, 1.5, -2.0};  // O1 C1 2x2
    // out_h = (4+2-2)/2+1 = 3, out_w = (3+0-2)/1+1 = 2
    std::vector<double> g = {1, 2, 3, 4, 5, 6};
    finite_difference_check(in, w, nullptr, g,
                            1, 1, 4, 3, 1, 2, 2, 2, 1, 1, 0,
                            "asymmetric_spatial");
}

// 12. Combined stride and padding.
void test_combined_stride_padding() {
    std::vector<double> in = {1, 2, 3, 4, 5, 6, 7, 8, 9};  // N1 C1 3x3
    std::vector<double> w = {0.25, -0.5, 0.75, -1.0};      // O1 C1 2x2
    // stride 2, padding 1 -> out 2x2
    std::vector<double> g = {-1.0, 0.5, 2.0, -3.0};
    finite_difference_check(in, w, nullptr, g,
                            1, 1, 3, 3, 1, 2, 2, 2, 2, 1, 1,
                            "combined_stride_padding");
}

// 13. Negative and fractional weights / upstream gradients.
void test_negative_and_fractional() {
    std::vector<double> in = {  // N1 C2 3x3
        -1.5, 2.25, 0.5, -3.0, 4.5, -0.25, 1.0, -2.0, 3.5,
        0.75, -1.25, 2.0, -0.5, 1.5, -3.5, 0.25, -0.75, 1.25};
    std::vector<double> w = {  // O2 C2 2x2
        0.5, -1.5, 2.0, -0.25,  -1.0, 0.75, -2.5, 1.25,
        1.75, -0.5, 0.3, -1.2,  -0.7, 2.4, -1.6, 0.9};
    // out (3-2+1) x (3-2+1) = 2 x 2, O=2
    std::vector<double> g = {-1.5, 0.25, 2.75, -0.5,  1.25, -3.0, 0.6, -2.2};
    finite_difference_check(in, w, nullptr, g,
                            1, 2, 3, 3, 2, 2, 2, 1, 1, 0, 0,
                            "negative_and_fractional");
}

// 14. Output initialization: garbage in grad_input must be fully reset.
void test_output_initialization() {
    std::vector<double> g = {1, 2, 3, 4};
    std::vector<double> w = {1, 0, 0, 1};
    std::vector<double> grad_input(9, -999.0);  // nonzero garbage
    tf::conv2d_input_backward_contiguous(
        g.data(), w.data(), grad_input.data(),
        1, 1, 3, 3, 1, 2, 2, 1, 1, 0, 0, 2, 2);
    expect(grad_input, {1, 2, 0,  3, 5, 2,  0, 3, 4}, "output_initialization");
}

// 15. Immutability: grad_output and weight are unchanged.
void test_inputs_unmodified() {
    std::vector<double> g = {1, 2, 3, 4, 5, 6, 7, 8};  // N1 O2 2x2
    std::vector<double> w = {2.0, -3.0};               // O2 C1 1x1
    const std::vector<double> g_before = g;
    const std::vector<double> w_before = w;
    (void)run_backward(g, w, 1, 1, 2, 2, 2, 1, 1, 1, 1, 0, 0);
    check(g == g_before, "grad_output unmodified");
    check(w == w_before, "weight unmodified");
}

// 16. Determinism: repeated execution is bit-identical.
void test_determinism() {
    std::vector<double> g = {-1.5, 2.5, 0.5, -3.0, 7.25, -0.125, 4.0, 9.5};
    std::vector<double> w = {0.5, -2.0, 1.25, 3.0};  // O2 C1 1x1... (O2)
    auto a = run_backward(g, w, 1, 1, 2, 2, 2, 1, 1, 1, 1, 0, 0);
    auto b = run_backward(g, w, 1, 1, 2, 2, 2, 1, 1, 1, 1, 0, 0);
    expect(a, b, "determinism");
}

// 17. Stable-framework parity: input gradient generated once from
// tensorforge.nn.Conv2d (N=1, C=2, H=4, W=3, O=2, 2x2 kernel, stride (2,1),
// padding (1,0)). Only the weight and upstream gradient feed the input
// gradient (the input values and bias do not), so only those are embedded.
void test_stable_parity() {
    const std::vector<double> w = {
        0.568, 0.784, -0.63, -0.222, 1.972, -0.186, 0.03, 0.441,
        -0.577, -0.39, 1.678, 1.296, -1.103, 0.065, -1.289, -1.532};
    const std::vector<double> g = {
        -0.082, 1.47, -1.057, 0.136, 0.081, 0.086,
        0.993, 0.573, -0.549, -1.351, -0.554, -0.021};
    const std::vector<double> want = {
        1.717914, 1.340526, 0.4162679999999999, -0.2836029999999999,
        0.242197, 0.633514, -0.255312, -2.8295079999999997,
        -1.781088, 0.365666, 0.340529, 0.075614,
        -1.2824369999999998, -2.251935, -0.22956599999999994,
        -1.4788569999999996, 1.919262, -0.113111,
        0.675951, 2.12045, 2.129708, 0.770794,
        0.14167899999999997, -0.017361};
    auto grad = run_backward(g, w, 1, 2, 4, 3, 2, 2, 2, 2, 1, 1, 0);
    if (grad.size() != want.size()) {
        check(false, "stable_parity size");
        return;
    }
    for (size_t i = 0; i < want.size(); ++i) {
        char label[64];
        std::snprintf(label, sizeof(label), "stable_parity[%zu]", i);
        check_close(grad[i], want[i], 1e-9, label);
    }
}

// 18. Finite-difference validation on a deterministic multi-channel case
// with stride and padding.
void test_finite_difference() {
    std::vector<double> in = {  // N2 C2 4x4
        1.0, -0.5, 2.0, 0.25, -1.0, 0.75, -2.0, 1.5,
        0.5, -1.5, 3.0, -0.25, 2.5, -0.75, 1.25, -3.5,
        -1.2, 0.6, 0.9, -0.3, 1.1, -2.1, 0.4, 1.9,
        -0.8, 2.2, -1.6, 0.7, 1.3, -0.9, 0.2, -1.4,
        // sample 2
        0.3, -1.3, 2.4, -0.6, 1.8, -0.2, -1.9, 0.9,
        -0.4, 1.6, -2.3, 0.8, 1.2, -1.1, 0.5, -0.7,
        2.1, -0.9, 1.4, -1.8, 0.6, 0.1, -0.3, 1.7,
        -1.5, 0.4, 2.0, -0.25, 0.75, -2.5, 1.0, -0.5};
    std::vector<double> w = {  // O3 C2 2x2
        0.5, -1.0, 1.5, -0.5,  0.25, 0.75, -1.25, 2.0,
        -0.6, 1.1, 0.3, -0.9,  1.4, -2.1, 0.7, -0.4,
        -1.7, 0.8, 2.2, -0.3,  0.9, -1.2, 0.6, 1.3};
    // stride 2, padding 1: out_h = (4+2-2)/2+1 = 3, out_w = 3. O=3.
    std::vector<double> g(2 * 3 * 3 * 3, 0.0);
    for (size_t k = 0; k < g.size(); ++k) {
        g[k] = 0.1 * static_cast<double>((k % 7)) - 0.3;  // deterministic
    }
    double bias[3] = {0.5, -1.0, 2.0};
    finite_difference_check(in, w, bias, g,
                            2, 2, 4, 4, 3, 2, 2, 2, 2, 1, 1,
                            "finite_difference");
}

// 19. No-bias equivalence: the input gradient is independent of the forward
// bias. The analytic gradient reads no bias; confirm the finite-difference
// gradient of the objective is identical for two different biases (and
// equals the analytic gradient), with the upstream gradient held fixed.
void test_no_bias_equivalence() {
    std::vector<double> in = {  // N1 C1 3x3
        0.4, -1.1, 2.3, -0.7, 1.6, -2.2, 0.9, -0.3, 1.8};
    std::vector<double> w = {0.6, -1.3, 0.8, -0.4};  // O1 C1 2x2
    std::vector<double> g = {1.0, -0.5, 2.0, -1.5};  // out 2x2

    const std::vector<double> analytic =
        run_backward(g, w, 1, 1, 3, 3, 1, 2, 2, 1, 1, 0, 0);

    // Central differences of the objective for two different biases must
    // both reproduce the (bias-independent) analytic gradient.
    double bias_a[1] = {0.0};
    double bias_b[1] = {7.25};
    const double eps = 1e-5;
    for (size_t idx = 0; idx < in.size(); ++idx) {
        double grad_a, grad_b;
        for (int which = 0; which < 2; ++which) {
            const double* bias = (which == 0) ? bias_a : bias_b;
            std::vector<double> x = in;
            const double original = x[idx];
            x[idx] = original + eps;
            const double plus = objective(x, w, bias, g,
                                          1, 1, 3, 3, 1, 2, 2, 1, 1, 0, 0);
            x[idx] = original - eps;
            const double minus = objective(x, w, bias, g,
                                           1, 1, 3, 3, 1, 2, 2, 1, 1, 0, 0);
            const double numeric = (plus - minus) / (2.0 * eps);
            if (which == 0) {
                grad_a = numeric;
            } else {
                grad_b = numeric;
            }
        }
        char label[96];
        std::snprintf(label, sizeof(label), "no_bias_equiv[a %zu]", idx);
        check_close(analytic[idx], grad_a, 1e-6, label);
        std::snprintf(label, sizeof(label), "no_bias_equiv[b %zu]", idx);
        check_close(analytic[idx], grad_b, 1e-6, label);
        std::snprintf(label, sizeof(label), "no_bias_equiv[a==b %zu]", idx);
        check_close(grad_a, grad_b, 1e-9, label);
    }
}

}  // namespace

int main() {
    test_simple_1x1();
    test_hand_computed_2x2();
    test_overlapping_windows();
    test_multiple_input_channels();
    test_multiple_output_channels();
    test_batch();
    test_stride_leaves_untouched_zero();
    test_symmetric_padding();
    test_rectangular_input();
    test_rectangular_kernel();
    test_asymmetric_spatial();
    test_combined_stride_padding();
    test_negative_and_fractional();
    test_output_initialization();
    test_inputs_unmodified();
    test_determinism();
    test_stable_parity();
    test_finite_difference();
    test_no_bias_equivalence();

    if (g_failures == 0) {
        std::printf("OK: all conv2d_input_backward tests passed\n");
        return 0;
    }
    std::printf("FAILED: %d conv2d_input_backward check(s)\n", g_failures);
    return 1;
}
