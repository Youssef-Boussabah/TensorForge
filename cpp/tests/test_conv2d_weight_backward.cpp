// Dependency-free C++ test for the internal Conv2d weight-gradient kernel
// (Phase D, milestone D5). No GoogleTest/Catch2 — a plain executable that
// prints failures and returns a nonzero exit code if any check fails, so
// CTest reports pass/fail.
//
// This binary compiles cpp/src/conv2d.cpp directly (see the CMake
// TF_BUILD_TESTS target): the kernels are internal, hidden-visibility
// symbols, not reachable through the shared library's public ABI. It never
// touches Python at runtime. Three independent oracles cross-check the
// kernel: exact hand-computed values, an explicit zero-padded
// materialization reference (a different strategy from the kernel's
// skip-on-out-of-bounds), the stable-framework parity values embedded from
// tensorforge.nn.Conv2d, and central finite differences of the scalar
// objective built from the already-tested internal forward kernel.

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

int64_t out_dim(int64_t in, int64_t k, int64_t s, int64_t p) {
    return (in + 2 * p - k) / s + 1;
}

// Row-major flat offset into a 4-D array of extents (·, d1, d2, d3).
int64_t idx4(int64_t i0, int64_t i1, int64_t i2, int64_t i3,
             int64_t d1, int64_t d2, int64_t d3) {
    return ((i0 * d1 + i1) * d2 + i2) * d3 + i3;
}

// Run the weight-gradient kernel; the output is (O, C, kh, kw) and the
// kernel zero-initializes it (we pre-fill garbage to prove the reset).
std::vector<double> run_weight_backward(
    const std::vector<double>& grad_output, const std::vector<double>& input,
    int64_t N, int64_t C, int64_t H, int64_t W, int64_t O,
    int64_t kh, int64_t kw, int64_t sh, int64_t sw, int64_t ph, int64_t pw) {
    std::vector<double> grad_weight(
        static_cast<size_t>(O * C * kh * kw), -98765.4321);  // garbage
    tf::conv2d_weight_backward_contiguous(
        grad_output.data(), input.data(), grad_weight.data(),
        N, C, H, W, O, kh, kw, sh, sw, ph, pw,
        out_dim(H, kh, sh, ph), out_dim(W, kw, sw, pw));
    return grad_weight;
}

// Independent oracle: materialize an explicit zero-padded input and
// accumulate grad_weight[o,c,p,q] += g[n,o,i,j] * padded[n,c, i*sh+p,
// j*sw+q] with NO skip logic (the padded array carries real zeros). A
// genuinely different strategy from the kernel's out-of-bounds skip, so
// agreement validates both the coordinate map and the padding handling.
std::vector<double> weight_grad_padded_reference(
    const std::vector<double>& input, const std::vector<double>& g,
    int64_t N, int64_t C, int64_t H, int64_t W, int64_t O,
    int64_t kh, int64_t kw, int64_t sh, int64_t sw, int64_t ph, int64_t pw) {
    const int64_t oh = out_dim(H, kh, sh, ph);
    const int64_t ow = out_dim(W, kw, sw, pw);
    const int64_t Hp = H + 2 * ph;
    const int64_t Wp = W + 2 * pw;
    std::vector<double> padded(static_cast<size_t>(N * C * Hp * Wp), 0.0);
    for (int64_t n = 0; n < N; ++n)
        for (int64_t c = 0; c < C; ++c)
            for (int64_t y = 0; y < H; ++y)
                for (int64_t x = 0; x < W; ++x)
                    padded[idx4(n, c, y + ph, x + pw, C, Hp, Wp)] =
                        input[idx4(n, c, y, x, C, H, W)];
    std::vector<double> gw(static_cast<size_t>(O * C * kh * kw), 0.0);
    for (int64_t n = 0; n < N; ++n)
        for (int64_t o = 0; o < O; ++o)
            for (int64_t i = 0; i < oh; ++i)
                for (int64_t j = 0; j < ow; ++j) {
                    const double gv = g[idx4(n, o, i, j, O, oh, ow)];
                    for (int64_t c = 0; c < C; ++c)
                        for (int64_t p = 0; p < kh; ++p)
                            for (int64_t q = 0; q < kw; ++q)
                                gw[idx4(o, c, p, q, C, kh, kw)] +=
                                    gv * padded[idx4(n, c, i * sh + p,
                                                     j * sw + q, C, Hp, Wp)];
                }
    return gw;
}

// Forward pass, used only by the finite-difference / bias objectives.
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

// Scalar objective L(weight) = sum_k g[k] * forward(weight)[k]; its exact
// gradient w.r.t. weight is what the backward kernel computes.
double objective(
    const std::vector<double>& input, const std::vector<double>& weight,
    const double* bias, const std::vector<double>& g,
    int64_t N, int64_t C, int64_t H, int64_t W, int64_t O,
    int64_t kh, int64_t kw, int64_t sh, int64_t sw, int64_t ph, int64_t pw) {
    std::vector<double> out =
        run_forward(input, weight, bias, N, C, H, W, O, kh, kw, sh, sw, ph, pw);
    double sum = 0.0;
    for (size_t k = 0; k < out.size(); ++k) sum += g[k] * out[k];
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

void expect_close(const std::vector<double>& got,
                  const std::vector<double>& want, double tol,
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
        check_close(got[i], want[i], tol, label);
    }
}

// Kernel-vs-padded-reference exact agreement for one configuration.
void expect_matches_reference(
    const std::vector<double>& input, const std::vector<double>& g,
    int64_t N, int64_t C, int64_t H, int64_t W, int64_t O,
    int64_t kh, int64_t kw, int64_t sh, int64_t sw, int64_t ph, int64_t pw,
    const char* what) {
    auto got = run_weight_backward(g, input, N, C, H, W, O, kh, kw, sh, sw, ph, pw);
    auto want = weight_grad_padded_reference(
        input, g, N, C, H, W, O, kh, kw, sh, sw, ph, pw);
    // Both accumulate in the same order over the same real values, so exact
    // equality is expected here.
    expect(got, want, what);
}

// Central finite-difference check of analytic grad_weight against the local
// objective. eps 1e-5, absolute tolerance 1e-6 (small O(1) values).
void finite_difference_weight_check(
    const std::vector<double>& input, std::vector<double> weight,
    const double* bias, const std::vector<double>& g,
    int64_t N, int64_t C, int64_t H, int64_t W, int64_t O,
    int64_t kh, int64_t kw, int64_t sh, int64_t sw, int64_t ph, int64_t pw,
    const char* what) {
    const std::vector<double> analytic =
        run_weight_backward(g, input, N, C, H, W, O, kh, kw, sh, sw, ph, pw);
    const double eps = 1e-5;
    for (size_t idx = 0; idx < weight.size(); ++idx) {
        const double original = weight[idx];
        weight[idx] = original + eps;
        const double plus = objective(input, weight, bias, g,
                                      N, C, H, W, O, kh, kw, sh, sw, ph, pw);
        weight[idx] = original - eps;
        const double minus = objective(input, weight, bias, g,
                                       N, C, H, W, O, kh, kw, sh, sw, ph, pw);
        weight[idx] = original;
        const double numeric = (plus - minus) / (2.0 * eps);
        char label[160];
        std::snprintf(label, sizeof(label), "%s[fd %zu]", what, idx);
        check_close(analytic[idx], numeric, 1e-6, label);
    }
}

// -- individual cases -------------------------------------------------

// 1. Simple 1x1 kernel: grad_weight = dot(input, grad_output).
void test_simple_1x1() {
    std::vector<double> in = {1, 2, 3, 4};  // N1 C1 2x2
    std::vector<double> g = {5, 6, 7, 8};   // N1 O1 2x2
    auto gw = run_weight_backward(g, in, 1, 1, 2, 2, 1, 1, 1, 1, 1, 0, 0);
    expect(gw, {70}, "simple_1x1");  // 1*5+2*6+3*7+4*8
}

// 2. Small hand-computed 2x2 kernel.
void test_hand_computed_2x2() {
    std::vector<double> in = {1, 2, 3, 4, 5, 6, 7, 8, 9};  // N1 C1 3x3
    std::vector<double> g = {1, 2, 3, 4};                  // N1 O1 2x2
    auto gw = run_weight_backward(g, in, 1, 1, 3, 3, 1, 2, 2, 1, 1, 0, 0);
    expect(gw, {37, 47, 67, 77}, "hand_computed_2x2");
}

// 3. Accumulation across batch.
void test_batch_accumulation() {
    std::vector<double> in = {1, 2, 3, 4,   10, 20, 30, 40};  // N2 C1 2x2
    std::vector<double> g = {1, 1, 1, 1,   2, 2, 2, 2};       // N2 O1 2x2
    auto gw = run_weight_backward(g, in, 2, 1, 2, 2, 1, 1, 1, 1, 1, 0, 0);
    expect(gw, {210}, "batch_accumulation");  // 10 + 200
}

// 4. Accumulation across output positions (2x2 kernel over 2x3 input).
void test_output_position_accumulation() {
    std::vector<double> in = {1, 2, 3, 4, 5, 6};  // N1 C1 2x3
    std::vector<double> g = {10, 20};             // N1 O1 out (1x2)
    auto gw = run_weight_backward(g, in, 1, 1, 2, 3, 1, 2, 2, 1, 1, 0, 0);
    expect(gw, {50, 80, 140, 170}, "output_position_accumulation");
}

// 5. Multiple input channels.
void test_multiple_input_channels() {
    std::vector<double> in = {1, 2, 3, 4,   10, 20, 30, 40};  // N1 C2 2x2
    std::vector<double> g = {1, 1, 1, 1};                     // N1 O1 2x2
    auto gw = run_weight_backward(g, in, 1, 2, 2, 2, 1, 1, 1, 1, 1, 0, 0);
    expect(gw, {10, 100}, "multiple_input_channels");  // O1 C2 1x1
}

// 6. Multiple output channels.
void test_multiple_output_channels() {
    std::vector<double> in = {1, 2, 3, 4};              // N1 C1 2x2
    std::vector<double> g = {1, 1, 1, 1,   1, 2, 3, 4};  // N1 O2 2x2
    auto gw = run_weight_backward(g, in, 1, 1, 2, 2, 2, 1, 1, 1, 1, 0, 0);
    // o0: 1*(1+2+3+4)=10 ; o1: 1+4+9+16=30
    expect(gw, {10, 30}, "multiple_output_channels");
}

// 7. Stride greater than one (vs padded reference).
void test_stride() {
    std::vector<double> in = {
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};  // N1 C1 4x4
    std::vector<double> g = {1, 2, 3, 4};  // stride 2 -> out 2x2
    expect_matches_reference(in, g, 1, 1, 4, 4, 1, 2, 2, 2, 2, 0, 0, "stride");
}

// 8. Symmetric padding (vs padded reference).
void test_symmetric_padding() {
    std::vector<double> in = {1, 2, 3, 4, 5, 6, 7, 8, 9};  // N1 C1 3x3
    std::vector<double> g(9, 0.0);
    for (size_t k = 0; k < g.size(); ++k) g[k] = 0.5 * (double)k - 1.0;
    // 3x3 kernel, pad 1, stride 1 -> out 3x3
    expect_matches_reference(in, g, 1, 1, 3, 3, 1, 3, 3, 1, 1, 1, 1,
                             "symmetric_padding");
}

// 9. Rectangular input (3x2), hand-computed.
void test_rectangular_input() {
    std::vector<double> in = {1, 2, 3, 4, 5, 6};  // N1 C1 3x2
    std::vector<double> g = {10, 20};             // out (2x1)
    auto gw = run_weight_backward(g, in, 1, 1, 3, 2, 1, 2, 2, 1, 1, 0, 0);
    expect(gw, {70, 100, 130, 160}, "rectangular_input");
}

// 10. Rectangular kernel (vs padded reference).
void test_rectangular_kernel() {
    std::vector<double> in = {  // N1 C1 3x4
        0.5, -1.5, 2.0, 0.25, -0.75, 1.25, 3.0, -2.0, 0.1, -0.2, 0.3, -0.4};
    std::vector<double> g = {1.0, -2.0, 0.5, 3.0};  // 2x3 kernel -> out 2x2
    expect_matches_reference(in, g, 1, 1, 3, 4, 1, 2, 3, 1, 1, 0, 0,
                             "rectangular_kernel");
}

// 11. Separate height/width stride and padding (vs padded reference).
void test_asymmetric_spatial() {
    std::vector<double> in = {  // N1 C1 4x3
        1, -2, 3, -4, 5, -6, 0.5, -0.5, 1.5, -1.5, 2.5, -2.5};
    std::vector<double> g = {1, 2, 3, 4, 5, 6};  // stride (2,1) pad (1,0) -> 3x2
    expect_matches_reference(in, g, 1, 1, 4, 3, 1, 2, 2, 2, 1, 1, 0,
                             "asymmetric_spatial");
}

// 12. Combined stride and padding (vs padded reference).
void test_combined_stride_padding() {
    std::vector<double> in = {1, 2, 3, 4, 5, 6, 7, 8, 9};  // N1 C1 3x3
    std::vector<double> g = {-1.0, 0.5, 2.0, -3.0};        // stride2 pad1 -> 2x2
    expect_matches_reference(in, g, 1, 1, 3, 3, 1, 2, 2, 2, 2, 1, 1,
                             "combined_stride_padding");
}

// 13. Negative and fractional values (vs padded reference).
void test_negative_and_fractional() {
    std::vector<double> in = {  // N1 C2 3x3
        -1.5, 2.25, 0.5, -3.0, 4.5, -0.25, 1.0, -2.0, 3.5,
        0.75, -1.25, 2.0, -0.5, 1.5, -3.5, 0.25, -0.75, 1.25};
    std::vector<double> g = {  // N1 O2 2x2
        -1.5, 0.25, 2.75, -0.5,   1.25, -3.0, 0.6, -2.2};
    expect_matches_reference(in, g, 1, 2, 3, 3, 2, 2, 2, 1, 1, 0, 0,
                             "negative_and_fractional");
}

// 14. Output initialization: garbage in grad_weight must be fully reset.
void test_output_initialization() {
    std::vector<double> in = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    std::vector<double> g = {1, 2, 3, 4};
    std::vector<double> gw(4, 777.0);  // nonzero garbage
    tf::conv2d_weight_backward_contiguous(
        g.data(), in.data(), gw.data(),
        1, 1, 3, 3, 1, 2, 2, 1, 1, 0, 0, 2, 2);
    expect(gw, {37, 47, 67, 77}, "output_initialization");
}

// 15. Immutability: input and grad_output unchanged.
void test_inputs_unmodified() {
    std::vector<double> in = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    std::vector<double> g = {1, 2, 3, 4};
    const std::vector<double> in_before = in;
    const std::vector<double> g_before = g;
    (void)run_weight_backward(g, in, 1, 1, 3, 3, 1, 2, 2, 1, 1, 0, 0);
    check(in == in_before, "input unmodified");
    check(g == g_before, "grad_output unmodified");
}

// 16. Determinism: repeated execution is bit-identical.
void test_determinism() {
    std::vector<double> in = {-1.5, 2.5, 0.5, -3.0, 7.25, -0.125, 4.0, 9.5, 6.0};
    std::vector<double> g = {0.5, -2.0, 1.25, 3.0};
    auto a = run_weight_backward(g, in, 1, 1, 3, 3, 1, 2, 2, 1, 1, 0, 0);
    auto b = run_weight_backward(g, in, 1, 1, 3, 3, 1, 2, 2, 1, 1, 0, 0);
    expect(a, b, "determinism");
}

// 17. Stable-framework parity: weight gradient generated once from
// tensorforge.nn.Conv2d (N=1, C=2, H=4, W=3, O=2, 2x2 kernel, stride (2,1),
// padding (1,0)). Weight gradient depends on the input and upstream
// gradient (not the bias), so only those are embedded.
void test_stable_parity() {
    const std::vector<double> in = {
        -1.232, 0.267, -0.007, 0.502, -1.327, 1.108, 0.094, -1.171,
        -1.358, -1.307, -0.718, 1.186, 0.895, -0.544, -0.547, 1.93,
        0.903, -0.803, -0.15, -0.43, 0.152, 0.879, -0.275, 0.369};
    const std::vector<double> g = {
        -0.082, 1.47, -1.057, 0.136, 0.081, 0.086,
        0.993, 0.573, -0.549, -1.351, -0.554, -0.021};
    const std::vector<double> want = {
        -0.878701, 1.597165, 0.2349, 1.020875,
        -1.8696529999999996, -1.0542200000000002, -0.773, -0.2843000000000001,
        2.256335, -0.39551900000000006, 0.46003000000000016, 2.7386570000000003,
        -2.7607140000000006, 0.733707, 1.240303, -0.822905};
    auto gw = run_weight_backward(g, in, 1, 2, 4, 3, 2, 2, 2, 2, 1, 1, 0);
    expect_close(gw, want, 1e-9, "stable_parity");
}

// 18. Finite-difference validation on a deterministic multi-channel case
// with stride and padding.
void test_finite_difference() {
    std::vector<double> in = {  // N2 C2 4x4
        1.0, -0.5, 2.0, 0.25, -1.0, 0.75, -2.0, 1.5,
        0.5, -1.5, 3.0, -0.25, 2.5, -0.75, 1.25, -3.5,
        -1.2, 0.6, 0.9, -0.3, 1.1, -2.1, 0.4, 1.9,
        -0.8, 2.2, -1.6, 0.7, 1.3, -0.9, 0.2, -1.4,
        0.3, -1.3, 2.4, -0.6, 1.8, -0.2, -1.9, 0.9,
        -0.4, 1.6, -2.3, 0.8, 1.2, -1.1, 0.5, -0.7,
        2.1, -0.9, 1.4, -1.8, 0.6, 0.1, -0.3, 1.7,
        -1.5, 0.4, 2.0, -0.25, 0.75, -2.5, 1.0, -0.5};
    std::vector<double> w = {  // O3 C2 2x2
        0.5, -1.0, 1.5, -0.5,  0.25, 0.75, -1.25, 2.0,
        -0.6, 1.1, 0.3, -0.9,  1.4, -2.1, 0.7, -0.4,
        -1.7, 0.8, 2.2, -0.3,  0.9, -1.2, 0.6, 1.3};
    std::vector<double> g(2 * 3 * 3 * 3, 0.0);  // stride2 pad1 -> out 3x3, O3
    for (size_t k = 0; k < g.size(); ++k)
        g[k] = 0.1 * (double)(k % 7) - 0.3;
    double bias[3] = {0.5, -1.0, 2.0};
    finite_difference_weight_check(in, w, bias, g,
                                   2, 2, 4, 4, 3, 2, 2, 2, 2, 1, 1,
                                   "finite_difference");
}

// 19. Bias independence: changing the forward bias must not change the
// finite-difference weight gradient (grad_weight reads no bias).
void test_bias_independence() {
    std::vector<double> in = {0.4, -1.1, 2.3, -0.7, 1.6, -2.2, 0.9, -0.3, 1.8};
    std::vector<double> w = {0.6, -1.3, 0.8, -0.4};
    std::vector<double> g = {1.0, -0.5, 2.0, -1.5};  // out 2x2
    const std::vector<double> analytic =
        run_weight_backward(g, in, 1, 1, 3, 3, 1, 2, 2, 1, 1, 0, 0);
    double bias_a[1] = {0.0};
    double bias_b[1] = {7.25};
    const double eps = 1e-5;
    for (size_t idx = 0; idx < w.size(); ++idx) {
        double grad_a = 0.0, grad_b = 0.0;
        for (int which = 0; which < 2; ++which) {
            const double* bias = (which == 0) ? bias_a : bias_b;
            std::vector<double> ww = w;
            const double original = ww[idx];
            ww[idx] = original + eps;
            const double plus = objective(in, ww, bias, g,
                                          1, 1, 3, 3, 1, 2, 2, 1, 1, 0, 0);
            ww[idx] = original - eps;
            const double minus = objective(in, ww, bias, g,
                                           1, 1, 3, 3, 1, 2, 2, 1, 1, 0, 0);
            const double numeric = (plus - minus) / (2.0 * eps);
            if (which == 0) grad_a = numeric; else grad_b = numeric;
        }
        char label[96];
        std::snprintf(label, sizeof(label), "bias_indep[a %zu]", idx);
        check_close(analytic[idx], grad_a, 1e-6, label);
        std::snprintf(label, sizeof(label), "bias_indep[b %zu]", idx);
        check_close(analytic[idx], grad_b, 1e-6, label);
        // grad_weight is exactly bias-independent (the kernel reads no
        // bias); the two *finite-difference* estimates agree only to the FD
        // accuracy floor, since a large bias inflates the objective and its
        // subtractive rounding. The analytic checks above are the tight
        // proof; this is the honest FD-vs-FD tolerance.
        std::snprintf(label, sizeof(label), "bias_indep[a==b %zu]", idx);
        check_close(grad_a, grad_b, 1e-6, label);
    }
}

// 20. No-bias equivalence: the forward objective's weight gradient is the
// same whether the forward used no bias (nullptr) or an explicit zero bias.
void test_no_bias_equivalence() {
    std::vector<double> in = {0.4, -1.1, 2.3, -0.7, 1.6, -2.2, 0.9, -0.3, 1.8};
    std::vector<double> w = {0.6, -1.3, 0.8, -0.4};
    std::vector<double> g = {1.0, -0.5, 2.0, -1.5};
    double zero_bias[1] = {0.0};
    const double eps = 1e-5;
    for (size_t idx = 0; idx < w.size(); ++idx) {
        std::vector<double> ww = w;
        const double original = ww[idx];
        ww[idx] = original + eps;
        const double plus_null = objective(in, ww, nullptr, g,
                                           1, 1, 3, 3, 1, 2, 2, 1, 1, 0, 0);
        const double plus_zero = objective(in, ww, zero_bias, g,
                                            1, 1, 3, 3, 1, 2, 2, 1, 1, 0, 0);
        ww[idx] = original - eps;
        const double minus_null = objective(in, ww, nullptr, g,
                                            1, 1, 3, 3, 1, 2, 2, 1, 1, 0, 0);
        const double minus_zero = objective(in, ww, zero_bias, g,
                                             1, 1, 3, 3, 1, 2, 2, 1, 1, 0, 0);
        const double grad_null = (plus_null - minus_null) / (2.0 * eps);
        const double grad_zero = (plus_zero - minus_zero) / (2.0 * eps);
        char label[96];
        std::snprintf(label, sizeof(label), "no_bias_equiv[%zu]", idx);
        check_close(grad_null, grad_zero, 1e-12, label);
    }
}

// 21. Boundary / padding-only contributions: with a kernel larger than the
// input and padding, many taps land on the pad border. The kernel's skip
// must match the explicit-zero padded reference exactly (padded taps add 0).
void test_padding_only_contributions() {
    std::vector<double> in = {1, 2, 3, 4};  // N1 C1 2x2
    std::vector<double> g;                  // 3x3 kernel, pad1 -> out 2x2
    g = {1.0, -2.0, 3.0, -4.0};
    expect_matches_reference(in, g, 1, 1, 2, 2, 1, 3, 3, 1, 1, 1, 1,
                             "padding_only_contributions");
}

}  // namespace

int main() {
    test_simple_1x1();
    test_hand_computed_2x2();
    test_batch_accumulation();
    test_output_position_accumulation();
    test_multiple_input_channels();
    test_multiple_output_channels();
    test_stride();
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
    test_bias_independence();
    test_no_bias_equivalence();
    test_padding_only_contributions();

    if (g_failures == 0) {
        std::printf("OK: all conv2d_weight_backward tests passed\n");
        return 0;
    }
    std::printf("FAILED: %d conv2d_weight_backward check(s)\n", g_failures);
    return 1;
}
