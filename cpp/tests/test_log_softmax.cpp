// Dependency-free C++ test for the fused native log-softmax (Phase E,
// milestone E4). No GoogleTest/Catch2 — a plain executable that prints
// failures and returns a nonzero exit code if any check fails, so CTest
// reports pass/fail.
//
// This binary compiles cpp/src/classification.cpp (and error.cpp)
// directly, so it exercises **both** layers of that translation unit, the
// way test_softmax.cpp does:
//
//   * tf::log_softmax_forward_contiguous — the internal, hidden compute
//     kernel (valid, pre-validated arguments only: it is the math, not a
//     validation boundary); and
//   * tf_core_log_softmax_forward — the exported guarded C ABI wrapper,
//     where every trust-boundary argument must be rejected, driven here
//     through plain tf::Storage handles and the thread-local error slot.
//
// E4 factored the two exports' identical precondition checks into one
// file-local validator, so this binary also re-drives the **softmax**
// rejection matrix and a softmax numerical spot-check as regression
// coverage that the sharing changed nothing about E3's behavior.
//
// The reference values are computed with the SAME maximum-shift /
// log-sum-exp order the kernel uses, not borrowed from another
// framework: the point is that the fused kernel agrees with the
// algorithm it claims to implement, including at IEEE edges where
// frameworks differ.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <vector>

#include "tf_classification_internal.h"
#include "tf_internal.h"  // TF_EXPORT, tf::Storage, error accessors, TfStatus

TF_EXPORT void tf_core_log_softmax_forward(
    const void* src_handle, int64_t src_offset,
    void* dst_handle,
    int64_t outer, int64_t axis_length, int64_t inner);

TF_EXPORT void tf_core_softmax_forward(
    const void* src_handle, int64_t src_offset,
    void* dst_handle,
    int64_t outer, int64_t axis_length, int64_t inner);

namespace {

int g_failures = 0;

const double kPosInf = std::numeric_limits<double>::infinity();
const double kNegInf = -std::numeric_limits<double>::infinity();
const double kNaN = std::numeric_limits<double>::quiet_NaN();
const double kPoison = 7777.5;

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

// The same maximum-shift / log-sum-exp algorithm, written independently
// here so a mistake in the kernel does not silently define its own
// reference. Note what it is NOT: log(softmax(x)).
std::vector<double> reference(const std::vector<double>& src,
                              int64_t outer, int64_t axis_length,
                              int64_t inner) {
    std::vector<double> out(src.size(), 0.0);
    for (int64_t o = 0; o < outer; ++o) {
        for (int64_t i = 0; i < inner; ++i) {
            const int64_t base = o * axis_length * inner + i;
            double maximum = src[static_cast<size_t>(base)];
            for (int64_t k = 1; k < axis_length; ++k) {
                const double value = src[static_cast<size_t>(base + k * inner)];
                if (value > maximum) {
                    maximum = value;
                }
            }
            double sum_exp = 0.0;
            for (int64_t k = 0; k < axis_length; ++k) {
                const double shifted =
                    src[static_cast<size_t>(base + k * inner)] - maximum;
                out[static_cast<size_t>(base + k * inner)] = shifted;
                sum_exp += std::exp(shifted);
            }
            const double log_denominator = std::log(sum_exp);
            for (int64_t k = 0; k < axis_length; ++k) {
                out[static_cast<size_t>(base + k * inner)] -= log_denominator;
            }
        }
    }
    return out;
}

// -- drivers ----------------------------------------------------------

std::vector<double> run_kernel(const std::vector<double>& src,
                               int64_t outer, int64_t axis_length,
                               int64_t inner) {
    std::vector<double> dst(src.size(), kPoison);
    tf::log_softmax_forward_contiguous(src.data(), dst.data(), outer,
                                       axis_length, inner);
    return dst;
}

int run_checked(const std::vector<double>& src, int64_t src_offset,
                std::vector<double>& dst,
                int64_t outer, int64_t axis_length, int64_t inner) {
    tf::Storage in{const_cast<double*>(src.data()),
                   static_cast<int64_t>(src.size())};
    tf::Storage out{dst.data(), static_cast<int64_t>(dst.size())};
    tf_core_log_softmax_forward(&in, src_offset, &out, outer, axis_length,
                                inner);
    return tf::last_error_code();
}

int run_checked_softmax(const std::vector<double>& src, int64_t src_offset,
                        std::vector<double>& dst,
                        int64_t outer, int64_t axis_length, int64_t inner) {
    tf::Storage in{const_cast<double*>(src.data()),
                   static_cast<int64_t>(src.size())};
    tf::Storage out{dst.data(), static_cast<int64_t>(dst.size())};
    tf_core_softmax_forward(&in, src_offset, &out, outer, axis_length, inner);
    return tf::last_error_code();
}

// Bit-for-bit equality that also accepts NaN == NaN, so exceptional
// slices can be compared against the same-order reference.
void check_eq_or_both_nan(double got, double want, const char* what) {
    if (std::isnan(got) && std::isnan(want)) {
        return;
    }
    check_eq(got, want, what);
}

void expect_matches_reference(const std::vector<double>& src,
                              int64_t outer, int64_t axis_length,
                              int64_t inner, const char* what) {
    const std::vector<double> got = run_kernel(src, outer, axis_length, inner);
    const std::vector<double> want = reference(src, outer, axis_length, inner);
    for (size_t i = 0; i < want.size(); ++i) {
        char label[128];
        std::snprintf(label, sizeof(label), "%s[%zu]", what, i);
        check_eq_or_both_nan(got[i], want[i], label);
    }
}

// exp() of every log-probability in a finite slice must sum to 1, and
// every log-probability must be <= 0 (up to rounding).
void expect_exp_slices_sum_to_one(const std::vector<double>& out,
                                  int64_t outer, int64_t axis_length,
                                  int64_t inner, const char* what) {
    for (int64_t o = 0; o < outer; ++o) {
        for (int64_t i = 0; i < inner; ++i) {
            const int64_t base = o * axis_length * inner + i;
            double total = 0.0;
            for (int64_t k = 0; k < axis_length; ++k) {
                const double value = out[static_cast<size_t>(base + k * inner)];
                check(value <= 1e-15, "a log-probability is non-positive");
                total += std::exp(value);
            }
            char label[160];
            std::snprintf(label, sizeof(label),
                          "%s exp(slice(%lld,%lld)) sums to 1", what,
                          static_cast<long long>(o),
                          static_cast<long long>(i));
            check_close(total, 1.0, 1e-12, label);
        }
    }
}

// -- finite correctness ------------------------------------------------

// 1. axis_length == 1: the only member of each slice takes all the mass,
// so its log-probability is exactly zero.
void test_axis_length_one() {
    std::vector<double> src = {5.0, -3.0, 100.0};
    auto out = run_kernel(src, 3, 1, 1);
    for (size_t i = 0; i < out.size(); ++i) {
        char label[64];
        std::snprintf(label, sizeof(label), "axis_length_one[%zu]", i);
        check_eq(out[i], 0.0, label);
    }
}

// 2. A simple vector against the reference and against hand values.
void test_simple_vector() {
    std::vector<double> src = {1.0, 2.0, 3.0};
    expect_matches_reference(src, 1, 3, 1, "simple_vector");
    auto out = run_kernel(src, 1, 3, 1);
    expect_exp_slices_sum_to_one(out, 1, 3, 1, "simple_vector");
    // y_i = x_i - log(sum_j exp(x_j)) with the shift applied by hand.
    const double denominator = std::exp(-2.0) + std::exp(-1.0) + 1.0;
    check_close(out[0], -2.0 - std::log(denominator), 1e-15,
                "simple_vector[0]");
    check_close(out[2], 0.0 - std::log(denominator), 1e-15,
                "simple_vector[2]");
    check(out[0] < out[1] && out[1] < out[2],
          "log_softmax is order preserving");
}

// 3. Equal values give -log(axis_length) everywhere, exactly.
void test_equal_values() {
    std::vector<double> src(8, 3.25);
    auto out = run_kernel(src, 2, 4, 1);
    for (size_t i = 0; i < out.size(); ++i) {
        char label[64];
        std::snprintf(label, sizeof(label), "equal_values[%zu]", i);
        check_eq(out[i], -std::log(4.0), label);
    }
}

// 4/5. A large common offset must not overflow and must not change the
// result — the whole point of the maximum shift. A naive
// log(sum(exp(x))) would give inf at +700 and -inf at -700.
void test_large_common_offsets() {
    const std::vector<double> base = {0.0, 1.0, 2.0, 3.0};
    const std::vector<double> want = run_kernel(base, 1, 4, 1);
    for (double offset : {700.0, -700.0, 1e5, -1e5, 1e10, -1e10}) {
        std::vector<double> shifted;
        for (double value : base) {
            shifted.push_back(value + offset);
        }
        auto out = run_kernel(shifted, 1, 4, 1);
        for (size_t i = 0; i < out.size(); ++i) {
            char label[128];
            std::snprintf(label, sizeof(label), "offset %.3g invariance[%zu]",
                          offset, i);
            // The addition itself rounds slightly at 1e10, so compare to
            // a tolerance rather than exactly.
            check_close(out[i], want[i], 1e-6, label);
            check(std::isfinite(out[i]), "offset output stays finite");
        }
        expect_exp_slices_sum_to_one(out, 1, 4, 1, "large_offset");
    }
    // At an absurd magnitude the offset swallows the differences in the
    // *input* (1e300 + 3.0 == 1e300 in float64), so the slice really is
    // uniform; the honest claim is that the kernel still returns finite,
    // normalized log-probabilities rather than inf or NaN.
    for (double offset : {1e300, -1e300}) {
        std::vector<double> saturated(4, offset);
        auto out = run_kernel(saturated, 1, 4, 1);
        for (size_t i = 0; i < out.size(); ++i) {
            char label[128];
            std::snprintf(label, sizeof(label), "saturated %.3g[%zu]", offset, i);
            check_eq(out[i], -std::log(4.0), label);
        }
        expect_exp_slices_sum_to_one(out, 1, 4, 1, "saturated_offset");
    }
}

// 6. Mixed magnitudes and signs, including the small-probability regime
// where the composed log(softmax(x)) form loses precision outright.
void test_mixed_values() {
    std::vector<double> src = {-50.0, 0.0, 12.5, -3.25, 7.0, 1.0};
    expect_matches_reference(src, 1, 6, 1, "mixed_values");
    auto out = run_kernel(src, 1, 6, 1);
    expect_exp_slices_sum_to_one(out, 1, 6, 1, "mixed_values");
    check(out[2] > out[4], "larger logit gets a larger log-probability");
    // A logit 800 below the maximum has probability ~e^-800, which
    // underflows float64 to exactly 0 — log(softmax(x)) would report
    // -inf there. The fused form reports a finite, accurate value.
    std::vector<double> extreme = {0.0, -800.0};
    auto tiny = run_kernel(extreme, 1, 2, 1);
    check(std::isfinite(tiny[1]),
          "a tiny probability keeps a finite log-probability");
    check_close(tiny[1], -800.0, 1e-9, "small-probability regime is accurate");
    // ...and the composed form really would have failed here.
    check(std::exp(-800.0) == 0.0, "the composed form would underflow to 0");
    check(std::log(std::exp(-800.0)) == kNegInf,
          "log(softmax(x)) would give -inf");
}

// 7-9. The three axis positions of a (2, 3, 4) tensor, each expressed as
// its own (outer, axis_length, inner) decomposition over the same buffer.
void test_axis_decompositions() {
    std::vector<double> src;
    for (int i = 0; i < 24; ++i) {
        src.push_back(0.25 * i - 3.0);
    }
    // axis 0: outer=1, axis_length=2, inner=12
    expect_matches_reference(src, 1, 2, 12, "axis_first");
    expect_exp_slices_sum_to_one(run_kernel(src, 1, 2, 12), 1, 2, 12,
                                 "axis_first");
    // axis 1: outer=2, axis_length=3, inner=4
    expect_matches_reference(src, 2, 3, 4, "axis_middle");
    expect_exp_slices_sum_to_one(run_kernel(src, 2, 3, 4), 2, 3, 4,
                                 "axis_middle");
    // axis 2: outer=6, axis_length=4, inner=1
    expect_matches_reference(src, 6, 4, 1, "axis_last");
    expect_exp_slices_sum_to_one(run_kernel(src, 6, 4, 1), 6, 4, 1,
                                 "axis_last");
    // outer == 1 and inner == 1 degenerate cases over the whole buffer.
    expect_matches_reference(src, 1, 24, 1, "single_slice");
    expect_matches_reference(src, 24, 1, 1, "all_slices_length_one");
}

// 10. Slices are independent: changing one must not disturb another.
void test_slices_are_independent() {
    std::vector<double> src = {1.0, 2.0,   0.0, 0.0};  // outer=2, len=2, inner=1
    auto first = run_kernel(src, 2, 2, 1);
    src[2] = 5.0;                                      // perturb slice 1 only
    auto second = run_kernel(src, 2, 2, 1);
    check_eq(second[0], first[0], "independent slice[0]");
    check_eq(second[1], first[1], "independent slice[1]");
    check(second[2] != first[2], "perturbed slice changed");
    expect_exp_slices_sum_to_one(second, 2, 2, 1, "independent");
}

// 11. An additive shift of one whole slice leaves that slice's result
// unchanged (log-softmax is shift invariant, like softmax).
void test_additive_shift_invariance() {
    const std::vector<double> base = {0.5, -1.5, 2.25};
    const auto want = run_kernel(base, 1, 3, 1);
    std::vector<double> shifted;
    for (double value : base) {
        shifted.push_back(value + 37.5);
    }
    const auto got = run_kernel(shifted, 1, 3, 1);
    for (size_t i = 0; i < want.size(); ++i) {
        char label[64];
        std::snprintf(label, sizeof(label), "shift_invariance[%zu]", i);
        check_close(got[i], want[i], 1e-13, label);
    }
}

// 12. A nonzero source offset through the checked wrapper.
void test_checked_nonzero_offset() {
    std::vector<double> src = {99.0, 99.0, 1.0, 2.0, 3.0};
    std::vector<double> dst(3, kPoison);
    check(run_checked(src, 2, dst, 1, 3, 1) == TF_OK, "offset status");
    const std::vector<double> tail = {1.0, 2.0, 3.0};
    const std::vector<double> want = reference(tail, 1, 3, 1);
    for (size_t i = 0; i < want.size(); ++i) {
        char label[64];
        std::snprintf(label, sizeof(label), "checked_offset[%zu]", i);
        check_eq(dst[i], want[i], label);
    }
    check_eq(src[0], 99.0, "offset call left the prefix untouched");
}

// 13. Determinism and non-mutation; the destination is fully written.
void test_determinism_and_non_mutation() {
    const std::vector<double> original = {0.5, -1.5, 2.25, 4.0, -0.75, 3.5};
    std::vector<double> src = original;
    auto first = run_kernel(src, 2, 3, 1);
    auto second = run_kernel(src, 2, 3, 1);
    for (size_t i = 0; i < first.size(); ++i) {
        char label[64];
        std::snprintf(label, sizeof(label), "determinism[%zu]", i);
        check_eq(first[i], second[i], label);
        std::snprintf(label, sizeof(label), "input_unmodified[%zu]", i);
        check_eq(src[i], original[i], label);
        std::snprintf(label, sizeof(label), "output_written[%zu]", i);
        check(first[i] != kPoison, label);
    }
}

// 14. exp(log_softmax(x)) agrees with softmax(x) to tolerance. Compared
// against the softmax *export* running over the same buffer, so this is
// a genuine cross-check of two independent kernels rather than of one
// implementation with itself. Bit equality is NOT claimed: the two
// kernels do different arithmetic.
void test_agrees_with_softmax() {
    const std::vector<double> src = {-2.0, 0.5, 3.25, 1.0, -0.75, 2.5};
    const auto logs = run_kernel(src, 2, 3, 1);
    std::vector<double> probabilities(src.size(), kPoison);
    check(run_checked_softmax(src, 0, probabilities, 2, 3, 1) == TF_OK,
          "softmax regression call succeeded");
    for (size_t i = 0; i < src.size(); ++i) {
        char label[96];
        std::snprintf(label, sizeof(label), "exp(log_softmax) == softmax[%zu]",
                      i);
        check_close(std::exp(logs[i]), probabilities[i], 1e-15, label);
    }
}

// -- exceptional values (plain IEEE, no special-casing) ----------------

// 15. NaN poisons its own slice and only its own slice.
void test_nan_propagates_within_its_slice() {
    std::vector<double> src = {1.0, kNaN, 2.0,   1.0, 2.0, 3.0};
    auto out = run_kernel(src, 2, 3, 1);
    for (size_t i = 0; i < 3; ++i) {
        char label[64];
        std::snprintf(label, sizeof(label), "nan_slice[%zu] is NaN", i);
        check(std::isnan(out[i]), label);
    }
    const std::vector<double> clean = {1.0, 2.0, 3.0};
    const std::vector<double> want = reference(clean, 1, 3, 1);
    for (size_t i = 0; i < 3; ++i) {
        char label[64];
        std::snprintf(label, sizeof(label), "clean_slice[%zu] unaffected", i);
        check_eq(out[3 + i], want[i], label);
    }
    expect_matches_reference(src, 2, 3, 1, "nan_reference");
}

// 16. +inf makes its slice NaN (inf - inf); -inf mixed with finite
// values gives -inf at that position while the finite positions stay
// governed by the stable computation; an all -inf slice is NaN.
void test_infinities() {
    std::vector<double> with_pos_inf = {kPosInf, 1.0, 2.0};
    auto pos = run_kernel(with_pos_inf, 1, 3, 1);
    for (size_t i = 0; i < 3; ++i) {
        char label[64];
        std::snprintf(label, sizeof(label), "pos_inf[%zu] is NaN", i);
        check(std::isnan(pos[i]), label);
    }

    std::vector<double> with_neg_inf = {kNegInf, 1.0, 2.0};
    auto neg = run_kernel(with_neg_inf, 1, 3, 1);
    check_eq(neg[0], kNegInf, "neg_inf member has log-probability -inf");
    check(std::isfinite(neg[1]) && std::isfinite(neg[2]),
          "the rest of the slice stays finite");
    check_close(std::exp(neg[1]) + std::exp(neg[2]), 1.0, 1e-15,
                "the finite part of a -inf slice still normalizes to 1");
    // The finite members match the stable computation over the finite
    // sub-slice exactly — -inf contributes exp(-inf) == 0 to the sum.
    const std::vector<double> finite_only = {1.0, 2.0};
    const auto finite_reference = reference(finite_only, 1, 2, 1);
    check_eq(neg[1], finite_reference[0], "finite member unaffected by -inf");
    check_eq(neg[2], finite_reference[1], "finite member unaffected by -inf");

    std::vector<double> all_neg_inf = {kNegInf, kNegInf};
    auto all_neg = run_kernel(all_neg_inf, 1, 2, 1);
    for (size_t i = 0; i < 2; ++i) {
        char label[64];
        std::snprintf(label, sizeof(label), "all_neg_inf[%zu] is NaN", i);
        check(std::isnan(all_neg[i]), label);  // -inf - (-inf) == NaN
    }
    expect_matches_reference(with_pos_inf, 1, 3, 1, "pos_inf_reference");
    expect_matches_reference(with_neg_inf, 1, 3, 1, "neg_inf_reference");
    expect_matches_reference(all_neg_inf, 1, 2, 1, "all_neg_inf_reference");
}

// 17. A numerically exceptional but structurally valid call is NOT an
// ABI failure: the error slot stays TF_OK.
void test_exceptional_values_leave_status_ok() {
    std::vector<double> src = {kNaN, kPosInf, kNegInf};
    std::vector<double> dst(3, kPoison);
    const int status = run_checked(src, 0, dst, 1, 3, 1);
    check(status == TF_OK, "exceptional values leave the error slot clear");
    for (size_t i = 0; i < 3; ++i) {
        check(dst[i] != kPoison, "destination still fully written");
    }
}

// -- validation at the trust boundary ----------------------------------

void expect_rejected(int status, const std::vector<double>& dst,
                     const char* what) {
    char label[160];
    std::snprintf(label, sizeof(label), "%s rejected with TF_ERROR_INVALID",
                  what);
    check(status == TF_ERROR_INVALID, label);
    for (size_t i = 0; i < dst.size(); ++i) {
        std::snprintf(label, sizeof(label), "%s left dst[%zu] untouched",
                      what, i);
        check_eq(dst[i], kPoison, label);
    }
    tf::clear_error();
}

// 18. Null handles.
void test_rejects_null_handles() {
    std::vector<double> src = {1.0, 2.0};
    std::vector<double> dst(2, kPoison);
    tf::Storage in{src.data(), 2};
    tf::Storage out{dst.data(), 2};
    tf_core_log_softmax_forward(nullptr, 0, &out, 1, 2, 1);
    expect_rejected(tf::last_error_code(), dst, "null source");
    tf_core_log_softmax_forward(&in, 0, nullptr, 1, 2, 1);
    check(tf::last_error_code() == TF_ERROR_INVALID, "null destination");
    tf::clear_error();
}

// 19. Non-positive and negative dimensional factors, and a negative
// offset.
void test_rejects_malformed_dimensions() {
    std::vector<double> src = {1.0, 2.0, 3.0, 4.0};
    std::vector<double> dst(4, kPoison);
    check(run_checked(src, 0, dst, 2, 2, 1) == TF_OK, "sane baseline");
    std::fill(dst.begin(), dst.end(), kPoison);

    expect_rejected(run_checked(src, 0, dst, 0, 2, 1), dst, "outer == 0");
    expect_rejected(run_checked(src, 0, dst, 2, 0, 1), dst, "axis_length == 0");
    expect_rejected(run_checked(src, 0, dst, 2, 2, 0), dst, "inner == 0");
    expect_rejected(run_checked(src, 0, dst, -1, 2, 1), dst, "negative outer");
    expect_rejected(run_checked(src, 0, dst, 2, -2, 1), dst,
                    "negative axis_length");
    expect_rejected(run_checked(src, 0, dst, 2, 2, -1), dst, "negative inner");
    expect_rejected(run_checked(src, -1, dst, 2, 2, 1), dst, "negative offset");
}

// 20. Spans and capacities.
void test_rejects_bad_spans() {
    std::vector<double> src = {1.0, 2.0, 3.0, 4.0};
    std::vector<double> dst(4, kPoison);
    // numel 6 > source capacity 4.
    expect_rejected(run_checked(src, 0, dst, 2, 3, 1), dst,
                    "source span too short");
    // offset pushes the run past the end.
    expect_rejected(run_checked(src, 2, dst, 1, 4, 1), dst,
                    "offset + numel past the source end");
    // A destination smaller than numel.
    std::vector<double> small(2, kPoison);
    expect_rejected(run_checked(src, 0, small, 2, 2, 1), small,
                    "destination capacity too short");
}

// 21. Overflow in the dimension product and in offset + numel.
void test_rejects_overflow() {
    std::vector<double> src = {1.0, 2.0};
    std::vector<double> dst(2, kPoison);
    const int64_t huge = INT64_MAX / 2 + 4;
    expect_rejected(run_checked(src, 0, dst, huge, huge, 1), dst,
                    "outer * axis_length overflow");
    expect_rejected(run_checked(src, 0, dst, huge, 1, huge), dst,
                    "product * inner overflow");
    expect_rejected(run_checked(src, INT64_MAX, dst, 1, 2, 1), dst,
                    "offset + numel overflow");
}

// 22. The guard clears the slot on entry, so neither a previous
// rejection nor an unrelated stale error contaminates a later call.
void test_guard_clears_previous_error() {
    std::vector<double> src = {1.0, 2.0};
    std::vector<double> dst(2, kPoison);
    expect_rejected(run_checked(src, 0, dst, 5, 5, 5), dst, "priming failure");
    tf::set_error(TF_ERROR_RUNTIME, "stale error from somewhere else");
    check(run_checked(src, 0, dst, 1, 2, 1) == TF_OK,
          "a valid call clears the stale error");
    const std::vector<double> want = reference(src, 1, 2, 1);
    check_eq(dst[0], want[0], "post-failure call still computes");
    check_close(std::exp(dst[0]) + std::exp(dst[1]), 1.0, 1e-15,
                "post-failure result normalized");
}

// -- regression: the shared validator did not change softmax -----------

// 23. E4 factored the two exports' preconditions into one file-local
// validator. Re-drive softmax's whole rejection matrix (and one numeric
// call) through the exported wrapper to prove that sharing weakened
// neither the checks nor the messages' effect.
void test_shared_validation_leaves_softmax_unchanged() {
    std::vector<double> src = {1.0, 2.0, 3.0, 4.0};
    std::vector<double> dst(4, kPoison);
    tf::Storage in{src.data(), 4};
    tf::Storage out{dst.data(), 4};

    tf_core_softmax_forward(nullptr, 0, &out, 2, 2, 1);
    expect_rejected(tf::last_error_code(), dst, "softmax null source");
    tf_core_softmax_forward(&in, 0, nullptr, 2, 2, 1);
    check(tf::last_error_code() == TF_ERROR_INVALID, "softmax null destination");
    tf::clear_error();

    expect_rejected(run_checked_softmax(src, 0, dst, 0, 2, 1), dst,
                    "softmax outer == 0");
    expect_rejected(run_checked_softmax(src, 0, dst, 2, 0, 1), dst,
                    "softmax axis_length == 0");
    expect_rejected(run_checked_softmax(src, 0, dst, 2, 2, 0), dst,
                    "softmax inner == 0");
    expect_rejected(run_checked_softmax(src, -1, dst, 2, 2, 1), dst,
                    "softmax negative offset");
    expect_rejected(run_checked_softmax(src, 0, dst, 2, 3, 1), dst,
                    "softmax source span too short");
    expect_rejected(run_checked_softmax(src, 2, dst, 1, 4, 1), dst,
                    "softmax offset + numel past the end");
    const int64_t huge = INT64_MAX / 2 + 4;
    expect_rejected(run_checked_softmax(src, 0, dst, huge, huge, 1), dst,
                    "softmax product overflow");
    expect_rejected(run_checked_softmax(src, INT64_MAX, dst, 1, 2, 1), dst,
                    "softmax offset + numel overflow");
    std::vector<double> small(2, kPoison);
    expect_rejected(run_checked_softmax(src, 0, small, 2, 2, 1), small,
                    "softmax destination capacity too short");

    // ...and softmax still computes exactly what E3 shipped.
    std::fill(dst.begin(), dst.end(), kPoison);
    check(run_checked_softmax(src, 0, dst, 1, 4, 1) == TF_OK,
          "softmax still succeeds on a valid call");
    double total = 0.0;
    for (size_t i = 0; i < dst.size(); ++i) {
        check(dst[i] > 0.0, "softmax probability is positive");
        total += dst[i];
    }
    check_close(total, 1.0, 1e-15, "softmax slice still sums to 1");
    check(dst[3] > dst[0], "softmax is still order preserving");
}

}  // namespace

int main() {
    test_axis_length_one();
    test_simple_vector();
    test_equal_values();
    test_large_common_offsets();
    test_mixed_values();
    test_axis_decompositions();
    test_slices_are_independent();
    test_additive_shift_invariance();
    test_checked_nonzero_offset();
    test_determinism_and_non_mutation();
    test_agrees_with_softmax();
    test_nan_propagates_within_its_slice();
    test_infinities();
    test_exceptional_values_leave_status_ok();
    test_rejects_null_handles();
    test_rejects_malformed_dimensions();
    test_rejects_bad_spans();
    test_rejects_overflow();
    test_guard_clears_previous_error();
    test_shared_validation_leaves_softmax_unchanged();

    if (g_failures == 0) {
        std::printf("OK: all log_softmax tests passed\n");
        return 0;
    }
    std::printf("FAILED: %d log_softmax check(s)\n", g_failures);
    return 1;
}
