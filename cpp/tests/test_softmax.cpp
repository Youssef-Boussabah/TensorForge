// Dependency-free C++ test for the fused native softmax (Phase E,
// milestone E3). No GoogleTest/Catch2 — a plain executable that prints
// failures and returns a nonzero exit code if any check fails, so CTest
// reports pass/fail.
//
// This binary compiles cpp/src/classification.cpp (and error.cpp)
// directly, so it exercises **both** layers of that translation unit, the
// way test_maxpool2d_backward.cpp does:
//
//   * tf::softmax_forward_contiguous — the internal, hidden compute
//     kernel (valid, pre-validated arguments only: it is the math, not a
//     validation boundary); and
//   * tf_core_softmax_forward — the exported guarded C ABI wrapper,
//     where every trust-boundary argument must be rejected, driven here
//     through plain tf::Storage handles and the thread-local error slot.
//
// The reference values are computed with the SAME maximum-shift order
// the kernel uses, not borrowed from another framework: the point is
// that the fused kernel agrees with the algorithm it claims to
// implement, including at IEEE edges where frameworks differ.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <vector>

#include "tf_classification_internal.h"
#include "tf_internal.h"  // TF_EXPORT, tf::Storage, error accessors, TfStatus

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

// The same maximum-shift algorithm, written independently here so a
// mistake in the kernel does not silently define its own reference.
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
            double total = 0.0;
            for (int64_t k = 0; k < axis_length; ++k) {
                const double e =
                    std::exp(src[static_cast<size_t>(base + k * inner)] - maximum);
                out[static_cast<size_t>(base + k * inner)] = e;
                total += e;
            }
            for (int64_t k = 0; k < axis_length; ++k) {
                out[static_cast<size_t>(base + k * inner)] /= total;
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
    tf::softmax_forward_contiguous(src.data(), dst.data(), outer, axis_length,
                                   inner);
    return dst;
}

int run_checked(const std::vector<double>& src, int64_t src_offset,
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

// Every slice of a finite input must sum to 1 within tolerance.
void expect_slices_sum_to_one(const std::vector<double>& out,
                              int64_t outer, int64_t axis_length,
                              int64_t inner, const char* what) {
    for (int64_t o = 0; o < outer; ++o) {
        for (int64_t i = 0; i < inner; ++i) {
            const int64_t base = o * axis_length * inner + i;
            double total = 0.0;
            for (int64_t k = 0; k < axis_length; ++k) {
                const double p = out[static_cast<size_t>(base + k * inner)];
                check(p >= 0.0, "probability is non-negative");
                total += p;
            }
            char label[128];
            std::snprintf(label, sizeof(label), "%s slice(%lld,%lld) sums to 1",
                          what, static_cast<long long>(o),
                          static_cast<long long>(i));
            check_close(total, 1.0, 1e-12, label);
        }
    }
}

// -- finite correctness ------------------------------------------------

// 1. axis_length == 1: the only member of each slice takes all the mass.
void test_axis_length_one() {
    std::vector<double> src = {5.0, -3.0, 100.0};
    auto out = run_kernel(src, 3, 1, 1);
    for (size_t i = 0; i < out.size(); ++i) {
        char label[64];
        std::snprintf(label, sizeof(label), "axis_length_one[%zu]", i);
        check_eq(out[i], 1.0, label);
    }
}

// 2. A simple vector against the reference and against hand values.
void test_simple_vector() {
    std::vector<double> src = {1.0, 2.0, 3.0};
    expect_matches_reference(src, 1, 3, 1, "simple_vector");
    auto out = run_kernel(src, 1, 3, 1);
    expect_slices_sum_to_one(out, 1, 3, 1, "simple_vector");
    // e^-2 : e^-1 : e^0 normalized.
    const double d = std::exp(-2.0) + std::exp(-1.0) + 1.0;
    check_close(out[0], std::exp(-2.0) / d, 1e-15, "simple_vector[0]");
    check_close(out[2], 1.0 / d, 1e-15, "simple_vector[2]");
    check(out[0] < out[1] && out[1] < out[2], "softmax is order preserving");
}

// 3. Equal values give a uniform distribution, exactly.
void test_equal_values_are_uniform() {
    std::vector<double> src(8, 3.25);
    auto out = run_kernel(src, 2, 4, 1);
    for (size_t i = 0; i < out.size(); ++i) {
        char label[64];
        std::snprintf(label, sizeof(label), "equal_values[%zu]", i);
        check_eq(out[i], 0.25, label);
    }
}

// 4/5. A large common offset must not overflow, and must not change the
// result — the whole point of the maximum shift. A naive
// exp-then-normalize would give inf/inf == NaN at +700 and 0/0 == NaN at
// -700; the shift keeps both exact.
void test_large_common_offsets() {
    const std::vector<double> base = {0.0, 1.0, 2.0, 3.0};
    const std::vector<double> want = run_kernel(base, 1, 4, 1);
    // Offsets where base + offset is still exactly representable, so the
    // *input* differences survive and invariance is a real claim.
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
        expect_slices_sum_to_one(out, 1, 4, 1, "large_offset");
    }
    // At an absurd magnitude the offset swallows the differences in the
    // *input* (1e300 + 3.0 == 1e300 in float64), so the slice really is
    // uniform. The honest claim there is that the kernel still returns a
    // finite, normalized distribution rather than inf/inf or 0/0.
    for (double offset : {1e300, -1e300}) {
        std::vector<double> saturated(4, offset);
        auto out = run_kernel(saturated, 1, 4, 1);
        for (size_t i = 0; i < out.size(); ++i) {
            char label[128];
            std::snprintf(label, sizeof(label), "saturated %.3g[%zu]", offset, i);
            check_eq(out[i], 0.25, label);
        }
        expect_slices_sum_to_one(out, 1, 4, 1, "saturated_offset");
    }
}

// 6. Mixed magnitudes and signs.
void test_mixed_values() {
    std::vector<double> src = {-50.0, 0.0, 12.5, -3.25, 7.0, 1.0};
    expect_matches_reference(src, 1, 6, 1, "mixed_values");
    auto out = run_kernel(src, 1, 6, 1);
    expect_slices_sum_to_one(out, 1, 6, 1, "mixed_values");
    check(out[2] > out[4], "larger logit gets more mass");
    check(out[0] > 0.0, "a very small logit still gets positive mass");
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
    expect_slices_sum_to_one(run_kernel(src, 1, 2, 12), 1, 2, 12, "axis_first");
    // axis 1: outer=2, axis_length=3, inner=4
    expect_matches_reference(src, 2, 3, 4, "axis_middle");
    expect_slices_sum_to_one(run_kernel(src, 2, 3, 4), 2, 3, 4, "axis_middle");
    // axis 2: outer=6, axis_length=4, inner=1
    expect_matches_reference(src, 6, 4, 1, "axis_last");
    expect_slices_sum_to_one(run_kernel(src, 6, 4, 1), 6, 4, 1, "axis_last");
    // outer == 1 and inner == 1 degenerate cases over the whole buffer.
    expect_matches_reference(src, 1, 24, 1, "single_slice");
    expect_matches_reference(src, 24, 1, 1, "all_slices_length_one");
}

// 10. Slices are independent: changing one must not disturb another.
void test_slices_are_independent() {
    std::vector<double> src = {1.0, 2.0,   0.0, 0.0};   // outer=2, len=2, inner=1
    auto first = run_kernel(src, 2, 2, 1);
    src[2] = 5.0;                                       // perturb slice 1 only
    auto second = run_kernel(src, 2, 2, 1);
    check_eq(second[0], first[0], "independent slice[0]");
    check_eq(second[1], first[1], "independent slice[1]");
    check(second[2] != first[2], "perturbed slice changed");
    expect_slices_sum_to_one(second, 2, 2, 1, "independent");
}

// 11. A nonzero source offset through the checked wrapper.
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

// 12. Determinism and non-mutation; the destination is fully written.
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

// -- exceptional values (plain IEEE, no special-casing) ----------------

// 13. NaN poisons its own slice and only its own slice.
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
    // The kernel agrees with the same-order reference at these edges too.
    expect_matches_reference(src, 2, 3, 1, "nan_reference");
}

// 14. +inf makes its slice NaN (inf - inf), -inf simply gets zero mass.
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
    check_eq(neg[0], 0.0, "neg_inf member gets zero mass");
    check(std::isfinite(neg[1]) && std::isfinite(neg[2]),
          "the rest of the slice stays finite");
    check_close(neg[1] + neg[2], 1.0, 1e-15, "neg_inf slice still sums to 1");
    std::vector<double> all_neg_inf = {kNegInf, kNegInf};
    auto all_neg = run_kernel(all_neg_inf, 1, 2, 1);
    for (size_t i = 0; i < 2; ++i) {
        char label[64];
        std::snprintf(label, sizeof(label), "all_neg_inf[%zu] is NaN", i);
        check(std::isnan(all_neg[i]), label);  // -inf - (-inf) == NaN
    }
    expect_matches_reference(with_pos_inf, 1, 3, 1, "pos_inf_reference");
    expect_matches_reference(with_neg_inf, 1, 3, 1, "neg_inf_reference");
}

// 15. A numerically exceptional but structurally valid call is NOT an
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

// 16. Null handles.
void test_rejects_null_handles() {
    std::vector<double> src = {1.0, 2.0};
    std::vector<double> dst(2, kPoison);
    tf::Storage in{src.data(), 2};
    tf::Storage out{dst.data(), 2};
    tf_core_softmax_forward(nullptr, 0, &out, 1, 2, 1);
    expect_rejected(tf::last_error_code(), dst, "null source");
    tf_core_softmax_forward(&in, 0, nullptr, 1, 2, 1);
    check(tf::last_error_code() == TF_ERROR_INVALID, "null destination");
    tf::clear_error();
}

// 17. Non-positive and negative dimensional factors, and a negative
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

// 18. Spans and capacities.
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

// 19. Overflow in the dimension product and in offset + numel.
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

// 20. The guard clears the slot on entry, so neither a previous
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
    check_close(dst[0] + dst[1], 1.0, 1e-15, "post-failure result normalized");
}

}  // namespace

int main() {
    test_axis_length_one();
    test_simple_vector();
    test_equal_values_are_uniform();
    test_large_common_offsets();
    test_mixed_values();
    test_axis_decompositions();
    test_slices_are_independent();
    test_checked_nonzero_offset();
    test_determinism_and_non_mutation();
    test_nan_propagates_within_its_slice();
    test_infinities();
    test_exceptional_values_leave_status_ok();
    test_rejects_null_handles();
    test_rejects_malformed_dimensions();
    test_rejects_bad_spans();
    test_rejects_overflow();
    test_guard_clears_previous_error();

    if (g_failures == 0) {
        std::printf("OK: all softmax tests passed\n");
        return 0;
    }
    std::printf("FAILED: %d softmax check(s)\n", g_failures);
    return 1;
}
