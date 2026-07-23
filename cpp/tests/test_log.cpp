// Dependency-free C++ test for the native logarithm (Phase E,
// milestone E2). No GoogleTest/Catch2 — a plain executable that prints
// failures and returns a nonzero exit code if any check fails, so CTest
// reports pass/fail.
//
// This binary compiles cpp/src/elementwise.cpp (and error.cpp) directly
// (see the CMake TF_BUILD_TESTS target), mirroring test_exp.cpp: the
// logarithm's compute is file-local to that translation unit, so the
// surface under test is the **exported guarded C ABI pair**:
//
//   * tf_core_log             — the generic strided (odometer) export
//   * tf_core_log_contiguous  — the flat contiguous fast-path export
//
// Both are driven through plain tf::Storage handles plus the
// thread-local error slot. E2 reuses E1's validators unchanged, so the
// rejection cases below also serve as regression coverage that sharing
// them with a second operation did not weaken either one.
//
// The defining E2 nuance this file pins: std::log's domain results
// (-inf for zero, NaN for negatives) are **values**, not ABI errors —
// they land in the destination and leave the error slot clear.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <vector>

#include "tf_internal.h"  // TF_EXPORT, tf::Storage, error accessors, TfStatus

// The exports under test, declared with the same TF_EXPORT macro the
// definitions use so the linkage matches exactly.
TF_EXPORT void tf_core_log(
    const void* src, void* dst,
    const int64_t* shape, const int64_t* strides, int64_t offset,
    int64_t ndim);
TF_EXPORT void tf_core_log_contiguous(
    const void* src, void* dst, int64_t numel, int64_t offset);

namespace {

int g_failures = 0;

const double kPosInf = std::numeric_limits<double>::infinity();
const double kNegInf = -std::numeric_limits<double>::infinity();
const double kNaN = std::numeric_limits<double>::quiet_NaN();

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

void check_close(double got, double want, double rel, const char* what) {
    const double scale = std::fabs(want) > 1.0 ? std::fabs(want) : 1.0;
    if (!(std::fabs(got - want) <= rel * scale)) {
        std::printf("FAIL: %s (got %.17g, want %.17g, |diff|=%.3g)\n",
                    what, got, want, std::fabs(got - want));
        ++g_failures;
    }
}

// -- drivers ----------------------------------------------------------

int run_contiguous(const std::vector<double>& src, std::vector<double>& dst,
                   int64_t numel, int64_t offset) {
    tf::Storage in{const_cast<double*>(src.data()),
                   static_cast<int64_t>(src.size())};
    tf::Storage out{dst.data(), static_cast<int64_t>(dst.size())};
    tf_core_log_contiguous(&in, &out, numel, offset);
    return tf::last_error_code();
}

int run_strided(const std::vector<double>& src, std::vector<double>& dst,
                const std::vector<int64_t>& shape,
                const std::vector<int64_t>& strides, int64_t offset) {
    tf::Storage in{const_cast<double*>(src.data()),
                   static_cast<int64_t>(src.size())};
    tf::Storage out{dst.data(), static_cast<int64_t>(dst.size())};
    tf_core_log(&in, &out, shape.empty() ? nullptr : shape.data(),
                strides.empty() ? nullptr : strides.data(), offset,
                static_cast<int64_t>(shape.size()));
    return tf::last_error_code();
}

// -- forward value cases ----------------------------------------------

// 1. A single element through the contiguous path: log(1) is exactly 0.
void test_contiguous_single_element() {
    std::vector<double> src = {1.0};
    std::vector<double> dst(1, 7777.5);
    check(run_contiguous(src, dst, 1, 0) == TF_OK, "single-element status");
    check_eq(dst[0], 0.0, "log(1) == 0 exactly");
}

// 2. Several values at once, against std::log elementwise, plus two
// independent references.
void test_contiguous_values() {
    std::vector<double> src = {1.0, 2.0, 10.0, 0.5, 0.25, 100.0, 2.718281828459045};
    std::vector<double> dst(src.size(), 7777.5);
    check(run_contiguous(src, dst, static_cast<int64_t>(src.size()), 0) == TF_OK,
          "values status");
    for (size_t i = 0; i < src.size(); ++i) {
        char label[96];
        std::snprintf(label, sizeof(label), "contiguous_values[%zu]", i);
        check_eq(dst[i], std::log(src[i]), label);
    }
    check_close(dst[1], 0.6931471805599453, 1e-15, "log(2)");
    check_close(dst[6], 1.0, 1e-15, "log(e) == 1");
    // Values in (0, 1) are negative and finite.
    check(dst[3] < 0.0 && std::isfinite(dst[3]), "log(0.5) negative finite");
    check(dst[4] < dst[3], "log is monotonic below 1");
}

// 3. A nonzero offset starts the contiguous run later in the storage.
void test_contiguous_offset() {
    std::vector<double> src = {-99.0, -99.0, 1.0, 2.0, 4.0};
    std::vector<double> dst(3, 7777.5);
    check(run_contiguous(src, dst, 3, 2) == TF_OK, "offset status");
    check_eq(dst[0], 0.0, "offset[0]");
    check_eq(dst[1], std::log(2.0), "offset[1]");
    check_eq(dst[2], std::log(4.0), "offset[2]");
}

// 4. The strided export over a plain row-major 2x3 view must agree with
// the contiguous export bit-for-bit.
void test_strided_matches_contiguous() {
    std::vector<double> src = {1.0, 2.0, 3.0, 0.5, 0.75, 8.0};
    std::vector<double> flat(6, 7777.5);
    std::vector<double> strided(6, 7777.5);
    check(run_contiguous(src, flat, 6, 0) == TF_OK, "flat status");
    check(run_strided(src, strided, {2, 3}, {3, 1}, 0) == TF_OK,
          "row-major strided status");
    for (size_t i = 0; i < flat.size(); ++i) {
        char label[96];
        std::snprintf(label, sizeof(label), "strided_matches_contiguous[%zu]", i);
        check_eq(strided[i], flat[i], label);
    }
}

// 5. A genuinely non-contiguous view: the transpose of a 2x3 buffer,
// walked as 3x2 with strides (1, 3).
void test_strided_transposed() {
    std::vector<double> src = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
    std::vector<double> dst(6, 7777.5);
    check(run_strided(src, dst, {3, 2}, {1, 3}, 0) == TF_OK,
          "transposed status");
    const double want[6] = {std::log(1.0), std::log(4.0),
                            std::log(2.0), std::log(5.0),
                            std::log(3.0), std::log(6.0)};
    for (size_t i = 0; i < 6; ++i) {
        char label[96];
        std::snprintf(label, sizeof(label), "strided_transposed[%zu]", i);
        check_eq(dst[i], want[i], label);
    }
}

// 6. A strided view with a nonzero offset (the "narrow then transpose"
// layout the Core wrapper produces).
void test_strided_offset() {
    std::vector<double> src = {9.0, 0.5, 1.0, 9.0, 1.5, 2.0};
    std::vector<double> dst(4, 7777.5);
    check(run_strided(src, dst, {2, 2}, {1, 3}, 1) == TF_OK,
          "strided offset status");
    check_eq(dst[0], std::log(0.5), "strided_offset[0]");
    check_eq(dst[1], std::log(1.5), "strided_offset[1]");
    check_eq(dst[2], std::log(1.0), "strided_offset[2]");
    check_eq(dst[3], std::log(2.0), "strided_offset[3]");
}

// 7. Negative strides walk backwards from an interior offset. The raw
// ABI supports them (the span validator bounds both directions), even
// though no Python view currently produces one.
void test_strided_negative_stride() {
    std::vector<double> src = {1.0, 2.0, 4.0, 8.0};
    std::vector<double> dst(4, 7777.5);
    // shape (4,), stride -1, offset 3 -> reads 8, 4, 2, 1.
    check(run_strided(src, dst, {4}, {-1}, 3) == TF_OK,
          "negative-stride status");
    check_eq(dst[0], std::log(8.0), "negative_stride[0]");
    check_eq(dst[1], std::log(4.0), "negative_stride[1]");
    check_eq(dst[2], std::log(2.0), "negative_stride[2]");
    check_eq(dst[3], std::log(1.0), "negative_stride[3]");
    // Mixed directions across two axes: shape (2,2), strides (-2, 1),
    // offset 2 -> rows {4, 8} then {1, 2}.
    std::vector<double> mixed(4, 7777.5);
    check(run_strided(src, mixed, {2, 2}, {-2, 1}, 2) == TF_OK,
          "mixed-direction status");
    check_eq(mixed[0], std::log(4.0), "mixed_stride[0]");
    check_eq(mixed[1], std::log(8.0), "mixed_stride[1]");
    check_eq(mixed[2], std::log(1.0), "mixed_stride[2]");
    check_eq(mixed[3], std::log(2.0), "mixed_stride[3]");
}

// 8. Rank 0 (a scalar view) reads exactly one element, at the offset.
void test_strided_rank_zero() {
    std::vector<double> src = {5.0, 1.0};
    std::vector<double> dst(1, 7777.5);
    check(run_strided(src, dst, {}, {}, 1) == TF_OK, "rank-0 status");
    check_eq(dst[0], 0.0, "rank-0 reads src[offset]");
}

// -- IEEE domain behavior (values, never ABI errors) -------------------

// 9. Zero, negative, infinity, and NaN — unclamped, and the error slot
// stays clear because these are numerical results.
void test_domain_values_are_not_errors() {
    std::vector<double> src = {0.0, -0.0, -1.0, -0.5, kPosInf, kNaN, 1.0};
    std::vector<double> dst(src.size(), 7777.5);
    const int status =
        run_contiguous(src, dst, static_cast<int64_t>(src.size()), 0);
    check(status == TF_OK, "domain values leave the error slot clear");
    check_eq(dst[0], kNegInf, "log(+0) == -inf");
    check_eq(dst[1], kNegInf, "log(-0) == -inf");
    check(std::isnan(dst[2]), "log(-1) is NaN");
    check(std::isnan(dst[3]), "log(-0.5) is NaN");
    check_eq(dst[4], kPosInf, "log(+inf) == +inf");
    check(std::isnan(dst[5]), "log(NaN) is NaN");
    check_eq(dst[6], 0.0, "log(1) == 0");
    // The same values through the strided path.
    std::vector<double> other(src.size(), 7777.5);
    check(run_strided(src, other, {static_cast<int64_t>(src.size())}, {1}, 0)
              == TF_OK,
          "domain values strided status");
    check_eq(other[0], kNegInf, "strided log(+0)");
    check(std::isnan(other[2]), "strided log(-1)");
    check_eq(other[4], kPosInf, "strided log(+inf)");
    check(std::isnan(other[5]), "strided log(NaN)");
    // -inf input is a negative value: NaN, not -inf.
    std::vector<double> neg = {kNegInf};
    std::vector<double> neg_out(1, 7777.5);
    check(run_contiguous(neg, neg_out, 1, 0) == TF_OK, "log(-inf) status");
    check(std::isnan(neg_out[0]), "log(-inf) is NaN");
}

// 10. Very large and very small positive finite values still compute.
void test_extreme_finite_values() {
    std::vector<double> src = {1e-300, 1e-8, 1e8, 1e300};
    std::vector<double> dst(src.size(), 7777.5);
    check(run_contiguous(src, dst, static_cast<int64_t>(src.size()), 0) == TF_OK,
          "extreme status");
    for (size_t i = 0; i < src.size(); ++i) {
        char label[96];
        std::snprintf(label, sizeof(label), "extreme[%zu]", i);
        check_eq(dst[i], std::log(src[i]), label);
        check(std::isfinite(dst[i]), "extreme value stays finite");
    }
}

// -- ownership / non-mutation ------------------------------------------

// 11. The input storage is never written, and every destination slot is
// overwritten on success.
void test_input_unmodified_and_output_fully_written() {
    const std::vector<double> original = {1.0, 2.0, 0.5, 4.0};
    std::vector<double> src = original;
    std::vector<double> dst(4, 7777.5);
    check(run_contiguous(src, dst, 4, 0) == TF_OK, "unmodified status");
    for (size_t i = 0; i < original.size(); ++i) {
        char label[96];
        std::snprintf(label, sizeof(label), "input_unmodified[%zu]", i);
        check_eq(src[i], original[i], label);
        std::snprintf(label, sizeof(label), "output_written[%zu]", i);
        check(dst[i] != 7777.5, label);
    }
}

// 12. Determinism: identical bits on repeat, and the two paths agree.
void test_determinism() {
    std::vector<double> src = {0.125, 0.25, 3.5, 4.75, 7.0};
    std::vector<double> first(5, 0.0), second(5, 0.0), strided(5, 0.0);
    check(run_contiguous(src, first, 5, 0) == TF_OK, "determinism status a");
    check(run_contiguous(src, second, 5, 0) == TF_OK, "determinism status b");
    check(run_strided(src, strided, {5}, {1}, 0) == TF_OK,
          "determinism status c");
    for (size_t i = 0; i < 5; ++i) {
        char label[96];
        std::snprintf(label, sizeof(label), "determinism[%zu]", i);
        check_eq(first[i], second[i], label);
        std::snprintf(label, sizeof(label), "path_agreement[%zu]", i);
        check_eq(first[i], strided[i], label);
    }
}

// -- validation at the trust boundary ----------------------------------

// A rejected call must leave the destination exactly as it was.
void expect_rejected(int status, const std::vector<double>& dst,
                     const char* what) {
    char label[128];
    std::snprintf(label, sizeof(label), "%s rejected with TF_ERROR_INVALID",
                  what);
    check(status == TF_ERROR_INVALID, label);
    for (size_t i = 0; i < dst.size(); ++i) {
        std::snprintf(label, sizeof(label), "%s left dst[%zu] untouched",
                      what, i);
        check_eq(dst[i], 7777.5, label);
    }
    tf::clear_error();
}

// 13. Null handles are rejected by both exports.
void test_rejects_null_handles() {
    std::vector<double> dst(2, 7777.5);
    tf::Storage out{dst.data(), 2};
    const int64_t shape[1] = {2};
    const int64_t strides[1] = {1};
    tf_core_log(nullptr, &out, shape, strides, 0, 1);
    expect_rejected(tf::last_error_code(), dst, "strided null source");
    tf_core_log_contiguous(nullptr, &out, 2, 0);
    expect_rejected(tf::last_error_code(), dst, "contiguous null source");
    std::vector<double> src = {1.0, 2.0};
    tf::Storage in{src.data(), 2};
    tf_core_log(&in, nullptr, shape, strides, 0, 1);
    check(tf::last_error_code() == TF_ERROR_INVALID, "strided null destination");
    tf::clear_error();
    tf_core_log_contiguous(&in, nullptr, 2, 0);
    check(tf::last_error_code() == TF_ERROR_INVALID,
          "contiguous null destination");
    tf::clear_error();
}

// 14. Malformed dimensional metadata.
void test_rejects_malformed_dimensions() {
    std::vector<double> src = {1.0, 2.0, 3.0, 4.0};
    std::vector<double> dst(4, 7777.5);
    check(run_strided(src, dst, {2, 2}, {2, 1}, 0) == TF_OK, "sane baseline");
    std::fill(dst.begin(), dst.end(), 7777.5);

    tf::Storage in{src.data(), 4};
    tf::Storage out{dst.data(), 4};
    const int64_t shape[2] = {2, 2};
    const int64_t strides[2] = {2, 1};
    tf_core_log(&in, &out, shape, strides, 0, -1);
    expect_rejected(tf::last_error_code(), dst, "negative ndim");

    const int64_t zero_shape[2] = {2, 0};
    tf_core_log(&in, &out, zero_shape, strides, 0, 2);
    expect_rejected(tf::last_error_code(), dst, "zero dimension");

    const int64_t negative_shape[2] = {2, -2};
    tf_core_log(&in, &out, negative_shape, strides, 0, 2);
    expect_rejected(tf::last_error_code(), dst, "negative dimension");

    tf_core_log(&in, &out, nullptr, strides, 0, 2);
    expect_rejected(tf::last_error_code(), dst, "null shape array");

    tf_core_log(&in, &out, shape, nullptr, 0, 2);
    expect_rejected(tf::last_error_code(), dst, "null stride array");
}

// 15. Spans that would read or write outside their storage.
void test_rejects_out_of_range_spans() {
    std::vector<double> src = {1.0, 2.0, 3.0, 4.0};
    std::vector<double> dst(4, 7777.5);

    expect_rejected(run_contiguous(src, dst, 4, 1), dst,
                    "contiguous span past the end");
    expect_rejected(run_contiguous(src, dst, 5, 0), dst,
                    "contiguous count past the end");
    expect_rejected(run_contiguous(src, dst, 2, -1), dst,
                    "contiguous negative offset");
    expect_rejected(run_contiguous(src, dst, -1, 0), dst,
                    "contiguous negative count");

    expect_rejected(run_strided(src, dst, {4}, {2}, 0), dst,
                    "strided walk past the end");
    expect_rejected(run_strided(src, dst, {2, 2}, {2, 1}, 1), dst,
                    "strided offset past the end");
    expect_rejected(run_strided(src, dst, {2}, {-1}, 0), dst,
                    "strided walk before the start");
    expect_rejected(run_strided(src, dst, {4}, {-1}, 2), dst,
                    "negative stride reaching before the start");
    expect_rejected(run_strided(src, dst, {2}, {1}, -1), dst,
                    "strided negative offset");
    expect_rejected(run_strided(src, dst, {}, {}, 4), dst,
                    "rank-0 offset past the end");

    std::vector<double> small(2, 7777.5);
    expect_rejected(run_contiguous(src, small, 4, 0), small,
                    "destination too small (contiguous)");
    expect_rejected(run_strided(src, small, {4}, {1}, 0), small,
                    "destination too small (strided)");
}

// 16. Overflowing dimensional arithmetic is rejected, never wrapped.
void test_rejects_overflow() {
    std::vector<double> src = {1.0, 2.0};
    std::vector<double> dst(2, 7777.5);
    const int64_t huge = INT64_MAX / 2 + 4;
    expect_rejected(run_strided(src, dst, {huge, huge}, {1, 1}, 0), dst,
                    "element-count overflow");
    expect_rejected(run_strided(src, dst, {3}, {INT64_MAX / 2 + 1}, 0), dst,
                    "stride-travel overflow");
    // INT64_MIN cannot be negated, so it is rejected before the magnitude
    // is taken (undefined behavior otherwise).
    expect_rejected(run_strided(src, dst, {2}, {INT64_MIN}, 0), dst,
                    "INT64_MIN stride");
    tf::Storage in{src.data(), 2};
    tf::Storage out{dst.data(), 2};
    tf_core_log_contiguous(&in, &out, INT64_MAX, INT64_MAX);
    expect_rejected(tf::last_error_code(), dst, "contiguous span overflow");
}

// 17. The guard clears the slot on entry, so neither a previous
// rejection nor an unrelated stale error contaminates a later call.
void test_guard_clears_previous_error() {
    std::vector<double> src = {1.0, 2.0};
    std::vector<double> dst(2, 7777.5);
    expect_rejected(run_contiguous(src, dst, 5, 0), dst, "priming failure");
    tf::set_error(TF_ERROR_RUNTIME, "stale error from somewhere else");
    check(run_contiguous(src, dst, 2, 0) == TF_OK,
          "a valid call clears the stale error");
    check_eq(dst[0], 0.0, "post-failure call still computes");
    check_eq(dst[1], std::log(2.0), "post-failure call still computes [1]");
    // Same through the strided export.
    tf::set_error(TF_ERROR_RUNTIME, "another stale error");
    std::vector<double> other(2, 7777.5);
    check(run_strided(src, other, {2}, {1}, 0) == TF_OK,
          "a valid strided call clears the stale error");
    check_eq(other[0], 0.0, "strided post-failure call still computes");
}

// 18. Sharing E1's validators with a second operation did not weaken
// them: exp-shaped rejections still behave identically for log, and a
// domain result is still not a rejection.
void test_shared_validator_regression() {
    std::vector<double> src = {1.0, 0.0, -1.0};
    std::vector<double> dst(3, 7777.5);
    // A domain-heavy input is accepted (not a validation failure)...
    check(run_contiguous(src, dst, 3, 0) == TF_OK, "domain input accepted");
    check_eq(dst[0], 0.0, "shared_validator log(1)");
    check_eq(dst[1], kNegInf, "shared_validator log(0)");
    check(std::isnan(dst[2]), "shared_validator log(-1)");
    // ...while a bad span on the very same data is still rejected.
    std::fill(dst.begin(), dst.end(), 7777.5);
    expect_rejected(run_contiguous(src, dst, 4, 0), dst,
                    "shared validator still rejects a bad span");
}

}  // namespace

int main() {
    test_contiguous_single_element();
    test_contiguous_values();
    test_contiguous_offset();
    test_strided_matches_contiguous();
    test_strided_transposed();
    test_strided_offset();
    test_strided_negative_stride();
    test_strided_rank_zero();
    test_domain_values_are_not_errors();
    test_extreme_finite_values();
    test_input_unmodified_and_output_fully_written();
    test_determinism();
    test_rejects_null_handles();
    test_rejects_malformed_dimensions();
    test_rejects_out_of_range_spans();
    test_rejects_overflow();
    test_guard_clears_previous_error();
    test_shared_validator_regression();

    if (g_failures == 0) {
        std::printf("OK: all log tests passed\n");
        return 0;
    }
    std::printf("FAILED: %d log check(s)\n", g_failures);
    return 1;
}
