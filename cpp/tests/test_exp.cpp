// Dependency-free C++ test for the native exponential (Phase E,
// milestone E1). No GoogleTest/Catch2 — a plain executable that prints
// failures and returns a nonzero exit code if any check fails, so CTest
// reports pass/fail.
//
// This binary compiles cpp/src/elementwise.cpp (and error.cpp) directly
// (see the CMake TF_BUILD_TESTS target). The exponential's compute lives
// in that translation unit's anonymous namespace, so — unlike the conv2d
// and pooling tests, which reach hidden tf:: kernels — the surface under
// test here is the **exported guarded C ABI pair**:
//
//   * tf_core_exp             — the generic strided (odometer) export
//   * tf_core_exp_contiguous  — the flat contiguous fast-path export
//
// Both are driven through plain tf::Storage handles plus the
// thread-local error slot, which is also how their argument validation
// is exercised: E1's exports validate handles, layout metadata, spans,
// and overflow themselves rather than trusting the Python wrapper.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <vector>

#include "tf_internal.h"  // TF_EXPORT, tf::Storage, error accessors, TfStatus

// The exports under test, declared with the same TF_EXPORT macro the
// definitions use so the linkage matches exactly.
TF_EXPORT void tf_core_exp(
    const void* src, void* dst,
    const int64_t* shape, const int64_t* strides, int64_t offset,
    int64_t ndim);
TF_EXPORT void tf_core_exp_contiguous(
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

// Relative comparison for values that are not exactly representable.
void check_close(double got, double want, double rel, const char* what) {
    const double tolerance = rel * (std::fabs(want) > 1.0 ? std::fabs(want) : 1.0);
    if (!(std::fabs(got - want) <= tolerance)) {
        std::printf("FAIL: %s (got %.17g, want %.17g, |diff|=%.3g)\n",
                    what, got, want, std::fabs(got - want));
        ++g_failures;
    }
}

// -- drivers ----------------------------------------------------------

// Run the contiguous export; returns the status code (TF_OK on success).
int run_contiguous(const std::vector<double>& src, std::vector<double>& dst,
                   int64_t numel, int64_t offset) {
    tf::Storage in{const_cast<double*>(src.data()),
                   static_cast<int64_t>(src.size())};
    tf::Storage out{dst.data(), static_cast<int64_t>(dst.size())};
    tf_core_exp_contiguous(&in, &out, numel, offset);
    return tf::last_error_code();
}

// Run the strided export; returns the status code (TF_OK on success).
int run_strided(const std::vector<double>& src, std::vector<double>& dst,
                const std::vector<int64_t>& shape,
                const std::vector<int64_t>& strides, int64_t offset) {
    tf::Storage in{const_cast<double*>(src.data()),
                   static_cast<int64_t>(src.size())};
    tf::Storage out{dst.data(), static_cast<int64_t>(dst.size())};
    tf_core_exp(&in, &out, shape.empty() ? nullptr : shape.data(),
                strides.empty() ? nullptr : strides.data(), offset,
                static_cast<int64_t>(shape.size()));
    return tf::last_error_code();
}

// -- forward value cases ----------------------------------------------

// 1. A single element through the contiguous path: exp(0) is exactly 1.
void test_contiguous_scalar() {
    std::vector<double> src = {0.0};
    std::vector<double> dst(1, 7777.5);
    check(run_contiguous(src, dst, 1, 0) == TF_OK, "scalar status");
    check_eq(dst[0], 1.0, "exp(0) == 1 exactly");
}

// 2. Several values at once, compared against std::exp elementwise.
void test_contiguous_values() {
    std::vector<double> src = {0.0, 1.0, -1.0, 2.5, -3.25, 0.5, -0.5, 10.0};
    std::vector<double> dst(src.size(), 7777.5);
    check(run_contiguous(src, dst, static_cast<int64_t>(src.size()), 0) == TF_OK,
          "values status");
    for (size_t i = 0; i < src.size(); ++i) {
        char label[96];
        std::snprintf(label, sizeof(label), "contiguous_values[%zu]", i);
        check_eq(dst[i], std::exp(src[i]), label);
    }
    // Known references, independent of std::exp on the left-hand side.
    check_close(dst[1], 2.718281828459045, 1e-15, "exp(1) == e");
    check_close(dst[2], 0.36787944117144233, 1e-15, "exp(-1) == 1/e");
}

// 3. A nonzero offset starts the contiguous run later in the storage and
// never reads the elements before it.
void test_contiguous_offset() {
    std::vector<double> src = {99.0, 99.0, 0.0, 1.0, 2.0};
    std::vector<double> dst(3, 7777.5);
    check(run_contiguous(src, dst, 3, 2) == TF_OK, "offset status");
    check_eq(dst[0], 1.0, "offset[0]");
    check_eq(dst[1], std::exp(1.0), "offset[1]");
    check_eq(dst[2], std::exp(2.0), "offset[2]");
}

// 4. The strided export over a plain row-major 2x3 view must agree with
// the contiguous export bit-for-bit — the fast path is a traversal
// choice, never a different computation.
void test_strided_matches_contiguous() {
    std::vector<double> src = {0.0, 0.25, -0.75, 1.5, -2.0, 3.0};
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
    std::vector<double> src = {0.0, 1.0, 2.0, 3.0, 4.0, 5.0};
    std::vector<double> dst(6, 7777.5);
    check(run_strided(src, dst, {3, 2}, {1, 3}, 0) == TF_OK,
          "transposed status");
    const double want[6] = {std::exp(0.0), std::exp(3.0),
                            std::exp(1.0), std::exp(4.0),
                            std::exp(2.0), std::exp(5.0)};
    for (size_t i = 0; i < 6; ++i) {
        char label[96];
        std::snprintf(label, sizeof(label), "strided_transposed[%zu]", i);
        check_eq(dst[i], want[i], label);
    }
}

// 6. A strided view with a nonzero offset (the "narrow then transpose"
// layout the Core wrapper produces): shape (2, 2), strides (1, 3),
// offset 1 over a 2x3 buffer.
void test_strided_offset() {
    std::vector<double> src = {9.0, 0.5, 1.0, 9.0, 1.5, 2.0};
    std::vector<double> dst(4, 7777.5);
    check(run_strided(src, dst, {2, 2}, {1, 3}, 1) == TF_OK,
          "strided offset status");
    check_eq(dst[0], std::exp(0.5), "strided_offset[0]");
    check_eq(dst[1], std::exp(1.5), "strided_offset[1]");
    check_eq(dst[2], std::exp(1.0), "strided_offset[2]");
    check_eq(dst[3], std::exp(2.0), "strided_offset[3]");
}

// 7. Rank 0 (a scalar view) reads exactly one element, at the offset.
void test_strided_rank_zero() {
    std::vector<double> src = {5.0, 0.0};
    std::vector<double> dst(1, 7777.5);
    check(run_strided(src, dst, {}, {}, 1) == TF_OK, "rank-0 status");
    check_eq(dst[0], 1.0, "rank-0 reads src[offset]");
}

// -- IEEE exceptional values -------------------------------------------

// 8. Infinities, NaN, and the overflow/underflow edges — unclamped.
void test_exceptional_values() {
    std::vector<double> src = {kPosInf, kNegInf, kNaN, 1000.0, -1000.0, -0.0};
    std::vector<double> dst(src.size(), 7777.5);
    check(run_contiguous(src, dst, static_cast<int64_t>(src.size()), 0) == TF_OK,
          "exceptional status");
    check_eq(dst[0], kPosInf, "exp(+inf) == +inf");
    check_eq(dst[1], 0.0, "exp(-inf) == 0");
    check(!std::signbit(dst[1]), "exp(-inf) == +0 (positive zero)");
    check(std::isnan(dst[2]), "exp(NaN) is NaN");
    check_eq(dst[3], kPosInf, "exp(1000) overflows to +inf");
    check_eq(dst[4], 0.0, "exp(-1000) underflows to 0");
    check_eq(dst[5], 1.0, "exp(-0.0) == 1");
    // The same values through the strided path.
    std::vector<double> other(src.size(), 7777.5);
    check(run_strided(src, other, {static_cast<int64_t>(src.size())}, {1}, 0)
              == TF_OK, "exceptional strided status");
    check_eq(other[0], kPosInf, "strided exp(+inf)");
    check_eq(other[1], 0.0, "strided exp(-inf)");
    check(std::isnan(other[2]), "strided exp(NaN)");
    check_eq(other[3], kPosInf, "strided exp(1000)");
    check_eq(other[4], 0.0, "strided exp(-1000)");
}

// -- ownership / non-mutation ------------------------------------------

// 9. The input storage is never written, and the output is filled in
// full (every slot of the pre-poisoned destination is overwritten).
void test_input_unmodified_and_output_fully_written() {
    const std::vector<double> original = {0.0, 1.0, -1.0, 2.0};
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

// 10. Determinism: the same input gives the same bits every time, and
// the two paths agree exactly.
void test_determinism() {
    std::vector<double> src = {0.125, -0.25, 3.5, -4.75, 7.0};
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

// 11. Null handles are rejected by both exports.
void test_rejects_null_handles() {
    std::vector<double> dst(2, 7777.5);
    tf::Storage out{dst.data(), 2};
    const int64_t shape[1] = {2};
    const int64_t strides[1] = {1};
    tf_core_exp(nullptr, &out, shape, strides, 0, 1);
    expect_rejected(tf::last_error_code(), dst, "strided null source");
    tf_core_exp_contiguous(nullptr, &out, 2, 0);
    expect_rejected(tf::last_error_code(), dst, "contiguous null source");
    std::vector<double> src = {0.0, 1.0};
    tf::Storage in{src.data(), 2};
    tf_core_exp(&in, nullptr, shape, strides, 0, 1);
    check(tf::last_error_code() == TF_ERROR_INVALID, "null destination");
    tf::clear_error();
}

// 12. Malformed dimensional metadata: negative ndim, a non-positive
// dimension, and a null shape/stride array with ndim > 0.
void test_rejects_malformed_dimensions() {
    std::vector<double> src = {0.0, 1.0, 2.0, 3.0};
    std::vector<double> dst(4, 7777.5);
    check(run_strided(src, dst, {2, 2}, {2, 1}, 0) == TF_OK, "sane baseline");
    std::fill(dst.begin(), dst.end(), 7777.5);

    tf::Storage in{src.data(), 4};
    tf::Storage out{dst.data(), 4};
    const int64_t shape[2] = {2, 2};
    const int64_t strides[2] = {2, 1};
    tf_core_exp(&in, &out, shape, strides, 0, -1);
    expect_rejected(tf::last_error_code(), dst, "negative ndim");

    const int64_t zero_shape[2] = {2, 0};
    tf_core_exp(&in, &out, zero_shape, strides, 0, 2);
    expect_rejected(tf::last_error_code(), dst, "zero dimension");

    const int64_t negative_shape[2] = {2, -2};
    tf_core_exp(&in, &out, negative_shape, strides, 0, 2);
    expect_rejected(tf::last_error_code(), dst, "negative dimension");

    tf_core_exp(&in, &out, nullptr, strides, 0, 2);
    expect_rejected(tf::last_error_code(), dst, "null shape array");

    tf_core_exp(&in, &out, shape, nullptr, 0, 2);
    expect_rejected(tf::last_error_code(), dst, "null stride array");
}

// 13. Spans that would read or write outside their storage.
void test_rejects_out_of_range_spans() {
    std::vector<double> src = {0.0, 1.0, 2.0, 3.0};
    std::vector<double> dst(4, 7777.5);

    // Contiguous: offset + numel past the end.
    expect_rejected(run_contiguous(src, dst, 4, 1), dst,
                    "contiguous span past the end");
    expect_rejected(run_contiguous(src, dst, 5, 0), dst,
                    "contiguous count past the end");
    expect_rejected(run_contiguous(src, dst, 2, -1), dst,
                    "contiguous negative offset");
    expect_rejected(run_contiguous(src, dst, -1, 0), dst,
                    "contiguous negative count");

    // Strided: a stride that walks past the last element.
    expect_rejected(run_strided(src, dst, {4}, {2}, 0), dst,
                    "strided walk past the end");
    expect_rejected(run_strided(src, dst, {2, 2}, {2, 1}, 1), dst,
                    "strided offset past the end");
    expect_rejected(run_strided(src, dst, {2}, {-1}, 0), dst,
                    "strided walk before the start");
    expect_rejected(run_strided(src, dst, {2}, {1}, -1), dst,
                    "strided negative offset");

    // Rank 0 reads exactly one element: an offset at or past the end is
    // out of range.
    expect_rejected(run_strided(src, dst, {}, {}, 4), dst,
                    "rank-0 offset past the end");

    // A destination smaller than the element count is rejected too.
    std::vector<double> small(2, 7777.5);
    expect_rejected(run_contiguous(src, small, 4, 0), small,
                    "destination too small (contiguous)");
    expect_rejected(run_strided(src, small, {4}, {1}, 0), small,
                    "destination too small (strided)");
}

// 14. Overflowing dimensional arithmetic is rejected rather than wrapped
// into a small span that would pass a naive bounds check.
void test_rejects_overflow() {
    std::vector<double> src = {0.0, 1.0};
    std::vector<double> dst(2, 7777.5);
    const int64_t huge = INT64_MAX / 2 + 4;
    // numel = huge * huge overflows int64.
    expect_rejected(run_strided(src, dst, {huge, huge}, {1, 1}, 0), dst,
                    "element-count overflow");
    // stride * (shape - 1) overflows int64.
    expect_rejected(run_strided(src, dst, {3}, {INT64_MAX / 2 + 1}, 0), dst,
                    "stride-travel overflow");
    // offset + numel overflows int64 on the contiguous path.
    tf::Storage in{src.data(), 2};
    tf::Storage out{dst.data(), 2};
    tf_core_exp_contiguous(&in, &out, INT64_MAX, INT64_MAX);
    expect_rejected(tf::last_error_code(), dst, "contiguous span overflow");
}

// 15. The error slot is cleared on entry, so a previous failure never
// contaminates a later successful call.
void test_guard_clears_previous_error() {
    std::vector<double> src = {0.0, 1.0};
    std::vector<double> dst(2, 7777.5);
    expect_rejected(run_contiguous(src, dst, 5, 0), dst, "priming failure");
    tf::set_error(TF_ERROR_RUNTIME, "stale error from somewhere else");
    check(run_contiguous(src, dst, 2, 0) == TF_OK,
          "a valid call clears the stale error");
    check_eq(dst[0], 1.0, "post-failure call still computes");
    check_eq(dst[1], std::exp(1.0), "post-failure call still computes [1]");
}

}  // namespace

int main() {
    test_contiguous_scalar();
    test_contiguous_values();
    test_contiguous_offset();
    test_strided_matches_contiguous();
    test_strided_transposed();
    test_strided_offset();
    test_strided_rank_zero();
    test_exceptional_values();
    test_input_unmodified_and_output_fully_written();
    test_determinism();
    test_rejects_null_handles();
    test_rejects_malformed_dimensions();
    test_rejects_out_of_range_spans();
    test_rejects_overflow();
    test_guard_clears_previous_error();

    if (g_failures == 0) {
        std::printf("OK: all exp tests passed\n");
        return 0;
    }
    std::printf("FAILED: %d exp check(s)\n", g_failures);
    return 1;
}
