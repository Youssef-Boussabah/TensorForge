// Dependency-free C++ test for the two matmul compute paths and the
// metadata predicate that dispatches between them (Phase H, milestone
// H2). No GoogleTest / Catch2 — a plain executable that prints failures
// and returns a nonzero exit code if any check fails, so CTest reports
// pass/fail.
//
// This binary compiles cpp/src/matmul.cpp (and error.cpp, for the
// thread-local error slot the guarded export references) directly, which
// is the only way to reach ``tf::matmul_prefers_row_sweep``,
// ``tf::matmul_generic_strided``, and ``tf::matmul_row_sweep``: they are
// hidden-visibility internals and the shared library exports none of
// them. That is deliberate, and it is what lets H2 prove which path a
// layout takes **without** adding a "which kernel ran" export, a
// block-size setter, a dispatch tracer, or any other runtime control to
// the shipped ABI.
//
// Three properties are established here, and they are the load-bearing
// ones:
//
//   1. **The predicate is exactly the documented table.** Total, pure, a
//      function of (m, n, p, b_stride1) alone. Every row of the table in
//      docs/native_cpu_performance_design.md §16.2 is asserted, including
//      both sides of each boundary.
//   2. **The two paths agree under H2's four-part numerical contract**,
//      compared as raw IEEE-754 bit patterns rather than by tolerance:
//        a. accumulation order is preserved exactly;
//        b. **every non-NaN result is bit-identical** — signed zeros,
//           infinities, denormals, the smallest normal, and the largest
//           finite magnitudes included;
//        c. NaN results occur in exactly the same positions on both
//           paths and are always **quiet** NaNs;
//        d. **NaN payload bits are outside the contract** and may differ.
//      (d) is a measured property, not a hedge: x86-64's ADDSD returns
//      the destination operand's NaN when both are NaN, and which addend
//      the compiler places there follows from the loop order rather than
//      from anything C++ can express. Ten source-level formulations were
//      measured and all ten i-k-j spellings agreed with each other and
//      differed from the i-j-k reference. The tests below therefore
//      enforce (b) and (c) exactly, and deliberately assert **nothing**
//      about (d) in either direction.
//   3. **The row sweep writes every destination element before reading
//      it.** Its destination is pre-filled with a poison pattern (the
//      same technique the Python suite uses, needing no library support)
//      and no poison may survive, nor may any output value differ from
//      the run over a differently poisoned buffer.

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <vector>

#include "tf_internal.h"
#include "tf_matmul_internal.h"

TF_EXPORT void* tf_storage_create(std::int64_t size);
TF_EXPORT void tf_storage_destroy(void* handle);
TF_EXPORT int tf_last_error_code();
TF_EXPORT void tf_matmul(const double* a, const double* b, double* out,
                         std::int64_t m, std::int64_t n, std::int64_t p);
TF_EXPORT void tf_core_matmul(
    const void* a_handle, const void* b_handle, void* dst_handle,
    std::int64_t m, std::int64_t n, std::int64_t p,
    std::int64_t a_stride0, std::int64_t a_stride1,
    std::int64_t b_stride0, std::int64_t b_stride1,
    std::int64_t a_offset, std::int64_t b_offset);

namespace {

int g_failures = 0;

void check(bool condition, const char* what) {
    if (!condition) {
        std::printf("FAIL: %s\n", what);
        ++g_failures;
    }
}

void check_at(bool condition, const char* what,
              std::int64_t m, std::int64_t n, std::int64_t p) {
    if (!condition) {
        std::printf("FAIL: %s at m=%lld n=%lld p=%lld\n", what,
                    static_cast<long long>(m), static_cast<long long>(n),
                    static_cast<long long>(p));
        ++g_failures;
    }
}

// Raw bit comparison. Two float64 buffers are "the same" here only when
// every bit agrees — so +0.0 differs from -0.0, and two NaNs differ
// unless their payloads match. Tolerance would hide exactly the
// differences H2 must not introduce.
bool same_bits(const std::vector<double>& left,
               const std::vector<double>& right) {
    if (left.size() != right.size()) {
        return false;
    }
    return left.empty() ||
           std::memcmp(left.data(), right.data(),
                       sizeof(double) * left.size()) == 0;
}

// H2's numerical contract, parts (b) and (c) together: every element is
// either bit-identical, or **both** sides are NaN. One NaN against a
// number, or two different non-NaN numbers, is a real difference and
// fails. ``nan_pairs`` counts the elements where only the payload
// differed — reported, never asserted, because part (d) puts payload
// bits outside the contract.
bool agrees_under_the_numerical_contract(const std::vector<double>& left,
                                         const std::vector<double>& right,
                                         int* nan_pairs) {
    if (left.size() != right.size()) {
        return false;
    }
    *nan_pairs = 0;
    for (std::size_t i = 0; i < left.size(); ++i) {
        std::uint64_t lb, rb;
        std::memcpy(&lb, &left[i], sizeof(lb));
        std::memcpy(&rb, &right[i], sizeof(rb));
        if (lb == rb) {
            continue;
        }
        if (std::isnan(left[i]) && std::isnan(right[i])) {
            ++(*nan_pairs);
            continue;
        }
        return false;
    }
    return true;
}

// A NaN is quiet when its most significant mantissa bit is set. Neither
// path may hand back a signaling NaN.
bool all_nans_are_quiet(const std::vector<double>& values) {
    for (double value : values) {
        if (!std::isnan(value)) {
            continue;
        }
        std::uint64_t bits;
        std::memcpy(&bits, &value, sizeof(bits));
        if ((bits & 0x0008000000000000ull) == 0) {
            return false;
        }
    }
    return true;
}

// A deterministic, reproducible value generator — no <random>, no clock,
// no global state, so this file's expectations are fixed forever.
double sample(std::int64_t index) {
    const double phase = static_cast<double>(index % 17) - 8.0;
    const double scale = 1.0 + static_cast<double>(index % 5) * 0.25;
    return phase * scale * 0.125;
}

std::vector<double> filled(std::int64_t count, std::int64_t seed) {
    std::vector<double> values(static_cast<std::size_t>(count));
    for (std::int64_t i = 0; i < count; ++i) {
        values[static_cast<std::size_t>(i)] = sample(i + seed);
    }
    return values;
}

// The poison patterns, matching tests/test_native_storage_allocation.py:
// a quiet NaN with a distinctive payload, and a large negative finite
// value that catches code special-casing NaN.
double poison_nan() {
    const std::uint64_t bits = 0x7FF8DEADBEEFCAFEull;
    double value;
    std::memcpy(&value, &bits, sizeof(value));
    return value;
}
const double kPoisonFinite = -1.2345678901234567e300;

// ---------------------------------------------------------------------------
// 1. The dispatch predicate is exactly the documented table
// ---------------------------------------------------------------------------

void test_predicate_table() {
    // Qualifying: unit column stride, n >= 1, p >= MATMUL_MIN_COLUMNS.
    check(tf::matmul_prefers_row_sweep(4, 4, 8, 1),
          "predicate rejected the canonical qualifying layout");
    check(tf::matmul_prefers_row_sweep(1, 1, 64, 1),
          "predicate rejected a single-row qualifying layout");
    check(tf::matmul_prefers_row_sweep(0, 3, 16, 1),
          "predicate rejected an empty-m qualifying layout");
    check(tf::matmul_prefers_row_sweep(1000, 1, 1000, 1),
          "predicate rejected n == 1");

    // b's column stride must be exactly 1. A transposed right operand
    // (b_stride0 == 1, b_stride1 == p) is the generic kernel's good case.
    check(!tf::matmul_prefers_row_sweep(64, 64, 64, 64),
          "predicate accepted a transposed right operand");
    check(!tf::matmul_prefers_row_sweep(64, 64, 64, 2),
          "predicate accepted a non-unit column stride");
    check(!tf::matmul_prefers_row_sweep(64, 64, 64, 0),
          "predicate accepted a zero column stride");
    check(!tf::matmul_prefers_row_sweep(64, 64, 64, -1),
          "predicate accepted a negative column stride");

    // n >= 1, because the k == 0 pass is what assigns the destination.
    check(!tf::matmul_prefers_row_sweep(4, 0, 64, 1),
          "predicate accepted n == 0");
    check(!tf::matmul_prefers_row_sweep(4, -1, 64, 1),
          "predicate accepted a negative n");

    // The p boundary, both sides, at the documented value.
    check(tf::MATMUL_MIN_COLUMNS == 8, "MATMUL_MIN_COLUMNS moved");
    check(tf::MATMUL_ROW_BLOCK == 4, "MATMUL_ROW_BLOCK moved");
    for (std::int64_t p = 0; p < tf::MATMUL_MIN_COLUMNS; ++p) {
        check_at(!tf::matmul_prefers_row_sweep(8, 8, p, 1),
                 "predicate accepted p below the threshold", 8, 8, p);
    }
    for (std::int64_t p = tf::MATMUL_MIN_COLUMNS;
         p <= tf::MATMUL_MIN_COLUMNS + 4; ++p) {
        check_at(tf::matmul_prefers_row_sweep(8, 8, p, 1),
                 "predicate rejected p at or above the threshold", 8, 8, p);
    }

    // Pure and repeatable: the same question always gets the same answer.
    for (int repeat = 0; repeat < 3; ++repeat) {
        check(tf::matmul_prefers_row_sweep(17, 23, 29, 1),
              "predicate is not repeatable");
    }
}

// ---------------------------------------------------------------------------
// 2. Bit identity between the two paths
// ---------------------------------------------------------------------------

// Run both paths over the same logical operands and compare raw bits.
// ``a`` is supplied through arbitrary strides; ``b`` is supplied twice —
// once row-major (column stride 1, the row sweep's layout) and once as
// the transposed buffer the generic path would receive for the same
// logical matrix — so the comparison really is "the same product, two
// paths", not "the same kernel twice".
void compare_paths(const char* label,
                   std::int64_t m, std::int64_t n, std::int64_t p,
                   const std::vector<double>& a_values,
                   std::int64_t a_stride0, std::int64_t a_stride1,
                   std::int64_t a_offset,
                   const std::vector<double>& b_row_major) {
    std::vector<double> generic(static_cast<std::size_t>(m * p), 0.0);
    std::vector<double> sweep(static_cast<std::size_t>(m * p), 0.0);

    tf::matmul_generic_strided(a_values.data(), b_row_major.data(),
                               generic.data(), m, n, p,
                               a_stride0, a_stride1, p, 1, a_offset, 0);
    tf::matmul_row_sweep(a_values.data(), b_row_major.data(), sweep.data(),
                         m, n, p, a_stride0, a_stride1, p, a_offset, 0);
    if (!same_bits(generic, sweep)) {
        std::printf("FAIL: %s: row sweep differs from the generic path\n",
                    label);
        ++g_failures;
    }

    // The same logical b, delivered as a transposed (p, n) buffer, which
    // is the layout the generic path takes in production. The generic
    // result must be bit-identical through both deliveries, so the
    // comparison above is not an artifact of one memory arrangement.
    std::vector<double> b_transposed(static_cast<std::size_t>(n * p), 0.0);
    for (std::int64_t k = 0; k < n; ++k) {
        for (std::int64_t j = 0; j < p; ++j) {
            b_transposed[static_cast<std::size_t>(j * n + k)] =
                b_row_major[static_cast<std::size_t>(k * p + j)];
        }
    }
    std::vector<double> viewed(static_cast<std::size_t>(m * p), 0.0);
    tf::matmul_generic_strided(a_values.data(), b_transposed.data(),
                               viewed.data(), m, n, p,
                               a_stride0, a_stride1, 1, n, a_offset, 0);
    if (!same_bits(generic, viewed)) {
        std::printf("FAIL: %s: the generic path differs across b layouts\n",
                    label);
        ++g_failures;
    }
}

void test_finite_bit_identity_across_shapes() {
    // Dimensions immediately below, at, and above the row-block and
    // column-threshold boundaries; primes; one-element dimensions;
    // rectangular and square; multiple blocks and a partial final block.
    const std::int64_t dims[] = {1, 2, 3, 4, 5, 7, 8, 9, 11, 13, 16, 17,
                                 31, 32, 33, 63, 64, 65};
    for (std::int64_t m : dims) {
        for (std::int64_t n : dims) {
            for (std::int64_t p : dims) {
                if (m * n * p > 40000) {
                    continue;  // keep the CTest fast; larger cases below
                }
                const std::vector<double> a = filled(m * n, m * 31 + n);
                const std::vector<double> b = filled(n * p, n * 17 + p + 5);
                std::vector<double> generic(
                    static_cast<std::size_t>(m * p), 0.0);
                std::vector<double> sweep(
                    static_cast<std::size_t>(m * p), 0.0);
                tf::matmul_generic_strided(a.data(), b.data(), generic.data(),
                                           m, n, p, n, 1, p, 1, 0, 0);
                tf::matmul_row_sweep(a.data(), b.data(), sweep.data(),
                                     m, n, p, n, 1, p, 0, 0);
                check_at(same_bits(generic, sweep),
                         "row sweep is not bit-identical", m, n, p);
            }
        }
    }
}

void test_finite_bit_identity_at_larger_shapes() {
    const std::int64_t shapes[][3] = {
        {128, 128, 128}, {127, 127, 127}, {129, 65, 33},
        {1, 512, 64}, {512, 1, 64}, {64, 512, 8}, {33, 8, 129},
    };
    for (const auto& shape : shapes) {
        const std::int64_t m = shape[0], n = shape[1], p = shape[2];
        const std::vector<double> a = filled(m * n, m + n * 3);
        const std::vector<double> b = filled(n * p, n + p * 7 + 11);
        compare_paths("large shape", m, n, p, a, n, 1, 0, b);
    }
}

void test_finite_bit_identity_through_strided_left_operands() {
    // A transposed left operand (a_stride1 == m, a_stride0 == 1) is the
    // exact shape `db = a.T @ upstream` feeds the kernel in the matmul
    // backward, and it qualifies for the row sweep whenever the right
    // operand is row-major. A narrowed left operand exercises a nonzero
    // offset and a row stride wider than the logical row.
    const std::int64_t m = 12, n = 9, p = 16;

    std::vector<double> a_transposed = filled(n * m, 3);   // logical (m, n)
    const std::vector<double> b = filled(n * p, 41);
    compare_paths("transposed left operand", m, n, p, a_transposed,
                  1, m, 0, b);

    // (m, n) window inside a (m + 3, n + 5) buffer, starting at (2, 3).
    const std::int64_t wide = n + 5;
    std::vector<double> a_wide = filled((m + 3) * wide, 77);
    compare_paths("narrowed left operand", m, n, p, a_wide,
                  wide, 1, 2 * wide + 3, b);

    // A non-unit row *and* column stride on the left operand: every
    // second row and every third column of a larger buffer.
    const std::int64_t tall = m * 2, broad = n * 3;
    std::vector<double> a_sparse = filled(tall * broad, 129);
    compare_paths("positive non-unit strides", m, n, p, a_sparse,
                  2 * broad, 3, 0, b);
}

// Build the "every special against every special" pair of operands from
// a value list, and run both paths over it.
void special_value_pair(const double* values, std::int64_t count,
                        std::int64_t p,
                        std::vector<double>& generic,
                        std::vector<double>& sweep) {
    const std::int64_t m = count, n = count;
    std::vector<double> a(static_cast<std::size_t>(m * n));
    for (std::int64_t i = 0; i < m; ++i) {
        for (std::int64_t k = 0; k < n; ++k) {
            a[static_cast<std::size_t>(i * n + k)] =
                values[static_cast<std::size_t>(i)];
        }
    }
    std::vector<double> b(static_cast<std::size_t>(n * p));
    for (std::int64_t k = 0; k < n; ++k) {
        for (std::int64_t j = 0; j < p; ++j) {
            b[static_cast<std::size_t>(k * p + j)] =
                values[static_cast<std::size_t>((k + j) % count)];
        }
    }
    generic.assign(static_cast<std::size_t>(m * p), 0.0);
    sweep.assign(static_cast<std::size_t>(m * p), 0.0);
    tf::matmul_generic_strided(a.data(), b.data(), generic.data(), m, n, p,
                               n, 1, p, 1, 0, 0);
    tf::matmul_row_sweep(a.data(), b.data(), sweep.data(), m, n, p,
                         n, 1, p, 0, 0);
}

void test_the_contract_comparator_is_capable_of_failing() {
    // The comparator that states the documented exception is checked
    // directly, because whether the exception is *exercised* by a given
    // build is a codegen property (the MSVC Release build differs on
    // some NaN payloads; the Debug build differs on none). The rule must
    // still be proved strict where it matters.
    std::uint64_t payload_a = 0x7FF8000000000000ull;
    std::uint64_t payload_b = 0x7FF8DEADBEEFCAFEull;
    double nan_a, nan_b;
    std::memcpy(&nan_a, &payload_a, sizeof(nan_a));
    std::memcpy(&nan_b, &payload_b, sizeof(nan_b));
    check(std::isnan(nan_a) && std::isnan(nan_b), "the probes are not NaN");

    int pairs = 0;
    check(agrees_under_the_numerical_contract({nan_a}, {nan_b}, &pairs),
          "two NaNs with different payloads were rejected");
    check(pairs == 1, "the NaN-payload counter did not count the pair");
    check(!agrees_under_the_numerical_contract({nan_a}, {1.0}, &pairs),
          "a NaN against a number was accepted");
    check(!agrees_under_the_numerical_contract({1.0}, {nan_a}, &pairs),
          "a number against a NaN was accepted");
    check(!agrees_under_the_numerical_contract({1.0}, {1.0000000000000002}, &pairs),
          "two different numbers were accepted");
    check(!agrees_under_the_numerical_contract({0.0}, {-0.0}, &pairs),
          "a signed-zero difference was accepted");
    check(agrees_under_the_numerical_contract({0.0}, {0.0}, &pairs),
          "identical values were rejected");
    check(pairs == 0, "the counter reported a pair where there was none");
    // A hand-built signaling NaN (mantissa MSB clear) must be rejected by
    // the quietness check, so that check is not vacuous either.
    std::uint64_t signaling_bits = 0x7FF0000000000001ull;
    double signaling;
    std::memcpy(&signaling, &signaling_bits, sizeof(signaling));
    check(std::isnan(signaling), "the signaling probe is not NaN");
    check(!all_nans_are_quiet({signaling}),
          "a signaling NaN passed the quietness check");
    check(all_nans_are_quiet({nan_a, 1.0, 0.0}),
          "the quietness check rejected quiet NaNs");
}

// A quiet NaN carrying a chosen payload, so several distinguishable NaNs
// can be placed in one problem.
double quiet_nan_with(std::uint64_t payload, bool negative = false) {
    std::uint64_t bits = 0x7FF8000000000000ull | payload;
    if (negative) {
        bits |= 0x8000000000000000ull;
    }
    double value;
    std::memcpy(&value, &bits, sizeof(value));
    return value;
}

// Run one problem through both paths and enforce the contract on it.
void require_contract(const char* label,
                      const std::vector<double>& a,
                      const std::vector<double>& b,
                      std::int64_t m, std::int64_t n, std::int64_t p,
                      bool expect_a_nan) {
    std::vector<double> generic(static_cast<std::size_t>(m * p), 0.0);
    std::vector<double> sweep(static_cast<std::size_t>(m * p), 0.0);
    tf::matmul_generic_strided(a.data(), b.data(), generic.data(), m, n, p,
                               n, 1, p, 1, 0, 0);
    tf::matmul_row_sweep(a.data(), b.data(), sweep.data(), m, n, p,
                         n, 1, p, 0, 0);
    int nan_pairs = 0;
    if (!agrees_under_the_numerical_contract(generic, sweep, &nan_pairs)) {
        std::printf("FAIL: %s: the paths differ by more than a NaN payload\n",
                    label);
        ++g_failures;
    }
    if (!all_nans_are_quiet(generic) || !all_nans_are_quiet(sweep)) {
        std::printf("FAIL: %s: a signaling NaN reached a result\n", label);
        ++g_failures;
    }
    bool saw_nan = false;
    for (std::size_t i = 0; i < generic.size(); ++i) {
        if (std::isnan(generic[i]) != std::isnan(sweep[i])) {
            std::printf("FAIL: %s: the paths disagree about where NaNs are\n",
                        label);
            ++g_failures;
            break;
        }
        saw_nan = saw_nan || std::isnan(generic[i]);
    }
    if (saw_nan != expect_a_nan) {
        std::printf("FAIL: %s: expected %s NaN in the result\n", label,
                    expect_a_nan ? "a" : "no");
        ++g_failures;
    }
    // When the result is NaN-free the claim is the strong one.
    if (!expect_a_nan && !same_bits(generic, sweep)) {
        std::printf("FAIL: %s: NaN-free results are not bit-identical\n",
                    label);
        ++g_failures;
    }
}

// Where the NaN sits matters: a NaN in the left operand multiplies a
// whole output row, a NaN in the right operand a whole column, and both
// together make two NaNs meet inside one accumulation — which is the only
// situation in which payload selection is even reachable.
void test_the_contract_with_nans_in_each_operand() {
    const std::int64_t m = 4, n = 6, p = 16;
    const double nan_1 = quiet_nan_with(0x0111111111111ull);
    const double nan_2 = quiet_nan_with(0x0222222222222ull);
    const double nan_3 = quiet_nan_with(0x0333333333333ull, /*negative=*/true);
    const double inf = std::numeric_limits<double>::infinity();

    std::vector<double> base_a = filled(m * n, 11);
    std::vector<double> base_b = filled(n * p, 29);

    // A quiet NaN in the left operand only.
    {
        std::vector<double> a = base_a;
        a[static_cast<std::size_t>(1 * n + 2)] = nan_1;
        require_contract("nan in the left operand", a, base_b, m, n, p, true);
    }
    // A quiet NaN in the right operand only.
    {
        std::vector<double> b = base_b;
        b[static_cast<std::size_t>(3 * p + 5)] = nan_2;
        require_contract("nan in the right operand", base_a, b, m, n, p, true);
    }
    // Several distinct payloads, including a negative NaN, meeting inside
    // one accumulation from both operands at once.
    {
        std::vector<double> a = base_a;
        std::vector<double> b = base_b;
        a[static_cast<std::size_t>(0 * n + 0)] = nan_1;
        a[static_cast<std::size_t>(0 * n + 4)] = nan_3;
        b[static_cast<std::size_t>(2 * p + 0)] = nan_2;
        b[static_cast<std::size_t>(5 * p + 0)] = nan_3;
        require_contract("multiple nan payloads, both operands",
                         a, b, m, n, p, true);
    }
    // A NaN manufactured by the arithmetic rather than supplied: 0 * inf
    // in one term and inf - inf across two.
    {
        std::vector<double> a = base_a;
        std::vector<double> b = base_b;
        a[static_cast<std::size_t>(2 * n + 1)] = 0.0;
        b[static_cast<std::size_t>(1 * p + 3)] = inf;
        a[static_cast<std::size_t>(3 * n + 0)] = inf;
        a[static_cast<std::size_t>(3 * n + 1)] = -inf;
        require_contract("nan manufactured by 0*inf and inf-inf",
                         a, b, m, n, p, true);
    }
    // Infinities without any NaN: the results are +-inf and finite, and
    // the strong bit-identity claim applies.
    {
        std::vector<double> a = base_a;
        std::vector<double> b = base_b;
        for (std::int64_t k = 0; k < n; ++k) {
            a[static_cast<std::size_t>(0 * n + k)] = 1.0;
        }
        for (std::int64_t j = 0; j < p; ++j) {
            b[static_cast<std::size_t>(0 * p + j)] = inf;
        }
        require_contract("infinities without a nan", a, b, m, n, p, false);
    }
    // Signed zeros, denormals, and the largest finite magnitudes, with no
    // overflow: strong bit identity again.
    {
        const double tiny = std::numeric_limits<double>::denorm_min();
        const double small = std::numeric_limits<double>::min();
        std::vector<double> a(static_cast<std::size_t>(m * n));
        std::vector<double> b(static_cast<std::size_t>(n * p));
        const double values[] = {0.0, -0.0, tiny, -tiny, small, 1e150,
                                 -1e150, 0.5};
        for (std::size_t i = 0; i < a.size(); ++i) a[i] = values[i % 8];
        for (std::size_t i = 0; i < b.size(); ++i) b[i] = values[(i + 3) % 8];
        require_contract("signed zeros, denormals, huge finite",
                         a, b, m, n, p, false);
    }
}

void test_the_numerical_contract_on_special_values() {
    const double inf = std::numeric_limits<double>::infinity();
    const double quiet = std::numeric_limits<double>::quiet_NaN();
    const double tiny = std::numeric_limits<double>::denorm_min();
    const double small = std::numeric_limits<double>::min();
    const double huge = std::numeric_limits<double>::max();

    // (a) No NaN anywhere — signed zeros, denormals, the smallest normal,
    // the largest finite magnitudes, and values that scale into and out of
    // the representable range. Here **exact bit equality is claimed and
    // asserted**, with p == 16 so the row sweep really runs.
    const double finite[] = {0.0, -0.0, 1.0, -1.0, tiny, -tiny, small,
                             huge, -huge, 1e-300, 1e300, -1e300};
    const std::int64_t finite_count =
        static_cast<std::int64_t>(sizeof(finite) / sizeof(finite[0]));
    std::vector<double> generic, sweep;
    special_value_pair(finite, finite_count, 16, generic, sweep);
    check(same_bits(generic, sweep),
          "the paths differ on non-NaN special values");

    // (b) The full set, NaN and infinities included. Infinities alone can
    // still manufacture a NaN (inf - inf, 0 * inf), so this is where the
    // documented exception lives: bits equal, or both NaN.
    const double specials[] = {0.0, -0.0, 1.0, -1.0, inf, -inf, quiet,
                               tiny, -tiny, small, huge, 1e-300, 1e300};
    const std::int64_t count =
        static_cast<std::int64_t>(sizeof(specials) / sizeof(specials[0]));
    special_value_pair(specials, count, 16, generic, sweep);
    int nan_pairs = 0;
    check(agrees_under_the_numerical_contract(generic, sweep, &nan_pairs),
          "the paths differ by more than a NaN payload");
    check(all_nans_are_quiet(generic) && all_nans_are_quiet(sweep),
          "a signaling NaN reached a result");
    // NaN-*ness* itself must agree exactly — that is not the documented
    // exception, and a path that produced a NaN where the other produced
    // a number would be a real defect.
    bool nan_positions_agree = true;
    for (std::size_t i = 0; i < generic.size(); ++i) {
        if (std::isnan(generic[i]) != std::isnan(sweep[i])) {
            nan_positions_agree = false;
        }
    }
    check(nan_positions_agree, "the paths disagree about where NaNs are");
    bool any_nan = false;
    for (double value : generic) {
        if (std::isnan(value)) {
            any_nan = true;
        }
    }
    check(any_nan, "the special-value case produced no NaN at all");
    // ``nan_pairs`` is *reported*, never asserted in either direction.
    // Part (d) of the contract puts payload bits outside it, so a build
    // that agrees on every payload and a build that agrees on none are
    // equally conforming. Measured on this toolchain: MSVC Release
    // differs on 162 of 208, MSVC Debug and Clang -O0 on 0 of 208.
    std::printf("note: %d of %zu results differ in NaN payload only "
                "(outside the contract either way)\n",
                nan_pairs, generic.size());

    // The signed-zero case on its own, because it is the single reason
    // the row sweep's k == 0 pass writes `0.0 + product` rather than the
    // product: 0.0 + (-0.0) is +0.0, and dropping the addition would
    // hand back -0.0 instead.
    const std::int64_t sm = 2, sn = 1, sp = 8;
    const std::vector<double> minus_one(static_cast<std::size_t>(sm * sn),
                                        -1.0);
    std::vector<double> zeros(static_cast<std::size_t>(sn * sp), 0.0);
    std::vector<double> zero_generic(static_cast<std::size_t>(sm * sp), 7.0);
    std::vector<double> zero_sweep(static_cast<std::size_t>(sm * sp), 7.0);
    tf::matmul_generic_strided(minus_one.data(), zeros.data(),
                               zero_generic.data(),
                               sm, sn, sp, sn, 1, sp, 1, 0, 0);
    tf::matmul_row_sweep(minus_one.data(), zeros.data(), zero_sweep.data(),
                         sm, sn, sp, sn, 1, sp, 0, 0);
    check(same_bits(zero_generic, zero_sweep),
          "signed zero differs between the paths");
    check(!std::signbit(zero_generic[0]),
          "the reference no longer produces +0.0 for -1.0 * 0.0");
    check(!std::signbit(zero_sweep[0]),
          "the row sweep produced -0.0 where the reference produces +0.0");
}

void test_agreement_with_the_legacy_raw_kernel() {
    // tf_matmul is an independent transcription of the same triple loop
    // over plain contiguous buffers, on no production path. Both shipped
    // paths must match it bit for bit, which makes the parity above a
    // three-way agreement rather than two readings of one implementation.
    const std::int64_t shapes[][3] = {{8, 8, 8}, {16, 9, 32}, {33, 17, 64}};
    for (const auto& shape : shapes) {
        const std::int64_t m = shape[0], n = shape[1], p = shape[2];
        const std::vector<double> a = filled(m * n, 5);
        const std::vector<double> b = filled(n * p, 23);
        std::vector<double> legacy(static_cast<std::size_t>(m * p), 0.0);
        std::vector<double> generic(static_cast<std::size_t>(m * p), 0.0);
        std::vector<double> sweep(static_cast<std::size_t>(m * p), 0.0);
        tf_matmul(a.data(), b.data(), legacy.data(), m, n, p);
        tf::matmul_generic_strided(a.data(), b.data(), generic.data(),
                                   m, n, p, n, 1, p, 1, 0, 0);
        tf::matmul_row_sweep(a.data(), b.data(), sweep.data(),
                             m, n, p, n, 1, p, 0, 0);
        check_at(same_bits(legacy, generic),
                 "generic path differs from tf_matmul", m, n, p);
        check_at(same_bits(legacy, sweep),
                 "row sweep differs from tf_matmul", m, n, p);
    }
}

// ---------------------------------------------------------------------------
// 3. The destination is fully written before it is read (H1)
// ---------------------------------------------------------------------------

void test_no_destination_contents_survive_or_leak() {
    const std::int64_t shapes[][3] = {
        {1, 1, 8}, {4, 4, 8}, {5, 3, 9}, {9, 7, 17}, {12, 5, 64},
    };
    for (const auto& shape : shapes) {
        const std::int64_t m = shape[0], n = shape[1], p = shape[2];
        const std::vector<double> a = filled(m * n, 61);
        const std::vector<double> b = filled(n * p, 97);
        const std::size_t count = static_cast<std::size_t>(m * p);

        // Poison one way, poison the other way, and require both that no
        // poison survives and that the two results are bit-identical —
        // which is the stronger statement: no output value depends in any
        // way on what the destination held beforehand.
        std::vector<double> from_nan(count, poison_nan());
        std::vector<double> from_finite(count, kPoisonFinite);
        std::vector<double> from_zero(count, 0.0);
        tf::matmul_row_sweep(a.data(), b.data(), from_nan.data(),
                             m, n, p, n, 1, p, 0, 0);
        tf::matmul_row_sweep(a.data(), b.data(), from_finite.data(),
                             m, n, p, n, 1, p, 0, 0);
        tf::matmul_row_sweep(a.data(), b.data(), from_zero.data(),
                             m, n, p, n, 1, p, 0, 0);
        check_at(same_bits(from_nan, from_finite),
                 "row sweep output depends on prior destination contents",
                 m, n, p);
        check_at(same_bits(from_nan, from_zero),
                 "row sweep output depends on prior destination contents",
                 m, n, p);
        bool survivor = false;
        for (std::size_t i = 0; i < count; ++i) {
            if (std::isnan(from_nan[i]) || from_finite[i] == kPoisonFinite) {
                survivor = true;
            }
        }
        check_at(!survivor, "poison survived the row sweep", m, n, p);
    }
}

void test_the_poison_detector_can_fail() {
    // The negative control. A deliberately incomplete sweep — the same
    // kernel told it has one column fewer than the destination really has
    // — must leave the last column of every row holding the poison, which
    // is how we know the check above is capable of failing.
    const std::int64_t m = 4, n = 3, p = 9;
    const std::vector<double> a = filled(m * n, 61);
    const std::vector<double> b = filled(n * p, 97);
    std::vector<double> destination(static_cast<std::size_t>(m * p),
                                    kPoisonFinite);
    // Writes only p - 1 of each row's p columns, walking b's rows at the
    // real stride so the arithmetic stays in bounds.
    tf::matmul_row_sweep(a.data(), b.data(), destination.data(),
                         m, n, p - 1, n, 1, p, 0, 0);
    int survivors = 0;
    for (std::size_t i = 0; i < destination.size(); ++i) {
        if (destination[i] == kPoisonFinite) {
            ++survivors;
        }
    }
    // The destination is addressed as (m, p - 1) = 32 written cells out of
    // 36, so exactly 4 cells keep the poison.
    check(survivors == 4, "the poison detector did not catch a partial write");
}

// ---------------------------------------------------------------------------
// 4. The exported wrapper routes through the predicate
// ---------------------------------------------------------------------------

void test_exported_wrapper_matches_the_selected_path() {
    // Drive the real guarded export through real Storage handles and
    // require its result to equal whichever internal path the predicate
    // says it should have taken — so the wrapper cannot silently choose
    // differently from the predicate this file pins.
    struct Case { std::int64_t m, n, p, b_stride0, b_stride1; };
    const Case cases[] = {
        {6, 5, 16, 16, 1},   // qualifies: row sweep
        {6, 5, 7, 7, 1},     // p below the threshold: generic
        {6, 5, 16, 1, 5},    // transposed right operand: generic
    };
    for (const Case& c : cases) {
        const std::int64_t a_count = c.m * c.n;
        const std::int64_t b_count = c.n * c.p;
        const std::int64_t out_count = c.m * c.p;
        const std::vector<double> a = filled(a_count, 13);
        const std::vector<double> b = filled(b_count, 29);

        void* a_handle = tf_storage_create(a_count);
        void* b_handle = tf_storage_create(b_count);
        void* out_handle = tf_storage_create(out_count);
        check(a_handle && b_handle && out_handle, "storage creation failed");
        if (!a_handle || !b_handle || !out_handle) {
            continue;
        }
        std::memcpy(tf::storage_f64(a_handle), a.data(),
                    sizeof(double) * static_cast<std::size_t>(a_count));
        std::memcpy(tf::storage_f64(b_handle), b.data(),
                    sizeof(double) * static_cast<std::size_t>(b_count));

        tf_core_matmul(a_handle, b_handle, out_handle, c.m, c.n, c.p,
                       c.n, 1, c.b_stride0, c.b_stride1, 0, 0);
        check(tf_last_error_code() == TF_OK, "the export reported an error");

        std::vector<double> expected(static_cast<std::size_t>(out_count), 0.0);
        if (tf::matmul_prefers_row_sweep(c.m, c.n, c.p, c.b_stride1)) {
            tf::matmul_row_sweep(a.data(), b.data(), expected.data(),
                                 c.m, c.n, c.p, c.n, 1, c.b_stride0, 0, 0);
        } else {
            tf::matmul_generic_strided(a.data(), b.data(), expected.data(),
                                       c.m, c.n, c.p, c.n, 1,
                                       c.b_stride0, c.b_stride1, 0, 0);
        }
        std::vector<double> produced(
            tf::storage_f64(out_handle),
            tf::storage_f64(out_handle) + out_count);
        check_at(same_bits(expected, produced),
                 "the export did not match the predicted path",
                 c.m, c.n, c.p);

        tf_storage_destroy(a_handle);
        tf_storage_destroy(b_handle);
        tf_storage_destroy(out_handle);
    }
}

void test_operands_are_not_mutated() {
    const std::int64_t m = 9, n = 6, p = 16;
    const std::vector<double> a_original = filled(m * n, 3);
    const std::vector<double> b_original = filled(n * p, 91);
    std::vector<double> a = a_original;
    std::vector<double> b = b_original;
    std::vector<double> out(static_cast<std::size_t>(m * p), 0.0);
    tf::matmul_row_sweep(a.data(), b.data(), out.data(), m, n, p, n, 1, p, 0, 0);
    check(same_bits(a, a_original), "the row sweep mutated its left operand");
    check(same_bits(b, b_original), "the row sweep mutated its right operand");
    tf::matmul_generic_strided(a.data(), b.data(), out.data(),
                               m, n, p, n, 1, p, 1, 0, 0);
    check(same_bits(a, a_original), "the generic path mutated its left operand");
    check(same_bits(b, b_original),
          "the generic path mutated its right operand");
}

void test_repeated_calls_are_identical() {
    const std::int64_t m = 11, n = 7, p = 13;
    const std::vector<double> a = filled(m * n, 45);
    const std::vector<double> b = filled(n * p, 67);
    std::vector<double> first(static_cast<std::size_t>(m * p), 0.0);
    tf::matmul_row_sweep(a.data(), b.data(), first.data(),
                         m, n, p, n, 1, p, 0, 0);
    for (int repeat = 0; repeat < 5; ++repeat) {
        std::vector<double> again(static_cast<std::size_t>(m * p),
                                  kPoisonFinite);
        tf::matmul_row_sweep(a.data(), b.data(), again.data(),
                             m, n, p, n, 1, p, 0, 0);
        check(same_bits(first, again), "repeated calls are not identical");
    }
}

}  // namespace

int main() {
    test_predicate_table();
    test_finite_bit_identity_across_shapes();
    test_finite_bit_identity_at_larger_shapes();
    test_finite_bit_identity_through_strided_left_operands();
    test_the_contract_comparator_is_capable_of_failing();
    test_the_contract_with_nans_in_each_operand();
    test_the_numerical_contract_on_special_values();
    test_agreement_with_the_legacy_raw_kernel();
    test_no_destination_contents_survive_or_leak();
    test_the_poison_detector_can_fail();
    test_exported_wrapper_matches_the_selected_path();
    test_operands_are_not_mutated();
    test_repeated_calls_are_identical();

    if (g_failures != 0) {
        std::printf("%d matmul check(s) failed\n", g_failures);
        return 1;
    }
    std::printf("all matmul checks passed\n");
    return 0;
}
